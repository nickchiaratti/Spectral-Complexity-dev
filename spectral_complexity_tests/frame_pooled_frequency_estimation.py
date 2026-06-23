# Gridless Frequency Estimation Module
import numpy as np
import matplotlib.pyplot as plt
import torch
import math
import h5py
import datetime
import warnings
import time
import os

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--location", type=str, default="Rochesterv2", help="Location name (e.g. Rochesterv2, Tait)")
args = parser.parse_args()

LOCATION = args.location
H5_PATH = f"C:/satelliteImagery/HLST30/HLST_{LOCATION}_Harmonized_SC_EM-7_Norm-bandCount.h5"
TARGET_METRIC = "sliding_volume_z_score"
START_DATE = "2022-01-01"
END_DATE = "2026-06-01"
OUTPUT_H5_PATH = f"C:/satelliteImagery/HLST30/Frequency Estimation Maps/{LOCATION}_Frequency_Estimation_{START_DATE}_{END_DATE}.h5"
LOAD_EXISTING_RESULTS = True

NDFT_MIN_CPY = 0.25
NDFT_MAX_CPY = 4.0
NDFT_GRID_BINS = 150
SECONDS_IN_YEAR = 365.25 * 24 * 3600

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 5000  # Safe batch size to prevent VRAM/RAM OOM

# --- Batched Tensor Algorithms ---

def batched_ndft_init(t, y, valid_mask, min_f, max_f, bins=1000, max_atoms=20):
    """Computes a batched Non-Uniform DFT to seed the continuous algorithms and provide grid baseline."""
    f_grid = torch.linspace(min_f, max_f, bins, device=DEVICE)
    omega = 2 * math.pi * f_grid
    
    E = torch.exp(-1j * omega.unsqueeze(1) * t.unsqueeze(0))
    y_masked = (y * valid_mask).to(torch.complex64)
    
    spectrum = torch.abs(torch.matmul(y_masked, E.T.conj()))
    
    top_amps, top_indices = torch.topk(spectrum, max_atoms, dim=1)
    f_init = f_grid[top_indices]
    
    # NDFT top frequency is simply the highest peak on the grid
    f_ndft_top = f_init[:, 0].cpu().numpy()
    
    return f_init, spectrum.max(dim=1)[0], f_ndft_top

def get_top_unique_frequencies(f_batch, amps_batch):
    """Prunes soft-thresholded atoms and deduplicates frequencies that converged into the same well."""
    N = f_batch.shape[0]
    top_f = np.full(N, np.nan)
    
    for i in range(N):
        f_arr = f_batch[i]
        a_arr = amps_batch[i]
        
        threshold = 0.05 * np.max(a_arr)
        active = a_arr > threshold
        f_active = f_arr[active]
        a_active = a_arr[active]
        
        if len(a_active) == 0:
            continue
            
        f_unique = []
        amps_unique = []
        
        sort_active = np.argsort(a_active)[::-1]
        for f, amp in zip(f_active[sort_active], a_active[sort_active]):
            if not any(abs(f - uf) < 0.05 for uf in f_unique):
                f_unique.append(f)
                amps_unique.append(amp)
        
        if len(f_unique) > 0:
            top_f[i] = f_unique[0]
            
    return top_f

def compute_batched_nomp(t, y, valid_mask, min_f=0.3, max_f=4.0, bins=1000, max_components=3):
    """Batched 3-Atom Newtonized Orthogonal Matching Pursuit (NOMP) with ridge-stabilized joint refinement."""
    B, T = y.shape
    f_grid = torch.linspace(min_f, max_f, bins, device=DEVICE)
    omega_grid = 2 * math.pi * f_grid
    
    E = torch.exp(-1j * omega_grid.unsqueeze(1) * t.unsqueeze(0))
    y_masked = (y * valid_mask).to(torch.complex64)
    
    frequencies = []
    
    def build_X(freqs_tensor):
        K = freqs_tensor.shape[1]
        omega = 2 * math.pi * freqs_tensor
        wt = omega.unsqueeze(2) * t.view(1, 1, T)
        A_real = torch.cos(wt)
        A_imag = torch.sin(wt)
        X = torch.cat([A_real.transpose(1, 2), A_imag.transpose(1, 2)], dim=2)
        return X * valid_mask.unsqueeze(2)

    for k in range(max_components):
        if len(frequencies) > 0:
            f_tensor = torch.stack(frequencies, dim=1)
            X = build_X(f_tensor)
            
            XtX = torch.bmm(X.transpose(1, 2), X)
            I = torch.eye(X.shape[2], device=DEVICE).unsqueeze(0)
            XtX += 1e-5 * I
            Xty = torch.bmm(X.transpose(1, 2), y_masked.real.unsqueeze(2))
            beta = torch.bmm(torch.inverse(XtX), Xty)
            
            y_pred = torch.bmm(X, beta).squeeze(2)
            residual = y_masked.real - y_pred
        else:
            residual = y_masked.real.clone()
            
        spectrum = torch.abs(torch.matmul(residual.to(torch.complex64), E.T.conj()))
        top_indices = torch.argmax(spectrum, dim=1)
        f_new = f_grid[top_indices]
        frequencies.append(f_new)
        
        f_tensor = torch.stack(frequencies, dim=1).detach().requires_grad_(True)
        optimizer = torch.optim.Adam([f_tensor], lr=0.01)
        
        for _ in range(30):
            optimizer.zero_grad()
            f_clamped = torch.clamp(f_tensor, min=min_f, max=max_f)
            X_opt = build_X(f_clamped)
            
            XtX_opt = torch.bmm(X_opt.transpose(1, 2), X_opt)
            I_opt = torch.eye(X_opt.shape[2], device=DEVICE).unsqueeze(0)
            XtX_opt = XtX_opt + 1e-4 * I_opt
            Xty_opt = torch.bmm(X_opt.transpose(1, 2), y_masked.real.unsqueeze(2))
            
            beta_opt = torch.bmm(torch.inverse(XtX_opt), Xty_opt)
            y_pred_opt = torch.bmm(X_opt, beta_opt).squeeze(2)
            
            loss = torch.sum(valid_mask * (y_masked.real - y_pred_opt)**2)
            loss.backward()
            optimizer.step()
            
        frequencies = [torch.clamp(f_tensor[:, i].detach(), min=min_f, max=max_f) for i in range(k+1)]
        
    f_final = torch.stack(frequencies, dim=1)
    X_final = build_X(f_final)
    XtX_final = torch.bmm(X_final.transpose(1, 2), X_final)
    I_final = torch.eye(X_final.shape[2], device=DEVICE).unsqueeze(0)
    XtX_final += 1e-5 * I_final
    Xty_final = torch.bmm(X_final.transpose(1, 2), y_masked.real.unsqueeze(2))
    beta_final = torch.bmm(torch.inverse(XtX_final), Xty_final).squeeze(2)
    
    amps = []
    for i in range(max_components):
        c = beta_final[:, i]
        s = beta_final[:, i + max_components]
        amps.append(torch.sqrt(c**2 + s**2))
        
    amps_final = torch.stack(amps, dim=1).cpu().numpy()
    f_final = f_final.cpu().numpy()
    
    return get_top_unique_frequencies(f_final, amps_final)

def compute_batched_cbpdn(t, y, valid_mask, f_init, max_spec, min_f, max_f, max_atoms=20):
    """Continuous Basis Pursuit Denoising (C-BPDN) optimized for batched tensor execution."""
    N, T = y.shape
    
    freqs = f_init.clone().detach().requires_grad_(True)
    a_real = torch.randn((N, max_atoms), dtype=torch.float32, device=DEVICE, requires_grad=True)
    a_imag = torch.randn((N, max_atoms), dtype=torch.float32, device=DEVICE, requires_grad=True)
    
    optimizer = torch.optim.Adam([freqs, a_real, a_imag], lr=0.05)
    lambda_reg = 0.15 * max_spec.unsqueeze(1)
    
    for _ in range(800):
        optimizer.zero_grad()
        omega = 2 * math.pi * torch.clamp(freqs, min=min_f, max=max_f)
        
        wt = omega.unsqueeze(2) * t.view(1, 1, T)
        A_real = torch.cos(wt)
        A_imag = torch.sin(wt)
        
        pred_real = torch.bmm(a_real.unsqueeze(1), A_real).squeeze(1)
        pred_imag = torch.bmm(a_imag.unsqueeze(1), A_imag).squeeze(1)
        
        y_pred = pred_real - pred_imag
        
        mse = torch.sum(valid_mask * (y - y_pred)**2, dim=1)
        
        l1 = lambda_reg.squeeze(1) * torch.sum(torch.sqrt(a_real**2 + a_imag**2 + 1e-8), dim=1)
        
        loss = torch.sum(mse + l1)
        loss.backward()
        optimizer.step()
        
    f_final = torch.clamp(freqs, min=min_f, max=max_f).detach().cpu().numpy()
    amps_final = torch.sqrt(a_real**2 + a_imag**2).detach().cpu().numpy()
    
    return get_top_unique_frequencies(f_final, amps_final)

def compute_batched_cirl(t, y, valid_mask, f_init, max_spec, min_f, max_f, max_atoms=20):
    """Continuous Iterative Reweighted L1 (CIRL) optimized for batched tensor execution."""
    N, T = y.shape
    
    freqs = f_init.clone().detach().requires_grad_(True)
    a_real = torch.randn((N, max_atoms), dtype=torch.float32, device=DEVICE, requires_grad=True)
    a_imag = torch.randn((N, max_atoms), dtype=torch.float32, device=DEVICE, requires_grad=True)
    
    optimizer = torch.optim.Adam([freqs, a_real, a_imag], lr=0.05)
    
    for step in range(800):
        optimizer.zero_grad()
        omega = 2 * math.pi * torch.clamp(freqs, min=min_f, max=max_f)
        
        wt = omega.unsqueeze(2) * t.view(1, 1, T)
        A_real = torch.cos(wt)
        A_imag = torch.sin(wt)
        
        pred_real = torch.bmm(a_real.unsqueeze(1), A_real).squeeze(1)
        pred_imag = torch.bmm(a_imag.unsqueeze(1), A_imag).squeeze(1)
        
        y_pred = pred_real - pred_imag
        mse = torch.sum(valid_mask * (y - y_pred)**2, dim=1)
        
        eps = 1e-3 if step < 400 else 1e-4
        amps = torch.sqrt(a_real**2 + a_imag**2 + 1e-8)
        gls_penalty = torch.sum(torch.log(amps + eps), dim=1) * 0.5 * max_spec
        
        loss = torch.sum(mse + gls_penalty)
        loss.backward()
        optimizer.step()
        
    f_final = torch.clamp(freqs, min=min_f, max=max_f).detach().cpu().numpy()
    amps_final = torch.sqrt(a_real**2 + a_imag**2).detach().cpu().numpy()
    
    return get_top_unique_frequencies(f_final, amps_final)

def main():
    if LOAD_EXISTING_RESULTS and os.path.exists(OUTPUT_H5_PATH):
        print(f"Loading existing results from {OUTPUT_H5_PATH}...")
        with h5py.File(OUTPUT_H5_PATH, 'r') as in_f:
            maps_grp = in_f['Spatial_Maps']
            top_f_ndft = maps_grp['NDFT_Frequency_CPY'][:]
            top_f_nomp = maps_grp['NOMP_Frequency_CPY'][:]
            top_f_cbpdn = maps_grp['CBPDN_Frequency_CPY'][:]
            top_f_cirl = maps_grp['CIRL_Frequency_CPY'][:]

            period_ndft = maps_grp['NDFT_Period_Days'][:]
            period_nomp = maps_grp['NOMP_Period_Days'][:]
            period_cbpdn = maps_grp['CBPDN_Period_Days'][:]
            period_cirl = maps_grp['CIRL_Period_Days'][:]
            
            stats_grp = in_f['Statistics']
            ndft_mean = stats_grp.attrs['NDFT_Mean_CPY']
            ndft_median = stats_grp.attrs['NDFT_Median_CPY']
            time_ndft = stats_grp.attrs['NDFT_Compute_s']

            nomp_mean = stats_grp.attrs['NOMP_Mean_CPY']
            nomp_median = stats_grp.attrs['NOMP_Median_CPY']
            time_nomp = stats_grp.attrs['NOMP_Compute_s']

            cbpdn_mean = stats_grp.attrs['CBPDN_Mean_CPY']
            cbpdn_median = stats_grp.attrs['CBPDN_Median_CPY']
            time_cbpdn = stats_grp.attrs['CBPDN_Compute_s']

            cirl_mean = stats_grp.attrs['CIRL_Mean_CPY']
            cirl_median = stats_grp.attrs['CIRL_Median_CPY']
            time_cirl = stats_grp.attrs['CIRL_Compute_s']

            num_pixels = np.sum(~np.isnan(top_f_ndft))

        _do_plotting(
            top_f_ndft, top_f_nomp, top_f_cbpdn, top_f_cirl,
            period_ndft, period_nomp, period_cbpdn, period_cirl,
            ndft_mean, ndft_median, time_ndft,
            nomp_mean, nomp_median, time_nomp,
            cbpdn_mean, cbpdn_median, time_cbpdn,
            cirl_mean, cirl_median, time_cirl,
            num_pixels
        )
        return

    print(f"Loading data from {H5_PATH}...")
    with h5py.File(H5_PATH, 'r') as f:
        data_grp = f['/HDFEOS/GRIDS/HARMONIZED/Data Fields']
        metric_ds = data_grp[TARGET_METRIC]
        acq_times = metric_ds.attrs['acquisition_time'][:]
        
        y_data = metric_ds[:]
        common_mask = data_grp['common_mask'][:]

    T_orig, H, W = y_data.shape
    print(f"Original dataset shape: {y_data.shape}")

    # Time filtering
    start_dt = datetime.datetime.strptime(START_DATE, "%Y-%m-%d").replace(tzinfo=datetime.timezone.utc)
    end_dt = datetime.datetime.strptime(END_DATE, "%Y-%m-%d").replace(tzinfo=datetime.timezone.utc)
    time_mask = (acq_times >= start_dt.timestamp()) & (acq_times <= end_dt.timestamp())

    acq_times = acq_times[time_mask]
    y_data = y_data[time_mask, :, :]
    common_mask = common_mask[time_mask, :, :]
    print(f"Time-filtered dataset shape: {y_data.shape}")

    # Reshape to (N, T) for batched processing
    N = H * W
    T_new = len(acq_times)
    y_flat = y_data.reshape(T_new, N).T
    mask_flat = common_mask.reshape(T_new, N).T

    valid_mask = (mask_flat == 0) & (~np.isnan(y_flat))
    y_flat = np.where(valid_mask, y_flat, 0.0)

    # Filter out pixels with insufficient data (Must be >= 10 for safe 3-Atom NOMP least squares)
    valid_pixel_counts = np.sum(valid_mask, axis=1)
    process_pixels_idx = np.where(valid_pixel_counts >= 10)[0]
    print(f"Found {len(process_pixels_idx)} / {N} pixels with sufficient valid data (>=10).")

    if len(process_pixels_idx) == 0:
        print("No valid pixels found. Exiting.")
        return

    y_process = y_flat[process_pixels_idx]
    mask_process = valid_mask[process_pixels_idx]

    # Mean center per pixel
    y_means = np.sum(y_process, axis=1, keepdims=True) / (np.sum(mask_process, axis=1, keepdims=True) + 1e-8)
    y_centered = np.where(mask_process, y_process - y_means, 0.0)

    t0 = acq_times[0]
    t_years = (acq_times - t0) / SECONDS_IN_YEAR
    t_tensor = torch.tensor(t_years, dtype=torch.float32, device=DEVICE)

    # Setup batch arrays
    top_f_ndft_process = np.full(len(process_pixels_idx), np.nan)
    top_f_nomp_process = np.full(len(process_pixels_idx), np.nan)
    top_f_cbpdn_process = np.full(len(process_pixels_idx), np.nan)
    top_f_cirl_process = np.full(len(process_pixels_idx), np.nan)

    time_ndft = 0.0
    time_nomp = 0.0
    time_cbpdn = 0.0
    time_cirl = 0.0

    num_batches = math.ceil(len(process_pixels_idx) / BATCH_SIZE)
    print(f"Executing PyTorch Estimators on device: {DEVICE} in {num_batches} batches...")

    for i in range(num_batches):
        start_idx = i * BATCH_SIZE
        end_idx = min((i + 1) * BATCH_SIZE, len(process_pixels_idx))
        print(f"  Processing batch {i+1}/{num_batches} (pixels {start_idx} to {end_idx})...")
        
        y_batch = torch.tensor(y_centered[start_idx:end_idx], dtype=torch.float32, device=DEVICE)
        mask_batch = torch.tensor(mask_process[start_idx:end_idx], dtype=torch.float32, device=DEVICE)
        
        # 1. NDFT (Grid-based Baseline)
        t0_alg = time.time()
        f_init, max_spec, f_ndft = batched_ndft_init(t_tensor, y_batch, mask_batch, NDFT_MIN_CPY, NDFT_MAX_CPY, bins=NDFT_GRID_BINS)
        top_f_ndft_process[start_idx:end_idx] = f_ndft
        time_ndft += time.time() - t0_alg
        
        # 2. NOMP (Greedy OMP + Continuous Refinement)
        t0_alg = time.time()
        f_nomp = compute_batched_nomp(t_tensor, y_batch, mask_batch, NDFT_MIN_CPY, NDFT_MAX_CPY, bins=NDFT_GRID_BINS, max_components=3)
        top_f_nomp_process[start_idx:end_idx] = f_nomp
        time_nomp += time.time() - t0_alg
        
        # 3. C-BPDN (Continuous L1)
        t0_alg = time.time()
        f_cbpdn = compute_batched_cbpdn(t_tensor, y_batch, mask_batch, f_init, max_spec, NDFT_MIN_CPY, NDFT_MAX_CPY)
        top_f_cbpdn_process[start_idx:end_idx] = f_cbpdn
        time_cbpdn += time.time() - t0_alg
        
        # 4. CIRL (Continuous Log-Sum)
        t0_alg = time.time()
        f_cirl = compute_batched_cirl(t_tensor, y_batch, mask_batch, f_init, max_spec, NDFT_MIN_CPY, NDFT_MAX_CPY)
        top_f_cirl_process[start_idx:end_idx] = f_cirl
        time_cirl += time.time() - t0_alg

    # Reconstruct 1D output back to 2D global frame (H, W)
    def reconstruct_map(process_array):
        global_map = np.full(N, np.nan)
        global_map[process_pixels_idx] = process_array
        return global_map.reshape(H, W)
        
    top_f_ndft = reconstruct_map(top_f_ndft_process)
    top_f_nomp = reconstruct_map(top_f_nomp_process)
    top_f_cbpdn = reconstruct_map(top_f_cbpdn_process)
    top_f_cirl = reconstruct_map(top_f_cirl_process)

    # Statistical Aggregation (Mean & Median)
    def calc_stats(f_map):
        valid = f_map[~np.isnan(f_map)]
        if len(valid) == 0: return np.nan, np.nan
        return np.mean(valid), np.median(valid)

    ndft_mean, ndft_median = calc_stats(top_f_ndft)
    nomp_mean, nomp_median = calc_stats(top_f_nomp)
    cbpdn_mean, cbpdn_median = calc_stats(top_f_cbpdn)
    cirl_mean, cirl_median = calc_stats(top_f_cirl)

    # --- Save Results to HDF5 ---
    print(f"Saving spatial maps and statistics to {OUTPUT_H5_PATH}...")
    os.makedirs(os.path.dirname(OUTPUT_H5_PATH), exist_ok=True)
    with h5py.File(OUTPUT_H5_PATH, 'w') as out_f:
        # Configuration Attributes
        out_f.attrs['LOCATION'] = LOCATION
        out_f.attrs['TARGET_METRIC'] = TARGET_METRIC
        out_f.attrs['START_DATE'] = START_DATE
        out_f.attrs['END_DATE'] = END_DATE
        out_f.attrs['NDFT_GRID_BINS'] = NDFT_GRID_BINS
        out_f.attrs['NDFT_MIN_CPY'] = NDFT_MIN_CPY
        out_f.attrs['NDFT_MAX_CPY'] = NDFT_MAX_CPY
        
        # Statistics Attributes
        stats_grp = out_f.create_group('Statistics')
        stats_grp.attrs['NDFT_Mean_CPY'] = ndft_mean
        stats_grp.attrs['NDFT_Median_CPY'] = ndft_median
        stats_grp.attrs['NDFT_Compute_s'] = time_ndft
        
        stats_grp.attrs['NOMP_Mean_CPY'] = nomp_mean
        stats_grp.attrs['NOMP_Median_CPY'] = nomp_median
        stats_grp.attrs['NOMP_Compute_s'] = time_nomp
        
        stats_grp.attrs['CBPDN_Mean_CPY'] = cbpdn_mean
        stats_grp.attrs['CBPDN_Median_CPY'] = cbpdn_median
        stats_grp.attrs['CBPDN_Compute_s'] = time_cbpdn
        
        stats_grp.attrs['CIRL_Mean_CPY'] = cirl_mean
        stats_grp.attrs['CIRL_Median_CPY'] = cirl_median
        stats_grp.attrs['CIRL_Compute_s'] = time_cirl

        # Spatial Map Datasets (Frequencies)
        maps_grp = out_f.create_group('Spatial_Maps')
        maps_grp.create_dataset('NDFT_Frequency_CPY', data=top_f_ndft, compression='gzip')
        maps_grp.create_dataset('NOMP_Frequency_CPY', data=top_f_nomp, compression='gzip')
        maps_grp.create_dataset('CBPDN_Frequency_CPY', data=top_f_cbpdn, compression='gzip')
        maps_grp.create_dataset('CIRL_Frequency_CPY', data=top_f_cirl, compression='gzip')

        # Spatial Map Datasets (Periods in Days)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            period_ndft = 365.25 / top_f_ndft
            period_nomp = 365.25 / top_f_nomp
            period_cbpdn = 365.25 / top_f_cbpdn
            period_cirl = 365.25 / top_f_cirl
            
        maps_grp.create_dataset('NDFT_Period_Days', data=period_ndft, compression='gzip')
        maps_grp.create_dataset('NOMP_Period_Days', data=period_nomp, compression='gzip')
        maps_grp.create_dataset('CBPDN_Period_Days', data=period_cbpdn, compression='gzip')
        maps_grp.create_dataset('CIRL_Period_Days', data=period_cirl, compression='gzip')

    print("HDF5 Save Complete!")

    num_pixels = len(process_pixels_idx)
    _do_plotting(
        top_f_ndft, top_f_nomp, top_f_cbpdn, top_f_cirl,
        period_ndft, period_nomp, period_cbpdn, period_cirl,
        ndft_mean, ndft_median, time_ndft,
        nomp_mean, nomp_median, time_nomp,
        cbpdn_mean, cbpdn_median, time_cbpdn,
        cirl_mean, cirl_median, time_cirl,
        num_pixels
    )

def _do_plotting(
    top_f_ndft, top_f_nomp, top_f_cbpdn, top_f_cirl,
    period_ndft, period_nomp, period_cbpdn, period_cirl,
    ndft_mean, ndft_median, time_ndft,
    nomp_mean, nomp_median, time_nomp,
    cbpdn_mean, cbpdn_median, time_cbpdn,
    cirl_mean, cirl_median, time_cirl,
    num_pixels
):
    # --- Plotting ---
    fig1, axes1 = plt.subplots(1, 4, figsize=(24, 6))
    fig1.canvas.manager.set_window_title(f"Per-Pixel Batched Frequency Estimation (CPY) - {LOCATION}")

    fig2, axes2 = plt.subplots(1, 4, figsize=(24, 6))
    fig2.canvas.manager.set_window_title(f"Per-Pixel Batched Period Estimation (Days) - {LOCATION}")

    def format_stat(mean, median):
        mean_d = 365.25/mean if not np.isnan(mean) else 0
        med_d = 365.25/median if not np.isnan(median) else 0
        return f"\nMean: {mean:.3f} cpy ({mean_d:.1f} d)\nMedian: {median:.3f} cpy ({med_d:.1f} d)"

    # Maps for Figure 1 (CPY)
    maps_cpy = [
        (top_f_ndft, "NDFT Top Freq (CPY)" + format_stat(ndft_mean, ndft_median)),
        (top_f_nomp, "NOMP Top Freq (CPY)" + format_stat(nomp_mean, nomp_median)),
        (top_f_cbpdn, "C-BPDN Top Freq (CPY)" + format_stat(cbpdn_mean, cbpdn_median)),
        (top_f_cirl, "CIRL Top Freq (CPY)" + format_stat(cirl_mean, cirl_median))
    ]

    for i, (m, title) in enumerate(maps_cpy):
        ax = axes1[i]
        im = ax.imshow(m, cmap='jet', vmin=NDFT_MIN_CPY, vmax=NDFT_MAX_CPY)
        ax.set_title(title, fontsize=12, fontweight='bold')
        fig1.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks([]); ax.set_yticks([])

    # Maps for Figure 2 (Days)
    maps_days = [
        (period_ndft, "NDFT Period (Days)" + format_stat(ndft_mean, ndft_median)),
        (period_nomp, "NOMP Period (Days)" + format_stat(nomp_mean, nomp_median)),
        (period_cbpdn, "C-BPDN Period (Days)" + format_stat(cbpdn_mean, cbpdn_median)),
        (period_cirl, "CIRL Period (Days)" + format_stat(cirl_mean, cirl_median))
    ]

    for i, (m, title) in enumerate(maps_days):
        ax = axes2[i]
        im = ax.imshow(m, cmap='jet', vmin=365.25/NDFT_MAX_CPY, vmax=365.25/NDFT_MIN_CPY)
        ax.set_title(title, fontsize=12, fontweight='bold')
        fig2.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks([]); ax.set_yticks([])

    suptitle_text = f"Batched Per-Pixel Statistics ({num_pixels:,} pixels) | Span: {START_DATE} to {END_DATE}"
    fig1.suptitle(suptitle_text, fontsize=14, fontweight='bold')
    fig2.suptitle(suptitle_text, fontsize=14, fontweight='bold')
             
    fig1.subplots_adjust(left=0.03, right=0.97, top=0.75, bottom=0.05, wspace=0.15)
    fig2.subplots_adjust(left=0.03, right=0.97, top=0.75, bottom=0.05, wspace=0.15)
    plt.show()

if __name__ == "__main__":
    main()