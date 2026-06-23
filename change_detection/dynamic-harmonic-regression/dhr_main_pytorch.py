import os
import h5py
import numpy as np
import datetime
import math
from tqdm import tqdm
import torch

# ==========================================
# 1. CONFIGURATION
# ==========================================
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--location", type=str, default="Tait", help="Location name")
args = parser.parse_args()

LOCATION = args.location
H5_PATH = f"C:/satelliteImagery/HLST30/HLST_{LOCATION}_Harmonized_SC_EM-7_Norm-bandCount.h5"

# 'ALFT' (Iterative Grid Search), 'NDFT' (Static Grid), 'NOMP', 'CBPDN', 'CIRL'
FREQUENCY_ESTIMATOR = 'NOMP'

START_DATE = "2020-01-01"
END_DATE = "2026-06-01"

TARGET_METRIC = 'sliding_volume_z_score'
IGNORE_COMMON_MASK = False # If True, utilizes noisy/cloudy pixels and relies on NDFT to filter noise
RMSE_MULTIPLIER = 2
CONSECUTIVE_ANOMALIES = 4
MAX_WINDOW_YEARS = 4.0
MIN_WINDOW_YEARS = 0.1
K_FREQUENCIES = 2
MIN_SAMPLES = 2 * K_FREQUENCIES + 1 + 3 # 8 parameters + 3 df
CHUNK_SIZE = 128 # Spatial block size
NDFT_MIN_CPY = 0.2
NDFT_MAX_CPY = 4.0
NDFT_GRID_BINS = 100
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def extract_fractional_years(acq_times):
    """Converts UNIX timestamps into continuous fractional years (t)."""
    frac_years = []
    for dt in acq_times:
        dt_obj = datetime.datetime.fromtimestamp(float(dt), tz=datetime.timezone.utc)
        year = dt_obj.year
        start_of_year = datetime.datetime(year, 1, 1, tzinfo=datetime.timezone.utc)
        start_of_next = datetime.datetime(year + 1, 1, 1, tzinfo=datetime.timezone.utc)
        year_duration = (start_of_next - start_of_year).total_seconds()
        elapsed = (dt_obj - start_of_year).total_seconds()
        frac_years.append(year + (elapsed / year_duration))
    return np.array(frac_years)

def batched_ndft_init(t, y, valid_mask, min_f, max_f, bins=1000, max_atoms=20):
    f_grid = torch.linspace(min_f, max_f, bins, device=DEVICE)
    omega = 2 * math.pi * f_grid
    
    E = torch.exp(-1j * omega.unsqueeze(1) * t.unsqueeze(0))
    y_masked = (y * valid_mask).to(torch.complex64)
    
    spectrum = torch.abs(torch.matmul(y_masked, E.T.conj()))
    
    top_amps, top_indices = torch.topk(spectrum, max_atoms, dim=1)
    f_init = f_grid[top_indices]
    
    return f_init, spectrum.max(dim=1)[0], f_init[:, 0]

def get_top_unique_frequencies(f_batch, amps_batch, k_freqs=3):
    N = f_batch.shape[0]
    top_f = torch.full((k_freqs, N), NDFT_MIN_CPY, device=DEVICE) # Fallback initialization
    
    for i in range(N):
        f_arr = f_batch[i]
        a_arr = amps_batch[i]
        
        threshold = 0.05 * torch.max(a_arr)
        active = a_arr > threshold
        f_active = f_arr[active]
        a_active = a_arr[active]
        
        if len(a_active) < k_freqs:
            # Fallback to no threshold
            f_active = f_arr
            a_active = a_arr
            
        f_unique = []
        amps_unique = []
        
        sort_active = torch.argsort(a_active, descending=True)
        for idx in sort_active:
            f = f_active[idx]
            amp = a_active[idx]
            if not any(abs(f - uf) < 0.05 for uf in f_unique):
                f_unique.append(f)
                amps_unique.append(amp)
                if len(f_unique) == k_freqs:
                    break
                    
        # Pad if still not enough unique frequencies
        while len(f_unique) < k_freqs:
            f_unique.append(f_arr[torch.argmax(a_arr)])
            
        for j in range(k_freqs):
            top_f[j, i] = f_unique[j]
            
    return top_f

def compute_batched_nomp(t, y, valid_mask, min_f=0.3, max_f=4.0, bins=1000, max_components=3):
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
        
    amps_final = torch.stack(amps, dim=1)
    
    return get_top_unique_frequencies(f_final, amps_final, max_components)

def compute_batched_cbpdn(t, y, valid_mask, f_init, max_spec, min_f, max_f, max_atoms=20, max_components=3):
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
        
    f_final = torch.clamp(freqs, min=min_f, max=max_f).detach()
    amps_final = torch.sqrt(a_real**2 + a_imag**2).detach()
    
    return get_top_unique_frequencies(f_final, amps_final, max_components)

def compute_batched_cirl(t, y, valid_mask, f_init, max_spec, min_f, max_f, max_atoms=20, max_components=3):
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
        
    f_final = torch.clamp(freqs, min=min_f, max=max_f).detach()
    amps_final = torch.sqrt(a_real**2 + a_imag**2).detach()
    
    return get_top_unique_frequencies(f_final, amps_final, max_components)

def main():
    _term_str = f"K{K_FREQUENCIES}"
    _win_str = f"W{int(MAX_WINDOW_YEARS)}"
    _mask_str = "_unmasked" if IGNORE_COMMON_MASK else ""
    output_h5 = f"C:/satelliteImagery/HLST30/DHR/{LOCATION}_DHR_Change_Detection_{FREQUENCY_ESTIMATOR}_{START_DATE}_{END_DATE}_{_term_str}_{_win_str}{_mask_str}.h5"

    print(f"Loading data from {H5_PATH}...")
    with h5py.File(H5_PATH, 'r') as f:
        data_grp = f['/HDFEOS/GRIDS/HARMONIZED/Data Fields']
        metric_ds = data_grp[TARGET_METRIC]
        
        acq_times = metric_ds.attrs['acquisition_time'][:]
        y_data = metric_ds[...]
        
        common_mask = data_grp['common_mask'][...]
        if IGNORE_COMMON_MASK:
            valid_mask = ~np.isnan(y_data)
        else:
            valid_mask = (common_mask == 0) & ~np.isnan(y_data)
        
        geo_transform = metric_ds.attrs.get('GeoTransform')
        spatial_ref = metric_ds.attrs.get('spatial_ref')
        
    num_frames, height, width = y_data.shape
    frac_years = extract_fractional_years(acq_times)
    
    # Sort chronologically
    sort_idx = np.argsort(acq_times)
    acq_times = acq_times[sort_idx]
    frac_years = frac_years[sort_idx]
    y_data = y_data[sort_idx, ...]
    valid_mask = valid_mask[sort_idx, ...]

    # Filter by date range
    start_ts = datetime.datetime.strptime(START_DATE, "%Y-%m-%d").replace(tzinfo=datetime.timezone.utc).timestamp()
    end_ts = datetime.datetime.strptime(END_DATE, "%Y-%m-%d").replace(tzinfo=datetime.timezone.utc).timestamp()
    
    date_mask = (acq_times >= start_ts) & (acq_times <= end_ts)
    acq_times = acq_times[date_mask]
    frac_years = frac_years[date_mask]
    y_data = y_data[date_mask, ...]
    valid_mask = valid_mask[date_mask, ...]
    
    num_frames = len(acq_times)
    
    print(f"Dataset shape: {num_frames} frames, {height}x{width} pixels")
    print(f"Using device: {DEVICE}")

    # Output arrays on CPU
    change_date_map = np.full((height, width), np.nan, dtype=np.float64)
    change_count_map = np.zeros((height, width), dtype=np.int32)
    predicted_series = np.full((num_frames, height, width), np.nan, dtype=np.float32)
    rmse_series = np.full((num_frames, height, width), np.nan, dtype=np.float32)
    dominant_frequencies_series = np.full((num_frames, K_FREQUENCIES, height, width), np.nan, dtype=np.float32)
    amplitude_series = np.full((num_frames, K_FREQUENCIES, height, width), np.nan, dtype=np.float32)
    anomaly_flags = np.zeros((num_frames, height, width), dtype=np.uint8)

    y_data_torch = torch.from_numpy(y_data).float()
    y_data_torch = torch.nan_to_num(y_data_torch, nan=0.0)
    valid_mask_torch = torch.from_numpy(valid_mask).bool()
    frac_years_torch = torch.from_numpy(frac_years).float().to(DEVICE)
    acq_times_torch = torch.from_numpy(acq_times).double().to(DEVICE)

    # Frequency Grid for NDFT
    f_grid = torch.linspace(NDFT_MIN_CPY, NDFT_MAX_CPY, NDFT_GRID_BINS, device=DEVICE)
    Omega = 2.0 * math.pi * f_grid
    
    print("\nExecuting Batched Dynamic Harmonic Regression...")
    
    y_chunks = list(range(0, height, CHUNK_SIZE))
    x_chunks = list(range(0, width, CHUNK_SIZE))
    total_chunks = len(y_chunks) * len(x_chunks)
    
    pbar = tqdm(total=total_chunks, desc="Spatial Chunks", position=0)
    
    for y_start in y_chunks:
        y_end = min(y_start + CHUNK_SIZE, height)
        for x_start in x_chunks:
            x_end = min(x_start + CHUNK_SIZE, width)
            
            chunk_h = y_end - y_start
            chunk_w = x_end - x_start
            P = chunk_h * chunk_w
            
            Y_chunk = y_data_torch[:, y_start:y_end, x_start:x_end].reshape(num_frames, P).to(DEVICE)
            M_chunk = valid_mask_torch[:, y_start:y_end, x_start:x_end].reshape(num_frames, P).to(DEVICE)
            
            # Precompute first valid time for each pixel
            any_valid = M_chunk.any(dim=0)
            first_valid_idx = M_chunk.to(torch.int8).argmax(dim=0)
            first_valid_time = frac_years_torch[first_valid_idx]
            first_valid_time[~any_valid] = float('inf')
            
            # State tracking arrays for this chunk
            chunk_consec = torch.zeros(P, dtype=torch.int32, device=DEVICE)
            streak_start = torch.zeros(P, dtype=torch.int32, device=DEVICE)
            chunk_count = torch.zeros(P, dtype=torch.int32, device=DEVICE)
            chunk_date = torch.full((P,), np.nan, dtype=torch.float64, device=DEVICE)
            
            c_pred = torch.full((num_frames, P), np.nan, dtype=torch.float32, device=DEVICE)
            c_rmse = torch.full((num_frames, P), np.nan, dtype=torch.float32, device=DEVICE)
            c_freq = torch.full((num_frames, K_FREQUENCIES, P), np.nan, dtype=torch.float32, device=DEVICE)
            c_amp = torch.full((num_frames, K_FREQUENCIES, P), np.nan, dtype=torch.float32, device=DEVICE)
            c_flags = torch.zeros((num_frames, P), dtype=torch.uint8, device=DEVICE)

            frame_pbar = tqdm(total=num_frames, desc="Frames", position=1, leave=False)
            for t in range(num_frames):
                target_time = frac_years_torch[t]
                
                window_start = target_time - MAX_WINDOW_YEARS
                in_window = (frac_years_torch >= window_start) & (frac_years_torch < target_time)
                W_indices = torch.where(in_window)[0]
                
                if len(W_indices) < MIN_SAMPLES:
                    continue
                    
                Y_win = Y_chunk[W_indices, :]
                M_win = M_chunk[W_indices, :]
                T_win = frac_years_torch[W_indices]
                
                N_valid = M_win.sum(dim=0)
                has_enough_samples = N_valid >= MIN_SAMPLES
                has_enough_span = (target_time - first_valid_time) >= MIN_WINDOW_YEARS
                
                valid_pixel_mask = has_enough_samples & has_enough_span
                active_indices = torch.where(valid_pixel_mask)[0]
                
                if len(active_indices) == 0:
                    continue
                    
                P_active = len(active_indices)
                Y_active = Y_win[:, active_indices]
                M_active = M_win[:, active_indices]
                
                # 1. Batched NDFT setup
                E = torch.exp(-1j * Omega.unsqueeze(1) * T_win.unsqueeze(0)) # [K_grid, W]
                Y_active_sum = (Y_active * M_active).sum(dim=0)
                M_active_sum = M_active.sum(dim=0)
                Y_active_mean = Y_active_sum / M_active_sum
                Y_active_centered = (Y_active - Y_active_mean.unsqueeze(0)) * M_active
                
                # 1 & 2. Frequency Extraction based on configured estimator
                if FREQUENCY_ESTIMATOR == 'ALFT':
                    Y_residual = Y_active_centered.clone()
                    Omega_active_list = []
                    
                    for k in range(K_FREQUENCIES):
                        Spectrum = torch.abs(torch.matmul(E, Y_residual.to(torch.complex64))) # [K_grid, P_active]
                        top1_indices = torch.argmax(Spectrum, dim=0) # [P_active]
                        Omega_k = Omega[top1_indices] # [P_active]
                        Omega_active_list.append(Omega_k)
                        
                        if k < K_FREQUENCIES - 1:
                            Omega_so_far = torch.stack(Omega_active_list, dim=0) # [k+1, P_active]
                            angles_so_far = T_win.unsqueeze(1).unsqueeze(2) * Omega_so_far.unsqueeze(0) # [W, k+1, P_active]
                            X_cos_so_far = torch.cos(angles_so_far)
                            X_sin_so_far = torch.sin(angles_so_far)
                            X_const_so_far = torch.ones(len(T_win), 1, P_active, device=DEVICE)
                            X_active_so_far = torch.cat([X_const_so_far, X_cos_so_far, X_sin_so_far], dim=1)
                            X_active_so_far = X_active_so_far.permute(2, 0, 1)
                            
                            M_active_expanded = M_active.transpose(0, 1).unsqueeze(-1)
                            X_masked_so_far = X_active_so_far * M_active_expanded
                            
                            F_so_far = 2 * (k + 1) + 1
                            XtX_so_far = torch.bmm(X_masked_so_far.transpose(1, 2), X_masked_so_far)
                            XtX_so_far += torch.eye(F_so_far, device=DEVICE) * 1e-5
                            
                            Y_orig_expanded = Y_active_centered.transpose(0, 1).unsqueeze(-1)
                            Xty_so_far = torch.bmm(X_masked_so_far.transpose(1, 2), Y_orig_expanded * M_active_expanded)
                            
                            beta_so_far = torch.linalg.solve(XtX_so_far, Xty_so_far)
                            
                            Y_pred_so_far = torch.bmm(X_active_so_far, beta_so_far).squeeze(-1).transpose(0, 1)
                            Y_residual = (Y_active_centered - Y_pred_so_far) * M_active
                    
                    Omega_active = torch.stack(Omega_active_list, dim=0) # [K, P_active]
                
                elif FREQUENCY_ESTIMATOR == 'NDFT':
                    # NDFT extracts K frequencies dynamically from the single grid spectrum
                    Spectrum = torch.abs(torch.matmul(E, Y_active_centered.to(torch.complex64)))
                    top_amps, top_indices = torch.topk(Spectrum, K_FREQUENCIES, dim=0)
                    Omega_active = Omega[top_indices] # [K, P_active]
                    
                elif FREQUENCY_ESTIMATOR == 'NOMP':
                    # Y_active is [W, P_active]. The batched functions expect [B, T].
                    # So we pass Y_active.T and M_active.T
                    f_nomp = compute_batched_nomp(T_win, Y_active_centered.transpose(0, 1), M_active.transpose(0, 1), 
                                                min_f=NDFT_MIN_CPY, max_f=NDFT_MAX_CPY, 
                                                bins=NDFT_GRID_BINS, max_components=K_FREQUENCIES)
                    Omega_active = 2 * math.pi * f_nomp # [K, P_active]
                    
                elif FREQUENCY_ESTIMATOR in ['CBPDN', 'CIRL']:
                    # First seed with NDFT
                    f_init, max_spec, _ = batched_ndft_init(T_win, Y_active_centered.transpose(0, 1), M_active.transpose(0, 1),
                                                            min_f=NDFT_MIN_CPY, max_f=NDFT_MAX_CPY, 
                                                            bins=NDFT_GRID_BINS, max_atoms=20)
                                                            
                    if FREQUENCY_ESTIMATOR == 'CBPDN':
                        f_continuous = compute_batched_cbpdn(T_win, Y_active_centered.transpose(0, 1), M_active.transpose(0, 1),
                                                             f_init, max_spec, min_f=NDFT_MIN_CPY, max_f=NDFT_MAX_CPY,
                                                             max_atoms=20, max_components=K_FREQUENCIES)
                    else: # CIRL
                        f_continuous = compute_batched_cirl(T_win, Y_active_centered.transpose(0, 1), M_active.transpose(0, 1),
                                                            f_init, max_spec, min_f=NDFT_MIN_CPY, max_f=NDFT_MAX_CPY,
                                                            max_atoms=20, max_components=K_FREQUENCIES)
                                                            
                    Omega_active = 2 * math.pi * f_continuous # [K, P_active]
                
                # 3. Design Matrix
                T_win_expanded = T_win.unsqueeze(1).unsqueeze(2) # [W, 1, 1]
                Omega_active_expanded = Omega_active.unsqueeze(0) # [1, K, P_active]
                angles = T_win_expanded * Omega_active_expanded # [W, K, P_active]
                
                X_cos = torch.cos(angles)
                X_sin = torch.sin(angles)
                X_const = torch.ones(len(T_win), 1, P_active, device=DEVICE)
                X_active = torch.cat([X_const, X_cos, X_sin], dim=1) # [W, F, P_active]
                X_active = X_active.permute(2, 0, 1) # [P_active, W, F]
                
                M_active_expanded = M_active.transpose(0, 1).unsqueeze(-1) # [P_active, W, 1]
                X_masked = X_active * M_active_expanded
                
                F = 2 * K_FREQUENCIES + 1
                XtX = torch.bmm(X_masked.transpose(1, 2), X_masked) # [P_active, F, F]
                XtX += torch.eye(F, device=DEVICE) * 1e-5
                
                Y_active_expanded = Y_active.transpose(0, 1).unsqueeze(-1) # [P_active, W, 1]
                Xty = torch.bmm(X_masked.transpose(1, 2), Y_active_expanded * M_active_expanded)
                
                beta = torch.linalg.solve(XtX, Xty) # [P_active, F, 1]
                
                # 4. Robust Training Variance (MAD)
                Y_train_pred = torch.bmm(X_active, beta) # [P_active, W, 1]
                e = Y_active_expanded - Y_train_pred
                e_masked = e * M_active_expanded
                
                # Replace zeros in e_masked with NaN for valid median calculation
                e_valid = torch.where(M_active_expanded.bool(), e_masked, torch.tensor(float('nan'), device=DEVICE))
                
                # Compute median of valid residuals along the time window axis (dim=1)
                med_e = torch.nanmedian(e_valid, dim=1, keepdim=True).values
                
                # Compute MAD
                mad_e = torch.nanmedian(torch.abs(e_valid - med_e), dim=1, keepdim=True).values
                
                # Convert MAD to robust standard deviation (1.4826 assumes asymptotic normality of the inliers)
                # Clamp to a small positive value to prevent zero variance if all residuals perfectly match
                sigma_robust = torch.clamp(1.4826 * mad_e, min=1e-5)
                
                # Robust Variance
                RMSE_sq = sigma_robust ** 2 # [P_active, 1, 1]
                RMSE_sq = RMSE_sq.squeeze(1) # [P_active, 1]
                
                # 5. Prediction & Uncertainty Bound
                target_angles = target_time * Omega_active # [K, P_active]
                x_t_cos = torch.cos(target_angles).transpose(0, 1) # [P_active, K]
                x_t_sin = torch.sin(target_angles).transpose(0, 1)
                x_t_const = torch.ones(P_active, 1, device=DEVICE)
                x_target = torch.cat([x_t_const, x_t_cos, x_t_sin], dim=1).unsqueeze(-1) # [P_active, F, 1]
                
                y_pred = torch.bmm(x_target.transpose(1, 2), beta).squeeze(-1) # [P_active, 1]
                
                XtX_inv_x = torch.linalg.solve(XtX, x_target)
                xt_XtXinv_x = torch.bmm(x_target.transpose(1, 2), XtX_inv_x).squeeze(-1)
                S_sq = RMSE_sq * (1.0 + xt_XtXinv_x)
                S = torch.sqrt(S_sq)
                
                # 6. Anomaly Detection
                y_actual = Y_chunk[t, active_indices].unsqueeze(-1)
                M_actual = M_chunk[t, active_indices].unsqueeze(-1)
                
                error = torch.abs(y_actual - y_pred)
                is_anomaly = (error > RMSE_MULTIPLIER * S) & M_actual.bool()
                
                # Update State
                c_pred[t, active_indices] = y_pred.squeeze(-1)
                c_rmse[t, active_indices] = S.squeeze(-1)
                c_freq[t, :, active_indices] = Omega_active
                
                # Extract amplitude from beta: sqrt(cos_coeff^2 + sin_coeff^2)
                # beta shape is [P_active, F, 1]
                cos_coeffs = beta[:, 1:K_FREQUENCIES+1, 0] # [P_active, K]
                sin_coeffs = beta[:, K_FREQUENCIES+1:2*K_FREQUENCIES+1, 0] # [P_active, K]
                amplitudes = torch.sqrt(cos_coeffs**2 + sin_coeffs**2).transpose(0, 1) # [K, P_active]
                c_amp[t, :, active_indices] = amplitudes
                
                anom_mask = is_anomaly.squeeze(-1)
                active_anom_indices = active_indices[anom_mask]
                active_norm_indices = active_indices[~anom_mask]
                
                c_flags[t, active_anom_indices] = 1
                
                # Consecutive tracking
                new_anom_mask = (chunk_consec[active_anom_indices] == 0)
                new_anom_indices = active_anom_indices[new_anom_mask]
                streak_start[new_anom_indices] = t
                
                chunk_consec[active_anom_indices] += 1
                chunk_consec[active_norm_indices] = 0
                
                trigger_mask = chunk_consec[active_anom_indices] >= CONSECUTIVE_ANOMALIES
                trigger_indices = active_anom_indices[trigger_mask]
                
                if len(trigger_indices) > 0:
                    chunk_count[trigger_indices] += 1
                    
                    first_time_mask = torch.isnan(chunk_date[trigger_indices])
                    first_time_indices = trigger_indices[first_time_mask]
                    if len(first_time_indices) > 0:
                        chunk_date[first_time_indices] = acq_times_torch[streak_start[first_time_indices]]
                        
                frame_pbar.update(1)
            frame_pbar.close()
                        
            # Write chunk back to CPU output arrays
            c_pred_cpu = c_pred.cpu().numpy().reshape(num_frames, chunk_h, chunk_w)
            c_rmse_cpu = c_rmse.cpu().numpy().reshape(num_frames, chunk_h, chunk_w)
            c_freq_cpu = c_freq.cpu().numpy().reshape(num_frames, K_FREQUENCIES, chunk_h, chunk_w)
            c_amp_cpu = c_amp.cpu().numpy().reshape(num_frames, K_FREQUENCIES, chunk_h, chunk_w)
            c_flags_cpu = c_flags.cpu().numpy().reshape(num_frames, chunk_h, chunk_w)
            c_date_cpu = chunk_date.cpu().numpy().reshape(chunk_h, chunk_w)
            c_count_cpu = chunk_count.cpu().numpy().reshape(chunk_h, chunk_w)
            
            predicted_series[:, y_start:y_end, x_start:x_end] = c_pred_cpu
            rmse_series[:, y_start:y_end, x_start:x_end] = c_rmse_cpu
            dominant_frequencies_series[:, :, y_start:y_end, x_start:x_end] = c_freq_cpu
            amplitude_series[:, :, y_start:y_end, x_start:x_end] = c_amp_cpu
            anomaly_flags[:, y_start:y_end, x_start:x_end] = c_flags_cpu
            change_date_map[y_start:y_end, x_start:x_end] = c_date_cpu
            change_count_map[y_start:y_end, x_start:x_end] = c_count_cpu
            
            pbar.update(1)
            
    pbar.close()

    os.makedirs(os.path.dirname(output_h5), exist_ok=True)
    print(f"\nSaving Results to {output_h5}...")
    with h5py.File(output_h5, 'w') as out_file:
        out_file.attrs['spatial_ref'] = spatial_ref
        out_file.attrs['GeoTransform'] = geo_transform
        out_file.attrs['RMSE_MULTIPLIER'] = RMSE_MULTIPLIER
        out_file.attrs['CONSECUTIVE_ANOMALIES'] = CONSECUTIVE_ANOMALIES
        out_file.attrs['MAX_WINDOW_YEARS'] = MAX_WINDOW_YEARS
        out_file.attrs['MIN_WINDOW_YEARS'] = MIN_WINDOW_YEARS
        out_file.attrs['MIN_SAMPLES'] = MIN_SAMPLES
        out_file.attrs['K_FREQUENCIES'] = K_FREQUENCIES
        out_file.attrs['TARGET_METRIC'] = TARGET_METRIC
        out_file.attrs['IGNORE_COMMON_MASK'] = IGNORE_COMMON_MASK
        out_file.attrs['SOURCE_DATA'] = H5_PATH
        
        out_file.create_dataset('predicted_series', data=predicted_series, compression='gzip')
        out_file.create_dataset('rmse_series', data=rmse_series, compression='gzip')
        out_file.create_dataset('dominant_frequencies_series', data=dominant_frequencies_series, compression='gzip')
        out_file.create_dataset('amplitude_series', data=amplitude_series, compression='gzip')
        out_file.create_dataset('anomaly_flags', data=anomaly_flags, compression='gzip')
        out_file.create_dataset('change_date_timestamp', data=change_date_map, compression='gzip')
        out_file.create_dataset('change_count', data=change_count_map, compression='gzip')
        
    print("DHR Pipeline Complete!")

if __name__ == "__main__":
    main()
