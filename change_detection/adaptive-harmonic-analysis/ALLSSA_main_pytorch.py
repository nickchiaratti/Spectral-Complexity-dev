import os
import h5py
import numpy as np
import datetime
import math
from tqdm import tqdm
import torch
import psutil

# ==========================================
# 1. CONFIGURATION
# ==========================================
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--location", type=str, default="Tait", help="Location name")
args = parser.parse_args()

LOCATION = args.location
H5_PATH = f"C:/satelliteImagery/HLST30/HLST_{LOCATION}_Harmonized_SC_EM-7_Norm-None.h5"

START_DATE = "2015-01-01"
END_DATE = "2026-06-01"

TARGET_METRIC = 'sliding_volume_z_score'
IGNORE_COMMON_MASK = False # If True, utilizes noisy/cloudy pixels and relies on NDFT to filter noise
RMSE_MULTIPLIER = 2.576
CONSECUTIVE_ANOMALIES = 4
MAX_WINDOW_YEARS = 3
MIN_WINDOW_YEARS = 3
K_FREQUENCIES = 2
MIN_SAMPLES = 2 * K_FREQUENCIES + 1 + 3 # 8 parameters + 3 df
CHUNK_SIZE = None # Spatial block size, dynamically computed based on VRAM/RAM
NDFT_MIN_CPY = 0.2
NDFT_MAX_CPY = 4.0
# Grid bins computed from Rayleigh criterion: df = max(1/(4*T_window), 0.02)
OVERSAMPLING_FACTOR = 4
FREQ_RESOLUTION_FLOOR = 0.02  # CPY
_df_step = max(1.0 / (OVERSAMPLING_FACTOR * MAX_WINDOW_YEARS), FREQ_RESOLUTION_FLOOR)
NDFT_GRID_BINS = int((NDFT_MAX_CPY - NDFT_MIN_CPY) / _df_step) + 1
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
def main():
    _term_str = f"K{K_FREQUENCIES}"
    _win_str = f"W{int(MAX_WINDOW_YEARS)}"
    _mask_str = "_unmasked" if IGNORE_COMMON_MASK else ""
    output_h5 = f"C:/satelliteImagery/HLST30/ALLSSA/{LOCATION}_ALLSSA_Change_Detection_{START_DATE}_{END_DATE}_{_term_str}_{_win_str}{_mask_str}.h5"

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
    amplitude_series = np.zeros((num_frames, K_FREQUENCIES, height, width), dtype=np.float32)
    anomaly_flags = np.zeros((num_frames, height, width), dtype=np.uint8)
    confirmed_flags = np.zeros((num_frames, height, width), dtype=np.uint8)

    y_data_torch = torch.from_numpy(y_data).float()
    y_data_torch = torch.nan_to_num(y_data_torch, nan=0.0)
    valid_mask_torch = torch.from_numpy(valid_mask).bool()
    frac_years_torch = torch.from_numpy(frac_years).float().to(DEVICE)
    acq_times_torch = torch.from_numpy(acq_times).double().to(DEVICE)

    # Frequency Grid (Rayleigh-criterion sized)
    f_grid = torch.linspace(NDFT_MIN_CPY, NDFT_MAX_CPY, NDFT_GRID_BINS, device=DEVICE)
    Omega = 2.0 * math.pi * f_grid
    G = len(f_grid)
    print(f"Frequency grid: {NDFT_GRID_BINS} bins, df={_df_step:.4f} CPY")
    
    print("\nExecuting Batched Adaptive Harmonic Analysis...")
    
    global CHUNK_SIZE
    if CHUNK_SIZE is None:
        def calculate_dynamic_chunk_size(n_frames, device_type, w, h):
            # Heuristic memory usage per pixel (bytes). Accounts for model weights, gradients, states
            bytes_per_pixel = n_frames * 400
            
            if device_type.type == 'cuda':
                free_mem, _ = torch.cuda.mem_get_info(device_type)
                target_mem = free_mem * 0.8
                print(f"GPU VRAM: {free_mem / 1e9:.2f} GB free, targeting {target_mem / 1e9:.2f} GB (80%)")
            else:
                if psutil is not None:
                    vm = psutil.virtual_memory()
                    target_mem = vm.available * 0.8
                    print(f"CPU RAM: {vm.available / 1e9:.2f} GB available, targeting {target_mem / 1e9:.2f} GB (80%)")
                else:
                    target_mem = 8 * 1024**3
                    print(f"psutil not found, using fallback CPU memory target: {target_mem / 1e9:.2f} GB")
                    
            max_pixels = max(1, int(target_mem / bytes_per_pixel))
            c_size = int(math.sqrt(max_pixels))
            c_size = min(c_size, max(w, h))
            c_size = max(16, c_size) # floor size at 16x16
            
            print(f"Estimated dynamic memory per pixel: {bytes_per_pixel} bytes")
            print(f"Dynamically calculated CHUNK_SIZE: {c_size}")
            return c_size

        CHUNK_SIZE = calculate_dynamic_chunk_size(num_frames, DEVICE, width, height)
    
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
            c_confirmed = torch.zeros((num_frames, P), dtype=torch.uint8, device=DEVICE)

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
                
                # 1. Masked mean centering
                Y_active_sum = (Y_active * M_active).sum(dim=0)
                M_active_sum = M_active.sum(dim=0)
                Y_active_mean = Y_active_sum / M_active_sum
                Y_active_centered = (Y_active - Y_active_mean.unsqueeze(0)) * M_active
                
                W = len(T_win)
                
                # 1 & 2. Frequency Extraction via True LSSA
                # True LSSA-based iterative frequency search (Ghaderpour 2018)
                # Evaluates the normalized least-squares spectrum at each grid 
                # frequency, accounting for non-orthogonality via the Schur complement.
                
                # Precompute test basis pairs for all grid frequencies: [G, 2, W]
                angles_grid = 2.0 * math.pi * f_grid.unsqueeze(1) * T_win.unsqueeze(0)  # [G, W]
                Phi_cos = torch.cos(angles_grid)  # [G, W]
                Phi_sin = torch.sin(angles_grid)  # [G, W]
                Phi = torch.stack([Phi_cos, Phi_sin], dim=1)  # [G, 2, W]
                
                # N22 = Phi @ Phi^T: [G, 2, 2] (shared across pixels)
                N22 = torch.bmm(Phi, Phi.transpose(1, 2))
                
                Omega_active_list = []
                
                for k in range(K_FREQUENCIES):
                    # Build per-pixel OLS design matrix D
                    # D has columns: [1, cos(2pi*f1*t), sin(2pi*f1*t), ...]
                    # Each pixel has its own extracted frequencies, so D is pixel-dependent.
                    nc = 1 + 2 * k
                    
                    # For batched processing: each pixel may have different known freqs.
                    # Build D per pixel: [P_active, W, nc]
                    D_batch = torch.zeros(P_active, W, nc, device=DEVICE)
                    D_batch[:, :, 0] = M_active.transpose(0, 1)  # masked constant
                    for i in range(k):
                        omega_i = Omega_active_list[i]  # [P_active]
                        angles_i = T_win.unsqueeze(0) * omega_i.unsqueeze(1)  # [P_active, W]
                        D_batch[:, :, 1 + 2*i]     = torch.cos(angles_i) * M_active.transpose(0, 1)
                        D_batch[:, :, 1 + 2*i + 1] = torch.sin(angles_i) * M_active.transpose(0, 1)
                    
                    # OLS fit: coeff = (D^T D)^-1 D^T f
                    DtD = torch.bmm(D_batch.transpose(1, 2), D_batch)  # [P_active, nc, nc]
                    DtD += torch.eye(nc, device=DEVICE) * 1e-8  # regularization
                    
                    Y_batch = (Y_active * M_active).transpose(0, 1).unsqueeze(-1)  # [P_active, W, 1]
                    Dty = torch.bmm(D_batch.transpose(1, 2), Y_batch)  # [P_active, nc, 1]
                    
                    coeff_batch = torch.linalg.solve(DtD, Dty)  # [P_active, nc, 1]
                    Ninv_batch = torch.linalg.inv(DtD)  # [P_active, nc, nc]
                    
                    # Residual per pixel
                    D_unmasked = torch.zeros(P_active, W, nc, device=DEVICE)
                    D_unmasked[:, :, 0] = 1.0
                    for i in range(k):
                        omega_i = Omega_active_list[i]
                        angles_i = T_win.unsqueeze(0) * omega_i.unsqueeze(1)
                        D_unmasked[:, :, 1 + 2*i]     = torch.cos(angles_i)
                        D_unmasked[:, :, 1 + 2*i + 1] = torch.sin(angles_i)
                    
                    Y_pred = torch.bmm(D_unmasked, coeff_batch)  # [P_active, W, 1]
                    r = Y_batch - Y_pred  # [P_active, W, 1]
                    r = r * M_active.transpose(0, 1).unsqueeze(-1)  # mask residual
                    norm_r_sq = (r ** 2).sum(dim=1).squeeze(-1)  # [P_active]
                    r_flat = r.squeeze(-1)  # [P_active, W]
                    
                    # Compute LSS at every grid frequency for every pixel
                    # Phi: [G, 2, W], r_flat: [P_active, W]
                    # Phi @ r: [P_active, G, 2]
                    Phi_r = torch.einsum('gkw,pw->pgk', Phi, r_flat)  # [P_active, G, 2]
                    
                    # N12 = D^T @ Phi^T: [P_active, G, nc, 2]
                    # D_batch: [P_active, W, nc], Phi: [G, 2, W]
                    # N12[p,g] = D_batch[p]^T @ Phi[g]^T
                    N12 = torch.einsum('pwn,gkw->pgnk', D_batch, Phi)  # [P_active, G, nc, 2]
                    
                    # Correction: N12^T @ Ninv @ N12: [P_active, G, 2, 2]
                    # Ninv_batch: [P_active, nc, nc]
                    Ninv_N12 = torch.einsum('pmn,pgnk->pgmk', Ninv_batch, N12)  # [P_active, G, nc, 2]
                    correction = torch.einsum('pgnk,pgnj->pgkj', N12, Ninv_N12)  # [P_active, G, 2, 2]
                    
                    # Schur complement: [P_active, G, 2, 2]
                    Schur = N22.unsqueeze(0) - correction  # broadcast N22 over P_active
                    
                    # Analytic 2x2 inverse
                    a = Schur[:, :, 0, 0]
                    b = Schur[:, :, 0, 1]
                    c = Schur[:, :, 1, 0]
                    d = Schur[:, :, 1, 1]
                    det = a * d - b * c
                    det = det.clamp(min=1e-12)  # prevent division by zero
                    
                    Schur_inv = torch.zeros_like(Schur)
                    Schur_inv[:, :, 0, 0] = d / det
                    Schur_inv[:, :, 0, 1] = -b / det
                    Schur_inv[:, :, 1, 0] = -c / det
                    Schur_inv[:, :, 1, 1] = a / det
                    
                    # c_hat = Schur_inv @ Phi_r: [P_active, G, 2]
                    c_hat = torch.einsum('pgkj,pgj->pgk', Schur_inv, Phi_r)
                    
                    # LSS = (Phi_r . c_hat) / norm_r_sq: [P_active, G]
                    LSS = (Phi_r * c_hat).sum(dim=-1) / norm_r_sq.unsqueeze(-1).clamp(min=1e-12)
                    LSS = LSS.clamp(0.0, 1.0)
                    
                    # Mask out already-known frequencies
                    for omega_prev in Omega_active_list:
                        # omega_prev is [P_active] in angular freq
                        f_prev = omega_prev / (2.0 * math.pi)  # [P_active]
                        # For each pixel, mask the grid bin closest to its known freq
                        dist = (f_grid.unsqueeze(0) - f_prev.unsqueeze(1)).abs()  # [P_active, G]
                        near_mask = dist < (_df_step * 0.5)
                        LSS[near_mask] = 0.0
                    
                    # Select peak frequency per pixel
                    best_indices = torch.argmax(LSS, dim=1)  # [P_active]
                    Omega_k = Omega[best_indices]  # [P_active]
                    Omega_active_list.append(Omega_k)
                
                Omega_active = torch.stack(Omega_active_list, dim=0)  # [K, P_active]
                
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
                valid_mask = M_actual.bool().squeeze(-1)
                active_anom_indices = active_indices[anom_mask]
                active_norm_indices = active_indices[(~anom_mask) & valid_mask]
                
                c_flags[t, active_anom_indices] = 1
                
                # Consecutive tracking (ignores clouded pixels)
                new_anom_mask = (chunk_consec[active_anom_indices] == 0)
                new_anom_indices = active_anom_indices[new_anom_mask]
                streak_start[new_anom_indices] = t
                
                chunk_consec[active_anom_indices] += 1
                chunk_consec[active_norm_indices] = 0
                
                trigger_mask = chunk_consec[active_anom_indices] >= CONSECUTIVE_ANOMALIES
                trigger_indices = active_anom_indices[trigger_mask]
                
                if len(trigger_indices) > 0:
                    c_confirmed[t, trigger_indices] = 1
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
            c_confirmed_cpu = c_confirmed.cpu().numpy().reshape(num_frames, chunk_h, chunk_w)
            c_date_cpu = chunk_date.cpu().numpy().reshape(chunk_h, chunk_w)
            c_count_cpu = chunk_count.cpu().numpy().reshape(chunk_h, chunk_w)
            
            predicted_series[:, y_start:y_end, x_start:x_end] = c_pred_cpu
            rmse_series[:, y_start:y_end, x_start:x_end] = c_rmse_cpu
            dominant_frequencies_series[:, :, y_start:y_end, x_start:x_end] = c_freq_cpu
            amplitude_series[:, :, y_start:y_end, x_start:x_end] = c_amp_cpu
            anomaly_flags[:, y_start:y_end, x_start:x_end] = c_flags_cpu
            confirmed_flags[:, y_start:y_end, x_start:x_end] = c_confirmed_cpu
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
        out_file.create_dataset('confirmed_change_flags', data=confirmed_flags, compression='gzip')
        out_file.create_dataset('change_date_timestamp', data=change_date_map, compression='gzip')
        out_file.create_dataset('change_count', data=change_count_map, compression='gzip')
        
    print("ALLSSA Pipeline Complete!")

if __name__ == "__main__":
    main()
