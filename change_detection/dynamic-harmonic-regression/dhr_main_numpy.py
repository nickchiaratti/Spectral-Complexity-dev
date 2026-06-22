import os
import h5py
import numpy as np
import datetime
import math
from tqdm import tqdm

# ==========================================
# 1. CONFIGURATION
# ==========================================
LOCATION = "Tait"
H5_PATH = f"C:/satelliteImagery/HLST30/HLST_{LOCATION}_Harmonized_SC_EM-7_Norm-bandCount.h5"

TARGET_METRIC = 'sliding_volume_z_score'
IGNORE_COMMON_MASK = True # If True, utilizes noisy/cloudy pixels and relies on NDFT to filter noise
RMSE_MULTIPLIER = 2
CONSECUTIVE_ANOMALIES = 4
MAX_WINDOW_YEARS = 5.0
MIN_WINDOW_YEARS = 1.0
K_FREQUENCIES = 2
MIN_SAMPLES = 1 * K_FREQUENCIES + 1 + 3 # 8 parameters + 3 df
CHUNK_SIZE = 128 # Spatial block size

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
    output_h5 = f"C:/satelliteImagery/HLST30/DHR/{LOCATION}_DHR_Change_Detection_{_term_str}_{_win_str}{_mask_str}_numpy.h5"

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

    print(f"Dataset shape: {num_frames} frames, {height}x{width} pixels")
    print("Using NumPy vectorization on CPU (PyTorch dependency removed)")

    # Output arrays
    change_date_map = np.full((height, width), np.nan, dtype=np.float64)
    change_count_map = np.zeros((height, width), dtype=np.int32)
    predicted_series = np.full((num_frames, height, width), np.nan, dtype=np.float32)
    rmse_series = np.full((num_frames, height, width), np.nan, dtype=np.float32)
    dominant_frequencies_series = np.full((num_frames, K_FREQUENCIES, height, width), np.nan, dtype=np.float32)
    amplitude_series = np.full((num_frames, K_FREQUENCIES, height, width), np.nan, dtype=np.float32)
    anomaly_flags = np.zeros((num_frames, height, width), dtype=np.uint8)

    y_data_np = np.nan_to_num(y_data.astype(np.float32), nan=0.0)
    valid_mask_np = valid_mask.astype(bool)
    frac_years_np = frac_years.astype(np.float32)
    acq_times_np = acq_times.astype(np.float64)

    # Frequency Grid for NDFT
    f_grid = np.linspace(0.2, 4.0, 150)
    Omega = 2.0 * math.pi * f_grid
    
    print("\nExecuting Batched Dynamic Harmonic Regression...")
    
    y_chunks = list(range(0, height, CHUNK_SIZE))
    x_chunks = list(range(0, width, CHUNK_SIZE))
    total_chunks = len(y_chunks) * len(x_chunks)
    
    pbar = tqdm(total=total_chunks, desc="Spatial Chunks")
    
    for y_start in y_chunks:
        y_end = min(y_start + CHUNK_SIZE, height)
        for x_start in x_chunks:
            x_end = min(x_start + CHUNK_SIZE, width)
            
            chunk_h = y_end - y_start
            chunk_w = x_end - x_start
            P = chunk_h * chunk_w
            
            Y_chunk = y_data_np[:, y_start:y_end, x_start:x_end].reshape(num_frames, P)
            M_chunk = valid_mask_np[:, y_start:y_end, x_start:x_end].reshape(num_frames, P)
            
            # Precompute first valid time for each pixel
            any_valid = M_chunk.any(axis=0)
            first_valid_idx = M_chunk.astype(np.int8).argmax(axis=0)
            first_valid_time = frac_years_np[first_valid_idx].copy()
            first_valid_time[~any_valid] = np.inf
            
            # State tracking arrays for this chunk
            chunk_consec = np.zeros(P, dtype=np.int32)
            streak_start = np.zeros(P, dtype=np.int32)
            chunk_count = np.zeros(P, dtype=np.int32)
            chunk_date = np.full((P,), np.nan, dtype=np.float64)
            
            c_pred = np.full((num_frames, P), np.nan, dtype=np.float32)
            c_rmse = np.full((num_frames, P), np.nan, dtype=np.float32)
            c_freq = np.full((num_frames, K_FREQUENCIES, P), np.nan, dtype=np.float32)
            c_amp = np.full((num_frames, K_FREQUENCIES, P), np.nan, dtype=np.float32)
            c_flags = np.zeros((num_frames, P), dtype=np.uint8)

            for t in range(num_frames):
                target_time = frac_years_np[t]
                
                window_start = target_time - MAX_WINDOW_YEARS
                in_window = (frac_years_np >= window_start) & (frac_years_np < target_time)
                W_indices = np.where(in_window)[0]
                
                if len(W_indices) < MIN_SAMPLES:
                    continue
                    
                Y_win = Y_chunk[W_indices, :]
                M_win = M_chunk[W_indices, :]
                T_win = frac_years_np[W_indices]
                
                N_valid = M_win.sum(axis=0)
                has_enough_samples = N_valid >= MIN_SAMPLES
                has_enough_span = (target_time - first_valid_time) >= MIN_WINDOW_YEARS
                
                valid_pixel_mask = has_enough_samples & has_enough_span
                active_indices = np.where(valid_pixel_mask)[0]
                
                if len(active_indices) == 0:
                    continue
                    
                P_active = len(active_indices)
                Y_active = Y_win[:, active_indices]
                M_active = M_win[:, active_indices]
                
                # 1. Batched NDFT setup
                E = np.exp(-1j * Omega[:, None] * T_win[None, :]) # [K_grid, W]
                Y_active_sum = (Y_active * M_active).sum(axis=0)
                M_active_sum = M_active.sum(axis=0)
                
                # Prevent divide by zero issues gracefully
                valid_means = M_active_sum > 0
                Y_active_mean = np.zeros_like(Y_active_sum)
                Y_active_mean[valid_means] = Y_active_sum[valid_means] / M_active_sum[valid_means]
                
                Y_active_centered = (Y_active - Y_active_mean[None, :]) * M_active
                
                # 1 & 2. Iterative Frequency Extraction (ALFT/OMP)
                Y_residual = Y_active_centered.copy()
                Omega_active_list = []
                
                for k in range(K_FREQUENCIES):
                    Spectrum = np.abs(np.matmul(E, Y_residual.astype(np.complex64))) # [K_grid, P_active]
                    top1_indices = np.argmax(Spectrum, axis=0) # [P_active]
                    Omega_k = Omega[top1_indices] # [P_active]
                    Omega_active_list.append(Omega_k)
                    
                    if k < K_FREQUENCIES - 1:
                        Omega_so_far = np.stack(Omega_active_list, axis=0) # [k+1, P_active]
                        angles_so_far = T_win[:, None, None] * Omega_so_far[None, :, :] # [W, k+1, P_active]
                        
                        X_cos_so_far = np.cos(angles_so_far)
                        X_sin_so_far = np.sin(angles_so_far)
                        X_const_so_far = np.ones((len(T_win), 1, P_active), dtype=np.float32)
                        X_active_so_far = np.concatenate([X_const_so_far, X_cos_so_far, X_sin_so_far], axis=1) # [W, F_so_far, P_active]
                        X_active_so_far = np.transpose(X_active_so_far, (2, 0, 1)) # [P_active, W, F_so_far]
                        
                        M_active_expanded = np.transpose(M_active, (1, 0))[:, :, None] # [P_active, W, 1]
                        X_masked_so_far = X_active_so_far * M_active_expanded
                        
                        F_so_far = 2 * (k + 1) + 1
                        XtX_so_far = np.matmul(np.transpose(X_masked_so_far, (0, 2, 1)), X_masked_so_far)
                        XtX_so_far += np.eye(F_so_far) * 1e-5
                        
                        Y_orig_expanded = np.transpose(Y_active_centered, (1, 0))[:, :, None]
                        Xty_so_far = np.matmul(np.transpose(X_masked_so_far, (0, 2, 1)), Y_orig_expanded * M_active_expanded)
                        
                        beta_so_far = np.linalg.solve(XtX_so_far, Xty_so_far)
                        
                        Y_pred_so_far = np.matmul(X_active_so_far, beta_so_far).squeeze(-1).T # [W, P_active]
                        Y_residual = (Y_active_centered - Y_pred_so_far) * M_active
                
                Omega_active = np.stack(Omega_active_list, axis=0) # [K, P_active]
                
                # 3. Design Matrix
                T_win_expanded = T_win[:, None, None] # [W, 1, 1]
                Omega_active_expanded = Omega_active[None, :, :] # [1, K, P_active]
                angles = T_win_expanded * Omega_active_expanded # [W, K, P_active]
                
                X_cos = np.cos(angles)
                X_sin = np.sin(angles)
                X_const = np.ones((len(T_win), 1, P_active), dtype=np.float32)
                X_active = np.concatenate([X_const, X_cos, X_sin], axis=1) # [W, F, P_active]
                X_active = np.transpose(X_active, (2, 0, 1)) # [P_active, W, F]
                
                M_active_expanded = np.transpose(M_active, (1, 0))[:, :, None] # [P_active, W, 1]
                X_masked = X_active * M_active_expanded
                
                F = 2 * K_FREQUENCIES + 1
                XtX = np.matmul(np.transpose(X_masked, (0, 2, 1)), X_masked) # [P_active, F, F]
                XtX += np.eye(F) * 1e-5
                
                Y_active_expanded = np.transpose(Y_active, (1, 0))[:, :, None] # [P_active, W, 1]
                Xty = np.matmul(np.transpose(X_masked, (0, 2, 1)), Y_active_expanded * M_active_expanded)
                
                beta = np.linalg.solve(XtX, Xty) # [P_active, F, 1]
                
                # 4. Robust Training Variance (MAD)
                Y_train_pred = np.matmul(X_active, beta) # [P_active, W, 1]
                e = Y_active_expanded - Y_train_pred
                e_masked = e * M_active_expanded
                
                # Replace zeros with NaN for valid median calculation
                e_valid = np.where(M_active_expanded.astype(bool), e_masked, np.nan)
                
                # Compute median of valid residuals along the time window axis (axis=1)
                med_e = np.nanmedian(e_valid, axis=1, keepdims=True)
                
                # Compute MAD
                mad_e = np.nanmedian(np.abs(e_valid - med_e), axis=1, keepdims=True)
                
                # Convert MAD to robust standard deviation
                sigma_robust = np.clip(1.4826 * mad_e, a_min=1e-5, a_max=None)
                
                # Robust Variance
                RMSE_sq = sigma_robust ** 2 # [P_active, 1, 1]
                RMSE_sq = np.squeeze(RMSE_sq, axis=1) # [P_active, 1]
                
                # 5. Prediction & Uncertainty Bound
                target_angles = target_time * Omega_active # [K, P_active]
                x_t_cos = np.transpose(np.cos(target_angles), (1, 0)) # [P_active, K]
                x_t_sin = np.transpose(np.sin(target_angles), (1, 0))
                x_t_const = np.ones((P_active, 1), dtype=np.float32)
                x_target = np.concatenate([x_t_const, x_t_cos, x_t_sin], axis=1)[:, :, None] # [P_active, F, 1]
                
                y_pred = np.matmul(np.transpose(x_target, (0, 2, 1)), beta).squeeze(-1) # [P_active, 1]
                
                XtX_inv_x = np.linalg.solve(XtX, x_target)
                xt_XtXinv_x = np.matmul(np.transpose(x_target, (0, 2, 1)), XtX_inv_x).squeeze(-1)
                S_sq = RMSE_sq * (1.0 + xt_XtXinv_x)
                S = np.sqrt(S_sq)
                
                # 6. Anomaly Detection
                y_actual = Y_chunk[t, active_indices][:, None]
                M_actual = M_chunk[t, active_indices][:, None]
                
                error = np.abs(y_actual - y_pred)
                is_anomaly = (error > RMSE_MULTIPLIER * S) & M_actual.astype(bool)
                
                # Update State
                c_pred[t, active_indices] = y_pred.squeeze(-1)
                c_rmse[t, active_indices] = S.squeeze(-1)
                
                # NumPy advanced indexing returns (P_active, K) when mixing integer and array indices separated by slice
                c_freq[t, :, active_indices] = Omega_active.T
                
                cos_coeffs = beta[:, 1:K_FREQUENCIES+1, 0] # [P_active, K]
                sin_coeffs = beta[:, K_FREQUENCIES+1:2*K_FREQUENCIES+1, 0] # [P_active, K]
                amplitudes = np.transpose(np.sqrt(cos_coeffs**2 + sin_coeffs**2), (1, 0)) # [K, P_active]
                c_amp[t, :, active_indices] = amplitudes.T
                
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
                    
                    first_time_mask = np.isnan(chunk_date[trigger_indices])
                    first_time_indices = trigger_indices[first_time_mask]
                    if len(first_time_indices) > 0:
                        chunk_date[first_time_indices] = acq_times_np[streak_start[first_time_indices]]
                        
            # Write chunk back to CPU output arrays
            c_pred_cpu = c_pred.reshape(num_frames, chunk_h, chunk_w)
            c_rmse_cpu = c_rmse.reshape(num_frames, chunk_h, chunk_w)
            c_freq_cpu = c_freq.reshape(num_frames, K_FREQUENCIES, chunk_h, chunk_w)
            c_amp_cpu = c_amp.reshape(num_frames, K_FREQUENCIES, chunk_h, chunk_w)
            c_flags_cpu = c_flags.reshape(num_frames, chunk_h, chunk_w)
            c_date_cpu = chunk_date.reshape(chunk_h, chunk_w)
            c_count_cpu = chunk_count.reshape(chunk_h, chunk_w)
            
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
