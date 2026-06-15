import os
import h5py
import numpy as np
import datetime
import math
from tqdm import tqdm
import multiprocessing
import joblib
import contextlib
from joblib import Parallel, delayed

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar."""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()

# ==========================================
# 1. CONFIGURATION
# ==========================================
LOCATION = "Tait"
H5_PATH = f"C:/satelliteImagery/HLST30/HLST_{LOCATION}_Harmonized_SC_EM-7_Norm-bandCount.h5"
OUTPUT_H5 = f"C:/satelliteImagery/HLST30/CCD/{LOCATION}_CCD_Harmonized_Change_Detection.h5"

TARGET_METRIC = 'sliding_volume_z_score'
RMSE_MULTIPLIER = 3.0
CONSECUTIVE_ANOMALIES = 4
TIME_WINDOW_YEARS = 3.0
ENABLE_ELASTIC_WINDOW = True  # Allows window to expand backwards to meet MIN_SAMPLES
MAX_ELASTIC_WINDOW_YEARS = TIME_WINDOW_YEARS + 2.0  # Maximum span to expand backwards
TEMPORAL_PERIODS = [2/3, 1/2, 1.0, 3.0]
MIN_SAMPLES = len(TEMPORAL_PERIODS) * 2 + 1 # Minimum required to solve OLS without being underdetermined

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

def build_harmonic_matrix(t, temporal_periods=TEMPORAL_PERIODS):
    """
    Constructs a Fourier basis matrix using specific temporal periods (no linear trend).
    """
    w = 2.0 * math.pi
    cols = [
        np.ones_like(t),  # Intercept
    ]
    for p in temporal_periods:
        cols.append(np.cos((w / p) * t))
        cols.append(np.sin((w / p) * t))
    return np.column_stack(cols)

def _process_row_chunk(chunk_args):
    y_start, y_end, width, y_data, valid_mask, frac_years, X_full, acq_times, min_samples, time_window_years, enable_elastic, max_elastic_years, rmse_mult, consec_anom = chunk_args
    
    num_frames = y_data.shape[0]
    chunk_height = y_end - y_start
    
    chunk_pred = np.full((num_frames, chunk_height, width), np.nan, dtype=np.float32)
    chunk_rmse = np.full((num_frames, chunk_height, width), np.nan, dtype=np.float32)
    chunk_flags = np.zeros((num_frames, chunk_height, width), dtype=np.uint8)
    chunk_date = np.full((chunk_height, width), np.nan, dtype=np.float64)
    chunk_count = np.zeros((chunk_height, width), dtype=np.int32)
    
    for y_local in range(chunk_height):
        y_global = y_start + y_local
        for x in range(width):
            pixel_valid = valid_mask[:, y_global, x]
            valid_indices = np.where(pixel_valid)[0]
            
            if len(valid_indices) <= min_samples:
                continue

            consecutive_count = 0
            first_valid_time = frac_years[valid_indices[0]]
            
            for i in range(min_samples, len(valid_indices)):
                target_idx = valid_indices[i]
                target_time = frac_years[target_idx]
                
                if target_time - first_valid_time < time_window_years:
                    continue
                
                window_start_time = target_time - time_window_years
                past_valid = valid_indices[:i]
                past_times = frac_years[past_valid]
                in_window_mask = past_times >= window_start_time
                
                if np.sum(in_window_mask) < min_samples:
                    if enable_elastic:
                        train_idx = past_valid[-min_samples:]
                        if target_time - frac_years[train_idx[0]] > max_elastic_years:
                            continue
                    else:
                        continue
                else:
                    train_idx = past_valid[in_window_mask]
                
                X_train = X_full[train_idx, :]
                Y_train = y_data[train_idx, y_global, x]
                
                # Fit model on window
                coeffs, residuals, rank, s = np.linalg.lstsq(X_train, Y_train, rcond=None)
                
                # Compute RMSE of the training window
                if len(residuals) > 0:
                    rmse = np.sqrt(residuals[0] / len(Y_train))
                else:
                    y_train_pred = X_train @ coeffs
                    rmse = np.sqrt(np.mean((Y_train - y_train_pred)**2))
                
                rmse = max(rmse, 1e-5)
                
                # Predict target step
                X_target = X_full[target_idx, :]
                y_pred = X_target @ coeffs
                
                chunk_pred[target_idx, y_local, x] = y_pred
                chunk_rmse[target_idx, y_local, x] = rmse
                
                # Check anomaly condition
                actual = y_data[target_idx, y_global, x]
                error = abs(actual - y_pred)
                
                is_anomaly = error > (rmse_mult * rmse)
                if is_anomaly:
                    chunk_flags[target_idx, y_local, x] = 1
                    consecutive_count += 1
                    
                    if consecutive_count >= consec_anom:
                        chunk_count[y_local, x] += 1
                        # If this is the first confirmed structural change, record the date
                        if np.isnan(chunk_date[y_local, x]):
                            first_anomaly_idx = valid_indices[i - consec_anom + 1]
                            chunk_date[y_local, x] = acq_times[first_anomaly_idx]
                else:
                    consecutive_count = 0

    return y_start, y_end, chunk_pred, chunk_rmse, chunk_flags, chunk_date, chunk_count

def main():
    print(f"Loading data from {H5_PATH}...")
    with h5py.File(H5_PATH, 'r') as f:
        data_grp = f['/HDFEOS/GRIDS/HARMONIZED/Data Fields']
        metric_ds = data_grp[TARGET_METRIC]
        
        acq_times = metric_ds.attrs['acquisition_time'][:]
        y_data = metric_ds[...]
        
        # Determine valid mask
        common_mask = data_grp['common_mask'][...]
        valid_mask = (common_mask == 0) & ~np.isnan(y_data)
        
        geo_transform = metric_ds.attrs.get('GeoTransform')
        spatial_ref = metric_ds.attrs.get('spatial_ref')
        
    num_frames, height, width = y_data.shape
    frac_years = extract_fractional_years(acq_times)
    
    # Sort chronologically just in case
    sort_idx = np.argsort(acq_times)
    acq_times = acq_times[sort_idx]
    frac_years = frac_years[sort_idx]
    y_data = y_data[sort_idx, ...]
    valid_mask = valid_mask[sort_idx, ...]

    print(f"Dataset shape: {num_frames} frames, {height}x{width} pixels")

    # Output arrays
    change_date_map = np.zeros((height, width), dtype=np.float64)
    change_date_map[:] = np.nan
    change_count_map = np.zeros((height, width), dtype=np.int32)
    
    predicted_series = np.full((num_frames, height, width), np.nan, dtype=np.float32)
    rmse_series = np.full((num_frames, height, width), np.nan, dtype=np.float32)
    anomaly_flags = np.zeros((num_frames, height, width), dtype=np.uint8)

    X_full = build_harmonic_matrix(frac_years)

    print("\nExecuting Sliding Window OLS Harmonic Regression...")
    
    n_jobs = multiprocessing.cpu_count()
    print(f"Using {n_jobs} cores for parallel processing.")
    
    # Divide the workload into roughly n_jobs * 4 chunks for load balancing
    num_chunks = max(1, n_jobs * 4) 
    chunk_size = max(1, math.ceil(height / num_chunks))
    
    chunks = []
    for y_start in range(0, height, chunk_size):
        y_end = min(y_start + chunk_size, height)
        chunk_args = (
            y_start, y_end, width, 
            y_data, valid_mask, 
            frac_years, X_full, acq_times, 
            MIN_SAMPLES, TIME_WINDOW_YEARS, 
            ENABLE_ELASTIC_WINDOW, MAX_ELASTIC_WINDOW_YEARS, 
            RMSE_MULTIPLIER, CONSECUTIVE_ANOMALIES
        )
        chunks.append(chunk_args)
        
    with tqdm_joblib(tqdm(desc="Processing row chunks", total=len(chunks))):
        results = Parallel(n_jobs=n_jobs, backend='loky')(
            delayed(_process_row_chunk)(chunk) for chunk in chunks
        )
    
    for y_start, y_end, c_pred, c_rmse, c_flags, c_date, c_count in results:
        predicted_series[:, y_start:y_end, :] = c_pred
        rmse_series[:, y_start:y_end, :] = c_rmse
        anomaly_flags[:, y_start:y_end, :] = c_flags
        change_date_map[y_start:y_end, :] = c_date
        change_count_map[y_start:y_end, :] = c_count

    os.makedirs(os.path.dirname(OUTPUT_H5), exist_ok=True)
    print(f"\nSaving Results to {OUTPUT_H5}...")
    with h5py.File(OUTPUT_H5, 'w') as out_file:
        out_file.attrs['spatial_ref'] = spatial_ref
        out_file.attrs['GeoTransform'] = geo_transform
        out_file.attrs['RMSE_MULTIPLIER'] = RMSE_MULTIPLIER
        out_file.attrs['CONSECUTIVE_ANOMALIES'] = CONSECUTIVE_ANOMALIES
        out_file.attrs['TIME_WINDOW_YEARS'] = TIME_WINDOW_YEARS
        out_file.attrs['ENABLE_ELASTIC_WINDOW'] = ENABLE_ELASTIC_WINDOW
        out_file.attrs['MAX_ELASTIC_WINDOW_YEARS'] = MAX_ELASTIC_WINDOW_YEARS
        out_file.attrs['MIN_SAMPLES'] = MIN_SAMPLES
        out_file.attrs['TEMPORAL_PERIODS'] = TEMPORAL_PERIODS
        out_file.attrs['TARGET_METRIC'] = TARGET_METRIC
        out_file.attrs['SOURCE_DATA'] = H5_PATH
        
        out_file.create_dataset('predicted_series', data=predicted_series, compression='gzip')
        out_file.create_dataset('rmse_series', data=rmse_series, compression='gzip')
        out_file.create_dataset('anomaly_flags', data=anomaly_flags, compression='gzip')
        out_file.create_dataset('change_date_timestamp', data=change_date_map, compression='gzip')
        out_file.create_dataset('change_count', data=change_count_map, compression='gzip')
        
    print("Harmonized CCD Pipeline Complete!")

if __name__ == "__main__":
    main()
