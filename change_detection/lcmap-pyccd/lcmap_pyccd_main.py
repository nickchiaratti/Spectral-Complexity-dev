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
import ccd
import ccd.math_utils
import scipy.stats

# Monkey patch mode to fix scipy 1.11+ breaking pyccd
original_mode = scipy.stats.mode
def safe_mode(*args, **kwargs):
    kwargs['keepdims'] = True
    return original_mode(*args, **kwargs)
ccd.math_utils.mode = safe_mode

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
H5_PATH = f"C:/satelliteImagery/HLST30/HLST_{LOCATION}_Harmonized_SC_EM-7_Norm-None.h5"

TARGET_METRIC = 'sliding_volume_z_score'
# lcmap-pyccd params
# Chi-square for 1 DOF at 99% is 6.63. We replicate 6 times -> 39.78
CHANGE_THRESHOLD = 39.78 
MIN_SAMPLES = 20

def get_ordinal_dates(acq_times):
    ord_dates = []
    for dt in acq_times:
        dt_obj = datetime.datetime.fromtimestamp(float(dt), tz=datetime.timezone.utc)
        ord_dates.append(dt_obj.toordinal())
    return np.array(ord_dates)

def predict_ccdc(model, dates):
    w = 2 * np.pi / 365.2425
    coefs = model['green']['coefficients']
    intercept = model['green']['intercept']
    
    t = dates
    pred = intercept + coefs[0] * t
    if len(coefs) > 1:
        pred += coefs[1] * np.cos(w * t) + coefs[2] * np.sin(w * t)
    if len(coefs) > 3:
        pred += coefs[3] * np.cos(2 * w * t) + coefs[4] * np.sin(2 * w * t)
    if len(coefs) > 5:
        pred += coefs[5] * np.cos(3 * w * t) + coefs[6] * np.sin(3 * w * t)
    return pred

def _process_row_chunk(chunk_args):
    y_start, y_end, width, y_data, valid_mask, ord_dates, acq_times, change_threshold, min_samples = chunk_args
    
    num_frames = y_data.shape[0]
    chunk_height = y_end - y_start
    
    chunk_pred = np.full((num_frames, chunk_height, width), np.nan, dtype=np.float32)
    chunk_rmse = np.full((num_frames, chunk_height, width), np.nan, dtype=np.float32)
    chunk_flags = np.zeros((num_frames, chunk_height, width), dtype=np.uint8)
    chunk_date = np.full((chunk_height, width), np.nan, dtype=np.float64)
    chunk_count = np.zeros((chunk_height, width), dtype=np.int32)
    
    params = {'CHANGE_THRESHOLD': change_threshold}
    
    for y_local in range(chunk_height):
        y_global = y_start + y_local
        for x in range(width):
            pixel_valid = valid_mask[:, y_global, x]
            
            # QA array: 66 = clear, 255 = fill/invalid
            pixel_qas = np.where(pixel_valid, 66, 255).astype(np.int16)
            
            valid_indices = np.where(pixel_valid)[0]
            if len(valid_indices) < min_samples:
                continue
                
            y_series = y_data[:, y_global, x].copy()
            y_series[np.isnan(y_series)] = 0.0  # safe fill for pyccd, ignored by QA=255
            
            # Map Z-scores to [2000, 4000] to satisfy pyccd filters
            mapped_series = (y_series * 100.0) + 3000.0
            mapped_series = np.clip(mapped_series, 1, 7069)
            
            try:
                # Replication Wrapper logic (duplicate the 1D series across all 6 expected bands)
                results = ccd.detect(ord_dates, mapped_series, mapped_series, mapped_series, mapped_series, mapped_series, mapped_series, mapped_series, pixel_qas, params=params)
                models = results.get('change_models', [])
                
                if not models:
                    continue
                    
                change_count_val = max(0, len(models) - 1)
                chunk_count[y_local, x] = change_count_val
                
                if change_count_val > 0:
                    first_break_ord = models[0]['break_day']
                    diffs = np.abs(ord_dates - first_break_ord)
                    best_idx = np.argmin(diffs)
                    chunk_date[y_local, x] = acq_times[best_idx]
                
                # Reconstruct models for visualization
                for m in models:
                    m_mask = (ord_dates >= m['start_day']) & (ord_dates <= m['end_day']) & pixel_valid
                    if np.any(m_mask):
                        preds = predict_ccdc(m, ord_dates[m_mask])
                        preds = (preds - 3000.0) / 100.0
                        chunk_pred[m_mask, y_local, x] = preds
                        chunk_rmse[m_mask, y_local, x] = m['green']['rmse'] / 100.0
                    
                    if m['break_day'] > 0:
                        b_idx = np.where(ord_dates == m['break_day'])[0]
                        if len(b_idx) > 0:
                            chunk_flags[b_idx[0], y_local, x] = 1

            except Exception as e:
                # If a specific pixel fails in the ccd library, skip gracefully
                pass
                
    return y_start, y_end, chunk_pred, chunk_rmse, chunk_flags, chunk_date, chunk_count

def main():
    output_h5 = f"C:/satelliteImagery/HLST30/CCD/{LOCATION}_CCD_lcmap_pyccd_Change_Detection_{TARGET_METRIC}.h5"

    print(f"Loading data from {H5_PATH}...")
    with h5py.File(H5_PATH, 'r') as f:
        data_grp = f['/HDFEOS/GRIDS/HARMONIZED/Data Fields']
        metric_ds = data_grp[TARGET_METRIC]
        
        acq_times = metric_ds.attrs['acquisition_time'][:]
        y_data = metric_ds[...]
        
        common_mask = data_grp['common_mask'][...]
        valid_mask = (common_mask == 0) & ~np.isnan(y_data)
        
        geo_transform = metric_ds.attrs.get('GeoTransform')
        spatial_ref = metric_ds.attrs.get('spatial_ref')
        
    num_frames, height, width = y_data.shape
    
    # Sort chronologically
    sort_idx = np.argsort(acq_times)
    acq_times = acq_times[sort_idx]
    y_data = y_data[sort_idx, ...]
    valid_mask = valid_mask[sort_idx, ...]
    
    ord_dates = get_ordinal_dates(acq_times)

    print(f"Dataset shape: {num_frames} frames, {height}x{width} pixels")

    change_date_map = np.zeros((height, width), dtype=np.float64)
    change_date_map[:] = np.nan
    change_count_map = np.zeros((height, width), dtype=np.int32)
    
    predicted_series = np.full((num_frames, height, width), np.nan, dtype=np.float32)
    rmse_series = np.full((num_frames, height, width), np.nan, dtype=np.float32)
    anomaly_flags = np.zeros((num_frames, height, width), dtype=np.uint8)

    print("\nExecuting lcmap-pyccd Replication Wrapper...")
    
    n_jobs = multiprocessing.cpu_count()
    print(f"Using {n_jobs} cores for parallel processing.")
    
    num_chunks = max(1, n_jobs * 4) 
    chunk_size = max(1, math.ceil(height / num_chunks))
    
    chunks = []
    for y_start in range(0, height, chunk_size):
        y_end = min(y_start + chunk_size, height)
        chunk_args = (
            y_start, y_end, width, 
            y_data, valid_mask, 
            ord_dates, acq_times, 
            CHANGE_THRESHOLD, MIN_SAMPLES
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

    os.makedirs(os.path.dirname(output_h5), exist_ok=True)
    print(f"\nSaving Results to {output_h5}...")
    with h5py.File(output_h5, 'w') as out_file:
        if spatial_ref is not None:
            out_file.attrs['spatial_ref'] = spatial_ref
        if geo_transform is not None:
            out_file.attrs['GeoTransform'] = geo_transform
        
        # Save dummy RMSE_MULTIPLIER for the visualization script to use for drawing bounds
        out_file.attrs['RMSE_MULTIPLIER'] = 1.0 
        out_file.attrs['MIN_SAMPLES'] = MIN_SAMPLES
        out_file.attrs['TARGET_METRIC'] = TARGET_METRIC
        out_file.attrs['SOURCE_DATA'] = H5_PATH
        out_file.attrs['CHANGE_THRESHOLD'] = CHANGE_THRESHOLD
        
        out_file.create_dataset('predicted_series', data=predicted_series, compression='gzip')
        out_file.create_dataset('rmse_series', data=rmse_series, compression='gzip')
        out_file.create_dataset('anomaly_flags', data=anomaly_flags, compression='gzip')
        out_file.create_dataset('change_date_timestamp', data=change_date_map, compression='gzip')
        out_file.create_dataset('change_count', data=change_count_map, compression='gzip')
        
    print("lcmap-pyccd Pipeline Complete!")

if __name__ == "__main__":
    main()
