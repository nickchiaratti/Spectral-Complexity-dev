import os
import h5py
import numpy as np
import datetime
import math
from tqdm import tqdm
from sklearn.linear_model import LassoLarsIC
import multiprocessing
import joblib
import contextlib
from scipy.stats import chi2
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
# Input/Output
LOCATION = "Tait"
H5_PATH = f"C:/satelliteImagery/HLST30/HLST_{LOCATION}_Harmonized_SC_EM-7_Norm-None.h5"
TARGET_METRIC = 'sliding_volume_z_score'

if TARGET_METRIC == 'sliding_volume_z_score':
    TARGET_NAME = 'Spectral Complexity (Z-Score)'
elif TARGET_METRIC == 'sliding_volume_robust_scale':
    TARGET_NAME = 'Spectral Complexity (Robust)'
elif TARGET_METRIC == 'ndvi':
    TARGET_NAME = 'NDVI'
elif TARGET_METRIC == 'ndbi':
    TARGET_NAME = 'NDBI'

# Anomaly Detection Thresholds
CHANGE_PROBABILITY = 0.99
CONSECUTIVE_ANOMALIES = 4
CHI2_DEGREES_OF_FREEDOM = 3

# Segment Initialization & Stability Constraints
MAX_RMSE = 1.0
MIN_RMSE_CLAMP = 1e-5
MIN_YEARS_FOR_INIT = 1.5
MIN_SAMPLES = 16

# Harmonic Regression Configurations
ENABLE_CONSTANT = True
ENABLE_LINEAR = False
ENABLE_QUADRATIC = False
TEMPORAL_PERIODS = [0.33, 0.5, 1]

# RLS Tuning
RLS_RIDGE_PENALTY = 1e-6
RLS_FORGETTING_FACTOR = 0.999



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

def build_harmonic_matrix(t, temporal_periods=TEMPORAL_PERIODS, 
                          enable_const=ENABLE_CONSTANT, 
                          enable_lin=ENABLE_LINEAR, 
                          enable_quad=ENABLE_QUADRATIC):
    """
    Constructs a Fourier basis matrix with configurable polynomial trends and temporal periods.
    """
    cols = []
    
    if enable_const:
        cols.append(np.ones_like(t))
    
    if enable_lin:
        cols.append(t)
    if enable_quad:
        cols.append(t**2)
        
    w = 2.0 * math.pi
    for p in temporal_periods:
        cols.append(np.cos((w / p) * t))
        cols.append(np.sin((w / p) * t))
        
    return np.column_stack(cols) if cols else np.zeros((len(t), 0))

def _process_row_chunk(chunk_args):
    y_start, y_end, width, y_data_chunk, valid_mask_chunk, frac_years, X_full, acq_times, min_samples, change_prob, consec_anom = chunk_args
    
    num_frames = y_data_chunk.shape[0]
    chunk_height = y_end - y_start
    num_features = X_full.shape[1]
    
    chunk_pred = np.full((num_frames, chunk_height, width), np.nan, dtype=np.float32)
    chunk_rmse = np.full((num_frames, chunk_height, width), np.nan, dtype=np.float32)
    chunk_coef = np.full((num_frames, chunk_height, width, num_features), np.nan, dtype=np.float32)
    chunk_flags = np.zeros((num_frames, chunk_height, width), dtype=np.uint8)
    chunk_date = np.full((chunk_height, width), np.nan, dtype=np.float64)
    chunk_count = np.zeros((chunk_height, width), dtype=np.int32)
    
    chi2_threshold = chi2.ppf(change_prob, df=CHI2_DEGREES_OF_FREEDOM)
    
    for y_local in range(chunk_height):
        y_global = y_start + y_local
        for x in range(width):
            pixel_valid = valid_mask_chunk[:, y_local, x]
            valid_indices = np.where(pixel_valid)[0]
            
            if len(valid_indices) <= min_samples:
                continue

            i = 0
            while i < len(valid_indices) - min_samples:
                # 1. Initialize segment
                segment_start_idx = i
                train_end = segment_start_idx + min_samples
                
                # 1-year constraint
                while train_end <= len(valid_indices):
                    time_span = frac_years[valid_indices[train_end - 1]] - frac_years[valid_indices[segment_start_idx]]
                    if time_span >= MIN_YEARS_FOR_INIT:
                        break
                    train_end += 1
                    
                if train_end > len(valid_indices):
                    # Could not find a 1-year window
                    break
                
                active_pool = list(range(segment_start_idx, train_end))
                min_dof = X_full.shape[1] + 1
                absolute_min_samples = max(min_dof, min_samples)
                
                initialization_successful = False
                lasso_model = LassoLarsIC(criterion='bic', fit_intercept=True)
                
                while len(active_pool) >= absolute_min_samples:
                    X_train = X_full[valid_indices[active_pool], :]
                    Y_train = y_data_chunk[valid_indices[active_pool], y_local, x]
                    
                    try:
                        lasso_model.fit(X_train, Y_train)
                        y_train_pred = lasso_model.predict(X_train)
                        rmse = np.sqrt(np.mean((Y_train - y_train_pred)**2))
                        rmse = max(rmse, MIN_RMSE_CLAMP)
                    except Exception:
                        break
                        
                    errors = np.abs(Y_train - y_train_pred)
                    max_error_idx = int(np.argmax(errors))
                    max_error = errors[max_error_idx]
                    
                    chi2_val_init = (max_error / rmse)**2
                    if chi2_val_init > chi2_threshold:
                        active_pool.pop(max_error_idx)
                    else:
                        initialization_successful = True
                        break
                
                if not initialization_successful:
                    i += 1
                    continue
                
                # Hand-off to OLS for RLS initialization
                active_features = np.where(lasso_model.coef_ != 0)[0]
                
                # If the constant term is enabled, force it to be active 
                # (LASSO zeroed it out internally due to fit_intercept=True)
                if ENABLE_CONSTANT and 0 not in active_features:
                    active_features = np.insert(active_features, 0, 0)
                    
                if len(active_features) == 0:
                    active_features = np.array([0]) 
                    
                X_train_active = X_full[valid_indices[active_pool]][:, active_features]
                Y_train_active = y_data_chunk[valid_indices[active_pool], y_local, x]
                
                # Initialize Full RLS State (track all terms)
                X_train_all = X_full[valid_indices[active_pool]]
                num_total_features = X_train_all.shape[1]
                
                try:
                    # OLS only on active features to prevent overfitting on initialization
                    P_active = np.linalg.inv(X_train_active.T @ X_train_active + np.eye(len(active_features)) * RLS_RIDGE_PENALTY)
                    theta_active = P_active @ X_train_active.T @ Y_train_active
                    
                    # Map to full theta vector (inactive terms start at 0)
                    theta = np.zeros(num_total_features)
                    theta[active_features] = theta_active
                    
                    # Initialize P matrix on all features so RLS can adapt them later
                    P = np.linalg.inv(X_train_all.T @ X_train_all + np.eye(num_total_features) * RLS_RIDGE_PENALTY)
                except np.linalg.LinAlgError:
                    i += 1
                    continue
                
                Y_pred_init = X_train_all @ theta
                SSE = np.sum((Y_train_active - Y_pred_init)**2)
                dof = len(active_features)
                n_points = len(active_pool)
                rmse = np.sqrt(SSE / max(1, n_points - dof))
                rmse = max(rmse, MIN_RMSE_CLAMP)
                
                # CCDC Transient Event Rejection: If the initialization window is 
                # highly volatile (e.g. caught in a transition), reject it and slide forward.
                if rmse > MAX_RMSE:
                    i += 1
                    continue
                    
                # Backfill predictions for the initialization window
                for global_idx in valid_indices[segment_start_idx:train_end]:
                    x_target = X_full[global_idx, :]
                    chunk_pred[global_idx, y_local, x] = np.dot(x_target, theta)
                    chunk_rmse[global_idx, y_local, x] = rmse
                    chunk_coef[global_idx, y_local, x, :] = theta
                
                consecutive_count = 0
                break_detected = False
                
                # 2. Forward Expanding Phase
                for j in range(train_end, len(valid_indices)):
                    target_idx = valid_indices[j]
                    
                    x_target = X_full[target_idx, :]
                    y_pred = np.dot(x_target, theta)
                    
                    actual = y_data_chunk[target_idx, y_local, x]
                    error = abs(actual - y_pred)
                    
                    chunk_pred[target_idx, y_local, x] = y_pred
                    # Keep RMSE frozen from initialization for stable break detection
                    chunk_rmse[target_idx, y_local, x] = rmse
                    chunk_coef[target_idx, y_local, x, :] = theta
                    
                    chi2_val = (error / rmse)**2
                    is_anomaly = chi2_val > chi2_threshold
                    
                    if is_anomaly:
                        chunk_flags[target_idx, y_local, x] = 1
                        consecutive_count += 1
                        
                        if consecutive_count >= consec_anom:
                            chunk_count[y_local, x] += 1
                            if np.isnan(chunk_date[y_local, x]):
                                first_anomaly_idx = valid_indices[j - consec_anom + 1]
                                chunk_date[y_local, x] = acq_times[first_anomaly_idx]
                                
                            # Break detected: end segment and prepare to start new one
                            break_detected = True
                            break
                    else:
                        consecutive_count = 0
                        # RLS Update Step
                        x_target_2d = x_target.reshape(-1, 1) # column vector
                        Px = P @ x_target_2d
                        denom = RLS_FORGETTING_FACTOR + (x_target_2d.T @ Px)[0, 0]
                        K = Px / denom
                        
                        e = actual - y_pred
                        
                        theta = theta + (K.flatten() * e)
                        P = (P - (K @ (x_target_2d.T @ P))) / RLS_FORGETTING_FACTOR
                        # Update SSE and mathematically track RMSE, but DO NOT use it 
                        # to overwrite the frozen `chunk_rmse` for the anomaly threshold.
                        SSE = SSE + (e**2) / denom
                        n_points += 1
                        new_rmse = np.sqrt(SSE / max(1, n_points - dof))
                
                if break_detected:
                    # Re-initialize after the break, starting from the first anomaly
                    i = j - consec_anom + 1
                else:
                    # Reached end of time series
                    break

    return y_start, y_end, chunk_pred, chunk_rmse, chunk_coef, chunk_flags, chunk_date, chunk_count

def main(enable_const=ENABLE_CONSTANT, 
         enable_lin=ENABLE_LINEAR, 
         enable_quad=ENABLE_QUADRATIC, 
         temporal_periods=None,
         target_metric=TARGET_METRIC,
         launch_vis=True):
    if temporal_periods is None:
        temporal_periods = TEMPORAL_PERIODS
        
    # Enforce global hyperparameter instead of dynamic CCDC equation
    min_samples = MIN_SAMPLES
    
    _term_str = f"C{int(enable_const)}L{int(enable_lin)}Q{int(enable_quad)}"
    _period_str = f"P{len(temporal_periods)}"
    output_h5 = f"C:/satelliteImagery/HLST30/CCD/{LOCATION}_CCD_Harmonized_Change_Detection_{target_metric}_{_term_str}_{_period_str}.h5"

    print(f"Loading data from {H5_PATH}...")
    with h5py.File(H5_PATH, 'r') as f:
        data_grp = f['/HDFEOS/GRIDS/HARMONIZED/Data Fields']
        metric_ds = data_grp[target_metric]
        
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

    X_full = build_harmonic_matrix(frac_years, temporal_periods=temporal_periods,
                                   enable_const=enable_const, enable_lin=enable_lin, enable_quad=enable_quad)
                                   
    num_features = X_full.shape[1]
    coef_series = np.full((num_frames, height, width, num_features), np.nan, dtype=np.float32)

    print("\nExecuting Sliding Window LASSO Harmonic Regression...")
    
    n_jobs = min(8, multiprocessing.cpu_count())
    print(f"Using {n_jobs} cores for parallel processing.")
    
    # Restrict chunk size to minimize per-worker memory allocation (ArrayMemoryError fix)
    chunk_size = 4
    num_chunks = max(1, math.ceil(height / chunk_size))
    
    chunks = []
    for y_start in range(0, height, chunk_size):
        y_end = min(y_start + chunk_size, height)
        chunk_args = (
            y_start, y_end, width, 
            y_data[:, y_start:y_end, :], valid_mask[:, y_start:y_end, :], 
            frac_years, X_full, acq_times, 
            min_samples, CHANGE_PROBABILITY, CONSECUTIVE_ANOMALIES
        )
        chunks.append(chunk_args)
        
    with tqdm_joblib(tqdm(desc="Processing row chunks", total=len(chunks))):
        results = Parallel(n_jobs=n_jobs, backend='loky', return_as='generator')(
            delayed(_process_row_chunk)(chunk) for chunk in chunks
        )
    
    for y_start, y_end, c_pred, c_rmse, c_coef, c_flags, c_date, c_count in results:
        predicted_series[:, y_start:y_end, :] = c_pred
        rmse_series[:, y_start:y_end, :] = c_rmse
        coef_series[:, y_start:y_end, :, :] = c_coef
        anomaly_flags[:, y_start:y_end, :] = c_flags
        change_date_map[y_start:y_end, :] = c_date
        change_count_map[y_start:y_end, :] = c_count

    os.makedirs(os.path.dirname(output_h5), exist_ok=True)
    print(f"\nSaving Results to {output_h5}...")
    with h5py.File(output_h5, 'w') as out_file:
        out_file.attrs['spatial_ref'] = spatial_ref
        out_file.attrs['GeoTransform'] = geo_transform
        out_file.attrs['CHANGE_PROBABILITY'] = CHANGE_PROBABILITY
        out_file.attrs['CONSECUTIVE_ANOMALIES'] = CONSECUTIVE_ANOMALIES
        out_file.attrs['CHI2_DEGREES_OF_FREEDOM'] = CHI2_DEGREES_OF_FREEDOM
        out_file.attrs['MAX_RMSE'] = MAX_RMSE
        out_file.attrs['MIN_RMSE_CLAMP'] = MIN_RMSE_CLAMP
        out_file.attrs['MIN_YEARS_FOR_INIT'] = MIN_YEARS_FOR_INIT
        out_file.attrs['MIN_SAMPLES'] = min_samples
        out_file.attrs['TEMPORAL_PERIODS'] = temporal_periods
        out_file.attrs['ENABLE_CONSTANT'] = enable_const
        out_file.attrs['ENABLE_LINEAR'] = enable_lin
        out_file.attrs['ENABLE_QUADRATIC'] = enable_quad
        out_file.attrs['TARGET_METRIC'] = target_metric
        out_file.attrs['SOURCE_DATA'] = H5_PATH
        
        out_file.create_dataset('predicted_series', data=predicted_series, compression='gzip')
        out_file.create_dataset('rmse_series', data=rmse_series, compression='gzip')
        out_file.create_dataset('coef_series', data=coef_series, compression='gzip')
        out_file.create_dataset('anomaly_flags', data=anomaly_flags, compression='gzip')
        out_file.create_dataset('change_date_timestamp', data=change_date_map, compression='gzip')
        out_file.create_dataset('change_count', data=change_count_map, compression='gzip')
        
    print("Harmonized CCD Pipeline Complete!")

    if launch_vis:
        print("\nLaunching visualization...")
        try:
            from harmonized_CCD_vis import plot_spatial_anomaly_overlay
            plot_spatial_anomaly_overlay(H5_PATH, output_h5)
        except ImportError as e:
            print(f"Could not import visualization module: {e}")
        except Exception as e:
            print(f"An error occurred while launching visualization: {e}")
            
    return output_h5

if __name__ == "__main__":
    main()
