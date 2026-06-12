import os
import h5py
import numpy as np
import datetime
import math
from tqdm import tqdm

# ==========================================
# 1. CONFIGURATION
# ==========================================
LOCATION = "Hurlingham"
H5_PATH = f"C:/satelliteImagery/HLST30/HLST_{LOCATION}_Harmonized_SC_EM-7_Norm-bandCount.h5"
OUTPUT_H5 = f"C:/satelliteImagery/HLST30/CCD/{LOCATION}_CCD_Harmonized_Change_Detection.h5"

TARGET_METRIC = 'sliding_volume_z_score'
RMSE_MULTIPLIER = 3.0
CONSECUTIVE_ANOMALIES = 4
TIME_WINDOW_YEARS = 3.0
ENABLE_ELASTIC_WINDOW = True  # Allows window to expand backwards to meet MIN_SAMPLES
MAX_ELASTIC_WINDOW_YEARS = TIME_WINDOW_YEARS + 2.0  # Maximum span to expand backwards
NUM_HARMONICS = 3  # Yields 8 parameters total: Intercept, Slope, Cos1, Sin1, Cos2, Sin2, Cos3, Sin3
MIN_SAMPLES = NUM_HARMONICS * 2 + 2 # Minimum required to solve OLS without being underdetermined

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

def build_harmonic_matrix(t, num_harmonics):
    """
    Constructs a Fourier basis matrix incorporating a linear trend.
    Columns: [Intercept, Slope, Cos(1x), Sin(1x), Cos(2x), Sin(2x), ...]
    """
    w = 2.0 * math.pi
    cols = [
        np.ones_like(t),  # Intercept
        t                 # Linear Trend
    ]
    for u in range(1, num_harmonics + 1):
        cols.append(np.cos(u * w * t))
        cols.append(np.sin(u * w * t))
    return np.column_stack(cols)

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

    X_full = build_harmonic_matrix(frac_years, NUM_HARMONICS)

    print("\nExecuting Sliding Window OLS Harmonic Regression...")
    for y in tqdm(range(height), desc="Scanning pixels"):
        for x in range(width):
            pixel_valid = valid_mask[:, y, x]
            valid_indices = np.where(pixel_valid)[0]
            
            if len(valid_indices) <= MIN_SAMPLES:
                continue

            consecutive_count = 0
            first_valid_time = frac_years[valid_indices[0]]
            
            # Start predicting from the observation after the initial minimum samples
            for i in range(MIN_SAMPLES, len(valid_indices)):
                target_idx = valid_indices[i]
                target_time = frac_years[target_idx]
                
                # Enforce TIME_WINDOW_YEARS as the initialization requirement
                if target_time - first_valid_time < TIME_WINDOW_YEARS:
                    continue
                
                # Subset past observations within the time window
                window_start_time = target_time - TIME_WINDOW_YEARS
                past_valid = valid_indices[:i]
                past_times = frac_years[past_valid]
                in_window_mask = past_times >= window_start_time
                
                # Elastic window: enforce minimum samples for OLS rank constraint
                if np.sum(in_window_mask) < MIN_SAMPLES:
                    if ENABLE_ELASTIC_WINDOW:
                        # Expand backwards to grab exactly MIN_SAMPLES
                        train_idx = past_valid[-MIN_SAMPLES:]
                        # Enforce maximum expansion duration
                        if target_time - frac_years[train_idx[0]] > MAX_ELASTIC_WINDOW_YEARS:
                            continue
                    else:
                        continue
                else:
                    train_idx = past_valid[in_window_mask]
                
                X_train = X_full[train_idx, :]
                Y_train = y_data[train_idx, y, x]
                
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
                
                predicted_series[target_idx, y, x] = y_pred
                rmse_series[target_idx, y, x] = rmse
                
                # Check anomaly condition
                actual = y_data[target_idx, y, x]
                error = abs(actual - y_pred)
                
                is_anomaly = error > (RMSE_MULTIPLIER * rmse)
                if is_anomaly:
                    anomaly_flags[target_idx, y, x] = 1
                    consecutive_count += 1
                    
                    if consecutive_count >= CONSECUTIVE_ANOMALIES:
                        change_count_map[y, x] += 1
                        # If this is the first confirmed structural change, record the date
                        if np.isnan(change_date_map[y, x]):
                            # We record the date of the FIRST anomaly in this 3-streak
                            first_anomaly_idx = valid_indices[i - CONSECUTIVE_ANOMALIES + 1]
                            change_date_map[y, x] = acq_times[first_anomaly_idx]
                else:
                    consecutive_count = 0

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
        out_file.attrs['NUM_HARMONICS'] = NUM_HARMONICS
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
