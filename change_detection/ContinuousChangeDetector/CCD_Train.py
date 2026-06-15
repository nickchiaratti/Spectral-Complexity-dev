import os
import h5py
import numpy as np
import datetime
import math
from tqdm import tqdm
from scipy import ndimage
import SpecComplex as sc

# ==========================================
# 1. CONFIGURATION
# ==========================================
Location = "Tait"

multisensor = True 
landsat_path = f"C:/satelliteImagery/LANDSAT/{Location}/LANDSAT_Stack_{Location}_GEE_2015_2025_WRS16_SC_EM-7_Gram-minEndmember_Norm-bandCount.h5"
tanager_path = f"C:/satelliteImagery/Tanager/{Location}/Tanager_Stack_{Location}_HDFEOS_SC_EM-7_Gram-minEndmember_Norm-bandCount.h5"

TARGET_METRIC = 'sliding_volume_z_score_masked'
suffix = "_WRS16"
if TARGET_METRIC == 'sliding_volume_z_score':
    suffix += '_zscore'
if TARGET_METRIC == 'sliding_volume_z_score_masked':
    suffix += '_maskedZscore'
elif TARGET_METRIC == 'sliding_volume_map':
    suffix += '_SC'
elif TARGET_METRIC == 'evi_map':
    suffix += '_EVI'

if multisensor:
    OUTPUT_H5 = f"C:/satelliteImagery/AnomalyDetector/CCD/{Location}/CCD_Multisensor_Change_Detection_{Location}" + suffix + ".h5"
else: 
    OUTPUT_H5 = f"C:/satelliteImagery/AnomalyDetector/CCD/{Location}/CCD_Landsat_Change_Detection_{Location}" + suffix + ".h5"
TRAIN_END_YEAR = 2022

# CCDC Statistical Change Thresholds
RMSE_MULTIPLIER = 3.5    # Deviation must exceed 3 * RMSE
CONSECUTIVE_ANOMALIES = 3 # Must stay anomalous for 3 consecutive clear observations

# Harmonic Configuration
TEMPORAL_PERIODS = [1/3, 1/2, 1.0, 2.0, 3.0, 4.0]
MIN_TRAINING_OBSERVATIONS = 15 # Minimum cloud-free days required for least squares fit (Zhu recommends 15 for 6 parameter model)

SUN_ELEVATION_THRESHOLD = 30
CLOUD_DILATION = 4
QA_REJECT_MASK = 0b11111
RADSAT_ACCEPT_VALUE = 0
AEROSOL_ACCEPT_LEVEL = 'medium' # 'low' 'medium' 'high' 'all'

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
        np.ones_like(t),  # a0 (Intercept / DC Component)
    ]
    
    for p in temporal_periods:
        cols.append(np.cos((w / p) * t))  # Real part
        cols.append(np.sin((w / p) * t))  # Imaginary part
        
    return np.column_stack(cols)

def generate_3d_mask(grid_name, data_grp, indices, height, width):
    """Routes masking sequentially to SpecComplex for either LANDSAT or TANAGER."""
    mask_3d = np.zeros((len(indices), height, width), dtype=bool)
    for i, original_idx in enumerate(tqdm(indices, desc=f"Generating {grid_name} Masks", leave=False)):
        if grid_name == 'LANDSAT':
            mask_3d[i] = sc.get_landsat_mask(
                data_grp, original_idx, (height, width),
                sun_elevation_threshold=SUN_ELEVATION_THRESHOLD,
                cloud_dilation=CLOUD_DILATION,
                qa_reject_mask=QA_REJECT_MASK,
                radsat_accept_value=RADSAT_ACCEPT_VALUE,
                aerosol_accept_level=AEROSOL_ACCEPT_LEVEL
            )
        elif grid_name == 'TANAGER':
            mask_3d[i] = sc.get_tanager_mask(
                data_grp, original_idx, (height, width),
                sun_elevation_threshold=SUN_ELEVATION_THRESHOLD,
                cloud_dilation=CLOUD_DILATION
            )
    return mask_3d

def load_and_preprocess_sensor(h5_path, grid_expected):
    """Loads data from a specific sensor, builds pure QA masks, and extracts timestamps."""
    with h5py.File(h5_path, 'r') as f:
        grids = list(f['/HDFEOS/GRIDS'].keys())
        if grid_expected not in grids:
            raise KeyError(f"CRITICAL ERROR: Expected grid {grid_expected} not found in {h5_path}. Halting pipeline.")
            
        data_grp = f[f'/HDFEOS/GRIDS/{grid_expected}/Data Fields']
        sr_ds = data_grp['surface_reflectance']
        
        height, width = sr_ds.shape[2], sr_ds.shape[3]
        
        acq_times = sr_ds.attrs.get('acquisition_time')
        if acq_times is None:
            raise ValueError(f"CRITICAL ERROR: 'acquisition_time' attribute missing in {h5_path}.")
            
        frac_years = extract_fractional_years(acq_times)
        num_frames = len(frac_years)
        
        # Extract raw data without artificial fill values
        y_data = data_grp[TARGET_METRIC][...]
        
        # Centralized SpecComplex masking
        indices = list(range(num_frames))
        valid_mask = generate_3d_mask(grid_expected, data_grp, indices, height, width)
        valid_mask &= ~np.isnan(y_data)
        
        geo_transform = sr_ds.attrs.get('GeoTransform')
        spatial_ref = sr_ds.attrs.get('spatial_ref')
        
        return frac_years, y_data, valid_mask, height, width, geo_transform, spatial_ref

def main(TRAIN_END_YEAR):
    print("Loading multi-sensor HDF5 data into memory...")
    
    if multisensor:
        print("\n--- Processing Primary Sensor: LANDSAT ---")
        l_frac, l_y, l_mask, l_h, l_w, l_gt, l_sr = load_and_preprocess_sensor(landsat_path, 'LANDSAT')
        
        print("\n--- Processing Secondary Sensor: TANAGER ---")
        t_frac, t_y, t_mask, t_h, t_w, t_gt, t_sr = load_and_preprocess_sensor(tanager_path, 'TANAGER')
        
        # CRITICAL ENFORCEMENT: Spatial grids must match exactly for 3D array concatenation
        if l_h != t_h or l_w != t_w:
            raise ValueError(f"CRITICAL ERROR: Spatial dimension mismatch. Landsat: {l_h}x{l_w}, Tanager: {t_h}x{t_w}")
            
        print("\nMerging datasets along the temporal axis...")
        all_frac_years = np.concatenate([l_frac, t_frac])
        all_y_data = np.concatenate([l_y, t_y], axis=0)
        all_valid_mask = np.concatenate([l_mask, t_mask], axis=0)
        
        height, width = l_h, l_w
        geo_transform, spatial_ref = l_gt, l_sr
    else:
        print("\n--- Processing Sensor: LANDSAT ---")
        all_frac_years, all_y_data, all_valid_mask, height, width, geo_transform, spatial_ref = load_and_preprocess_sensor(landsat_path, 'LANDSAT')

    print("\nSorting merged time-series chronologically...")
    sort_idx = np.argsort(all_frac_years)
    
    all_frac_years = all_frac_years[sort_idx]
    all_y_data = all_y_data[sort_idx, ...]
    all_valid_mask = all_valid_mask[sort_idx, ...]

    # 1. Filter for Training Period
    train_indices = np.where(all_frac_years < (TRAIN_END_YEAR + 1.0))[0]
    train_frac_years = all_frac_years[train_indices]
    train_y_data = all_y_data[train_indices, ...]
    train_valid_mask = all_valid_mask[train_indices, ...]
    
    print(f"\nIsolated {len(train_indices)} chronologically sorted frames for baseline training (<= {TRAIN_END_YEAR}).")
    print(f"Train end: {all_frac_years[train_indices[-1]]}")
    

    # 3. Prepare Coefficient Output Arrays
    num_coeffs = 1 + (2 * len(TEMPORAL_PERIODS)) # Intercept + 2 for each period
    baseline_coefficients = np.full((num_coeffs, height, width), np.nan, dtype=np.float32)
    baseline_rmse = np.full((height, width), np.nan, dtype=np.float32)

    # Gather Test Data
    test_indices = np.where(all_frac_years >= (TRAIN_END_YEAR + 1.0))[0]
    print(f"Test start: {all_frac_years[test_indices[0]]}")
    test_frac_years = all_frac_years[test_indices]
    test_valid_mask = all_valid_mask[test_indices, ...]
    test_y_data = all_y_data[test_indices, ...]

    # 4. Pixel-by-Pixel Vectorized OLS Fitting
    print("\nExecuting Ordinary Least Squares (OLS) Harmonic Regression...")

    X_full = build_harmonic_matrix(train_frac_years)
    for y in tqdm(range(height), desc="Fitting CCDC rows"):
        for x in range(width):
            valid_t = train_valid_mask[:, y, x]

            if np.sum(valid_t) < MIN_TRAINING_OBSERVATIONS: 
                continue # Leave as NaN

            X_pixel = X_full[valid_t, :]
            Y_pixel = train_y_data[valid_t, y, x]

            coeffs, residuals, rank, s = np.linalg.lstsq(X_pixel, Y_pixel, rcond=None)

            baseline_coefficients[:, y, x] = coeffs

            # Calculate True RMSE
            if len(residuals) > 0:
                rmse = np.sqrt(residuals[0] / len(Y_pixel))
            else:
                # If exact fit, calculate manually
                y_pred = X_pixel @ coeffs
                rmse = np.sqrt(np.mean((Y_pixel - y_pred)**2))

            baseline_rmse[y, x] = max(rmse, 1e-5) # Prevent 0.0 RMSE for stability
            
    
    # Pre-calculate the design matrix for the future dates
    X_test = build_harmonic_matrix(test_frac_years)
    # Initialize output maps
    change_detected_map = np.zeros((height, width), dtype=np.uint8)
    change_date_map = np.zeros((height, width), dtype=np.float32) # Stores the fractional year of change
    predicted_series = np.full((len(test_indices), height, width), np.nan, dtype=np.float32)
    
    

    for y in tqdm(range(height), desc="Scanning for Anomalies"):
        for x in range(width):
            
            # Skip if there's no baseline trained for this pixel (e.g., constant clouds)
            if np.isnan(baseline_rmse[y, x]):
                continue
                
            pixel_coeffs = baseline_coefficients[:, y, x]
            pixel_rmse = baseline_rmse[y, x]
            
            # Predict the expected baseline for all test dates using dot product
            # X_future is [T, N_coeffs], pixel_coeffs is [N_coeffs,] -> Result is [T,]
            y_pred = X_test @ pixel_coeffs
            predicted_series[:, y, x] = y_pred
            
            pixel_valid_mask = test_valid_mask[:, y, x]
            pixel_actual = test_y_data[:, y, x]
            
            # Mathematical condition: |Actual - Predicted| > 3 * RMSE
            # np.abs safely ignores NaNs here because pixel_valid_mask will override them anyway
            residuals = np.abs(pixel_actual - y_pred)
            is_anomalous = residuals > (RMSE_MULTIPLIER * pixel_rmse)
            
            # Only consider valid observations
            is_anomalous[~pixel_valid_mask] = False
            
            # Apply Temporal Persistence Filter (CCDC Logic)
            consecutive_count = 0
            streak_start_idx = -1 # DOCTORAL UPDATE: Explicitly track the true start of the sequence
            
            for t_idx in range(len(test_indices)):
                if not pixel_valid_mask[t_idx]:
                    continue # Ignore cloudy days AND NaNs; don't break the streak, but don't count them
                    
                if is_anomalous[t_idx]:
                    if consecutive_count == 0:
                        streak_start_idx = t_idx # Lock in the exact index of the first break
                        
                    consecutive_count += 1
                    if consecutive_count >= CONSECUTIVE_ANOMALIES:
                        # Structural change officially confirmed
                        change_detected_map[y, x] = 1
                        
                        # Extract the exact fractional year using the locked index
                        change_date_map[y, x] = test_frac_years[streak_start_idx]
                        break # Stop checking this pixel; baseline is broken
                else:
                    consecutive_count = 0 # Streak broken by normal behavior
                    streak_start_idx = -1

    print(f"\nTotal pixels with confirmed structural change: {np.sum(change_detected_map)}")

    # 5. Save the Baseline Map
    print(f"\nSaving Harmonic Baseline to {OUTPUT_H5}...")
    with h5py.File(OUTPUT_H5, 'w') as out_file:
        out_file.create_dataset('coefficients', data=baseline_coefficients, compression='gzip')
        out_file.create_dataset('rmse', data=baseline_rmse, compression='gzip')
        out_file.attrs['spatial_ref'] = spatial_ref
        out_file.attrs['GeoTransform'] = geo_transform
        
        # Dynamically build the order string for HDF5 metadata depending on TEMPORAL_PERIODS
        coeff_names = ["Intercept"] + [f"{func}({p:.2f}y)" for p in TEMPORAL_PERIODS for func in ["Cos", "Sin"]]
        out_file.attrs['coefficient_order'] = ", ".join(coeff_names)
        
        out_file.attrs['train_end_year'] = TRAIN_END_YEAR
        out_file.attrs['rmse_multiplier'] = RMSE_MULTIPLIER
        out_file.attrs['consecutive_anomalies'] = CONSECUTIVE_ANOMALIES
        out_file.attrs['temporal_periods'] = TEMPORAL_PERIODS
        out_file.attrs['min_training_observations'] = MIN_TRAINING_OBSERVATIONS
        out_file.attrs['sun_elevation_threshold'] = SUN_ELEVATION_THRESHOLD
        out_file.attrs['cloud_dilation'] = CLOUD_DILATION
        out_file.attrs['qa_reject_mask'] = QA_REJECT_MASK
        out_file.attrs['radsat_accept_value'] = RADSAT_ACCEPT_VALUE
        out_file.attrs['aerosol_accept_level'] = AEROSOL_ACCEPT_LEVEL
        out_file.attrs['target_metric'] = TARGET_METRIC
        out_file.attrs['multisensor'] = multisensor
        out_file.attrs['landsat_path'] = landsat_path
        out_file.attrs['tanager_path'] = tanager_path
        out_file.create_dataset('change_mask', data=change_detected_map, compression='gzip')
        out_file.create_dataset('change_date_frac_year', data=change_date_map, compression='gzip')
        out_file.create_dataset('predicted_baseline', data=predicted_series, compression='gzip')
    print("CCDC Baseline Training Complete!")

if __name__ == "__main__":
    main(TRAIN_END_YEAR)