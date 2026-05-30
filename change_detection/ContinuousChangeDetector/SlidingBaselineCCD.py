import os
import h5py
import numpy as np
import datetime
import math
from tqdm import tqdm
import SpecComplex as sc

# ==========================================
# 1. CONFIGURATION
# ==========================================
Location = "Tait"
Frame_Reg = "WRS16" #"CoReg"

H5_RAW_PATH = f"C:/satelliteImagery/LANDSAT/{Location}/LANDSAT_Stack_{Location}_GEE_2015_2025_{Frame_Reg}_SC_EM-7_Gram-minEndmember_Norm-bandCount.h5"

TARGET_METRIC = 'sliding_volume_z_score_masked'
suffix = f'_{Frame_Reg}'
if TARGET_METRIC == 'sliding_volume_z_score':
    suffix += '_zscore'
if TARGET_METRIC == 'sliding_volume_z_score_masked':
    suffix += '_maskedZscore'
elif TARGET_METRIC == 'sliding_volume_map':
    suffix += '_SC'
elif TARGET_METRIC == 'evi_map':
    suffix += '_EVI'

OUTPUT_H5 = f"C:/satelliteImagery/LANDSAT/{Location}/SlidingBaselineCCD_Change_Detection_{Location}{suffix}.h5"

# --- Continuous CCDC Hyperparameters ---
MIN_VALID_OBSERVATIONS = 15
RMSE_MULTIPLIER = 3.0       
CONSECUTIVE_ANOMALIES = 3   

# --- Centralized Masking Configuration ---
SUN_ELEVATION_THRESHOLD = 30
CLOUD_DILATION = 1
QA_REJECT_MASK = 0b111111
AEROSOL_ACCEPT_LEVEL = 'medium' 

# ==========================================
# 2. UTILITIES
# ==========================================
def extract_fractional_years(acq_times):
    frac_years = []
    for dt in acq_times:
        try:
            dt_obj = datetime.datetime.fromtimestamp(float(dt), tz=datetime.timezone.utc)
        except ValueError:
            dt_str = dt.decode('utf-8') if isinstance(dt, bytes) else str(dt)
            dt_obj = datetime.datetime.strptime(dt_str[:10], "%Y-%m-%d").replace(tzinfo=datetime.timezone.utc)
            
        year = dt_obj.year
        start_of_year = datetime.datetime(year, 1, 1, tzinfo=datetime.timezone.utc)
        start_of_next = datetime.datetime(year + 1, 1, 1, tzinfo=datetime.timezone.utc)
        year_duration = (start_of_next - start_of_year).total_seconds()
        elapsed = (dt_obj - start_of_year).total_seconds()
        frac_years.append(year + (elapsed / year_duration))
    return np.array(frac_years)

def build_harmonic_matrix(t):
    w = 2.0 * math.pi
    return np.column_stack([
        np.ones_like(t), t, np.cos(w * t), np.sin(w * t), np.cos(2 * w * t), np.sin(2 * w * t)
    ])

def fit_ols_model(t_array, y_array):
    X = build_harmonic_matrix(t_array)
    coeffs, residuals, _, _ = np.linalg.lstsq(X, y_array, rcond=None)
    
    if len(residuals) > 0:
        rmse = np.sqrt(residuals[0] / len(y_array))
    else:
        y_pred = X @ coeffs
        rmse = np.sqrt(np.mean((y_array - y_pred)**2))
        
    return coeffs, max(rmse, 0.001)

# ==========================================
# 3. CONTINUOUS CCDC ENGINE
# ==========================================
def main():
    print("Loading HDF5 data into memory...")
    with h5py.File(H5_RAW_PATH, 'r') as f:
        
        # Dynamic Sensor Routing
        grids = list(f['/HDFEOS/GRIDS'].keys())
        if 'LANDSAT' in grids:
            grid_name = 'LANDSAT'
        elif 'TANAGER' in grids:
            grid_name = 'TANAGER'
        else:
            raise KeyError("Neither LANDSAT nor TANAGER grid found in HDF5 file.")
            
        print(f"Detected {grid_name} dataset. Routing to centralized masking functions...")
        data_grp = f[f'/HDFEOS/GRIDS/{grid_name}/Data Fields']
        sr_ds = data_grp['surface_reflectance']
        
        acq_times = sr_ds.attrs.get('acquisition_time')
        global_t = extract_fractional_years(acq_times)
        num_frames = len(global_t)
        
        y_data = np.nan_to_num(data_grp[TARGET_METRIC][...], nan=0.0)
        
        print("Applying unified QA, Cloud, and Aerosol masks via SpecComplex...")
        height, width = sr_ds.shape[2], sr_ds.shape[3]
        valid_mask = np.ones((num_frames, height, width), dtype=bool)
        
        for i in tqdm(range(num_frames), desc="Generating Masks"):
            if grid_name == 'LANDSAT':
                valid_mask[i] = sc.get_landsat_mask(
                    data_grp, i, (height, width),
                    sun_elevation_threshold=SUN_ELEVATION_THRESHOLD,
                    cloud_dilation=CLOUD_DILATION,
                    qa_reject_mask=QA_REJECT_MASK,
                    radsat_accept_value=0,
                    aerosol_accept_level=AEROSOL_ACCEPT_LEVEL
                )
            elif grid_name == 'TANAGER':
                valid_mask[i] = sc.get_tanager_mask(
                    data_grp, i, (height, width),
                    sun_elevation_threshold=SUN_ELEVATION_THRESHOLD,
                    cloud_dilation=CLOUD_DILATION
                )

        spatial_ref = sr_ds.attrs.get('spatial_ref')
        geo_transform = sr_ds.attrs.get('GeoTransform')

    out_coeffs = np.full((6, height, width), np.nan, dtype=np.float32)
    out_rmse = np.full((height, width), np.nan, dtype=np.float32)
    change_mask = np.zeros((height, width), dtype=np.uint8)
    change_date = np.full((height, width), np.nan, dtype=np.float32)
    predicted_series = np.full((num_frames, height, width), np.nan, dtype=np.float32)

    print("Executing Pixel-Wise Continuous Sliding Baseline...")
    X_global = build_harmonic_matrix(global_t)

    for y in tqdm(range(height), desc="Processing spatial grid"):
        for x in range(width):
            pixel_valid_idx = np.where(valid_mask[:, y, x])[0]
            
            if len(pixel_valid_idx) < MIN_VALID_OBSERVATIONS:
                continue
                
            p_t = global_t[pixel_valid_idx]
            p_y = y_data[pixel_valid_idx, y, x]
            
            hist_start = 0
            hist_end = MIN_VALID_OBSERVATIONS
            
            current_coeffs, current_rmse = fit_ols_model(p_t[hist_start:hist_end], p_y[hist_start:hist_end])
            
            out_coeffs[:, y, x] = current_coeffs
            out_rmse[y, x] = current_rmse
            
            streak = 0
            first_change_found = False
            
            i = hist_end
            while i < len(p_t):
                test_idx = pixel_valid_idx[i]
                
                X_test = X_global[test_idx, :]
                y_pred = X_test @ current_coeffs
                predicted_series[test_idx, y, x] = y_pred
                
                residual = abs(p_y[i] - y_pred)
                
                if residual > (RMSE_MULTIPLIER * current_rmse):
                    streak += 1
                    if streak >= CONSECUTIVE_ANOMALIES and not first_change_found:
                        change_idx = pixel_valid_idx[i - CONSECUTIVE_ANOMALIES + 1]
                        change_mask[y, x] = 1
                        change_date[y, x] = global_t[change_idx]
                        first_change_found = True
                        break 
                else:
                    streak = 0
                    hist_end = i + 1
                    current_coeffs, current_rmse = fit_ols_model(p_t[hist_start:hist_end], p_y[hist_start:hist_end])
                    
                    if not first_change_found:
                        out_coeffs[:, y, x] = current_coeffs
                        out_rmse[y, x] = current_rmse
                        
                i += 1
                
            if not np.isnan(out_coeffs[0, y, x]):
                predicted_series[:, y, x] = X_global @ out_coeffs[:, y, x]

    print(f"\nTotal pixels with confirmed structural change: {np.sum(change_mask)}")
    
    print(f"Saving CCDC Maps to {OUTPUT_H5}...")
    with h5py.File(OUTPUT_H5, 'w') as out_f:
        out_f.create_dataset('coefficients', data=out_coeffs, compression='gzip')
        out_f.create_dataset('rmse', data=out_rmse, compression='gzip')
        out_f.create_dataset('change_mask', data=change_mask, compression='gzip')
        out_f.create_dataset('change_date_frac_year', data=change_date, compression='gzip')
        out_f.create_dataset('predicted_baseline', data=predicted_series, compression='gzip')
        
        if spatial_ref is not None:
            out_f.attrs['spatial_ref'] = spatial_ref
        if geo_transform is not None:
            out_f.attrs['GeoTransform'] = geo_transform

if __name__ == "__main__":
    main()