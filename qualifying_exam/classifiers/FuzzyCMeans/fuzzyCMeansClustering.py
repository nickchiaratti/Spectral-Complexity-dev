import os
import shutil
import h5py
import numpy as np
import tkinter as tk
from tkinter import filedialog
from datetime import datetime, timezone
import warnings
import skfuzzy as fuzz
from scipy import ndimage

# --- Configuration ---
MIN_CLUSTERS = 4
MAX_CLUSTERS = 10
FUZZY_EXPONENT = 1.5
MAX_FCM_ITERATIONS = 150
FCM_TOLERANCE = 1e-4

# FCM Training Sample Size (limits memory usage during cluster center calculation)
TRAIN_SAMPLE_SIZE = 10000000 

# Flexibility for changing features
# FEATURES_TO_USE = ['NDVI', 'NDBI', 'SWIR1', 'NIR', 'DOY_sin', 'DOY_cos', 'complexity']
FEATURES_TO_USE = ['NDBI', 'complexity']
STANDARDIZE_FEATURES = True

# Target Wavelengths (µm)
TARGET_WLS = {
    'RED': 0.655,
    'NIR': 0.865,
    'SWIR1': 1.609
}

# --- Pixel Masking Configuration ---
SUN_ELEVATION_THRESHOLD = 30
CLOUD_DILATION = 2

# TANAGER Masking
TANAGER_AEROSOL_DEPTH_THRESHOLD = 0.3
TANAGER_SR_UNCERTAINTY_THRESHOLD = 0.05

# LANDSAT Masking
QA_REJECT_MASK = 0b111111
RADSAT_ACCEPT_VALUE = 0
AEROSOL_ACCEPT_VALUES = [2, 4, 32, 66, 68, 96, 100, 130, 132, 160, 164] # 'medium' level

def find_closest_band(wavelengths, target_wl):
    """Finds the index of the closest wavelength."""
    if np.nanmax(wavelengths) > 100:
        target_wl = target_wl * 1000.0
    return (np.abs(wavelengths - target_wl)).argmin()

def _get_landsat_mask(data_grp, num_frames, height, width):
    """Generates a boolean mask for LANDSAT data based on active filters."""
    valid_mask = np.ones((num_frames, height, width), dtype=bool)
    sun_elev_arr = data_grp['surface_reflectance'].attrs.get('sun_elevation')
    kernel = np.ones((3, 3), dtype=bool)

    for f_idx in range(num_frames):
        if sun_elev_arr is not None and f_idx < len(sun_elev_arr):
            if sun_elev_arr[f_idx] < SUN_ELEVATION_THRESHOLD:
                valid_mask[f_idx] = False
                continue

        # QA Reject Mask
        if 'QUALITY_L1_PIXEL' in data_grp:
            qa_pixel = data_grp['QUALITY_L1_PIXEL'][f_idx, ...]
            bad_qa_mask = (qa_pixel & QA_REJECT_MASK) != 0
            if CLOUD_DILATION > 0:
                bad_qa_mask = ndimage.binary_dilation(bad_qa_mask, structure=kernel, iterations=CLOUD_DILATION)
            valid_mask[f_idx] &= ~bad_qa_mask

        # RADSAT Accept Value
        if 'RADIOMETRIC_SATURATION' in data_grp:
            bad_radsat = data_grp['RADIOMETRIC_SATURATION'][f_idx, ...] != RADSAT_ACCEPT_VALUE
            if CLOUD_DILATION > 0:
                bad_radsat = ndimage.binary_dilation(bad_radsat, structure=kernel, iterations=CLOUD_DILATION)
            valid_mask[f_idx] &= ~bad_radsat

        # Aerosol Accept Values
        if 'QUALITY_L2_AEROSOL' in data_grp:
            aerosol = data_grp['QUALITY_L2_AEROSOL'][f_idx, ...]
            invalid_aerosol = ~np.isin(aerosol, AEROSOL_ACCEPT_VALUES)
            if CLOUD_DILATION > 0:
                invalid_aerosol = ndimage.binary_dilation(invalid_aerosol, structure=kernel, iterations=CLOUD_DILATION)
            valid_mask[f_idx] &= ~invalid_aerosol

    return valid_mask

def _get_tanager_mask(data_grp, num_frames, height, width):
    """Generates a boolean mask for TANAGER data based on active filters."""
    valid_mask = np.ones((num_frames, height, width), dtype=bool)
    kernel = np.ones((3, 3), dtype=bool)
    gw_mask_all = data_grp['surface_reflectance'].attrs.get('all_good_wavelengths')

    for f_idx in range(num_frames):
        # Cloud Mask Check
        if 'beta_cloud_mask' in data_grp:
            cloud_mask = (data_grp['beta_cloud_mask'][f_idx, ...] == 1)
            cirrus_mask = (data_grp['beta_cirrus_mask'][f_idx, ...] == 1)
            combined_cloud = cloud_mask | cirrus_mask
            if CLOUD_DILATION > 0:
                combined_cloud = ndimage.binary_dilation(combined_cloud, structure=kernel, iterations=CLOUD_DILATION)
            valid_mask[f_idx] &= ~combined_cloud
        
        # Sun Elevation Check
        if 'sun_zenith' in data_grp:
            zenith = data_grp['sun_zenith'][f_idx, ...]
            valid_mask[f_idx] &= ((90.0 - zenith) >= SUN_ELEVATION_THRESHOLD)
            
        # Aerosol Optical Depth Check
        if 'aerosol_optical_depth' in data_grp:
            bad_aod_mask = data_grp['aerosol_optical_depth'][f_idx, ...] >= TANAGER_AEROSOL_DEPTH_THRESHOLD
            if TANAGER_AEROSOL_DEPTH_THRESHOLD > 0 and CLOUD_DILATION > 0:
                bad_aod_mask = ndimage.binary_dilation(bad_aod_mask, structure=kernel, iterations=CLOUD_DILATION)
            valid_mask[f_idx] &= ~bad_aod_mask
            
        # Surface Reflectance Uncertainty Check
        if 'surface_reflectance_uncertainty' in data_grp and gw_mask_all is not None:
            valid_bands = gw_mask_all[f_idx].astype(bool)
            unc_mask = np.nanmax(data_grp['surface_reflectance_uncertainty'][f_idx, valid_bands, ...], axis=0) >= TANAGER_SR_UNCERTAINTY_THRESHOLD
            if TANAGER_SR_UNCERTAINTY_THRESHOLD > 0 and CLOUD_DILATION > 0:
                unc_mask = ndimage.binary_dilation(unc_mask, structure=kernel, iterations=CLOUD_DILATION)
            valid_mask[f_idx] &= ~unc_mask
            
    return valid_mask

def process_file(filepath):
    print(f"\nProcessing: {filepath}")
    
    with h5py.File(filepath, 'r') as h5:
        grid_name = list(h5['/HDFEOS/GRIDS'].keys())[0]
        base_fields_path = f"/HDFEOS/GRIDS/{grid_name}/Data Fields"
        data_grp = h5[base_fields_path]
            
        ds_surfRef = data_grp["surface_reflectance"]
        ds_complexity = data_grp["sliding_volume_map"]
        num_frames, num_bands, height, width = ds_surfRef.shape
        
        # --- 1. Extract Wavelengths and Identify Bands ---
        wl = ds_surfRef.attrs['wavelengths'][:]
        if grid_name == 'TANAGER':
                wl = wl / 1000.0
        idx_red = find_closest_band(wl, TARGET_WLS['RED'])
        idx_nir = find_closest_band(wl, TARGET_WLS['NIR'])
        idx_swir1 = find_closest_band(wl, TARGET_WLS['SWIR1'])
            
        print(f"Using Band Indices -> RED: {idx_red}, NIR: {idx_nir}, SWIR1: {idx_swir1}")

        # --- 2. Generate Spatial Masks ---
        print("Generating spatial valid masks...")
        if grid_name == 'LANDSAT':
            spatial_mask = _get_landsat_mask(data_grp, num_frames, height, width)
        elif grid_name == 'TANAGER':
            spatial_mask = _get_tanager_mask(data_grp, num_frames, height, width)

        # --- 3. Load Data Frame-by-Frame (Minimizing Memory Footprint) ---
        print("Loading reflectance data into memory (applying masks)...")
        red = np.full((num_frames, height, width), np.nan, dtype=np.float32)
        nir = np.full((num_frames, height, width), np.nan, dtype=np.float32)
        swir1 = np.full((num_frames, height, width), np.nan, dtype=np.float32)
        complexity = np.full((num_frames, height, width), np.nan, dtype=np.float32)


        for f_idx in range(num_frames):
            frame_mask = spatial_mask[f_idx]
            if np.any(frame_mask):
                # Only load the relevant bands for this specific frame into memory
                frame_red = ds_surfRef[f_idx, idx_red, :, :]
                frame_nir = ds_surfRef[f_idx, idx_nir, :, :]
                frame_swir1 = ds_surfRef[f_idx, idx_swir1, :, :]
                frame_complexity = ds_complexity[f_idx, :, :]
                
                # Apply mask immediately to drop bad data
                red[f_idx, frame_mask] = frame_red[frame_mask]
                nir[f_idx, frame_mask] = frame_nir[frame_mask]
                swir1[f_idx, frame_mask] = frame_swir1[frame_mask]
                complexity[f_idx, frame_mask] = frame_complexity[frame_mask]
        
        print("Calculating NDVI and NDBI...")
        with np.errstate(divide='ignore', invalid='ignore'):
            ndvi = (nir - red) / (nir + red)
            ndbi = (swir1 - nir) / (swir1 + nir)
            
        # Clean invalid math results
        ndvi = np.nan_to_num(ndvi, nan=0.0, posinf=1.0, neginf=-1.0)
        ndbi = np.nan_to_num(ndbi, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Enforce NaNs on masked areas for clustering calculations downstream
        ndvi[~spatial_mask] = np.nan
        ndbi[~spatial_mask] = np.nan

        # --- 4. Calculate Temporal Features (DOY) ---
        print("Extracting Temporal Features...")
        acq_times = ds_surfRef.attrs.get('acquisition_time')
        doy_sin = np.zeros(num_frames)
        doy_cos = np.zeros(num_frames)
        
        for t, ts in enumerate(acq_times):
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            doy = dt.timetuple().tm_yday
            doy_sin[t] = np.sin(2 * np.pi * doy / 365.25)
            doy_cos[t] = np.cos(2 * np.pi * doy / 365.25)
            
        doy_sin_grid = doy_sin[:, np.newaxis, np.newaxis] * np.ones((num_frames, height, width))
        doy_cos_grid = doy_cos[:, np.newaxis, np.newaxis] * np.ones((num_frames, height, width))

        # --- 5. Build Feature Set ---
        available_features = {
            'NDVI': ndvi,
            'NDBI': ndbi,
            'SWIR1': swir1,
            'NIR': nir,
            'RED': red,
            'DOY_sin': doy_sin_grid,
            'DOY_cos': doy_cos_grid,
            'complexity': complexity

        }
        
        feature_layers = [available_features[f] for f in FEATURES_TO_USE]
        feature_cube = np.stack(feature_layers, axis=-1)
        
        # --- 6. Valid Pixel Extraction ---
        # The mask naturally propagated through NaNs in the NIR band
        valid_mask_final = ~np.isnan(nir)
        X_valid = feature_cube[valid_mask_final]
        
    # --- END OF READ-ONLY BLOCK ---
        
    if STANDARDIZE_FEATURES:
        print("Standardizing features (Z-score)...")
        feature_means = np.nanmean(X_valid, axis=0)
        feature_stds = np.nanstd(X_valid, axis=0)
        feature_stds[feature_stds == 0] = 1.0 
        X_valid = (X_valid - feature_means) / feature_stds

    if X_valid.shape[0] < MIN_CLUSTERS:
        print("Not enough valid pixels to cluster. Aborting.")
        return

    # --- 7. Fuzzy C-Means Optimization (scikit-fuzzy) ---
    print(f"Optimizing Fuzzy C-Means (Testing c={MIN_CLUSTERS} to {MAX_CLUSTERS})...")
    sample_size = min(TRAIN_SAMPLE_SIZE, X_valid.shape[0])
    print(f"Using sample size {sample_size}.")
    idx_sample = np.random.choice(X_valid.shape[0], sample_size, replace=False)
    X_train = X_valid[idx_sample]
    
    best_fpc = -1
    best_c = -1
    best_centers = None
    
    for c in range(MIN_CLUSTERS, MAX_CLUSTERS + 1):
        cluster_centers, u, u0, d, jm, p, fpc = fuzz.cmeans(
            X_train.T, c=c, m=FUZZY_EXPONENT, 
            error=FCM_TOLERANCE, maxiter=MAX_FCM_ITERATIONS, init=None
        )
        print(f"  Tested c={c:02d} | FPC: {fpc:.4f}")
        if fpc > best_fpc:
            best_fpc = fpc
            best_c = c
            best_centers = cluster_centers
            
    NUM_CLUSTERS = best_c
    print(f"\nOptimal clusters identified: c={NUM_CLUSTERS} (FPC = {best_fpc:.4f})")
    
    print("Predicting memberships for all valid pixels...")
    u_pred, u0, d, jm, p, fpc = fuzz.cmeans_predict(
        X_valid.T, cntr_trained=best_centers, m=FUZZY_EXPONENT, 
        error=FCM_TOLERANCE, maxiter=MAX_FCM_ITERATIONS
    )
    U_pred = u_pred.T
    
    # --- 8. Reconstruct Maps ---
    full_memberships = np.full((num_frames, height, width, NUM_CLUSTERS), np.nan, dtype=np.float32)
    full_hard_clusters = np.full((num_frames, height, width), -1, dtype=np.int8) 
    
    full_memberships[valid_mask_final] = U_pred
    full_hard_clusters[valid_mask_final] = np.argmax(U_pred, axis=1)
    
    # --- 9. Save Results to HDF5 ---
    suffix = f"_FCM-Clusters-{NUM_CLUSTERS}"
    out_path = filepath.replace(".h5", f"{suffix}.h5")
    
    print(f"\nDuplicating to Output Path: {out_path}")
    shutil.copy2(filepath, out_path)

    print("Writing results to HDF5...")
    with h5py.File(out_path, 'r+') as h5:
        if 'fcm_memberships' in h5[base_fields_path]: del h5[f"{base_fields_path}/fcm_memberships"]
        if 'fcm_hard_clusters' in h5[base_fields_path]: del h5[f"{base_fields_path}/fcm_hard_clusters"]
        
        ds_mem = h5[base_fields_path].create_dataset(
            'fcm_memberships', data=full_memberships, compression="gzip"
        )
        ds_hard = h5[base_fields_path].create_dataset(
            'fcm_hard_clusters', data=full_hard_clusters, compression="gzip", fillvalue=-1
        )
        
        if STANDARDIZE_FEATURES:
            readable_centers = (best_centers * feature_stds) + feature_means
        else:
            readable_centers = best_centers
            
        if 'fcm_cluster_centers' in h5[base_fields_path]: del h5[f"{base_fields_path}/fcm_cluster_centers"]
        ds_cen = h5[base_fields_path].create_dataset('fcm_cluster_centers', data=readable_centers)

        # Attributes
        ds_mem.attrs['description'] = "Fuzzy C-Means membership probabilities per cluster."
        ds_mem.attrs['num_clusters'] = NUM_CLUSTERS
        ds_mem.attrs['features'] = np.array(FEATURES_TO_USE, dtype='S')
        ds_mem.attrs['fuzzy_exponent_m'] = FUZZY_EXPONENT
        ds_mem.attrs['cloud_dilation'] = CLOUD_DILATION
        ds_mem.attrs['fpc_score'] = best_fpc
        
        ds_hard.attrs['description'] = "Hard classifications derived from argmax of FCM memberships."
        ds_hard.attrs['num_clusters'] = NUM_CLUSTERS
        ds_cen.attrs['features'] = np.array(FEATURES_TO_USE, dtype='S')

    print(f"\nClustering Complete! File saved successfully to:\n{out_path}")

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    
    print("Please select the HDF5 Image Stack...")
    file_path = filedialog.askopenfilename(
        title="Select HDF5 Image Stack",
        filetypes=[("HDF5 files", "*.h5")]
    )
    
    if file_path:
        process_file(file_path)
    else:
        print("No file selected.")
    
    root.destroy()