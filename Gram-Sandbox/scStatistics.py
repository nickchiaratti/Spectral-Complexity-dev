import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from datetime import datetime, timezone
import tkinter as tk
from tkinter import filedialog

# ==========================================
# 1. CONFIGURATION
# ==========================================

# Temporal Filtering
START_DATE = datetime(2015, 1, 1, tzinfo=timezone.utc)
END_DATE = datetime(2026, 12, 31, tzinfo=timezone.utc)

LANDSAT_MAX_SC = 7.128

# Combined Pixel Mask Configuration
SUN_ELEVATION_THRESHOLD = 30
CLOUD_DILATION = 2

# Tanager Pixel Mask Configuration
TANAGER_AEROSOL_DEPTH_THRESHOLD = 0.3
TANAGER_SR_UNCERTAINTY_THRESHOLD = 0.05

# LANDSAT Pixel Mask Configuration
QA_REJECT_MASK = 0b111111
RADSAT_ACCEPT_VALUE = 0
AEROSOL_ACCEPT_LEVEL = 'medium' #'low' 'medium' 'high'

AEROSOL_DICT = {
    'low': [2, 4, 32, 66, 68, 96, 100],
    'medium': [2, 4, 32, 66, 68, 96, 100, 130, 132, 160, 164],
    'high': [2, 4, 32, 66, 68, 96, 100, 130, 132, 160, 164, 192, 194, 196, 224, 228]
}

# ==========================================
# 2. MASKING FUNCTIONS (Mirrored from Viewer)
# ==========================================

def get_landsat_mask(data_grp, f_idx, shape):
    """Generates a boolean mask for LANDSAT data."""
    valid_mask = np.ones(shape, dtype=bool)
    
    sun_elev_arr = data_grp['surface_reflectance'].attrs.get('sun_elevation')
    if f_idx < len(sun_elev_arr):
        if sun_elev_arr[f_idx] < SUN_ELEVATION_THRESHOLD:
            return np.zeros(shape, dtype=bool)

    if 'QUALITY_L1_PIXEL' in data_grp:
        qa_pixel = data_grp['QUALITY_L1_PIXEL'][f_idx, ...]
        bad_qa_mask = (qa_pixel & QA_REJECT_MASK) != 0
        if CLOUD_DILATION > 0:
            kernel = np.ones((3, 3), dtype=bool)
            bad_qa_mask = ndimage.binary_dilation(bad_qa_mask, structure=kernel, iterations=CLOUD_DILATION)
        valid_mask &= ~bad_qa_mask

    if 'RADIOMETRIC_SATURATION' in data_grp:
        bad_radsat = data_grp['RADIOMETRIC_SATURATION'][f_idx, ...] != RADSAT_ACCEPT_VALUE
        kernel = np.ones((3, 3), dtype=bool)
        bad_radsat = ndimage.binary_dilation(bad_radsat, structure=kernel, iterations=1)
        valid_mask &= ~bad_radsat

    if 'QUALITY_L2_AEROSOL' in data_grp:
        if AEROSOL_ACCEPT_LEVEL != 'all':
            aerosol = data_grp['QUALITY_L2_AEROSOL'][f_idx, ...]
            invalid_aerosol = ~np.isin(aerosol, AEROSOL_DICT[AEROSOL_ACCEPT_LEVEL])
            kernel = np.ones((3, 3), dtype=bool)
            invalid_aerosol = ndimage.binary_dilation(invalid_aerosol, structure=kernel, iterations=1)
            valid_mask &= ~invalid_aerosol

    return valid_mask

def get_tanager_mask(data_grp, f_idx, shape):
    """Generates a boolean mask for TANAGER data."""
    valid_mask = np.ones(shape, dtype=bool)

    if 'beta_cloud_mask' in data_grp:
        cloud_mask = (data_grp['beta_cloud_mask'][f_idx, ...]==1)
        cirrus_mask = (data_grp['beta_cirrus_mask'][f_idx, ...]==1)
        combined_cloud = cloud_mask | cirrus_mask
        if CLOUD_DILATION > 0:
            kernel = np.ones((3, 3), dtype=bool)
            combined_cloud = ndimage.binary_dilation(combined_cloud, structure=kernel, iterations=CLOUD_DILATION)
        valid_mask &= ~combined_cloud
    
    if 'sun_zenith' in data_grp:
        zenith = data_grp['sun_zenith'][f_idx, ...]
        valid_mask &= (zenith != -9999.0) & ((90.0 - zenith) >= SUN_ELEVATION_THRESHOLD)
        
    if 'aerosol_optical_depth' in data_grp:
        aod = data_grp['aerosol_optical_depth'][f_idx, ...]
        bad_aod_mask = (aod == -9999.0) | (aod >= TANAGER_AEROSOL_DEPTH_THRESHOLD)
        if TANAGER_AEROSOL_DEPTH_THRESHOLD > 0:
            kernel = np.ones((3, 3), dtype=bool)
            bad_aod_mask = ndimage.binary_dilation(bad_aod_mask, structure=kernel, iterations=1)
        valid_mask &= ~bad_aod_mask
        
    if 'surface_reflectance_uncertainty' in data_grp:
        gw_mask = data_grp['surface_reflectance'].attrs.get('all_good_wavelengths')
        if gw_mask is not None:
            valid_bands = gw_mask[f_idx].astype(bool)
            unc = np.nanmax(data_grp['surface_reflectance_uncertainty'][f_idx, valid_bands, ...], axis=0)
        else:
            unc = np.nanmax(data_grp['surface_reflectance_uncertainty'][f_idx, ...], axis=0)
            
        unc_mask = (unc == -9999.0) | (unc >= TANAGER_SR_UNCERTAINTY_THRESHOLD)
        if TANAGER_SR_UNCERTAINTY_THRESHOLD > 0:
            kernel = np.ones((3, 3), dtype=bool)
            unc_mask = ndimage.binary_dilation(unc_mask, structure=kernel, iterations=1)
        valid_mask &= ~unc_mask
        
    return valid_mask

# ==========================================
# 3. MAIN EXECUTION
# ==========================================

def main():
    root = tk.Tk()
    root.withdraw()
    print("Select an HDF5 file for statistical analysis...")
    file_path = filedialog.askopenfilename(title="Select HDF5 Dataset", filetypes=[("HDF5", "*.h5")])
    
    if not file_path:
        print("Selection cancelled.")
        return

    print(f"\nLoading: {os.path.basename(file_path)}")
    
    h5 = h5py.File(file_path, 'r')
    source_name = list(h5['/HDFEOS/GRIDS'].keys())[0]
    data_grp = h5[f'HDFEOS/GRIDS/{source_name}/Data Fields']
        
    vol_dset = data_grp['sliding_volume_map']
    #vol_dset = data_grp['sliding_volume_local_z_score']
    acq_times = data_grp['surface_reflectance'].attrs.get('acquisition_time')
    num_frames = vol_dset.shape[0]
    
    valid_frames = 0
    master_volumes = []

    print(f"Filtering frames between {START_DATE.strftime('%Y-%m-%d')} and {END_DATE.strftime('%Y-%m-%d')}...")

    for f_idx in range(num_frames):
        dt = datetime.fromtimestamp(acq_times[f_idx], tz=timezone.utc)
        
        if START_DATE <= dt <= END_DATE:
            shape = vol_dset[f_idx].shape
            
            # Generate appropriate mask
            if 'LANDSAT' in source_name.upper():
                mask = get_landsat_mask(data_grp, f_idx, shape)
            else:
                mask = get_tanager_mask(data_grp, f_idx, shape)
            
            # Extract valid frame data
            frame_data = vol_dset[f_idx, ...]
            
            # Combine Spatial Mask, NaN rejection, and ensure strictly positive values for Log10
            valid_pixels = frame_data[mask & ~np.isnan(frame_data) & (frame_data > 0)]
            
            if len(valid_pixels) > 0:
                master_volumes.append(valid_pixels)
                valid_frames += 1
                
    if not master_volumes:
        print("No valid pixels found in the specified date range.")
        return
        
    print("Aggregating array data... (This may take a moment for large stacks)")
    all_vols = np.concatenate(master_volumes)
    total_pixels = len(all_vols)
    print(f"Extracted {total_pixels:,} valid pixels across {valid_frames} frames.")

    # --- Statistical Calculations ---
    mean_v = np.mean(all_vols)
    med_v = np.median(all_vols)
    var_v = np.var(all_vols)
    
    # Mathematical transformation: log10(sc) - log10(mean(sc))
    log_mean = np.log10(mean_v)
    scaled_vols = np.log10(all_vols) - log_mean

    # --- Plotting ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.canvas.manager.set_window_title(f"Global Statistics: {source_name}")
    
    color_theme = 'tab:purple' if 'LANDSAT' in source_name.upper() else 'tab:orange'

    # Plot 1: Raw Volume Distribution (Log-Scaled Bins)
    bins_raw = np.logspace(np.log10(np.min(all_vols)), np.log10(np.max(all_vols)), 256)
    ax1.hist(all_vols, bins=bins_raw, color=color_theme, alpha=0.75, edgecolor='black', linewidth=0.2)
    ax1.set_xscale('log')
    ax1.set_title(f"Raw Spatial Volume Distribution\n({total_pixels:,} pixels)")
    ax1.set_xlabel("Complexity Volume (Log Scale)")
    ax1.set_ylabel("Pixel Frequency")
    ax1.grid(True, alpha=0.3, which="both", ls="--")
    
    stats_raw = (f"Dataset: {source_name}\n"
                 f"Frames Analyzed: {valid_frames}\n\n"
                 f"Mean: {mean_v:.4e}\n"
                 f"Median: {med_v:.4e}\n"
                 f"Variance: {var_v:.4e}")
    ax1.text(0.95, 0.95, stats_raw, transform=ax1.transAxes, ha='right', va='top', 
             fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    # Plot 2: Scaled Volume Distribution (Linear Bins)
    # Scaled data is linear in log-space, so linear bins are correct here.
    ax2.hist(scaled_vols, bins=256, color='tab:cyan', alpha=0.75, edgecolor='black', linewidth=0.2)
    ax2.set_title("Mean-Normalized Log-Ratio Distribution\nTransformation: log10(v) - log10(μ)")
    
    # Explicitly labeling the inverted ordinality for correct interpretation
    ax2.set_xlabel("Scaled Value")
    ax2.set_ylabel("Pixel Frequency")
    ax2.grid(True, alpha=0.3)
    
    stats_scaled = (f"Mean Scaled: {np.mean(scaled_vols):.4f}\n"
                    f"Median Scaled: {np.median(scaled_vols):.4f}\n"
                    f"Variance Scaled: {np.var(scaled_vols):.4f}")
    ax2.text(0.95, 0.95, stats_scaled, transform=ax2.transAxes, ha='right', va='top', 
             fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    plt.suptitle(f"Global Spectral Complexity Analysis | {START_DATE.strftime('%Y-%m-%d')} to {END_DATE.strftime('%Y-%m-%d')}\n{os.path.basename(file_path)}", fontsize=14)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()