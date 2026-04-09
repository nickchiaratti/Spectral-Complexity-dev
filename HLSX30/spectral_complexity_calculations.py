import os
import shutil
import h5py
import numpy as np
import tkinter as tk
from tkinter import filedialog
from scipy.ndimage import binary_dilation
import SpecComplex as sc
import warnings

# ==========================================
# 1. CONFIGURATION
# ==========================================
TILE_SIZE = 3          # Size of the window (NxN pixels) for volume calc
SLIDING_STRIDE = 1      # Stride for sliding window
Z_SCORE_WINDOW_SIZE = 11

# --- Parameters for Spectral Complexity ---
NUM_ENDMEMBERS = 7
NORM_PARAM = 'bandCount'

# --- Pixel Mask Configuration ---
MASKING = True
SUN_ELEVATION_THRESHOLD = 30
CLOUD_DILATION = 0

# HLS Specific Configuration (Unified Fmask for both S30 and L30)
# Bits 0-5: cirrus, cloud, adj cloud/shadow, cloud shadow, snow/ice, water
QA_REJECT_MASK = 0b111111 
AEROSOL_ACCEPT_LEVEL = 'medium' # 'low' (0-1), 'medium' (0-2), 'high' (0-3)

# TANAGER Specific Configuration
TANAGER_CLOUD_MASK = True
TANAGER_UNCERTAINTY_THRESHOLD = 0.1
TANAGER_AEROSOL_THRESHOLD = 0.3

# ==========================================
# 2. FILE MANAGEMENT & I/O
# ==========================================
def process_ard_cube(filepath):
    print(f"Loading ARD Master Cube: {filepath}")
    
    # Construct Output Filename
    suffix = f"_SC_EM-{NUM_ENDMEMBERS}_Norm-{NORM_PARAM}"
    out_path = filepath.replace(".h5", f"{suffix}.h5")
    
    print(f"Cloning to Target File: {out_path}")
    shutil.copy2(filepath, out_path)

    with h5py.File(out_path, 'r+') as h5:
        # Fails hard if ARD structure is corrupted
        if '/HDFEOS/GRIDS' not in h5:
            raise ValueError(f"CRITICAL ERROR: No /HDFEOS/GRIDS group found in {filepath}. Not a valid ARD Cube.")
            
        grids = list(h5['/HDFEOS/GRIDS'].keys())
        print(f"Detected {len(grids)} Harmonized Grids: {grids}")
        
        for grid_name in grids:
            print(f"\n{'='*40}\nInitializing Calculation Engine for {grid_name}\n{'='*40}")
            process_grid_stack(h5, grid_name)

    print(f"\nMulti-Sensor Spectral Complexity Calculation Complete.\nSaved to: {out_path}")

def overwrite_dset(data_grp, name, shape, dtype='float32', spatial_ref=None, geo_transform=None, **kwargs):
    """Safely overwrites datasets and automatically ports strict GIS georeferencing attributes."""
    if name in data_grp:
        del data_grp[name]
    
    ds = data_grp.create_dataset(name, shape=shape, dtype=dtype, compression="gzip", **kwargs)
    
    # ARD Compliance: Ensure output derived datasets map correctly in GIS software
    if spatial_ref is not None: ds.attrs['spatial_ref'] = spatial_ref
    if geo_transform is not None: ds.attrs['GeoTransform'] = geo_transform
        
    return ds
# ==========================================
# 4. CORE CALCULATION ENGINE
# ==========================================
def process_grid_stack(h5, grid_name):
    base_fields_path = f"/HDFEOS/GRIDS/{grid_name}/Data Fields"
    if base_fields_path not in h5:
        raise ValueError(f"CRITICAL ERROR: Data Fields missing for {grid_name}")
        
    data_grp = h5[base_fields_path]
    
    if "surface_reflectance" not in data_grp:
        raise ValueError(f"CRITICAL ERROR: 'surface_reflectance' missing in {grid_name}")
        
    ds_surfRef = data_grp["surface_reflectance"]
    num_frames, num_bands, height, width = ds_surfRef.shape
    
    spatial_ref = ds_surfRef.attrs.get('spatial_ref')
    geo_transform = ds_surfRef.attrs.get('GeoTransform')

    # Sensor Physics Mapping
    if "HLSS30" in grid_name:
        red_idx, nir_idx, swir_idx = 3, 7, 11
        sensor_type = "HLS"
        gw_mask = None
    elif "HLSL30" in grid_name:
        red_idx, nir_idx, swir_idx = 3, 4, 5
        sensor_type = "HLS"
        gw_mask = None
    elif "TANAGER" in grid_name:
        red_idx, nir_idx, swir_idx = 59, 97, 244
        sensor_type = "TANAGER"
        # Fails hard if Tanager quality flags are missing
        if "all_good_wavelengths" not in ds_surfRef.attrs:
             raise ValueError(f"CRITICAL ERROR: 'all_good_wavelengths' missing for {grid_name}")
        gw_mask = ds_surfRef.attrs.get("all_good_wavelengths").astype(bool)
    else:
        raise ValueError(f"CRITICAL ERROR: Unrecognized Sensor Grid Architecture: {grid_name}")

    # Initialize ARD Output Datasets
    ds_endmembers = overwrite_dset(data_grp, 'frame_endmembers', (num_frames, num_bands, NUM_ENDMEMBERS), spatial_ref=spatial_ref, geo_transform=geo_transform)
    ds_endmember_indices = overwrite_dset(data_grp, 'frame_endmember_indices', (num_frames, NUM_ENDMEMBERS), dtype='int32')
    ds_vol_curve = overwrite_dset(data_grp, 'frame_endmember_volumes', (num_frames, NUM_ENDMEMBERS))
    ds_slide = overwrite_dset(data_grp, 'sliding_volume_map', (num_frames, height, width), spatial_ref=spatial_ref, geo_transform=geo_transform)
    ds_ndvi = overwrite_dset(data_grp, 'ndvi_map', (num_frames, height, width), spatial_ref=spatial_ref, geo_transform=geo_transform)
    ds_ndbi = overwrite_dset(data_grp, 'ndbi_map', (num_frames, height, width), spatial_ref=spatial_ref, geo_transform=geo_transform)
    ds_msd = overwrite_dset(data_grp, 'msd_map', (num_frames, height, width), spatial_ref=spatial_ref, geo_transform=geo_transform)
    ds_slideZ = overwrite_dset(data_grp, 'sliding_volume_z_score', (num_frames, height, width), spatial_ref=spatial_ref, geo_transform=geo_transform)

    for t in range(num_frames):
        print(f"  [{grid_name}] Processing Frame {t+1}/{num_frames} ...")
        frame_sr = ds_surfRef[t, ...]
        
        # EVIDENCE-BASED FIX: Calculate Spectral Indices BEFORE Pruning
        # Calculating on the raw array ensures the physical red/nir/swir wavelengths 
        # map exactly to the un-shifted indices.
        ds_ndvi[t, ...] = sc.calc_ndvi_frame(frame_sr, red_idx=red_idx, nir_idx=nir_idx)
        ds_ndbi[t, ...] = sc.calc_ndbi_frame(frame_sr, swir_idx=swir_idx, nir_idx=nir_idx)

        # Generate spatial mask utilizing unified SpecComplex functions
        if MASKING:
            if sensor_type == "HLS":
                valid_mask = sc.get_hls_mask(data_grp, t, 
                                          sun_elevation_threshold=SUN_ELEVATION_THRESHOLD,
                                          cloud_dilation=CLOUD_DILATION,
                                          qa_reject_mask=QA_REJECT_MASK,
                                          aerosol_accept_level=AEROSOL_ACCEPT_LEVEL)
            elif sensor_type == "TANAGER":
                valid_mask = sc.get_tanager_mask(data_grp, t, (height, width),
                                                 sun_elevation_threshold=SUN_ELEVATION_THRESHOLD,
                                                 cloud_dilation=CLOUD_DILATION,
                                                 apply_cloud_mask=TANAGER_CLOUD_MASK,
                                                 uncertainty_threshold=TANAGER_UNCERTAINTY_THRESHOLD,
                                                 aerosol_depth_threshold=TANAGER_AEROSOL_THRESHOLD)
        else:
            valid_mask = np.ones((height, width), dtype=bool)

        # Prune dead bands for mathematical volume extraction
        if sensor_type == "TANAGER":
            frame_sr = np.delete(frame_sr, np.where(~gw_mask[t]), axis=0)

        # Execute Core Complexity Mathematics
        endmembers, endmember_idx, vol_curve = sc.process_volume_frame(frame_sr, NUM_ENDMEMBERS, 'minEndmember', NORM_PARAM)

        # Repopulate Tanager endmembers into the full spectral profile
        if sensor_type == "TANAGER":
            em_full = np.full((num_bands, NUM_ENDMEMBERS), np.nan, dtype=np.float32)
            em_full[gw_mask[t]==1, :] = endmembers
            ds_endmembers[t, ...] = em_full
        else:
            ds_endmembers[t, ...] = endmembers

        ds_endmember_indices[t, ...] = endmember_idx
        ds_vol_curve[t, ...] = vol_curve
        
        # Execute Spatial Sliding Maps
        ds_slide[t, ...] = sc.process_volume_sliding_tile(frame_sr, TILE_SIZE, SLIDING_STRIDE, NUM_ENDMEMBERS, 'minEndmember', NORM_PARAM)
        ds_msd[t, ...] = sc.process_msd_sliding_tile(frame_sr, TILE_SIZE, SLIDING_STRIDE)
        ds_slideZ[t, ...] = sc.calculate_global_z_score(ds_slide[t, ...], valid_mask)
            
    # --- Metadata Provenance Updates ---
    ds_vol_curve.attrs['description'] = "Full volume curve (Volume vs Endmember Count) for entire frame"
    ds_vol_curve.attrs['gram_type'] = 'minEndmember'
    ds_vol_curve.attrs['num_endmembers'] = NUM_ENDMEMBERS
    
    ds_endmembers.attrs['description'] = "Endmembers for each pixel"
    ds_endmembers.attrs['num_endmembers'] = NUM_ENDMEMBERS
    ds_endmember_indices.attrs['description'] = "Endmember indices for each pixel"
    
    norm_attr = NORM_PARAM if NORM_PARAM else "None"
    ds_endmembers.attrs['Normalization'] = norm_attr
    ds_vol_curve.attrs['Normalization'] = norm_attr
    ds_slide.attrs['Normalization'] = norm_attr
    
    ds_slide.attrs['description'] = f"Volume of convex hull of spectral data within each sliding {TILE_SIZE}x{TILE_SIZE} tile"
    ds_slide.attrs['tile_size'] = TILE_SIZE
    ds_slide.attrs['sliding_stride'] = SLIDING_STRIDE
    ds_slide.attrs['gram_type'] = 'minEndmember'
    ds_slide.attrs['num_endmembers'] = NUM_ENDMEMBERS
    
    ds_msd.attrs['description'] = "MSD for each pixel"
    ds_msd.attrs['tile_size'] = TILE_SIZE
    ds_msd.attrs['sliding_stride'] = SLIDING_STRIDE
    
    ds_slideZ.attrs['description'] = "Global Spectral Complexity Z-score. Sensor-masked pixels excluded from background stats."
    ds_slideZ.attrs['MASKING_APPLIED'] = MASKING
    ds_slideZ.attrs['SUN_ELEVATION_THRESHOLD'] = SUN_ELEVATION_THRESHOLD
    ds_slideZ.attrs['CLOUD_DILATION'] = CLOUD_DILATION
    
    ds_ndvi.attrs['description'] = "NDVI for each pixel"
    ds_ndvi.attrs['red_idx'] = red_idx
    ds_ndvi.attrs['nir_idx'] = nir_idx
    
    ds_ndbi.attrs['description'] = "NDBI for each pixel"
    ds_ndbi.attrs['swir_idx'] = swir_idx
    ds_ndbi.attrs['nir_idx'] = nir_idx
    
    if sensor_type == "HLS":
        ds_slideZ.attrs['QA_REJECT_MASK'] = QA_REJECT_MASK
        ds_slideZ.attrs['AEROSOL_ACCEPT_LEVEL'] = AEROSOL_ACCEPT_LEVEL
    elif sensor_type == "TANAGER":
        ds_slideZ.attrs['TANAGER_CLOUD_MASK'] = TANAGER_CLOUD_MASK
        ds_slideZ.attrs['TANAGER_UNCERTAINTY_THRESHOLD'] = TANAGER_UNCERTAINTY_THRESHOLD
        ds_slideZ.attrs['TANAGER_AEROSOL_THRESHOLD'] = TANAGER_AEROSOL_THRESHOLD

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    
    print("Please select the ARD Master Grid HDF5 Cube...")
    file_path = tk.filedialog.askopenfilename(
        title="Select ARD Master Grid HDF5 Cube",
        filetypes=[("HDF5 files", "*.h5")]
    )
    
    if file_path:
        process_ard_cube(file_path)
    else:
        print("No file selected. Exiting.")
    
    root.destroy()