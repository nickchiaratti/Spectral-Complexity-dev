import os
import shutil
import h5py
import numpy as np
import tkinter as tk
from tkinter import filedialog
import SpecComplex as sc
import warnings
from datetime import datetime, timezone

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
# Set to True to enforce the pre-calculated 'common_mask' stored natively in the ARD Cube.
# Set to False to run calculations on all pixels (ignoring clouds, shadows, etc.)
MASKING = True 

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
            
        print("\n" + "="*50)
        print("Initializing Chronological Multi-Sensor Fusion")
        print("="*50)
        process_global_timeline(h5)

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
# 3. CORE CALCULATION ENGINE
# ==========================================
def process_global_timeline(h5):
    """
    Constructs a unified temporal index across all available sensors, extracting and 
    fusing the derived arrays into a single chronologically sequenced HARMONIZED group.
    """
    grids = [g for g in h5['/HDFEOS/GRIDS'].keys() if g != 'HARMONIZED']
    if not grids:
        raise ValueError("CRITICAL ERROR: No sensor grids found to process.")

    # 1. Build Global Timeline
    timeline = []
    for grid in grids:
        base_path = f"/HDFEOS/GRIDS/{grid}/Data Fields"
        if base_path not in h5:
            raise ValueError(f"CRITICAL ERROR: Data Fields missing for {grid}")
            
        data_grp = h5[base_path]
        
        # Enforce strict dataset availability requirements
        for req_ds in ["surface_reflectance", "common_mask"]:
            if req_ds not in data_grp:
                raise ValueError(f"CRITICAL ERROR: '{req_ds}' missing in {grid}. Ensure ARD Harmonizer ran successfully.")
                
        acq_times = data_grp["surface_reflectance"].attrs['acquisition_time']
        spacecraft_ids = data_grp["surface_reflectance"].attrs['spacecraft_id']
        
        for i, ts in enumerate(acq_times):
            sp_id = spacecraft_ids[i]
            sp_str = sp_id.decode('utf-8') if isinstance(sp_id, bytes) else str(sp_id)
            
            timeline.append({
                'time': ts,
                'grid': grid,
                'local_idx': i,
                'spacecraft': sp_str
            })

    # Sort strictly by UTC acquisition time
    timeline.sort(key=lambda x: x['time'])
    total_frames = len(timeline)
    print(f"Global Timeline Established: {total_frames} frames across {len(grids)} sensors.")

    # 2. Extract Master Geometric Provenance
    ref_sr = h5[f"/HDFEOS/GRIDS/{grids[0]}/Data Fields/surface_reflectance"]
    _, _, height, width = ref_sr.shape
    spatial_ref = ref_sr.attrs.get('spatial_ref')
    geo_transform = ref_sr.attrs.get('GeoTransform')

    # 3. Initialize HARMONIZED Data Group
    harm_path = '/HDFEOS/GRIDS/HARMONIZED/Data Fields'
    if harm_path in h5:
        harm_grp = h5[harm_path]
    else:
        if '/HDFEOS/GRIDS/HARMONIZED' not in h5:
            h5.create_group('/HDFEOS/GRIDS/HARMONIZED')
        harm_grp = h5.create_group(harm_path)

    # 4. Pre-allocate Unified Analytical Datasets
    ds_harm_mask = overwrite_dset(harm_grp, 'common_mask', (total_frames, height, width), dtype='uint8', spatial_ref=spatial_ref, geo_transform=geo_transform)
    ds_harm_slide = overwrite_dset(harm_grp, 'sliding_volume_map', (total_frames, height, width), spatial_ref=spatial_ref, geo_transform=geo_transform)
    ds_harm_ndvi = overwrite_dset(harm_grp, 'ndvi_map', (total_frames, height, width), spatial_ref=spatial_ref, geo_transform=geo_transform)
    ds_harm_ndbi = overwrite_dset(harm_grp, 'ndbi_map', (total_frames, height, width), spatial_ref=spatial_ref, geo_transform=geo_transform)
    ds_harm_msd = overwrite_dset(harm_grp, 'msd_map', (total_frames, height, width), spatial_ref=spatial_ref, geo_transform=geo_transform)
    ds_harm_z = overwrite_dset(harm_grp, 'sliding_volume_z_score', (total_frames, height, width), spatial_ref=spatial_ref, geo_transform=geo_transform)

    # 5. Pre-allocate Sensor-Specific Output Datasets
    # Endmembers remain in native grids due to varying band-dimension physics
    sensor_dsets = {}
    for grid in grids:
        data_grp = h5[f"/HDFEOS/GRIDS/{grid}/Data Fields"]
        sr_shape = data_grp["surface_reflectance"].shape
        n_frames, n_bands = sr_shape[0], sr_shape[1]
        
        em_ds = overwrite_dset(data_grp, 'frame_endmembers', (n_frames, n_bands, NUM_ENDMEMBERS), spatial_ref=spatial_ref, geo_transform=geo_transform)
        idx_ds = overwrite_dset(data_grp, 'frame_endmember_indices', (n_frames, NUM_ENDMEMBERS), dtype='int32')
        vol_ds = overwrite_dset(data_grp, 'frame_endmember_volumes', (n_frames, NUM_ENDMEMBERS))
        
        sensor_dsets[grid] = {'em': em_ds, 'idx': idx_ds, 'vol': vol_ds}

    # 6. Execute Multi-Sensor Fusion Loop
    for global_idx, meta in enumerate(timeline):
        grid_name = meta['grid']
        t_local = meta['local_idx']
        
        dt_str = datetime.fromtimestamp(meta['time'], tz=timezone.utc).strftime('%Y-%m-%d %H:%M')
        print(f"  [{global_idx+1}/{total_frames}] Processing {grid_name} [{dt_str}]...")
        
        data_grp = h5[f"/HDFEOS/GRIDS/{grid_name}/Data Fields"]
        frame_sr = data_grp["surface_reflectance"][t_local, ...]
        frame_mask = data_grp["common_mask"][t_local, ...]
        
        # Sensor Physics Mapping for specific indices
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
            gw_mask = data_grp["surface_reflectance"].attrs.get("all_good_wavelengths").astype(bool)
        else:
            raise ValueError(f"CRITICAL ERROR: Unrecognized Sensor Grid Architecture: {grid_name}")

        # Store Common Mask into Harmonized Cube
        ds_harm_mask[global_idx, ...] = frame_mask
        
        valid_mask = frame_mask == 1 if MASKING else np.ones((height, width), dtype=bool)

        # Compute & Store Unified Analytical Arrays
        ds_harm_ndvi[global_idx, ...] = sc.calc_ndvi_frame(frame_sr, red_idx=red_idx, nir_idx=nir_idx)
        ds_harm_ndbi[global_idx, ...] = sc.calc_ndbi_frame(frame_sr, swir_idx=swir_idx, nir_idx=nir_idx)

        # Prune dead bands for mathematical volume extraction
        if sensor_type == "TANAGER":
            frame_sr = np.delete(frame_sr, np.where(~gw_mask[t_local]), axis=0)

        # Execute Core Complexity Mathematics
        endmembers, endmember_idx, vol_curve = sc.process_volume_frame(frame_sr, NUM_ENDMEMBERS, 'minEndmember', NORM_PARAM)
        
        slide_map = sc.process_volume_sliding_tile(frame_sr, TILE_SIZE, SLIDING_STRIDE, NUM_ENDMEMBERS, 'minEndmember', NORM_PARAM)
        ds_harm_slide[global_idx, ...] = slide_map
        ds_harm_msd[global_idx, ...] = sc.process_msd_sliding_tile(frame_sr, TILE_SIZE, SLIDING_STRIDE)
        ds_harm_z[global_idx, ...] = sc.calculate_global_z_score(slide_map, valid_mask)

        # Store Sensor-Specific Outputs
        if sensor_type == "TANAGER":
            em_full = np.full((data_grp["surface_reflectance"].shape[1], NUM_ENDMEMBERS), np.nan, dtype=np.float32)
            em_full[gw_mask[t_local]==1, :] = endmembers
            sensor_dsets[grid_name]['em'][t_local, ...] = em_full
        else:
            sensor_dsets[grid_name]['em'][t_local, ...] = endmembers

        sensor_dsets[grid_name]['idx'][t_local, ...] = endmember_idx
        sensor_dsets[grid_name]['vol'][t_local, ...] = vol_curve

    # ==========================================
    # 7. APPLY STRICT DATA PROVENANCE ATTRIBUTES
    # ==========================================
    print("\nApplying Absolute Data Provenance Attributes...")
    dt_str = h5py.string_dtype(encoding='ascii')
    
    prov_grid = np.array([m['grid'] for m in timeline], dtype=dt_str)
    prov_space = np.array([m['spacecraft'] for m in timeline], dtype=dt_str)
    prov_time = np.array([m['time'] for m in timeline], dtype='float64')
    prov_idx = np.array([m['local_idx'] for m in timeline], dtype='int32')
    
    # Apply to all harmonized datasets
    for ds in [ds_harm_mask, ds_harm_slide, ds_harm_ndvi, ds_harm_ndbi, ds_harm_msd, ds_harm_z]:
        ds.attrs.create('source_grid', data=prov_grid)
        ds.attrs.create('source_spacecraft', data=prov_space)
        ds.attrs['acquisition_time'] = prov_time
        ds.attrs['source_frame_index'] = prov_idx
        
        # Add normalization lineage tracking
        if ds.name.endswith('sliding_volume_map'):
            ds.attrs['description'] = f"Volume of convex hull within sliding {TILE_SIZE}x{TILE_SIZE} tile"
            ds.attrs['tile_size'] = TILE_SIZE
            ds.attrs['sliding_stride'] = SLIDING_STRIDE
            ds.attrs['gram_type'] = 'minEndmember'
            ds.attrs['num_endmembers'] = NUM_ENDMEMBERS
            ds.attrs['Normalization'] = NORM_PARAM if NORM_PARAM else "None"
        elif ds.name.endswith('msd_map'):
            ds.attrs['description'] = "Mean Spectral Distance (MSD) for each pixel"
            ds.attrs['tile_size'] = TILE_SIZE
            ds.attrs['sliding_stride'] = SLIDING_STRIDE
        elif ds.name.endswith('sliding_volume_z_score'):
            ds.attrs['description'] = "Global Spectral Complexity Z-score. ARD Masked pixels excluded from background stats."
            ds.attrs['MASKING_APPLIED'] = MASKING
            ds.attrs['MASK_SOURCE'] = "HARMONIZED_common_mask"

    # Attach Attributes to specific sensor outputs
    for grid in grids:
        dsets = sensor_dsets[grid]
        dsets['vol'].attrs['description'] = "Full volume curve (Volume vs Endmember Count) for entire frame"
        dsets['vol'].attrs['gram_type'] = 'minEndmember'
        dsets['vol'].attrs['num_endmembers'] = NUM_ENDMEMBERS
        dsets['vol'].attrs['Normalization'] = NORM_PARAM if NORM_PARAM else "None"
        
        dsets['em'].attrs['description'] = "Endmembers extracted for frame"
        dsets['em'].attrs['num_endmembers'] = NUM_ENDMEMBERS
        dsets['em'].attrs['Normalization'] = NORM_PARAM if NORM_PARAM else "None"
        dsets['idx'].attrs['description'] = "Spatial 1D indices for extracted endmembers"

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