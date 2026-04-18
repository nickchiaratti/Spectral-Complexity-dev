import os
import shutil
import h5py
import numpy as np
import tkinter as tk
from tkinter import filedialog
import SpecComplex as sc
import warnings
from datetime import datetime, timezone
import concurrent.futures
import multiprocessing

# ==========================================
# 1. CONFIGURATION
# ==========================================
TILE_SIZE = 3          
SLIDING_STRIDE = 1      
Z_SCORE_WINDOW_SIZE = 11

NUM_ENDMEMBERS = 7
NORM_PARAM = 'bandCount'
MASKING = True 

# ==========================================
# 2. WORKER FUNCTION (PARALLEL EXECUTION)
# ==========================================
def compute_frame_metrics(payload):
    """
    Independent worker function. Bypasses the GIL by executing entirely 
    in its own process space. Reads directly from the untouched source HDF5 
    to prevent Windows SWMR (Single Writer Multiple Reader) file lock crashes.
    """
    orig_filepath = payload['orig_filepath']
    grid_name = payload['grid_name']
    t_local = payload['t_local']
    sensor_type = payload['sensor_type']
    red_idx = payload['red_idx']
    nir_idx = payload['nir_idx']
    swir_idx = payload['swir_idx']
    num_bands = payload['num_bands']
    height = payload['height']
    width = payload['width']
    
    # 1. Read Data (Independent File Handle)
    with h5py.File(orig_filepath, 'r') as h5_in:
        data_grp = h5_in[f"/HDFEOS/GRIDS/{grid_name}/Data Fields"]
        frame_sr = data_grp["surface_reflectance"][t_local, ...]
        frame_mask = data_grp["common_mask"][t_local, ...]

        if sensor_type == "TANAGER":
            gw_mask = data_grp["surface_reflectance"].attrs.get("all_good_wavelengths")[t_local].astype(bool)
        else:
            gw_mask = None

    # 2. Core Computations
    valid_mask = frame_mask == 1 if MASKING else np.ones((height, width), dtype=bool)

    ndvi = sc.calc_ndvi_frame(frame_sr, red_idx=red_idx, nir_idx=nir_idx)
    ndbi = sc.calc_ndbi_frame(frame_sr, swir_idx=swir_idx, nir_idx=nir_idx)

    # Strict physical band pruning for Tanager
    if sensor_type == "TANAGER":
        frame_sr = np.delete(frame_sr, np.where(~gw_mask), axis=0)

    endmembers, endmember_idx, vol_curve = sc.process_volume_frame(frame_sr, NUM_ENDMEMBERS, 'minEndmember', NORM_PARAM)
    
    slide_map = sc.process_volume_sliding_tile(frame_sr, TILE_SIZE, SLIDING_STRIDE, NUM_ENDMEMBERS, 'minEndmember', NORM_PARAM)
    msd_map = sc.process_msd_sliding_tile(frame_sr, TILE_SIZE, SLIDING_STRIDE)
    z_map = sc.calculate_global_z_score(slide_map, valid_mask)

    # 3. Package Results
    if sensor_type == "TANAGER":
        em_full = np.full((num_bands, NUM_ENDMEMBERS), np.nan, dtype=np.float32)
        em_full[gw_mask==1, :] = endmembers
        em_out = em_full
    else:
        em_out = endmembers

    return {
        'global_idx': payload['global_idx'],
        'grid_name': grid_name,
        't_local': t_local,
        'mask': frame_mask,
        'ndvi': ndvi,
        'ndbi': ndbi,
        'slide': slide_map,
        'msd': msd_map,
        'z_map': z_map,
        'em': em_out,
        'em_idx': endmember_idx,
        'vol': vol_curve
    }

# ==========================================
# 3. FILE MANAGEMENT & I/O
# ==========================================
def process_ard_cube(filepath):
    print(f"Loading ARD Master Cube: {filepath}")
    
    suffix = f"_SC_EM-{NUM_ENDMEMBERS}_Norm-{NORM_PARAM}"
    out_path = filepath.replace(".h5", f"{suffix}.h5")
    
    print(f"Cloning to Target File: {out_path}")
    shutil.copy2(filepath, out_path)

    # Pass the original (read-only) filepath to the workers to prevent write-lock collisions
    with h5py.File(out_path, 'r+') as h5_out:
        if '/HDFEOS/GRIDS' not in h5_out:
            raise ValueError(f"CRITICAL ERROR: No /HDFEOS/GRIDS group found in {filepath}. Not a valid ARD Cube.")
            
        print("\n" + "="*50)
        print("Initializing Chronological Multi-Sensor Fusion (Parallel)")
        print("="*50)
        process_global_timeline(h5_out, filepath)

    print(f"\nMulti-Sensor Spectral Complexity Calculation Complete.\nSaved to: {out_path}")

def overwrite_dset(data_grp, name, shape, dtype='float32', spatial_ref=None, geo_transform=None, chunks=None, **kwargs):
    """
    Safely overwrites datasets. 
    EVIDENCE-BASED OPTIMIZATION: Now strictly enforces requested spatial chunking and 
    optimized compression to prevent I/O write thrashing.
    """
    if name in data_grp:
        del data_grp[name]
    
    ds = data_grp.create_dataset(name, shape=shape, dtype=dtype, compression="gzip", compression_opts=4, chunks=chunks, **kwargs)
    
    if spatial_ref is not None: ds.attrs['spatial_ref'] = spatial_ref
    if geo_transform is not None: ds.attrs['GeoTransform'] = geo_transform
        
    return ds

# ==========================================
# 4. MASTER THREAD DISPATCHER
# ==========================================
def process_global_timeline(h5_out, orig_filepath):
    grids = [g for g in h5_out['/HDFEOS/GRIDS'].keys() if g != 'HARMONIZED']
    if not grids:
        raise ValueError("CRITICAL ERROR: No sensor grids found to process.")

    # 1. Build Global Timeline
    timeline = []
    for grid in grids:
        base_path = f"/HDFEOS/GRIDS/{grid}/Data Fields"
        if base_path not in h5_out:
            raise ValueError(f"CRITICAL ERROR: Data Fields missing for {grid}")
            
        data_grp = h5_out[base_path]
        
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

    timeline.sort(key=lambda x: x['time'])
    total_frames = len(timeline)
    print(f"Global Timeline Established: {total_frames} frames across {len(grids)} sensors.")

    # 2. Extract Master Geometric Provenance
    ref_sr = h5_out[f"/HDFEOS/GRIDS/{grids[0]}/Data Fields/surface_reflectance"]
    _, _, height, width = ref_sr.shape
    spatial_ref = ref_sr.attrs.get('spatial_ref')
    geo_transform = ref_sr.attrs.get('GeoTransform')

    # EVIDENCE-BASED OPTIMIZATION: Spatial Chunking
    chunk_h, chunk_w = min(height, 256), min(width, 256)
    chunks_3d = (1, chunk_h, chunk_w)

    # 3. Initialize HARMONIZED Data Group
    harm_path = '/HDFEOS/GRIDS/HARMONIZED/Data Fields'
    if harm_path in h5_out:
        harm_grp = h5_out[harm_path]
    else:
        if '/HDFEOS/GRIDS/HARMONIZED' not in h5_out:
            h5_out.create_group('/HDFEOS/GRIDS/HARMONIZED')
        harm_grp = h5_out.create_group(harm_path)

    # 4. Pre-allocate Unified Analytical Datasets
    ds_harm_mask = overwrite_dset(harm_grp, 'common_mask', (total_frames, height, width), dtype='uint8', spatial_ref=spatial_ref, geo_transform=geo_transform, chunks=chunks_3d)
    ds_harm_slide = overwrite_dset(harm_grp, 'sliding_volume_map', (total_frames, height, width), spatial_ref=spatial_ref, geo_transform=geo_transform, chunks=chunks_3d)
    ds_harm_ndvi = overwrite_dset(harm_grp, 'ndvi_map', (total_frames, height, width), spatial_ref=spatial_ref, geo_transform=geo_transform, chunks=chunks_3d)
    ds_harm_ndbi = overwrite_dset(harm_grp, 'ndbi_map', (total_frames, height, width), spatial_ref=spatial_ref, geo_transform=geo_transform, chunks=chunks_3d)
    ds_harm_msd = overwrite_dset(harm_grp, 'msd_map', (total_frames, height, width), spatial_ref=spatial_ref, geo_transform=geo_transform, chunks=chunks_3d)
    ds_harm_z = overwrite_dset(harm_grp, 'sliding_volume_z_score', (total_frames, height, width), spatial_ref=spatial_ref, geo_transform=geo_transform, chunks=chunks_3d)

    # 5. Pre-allocate Sensor-Specific Output Datasets
    sensor_dsets = {}
    for grid in grids:
        data_grp = h5_out[f"/HDFEOS/GRIDS/{grid}/Data Fields"]
        sr_shape = data_grp["surface_reflectance"].shape
        n_frames, n_bands = sr_shape[0], sr_shape[1]
        
        em_ds = overwrite_dset(data_grp, 'frame_endmembers', (n_frames, n_bands, NUM_ENDMEMBERS), spatial_ref=spatial_ref, geo_transform=geo_transform, chunks=(1, n_bands, NUM_ENDMEMBERS))
        idx_ds = overwrite_dset(data_grp, 'frame_endmember_indices', (n_frames, NUM_ENDMEMBERS), dtype='int32', chunks=(1, NUM_ENDMEMBERS))
        vol_ds = overwrite_dset(data_grp, 'frame_endmember_volumes', (n_frames, NUM_ENDMEMBERS), chunks=(1, NUM_ENDMEMBERS))
        
        sensor_dsets[grid] = {'em': em_ds, 'idx': idx_ds, 'vol': vol_ds, 'num_bands': n_bands}

    # 6. Generate Parallel Payloads
    print(f"Spooling {total_frames} frames into Parallel Compute Cluster...")
    payloads = []
    for global_idx, meta in enumerate(timeline):
        grid_name = meta['grid']
        
        if "HLSS30" in grid_name:
            red_idx, nir_idx, swir_idx = 3, 7, 11
            sensor_type = "HLS"
        elif "HLSL30" in grid_name:
            red_idx, nir_idx, swir_idx = 3, 4, 5
            sensor_type = "HLS"
        elif "TANAGER" in grid_name:
            red_idx, nir_idx, swir_idx = 59, 97, 244
            sensor_type = "TANAGER"
        else:
            raise ValueError(f"CRITICAL ERROR: Unrecognized Sensor Grid Architecture: {grid_name}")

        payloads.append({
            'orig_filepath': orig_filepath,
            'global_idx': global_idx,
            'grid_name': grid_name,
            't_local': meta['local_idx'],
            'sensor_type': sensor_type,
            'red_idx': red_idx,
            'nir_idx': nir_idx,
            'swir_idx': swir_idx,
            'num_bands': sensor_dsets[grid_name]['num_bands'],
            'height': height,
            'width': width
        })

    # 7. Execute Multi-Sensor Fusion Loop (Temporal Threading)
    completed_frames = 0
    max_workers = max(1, multiprocessing.cpu_count() - 2) # Leave 2 cores for OS overhead
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all futures
        future_to_idx = {executor.submit(compute_frame_metrics, p): p['global_idx'] for p in payloads}
        
        for future in concurrent.futures.as_completed(future_to_idx):
            global_idx = future_to_idx[future]
            result = future.result() 
            
            grid_name = result['grid_name']
            t_local = result['t_local']
            
            # Write returned metrics sequentially into the locked output file
            ds_harm_mask[global_idx, ...] = result['mask']
            ds_harm_ndvi[global_idx, ...] = result['ndvi']
            ds_harm_ndbi[global_idx, ...] = result['ndbi']
            ds_harm_slide[global_idx, ...] = result['slide']
            ds_harm_msd[global_idx, ...] = result['msd']
            ds_harm_z[global_idx, ...] = result['z_map']
            
            sensor_dsets[grid_name]['em'][t_local, ...] = result['em']
            sensor_dsets[grid_name]['idx'][t_local, ...] = result['em_idx']
            sensor_dsets[grid_name]['vol'][t_local, ...] = result['vol']
            
            completed_frames += 1
            print(f"  [{completed_frames}/{total_frames}] Completed computation for {grid_name} (Global Index {global_idx})")

    # ==========================================
    # 8. APPLY STRICT DATA PROVENANCE ATTRIBUTES
    # ==========================================
    print("\nApplying Absolute Data Provenance Attributes...")
    dt_str = h5py.string_dtype(encoding='ascii')
    
    prov_grid = np.array([m['grid'] for m in timeline], dtype=dt_str)
    prov_space = np.array([m['spacecraft'] for m in timeline], dtype=dt_str)
    prov_time = np.array([m['time'] for m in timeline], dtype='float64')
    prov_idx = np.array([m['local_idx'] for m in timeline], dtype='int32')
    
    for ds in [ds_harm_mask, ds_harm_slide, ds_harm_ndvi, ds_harm_ndbi, ds_harm_msd, ds_harm_z]:
        ds.attrs.create('source_grid', data=prov_grid)
        ds.attrs.create('source_spacecraft', data=prov_space)
        ds.attrs['acquisition_time'] = prov_time
        ds.attrs['source_frame_index'] = prov_idx
        
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
    # The Windows if __name__ == '__main__' guard ensures the ProcessPoolExecutor 
    # child processes do not accidentally spawn infinite Tkinter dialog windows.
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