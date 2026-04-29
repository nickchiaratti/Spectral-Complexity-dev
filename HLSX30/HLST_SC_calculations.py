'''
Calculates derivative metrics for HDFEOS compliant grids. Stores calculated values in a 
combined HARMONIZED grid to produce virtual constellation results. 
'''
import os
import shutil
import h5py
import numpy as np
import tkinter as tk
from tkinter import filedialog
import SpecComplex as sc
import SpecComplexQR as scQR
import warnings
import time
from datetime import datetime, timezone
import concurrent.futures
import multiprocessing

# ==========================================
# 1. CONFIGURATION & FEATURE TOGGLES
# ==========================================
TILE_SIZE = 2         
SLIDING_STRIDE = 1      
#Z_SCORE_WINDOW_SIZE = 11

NUM_ENDMEMBERS = 4
CUSTOM_SUFFIX = '_2x2'
NORM_PARAM = 'bandCount'
MASKING = True 

# --- ANALYTICAL FEATURE TOGGLES ---
# Set to True to calculate and overwrite. Set to False to skip processing and 
# retain existing datasets in the ARD cube.
CALC_NDVI = False
CALC_NDBI = False
CALC_MSD = False
CALC_GLOBAL_ENDMEMBERS = False
CALC_SLIDING_VOLUME = True
CALC_Z_SCORE = True

# ==========================================
# 2. WORKER FUNCTION (PARALLEL EXECUTION WITH TELEMETRY)
# ==========================================
def compute_frame_metrics(payload):
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
    flags = payload['calc_flags']
    
    telemetry = {}
    t_start_total = time.perf_counter()
    
    with warnings.catch_warnings(), np.errstate(all='ignore'):
        warnings.simplefilter("ignore")
        
        # --- 1. Data Ingestion ---
        t0 = time.perf_counter()
        with h5py.File(orig_filepath, 'r') as h5_in:
            data_grp = h5_in[f"/HDFEOS/GRIDS/{grid_name}/Data Fields"]
            frame_sr = data_grp["surface_reflectance"][t_local, ...]
            frame_mask = data_grp["common_mask"][t_local, ...]

            if sensor_type == "TANAGER":
                gw_mask = data_grp["surface_reflectance"].attrs.get("all_good_wavelengths")[t_local].astype(bool)
            else:
                gw_mask = None
                
            # Dependency Resolution: Read existing volume map if Z-Score requires it but we aren't calculating it
            if flags['z_score'] and not flags['volume']:
                slide_map = h5_in["/HDFEOS/GRIDS/HARMONIZED/Data Fields/sliding_volume_map"][payload['global_idx'], ...]
            else:
                slide_map = None
                
        telemetry['I/O_Read'] = time.perf_counter() - t0
        valid_mask = frame_mask == 1 if MASKING else np.ones((height, width), dtype=bool)

        # --- 2. Spectral Indices ---
        ndvi, ndbi = None, None
        if flags['ndvi'] or flags['ndbi']:
            t0 = time.perf_counter()
            if flags['ndvi']: ndvi = sc.calc_ndvi_frame(frame_sr, red_idx=red_idx, nir_idx=nir_idx)
            if flags['ndbi']: ndbi = sc.calc_ndbi_frame(frame_sr, swir_idx=swir_idx, nir_idx=nir_idx)
            telemetry['Spectral_Indices'] = time.perf_counter() - t0

        # --- 3. Global Endmembers ---
        endmembers, endmember_idx, vol_curve, em_out = None, None, None, None
        if flags['endmembers']:
            t0 = time.perf_counter()
            # Tanager PRUNING specifically for Global Endmember calculation
            eval_sr = np.delete(frame_sr, np.where(~gw_mask), axis=0) if sensor_type == "TANAGER" else frame_sr
            endmembers, endmember_idx, vol_curve = sc.process_volume_frame(eval_sr, NUM_ENDMEMBERS, 'minEndmember', NORM_PARAM)
            
            if sensor_type == "TANAGER":
                em_full = np.full((num_bands, NUM_ENDMEMBERS), np.nan, dtype=np.float32)
                em_full[gw_mask==1, :] = endmembers
                em_out = em_full
            else:
                em_out = endmembers
            telemetry['Global_Endmembers'] = time.perf_counter() - t0
        
        # --- 4. Sliding Window Complexity ---
        if flags['volume']:
            t0 = time.perf_counter()
            # Note: process_volume_sliding_tile prunes invalid pixels internally
            slide_map = scQR.process_volume_sliding_tile(frame_sr, TILE_SIZE, SLIDING_STRIDE, NUM_ENDMEMBERS, 'minEndmember', NORM_PARAM)
            telemetry['Sliding_Volume_Map'] = time.perf_counter() - t0
        
        # --- 5. Mean Spectral Distance ---
        msd_map = None
        if flags['msd']:
            t0 = time.perf_counter()
            msd_map = sc.process_msd_sliding_tile(frame_sr, TILE_SIZE, SLIDING_STRIDE)
            telemetry['MSD_Map'] = time.perf_counter() - t0
        
        # --- 6. Global Z-Score ---
        z_map = None
        if flags['z_score']:
            t0 = time.perf_counter()
            z_map = sc.calculate_global_z_score(slide_map, valid_mask)
            telemetry['Z_Score'] = time.perf_counter() - t0

        telemetry['Total_Worker_Time'] = time.perf_counter() - t_start_total

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
            'vol': vol_curve,
            'telemetry': telemetry
        }

# ==========================================
# 3. FILE MANAGEMENT & I/O
# ==========================================
def process_ard_cube(filepath):
    print(f"\nEvaluating Initialization Directives for: {filepath}")
    suffix = f"_SC_EM-{NUM_ENDMEMBERS}_Norm-{NORM_PARAM}{CUSTOM_SUFFIX}"
    
    # OS Thread-Lock Isolation: If updating an existing file, create a strictly 
    # read-only cache for the parallel workers to prevent Windows SWMR crashes.
    worker_cache_path = None
    
    if suffix in filepath:
        out_path = filepath
        worker_cache_path = filepath.replace(".h5", "_worker_read_cache.h5")
        print(f"Modifying Existing Pipeline Output.")
        print(f"Cloning read-only cache for parallel workers: {worker_cache_path}")
        shutil.copy2(filepath, worker_cache_path)
    else:
        out_path = filepath.replace(".h5", f"{suffix}.h5")
        worker_cache_path = filepath
        print(f"New Pipeline Run. Cloning to Target Output File: {out_path}")
        shutil.copy2(filepath, out_path)

    with h5py.File(out_path, 'r+') as h5_out:
        if '/HDFEOS/GRIDS' not in h5_out:
            raise ValueError(f"CRITICAL ERROR: No /HDFEOS/GRIDS group found. Not a valid ARD Cube.")
            
        print("\n" + "="*50)
        print("Initializing Chronological Multi-Sensor Fusion (Parallel)")
        print("="*50)
        process_global_timeline(h5_out, worker_cache_path)

    # Cleanup the throwaway cache file
    if suffix in filepath and worker_cache_path and os.path.exists(worker_cache_path):
        os.remove(worker_cache_path)
        print("\nCleaned up throwaway worker cache.")

    print(f"Multi-Sensor Analytical Processing Complete.\nSaved to: {out_path}")

def overwrite_dset(data_grp, name, shape, dtype='float32', spatial_ref=None, geo_transform=None, chunks=None, **kwargs):
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
    global CALC_SLIDING_VOLUME
    
    grids = [g for g in h5_out['/HDFEOS/GRIDS'].keys() if g != 'HARMONIZED']
    if not grids:
        raise ValueError("CRITICAL ERROR: No sensor grids found to process.")

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
            timeline.append({'time': ts, 'grid': grid, 'local_idx': i, 'spacecraft': sp_str})

    timeline.sort(key=lambda x: x['time'])
    total_frames = len(timeline)
    print(f"Global Timeline Established: {total_frames} frames across {len(grids)} sensors.")

    # Mathematical Dependency Resolution
    harm_path = '/HDFEOS/GRIDS/HARMONIZED/Data Fields'
    if CALC_Z_SCORE and not CALC_SLIDING_VOLUME:
        if harm_path in h5_out and 'sliding_volume_map' in h5_out[harm_path]:
            print("  -> Dependency Resolved: 'sliding_volume_map' found in existing file. Bypassing volume calculation.")
        else:
            print("  -> Dependency Warning: Z-score requested but 'sliding_volume_map' not found in file. Forcing CALC_SLIDING_VOLUME = True.")
            CALC_SLIDING_VOLUME = True

    ref_sr = h5_out[f"/HDFEOS/GRIDS/{grids[0]}/Data Fields/surface_reflectance"]
    _, _, height, width = ref_sr.shape
    spatial_ref = ref_sr.attrs.get('spatial_ref')
    geo_transform = ref_sr.attrs.get('GeoTransform')

    chunk_h, chunk_w = min(height, 256), min(width, 256)
    chunks_3d = (1, chunk_h, chunk_w)

    if harm_path in h5_out:
        harm_grp = h5_out[harm_path]
    else:
        h5_out.create_group('/HDFEOS/GRIDS/HARMONIZED')
        harm_grp = h5_out.create_group(harm_path)

    # Pre-allocate ONLY requested datasets
    ds_harm_mask = overwrite_dset(harm_grp, 'common_mask', (total_frames, height, width), dtype='uint8', spatial_ref=spatial_ref, geo_transform=geo_transform, chunks=chunks_3d)
    ds_harm_slide = overwrite_dset(harm_grp, 'sliding_volume_map', (total_frames, height, width), spatial_ref=spatial_ref, geo_transform=geo_transform, chunks=chunks_3d) if CALC_SLIDING_VOLUME else None
    ds_harm_ndvi = overwrite_dset(harm_grp, 'ndvi_map', (total_frames, height, width), spatial_ref=spatial_ref, geo_transform=geo_transform, chunks=chunks_3d) if CALC_NDVI else None
    ds_harm_ndbi = overwrite_dset(harm_grp, 'ndbi_map', (total_frames, height, width), spatial_ref=spatial_ref, geo_transform=geo_transform, chunks=chunks_3d) if CALC_NDBI else None
    ds_harm_msd = overwrite_dset(harm_grp, 'msd_map', (total_frames, height, width), spatial_ref=spatial_ref, geo_transform=geo_transform, chunks=chunks_3d) if CALC_MSD else None
    ds_harm_z = overwrite_dset(harm_grp, 'sliding_volume_z_score', (total_frames, height, width), spatial_ref=spatial_ref, geo_transform=geo_transform, chunks=chunks_3d) if CALC_Z_SCORE else None

    sensor_dsets = {}
    if CALC_GLOBAL_ENDMEMBERS:
        for grid in grids:
            data_grp = h5_out[f"/HDFEOS/GRIDS/{grid}/Data Fields"]
            sr_shape = data_grp["surface_reflectance"].shape
            n_frames, n_bands = sr_shape[0], sr_shape[1]
            
            em_ds = overwrite_dset(data_grp, 'frame_endmembers', (n_frames, n_bands, NUM_ENDMEMBERS), spatial_ref=spatial_ref, geo_transform=geo_transform, chunks=(1, n_bands, NUM_ENDMEMBERS))
            idx_ds = overwrite_dset(data_grp, 'frame_endmember_indices', (n_frames, NUM_ENDMEMBERS), dtype='int32', chunks=(1, NUM_ENDMEMBERS))
            vol_ds = overwrite_dset(data_grp, 'frame_endmember_volumes', (n_frames, NUM_ENDMEMBERS), chunks=(1, NUM_ENDMEMBERS))
            
            sensor_dsets[grid] = {'em': em_ds, 'idx': idx_ds, 'vol': vol_ds, 'num_bands': n_bands}

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
            'num_bands': sensor_dsets[grid_name]['num_bands'] if CALC_GLOBAL_ENDMEMBERS else 0,
            'height': height,
            'width': width,
            'calc_flags': {
                'ndvi': CALC_NDVI, 'ndbi': CALC_NDBI, 'msd': CALC_MSD,
                'endmembers': CALC_GLOBAL_ENDMEMBERS, 'volume': CALC_SLIDING_VOLUME, 'z_score': CALC_Z_SCORE
            }
        })

    # Dynamic Telemetry Aggregator
    agg_telemetry = {'Total_Worker_Time': []}

    completed_frames = 0
    max_workers = 2
    
    print(f"Spooling {total_frames} frames into Compute Cluster (Max Workers: {max_workers})...")
    t_start_pipeline = time.perf_counter()
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {executor.submit(compute_frame_metrics, p): p['global_idx'] for p in payloads}
        
        for future in concurrent.futures.as_completed(future_to_idx):
            global_idx = future_to_idx[future]
            result = future.result() 
            
            grid_name = result['grid_name']
            t_local = result['t_local']
            
            # Dynamically aggregate telemetry
            for key, val in result['telemetry'].items():
                if key not in agg_telemetry: agg_telemetry[key] = []
                agg_telemetry[key].append(val)
            
            # Write selectively based on configuration
            ds_harm_mask[global_idx, ...] = result['mask']
            if CALC_NDVI: ds_harm_ndvi[global_idx, ...] = result['ndvi']
            if CALC_NDBI: ds_harm_ndbi[global_idx, ...] = result['ndbi']
            if CALC_MSD: ds_harm_msd[global_idx, ...] = result['msd']
            if CALC_SLIDING_VOLUME: ds_harm_slide[global_idx, ...] = result['slide']
            if CALC_Z_SCORE: ds_harm_z[global_idx, ...] = result['z_map']
            
            if CALC_GLOBAL_ENDMEMBERS:
                sensor_dsets[grid_name]['em'][t_local, ...] = result['em']
                sensor_dsets[grid_name]['idx'][t_local, ...] = result['em_idx']
                sensor_dsets[grid_name]['vol'][t_local, ...] = result['vol']
            
            completed_frames += 1
            print(f"  [{completed_frames}/{total_frames}] {grid_name} (Global Index {global_idx}) processed in {result['telemetry']['Total_Worker_Time']:.2f}s")

    t_end_pipeline = time.perf_counter()

    # ==========================================
    # PERFORMANCE METRICS (APM OUTPUT)
    # ==========================================
    print("\n" + "="*50)
    print("ALGORITHMIC PERFORMANCE PROFILE (Mean per Frame)")
    print("="*50)
    
    total_mean = np.mean(agg_telemetry['Total_Worker_Time'])
    for key in agg_telemetry:
        if key == 'Total_Worker_Time': continue
        mean_time = np.mean(agg_telemetry[key])
        pct = (mean_time / total_mean) * 100
        print(f" {key.ljust(22)} : {mean_time:7.3f} sec  ({pct:5.1f}%)")
        
    print(f"\n Total Compute Time       : {(t_end_pipeline - t_start_pipeline)/60:.2f} minutes")
    print(f" Equivalent Serial Time   : {(total_mean * total_frames)/60:.2f} minutes")
    print("="*50)

    # ==========================================
    # 8. APPLY STRICT DATA PROVENANCE ATTRIBUTES
    # ==========================================
    print("\nApplying Absolute Data Provenance Attributes...")
    dt_str = h5py.string_dtype(encoding='ascii')
    
    prov_grid = np.array([m['grid'] for m in timeline], dtype=dt_str)
    prov_space = np.array([m['spacecraft'] for m in timeline], dtype=dt_str)
    prov_time = np.array([m['time'] for m in timeline], dtype='float64')
    prov_idx = np.array([m['local_idx'] for m in timeline], dtype='int32')
    
    # Only tag the datasets that were actually generated in this run
    created_harm_dsets = [ds for ds in [ds_harm_mask, ds_harm_slide, ds_harm_ndvi, ds_harm_ndbi, ds_harm_msd, ds_harm_z] if ds is not None]
    
    for ds in created_harm_dsets:
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

    if CALC_GLOBAL_ENDMEMBERS:
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
    
    print("Please select the HLST ARD Master Grid HDF5 Cube...")
    file_path = tk.filedialog.askopenfilename(
        title="Select HLST ARD Master Grid HDF5 Cube",
        filetypes=[("HDF5 files", "*.h5")]
    )
    
    if file_path:
        process_ard_cube(file_path)
    else:
        print("No file selected. Exiting.")
    
    root.destroy()