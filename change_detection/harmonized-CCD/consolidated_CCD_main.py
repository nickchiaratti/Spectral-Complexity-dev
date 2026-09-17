import os
import sys
import h5py
import numpy as np
import json
from pathlib import Path

script_dir = Path(__file__).resolve().parent
if str(script_dir.parent) not in sys.path:
    sys.path.insert(0, str(script_dir.parent))
    
# Import original CCD main but override the data loading logic
from change_detection.harmonized_CCD.harmonized_CCD_main import *
import change_detection.harmonized_CCD.harmonized_CCD_main as orig

# We override the execute_ccd_pipeline function
def execute_ccd_pipeline(
         enable_const=ENABLE_CONSTANT, 
         enable_lin=ENABLE_LINEAR, 
         enable_quad=ENABLE_QUADRATIC, 
         temporal_periods=None,
         target_metric=orig.TARGET_METRIC,
         launch_vis=True):
    
    if temporal_periods is None:
        temporal_periods = orig.TEMPORAL_PERIODS
        
    min_samples = orig.MIN_SAMPLES
    
    _term_str = f"C{int(enable_const)}L{int(enable_lin)}Q{int(enable_quad)}"
    _period_str = f"P{len(temporal_periods)}"
    
    # Define paths
    H5_PATH = f"C:/satelliteImagery/MGRS30mConstellation/SC_Constellation_{orig.LOCATION}.h5"
    base_dir = os.path.dirname(H5_PATH)
    ccd_dir = os.path.join(base_dir, "CCD")
    os.makedirs(ccd_dir, exist_ok=True)
    
    output_h5 = os.path.join(ccd_dir, f"{orig.LOCATION}_CCD_Consolidated_{target_metric}_{_term_str}_{_period_str}.h5")

    print(f"Loading data from {H5_PATH} via index maps...")
    with h5py.File(H5_PATH, 'r') as f:
        timeline = f['/timeline/acquisition_time'][:]
        sensors = f['/timeline/source_sensor'][:]
        frame_indices = f['/timeline/source_frame_index'][:]
        
        target_h = f['target_grid'].attrs['height']
        target_w = f['target_grid'].attrs['width']
        geo_transform = f['target_grid'].attrs['GeoTransform']
        spatial_ref = f['target_grid'].attrs['spatial_ref']
        
        y_data = np.full((len(timeline), target_h, target_w), np.nan, dtype=np.float32)
        valid_mask = np.zeros((len(timeline), target_h, target_w), dtype=bool)
        
        for i, (sensor_csv, fidx_csv) in enumerate(zip(sensors, frame_indices)):
            sensor_keys = sensor_csv.decode('utf-8').split(',')
            fidxs = json.loads(fidx_csv.decode('utf-8'))
            
            for sk, fidx in zip(sensor_keys, fidxs):
                sg = f[f'sensors/{sk}']
                row_map = sg['row_map'][:]
                col_map = sg['col_map'][:]
                valid = sg['valid_mask'][:]
                
                try:
                    metric_ds = sg['derived_fields'][target_metric]
                except KeyError:
                    metric_ds = sg['data_fields'][target_metric]
                
                native_frame = metric_ds[fidx]
                
                # Apply scale if it's int16
                if 'scale_to_float' in metric_ds.attrs:
                    scale = metric_ds.attrs['scale_to_float']
                    fill = metric_ds.attrs.get('_FillValue', -9999)
                    nf_float = native_frame.astype(np.float32)
                    valid_nf = (native_frame != fill)
                    nf_float[valid_nf] *= scale
                    nf_float[~valid_nf] = np.nan
                    native_frame = nf_float
                
                native_mask = sg['data_fields']['common_mask'][fidx]
                
                # Map to target grid
                y_data[i, valid] = native_frame[row_map[valid], col_map[valid]]
                
                is_clear = (native_mask[row_map[valid], col_map[valid]] == 0)
                is_not_nan = ~np.isnan(y_data[i, valid])
                valid_mask[i, valid] = is_clear & is_not_nan

    num_frames = len(timeline)
    frac_years = orig.extract_fractional_years(timeline)
    
    sort_idx = np.argsort(timeline)
    acq_times = timeline[sort_idx]
    frac_years = frac_years[sort_idx]
    y_data = y_data[sort_idx, ...]
    valid_mask = valid_mask[sort_idx, ...]

    print(f"Dataset shape: {num_frames} frames, {target_h}x{target_w} pixels")

    X_full = orig.build_harmonic_matrix(frac_years, temporal_periods=temporal_periods,
                                   enable_const=enable_const, enable_lin=enable_lin, enable_quad=enable_quad)

    print("\nInitializing Output HDF5 on disk...")
    os.makedirs(os.path.dirname(output_h5), exist_ok=True)
    with h5py.File(output_h5, 'w') as f_out:
        orig._initialize_output_h5(f_out, num_frames, target_h, target_w, target_metric, geo_transform, spatial_ref, acq_times, frac_years)

    print(f"\nStarting Row-Chunked Processing... (Chunk Size: {orig.CHUNK_ROWS} rows)")
    import multiprocessing as mp
    
    total_chunks = int(np.ceil(target_h / orig.CHUNK_ROWS))
    manager = mp.Manager()
    progress_queue = manager.Queue()
    
    pool_args = []
    for chunk_idx in range(total_chunks):
        r_start = chunk_idx * orig.CHUNK_ROWS
        r_end = min(r_start + orig.CHUNK_ROWS, target_h)
        y_chunk = y_data[:, r_start:r_end, :]
        v_chunk = valid_mask[:, r_start:r_end, :]
        pool_args.append((chunk_idx, r_start, r_end, y_chunk, v_chunk, X_full, min_samples, progress_queue))

    import time
    start_time = time.time()
    
    with h5py.File(output_h5, 'r+') as f_out:
        with mp.Pool(processes=orig.NUM_WORKERS) as pool:
            results = pool.map_async(orig._process_row_chunk_wrapper, pool_args)
            
            completed = 0
            while not results.ready():
                while not progress_queue.empty():
                    msg = progress_queue.get()
                    completed += 1
                    orig.print_progress_bar(completed, total_chunks, prefix='Progress:', suffix='Complete', length=50)
                time.sleep(0.5)
                
            final_results = results.get()
            
            # Ensure final progress bar
            while not progress_queue.empty():
                progress_queue.get()
                completed += 1
            orig.print_progress_bar(total_chunks, total_chunks, prefix='Progress:', suffix='Complete', length=50)
            print()
            
            for chunk_idx, r_start, r_end, res_dict in final_results:
                orig._write_chunk_results(f_out, r_start, r_end, res_dict)

    elapsed = time.time() - start_time
    print(f"\nProcessing Complete in {elapsed/60:.2f} minutes.")
    print(f"Results saved to: {output_h5}")

    if launch_vis:
        print("\nSkipping automatic viewer launch for now.")

if __name__ == '__main__':
    orig.execute_ccd_pipeline = execute_ccd_pipeline
    orig.execute_ccd_pipeline()
