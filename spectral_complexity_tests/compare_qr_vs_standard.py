import os
import sys
import time
import platform
import warnings
import h5py
import numpy as np

# Monkeypatch platform._wmi_query to raise OSError immediately, bypassing Windows WMI hangs/KeyErrors in multiprocessing child processes
def _dummy_wmi_query(*args, **kwargs):
    raise OSError("WMI disabled to prevent hangs")
platform._wmi_query = _dummy_wmi_query

# Add current directory to path to ensure modules can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import SpecComplex as sc
import SpecComplexQR as scQR

# Configuration based on HLST_SC_calculations.py defaults
TILE_SIZE = 3        
SLIDING_STRIDE = 1      
NUM_ENDMEMBERS = 7
GRAM_TYPE = 'minEndmember'
NORM_PARAM = 'bandCount'

DATA_PATH = r"C:\satelliteImagery\HLS30\HLS_Tait_STAC_Native_2025.h5"

def main():
    print(f"===========================================================")
    print(f" Validation: SpecComplex (Standard) vs SpecComplexQR (QR)")
    print(f"===========================================================")
    
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Sample data file not found at {DATA_PATH}")
        return

    print(f"Loading sample data from: {DATA_PATH}")
    
    total_time_standard = 0
    total_time_qr = 0
    global_max_rel_diff = 0
    global_mean_rel_diff_sum = 0
    global_frames_evaluated = 0
    global_valid_pixels = 0
    pixels_over_tol_total = 0
    total_pixels_compared = 0
    tolerance = 0.01 # 1%

    # Load data similarly to HLST_SC_calculations.py
    with h5py.File(DATA_PATH, 'r') as h5_in:
        grids = [g for g in h5_in['/HDFEOS/GRIDS'].keys() if g != 'HARMONIZED']
        if not grids:
            print("ERROR: No sensor grids found in the HDF5 file.")
            return
            
        for grid_name in grids:
            data_grp = h5_in[f"/HDFEOS/GRIDS/{grid_name}/Data Fields"]
            num_frames = data_grp["surface_reflectance"].shape[0]
            print(f"\nProcessing Grid: {grid_name} ({num_frames} frames)")
            
            for t_local in range(num_frames):
                print(f"  Frame {t_local+1}/{num_frames}...")
                frame_sr = data_grp["surface_reflectance"][t_local, ...]
                
                # --- 1. Running Standard Method (SpecComplex.py) ---
                start_time = time.perf_counter()
                with warnings.catch_warnings(), np.errstate(all='ignore'):
                    warnings.simplefilter("ignore")
                    map_standard = sc.process_volume_sliding_tile(
                        frame_sr, TILE_SIZE, SLIDING_STRIDE, NUM_ENDMEMBERS, GRAM_TYPE, NORM_PARAM
                    )
                time_standard = time.perf_counter() - start_time
                total_time_standard += time_standard

                # --- 2. Running QR Method (SpecComplexQR.py) ---
                start_time = time.perf_counter()
                with warnings.catch_warnings(), np.errstate(all='ignore'):
                    warnings.simplefilter("ignore")
                    map_qr = scQR.process_volume_sliding_tile(
                        frame_sr, TILE_SIZE, SLIDING_STRIDE, NUM_ENDMEMBERS, GRAM_TYPE, NORM_PARAM
                    )
                time_qr = time.perf_counter() - start_time
                total_time_qr += time_qr
                
                # --- 3. Validate Differences ---
                valid_mask_std = ~np.isnan(map_standard)
                valid_mask_qr = ~np.isnan(map_qr)
                
                common_valid = valid_mask_std & valid_mask_qr
                
                if np.sum(common_valid) == 0:
                    continue
                    
                std_vals = map_standard[common_valid]
                qr_vals = map_qr[common_valid]
                
                abs_diff = np.abs(std_vals - qr_vals)
                
                # Relative difference: |A - B| / max(|A|, |B|, epsilon)
                epsilon = 1e-10
                denominator = np.maximum(np.maximum(np.abs(std_vals), np.abs(qr_vals)), epsilon)
                rel_diff = abs_diff / denominator
                
                max_rel_diff = np.max(rel_diff)
                global_max_rel_diff = max(global_max_rel_diff, max_rel_diff)
                
                global_mean_rel_diff_sum += np.sum(rel_diff)
                global_valid_pixels += len(rel_diff)
                
                pixels_over_tol = np.sum(rel_diff > tolerance)
                pixels_over_tol_total += pixels_over_tol
                total_pixels_compared += len(rel_diff)
                global_frames_evaluated += 1

    print(f"\n===========================================================")
    print(f" FINAL SUMMARY ({global_frames_evaluated} frames evaluated)")
    print(f"===========================================================")
    print(f"Total Standard Time : {total_time_standard:.2f} seconds")
    print(f"Total QR Time       : {total_time_qr:.2f} seconds")
    if total_time_qr > 0:
        print(f"Overall Speedup     : {total_time_standard / total_time_qr:.2f}x faster")
        
    if global_valid_pixels > 0:
        overall_mean_rel_diff = global_mean_rel_diff_sum / global_valid_pixels
        print(f"Valid Pixels Eval   : {global_valid_pixels}")
        print(f"Mean Relative Diff  : {overall_mean_rel_diff * 100:.6f}%")
        print(f"Max Relative Diff   : {global_max_rel_diff * 100:.6f}%")
        
        if global_max_rel_diff <= tolerance:
            print(f"\nSUCCESS: The results are equivalent across all frames! Maximum relative difference is less than {tolerance*100}%.")
        else:
            print(f"\nWARNING: Some pixels have a relative difference greater than {tolerance*100}%.")
            print(f"Pixels over {tolerance*100}% tolerance: {pixels_over_tol_total} out of {total_pixels_compared} ({pixels_over_tol_total/total_pixels_compared*100:.4f}%)")
    else:
        print("\nERROR: No valid pixels found across any frames.")

if __name__ == '__main__':
    # Required for safe multiprocessing execution in Windows
    import multiprocessing
    multiprocessing.freeze_support()
    main()
