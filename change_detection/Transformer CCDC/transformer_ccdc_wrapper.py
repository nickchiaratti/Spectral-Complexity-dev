"""
Main Execution Wrapper for Transformer-based CCDC Analysis
Integrates multi-sensor data loading with temporal change detection.
"""

import os
import logging
import h5py
import numpy as np
import data_loader as DL
from transformer_ccdc import run_transformer_ccdc, extract_valid_pixel_history

# ==========================================
# 1. PIPELINE CONFIGURATION
# ==========================================
# Define a small spatial subset for initial testing
TARGET_ROWS = [70, 75]
TARGET_COLS = [20, 40]

LOCATION = "Tait"
LANDSAT_PATH = f"C:/satelliteImagery/LANDSAT/{LOCATION}/LANDSAT_Stack_{LOCATION}_GEE_2015_2025_WRS16_SC_EM-7_Gram-minEndmember_Norm-bandCount.h5"
TANAGER_PATH = f"C:/satelliteImagery/Tanager/{LOCATION}/Tanager_Stack_{LOCATION}_HDFEOS_SC_EM-7_Gram-minEndmember_Norm-bandCount.h5"

SAVE_DIR = f"C:/satelliteImagery/transformerModel/{LOCATION}"
MODEL_SAVE_DIR = os.path.join(SAVE_DIR, "models")

# CCDC Hyperparameters
WINDOW_YEARS = 3.0
K_CONSECUTIVE = 5
LAMBDA_THRESH = 3.0

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    logging.info("Initializing Continuous Change Detection Pipeline...")
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    
    # ==========================================
    # 2. DATA INGESTION
    # ==========================================
    logging.info("Loading merged multi-sensor datacube. This may take a moment...")
    cube, h5_l, h5_t = DL.load_merged_datacube(LANDSAT_PATH, TANAGER_PATH)
    logging.info(f"Datacube loaded successfully. Total frames: {len(cube)}")
    
    # Pre-calculate global time fractions to map detected changes back to the chronological datacube index
    start_time = cube[0]['datetime_utc']
    _, frame_cols, frame_rows = cube[0]['ortho_visual'].shape
    cube_time_fractions = np.array([
        (frame['datetime_utc'] - start_time).total_seconds() / (365.25 * 24 * 3600) 
        for frame in cube
    ])
    
    # ==========================================
    # 3. SPATIAL ITERATION & INFERENCE
    # ==========================================
    changed_pixels_mask = []  # To hold the structure: [time_index, x (col), y (row)]
    
    for r in range(TARGET_ROWS[0], TARGET_ROWS[1]):
        for c in range(TARGET_COLS[0], TARGET_COLS[1]):
            pixel_id = f"r{r}_c{c}"
            logging.info(f"--- Processing pixel {pixel_id} ---")
            
            try:
                # Extract strictly valid, unmasked temporal history. 
                # This will raise a ValueError if data is insufficient or corrupted (NaNs).
                times_ary, values_ary = extract_valid_pixel_history(cube, r, c)
                
                logging.info(f"Found {len(times_ary)} valid observations. Commencing Transformer inference...")
                
                # Execute the Transformer-CCDC logic
                changes = run_transformer_ccdc(
                    times=times_ary,
                    values=values_ary,
                    window_years=WINDOW_YEARS,
                    k_consecutive=K_CONSECUTIVE,
                    lambda_thresh=LAMBDA_THRESH,
                    save_model_dir=MODEL_SAVE_DIR,
                    pixel_id=pixel_id
                )
                
                if changes:
                    logging.info(f"Detected {len(changes)} structural changes for pixel {pixel_id}.")
                    for idx, change in enumerate(changes):
                        logging.info(f"  -> Change {idx+1}: Fractional Year {change['time_fractional_year']:.3f} | Sigma: {change['prior_sigma']:.4f}")
                        
                        # Rigorously map the fractional year back to the chronological frame index
                        time_diffs = np.abs(cube_time_fractions - change['time_fractional_year'])
                        time_index = int(np.argmin(time_diffs))
                        
                        # Append [time_index, x_position (col), y_position (row)]
                        changed_pixels_mask.append([time_index, c, r])
                else:
                    logging.info(f"No prolonged structural deviations detected for pixel {pixel_id}.")

            except ValueError as ve:
                # Handles intentional fail-fast conditions to prevent silent assumptions
                logging.warning(f"Skipping pixel {pixel_id}: {ve}")
                
    # ==========================================
    # 4. RESULTS AGGREGATION
    # ==========================================
    # Save the changed pixel mask to an HDF5 dataset structure
    h5_output_path = os.path.join(SAVE_DIR, "xfmrCCDC_mask.h5")
    with h5py.File(h5_output_path, 'w') as h5_out:
        grp = h5_out.create_group('/HDFEOS/GRIDS/xfmrCCDC/Data Fields/')
        
        # Enforce strict array dimensions (N, 3), allocating an empty shape if no anomalies were found
        mask_array = np.array(changed_pixels_mask, dtype=np.int32) if changed_pixels_mask else np.empty((0, 3), dtype=np.int32)
        
        dataset = grp.create_dataset('changed_pixel_mask', data=mask_array, compression="gzip", compression_opts=4)
        dataset.attrs['description'] = "Array of detected structural anomalies in the time series"
        dataset.attrs['format'] = "[time_index, x_position (col), y_position (row)]"
        
    logging.info(f"Pipeline complete. HDF5 Pixel Mask safely committed to {h5_output_path}")
    
    # Safely close HDF5 file handles
    h5_l.close()
    h5_t.close()

if __name__ == "__main__":
    main()