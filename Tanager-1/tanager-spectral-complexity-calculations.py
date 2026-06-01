import os
import shutil
import h5py
import numpy as np
import tkinter as tk
from tkinter import filedialog
from tqdm import tqdm
# Assuming these are available in your environment
import MaxD_Gram as maxD 
import NSC_toolbox as nsc

# --- Configuration ---
TILE_SIZE = 3          # Size of the window (NxN pixels) for volume calc
SLIDING_STRIDE = 1      # Stride for sliding window (1 = every pixel, higher = faster)

# --- Parameters for Maximum-Distance ---
MAX_DIST_P1 = min(TILE_SIZE**2-1, 10)
MAX_DIST_P2 = 0
MAX_DIST_P3 = 'local'

def _process_volume_ROI(frame_data, valid_mask=None):
    """
    Process the ROI image to identify endmembers for the entire ROI.
    Pixel Filtering: Only valid pixels are extracted into the 2D matrix.
    Returns the full volume curve, endmembers, and indices.
    """
    bands, height, width = frame_data.shape
    img = np.transpose(frame_data, (1, 2, 0))
    image2D = np.reshape(img, (height * width, bands))

    if valid_mask is not None:
        # Flatten mask and extract only valid spectral signatures
        flat_mask = valid_mask.flatten()
        image2D = image2D[flat_mask]
        
    if image2D.shape[0] < MAX_DIST_P1:
        return np.zeros(MAX_DIST_P1), np.zeros((bands, MAX_DIST_P1)), np.zeros(MAX_DIST_P1)

    # Calculate using NSC toolbox
    endmembers, endmember_indices, volume = nsc.maximumDistance(image2D, MAX_DIST_P1, MAX_DIST_P2, MAX_DIST_P3)
    
    # Return full volume array (curve) instead of just the maximum
    return volume, endmembers, endmember_indices

def _process_volume_tiles(frame_data, tile_size, valid_mask):
    """
    Grid-based processing (Non-overlapping tiles).
    Strict Validity: Window is only processed if ALL pixels are valid.
    Any pixel that is part of an invalid tile is set to NaN.
    """
    bands, height, width = frame_data.shape
    img = np.transpose(frame_data, (1, 2, 0))
    output_map = np.full((height, width), np.nan, dtype=np.float32)
    
    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            y_end, x_end = min(y + tile_size, height), min(x + tile_size, width)
            
            # Check mask for this tile
            m_chunk = valid_mask[y:y_end, x:x_end]
            
            # REQUIREMENT: Window must be 100% valid (no clouds/shadows/nodata)
            if not np.all(m_chunk):
                continue 
            
            chunk = img[y:y_end, x:x_end, :]
            chunk_2d = np.reshape(chunk, (-1, bands))
            
            if chunk_2d.shape[0] >= MAX_DIST_P1:
                _, _, volume = nsc.maximumDistance(chunk_2d, MAX_DIST_P1, MAX_DIST_P2, MAX_DIST_P3)
                output_map[y:y_end, x:x_end] = np.max(volume[3:])
    
    # Explicitly enforce spatial mask on final output
    output_map[valid_mask == 0] = np.nan
    return output_map

def _process_volume_sliding_tile(frame_data, tile_size, stride, valid_mask):
    """
    Sliding window processing.
    Strict Validity: Window is only processed if ALL pixels are valid.
    Output is masked with NaN for any pixel identified as invalid.
    """
    bands, height, width = frame_data.shape
    img = np.transpose(frame_data, (1, 2, 0))
    
    sum_map = np.zeros((height, width), dtype=np.float32)
    count_map = np.zeros((height, width), dtype=np.float32)
    
    for y_start in tqdm(range(0, height - tile_size + 1, stride), desc="Sliding Window"):
        for x_start in range(0, width - tile_size + 1, stride):
            y_end, x_end = y_start + tile_size, x_start + tile_size
            
            # Use valid mask to verify window integrity
            m_chunk = valid_mask[y_start:y_end, x_start:x_end]
            
            if not np.all(m_chunk):
                continue 

            tile_cube = img[y_start:y_end, x_start:x_end, :]
            tile_2d = np.reshape(tile_cube, (-1, bands))
            
            if tile_2d.shape[0] >= MAX_DIST_P1:
                _, _, volume = nsc.maximumDistance(tile_2d, MAX_DIST_P1, MAX_DIST_P2, MAX_DIST_P3)
                vol_val = np.max(volume[3:])
                sum_map[y_start:y_end, x_start:x_end] += vol_val
                count_map[y_start:y_end, x_start:x_end] += 1.0
            
    # Finalize normalization
    result = np.full((height, width), np.nan, dtype=np.float32)
    valid_pixels = (count_map > 0)
    result[valid_pixels] = sum_map[valid_pixels] / count_map[valid_pixels]
    
    # Explicitly enforce spatial mask on final output
    result[valid_mask == 0] = np.nan
    return result

def process_file(filepath):
    print(f"Processing: {filepath}")
    out_path = filepath.replace(".h5", "_calculated.h5")
    if out_path == filepath: out_path = filepath + "_calc.h5"
    
    shutil.copy2(filepath, out_path)

    with h5py.File(out_path, 'r+') as h5:
        grid_name = "TANAGER"
        base_fields_path = f"/HDFEOS/GRIDS/{grid_name}/Data Fields"
        data_path = f"{base_fields_path}/surface_reflectance"
        
        if data_path not in h5:
            print(f"Error: Could not find surface_reflectance in {data_path}.")
            return

        sr_dset = h5[data_path]
        num_frames, num_bands, height, width = sr_dset.shape

        # Fetch per-frame spectral quality mask
        all_gw = sr_dset.attrs.get("all_good_wavelengths")

        # Helper to manage output datasets
        def overwrite_dset(name, shape, dtype='float32'):
            path = f"{base_fields_path}/{name}"
            if name in h5[base_fields_path]: del h5[path]
            return h5[base_fields_path].create_dataset(name, shape=shape, dtype=dtype, compression="gzip")

        # --- Datasets ---
        ds_ROI = overwrite_dset("ROI_volume", (num_frames, 1))
        ds_vol_curve = overwrite_dset("endmember_volumes", (num_frames, MAX_DIST_P1))
        
        ds_endmembers = overwrite_dset("endmembers", (num_frames, num_bands, MAX_DIST_P1))
        ds_endmember_indices = overwrite_dset("endmember_indices", (num_frames, MAX_DIST_P1), dtype='int32')
        ds_tile = overwrite_dset("tile_volume_map", (num_frames, height, width))
        ds_slide = overwrite_dset("sliding_volume_map", (num_frames, height, width))

        for t in range(num_frames):
            print(f"\n--- Frame {t+1}/{num_frames} ---")
            
            # --- Spatial Validity Mask ---
            # Includes cloud, cirrus, and nodata
            try:
                cloud = h5[f"{base_fields_path}/beta_cloud_mask"][t, ...]
                cirrus = h5[f"{base_fields_path}/beta_cirrus_mask"][t, ...]
                nodata = h5[f"{base_fields_path}/nodata_pixels"][t, ...]
                
                # Logic: 0 is valid for all masks
                valid_spatial_mask = (cloud == 0) & (cirrus == 0) & (nodata == 0)
                print(f"Spatial Mask: {np.sum(valid_spatial_mask)}/{height*width} clear pixels.")
            except KeyError as e:
                print(f"Warning: Spatial mask dataset missing ({e}). Assuming all pixels valid.")
                valid_spatial_mask = np.ones((height, width), dtype=bool)

            frame_data = sr_dset[t, ...] 
            
            # Band Filtering (Spectral)
            if all_gw is not None:
                b_mask = all_gw[t] == 1
                filtered_data = frame_data[b_mask, :, :]
            else:
                filtered_data = frame_data
                b_mask = np.ones(num_bands, dtype=bool)

            # 1. ROI Volume and Volume Curve
            print("Calculating Full ROI Volume and Endmember Curve...")
            vol_curve, em_subset, em_idx = _process_volume_ROI(filtered_data, valid_mask=valid_spatial_mask)
            
            # Store maximum volume (scalar) and full curve (vector)
            ds_ROI[t, ...] = np.max(vol_curve[3:]) if len(vol_curve) > 3 else vol_curve[-1]
            ds_vol_curve[t, :] = vol_curve
            
            # Map endmembers back with NaN for bad bands
            em_full = np.full((num_bands, MAX_DIST_P1), np.nan, dtype=np.float32)
            em_full[b_mask, :] = em_subset
            ds_endmembers[t, ...] = em_full
            ds_endmember_indices[t, ...] = em_idx
            
            # 2. Tiled Volume Map
            print(f"Calculating Tiled Volume Map...")
            ds_tile[t, ...] = _process_volume_tiles(filtered_data, TILE_SIZE, valid_spatial_mask)
            
            # 3. Sliding Volume Map
            print(f"Calculating Sliding Volume Map...")
            ds_slide[t, ...] = _process_volume_sliding_tile(filtered_data, TILE_SIZE, SLIDING_STRIDE, valid_spatial_mask)
            
            h5.flush()

        # Add Attributes
        ds_vol_curve.attrs['description'] = "Full volume curve (Volume vs Endmember Count) for entire ROI"
        ds_ROI.attrs['description'] = "Maximum Spectral Complexity for ROI (Clouds/NoData filtered)"
        ds_slide.attrs['description'] = "Sliding Window Volume (Strict Validity windows: Clouds/NoData filtered)"
            
    print(f"\nCalculation Complete. Saved to: {out_path}")

if __name__ == "__main__":
    # Hide Tkinter root
    root = tk.Tk()
    root.withdraw()
    
    print("Please select the Tanager Stack HDF5 file...")
    file_path = filedialog.askopenfilename(
        title="Select Tanager HDF5 Stack",
        filetypes=[("HDF5 files", "*.h5")]
    )
    
    if file_path:
        process_file(file_path)
    else:
        print("No file selected.")
    
    root.destroy()