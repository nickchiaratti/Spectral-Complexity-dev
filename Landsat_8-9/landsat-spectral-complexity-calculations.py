import h5py
import numpy as np
import os
import shutil
import tkinter as tk
from tkinter import filedialog
from tqdm import tqdm
import NSC_toolbox as nsc

# --- CONFIGURATION ---
# Processing Parameters
TILE_SIZE = 3          # Size of the window (NxN pixels) for volume calc
SLIDING_STRIDE = 1      # Stride for sliding window (1 = every pixel, higher = faster)
VOLUME_METHOD = 'peak'  # 'sum' or 'peak'

# Maximum Distance Parameters
MAX_DIST_P1_TILED = min(TILE_SIZE**2-1, 7) # For Grid/Sliding
MAX_DIST_P1_FULL = 7                       # Default for Full ROI
MAX_DIST_P2 = 0
MAX_DIST_P3 = 'local'
MAX_DIST_NORMALIZE = False

# --- QA Bitmasks (Landsat Collection 2) ---
# Bit 0: Fill, 1: Dilated Cloud, 2: Cirrus, 3: Cloud, 4: Cloud Shadow, 5: Snow
QA_BAD_MASK = (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3) | (1 << 4) | (1 << 5)

def _process_volume_ROI(frame_data, valid_mask):
    """
    Process the ROI image to identify endmembers for the entire ROI.
    Pixel Filtering: Only valid pixels are extracted into the 2D matrix.
    """
    bands, height, width = frame_data.shape
    img = np.transpose(frame_data, (1, 2, 0))
    image2D = np.reshape(img, (height * width, bands))

    # Flatten mask and extract only valid spectral signatures
    flat_mask = valid_mask.flatten()
    image2D = image2D[flat_mask]
        
    if image2D.shape[0] < MAX_DIST_P1_FULL:
        return 0.0, np.zeros((bands, MAX_DIST_P1_FULL)), np.zeros(MAX_DIST_P1_FULL), np.zeros(MAX_DIST_P1_FULL)

    # Calculate using NSC toolbox
    # nsc.maximumDistance returns: endmembers (bands, P1), indices (1, P1), volume (P1,)
    if MAX_DIST_NORMALIZE:
        endmembers, endmember_indices, volume_curve = nsc.maximumDistanceNormalized(image2D, MAX_DIST_P1_FULL, MAX_DIST_P2, MAX_DIST_P3)
    else:
        endmembers, endmember_indices, volume_curve = nsc.maximumDistance(image2D, MAX_DIST_P1_FULL, MAX_DIST_P2, MAX_DIST_P3)
    
    ROI_volume = np.max(volume_curve[3:]) if VOLUME_METHOD == 'peak' else np.sum(volume_curve[3:])
    
    # Flatten endmember_indices to ensure it is 1D (shape [P1,])
    return ROI_volume, endmembers, endmember_indices.flatten(), volume_curve

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
            if not np.all(m_chunk) or m_chunk.size < (tile_size**2):
                continue 
            
            chunk = img[y:y_end, x:x_end, :]
            chunk_2d = np.reshape(chunk, (-1, bands))
            
            if chunk_2d.shape[0] >= MAX_DIST_P1_TILED:
                if MAX_DIST_NORMALIZE:
                    _, _, volume = nsc.maximumDistanceNormalized(chunk_2d, MAX_DIST_P1_TILED, MAX_DIST_P2, MAX_DIST_P3)
                else:
                    _, _, volume = nsc.maximumDistance(chunk_2d, MAX_DIST_P1_TILED, MAX_DIST_P2, MAX_DIST_P3)
                output_map[y:y_end, x:x_end] = np.max(volume[3:]) if VOLUME_METHOD == 'peak' else np.sum(volume[3:])
    
    # Explicitly enforce spatial mask on final output
    output_map[valid_mask == 0] = np.nan
    return output_map

def _process_volume_sliding_tile(frame_data, tile_size, stride, valid_mask):
    """
    Sliding window processing.
    Strict Validity: Window is only processed if ALL pixels are valid.
    """
    bands, height, width = frame_data.shape
    img = np.transpose(frame_data, (1, 2, 0))
    
    sum_map = np.zeros((height, width), dtype=np.float32)
    count_map = np.zeros((height, width), dtype=np.float32)
    
    for y_start in range(0, height - tile_size + 1, stride):
        for x_start in range(0, width - tile_size + 1, stride):
            y_end, x_end = y_start + tile_size, x_start + tile_size
            
            # Check window integrity
            m_chunk = valid_mask[y_start:y_end, x_start:x_end]
            
            if not np.all(m_chunk):
                continue 

            tile_cube = img[y_start:y_end, x_start:x_end, :]
            tile_2d = np.reshape(tile_cube, (-1, bands))
            
            if tile_2d.shape[0] >= MAX_DIST_P1_TILED:
                if MAX_DIST_NORMALIZE:
                    _, _, volume = nsc.maximumDistanceNormalized(tile_2d, MAX_DIST_P1_TILED, MAX_DIST_P2, MAX_DIST_P3)
                else:
                    _, _, volume = nsc.maximumDistance(tile_2d, MAX_DIST_P1_TILED, MAX_DIST_P2, MAX_DIST_P3)
                vol_val = np.max(volume[3:]) if VOLUME_METHOD == 'peak' else np.sum(volume[3:])
                sum_map[y_start:y_end, x_start:x_end] += vol_val
                count_map[y_start:y_end, x_start:x_end] += 1.0
            
    result = np.full((height, width), np.nan, dtype=np.float32)
    valid_pixels = (count_map > 0)
    result[valid_pixels] = sum_map[valid_pixels] / count_map[valid_pixels]
    
    # Mask final output
    result[valid_mask == 0] = np.nan
    return result

def process_hdf5_stack(input_h5, output_h5):
    """
    Reads a Landsat HDF5 stack, performs spectral complexity analysis with QA masking, 
    and saves results back into the Data Fields group.
    """
    print(f"Initializing processing for: {input_h5}")
    
    if os.path.exists(output_h5):
        os.remove(output_h5)
    
    shutil.copy2(input_h5, output_h5)
    print(f"Created working copy: {output_h5}")

    with h5py.File(output_h5, 'r+') as f:
        try:
            data_grp_path = 'HDFEOS/GRIDS/LANDSAT/Data Fields'
            data_grp = f[data_grp_path]
            sr_dset = data_grp['surface_reflectance']
            qa_dset = data_grp['QUALITY_L1_PIXEL']
        except KeyError as e:
            print(f"Error: Missing required dataset in HDF5: {e}")
            return

        num_frames, num_bands, height, width = sr_dset.shape

        def overwrite_dset(name, shape, dtype='float32'):
            if name in data_grp: del data_grp[name]
            return data_grp.create_dataset(name, shape=shape, dtype=dtype, compression="gzip")

        # Initialize Results
        ds_full = overwrite_dset('spectral_complexity_full', (num_frames, 1))
        ds_endmembers = overwrite_dset('endmembers', (num_frames, num_bands, MAX_DIST_P1_FULL))
        ds_indices = overwrite_dset('endmember_indices', (num_frames, MAX_DIST_P1_FULL), dtype='int32')
        ds_vol_curve = overwrite_dset('endmember_volumes', (num_frames, MAX_DIST_P1_FULL))
        ds_grid = overwrite_dset('spectral_complexity_grid', (num_frames, height, width))
        ds_slide = overwrite_dset('spectral_complexity_sliding', (num_frames, height, width))

        for t in tqdm(range(num_frames), desc="Analyzing Frames"):
            # Load Data
            frame_sr = sr_dset[t, ...]
            frame_qa = qa_dset[t, ...]
            
            # Generate Spatial Validity Mask
            # 0 = Masked (Bad), 1 = Valid (Clear)
            valid_spatial_mask = (frame_qa & QA_BAD_MASK) == 0
            
            # 1. ROI Volume
            vol_ROI, em_subset, em_idx, vol_curve = _process_volume_ROI(frame_sr, valid_spatial_mask)
            ds_full[t, 0] = vol_ROI
            
            # em_subset is (num_bands, P1) - matches ds_endmembers[t] shape
            ds_endmembers[t, :, :] = em_subset
            
            # em_idx is (P1,) - matches ds_indices[t] shape
            ds_indices[t, :] = em_idx
            
            # vol_curve is (P1,) - matches ds_vol_curve[t] shape
            ds_vol_curve[t, :] = vol_curve
            
            # 2. Tiled Map
            ds_grid[t, :, :] = _process_volume_tiles(frame_sr, TILE_SIZE, valid_spatial_mask)
            
            # 3. Sliding Map
            ds_slide[t, :, :] = _process_volume_sliding_tile(frame_sr, TILE_SIZE, SLIDING_STRIDE, valid_spatial_mask)
            
            f.flush()

        # Finalize Attributes
        common_desc = f"Spectral Complexity (Volume, {VOLUME_METHOD} method)"
        ds_full.attrs['description'] = f"{common_desc} - Entire ROI (Clear pixels only)"
        ds_endmembers.attrs['description'] = "Endmember Spectra from clear ROI pixels"
        ds_indices.attrs['description'] = "Pixel indices of endmembers"
        ds_vol_curve.attrs['description'] = "Iterative volume curve for ROI endmembers"
        ds_grid.attrs['description'] = f"{common_desc} - Grid Tiling (Strict: windows must be 100% clear)"
        ds_grid.attrs['tile_size'] = tile_size = TILE_SIZE
        ds_slide.attrs['description'] = f"{common_desc} - Sliding Window (Strict: windows must be 100% clear)"
        ds_slide.attrs['tile_size'] = TILE_SIZE
        ds_slide.attrs['sliding_stride'] = SLIDING_STRIDE
        ds_slide.attrs['endmember_normalization'] = MAX_DIST_NORMALIZE

        # TODO: Update StructMetadata.0 ODL string to include these new DataFields 

    print(f"\nProcessing Complete. Saved results to: {output_h5}")

if __name__ == "__main__":
    root = tk.Tk(); root.withdraw()
    input_file = filedialog.askopenfilename(title="Select Stacker HDF5 File", filetypes=[("HDF5 files", "*.h5")])
    if input_file:
        path_base, path_ext = os.path.splitext(input_file)
        if MAX_DIST_NORMALIZE:
            output_file = f"{path_base}_Complexity_Analysis_Gram_{MAX_DIST_P3}_Normalized{path_ext}"
        else:
            output_file = f"{path_base}_Complexity_Analysis_Gram_{MAX_DIST_P3}{path_ext}"
        process_hdf5_stack(input_file, output_file)
    root.destroy()