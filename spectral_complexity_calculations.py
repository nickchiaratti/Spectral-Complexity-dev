import shutil
import h5py
import numpy as np
import tkinter as tk
from tkinter import filedialog
import SpecComplex as sc


# --- Configuration ---
TILE_SIZE = 3          # Size of the window (NxN pixels) for volume calc
SLIDING_STRIDE = 1      # Stride for sliding window (1 = every pixel, higher = faster)

# --- Parameters for Maximum-Distance ---
num_endmembers = 7
MAX_DIST_P2 = 0
gram_type = 'local'
#SC_Param_Norm = None #'magnitude' 'band_count' None 'dimensionality' 'simplex'

def process_file(filepath, norm_param=None):
    print(f"Processing: {filepath}")
    out_path = filepath.replace(".h5", f"_SC_EM-{num_endmembers}_Gram-{gram_type}_Norm-{norm_param}.h5")
    print(f"Output Path: {out_path}")
    shutil.copy2(filepath, out_path)

    with h5py.File(out_path, 'r+') as h5:

        grid_name = list(h5['/HDFEOS/GRIDS'].keys())[0]

        if grid_name not in ['TANAGER','LANDSAT']:
            print(f"Error: Unknown grid name: {grid_name}")
            return
        process_image_stack(h5,grid_name, norm_param)
    print(f"\nCalculation Complete. Saved to: {out_path}")

# Helper to manage output datasets
def overwrite_dset(h5, base_fields_path, name, shape, dtype='float32'):
    path = f"{base_fields_path}/{name}"
    if name in h5[base_fields_path]: del h5[path]
    return h5[base_fields_path].create_dataset(name, shape=shape, dtype=dtype, compression="gzip")

def process_image_stack(h5,sourceName, norm_param=None):

    grid_name = sourceName
    base_fields_path = f"/HDFEOS/GRIDS/{grid_name}/Data Fields"
    ds_surfRef = h5[f"{base_fields_path}/surface_reflectance"]
    num_frames, num_bands, height, width = ds_surfRef.shape

    if sourceName == "LANDSAT":
        # --- QA Bitmasks (Landsat Collection 2) ---
        # Bit 0: Fill, 1: Dilated Cloud, 2: Cirrus, 3: Cloud, 4: Cloud Shadow, 5: Snow
        QA_BAD_MASK = (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3) | (1 << 4) | (1 << 5)
        ds_quality = h5[f"{base_fields_path}/QUALITY_L1_PIXEL"]
    elif sourceName == "TANAGER":
        gw_mask = ds_surfRef.attrs.get("all_good_wavelengths")
        invalid_mask = h5[f"{base_fields_path}/sr_invalid"]
        cloud_mask = h5[f"{base_fields_path}/beta_cloud_mask"]
        cirrus_mask = h5[f"{base_fields_path}/beta_cirrus_mask"]
        nodata_mask = h5[f"{base_fields_path}/nodata_pixels"]
        

    # Initialize Results
    ds_endmembers = overwrite_dset(h5, base_fields_path, 'frame_endmembers', (num_frames, num_bands, num_endmembers))
    ds_endmember_indices = overwrite_dset(h5, base_fields_path, 'frame_endmember_indices', (num_frames, num_endmembers), dtype='int32')
    ds_vol_curve = overwrite_dset(h5, base_fields_path, 'frame_endmember_volumes', (num_frames, num_endmembers))
    ds_tile = overwrite_dset(h5, base_fields_path, 'tile_volume_map', (num_frames, height, width))
    ds_slide = overwrite_dset(h5, base_fields_path, 'sliding_volume_map', (num_frames, height, width))

    for t in range(num_frames):
        print(f"\n--- Frame {t+1}/{num_frames} ---")
        if sourceName == "LANDSAT":
            frame_sr = ds_surfRef[t, ...]
            valid_spatial_mask = (ds_quality[t, ...] & QA_BAD_MASK) == 0
        elif sourceName == "TANAGER":
            frame_sr = ds_surfRef[t,gw_mask[t]==1, ...]
            valid_spatial_mask = (invalid_mask[t, ...] == 1) & (cloud_mask[t, ...] == 0) & (cirrus_mask[t, ...] == 0) & (nodata_mask[t, ...] == 0)

        print(f"Calculating full frame Volume for frame {t+1}/{num_frames}")
        endmembers, endmember_idx, vol_curve = sc.process_volume_frame(frame_sr, num_endmembers, gram_type, valid_spatial_mask, norm_param)

        if sourceName == "TANAGER":
            em_full = np.full((num_bands, num_endmembers), np.nan, dtype=np.float32)
            em_full[gw_mask[t]==1, :] = endmembers
            ds_endmembers[t, ...] = em_full
        else:
            ds_endmembers[t, ...] = endmembers

        ds_endmember_indices[t, ...] = endmember_idx
        ds_vol_curve[t, ...] = vol_curve
        print(f"Calculating Tile Volume for frame {t+1}/{num_frames}")
        ds_tile[t, ...] = sc.process_volume_tiles(frame_sr, TILE_SIZE, num_endmembers, gram_type, valid_spatial_mask, norm_param)
        print(f"Calculating Sliding Tile Volume for frame {t+1}/{num_frames}")
        ds_slide[t, ...] = sc.process_volume_sliding_tile(frame_sr, TILE_SIZE, SLIDING_STRIDE, num_endmembers, gram_type, valid_spatial_mask, norm_param)
            
    ds_vol_curve.attrs['description'] = "Full volume curve (Volume vs Endmember Count) for entire frame"
    ds_endmember_indices.attrs['description'] = "Endmember indices for each pixel"
    ds_endmembers.attrs['description'] = "Endmembers for each pixel"
    if norm_param == 'magnitude':
        ds_endmembers.attrs['Normalization'] = "Surface Reflectance Magnitude"
        ds_vol_curve.attrs['Normalization'] = "Surface Reflectance Magnitude"
        ds_tile.attrs['Normalization'] = "Surface Reflectance Magnitude"
        ds_slide.attrs['Normalization'] = "Surface Reflectance Magnitude"
    elif norm_param == None:
        ds_endmembers.attrs['Normalization'] = "None"
        ds_vol_curve.attrs['Normalization'] = "None"
        ds_tile.attrs['Normalization'] = "None"
        ds_slide.attrs['Normalization'] = "None"
    
    ds_tile.attrs['description'] = "Volume of convex hull of spectral data within each NxN tile"
    ds_slide.attrs['description'] = "Volume of convex hull of spectral data within each sliding NxN tile"
    ds_tile.attrs['tile_size'] = TILE_SIZE
    ds_slide.attrs['tile_size'] = TILE_SIZE
    ds_slide.attrs['sliding_stride'] = SLIDING_STRIDE
    ds_tile.attrs['gram_type'] = gram_type
    ds_slide.attrs['gram_type'] = gram_type

    h5.close()
        

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    
    print("Please select the HDF5 Image Stack...")
    file_path = tk.filedialog.askopenfilename(
        title="Select HDF5 Image Stack",
        filetypes=[("HDF5 files", "*.h5")]
    )
    
    if file_path:
        process_file(file_path)
    else:
        print("No file selected.")
    
    root.destroy()