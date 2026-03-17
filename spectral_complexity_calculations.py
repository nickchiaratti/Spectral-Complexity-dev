'''
todo: implement LANDSAT radsat exclusion

'''

import shutil
import h5py
import numpy as np
import tkinter as tk
from tkinter import filedialog
import SpecComplex as sc

# --- Configuration ---

QA_REJECT_MASK = 0b111111
SUN_ELEVATION_THRESHOLD = 40
QA_AEROSOL_ACCEPT_VALUES = [2, 4, 32, 66, 68, 96, 100]
AEROSOL_ACCEPT_LEVEL = 'low' #'low' 'medium' 'high'
AEROSOL_ACCEPT_VALUES = [2, 4, 32, 66, 68, 96, 100]

# Flattened list creation to prevent h5py attribute assignment errors
if AEROSOL_ACCEPT_LEVEL == 'medium':
    AEROSOL_ACCEPT_VALUES.extend([130, 132, 160, 164])
elif AEROSOL_ACCEPT_LEVEL == 'high':
    AEROSOL_ACCEPT_VALUES.extend([130, 132, 160, 164, 192, 194, 196, 224, 228])


TILE_SIZE = 3          # Size of the window (NxN pixels) for volume calc
SLIDING_STRIDE = 1      # Stride for sliding window (1 = every pixel, higher = faster)

# --- Parameters for Maximum-Distance ---
num_endmembers = 7
MAX_DIST_P2 = 0
#gram_type = 'datasetMean' # 'general
#SC_Param_Norm = 'bandCount' #'bandCount' None 

def process_file(filepath, norm_param='bandCount', gram_type='datasetMean'):
    print(f"Processing: {filepath}")
    
    # Construct Output Filename
    suffix = f"_SC_EM-{num_endmembers}_Gram-{gram_type}_Norm-{norm_param}_Aerosol-{AEROSOL_ACCEPT_LEVEL}_QA-AllFrames"
    out_path = filepath.replace(".h5", f"{suffix}_sunElMin-{SUN_ELEVATION_THRESHOLD}.h5")
    
    print(f"Output Path: {out_path}")
    shutil.copy2(filepath, out_path)

    with h5py.File(out_path, 'r+') as h5:
        grid_name = list(h5['/HDFEOS/GRIDS'].keys())[0]

        if grid_name not in ['TANAGER','LANDSAT']:
            print(f"Error: Unknown grid name: {grid_name}")
            return
            
        # Start Processing
        process_image_stack(h5, grid_name, norm_param, gram_type)

    print(f"\nCalculation Complete. Saved to: {out_path}")

# Helper to manage output datasets
def overwrite_dset(h5, base_fields_path, name, shape, dtype='float32'):
    path = f"{base_fields_path}/{name}"
    if name in h5[base_fields_path]: del h5[path]
    return h5[base_fields_path].create_dataset(name, shape=shape, dtype=dtype, compression="gzip")

def process_image_stack(h5, sourceName, norm_param, gram_type):

    grid_name = sourceName
    base_fields_path = f"/HDFEOS/GRIDS/{grid_name}/Data Fields"
    ds_surfRef = h5[f"{base_fields_path}/surface_reflectance"]
    num_frames, num_bands, height, width = ds_surfRef.shape

    if sourceName == "LANDSAT":
        # Apply the exact same mask here to handle pixel-level exclusions if Strict QA is False
        ds_quality = h5[f"{base_fields_path}/QUALITY_L1_PIXEL"]
        ds_aerosol = h5[f"{base_fields_path}/QUALITY_L2_AEROSOL"]
        ds_radsat = h5[f"{base_fields_path}/RADIOMETRIC_SATURATION"]
    elif sourceName == "TANAGER":
        gw_mask = ds_surfRef.attrs.get("all_good_wavelengths").astype(bool)
        invalid_mask = h5[f"{base_fields_path}/sr_invalid"]
        cloud_mask = h5[f"{base_fields_path}/beta_cloud_mask"]
        cirrus_mask = h5[f"{base_fields_path}/beta_cirrus_mask"]
        nodata_mask = h5[f"{base_fields_path}/nodata_pixels"]
        sun_zenith_ds = h5[f"{base_fields_path}/sun_zenith"]
        

    # Initialize Results
    ds_endmembers = overwrite_dset(h5, base_fields_path, 'frame_endmembers', (num_frames, num_bands, num_endmembers))
    ds_endmember_indices = overwrite_dset(h5, base_fields_path, 'frame_endmember_indices', (num_frames, num_endmembers), dtype='int32')
    ds_vol_curve = overwrite_dset(h5, base_fields_path, 'frame_endmember_volumes', (num_frames, num_endmembers))
    ds_tile = overwrite_dset(h5, base_fields_path, 'tile_volume_map', (num_frames, height, width))
    ds_slide = overwrite_dset(h5, base_fields_path, 'sliding_volume_map', (num_frames, height, width))
    ds_evi = overwrite_dset(h5, base_fields_path, 'evi_map', (num_frames, height, width))
    ds_msd = overwrite_dset(h5, base_fields_path, 'msd_map', (num_frames, height, width))

    for t in range(num_frames):
        print(f"\n--- Frame {t+1}/{num_frames} ---")
        frame_sr = ds_surfRef[t, ...]
        if sourceName == "TANAGER":
            frame_sr = np.delete(frame_sr, np.where(~gw_mask[t]), axis=0)

        print(f"Calculating full frame Volume for frame {t+1}/{num_frames}")
        endmembers, endmember_idx, vol_curve = sc.process_volume_frame(frame_sr, num_endmembers, gram_type, norm_param)

        if sourceName == "TANAGER":
            em_full = np.full((num_bands, num_endmembers), np.nan, dtype=np.float32)
            em_full[gw_mask[t]==1, :] = endmembers
            ds_endmembers[t, ...] = em_full
        else:
            ds_endmembers[t, ...] = endmembers

        ds_endmember_indices[t, ...] = endmember_idx
        ds_vol_curve[t, ...] = vol_curve
        print(f"Calculating Tile Volume for frame {t+1}/{num_frames}")
        ds_tile[t, ...] = sc.process_volume_tiles(frame_sr, TILE_SIZE, num_endmembers, gram_type, norm_param)
        print(f"Calculating Sliding Tile Volume for frame {t+1}/{num_frames}")
        ds_slide[t, ...] = sc.process_volume_sliding_tile(frame_sr, TILE_SIZE, SLIDING_STRIDE, num_endmembers, gram_type, norm_param)
        ds_evi[t, ...] = sc.calc_evi_frame(frame_sr)
        ds_msd[t, ...] = sc.process_msd_sliding_tile(frame_sr, TILE_SIZE, SLIDING_STRIDE)
            
    ds_vol_curve.attrs['description'] = "Full volume curve (Volume vs Endmember Count) for entire frame"
    ds_vol_curve.attrs['gram_type'] = gram_type
    ds_vol_curve.attrs['num_endmembers'] = num_endmembers
    ds_endmember_indices.attrs['description'] = "Endmember indices for each pixel"
    ds_endmembers.attrs['description'] = "Endmembers for each pixel"
    ds_endmembers.attrs['num_endmembers'] = num_endmembers
    
    # Store configuration attributes on datasets resulting from process_volume_frame
    for ds in [ds_endmembers, ds_endmember_indices, ds_vol_curve]:
        ds.attrs['SUN_ELEVATION_THRESHOLD'] = SUN_ELEVATION_THRESHOLD
        ds.attrs['QA_REJECT_MASK'] = QA_REJECT_MASK
        ds.attrs['AEROSOL_ACCEPT_VALUES'] = AEROSOL_ACCEPT_VALUES

    if norm_param:
        ds_endmembers.attrs['Normalization'] = norm_param
        ds_vol_curve.attrs['Normalization'] = norm_param
        ds_tile.attrs['Normalization'] = norm_param
        ds_slide.attrs['Normalization'] = norm_param
    else:
        ds_endmembers.attrs['Normalization'] = "None"
        ds_vol_curve.attrs['Normalization'] = "None"
        ds_tile.attrs['Normalization'] = "None"
        ds_slide.attrs['Normalization'] = "None"
    
    ds_tile.attrs['description'] = "Volume of convex hull of spectral data within each NxN tile"
    ds_slide.attrs['description'] = "Volume of convex hull of spectral data within each sliding NxN tile"
    ds_tile.attrs['tile_size'] = TILE_SIZE
    ds_slide.attrs['tile_size'] = TILE_SIZE
    ds_evi.attrs['description'] = "EVI for each pixel"
    ds_msd.attrs['description'] = "MSD for each pixel"
    ds_msd.attrs['tile_size'] = TILE_SIZE
    ds_msd.attrs['sliding_stride'] = SLIDING_STRIDE
    ds_slide.attrs['sliding_stride'] = SLIDING_STRIDE
    ds_tile.attrs['gram_type'] = gram_type
    ds_slide.attrs['gram_type'] = gram_type
    ds_tile.attrs['num_endmembers'] = num_endmembers
    ds_slide.attrs['num_endmembers'] = num_endmembers
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