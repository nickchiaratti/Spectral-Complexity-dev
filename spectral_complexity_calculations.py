import shutil
import h5py
import numpy as np
import tkinter as tk
from tkinter import filedialog
import SpecComplex as sc

# --- Configuration ---
TILE_SIZE = 3          # Size of the window (NxN pixels) for volume calc
SLIDING_STRIDE = 1      # Stride for sliding window
Z_SCORE_WINDOW_SIZE = 11

# --- Parameters for Maximum-Distance ---
num_endmembers = 7
MAX_DIST_P2 = 0
#gram_type = 'datasetMean' # 'general
#SC_Param_Norm = 'bandCount' #'bandCount' None 

# Pixel Mask Configuration
MASKING = True
SUN_ELEVATION_THRESHOLD = 30
CLOUD_DILATION = 2

# LANDSAT Specific Configuration
QA_REJECT_MASK = 0b111111
RADSAT_ACCEPT_VALUE = 0
AEROSOL_ACCEPT_LEVEL = 'medium' #'low' 'medium' 'high'

# TANAGER Specific Configuration
TANAGER_CLOUD_MASK = True
TANAGER_UNCERTAINTY_THRESHOLD = 0.1
TANAGER_AEROSOL_THRESHOLD = 0.3


def process_file(filepath, norm_param='bandCount', gram_type='minEndmember'):
    print(f"Processing: {filepath}")
    
    # Construct Output Filename
    suffix = f"_SC_EM-{num_endmembers}_Gram-{gram_type}_Norm-{norm_param}"
    out_path = filepath.replace(".h5", f"{suffix}.h5")
    
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
def overwrite_dset(h5, base_fields_path, name, shape, dtype='float32', **kwargs):
    path = f"{base_fields_path}/{name}"
    if name in h5[base_fields_path]: del h5[path]
    return h5[base_fields_path].create_dataset(name, shape=shape, dtype=dtype, compression="gzip", **kwargs)

def process_image_stack(h5, sourceName, norm_param, gram_type):

    grid_name = sourceName
    base_fields_path = f"/HDFEOS/GRIDS/{grid_name}/Data Fields"
    data_grp = h5[base_fields_path]
    ds_surfRef = data_grp["surface_reflectance"]
    num_frames, num_bands, height, width = ds_surfRef.shape

    # Pre-fetch the gw_mask for Tanager processing
    if sourceName == "TANAGER":
        gw_mask = ds_surfRef.attrs.get("all_good_wavelengths").astype(bool)

    # Initialize Results
    ds_endmembers = overwrite_dset(h5, base_fields_path, 'frame_endmembers', (num_frames, num_bands, num_endmembers))
    ds_endmember_indices = overwrite_dset(h5, base_fields_path, 'frame_endmember_indices', (num_frames, num_endmembers), dtype='int32')
    ds_vol_curve = overwrite_dset(h5, base_fields_path, 'frame_endmember_volumes', (num_frames, num_endmembers))
    ds_tile = overwrite_dset(h5, base_fields_path, 'tile_volume_map', (num_frames, height, width))
    ds_slide = overwrite_dset(h5, base_fields_path, 'sliding_volume_map', (num_frames, height, width))
    ds_evi = overwrite_dset(h5, base_fields_path, 'evi_map', (num_frames, height, width))
    ds_msd = overwrite_dset(h5, base_fields_path, 'msd_map', (num_frames, height, width))
    ds_slideZ = overwrite_dset(h5, base_fields_path, 'sliding_volume_z_score', (num_frames, height, width))
    ds_slideZ_masked = overwrite_dset(h5, base_fields_path, 'sliding_volume_z_score_masked', (num_frames, height, width))


    for t in range(num_frames):
        print(f"\n--- Frame {t+1}/{num_frames} ---")
        frame_sr = ds_surfRef[t, ...]
        
        if sourceName == "TANAGER":
            frame_sr = np.delete(frame_sr, np.where(~gw_mask[t]), axis=0)

        # Generate spatial mask utilizing centralized SpecComplex functions
        if MASKING:
            if sourceName == "LANDSAT":
                valid_mask = sc.get_landsat_mask(data_grp, t, (height, width), 
                                                 sun_elevation_threshold=SUN_ELEVATION_THRESHOLD,
                                                 cloud_dilation=CLOUD_DILATION,
                                                 qa_reject_mask=QA_REJECT_MASK,
                                                 radsat_accept_value=RADSAT_ACCEPT_VALUE,
                                                 aerosol_accept_level=AEROSOL_ACCEPT_LEVEL)
            elif sourceName == "TANAGER":
                valid_mask = sc.get_tanager_mask(data_grp, t, (height, width),
                                                 sun_elevation_threshold=SUN_ELEVATION_THRESHOLD,
                                                 cloud_dilation=CLOUD_DILATION,
                                                 apply_cloud_mask=TANAGER_CLOUD_MASK,
                                                 uncertainty_threshold=TANAGER_UNCERTAINTY_THRESHOLD,
                                                 aerosol_depth_threshold=TANAGER_AEROSOL_THRESHOLD)
        else:
            valid_mask = np.ones((height, width), dtype=bool)

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
        
        # Standard Z-Score (Background calculated on entire valid frame > 0.0)
        ds_slideZ[t,...] = sc.calculate_global_z_score(ds_slide[t, ...], np.ones((height, width), dtype=bool))
        
        # Masked Z-Score (Background cleanly calculated from sensor-valid pixels only)
        ds_slideZ_masked[t,...] = sc.calculate_global_z_score(ds_slide[t, ...], valid_mask)
            
    ds_vol_curve.attrs['description'] = "Full volume curve (Volume vs Endmember Count) for entire frame"
    ds_vol_curve.attrs['gram_type'] = gram_type
    ds_vol_curve.attrs['num_endmembers'] = num_endmembers
    ds_endmember_indices.attrs['description'] = "Endmember indices for each pixel"
    ds_endmembers.attrs['description'] = "Endmembers for each pixel"
    ds_endmembers.attrs['num_endmembers'] = num_endmembers
    
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
    ds_slideZ.attrs['description'] = "Global Spectral Complexity Z-score. (Log(volume)-Log(mean_volume))/Std(Log(volume))"
    
    ds_slideZ_masked.attrs['description'] = "Global Spectral Complexity Z-score. Sensor-masked pixels excluded from background stats."
    ds_slideZ_masked.attrs['MASKING_APPLIED'] = MASKING
    ds_slideZ_masked.attrs['SUN_ELEVATION_THRESHOLD'] = SUN_ELEVATION_THRESHOLD
    ds_slideZ_masked.attrs['CLOUD_DILATION'] = CLOUD_DILATION
    
    if sourceName == "LANDSAT":
        ds_slideZ_masked.attrs['QA_REJECT_MASK'] = QA_REJECT_MASK
        ds_slideZ_masked.attrs['RADSAT_ACCEPT_VALUE'] = RADSAT_ACCEPT_VALUE
        ds_slideZ_masked.attrs['AEROSOL_ACCEPT_LEVEL'] = AEROSOL_ACCEPT_LEVEL
    elif sourceName == "TANAGER":
        ds_slideZ_masked.attrs['TANAGER_CLOUD_MASK'] = TANAGER_CLOUD_MASK
        ds_slideZ_masked.attrs['TANAGER_UNCERTAINTY_THRESHOLD'] = TANAGER_UNCERTAINTY_THRESHOLD
        ds_slideZ_masked.attrs['TANAGER_AEROSOL_THRESHOLD'] = TANAGER_AEROSOL_THRESHOLD

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