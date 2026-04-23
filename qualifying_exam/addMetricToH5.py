import h5py
import numpy as np
import warnings
import tkinter as tk
from tkinter import filedialog
from scipy import ndimage
import SpecComplex as sc

# --- Configuration ---
# Combined Pixel Mask Configuration
MASKING = True
SUN_ELEVATION_THRESHOLD = 30
CLOUD_DILATION = 1

# LANDSAT Pixel Mask Configuration
QA_REJECT_MASK = 0b111111
RADSAT_ACCEPT_VALUE = 0
AEROSOL_ACCEPT_LEVEL = 'medium' #'low' 'medium' 'high'

# TANAGER Specific Configuration
TANAGER_CLOUD_MASK = True
TANAGER_UNCERTAINTY_THRESHOLD = 0.1
TANAGER_AEROSOL_THRESHOLD = 0.3

# Size of the sliding window for local background estimation (must be odd, e.g., 11, 21, 51)
# 21x21 pixels at 30m resolution provides a ~600m x 600m local background context.
Z_SCORE_WINDOW_SIZE = 7
SLIDING_STRIDE = 1

def process_file_rgb(filepath):
    print(f"Opening HDF5 File: {filepath}")
    
    with h5py.File(filepath, 'r+') as h5:
        # Dynamically determine the grid name (LANDSAT or TANAGER)
        grid_name = list(h5['/HDFEOS/GRIDS'].keys())[0]
        base_fields_path = f"/HDFEOS/GRIDS/{grid_name}/Data Fields"
        
        target_ds_name = "surface_reflectance"
        if target_ds_name not in h5[base_fields_path]:
            print(f"Error: Dataset '{target_ds_name}' not found in file.")
            return
        
        ds_surface_reflectance = h5[f"{base_fields_path}/{target_ds_name}"]
        num_frames, bands, height, width = ds_surface_reflectance.shape

        # Create or overwrite the RGB dataset
        out_ds_name = "ortho_visual"
        if out_ds_name in h5[base_fields_path]:
            print(f"Overwriting existing dataset: {out_ds_name}")
            del h5[f"{base_fields_path}/{out_ds_name}"]
            
        ds_ortho_visual = h5[base_fields_path].create_dataset(out_ds_name,shape=(num_frames, 4, height, width),dtype='uint8',compression="gzip")
        
        print(f"Generating RGB frames...")
        for t in range(num_frames):
            frame_sr = ds_surface_reflectance[t, ...]
            rgba_img = sc.generate_rgba_image(frame_sr)
            ds_ortho_visual[t, ...] = np.transpose(rgba_img, (2, 0, 1))
            print(f"  Frame {t+1}/{num_frames} processed.")
            
        ds_ortho_visual.attrs['description'] = "RGBalpha frame of the surface reflectance (B4, B3, B2, Alpha)"


    print("\nRGB frame generation complete and saved to file.")

def get_landsat_mask(data_grp, f_idx, shape):
    """Generates a boolean mask for LANDSAT data."""
    valid_mask = np.ones(shape, dtype=bool)
    kernel = np.ones((3, 3), dtype=bool)
    
    # Sun Elevation Check
    if 'surface_reflectance' in data_grp:
        sun_elev_arr = data_grp['surface_reflectance'].attrs.get('sun_elevation')
        if sun_elev_arr is not None and sun_elev_arr[f_idx] < SUN_ELEVATION_THRESHOLD:
            return np.zeros(shape, dtype=bool)

    # QA Reject Mask
    if 'QUALITY_L1_PIXEL' in data_grp:
        qa_pixel = data_grp['QUALITY_L1_PIXEL'][f_idx, ...]
        # True represents BAD pixels (Clouds/Shadows)
        bad_qa_mask = (qa_pixel & QA_REJECT_MASK) != 0
        if CLOUD_DILATION > 0:
            bad_qa_mask = ndimage.binary_dilation(bad_qa_mask, structure=kernel, iterations=CLOUD_DILATION)
        # Valid pixels are where bad_qa_mask is False
        valid_mask &= ~bad_qa_mask

    # RADSAT Accept Value
    if 'RADIOMETRIC_SATURATION' in data_grp:
        bad_radsat = data_grp['RADIOMETRIC_SATURATION'][f_idx, ...] != RADSAT_ACCEPT_VALUE
        valid_mask &= ~bad_radsat

    # Aerosol Accept Values
    if AEROSOL_ACCEPT_LEVEL != 'all' and 'QUALITY_L2_AEROSOL' in data_grp:
        aerosol = data_grp['QUALITY_L2_AEROSOL'][f_idx, ...]
        invalid_aerosol = ~np.isin(aerosol, AEROSOL_DICT.get(AEROSOL_ACCEPT_LEVEL, AEROSOL_DICT['medium']))
        if CLOUD_DILATION > 0:
            invalid_aerosol = ndimage.binary_dilation(invalid_aerosol, structure=kernel, iterations=1)
        valid_mask &= ~invalid_aerosol

    return valid_mask


def process_file_z_score(filepath):
    print(f"Opening HDF5 File: {filepath}")
    
    with h5py.File(filepath, 'r+') as h5:
        # Dynamically determine the grid name (LANDSAT or TANAGER)
        grid_name = list(h5['/HDFEOS/GRIDS'].keys())[0]
        base_fields_path = f"/HDFEOS/GRIDS/{grid_name}/Data Fields"
        data_grp = h5[base_fields_path]
        
        # Verify the target dataset exists
        target_ds_name = "sliding_volume_map"
        if target_ds_name not in data_grp:
            print(f"Error: Dataset '{target_ds_name}' not found in file. Please run the complexity calculation first.")
            return
            
        ds_sliding_vol = data_grp[target_ds_name]
        num_frames, height, width = ds_sliding_vol.shape

        global_ds_name = "sliding_volume_z_score_masked"
        if global_ds_name in data_grp:
            print(f"Overwriting existing dataset: {global_ds_name}")
            del data_grp[global_ds_name]
            
        ds_global_z_score = data_grp.create_dataset(global_ds_name, shape=(num_frames, height, width), dtype='float32', compression="gzip")
        
        print(f"Calculating Global Spectral Complexity Z-scores with Censored Background Modeling...")
        for t in range(num_frames):
            frame_vols = ds_sliding_vol[t, ...]
            
            # Fetch valid pixel mask based on grid type and configuration
            if MASKING:
                if grid_name == 'LANDSAT':
                    valid_mask = sc.get_landsat_mask(data_grp, t, (height, width), 
                                                 sun_elevation_threshold=SUN_ELEVATION_THRESHOLD,
                                                 cloud_dilation=CLOUD_DILATION,
                                                 qa_reject_mask=QA_REJECT_MASK,
                                                 radsat_accept_value=RADSAT_ACCEPT_VALUE,
                                                 aerosol_accept_level=AEROSOL_ACCEPT_LEVEL)
                elif grid_name == 'TANAGER':
                    valid_mask = sc.get_tanager_mask(data_grp, t, (height, width),
                                                 sun_elevation_threshold=SUN_ELEVATION_THRESHOLD,
                                                 cloud_dilation=CLOUD_DILATION,
                                                 apply_cloud_mask=TANAGER_CLOUD_MASK,
                                                 uncertainty_threshold=TANAGER_UNCERTAINTY_THRESHOLD,
                                                 aerosol_depth_threshold=TANAGER_AEROSOL_THRESHOLD)
            else:
                valid_mask = np.ones((height, width), dtype=bool)

            # Strict error handling: SpecComplex.py will raise ValueError if invalid
            z_frame = sc.calculate_global_z_score(frame_vols, valid_mask)
            ds_global_z_score[t, ...] = z_frame
            print(f"  Frame {t+1}/{num_frames} processed.")
            
        ds_global_z_score.attrs['description'] = "Global Spectral Complexity Z-score. Masked pixels excluded from background stats. (Log(volume)-Log(mean_volume))/Std(Log(volume))"
        ds_global_z_score.attrs['MASKING_APPLIED'] = MASKING
        ds_global_z_score.attrs['SUN_ELEVATION_THRESHOLD'] = SUN_ELEVATION_THRESHOLD
        ds_global_z_score.attrs['CLOUD_DILATION'] = CLOUD_DILATION
        ds_global_z_score.attrs['QA_REJECT_MASK'] = QA_REJECT_MASK
        ds_global_z_score.attrs['RADSAT_ACCEPT_VALUE'] = RADSAT_ACCEPT_VALUE
        ds_global_z_score.attrs['AEROSOL_ACCEPT_LEVEL'] = AEROSOL_ACCEPT_LEVEL

        ## Create or overwrite the Z-score dataset
        #out_ds_name = "sliding_volume_local_z_score"
        #if out_ds_name in h5[base_fields_path]:
        #    print(f"Overwriting existing dataset: {out_ds_name}")
        #    del h5[f"{base_fields_path}/{out_ds_name}"]
        #    
        #ds_local_z_score = h5[base_fields_path].create_dataset(out_ds_name,shape=(num_frames, height, width),dtype='float32',compression="gzip")
        #
        #print(f"Calculating Local Sliding Z-scores (Window Size = {Z_SCORE_WINDOW_SIZE}x{Z_SCORE_WINDOW_SIZE}, Stride = {SLIDING_STRIDE})...")
        #for t in range(num_frames):
        #    frame_vols = ds_sliding_vol[t, ...]
        #    
        #    z_frame = sc.calculate_local_z_score(frame_vols, Z_SCORE_WINDOW_SIZE, SLIDING_STRIDE)
        #        
        #    ds_local_z_score[t, ...] = z_frame
        #    print(f"  Frame {t+1}/{num_frames} processed.")
        #    
        #ds_local_z_score.attrs['description'] = "Local Adaptive Z-score of the local spectral complexity. (Log(volume)-Log(mean_volume))/Std(Log(volume))"
        #ds_local_z_score.attrs['z_score_window_size'] = Z_SCORE_WINDOW_SIZE
        #ds_local_z_score.attrs['z_score_sliding_stride'] = SLIDING_STRIDE

    print("\nZ-score calculation complete and saved to file.")

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    
    print("Please select the processed HDF5 Image Stack...")
    file_path = filedialog.askopenfilename(
        title="Select HDF5 Image Stack",
        filetypes=[("HDF5 files", "*.h5")]
    )
    
    if file_path:
        process_file_z_score(file_path)
    else:
        print("No file selected.")
    
    root.destroy()