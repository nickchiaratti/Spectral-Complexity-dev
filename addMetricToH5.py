import h5py
import numpy as np
import warnings
import tkinter as tk
from tkinter import filedialog
import SpecComplex as sc

# --- Configuration ---
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


def process_file_z_score(filepath):
    print(f"Opening HDF5 File: {filepath}")
    
    with h5py.File(filepath, 'r+') as h5:
        # Dynamically determine the grid name (LANDSAT or TANAGER)
        grid_name = list(h5['/HDFEOS/GRIDS'].keys())[0]
        base_fields_path = f"/HDFEOS/GRIDS/{grid_name}/Data Fields"
        
        # Verify the target dataset exists
        target_ds_name = "sliding_volume_map"
        if target_ds_name not in h5[base_fields_path]:
            print(f"Error: Dataset '{target_ds_name}' not found in file. Please run the complexity calculation first.")
            return
            
        ds_sliding_vol = h5[f"{base_fields_path}/{target_ds_name}"]
        num_frames, height, width = ds_sliding_vol.shape

        global_ds_name = "sliding_volume_z_score"
        if global_ds_name in h5[base_fields_path]:
            print(f"Overwriting existing dataset: {global_ds_name}")
            del h5[f"{base_fields_path}/{global_ds_name}"]
        ds_global_z_score = h5[base_fields_path].create_dataset(global_ds_name,shape=(num_frames, height, width),dtype='float32',compression="gzip")
        print(f"Calculating Global Spectral Complexity Z-scores...")
        for t in range(num_frames):
            frame_vols = ds_sliding_vol[t, ...]
            z_frame = sc.calculate_global_z_score(frame_vols)
            ds_global_z_score[t, ...] = z_frame
            print(f"  Frame {t+1}/{num_frames} processed.")
        

        # Create or overwrite the Z-score dataset
        out_ds_name = "sliding_volume_local_z_score"
        if out_ds_name in h5[base_fields_path]:
            print(f"Overwriting existing dataset: {out_ds_name}")
            del h5[f"{base_fields_path}/{out_ds_name}"]
            
        ds_local_z_score = h5[base_fields_path].create_dataset(out_ds_name,shape=(num_frames, height, width),dtype='float32',compression="gzip")
        
        print(f"Calculating Local Sliding Z-scores (Window Size = {Z_SCORE_WINDOW_SIZE}x{Z_SCORE_WINDOW_SIZE}, Stride = {SLIDING_STRIDE})...")
        for t in range(num_frames):
            frame_vols = ds_sliding_vol[t, ...]
            
            z_frame = sc.calculate_local_z_score(frame_vols, Z_SCORE_WINDOW_SIZE, SLIDING_STRIDE)
                
            ds_local_z_score[t, ...] = z_frame
            print(f"  Frame {t+1}/{num_frames} processed.")
            
        # Copy original attributes and add statistical metadata
        for k, v in ds_sliding_vol.attrs.items():
            ds_local_z_score.attrs[k] = v
            ds_global_z_score.attrs[k] = v
            
        ds_local_z_score.attrs['description'] = "Local Adaptive Z-score of the local spectral complexity. (Log(volume)-Log(mean_volume))/Std(Log(volume))"
        ds_local_z_score.attrs['z_score_window_size'] = Z_SCORE_WINDOW_SIZE
        ds_local_z_score.attrs['z_score_sliding_stride'] = SLIDING_STRIDE
        ds_global_z_score.attrs['description'] = "Global Spectral Complexity Z-score. (Log(volume)-Log(mean_volume))/Std(Log(volume))"


    print("\nLocal Z-score calculation complete and saved to file.")

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    
    print("Please select the processed HDF5 Image Stack...")
    file_path = filedialog.askopenfilename(
        title="Select HDF5 Image Stack",
        filetypes=[("HDF5 files", "*.h5")]
    )
    
    if file_path:
        process_file_rgb(file_path)
    else:
        print("No file selected.")
    
    root.destroy()