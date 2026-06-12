import os
from pathlib import Path
import h5py
import numpy as np
from scipy.signal import savgol_filter

SOURCE_DIR = "C:/satelliteImagery/Tanager/Rochesterv2_SourceData"

def apply_sg_filter():
    source_path = Path(SOURCE_DIR)
    py6s_files = list(source_path.rglob("*_basic_6Ssr_hdf5.h5"))
    
    if not py6s_files:
        print("No processed 6S basic swath files found.")
        return
        
    # SG Filter parameters
    window_length = 15 # Must be odd, e.g., 15 bands
    polyorder = 3
    
    for py6s_path in py6s_files:
        print(f"Processing {py6s_path.name}...")
        
        with h5py.File(py6s_path, 'r+') as f:
            grp = f['HDFEOS/SWATHS/HYP/Data Fields']
            
            if 'surface_reflectance' not in grp:
                print("  Missing 'surface_reflectance' dataset. Skipping.")
                continue
                
            sr_ds = grp['surface_reflectance']
            fill_val = sr_ds.attrs.get('_FillValue', -9999.0)
            if isinstance(fill_val, (np.ndarray, list, tuple)):
                fill_val = fill_val[0]
                
            print("  Reading data...")
            sr_data = sr_ds[()]
                
            if 'surface_reflectance_SG' in grp:
                print("  'surface_reflectance_SG' already exists. Overwriting in-place...")
                sg_ds = grp['surface_reflectance_SG']
            else:
                sg_ds = grp.create_dataset(
                    "surface_reflectance_SG",
                    shape=sr_data.shape,
                    dtype=sr_data.dtype,
                    chunks=sr_ds.chunks,
                    compression="gzip",
                    fillvalue=fill_val
                )
            
            import pandas as pd
            
            sg_data = np.full_like(sr_data, fill_val)
            
            # Find valid spatial pixels (where data has ANY valid bands)
            valid_spatial_mask = (sr_data != fill_val).any(axis=0)
            
            print("  Interpolating missing bands and applying Savitzky-Golay filter...")
            valid_pixels = sr_data[:, valid_spatial_mask]
            
            # Temporarily replace fill_val with NaN for interpolation
            valid_pixels[valid_pixels == fill_val] = np.nan
            
            # Pandas is significantly faster interpolating across columns (axis=1) on a transposed dataset
            df = pd.DataFrame(valid_pixels.T)
            df.interpolate(method='linear', axis=1, inplace=True)
            df.bfill(axis=1, inplace=True)
            df.ffill(axis=1, inplace=True)
            interpolated_pixels = df.values.T
            
            # Apply filter over axis=0 (spectral axis)
            smoothed_pixels = savgol_filter(interpolated_pixels, window_length=window_length, polyorder=polyorder, axis=0)
            
            # Restore original fill values
            original_fill_mask = sr_data[:, valid_spatial_mask] == fill_val
            smoothed_pixels[original_fill_mask] = fill_val
            
            # Put back into output array
            sg_data[:, valid_spatial_mask] = smoothed_pixels
            
            print("  Saving 'surface_reflectance_SG'...")
            sg_ds[...] = sg_data
            
            # Copy attributes
            for k, v in sr_ds.attrs.items():
                sg_ds.attrs[k] = v
                
        print(f"  Done with {py6s_path.name}\n")

if __name__ == "__main__":
    apply_sg_filter()
