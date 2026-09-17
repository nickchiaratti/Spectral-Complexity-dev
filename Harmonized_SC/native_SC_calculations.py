import os
import h5py
import numpy as np
import time
import argparse
import sys
import tkinter as tk
from tkinter import filedialog
from pathlib import Path

script_dir = Path(__file__).resolve().parent
if str(script_dir.parent) not in sys.path:
    sys.path.insert(0, str(script_dir.parent))
import SpecComplex as sc
import SpecComplexTorch as scTorch
import warnings

def overwrite_dset(data_grp, name, shape, dtype='float32', spatial_ref=None, geo_transform=None, chunks=None, scale_factor=None, fill_value=None, **kwargs):
    if name in data_grp:
        del data_grp[name]

    if fill_value is not None:
        kwargs['fillvalue'] = fill_value
        
    ds = data_grp.create_dataset(name, shape=shape, dtype=dtype, compression="gzip", compression_opts=5, chunks=chunks, shuffle=True, **kwargs)

    if spatial_ref is not None: ds.attrs['spatial_ref'] = spatial_ref
    if geo_transform is not None: ds.attrs['GeoTransform'] = geo_transform
    
    if scale_factor is not None: ds.attrs['scale_factor'] = scale_factor
    if fill_value is not None: ds.attrs['_FillValue'] = fill_value
    
    return ds

def process_native_cube(filepath):
    print(f"\nProcessing Native Sensor Cube: {filepath}")
    t_start = time.perf_counter()
    
    NUM_ENDMEMBERS = 7
    scale_fac_z = 1000.0
    fill_val = -32768
    
    with h5py.File(filepath, 'r+') as h5_file:
        grids_grp = h5_file['/HDFEOS/GRIDS']
        grids = [g for g in grids_grp.keys() if g != 'HARMONIZED']
            
        if not grids:
            raise ValueError("No sensor grids found.")
            
        sensor = grids[0]
        print(f"Detected sensor: {sensor}")
        
        data_grp_path = f"/HDFEOS/GRIDS/{sensor}/Data Fields"
        if data_grp_path not in h5_file:
            raise ValueError(f"Data Fields missing for {sensor}")
            
        data_grp = h5_file[data_grp_path]
        
        derived_grp_path = f"/HDFEOS/GRIDS/{sensor}/Derived Fields"
        if derived_grp_path not in h5_file:
            derived_grp = h5_file.create_group(derived_grp_path)
        else:
            derived_grp = h5_file[derived_grp_path]
            
        if 'surface_reflectance' not in data_grp or 'common_mask' not in data_grp:
            print(f"Skipping {file_path} because surface_reflectance or common_mask is missing (native stack likely incomplete).")
            return
            
        sr_ds = data_grp['surface_reflectance']
        mask_ds = data_grp['common_mask']
        
        n_frames, n_bands, height, width = sr_ds.shape
        spatial_ref = sr_ds.attrs.get('spatial_ref')
        geo_transform = sr_ds.attrs.get('GeoTransform')
        
        chunk_h, chunk_w = min(height, 256), min(width, 256)
        chunks_3d = (1, chunk_h, chunk_w)
        
        if "ENMAP" in sensor.upper():
            sensor_type = "ENMAP"
        elif "DRAGONETTE" in sensor.upper():
            sensor_type = "DRAGONETTE"
        elif "TANAGER" in sensor.upper():
            sensor_type = "TANAGER"
        else:
            sensor_type = "HLS"
            
        print(f"Total frames: {n_frames}")
        
        # Datasets
        vol_3x3_ds = overwrite_dset(derived_grp, 'sliding_volume_map_3x3', (n_frames, height, width), spatial_ref=spatial_ref, geo_transform=geo_transform, chunks=chunks_3d)
        vol_3x3_ds.attrs['description'] = "Volume of convex hull within sliding 3x3 tile"
        vol_3x3_ds.attrs['tile_size'] = 3
        vol_3x3_ds.attrs['sliding_stride'] = 1
        vol_3x3_ds.attrs['num_endmembers'] = NUM_ENDMEMBERS
        
        vol_18x18_ds = None
        if sensor_type == "DRAGONETTE":
            vol_18x18_ds = overwrite_dset(derived_grp, 'sliding_volume_map_18x18', (n_frames, height, width), spatial_ref=spatial_ref, geo_transform=geo_transform, chunks=chunks_3d)
            vol_18x18_ds.attrs['description'] = "Volume of convex hull within sliding 18x18 tile"
            vol_18x18_ds.attrs['tile_size'] = 18
            vol_18x18_ds.attrs['sliding_stride'] = 1
            vol_18x18_ds.attrs['num_endmembers'] = NUM_ENDMEMBERS
            
        z_3x3_ds = overwrite_dset(derived_grp, 'sliding_volume_3x3_zscore', (n_frames, height, width), dtype='int16', scale_factor=scale_fac_z, fill_value=fill_val, spatial_ref=spatial_ref, geo_transform=geo_transform, chunks=chunks_3d)
        z_3x3_ds.attrs['description'] = "Global Spatial Z-score of 3x3 sliding volume evaluated for the frame"
        z_3x3_ds.attrs['tile_size'] = 3
        
        ndvi_ds = overwrite_dset(derived_grp, 'ndvi', (n_frames, height, width), dtype='int16', scale_factor=10000, fill_value=fill_val, spatial_ref=spatial_ref, geo_transform=geo_transform, chunks=chunks_3d)
        ndvi_ds.attrs['description'] = "Normalized Difference Vegetation Index"
        
        # Get sensor indices for NDVI
        if sensor_type == "ENMAP":
            wl = sr_ds.attrs.get('wavelengths', np.zeros(n_bands))
            red_idx = int(np.argmin(np.abs(wl - 650)))
            nir_idx = int(np.argmin(np.abs(wl - 850)))
        elif sensor_type == "DRAGONETTE":
            wl = sr_ds.attrs.get('wavelengths', np.zeros(n_bands))
            red_idx = int(np.argmin(np.abs(wl - 650)))
            nir_idx = int(np.argmin(np.abs(wl - 850)))
        elif sensor_type == "TANAGER":
            red_idx, nir_idx = 59, 97
        else:
            red_idx, nir_idx = 3, 4
        
        # Process frames
        for t in range(n_frames):
            t0 = time.perf_counter()
            print(f"Processing frame {t+1}/{n_frames}...", end="", flush=True)
            
            with warnings.catch_warnings(), np.errstate(all='ignore'):
                warnings.simplefilter("ignore")
                
                if sensor_type == "ENMAP":
                    frame_sr, _ = sc.load_enmap_sr(sr_ds, t, np.s_[:, :])
                elif sensor_type == "TANAGER":
                    frame_sr, _ = sc.load_tanager_sr(sr_ds, t, np.s_[:, :])
                else:
                    frame_sr = sc.load_scaled_reflectance(sr_ds, np.s_[t, :, :, :])
                    
                frame_mask = mask_ds[t, :, :]
                
                # Exclude common_mask pixels from the metric kernel
                # The shape of frame_sr is (bands, height, width)
                frame_sr[:, frame_mask == 1] = np.nan
                
                # 3x3 Volume
                slide_result = scTorch.process_volume_sliding_tile(
                    frame_sr, tile_size=3, stride=1, num_endmembers=NUM_ENDMEMBERS
                )
                vol_3x3_ds[t, :, :] = slide_result[0]
                
                # Z-score
                valid_pixel_mask = (frame_mask == 0)
                z_score_frame, _, _ = sc.calculate_global_z_score(slide_result[0], valid_pixel_mask)
                z_3x3_ds[t, :, :] = sc.scale_to_int16(z_score_frame, scale_factor=scale_fac_z)
                
                # NDVI
                ndvi_frame = sc.calc_ndvi_frame(frame_sr, red_idx=red_idx, nir_idx=nir_idx)
                ndvi_ds[t, :, :] = sc.scale_to_int16(ndvi_frame, scale_factor=10000)
                
                if sensor_type == "DRAGONETTE":
                    slide_result_18 = scTorch.process_volume_sliding_tile(
                        frame_sr, tile_size=18, stride=1, num_endmembers=NUM_ENDMEMBERS
                    )
                    vol_18x18_ds[t, :, :] = slide_result_18[0]
            
            print(f" done in {time.perf_counter() - t0:.2f}s")
                
        # Z-Scores now computed per-frame inside loop
        print(f"Total Compute Time: {(time.perf_counter() - t_start) / 60:.2f} minutes")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate Native SC Metrics")
    parser.add_argument("--file", type=str, help="Path to Native HDF5 Cube")
    args = parser.parse_args()
    
    file_path = args.file
    
    if not file_path:
        root = tk.Tk()
        root.withdraw()
        print("Please select the Native Sensor HDF5 Cube...")
        file_path = filedialog.askopenfilename(
            title="Select Native Sensor HDF5 Cube",
            filetypes=[("HDF5 files", "*.h5")]
        )
        root.destroy()

    if file_path and os.path.exists(file_path):
        process_native_cube(file_path)
    else:
        print("No valid file selected. Exiting.")
