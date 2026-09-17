import os
import h5py
import numpy as np
import yaml
from pathlib import Path
from rasterio.transform import Affine
from pyproj import CRS, Transformer

LOCATIONS_YAML = r"f:\Resilio\IMGS 890 Research\Spectral-Complexity-dev\locations_config.yaml"
BASE_DIR = r"C:\satelliteImagery"

def distribute_water_mask(location):
    print(f"Distributing water mask for {location}...")
    
    # 1. Find the HLS files and load the persistent water mask
    hls_files = list(Path(f"{BASE_DIR}/HLS30").glob(f"HLS_{location}_*.h5"))
    if not hls_files:
        print("No HLS stack found, cannot distribute water mask.")
        return
        
    # Collect all HLS tiles' water masks
    hls_masks_info = []
    
    for hf in hls_files:
        with h5py.File(hf, 'r') as f:
            for grp in f['/HDFEOS/GRIDS']:
                if 'persistent_water_mask' in f[f'/HDFEOS/GRIDS/{grp}/Data Fields']:
                    hls_mask = f[f'/HDFEOS/GRIDS/{grp}/Data Fields/persistent_water_mask'][:]
                    sr_attrs = f[f'/HDFEOS/GRIDS/{grp}/Data Fields/surface_reflectance'].attrs
                    hls_transform = Affine.from_gdal(*sr_attrs['GeoTransform'])
                    hls_crs = CRS.from_wkt(sr_attrs['spatial_ref'])
                    hls_masks_info.append({
                        'mask': hls_mask,
                        'transform': hls_transform,
                        'crs': hls_crs
                    })
            
    if not hls_masks_info:
        print("No persistent_water_mask found in HLS files.")
        return
        
    # 2. Iterate through other sensors
    other_sensors = [
        Path(f"{BASE_DIR}/Tanager/Tanager_Native_Stack_{location}.h5"),
        Path(f"{BASE_DIR}/enmap/EnMAP_Native_Stack_{location}.h5"),
        Path(f"{BASE_DIR}/dragonette/Dragonette_Native_Stack_{location}.h5")
    ]
    
    for sf in other_sensors:
        if not sf.exists(): continue
        
        with h5py.File(sf, 'r+') as f:
            for grp in f['/HDFEOS/GRIDS']:
                df = f[f'/HDFEOS/GRIDS/{grp}/Data Fields']
                
                sr_attrs = df['surface_reflectance'].attrs
                if 'GeoTransform' not in sr_attrs or 'spatial_ref' not in sr_attrs:
                    print(f"Skipping {sf.name} ({grp}) because it is missing GeoTransform or spatial_ref. The native stack may be incomplete.")
                    continue
                
                native_transform = Affine.from_gdal(*sr_attrs['GeoTransform'])
                spatial_ref_str = sr_attrs['spatial_ref']
                if isinstance(spatial_ref_str, bytes):
                    spatial_ref_str = spatial_ref_str.decode('utf-8')
                    
                if spatial_ref_str.startswith('EPSG:'):
                    native_crs = CRS.from_string(spatial_ref_str)
                else:
                    native_crs = CRS.from_wkt(spatial_ref_str)
                
                shape = df['surface_reflectance'].shape
                if len(shape) == 4:
                    num_frames, _, native_h, native_w = shape
                else:
                    num_frames, native_h, native_w = shape
                
                # Build target native coordinates
                rows, cols = np.meshgrid(np.arange(native_h), np.arange(native_w), indexing='ij')
                native_x = native_transform.c + cols * native_transform.a
                native_y = native_transform.f + rows * native_transform.e
                
                reprojected_water = np.zeros((native_h, native_w), dtype=np.uint8)
                
                # Project native coordinates to each HLS CRS and accumulate
                for hls_info in hls_masks_info:
                    hls_crs = hls_info['crs']
                    hls_transform = hls_info['transform']
                    hls_mask = hls_info['mask']
                    hls_h, hls_w = hls_mask.shape
                    
                    if native_crs != hls_crs:
                        proj = Transformer.from_crs(native_crs, hls_crs, always_xy=True)
                        hls_x, hls_y = proj.transform(native_x, native_y)
                    else:
                        hls_x, hls_y = native_x, native_y
                        
                    # Apply inverse HLS affine to get pixel coordinates in HLS grid
                    inv = ~hls_transform
                    hls_col_f, hls_row_f = inv * (hls_x, hls_y)
                    
                    row_map = np.round(hls_row_f).astype(np.int32)
                    col_map = np.round(hls_col_f).astype(np.int32)
                    
                    valid = (row_map >= 0) & (row_map < hls_h) & (col_map >= 0) & (col_map < hls_w)
                    
                    # Extract the water mask mapped to the native grid (logical OR)
                    reprojected_water[valid] |= hls_mask[row_map[valid], col_map[valid]]
                
                # Merge with existing common_mask
                if 'common_mask' in df:
                    existing = df['common_mask'][:]
                    # Convert reprojected_water to bool matching existing
                    reprojected_water_3d = np.broadcast_to(reprojected_water == 1, existing.shape)
                    merged_mask = existing | reprojected_water_3d
                    
                    # Ensure attributes are preserved
                    desc = df['common_mask'].attrs.get('description', '')
                    c_dil = df['common_mask'].attrs.get('cloud_dilation')
                    sun_elev = df['common_mask'].attrs.get('sun_elevation_threshold')
                    
                    del df['common_mask']
                    dset = df.create_dataset('common_mask', data=merged_mask, compression='gzip')
                    
                    dset.attrs['description'] = desc + " Merged with HLS water mask."
                    if c_dil is not None: dset.attrs['cloud_dilation'] = c_dil
                    if sun_elev is not None: dset.attrs['sun_elevation_threshold'] = sun_elev
                else:
                    common_mask = np.broadcast_to(reprojected_water == 1, (num_frames, native_h, native_w))
                    df.create_dataset('common_mask', data=common_mask, compression='gzip')
                print(f"Written merged common_mask to {sf.name} ({grp})")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--location")
    args = parser.parse_args()
    distribute_water_mask(args.location)
