import os
import json
import h5py
import numpy as np
import rasterio
from pyproj import Transformer, CRS
from rasterio.warp import reproject, Resampling
from rasterio.transform import Affine
from pathlib import Path
from datetime import datetime, timezone
from tqdm import tqdm
import math
import sys
import yaml
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import hdfeos_odl
import SpecComplex as sc
from Harmonized_SC.harmonize_hls import process_hls_master_stack

# --- Configuration ---
TIME_THRESHOLD_SECONDS = 120
SOURCE_DIR = "C:/satelliteImagery/HLS30"

TARGET_RESOLUTION = 30.0

MIN_ROI_COVERAGE_PERCENT = 20.0 
SUN_ELEVATION_THRESHOLD = 20
HLS_CLOUD_DILATION = 0
QA_REJECT_MASK = 0b11111
AEROSOL_ACCEPT_LEVEL = 'medium'

S30_WAVELENGTHS = np.array([0.443, 0.490, 0.560, 0.665, 0.705, 0.740, 0.783, 0.842, 1.610, 2.190], dtype=np.float32)
L30_WAVELENGTHS = np.array([0.443, 0.482, 0.561, 0.655, 0.865, 1.609, 2.201], dtype=np.float32)

def get_utm_epsg_from_lonlat(lon, lat):
    zone = int((lon + 180) / 6) + 1
    if lat >= 0:
        return 32600 + zone
    else:
        return 32700 + zone

def calculate_mgrs_aligned_grid(roi_bbox):
    min_lon, min_lat, max_lon, max_lat = roi_bbox
    center_lon = (min_lon + max_lon) / 2.0
    center_lat = (min_lat + max_lat) / 2.0
    
    epsg_code = get_utm_epsg_from_lonlat(center_lon, center_lat)
    target_crs = f"EPSG:{epsg_code}"
    
    corners = [
        (min_lon, max_lat),
        (max_lon, max_lat),
        (max_lon, min_lat),
        (min_lon, min_lat)
    ]
    
    transformer = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
    xs, ys = zip(*corners)
    proj_xs, proj_ys = transformer.transform(xs, ys)
    
    min_x, max_x = min(proj_xs), max(proj_xs)
    min_y, max_y = min(proj_ys), max(proj_ys)
    
    # Snap boundaries to exact 30-meter multiples
    ul_x = math.floor(min_x / 30.0) * 30.0
    ul_y = math.ceil(max_y / 30.0) * 30.0
    lr_x = math.ceil(max_x / 30.0) * 30.0
    lr_y = math.floor(min_y / 30.0) * 30.0
    
    width = int((lr_x - ul_x) / 30.0)
    height = int((ul_y - lr_y) / 30.0)
    
    target_transform = Affine.translation(ul_x, ul_y) * Affine.scale(30.0, -30.0)
    return target_crs, target_transform, width, height

def fetch_native_hls_groups(native_h5_path, sensor_prefix):
    if not os.path.exists(native_h5_path):
        raise FileNotFoundError(f"Native HLS Truth file missing at {native_h5_path}")
    
    daily_groups = {}
    unique_tiles = set()

    with h5py.File(native_h5_path, 'r') as h5f:
        grid_groups = [k for k in h5f['HDFEOS/GRIDS'].keys() if k.startswith(sensor_prefix)]
        for grid_id in grid_groups:
            tile_name = grid_id.split('_')[1] 
            unique_tiles.add(tile_name)
        
            sr_ds = h5f[f'HDFEOS/GRIDS/{grid_id}/Data Fields/surface_reflectance']
            acq_times = sr_ds.attrs['acquisition_time']
        
            for f_idx, ts in enumerate(acq_times):
                dt_str = datetime.fromtimestamp(ts, tz=timezone.utc).strftime('%Y-%m-%d')
                if dt_str not in daily_groups: daily_groups[dt_str] = []
                daily_groups[dt_str].append({'tile': tile_name, 'grid_id': grid_id, 'frame_idx': f_idx})
            
    return daily_groups, unique_tiles

def write_hdf_sensor_group(h5f, group_path, data_dict, wavelengths, crs_wkt, transform, target_location):
    if not data_dict or data_dict['count'] == 0: return
    grp = h5f.create_group(group_path)
    gdal_transform = np.array([transform.c, transform.a, transform.b, transform.f, transform.d, transform.e], dtype='float64')
    dt = h5py.string_dtype(encoding='ascii')
    
    num_frames, bands, h, w = data_dict['sr'].shape
    chunk_h, chunk_w = min(h, 256), min(w, 256)

    sr_ds = grp.create_dataset('surface_reflectance', data=data_dict['sr'], dtype='int16', compression='gzip', compression_opts=5, shuffle=True, fillvalue=-9999, chunks=(1, bands, chunk_h, chunk_w))
    sr_ds.attrs['units'] = "Reflectance"
    sr_ds.attrs['_FillValue'] = -9999
    sr_ds.attrs['wavelengths'] = wavelengths
    sr_ds.attrs['spatial_ref'] = crs_wkt
    sr_ds.attrs['GeoTransform'] = gdal_transform

    fmask_ds = grp.create_dataset('Fmask', data=data_dict['fm'][:, 0, :, :], dtype='uint8', compression='gzip', shuffle=True, compression_opts=5, chunks=(1, chunk_h, chunk_w))
    fmask_ds.attrs['_FillValue'] = 255
    
    ang_ds = grp.create_dataset('solar_view_angles', data=data_dict['ag'], compression='gzip', shuffle=True, compression_opts=5, chunks=(1, 4, chunk_h, chunk_w))
    ang_ds.attrs['_FillValue'] = np.nan
    ang_ds.attrs['band_order'] = ["SZA", "SAA", "VZA", "VAA"]

    vis_ds = grp.create_dataset('ortho_visual', data=data_dict['vis'], dtype='uint8', compression='gzip', shuffle=True, compression_opts=5, chunks=(1, 4, chunk_h, chunk_w))
    vis_ds.attrs['spatial_ref'] = crs_wkt
    vis_ds.attrs['GeoTransform'] = gdal_transform

    mask_ds = grp.create_dataset('common_mask', data=data_dict['mask'], dtype=bool, compression='gzip', compression_opts=5, shuffle=True, chunks=(1, chunk_h, chunk_w))
    mask_ds.attrs['description'] = "True = Invalid/Masked, False = Valid."
    mask_ds.attrs['spatial_ref'] = crs_wkt
    mask_ds.attrs['GeoTransform'] = gdal_transform
    mask_ds.attrs['qa_reject_mask'] = QA_REJECT_MASK
    mask_ds.attrs['cloud_dilation'] = HLS_CLOUD_DILATION
    mask_ds.attrs['aerosol_accept_level'] = AEROSOL_ACCEPT_LEVEL
    mask_ds.attrs['sun_elevation_threshold'] = SUN_ELEVATION_THRESHOLD
    
    # WATER MASK Generation
    # Extract Bit 5 (value 32) from Fmask
    fmask_data = data_dict['fm'][:, 0, :, :]
    water_mask_data = (fmask_data & 0b100000) != 0
    water_ds = grp.create_dataset('water_mask', data=water_mask_data, dtype=bool, compression='gzip', shuffle=True, compression_opts=5, chunks=(1, chunk_h, chunk_w))
    water_ds.attrs['description'] = "True = Water, False = Non-Water. Derived from HLS Fmask Bit 5."
    water_ds.attrs['spatial_ref'] = crs_wkt
    water_ds.attrs['GeoTransform'] = gdal_transform

    sr_ds.attrs.create('spacecraft_id', data=np.array(data_dict['meta']['space'], dtype=dt))
    sr_ds.attrs['acquisition_time'] = np.array(data_dict['meta']['acq'], dtype='float64') 
    sr_ds.attrs['sun_azimuth'] = np.array(data_dict['meta']['saz'], dtype='float32')
    sr_ds.attrs['sun_elevation'] = np.array(data_dict['meta']['sel'], dtype='float32')
    sr_ds.attrs['cloud_cover'] = np.array(data_dict['meta']['cc'], dtype='float32')
    if 'scale_to_float' in data_dict['meta']:
        sr_ds.attrs['scale_to_float'] = data_dict['meta']['scale_to_float']
    else:
        warnings.warn(f"scale_to_float not found in metadata for {group_path}")
    
    # Export PNGs
    print(f"  Exporting visual frames as PNGs for {group_path}...")
    sensor_name = group_path.split('/')[3]
    location_dir = os.path.join(SOURCE_DIR, f"{target_location}_{sensor_name}")
    if not os.path.exists(location_dir):
        os.makedirs(location_dir)
        
    for t_idx in range(num_frames):
        acq_ts = data_dict['meta']['acq'][t_idx]
        pass_ts = datetime.fromtimestamp(acq_ts, tz=timezone.utc).strftime("%Y%m%dT%H%M%S")
        
        rgba_frame = np.transpose(data_dict['vis'][t_idx, ...], (1, 2, 0))
        img = Image.fromarray(rgba_frame, 'RGBA')
        png_filename = f"{sensor_name}_{target_location}_{pass_ts}_RGB.png"
        img.save(os.path.join(location_dir, png_filename))
        
        frame_mask = data_dict['mask'][t_idx, ...]
        overlay = Image.new('RGBA', img.size, (255, 166, 0, 0))
        overlay_data = np.array(overlay)
        overlay_data[frame_mask == True] = [255, 166, 0, 128]
        overlay_img = Image.fromarray(overlay_data, 'RGBA')
        masked_img = Image.alpha_composite(img, overlay_img)
        
        masked_png_name = f"{sensor_name}_{target_location}_{pass_ts}_RGB_masked.png"
        masked_img.save(os.path.join(location_dir, masked_png_name))
        print(f"    Saved: {masked_png_name}")

def process_hls_mgrs_stack(target_location):
    script_dir = Path(__file__).resolve().parent
    config_path = os.path.join(script_dir.parent, "locations_config.yaml")
    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)
        
    loc_config = config_data['locations'].get(target_location)
    if not loc_config:
        raise ValueError(f"Location {target_location} not found in locations_config.yaml")
        
    roi_bbox = (loc_config['ROI_LON_MIN'], loc_config['ROI_LAT_MIN'], loc_config['ROI_LON_MAX'], loc_config['ROI_LAT_MAX'])
    
    target_crs_str, target_transform, width, height = calculate_mgrs_aligned_grid(roi_bbox)
    target_crs = CRS.from_string(target_crs_str)
    
    input_native_h5 = os.path.join(SOURCE_DIR, f"HLS_{target_location}_STAC_Native.h5")
    output_mgrs_h5 = os.path.join(SOURCE_DIR, f"HLS_{target_location}_MGRS_Stack.h5")
    
    print(f"MGRS Grid Established: {width}x{height} at {target_crs_str}")
    
    s30_daily, s30_tiles = fetch_native_hls_groups(input_native_h5, "HLSS30")
    l30_daily, l30_tiles = fetch_native_hls_groups(input_native_h5, "HLSL30")
    
    with h5py.File(output_mgrs_h5, 'w') as h5f:
        print("Processing HLSS30 MGRS Stack...")
        s30_master_data = process_hls_master_stack(
            input_native_h5, s30_daily, 10, height, width, target_transform, target_crs,
            MIN_ROI_COVERAGE_PERCENT, SUN_ELEVATION_THRESHOLD, HLS_CLOUD_DILATION, QA_REJECT_MASK, AEROSOL_ACCEPT_LEVEL
        )
        if s30_master_data:
            write_hdf_sensor_group(h5f, '/HDFEOS/GRIDS/HLSS30/Data Fields', s30_master_data, S30_WAVELENGTHS, target_crs.to_wkt(), target_transform, target_location)
            del s30_master_data
            import gc
            gc.collect()
            
        print("Processing HLSL30 MGRS Stack...")
        l30_master_data = process_hls_master_stack(
            input_native_h5, l30_daily, 7, height, width, target_transform, target_crs,
            MIN_ROI_COVERAGE_PERCENT, SUN_ELEVATION_THRESHOLD, HLS_CLOUD_DILATION, QA_REJECT_MASK, AEROSOL_ACCEPT_LEVEL
        )
        if l30_master_data:
            write_hdf_sensor_group(h5f, '/HDFEOS/GRIDS/HLSL30/Data Fields', l30_master_data, L30_WAVELENGTHS, target_crs.to_wkt(), target_transform, target_location)

    print(f"\nHLS MGRS Stack completed: {output_mgrs_h5}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--location", type=str, required=True, help="Target location prefix")
    args = parser.parse_args()
    process_hls_mgrs_stack(args.location)
