import os
import h5py
import rasterio
import numpy as np
from datetime import datetime, timezone
from rasterio.windows import from_bounds
from rasterio.transform import from_bounds as transform_from_bounds
from rasterio.warp import reproject, Resampling
from pyproj import Transformer, CRS
import pystac_client
import earthaccess
import SpecComplex as sc
import json
import concurrent.futures
import warnings
from pathlib import Path
import re

# ==========================================
# 1. CONFIGURATION & AUTHENTICATION
# ==========================================
cloud_threshold = 80

print("Authenticating with NASA Earthdata...")
earthaccess.login(strategy="all", persist=True)

Location = "Rochesterv2"
SOURCE_CACHE = "Rochesterv2" 

if Location == "Rochesterv2":
    ROI_LON_MIN = -77.770166; ROI_LON_MAX = -77.376776
    ROI_LAT_MIN = 42.961778; ROI_LAT_MAX = 43.342135
    START_DATE = '2025-01-01'
    END_DATE = '2025-12-31'

HLSS30_OUTPUT_DIR = r"C:\satelliteImagery\HLSX30\HLSS30-SourceData"
HLSL30_OUTPUT_DIR = r"C:\satelliteImagery\HLSX30\HLSL30-SourceData"
COMBINED_OUTPUT_DIR = r"C:\satelliteImagery\HLSX30"

if SOURCE_CACHE:
    S30_TEMP_DIR = os.path.join(HLSS30_OUTPUT_DIR, f"{SOURCE_CACHE}/STAC_CACHE")
    L30_TEMP_DIR = os.path.join(HLSL30_OUTPUT_DIR, f"{SOURCE_CACHE}/STAC_CACHE")
else:
    S30_TEMP_DIR = os.path.join(HLSS30_OUTPUT_DIR, f"{Location}/STAC_CACHE")
    L30_TEMP_DIR = os.path.join(HLSL30_OUTPUT_DIR, f"{Location}/STAC_CACHE")

os.makedirs(S30_TEMP_DIR, exist_ok=True)
os.makedirs(L30_TEMP_DIR, exist_ok=True)
os.makedirs(COMBINED_OUTPUT_DIR, exist_ok=True)

OUTPUT_NATIVE_HDF5 = os.path.join(COMBINED_OUTPUT_DIR, f"HLS_Combined_Stack_{Location}_STAC_Native_2025.h5")

ASSETS_S30 = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12', 'Fmask', 'SZA', 'SAA', 'VZA', 'VAA']
ASSETS_L30 = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B09', 'B10', 'B11', 'Fmask', 'SZA', 'SAA', 'VZA', 'VAA']

S30_WAVELENGTHS = [0.443, 0.490, 0.560, 0.665, 0.705, 0.740, 0.783, 0.842, 0.865, 0.945, 1.375, 1.610, 2.190]
L30_SR_WAVELENGTHS = [0.443, 0.482, 0.561, 0.655, 0.865, 1.609, 2.201, 1.373] 
L30_TIRS_WAVELENGTHS = [10.9, 12.0]

# Defensive Topology Enforcement for STAC
safe_bbox = [
    min(ROI_LON_MIN, ROI_LON_MAX), max(ROI_LAT_MIN, ROI_LAT_MAX), 
    max(ROI_LON_MIN, ROI_LON_MAX), min(ROI_LAT_MIN, ROI_LAT_MAX)
]
safe_bbox = [min(safe_bbox[0], safe_bbox[2]), min(safe_bbox[1], safe_bbox[3]), max(safe_bbox[0], safe_bbox[2]), max(safe_bbox[1], safe_bbox[3])]

# ==========================================
# 2. NASA STAC QUERY & NATIVE VIRTUAL READ (HLS)
# ==========================================
def stac_native_window_read(collection_id, assets_list, temp_dir):
    print(f"\nQuerying NASA CMR STAC for {collection_id}...")
    catalog = pystac_client.Client.open("https://cmr.earthdata.nasa.gov/stac/LPCLOUD")
    search = catalog.search(collections=[collection_id], bbox=safe_bbox, datetime=f"{START_DATE}/{END_DATE}", limit=500)
    filtered_items = [i for i in list(search.items()) if i.properties.get('eo:cloud_cover', 100) < cloud_threshold]
    
    tile_collections = {}
    gdal_env = {
        'GDAL_HTTP_COOKIEFILE': os.path.expanduser('~/.urs_cookies'),
        'GDAL_HTTP_COOKIEJAR': os.path.expanduser('~/.urs_cookies'),
        'GDAL_DISABLE_READDIR_ON_OPEN': 'EMPTY_DIR',
        'CPL_VSIL_CURL_ALLOWED_EXTENSIONS': 'tif',
        'VSI_CACHE': True,
        'GDAL_HTTP_MULTIPLEX': 'YES'
    }
    
    def _fetch_single_band(idx, asset_key, url, window_obj):
        if asset_key == 'Fmask': fill_val = 255
        elif asset_key in ['SZA', 'SAA', 'VZA', 'VAA']: fill_val = 40000
        else: fill_val = -9999
        with rasterio.open(url) as b_src:
            data = b_src.read(1, window=window_obj, boundless=True, fill_value=fill_val)
            if asset_key in ['SZA', 'SAA', 'VZA', 'VAA']:
                data = np.where(data == 40000, -9999, data).astype(np.int16)
        return idx, data

    with rasterio.Env(**gdal_env):
        for item in filtered_items:
            img_id = item.id 
            parsed_mgrs_tile = img_id.split('.')[2] 
            cloud_cov = item.properties.get('eo:cloud_cover')
            
            if cloud_cov is None: continue
            
            if parsed_mgrs_tile not in tile_collections:
                tile_collections[parsed_mgrs_tile] = {'items': {}, 'transform': None, 'crs': None, 'width': None, 'height': None, 'zone': None}
                
            tile_data = tile_collections[parsed_mgrs_tile]
            out_tif = os.path.join(temp_dir, f"{img_id}.tif")
            
            tile_data['items'][img_id] = {
                'acquisition_time': item.datetime.timestamp(),
                'spacecraft_id': item.properties.get('platform', 'UNKNOWN'),
                'cloud_cover': cloud_cov,
                'filepath': out_tif
            }
            
            if os.path.exists(out_tif) and os.path.getsize(out_tif) > 0:
                if tile_data['transform'] is None:
                    with rasterio.open(out_tif) as cached_src:
                        tile_data['transform'] = cached_src.transform
                        tile_data['crs'] = cached_src.crs
                        tile_data['width'] = cached_src.width
                        tile_data['height'] = cached_src.height
                        tile_data['zone'] = CRS.from_user_input(cached_src.crs).to_epsg() - 32600
                continue
                
            try:
                asset_key_ref = assets_list[0]
                with rasterio.open(item.assets[asset_key_ref].href) as src:
                    if tile_data.get('window') is None:
                        transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
                        xs, ys = transformer.transform([safe_bbox[0], safe_bbox[2], safe_bbox[2], safe_bbox[0]], [safe_bbox[3], safe_bbox[3], safe_bbox[1], safe_bbox[1]])
                        roi_minx, roi_maxx, roi_miny, roi_maxy = min(xs), max(xs), min(ys), max(ys)
                        window = from_bounds(roi_minx, roi_miny, roi_maxx, roi_maxy, transform=src.transform).round_offsets().round_lengths()
                        
                        tile_data['transform'] = src.window_transform(window)
                        tile_data['crs'] = src.crs
                        tile_data['width'] = window.width
                        tile_data['height'] = window.height
                        tile_data['zone'] = CRS.from_user_input(src.crs).to_epsg() - 32600
                        tile_data['window'] = window
                    
                    window = tile_data['window']
                
                num_assets = len(assets_list)
                compiled_array = np.zeros((num_assets, tile_data['height'], tile_data['width']), dtype=np.int16)
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=len(assets_list)) as executor:
                    futures = []
                    for idx, asset_key in enumerate(assets_list):
                        url = item.assets[asset_key].href
                        futures.append(executor.submit(_fetch_single_band, idx, asset_key, url, window))
                    for future in concurrent.futures.as_completed(futures):
                        b_idx, b_data = future.result()
                        compiled_array[b_idx, :, :] = b_data
                
                profile = {'driver': 'GTiff', 'height': tile_data['height'], 'width': tile_data['width'], 'count': num_assets, 'dtype': 'int16', 'crs': tile_data['crs'], 'transform': tile_data['transform'], 'compress': 'deflate'}
                with rasterio.open(out_tif, 'w', **profile) as dst: dst.write(compiled_array)
                    
            except Exception as e:
                print(f"Failed retrieval for {img_id}: {e}")
                del tile_data['items'][img_id]
                
    return tile_collections

s30_collections = stac_native_window_read("HLSS30.v2.0", ASSETS_S30, S30_TEMP_DIR)
l30_collections = stac_native_window_read("HLSL30.v2.0", ASSETS_L30, L30_TEMP_DIR)

# ==========================================
# 4. LOCAL DATASET PARSING & SUBSETTING (TANAGER)
# ==========================================
def is_roi_intersecting(json_path):
    with open(json_path, 'r') as f: data = json.load(f)
    s_min_lon, s_min_lat, s_max_lon, s_max_lat = data['bbox']
    overlap = not (safe_bbox[2] < s_min_lon or safe_bbox[0] > s_max_lon or safe_bbox[3] < s_min_lat or safe_bbox[1] > s_max_lat)
    return overlap, data

def extract_georeferencing_from_h5(h5_path):
    """Extracts internal georeferencing. Fails hard if ODL is missing or malformed."""
    with h5py.File(h5_path, 'r') as f:
        meta_path = "HDFEOS INFORMATION/StructMetadata.0"
        if meta_path not in f:
            raise ValueError(f"CRITICAL ERROR: StructMetadata.0 missing in {h5_path}")
        
        meta_data = f[meta_path][()]
        if isinstance(meta_data, (np.ndarray, list)): meta_data = meta_data[0]
        odl = meta_data.decode('ascii') if isinstance(meta_data, bytes) else str(meta_data)
        
        ul_match = re.search(r'UpperLeftPointMtrs=\(\s*(-?[\d\.]+)\s*,\s*(-?[\d\.]+)\s*\)', odl)
        lr_match = re.search(r'LowerRightMtrs=\(\s*(-?[\d\.]+)\s*,\s*(-?[\d\.]+)\s*\)', odl)
        x_match = re.search(r'XDim=(\d+)', odl)
        y_match = re.search(r'YDim=(\d+)', odl)
        zone_match = re.search(r'ZoneCode=(\d+)', odl)
        
        if all([ul_match, lr_match, x_match, y_match]):
            ul_x, ul_y = float(ul_match.group(1)), float(ul_match.group(2))
            lr_x, lr_y = float(lr_match.group(1)), float(lr_match.group(2))
            x_dim, y_dim = int(x_match.group(1)), int(y_match.group(1))
            zone = int(zone_match.group(1)) if zone_match else 18
            return from_bounds(ul_x, lr_y, lr_x, ul_y, x_dim, y_dim), CRS.from_dict({'proj': 'utm', 'zone': zone, 'datum': 'WGS84'})
        else:
            raise ValueError(f"CRITICAL ERROR: Incomplete ODL bounding geometry in {h5_path}")

def group_tanager_scenes():
    print(f"\nLocating Tanager Hyperspectral Data in: {TANAGER_SOURCE_DIR}")
    root_path = Path(TANAGER_SOURCE_DIR)
    raw_scenes = []
    
    if not root_path.exists():
        print("WARNING: Tanager source directory not found. Skipping Tanager ingestion.")
        return []

    for subfolder in root_path.iterdir():
        if not subfolder.is_dir(): continue
        json_path = list(subfolder.glob("*.json"))
        h5_path = list(subfolder.glob("*_ortho_sr_hdf5.h5"))
        vis_path = list(subfolder.glob("*_ortho_visual.tif"))
        
        if json_path and h5_path:
            is_overlap, stac_data = is_roi_intersecting(json_path[0])
            if is_overlap:
                dt = datetime.fromisoformat(stac_data['properties']['datetime'].replace('Z', '+00:00'))
                raw_scenes.append({'h5_file': str(h5_path[0]), 'vis_file': str(vis_path[0]) if vis_path else None, 'time': dt, 'json': stac_data})

    if not raw_scenes: return []
    raw_scenes.sort(key=lambda x: x['time'])

    grouped_scenes, current_group = [], [raw_scenes[0]]
    for i in range(1, len(raw_scenes)):
        if (raw_scenes[i]['time'] - current_group[-1]['time']).total_seconds() <= TANAGER_TIME_THRESHOLD:
            current_group.append(raw_scenes[i])
        else:
            grouped_scenes.append(current_group); current_group = [raw_scenes[i]]
    grouped_scenes.append(current_group)
    print(f"Found {len(raw_scenes)} valid Tanager frames, grouped into {len(grouped_scenes)} temporal mosaics.")
    return grouped_scenes

# ==========================================
# 5. HDFEOS5 ODL UTILITIES (MULTI-SENSOR)
# ==========================================
def generate_odl_grid_string(grid_name, width, height, transform, proj_code, zone, proj_params, num_sr_bands, num_frames, has_thermal, has_tile_mask=False):
    ul_x, ul_y = transform.c, transform.f
    lr_x = transform.c + (transform.a * width)
    lr_y = transform.f + (transform.e * height)
    p_str = str(tuple(proj_params)).replace(' ', '').replace('(', '').replace(')', '')
    
    thermal_dim = f"""            OBJECT=Dimension_3\n                DimensionName="ThermalBands"\n                Size=2\n            END_OBJECT=Dimension_3""" if has_thermal else ""
    thermal_field = f"""            OBJECT=DataField_2\n                DataFieldName="thermal_infrared"\n                DataType=HDF5T_NATIVE_FLOAT\n                DimList=("Time","ThermalBands","YDim","XDim")\n            END_OBJECT=DataField_2""" if has_thermal else ""
    fmask_idx = 3 if has_thermal else 2
    ang_idx = 4 if has_thermal else 3
    tm_idx = 5 if has_thermal else 4
    
    tile_mask_field = f"""            OBJECT=DataField_{tm_idx}\n                DataFieldName="source_tile_mask"\n                DataType=HDF5T_NATIVE_UINT8\n                DimList=("Time","YDim","XDim")\n            END_OBJECT=DataField_{tm_idx}""" if has_tile_mask else ""
    
    return f"""    GROUP={grid_name}
        GridName="{grid_name}"
        XDim={width}
        YDim={height}
        UpperLeftPointMtrs=({ul_x:.6f},{ul_y:.6f})
        LowerRightMtrs=({lr_x:.6f},{lr_y:.6f})
        Projection={proj_code}
        ZoneCode={zone}
        SphereCode=12
        ProjParams={p_str}
        GROUP=Dimension
            OBJECT=Dimension_1
                DimensionName="Time"
                Size={num_frames}
            END_OBJECT=Dimension_1
            OBJECT=Dimension_2
                DimensionName="Bands"
                Size={num_sr_bands}
            END_OBJECT=Dimension_2
{thermal_dim}
            OBJECT=Dimension_4
                DimensionName="YDim"
                Size={height}
            END_OBJECT=Dimension_4
            OBJECT=Dimension_5
                DimensionName="XDim"
                Size={width}
            END_OBJECT=Dimension_5
            OBJECT=Dimension_6
                DimensionName="AngleBands"
                Size=4
            END_OBJECT=Dimension_6
        END_GROUP=Dimension
        GROUP=DataField
            OBJECT=DataField_1
                DataFieldName="surface_reflectance"
                DataType=HDF5T_NATIVE_FLOAT
                DimList=("Time","Bands","YDim","XDim")
            END_OBJECT=DataField_1
{thermal_field}
            OBJECT=DataField_{fmask_idx}
                DataFieldName="Fmask"
                DataType=HDF5T_NATIVE_UINT8
                DimList=("Time","YDim","XDim")
            END_OBJECT=DataField_{fmask_idx}
            OBJECT=DataField_{ang_idx}
                DataFieldName="solar_view_angles"
                DataType=HDF5T_NATIVE_FLOAT
                DimList=("Time","AngleBands","YDim","XDim")
            END_OBJECT=DataField_{ang_idx}
{tile_mask_field}
        END_GROUP=DataField
        GROUP=MergedFields
        END_GROUP=MergedFields
    END_GROUP={grid_name}"""

def generate_tanager_odl_string(grid_name, width, height, transform, proj_code, zone, proj_params, datasets_info, n_times, n_bands):
    """Dynamically generates ODL for Tanager's hyperspectral dimensionality."""
    ul_x, ul_y = transform.c, transform.f
    lr_x = transform.c + (transform.a * width)
    lr_y = transform.f + (transform.e * height)
    p_str = str(tuple(proj_params)).replace(' ', '').replace('(', '').replace(')', '')
    
    data_fields_blocks = []
    for i, (name, dtype, rank, dim_names) in enumerate(datasets_info):
        eos_type = "HDF5T_NATIVE_FLOAT"
        if "uint8" in str(dtype): eos_type = "HDF5T_NATIVE_UINT8"
        elif "uint16" in str(dtype): eos_type = "HDF5T_NATIVE_UINT16"
        elif "uint" in str(dtype): eos_type = "HDF5T_NATIVE_UINT"
        elif "int" in str(dtype): eos_type = "HDF5T_NATIVE_INT"
        elif "float64" in str(dtype) or "double" in str(dtype): eos_type = "HDF5T_NATIVE_DOUBLE"
        
        dims_list = ",".join([f"\"{d}\"" for d in dim_names])
        block = f"""            OBJECT=DataField_{i+1}
                DataFieldName="{name}"
                DataType={eos_type}
                DimList=({dims_list})
            END_OBJECT=DataField_{i+1}"""
        data_fields_blocks.append(block)
        
    return f"""    GROUP={grid_name}
        GridName="{grid_name}"
        XDim={width}
        YDim={height}
        UpperLeftPointMtrs=({ul_x:.6f},{ul_y:.6f})
        LowerRightMtrs=({lr_x:.6f},{lr_y:.6f})
        Projection={proj_code}
        ZoneCode={zone}
        SphereCode=12
        ProjParams={p_str}
        GROUP=Dimension
            OBJECT=Dimension_1
                DimensionName="Time"
                Size={n_times}
            END_OBJECT=Dimension_1
            OBJECT=Dimension_2
                DimensionName="Band"
                Size={n_bands}
            END_OBJECT=Dimension_2
            OBJECT=Dimension_3
                DimensionName="YDim"
                Size={height}
            END_OBJECT=Dimension_3
            OBJECT=Dimension_4
                DimensionName="XDim"
                Size={width}
            END_OBJECT=Dimension_4
            OBJECT=Dimension_5
                DimensionName="VisBand"
                Size=4
            END_OBJECT=Dimension_5
        END_GROUP=Dimension
        GROUP=DataField
{"\n".join(data_fields_blocks)}
        END_GROUP=DataField
        GROUP=MergedFields
        END_GROUP=MergedFields
    END_GROUP={grid_name}"""

def write_hdf_sensor_group(h5f, group_path, data_dict, wavelengths, crs, transform, thermal_wavelengths=None, tile_name=None, tile_mapping_json=None):
    if not data_dict or data_dict['count'] == 0: return
    grp = h5f.create_group(group_path)
    gdal_transform = np.array([transform.c, transform.a, transform.b, transform.f, transform.d, transform.e], dtype='float64')
    dt = h5py.string_dtype(encoding='ascii')
    
    sr_ds = grp.create_dataset('surface_reflectance', data=data_dict['sr'], compression='gzip', compression_opts=6)
    sr_ds.attrs['units'] = "Reflectance"; sr_ds.attrs['_FillValue'] = np.nan; sr_ds.attrs['wavelengths'] = wavelengths
    sr_ds.attrs['spatial_ref'] = crs.to_wkt(); sr_ds.attrs['GeoTransform'] = gdal_transform
    
    if thermal_wavelengths and data_dict['th'] is not None:
        th_ds = grp.create_dataset('thermal_infrared', data=data_dict['th'], compression='gzip', compression_opts=6)
        th_ds.attrs['units'] = "Kelvin/Celsius Apparent"; th_ds.attrs['_FillValue'] = np.nan; th_ds.attrs['wavelengths'] = thermal_wavelengths
        
    fmask_ds = grp.create_dataset('Fmask', data=data_dict['fm'][:, 0, :, :], dtype='uint8', compression='gzip', compression_opts=6)
    fmask_ds.attrs['_FillValue'] = 255
    ang_ds = grp.create_dataset('solar_view_angles', data=data_dict['ag'], compression='gzip', compression_opts=6)
    ang_ds.attrs['_FillValue'] = np.nan; ang_ds.attrs['band_order'] = ["SZA", "SAA", "VZA", "VAA"]
    
    if 'tm' in data_dict and data_dict['tm'] is not None:
        tm_ds = grp.create_dataset('source_tile_mask', data=data_dict['tm'][:, 0, :, :], dtype='uint8', compression='gzip', compression_opts=6)
        tm_ds.attrs['_FillValue'] = 0
        tm_ds.attrs['description'] = "Integer mapping to source MGRS tile to track radiometric provenance."
        if tile_mapping_json: tm_ds.attrs['tile_mapping'] = tile_mapping_json
            
    sr_ds.attrs.create('spacecraft_id', data=data_dict['meta']['space'], dtype=dt)
    if tile_name: sr_ds.attrs.create('mgrs_tile', data=tile_name, dtype=dt)
    sr_ds.attrs['acquisition_time'] = np.array(data_dict['meta']['acq'], dtype='float64') 
    sr_ds.attrs['sun_azimuth'] = np.array(data_dict['meta']['saz'], dtype='float32')
    sr_ds.attrs['sun_elevation'] = np.array(data_dict['meta']['sel'], dtype='float32')
    sr_ds.attrs['cloud_cover'] = np.array(data_dict['meta']['cc'], dtype='float32')

# ==========================================
# 6. PHASE 1: BUILD NATIVE HDF5 (HLS UNPROJECTED TRUTH)
# ==========================================
print(f"\nBuilding Native Truth Data HDF5 (HLS Only): {OUTPUT_NATIVE_HDF5}")

def process_native_stack(tile_data, expected_sr, expected_thermal, expected_fmask_idx, expect_thermal_flag=False):
    items = tile_data['items']
    sorted_ids = sorted(items.keys(), key=lambda k: items[k]['acquisition_time'])
    num_frames = len(sorted_ids)
    if num_frames == 0: return None
    
    w, h = tile_data['width'], tile_data['height']
    stk_sr = np.zeros((num_frames, expected_sr, h, w), dtype=np.float32)
    stk_th = np.zeros((num_frames, expected_thermal, h, w), dtype=np.float32) if expect_thermal_flag else None
    stk_fm = np.zeros((num_frames, 1, h, w), dtype=np.uint8)
    stk_ag = np.zeros((num_frames, 4, h, w), dtype=np.float32)
    meta_arrays = {'acq': [], 'space': [], 'saz': [], 'sel': [], 'cc': []}
    
    valid_idx = 0
    for img_id in sorted_ids:
        meta = items[img_id]
        with rasterio.open(meta['filepath']) as src:
            t_sr = src.read(list(range(1, expected_sr+1)))
            stk_sr[valid_idx] = np.where(t_sr != -9999, t_sr.astype(np.float32) * 0.0001, np.nan)
            if expect_thermal_flag:
                off = expected_sr + 1
                t_th = src.read(list(range(off, off+expected_thermal)))
                stk_th[valid_idx] = np.where(t_th != -9999, t_th.astype(np.float32) * 0.01, np.nan)
            stk_fm[valid_idx] = src.read(expected_fmask_idx).astype(np.uint8)[np.newaxis, ...]
            t_ag = src.read(list(range(expected_fmask_idx+1, expected_fmask_idx+5)))
            stk_ag[valid_idx] = np.where(t_ag != -9999, t_ag.astype(np.float32) * 0.01, np.nan)

            for k, v in meta.items():
                if v is None and k != 'filepath': raise ValueError(f"CRITICAL ERROR: Metadata '{k}' for '{img_id}' is null.")

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                mean_sza = np.nanmean(stk_ag[valid_idx, 0])
                mean_saa = np.nanmean(stk_ag[valid_idx, 1])

            meta_arrays['acq'].append(meta['acquisition_time']); meta_arrays['space'].append(meta['spacecraft_id'])
            meta_arrays['saz'].append(mean_saa); meta_arrays['sel'].append(90.0 - mean_sza)
            meta_arrays['cc'].append(meta['cloud_cover'])
            valid_idx += 1

    return {'sr': stk_sr[:valid_idx], 'th': stk_th[:valid_idx] if expect_thermal_flag else None, 'fm': stk_fm[:valid_idx], 'ag': stk_ag[:valid_idx], 'meta': meta_arrays, 'count': valid_idx}

with h5py.File(OUTPUT_NATIVE_HDF5, 'w') as h5f:
    info_grp = h5f.create_group("HDFEOS INFORMATION")
    info_grp.attrs["HDFEOSVersion"] = "HDFEOS_5.1.16"
    odl_blocks = []
    
    for mgrs_tile, tile_meta in s30_collections.items():
        p_data = process_native_stack(tile_meta, 13, 0, 14, False)
        if p_data:
            write_hdf_sensor_group(h5f, f'/HDFEOS/GRIDS/HLSS30_{mgrs_tile}/Data Fields', p_data, S30_WAVELENGTHS, tile_meta['crs'], tile_meta['transform'], tile_name=mgrs_tile)
            odl_blocks.append(generate_odl_grid_string(f"HLSS30_{mgrs_tile}", tile_meta['width'], tile_meta['height'], tile_meta['transform'], "GCTP_UTM", tile_meta['zone'], [0.0]*13, 13, p_data['count'], False))

    for mgrs_tile, tile_meta in l30_collections.items():
        p_data = process_native_stack(tile_meta, 8, 2, 11, True)
        if p_data:
            write_hdf_sensor_group(h5f, f'/HDFEOS/GRIDS/HLSL30_{mgrs_tile}/Data Fields', p_data, L30_SR_WAVELENGTHS, tile_meta['crs'], tile_meta['transform'], L30_TIRS_WAVELENGTHS, tile_name=mgrs_tile)
            odl_blocks.append(generate_odl_grid_string(f"HLSL30_{mgrs_tile}", tile_meta['width'], tile_meta['height'], tile_meta['transform'], "GCTP_UTM", tile_meta['zone'], [0.0]*13, 8, p_data['count'], True))

    full_odl = "GROUP=SwathStructure\nEND_GROUP=SwathStructure\nGROUP=GridStructure\n" + "\n".join(odl_blocks) + "\nEND_GROUP=GridStructure\nGROUP=PointStructure\nEND_GROUP=PointStructure\nGROUP=ZaStructure\nEND_GROUP=ZaStructure\nEND\n"
    info_grp.create_dataset("StructMetadata.0", shape=(1,), dtype=h5py.string_dtype(encoding='ascii'), data=full_odl)

print(f"\nPipeline Complete. Native Truth Data HDF5 generated successfully: {OUTPUT_NATIVE_HDF5}")