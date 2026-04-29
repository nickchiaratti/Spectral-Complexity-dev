'''
Downloads HLS30 data for a specified ROI and time range from NASA Earth Access. 
Merges downloaded imagery into a consolidated h5 file in native CRS grid. 
'''
import os
import h5py
import rasterio
import numpy as np
from datetime import datetime, timezone
from rasterio.windows import from_bounds
from rasterio.transform import from_bounds as transform_from_bounds
from pyproj import Transformer, CRS
import pystac_client
import earthaccess
import json
import concurrent.futures
import warnings
from pathlib import Path
import re

# ==========================================
# 1. CONFIGURATION & AUTHENTICATION
# ==========================================
cloud_threshold = 40

print("Authenticating with NASA Earthdata...")
earthaccess.login(strategy="all", persist=True)

Location = "Tait"

# Define exactly which MGRS tiles cover the Rochester ROI. 
# Excludes marginal edge-collision tiles like T18TUN and T18TUP.


if Location == "Rochesterv2":
    SOURCE_CACHE = "Rochesterv2"
    ROI_LON_MIN = -77.770166; ROI_LON_MAX = -77.376776
    ROI_LAT_MIN = 42.961778; ROI_LAT_MAX = 43.342135
    START_DATE = '2022-01-01'
    END_DATE = '2026-03-31'
    ALLOWED_MGRS_TILES = ['T17TQH'] 
if Location == "Tait":
    SOURCE_CACHE = "Rochesterv2"
    ROI_LON_MIN = -77.516127; ROI_LON_MAX = -77.461968
    ROI_LAT_MIN = 43.127698; ROI_LAT_MAX = 43.159168
    START_DATE = '2014-01-01'
    END_DATE = '2026-03-31' 
    ALLOWED_MGRS_TILES = ['T17TQH'] 
if Location == 'Guatemala-Debris':
    SOURCE_CACHE = None
    ROI_LON_MIN = -88.222000; ROI_LON_MAX = -87.822000
    ROI_LAT_MIN = 15.636200; ROI_LAT_MAX = 16.036200
    START_DATE = '2020-08-01'
    END_DATE = '2020-10-31' 
    ALLOWED_MGRS_TILES = ['T16PCC'] 
if Location == "MtEtna":
    SOURCE_CACHE = "MtEtna-Catania"
    ROI_LON_MIN = 14.9100; ROI_LON_MAX = 15.0900
    ROI_LAT_MIN = 37.6900; ROI_LAT_MAX = 37.8300
    START_DATE = '2020-08-01'
    END_DATE = '2021-07-31' 
    ALLOWED_MGRS_TILES = ['T33SVB','T33SWB'] 
if Location == "MtEtna-Catania":
    SOURCE_CACHE = "MtEtna-Catania"
    ROI_LON_MIN = 14.800; ROI_LON_MAX = 15.35
    ROI_LAT_MIN = 37.400; ROI_LAT_MAX = 37.9
    START_DATE = '2024-01-01'
    END_DATE = '2025-01-01' 
    ALLOWED_MGRS_TILES = ['T33SVB','T33SWB'] 


HLSS30_OUTPUT_DIR = r"C:\satelliteImagery\HLS30\HLSS30-SourceData"
HLSL30_OUTPUT_DIR = r"C:\satelliteImagery\HLS30\HLSL30-SourceData"
COMBINED_OUTPUT_DIR = r"C:\satelliteImagery\HLS30"

if SOURCE_CACHE:
    S30_TEMP_DIR = os.path.join(HLSS30_OUTPUT_DIR, f"{SOURCE_CACHE}/STAC_CACHE")
    L30_TEMP_DIR = os.path.join(HLSL30_OUTPUT_DIR, f"{SOURCE_CACHE}/STAC_CACHE")
else:
    S30_TEMP_DIR = os.path.join(HLSS30_OUTPUT_DIR, f"{Location}/STAC_CACHE")
    L30_TEMP_DIR = os.path.join(HLSL30_OUTPUT_DIR, f"{Location}/STAC_CACHE")

os.makedirs(S30_TEMP_DIR, exist_ok=True)
os.makedirs(L30_TEMP_DIR, exist_ok=True)
os.makedirs(COMBINED_OUTPUT_DIR, exist_ok=True)

OUTPUT_NATIVE_HDF5 = os.path.join(COMBINED_OUTPUT_DIR, f"HLS_{Location}_STAC_Native_2025.h5")

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
    
    total_items = len(filtered_items)
    print(f"Identified {total_items} STAC items for {collection_id} within temporal bounds and cloud thresholds.")
    
    tile_collections = {}
    
    # Environment configs to maximize throughput for GDAL's virtual file system
    gdal_env = {
        'GDAL_HTTP_COOKIEFILE': os.path.expanduser('~/.urs_cookies'),
        'GDAL_HTTP_COOKIEJAR': os.path.expanduser('~/.urs_cookies'),
        'GDAL_DISABLE_READDIR_ON_OPEN': 'EMPTY_DIR',
        'CPL_VSIL_CURL_ALLOWED_EXTENSIONS': 'tif',
        'VSI_CACHE': True,
        'GDAL_HTTP_MULTIPLEX': 'YES'
    }
    
    def _fetch_single_band(idx, asset_key, url, window_obj):
        """Thread-safe worker block for fetching independent spectral bands."""
        if asset_key == 'Fmask': fill_val = 255
        elif asset_key in ['SZA', 'SAA', 'VZA', 'VAA']: fill_val = 40000
        else: fill_val = -9999
        
        with rasterio.open(url) as b_src:
            data = b_src.read(1, window=window_obj, boundless=True, fill_value=fill_val)
            if asset_key in ['SZA', 'SAA', 'VZA', 'VAA']:
                data = np.where(data == 40000, -9999, data).astype(np.int16)
        return idx, data

    with rasterio.Env(**gdal_env):
        for i, item in enumerate(filtered_items, 1):
            img_id = item.id 
            parsed_mgrs_tile = img_id.split('.')[2] 
            
            # Strict Filtering Directive: Drop unapproved marginal tiles
            if parsed_mgrs_tile not in ALLOWED_MGRS_TILES:
                print(f"  [{i}/{total_items}] [{img_id}] Dropping frame: Tile {parsed_mgrs_tile} is not in the approved whitelist.")
                continue
                
            cloud_cov = item.properties.get('eo:cloud_cover')
            
            if cloud_cov is None: 
                print(f"  [{i}/{total_items}] [{img_id}] WARNING: STAC metadata is null. Excluding frame.")
                continue
                
            if parsed_mgrs_tile not in tile_collections:
                tile_collections[parsed_mgrs_tile] = {'items': {}, 'transform': None, 'crs': None, 'width': None, 'height': None, 'zone': None}
                
            tile_data = tile_collections[parsed_mgrs_tile]
            out_tif = os.path.join(temp_dir, f"{img_id}.tif")
            
            # STAC items from LPCLOUD for HLS often lack the 'platform' property
            # We can reliably derive it from the product ID (e.g., HLS.S30... or HLS.L30...)
            sensor_code = img_id.split('.')[1] if len(img_id.split('.')) > 1 else ''
            if sensor_code == 'S30':
                platform_id = 'Sentinel-2'
            elif sensor_code == 'L30':
                platform_id = 'Landsat'
            else:
                platform_id = item.properties.get('platform', 'UNKNOWN')
                
            tile_data['items'][img_id] = {
                'acquisition_time': item.datetime.timestamp(),
                'spacecraft_id': platform_id,
                'cloud_cover': cloud_cov,
                'filepath': out_tif
            }
            
            if os.path.exists(out_tif) and os.path.getsize(out_tif) > 0:
                print(f"  [{i}/{total_items}] [{img_id}] Valid cache located. Skipping STAC download.")
                if tile_data['transform'] is None:
                    with rasterio.open(out_tif) as cached_src:
                        tile_data['transform'] = cached_src.transform
                        tile_data['crs'] = cached_src.crs
                        tile_data['width'] = cached_src.width
                        tile_data['height'] = cached_src.height
                        
                        epsg_code = CRS.from_user_input(cached_src.crs).to_epsg()
                        if epsg_code is not None:
                            tile_data['zone'] = epsg_code % 100
                        else:
                            zone_match = re.search(r'T(\d+)', parsed_mgrs_tile)
                            if zone_match:
                                tile_data['zone'] = int(zone_match.group(1))
                            else:
                                tile_data['zone'] = cached_src.crs.to_dict().get('zone')
                continue
            
            print(f"  [{i}/{total_items}] [{img_id}] Downloading {len(assets_list)} STAC assets via Concurrent Window Read...")    
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
                        
                        epsg_code = CRS.from_user_input(src.crs).to_epsg()
                        if epsg_code is not None:
                            tile_data['zone'] = epsg_code % 100
                        else:
                            zone_match = re.search(r'T(\d+)', parsed_mgrs_tile)
                            if zone_match:
                                tile_data['zone'] = int(zone_match.group(1))
                            else:
                                tile_data['zone'] = src.crs.to_dict().get('zone')
                                
                        tile_data['window'] = window
                    
                    window = tile_data['window']
                
                num_assets = len(assets_list)
                compiled_array = np.zeros((num_assets, tile_data['height'], tile_data['width']), dtype=np.int16)
                
                # Bypasses Python GIL IO blocking for massive concurrent fetch speeds
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
                print(f"  [{i}/{total_items}] Failed retrieval for {img_id}: {e}")
                # Ensure corrupted or failed downloads are not retained in the manifest
                if img_id in tile_data['items']:
                    del tile_data['items'][img_id]
                
    # Strict Failure Handling: Halt if whitelist + bounds yield no data
    if not tile_collections:
        raise ValueError(f"CRITICAL ERROR: No valid tiles found for {collection_id} after enforcing ALLOWED_MGRS_TILES whitelist.")
        
    return tile_collections

s30_collections = stac_native_window_read("HLSS30.v2.0", ASSETS_S30, S30_TEMP_DIR)
l30_collections = stac_native_window_read("HLSL30.v2.0", ASSETS_L30, L30_TEMP_DIR)

# ==========================================
# 3. HDFEOS5 ODL GENERATOR
# ==========================================
def generate_odl_grid_string(grid_name, width, height, transform, proj_code, zone, proj_params, num_sr_bands, num_frames, has_thermal):
    ul_x, ul_y = transform.c, transform.f
    lr_x = transform.c + (transform.a * width)
    lr_y = transform.f + (transform.e * height)
    p_str = str(tuple(proj_params)).replace(' ', '').replace('(', '').replace(')', '')
    
    thermal_dim = f"""            OBJECT=Dimension_3\n                DimensionName="ThermalBands"\n                Size=2\n            END_OBJECT=Dimension_3""" if has_thermal else ""
    thermal_field = f"""            OBJECT=DataField_2\n                DataFieldName="thermal_infrared"\n                DataType=HDF5T_NATIVE_FLOAT\n                DimList=("Time","ThermalBands","YDim","XDim")\n            END_OBJECT=DataField_2""" if has_thermal else ""
    fmask_idx = 3 if has_thermal else 2
    ang_idx = 4 if has_thermal else 3
    
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
        END_GROUP=DataField
        GROUP=MergedFields
        END_GROUP=MergedFields
    END_GROUP={grid_name}"""

# ==========================================
# 4. PHASE 1: BUILD NATIVE HDF5 (OUT-OF-CORE STREAMING)
# ==========================================
print(f"\nBuilding Native Truth Data HDF5 (HLS Only): {OUTPUT_NATIVE_HDF5}")

def stream_native_stack_to_hdf5(h5f, group_path, tile_data, expected_sr, expected_thermal, expected_fmask_idx, wavelengths, thermal_wavelengths=None, tile_name=None):
    """
    EVIDENCE-BASED OPTIMIZATION: Spatial Chunking & Fast Compression
    Bypasses I/O Thrashing by aligning the HDF5 chunks with the chronological 
    write-pattern (Frame-by-Frame). Uses gzip level 4 to balance size and speed.
    """
    items = tile_data['items']
    sorted_ids = sorted(items.keys(), key=lambda k: items[k]['acquisition_time'])
    num_frames = len(sorted_ids)
    if num_frames == 0: return 0

    w, h = tile_data['width'], tile_data['height']
    grp = h5f.create_group(group_path)

    # 1. Pre-allocate HDF5 Datasets directly on the physical disk
    chunk_h, chunk_w = min(h, 256), min(w, 256)
    
    sr_ds = grp.create_dataset('surface_reflectance', shape=(num_frames, expected_sr, h, w), 
                               chunks=(1, expected_sr, chunk_h, chunk_w),
                               dtype=np.float32, compression='gzip', compression_opts=4)
    
    fm_ds = grp.create_dataset('Fmask', shape=(num_frames, 1, h, w), 
                               chunks=(1, 1, chunk_h, chunk_w),
                               dtype=np.uint8, compression='gzip', compression_opts=4)
                               
    ag_ds = grp.create_dataset('solar_view_angles', shape=(num_frames, 4, h, w), 
                               chunks=(1, 4, chunk_h, chunk_w),
                               dtype=np.float32, compression='gzip', compression_opts=4)

    th_ds = None
    if expected_thermal > 0:
        th_ds = grp.create_dataset('thermal_infrared', shape=(num_frames, expected_thermal, h, w), 
                                   chunks=(1, expected_thermal, chunk_h, chunk_w),
                                   dtype=np.float32, compression='gzip', compression_opts=4)

    meta_arrays = {'acq': [], 'space': [], 'saz': [], 'sel': [], 'cc': []}

    # 2. Stream Data to Disk Frame-by-Frame
    for idx, img_id in enumerate(sorted_ids):
        meta = items[img_id]

        # Strict Failure Directive: Halt if catalog metadata is missing.
        for k, v in meta.items():
            if v is None and k != 'filepath': 
                raise ValueError(f"CRITICAL ERROR: Metadata '{k}' for '{img_id}' is null. Halting pipeline to prevent data assumptions.")

        with rasterio.open(meta['filepath']) as src:
            from rasterio.windows import Window
            target_transform = tile_data['transform']
            w, h = tile_data['width'], tile_data['height']
            target_left, target_top = target_transform * (0, 0)
            col_off_float, row_off_float = ~src.transform * (target_left, target_top)
            col_off, row_off = int(round(col_off_float)), int(round(row_off_float))
            read_window = Window(col_off, row_off, w, h)

            t_sr = src.read(list(range(1, expected_sr+1)), window=read_window, boundless=True, fill_value=-9999)
            sr_ds[idx, ...] = np.where(t_sr != -9999, t_sr.astype(np.float32) * 0.0001, np.nan)
            
            if expected_thermal > 0:
                off = expected_sr + 1
                t_th = src.read(list(range(off, off+expected_thermal)), window=read_window, boundless=True, fill_value=-9999)
                th_ds[idx, ...] = np.where(t_th != -9999, t_th.astype(np.float32) * 0.01, np.nan)
                
            fm_ds[idx, ...] = src.read(expected_fmask_idx, window=read_window, boundless=True, fill_value=255).astype(np.uint8)[np.newaxis, ...]
            
            t_ag = src.read(list(range(expected_fmask_idx+1, expected_fmask_idx+5)), window=read_window, boundless=True, fill_value=-9999)
            ag_mapped = np.where(t_ag != -9999, t_ag.astype(np.float32) * 0.01, np.nan)
            ag_ds[idx, ...] = ag_mapped

            # Derive global angles strictly from valid real data
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                mean_sza = np.nanmean(ag_mapped[0])
                mean_saa = np.nanmean(ag_mapped[1])

            # Strict Failure Directive: Halt if array yields NaN angles.
            if np.isnan(mean_sza) or np.isnan(mean_saa):
                raise ValueError(f"CRITICAL ERROR: Raster-derived mean sun angles are NaN for '{img_id}'. Halting pipeline.")

            meta_arrays['acq'].append(meta['acquisition_time'])
            meta_arrays['space'].append(meta['spacecraft_id'])
            meta_arrays['saz'].append(mean_saa)
            meta_arrays['sel'].append(90.0 - mean_sza)
            meta_arrays['cc'].append(meta['cloud_cover'])

    # 3. Apply Mandatory ARD Attributes
    dt_str = h5py.string_dtype(encoding='ascii')
    gdal_transform = np.array([tile_data['transform'].c, tile_data['transform'].a, tile_data['transform'].b, 
                               tile_data['transform'].f, tile_data['transform'].d, tile_data['transform'].e], dtype='float64')
                               
    sr_ds.attrs['units'] = "Reflectance"
    sr_ds.attrs['_FillValue'] = np.nan
    sr_ds.attrs['wavelengths'] = wavelengths
    sr_ds.attrs['spatial_ref'] = tile_data['crs'].to_wkt()
    sr_ds.attrs['GeoTransform'] = gdal_transform
    
    if th_ds:
        th_ds.attrs['units'] = "Kelvin/Celsius Apparent"
        th_ds.attrs['_FillValue'] = np.nan
        th_ds.attrs['wavelengths'] = thermal_wavelengths
        
    fm_ds.attrs['_FillValue'] = 255
    ag_ds.attrs['_FillValue'] = np.nan
    ag_ds.attrs['band_order'] = ["SZA", "SAA", "VZA", "VAA"]
    
    sr_ds.attrs.create('spacecraft_id', data=meta_arrays['space'], dtype=dt_str)
    if tile_name: sr_ds.attrs.create('mgrs_tile', data=tile_name, dtype=dt_str)
    sr_ds.attrs['acquisition_time'] = np.array(meta_arrays['acq'], dtype='float64') 
    sr_ds.attrs['sun_azimuth'] = np.array(meta_arrays['saz'], dtype='float32')
    sr_ds.attrs['sun_elevation'] = np.array(meta_arrays['sel'], dtype='float32')
    sr_ds.attrs['cloud_cover'] = np.array(meta_arrays['cc'], dtype='float32')
    
    return num_frames

with h5py.File(OUTPUT_NATIVE_HDF5, 'w') as h5f:
    info_grp = h5f.create_group("HDFEOS INFORMATION")
    info_grp.attrs["HDFEOSVersion"] = "HDFEOS_5.1.16"
    odl_blocks = []
    
    for mgrs_tile, tile_meta in s30_collections.items():
        num_f = stream_native_stack_to_hdf5(h5f, f'/HDFEOS/GRIDS/HLSS30_{mgrs_tile}/Data Fields', tile_meta, 13, 0, 14, S30_WAVELENGTHS, tile_name=mgrs_tile)
        if num_f > 0:
            odl_blocks.append(generate_odl_grid_string(f"HLSS30_{mgrs_tile}", tile_meta['width'], tile_meta['height'], tile_meta['transform'], "GCTP_UTM", tile_meta['zone'], [0.0]*13, 13, num_f, False))

    for mgrs_tile, tile_meta in l30_collections.items():
        num_f = stream_native_stack_to_hdf5(h5f, f'/HDFEOS/GRIDS/HLSL30_{mgrs_tile}/Data Fields', tile_meta, 8, 2, 11, L30_SR_WAVELENGTHS, thermal_wavelengths=L30_TIRS_WAVELENGTHS, tile_name=mgrs_tile)
        if num_f > 0:
            odl_blocks.append(generate_odl_grid_string(f"HLSL30_{mgrs_tile}", tile_meta['width'], tile_meta['height'], tile_meta['transform'], "GCTP_UTM", tile_meta['zone'], [0.0]*13, 8, num_f, True))

    full_odl = "GROUP=SwathStructure\nEND_GROUP=SwathStructure\nGROUP=GridStructure\n" + "\n".join(odl_blocks) + "\nEND_GROUP=GridStructure\nGROUP=PointStructure\nEND_GROUP=PointStructure\nGROUP=ZaStructure\nEND_GROUP=ZaStructure\nEND\n"
    info_grp.create_dataset("StructMetadata.0", shape=(1,), dtype=h5py.string_dtype(encoding='ascii'), data=full_odl)

print(f"\nPipeline Complete. Native Truth Data HDF5 generated successfully: {OUTPUT_NATIVE_HDF5}")