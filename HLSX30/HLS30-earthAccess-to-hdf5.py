'''
Downloads HLS30 data for a specified ROI and time range from NASA Earth Access. 
Detects images originating from the exact same orbital pass and geometrically 
fuses adjacent MGRS tiles back into continuous sensor swaths.
Merges downloaded imagery into a consolidated, unified h5 file. 
'''
import os
import h5py
import rasterio
import numpy as np
import collections
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

import yaml

# ==========================================
# 1. CONFIGURATION & AUTHENTICATION
# ==========================================
cloud_threshold = 40

print("Authenticating with NASA Earthdata...")
earthaccess.login(strategy="all", persist=True)

# Load Configuration
script_dir = Path(__file__).resolve().parent
with open(os.path.join(script_dir, "locations_config.yaml"), "r") as f:
    config_data = yaml.safe_load(f)

Location = config_data.get("current_run", {}).get("location", "Palisades")
config = config_data["locations"][Location]

SOURCE_CACHE = config.get("SOURCE_CACHE")
ROI_LON_MIN = config["ROI_LON_MIN"]
ROI_LON_MAX = config["ROI_LON_MAX"]
ROI_LAT_MIN = config["ROI_LAT_MIN"]
ROI_LAT_MAX = config["ROI_LAT_MAX"]
START_DATE = config["START_DATE"]
END_DATE = config["END_DATE"]


if SOURCE_CACHE and SOURCE_CACHE in config_data["locations"]:
    cache_config = config_data["locations"][SOURCE_CACHE]
    cache_bbox = [
        min(cache_config["ROI_LON_MIN"], cache_config["ROI_LON_MAX"]), 
        max(cache_config["ROI_LAT_MIN"], cache_config["ROI_LAT_MAX"]),
        max(cache_config["ROI_LON_MIN"], cache_config["ROI_LON_MAX"]), 
        min(cache_config["ROI_LAT_MIN"], cache_config["ROI_LAT_MAX"])
    ]
else:
    cache_bbox = [
        min(ROI_LON_MIN, ROI_LON_MAX), max(ROI_LAT_MIN, ROI_LAT_MAX), 
        max(ROI_LON_MIN, ROI_LON_MAX), min(ROI_LAT_MIN, ROI_LAT_MAX)
    ]
cache_bbox = [min(cache_bbox[0], cache_bbox[2]), min(cache_bbox[1], cache_bbox[3]), max(cache_bbox[0], cache_bbox[2]), max(cache_bbox[1], cache_bbox[3])]



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

ASSETS_S30 = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B11', 'B12', 'Fmask', 'SZA', 'SAA', 'VZA', 'VAA']
ASSETS_L30 = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'Fmask', 'SZA', 'SAA', 'VZA', 'VAA']

S30_WAVELENGTHS = [0.443, 0.490, 0.560, 0.665, 0.705, 0.740, 0.783, 0.842, 1.610, 2.190]
L30_SR_WAVELENGTHS = [0.443, 0.482, 0.561, 0.655, 0.865, 1.609, 2.201] 

# Defensive Topology Enforcement for Output Grid (Location ROI)
safe_bbox = [
    min(ROI_LON_MIN, ROI_LON_MAX), max(ROI_LAT_MIN, ROI_LAT_MAX), 
    max(ROI_LON_MIN, ROI_LON_MAX), min(ROI_LAT_MIN, ROI_LAT_MAX)
]
safe_bbox = [min(safe_bbox[0], safe_bbox[2]), min(safe_bbox[1], safe_bbox[3]), max(safe_bbox[0], safe_bbox[2]), max(safe_bbox[1], safe_bbox[3])]

# ==========================================
# 2. DYNAMIC NATIVE GRID METROLOGY
# ==========================================
def establish_native_grid():
    """Derives a continuous localized UTM Grid bounding box to host the mosaicked passes."""
    central_lon = (ROI_LON_MIN + ROI_LON_MAX) / 2.0
    central_lat = (ROI_LAT_MIN + ROI_LAT_MAX) / 2.0
    utm_zone = int((central_lon + 180) / 6) + 1
    hemisphere_prefix = 32600 if central_lat >= 0 else 32700
    native_epsg = f"EPSG:{hemisphere_prefix + utm_zone}"
    
    transformer = Transformer.from_crs("EPSG:4326", native_epsg, always_xy=True)
    xs, ys = transformer.transform(
        [ROI_LON_MIN, ROI_LON_MAX, ROI_LON_MAX, ROI_LON_MIN], 
        [ROI_LAT_MAX, ROI_LAT_MAX, ROI_LAT_MIN, ROI_LAT_MIN]
    )
    
    # Snap to strict 30m intervals
    minx = np.floor(min(xs) / 30.0) * 30.0
    maxx = np.ceil(max(xs) / 30.0) * 30.0
    miny = np.floor(min(ys) / 30.0) * 30.0
    maxy = np.ceil(max(ys) / 30.0) * 30.0
    
    width = int((maxx - minx) / 30.0)
    height = int((maxy - miny) / 30.0)
    transform = transform_from_bounds(minx, miny, maxx, maxy, width, height)
    
    return native_epsg, transform, width, height, utm_zone

master_epsg, master_transform, master_width, master_height, master_zone = establish_native_grid()
print(f"Unified Native Grid Established: {master_width}x{master_height} px at {master_epsg}")

# ==========================================
# 3. NASA STAC QUERY & NATIVE VIRTUAL READ (HLS)
# ==========================================
def stac_native_window_read(collection_id, assets_list, temp_dir):
    print(f"\nQuerying NASA CMR STAC for {collection_id}...")
    catalog = pystac_client.Client.open("https://cmr.earthdata.nasa.gov/stac/LPCLOUD")
    search = catalog.search(collections=[collection_id], bbox=cache_bbox, datetime=f"{START_DATE}/{END_DATE}", limit=500)
    filtered_items = [i for i in list(search.items()) if i.properties.get('eo:cloud_cover', 100) < cloud_threshold]
    
    total_items = len(filtered_items)
    print(f"Identified {total_items} STAC items for {collection_id} within temporal bounds and cloud thresholds.")
    
    platform_mapping = {}
    item_ids = [i.id for i in filtered_items]
    if item_ids:
        print(f"Fetching platform metadata via earthaccess for {len(item_ids)} items...")
        short_name = collection_id.split('.')[0]
        for chunk_start in range(0, len(item_ids), 100):
            chunk = item_ids[chunk_start:chunk_start + 100]
            try:
                ea_results = earthaccess.search_data(short_name=short_name, granule_ur=chunk, count=len(chunk))
                for g in ea_results:
                    plats = g.get('umm', {}).get('Platforms', [])
                    if plats:
                        platform_mapping[g['umm']['GranuleUR']] = plats[0].get('ShortName', 'UNKNOWN')
            except Exception as e:
                print(f"Warning: Failed to fetch earthaccess metadata for chunk: {e}")

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
            # Broad casting to int32 to safely accommodate both -9999 and 40000 without overflow
            data = b_src.read(1, window=window_obj, boundless=True, fill_value=fill_val).astype(np.int32)
        return idx, data

    with rasterio.Env(**gdal_env):
        for i, item in enumerate(filtered_items, 1):
            img_id = item.id 
            parsed_mgrs_tile = img_id.split('.')[2] 
            
                
            cloud_cov = item.properties.get('eo:cloud_cover')
            
            if cloud_cov is None: 
                print(f"  [{i}/{total_items}] [{img_id}] WARNING: STAC metadata is null. Excluding frame.")
                continue
                
            if parsed_mgrs_tile not in tile_collections:
                tile_collections[parsed_mgrs_tile] = {'items': {}, 'transform': None, 'crs': None, 'width': None, 'height': None, 'zone': None}
                
            tile_data = tile_collections[parsed_mgrs_tile]
            out_tif = os.path.join(temp_dir, f"{img_id}.tif")
            
            tile_data['items'][img_id] = {
                'acquisition_time': item.datetime.timestamp(),
                'spacecraft_id': platform_mapping.get(img_id, item.properties.get('platform', 'UNKNOWN')),
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
                        # We use safe_bbox (the location subset) to generate the window to be read and stored in local cache
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
                # Ensure local cache utilizes int32 to hold the large 40000 fill value without overflow
                compiled_array = np.zeros((num_assets, tile_data['height'], tile_data['width']), dtype=np.int32)
                
                # Bypasses Python GIL IO blocking for massive concurrent fetch speeds
                with concurrent.futures.ThreadPoolExecutor(max_workers=len(assets_list)) as executor:
                    futures = []
                    for idx, asset_key in enumerate(assets_list):
                        url = item.assets[asset_key].href
                        futures.append(executor.submit(_fetch_single_band, idx, asset_key, url, window))
                    for future in concurrent.futures.as_completed(futures):
                        b_idx, b_data = future.result()
                        compiled_array[b_idx, :, :] = b_data
                
                profile = {'driver': 'GTiff', 'height': tile_data['height'], 'width': tile_data['width'], 'count': num_assets, 'dtype': 'int32', 'crs': tile_data['crs'], 'transform': tile_data['transform'], 'compress': 'deflate'}
                with rasterio.open(out_tif, 'w', **profile) as dst: dst.write(compiled_array)
                    
            except Exception as e:
                print(f"  [{i}/{total_items}] Failed retrieval for {img_id}: {e}")
                # Ensure corrupted or failed downloads are not retained in the manifest
                if img_id in tile_data['items']:
                    del tile_data['items'][img_id]
                
    if not tile_collections:
        raise ValueError(f"CRITICAL ERROR: No valid tiles found for {collection_id} after enforcing spatial bounds.")
        
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

def group_into_passes(tile_collections, time_threshold_sec=3000):
    all_items = []
    for tile, tile_data in tile_collections.items():
        for img_id, meta in tile_data['items'].items():
            all_items.append((img_id, meta, tile_data))
            
    all_items.sort(key=lambda x: x[1]['acquisition_time'])
    
    pass_clusters = []
    for item in all_items:
        img_id, meta, tile_data = item
        placed = False
        for cluster in pass_clusters:
            if cluster['spacecraft_id'] == meta['spacecraft_id']:
                if abs(cluster['mean_time'] - meta['acquisition_time']) < time_threshold_sec:
                    cluster['items'].append(item)
                    cluster['mean_time'] = sum(x[1]['acquisition_time'] for x in cluster['items']) / len(cluster['items'])
                    placed = True
                    break
        if not placed:
            pass_clusters.append({
                'spacecraft_id': meta['spacecraft_id'],
                'mean_time': meta['acquisition_time'],
                'items': [item]
            })
            
    pass_clusters.sort(key=lambda c: c['mean_time'])
    return pass_clusters

def stream_fused_stack_to_hdf5(h5f, group_path, pass_clusters, expected_sr, expected_fmask_idx, wavelengths):
    from rasterio.warp import reproject, Resampling
    from rasterio.crs import CRS
    
    num_frames = len(pass_clusters)
    if num_frames == 0: return 0

    w, h = master_width, master_height
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



    meta_arrays = {'acq': [], 'space': [], 'saz': [], 'sel': [], 'cc': []}

    # 2. Stream Data to Disk Pass-by-Pass
    for idx, cluster in enumerate(pass_clusters):
        sr_pass = np.full((expected_sr, h, w), -9999, dtype=np.int32)
        fm_pass = np.full((1, h, w), 255, dtype=np.uint8)
        ag_pass = np.full((4, h, w), 40000, dtype=np.uint16)

        for img_id, meta, tile_data in cluster['items']:
            # Strict Failure Directive: Halt if catalog metadata is missing.
            for k, v in meta.items():
                if v is None and k != 'filepath': 
                    raise ValueError(f"CRITICAL ERROR: Metadata '{k}' for '{img_id}' is null. Halting pipeline to prevent data assumptions.")

            with rasterio.open(meta['filepath']) as src:
                # SR bands
                t_sr = src.read(list(range(1, expected_sr+1)))
                temp_sr = np.full((expected_sr, h, w), -9999, dtype=np.int32)
                reproject(
                    source=t_sr, destination=temp_sr,
                    src_transform=src.transform, src_crs=src.crs,
                    dst_transform=master_transform, dst_crs=master_epsg,
                    resampling=Resampling.nearest, src_nodata=-9999, dst_nodata=-9999
                )
                sr_pass = np.where(temp_sr != -9999, temp_sr, sr_pass)


                # Fmask
                t_fm = src.read(expected_fmask_idx)
                temp_fm = np.full((h, w), 255, dtype=np.uint8)
                reproject(
                    source=t_fm, destination=temp_fm,
                    src_transform=src.transform, src_crs=src.crs,
                    dst_transform=master_transform, dst_crs=master_epsg,
                    resampling=Resampling.nearest, src_nodata=255, dst_nodata=255
                )
                
                # Derive valid footprint from SR to prevent swath-edge Fmask artifacts 
                # (e.g. false cloud/shadows at the boundary) from overwriting valid tiles
                sr_valid = temp_sr[0] != -9999
                fm_pass[0] = np.where((temp_fm != 255) & sr_valid, temp_fm, fm_pass[0])

                # Angles
                t_ag = src.read(list(range(expected_fmask_idx+1, expected_fmask_idx+5)))
                
                temp_ag = np.full((4, h, w), 40000, dtype=np.uint16)
                reproject(
                    source=t_ag, destination=temp_ag,
                    src_transform=src.transform, src_crs=src.crs,
                    dst_transform=master_transform, dst_crs=master_epsg,
                    resampling=Resampling.nearest, src_nodata=40000, dst_nodata=40000
                )
                ag_pass = np.where((temp_ag != 40000) & sr_valid, temp_ag, ag_pass)

        sr_ds[idx, ...] = np.where(sr_pass != -9999, sr_pass.astype(np.float32) * 0.0001, np.nan)
        fm_ds[idx, ...] = fm_pass
        
        ag_mapped = np.where(ag_pass != 40000, ag_pass * 0.01, np.nan)
        ag_ds[idx, ...] = ag_mapped

        # Derive global angles strictly from valid real data
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean_sza = np.nanmean(ag_mapped[0])
            mean_saa = np.nanmean(ag_mapped[1])

        if np.isnan(mean_sza) or np.isnan(mean_saa):
            print(f"WARNING: Raster-derived mean sun angles are NaN for pass {idx}. Using fallback 0.0.")
            mean_sza = 0.0
            mean_saa = 0.0

        avg_cc = sum(m['cloud_cover'] for _, m, _ in cluster['items']) / len(cluster['items'])

        meta_arrays['acq'].append(cluster['mean_time'])
        meta_arrays['space'].append(cluster['spacecraft_id'])
        meta_arrays['saz'].append(mean_saa)
        meta_arrays['sel'].append(90.0 - mean_sza)
        meta_arrays['cc'].append(avg_cc)

    # 3. Apply Mandatory ARD Attributes
    dt_str = h5py.string_dtype(encoding='ascii')
    gdal_transform = np.array([master_transform.c, master_transform.a, master_transform.b, 
                               master_transform.f, master_transform.d, master_transform.e], dtype='float64')
                               
    sr_ds.attrs['units'] = "Reflectance"
    sr_ds.attrs['_FillValue'] = np.nan
    sr_ds.attrs['wavelengths'] = wavelengths
    
    dst_crs_obj = CRS.from_string(master_epsg)
    sr_ds.attrs['spatial_ref'] = dst_crs_obj.to_wkt()
    sr_ds.attrs['GeoTransform'] = gdal_transform
    

        
    fm_ds.attrs['_FillValue'] = 255
    ag_ds.attrs['_FillValue'] = np.nan
    ag_ds.attrs['band_order'] = ["SZA", "SAA", "VZA", "VAA"]
    
    sr_ds.attrs.create('spacecraft_id', data=meta_arrays['space'], dtype=dt_str)
    sr_ds.attrs['acquisition_time'] = np.array(meta_arrays['acq'], dtype='float64') 
    sr_ds.attrs['sun_azimuth'] = np.array(meta_arrays['saz'], dtype='float32')
    sr_ds.attrs['sun_elevation'] = np.array(meta_arrays['sel'], dtype='float32')
    sr_ds.attrs['cloud_cover'] = np.array(meta_arrays['cc'], dtype='float32')
    
    return num_frames

with h5py.File(OUTPUT_NATIVE_HDF5, 'w') as h5f:
    info_grp = h5f.create_group("HDFEOS INFORMATION")
    info_grp.attrs["HDFEOSVersion"] = "HDFEOS_5.1.16"
    
    # Store Configuration Metadata
    meta_grp = h5f.create_group("METADATA/PIPELINE_CONFIG")
    meta_grp.attrs["Location"] = Location
    meta_grp.attrs["config_yaml"] = yaml.dump(config_data)

    odl_blocks = []
    
    s30_passes = group_into_passes(s30_collections)
    if s30_passes:
        num_f = stream_fused_stack_to_hdf5(h5f, '/HDFEOS/GRIDS/HLSS30_Merged/Data Fields', s30_passes, 10, 11, S30_WAVELENGTHS)
        if num_f > 0:
            odl_blocks.append(generate_odl_grid_string("HLSS30_Merged", master_width, master_height, master_transform, "GCTP_UTM", master_zone, [0.0]*13, 10, num_f, False))

    l30_passes = group_into_passes(l30_collections)
    if l30_passes:
        num_f = stream_fused_stack_to_hdf5(h5f, '/HDFEOS/GRIDS/HLSL30_Merged/Data Fields', l30_passes, 7, 8, L30_SR_WAVELENGTHS)
        if num_f > 0:
            odl_blocks.append(generate_odl_grid_string("HLSL30_Merged", master_width, master_height, master_transform, "GCTP_UTM", master_zone, [0.0]*13, 7, num_f, False))

    full_odl = "GROUP=SwathStructure\nEND_GROUP=SwathStructure\nGROUP=GridStructure\n" + "\n".join(odl_blocks) + "\nEND_GROUP=GridStructure\nGROUP=PointStructure\nEND_GROUP=PointStructure\nGROUP=ZaStructure\nEND_GROUP=ZaStructure\nEND\n"
    info_grp.create_dataset("StructMetadata.0", shape=(1,), dtype=h5py.string_dtype(encoding='ascii'), data=full_odl)

print(f"\nPipeline Complete. Native Truth Data HDF5 generated successfully: {OUTPUT_NATIVE_HDF5}")