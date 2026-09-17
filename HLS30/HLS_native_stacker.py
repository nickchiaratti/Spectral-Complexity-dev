'''
HLS_native_stacker.py
Downloads HLS30 STAC assets and stacks them per MGRS tile, producing HLS_{Location}_{Sensor}_{Tile}.h5.
'''
import os
import h5py
import rasterio
import numpy as np
import collections
from datetime import datetime, timezone
from pyproj import Transformer, CRS
import pystac_client
import earthaccess
import json
import concurrent.futures
import warnings
import math
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import re
import gc
import yaml
import sys
import pystac

script_dir = Path(__file__).resolve().parent
if str(script_dir.parent) not in sys.path:
    sys.path.insert(0, str(script_dir.parent))
import SpecComplex as sc
from Harmonized_SC.sc_data_utils import scale_reflectance_for_storage

LOCATION_DEFAULT = "SantaBarbara"

# --- Configuration ---
COMBINED_OUTPUT_DIR = r"C:\satelliteImagery\HLS30"
HLSS30_OUTPUT_DIR = r"C:\satelliteImagery\HLS30\HLSS30-SourceData"
HLSL30_OUTPUT_DIR = r"C:\satelliteImagery\HLS30\HLSL30-SourceData"

TARGET_RESOLUTION = 30.0
CLOUD_THRESHOLD = 85

MIN_ROI_COVERAGE_PERCENT = 20.0
SUN_ELEVATION_THRESHOLD = 20
HLS_CLOUD_DILATION = 0
QA_REJECT_MASK = 0b11111
AEROSOL_ACCEPT_LEVEL = 'medium'

S30_WAVELENGTHS = np.array([0.443, 0.490, 0.560, 0.665, 0.705, 0.740, 0.783, 0.842, 1.610, 2.190], dtype=np.float32)
L30_WAVELENGTHS = np.array([0.443, 0.482, 0.561, 0.655, 0.865, 1.609, 2.201], dtype=np.float32)

ASSETS_S30 = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B8A', 'B11', 'B12', 'Fmask', 'SZA', 'SAA', 'VZA', 'VAA']
ASSETS_L30 = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'Fmask', 'SZA', 'SAA', 'VZA', 'VAA']

LP_DAAC_BASE = "https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected"

def validate_asset_url(item_id, asset_key, href, collection_id):
    collection_dir = collection_id.replace('.v', '.0')
    if item_id in href:
        return href
    corrected = f"{LP_DAAC_BASE}/{collection_dir}/{item_id}/{item_id}.{asset_key}.tif"
    # print(f"    WARNING: Asset URL mismatch for {item_id}.{asset_key}. Correcting href.")
    return corrected

def _fetch_single_band(idx, asset_key, url, window_obj):
    if asset_key == 'Fmask':
        fill_val = 255
    elif asset_key in ['SZA', 'SAA', 'VZA', 'VAA']:
        fill_val = 40000
    else:
        fill_val = -9999
    with rasterio.open(url) as b_src:
        if window_obj is not None:
            data = b_src.read(1, window=window_obj, boundless=True, fill_value=fill_val).astype(np.int32)
        else:
            data = b_src.read(1, fill_value=fill_val).astype(np.int32)
    return idx, data

def _process_single_tile(tile_id, t_items_dicts, collection_id, assets_list, temp_dir, expected_sr, expected_fmask_idx,
                         Location, TARGET_RESOLUTION, wavelengths, gdal_env):
    
    import multiprocessing
    
    try:
        worker_id = multiprocessing.current_process()._identity[0]
    except (IndexError, AttributeError):
        worker_id = 0

    OUTPUT_HDF5 = os.path.join(COMBINED_OUTPUT_DIR, f"HLS_{Location}_{collection_id.split('.')[0]}_{tile_id}.h5")
    if os.path.exists(OUTPUT_HDF5):
        return f"{tile_id}: Skipped (HDF5 already exists)"

    with rasterio.Env(**gdal_env):
        # --- DOWNLOAD STAGE ---
        item_manifest = []
        for i, item in enumerate(tqdm(t_items_dicts, desc=f"DL {tile_id}", position=worker_id, leave=False, unit="img")):
            img_id = item['id']
            cloud_cov = item['properties'].get('eo:cloud_cover')
            if cloud_cov is None: continue
            out_tif = os.path.join(temp_dir, f"{img_id}.tif")
            spacecraft = item['properties'].get('platform')

            entry = {
                'img_id': img_id, 'tile': tile_id,
                'filepath': out_tif, 'acquisition_time': item['datetime_timestamp'],
                'spacecraft': spacecraft, 'cloud_cover': cloud_cov
            }

            if os.path.exists(out_tif) and os.path.getsize(out_tif) > 0:
                item_manifest.append(entry)
                continue
            try:
                first_asset_key = assets_list[0]
                first_url = validate_asset_url(img_id, first_asset_key, item['assets'][first_asset_key]['href'], collection_id)
                with rasterio.open(first_url) as src:
                    tw_height, tw_width = src.height, src.width
                    tw_crs, tw_transform = src.crs, src.transform
                    
                compiled_array = np.zeros((len(assets_list), tw_height, tw_width), dtype=np.int32)
                with concurrent.futures.ThreadPoolExecutor(max_workers=len(assets_list)) as executor:
                    futures = []
                    for idx, asset_key in enumerate(assets_list):
                        url = validate_asset_url(img_id, asset_key, item['assets'][asset_key]['href'], collection_id)
                        futures.append(executor.submit(_fetch_single_band, idx, asset_key, url, None))
                    for future in concurrent.futures.as_completed(futures):
                        b_idx, b_data = future.result()
                        compiled_array[b_idx, :, :] = b_data

                profile = {
                    'driver': 'GTiff', 'height': tw_height, 'width': tw_width,
                    'count': len(assets_list), 'dtype': 'int32',
                    'crs': tw_crs, 'transform': tw_transform, 'compress': 'deflate'
                }
                with rasterio.open(out_tif, 'w', **profile) as dst:
                    dst.write(compiled_array)

                item_manifest.append(entry)
            except Exception as e:
                tqdm.write(f"[{tile_id}] Failed retrieval for {img_id}: {e}")

        if not item_manifest:
            return f"{tile_id}: No valid items"

        # --- GROUP BY DATE ---
        daily_groups = {}
        for entry in item_manifest:
            dt_str = datetime.fromtimestamp(entry['acquisition_time'], tz=timezone.utc).strftime('%Y-%m-%d')
            spacecraft = entry.get('spacecraft', collection_id)
            group_key = f"{dt_str}_{spacecraft}"
            if group_key not in daily_groups:
                daily_groups[group_key] = []
            daily_groups[group_key].append(entry)
        
        sorted_dates = sorted(daily_groups.keys())
        num_valid = len(sorted_dates)
        if num_valid == 0: 
            return f"{tile_id}: No valid items"

        # --- GET SHAPE ---
        with rasterio.open(item_manifest[0]['filepath']) as src:
            master_height, master_width = src.height, src.width
            master_crs = src.crs
            master_transform = src.transform

        # --- HDF5 CREATION ---
        with h5py.File(OUTPUT_HDF5, 'w') as h5f:
            group_path = f'/HDFEOS/GRIDS/{collection_id.split(".")[0]}_{tile_id}/Data Fields'
            grp = h5f.create_group(group_path)
            
            crs_wkt = master_crs.to_wkt()
            gdal_transform = np.array([master_transform.c, master_transform.a, master_transform.b, 
                                       master_transform.f, master_transform.d, master_transform.e], dtype='float64')
            dt = h5py.string_dtype(encoding='ascii')

            chunk_h, chunk_w = min(master_height, 256), min(master_width, 256)

            sr_ds = grp.create_dataset('surface_reflectance', shape=(num_valid, expected_sr, master_height, master_width),
                                       dtype='int16', compression='gzip', compression_opts=5, shuffle=True,
                                       chunks=(1, expected_sr, chunk_h, chunk_w), fillvalue=-9999)
            sr_ds.attrs['scale_to_float'] = 0.0001
            sr_ds.attrs['units'] = "Reflectance"
            sr_ds.attrs['wavelengths'] = wavelengths
            sr_ds.attrs['spatial_ref'] = crs_wkt
            sr_ds.attrs['GeoTransform'] = gdal_transform

            fmask_ds = grp.create_dataset('Fmask', shape=(num_valid, master_height, master_width),
                                          dtype='uint8', compression='gzip', compression_opts=5,
                                          chunks=(1, chunk_h, chunk_w), fillvalue=255)
            fmask_ds.attrs['_FillValue'] = 255

            ang_ds = grp.create_dataset('solar_view_angles', shape=(num_valid, 4, master_height, master_width),
                                        dtype='float32', compression='gzip', compression_opts=5,
                                        chunks=(1, 4, chunk_h, chunk_w), fillvalue=np.nan)
            ang_ds.attrs['_FillValue'] = np.nan
            ang_ds.attrs['band_order'] = ["SZA", "SAA", "VZA", "VAA"]

            vis_ds = grp.create_dataset('ortho_visual', shape=(num_valid, 4, master_height, master_width),
                                        dtype='uint8', compression='gzip', compression_opts=5,
                                        chunks=(1, 4, chunk_h, chunk_w))
            vis_ds.attrs['spatial_ref'] = crs_wkt
            vis_ds.attrs['GeoTransform'] = gdal_transform

            mask_ds = grp.create_dataset('common_mask', shape=(num_valid, master_height, master_width),
                                         dtype=bool, compression='gzip', compression_opts=5,
                                         chunks=(1, chunk_h, chunk_w))
            mask_ds.attrs['description'] = "True = Invalid/Masked, False = Valid."
            mask_ds.attrs['spatial_ref'] = crs_wkt
            mask_ds.attrs['GeoTransform'] = gdal_transform
            mask_ds.attrs['qa_reject_mask'] = QA_REJECT_MASK
            mask_ds.attrs['cloud_dilation'] = HLS_CLOUD_DILATION
            mask_ds.attrs['aerosol_accept_level'] = AEROSOL_ACCEPT_LEVEL
            mask_ds.attrs['sun_elevation_threshold'] = SUN_ELEVATION_THRESHOLD

            meta_arrays = {'acq': [], 'space': [], 'saz': [], 'sel': [], 'cc': []}
            
            water_counts = np.zeros((master_height, master_width), dtype=np.uint32)
            clear_counts = np.zeros((master_height, master_width), dtype=np.uint32)

            # PNG setup
            sensor_tile = f"{collection_id.split('.')[0]}_{tile_id}"
            location_dir = os.path.join(COMBINED_OUTPUT_DIR, f"{Location}_{sensor_tile}")
            os.makedirs(location_dir, exist_ok=True)

            # --- FRAME LOOP ---
            for out_idx, date_str in enumerate(tqdm(sorted_dates, desc=f"Ext {tile_id}", position=worker_id, leave=False, unit="frame")):
                entries = daily_groups[date_str]
                base_entry = entries[0]
                
                meta_arrays['acq'].append(base_entry['acquisition_time'])
                meta_arrays['space'].append(base_entry['spacecraft'])
                meta_arrays['cc'].append(base_entry['cloud_cover'])
                
                with rasterio.open(base_entry['filepath']) as src:
                    raw_sr = src.read(list(range(1, expected_sr + 1)))
                    raw_fm = src.read([expected_fmask_idx])[0]
                    raw_ag = src.read(list(range(expected_fmask_idx + 1, expected_fmask_idx + 5)))
                    
                sr_valid = raw_sr[0] != -9999
                
                float_sr = np.where(raw_sr != -9999, raw_sr.astype(np.float32) * 0.0001, np.nan)
                scaled_sr = scale_reflectance_for_storage(float_sr, scale_factor=10000.0, fill_value_in=np.nan, fill_value_out=-9999)
                
                fm_pass = np.where((raw_fm != 255) & sr_valid, raw_fm, 255).astype(np.uint8)
                ag_pass = np.where((raw_ag != 40000) & sr_valid, raw_ag * 0.01, np.nan).astype(np.float32)
                
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    mean_sza = np.nanmean(ag_pass[0])
                    mean_saa = np.nanmean(ag_pass[1])
                meta_arrays['saz'].append(mean_saa)
                meta_arrays['sel'].append(90.0 - mean_sza)

                temp_grp = {'Fmask': np.expand_dims(fm_pass, 0), 'solar_view_angles': np.expand_dims(ag_pass, 0)}
                frame_mask = sc.get_hls_mask(temp_grp, 0,
                                             sun_elevation_threshold=SUN_ELEVATION_THRESHOLD,
                                             cloud_dilation=HLS_CLOUD_DILATION,
                                             qa_reject_mask=QA_REJECT_MASK,
                                             aerosol_accept_level=AEROSOL_ACCEPT_LEVEL).astype(bool)

                rgba_img = sc.generate_rgba_image(
                    r_band=scaled_sr[3, :, :],
                    g_band=scaled_sr[2, :, :],
                    b_band=scaled_sr[1, :, :],
                    nodata=-9999)
                vis_frame = np.transpose(rgba_img, (2, 0, 1))

                # Incrementally write to HDF5
                sr_ds[out_idx, ...] = scaled_sr
                fmask_ds[out_idx, ...] = fm_pass
                ang_ds[out_idx, ...] = ag_pass
                vis_ds[out_idx, ...] = vis_frame
                mask_ds[out_idx, ...] = frame_mask

                # Update persistent water counts
                water_flags = (fm_pass & 0b100000) != 0
                clear_obs = ~frame_mask & (fm_pass != 255)
                water_counts += (water_flags & clear_obs)
                clear_counts += clear_obs


            # --- POST LOOP: Metadata & Water Mask ---
            persistent_water = np.zeros((master_height, master_width), dtype=np.uint8)
            valid_counts = clear_counts > 0
            ratio = np.zeros((master_height, master_width), dtype=np.float32)
            ratio[valid_counts] = water_counts[valid_counts] / clear_counts[valid_counts]
            persistent_water[ratio >= 1/3.0] = 1

            pwater_ds = grp.create_dataset('persistent_water_mask', data=persistent_water,
                                           dtype='uint8', compression='gzip', compression_opts=5,
                                           chunks=(chunk_h, chunk_w))
            pwater_ds.attrs['description'] = "1 = Water >= 1/3 clear obs. Derived from Fmask Bit 5."
            pwater_ds.attrs['spatial_ref'] = crs_wkt
            pwater_ds.attrs['GeoTransform'] = gdal_transform

            sr_ds.attrs.create('spacecraft_id', data=np.array(meta_arrays['space'], dtype=dt))
            sr_ds.attrs['acquisition_time'] = np.array(meta_arrays['acq'], dtype='float64')
            sr_ds.attrs['sun_azimuth'] = np.array(meta_arrays['saz'], dtype='float32')
            sr_ds.attrs['sun_elevation'] = np.array(meta_arrays['sel'], dtype='float32')
            sr_ds.attrs['cloud_cover'] = np.array(meta_arrays['cc'], dtype='float32')

    # Post-processing: generate PNGs from the completed HDF5
    try:
        with h5py.File(OUTPUT_HDF5, 'r') as h5f:
            grp = h5f[group_path]
            vis_ds = grp['ortho_visual']
            mask_ds = grp['common_mask']
            acq_times = grp['surface_reflectance'].attrs.get('acquisition_time', [])
            
            for i in tqdm(range(num_valid), desc=f"PNG {tile_id}", position=worker_id, leave=False, unit="png"):
                rgba_img = vis_ds[i]
                rgba_img = np.transpose(rgba_img, (1, 2, 0))
                frame_mask = mask_ds[i]
                
                acq_ts = acq_times[i] if i < len(acq_times) else 0
                pass_ts = datetime.fromtimestamp(acq_ts, tz=timezone.utc).strftime("%Y%m%dT%H%M%S")
                
                img = Image.fromarray(rgba_img, 'RGBA')
                png_filename = f"{sensor_tile}_{Location}_{pass_ts}_RGB.png"
                img.save(os.path.join(location_dir, png_filename))
                
                overlay = Image.new('RGBA', img.size, (255, 166, 0, 0))
                overlay_data = np.array(overlay)
                overlay_data[frame_mask == True] = [255, 166, 0, 128]
                overlay_img = Image.fromarray(overlay_data, 'RGBA')
                
                masked_img = Image.alpha_composite(img, overlay_img)
                masked_png_name = f"{sensor_tile}_{Location}_{pass_ts}_RGB_masked.png"
                masked_img.save(os.path.join(location_dir, masked_png_name))
    except Exception as e:
        print(f"\nError generating PNGs for {tile_id}: {e}")

    return f"{tile_id}: Success ({num_valid} frames)"

def main(target_location=None):
    print("Authenticating with NASA Earthdata...")
    earthaccess.login(strategy="all", persist=True)

    config_path = os.path.join(script_dir, "locations_config.yaml")
    if not os.path.exists(config_path):
        config_path = os.path.join(script_dir.parent, "locations_config.yaml")
    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)

    if target_location is not None:
        Location = target_location
    else:
        Location = config_data.get("current_run", {}).get("location", LOCATION_DEFAULT)
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
    cache_bbox = [min(cache_bbox[0], cache_bbox[2]), min(cache_bbox[1], cache_bbox[3]),
                  max(cache_bbox[0], cache_bbox[2]), max(cache_bbox[1], cache_bbox[3])]

    if SOURCE_CACHE:
        S30_TEMP_DIR = os.path.join(HLSS30_OUTPUT_DIR, f"{SOURCE_CACHE}/STAC_CACHE")
        L30_TEMP_DIR = os.path.join(HLSL30_OUTPUT_DIR, f"{SOURCE_CACHE}/STAC_CACHE")
    else:
        S30_TEMP_DIR = os.path.join(HLSS30_OUTPUT_DIR, f"{Location}/STAC_CACHE")
        L30_TEMP_DIR = os.path.join(HLSL30_OUTPUT_DIR, f"{Location}/STAC_CACHE")

    os.makedirs(S30_TEMP_DIR, exist_ok=True)
    os.makedirs(L30_TEMP_DIR, exist_ok=True)
    os.makedirs(COMBINED_OUTPUT_DIR, exist_ok=True)

    safe_bbox = [
        min(ROI_LON_MIN, ROI_LON_MAX), min(ROI_LAT_MIN, ROI_LAT_MAX),
        max(ROI_LON_MIN, ROI_LON_MAX), max(ROI_LAT_MIN, ROI_LAT_MAX)
    ]

    def process_sensor_per_tile(collection_id, assets_list, temp_dir, expected_sr, expected_fmask_idx, wavelengths):
        STAC_METADATA_CACHE_DIR = os.path.join(COMBINED_OUTPUT_DIR, "STAC_METADATA_CACHE")
        os.makedirs(STAC_METADATA_CACHE_DIR, exist_ok=True)

        cache_name = SOURCE_CACHE if SOURCE_CACHE else Location
        cache_filename = f"{cache_name}_{collection_id}_{START_DATE}_{END_DATE}_c{CLOUD_THRESHOLD}.json"
        cache_filepath = os.path.join(STAC_METADATA_CACHE_DIR, cache_filename)

        all_items = []

        if os.path.exists(cache_filepath):
            print(f"\nLoading NASA STAC metadata from local cache for {collection_id}...")
            with open(cache_filepath, 'r') as f:
                cached_data = json.load(f)
                all_items = [pystac.Item.from_dict(d) for d in cached_data]
        else:
            print(f"\nQuerying NASA CMR STAC for {collection_id} over the SOURCE_CACHE extent...")
            catalog = pystac_client.Client.open("https://cmr.earthdata.nasa.gov/stac/LPCLOUD")
            search = catalog.search(collections=[collection_id], bbox=cache_bbox,
                                    datetime=f"{START_DATE}/{END_DATE}", limit=500)

            all_items = [i for i in list(search.items())
                         if i.properties.get('eo:cloud_cover', 100) < CLOUD_THRESHOLD]
            total_items = len(all_items)
            print(f"Identified {total_items} STAC items for {collection_id}.")

            platform_mapping = {}
            item_ids = [i.id for i in all_items]
            if item_ids:
                print(f"Fetching platform metadata via earthaccess for {len(item_ids)} items...")
                short_name = collection_id.split('.')[0]
                for chunk_start in range(0, len(item_ids), 100):
                    chunk = item_ids[chunk_start:chunk_start + 100]
                    try:
                        ea_results = earthaccess.search_data(short_name=short_name,
                                                            granule_ur=chunk, count=len(chunk))
                        for g in ea_results:
                            plats = g.get('umm', {}).get('Platforms', [])
                            if plats:
                                platform_mapping[g['umm']['GranuleUR']] = plats[0].get('ShortName')
                    except Exception as e:
                        print(f"Warning: Failed to fetch earthaccess metadata for chunk: {e}")

            serialized_items = []
            for item in all_items:
                item.properties['platform'] = platform_mapping.get(
                    item.id, item.properties.get('platform'))
                serialized_items.append(item.to_dict())

            with open(cache_filepath, 'w') as f:
                json.dump(serialized_items, f)

        roi_minx, roi_miny, roi_maxx, roi_maxy = safe_bbox
        filtered_items = []
        for item in all_items:
            if not item.bbox: continue
            item_minx, item_miny, item_maxx, item_maxy = item.bbox
            if (item_minx <= roi_maxx and item_maxx >= roi_minx and
                item_miny <= roi_maxy and item_maxy >= roi_miny):
                filtered_items.append(item)

        total_filtered = len(filtered_items)
        print(f"After local filtering, {total_filtered} items intersect {Location}.")

        gdal_env = {
            'GDAL_HTTP_COOKIEFILE': os.path.expanduser('~/.urs_cookies'),
            'GDAL_HTTP_COOKIEJAR': os.path.expanduser('~/.urs_cookies'),
            'GDAL_DISABLE_READDIR_ON_OPEN': 'EMPTY_DIR',
            'CPL_VSIL_CURL_ALLOWED_EXTENSIONS': 'tif',
            'VSI_CACHE': True,
            'GDAL_HTTP_MULTIPLEX': 'YES'
        }

        items_by_tile = collections.defaultdict(list)
        for item in filtered_items:
            parsed_mgrs_tile = item.id.split('.')[2]
            items_by_tile[parsed_mgrs_tile].append(item)

        args_list = []
        for tile_id, t_items in items_by_tile.items():
            t_items_dicts = []
            for item in t_items:
                assets = {}
                for k in assets_list:
                    if k in item.assets:
                        assets[k] = {'href': item.assets[k].href}
                d = {
                    'id': item.id,
                    'properties': {
                        'eo:cloud_cover': item.properties.get('eo:cloud_cover'),
                        'platform': item.properties.get('platform')
                    },
                    'datetime_timestamp': item.datetime.timestamp(),
                    'assets': assets
                }
                t_items_dicts.append(d)
            
            args = (tile_id, t_items_dicts, collection_id, assets_list, temp_dir, expected_sr, expected_fmask_idx,
                    Location, TARGET_RESOLUTION, wavelengths, gdal_env)
            args_list.append(args)

        print(f"Dispatching {len(args_list)} tiles to ProcessPoolExecutor...")
        with concurrent.futures.ProcessPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(_process_single_tile, *arg) for arg in args_list]
            for future in concurrent.futures.as_completed(futures):
                try:
                    res = future.result()
                    print(res)
                except Exception as e:
                    print(f"Tile processing failed: {e}")

    print(f"\nProcessing HLSS30...")
    process_sensor_per_tile("HLSS30.v2.0", ASSETS_S30, S30_TEMP_DIR, 10, 11, S30_WAVELENGTHS)

    print(f"\nProcessing HLSL30...")
    process_sensor_per_tile("HLSL30.v2.0", ASSETS_L30, L30_TEMP_DIR, 7, 8, L30_WAVELENGTHS)

    print(f"\nPipeline Complete.")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Unified HLS STAC-to-Native Pipeline")
    parser.add_argument("--location", type=str, default=None,
                        help="Target location from locations_config.yaml")
    args = parser.parse_args()
    main(target_location=args.location)
