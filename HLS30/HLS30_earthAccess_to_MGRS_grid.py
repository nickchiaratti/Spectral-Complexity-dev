'''
HLS30_earthAccess_to_MGRS_grid.py
Unified pipeline: downloads HLS30 STAC assets and directly reprojects + mosaics
them onto a target MGRS-aligned grid, producing HLS_{Location}_MGRS_Stack.h5
without writing the intermediate HLS_{Location}_STAC_Native.h5.

Merges the functionality of:
  - HLS30_earthAccess_to_hdf5.py  (STAC download, TIFF caching, metadata)
  - HLS_MGRS_stacker.py           (MGRS reprojection, QA masking, HDF5 output)
  - harmonize_hls.py              (multi-tile mosaicking and coverage filtering)
'''
import os
import h5py
import rasterio
import numpy as np
import collections
from datetime import datetime, timezone
from rasterio.windows import from_bounds
from rasterio.warp import reproject, Resampling
from rasterio.transform import Affine
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

script_dir = Path(__file__).resolve().parent
if str(script_dir.parent) not in sys.path:
    sys.path.insert(0, str(script_dir.parent))
import SpecComplex as sc

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


# ==========================================
# UTILITY FUNCTIONS
# ==========================================




def validate_asset_url(item_id, asset_key, href, collection_id):
    """
    Guard against upstream STAC catalog pointer misalignment.
    If the asset href does not contain the item's own granule ID,
    construct the deterministic URL from the item ID.
    """
    # Extract the collection short code (e.g. "HLSS30.020" from "HLSS30.v2.0")
    collection_dir = collection_id.replace('.v', '.0')
    if item_id in href:
        return href
    # Construct the correct deterministic URL
    corrected = f"{LP_DAAC_BASE}/{collection_dir}/{item_id}/{item_id}.{asset_key}.tif"
    print(f"    WARNING: Asset URL mismatch for {item_id}.{asset_key}. Correcting href.")
    return corrected


def _fetch_single_band(idx, asset_key, url, window_obj):
    """Thread-safe worker for fetching independent spectral bands."""
    if asset_key == 'Fmask':
        fill_val = 255
    elif asset_key in ['SZA', 'SAA', 'VZA', 'VAA']:
        fill_val = 40000
    else:
        fill_val = -9999

    with rasterio.open(url) as b_src:
        data = b_src.read(1, window=window_obj, boundless=True, fill_value=fill_val).astype(np.int32)
    return idx, data


# ==========================================
# HDF5 OUTPUT
# ==========================================

def write_hdf_sensor_group(h5f, group_path, data_dict, wavelengths, crs_wkt, transform, target_location):
    """Writes a complete sensor group to the output MGRS Stack HDF5."""
    if not data_dict or data_dict['count'] == 0:
        return
    grp = h5f.create_group(group_path)
    gdal_transform = np.array([transform.c, transform.a, transform.b, transform.f, transform.d, transform.e], dtype='float64')
    dt = h5py.string_dtype(encoding='ascii')

    num_frames, bands, h, w = data_dict['sr'].shape
    chunk_h, chunk_w = min(h, 256), min(w, 256)

    sr_ds = grp.create_dataset('surface_reflectance', data=data_dict['sr'],
                               compression='gzip', compression_opts=5, shuffle=True,
                               chunks=(1, bands, chunk_h, chunk_w),
                               fillvalue=-9999)
    sr_ds.attrs['scale_to_float'] = 0.0001
    sr_ds.attrs['units'] = "Reflectance"
    sr_ds.attrs['wavelengths'] = wavelengths
    sr_ds.attrs['spatial_ref'] = crs_wkt
    sr_ds.attrs['GeoTransform'] = gdal_transform

    fmask_ds = grp.create_dataset('Fmask', data=data_dict['fm'][:, 0, :, :],
                                  dtype='uint8', compression='gzip', compression_opts=5,
                                  chunks=(1, chunk_h, chunk_w))
    fmask_ds.attrs['_FillValue'] = 255

    ang_ds = grp.create_dataset('solar_view_angles', data=data_dict['ag'],
                                compression='gzip', compression_opts=5,
                                chunks=(1, 4, chunk_h, chunk_w))
    ang_ds.attrs['_FillValue'] = np.nan
    ang_ds.attrs['band_order'] = ["SZA", "SAA", "VZA", "VAA"]

    vis_ds = grp.create_dataset('ortho_visual', data=data_dict['vis'],
                                dtype='uint8', compression='gzip', compression_opts=5,
                                chunks=(1, 4, chunk_h, chunk_w))
    vis_ds.attrs['spatial_ref'] = crs_wkt
    vis_ds.attrs['GeoTransform'] = gdal_transform

    mask_ds = grp.create_dataset('common_mask', data=data_dict['mask'],
                                 dtype=bool, compression='gzip', compression_opts=5,
                                 chunks=(1, chunk_h, chunk_w))
    mask_ds.attrs['description'] = "True = Invalid/Masked, False = Valid."
    mask_ds.attrs['spatial_ref'] = crs_wkt
    mask_ds.attrs['GeoTransform'] = gdal_transform
    mask_ds.attrs['qa_reject_mask'] = QA_REJECT_MASK
    mask_ds.attrs['cloud_dilation'] = HLS_CLOUD_DILATION
    mask_ds.attrs['aerosol_accept_level'] = AEROSOL_ACCEPT_LEVEL
    mask_ds.attrs['sun_elevation_threshold'] = SUN_ELEVATION_THRESHOLD

    # Water mask from Fmask Bit 5
    fmask_data = data_dict['fm'][:, 0, :, :]
    water_mask_data = (fmask_data & 0b100000) != 0
    water_ds = grp.create_dataset('water_mask', data=water_mask_data,
                                  dtype=bool, compression='gzip', compression_opts=5,
                                  chunks=(1, chunk_h, chunk_w))
    water_ds.attrs['description'] = "True = Water, False = Non-Water. Derived from HLS Fmask Bit 5."
    water_ds.attrs['spatial_ref'] = crs_wkt
    water_ds.attrs['GeoTransform'] = gdal_transform

    sr_ds.attrs.create('spacecraft_id', data=np.array(data_dict['meta']['space'], dtype=dt))
    sr_ds.attrs['acquisition_time'] = np.array(data_dict['meta']['acq'], dtype='float64')
    sr_ds.attrs['sun_azimuth'] = np.array(data_dict['meta']['saz'], dtype='float32')
    sr_ds.attrs['sun_elevation'] = np.array(data_dict['meta']['sel'], dtype='float32')
    sr_ds.attrs['cloud_cover'] = np.array(data_dict['meta']['cc'], dtype='float32')

    # Export PNGs
    print(f"  Exporting visual frames as PNGs for {group_path}...")
    sensor_name = group_path.split('/')[3]
    location_dir = os.path.join(COMBINED_OUTPUT_DIR, f"{target_location}_{sensor_name}")
    os.makedirs(location_dir, exist_ok=True)

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


# ==========================================
# MAIN PIPELINE
# ==========================================

def main(target_location=None):
    # ==========================================
    # 1. CONFIGURATION & AUTHENTICATION
    # ==========================================
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

    # Cache bbox (may be broader than ROI if SOURCE_CACHE is set)
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

    # TIFF cache directories (shared with existing pipeline)
    if SOURCE_CACHE:
        S30_TEMP_DIR = os.path.join(HLSS30_OUTPUT_DIR, f"{SOURCE_CACHE}/STAC_CACHE")
        L30_TEMP_DIR = os.path.join(HLSL30_OUTPUT_DIR, f"{SOURCE_CACHE}/STAC_CACHE")
    else:
        S30_TEMP_DIR = os.path.join(HLSS30_OUTPUT_DIR, f"{Location}/STAC_CACHE")
        L30_TEMP_DIR = os.path.join(HLSL30_OUTPUT_DIR, f"{Location}/STAC_CACHE")

    os.makedirs(S30_TEMP_DIR, exist_ok=True)
    os.makedirs(L30_TEMP_DIR, exist_ok=True)
    os.makedirs(COMBINED_OUTPUT_DIR, exist_ok=True)

    OUTPUT_MGRS_HDF5 = os.path.join(COMBINED_OUTPUT_DIR, f"HLS_{Location}_MGRS_Stack.h5")

    # Defensive topology enforcement for ROI
    safe_bbox = [
        min(ROI_LON_MIN, ROI_LON_MAX), min(ROI_LAT_MIN, ROI_LAT_MAX),
        max(ROI_LON_MIN, ROI_LON_MAX), max(ROI_LAT_MIN, ROI_LAT_MAX)
    ]

    # Fetch MGRS-aligned master grid from config
    target_crs_str = f"EPSG:{config['MGRS_EPSG']}"
    master_width = config['MGRS_WIDTH']
    master_height = config['MGRS_HEIGHT']
    ul_x = config['MGRS_UL_X']
    ul_y = config['MGRS_UL_Y']
    target_gsd = config.get('TARGET_GSD', 30.0)
    
    master_transform = Affine.translation(ul_x, ul_y) * Affine.scale(target_gsd, -target_gsd)
    master_crs = CRS.from_string(target_crs_str)
    print(f"MGRS Grid Established: {master_width}x{master_height} at {target_crs_str}")

    # ==========================================
    # 2. STAC QUERY, DOWNLOAD & MGRS PROJECTION
    # ==========================================
    def process_sensor_direct(collection_id, assets_list, temp_dir, expected_sr, expected_fmask_idx):
        """
        Unified pipeline: queries STAC, downloads GeoTIFFs, reprojects to master grid,
        and returns the same data_dict structure as process_hls_master_stack().
        """
        import pystac

        # --- 2a. STAC Metadata Caching ---
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
            print(f"Loaded {len(all_items)} items from cache.")
        else:
            print(f"\nQuerying NASA CMR STAC for {collection_id} over the SOURCE_CACHE extent...")
            catalog = pystac_client.Client.open("https://cmr.earthdata.nasa.gov/stac/LPCLOUD")
            search = catalog.search(collections=[collection_id], bbox=cache_bbox,
                                    datetime=f"{START_DATE}/{END_DATE}", limit=500)

            all_items = [i for i in list(search.items())
                         if i.properties.get('eo:cloud_cover', 100) < CLOUD_THRESHOLD]
            total_items = len(all_items)
            print(f"Identified {total_items} STAC items for {collection_id}.")

            # Fetch EarthAccess platform metadata
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
                                platform_mapping[g['umm']['GranuleUR']] = plats[0].get('ShortName', 'UNKNOWN')
                    except Exception as e:
                        print(f"Warning: Failed to fetch earthaccess metadata for chunk: {e}")

            serialized_items = []
            for item in all_items:
                item.properties['platform'] = platform_mapping.get(
                    item.id, item.properties.get('platform', 'UNKNOWN'))
                serialized_items.append(item.to_dict())

            with open(cache_filepath, 'w') as f:
                json.dump(serialized_items, f)
            print(f"Saved STAC metadata to local cache: {cache_filename}")

        # --- 2b. Spatial Filtering to safe_bbox ---
        roi_minx, roi_miny, roi_maxx, roi_maxy = safe_bbox
        filtered_items = []
        for item in all_items:
            if not item.bbox:
                continue
            item_minx, item_miny, item_maxx, item_maxy = item.bbox
            intersects = (item_minx <= roi_maxx and item_maxx >= roi_minx and
                          item_miny <= roi_maxy and item_maxy >= roi_miny)
            if intersects:
                filtered_items.append(item)

        total_filtered = len(filtered_items)
        print(f"After local filtering, {total_filtered} items intersect {Location}.")

        # --- 2c. Download GeoTIFFs to Cache ---
        # Track per-tile window/transform for reading
        tile_windows = {}  # parsed_mgrs_tile -> {'window', 'transform', 'crs'}

        gdal_env = {
            'GDAL_HTTP_COOKIEFILE': os.path.expanduser('~/.urs_cookies'),
            'GDAL_HTTP_COOKIEJAR': os.path.expanduser('~/.urs_cookies'),
            'GDAL_DISABLE_READDIR_ON_OPEN': 'EMPTY_DIR',
            'CPL_VSIL_CURL_ALLOWED_EXTENSIONS': 'tif',
            'VSI_CACHE': True,
            'GDAL_HTTP_MULTIPLEX': 'YES'
        }

        # Build per-item metadata and ensure GeoTIFFs are cached
        item_manifest = []  # List of {item, img_id, tile, filepath, acq_time, spacecraft, cloud_cover}

        with rasterio.Env(**gdal_env):
            for i, item in enumerate(filtered_items, 1):
                img_id = item.id
                parsed_mgrs_tile = img_id.split('.')[2]

                cloud_cov = item.properties.get('eo:cloud_cover')
                if cloud_cov is None:
                    print(f"  [{i}/{total_filtered}] [{img_id}] WARNING: cloud_cover is null. Excluding.")
                    continue

                out_tif = os.path.join(temp_dir, f"{img_id}.tif")
                spacecraft = item.properties.get('platform', 'UNKNOWN')

                entry = {
                    'item': item,
                    'img_id': img_id,
                    'tile': parsed_mgrs_tile,
                    'filepath': out_tif,
                    'acquisition_time': item.datetime.timestamp(),
                    'spacecraft': spacecraft,
                    'cloud_cover': cloud_cov
                }

                # Check if valid cache exists
                if os.path.exists(out_tif) and os.path.getsize(out_tif) > 0:
                    cache_valid = False
                    with rasterio.open(out_tif) as cached_src:
                        transformer = Transformer.from_crs("EPSG:4326", cached_src.crs, always_xy=True)
                        xs, ys = transformer.transform(
                            [cache_bbox[0], cache_bbox[2], cache_bbox[2], cache_bbox[0]],
                            [cache_bbox[3], cache_bbox[3], cache_bbox[1], cache_bbox[1]]
                        )
                        c_minx, c_maxx, c_miny, c_maxy = min(xs), max(xs), min(ys), max(ys)
                        cache_valid = (
                            cached_src.bounds.left <= c_minx + 40 and
                            cached_src.bounds.right >= c_maxx - 40 and
                            cached_src.bounds.bottom <= c_miny + 40 and
                            cached_src.bounds.top >= c_maxy - 40
                        )

                    if cache_valid:
                        print(f"  [{i}/{total_filtered}] [{img_id}] Valid cache located. Skipping download.")
                        item_manifest.append(entry)
                        continue
                    else:
                        print(f"  [{i}/{total_filtered}] [{img_id}] Cache bounds insufficient. Re-downloading.")

                # Download from STAC
                print(f"  [{i}/{total_filtered}] [{img_id}] Downloading {len(assets_list)} assets...")
                try:
                    # Use the first asset to establish window geometry
                    first_asset_key = assets_list[0]
                    first_url = validate_asset_url(img_id, first_asset_key, item.assets[first_asset_key].href, collection_id)

                    with rasterio.open(first_url) as src:
                        if parsed_mgrs_tile not in tile_windows:
                            transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
                            xs, ys = transformer.transform(
                                [cache_bbox[0], cache_bbox[2], cache_bbox[2], cache_bbox[0]],
                                [cache_bbox[3], cache_bbox[3], cache_bbox[1], cache_bbox[1]]
                            )
                            r_minx, r_maxx, r_miny, r_maxy = min(xs), max(xs), min(ys), max(ys)
                            window = from_bounds(r_minx, r_miny, r_maxx, r_maxy,
                                                 transform=src.transform).round_offsets().round_lengths()
                            tile_windows[parsed_mgrs_tile] = {
                                'window': window,
                                'transform': src.window_transform(window),
                                'crs': src.crs,
                                'width': window.width,
                                'height': window.height
                            }
                        window = tile_windows[parsed_mgrs_tile]['window']

                    num_assets = len(assets_list)
                    tw = tile_windows[parsed_mgrs_tile]
                    compiled_array = np.zeros((num_assets, tw['height'], tw['width']), dtype=np.int32)

                    with concurrent.futures.ThreadPoolExecutor(max_workers=num_assets) as executor:
                        futures = []
                        for idx, asset_key in enumerate(assets_list):
                            url = validate_asset_url(img_id, asset_key,
                                                     item.assets[asset_key].href, collection_id)
                            futures.append(executor.submit(_fetch_single_band, idx, asset_key, url, window))
                        for future in concurrent.futures.as_completed(futures):
                            b_idx, b_data = future.result()
                            compiled_array[b_idx, :, :] = b_data

                    profile = {
                        'driver': 'GTiff',
                        'height': tw['height'], 'width': tw['width'],
                        'count': num_assets, 'dtype': 'int32',
                        'crs': tw['crs'], 'transform': tw['transform'],
                        'compress': 'deflate'
                    }
                    with rasterio.open(out_tif, 'w', **profile) as dst:
                        dst.write(compiled_array)

                    item_manifest.append(entry)

                except Exception as e:
                    print(f"  [{i}/{total_filtered}] Failed retrieval for {img_id}: {e}")

        if not item_manifest:
            print(f"No valid items downloaded for {collection_id}.")
            return None

        # --- 2d. Group by Calendar Date and Spacecraft ---
        daily_groups = {}
        for entry in item_manifest:
            dt_str = datetime.fromtimestamp(entry['acquisition_time'], tz=timezone.utc).strftime('%Y-%m-%d')
            spacecraft = entry.get('spacecraft_id', collection_id)
            group_key = f"{dt_str}_{spacecraft}"
            if group_key not in daily_groups:
                daily_groups[group_key] = []
            daily_groups[group_key].append(entry)

        sorted_dates = sorted(daily_groups.keys())
        print(f"\n{collection_id}: {len(sorted_dates)} unique calendar dates from {len(item_manifest)} tiles.")

        # --- 2e. PASS 1: Coverage Check ---
        print("  [Pass 1] Evaluating ROI coverage...", flush=True)
        valid_dates = []
        for date_str in sorted_dates:
            entries = daily_groups[date_str]
            accum_band0 = np.full((master_height, master_width), np.nan, dtype=np.float32)

            for entry in entries:
                with rasterio.open(entry['filepath']) as src:
                    src_tf = src.transform
                    src_crs = src.crs
                    # Read only band 1 (first SR band)
                    raw_b0 = src.read(1, boundless=True)

                # Convert to reflectance for valid pixel detection
                b0_refl = np.where(raw_b0 != -9999, raw_b0.astype(np.float32) * 0.0001, np.nan)

                tmp = np.full((master_height, master_width), np.nan, dtype=np.float32)
                reproject(source=b0_refl, destination=tmp,
                          src_transform=src_tf, src_crs=src_crs,
                          dst_transform=master_transform, dst_crs=master_crs,
                          resampling=Resampling.nearest,
                          src_nodata=np.nan, dst_nodata=np.nan)

                mask = ~np.isnan(tmp)
                accum_band0[mask] = tmp[mask]

            valid_pixels = np.sum(~np.isnan(accum_band0))
            coverage = (valid_pixels / (master_height * master_width)) * 100
            if coverage >= MIN_ROI_COVERAGE_PERCENT:
                valid_dates.append(date_str)
            else:
                print(f"  Skipping {date_str} (Coverage: {coverage:.1f}% < {MIN_ROI_COVERAGE_PERCENT}%)", flush=True)

        num_valid = len(valid_dates)
        if num_valid == 0:
            print(f"No valid dates for {collection_id} after coverage filtering.")
            return None

        # --- 2f. PASS 2: Full Extraction & Reprojection ---
        print(f"  [Pass 2] Processing {num_valid} valid dates...", flush=True)

        stk_sr = np.full((num_valid, expected_sr, master_height, master_width), -9999, dtype=np.int16)
        stk_fm = np.full((num_valid, 1, master_height, master_width), 255, dtype=np.uint8)
        stk_ag = np.full((num_valid, 4, master_height, master_width), np.nan, dtype=np.float32)
        stk_mask = np.ones((num_valid, master_height, master_width), dtype=bool)
        vis_data = np.zeros((num_valid, 4, master_height, master_width), dtype=np.uint8)
        meta_arrays = {'acq': [], 'space': [], 'saz': [], 'sel': [], 'cc': []}

        for out_idx, date_str in enumerate(tqdm(valid_dates, desc="  Reprojecting Frames", unit="frame")):
            entries = daily_groups[date_str]

            # Use the first entry's metadata as the frame's representative
            base_entry = entries[0]
            meta_arrays['acq'].append(base_entry['acquisition_time'])
            meta_arrays['space'].append(base_entry['spacecraft'])
            meta_arrays['cc'].append(base_entry['cloud_cover'])

            for entry in entries:
                with rasterio.open(entry['filepath']) as src:
                    src_tf = src.transform
                    src_crs = src.crs

                    # Read SR bands (bands 1..expected_sr)
                    raw_sr = src.read(list(range(1, expected_sr + 1)))
                    # Read Fmask
                    raw_fm = src.read(expected_fmask_idx)
                    # Read angle bands (SZA, SAA, VZA, VAA)
                    raw_ag = src.read(list(range(expected_fmask_idx + 1, expected_fmask_idx + 5)))

                # Convert SR to reflectance
                sr_valid = raw_sr[0] != -9999
                # Keep raw_sr as int16
                sr_refl = raw_sr

                # Convert Fmask: keep valid, fill=255
                fm_pass = np.where((raw_fm != 255) & sr_valid, raw_fm, 255).astype(np.uint8)
                fm_pass = fm_pass[np.newaxis, :, :]  # Add band dimension for reproject

                # Convert angles: scale by 0.01, fill 40000 → NaN
                ag_pass = np.where((raw_ag != 40000) & sr_valid, raw_ag * 0.01, np.nan).astype(np.float32)

                # Reproject SR
                tmp_sr = np.full((expected_sr, master_height, master_width), -9999, dtype=np.int16)
                reproject(source=sr_refl, destination=tmp_sr,
                          src_transform=src_tf, src_crs=src_crs,
                          dst_transform=master_transform, dst_crs=master_crs,
                          resampling=Resampling.nearest,
                          src_nodata=-9999, dst_nodata=-9999)
                mask_sr = (tmp_sr != -9999)
                stk_sr[out_idx][mask_sr] = tmp_sr[mask_sr]

                # Reproject Fmask
                tmp_fm = np.full((1, master_height, master_width), 255, dtype=np.uint8)
                reproject(source=fm_pass, destination=tmp_fm,
                          src_transform=src_tf, src_crs=src_crs,
                          dst_transform=master_transform, dst_crs=master_crs,
                          resampling=Resampling.nearest,
                          src_nodata=255, dst_nodata=255)
                mask_fm = (tmp_fm != 255)
                stk_fm[out_idx][mask_fm] = tmp_fm[mask_fm]

                # Reproject angles
                tmp_ag = np.full((4, master_height, master_width), np.nan, dtype=np.float32)
                reproject(source=ag_pass, destination=tmp_ag,
                          src_transform=src_tf, src_crs=src_crs,
                          dst_transform=master_transform, dst_crs=master_crs,
                          resampling=Resampling.nearest,
                          src_nodata=np.nan, dst_nodata=np.nan)
                mask_ag = ~np.isnan(tmp_ag)
                stk_ag[out_idx][mask_ag] = tmp_ag[mask_ag]

            # Compute mean sun angles from reprojected data
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                mean_sza = np.nanmean(stk_ag[out_idx, 0])
                mean_saa = np.nanmean(stk_ag[out_idx, 1])

            meta_arrays['saz'].append(mean_saa)
            meta_arrays['sel'].append(90.0 - mean_sza)

            # Compute QA mask via SpecComplex
            # sc.get_hls_mask expects dict-like access: data_grp["Fmask"][t, :, :] and data_grp["solar_view_angles"][t, :, :, :]
            temp_grp = {'Fmask': stk_fm[:, 0, :, :], 'solar_view_angles': stk_ag}
            stk_mask[out_idx] = sc.get_hls_mask(temp_grp, out_idx,
                                                 sun_elevation_threshold=SUN_ELEVATION_THRESHOLD,
                                                 cloud_dilation=HLS_CLOUD_DILATION,
                                                 qa_reject_mask=QA_REJECT_MASK,
                                                 aerosol_accept_level=AEROSOL_ACCEPT_LEVEL).astype(bool)

            # Generate RGBA visual
            rgba_img = sc.generate_rgba_image(
                r_band=stk_sr[out_idx, 3, :, :],
                g_band=stk_sr[out_idx, 2, :, :],
                b_band=stk_sr[out_idx, 1, :, :],
                nodata=-9999)
            vis_data[out_idx, ...] = np.transpose(rgba_img, (2, 0, 1))

        return {
            'sr': stk_sr, 'fm': stk_fm, 'ag': stk_ag,
            'vis': vis_data, 'mask': stk_mask,
            'meta': meta_arrays, 'count': num_valid
        }

    # ==========================================
    # 3. EXECUTE PIPELINE
    # ==========================================
    print(f"\nProcessing HLSS30...")
    s30_data = process_sensor_direct("HLSS30.v2.0", ASSETS_S30, S30_TEMP_DIR,
                                     expected_sr=10, expected_fmask_idx=11)

    print(f"\nProcessing HLSL30...")
    l30_data = process_sensor_direct("HLSL30.v2.0", ASSETS_L30, L30_TEMP_DIR,
                                     expected_sr=7, expected_fmask_idx=8)

    # ==========================================
    # 4. WRITE OUTPUT HDF5
    # ==========================================
    print(f"\nWriting MGRS Stack HDF5: {OUTPUT_MGRS_HDF5}")
    with h5py.File(OUTPUT_MGRS_HDF5, 'w') as h5f:
        if s30_data:
            write_hdf_sensor_group(h5f, '/HDFEOS/GRIDS/HLSS30/Data Fields',
                                   s30_data, S30_WAVELENGTHS,
                                   master_crs.to_wkt(), master_transform, Location)
            del s30_data
            gc.collect()

        if l30_data:
            write_hdf_sensor_group(h5f, '/HDFEOS/GRIDS/HLSL30/Data Fields',
                                   l30_data, L30_WAVELENGTHS,
                                   master_crs.to_wkt(), master_transform, Location)

    print(f"\nPipeline Complete. MGRS Stack generated: {OUTPUT_MGRS_HDF5}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Unified HLS STAC-to-MGRS Pipeline")
    parser.add_argument("--location", type=str, default=None,
                        help="Target location from locations_config.yaml")
    args = parser.parse_args()
    main(target_location=args.location)
