import os
import json
import h5py
import numpy as np
import rasterio
import xml.etree.ElementTree as ET
from pyproj import Transformer
from rasterio.warp import reproject, Resampling
from rasterio.transform import Affine
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import math
import sys

# Import centralized ODL generator
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import hdfeos_odl
import SpecComplex as sc
from PIL import Image

# --- Configuration ---
LOCATION = 'Tait'
TIME_THRESHOLD_SECONDS = 120  # Group acquisitions within 2 minutes into the same temporal pass
SOURCE_DIR = "C:/satelliteImagery/enmap"
OUTPUT_DIR = SOURCE_DIR

TARGET_RED_NM = 680.0
TARGET_GREEN_NM = 540.0
TARGET_BLUE_NM = 480.0

SUN_ELEVATION_THRESHOLD = 30
ENMAP_CLOUD_DILATION = 4
ENMAP_REJECT_CLOUD = True
ENMAP_REJECT_CLOUD_SHADOW = True
ENMAP_REJECT_HAZE = True
ENMAP_REJECT_CIRRUS = True
ENMAP_REJECT_SNOW = True
ENMAP_REJECT_DEFECTIVE = True
ENMAP_REJECT_WATER = True

def get_utm_epsg_from_lonlat(lon, lat):
    """Automatically determine the UTM EPSG code based on a central coordinate."""
    zone = int((lon + 180) / 6) + 1
    if lat >= 0:
        return 32600 + zone
    else:
        return 32700 + zone

def parse_enmap_scene(json_path):
    """
    Extracts rich metrology and file paths using both the STAC JSON (for asset links, 
    band wavelengths, etc.) and the METADATA.XML (for exact spatial footprints).
    """
    # 1. Parse STAC JSON for high-level structure and assets
    with open(json_path, 'r') as f:
        stac = json.load(f)
    
    dt_str = stac['properties']['datetime']
    acq_time = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
    
    assets = stac['assets']
    base_dir = os.path.dirname(json_path)
    
    def resolve_path(href):
        return os.path.join(base_dir, os.path.basename(href))
    
    # 2. Extract band metadata
    eo_bands = assets.get('image', {}).get('eo:bands', [])
    if eo_bands and 'center_wavelength' in eo_bands[0]:
        wavelengths = [b['center_wavelength'] for b in eo_bands]
        fwhms = [b.get('full_width_half_max', 0) for b in eo_bands]
    else:
        wavelengths = [0.0] * 224
        fwhms = [0.0] * 224
    
    sun_elev = stac['properties'].get('view:sun_elevation', 0.0)
    sun_azim = stac['properties'].get('view:sun_azimuth', 0.0)
    off_nadir = float(stac['properties'].get('enmap:acrossOffNadirAngle', 0.0))
    inc_angle = 0.0
    view_azim = float(stac['properties'].get('enmap:sceneAzimuthAngle', 0.0))
    nodata_val = float(stac['properties'].get('enmap:nodata', -32768))

    mask_keys = ['quality_classes', 'quality_cloud', 'quality_cloud_shadow', 'quality_haze', 'quality_cirrus', 'quality_snow', 'quality_testflags', 'defective_pixel_mask']
    mask_tifs = {mk: "" for mk in mask_keys}
    
    for mk in mask_keys:
        if mk in assets and 'href' in assets[mk]:
            mask_tifs[mk] = resolve_path(assets[mk]['href'])

    image_tif = resolve_path(assets['image']['href'])
    
    # 3. Locate and Parse METADATA.XML for exact spatial coverage
    xml_filename = stac['id'] + "-METADATA.XML"
    xml_path = os.path.join(base_dir, xml_filename)
    
    xml_coords = []
    if os.path.exists(xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        poly_points = root.findall('.//base/spatialCoverage/boundingPolygon/point')
        for pt in poly_points:
            frame = pt.findtext('frame')
            if frame != 'center':
                lat = float(pt.findtext('latitude'))
                lon = float(pt.findtext('longitude'))
                xml_coords.append((lon, lat))
    
    if not xml_coords:
        # Fallback to STAC generic BBOX if XML fails or is missing
        bbox = stac['bbox']
        xml_coords = [
            (bbox[0], bbox[1]), (bbox[2], bbox[1]), 
            (bbox[2], bbox[3]), (bbox[0], bbox[3])
        ]

    ret = {
        'id': stac['id'],
        'time': acq_time,
        'platform': stac['properties'].get('platform', 'enmap'),
        'reflectance_tif': image_tif,
        'wavelengths': np.array(wavelengths, dtype=np.float32),
        'fwhm': np.array(fwhms, dtype=np.float32),
        'sun_elevation': sun_elev,
        'sun_azimuth': sun_azim,
        'view_off_nadir': off_nadir,
        'view_incidence_angle': inc_angle,
        'view_azimuth': view_azim,
        'nodata': nodata_val,
        'stac_dict': stac,
        'bounds_lonlat': xml_coords
    }
    ret.update(mask_tifs)
    return ret

def intersects(bbox1, bbox2):
    """Evaluates whether two [min_lon, min_lat, max_lon, max_lat] bounding boxes intersect."""
    return not (bbox1[2] < bbox2[0] or bbox1[0] > bbox2[2] or
                bbox1[3] < bbox2[1] or bbox1[1] > bbox2[3])

def calculate_mgrs_aligned_grid(scenes, roi_bbox=None):
    """
    Computes a collective 30-meter UTM grid mathematically matching the MGRS format used by HLS.
    If roi_bbox is provided, forces the grid to strictly encapsulate the ROI instead of the scenes.
    """
    # 1. Aggregate all geographic coordinates to find the centroid and UTM zone
    all_lons = []
    all_lats = []
    
    if roi_bbox:
        min_lon, min_lat, max_lon, max_lat = roi_bbox
        all_lons = [min_lon, max_lon]
        all_lats = [min_lat, max_lat]
        corners = [
            (min_lon, max_lat),
            (max_lon, max_lat),
            (max_lon, min_lat),
            (min_lon, min_lat)
        ]
    else:
        for scene in scenes:
            for lon, lat in scene['bounds_lonlat']:
                all_lons.append(lon)
                all_lats.append(lat)
            
    center_lon = np.mean(all_lons)
    center_lat = np.mean(all_lats)
    
    epsg_code = get_utm_epsg_from_lonlat(center_lon, center_lat)
    target_crs = f"EPSG:{epsg_code}"
    
    # 2. Transform all corner points into the target UTM CRS
    transformer = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
    all_xs = []
    all_ys = []

    if roi_bbox:
        xs, ys = zip(*corners)
        proj_xs, proj_ys = transformer.transform(xs, ys)
        all_xs.extend(proj_xs)
        all_ys.extend(proj_ys)
    else:
        for scene in scenes:
            xs, ys = zip(*scene['bounds_lonlat'])
            proj_xs, proj_ys = transformer.transform(xs, ys)
            all_xs.extend(proj_xs)
            all_ys.extend(proj_ys)
        
    min_x, max_x = min(all_xs), max(all_xs)
    min_y, max_y = min(all_ys), max(all_ys)
    
    # 3. Snap boundaries to exact 30-meter multiples (MGRS HLS standard)
    ul_x = math.floor(min_x / 30.0) * 30.0
    ul_y = math.ceil(max_y / 30.0) * 30.0
    lr_x = math.ceil(max_x / 30.0) * 30.0
    lr_y = math.floor(min_y / 30.0) * 30.0
    
    width = int((lr_x - ul_x) / 30.0)
    height = int((ul_y - lr_y) / 30.0)
    
    target_transform = Affine.translation(ul_x, ul_y) * Affine.scale(30.0, -30.0)
    
    return target_transform, width, height, target_crs, (ul_x, ul_y), (lr_x, lr_y)

def process_mgrs_stack(target_location):
    print(f"Discovering EnMAP collections for location: {target_location}...")
    import yaml
    
    script_dir = Path(__file__).resolve().parent
    config_path = os.path.join(script_dir, "locations_config.yaml")
    if not os.path.exists(config_path):
        config_path = os.path.join(script_dir.parent, "locations_config.yaml")
        
    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)
        
    loc_data = config_data.get("locations", {}).get(target_location)
    if not loc_data:
        raise ValueError(f"CRITICAL: Location {target_location} not found in locations_config.yaml")
        
    roi_bbox = [
        loc_data["ROI_LON_MIN"],
        loc_data["ROI_LAT_MIN"],
        loc_data["ROI_LON_MAX"],
        loc_data["ROI_LAT_MAX"]
    ]
    
    source_cache = loc_data.get("SOURCE_CACHE") or target_location
    location_dir = os.path.join(SOURCE_DIR, f"{source_cache}_SourceData")
    if not os.path.exists(location_dir):
        # Check alternative folder name (from older stacker)
        alt_dir = os.path.join(SOURCE_DIR, f"{source_cache}_EnMAP")
        if os.path.exists(alt_dir):
            location_dir = alt_dir
        else:
            raise FileNotFoundError(f"Location directory not found: {location_dir} or {alt_dir}")

    # 1. Recursive Data Discovery
    root_path = Path(location_dir)
    json_files = list(root_path.rglob("*.json"))
    json_files = [f for f in json_files if "catalog" not in f.name.lower()]
    valid_json_files = [f for f in json_files if any(f.parent.glob("*.TIF")) or any(f.parent.glob("*.tif"))]
    
    if not valid_json_files:
        raise FileNotFoundError(f"CRITICAL: No valid EnMAP JSON files with adjacent TIFFs found in {location_dir}")

    raw_scenes = []
    for j_path in valid_json_files:
        try:
            scene_data = parse_enmap_scene(str(j_path))
            if not os.path.exists(scene_data['reflectance_tif']):
                continue
                
            # Filter by bounding box intersection
            lons = [pt[0] for pt in scene_data['bounds_lonlat']]
            lats = [pt[1] for pt in scene_data['bounds_lonlat']]
            scene_bbox = [min(lons), min(lats), max(lons), max(lats)]
            
            if intersects(scene_bbox, roi_bbox):
                raw_scenes.append(scene_data)
        except Exception as e:
            print(f"Warning: Skipping {j_path.name} due to parsing error: {e}")
            continue

    if not raw_scenes:
        raise FileNotFoundError(f"CRITICAL: No complete EnMAP scenes found.")

    raw_scenes.sort(key=lambda x: x['time'])

    # 2. Temporal Grouping
    grouped_scenes = []
    current_group = [raw_scenes[0]]
    for i in range(1, len(raw_scenes)):
        delta = (raw_scenes[i]['time'] - current_group[-1]['time']).total_seconds()
        if delta <= TIME_THRESHOLD_SECONDS:
            current_group.append(raw_scenes[i])
        else:
            grouped_scenes.append(current_group)
            current_group = [raw_scenes[i]]
    grouped_scenes.append(current_group)

    print(f"Aggregated {len(raw_scenes)} scenes into {len(grouped_scenes)} temporal passes.")

    # 3. Establish Universal MGRS-Aligned Grid
    tf_target, width, height, crs_wkt, ul_coords, lr_coords = calculate_mgrs_aligned_grid(raw_scenes, roi_bbox=roi_bbox)
    print(f"MGRS Target Grid: {width}x{height} 30m pixels (CRS: {crs_wkt})")
    print(f"Origin (UL): {ul_coords}, Lower Right: {lr_coords}")

    base_wv = raw_scenes[0]['wavelengths']
    n_times = len(grouped_scenes)
    n_bands = len(base_wv)

    output_file = os.path.join(OUTPUT_DIR, f"EnMAP_MGRS_Stack_{target_location}.h5")

    # 4. HDF5 Tensor Construction
    with h5py.File(output_file, 'w') as out_h5:
        grp_enmap = out_h5.create_group("HDFEOS/GRIDS/ENMAP/Data Fields")
        meta_grp = out_h5.create_group("METADATA")
        info_grp = out_h5.create_group("HDFEOS INFORMATION")
        
        datasets_info = [
            ("surface_reflectance", np.dtype('float32'), 4, ["Time", "Band", "YDim", "XDim"]),
            ("ortho_visual", np.dtype('uint8'), 4, ["Time", "RGBABand", "YDim", "XDim"]),
            ("common_mask", np.dtype('bool'), 3, ["Time", "YDim", "XDim"])
        ]
        mask_keys = ['quality_classes', 'quality_cloud', 'quality_cloud_shadow', 'quality_haze', 'quality_cirrus', 'quality_snow', 'quality_testflags', 'defective_pixel_mask']
        for mk in mask_keys:
            datasets_info.append((mk, np.dtype('uint8'), 3, ["Time", "YDim", "XDim"]))
        
        rad_nodata = raw_scenes[0]['nodata']
        ds_rad = grp_enmap.create_dataset("surface_reflectance", shape=(n_times, n_bands, height, width), dtype='float32', compression="gzip", fillvalue=rad_nodata)
        mask_ds_dict = {}
        for mk in mask_keys:
            mask_ds_dict[mk] = grp_enmap.create_dataset(mk, shape=(n_times, height, width), dtype='uint8', compression="gzip", fillvalue=255)
        ds_vis = grp_enmap.create_dataset("ortho_visual", shape=(n_times, 4, height, width), dtype='uint8', compression="gzip", fillvalue=0)

        acq_time_array = np.zeros(n_times, dtype='float64')
        platform_array = []
        sun_elev_array = np.zeros(n_times, dtype='float32')
        sun_azim_array = np.zeros(n_times, dtype='float32')
        off_nadir_array = np.zeros(n_times, dtype='float32')
        inc_angle_array = np.zeros(n_times, dtype='float32')
        view_azim_array = np.zeros(n_times, dtype='float32')
        good_wavelengths_array = []

        gdal_transform = [tf_target.c, tf_target.a, tf_target.b, tf_target.f, tf_target.d, tf_target.e]

        # 5. Iterative Data Assimilation
        for t_idx, group in enumerate(grouped_scenes):
            pass_ts = group[0]['time'].isoformat()
            print(f"  [Pass {t_idx+1}/{n_times}] Assimilating Swath: {pass_ts}...")
            
            canvas_rad = np.full((n_bands, height, width), rad_nodata, dtype='float32')
            mask_canvases = {mk: np.full((1, height, width), 255, dtype='uint8') for mk in mask_keys}
            
            for scene in group:
                scene_valid_mask_2d = None
                try:
                    with rasterio.open(scene['reflectance_tif']) as src:
                        incoming_rad = np.full((n_bands, height, width), rad_nodata, dtype='float32')
                        reproject(
                            source=rasterio.band(src, list(range(1, src.count + 1))),
                            destination=incoming_rad,
                            src_transform=src.transform, src_crs=src.crs, dst_transform=tf_target, dst_crs=crs_wkt,
                            resampling=Resampling.nearest,
                            src_nodata=rad_nodata, dst_nodata=rad_nodata
                        )
                        valid_mask = (incoming_rad != rad_nodata)
                        canvas_rad[valid_mask] = incoming_rad[valid_mask]
                        scene_valid_mask_2d = (incoming_rad[0] != rad_nodata)
                except rasterio.errors.RasterioIOError:
                    print(f"    Warning: Failed to read primary imagery for {scene['id']}")
                    
                for mk in mask_keys:
                    if scene.get(mk) and os.path.exists(scene[mk]):
                        try:
                            with rasterio.open(scene[mk]) as src:
                                incoming_mask = np.full((1, height, width), 255, dtype='uint8')
                                reproject(
                                    source=rasterio.band(src, 1),
                                    destination=incoming_mask,
                                    src_transform=src.transform, src_crs=src.crs, dst_transform=tf_target, dst_crs=crs_wkt,
                                    resampling=Resampling.nearest,
                                    src_nodata=255, dst_nodata=255
                                )
                                if scene_valid_mask_2d is not None:
                                    # Use the imagery's valid footprint to prevent invalid corner overlaps
                                    mask_canvases[mk][0, scene_valid_mask_2d] = incoming_mask[0, scene_valid_mask_2d]
                                else:
                                    valid_mask = (incoming_mask != 255)
                                    mask_canvases[mk][valid_mask] = incoming_mask[valid_mask]
                        except rasterio.errors.RasterioIOError:
                            pass
            
            frame_good_wv = ~np.all((canvas_rad == rad_nodata) | np.isnan(canvas_rad), axis=(1, 2))
            good_wavelengths_array.append(frame_good_wv)
            
            ds_rad[t_idx, ...] = canvas_rad
            
            # Dynamically compute defective pixel mask (pixels evaluating to exactly EnMAP hardware nodata)
            bad_pixel_locs = (canvas_rad == -32768)
            spatial_bad_pixels = np.any(bad_pixel_locs, axis=0)
            if 'defective_pixel_mask' in mask_canvases:
                mask_canvases['defective_pixel_mask'][0, spatial_bad_pixels] = 1

            for mk in mask_keys:
                mask_ds_dict[mk][t_idx, ...] = mask_canvases[mk][0, ...]
            
            acq_time_array[t_idx] = group[0]['time'].timestamp()
            platform_array.append(group[0]['platform'])
            sun_elev_array[t_idx] = group[0]['sun_elevation']
            sun_azim_array[t_idx] = group[0]['sun_azimuth']
            off_nadir_array[t_idx] = group[0]['view_off_nadir']
            inc_angle_array[t_idx] = group[0]['view_incidence_angle']
            view_azim_array[t_idx] = group[0]['view_azimuth']

            meta_grp.attrs[f"frame_{t_idx}_json"] = json.dumps(group[0]['stac_dict'])

        # Create common_mask dataset
        print("  Generating Common Mask for EnMAP on MGRS Grid...")
        mask_ds = grp_enmap.create_dataset('common_mask', shape=(n_times, height, width), dtype=bool, compression="gzip", fillvalue=True)
        mask_ds.attrs['description'] = "True = Invalid/Masked, False = Valid. Generated from SpecComplex ARD rules."
        
        from tqdm import tqdm
        for t_idx in tqdm(range(n_times), desc="  Computing ARD masks"):
            valid_mask = sc.get_enmap_mask(
                grp_enmap, t_idx, (height, width), 
                sun_elevation_threshold=SUN_ELEVATION_THRESHOLD,
                cloud_dilation=ENMAP_CLOUD_DILATION,
                reject_cloud=ENMAP_REJECT_CLOUD,
                reject_shadow=ENMAP_REJECT_CLOUD_SHADOW,
                reject_haze=ENMAP_REJECT_HAZE,
                reject_cirrus=ENMAP_REJECT_CIRRUS,
                reject_snow=ENMAP_REJECT_SNOW,
                reject_defective=ENMAP_REJECT_DEFECTIVE,
                reject_water=ENMAP_REJECT_WATER
            )
            mask_ds[t_idx, ...] = valid_mask

        # Create ortho visual (RGBA) datasets
        r_idx = int(np.argmin(np.abs(base_wv - TARGET_RED_NM)))
        g_idx = int(np.argmin(np.abs(base_wv - TARGET_GREEN_NM)))
        b_idx = int(np.argmin(np.abs(base_wv - TARGET_BLUE_NM)))
        
        for t_idx in tqdm(range(n_times), desc="  Writing visual channels"):
            r_band = ds_rad[t_idx, r_idx, :, :]
            g_band = ds_rad[t_idx, g_idx, :, :]
            b_band = ds_rad[t_idx, b_idx, :, :]
            
            r_input = np.where((r_band == rad_nodata) | np.isnan(r_band), np.nan, r_band)
            g_input = np.where((g_band == rad_nodata) | np.isnan(g_band), np.nan, g_band)
            b_input = np.where((b_band == rad_nodata) | np.isnan(b_band), np.nan, b_band)
            
            rgba_img = sc.generate_rgba_image(r_input, g_input, b_input)
            ds_vis[t_idx, ...] = np.transpose(rgba_img, (2, 0, 1))

        datasets_to_attr = [ds_rad, ds_vis, mask_ds] + list(mask_ds_dict.values())
        for dataset in datasets_to_attr:
            dataset.attrs['spatial_ref'] = crs_wkt
            dataset.attrs['GeoTransform'] = np.array(gdal_transform, dtype='float64')

        dt_str = h5py.string_dtype(encoding='ascii')
        ds_rad.attrs["acquisition_time"] = acq_time_array
        ds_rad.attrs["sun_elevation"] = sun_elev_array
        ds_rad.attrs["sun_azimuth"] = sun_azim_array
        ds_rad.attrs["view_off_nadir"] = off_nadir_array
        ds_rad.attrs["view_incidence_angle"] = inc_angle_array
        ds_rad.attrs["view_azimuth"] = view_azim_array
        ds_rad.attrs.create("spacecraft_id", data=np.array(platform_array, dtype=dt_str))
        ds_rad.attrs["unit"] = "reflectance"
        ds_rad.attrs["wavelengths"] = base_wv
        ds_rad.attrs["fwhm"] = raw_scenes[0]['fwhm']
        if len(good_wavelengths_array) == n_times:
            ds_rad.attrs["all_good_wavelengths"] = np.array(good_wavelengths_array, dtype=bool)

        if 'quality_classes' in mask_ds_dict:
            mask_ds_dict['quality_classes'].attrs["description"] = "EnMAP L2A Quality Classes Mask"

        # UTM zone extraction for ODL metadata
        zone = int(crs_wkt.split(":")[-1]) % 100
        proj_code = "HE5_GCTP_UTM"
        
        odl_grid = hdfeos_odl.generate_dynamic_odl_grid_string("ENMAP", width, height, tf_target, proj_code, zone, (0,)*13, datasets_info, n_times, n_bands)
        struct_meta = f"GROUP=SwathStructure\nEND_GROUP=SwathStructure\nGROUP=GridStructure\n{odl_grid}\nEND_GROUP=GridStructure\n"
        dt_str = h5py.string_dtype(encoding='ascii')
        info_grp.create_dataset("StructMetadata.0", (1,), dtype=dt_str, data=struct_meta)

        # 6. Export Visual Channels as PNGs
        print("  Exporting visual frames as PNGs...")
        for t_idx in range(n_times):
            pass_ts_safe = grouped_scenes[t_idx][0]['time'].strftime("%Y%m%dT%H%M%S")
            png_filename = f"EnMAP_{target_location}_{pass_ts_safe}_RGB.png"
            png_path = os.path.join(location_dir, png_filename)
            rgba_frame = np.transpose(ds_vis[t_idx, ...], (1, 2, 0))
            img = Image.fromarray(rgba_frame, 'RGBA')
            img.save(png_path)
            print(f"    Saved: {png_filename}")
            
            frame_mask = mask_ds[t_idx, ...]
            overlay = Image.new('RGBA', img.size, (255, 166, 0, 0))
            overlay_data = np.array(overlay)
            overlay_data[frame_mask == True] = [255, 166, 0, 128]
            overlay_img = Image.fromarray(overlay_data, 'RGBA')
            masked_img = Image.alpha_composite(img, overlay_img)
            
            masked_png_name = f"EnMAP_{target_location}_{pass_ts_safe}_RGB_masked.png"
            masked_png_path = os.path.join(location_dir, masked_png_name)
            masked_img.save(masked_png_path)
            print(f"    Saved: {masked_png_name}")

    print(f"\nTensor Synthesis Complete. MGRS Stack Stored at: {output_file}")

if __name__ == "__main__":
    process_mgrs_stack(LOCATION)
