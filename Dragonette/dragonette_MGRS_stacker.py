import os
import sys
import glob
import math
import json
import numpy as np
import h5py
from pathlib import Path
from pyproj import CRS, Transformer
from rasterio.warp import reproject, Resampling
from rasterio.transform import Affine
from datetime import datetime
import rasterio
from PIL import Image

LOCATION='Tait'
# Add parent folder to sys.path to find SpecComplex
script_dir = Path(__file__).resolve().parent
if str(script_dir.parent) not in sys.path:
    sys.path.insert(0, str(script_dir.parent))
import SpecComplex as sc

SOURCE_DIR = r"C:\satelliteImagery\dragonette"
OUTPUT_DIR = r"C:\satelliteImagery\dragonette"

def get_utm_epsg_from_lonlat(lon, lat):
    utm_band = str((math.floor((lon + 180) / 6) % 60) + 1)
    if len(utm_band) == 1:
        utm_band = '0' + utm_band
    if lat >= 0:
        epsg_code = '326' + utm_band
    else:
        epsg_code = '327' + utm_band
    return int(epsg_code)

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

def parse_wyvern_stac(json_path):
    """Extracts metrology and file paths from the Wyvern STAC JSON."""
    with open(json_path, 'r') as f:
        stac = json.load(f)
    
    dt_str = stac['properties']['datetime']
    acq_time = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
    
    epsg = stac['properties']['proj:epsg']
    
    if stac['properties']['processing:level'] != 'L2A':
        raise ValueError(f"CRITICAL: {json_path} has unsupported processing level {stac['properties'].get('processing:level')}. Expected strictly L2A.")
    
    assets = stac['assets']
    base_dir = os.path.dirname(json_path)
    
    def resolve_path(href):
        return os.path.join(base_dir, os.path.basename(href))
    
    eo_bands = assets['Cloud optimized GeoTiff']['eo:bands']
    wavelengths = [b['center_wavelength'] * 1000 for b in eo_bands]
    fwhms = [b['full_width_half_max'] * 1000 for b in eo_bands]
    esun = [b['solar_illumination'] for b in eo_bands]

    sun_elev = stac['properties']['view:sun_elevation']
    sun_azim = stac['properties']['view:sun_azimuth']
    off_nadir = stac['properties']['view:off_nadir']
    inc_angle = stac['properties']['view:incidence_angle']
    view_azim = stac['properties']['view:azimuth']
    
    nodata_val = assets['Cloud optimized GeoTiff']['raster:bands'][0].get('nodata', -9999)

    radiance_tif = resolve_path(assets['Cloud optimized GeoTiff']['href'])
    
    with rasterio.open(radiance_tif) as src:
        b = src.bounds
        if src.crs != CRS.from_epsg(4326):
            transformer = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
            left, bottom = transformer.transform(b.left, b.bottom)
            right, top = transformer.transform(b.right, b.top)
            bounds_lonlat = [(left, bottom), (left, top), (right, top), (right, bottom)]
        else:
            bounds_lonlat = [(b.left, b.bottom), (b.left, b.top), (b.right, b.top), (b.right, b.bottom)]

    return {
        'id': stac['id'],
        'json_path': json_path,
        'time': acq_time,
        'epsg': epsg,
        'platform': stac['properties']['platform'],
        'radiance_tif': radiance_tif,
        'data_mask_tif': resolve_path(assets['Data Mask']['href']),
        'pixel_quality_tif': resolve_path(assets['Pixel Quality Mask']['href']),
        'bounds_lonlat': bounds_lonlat,
        'wavelengths': np.array(wavelengths, dtype=np.float32),
        'fwhm': np.array(fwhms, dtype=np.float32),
        'solar_illumination': np.array(esun, dtype=np.float32),
        'sun_elevation': sun_elev,
        'sun_azimuth': sun_azim,
        'view_off_nadir': off_nadir,
        'view_incidence_angle': inc_angle,
        'view_azimuth': view_azim,
        'nodata': nodata_val,
        'stac_dict': stac
    }

def process_dragonette_mgrs_stack(target_location):
    print(f"Discovering Wyvern STAC collections for location: {target_location}...")
    import yaml
    
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
        raise FileNotFoundError(f"CRITICAL: Directory {location_dir} does not exist.")
        
    root_path = Path(location_dir)
    json_files = list(root_path.rglob("*.json"))
    json_files = [f for f in json_files if "catalog" not in f.name.lower()]
    valid_json_files = [f for f in json_files if any(f.parent.glob("*.tiff")) or any(f.parent.glob("*.tif"))]
    
    if not valid_json_files:
        raise FileNotFoundError(f"CRITICAL: No valid Wyvern STAC JSON files with adjacent TIFFs found in {location_dir}")

    raw_scenes = []
    for j_path in valid_json_files:
        try:
            scene_data = parse_wyvern_stac(str(j_path))
            
            # Filter by bounding box intersection
            lons = [pt[0] for pt in scene_data['bounds_lonlat']]
            lats = [pt[1] for pt in scene_data['bounds_lonlat']]
            scene_bbox = [min(lons), min(lats), max(lons), max(lats)]
            
            if intersects(scene_bbox, roi_bbox):
                raw_scenes.append(scene_data)
                
        except Exception as e:
            print(f"Warning: Skipping {j_path.name} due to parsing error: {e}")

    if not raw_scenes:
        raise ValueError(f"CRITICAL: No scenes intersect the bounding box for {target_location}.")

    raw_scenes.sort(key=lambda x: x['time'])
    
    # Dragonette does not have a chunking strategy; each scene is treated as a distinct temporal pass
    n_times = len(raw_scenes)
    print(f"Discovered {n_times} distinct Dragonette scenes intersecting the ROI.")

    # Establish MGRS-Aligned Grid strictly bound to the ROI
    tf_target, width, height, crs_wkt, ul_coords, lr_coords = calculate_mgrs_aligned_grid(raw_scenes, roi_bbox=roi_bbox)
    print(f"MGRS Target Grid: {width}x{height} 30m pixels (CRS: {crs_wkt})")
    print(f"Origin (UL): {ul_coords}, Lower Right: {lr_coords}")
    
    base_wv = raw_scenes[0]['wavelengths']
    for scn in raw_scenes:
        if len(scn['wavelengths']) != len(base_wv) or not np.allclose(scn['wavelengths'], base_wv, atol=1.0):
            raise ValueError(f"CRITICAL SPECTRAL MISMATCH: Dataset {scn['id']} wavelength array does not match stack standard.")
    n_bands = len(base_wv)

    output_file = os.path.join(OUTPUT_DIR, f"Dragonette_MGRS_Stack_{target_location}.h5")
    
    with h5py.File(output_file, 'w') as out_h5:
        grp_wyvern = out_h5.create_group("HDFEOS/GRIDS/DRAGONETTE/Data Fields")
        
        rad_nodata = raw_scenes[0]['nodata']
        ds_rad = grp_wyvern.create_dataset("surface_reflectance", shape=(n_times, n_bands, height, width), dtype='float32', compression="gzip", compression_opts=5, fillvalue=rad_nodata)
        ds_dmask = grp_wyvern.create_dataset("data_mask", shape=(n_times, 4, height, width), dtype='uint8', compression="gzip", compression_opts=5, fillvalue=255)
        ds_qmask = grp_wyvern.create_dataset("pixel_quality_mask", shape=(n_times, n_bands, height, width), dtype='uint8', compression="gzip", compression_opts=5, fillvalue=255)
        ds_vis = grp_wyvern.create_dataset("ortho_visual", shape=(n_times, 4, height, width), dtype='uint8', compression="gzip", compression_opts=5, fillvalue=0)
        ds_common_mask = grp_wyvern.create_dataset("common_mask", shape=(n_times, height, width), dtype=bool, compression="gzip", compression_opts=5, fillvalue=False)
        
        # Meta lists for attributes
        good_wavelengths_array = []
        acq_time_array = np.zeros(n_times, dtype='float64')
        platform_array = []
        sun_elev_array = np.zeros(n_times, dtype='float64')
        sun_azim_array = np.zeros(n_times, dtype='float64')
        off_nadir_array = np.zeros(n_times, dtype='float64')
        inc_angle_array = np.zeros(n_times, dtype='float64')
        view_azim_array = np.zeros(n_times, dtype='float64')
        solar_illum_array = np.zeros((n_times, n_bands), dtype='float32')

        gdal_transform = [tf_target.c, tf_target.a, tf_target.b, tf_target.f, tf_target.d, tf_target.e]

        for t_idx, scene in enumerate(raw_scenes):
            print(f"  [Pass {t_idx+1}/{n_times}] Assimilating Swath: {scene['time'].isoformat()}...")
            
            with rasterio.open(scene['radiance_tif']) as src:
                incoming_rad = np.full((n_bands, height, width), rad_nodata, dtype='float32')
                reproject(
                    source=rasterio.band(src, list(range(1, src.count + 1))),
                    destination=incoming_rad,
                    src_transform=src.transform, src_crs=src.crs, dst_transform=tf_target, dst_crs=crs_wkt,
                    resampling=Resampling.nearest,
                    src_nodata=rad_nodata, dst_nodata=rad_nodata
                )
                ds_rad[t_idx, ...] = incoming_rad
                
            with rasterio.open(scene['data_mask_tif']) as src:
                incoming_dmask = np.full((4, height, width), 255, dtype='uint8')
                reproject(
                    source=rasterio.band(src, list(range(1, src.count + 1))),
                    destination=incoming_dmask,
                    src_transform=src.transform, src_crs=src.crs, dst_transform=tf_target, dst_crs=crs_wkt,
                    resampling=Resampling.nearest,
                    src_nodata=255, dst_nodata=255
                )
                ds_dmask[t_idx, ...] = incoming_dmask

            with rasterio.open(scene['pixel_quality_tif']) as src:
                incoming_qmask = np.full((n_bands, height, width), 255, dtype='uint8')
                reproject(
                    source=rasterio.band(src, list(range(1, src.count + 1))),
                    destination=incoming_qmask,
                    src_transform=src.transform, src_crs=src.crs, dst_transform=tf_target, dst_crs=crs_wkt,
                    resampling=Resampling.nearest,
                    src_nodata=255, dst_nodata=255
                )
                ds_qmask[t_idx, ...] = incoming_qmask

            # Calculate good_wavelengths for this scene based on pixel_quality_mask
            # A band is "good" (True) if it has valid data and is less than 10% interpolated
            valid_pixels_per_band = np.sum(incoming_qmask != 255, axis=(1, 2))
            interpolated_per_band = np.sum(incoming_qmask == 1, axis=(1, 2))
            
            with np.errstate(divide='ignore', invalid='ignore'):
                interp_ratio = interpolated_per_band / valid_pixels_per_band
            
            frame_good_wv = (valid_pixels_per_band > 0) & (np.nan_to_num(interp_ratio) < 0.10)
            good_wavelengths_array.append(frame_good_wv)

            # Generate boolean common_mask
            # Band 0: QA_CLEAR_MASK, Band 1: QA_CLOUD_MASK, Band 2: QA_HAZE_MASK, Band 3: QA_CLOUD_SHADOW_MASK
            qa_clear = incoming_dmask[0, :, :]
            qa_cloud = incoming_dmask[1, :, :]
            qa_haze = incoming_dmask[2, :, :]
            qa_shadow = incoming_dmask[3, :, :]
            
            # We ONLY mask the entire pixel for spatial obstructions (Clouds/Haze/Shadows).
            # Interpolated/Bad bands are left to the 3D pixel_quality_mask to prevent throwing out the entire pixel.
            invalid_mask = (qa_cloud == 1) | (qa_haze == 1) | (qa_shadow == 1) | (qa_clear == 255)
            ds_common_mask[t_idx, ...] = invalid_mask

            # Ortho visual (RGB)
            r_idx = int(np.argmin(np.abs(base_wv - 680.0)))
            g_idx = int(np.argmin(np.abs(base_wv - 534.0)))
            b_idx = int(np.argmin(np.abs(base_wv - 480.0)))
            
            r_band = incoming_rad[r_idx, :, :]
            g_band = incoming_rad[g_idx, :, :]
            b_band = incoming_rad[b_idx, :, :]
            
            r_input = np.where(r_band == rad_nodata, np.nan, r_band)
            g_input = np.where(g_band == rad_nodata, np.nan, g_band)
            b_input = np.where(b_band == rad_nodata, np.nan, b_band)
            
            rgba_img = sc.generate_rgba_image(r_input, g_input, b_input)
            ds_vis[t_idx, ...] = np.transpose(rgba_img, (2, 0, 1))
            
            # Store metadata
            acq_time_array[t_idx] = scene['time'].timestamp()
            platform_array.append(scene['platform'])
            sun_elev_array[t_idx] = scene['sun_elevation']
            sun_azim_array[t_idx] = scene['sun_azimuth']
            off_nadir_array[t_idx] = scene['view_off_nadir']
            inc_angle_array[t_idx] = scene['view_incidence_angle']
            view_azim_array[t_idx] = scene['view_azimuth']
            solar_illum_array[t_idx, :] = scene['solar_illumination']

        # Save Attributes directly onto 'radiance' to mimic 'surface_reflectance' architecture
        dt_str = h5py.string_dtype(encoding='ascii')
        ds_rad.attrs['spatial_ref'] = crs_wkt
        ds_rad.attrs['GeoTransform'] = np.array(gdal_transform, dtype='float64')
        ds_rad.attrs['wavelengths'] = base_wv
        ds_rad.attrs['fwhm'] = scene['fwhm']
        ds_rad.attrs['solar_illumination'] = solar_illum_array
        ds_rad.attrs['acquisition_time'] = acq_time_array
        ds_rad.attrs.create('spacecraft_id', data=np.array(platform_array, dtype=dt_str))
        ds_rad.attrs['sun_elevation'] = sun_elev_array
        ds_rad.attrs['sun_azimuth'] = sun_azim_array
        ds_rad.attrs['view_off_nadir'] = off_nadir_array
        ds_rad.attrs['view_incidence_angle'] = inc_angle_array
        ds_rad.attrs['view_azimuth'] = view_azim_array
        if len(good_wavelengths_array) == n_times:
            ds_rad.attrs['all_good_wavelengths'] = np.array(good_wavelengths_array, dtype=bool)
        
        # Add to other datasets for completeness
        for name in ["data_mask", "pixel_quality_mask", "ortho_visual", "common_mask"]:
            grp_wyvern[name].attrs['spatial_ref'] = crs_wkt
            grp_wyvern[name].attrs['GeoTransform'] = np.array(gdal_transform, dtype='float64')
            
        ds_common_mask.attrs['description'] = "True = Invalid/Masked, False = Valid. Computed from Wyvern QA Data Mask (Cloud/Haze/Shadow)."

    print("  Exporting visual frames as PNGs...")
    with h5py.File(output_file, 'r') as out_h5:
        vis = out_h5["HDFEOS/GRIDS/DRAGONETTE/Data Fields/ortho_visual"]
        mask = out_h5["HDFEOS/GRIDS/DRAGONETTE/Data Fields/common_mask"]
        for t_idx in range(n_times):
            pass_ts = raw_scenes[t_idx]['time'].strftime("%Y%m%dT%H%M%S")
            frame_rgba = vis[t_idx, ...]
            frame_rgba = np.transpose(frame_rgba, (1, 2, 0))
            img = Image.fromarray(frame_rgba, 'RGBA')
            png_name = f"Dragonette_{target_location}_{pass_ts}_RGB.png"
            png_path = os.path.join(location_dir, png_name)
            img.save(png_path)
            print(f"    Saved: {png_name}")
            
            frame_mask = mask[t_idx, ...]
            overlay = Image.new('RGBA', img.size, (255, 166, 0, 0))
            overlay_data = np.array(overlay)
            overlay_data[frame_mask == True] = [255, 166, 0, 128]
            overlay_img = Image.fromarray(overlay_data, 'RGBA')
            masked_img = Image.alpha_composite(img, overlay_img)
            
            masked_png_name = f"Dragonette_{target_location}_{pass_ts}_RGB_masked.png"
            masked_png_path = os.path.join(location_dir, masked_png_name)
            masked_img.save(masked_png_path)
            print(f"    Saved: {masked_png_name}")

    print(f"\nTensor Synthesis Complete. MGRS Stack Stored at: {output_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--location", type=str, default=LOCATION, help="Target location prefix")
    args = parser.parse_args()
    process_dragonette_mgrs_stack(args.location)
