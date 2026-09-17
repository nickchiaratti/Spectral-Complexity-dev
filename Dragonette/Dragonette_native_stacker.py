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

LOCATION='Rochesterv2'
# Add parent folder to sys.path to find SpecComplex
script_dir = Path(__file__).resolve().parent
if str(script_dir.parent) not in sys.path:
    sys.path.insert(0, str(script_dir.parent))
import SpecComplex as sc
from Harmonized_SC.sc_data_utils import scale_reflectance_for_storage, add_reflectance_attributes

SOURCE_DIR = r"C:\satelliteImagery\dragonette"
OUTPUT_DIR = r"C:\satelliteImagery\dragonette"



def intersects(bbox1, bbox2):
    """Evaluates whether two [min_lon, min_lat, max_lon, max_lat] bounding boxes intersect."""
    return not (bbox1[2] < bbox2[0] or bbox1[0] > bbox2[2] or
                bbox1[3] < bbox2[1] or bbox1[1] > bbox2[3])



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
    wavelengths = [b['center_wavelength'] for b in eo_bands]
    fwhms = [b['full_width_half_max'] for b in eo_bands]
    esun = [b['solar_illumination'] for b in eo_bands]

    sun_elev = stac['properties']['view:sun_elevation']
    sun_azim = stac['properties']['view:sun_azimuth']
    off_nadir = stac['properties']['view:off_nadir']
    inc_angle = stac['properties']['view:incidence_angle']
    view_azim = stac['properties']['view:azimuth']
    
    stac_gsd = stac['properties'].get('gsd', None)
    nodata_val = assets['Cloud optimized GeoTiff']['raster:bands'][0].get('nodata', -9999)

    radiance_tif_href = assets['Cloud optimized GeoTiff']['href']
    tiff_name = os.path.basename(radiance_tif_href)
    radiance_tifs = list(Path(base_dir).rglob(tiff_name))
    radiance_tif = str(radiance_tifs[0]) if radiance_tifs else resolve_path(radiance_tif_href)

    dmask_name = os.path.basename(assets['Data Mask']['href'])
    dmasks = list(Path(base_dir).rglob(dmask_name))
    data_mask_tif = str(dmasks[0]) if dmasks else resolve_path(assets['Data Mask']['href'])
    
    qmask_name = os.path.basename(assets['Pixel Quality Mask']['href'])
    qmasks = list(Path(base_dir).rglob(qmask_name))
    pixel_quality_tif = str(qmasks[0]) if qmasks else resolve_path(assets['Pixel Quality Mask']['href'])
    
    with rasterio.open(radiance_tif) as src:
        b = src.bounds
        if src.crs != CRS.from_epsg(4326):
            transformer = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
            left, bottom = transformer.transform(b.left, b.bottom)
            right, top = transformer.transform(b.right, b.top)
            bounds_lonlat = [(left, bottom), (left, top), (right, top), (right, bottom)]
        else:
            bounds_lonlat = [(b.left, b.bottom), (b.left, b.top), (b.right, b.top), (b.right, b.bottom)]
        
        if stac_gsd is None:
            stac_gsd = (abs(src.transform.a) + abs(src.transform.e)) / 2.0

    return {
        'id': stac['id'],
        'json_path': json_path,
        'time': acq_time,
        'epsg': epsg,
        'platform': stac['properties']['platform'],
        'radiance_tif': radiance_tif,
        'data_mask_tif': data_mask_tif,
        'pixel_quality_tif': pixel_quality_tif,
        'bounds_lonlat': bounds_lonlat,
        'gsd': stac_gsd,
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

def process_dragonette_native_stack(target_location):
    output_file = os.path.join(OUTPUT_DIR, f"Dragonette_Native_Stack_{target_location}.h5")
    if os.path.exists(output_file):
        print(f"Skipping Dragonette native stacking for {target_location}: {output_file} already exists.")
        return

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
        print(f"Warning: Dragonette directory {location_dir} does not exist for {target_location}. Skipping.")
        return None
        
    root_path = Path(location_dir)
    json_files = list(root_path.rglob("*.json"))
    json_files = [f for f in json_files if "catalog" not in f.name.lower()]
    valid_json_files = json_files
    
    if not valid_json_files:
        print(f"Warning: No valid Wyvern STAC JSON files found in {location_dir} for {target_location}. Skipping.")
        return None

    raw_scenes = []
    for j_path in valid_json_files:
        try:
            scene_data = parse_wyvern_stac(str(j_path))
            lons = [pt[0] for pt in scene_data['bounds_lonlat']]
            lats = [pt[1] for pt in scene_data['bounds_lonlat']]
            scene_bbox = [min(lons), min(lats), max(lons), max(lats)]
            
            if intersects(scene_bbox, roi_bbox):
                raw_scenes.append(scene_data)
                
        except Exception as e:
            print(f"Warning: Skipping {j_path.name} due to parsing error: {e}")

    if not raw_scenes:
        print(f"Warning: No scenes intersect the bounding box for {target_location}. Skipping.")
        return None

    raw_scenes.sort(key=lambda x: x['time'])
    n_times = len(raw_scenes)
    print(f"Discovered {n_times} distinct Dragonette scenes intersecting the ROI.")

    avg_gsd = round(sum(s['gsd'] for s in raw_scenes) / n_times, 2)
    print(f"Calculated average native GSD: {avg_gsd}m")

    # Establish MGRS-Aligned Grid from Config for 30m
    width_30 = loc_data['MGRS_WIDTH']
    height_30 = loc_data['MGRS_HEIGHT']
    crs_wkt = f"EPSG:{loc_data['MGRS_EPSG']}"
    ul_x = loc_data['MGRS_UL_X']
    ul_y = loc_data['MGRS_UL_Y']
    target_gsd_30 = loc_data.get('TARGET_GSD', 30.0)
    
    tf_target_30 = Affine.translation(ul_x, ul_y) * Affine.scale(target_gsd_30, -target_gsd_30)
    
    extent_x = width_30 * target_gsd_30
    extent_y = height_30 * target_gsd_30
    
    target_gsd_5 = avg_gsd
    width_5 = math.ceil(extent_x / target_gsd_5)
    height_5 = math.ceil(extent_y / target_gsd_5)
    tf_target_5 = Affine.translation(ul_x, ul_y) * Affine.scale(target_gsd_5, -target_gsd_5)
    
    print(f"MGRS Target Grid 30m: {width_30}x{height_30} {target_gsd_30}m pixels (CRS: {crs_wkt})")
    print(f"MGRS Target Grid {avg_gsd}m: {width_5}x{height_5} pixels (CRS: {crs_wkt})")
    
    base_wv = raw_scenes[0]['wavelengths']
    for scn in raw_scenes:
        if len(scn['wavelengths']) != len(base_wv) or not np.allclose(scn['wavelengths'], base_wv, atol=1.0):
            raise ValueError(f"CRITICAL SPECTRAL MISMATCH: Dataset {scn['id']} wavelength array does not match stack standard.")
    n_bands = len(base_wv)
    

    with h5py.File(output_file, 'w') as out_h5:
        grp_wyvern_5 = out_h5.create_group("HDFEOS/GRIDS/DRAGONETTE/Data Fields")
        grp_wyvern_30 = out_h5.create_group("HDFEOS/GRIDS/DRAGONETTE_30M/Data Fields")
        
        rad_nodata = -9999
        
        ds_rad_5 = grp_wyvern_5.create_dataset("surface_reflectance", shape=(n_times, n_bands, height_5, width_5), dtype='int16', compression="gzip", compression_opts=5, shuffle=True, fillvalue=rad_nodata)
        ds_dmask_5 = grp_wyvern_5.create_dataset("data_mask", shape=(n_times, 4, height_5, width_5), dtype='uint8', compression="gzip", compression_opts=5, shuffle=True, fillvalue=255)
        ds_qmask_5 = grp_wyvern_5.create_dataset("pixel_quality_mask", shape=(n_times, n_bands, height_5, width_5), dtype='uint8', compression="gzip", shuffle=True, compression_opts=5, fillvalue=255)
        ds_vis_5 = grp_wyvern_5.create_dataset("ortho_visual", shape=(n_times, 4, height_5, width_5), dtype='uint8', compression="gzip", compression_opts=5, shuffle=True, fillvalue=0)
        ds_common_mask_5 = grp_wyvern_5.create_dataset("common_mask", shape=(n_times, height_5, width_5), dtype=bool, compression="gzip", compression_opts=5, shuffle=True, fillvalue=False)
        
        ds_rad_30 = grp_wyvern_30.create_dataset("surface_reflectance", shape=(n_times, n_bands, height_30, width_30), dtype='int16', compression="gzip", compression_opts=5, shuffle=True, fillvalue=rad_nodata)
        ds_dmask_30 = grp_wyvern_30.create_dataset("data_mask", shape=(n_times, 4, height_30, width_30), dtype='uint8', compression="gzip", compression_opts=5, shuffle=True, fillvalue=255)
        ds_qmask_30 = grp_wyvern_30.create_dataset("pixel_quality_mask", shape=(n_times, n_bands, height_30, width_30), dtype='uint8', compression="gzip", shuffle=True, compression_opts=5, fillvalue=255)
        ds_vis_30 = grp_wyvern_30.create_dataset("ortho_visual", shape=(n_times, 4, height_30, width_30), dtype='uint8', compression="gzip", compression_opts=5, shuffle=True, fillvalue=0)
        ds_common_mask_30 = grp_wyvern_30.create_dataset("common_mask", shape=(n_times, height_30, width_30), dtype=bool, compression="gzip", compression_opts=5, shuffle=True, fillvalue=False)

        good_wavelengths_array = []
        acq_time_array = np.zeros(n_times, dtype='float64')
        platform_array = []
        sun_elev_array = np.zeros(n_times, dtype='float64')
        sun_azim_array = np.zeros(n_times, dtype='float64')
        off_nadir_array = np.zeros(n_times, dtype='float64')
        inc_angle_array = np.zeros(n_times, dtype='float64')
        view_azim_array = np.zeros(n_times, dtype='float64')
        solar_illum_array = np.zeros((n_times, n_bands), dtype='float32')

        gdal_transform_5 = [tf_target_5.c, tf_target_5.a, tf_target_5.b, tf_target_5.f, tf_target_5.d, tf_target_5.e]
        gdal_transform_30 = [tf_target_30.c, tf_target_30.a, tf_target_30.b, tf_target_30.f, tf_target_30.d, tf_target_30.e]

        ds_rad_5.attrs['spatial_ref'] = crs_wkt
        ds_rad_5.attrs['GeoTransform'] = gdal_transform_5
        ds_rad_30.attrs['spatial_ref'] = crs_wkt
        ds_rad_30.attrs['GeoTransform'] = gdal_transform_30

        for t_idx, scene in enumerate(raw_scenes):
            print(f"  [Pass {t_idx+1}/{n_times}] Assimilating Swath: {scene['time'].isoformat()}...")
            
            src_rad_nodata = scene['nodata']
            
            with rasterio.open(scene['radiance_tif']) as src:
                src_arr_float = src.read().astype(np.float32)
                valid_src = src_arr_float[src_arr_float != src_rad_nodata]
                if len(valid_src) > 0 and np.nanmax(valid_src) > 10.0:
                    src_arr_float[src_arr_float != src_rad_nodata] /= 10000.0

                incoming_rad_5m_float = np.full((n_bands, height_5, width_5), np.nan, dtype='float32')
                reproject(
                    source=src_arr_float,
                    destination=incoming_rad_5m_float,
                    src_transform=src.transform, src_crs=src.crs, dst_transform=tf_target_5, dst_crs=crs_wkt,
                    resampling=Resampling.nearest,
                    src_nodata=src_rad_nodata, dst_nodata=np.nan
                )
                
                scaled_5m = scale_reflectance_for_storage(incoming_rad_5m_float, scale_factor=10000.0, fill_value_in=np.nan, fill_value_out=rad_nodata)
                ds_rad_5[t_idx, ...] = scaled_5m

                incoming_rad_30m_float = np.full((n_bands, height_30, width_30), np.nan, dtype='float32')
                reproject(
                    source=incoming_rad_5m_float,
                    destination=incoming_rad_30m_float,
                    src_transform=tf_target_5, src_crs=crs_wkt, dst_transform=tf_target_30, dst_crs=crs_wkt,
                    resampling=Resampling.average,
                    src_nodata=np.nan, dst_nodata=np.nan
                )
                
                scaled_30m = scale_reflectance_for_storage(incoming_rad_30m_float, scale_factor=10000.0, fill_value_in=np.nan, fill_value_out=rad_nodata)
                ds_rad_30[t_idx, ...] = scaled_30m

            with rasterio.open(scene['data_mask_tif']) as src:
                incoming_dmask_5m = np.full((4, height_5, width_5), 255, dtype='uint8')
                reproject(
                    source=rasterio.band(src, list(range(1, src.count + 1))),
                    destination=incoming_dmask_5m,
                    src_transform=src.transform, src_crs=src.crs, dst_transform=tf_target_5, dst_crs=crs_wkt,
                    resampling=Resampling.nearest,
                    src_nodata=255, dst_nodata=255
                )
                ds_dmask_5[t_idx, ...] = incoming_dmask_5m
                
                incoming_dmask_30m = np.full((4, height_30, width_30), 255, dtype='uint8')
                reproject(
                    source=incoming_dmask_5m,
                    destination=incoming_dmask_30m,
                    src_transform=tf_target_5, src_crs=crs_wkt, dst_transform=tf_target_30, dst_crs=crs_wkt,
                    resampling=Resampling.nearest,
                    src_nodata=255, dst_nodata=255
                )
                ds_dmask_30[t_idx, ...] = incoming_dmask_30m

            with rasterio.open(scene['pixel_quality_tif']) as src:
                incoming_qmask_5m = np.full((n_bands, height_5, width_5), 255, dtype='uint8')
                reproject(
                    source=rasterio.band(src, list(range(1, src.count + 1))),
                    destination=incoming_qmask_5m,
                    src_transform=src.transform, src_crs=src.crs, dst_transform=tf_target_5, dst_crs=crs_wkt,
                    resampling=Resampling.nearest,
                    src_nodata=255, dst_nodata=255
                )
                ds_qmask_5[t_idx, ...] = incoming_qmask_5m
                
                incoming_qmask_30m = np.full((n_bands, height_30, width_30), 255, dtype='uint8')
                reproject(
                    source=incoming_qmask_5m,
                    destination=incoming_qmask_30m,
                    src_transform=tf_target_5, src_crs=crs_wkt, dst_transform=tf_target_30, dst_crs=crs_wkt,
                    resampling=Resampling.nearest,
                    src_nodata=255, dst_nodata=255
                )
                ds_qmask_30[t_idx, ...] = incoming_qmask_30m

            valid_pixels_per_band = np.sum(incoming_qmask_5m != 255, axis=(1, 2))
            interpolated_per_band = np.sum(incoming_qmask_5m == 1, axis=(1, 2))
            with np.errstate(divide='ignore', invalid='ignore'):
                interp_ratio = interpolated_per_band / valid_pixels_per_band
            frame_good_wv = (valid_pixels_per_band > 0) & (np.nan_to_num(interp_ratio) < 0.10)
            good_wavelengths_array.append(frame_good_wv)

            qa_clear_5 = incoming_dmask_5m[0, :, :]
            qa_cloud_5 = incoming_dmask_5m[1, :, :]
            qa_haze_5 = incoming_dmask_5m[2, :, :]
            qa_shadow_5 = incoming_dmask_5m[3, :, :]
            invalid_mask_5 = (qa_cloud_5 == 1) | (qa_haze_5 == 1) | (qa_shadow_5 == 1) | (qa_clear_5 == 255)
            ds_common_mask_5[t_idx, ...] = invalid_mask_5

            qa_clear_30 = incoming_dmask_30m[0, :, :]
            qa_cloud_30 = incoming_dmask_30m[1, :, :]
            qa_haze_30 = incoming_dmask_30m[2, :, :]
            qa_shadow_30 = incoming_dmask_30m[3, :, :]
            invalid_mask_30 = (qa_cloud_30 == 1) | (qa_haze_30 == 1) | (qa_shadow_30 == 1) | (qa_clear_30 == 255)
            ds_common_mask_30[t_idx, ...] = invalid_mask_30

            r_idx = np.argmin(np.abs(base_wv - 650.0))
            g_idx = np.argmin(np.abs(base_wv - 550.0))
            b_idx = np.argmin(np.abs(base_wv - 470.0))
            
            rgba_img_5 = sc.generate_rgba_image(
                r_band=scaled_5m[r_idx, :, :],
                g_band=scaled_5m[g_idx, :, :],
                b_band=scaled_5m[b_idx, :, :],
                nodata=rad_nodata
            )
            ds_vis_5[t_idx, ...] = np.transpose(rgba_img_5, (2, 0, 1))

            rgba_img_30 = sc.generate_rgba_image(
                r_band=scaled_30m[r_idx, :, :],
                g_band=scaled_30m[g_idx, :, :],
                b_band=scaled_30m[b_idx, :, :],
                nodata=rad_nodata
            )
            ds_vis_30[t_idx, ...] = np.transpose(rgba_img_30, (2, 0, 1))
            
            acq_time_array[t_idx] = scene['time'].timestamp()
            platform_array.append(scene['platform'])
            sun_elev_array[t_idx] = scene['sun_elevation']
            sun_azim_array[t_idx] = scene['sun_azimuth']
            off_nadir_array[t_idx] = scene['view_off_nadir']
            inc_angle_array[t_idx] = scene['view_incidence_angle']
            view_azim_array[t_idx] = scene['view_azimuth']
            solar_illum_array[t_idx, :] = scene['solar_illumination']

        dt_str = h5py.string_dtype(encoding='ascii')
        for dset, tf, sz in [(ds_rad_5, gdal_transform_5, "5m"), (ds_rad_30, gdal_transform_30, "30m")]:
            add_reflectance_attributes(dset, scale_to_float=0.0001, fill_value=rad_nodata)
            dset.attrs['spatial_ref'] = crs_wkt
            dset.attrs['GeoTransform'] = np.array(tf, dtype='float64')
            dset.attrs['wavelengths'] = base_wv
            dset.attrs['fwhm'] = scene['fwhm']
            dset.attrs['solar_illumination'] = solar_illum_array
            dset.attrs['acquisition_time'] = acq_time_array
            dset.attrs.create('spacecraft_id', data=np.array(platform_array, dtype=dt_str))
            dset.attrs['sun_elevation'] = sun_elev_array
            dset.attrs['sun_azimuth'] = sun_azim_array
            dset.attrs['view_off_nadir'] = off_nadir_array
            dset.attrs['view_incidence_angle'] = inc_angle_array
            dset.attrs['view_azimuth'] = view_azim_array
            if len(good_wavelengths_array) == n_times:
                dset.attrs['all_good_wavelengths'] = np.array(good_wavelengths_array, dtype=bool)
                
        for grp, tf in [(grp_wyvern_5, gdal_transform_5), (grp_wyvern_30, gdal_transform_30)]:
            for name in ["data_mask", "pixel_quality_mask", "ortho_visual", "common_mask"]:
                grp[name].attrs['spatial_ref'] = crs_wkt
                grp[name].attrs['GeoTransform'] = np.array(tf, dtype='float64')
            grp["common_mask"].attrs['description'] = "True = Invalid/Masked, False = Valid. Computed from Wyvern QA Data Mask (Cloud/Haze/Shadow)."

    print(f"\nTensor Synthesis Complete. Native Stack Stored at: {output_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--location", type=str, default=LOCATION, help="Target location prefix")
    args = parser.parse_args()
    process_dragonette_native_stack(args.location)
