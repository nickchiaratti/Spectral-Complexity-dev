import os
import json
import h5py
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import Affine
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import sys

# Import centralized ODL generator
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import hdfeos_odl

# --- Configuration ---
LOCATION='CentralGreece'
TIME_THRESHOLD_SECONDS = 300  # Group acquisitions within 2 minutes into the same temporal pass
SOURCE_DIR = "C:/satelliteImagery/enmap"
OUTPUT_DIR = SOURCE_DIR
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# RGB Wavelength Targets for Ortho Visual generation
TARGET_RED_NM = 680.0
TARGET_GREEN_NM = 540.0
TARGET_BLUE_NM = 480.0



def parse_enmap_stac(json_path):
    """Extracts metrology and file paths from the EnMAP STAC JSON."""
    with open(json_path, 'r') as f:
        stac = json.load(f)
    
    dt_str = stac['properties']['datetime']
    acq_time = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
    
    epsg = 4326 # We will reproject all EnMAP EPSG (UTM) into WGS84 for the unified grid
    
    assets = stac['assets']
    base_dir = os.path.dirname(json_path)
    
    def resolve_path(href):
        return os.path.join(base_dir, os.path.basename(href))
    
    # Extract band metadata from the image asset
    eo_bands = assets.get('image', {}).get('eo:bands', [])
    wavelengths = []
    fwhms = []
    if eo_bands:
        wavelengths = [b.get('eo:center_wavelength') if 'eo:center_wavelength' in b else b.get('center_wavelength', 0.0) for b in eo_bands]
        fwhms = [b.get('eo:full_width_half_max') if 'eo:full_width_half_max' in b else b.get('full_width_half_max', 0.0) for b in eo_bands]
    
    # Locate METADATA.XML for band metadata fallback
    xml_filename = stac['id'] + "-METADATA.XML"
    xml_path = os.path.join(base_dir, xml_filename)
    
    if not any(w > 0 for w in wavelengths) and os.path.exists(xml_path):
        try:
            import xml.etree.ElementTree as ET
            tree_temp = ET.parse(xml_path)
            root_temp = tree_temp.getroot()
            xml_wvs = [float(e.text) for e in root_temp.iter() if e.tag.lower().endswith('wavelengthcenterofband') and e.text]
            xml_fwhms = [float(e.text) for e in root_temp.iter() if e.tag.lower().endswith('fwhmofband') and e.text]
            if len(xml_wvs) == 224:
                wavelengths = xml_wvs
                fwhms = xml_fwhms if len(xml_fwhms) == 224 else [0.0] * 224
        except Exception as e:
            print(f"Warning: Failed to parse band wavelengths from {xml_path}: {e}")

    if not wavelengths or not any(w > 0 for w in wavelengths):
        wavelengths = [0.0] * 224
        fwhms = [0.0] * 224
    
    sun_elev = stac['properties'].get('view:sun_elevation', 0.0)
    sun_azim = stac['properties'].get('view:sun_azimuth', 0.0)
    off_nadir = float(stac['properties'].get('enmap:acrossOffNadirAngle', 0.0))
    inc_angle = 0.0 # EnMAP JSON doesn't directly provide incidence angle
    view_azim = float(stac['properties'].get('enmap:sceneAzimuthAngle', 0.0))
    
    # Extract Fill Value
    nodata_val = -32768
    if 'enmap:nodata' in stac['properties']:
        nodata_val = float(stac['properties']['enmap:nodata'])

    # Find the quality classes tif
    quality_classes_tif = ""
    for asset_key, asset_val in assets.items():
        if "QUALITY_CLASSES_COG.TIF" in asset_val.get('href', ''):
            quality_classes_tif = resolve_path(asset_val['href'])
            break

    image_tif = resolve_path(assets['image']['href'])

    return {
        'id': stac['id'],
        'json_path': json_path,
        'time': acq_time,
        'platform': stac['properties'].get('platform', 'enmap'),
        'reflectance_tif': image_tif,
        'quality_classes_tif': quality_classes_tif,
        'wavelengths': np.array(wavelengths, dtype=np.float32),
        'fwhm': np.array(fwhms, dtype=np.float32),
        'sun_elevation': sun_elev,
        'sun_azimuth': sun_azim,
        'view_off_nadir': off_nadir,
        'view_incidence_angle': inc_angle,
        'view_azimuth': view_azim,
        'nodata': nodata_val,
        'stac_dict': stac
    }

def calculate_global_geographic_grid(scenes):
    """
    Computes a strict bounding box union aligning to the native angular resolution 
    (Lon/Lat) of the source data.
    """
    global_min_x, global_min_y = float('inf'), float('inf')
    global_max_x, global_max_y = float('-inf'), float('-inf')
    
    # Since EnMAP comes in UTM, we want to establish a uniform WGS84 grid.
    # 30m in degrees is roughly 0.00026949
    x_res = 0.00026949
    y_res = 0.00026949

    for scene in scenes:
        try:
            with rasterio.open(scene['reflectance_tif']) as src:
                # We get bounds in the native CRS, then reproject bounds to WGS84
                bounds = src.bounds
                from rasterio.warp import transform_bounds
                wgs84_bounds = transform_bounds(src.crs, 'EPSG:4326', *bounds)
                global_min_x = min(global_min_x, wgs84_bounds[0])
                global_min_y = min(global_min_y, wgs84_bounds[1])
                global_max_x = max(global_max_x, wgs84_bounds[2])
                global_max_y = max(global_max_y, wgs84_bounds[3])
        except rasterio.errors.RasterioIOError:
            raise RuntimeError(f"CRITICAL: Failed to read {scene['reflectance_tif']}. It may be an HTML login page instead of a valid TIFF due to failed authentication during download.")

    width = int(np.ceil((global_max_x - global_min_x) / x_res))
    height = int(np.ceil((global_max_y - global_min_y) / y_res))

    target_transform = Affine.translation(global_min_x, global_max_y) * Affine.scale(x_res, -y_res)
    
    return target_transform, width, height, (global_min_x, global_max_y), (global_max_x, global_min_y)

def process_native_stack(target_location):
    print(f"Discovering EnMAP STAC collections for location: {target_location}...")
    
    location_dir = os.path.join(SOURCE_DIR, f"{target_location}_SourceData")
    if not os.path.exists(location_dir):
        raise FileNotFoundError(f"Location directory not found: {location_dir}")

    # 1. Recursive Data Discovery
    root_path = Path(location_dir)
    json_files = list(root_path.rglob("*.json"))
    json_files = [f for f in json_files if "catalog" not in f.name.lower()]
    
    # Filter out top-level downloaded JSONs by ensuring the json is adjacent to TIFF data
    valid_json_files = [f for f in json_files if any(f.parent.glob("*.TIF")) or any(f.parent.glob("*.tif"))]
    
    if not valid_json_files:
        raise FileNotFoundError(f"CRITICAL: No valid EnMAP STAC JSON files with adjacent TIFFs found in {location_dir}")

    raw_scenes = []
    for j_path in valid_json_files:
        try:
            scene_data = parse_enmap_stac(str(j_path))
            if not os.path.exists(scene_data['reflectance_tif']):
                print(f"Warning: Skipping item {scene_data['id']} - missing primary reflectance TIFF.")
                continue
            raw_scenes.append(scene_data)
        except Exception as e:
            print(f"Warning: Skipping {j_path.name} due to discovery error: {e}")
            continue

    if not raw_scenes:
        raise FileNotFoundError(f"CRITICAL: No complete EnMAP scenes with valid reflectance TIFFs found in {location_dir}")

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

    # 3. Establish Universal Grid & Spectral Alignment
    tf_target, width, height, ul_coords, lr_coords = calculate_global_geographic_grid(raw_scenes)
    print(f"Global Geographic Grid: {width}x{height} angular pixels")

    base_wv = raw_scenes[0]['wavelengths']
    n_times = len(grouped_scenes)
    n_bands = len(base_wv)

    output_file = os.path.join(location_dir, f"EnMAP_Native_Stack_{target_location}.h5")

    # 4. HDF5 Tensor Construction
    with h5py.File(output_file, 'w') as out_h5:
        # Establish structural hierarchy
        grp_enmap = out_h5.create_group("HDFEOS/GRIDS/ENMAP/Data Fields")
        meta_grp = out_h5.create_group("METADATA")
        info_grp = out_h5.create_group("HDFEOS INFORMATION")
        
        # Dataset Registration (Shape/Type/Rank/Dimensions)
        datasets_info = [
            ("surface_reflectance", np.dtype('float32'), 4, ["Time", "Band", "YDim", "XDim"]),
            ("quality_classes", np.dtype('uint8'), 3, ["Time", "YDim", "XDim"]), 
            ("ortho_visual", np.dtype('float32'), 4, ["Time", "RGBBand", "YDim", "XDim"]),
            ("wavelength", np.dtype('float32'), 2, ["Time", "Band"]),
            ("fwhm", np.dtype('float32'), 2, ["Time", "Band"])
        ]
        
        # Allocate Datasets in BSQ Layout
        rad_nodata = raw_scenes[0]['nodata']
        ds_rad = grp_enmap.create_dataset("surface_reflectance", shape=(n_times, n_bands, height, width), dtype='float32', compression="gzip", fillvalue=rad_nodata)
        ds_qmask = grp_enmap.create_dataset("quality_classes", shape=(n_times, height, width), dtype='uint8', compression="gzip", fillvalue=255)
        ds_vis = grp_enmap.create_dataset("ortho_visual", shape=(n_times, 3, height, width), dtype='float32', compression="gzip", fillvalue=rad_nodata)
        ds_wv = grp_enmap.create_dataset("wavelength", shape=(n_times, n_bands), dtype='float32', compression="gzip")
        ds_fwhm = grp_enmap.create_dataset("fwhm", shape=(n_times, n_bands), dtype='float32', compression="gzip")
        
        acq_time_array = np.zeros(n_times, dtype='float64')
        platform_array = []
        sun_elev_array = np.zeros(n_times, dtype='float64')
        sun_azim_array = np.zeros(n_times, dtype='float64')
        off_nadir_array = np.zeros(n_times, dtype='float64')
        inc_angle_array = np.zeros(n_times, dtype='float64')
        view_azim_array = np.zeros(n_times, dtype='float64')

        gdal_transform = [tf_target.c, tf_target.a, tf_target.b, tf_target.f, tf_target.d, tf_target.e]
        crs_wkt = "EPSG:4326"

        for t_idx, group in enumerate(grouped_scenes):
            print(f"  Processing Temporal Pass {t_idx+1}/{n_times} ({group[0]['time'].isoformat()})...")
            
            canvas_rad = np.full((n_bands, height, width), rad_nodata, dtype='float32')
            canvas_qmask = np.full((1, height, width), 255, dtype='uint8')

            for scene in group:
                try:
                    with rasterio.open(scene['reflectance_tif']) as src:
                        if t_idx == 0 and scene == group[0] and base_wv[0] == 0.0:
                            # Initialize dummy wavelengths if not provided by STAC metadata
                            group[0]['wavelengths'] = np.zeros(n_bands, dtype=np.float32)
                            group[0]['fwhm'] = np.zeros(n_bands, dtype=np.float32)
                            base_wv = group[0]['wavelengths']
                        reproject(
                            source=rasterio.band(src, list(range(1, src.count + 1))),
                            destination=canvas_rad,
                            src_transform=src.transform, src_crs=src.crs, dst_transform=tf_target, dst_crs=crs_wkt,
                            resampling=Resampling.nearest,
                            src_nodata=rad_nodata, dst_nodata=rad_nodata
                        )
                except rasterio.errors.RasterioIOError:
                    raise RuntimeError(f"CRITICAL: Failed to read {scene['reflectance_tif']}. It may be an HTML login page instead of a valid TIFF due to failed authentication during download.")
                
                if scene['quality_classes_tif'] and os.path.exists(scene['quality_classes_tif']):
                    try:
                        with rasterio.open(scene['quality_classes_tif']) as src:
                            reproject(
                                source=rasterio.band(src, 1),
                                destination=canvas_qmask,
                                src_transform=src.transform, src_crs=src.crs, dst_transform=tf_target, dst_crs=crs_wkt,
                                resampling=Resampling.nearest,
                                src_nodata=255, dst_nodata=255
                            )
                    except rasterio.errors.RasterioIOError:
                        pass # Ignore if quality classes is also corrupt
            
            if np.all(canvas_rad == rad_nodata):
                raise ValueError(f"CRITICAL: Data assimilation failed for temporal pass {t_idx}. Spatial frame is completely NoData.")

            ds_rad[t_idx, ...] = canvas_rad
            ds_qmask[t_idx, ...] = canvas_qmask[0, ...]
            
            acq_time_array[t_idx] = group[0]['time'].timestamp()
            platform_array.append(group[0]['platform'])
            sun_elev_array[t_idx] = group[0]['sun_elevation']
            sun_azim_array[t_idx] = group[0]['sun_azimuth']
            off_nadir_array[t_idx] = group[0]['view_off_nadir']
            inc_angle_array[t_idx] = group[0]['view_incidence_angle']
            view_azim_array[t_idx] = group[0]['view_azimuth']
            
            ds_wv[t_idx, :] = group[0]['wavelengths']
            ds_fwhm[t_idx, :] = group[0]['fwhm']

            meta_grp.attrs[f"frame_{t_idx}_json"] = json.dumps(group[0]['stac_dict'])

        print("  Extracting 'ortho_visual' specific bands...")
        r_idx = int(np.argmin(np.abs(base_wv - TARGET_RED_NM)))
        g_idx = int(np.argmin(np.abs(base_wv - TARGET_GREEN_NM)))
        b_idx = int(np.argmin(np.abs(base_wv - TARGET_BLUE_NM)))
        
        for t_idx in tqdm(range(n_times), desc="  Writing visual channels"):
            ds_vis[t_idx, 0, :, :] = ds_rad[t_idx, r_idx, :, :]
            ds_vis[t_idx, 1, :, :] = ds_rad[t_idx, g_idx, :, :]
            ds_vis[t_idx, 2, :, :] = ds_rad[t_idx, b_idx, :, :]

        platform_utf8 = np.array(platform_array, dtype='S20')
        
        for dataset in [ds_rad, ds_qmask, ds_vis]:
            dataset.attrs['spatial_ref'] = crs_wkt
            dataset.attrs['GeoTransform'] = np.array(gdal_transform, dtype='float64')

        ds_wv.attrs['unit'] = 'nm'
        ds_fwhm.attrs['unit'] = 'nm'

        ds_rad.attrs["acquisition_time"] = acq_time_array
        ds_rad.attrs["sun_elevation"] = sun_elev_array
        ds_rad.attrs["sun_azimuth"] = sun_azim_array
        ds_rad.attrs["view_off_nadir"] = off_nadir_array
        ds_rad.attrs["view_incidence_angle"] = inc_angle_array
        ds_rad.attrs["view_azimuth"] = view_azim_array
        ds_rad.attrs["spacecraft_id"] = platform_utf8
        ds_rad.attrs["unit"] = "reflectance"

        ds_qmask.attrs["description"] = "EnMAP L2A Quality Classes Mask"

        odl_grid = hdfeos_odl.generate_dynamic_odl_grid_string("ENMAP", width, height, tf_target, "HE5_GCTP_GEO", 0, (0,)*13, datasets_info, n_times, n_bands)
        struct_meta = f"GROUP=SwathStructure\nEND_GROUP=SwathStructure\nGROUP=GridStructure\n{odl_grid}\nEND_GROUP=GridStructure\n"
        dt_str = h5py.string_dtype(encoding='ascii')
        info_grp.create_dataset("StructMetadata.0", (1,), dtype=dt_str, data=struct_meta)

    print(f"\nTensor Synthesis Complete. Stored at: {output_file}")

if __name__ == "__main__":
    process_native_stack(LOCATION)
