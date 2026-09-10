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
LOCATION = 'LakeFire2024'
TIME_THRESHOLD_SECONDS = 120  # Group acquisitions within 2 minutes into the same temporal pass
SOURCE_DIR = "C:/satelliteImagery/enmap"
OUTPUT_DIR = SOURCE_DIR

TARGET_RED_NM = 680.0
TARGET_GREEN_NM = 540.0
TARGET_BLUE_NM = 480.0

def apply_atmospheric_correction(radiance_cube, group_metadata):
    """
    Placeholder for SICOR / ISOFIT atmospheric correction.
    
    The radiance_cube provided here is in pure sensor geometry (along-track stacked chunks).
    This allows the radiative transfer model to process the entire datatake seamlessly, 
    avoiding tile boundaries.
    """
    print("    -> [STUB] Running SICOR/ISOFIT Atmospheric Correction on Merged Pass...")
    
    # Example scaling (simulating conversion from radiance to reflectance)
    # This must be replaced with the actual SICOR/ISOFIT inversion.
    boa_reflectance = radiance_cube * 0.5  
    
    return boa_reflectance


def parse_enmap_l1b_scene(json_path):
    """
    Extracts L1B metadata. L1B data usually splits VNIR and SWIR.
    """
    with open(json_path, 'r') as f:
        stac = json.load(f)
    
    dt_str = stac['properties']['datetime']
    acq_time = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
    
    assets = stac['assets']
    base_dir = os.path.dirname(json_path)
    
    def resolve_path(href):
        return os.path.join(base_dir, os.path.basename(href))
    
    # In EnMAP L1B STAC, assets might be named differently (e.g., vnir_image, swir_image)
    # We will look for standard L1B TIFFs.
    vnir_tif = None
    swir_tif = None
    
    for key, asset in assets.items():
        href = asset.get('href', '').upper()
        if 'VNIR' in href and href.endswith('.TIF') and 'QUICKLOOK' not in href:
            vnir_tif = resolve_path(asset['href'])
        elif 'SWIR' in href and href.endswith('.TIF') and 'QUICKLOOK' not in href:
            swir_tif = resolve_path(asset['href'])
            
    # Fallback if standard keys aren't matched
    if not vnir_tif and 'vnir_image' in assets:
        vnir_tif = resolve_path(assets['vnir_image']['href'])
    if not swir_tif and 'swir_image' in assets:
        swir_tif = resolve_path(assets['swir_image']['href'])

    # Wavelength extraction (Simplified for L1B)
    wavelengths = []
    fwhms = []
    
    # Look for metadata XML
    xml_filename = stac['id'] + "-METADATA.XML"
    xml_path = os.path.join(base_dir, xml_filename)
    xml_coords = []
    
    if os.path.exists(xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        xml_wvs = [float(e.text) for e in root.iter() if e.tag.lower().endswith('wavelengthcenterofband') and e.text]
        xml_fwhms = [float(e.text) for e in root.iter() if e.tag.lower().endswith('fwhmofband') and e.text]
        if len(xml_wvs) == 224:
            wavelengths = xml_wvs
            fwhms = xml_fwhms
            
        poly_points = root.findall('.//base/spatialCoverage/boundingPolygon/point')
        for pt in poly_points:
            frame = pt.findtext('frame')
            if frame != 'center':
                lat = float(pt.findtext('latitude'))
                lon = float(pt.findtext('longitude'))
                xml_coords.append((lon, lat))

    if not wavelengths:
        # Default fallback
        wavelengths = [0.0] * 224
        fwhms = [0.0] * 224

    if not xml_coords:
        bbox = stac['bbox']
        xml_coords = [(bbox[0], bbox[1]), (bbox[2], bbox[1]), (bbox[2], bbox[3]), (bbox[0], bbox[3])]

    sun_elev = stac['properties'].get('view:sun_elevation', 0.0)
    sun_azim = stac['properties'].get('view:sun_azimuth', 0.0)
    off_nadir = float(stac['properties'].get('enmap:acrossOffNadirAngle', 0.0))
    view_azim = float(stac['properties'].get('enmap:sceneAzimuthAngle', 0.0))
    nodata_val = float(stac['properties'].get('enmap:nodata', -32768))

    return {
        'id': stac['id'],
        'time': acq_time,
        'platform': stac['properties'].get('platform', 'enmap'),
        'vnir_tif': vnir_tif,
        'swir_tif': swir_tif,
        'wavelengths': np.array(wavelengths, dtype=np.float32),
        'fwhm': np.array(fwhms, dtype=np.float32),
        'sun_elevation': sun_elev,
        'sun_azimuth': sun_azim,
        'view_off_nadir': off_nadir,
        'view_azimuth': view_azim,
        'nodata': nodata_val,
        'stac_dict': stac,
        'bounds_lonlat': xml_coords
    }

def intersects(bbox1, bbox2):
    return not (bbox1[2] < bbox2[0] or bbox1[0] > bbox2[2] or
                bbox1[3] < bbox2[1] or bbox1[1] > bbox2[3])

def process_mgrs_l1b_stack(target_location):
    print(f"Discovering EnMAP L1B collections for location: {target_location}...")
    import yaml
    
    script_dir = Path(__file__).resolve().parent
    config_path = os.path.join(script_dir.parent, "locations_config.yaml")
        
    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)
        
    loc_data = config_data.get("locations", {}).get(target_location)
    if not loc_data:
        raise ValueError(f"CRITICAL: Location {target_location} not found.")
        
    roi_bbox = [loc_data["ROI_LON_MIN"], loc_data["ROI_LAT_MIN"], loc_data["ROI_LON_MAX"], loc_data["ROI_LAT_MAX"]]
    source_cache = loc_data.get("SOURCE_CACHE") or target_location
    location_dir = os.path.join(SOURCE_DIR, f"{source_cache}_SourceData")

    root_path = Path(location_dir)
    json_files = list(root_path.rglob("*.json"))
    valid_json_files = [f for f in json_files if "catalog" not in f.name.lower()]

    raw_scenes = []
    for j_path in valid_json_files:
        try:
            scene_data = parse_enmap_l1b_scene(str(j_path))
            if not scene_data['vnir_tif'] or not os.path.exists(scene_data['vnir_tif']):
                continue
                
            lons = [pt[0] for pt in scene_data['bounds_lonlat']]
            lats = [pt[1] for pt in scene_data['bounds_lonlat']]
            scene_bbox = [min(lons), min(lats), max(lons), max(lats)]
            
            if intersects(scene_bbox, roi_bbox):
                raw_scenes.append(scene_data)
        except Exception as e:
            continue

    if not raw_scenes:
        print("Warning: No L1B scenes found.")
        return

    raw_scenes.sort(key=lambda x: x['time'])

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

    width = loc_data['MGRS_WIDTH']
    height = loc_data['MGRS_HEIGHT']
    crs_wkt = f"EPSG:{loc_data['MGRS_EPSG']}"
    ul_x = loc_data['MGRS_UL_X']
    ul_y = loc_data['MGRS_UL_Y']
    target_gsd = loc_data.get('TARGET_GSD', 30.0)
    
    tf_target = Affine.translation(ul_x, ul_y) * Affine.scale(target_gsd, -target_gsd)
    
    base_wv = raw_scenes[0]['wavelengths']
    n_times = len(grouped_scenes)
    n_bands = 224 # VNIR + SWIR
    
    output_file = os.path.join(OUTPUT_DIR, f"EnMAP_MGRS_Stack_L1B_Merged_{target_location}.h5")

    with h5py.File(output_file, 'w') as out_h5:
        grp_enmap = out_h5.create_group("HDFEOS/GRIDS/ENMAP/Data Fields")
        meta_grp = out_h5.create_group("METADATA")
        info_grp = out_h5.create_group("HDFEOS INFORMATION")
        
        datasets_info = [
            ("surface_reflectance", np.dtype('int16'), 4, ["Time", "Band", "YDim", "XDim"]),
        ]
        
        rad_nodata = raw_scenes[0]['nodata']
        ds_rad = grp_enmap.create_dataset("surface_reflectance", shape=(n_times, n_bands, height, width), dtype='int16', compression="gzip", fillvalue=rad_nodata)

        acq_time_array = np.zeros(n_times, dtype='float64')

        for t_idx, group in enumerate(grouped_scenes):
            print(f"  [Pass {t_idx+1}/{n_times}] Assembling Sensor Geometry Swath...")
            
            pass_vnir_list = []
            pass_swir_list = []
            
            for scene in group:
                if scene['vnir_tif'] and os.path.exists(scene['vnir_tif']):
                    with rasterio.open(scene['vnir_tif']) as src:
                        pass_vnir_list.append(src.read())
                if scene['swir_tif'] and os.path.exists(scene['swir_tif']):
                    with rasterio.open(scene['swir_tif']) as src:
                        pass_swir_list.append(src.read())

            if not pass_vnir_list:
                continue
                
            # 1. Stitch chunks vertically (along Y axis / along-track) to form continuous sensor swath
            full_pass_vnir = np.concatenate(pass_vnir_list, axis=1)
            full_pass_swir = np.concatenate(pass_swir_list, axis=1)
            
            # Form full L1B radiance cube (VNIR + SWIR)
            full_pass_radiance = np.concatenate([full_pass_vnir, full_pass_swir], axis=0).astype('float32')

            # 2. Apply Atmospheric Correction on the continuous sensor-geometry swath
            full_pass_boa = apply_atmospheric_correction(full_pass_radiance, group)

            # 3. Project the corrected chunks to the MGRS canvas incrementally
            print(f"  [Pass {t_idx+1}/{n_times}] Projecting BOA Reflectance to MGRS Grid...")
            canvas_boa = np.full((n_bands, height, width), rad_nodata, dtype='float32')
            
            y_offset = 0
            for idx, scene in enumerate(group):
                chunk_height = pass_vnir_list[idx].shape[1]
                
                # Extract the corrected chunk from the full strip
                corrected_chunk = full_pass_boa[:, y_offset:y_offset+chunk_height, :]
                y_offset += chunk_height
                
                # Project this chunk to MGRS using the original L1B spatial metadata
                if scene['vnir_tif'] and os.path.exists(scene['vnir_tif']):
                    with rasterio.open(scene['vnir_tif']) as src:
                        incoming_boa = np.full((n_bands, height, width), rad_nodata, dtype='float32')
                        reproject(
                            source=corrected_chunk,
                            destination=incoming_boa,
                            src_transform=src.transform, src_crs=src.crs, dst_transform=tf_target, dst_crs=crs_wkt,
                            resampling=Resampling.nearest,
                            src_nodata=rad_nodata, dst_nodata=rad_nodata
                        )
                        valid = (incoming_boa != rad_nodata)
                        canvas_boa[valid] = incoming_boa[valid]
                        
            # Cast back to int16 for HDF5 storage
            canvas_boa[np.isnan(canvas_boa)] = rad_nodata
            ds_rad[t_idx, ...] = canvas_boa.astype('int16')
            acq_time_array[t_idx] = group[0]['time'].timestamp()

        # Add metadata
        ds_rad.attrs["acquisition_time"] = acq_time_array
        ds_rad.attrs["wavelengths"] = base_wv
        ds_rad.attrs["fwhm"] = raw_scenes[0]['fwhm']

    print(f"\nTensor Synthesis Complete. Stored at: {output_file}")

if __name__ == "__main__":
    process_mgrs_l1b_stack(LOCATION)
