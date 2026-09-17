import os
import h5py
import rasterio
import numpy as np
import warnings
import sys
from pathlib import Path
import json
import yaml
import argparse
from datetime import datetime, timezone
import rasterio
from PIL import Image

script_dir = Path(__file__).resolve().parent
if str(script_dir.parent) not in sys.path:
    sys.path.insert(0, str(script_dir.parent))
import SpecComplex as sc

SUN_ELEVATION_THRESHOLD = 30
ENMAP_CLOUD_DILATION = 4
ENMAP_REJECT_CLOUD = True
ENMAP_REJECT_CLOUD_SHADOW = True
ENMAP_REJECT_HAZE = True
ENMAP_REJECT_CIRRUS = True
ENMAP_REJECT_SNOW = True
ENMAP_REJECT_DEFECTIVE = True
ENMAP_REJECT_WATER = True
if str(script_dir.parent) not in sys.path:
    sys.path.insert(0, str(script_dir.parent))
from Harmonized_SC import sc_data_utils

TIME_THRESHOLD_SECONDS = 120

def load_default_enmap_wavelengths():
    excel_path = script_dir.parent / "wavelengths" / "EnMAP_Spectral_Bands_update.xlsx"
    if excel_path.exists():
        import pandas as pd
        try:
            df_vnir = pd.read_excel(excel_path, sheet_name='VNIR')
            df_swir = pd.read_excel(excel_path, sheet_name='SWIR')
            df = pd.concat([df_vnir, df_swir], ignore_index=True)
            if 'CW (nm)' in df.columns:
                return df['CW (nm)'].values.astype(np.float32)
        except:
            pass
    return np.zeros(224, dtype=np.float32)

def parse_enmap_stac(json_path):
    with open(json_path, 'r') as f:
        stac = json.load(f)
    dt_str = stac['properties']['datetime']
    acq_time = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
    assets = stac['assets']
    base_dir = os.path.dirname(json_path)
    
    def resolve_path(href):
        return os.path.join(base_dir, os.path.basename(href))
        
    eo_bands = assets.get('image', {}).get('eo:bands', [])
    wavelengths = []
    if eo_bands:
        for b in eo_bands:
            if 'eo:center_wavelength' in b: wavelengths.append(b['eo:center_wavelength'])
            elif 'center_wavelength' in b: wavelengths.append(b['center_wavelength'])
            
    if len(wavelengths) != 224 or any(w == 0.0 for w in wavelengths):
        wavelengths = load_default_enmap_wavelengths()
        
    nodata_val = float(stac['properties'].get('enmap:nodata', -32768))
    image_tif = resolve_path(assets['image']['href'])
    
    mask_keys = ['quality_classes', 'quality_cloud', 'quality_cloud_shadow', 'quality_haze', 'quality_cirrus', 'quality_snow', 'quality_testflags', 'defective_pixel_mask']
    mask_tifs = {mk: "" for mk in mask_keys}
    for mk in mask_keys:
        if mk in assets and 'href' in assets[mk]:
            mask_tifs[mk] = resolve_path(assets[mk]['href'])
    
    ret = {
        'time': acq_time,
        'reflectance_tif': image_tif,
        'wavelengths': np.array(wavelengths, dtype=np.float32),
        'nodata': nodata_val,
        'platform': stac['properties'].get('platform', 'enmap')
    }
    ret.update(mask_tifs)
    return ret

def process_enmap(location):
    config_path = os.path.join(script_dir.parent, "locations_config.yaml")
    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)
    
    if location not in config_data["locations"]:
        raise ValueError(f"Location {location} not found in config.")
        
    config = config_data["locations"][location]
    source_cache = config.get("SOURCE_CACHE") or location
    enmap_source_dir = f"C:/satelliteImagery/enmap/{source_cache}_SourceData"
    
    gsd = config["TARGET_GSD"]
    master_ul_x = config["MGRS_UL_X"]
    master_ul_y = config["MGRS_UL_Y"]
    master_height = config["MGRS_HEIGHT"]
    master_width = config["MGRS_WIDTH"]
    master_epsg = config["MGRS_EPSG"]
    
    out_dir = "C:/satelliteImagery/enmap"
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"EnMAP_Native_Stack_{location}.h5")
    if os.path.exists(out_file):
        print(f"Skipping EnMAP native stacking for {location}: {out_file} already exists.")
        return
    
    root_path = Path(enmap_source_dir)
    json_files = [f for f in root_path.rglob("*.json") if "catalog" not in f.name.lower()]
    valid_json_files = [f for f in json_files if any(f.parent.glob("*.TIF")) or any(f.parent.glob("*.tif"))]
    
    if not valid_json_files:
        print(f"No valid EnMAP scenes found in {enmap_source_dir}.")
        return
        
    raw_scenes = []
    for j_path in valid_json_files:
        try:
            scene_data = parse_enmap_stac(str(j_path))
            if os.path.exists(scene_data['reflectance_tif']):
                raw_scenes.append(scene_data)
        except Exception as e:
            continue
            
    valid_intersecting_scenes = []
    import rasterio
    for scene in raw_scenes:
        try:
            with rasterio.open(scene['reflectance_tif']) as src:
                col_off = int(round((src.transform.c - master_ul_x) / gsd))
                row_off = int(round((master_ul_y - src.transform.f) / gsd))
                src_h, src_w = src.height, src.width
                
                dst_col_start = max(0, col_off)
                dst_col_end = min(master_width, col_off + src_w)
                dst_row_start = max(0, row_off)
                dst_row_end = min(master_height, row_off + src_h)
                
                if dst_col_start < dst_col_end and dst_row_start < dst_row_end:
                    valid_intersecting_scenes.append(scene)
        except Exception:
            continue
            
    if not valid_intersecting_scenes:
        print(f"No valid EnMAP scenes intersect the ROI for {location}.")
        return
        
    raw_scenes = valid_intersecting_scenes
        
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
    
    total_num_frames = len(grouped_scenes)
    base_wv = grouped_scenes[0][0]['wavelengths']
    rad_nodata = grouped_scenes[0][0]['nodata']
    
    # Strip bad wavelengths
    bad_mask = ((base_wv >= 1350) & (base_wv <= 1440)) | ((base_wv >= 1800) & (base_wv <= 1950))
    good_mask = ~bad_mask
    filtered_wv = base_wv[good_mask]
    new_bands = len(filtered_wv)
    
    has_valid_data = False
    with h5py.File(out_file, 'w') as h5f:
        grp = h5f.create_group('/HDFEOS/GRIDS/EnMAP/Data Fields')
        sr_ds = grp.create_dataset(
            "surface_reflectance", 
            shape=(total_num_frames, new_bands, master_height, master_width), 
            dtype=np.int16, 
            compression="gzip", compression_opts=5, shuffle=True, 
            fillvalue=-9999, 
            chunks=(1, new_bands, min(master_height, 256), min(master_width, 256))
        )
        sr_ds.attrs['wavelengths'] = filtered_wv
        
        from pyproj import CRS
        crs = CRS.from_epsg(master_epsg)
        sr_ds.attrs['spatial_ref'] = crs.to_wkt()
        sr_ds.attrs['GeoTransform'] = [master_ul_x, gsd, 0.0, master_ul_y, 0.0, -gsd]
        
        sc_data_utils.add_reflectance_attributes(sr_ds, scale_to_float=0.0001, fill_value=-9999)
        
        mask_ds = grp.create_dataset(
            "common_mask",
            shape=(total_num_frames, master_height, master_width),
            dtype=bool,
            compression="gzip", compression_opts=5, shuffle=True,
            fillvalue=False,
            chunks=(1, min(master_height, 256), min(master_width, 256))
        )
        
        mask_ds.attrs['description'] = "True = Invalid/Masked, False = Valid. Generated from SpecComplex ARD rules."
        
        mask_keys = ['quality_classes', 'quality_cloud', 'quality_cloud_shadow', 'quality_haze', 'quality_cirrus', 'quality_snow', 'quality_testflags', 'defective_pixel_mask']
        mask_ds_dict = {}
        for mk in mask_keys:
            mask_ds_dict[mk] = grp.create_dataset(mk, shape=(total_num_frames, master_height, master_width), dtype='uint8', compression="gzip", shuffle=True, fillvalue=255)
        
        for t_idx, group in enumerate(grouped_scenes):
            print(f"Processing EnMAP pass {t_idx+1}/{total_num_frames}")
            # EnMAP natively comes in scale factor 10000, so it's already int16 scaled. 
            # We'll need to read it as float, strip bands, and use our function to handle properly.
            
            # Pre-allocate canvas for the scene
            canvas = np.full((len(base_wv), master_height, master_width), rad_nodata, dtype=np.int16)
            mask_canvases = {mk: np.full((master_height, master_width), 255, dtype=np.uint8) for mk in mask_keys}
            
            for scene in group:
                try:
                    with rasterio.open(scene['reflectance_tif']) as src:
                        # integer pixel offset
                        col_off = int(round((src.transform.c - master_ul_x) / gsd))
                        row_off = int(round((master_ul_y - src.transform.f) / gsd))
                        
                        src_data = src.read()
                        _, src_h, src_w = src_data.shape
                        
                        # Calculate bounds of intersection
                        dst_col_start = max(0, col_off)
                        dst_col_end = min(master_width, col_off + src_w)
                        dst_row_start = max(0, row_off)
                        dst_row_end = min(master_height, row_off + src_h)
                        
                        src_col_start = max(0, -col_off)
                        src_col_end = src_col_start + (dst_col_end - dst_col_start)
                        src_row_start = max(0, -row_off)
                        src_row_end = src_row_start + (dst_row_end - dst_row_start)
                        
                        if dst_col_start < dst_col_end and dst_row_start < dst_row_end:
                            valid_mask = src_data[:, src_row_start:src_row_end, src_col_start:src_col_end] != rad_nodata
                            
                            if np.any(valid_mask):
                                has_valid_data = True
                                
                            # Place data in canvas
                            canvas_slice = canvas[:, dst_row_start:dst_row_end, dst_col_start:dst_col_end]
                            src_slice = src_data[:, src_row_start:src_row_end, src_col_start:src_col_end]
                            
                            canvas_slice[valid_mask] = src_slice[valid_mask]
                            canvas[:, dst_row_start:dst_row_end, dst_col_start:dst_col_end] = canvas_slice
                            
                            # Now map the quality masks
                            for mk in mask_keys:
                                if mk in scene and scene[mk] and os.path.exists(scene[mk]):
                                    try:
                                        with rasterio.open(scene[mk]) as msrc:
                                            mdata = msrc.read([1])[0]
                                            m_valid = valid_mask[0] # taking 2D valid mask from reflectance
                                            mslice = mask_canvases[mk][dst_row_start:dst_row_end, dst_col_start:dst_col_end]
                                            msrc_slice = mdata[src_row_start:src_row_end, src_col_start:src_col_end]
                                            
                                            # Validate shapes to prevent index errors
                                            if m_valid.shape == mslice.shape and m_valid.shape == msrc_slice.shape:
                                                mslice[m_valid] = msrc_slice[m_valid]
                                                mask_canvases[mk][dst_row_start:dst_row_end, dst_col_start:dst_col_end] = mslice
                                    except rasterio.errors.RasterioIOError:
                                        pass
                                    
                except rasterio.errors.RasterioIOError:
                    print(f"Failed to read {scene['reflectance_tif']}")
                    
            # Filter bad wavelengths
            filtered_data = canvas[good_mask, ...]
            
            # EnMAP natively is int16 scaled by 10000. 
            # We want to use `scale_reflectance_for_storage` so we convert to float first
            float_data = np.where(filtered_data != rad_nodata, filtered_data.astype(np.float32) / 10000.0, np.nan)
            
            int_data = sc_data_utils.scale_reflectance_for_storage(float_data, scale_factor=10000.0, fill_value_in=np.nan, fill_value_out=-9999)
            
            sr_ds[t_idx, ...] = int_data
            
            for mk in mask_keys:
                mask_ds_dict[mk][t_idx, ...] = mask_canvases[mk]
                
            # Generate common_mask
            valid_mask = sc.get_enmap_mask(
                grp, t_idx, (master_height, master_width), 
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

    if not has_valid_data:
        if os.path.exists(out_file):
            os.remove(out_file)
        print(f"No valid EnMAP data intersected the ROI for {location}. Removed empty stack.")
        return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create native stack for EnMAP.")
    parser.add_argument('--location', type=str, required=True, help="Location name from config.")
    args = parser.parse_args()
    process_enmap(args.location)
