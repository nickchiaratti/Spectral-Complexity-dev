import os
import h5py
import rasterio
import numpy as np
from rasterio.warp import reproject, Resampling
from rasterio.transform import Affine
from pathlib import Path
import json
import glob
import sys
import yaml
import argparse
from datetime import datetime, timezone
import warnings

# Add parent folder to sys.path to find SpecComplex
script_dir = Path(__file__).resolve().parent
if str(script_dir.parent) not in sys.path:
    sys.path.insert(0, str(script_dir.parent))
import SpecComplex as sc

MIN_ROI_COVERAGE_PERCENT = 25.0 
SUN_ELEVATION_THRESHOLD = 30
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
        except Exception as e:
            print(f"Warning: Failed to load default EnMAP wavelengths from Excel: {e}")
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
            if 'eo:center_wavelength' in b:
                wavelengths.append(b['eo:center_wavelength'])
            elif 'center_wavelength' in b:
                wavelengths.append(b['center_wavelength'])
    
    if len(wavelengths) != 224 or any(w == 0.0 for w in wavelengths):
        print(f"  Warning: STAC JSON '{os.path.basename(json_path)}' omits complete band wavelength metadata. Falling back to default EnMAP 224-band table.")
        wavelengths = load_default_enmap_wavelengths()
        
    sun_elev = stac['properties'].get('view:sun_elevation', 0.0)
    sun_azim = stac['properties'].get('view:sun_azimuth', 0.0)
    off_nadir = float(stac['properties'].get('enmap:acrossOffNadirAngle', 0.0))
    inc_angle = 0.0 
    view_azim = float(stac['properties'].get('enmap:sceneAzimuthAngle', 0.0))
    
    nodata_val = -32768
    if 'enmap:nodata' in stac['properties']:
        nodata_val = float(stac['properties']['enmap:nodata'])

    mask_keys = ['quality_classes', 'quality_cloud', 'quality_cloud_shadow', 'quality_haze', 'quality_cirrus', 'quality_snow', 'quality_testflags', 'defective_pixel_mask']
    mask_tifs = {mk: "" for mk in mask_keys}
    
    for mk in mask_keys:
        if mk in assets and 'href' in assets[mk]:
            mask_tifs[mk] = resolve_path(assets[mk]['href'])

    image_tif = resolve_path(assets['image']['href'])

    ret = {
        'id': stac['id'],
        'time': acq_time,
        'platform': stac['properties'].get('platform', 'enmap'),
        'reflectance_tif': image_tif,
        'wavelengths': np.array(wavelengths, dtype=np.float32),
        'sun_elevation': sun_elev,
        'sun_azimuth': sun_azim,
        'view_off_nadir': off_nadir,
        'view_incidence_angle': inc_angle,
        'view_azimuth': view_azim,
        'nodata': nodata_val,
        'stac_dict': stac
    }
    ret.update(mask_tifs)
    return ret

def process_enmap_scenes_to_grid(h5f, enmap_source_dir, master_height, master_width, master_crs, master_transform, min_roi_coverage=MIN_ROI_COVERAGE_PERCENT, sun_elev_thresh=SUN_ELEVATION_THRESHOLD, cloud_dil=5, reject_cloud=True, reject_shadow=True, reject_haze=True, reject_cirrus=True, reject_snow=True, reject_defective=True, reject_water=True):
    root_path = Path(enmap_source_dir)
    json_files = list(root_path.rglob("*.json"))
    json_files = [f for f in json_files if "catalog" not in f.name.lower()]
    valid_json_files = [f for f in json_files if any(f.parent.glob("*.TIF")) or any(f.parent.glob("*.tif"))]

    if not valid_json_files:
        print(f"\nWARNING: No valid EnMAP STAC JSON files found in {enmap_source_dir}. Exiting.")
        return None

    if 'HDFEOS/GRIDS/ENMAP' in h5f:
        print("Removing existing ENMAP group from HDF5...")
        del h5f['HDFEOS/GRIDS/ENMAP']

    print(f"\nHarmonizing EnMAP Hyperspectral Arrays from Downloaded Scenes (Found {len(valid_json_files)} scenes)")
    
    raw_scenes = []
    for j_path in valid_json_files:
        try:
            scene_data = parse_enmap_stac(str(j_path))
            if not os.path.exists(scene_data['reflectance_tif']):
                continue
            raw_scenes.append(scene_data)
        except Exception as e:
            continue

    if not raw_scenes:
        print(f"\nWARNING: No valid EnMAP scenes found after parsing in {enmap_source_dir}.")
        return None

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
    print(f"Aggregated {len(raw_scenes)} scenes into {total_num_frames} temporal passes.")

    base_wv = grouped_scenes[0][0]['wavelengths']
    band_count = len(base_wv)
    
    datasets_created_info = []
    meta_lists = {'acq_time': [], 'space_id': [], 'good_wavelengths': []}
    valid_t_indices = []

    grp_enmap = h5f.create_group("HDFEOS/GRIDS/ENMAP/Data Fields")
    chunk_h, chunk_w = min(master_height, 256), min(master_width, 256)
    gdal_transform = np.array([master_transform.c, master_transform.a, master_transform.b, master_transform.f, master_transform.d, master_transform.e], dtype='float64')

    rad_nodata = grouped_scenes[0][0]['nodata']
    
    sr_ds = grp_enmap.create_dataset("surface_reflectance", shape=(total_num_frames, band_count, master_height, master_width), dtype='float32', compression="gzip", compression_opts=5, fillvalue=rad_nodata, chunks=(1, band_count, chunk_h, chunk_w))
    datasets_created_info.append(("surface_reflectance", np.dtype('float32'), 4, ["Time", "Band", "YDim", "XDim"]))
    
    mask_keys = ['quality_classes', 'quality_cloud', 'quality_cloud_shadow', 'quality_haze', 'quality_cirrus', 'quality_snow', 'quality_testflags', 'defective_pixel_mask']
    mask_ds_dict = {}
    for mk in mask_keys:
        ds = grp_enmap.create_dataset(mk, shape=(total_num_frames, master_height, master_width), dtype='uint8', compression="gzip", compression_opts=5, fillvalue=255, chunks=(1, chunk_h, chunk_w))
        mask_ds_dict[mk] = ds
        datasets_created_info.append((mk, np.dtype('uint8'), 3, ["Time", "YDim", "XDim"]))

    sun_elev_arr = np.zeros(total_num_frames, dtype='float32')

    for t_idx, group in enumerate(grouped_scenes):
        pass_ts = group[0]['time'].isoformat()
        print(f"  [EnMAP {t_idx+1}/{total_num_frames}] Translating Swath Pass: {pass_ts}...")
        
        meta_lists['space_id'].append(group[0]['platform'])
        meta_lists['acq_time'].append(group[0]['time'].timestamp())
        sun_elev_arr[t_idx] = group[0]['sun_elevation']
        
        canvas_rad = np.full((band_count, master_height, master_width), rad_nodata, dtype='float32')
        mask_canvases = {mk: np.full((1, master_height, master_width), 255, dtype='uint8') for mk in mask_keys}
        
        for scene in group:
            try:
                with rasterio.open(scene['reflectance_tif']) as src:
                    if t_idx == 0 and scene == group[0] and base_wv[0] == 0.0:
                        scene['wavelengths'] = np.zeros(band_count, dtype=np.float32)
                        base_wv = scene['wavelengths']
                    reproject(
                        source=rasterio.band(src, list(range(1, src.count + 1))),
                        destination=canvas_rad,
                        src_transform=src.transform, src_crs=src.crs, dst_transform=master_transform, dst_crs=master_crs,
                        resampling=Resampling.nearest,
                        src_nodata=rad_nodata, dst_nodata=rad_nodata
                    )
            except rasterio.errors.RasterioIOError:
                print(f"    Failed to read {scene['reflectance_tif']}")
                continue
                
            for mk in mask_keys:
                if scene.get(mk) and os.path.exists(scene[mk]):
                    try:
                        with rasterio.open(scene[mk]) as src:
                            reproject(
                                source=rasterio.band(src, 1),
                                destination=mask_canvases[mk],
                                src_transform=src.transform, src_crs=src.crs, dst_transform=master_transform, dst_crs=master_crs,
                                resampling=Resampling.nearest,
                                src_nodata=255, dst_nodata=255
                            )
                    except rasterio.errors.RasterioIOError:
                        pass 
        
        valid = ~np.isclose(canvas_rad[0], rad_nodata, equal_nan=True)
        sr_valid_pixels = np.sum(valid)
        
        frame_good_wv = ~np.all(np.isclose(canvas_rad, rad_nodata, equal_nan=True), axis=(1, 2))
        meta_lists['good_wavelengths'].append(frame_good_wv)
        
        # Check for -32768 (EnMAP hardcoded bad pixel) and add to defective_pixel_mask
        bad_pixel_locs = (canvas_rad == -32768)
        spatial_bad_pixels = np.any(bad_pixel_locs, axis=0)
        
        if 'defective_pixel_mask' in mask_canvases:
            mask_canvases['defective_pixel_mask'][0, spatial_bad_pixels] = 1
        
        # EnMAP Native Reflectance is stored as scaled Int16 (scale factor 10000)
        # We must convert valid pixels to a [0.0, 1.0] float representation
        valid_rad_3d = ~np.isclose(canvas_rad, rad_nodata, equal_nan=True) & ~bad_pixel_locs
        canvas_rad[valid_rad_3d] /= 10000.0
        
        # Set bad pixels to nodata to prevent skewing downstream numerics
        canvas_rad[bad_pixel_locs] = rad_nodata
        
        sr_ds[t_idx, ...] = canvas_rad
        for mk in mask_keys:
            mask_ds_dict[mk][t_idx, ...] = mask_canvases[mk][0, ...]
        
        coverage = (sr_valid_pixels / (master_height * master_width)) * 100
        if coverage >= min_roi_coverage:
            valid_t_indices.append(t_idx)
        else:
            print(f"    Warning: EnMAP pass {pass_ts} coverage ({coverage:.1f}%) < {min_roi_coverage}%")

    dt_str = h5py.string_dtype(encoding='ascii')
    sr_ds.attrs['acquisition_time'] = np.array(meta_lists['acq_time'], dtype='float64')
    sr_ds.attrs.create('spacecraft_id', data=np.array(meta_lists['space_id'], dtype=dt_str))
    sr_ds.attrs['wavelengths'] = base_wv
    sr_ds.attrs['sun_elevation'] = sun_elev_arr
    if len(meta_lists['good_wavelengths']) == total_num_frames:
        sr_ds.attrs['all_good_wavelengths'] = np.array(meta_lists['good_wavelengths'], dtype=bool)

    num_frames = len(valid_t_indices)
    if num_frames > 0:
        print("  Generating Common Mask for EnMAP on Master Grid...")
        mask_ds = grp_enmap.create_dataset('common_mask', shape=(total_num_frames, master_height, master_width), dtype=bool, compression="gzip", compression_opts=5, chunks=(1, chunk_h, chunk_w))
        datasets_created_info.append(("common_mask", bool, 3, ["Time", "YDim", "XDim"]))
        mask_ds.attrs['spatial_ref'] = master_crs.to_wkt()
        mask_ds.attrs['GeoTransform'] = gdal_transform
        mask_ds.attrs['description'] = "True = Invalid/Masked, False = Valid. Generated from SpecComplex ARD rules."
        h5f.flush()
        
        for out_idx in range(total_num_frames):
            valid_mask = sc.get_enmap_mask(
                grp_enmap, out_idx, (master_height, master_width), 
                sun_elevation_threshold=sun_elev_thresh,
                cloud_dilation=cloud_dil,
                reject_cloud=reject_cloud,
                reject_shadow=reject_shadow,
                reject_haze=reject_haze,
                reject_cirrus=reject_cirrus,
                reject_snow=reject_snow,
                reject_defective=reject_defective,
                reject_water=reject_water
            )
            mask_ds[out_idx, ...] = valid_mask
        
        print("  Generating strict 'ortho_visual' RGB composite from SR...")
        # EnMAP bands approximate: 650nm(Red), 550nm(Green), 450nm(Blue)
        r_idx = int(np.argmin(np.abs(base_wv - 650)))
        g_idx = int(np.argmin(np.abs(base_wv - 550)))
        b_idx = int(np.argmin(np.abs(base_wv - 450)))
        
        ortho_vis_dset = grp_enmap.create_dataset("ortho_visual", shape=(total_num_frames, 4, master_height, master_width), dtype='uint8', compression="gzip", compression_opts=5, fillvalue=0, chunks=(1, 4, chunk_h, chunk_w))
        datasets_created_info.append(("ortho_visual", np.dtype('uint8'), 4, ["Time", "RGBABand", "YDim", "XDim"]))
        ortho_vis_dset.attrs['spatial_ref'] = master_crs.to_wkt()
        ortho_vis_dset.attrs['GeoTransform'] = gdal_transform
        
        for out_idx in range(total_num_frames):
            r_band = sr_ds[out_idx, r_idx, :, :]
            g_band = sr_ds[out_idx, g_idx, :, :]
            b_band = sr_ds[out_idx, b_idx, :, :]
            
            r_input = np.where(r_band == rad_nodata, np.nan, r_band)
            g_input = np.where(g_band == rad_nodata, np.nan, g_band)
            b_input = np.where(b_band == rad_nodata, np.nan, b_band)
            
            rgba_img = sc.generate_rgba_image(r_input, g_input, b_input)
            ortho_vis_dset[out_idx, ...] = np.transpose(rgba_img, (2, 0, 1))

        return datasets_created_info, total_num_frames, band_count
    else:
        print("  No EnMAP passes met the minimum coverage threshold.")
        return None
