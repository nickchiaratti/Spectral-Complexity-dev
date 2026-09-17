import os
import h5py
import rasterio
import numpy as np
from rasterio.warp import reproject, Resampling
from pyproj import CRS
from rasterio.transform import Affine
from rasterio.control import GroundControlPoint
from pathlib import Path
import glob
import sys
import yaml
import argparse
from datetime import datetime, timezone
import warnings

script_dir = Path(__file__).resolve().parent
if str(script_dir.parent) not in sys.path:
    sys.path.insert(0, str(script_dir.parent))
import SpecComplex as sc

SUN_ELEVATION_THRESHOLD = 30
TANAGER_CLOUD_DILATION = 4
TANAGER_UNCERTAINTY_THRESHOLD = 0.1
TANAGER_AEROSOL_THRESHOLD = 0.35
if str(script_dir.parent) not in sys.path:
    sys.path.insert(0, str(script_dir.parent))
from Harmonized_SC import sc_data_utils

def process_tanager(location):
    config_path = os.path.join(script_dir.parent, "locations_config.yaml")
    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)
    
    if location not in config_data["locations"]:
        raise ValueError(f"Location {location} not found in config.")
        
    config = config_data["locations"][location]
    source_cache = config.get("SOURCE_CACHE") or location
    tanager_source_dir = f"C:/satelliteImagery/Tanager/{source_cache}_SourceData"
    
    master_crs = CRS.from_epsg(config["MGRS_EPSG"])
    gsd = config["TARGET_GSD"]
    master_transform = Affine(gsd, 0.0, config["MGRS_UL_X"], 0.0, -gsd, config["MGRS_UL_Y"])
    master_height = config["MGRS_HEIGHT"]
    master_width = config["MGRS_WIDTH"]
    
    out_dir = "C:/satelliteImagery/Tanager"
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"Tanager_Native_Stack_{location}.h5")
    if os.path.exists(out_file):
        print(f"Skipping Tanager native stacking for {location}: {out_file} already exists.")
        return
    
    basic_files = glob.glob(os.path.join(tanager_source_dir, "**", "*_basic_sr_hdf5.h5"), recursive=True)
    if not basic_files:
        print(f"No basic_sr_hdf5 files found in {tanager_source_dir}")
        return
        
    passes = {}
    for f in basic_files:
        basename = os.path.basename(f)
        parts = basename.split('_')
        if len(parts) >= 1:
            pass_ts = parts[0]
            if pass_ts not in passes: passes[pass_ts] = []
            passes[pass_ts].append(f)
            
    pass_keys = sorted(list(passes.keys()))
    total_num_frames = len(pass_keys)
    if total_num_frames == 0:
        return
        
    with h5py.File(out_file, 'w') as h5f:
        with h5py.File(passes[pass_keys[0]][0], 'r') as f_test:
            src_df = f_test['HDFEOS/SWATHS/HYP/Data Fields']
            wv = src_df['surface_reflectance'].attrs.get('wavelengths')
            if wv is None:
                raise ValueError("Could not find wavelengths attribute.")
            
            # Identify bad bands
            bad_mask = ((wv >= 1350) & (wv <= 1440)) | ((wv >= 1800) & (wv <= 1950))
            good_mask = ~bad_mask
            filtered_wv = wv[good_mask]
            new_bands = len(filtered_wv)
            
            grp = h5f.create_group('/HDFEOS/GRIDS/Tanager/Data Fields')
            for name in src_df.keys():
                src_dset = src_df[name]
                is_3d = len(src_dset.shape) == 3
                
                if name == 'surface_reflectance':
                    out_shape = (total_num_frames, new_bands, master_height, master_width)
                    chunks = (1, new_bands, min(master_height, 256), min(master_width, 256))
                    dtype = np.int16
                    fill_val = -9999
                else:
                    bands = src_dset.shape[0] if is_3d else None
                    out_shape = (total_num_frames, bands, master_height, master_width) if is_3d else (total_num_frames, master_height, master_width)
                    chunks = (1, bands, min(master_height, 256), min(master_width, 256)) if is_3d else (1, min(master_height, 256), min(master_width, 256))
                    dtype = src_dset.dtype
                    fill_val = src_dset.attrs.get("_FillValue", -9999)
                    
                out_dset = grp.create_dataset(name, shape=out_shape, dtype=dtype, compression="gzip", compression_opts=5, fillvalue=fill_val, chunks=chunks)
                for attr_name, attr_val in src_dset.attrs.items():
                    out_dset.attrs[attr_name] = attr_val
                if name == 'surface_reflectance':
                    grp[name].attrs['wavelengths'] = filtered_wv
                    
                    crs = CRS.from_epsg(config["MGRS_EPSG"])
                    grp[name].attrs['spatial_ref'] = crs.to_wkt()
                    grp[name].attrs['GeoTransform'] = [config["MGRS_UL_X"], config["TARGET_GSD"], 0.0, config["MGRS_UL_Y"], 0.0, -config["TARGET_GSD"]]
                    
                    sc_data_utils.add_reflectance_attributes(grp[name], scale_to_float=0.0001, fill_value=-9999)
            
            # Ensure common_mask exists
            if 'common_mask' not in grp:
                mask_ds = grp.create_dataset(
                    "common_mask",
                    shape=(total_num_frames, master_height, master_width),
                    dtype=bool,
                    compression="gzip", compression_opts=5, shuffle=True,
                    fillvalue=False,
                    chunks=(1, min(master_height, 256), min(master_width, 256))
                )
                mask_ds.attrs['description'] = "True = Invalid/Masked, False = Valid. Generated from SpecComplex ARD rules."
                mask_ds.attrs['cloud_dilation'] = TANAGER_CLOUD_DILATION
                mask_ds.attrs['uncertainty_threshold'] = TANAGER_UNCERTAINTY_THRESHOLD
                mask_ds.attrs['aerosol_depth_threshold'] = TANAGER_AEROSOL_THRESHOLD
                mask_ds.attrs['sun_elevation_threshold'] = SUN_ELEVATION_THRESHOLD

        meta_lists = {'acq_time': [], 'space_id': []}
        for t_idx, pass_ts in enumerate(pass_keys):
            print(f"Processing Tanager pass {t_idx+1}/{total_num_frames}: {pass_ts}")
            chunks_files = passes[pass_ts]
            
            pass_canvases = {}
            pass_times = []
            for name in grp.keys():
                is_3d = len(grp[name].shape) == 4
                if name == 'surface_reflectance':
                    pass_canvases[name] = np.full((len(wv), master_height, master_width), np.nan, dtype=np.float32)
                else:
                    bands = grp[name].shape[1] if is_3d else None
                    canvas_shape = (bands, master_height, master_width) if is_3d else (master_height, master_width)
                    pass_canvases[name] = np.full(canvas_shape, grp[name].fillvalue, dtype=grp[name].dtype)
                    
            for chunk_file in chunks_files:
                with h5py.File(chunk_file, 'r') as f_chunk:
                    df_grp = f_chunk['HDFEOS/SWATHS/HYP/Data Fields']
                    geo_grp = f_chunk['HDFEOS/SWATHS/HYP/Geolocation Fields']
                    lat = geo_grp['Latitude'][:]
                    lon = geo_grp['Longitude'][:]
                    pass_times.extend(geo_grp['Time'][:].tolist())
                    
                    gcps = []
                    step = 10
                    rows = list(range(0, lat.shape[0], step))
                    if rows[-1] != lat.shape[0] - 1: rows.append(lat.shape[0] - 1)
                    cols = list(range(0, lat.shape[1], step))
                    if cols[-1] != lat.shape[1] - 1: cols.append(lat.shape[1] - 1)
                    for r in rows:
                        for c in cols:
                            gcps.append(GroundControlPoint(row=r, col=c, x=lon[r, c], y=lat[r, c]))
                    
                    for name in df_grp.keys():
                        src_data = df_grp[name][:]
                        is_3d = len(src_data.shape) == 3
                        if not is_3d: src_data = src_data[np.newaxis, ...]
                        
                        if name == 'surface_reflectance':
                            src_fill = df_grp[name].attrs.get("_FillValue", -9999.0)
                            incoming = np.full((len(wv), master_height, master_width), np.nan, dtype=np.float32)
                            reproject(
                                source=src_data, destination=incoming, src_transform=None, gcps=gcps,
                                src_crs="EPSG:4326", dst_transform=master_transform, dst_crs=master_crs,
                                resampling=Resampling.nearest, src_nodata=src_fill, dst_nodata=np.nan, tps=True
                            )
                            valid_mask = ~np.isnan(incoming)
                            pass_canvases[name][valid_mask] = incoming[valid_mask]
                        else:
                            src_fill = df_grp[name].attrs.get("_FillValue")
                            dtype = df_grp[name].dtype
                            bands = src_data.shape[0]
                            fill_val = grp[name].fillvalue
                            incoming = np.full((bands, master_height, master_width), fill_val, dtype=dtype)
                            resample_algo = Resampling.nearest if dtype.kind in ['i', 'u', 'b'] else Resampling.bilinear
                            reproject(
                                source=src_data, destination=incoming, src_transform=None, gcps=gcps,
                                src_crs="EPSG:4326", dst_transform=master_transform, dst_crs=master_crs,
                                resampling=resample_algo, src_nodata=src_fill, dst_nodata=fill_val, tps=True
                            )
                            if dtype.kind in ['f', 'c'] and (fill_val is None or np.isnan(fill_val)):
                                valid_mask = ~np.isnan(incoming)
                            else:
                                valid_mask = ~np.isclose(incoming, fill_val, equal_nan=True)
                                
                            if not is_3d:
                                pass_canvases[name][valid_mask[0]] = incoming[0][valid_mask[0]]
                            else:
                                pass_canvases[name][valid_mask] = incoming[valid_mask]
                            
            if len(pass_times) > 0:
                meta_lists['acq_time'].append(np.mean(pass_times))
            else:
                meta_lists['acq_time'].append(0.0)
            meta_lists['space_id'].append('Tanager-1')

            for name in pass_canvases.keys():
                if name == 'surface_reflectance':
                    # Strip bad wavelengths
                    filtered_data = pass_canvases[name][good_mask, ...]
                    # Scale to int16
                    int_data = sc_data_utils.scale_reflectance_for_storage(filtered_data, scale_factor=10000.0, fill_value_in=np.nan, fill_value_out=-9999)
                    grp[name][t_idx, ...] = int_data
                else:
                    final_arr = pass_canvases[name]
                    if len(grp[name].shape) == 3:
                        final_arr = final_arr[0]
                    grp[name][t_idx, ...] = final_arr
                    
            # Generate common_mask after all native datasets are written for this pass
            valid_mask = sc.get_tanager_mask(grp, t_idx, (master_height, master_width),
                                             sun_elevation_threshold=SUN_ELEVATION_THRESHOLD,
                                             cloud_dilation=TANAGER_CLOUD_DILATION,
                                             apply_cloud_mask=True,
                                             uncertainty_threshold=TANAGER_UNCERTAINTY_THRESHOLD,
                                             aerosol_depth_threshold=TANAGER_AEROSOL_THRESHOLD)
            grp['common_mask'][t_idx, ...] = valid_mask

        dt_str = h5py.string_dtype(encoding='ascii')
        grp['surface_reflectance'].attrs['acquisition_time'] = np.array(meta_lists['acq_time'], dtype='float64')
        grp['surface_reflectance'].attrs.create('spacecraft_id', data=np.array(meta_lists['space_id'], dtype=dt_str))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create native stack for Tanager.")
    parser.add_argument('--location', type=str, required=True, help="Location name from config.")
    args = parser.parse_args()
    process_tanager(args.location)
