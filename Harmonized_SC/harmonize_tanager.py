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

# Add parent folder to sys.path to find SpecComplex
script_dir = Path(__file__).resolve().parent
if str(script_dir.parent) not in sys.path:
    sys.path.insert(0, str(script_dir.parent))
import SpecComplex as sc

MIN_ROI_COVERAGE_PERCENT = 25.0 
SUN_ELEVATION_THRESHOLD = 30
TANAGER_CLOUD_DILATION = 4
TANAGER_UNCERTAINTY_THRESHOLD = 0.1
TANAGER_AEROSOL_THRESHOLD = 0.35

def process_tanager_swaths_to_grid(h5f, tanager_source_dir, master_height, master_width, master_crs, master_transform, min_roi_coverage=MIN_ROI_COVERAGE_PERCENT, sun_elev_thresh=SUN_ELEVATION_THRESHOLD, cloud_dil=TANAGER_CLOUD_DILATION, uncert_thresh=TANAGER_UNCERTAINTY_THRESHOLD, aero_thresh=TANAGER_AEROSOL_THRESHOLD):
    basic_files = glob.glob(os.path.join(tanager_source_dir, "**", "*_basic_sr_hdf5.h5"), recursive=True)
    if not basic_files:
        print(f"\nWARNING: No basic_sr_hdf5 files found in {tanager_source_dir}. Exiting.")
        return None

    if 'HDFEOS/GRIDS/TANAGER' in h5f:
        print("Removing existing TANAGER group from HDF5...")
        del h5f['HDFEOS/GRIDS/TANAGER']

    print(f"\nHarmonizing Tanager Hyperspectral Arrays from Basic Swaths (Found {len(basic_files)} chunks)")
    passes = {}
    for f in basic_files:
        basename = os.path.basename(f)
        parts = basename.split('_')
        if len(parts) >= 1:
            pass_ts = parts[0]
            if pass_ts not in passes: passes[pass_ts] = []
            passes[pass_ts].append(f)
    
    valid_t_indices = []
    pass_keys = sorted(list(passes.keys()))
    total_num_frames = len(pass_keys)
    
    band_count = 0
    datasets_created_info = []
    meta_lists = {'acq_time': [], 'space_id': [], 'good_wavelengths': []}
    
    if total_num_frames > 0:
        with h5py.File(passes[pass_keys[0]][0], 'r') as f_test:
            sr_test = f_test['HDFEOS/SWATHS/HYP/Data Fields/surface_reflectance']
            band_count = sr_test.shape[0]
        grp_tanager = h5f.create_group("HDFEOS/GRIDS/TANAGER/Data Fields")
        chunk_h, chunk_w = min(master_height, 256), min(master_width, 256)
        gdal_transform = np.array([master_transform.c, master_transform.a, master_transform.b, master_transform.f, master_transform.d, master_transform.e], dtype='float64')

        with h5py.File(passes[pass_keys[0]][0], 'r') as f_meta:
            src_df = f_meta['HDFEOS/SWATHS/HYP/Data Fields']
            for name in src_df.keys():
                src_dset = src_df[name]
                dtype = src_dset.dtype
                is_3d = len(src_dset.shape) == 3
                bands = src_dset.shape[0] if is_3d else None
            
                out_shape = (total_num_frames, bands, master_height, master_width) if is_3d else (total_num_frames, master_height, master_width)
                chunks = (1, bands, chunk_h, chunk_w) if is_3d else (1, chunk_h, chunk_w)
            
                if "_FillValue" not in src_dset.attrs:
                    raise ValueError(f"Missing _FillValue attribute in dataset {name}.")
                fill_val = src_dset.attrs["_FillValue"]
                out_dset = grp_tanager.create_dataset(name, shape=out_shape, dtype=dtype, compression="gzip", compression_opts=5, fillvalue=fill_val, chunks=chunks)
                datasets_created_info.append((name, dtype, len(out_shape), ["Time", "Band", "YDim", "XDim"] if is_3d else ["Time", "YDim", "XDim"]))

        for t_idx, pass_ts in enumerate(pass_keys):
            print(f"  [Tanager {t_idx+1}/{total_num_frames}] Translating Swath Pass: {pass_ts}...")
            chunks_files = passes[pass_ts]
            
            pass_canvases = {}
            pass_times = []
            for name in grp_tanager.keys():
                dtype = grp_tanager[name].dtype
                is_3d = len(grp_tanager[name].shape) == 4
                bands = grp_tanager[name].shape[1] if is_3d else None
                canvas_shape = (bands, master_height, master_width) if is_3d else (master_height, master_width)
                fill_val = grp_tanager[name].fillvalue                    
                pass_canvases[name] = np.full(canvas_shape, fill_val, dtype=dtype)

            meta_lists['space_id'].append('Tanager-1')
            gw_found = False
            for chunk_idx, chunk_file in enumerate(chunks_files):
                with h5py.File(chunk_file, 'r') as f_chunk:
                    df_grp = f_chunk['HDFEOS/SWATHS/HYP/Data Fields']
                    geo_grp = f_chunk['HDFEOS/SWATHS/HYP/Geolocation Fields']
                    lat = geo_grp['Latitude'][:]
                    lon = geo_grp['Longitude'][:]
                    nodata_mask = df_grp['nodata_pixels'][:]
                    pass_times.extend(geo_grp['Time'][:].tolist())
                    
                    if chunk_idx == 0:
                        gw = df_grp['surface_reflectance'].attrs.get('good_wavelengths')
                        if gw is not None:
                            meta_lists['good_wavelengths'].append(gw)
                            gw_found = True
                
                    gcps = []
                    step = 10
                    rows = list(range(0, lat.shape[0], step))
                    if rows[-1] != lat.shape[0] - 1:
                        rows.append(lat.shape[0] - 1)
                    cols = list(range(0, lat.shape[1], step))
                    if cols[-1] != lat.shape[1] - 1:
                        cols.append(lat.shape[1] - 1)
                        
                    for r in rows:
                        for c in cols:
                            gcps.append(GroundControlPoint(row=r, col=c, x=lon[r, c], y=lat[r, c]))
                
                    for name in df_grp.keys():
                        if chunk_idx == 0:
                            for attr_name, attr_val in df_grp[name].attrs.items():
                                if attr_name not in grp_tanager[name].attrs:
                                    grp_tanager[name].attrs[attr_name] = attr_val
                        is_3d = len(grp_tanager[name].shape) == 4
                        bands = grp_tanager[name].shape[1] if is_3d else None
                        dtype = df_grp[name].dtype
                        
                        fill_val = grp_tanager[name].fillvalue
                        resample_algo = Resampling.average
                        if dtype.kind in ['i', 'u', 'b']:
                            resample_algo = Resampling.nearest
                        
                        src_data = df_grp[name][:]
                        
                        if not is_3d:
                            src_data = src_data[np.newaxis, ...]
                            incoming = np.full((1, master_height, master_width), fill_val, dtype=dtype)
                        else:
                            incoming = np.full((bands, master_height, master_width), fill_val, dtype=dtype)
                            
                        reprojected, _ = reproject(
                            source=src_data,
                            destination=incoming,
                            src_transform=None,
                            gcps=gcps,
                            src_crs="EPSG:4326",
                            dst_transform=master_transform,
                            dst_crs=master_crs,
                            resampling=resample_algo,
                            src_nodata=fill_val,
                            dst_nodata=fill_val,
                            tps=True
                        )
                        
                        if dtype.kind in ['f', 'c'] and np.isnan(fill_val):
                            valid_mask = ~np.isnan(incoming)
                        else:
                            valid_mask = ~np.isclose(incoming, fill_val, equal_nan=True)
                            
                        if not is_3d:
                            valid_mask = valid_mask[0]
                            pass_canvases[name][valid_mask] = incoming[0][valid_mask]
                        else:
                            pass_canvases[name][valid_mask] = incoming[valid_mask]

            if len(pass_times) > 0:
                meta_lists['acq_time'].append(np.mean(pass_times))
            else:
                meta_lists['acq_time'].append(0.0)
                
            if not gw_found:
                meta_lists['good_wavelengths'].append(np.zeros(band_count, dtype=bool))

            sr_valid_pixels = 0
            sr_fill = grp_tanager['surface_reflectance'].fillvalue
            sr_canvas = pass_canvases['surface_reflectance']
            valid = ~np.isclose(sr_canvas[0], sr_fill, equal_nan=True)
            sr_valid_pixels = np.sum(valid)

            for name in pass_canvases.keys():
                dtype = grp_tanager[name].dtype
                fill_val = grp_tanager[name].fillvalue
                if isinstance(fill_val, (np.ndarray, list)): fill_val = fill_val[0]
                final_arr = pass_canvases[name]
                grp_tanager[name][t_idx, ...] = final_arr
                
                
            coverage = (sr_valid_pixels / (master_height * master_width)) * 100
            if coverage >= min_roi_coverage:
                valid_t_indices.append(t_idx)
            else:
                print(f"    Warning: Tanager pass {pass_ts} coverage ({coverage:.1f}%) < {min_roi_coverage}%")
    
        # Write global metadata arrays
        dt_str = h5py.string_dtype(encoding='ascii')
        grp_tanager['surface_reflectance'].attrs['acquisition_time'] = np.array(meta_lists['acq_time'], dtype='float64')
        grp_tanager['surface_reflectance'].attrs.create('spacecraft_id', data=np.array(meta_lists['space_id'], dtype=dt_str))
        if len(meta_lists['good_wavelengths']) == total_num_frames:
            grp_tanager['surface_reflectance'].attrs['all_good_wavelengths'] = np.array(meta_lists['good_wavelengths'], dtype=bool)

        num_frames = len(valid_t_indices)
        if num_frames > 0:
            print("  Generating Common Mask for Tanager on Master Grid...")
            mask_ds = grp_tanager.create_dataset('common_mask', shape=(total_num_frames, master_height, master_width), dtype=bool, compression="gzip", compression_opts=5, chunks=(1, chunk_h, chunk_w))
            datasets_created_info.append(("common_mask", bool, 3, ["Time", "YDim", "XDim"]))
            mask_ds.attrs['spatial_ref'] = master_crs.to_wkt()
            mask_ds.attrs['GeoTransform'] = gdal_transform
            mask_ds.attrs['description'] = "True = Invalid/Masked, False = Valid. Generated from SpecComplex ARD rules."
            h5f.flush()
        
            for out_idx in range(total_num_frames):
                valid_mask = sc.get_tanager_mask(grp_tanager, out_idx, (master_height, master_width),
                                                 sun_elevation_threshold=sun_elev_thresh,
                                                 cloud_dilation=cloud_dil,
                                                 apply_cloud_mask=True,
                                                 uncertainty_threshold=uncert_thresh,
                                                 aerosol_depth_threshold=aero_thresh)
                mask_ds[out_idx, ...] = valid_mask
            
            
        
            print("  Generating strict 'ortho_visual' RGB composite from SR...")
            wavelengths = grp_tanager['surface_reflectance'].attrs['wavelengths']
            r_idx = int(np.argmin(np.abs(wavelengths - 650)))
            g_idx = int(np.argmin(np.abs(wavelengths - 550)))
            b_idx = int(np.argmin(np.abs(wavelengths - 450)))
            ortho_vis_dset = grp_tanager.create_dataset("ortho_visual", shape=(total_num_frames, 4, master_height, master_width), dtype='uint8', compression="gzip", compression_opts=5, fillvalue=0, chunks=(1, 4, chunk_h, chunk_w))
            datasets_created_info.append(("ortho_visual", np.dtype('uint8'), 4, ["Time", "RGBABand", "YDim", "XDim"]))
            ortho_vis_dset.attrs['spatial_ref'] = master_crs.to_wkt()
            ortho_vis_dset.attrs['GeoTransform'] = gdal_transform
            
            sr_dset_ref = grp_tanager["surface_reflectance"]
            sr_fill = sr_dset_ref.fillvalue
            if isinstance(sr_fill, (np.ndarray, list)): sr_fill = sr_fill[0]
            for out_idx in range(total_num_frames):
                r_band = sr_dset_ref[out_idx, r_idx, :, :]
                g_band = sr_dset_ref[out_idx, g_idx, :, :]
                b_band = sr_dset_ref[out_idx, b_idx, :, :]
                
                r_input = np.where(r_band < -1, np.nan, r_band)
                g_input = np.where(g_band < -1, np.nan, g_band)
                b_input = np.where(b_band < -1, np.nan, b_band)
                
                rgba_img = sc.generate_rgba_image(r_input, g_input, b_input)
                ortho_vis_dset[out_idx, ...] = np.transpose(rgba_img, (2, 0, 1))
        
        return datasets_created_info, total_num_frames, band_count
    else:
        print("  No Tanager passes met the minimum coverage threshold.")
        return None

def main():
    parser = argparse.ArgumentParser(description="Run Tanager pipeline standalone on an existing Harmonized HDF5 file.")
    parser.add_argument('--hdf5_path', type=str, required=True, help="Path to the Harmonized HDF5 file.")
    args = parser.parse_args()

    config_path = os.path.join(script_dir.parent, "locations_config.yaml")
    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)

    with h5py.File(args.hdf5_path, 'r+') as h5f:
        if 'METADATA/PIPELINE_CONFIG' in h5f:
            Location = h5f['METADATA/PIPELINE_CONFIG'].attrs['Location']
        else:
            Location = config_data.get("current_run", {}).get("location", "Palisades")
            
        config = config_data["locations"][Location]
        SOURCE_CACHE = config.get("SOURCE_CACHE", Location)
        if SOURCE_CACHE is None: SOURCE_CACHE = Location
        
        TANAGER_SOURCE_DIR = f"C:/satelliteImagery/Tanager/{SOURCE_CACHE}_SourceData"

        # Determine Master Grid properties
        sr_dset = None
        if 'HDFEOS/GRIDS/HLSS30/Data Fields/surface_reflectance' in h5f:
            sr_dset = h5f['HDFEOS/GRIDS/HLSS30/Data Fields/surface_reflectance']
        elif 'HDFEOS/GRIDS/HLSL30/Data Fields/surface_reflectance' in h5f:
            sr_dset = h5f['HDFEOS/GRIDS/HLSL30/Data Fields/surface_reflectance']
            
        if sr_dset is None:
            raise ValueError("No existing HLS surface_reflectance dataset found to derive master grid.")
            
        master_height, master_width = sr_dset.shape[-2:]
        master_crs = CRS.from_wkt(sr_dset.attrs['spatial_ref'])
        master_transform = Affine.from_gdal(*sr_dset.attrs['GeoTransform'])
        
        master_proj = "ALBERS"
        master_zone = 0
        master_gctp = [0] * 15

        res = process_tanager_swaths_to_grid(h5f, TANAGER_SOURCE_DIR, master_height, master_width, master_crs, master_transform)
        if res:
            datasets_created_info, total_num_frames, band_count = res
            
            import hdfeos_odl
            tanager_odl = hdfeos_odl.generate_dynamic_odl_grid_string("TANAGER", master_width, master_height, master_transform, master_proj, master_zone, master_gctp, datasets_created_info, total_num_frames, band_count)
            
            if "HDFEOS INFORMATION" in h5f and "StructMetadata.0" in h5f["HDFEOS INFORMATION"]:
                info_grp = h5f["HDFEOS INFORMATION"]
                current_odl = info_grp["StructMetadata.0"][()].decode("ascii")
                if "GROUP=TANAGER" not in current_odl and "END" in current_odl:
                    new_odl = current_odl.replace("\nEND\n", f"\n{tanager_odl}\nEND\n")
                    del info_grp["StructMetadata.0"]
                    info_grp.create_dataset("StructMetadata.0", data=np.string_(new_odl))

if __name__ == "__main__":
    main()
