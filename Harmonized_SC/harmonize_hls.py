import numpy as np
import h5py
from datetime import datetime, timezone
import warnings
from rasterio.warp import reproject, Resampling
from pyproj import CRS
from rasterio.transform import Affine
import sys
from pathlib import Path

script_dir = Path(__file__).resolve().parent
if str(script_dir.parent) not in sys.path:
    sys.path.insert(0, str(script_dir.parent))
import SpecComplex as sc


def process_hls_master_stack(
    native_h5_path, 
    daily_groups, 
    expected_sr, 
    master_height, 
    master_width, 
    master_transform, 
    master_crs, 
    min_roi_coverage, 
    sun_elev_thresh, 
    cloud_dil, 
    qa_reject_mask, 
    aerosol_accept_level
):
    """Harmonizes unprojected native arrays into the Master Grid directly in-memory."""
    sorted_dates = sorted(daily_groups.keys())
    if len(sorted_dates) == 0: return None

    # ---- PASS 1: Identify Valid Dates ----
    valid_dates = []
    with h5py.File(native_h5_path, 'r') as h5f:
        print("    [Pass 1] Evaluating ROI coverage...", flush=True)
        for date_str in sorted_dates:
            entries = daily_groups[date_str]
            accum_sr_band0 = np.full((1, master_height, master_width), np.nan, dtype=np.float32)
            
            for entry in entries:
                fidx = entry['frame_idx']
                grid_id = entry['grid_id']
                df_path = f'HDFEOS/GRIDS/{grid_id}/Data Fields'
                
                sr_node = h5f[f'{df_path}/surface_reflectance']
                src_tf = Affine.from_gdal(*sr_node.attrs['GeoTransform'])
                src_crs = CRS.from_wkt(sr_node.attrs['spatial_ref'])
                
                # Load only the first band
                src_sr = sr_node[fidx, 0:1, :, :]
                
                tmp_sr_band0 = np.full((1, master_height, master_width), np.nan, dtype=np.float32)
                reproject(source=src_sr, destination=tmp_sr_band0, src_transform=src_tf, src_crs=src_crs, dst_transform=master_transform, dst_crs=master_crs, resampling=Resampling.cubic, src_nodata=np.nan, dst_nodata=np.nan)
                
                mask_sr = ~np.isnan(tmp_sr_band0)
                accum_sr_band0[mask_sr] = tmp_sr_band0[mask_sr]
            
            valid_pixels = np.sum(~np.isnan(accum_sr_band0[0]))
            coverage = (valid_pixels / (master_height * master_width)) * 100
            if coverage >= min_roi_coverage:
                valid_dates.append(date_str)
            else:
                print(f"    Skipping HLS frame {date_str} (Coverage: {coverage:.1f}% < {min_roi_coverage}%)", flush=True)

    num_valid = len(valid_dates)
    if num_valid == 0:
        return None

    # ---- PASS 2: Extract and process ONLY valid frames ----
    print(f"    [Pass 2] Allocating memory for {num_valid} valid frames...", flush=True)
    stk_sr = np.full((num_valid, expected_sr, master_height, master_width), np.nan, dtype=np.float32)
    stk_fm = np.full((num_valid, 1, master_height, master_width), 255, dtype=np.uint8)
    stk_ag = np.full((num_valid, 4, master_height, master_width), np.nan, dtype=np.float32)
    stk_mask = np.ones((num_valid, master_height, master_width), dtype=bool)
    vis_data = np.zeros((num_valid, 4, master_height, master_width), dtype=np.uint8)
    meta_arrays = {'acq': [], 'space': [], 'saz': [], 'sel': [], 'cc': []}

    with h5py.File(native_h5_path, 'r') as h5f:
        for idx, date_str in enumerate(valid_dates):
            entries = daily_groups[date_str]
        
            base_grid = entries[0]['grid_id']
            base_fidx = entries[0]['frame_idx']
            base_path = f'HDFEOS/GRIDS/{base_grid}/Data Fields/surface_reflectance'
            meta_arrays['acq'].append(h5f[base_path].attrs['acquisition_time'][base_fidx])
        
            raw_spacecraft = h5f[base_path].attrs['spacecraft_id'][base_fidx]
            spacecraft_str = raw_spacecraft.decode('utf-8') if isinstance(raw_spacecraft, bytes) else str(raw_spacecraft)
            meta_arrays['space'].append(spacecraft_str)
            meta_arrays['cc'].append(h5f[base_path].attrs['cloud_cover'][base_fidx])
        
            for entry in entries:
                tile = entry['tile']
                fidx = entry['frame_idx']
                grid_id = entry['grid_id']
                df_path = f'HDFEOS/GRIDS/{grid_id}/Data Fields'
            
                sr_node = h5f[f'{df_path}/surface_reflectance']
                src_tf = Affine.from_gdal(*sr_node.attrs['GeoTransform'])
                src_crs = CRS.from_wkt(sr_node.attrs['spatial_ref'])
            
                src_sr = sr_node[fidx]
                tmp_sr = np.full((expected_sr, master_height, master_width), np.nan, dtype=np.float32)
                reproject(source=src_sr, destination=tmp_sr, src_transform=src_tf, src_crs=src_crs, dst_transform=master_transform, dst_crs=master_crs, resampling=Resampling.cubic, src_nodata=np.nan, dst_nodata=np.nan)
                mask_sr = ~np.isnan(tmp_sr)
                stk_sr[idx][mask_sr] = tmp_sr[mask_sr]
            
                src_fm = h5f[f'{df_path}/Fmask'][fidx]
                tmp_fm = np.full((1, master_height, master_width), 255, dtype=np.uint8)
                reproject(source=src_fm, destination=tmp_fm, src_transform=src_tf, src_crs=src_crs, dst_transform=master_transform, dst_crs=master_crs, resampling=Resampling.nearest, src_nodata=255, dst_nodata=255)
                mask_fm = (tmp_fm != 255)
                stk_fm[idx][mask_fm] = tmp_fm[mask_fm]
            
                src_ag = h5f[f'{df_path}/solar_view_angles'][fidx]
                tmp_ag = np.full((4, master_height, master_width), np.nan, dtype=np.float32)
                reproject(source=src_ag, destination=tmp_ag, src_transform=src_tf, src_crs=src_crs, dst_transform=master_transform, dst_crs=master_crs, resampling=Resampling.nearest, src_nodata=np.nan, dst_nodata=np.nan)
                mask_ag = ~np.isnan(tmp_ag)
                stk_ag[idx][mask_ag] = tmp_ag[mask_ag]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                mean_sza = np.nanmean(stk_ag[idx, 0])
                mean_saa = np.nanmean(stk_ag[idx, 1])
            
            meta_arrays['saz'].append(mean_saa)
            meta_arrays['sel'].append(90.0 - mean_sza)
        
            temp_grp = {'Fmask': stk_fm, 'solar_view_angles': stk_ag}
            stk_mask[idx] = sc.get_hls_mask(temp_grp, idx, 
                                            sun_elevation_threshold=sun_elev_thresh,
                                            cloud_dilation=cloud_dil,
                                            qa_reject_mask=qa_reject_mask,
                                            aerosol_accept_level=aerosol_accept_level).astype(bool)
        
            rgba_img = sc.generate_rgba_image(r_band = stk_sr[idx, 3, :, :], g_band = stk_sr[idx, 2, :, :], b_band = stk_sr[idx, 1, :, :])
            vis_data[idx, ...] = np.transpose(rgba_img, (2, 0, 1))

    return {'sr': stk_sr, 'fm': stk_fm, 'ag': stk_ag, 'vis': vis_data, 'mask': stk_mask, 'meta': meta_arrays, 'count': num_valid}
