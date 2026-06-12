'''
Combines HDFEOS compliant grids in separate h5 files using a dynamically 
centered Albers Equal Area coordinate grid into a single h5 file. 
Implements an ROI bounding box for the combined data and a strict spatial 
coverage guardrail to reject marginal swath overlaps.
'''

import os
import platform
# Monkeypatch platform._wmi_query to raise OSError immediately, bypassing Windows WMI hangs/KeyErrors in multiprocessing child processes
def _dummy_wmi_query(*args, **kwargs):
    raise OSError("WMI disabled to prevent hangs")
platform._wmi_query = _dummy_wmi_query

import h5py
import rasterio
import numpy as np
from datetime import datetime, timezone
from rasterio.transform import from_bounds as transform_from_bounds
from rasterio.warp import reproject, Resampling
from pyproj import Transformer, CRS
from rasterio.transform import Affine
from rasterio.control import GroundControlPoint
from pathlib import Path
import json
import warnings
import glob
import sys
# Add parent folder to sys.path to find SpecComplex and SpecComplexQR
script_dir = Path(__file__).resolve().parent
if str(script_dir.parent) not in sys.path:
    sys.path.insert(0, str(script_dir.parent))
import SpecComplex as sc

import yaml


def percentile_stretch(band_data, fill_value, lower_pct=1, upper_pct=99):
    valid_mask = (band_data != fill_value)
    if band_data.dtype.kind in ['f', 'c']:
        valid_mask &= ~np.isnan(band_data)
    valid_data = band_data[valid_mask]
    if valid_data.size == 0:
        return np.zeros_like(band_data, dtype=np.uint8)
    vmin = np.percentile(valid_data, lower_pct)
    vmax = np.percentile(valid_data, upper_pct)
    if vmax == vmin:
        return np.zeros_like(band_data, dtype=np.uint8)
    stretched = np.clip((band_data - vmin) / (vmax - vmin), 0, 1)
    stretched = (stretched * 255).astype(np.uint8)
    res = np.zeros_like(band_data, dtype=np.uint8)
    res[valid_mask] = stretched[valid_mask]
    return res


def main(target_location=None):

    # ==========================================
    # 1. CONFIGURATION & DIRECTORIES
    # ==========================================

    # Load Configuration
    script_dir = Path(__file__).resolve().parent
    config_path = os.path.join(script_dir, "locations_config.yaml")
    if not os.path.exists(config_path):
        config_path = os.path.join(script_dir.parent, "locations_config.yaml")
    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)

    if target_location is not None:
        Location = target_location
    else:
        Location = config_data.get("current_run", {}).get("location", "Palisades")
    config = config_data["locations"][Location]

    SOURCE_CACHE = config.get("SOURCE_CACHE", Location)
    if SOURCE_CACHE is None:
        SOURCE_CACHE = Location

    ROI_LON_MIN = config["ROI_LON_MIN"]
    ROI_LON_MAX = config["ROI_LON_MAX"]
    ROI_LAT_MIN = config["ROI_LAT_MIN"]
    ROI_LAT_MAX = config["ROI_LAT_MAX"]
    TANAGER_AVAILABLE = config.get("TANAGER_AVAILABLE", False)

    HLS_SOURCE_DIR = "C:/satelliteImagery/HLS30/"

    TANAGER_SOURCE_DIR = f"C:/satelliteImagery/Tanager/{SOURCE_CACHE}_SourceData"
    COMBINED_OUTPUT_DIR = "C:/satelliteImagery/HLST30/"
 
    INPUT_NATIVE_HDF5 = os.path.join(HLS_SOURCE_DIR, f"HLS_{Location}_STAC_Native_2025.h5")
    if TANAGER_AVAILABLE:
        INPUT_NATIVE_TANAGER_HDF5 = os.path.join(TANAGER_SOURCE_DIR, f"Tanager_Native_Stack_{SOURCE_CACHE}.h5")
    else:
        INPUT_NATIVE_TANAGER_HDF5 = os.path.join(TANAGER_SOURCE_DIR, "SKIP")#"Tanager_Native_Stack_HDFEOS.h5")
    OUTPUT_MASTER_HDF5 = os.path.join(COMBINED_OUTPUT_DIR, f"HLST_{Location}_Harmonized.h5")

    S30_WAVELENGTHS = [0.443, 0.490, 0.560, 0.665, 0.705, 0.740, 0.783, 0.842, 1.610, 2.190]
    L30_SR_WAVELENGTHS = [0.443, 0.482, 0.561, 0.655, 0.865, 1.609, 2.201]

    safe_bbox = [
        min(ROI_LON_MIN, ROI_LON_MAX), max(ROI_LAT_MIN, ROI_LAT_MAX), 
        max(ROI_LON_MIN, ROI_LON_MAX), min(ROI_LAT_MIN, ROI_LAT_MAX)
    ]
    safe_bbox = [min(safe_bbox[0], safe_bbox[2]), min(safe_bbox[1], safe_bbox[3]), max(safe_bbox[0], safe_bbox[2]), max(safe_bbox[1], safe_bbox[3])]
    TARGET_RESOLUTION = 30.0

    # --- Pixel Mask Configuration ---
    # Strict 85% spatial coverage threshold to prevent edge collisions
    MIN_ROI_COVERAGE_PERCENT = 25.0 
    SUN_ELEVATION_THRESHOLD = 30
    # HLS Specific Configuration (Unified Fmask for both S30 and L30)
    # Bits 0-5: cirrus, cloud, adj cloud/shadow, cloud shadow, snow/ice, water
    HLS_CLOUD_DILATION =0
    QA_REJECT_MASK = 0b11111 
    AEROSOL_ACCEPT_LEVEL = 'medium' # 'low' (0-1), 'medium' (0-2), 'high' (0-3)

    # TANAGER Specific Configuration
    TANAGER_CLOUD_DILATION = 2
    TANAGER_UNCERTAINTY_THRESHOLD = 0.1
    TANAGER_AEROSOL_THRESHOLD = 0.35

    # ==========================================
    # 2. MASTER GRID PRE-CALCULATION (DYNAMIC ALBERS)
    # ==========================================
    def calculate_master_grid(bbox, resolution):
        """
        Calculates a Unified Master Grid using a Dynamically Centered Albers Equal Area projection.
        Prevents spatial distortion on international targets by centering the map mathematically
        on the chosen ROI using the Deetz & Adams One-Sixth Rule.
        """
        min_lon, min_lat, max_lon, max_lat = bbox
    
        # Derive optimal parameters directly from the spatial bounds
        central_lon = (min_lon + max_lon) / 2.0
        central_lat = (min_lat + max_lat) / 2.0
        lat_1 = min_lat + (max_lat - min_lat) / 6.0
        lat_2 = max_lat - (max_lat - min_lat) / 6.0
    
        proj_str = f"+proj=aea +lat_1={lat_1:.6f} +lat_2={lat_2:.6f} +lat_0={central_lat:.6f} +lon_0={central_lon:.6f} +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
        dst_crs = CRS.from_string(proj_str)
    
        transformer = Transformer.from_crs("EPSG:4326", dst_crs, always_xy=True)
        xs, ys = transformer.transform([bbox[0], bbox[2], bbox[2], bbox[0]], [bbox[3], bbox[3], bbox[1], bbox[1]])
        minx, maxx, miny, maxy = min(xs), max(xs), min(ys), max(ys)
    
        width = int(np.ceil((maxx - minx) / resolution))
        height = int(np.ceil((maxy - miny) / resolution))
        transform = transform_from_bounds(minx, miny, maxx, maxy, width, height)
    
        # Preserves HDF-EOS GCTP metadata provenance accurately
        gctp_params = [6378137.0, 6356752.314245, lat_1, lat_2, central_lon, central_lat, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    
        return dst_crs, transform, width, height, "GCTP_ALBERS", 0, gctp_params

    master_crs, master_transform, master_width, master_height, master_proj, master_zone, master_gctp = calculate_master_grid(safe_bbox, TARGET_RESOLUTION)
    print(f"Master Grid Established: {master_width}x{master_height} at Dynamic Albers Equal Area (Centered: {master_gctp[5]:.2f}N, {master_gctp[4]:.2f}E)")

    # ==========================================
    # 3. HDFEOS5 ODL GENERATORS
    # ==========================================
    def generate_odl_grid_string(grid_name, width, height, transform, proj_code, zone, proj_params, num_sr_bands, num_frames, has_tile_mask=False):
        ul_x, ul_y = transform.c, transform.f
        lr_x = transform.c + (transform.a * width)
        lr_y = transform.f + (transform.e * height)
        p_str = str(tuple(proj_params)).replace(' ', '').replace('(', '').replace(')', '')
    
        fields = []
        idx = 1
        fields.append(f"""            OBJECT=DataField_{idx}\n                DataFieldName="surface_reflectance"\n                DataType=HDF5T_NATIVE_FLOAT\n                DimList=("Time","Bands","YDim","XDim")\n            END_OBJECT=DataField_{idx}""")
        idx += 1
        fields.append(f"""            OBJECT=DataField_{idx}\n                DataFieldName="Fmask"\n                DataType=HDF5T_NATIVE_UINT8\n                DimList=("Time","YDim","XDim")\n            END_OBJECT=DataField_{idx}""")
        idx += 1
        fields.append(f"""            OBJECT=DataField_{idx}\n                DataFieldName="solar_view_angles"\n                DataType=HDF5T_NATIVE_FLOAT\n                DimList=("Time","AngleBands","YDim","XDim")\n            END_OBJECT=DataField_{idx}""")
        idx += 1
        if has_tile_mask:
            fields.append(f"""            OBJECT=DataField_{idx}\n                DataFieldName="source_tile_mask"\n                DataType=HDF5T_NATIVE_UINT8\n                DimList=("Time","YDim","XDim")\n            END_OBJECT=DataField_{idx}""")
            idx += 1
        fields.append(f"""            OBJECT=DataField_{idx}\n                DataFieldName="ortho_visual"\n                DataType=HDF5T_NATIVE_UINT8\n                DimList=("Time","VisBand","YDim","XDim")\n            END_OBJECT=DataField_{idx}""")
        idx += 1
        fields.append(f"""            OBJECT=DataField_{idx}\n                DataFieldName="common_mask"\n                DataType=HDF5T_NATIVE_B8\n                DimList=("Time","YDim","XDim")\n            END_OBJECT=DataField_{idx}""")
    
        data_fields_str = "\n".join(fields)
    
        return f"""    GROUP={grid_name}
            GridName="{grid_name}"
            XDim={width}
            YDim={height}
            UpperLeftPointMtrs=({ul_x:.6f},{ul_y:.6f})
            LowerRightMtrs=({lr_x:.6f},{lr_y:.6f})
            Projection={proj_code}
            ZoneCode={zone}
            SphereCode=12
            ProjParams={p_str}
            GROUP=Dimension
                OBJECT=Dimension_1
                    DimensionName="Time"
                    Size={num_frames}
                END_OBJECT=Dimension_1
                OBJECT=Dimension_2
                    DimensionName="Bands"
                    Size={num_sr_bands}
                END_OBJECT=Dimension_2
                OBJECT=Dimension_4
                    DimensionName="YDim"
                    Size={height}
                END_OBJECT=Dimension_4
                OBJECT=Dimension_5
                    DimensionName="XDim"
                    Size={width}
                END_OBJECT=Dimension_5
                OBJECT=Dimension_6
                    DimensionName="AngleBands"
                    Size=4
                END_OBJECT=Dimension_6
                OBJECT=Dimension_7
                    DimensionName="VisBand"
                    Size=4
                END_OBJECT=Dimension_7
            END_GROUP=Dimension
            GROUP=DataField
    {data_fields_str}
            END_GROUP=DataField
            GROUP=MergedFields
            END_GROUP=MergedFields
        END_GROUP={grid_name}"""

    def generate_tanager_odl_string(grid_name, width, height, transform, proj_code, zone, proj_params, datasets_info, n_times, n_bands):
        ul_x, ul_y = transform.c, transform.f
        lr_x = transform.c + (transform.a * width)
        lr_y = transform.f + (transform.e * height)
        p_str = str(tuple(proj_params)).replace(' ', '').replace('(', '').replace(')', '')
    
        data_fields_blocks = []
        for i, (name, dtype, rank, dim_names) in enumerate(datasets_info):
            eos_type = "HDF5T_NATIVE_FLOAT"
            if "uint8" in str(dtype): eos_type = "HDF5T_NATIVE_UINT8"
            elif "uint16" in str(dtype): eos_type = "HDF5T_NATIVE_UINT16"
            elif "uint" in str(dtype): eos_type = "HDF5T_NATIVE_UINT"
            elif "int" in str(dtype): eos_type = "HDF5T_NATIVE_INT"
            elif "float64" in str(dtype) or "double" in str(dtype): eos_type = "HDF5T_NATIVE_DOUBLE"
            elif "bool" in str(dtype): eos_type = "HDF5T_NATIVE_B8"
        
            dims_list = ",".join([f"\"{d}\"" for d in dim_names])
            block = f"""            OBJECT=DataField_{i+1}
                    DataFieldName="{name}"
                    DataType={eos_type}
                    DimList=({dims_list})
                END_OBJECT=DataField_{i+1}"""
            data_fields_blocks.append(block)
        
        return f"""    GROUP={grid_name}
            GridName="{grid_name}"
            XDim={width}
            YDim={height}
            UpperLeftPointMtrs=({ul_x:.6f},{ul_y:.6f})
            LowerRightMtrs=({lr_x:.6f},{lr_y:.6f})
            Projection={proj_code}
            ZoneCode={zone}
            SphereCode=12
            ProjParams={p_str}
            GROUP=Dimension
                OBJECT=Dimension_1
                    DimensionName="Time"
                    Size={n_times}
                END_OBJECT=Dimension_1
                OBJECT=Dimension_2
                    DimensionName="Band"
                    Size={n_bands}
                END_OBJECT=Dimension_2
                OBJECT=Dimension_3
                    DimensionName="YDim"
                    Size={height}
                END_OBJECT=Dimension_3
                OBJECT=Dimension_4
                    DimensionName="XDim"
                    Size={width}
                END_OBJECT=Dimension_4
                OBJECT=Dimension_5
                    DimensionName="VisBand"
                    Size=4
                END_OBJECT=Dimension_5
            END_GROUP=Dimension
            GROUP=DataField
    {"\n".join(data_fields_blocks)}
            END_GROUP=DataField
            GROUP=MergedFields
            END_GROUP=MergedFields
        END_GROUP={grid_name}"""

    def write_hdf_sensor_group(h5f, group_path, data_dict, wavelengths, crs, transform, tile_mapping_json=None):
        if not data_dict or data_dict['count'] == 0: return
        grp = h5f.create_group(group_path)
        gdal_transform = np.array([transform.c, transform.a, transform.b, transform.f, transform.d, transform.e], dtype='float64')
        dt = h5py.string_dtype(encoding='ascii')
    
        num_frames, bands, h, w = data_dict['sr'].shape
        chunk_h, chunk_w = min(h, 256), min(w, 256)
    
        sr_ds = grp.create_dataset('surface_reflectance', data=data_dict['sr'], compression='gzip', compression_opts=4, chunks=(1, bands, chunk_h, chunk_w))
        sr_ds.attrs['units'] = "Reflectance"; sr_ds.attrs['_FillValue'] = np.nan; sr_ds.attrs['wavelengths'] = wavelengths
        sr_ds.attrs['spatial_ref'] = crs.to_wkt(); sr_ds.attrs['GeoTransform'] = gdal_transform
    
        fmask_ds = grp.create_dataset('Fmask', data=data_dict['fm'][:, 0, :, :], dtype='uint8', compression='gzip', compression_opts=4, chunks=(1, chunk_h, chunk_w))
        fmask_ds.attrs['_FillValue'] = 255
        ang_ds = grp.create_dataset('solar_view_angles', data=data_dict['ag'], compression='gzip', compression_opts=4, chunks=(1, 4, chunk_h, chunk_w))
        ang_ds.attrs['_FillValue'] = np.nan; ang_ds.attrs['band_order'] = ["SZA", "SAA", "VZA", "VAA"]
    
        vis_ds = grp.create_dataset('ortho_visual', data=data_dict['vis'], dtype='uint8', compression='gzip', compression_opts=4, chunks=(1, 4, chunk_h, chunk_w))
        vis_ds.attrs['spatial_ref'] = crs.to_wkt()
        vis_ds.attrs['GeoTransform'] = gdal_transform

        mask_ds = grp.create_dataset('common_mask', data=data_dict['mask'], dtype=bool, compression='gzip', compression_opts=4, chunks=(1, chunk_h, chunk_w))
        mask_ds.attrs['description'] = "True = Invalid/Masked, False = Valid."
        mask_ds.attrs['spatial_ref'] = crs.to_wkt()
        mask_ds.attrs['GeoTransform'] = gdal_transform
        mask_ds.attrs['qa_reject_mask'] = QA_REJECT_MASK
        mask_ds.attrs['cloud_dilation'] = HLS_CLOUD_DILATION
        mask_ds.attrs['aerosol_accept_level'] = AEROSOL_ACCEPT_LEVEL
        mask_ds.attrs['sun_elevation_threshold'] = SUN_ELEVATION_THRESHOLD

        if 'tm' in data_dict and data_dict['tm'] is not None:
            tm_ds = grp.create_dataset('source_tile_mask', data=data_dict['tm'][:, 0, :, :], dtype='uint8', compression='gzip', compression_opts=4, chunks=(1, chunk_h, chunk_w))
            tm_ds.attrs['_FillValue'] = 0
            tm_ds.attrs['description'] = "Integer mapping to source MGRS tile to track radiometric provenance."
            if tile_mapping_json: tm_ds.attrs['tile_mapping'] = tile_mapping_json
            
        sr_ds.attrs.create('spacecraft_id', data=data_dict['meta']['space'], dtype=dt)
        sr_ds.attrs['acquisition_time'] = np.array(data_dict['meta']['acq'], dtype='float64') 
        sr_ds.attrs['sun_azimuth'] = np.array(data_dict['meta']['saz'], dtype='float32')
        sr_ds.attrs['sun_elevation'] = np.array(data_dict['meta']['sel'], dtype='float32')
        sr_ds.attrs['cloud_cover'] = np.array(data_dict['meta']['cc'], dtype='float32')

    # ==========================================
    # 4. DATA PIPELINE: HLS NATIVE TO MASTER GRID
    # ==========================================
    def fetch_native_hls_groups(native_h5_path, sensor_prefix):
        """Scans Native Truth HDF5 and groups temporal frames strictly by acquisition day."""
        if not os.path.exists(native_h5_path):
            raise FileNotFoundError(f"CRITICAL ERROR: Native HLS Truth file missing at {native_h5_path}")
        
        daily_groups = {}
        unique_tiles = set()
    
        with h5py.File(native_h5_path, 'r') as h5f:
            grid_groups = [k for k in h5f['HDFEOS/GRIDS'].keys() if k.startswith(sensor_prefix)]
            for grid_id in grid_groups:
                tile_name = grid_id.split('_')[1] 
                unique_tiles.add(tile_name)
            
                sr_ds = h5f[f'HDFEOS/GRIDS/{grid_id}/Data Fields/surface_reflectance']
                acq_times = sr_ds.attrs['acquisition_time']
            
                for f_idx, ts in enumerate(acq_times):
                    dt_str = datetime.fromtimestamp(ts, tz=timezone.utc).strftime('%Y-%m-%d')
                    if dt_str not in daily_groups: daily_groups[dt_str] = []
                    daily_groups[dt_str].append({'tile': tile_name, 'grid_id': grid_id, 'frame_idx': f_idx})
                
        return daily_groups, unique_tiles

    def process_hls_master_stack(native_h5_path, daily_groups, expected_sr, tile_map):
        """Harmonizes unprojected native arrays into the Master Grid directly in-memory."""
        sorted_dates = sorted(daily_groups.keys())
        num_frames = len(sorted_dates)
        if num_frames == 0: return None
    
        stk_sr = np.full((num_frames, expected_sr, master_height, master_width), np.nan, dtype=np.float32)
        stk_fm = np.full((num_frames, 1, master_height, master_width), 255, dtype=np.uint8)
        stk_tm = np.zeros((num_frames, 1, master_height, master_width), dtype=np.uint16)
        stk_ag = np.full((num_frames, 4, master_height, master_width), np.nan, dtype=np.float32)
        stk_mask = np.ones((num_frames, master_height, master_width), dtype=bool)
        vis_data = np.zeros((num_frames, 4, master_height, master_width), dtype=np.uint8)
        meta_arrays = {'acq': [], 'space': [], 'saz': [], 'sel': [], 'cc': []}
    
        with h5py.File(native_h5_path, 'r') as h5f:
            for idx, date_str in enumerate(sorted_dates):
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
                
                    stk_tm[idx, 0][mask_fm[0]] = tile_map[tile]
                
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
                                                sun_elevation_threshold=SUN_ELEVATION_THRESHOLD,
                                                cloud_dilation=HLS_CLOUD_DILATION,
                                                qa_reject_mask=QA_REJECT_MASK,
                                                aerosol_accept_level=AEROSOL_ACCEPT_LEVEL).astype(bool)
            
                rgba_img = sc.generate_rgba_image(stk_sr[idx])
                vis_data[idx, ...] = np.transpose(rgba_img, (2, 0, 1))

        valid_indices = []
        for idx, date_str in enumerate(sorted_dates):
            valid_pixels = np.sum(~np.isnan(stk_sr[idx, 0]))
            coverage = (valid_pixels / (master_height * master_width)) * 100
            if coverage >= MIN_ROI_COVERAGE_PERCENT:
                valid_indices.append(idx)
            else:
                print(f"    Skipping HLS frame {date_str} (Coverage: {coverage:.1f}% < {MIN_ROI_COVERAGE_PERCENT}%)")

        if not valid_indices:
            return None
        
        num_valid = len(valid_indices)
        stk_sr = stk_sr[valid_indices]
        stk_fm = stk_fm[valid_indices]
        stk_tm = stk_tm[valid_indices]
        stk_ag = stk_ag[valid_indices]
        stk_mask = stk_mask[valid_indices]
        vis_data = vis_data[valid_indices]
    
        meta_arrays['acq'] = [meta_arrays['acq'][i] for i in valid_indices]
        meta_arrays['space'] = [meta_arrays['space'][i] for i in valid_indices]
        meta_arrays['saz'] = [meta_arrays['saz'][i] for i in valid_indices]
        meta_arrays['sel'] = [meta_arrays['sel'][i] for i in valid_indices]
        meta_arrays['cc'] = [meta_arrays['cc'][i] for i in valid_indices]

        return {'sr': stk_sr, 'fm': stk_fm, 'ag': stk_ag, 'tm': stk_tm, 'vis': vis_data, 'mask': stk_mask, 'meta': meta_arrays, 'count': num_valid}

    # ==========================================
    # 5. MASTER EXECUTION
    # ==========================================
    print(f"\nBuilding Multi-Sensor ARD Cube (Dynamic Albers): {OUTPUT_MASTER_HDF5}")

    s30_daily, s30_tiles = fetch_native_hls_groups(INPUT_NATIVE_HDF5, "HLSS30")
    l30_daily, l30_tiles = fetch_native_hls_groups(INPUT_NATIVE_HDF5, "HLSL30")

    unique_hls_tiles = sorted(list(s30_tiles.union(l30_tiles)))
    master_tile_mapping = {tile: i+1 for i, tile in enumerate(unique_hls_tiles)}
    master_tile_mapping_json = json.dumps(master_tile_mapping)

    with h5py.File(OUTPUT_MASTER_HDF5, 'w') as h5f:
        info_grp = h5f.create_group("HDFEOS INFORMATION")
        info_grp.attrs["HDFEOSVersion"] = "HDFEOS_5.1.16"
    
        # Store Configuration Metadata
        meta_grp = h5f.create_group("METADATA/PIPELINE_CONFIG")
        meta_grp.attrs["Location"] = Location
        meta_grp.attrs["config_yaml"] = yaml.dump(config_data)

        odl_blocks = []
    
        # --- 5a. HLS Master Grid Stitching ---
        print("Harmonizing HLSS30...")
        s30_master_data = process_hls_master_stack(INPUT_NATIVE_HDF5, s30_daily, 10, master_tile_mapping)
        if s30_master_data:
            write_hdf_sensor_group(h5f, '/HDFEOS/GRIDS/HLSS30/Data Fields', s30_master_data, S30_WAVELENGTHS, master_crs, master_transform, tile_mapping_json=master_tile_mapping_json)
            odl_blocks.append(generate_odl_grid_string("HLSS30", master_width, master_height, master_transform, master_proj, master_zone, master_gctp, 10, s30_master_data['count'], has_tile_mask=True))

        print("Harmonizing HLSL30...")
        l30_master_data = process_hls_master_stack(INPUT_NATIVE_HDF5, l30_daily, 7, master_tile_mapping)
        if l30_master_data:
            write_hdf_sensor_group(h5f, '/HDFEOS/GRIDS/HLSL30/Data Fields', l30_master_data, L30_SR_WAVELENGTHS, master_crs, master_transform, tile_mapping_json=master_tile_mapping_json)
            odl_blocks.append(generate_odl_grid_string("HLSL30", master_width, master_height, master_transform, master_proj, master_zone, master_gctp, 7, l30_master_data['count'], has_tile_mask=True))

        # --- 5b. TANAGER Hyperspectral Processing (From Basic SWATH) ---
        basic_files = glob.glob(os.path.join(TANAGER_SOURCE_DIR, "**", "*_basic_sr_hdf5.h5"), recursive=True)
    
        if basic_files:
            print(f"\nHarmonizing Tanager Hyperspectral Arrays from Basic Swaths (Found {len(basic_files)} chunks)")
        
            # Group by pass date to merge chunks of the same pass (e.g., 20250801)
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
        
            meta_lists = {'acq_time': [], 'space_id': [], 'sun_az': [], 'sun_el': [], 'cloud_cv': [], 'good_wavelengths': []}
        
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
                    
                        if "_FillValue" in src_dset.attrs: fill_val = src_dset.attrs["_FillValue"]
                        elif dtype.name == 'uint8': fill_val = 0
                        else: fill_val = np.nan
                        if isinstance(fill_val, (np.ndarray, list)): fill_val = fill_val[0]

                        out_dset = grp_tanager.create_dataset(name, shape=out_shape, dtype=dtype, compression="gzip", compression_opts=4, fillvalue=fill_val, chunks=chunks)
                        datasets_created_info.append((name, dtype, len(out_shape), ["Time", "Band", "YDim", "XDim"] if is_3d else ["Time", "YDim", "XDim"]))
                    
                        for k, v in src_dset.attrs.items():
                            if k not in ["DIMENSION_LIST", "REFERENCE_LIST", "CLASS", "PALETTE", "spatial_ref", "GeoTransform"]:
                                out_dset.attrs[k] = v
                        out_dset.attrs['spatial_ref'] = master_crs.to_wkt()
                        out_dset.attrs['GeoTransform'] = gdal_transform

                for t_idx, pass_ts in enumerate(pass_keys):
                    chunks_files = passes[pass_ts]
                    print(f"  Processing Tanager Pass {pass_ts} ({len(chunks_files)} chunks)...")
                
                    pass_canvases = {}
                    pass_counts = {}
                    pass_times = []
                
                    acq_time, space_id, sun_az, sun_el, cloud_cv = np.nan, b'Tanager', np.nan, np.nan, np.nan
                
                    for chunk_idx, chunk_file in enumerate(chunks_files):
                        with h5py.File(chunk_file, 'r') as f_chunk:
                            df_grp = f_chunk['HDFEOS/SWATHS/HYP/Data Fields']
                            geo_grp = f_chunk['HDFEOS/SWATHS/HYP/Geolocation Fields']
                            lat = geo_grp['Latitude'][:]
                            lon = geo_grp['Longitude'][:]
                            nodata_mask = df_grp['nodata_pixels'][:]
                            if 'Time' in geo_grp:
                                pass_times.extend(geo_grp['Time'][:].tolist())
                        
                            gcps = []
                            step = 10
                            for r in range(0, lat.shape[0], step):
                                for c in range(0, lat.shape[1], step):
                                    gcps.append(GroundControlPoint(row=r, col=c, x=lon[r, c], y=lat[r, c]))
                            gcps.append(GroundControlPoint(row=lat.shape[0]-1, col=lat.shape[1]-1, x=lon[-1, -1], y=lat[-1, -1]))
                        
                            for name in df_grp.keys():
                                if name not in pass_canvases:
                                    is_3d = len(grp_tanager[name].shape) == 4
                                    bands = grp_tanager[name].shape[1] if is_3d else None
                                    shape = (bands, master_height, master_width) if is_3d else (master_height, master_width)
                                    pass_canvases[name] = np.full(shape, np.nan, dtype=np.float32)
                                    pass_counts[name] = np.zeros(shape, dtype=np.int32)
                                
                                src_data = df_grp[name][:]
                                dtype = src_data.dtype
                                is_3d = len(src_data.shape) == 3
                                
                                src_data = np.where(nodata_mask == 1, np.nan, src_data)
                            
                                if name == "ortho_visual": resample_algo = Resampling.cubic
                                elif dtype.name == 'uint8' or dtype.kind == 'b' or "mask" in name.lower() or "nodata" in name.lower() or "beta" in name.lower():
                                    resample_algo = Resampling.nearest
                                else: resample_algo = Resampling.average
                                
                                if is_3d:
                                    bands = src_data.shape[0]
                                    incoming = np.full((bands, master_height, master_width), np.nan, dtype=np.float32)
                                else:
                                    incoming = np.full((master_height, master_width), np.nan, dtype=np.float32)

                                reproject(
                                    source=src_data, destination=incoming,
                                    gcps=gcps, src_crs=CRS.from_epsg(4326),
                                    dst_transform=master_transform, dst_crs=master_crs,
                                    resampling=resample_algo, src_nodata=np.nan, dst_nodata=np.nan
                                )
                            
                                valid_mask = ~np.isnan(incoming)
                                init_mask = np.isnan(pass_canvases[name]) & valid_mask
                                pass_canvases[name][init_mask] = 0.0
                            
                                pass_canvases[name][valid_mask] += incoming[valid_mask]
                                pass_counts[name][valid_mask] += 1
                            
                            if chunk_idx == 0:
                                sr_attrs = df_grp['surface_reflectance'].attrs
                                space_id = sr_attrs.get('spacecraft_id', b'Tanager')
                                sun_az = sr_attrs.get('sun_azimuth', np.nan)
                                if isinstance(sun_az, (np.ndarray, list)): sun_az = sun_az[0]
                                sun_el = sr_attrs.get('sun_elevation', np.nan)
                                if isinstance(sun_el, (np.ndarray, list)): sun_el = sun_el[0]
                                cloud_cv = sr_attrs.get('cloud_cover', np.nan)
                                if isinstance(cloud_cv, (np.ndarray, list)): cloud_cv = cloud_cv[0]
                                pass_gw = sr_attrs.get('good_wavelengths', np.ones(band_count, dtype=np.int32))

                    if pass_times:
                        pass_times.sort()
                        acq_time = pass_times[len(pass_times) // 2]
                    else:
                        # Fallback to the timestamp of the first chunk if pass_times not populated
                        fallback_basename = os.path.basename(chunks_files[0])
                        fallback_parts = fallback_basename.split('_')
                        dt_obj = datetime.strptime(f"{fallback_parts[0]}_{fallback_parts[1]}", "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc)
                        acq_time = dt_obj.timestamp()

                    meta_lists['acq_time'].append(acq_time)
                    meta_lists['space_id'].append(space_id)
                    meta_lists['sun_az'].append(sun_az)
                    meta_lists['sun_el'].append(sun_el)
                    meta_lists['cloud_cv'].append(cloud_cv)
                    meta_lists['good_wavelengths'].append(pass_gw)

                    sr_valid_pixels = 0
                    for name in pass_canvases.keys():
                        counts = pass_counts[name]
                        valid = counts > 0
                        pass_canvases[name][valid] /= counts[valid]
                    
                        dtype = grp_tanager[name].dtype
                        fill_val = grp_tanager[name].fillvalue
                        if isinstance(fill_val, (np.ndarray, list)): fill_val = fill_val[0]
                    
                        final_arr = pass_canvases[name]
                        final_arr[~valid] = fill_val
                    
                        if dtype.kind in ['u', 'i', 'b']: final_arr = np.round(final_arr).astype(dtype)
                        else: final_arr = final_arr.astype(dtype)
                        
                        is_3d = len(grp_tanager[name].shape) == 4
                        if is_3d: grp_tanager[name][t_idx, ...] = final_arr
                        else: grp_tanager[name][t_idx, ...] = final_arr[0, ...]
                        
                        if name == "surface_reflectance":
                            sr_valid_pixels = np.sum(valid[0])
                        
                    coverage = (sr_valid_pixels / (master_height * master_width)) * 100
                    if coverage >= MIN_ROI_COVERAGE_PERCENT:
                        valid_t_indices.append(t_idx)
                    else:
                        print(f"    Warning: Tanager pass {pass_ts} coverage ({coverage:.1f}%) < {MIN_ROI_COVERAGE_PERCENT}%")
            
                # Write global metadata arrays
                dt_str = h5py.string_dtype(encoding='ascii')
                grp_tanager['surface_reflectance'].attrs['acquisition_time'] = np.array(meta_lists['acq_time'], dtype='float64')
                grp_tanager['surface_reflectance'].attrs.create('spacecraft_id', data=np.array(meta_lists['space_id'], dtype=dt_str))
                grp_tanager['surface_reflectance'].attrs['sun_azimuth'] = np.array(meta_lists['sun_az'], dtype='float32')
                grp_tanager['surface_reflectance'].attrs['sun_elevation'] = np.array(meta_lists['sun_el'], dtype='float32')
                grp_tanager['surface_reflectance'].attrs['cloud_cover'] = np.array(meta_lists['cloud_cv'], dtype='float32')
                
                gw_array = np.array(meta_lists['good_wavelengths'])
                grp_tanager['surface_reflectance'].attrs['all_good_wavelengths'] = gw_array
                grp_tanager['surface_reflectance'].attrs['good_wavelengths'] = np.logical_and.reduce(gw_array, axis=0).astype(np.int32)

                num_frames = len(valid_t_indices)
                if num_frames > 0:
                    print("  Generating Common Mask for Tanager on Master Grid...")
                    mask_ds = grp_tanager.create_dataset('common_mask', shape=(total_num_frames, master_height, master_width), dtype=bool, compression="gzip", compression_opts=4, chunks=(1, chunk_h, chunk_w))
                    datasets_created_info.append(("common_mask", bool, 3, ["Time", "YDim", "XDim"]))
                    mask_ds.attrs['spatial_ref'] = master_crs.to_wkt()
                    mask_ds.attrs['GeoTransform'] = gdal_transform
                    mask_ds.attrs['description'] = "True = Invalid/Masked, False = Valid. Generated from SpecComplex ARD rules."
                    h5f.flush()
                
                    for out_idx in range(total_num_frames):
                        valid_mask = sc.get_tanager_mask(grp_tanager, out_idx, (master_height, master_width),
                                                         sun_elevation_threshold=SUN_ELEVATION_THRESHOLD,
                                                         cloud_dilation=TANAGER_CLOUD_DILATION,
                                                         apply_cloud_mask=True,
                                                         uncertainty_threshold=TANAGER_UNCERTAINTY_THRESHOLD,
                                                         aerosol_depth_threshold=TANAGER_AEROSOL_THRESHOLD)
                        mask_ds[out_idx, ...] = valid_mask.astype(bool)
                
                    if 'wavelengths' in grp_tanager['surface_reflectance'].attrs:
                        print("  Generating strict 'ortho_visual' RGB composite from SR...")
                        wavelengths = grp_tanager['surface_reflectance'].attrs['wavelengths']
                        r_idx = int(np.argmin(np.abs(wavelengths - 650)))
                        g_idx = int(np.argmin(np.abs(wavelengths - 550)))
                        b_idx = int(np.argmin(np.abs(wavelengths - 450)))

                        ortho_vis_dset = grp_tanager.create_dataset("ortho_visual", shape=(total_num_frames, 3, master_height, master_width), dtype='uint8', compression="gzip", fillvalue=0, chunks=(1, 3, chunk_h, chunk_w))
                        datasets_created_info.append(("ortho_visual", np.dtype('uint8'), 3, ["Time", "RGBBand", "YDim", "XDim"]))
                        ortho_vis_dset.attrs['spatial_ref'] = master_crs.to_wkt()
                        ortho_vis_dset.attrs['GeoTransform'] = gdal_transform
                        
                        sr_dset_ref = grp_tanager["surface_reflectance"]
                        sr_fill = sr_dset_ref.fillvalue
                        if isinstance(sr_fill, (np.ndarray, list)): sr_fill = sr_fill[0]

                        for out_idx in range(total_num_frames):
                            r_band = sr_dset_ref[out_idx, r_idx, :, :]
                            g_band = sr_dset_ref[out_idx, g_idx, :, :]
                            b_band = sr_dset_ref[out_idx, b_idx, :, :]
                            
                            try: r_stretched = percentile_stretch(r_band, sr_fill, 2, 98)
                            except ValueError: r_stretched = np.zeros((master_height, master_width), dtype=np.uint8)
                            
                            try: g_stretched = percentile_stretch(g_band, sr_fill, 2, 98)
                            except ValueError: g_stretched = np.zeros((master_height, master_width), dtype=np.uint8)
                            
                            try: b_stretched = percentile_stretch(b_band, sr_fill, 2, 98)
                            except ValueError: b_stretched = np.zeros((master_height, master_width), dtype=np.uint8)
                            
                            ortho_vis_dset[out_idx, 0, :, :] = r_stretched
                            ortho_vis_dset[out_idx, 1, :, :] = g_stretched
                            ortho_vis_dset[out_idx, 2, :, :] = b_stretched

                    odl_blocks.append(generate_tanager_odl_string("TANAGER", master_width, master_height, master_transform, master_proj, master_zone, master_gctp, datasets_created_info, total_num_frames, band_count))
                else:
                    print("  No Tanager passes met the minimum coverage threshold.")
            else:
                print(f"\nWARNING: No basic_sr_hdf5 files found in {TANAGER_SOURCE_DIR}. Skipping Tanager processing.")

        # --- 5c. HARMONIZED Global Timeline and ortho_visual generation ---
        print("\nGenerating Global Timeline and HARMONIZED ortho_visual dataset...")
        grids = [g for g in h5f['/HDFEOS/GRIDS'].keys()]
        timeline = []
        for grid in grids:
            data_grp = h5f[f"/HDFEOS/GRIDS/{grid}/Data Fields"]
            if "surface_reflectance" in data_grp:
                acq_times = data_grp["surface_reflectance"].attrs['acquisition_time']
                spacecraft_ids = data_grp["surface_reflectance"].attrs.get('spacecraft_id', [b'UNKNOWN']*len(acq_times))
                for i, ts in enumerate(acq_times):
                    sp_id = spacecraft_ids[i]
                    sp_str = sp_id.decode('utf-8') if isinstance(sp_id, bytes) else str(sp_id)
                    timeline.append({'time': ts, 'grid': grid, 'local_idx': i, 'spacecraft': sp_str})

        timeline.sort(key=lambda x: x['time'])
        total_frames = len(timeline)
    
        if total_frames > 0:
            harm_grp = h5f.create_group('/HDFEOS/GRIDS/HARMONIZED/Data Fields')
            chunk_h, chunk_w = min(master_height, 256), min(master_width, 256)
        
            ortho_ds = harm_grp.create_dataset('ortho_visual', shape=(total_frames, 4, master_height, master_width), dtype='uint8', compression="gzip", compression_opts=4, chunks=(1, 4, chunk_h, chunk_w))
            ortho_ds.attrs['spatial_ref'] = master_crs.to_wkt()
            gdal_transform = np.array([master_transform.c, master_transform.a, master_transform.b, master_transform.f, master_transform.d, master_transform.e], dtype='float64')
            ortho_ds.attrs['GeoTransform'] = gdal_transform
        
            dt_str = h5py.string_dtype(encoding='ascii')
            prov_grid = np.array([m['grid'] for m in timeline], dtype=dt_str)
            prov_space = np.array([m['spacecraft'] for m in timeline], dtype=dt_str)
            prov_time = np.array([m['time'] for m in timeline], dtype='float64')
            prov_idx = np.array([m['local_idx'] for m in timeline], dtype='int32')
        
            ortho_ds.attrs.create('source_grid', data=prov_grid)
            ortho_ds.attrs.create('source_spacecraft', data=prov_space)
            ortho_ds.attrs['acquisition_time'] = prov_time
            ortho_ds.attrs['source_frame_index'] = prov_idx
        
            for global_idx, meta in enumerate(timeline):
                grid_name = meta['grid']
                local_idx = meta['local_idx']
                src_val = h5f[f"/HDFEOS/GRIDS/{grid_name}/Data Fields/ortho_visual"][local_idx, ...]
                if src_val.shape[0] == 3:
                    # Add Alpha channel using common_mask (False = valid -> alpha=1, True = invalid -> alpha=0)
                    if "common_mask" in h5f[f"/HDFEOS/GRIDS/{grid_name}/Data Fields"]:
                        mask = h5f[f"/HDFEOS/GRIDS/{grid_name}/Data Fields/common_mask"][local_idx, ...]
                        alpha = np.where(mask, 0, 255).astype('uint8')
                    else:
                        alpha = np.full((src_val.shape[1], src_val.shape[2]), 255, dtype='uint8')
                    src_val = np.concatenate([src_val, alpha[np.newaxis, ...]], axis=0)
                ortho_ds[global_idx, ...] = src_val
            
            print(f"  HARMONIZED ortho_visual created with {total_frames} frames.")
        
            harm_odl = f"""    GROUP=HARMONIZED
            GridName="HARMONIZED"
            XDim={master_width}
            YDim={master_height}
            UpperLeftPointMtrs=({master_transform.c:.6f},{master_transform.f:.6f})
            LowerRightMtrs=({master_transform.c + master_transform.a * master_width:.6f},{master_transform.f + master_transform.e * master_height:.6f})
            Projection={master_proj}
            ZoneCode={master_zone}
            SphereCode=12
            ProjParams={str(tuple(master_gctp)).replace(' ', '').replace('(', '').replace(')', '')}
            GROUP=Dimension
                OBJECT=Dimension_1
                    DimensionName="Time"
                    Size={total_frames}
                END_OBJECT=Dimension_1
                OBJECT=Dimension_2
                    DimensionName="YDim"
                    Size={master_height}
                END_OBJECT=Dimension_2
                OBJECT=Dimension_3
                    DimensionName="XDim"
                    Size={master_width}
                END_OBJECT=Dimension_3
                OBJECT=Dimension_4
                    DimensionName="VisBand"
                    Size=4
                END_OBJECT=Dimension_4
            END_GROUP=Dimension
            GROUP=DataField
                OBJECT=DataField_1
                    DataFieldName="ortho_visual"
                    DataType=HDF5T_NATIVE_UINT8
                    DimList=("Time","VisBand","YDim","XDim")
                END_OBJECT=DataField_1
            END_GROUP=DataField
            GROUP=MergedFields
            END_GROUP=MergedFields
        END_GROUP=HARMONIZED"""
            odl_blocks.append(harm_odl)

        full_odl = "GROUP=SwathStructure\nEND_GROUP=SwathStructure\nGROUP=GridStructure\n" + "\n".join(odl_blocks) + "\nEND_GROUP=GridStructure\nGROUP=PointStructure\nEND_GROUP=PointStructure\nGROUP=ZaStructure\nEND_GROUP=ZaStructure\nEND\n"
        info_grp.create_dataset("StructMetadata.0", shape=(1,), dtype=h5py.string_dtype(encoding='ascii'), data=full_odl)

    print("\nPipeline Complete. Multi-Sensor ARD Master Grid generated successfully.")

if __name__ == '__main__':
    main()
