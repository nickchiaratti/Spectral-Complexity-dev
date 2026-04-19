import os
import h5py
import rasterio
import numpy as np
from datetime import datetime, timezone
from rasterio.transform import from_bounds as transform_from_bounds
from rasterio.warp import reproject, Resampling
from pyproj import Transformer, CRS
from rasterio.transform import Affine
from pathlib import Path
import json
import warnings
import SpecComplex as sc

# ==========================================
# 1. CONFIGURATION & DIRECTORIES
# ==========================================
Location = "MtEtna"

if Location == "Rochesterv2":
    ROI_LON_MIN = -77.770166; ROI_LON_MAX = -77.376776
    ROI_LAT_MIN = 42.961778; ROI_LAT_MAX = 43.342135
elif Location == "Tait":
    ROI_LON_MIN = -77.516127; ROI_LON_MAX = -77.461968
    ROI_LAT_MIN = 43.127698; ROI_LAT_MAX = 43.159168
if Location == 'Guatemala-Debris':
    ROI_LON_MIN = -88.222000; ROI_LON_MAX = -87.822000
    ROI_LAT_MIN = 15.636200; ROI_LAT_MAX = 16.036200
if Location == "MtEtna":
    ROI_LON_MIN = 14.9100; ROI_LON_MAX = 15.0900
    ROI_LAT_MIN = 37.6900; ROI_LAT_MAX = 37.8300
    
HLS_SOURCE_DIR = r"C:\satelliteImagery\HLS30"
TANAGER_SOURCE_DIR = r"C:\satelliteImagery\Tanager\SourceData"
COMBINED_OUTPUT_DIR = r"C:\satelliteImagery\HLST30"

INPUT_NATIVE_HDF5 = os.path.join(HLS_SOURCE_DIR, f"HLS_{Location}_STAC_Native_2025.h5")
# New modular ingestion source for Tanager Hyperspectral data
INPUT_NATIVE_TANAGER_HDF5 = os.path.join(TANAGER_SOURCE_DIR, "SKIP")#"Tanager_Native_Stack_HDFEOS.h5")

OUTPUT_MASTER_HDF5 = os.path.join(COMBINED_OUTPUT_DIR, f"HLST_{Location}_Harmonized_2025.h5")

S30_WAVELENGTHS = [0.443, 0.490, 0.560, 0.665, 0.705, 0.740, 0.783, 0.842, 0.865, 0.945, 1.375, 1.610, 2.190]
L30_SR_WAVELENGTHS = [0.443, 0.482, 0.561, 0.655, 0.865, 1.609, 2.201, 1.373] 
L30_TIRS_WAVELENGTHS = [10.9, 12.0]

safe_bbox = [
    min(ROI_LON_MIN, ROI_LON_MAX), max(ROI_LAT_MIN, ROI_LAT_MAX), 
    max(ROI_LON_MIN, ROI_LON_MAX), min(ROI_LAT_MIN, ROI_LAT_MAX)
]
safe_bbox = [min(safe_bbox[0], safe_bbox[2]), min(safe_bbox[1], safe_bbox[3]), max(safe_bbox[0], safe_bbox[2]), max(safe_bbox[1], safe_bbox[3])]
TARGET_RESOLUTION = 30.0

# --- Pixel Mask Configuration ---
SUN_ELEVATION_THRESHOLD = 30
# HLS Specific Configuration (Unified Fmask for both S30 and L30)
# Bits 0-5: cirrus, cloud, adj cloud/shadow, cloud shadow, snow/ice, water
HLS_CLOUD_DILATION =0
QA_REJECT_MASK = 0b111111 
AEROSOL_ACCEPT_LEVEL = 'medium' # 'low' (0-1), 'medium' (0-2), 'high' (0-3)

# TANAGER Specific Configuration
TANAGER_CLOUD_DILATION = 2
TANAGER_UNCERTAINTY_THRESHOLD = 0.1
TANAGER_AEROSOL_THRESHOLD = 0.35

# ==========================================
# 2. MASTER GRID PRE-CALCULATION (USGS ARD)
# ==========================================
def calculate_master_grid(bbox, resolution):
    """Calculates a Unified Master Grid using USGS CONUS Albers Equal Area."""
    lat_1, lat_2, central_lat, central_lon = 29.5, 45.5, 23.0, -96.0
    proj_str = f"+proj=aea +lat_1={lat_1} +lat_2={lat_2} +lat_0={central_lat} +lon_0={central_lon} +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
    dst_crs = CRS.from_string(proj_str)
    
    transformer = Transformer.from_crs("EPSG:4326", dst_crs, always_xy=True)
    xs, ys = transformer.transform([bbox[0], bbox[2], bbox[2], bbox[0]], [bbox[3], bbox[3], bbox[1], bbox[1]])
    minx, maxx, miny, maxy = min(xs), max(xs), min(ys), max(ys)
    
    width = int(np.ceil((maxx - minx) / resolution))
    height = int(np.ceil((maxy - miny) / resolution))
    transform = transform_from_bounds(minx, miny, maxx, maxy, width, height)
    gctp_params = [6378137.0, 6356752.314245, lat_1, lat_2, central_lon, central_lat, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    return dst_crs, transform, width, height, "GCTP_ALBERS", 0, gctp_params

master_crs, master_transform, master_width, master_height, master_proj, master_zone, master_gctp = calculate_master_grid(safe_bbox, TARGET_RESOLUTION)
print(f"Master Grid Established: {master_width}x{master_height} at USGS CONUS Albers Equal Area")

# ==========================================
# 3. HDFEOS5 ODL GENERATORS
# ==========================================
def generate_odl_grid_string(grid_name, width, height, transform, proj_code, zone, proj_params, num_sr_bands, num_frames, has_thermal, has_tile_mask=False):
    ul_x, ul_y = transform.c, transform.f
    lr_x = transform.c + (transform.a * width)
    lr_y = transform.f + (transform.e * height)
    p_str = str(tuple(proj_params)).replace(' ', '').replace('(', '').replace(')', '')
    
    thermal_dim = f"""            OBJECT=Dimension_3\n                DimensionName="ThermalBands"\n                Size=2\n            END_OBJECT=Dimension_3""" if has_thermal else ""
    
    fields = []
    idx = 1
    fields.append(f"""            OBJECT=DataField_{idx}\n                DataFieldName="surface_reflectance"\n                DataType=HDF5T_NATIVE_FLOAT\n                DimList=("Time","Bands","YDim","XDim")\n            END_OBJECT=DataField_{idx}""")
    idx += 1
    if has_thermal:
        fields.append(f"""            OBJECT=DataField_{idx}\n                DataFieldName="thermal_infrared"\n                DataType=HDF5T_NATIVE_FLOAT\n                DimList=("Time","ThermalBands","YDim","XDim")\n            END_OBJECT=DataField_{idx}""")
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
    fields.append(f"""            OBJECT=DataField_{idx}\n                DataFieldName="common_mask"\n                DataType=HDF5T_NATIVE_UINT8\n                DimList=("Time","YDim","XDim")\n            END_OBJECT=DataField_{idx}""")
    
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
{thermal_dim}
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

def write_hdf_sensor_group(h5f, group_path, data_dict, wavelengths, crs, transform, thermal_wavelengths=None, tile_mapping_json=None):
    if not data_dict or data_dict['count'] == 0: return
    grp = h5f.create_group(group_path)
    gdal_transform = np.array([transform.c, transform.a, transform.b, transform.f, transform.d, transform.e], dtype='float64')
    dt = h5py.string_dtype(encoding='ascii')
    
    num_frames, bands, h, w = data_dict['sr'].shape
    chunk_h, chunk_w = min(h, 256), min(w, 256)
    
    sr_ds = grp.create_dataset('surface_reflectance', data=data_dict['sr'], compression='gzip', compression_opts=4, chunks=(1, bands, chunk_h, chunk_w))
    sr_ds.attrs['units'] = "Reflectance"; sr_ds.attrs['_FillValue'] = np.nan; sr_ds.attrs['wavelengths'] = wavelengths
    sr_ds.attrs['spatial_ref'] = crs.to_wkt(); sr_ds.attrs['GeoTransform'] = gdal_transform
    
    if thermal_wavelengths and data_dict['th'] is not None:
        th_bands = data_dict['th'].shape[1]
        th_ds = grp.create_dataset('thermal_infrared', data=data_dict['th'], compression='gzip', compression_opts=4, chunks=(1, th_bands, chunk_h, chunk_w))
        th_ds.attrs['units'] = "Kelvin/Celsius Apparent"; th_ds.attrs['_FillValue'] = np.nan; th_ds.attrs['wavelengths'] = thermal_wavelengths
        
    fmask_ds = grp.create_dataset('Fmask', data=data_dict['fm'][:, 0, :, :], dtype='uint8', compression='gzip', compression_opts=4, chunks=(1, chunk_h, chunk_w))
    fmask_ds.attrs['_FillValue'] = 255
    ang_ds = grp.create_dataset('solar_view_angles', data=data_dict['ag'], compression='gzip', compression_opts=4, chunks=(1, 4, chunk_h, chunk_w))
    ang_ds.attrs['_FillValue'] = np.nan; ang_ds.attrs['band_order'] = ["SZA", "SAA", "VZA", "VAA"]
    
    vis_ds = grp.create_dataset('ortho_visual', data=data_dict['vis'], dtype='uint8', compression='gzip', compression_opts=4, chunks=(1, 4, chunk_h, chunk_w))
    vis_ds.attrs['spatial_ref'] = crs.to_wkt()
    vis_ds.attrs['GeoTransform'] = gdal_transform

    mask_ds = grp.create_dataset('common_mask', data=data_dict['mask'], dtype='uint8', compression='gzip', compression_opts=4, chunks=(1, chunk_h, chunk_w))
    mask_ds.attrs['_FillValue'] = 0
    mask_ds.attrs['description'] = "0 = Invalid/Masked, 1 = Valid. Generated from SpecComplex ARD rules."
    mask_ds.attrs['spatial_ref'] = crs.to_wkt()
    mask_ds.attrs['GeoTransform'] = gdal_transform

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

def process_hls_master_stack(native_h5_path, daily_groups, expected_sr, expected_thermal, tile_map):
    """Harmonizes unprojected native arrays into the Master Grid directly in-memory."""
    sorted_dates = sorted(daily_groups.keys())
    num_frames = len(sorted_dates)
    if num_frames == 0: return None
    
    stk_sr = np.full((num_frames, expected_sr, master_height, master_width), np.nan, dtype=np.float32)
    stk_th = np.full((num_frames, expected_thermal, master_height, master_width), np.nan, dtype=np.float32) if expected_thermal > 0 else None
    stk_fm = np.full((num_frames, 1, master_height, master_width), 255, dtype=np.uint8)
    stk_tm = np.full((num_frames, 1, master_height, master_width), 0, dtype=np.uint8)
    stk_ag = np.full((num_frames, 4, master_height, master_width), np.nan, dtype=np.float32)
    stk_mask = np.zeros((num_frames, master_height, master_width), dtype=np.uint8)
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
                
                if expected_thermal > 0:
                    src_th = h5f[f'{df_path}/thermal_infrared'][fidx]
                    tmp_th = np.full((expected_thermal, master_height, master_width), np.nan, dtype=np.float32)
                    reproject(source=src_th, destination=tmp_th, src_transform=src_tf, src_crs=src_crs, dst_transform=master_transform, dst_crs=master_crs, resampling=Resampling.cubic, src_nodata=np.nan, dst_nodata=np.nan)
                    mask_th = ~np.isnan(tmp_th)
                    stk_th[idx][mask_th] = tmp_th[mask_th]

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
                                            aerosol_accept_level=AEROSOL_ACCEPT_LEVEL).astype(np.uint8)
            
            #try:
            rgba_img = sc.generate_rgba_image(stk_sr[idx])
            vis_data[idx, ...] = np.transpose(rgba_img, (2, 0, 1))
            #except Exception:
            #    vis_data[idx, 0] = np.clip(np.nan_to_num(stk_sr[idx, 3]) * 255 * 3, 0, 255).astype(np.uint8) 
            #    vis_data[idx, 1] = np.clip(np.nan_to_num(stk_sr[idx, 2]) * 255 * 3, 0, 255).astype(np.uint8) 
            #    vis_data[idx, 2] = np.clip(np.nan_to_num(stk_sr[idx, 1]) * 255 * 3, 0, 255).astype(np.uint8) 
            #    vis_data[idx, 3] = 255

    return {'sr': stk_sr, 'th': stk_th, 'fm': stk_fm, 'ag': stk_ag, 'tm': stk_tm, 'vis': vis_data, 'mask': stk_mask, 'meta': meta_arrays, 'count': num_frames}

# ==========================================
# 5. MASTER EXECUTION
# ==========================================
print(f"\nBuilding Multi-Sensor ARD Cube (CONUS Albers): {OUTPUT_MASTER_HDF5}")

s30_daily, s30_tiles = fetch_native_hls_groups(INPUT_NATIVE_HDF5, "HLSS30")
l30_daily, l30_tiles = fetch_native_hls_groups(INPUT_NATIVE_HDF5, "HLSL30")

unique_hls_tiles = sorted(list(s30_tiles.union(l30_tiles)))
master_tile_mapping = {tile: i+1 for i, tile in enumerate(unique_hls_tiles)}
master_tile_mapping_json = json.dumps(master_tile_mapping)

with h5py.File(OUTPUT_MASTER_HDF5, 'w') as h5f:
    info_grp = h5f.create_group("HDFEOS INFORMATION")
    info_grp.attrs["HDFEOSVersion"] = "HDFEOS_5.1.16"
    odl_blocks = []
    
    # --- 5a. HLS Master Grid Stitching ---
    print("Harmonizing HLSS30...")
    s30_master_data = process_hls_master_stack(INPUT_NATIVE_HDF5, s30_daily, 13, 0, master_tile_mapping)
    if s30_master_data:
        write_hdf_sensor_group(h5f, '/HDFEOS/GRIDS/HLSS30/Data Fields', s30_master_data, S30_WAVELENGTHS, master_crs, master_transform, tile_mapping_json=master_tile_mapping_json)
        odl_blocks.append(generate_odl_grid_string("HLSS30", master_width, master_height, master_transform, master_proj, master_zone, master_gctp, 13, s30_master_data['count'], False, has_tile_mask=True))

    print("Harmonizing HLSL30...")
    l30_master_data = process_hls_master_stack(INPUT_NATIVE_HDF5, l30_daily, 8, 2, master_tile_mapping)
    if l30_master_data:
        write_hdf_sensor_group(h5f, '/HDFEOS/GRIDS/HLSL30/Data Fields', l30_master_data, L30_SR_WAVELENGTHS, master_crs, master_transform, L30_TIRS_WAVELENGTHS, tile_mapping_json=master_tile_mapping_json)
        odl_blocks.append(generate_odl_grid_string("HLSL30", master_width, master_height, master_transform, master_proj, master_zone, master_gctp, 8, l30_master_data['count'], True, has_tile_mask=True))

    # --- 5b. TANAGER Hyperspectral Processing (From Native Stack) ---
    if os.path.exists(INPUT_NATIVE_TANAGER_HDF5):
        print(f"\nHarmonizing Tanager Hyperspectral Arrays from Pre-Compiled Native Stack: {INPUT_NATIVE_TANAGER_HDF5}")
        with h5py.File(INPUT_NATIVE_TANAGER_HDF5, 'r') as f_tan_native:
            src_grp = f_tan_native["HDFEOS/GRIDS/TANAGER/Data Fields"]
            num_frames = src_grp["surface_reflectance"].shape[0]
            band_count = src_grp["surface_reflectance"].shape[1]

            if num_frames > 0:
                datasets_created_info = []
                grp_tanager = h5f.create_group("HDFEOS/GRIDS/TANAGER/Data Fields")
                
                # Clone METADATA group to preserve deep JSON origin provenance 
                if "METADATA" in f_tan_native:
                    h5f.copy(f_tan_native["METADATA"], "METADATA_TANAGER")

                chunk_h, chunk_w = min(master_height, 256), min(master_width, 256)
                gdal_transform = np.array([master_transform.c, master_transform.a, master_transform.b, master_transform.f, master_transform.d, master_transform.e], dtype='float64')

                # Dynamically iterate and reproject all Native datasets (including pre-stretched ortho_visual)
                for name in src_grp.keys():
                    src_dset = src_grp[name]
                    dtype = src_dset.dtype
                    is_3d = len(src_dset.shape) == 4 # (Time, Bands, Y, X)
                    bands = src_dset.shape[1] if is_3d else None
                    
                    out_shape = (num_frames, bands, master_height, master_width) if is_3d else (num_frames, master_height, master_width)
                    chunks = (1, bands, chunk_h, chunk_w) if is_3d else (1, chunk_h, chunk_w)

                    # EVIDENCE-BASED FIX: Intrinsic vs Attribute Fill Value Resolution
                    # Resolves HDF5 intrinsic fill properties vs GDAL user-defined attributes, 
                    # while maintaining strict data purity requirements for scientific float arrays.
                    if "_FillValue" in src_dset.attrs:
                        fill_val = src_dset.attrs["_FillValue"]
                    elif src_dset.fillvalue is not None:
                        fill_val = src_dset.fillvalue
                    elif dtype.name == 'uint8':
                        fill_val = 0  # Standard background for 8-bit visual and binary mask arrays
                    else:
                        raise AttributeError(f"CRITICAL ERROR: Scientific Dataset '{name}' missing fill value in Native Tanager Stack.")
                        
                    if isinstance(fill_val, (np.ndarray, list)): fill_val = fill_val[0]

                    print(f"  Reprojecting dataset: {name}")
                    out_dset = grp_tanager.create_dataset(name, shape=out_shape, dtype=dtype, compression="gzip", compression_opts=4, fillvalue=fill_val, chunks=chunks)
                    datasets_created_info.append((name, dtype, len(out_shape), ["Time", "Band", "YDim", "XDim"] if is_3d else ["Time", "YDim", "XDim"]))

                    # Dynamically port all scientific attributes (wavelengths, timestamps, IDs)
                    for k, v in src_dset.attrs.items():
                        if k not in ["DIMENSION_LIST", "REFERENCE_LIST", "CLASS", "PALETTE", "spatial_ref", "GeoTransform"]:
                            out_dset.attrs[k] = v
                            
                    out_dset.attrs['spatial_ref'] = master_crs.to_wkt()
                    out_dset.attrs['GeoTransform'] = gdal_transform

                    # Extract Native Georeferencing
                    src_crs_str = src_dset.attrs['spatial_ref']
                    src_crs_str = src_crs_str.decode('utf-8') if isinstance(src_crs_str, bytes) else str(src_crs_str)
                    src_crs = CRS.from_user_input(src_crs_str)
                    
                    src_gt = src_dset.attrs['GeoTransform']
                    src_tf = Affine.from_gdal(*src_gt)

                    # Strict Interpolation Rule: Nearest neighbor for masks/RGB, Cubic for continuous signals
                    resampling_algo = Resampling.nearest if dtype.name == 'uint8' else Resampling.cubic

                    for t in range(num_frames):
                        src_data = src_dset[t, ...]
                        
                        # Pad the spatial arrays with a temporary band dimension to satisfy rasterio.warp requirements
                        if not is_3d: src_data = src_data[np.newaxis, ...]
                            
                        incoming = np.full((bands if is_3d else 1, master_height, master_width), fill_val, dtype=dtype)
                        
                        reproject(
                            source=src_data, destination=incoming,
                            src_transform=src_tf, src_crs=src_crs,
                            dst_transform=master_transform, dst_crs=master_crs,
                            resampling=resampling_algo, src_nodata=fill_val, dst_nodata=fill_val
                        )
                        
                        out_dset[t, ...] = incoming if is_3d else incoming[0, ...]

                # Generate the final Harmonized 'common_mask' utilizing the master grid data
                print("  Generating Common Mask for Tanager on Master Grid...")
                mask_dset = grp_tanager.create_dataset('common_mask', shape=(num_frames, master_height, master_width), dtype='uint8', compression="gzip", compression_opts=4, fillvalue=0, chunks=(1, chunk_h, chunk_w))
                mask_dset.attrs['spatial_ref'] = master_crs.to_wkt()
                mask_dset.attrs['GeoTransform'] = gdal_transform
                mask_dset.attrs['description'] = "0 = Invalid/Masked, 1 = Valid. Generated from SpecComplex ARD rules."
                datasets_created_info.append(("common_mask", np.uint8, 3, ["Time", "YDim", "XDim"]))

                for t_idx in range(num_frames):
                    valid_mask = sc.get_tanager_mask(grp_tanager, t_idx, (master_height, master_width),
                                                     sun_elevation_threshold=SUN_ELEVATION_THRESHOLD,
                                                     cloud_dilation=TANAGER_CLOUD_DILATION,
                                                     apply_cloud_mask=True,
                                                     uncertainty_threshold=TANAGER_UNCERTAINTY_THRESHOLD,
                                                     aerosol_depth_threshold=TANAGER_AEROSOL_THRESHOLD)
                    mask_dset[t_idx, ...] = valid_mask.astype(np.uint8)

                odl_blocks.append(generate_tanager_odl_string("TANAGER", master_width, master_height, master_transform, master_proj, master_zone, master_gctp, datasets_created_info, num_frames, band_count))
    else:
        print(f"\nWARNING: Native Tanager Stack not found at {INPUT_NATIVE_TANAGER_HDF5}. Skipping Tanager processing.")

    full_odl = "GROUP=SwathStructure\nEND_GROUP=SwathStructure\nGROUP=GridStructure\n" + "\n".join(odl_blocks) + "\nEND_GROUP=GridStructure\nGROUP=PointStructure\nEND_GROUP=PointStructure\nGROUP=ZaStructure\nEND_GROUP=ZaStructure\nEND\n"
    info_grp.create_dataset("StructMetadata.0", shape=(1,), dtype=h5py.string_dtype(encoding='ascii'), data=full_odl)

print("\nPipeline Complete. Multi-Sensor ARD Master Grid generated successfully.")