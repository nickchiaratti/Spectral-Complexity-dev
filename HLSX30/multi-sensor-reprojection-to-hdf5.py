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
import re
import warnings
import SpecComplex as sc

# ==========================================
# 1. CONFIGURATION & DIRECTORIES
# ==========================================
Location = "Rochesterv2"

if Location == "Rochesterv2":
    ROI_LON_MIN = -77.716163; ROI_LON_MAX = -77.751438
    ROI_LAT_MIN = 42.961035; ROI_LAT_MAX = 43.333724
    
COMBINED_OUTPUT_DIR = r"C:\satelliteImagery\HLSX30"
TANAGER_SOURCE_DIR = r"C:\satelliteImagery\Tanager\SourceData"

INPUT_NATIVE_HDF5 = os.path.join(COMBINED_OUTPUT_DIR, f"HLS_Combined_Stack_{Location}_STAC_Native_2025.h5")
OUTPUT_MASTER_HDF5 = os.path.join(COMBINED_OUTPUT_DIR, f"ARD_Cube_{Location}_MasterGrid_2025.h5")

S30_WAVELENGTHS = [0.443, 0.490, 0.560, 0.665, 0.705, 0.740, 0.783, 0.842, 0.865, 0.945, 1.375, 1.610, 2.190]
L30_SR_WAVELENGTHS = [0.443, 0.482, 0.561, 0.655, 0.865, 1.609, 2.201, 1.373] 
L30_TIRS_WAVELENGTHS = [10.9, 12.0]

safe_bbox = [
    min(ROI_LON_MIN, ROI_LON_MAX), max(ROI_LAT_MIN, ROI_LAT_MAX), 
    max(ROI_LON_MIN, ROI_LON_MAX), min(ROI_LAT_MIN, ROI_LAT_MAX)
]
safe_bbox = [min(safe_bbox[0], safe_bbox[2]), min(safe_bbox[1], safe_bbox[3]), max(safe_bbox[0], safe_bbox[2]), max(safe_bbox[1], safe_bbox[3])]
TARGET_RESOLUTION = 30.0
TANAGER_TIME_THRESHOLD = 60 

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
    thermal_field = f"""            OBJECT=DataField_2\n                DataFieldName="thermal_infrared"\n                DataType=HDF5T_NATIVE_FLOAT\n                DimList=("Time","ThermalBands","YDim","XDim")\n            END_OBJECT=DataField_2""" if has_thermal else ""
    fmask_idx = 3 if has_thermal else 2
    ang_idx = 4 if has_thermal else 3
    tm_idx = 5 if has_thermal else 4
    
    tile_mask_field = f"""            OBJECT=DataField_{tm_idx}\n                DataFieldName="source_tile_mask"\n                DataType=HDF5T_NATIVE_UINT8\n                DimList=("Time","YDim","XDim")\n            END_OBJECT=DataField_{tm_idx}""" if has_tile_mask else ""
    
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
        END_GROUP=Dimension
        GROUP=DataField
            OBJECT=DataField_1
                DataFieldName="surface_reflectance"
                DataType=HDF5T_NATIVE_FLOAT
                DimList=("Time","Bands","YDim","XDim")
            END_OBJECT=DataField_1
{thermal_field}
            OBJECT=DataField_{fmask_idx}
                DataFieldName="Fmask"
                DataType=HDF5T_NATIVE_UINT8
                DimList=("Time","YDim","XDim")
            END_OBJECT=DataField_{fmask_idx}
            OBJECT=DataField_{ang_idx}
                DataFieldName="solar_view_angles"
                DataType=HDF5T_NATIVE_FLOAT
                DimList=("Time","AngleBands","YDim","XDim")
            END_OBJECT=DataField_{ang_idx}
{tile_mask_field}
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
    
    sr_ds = grp.create_dataset('surface_reflectance', data=data_dict['sr'], compression='gzip', compression_opts=6)
    sr_ds.attrs['units'] = "Reflectance"; sr_ds.attrs['_FillValue'] = np.nan; sr_ds.attrs['wavelengths'] = wavelengths
    sr_ds.attrs['spatial_ref'] = crs.to_wkt(); sr_ds.attrs['GeoTransform'] = gdal_transform
    
    if thermal_wavelengths and data_dict['th'] is not None:
        th_ds = grp.create_dataset('thermal_infrared', data=data_dict['th'], compression='gzip', compression_opts=6)
        th_ds.attrs['units'] = "Kelvin/Celsius Apparent"; th_ds.attrs['_FillValue'] = np.nan; th_ds.attrs['wavelengths'] = thermal_wavelengths
        
    fmask_ds = grp.create_dataset('Fmask', data=data_dict['fm'][:, 0, :, :], dtype='uint8', compression='gzip', compression_opts=6)
    fmask_ds.attrs['_FillValue'] = 255
    ang_ds = grp.create_dataset('solar_view_angles', data=data_dict['ag'], compression='gzip', compression_opts=6)
    ang_ds.attrs['_FillValue'] = np.nan; ang_ds.attrs['band_order'] = ["SZA", "SAA", "VZA", "VAA"]
    
    vis_ds = grp.create_dataset('ortho_visual', data=data_dict['vis'], dtype='uint8', compression='gzip', compression_opts=6)
    vis_ds.attrs['spatial_ref'] = crs.to_wkt()
    vis_ds.attrs['GeoTransform'] = gdal_transform

    if 'tm' in data_dict and data_dict['tm'] is not None:
        tm_ds = grp.create_dataset('source_tile_mask', data=data_dict['tm'][:, 0, :, :], dtype='uint8', compression='gzip', compression_opts=6)
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
            tile_name = grid_id.split('_')[1] # e.g. T17TQJ
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
    vis_data = np.zeros((num_frames, 4, master_height, master_width), dtype=np.uint8)
    meta_arrays = {'acq': [], 'space': [], 'saz': [], 'sel': [], 'cc': []}
    
    with h5py.File(native_h5_path, 'r') as h5f:
        for idx, date_str in enumerate(sorted_dates):
            entries = daily_groups[date_str]
            
            # Port foundational metadata from the first valid frame of the swath
            base_grid = entries[0]['grid_id']
            base_fidx = entries[0]['frame_idx']
            base_path = f'HDFEOS/GRIDS/{base_grid}/Data Fields/surface_reflectance'
            meta_arrays['acq'].append(h5f[base_path].attrs['acquisition_time'][base_fidx])
            
            # EVIDENCE-BASED FIX: Defensive String Extraction
            # Handles h5py 2.x (returns bytes) and 3.x+ (returns str) seamlessly
            raw_spacecraft = h5f[base_path].attrs['spacecraft_id'][base_fidx]
            spacecraft_str = raw_spacecraft.decode('utf-8') if isinstance(raw_spacecraft, bytes) else str(raw_spacecraft)
            meta_arrays['space'].append(spacecraft_str)
            
            meta_arrays['cc'].append(h5f[base_path].attrs['cloud_cover'][base_fidx])
            
            for entry in entries:
                tile = entry['tile']
                fidx = entry['frame_idx']
                grid_id = entry['grid_id']
                df_path = f'HDFEOS/GRIDS/{grid_id}/Data Fields'
                
                # Extract native geographic geometries
                sr_node = h5f[f'{df_path}/surface_reflectance']
                src_tf = Affine.from_gdal(*sr_node.attrs['GeoTransform'])
                src_crs = CRS.from_wkt(sr_node.attrs['spatial_ref'])
                
                # 1. Surface Reflectance (Cubic)
                src_sr = sr_node[fidx]
                tmp_sr = np.full((expected_sr, master_height, master_width), np.nan, dtype=np.float32)
                reproject(source=src_sr, destination=tmp_sr, src_transform=src_tf, src_crs=src_crs, dst_transform=master_transform, dst_crs=master_crs, resampling=Resampling.cubic, src_nodata=np.nan, dst_nodata=np.nan)
                mask_sr = ~np.isnan(tmp_sr)
                stk_sr[idx][mask_sr] = tmp_sr[mask_sr]
                
                # 2. Thermal (Cubic)
                if expected_thermal > 0:
                    src_th = h5f[f'{df_path}/thermal_infrared'][fidx]
                    tmp_th = np.full((expected_thermal, master_height, master_width), np.nan, dtype=np.float32)
                    reproject(source=src_th, destination=tmp_th, src_transform=src_tf, src_crs=src_crs, dst_transform=master_transform, dst_crs=master_crs, resampling=Resampling.cubic, src_nodata=np.nan, dst_nodata=np.nan)
                    mask_th = ~np.isnan(tmp_th)
                    stk_th[idx][mask_th] = tmp_th[mask_th]

                # 3. Fmask (Nearest)
                src_fm = h5f[f'{df_path}/Fmask'][fidx]
                tmp_fm = np.full((1, master_height, master_width), 255, dtype=np.uint8)
                reproject(source=src_fm, destination=tmp_fm, src_transform=src_tf, src_crs=src_crs, dst_transform=master_transform, dst_crs=master_crs, resampling=Resampling.nearest, src_nodata=255, dst_nodata=255)
                mask_fm = (tmp_fm != 255)
                stk_fm[idx][mask_fm] = tmp_fm[mask_fm]
                
                # Assign Source Tile ID to mask
                stk_tm[idx, 0][mask_fm[0]] = tile_map[tile]
                
                # 4. Angles (Nearest)
                src_ag = h5f[f'{df_path}/solar_view_angles'][fidx]
                tmp_ag = np.full((4, master_height, master_width), np.nan, dtype=np.float32)
                reproject(source=src_ag, destination=tmp_ag, src_transform=src_tf, src_crs=src_crs, dst_transform=master_transform, dst_crs=master_crs, resampling=Resampling.nearest, src_nodata=np.nan, dst_nodata=np.nan)
                mask_ag = ~np.isnan(tmp_ag)
                stk_ag[idx][mask_ag] = tmp_ag[mask_ag]

            # Re-derive holistic Solar metadata across the harmonized footprint
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                mean_sza = np.nanmean(stk_ag[idx, 0])
                mean_saa = np.nanmean(stk_ag[idx, 1])
                
            meta_arrays['saz'].append(mean_saa)
            meta_arrays['sel'].append(90.0 - mean_sza)
            
            # Generate RGBA Visual Fallback
            try:
                rgba_img = sc.generate_rgba_image(stk_sr[idx])
                vis_data[idx, ...] = np.transpose(rgba_img, (2, 0, 1))
            except Exception:
                vis_data[idx, 0] = np.clip(np.nan_to_num(stk_sr[idx, 3]) * 255 * 3, 0, 255).astype(np.uint8) 
                vis_data[idx, 1] = np.clip(np.nan_to_num(stk_sr[idx, 2]) * 255 * 3, 0, 255).astype(np.uint8) 
                vis_data[idx, 2] = np.clip(np.nan_to_num(stk_sr[idx, 1]) * 255 * 3, 0, 255).astype(np.uint8) 
                vis_data[idx, 3] = 255

    return {'sr': stk_sr, 'th': stk_th, 'fm': stk_fm, 'ag': stk_ag, 'tm': stk_tm, 'vis': vis_data, 'meta': meta_arrays, 'count': num_frames}

# ==========================================
# 5. DATA PIPELINE: TANAGER TO MASTER GRID
# ==========================================
def is_roi_intersecting(json_path):
    with open(json_path, 'r') as f: data = json.load(f)
    s_min_lon, s_min_lat, s_max_lon, s_max_lat = data['bbox']
    overlap = not (safe_bbox[2] < s_min_lon or safe_bbox[0] > s_max_lon or safe_bbox[3] < s_min_lat or safe_bbox[1] > s_max_lat)
    return overlap, data

def extract_georeferencing_from_h5(h5_path):
    with h5py.File(h5_path, 'r') as f:
        meta_path = "HDFEOS INFORMATION/StructMetadata.0"
        if meta_path not in f: raise ValueError(f"CRITICAL ERROR: StructMetadata.0 missing in {h5_path}")
        
        meta_data = f[meta_path][()]
        if isinstance(meta_data, (np.ndarray, list)): meta_data = meta_data[0]
        odl = meta_data.decode('ascii') if isinstance(meta_data, bytes) else str(meta_data)
        
        ul_match = re.search(r'UpperLeftPointMtrs=\(\s*(-?[\d\.]+)\s*,\s*(-?[\d\.]+)\s*\)', odl)
        lr_match = re.search(r'LowerRightMtrs=\(\s*(-?[\d\.]+)\s*,\s*(-?[\d\.]+)\s*\)', odl)
        x_match = re.search(r'XDim=(\d+)', odl)
        y_match = re.search(r'YDim=(\d+)', odl)
        zone_match = re.search(r'ZoneCode=(\d+)', odl)
        
        if all([ul_match, lr_match, x_match, y_match]):
            ul_x, ul_y = float(ul_match.group(1)), float(ul_match.group(2))
            lr_x, lr_y = float(lr_match.group(1)), float(lr_match.group(2))
            x_dim, y_dim = int(x_match.group(1)), int(y_match.group(1))
            zone = int(zone_match.group(1)) if zone_match else 18
            
            # EVIDENCE-BASED FIX: Namespace Alignment
            # Call the securely aliased 'transform_from_bounds' defined in the script imports.
            return transform_from_bounds(ul_x, lr_y, lr_x, ul_y, x_dim, y_dim), CRS.from_dict({'proj': 'utm', 'zone': zone, 'datum': 'WGS84'})
        else: raise ValueError(f"CRITICAL ERROR: Incomplete ODL bounding geometry in {h5_path}")

def group_tanager_scenes():
    print(f"\nLocating Tanager Hyperspectral Data in: {TANAGER_SOURCE_DIR}")
    root_path = Path(TANAGER_SOURCE_DIR)
    raw_scenes = []
    
    if not root_path.exists():
        print("WARNING: Tanager source directory not found. Skipping Tanager ingestion.")
        return []

    for subfolder in root_path.iterdir():
        if not subfolder.is_dir(): continue
        json_path = list(subfolder.glob("*.json"))
        h5_path = list(subfolder.glob("*_ortho_sr_hdf5.h5"))
        vis_path = list(subfolder.glob("*_ortho_visual.tif"))
        
        if json_path and h5_path:
            is_overlap, stac_data = is_roi_intersecting(json_path[0])
            if is_overlap:
                dt = datetime.fromisoformat(stac_data['properties']['datetime'].replace('Z', '+00:00'))
                raw_scenes.append({'h5_file': str(h5_path[0]), 'vis_file': str(vis_path[0]) if vis_path else None, 'time': dt, 'json': stac_data})

    if not raw_scenes: return []
    raw_scenes.sort(key=lambda x: x['time'])

    grouped_scenes, current_group = [], [raw_scenes[0]]
    for i in range(1, len(raw_scenes)):
        if (raw_scenes[i]['time'] - current_group[-1]['time']).total_seconds() <= TANAGER_TIME_THRESHOLD:
            current_group.append(raw_scenes[i])
        else:
            grouped_scenes.append(current_group); current_group = [raw_scenes[i]]
    grouped_scenes.append(current_group)
    print(f"Found {len(raw_scenes)} valid Tanager frames, grouped into {len(grouped_scenes)} temporal mosaics.")
    return grouped_scenes

# ==========================================
# 6. MASTER EXECUTION
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
    
    # --- 6a. HLS Master Grid Stitching ---
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

    # --- 6b. TANAGER Hyperspectral Processing ---
    grouped_tanager_scenes = group_tanager_scenes()
    if grouped_tanager_scenes:
        print("Harmonizing Tanager Hyperspectral Arrays...")
        first_h5 = grouped_tanager_scenes[0][0]['h5_file']
        dataset_info_list = []
        band_count = 0
        with h5py.File(first_h5, 'r') as f_tan:
            src_grp = f_tan["HDFEOS/GRIDS/HYP/Data Fields"]
            for name in src_grp.keys():
                dset = src_grp[name]
                dataset_info_list.append({'name': name, 'h5_path': f"HDFEOS/GRIDS/HYP/Data Fields/{name}", 'dtype': dset.dtype, 'shape': dset.shape, 'fill': dset.attrs.get("_FillValue")})
                if name == "surface_reflectance": band_count = dset.shape[0]

        datasets_created_info = []
        grp_tanager = h5f.create_group("HDFEOS/GRIDS/TANAGER/Data Fields")
        meta_grp = h5f.create_group("METADATA_TANAGER")
        
        acqTime_attr = np.zeros(len(grouped_tanager_scenes), dtype='float64')
        gdal_transform = np.array([master_transform.c, master_transform.a, master_transform.b, master_transform.f, master_transform.d, master_transform.e], dtype='float64')
        
        for d_info in dataset_info_list:
            name = d_info['name']
            is_3d = len(d_info['shape']) == 3
            out_shape = (len(grouped_tanager_scenes), d_info['shape'][0], master_height, master_width) if is_3d else (len(grouped_tanager_scenes), master_height, master_width)
            
            out_dset = grp_tanager.create_dataset(name, shape=out_shape, dtype=d_info['dtype'], compression="gzip", fillvalue=d_info['fill'])
            datasets_created_info.append((name, d_info['dtype'], len(out_shape), ["Time", "Band", "YDim", "XDim"] if is_3d else ["Time", "YDim", "XDim"]))

            ds_invalid = None
            if name == "surface_reflectance":
                ds_invalid = grp_tanager.create_dataset("sr_invalid", shape=(len(grouped_tanager_scenes), master_height, master_width), dtype='uint8', compression="gzip", fillvalue=0)
                datasets_created_info.append(("sr_invalid", np.dtype('uint8'), 3, ["Time", "YDim", "XDim"]))

            per_frame_good_wavelengths = []

            for t_idx, group in enumerate(grouped_tanager_scenes):
                pass_canvas = np.full(out_shape[1:], d_info['fill'], dtype=d_info['dtype'])
                for scene in group:
                    fpath = scene["h5_file"].replace("\\", "/")
                    src_tf, src_crs_info = extract_georeferencing_from_h5(fpath)

                    # EVIDENCE-BASED FIX: Direct Memory Extraction via h5py
                    # Bypasses unstable GDAL HDF5 driver subdataset parsing by extracting 
                    # the literal numpy array into memory and passing it directly to the warp engine.
                    with h5py.File(fpath, 'r') as h5_src:
                        source_array = h5_src[d_info['h5_path']][()]
                        
                        incoming = np.full(out_shape[1:], d_info['fill'], dtype=d_info['dtype'])
                        resampling_algo = Resampling.nearest if (d_info['dtype'].name == 'uint8') else Resampling.cubic
                        
                        reproject(source=source_array, destination=incoming, 
                                  src_transform=src_tf, src_crs=src_crs_info, dst_transform=master_transform, dst_crs=master_crs,
                                  resampling=resampling_algo, src_nodata=d_info['fill'], dst_nodata=d_info['fill'])
                        
                        mask = (incoming != d_info['fill'])
                        pass_canvas[mask] = incoming[mask]
                
                out_dset[t_idx, ...] = pass_canvas
                if name == 'time': acqTime_attr[t_idx] = pass_canvas[master_height // 2, master_width // 2]
                
                if name == "surface_reflectance" and ds_invalid is not None:
                    invalid_mask = np.logical_or(np.any(pass_canvas < 0, axis=0), np.any(pass_canvas > 1, axis=0)).astype(np.uint8)
                    ds_invalid[t_idx, ...] = invalid_mask
                    with h5py.File(group[0]["h5_file"], 'r') as f_attr:
                        per_frame_good_wavelengths.append(f_attr[d_info['h5_path']].attrs.get("good_wavelengths"))
                    meta_grp.attrs[f"frame_{t_idx}_json"] = json.dumps(group[0]['json'])

            # Port standard attributes
            with h5py.File(first_h5, 'r') as f0:
                src_ds = f0[d_info['h5_path']]
                for k, v in src_ds.attrs.items():
                    if k not in ["DIMENSION_LIST", "REFERENCE_LIST", "CLASS", "PALETTE", "good_wavelengths"]: out_dset.attrs[k] = v
                if name == "surface_reflectance" and per_frame_good_wavelengths:
                    gw_array = np.array(per_frame_good_wavelengths)
                    out_dset.attrs["all_good_wavelengths"] = gw_array
                    out_dset.attrs["good_wavelengths"] = np.logical_and.reduce(gw_array, axis=0).astype(np.int32)
                out_dset.attrs['spatial_ref'] = master_crs.to_wkt()
                out_dset.attrs['GeoTransform'] = gdal_transform

            if name == "surface_reflectance" and ds_invalid is not None:
                ds_invalid.attrs['spatial_ref'] = master_crs.to_wkt()
                ds_invalid.attrs['GeoTransform'] = gdal_transform

        if "surface_reflectance" in grp_tanager:
            grp_tanager["surface_reflectance"].attrs["acquisition_time"] = acqTime_attr
            grp_tanager["surface_reflectance"].attrs["spacecraft_id"] = ['Tanager-1']*len(grouped_tanager_scenes)

        vis_dset = grp_tanager.create_dataset("ortho_visual", shape=(len(grouped_tanager_scenes), 4, master_height, master_width), dtype='uint8', compression="gzip", fillvalue=0)
        vis_dset.attrs['spatial_ref'] = master_crs.to_wkt()
        vis_dset.attrs['GeoTransform'] = gdal_transform
        datasets_created_info.append(("ortho_visual", np.uint8, 4, ["Time", "VisBand", "YDim", "XDim"]))
        
        for t_idx, group in enumerate(grouped_tanager_scenes):
            pass_vis = np.zeros((4, master_height, master_width), dtype='uint8')
            for scene in group:
                if scene['vis_file']:
                    with rasterio.open(scene['vis_file']) as src:
                        incoming = np.zeros((4, master_height, master_width), dtype='uint8')
                        reproject(rasterio.band(src, [1, 2, 3, 4]), incoming, src_transform=src.transform, src_crs=src.crs, dst_transform=master_transform, dst_crs=master_crs, resampling=Resampling.cubic)
                        mask = (incoming[3] > 0); pass_vis[:, mask] = incoming[:, mask]
            vis_dset[t_idx, ...] = pass_vis

        odl_blocks.append(generate_tanager_odl_string("TANAGER", master_width, master_height, master_transform, master_proj, master_zone, master_gctp, datasets_created_info, len(grouped_tanager_scenes), band_count))

    full_odl = "GROUP=SwathStructure\nEND_GROUP=SwathStructure\nGROUP=GridStructure\n" + "\n".join(odl_blocks) + "\nEND_GROUP=GridStructure\nGROUP=PointStructure\nEND_GROUP=PointStructure\nGROUP=ZaStructure\nEND_GROUP=ZaStructure\nEND\n"
    info_grp.create_dataset("StructMetadata.0", shape=(1,), dtype=h5py.string_dtype(encoding='ascii'), data=full_odl)

print("\nPipeline Complete. Multi-Sensor ARD Master Grid generated successfully.")