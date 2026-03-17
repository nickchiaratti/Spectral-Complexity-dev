import os
import json
import h5py
import numpy as np
import rasterio
import re
import matplotlib.pyplot as plt
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_bounds, Affine
from pyproj import Transformer, CRS
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import warnings

# --- Configuration & ROI Definitions ---
TIME_THRESHOLD_SECONDS = 60  # Group frames taken within 1 minute
TARGET_RESOLUTION = 30.0     # Meters (Standard Tanager Product Spec)
# Bounding Box (Longitude/Latitude)
Location = "Rochester"
if Location == "Rochester":
    ROI_LON_MIN = -77.72; ROI_LON_MAX = -77.4450
    ROI_LAT_MIN = 43.0450; ROI_LAT_MAX = 43.28
elif Location == "Tait":
    ROI_LON_MIN = -77.516127; ROI_LON_MAX = -77.461968
    ROI_LAT_MIN = 43.127698; ROI_LAT_MAX = 43.159168
elif Location == "RIT":
    ROI_LON_MIN = -77.688990; ROI_LON_MAX = -77.660365
    ROI_LAT_MIN = 43.072486; ROI_LAT_MAX = 43.093298
elif Location == "Tait-I-490":
    ROI_LON_MIN = -77.516127; ROI_LON_MAX = -77.4450
    ROI_LAT_MIN = 43.0450; ROI_LAT_MAX = 43.159168

# Main directory
SOURCE_DIR = "C:/satelliteImagery/Tanager/SourceData"
OUTPUT_DIR = f"C:/satelliteImagery/Tanager/{Location}"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, f"Tanager_Stack_{Location}_HDFEOS.h5")

def calculate_target_grid(lon_min, lon_max, lat_min, lat_max, resolution):
    """
    Projects WGS84 ROI to the correct UTM Zone and defines the target grid.
    Matches Landsat L2 dimensions and UTM Zone selection logic.
    """
    central_lon = (lon_min + lon_max) / 2
    utm_zone = int((central_lon + 180) / 6) + 1
    epsg_code = 32600 + utm_zone
    dst_crs = CRS.from_epsg(epsg_code)
    
    transformer = Transformer.from_crs("EPSG:4326", dst_crs, always_xy=True)
    xs, ys = transformer.transform([lon_min, lon_max, lon_max, lon_min], 
                                   [lat_max, lat_max, lat_min, lat_min])
    
    dst_min_x, dst_max_x = min(xs), max(xs)
    dst_min_y, dst_max_y = min(ys), max(ys)
    
    width = int(np.ceil((dst_max_x - dst_min_x) / resolution))
    height = int(np.ceil((dst_max_y - dst_min_y) / resolution))
    
    target_transform = from_bounds(dst_min_x, dst_min_y, dst_max_x, dst_max_y, width, height)
    
    return dst_crs, target_transform, width, height, (dst_min_x, dst_max_y), (dst_max_x, dst_min_y), utm_zone

def is_roi_intersecting(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    s_min_lon, s_min_lat, s_max_lon, s_max_lat = data['bbox']
    overlap = not (ROI_LON_MAX < s_min_lon or ROI_LON_MIN > s_max_lon or 
                   ROI_LAT_MAX < s_min_lat or ROI_LAT_MIN > s_max_lat)
    return overlap, data

def extract_georeferencing_from_h5(h5_path):
    """
    Parses internal StructMetadata.0 ODL string from source HDF5.
    Robust against HE5_ prefixes, spaces, and varied number formatting.
    """
    try:
        with h5py.File(h5_path, 'r') as f:
            meta_path = "HDFEOS INFORMATION/StructMetadata.0"
            if meta_path not in f:
                return None, None
            
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
                x_dim = int(x_match.group(1))
                y_dim = int(y_match.group(1))
                
                src_transform = from_bounds(ul_x, lr_y, lr_x, ul_y, x_dim, y_dim)
                zone = int(zone_match.group(1)) if zone_match else 18
                src_crs = CRS.from_dict({'proj': 'utm', 'zone': zone, 'datum': 'WGS84'})
                
                return src_transform, src_crs
    except Exception as e:
        print(f"      Warning: Failed to parse ODL from {os.path.basename(h5_path)}: {e}")
    return None, None

def generate_struct_metadata(grid_name, width, height, ul_mtrs, lr_mtrs, datasets_info, n_times, n_bands, utm_zone):
    """
    Generates HDF-EOS5 StructMetadata.0 ODL string for the stack.
    """
    data_fields_blocks = []
    for i, (name, dtype, rank, dim_names) in enumerate(datasets_info):
        eos_type = "H5T_NATIVE_FLOAT"
        if "uint8" in str(dtype): eos_type = "H5T_NATIVE_UINT8"
        elif "uint16" in str(dtype): eos_type = "H5T_NATIVE_UINT16"
        elif "uint" in str(dtype): eos_type = "H5T_NATIVE_UINT"
        elif "int" in str(dtype): eos_type = "H5T_NATIVE_INT"
        elif "float64" in str(dtype) or "double" in str(dtype): eos_type = "H5T_NATIVE_DOUBLE"
        
        dims_list = ",".join([f"\"{d}\"" for d in dim_names])
        block = f"""            OBJECT=DataField_{i+1}
                DataFieldName="{name}"
                DataType={eos_type}
                DimList=({dims_list})
                MaxdimList=({dims_list})
                CompressionType=HE5_HDFE_COMP_DEFLATE
                DeflateLevel=4
            END_OBJECT=DataField_{i+1}"""
        data_fields_blocks.append(block)
    
    odl = f"""GROUP=SwathStructure
END_GROUP=SwathStructure
GROUP=GridStructure
    GROUP=GRID_1
        GridName="{grid_name}"
        XDim={width}
        YDim={height}
        UpperLeftPointMtrs=({ul_mtrs[0]:.6f},{ul_mtrs[1]:.6f})
        LowerRightMtrs=({lr_mtrs[0]:.6f},{lr_mtrs[1]:.6f})
        Projection=HE5_GCTP_UTM
        ZoneCode={utm_zone}
        SphereCode=12
        CompressionType=HE5_HDFE_COMP_DEFLATE
        DeflateLevel=4
        PixelRegistration=HE5_HDFE_CORNER
        GridOrigin=HE5_HDFE_GD_UL

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
    END_GROUP=GRID_1
END_GROUP=GridStructure
GROUP=PointStructure
END_GROUP=PointStructure
GROUP=ZaStructure
END_GROUP=ZaStructure
END
"""
    return odl

def process_tanager_stack():
    print(f"Starting Rasterio-based Processing for: {Location}")
    
    root_path = Path(SOURCE_DIR)
    raw_scenes = []
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

    if not raw_scenes: return
    raw_scenes.sort(key=lambda x: x['time'])

    grouped_scenes = []
    current_group = [raw_scenes[0]]
    for i in range(1, len(raw_scenes)):
        if (raw_scenes[i]['time'] - current_group[-1]['time']).total_seconds() <= TIME_THRESHOLD_SECONDS:
            current_group.append(raw_scenes[i])
        else:
            grouped_scenes.append(current_group); current_group = [raw_scenes[i]]
    grouped_scenes.append(current_group)

    dst_crs, tf_target, width, height, ul_coords, lr_coords, utm_zone = calculate_target_grid(ROI_LON_MIN, ROI_LON_MAX, ROI_LAT_MIN, ROI_LAT_MAX, TARGET_RESOLUTION)

    first_h5 = grouped_scenes[0][0]['h5_file']
    dataset_info_list = []
    band_count = 0
    with h5py.File(first_h5, 'r') as f:
        src_grp = f["HDFEOS/GRIDS/HYP/Data Fields"]
        for name in src_grp.keys():
            dset = src_grp[name]
            dataset_info_list.append({'name': name, 'h5_path': f"HDFEOS/GRIDS/HYP/Data Fields/{name}", 'dtype': dset.dtype, 'shape': dset.shape, 'fill': dset.attrs.get("_FillValue")})
            if name == "surface_reflectance": band_count = dset.shape[0]

    datasets_created_info = []
    gw_to_plot = None
    
    # Initialize array for Unix UTC acquisition times
    acqTime_attr = np.zeros(len(grouped_scenes), dtype='float64')

    with h5py.File(OUTPUT_FILE, 'w') as out_h5:
        grp_tanager = out_h5.create_group("HDFEOS/GRIDS/TANAGER/Data Fields")
        meta_grp = out_h5.create_group("METADATA")
        info_grp = out_h5.create_group("HDFEOS INFORMATION")
        
        for d_info in dataset_info_list:
            name = d_info['name']
            print(f"  Processing dataset: {name}")
            is_3d = len(d_info['shape']) == 3
            out_shape = (len(grouped_scenes), d_info['shape'][0], height, width) if is_3d else (len(grouped_scenes), height, width)
            out_dset = grp_tanager.create_dataset(name, shape=out_shape, dtype=d_info['dtype'], compression="gzip", fillvalue=d_info['fill'])
            datasets_created_info.append((name, d_info['dtype'], len(out_shape), ["Time", "Band", "YDim", "XDim"] if is_3d else ["Time", "YDim", "XDim"]))

            # Handle sr_invalid creation if we are on surface_reflectance
            ds_invalid = None
            if name == "surface_reflectance":
                ds_invalid = grp_tanager.create_dataset("sr_invalid", shape=(len(grouped_scenes), height, width), dtype='uint8', compression="gzip", fillvalue=0)
                datasets_created_info.append(("sr_invalid", np.dtype('uint8'), 3, ["Time", "YDim", "XDim"]))

            per_frame_good_wavelengths = []

            for t_idx, group in enumerate(grouped_scenes):
                pass_canvas = np.full(out_shape[1:], d_info['fill'], dtype=d_info['dtype'])
                
                for scene in group:
                    fpath = scene["h5_file"].replace("\\", "/")
                    src_tf, src_crs_info = extract_georeferencing_from_h5(fpath)

                    handle = rasterio.open(f'HDF5:"{fpath}"://{d_info["h5_path"].replace(" ", "_")}')
                    if handle:
                        try:
                            incoming = np.full(out_shape[1:], d_info['fill'], dtype=d_info['dtype'])
                            reproject(rasterio.band(handle, list(range(1, handle.count + 1))), incoming, 
                                      src_transform=src_tf, src_crs=src_crs_info, dst_transform=tf_target, dst_crs=dst_crs,
                                      resampling=Resampling.nearest if (d_info['dtype'].name == 'uint8') else Resampling.cubic_spline,
                                      src_nodata=d_info['fill'], dst_nodata=d_info['fill'])
                            
                            mask = (incoming != d_info['fill'])
                            pass_canvas[mask] = incoming[mask]
                        finally: handle.close()
                    else:
                        print(f"    Warning: Could not open {name} in {os.path.basename(fpath)}")
                
                # Write final pass canvas
                out_dset[t_idx, ...] = pass_canvas
                
                # Save center timestamp value if this is the 'time' dataset
                if name == 'time':
                    center_y, center_x = height // 2, width // 2
                    acqTime_attr[t_idx] = pass_canvas[center_y, center_x]
                
                # Compute sr_invalid mask if processing reflectance
                if name == "surface_reflectance" and ds_invalid is not None:
                    invalid_mask = np.logical_or(np.any(pass_canvas < 0, axis=0), np.any(pass_canvas > 1, axis=0)).astype(np.uint8)
                    ds_invalid[t_idx, ...] = invalid_mask
                
                if name == "surface_reflectance":
                    with h5py.File(group[0]["h5_file"], 'r') as f_attr:
                        per_frame_good_wavelengths.append(f_attr[d_info['h5_path']].attrs.get("good_wavelengths"))
                    meta_grp.attrs[f"frame_{t_idx}_json"] = json.dumps(group[0]['json'])

            with h5py.File(first_h5, 'r') as f0:
                src_ds = f0[d_info['h5_path']]
                for k, v in src_ds.attrs.items():
                    if k not in ["DIMENSION_LIST", "REFERENCE_LIST", "CLASS", "PALETTE", "good_wavelengths"]:
                        out_dset.attrs[k] = v
                if name == "surface_reflectance" and per_frame_good_wavelengths:
                    gw_array = np.array(per_frame_good_wavelengths)
                    out_dset.attrs["all_good_wavelengths"] = gw_array
                    gw_to_plot = gw_array
                    out_dset.attrs["good_wavelengths"] = np.logical_and.reduce(gw_array, axis=0).astype(np.int32)

                # --- Standardized Spatial Metadata ---
                out_dset.attrs['spatial_ref'] = dst_crs.to_wkt()
                gdal_transform = [tf_target.c, tf_target.a, tf_target.b, tf_target.f, tf_target.d, tf_target.e]
                out_dset.attrs['GeoTransform'] = np.array(gdal_transform, dtype='float64')

            # Also apply to the invalid mask if it was generated alongside surface_reflectance
            if name == "surface_reflectance" and ds_invalid is not None:
                ds_invalid.attrs['spatial_ref'] = dst_crs.to_wkt()
                ds_invalid.attrs['GeoTransform'] = np.array(gdal_transform, dtype='float64')

        # After processing all datasets, add acquisition_time to surface_reflectance
        if "surface_reflectance" in grp_tanager:
            grp_tanager["surface_reflectance"].attrs["acquisition_time"] = acqTime_attr
            grp_tanager["surface_reflectance"].attrs["spacecraft_id"] = ['Tanager-1']*len(grouped_scenes)

        # Visuals
        vis_dset = grp_tanager.create_dataset("ortho_visual", shape=(len(grouped_scenes), 4, height, width), dtype='uint8', compression="gzip", fillvalue=0)
        
        # --- Standardized Spatial Metadata ---
        vis_dset.attrs['spatial_ref'] = dst_crs.to_wkt()
        gdal_transform = [tf_target.c, tf_target.a, tf_target.b, tf_target.f, tf_target.d, tf_target.e]
        vis_dset.attrs['GeoTransform'] = np.array(gdal_transform, dtype='float64')
        
        datasets_created_info.append(("ortho_visual", np.uint8, 4, ["Time", "VisBand", "YDim", "XDim"]))
        for t_idx, group in enumerate(grouped_scenes):
            pass_vis = np.zeros((4, height, width), dtype='uint8')
            for scene in group:
                with rasterio.open(scene['vis_file']) as src:
                    incoming = np.zeros((4, height, width), dtype='uint8')
                    reproject(rasterio.band(src, [1, 2, 3, 4]), incoming, src_transform=src.transform, 
                              src_crs=src.crs, dst_transform=tf_target, dst_crs=dst_crs, resampling=Resampling.cubic)
                    mask = (incoming[3] > 0); pass_vis[:, mask] = incoming[:, mask]
            vis_dset[t_idx, ...] = pass_vis

        struct_meta = generate_struct_metadata("TANAGER", width, height, ul_coords, lr_coords, datasets_created_info, len(grouped_scenes), band_count, utm_zone)
        dt_str = h5py.string_dtype(encoding='ascii')
        info_grp.create_dataset("StructMetadata.0", (1,), dtype=dt_str, data=struct_meta)

    if gw_to_plot is not None:
        plt.figure(figsize=(12, 6))
        plt.imshow(gw_to_plot, aspect='auto', interpolation='none', cmap='binary_r', origin='lower')
        plt.title(f"Spectral Quality Mask Across Stack | {Location}"); plt.tight_layout(); plt.show()

    print(f"\nProcessing Complete. Saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    process_tanager_stack()