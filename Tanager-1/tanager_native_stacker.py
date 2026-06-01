import os
import json
import h5py
import numpy as np
import rasterio
import re
from rasterio.warp import reproject, Resampling
from rasterio.transform import Affine
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# --- Configuration ---
TIME_THRESHOLD_SECONDS = 60
TARGET_RESOLUTION = 30.0
TARGET_RED_NM = 670.0
TARGET_GREEN_NM = 550.0
TARGET_BLUE_NM = 480.0

SOURCE_DIR = "C:/satelliteImagery/Tanager/BuenosAires_SourceData"
OUTPUT_DIR = SOURCE_DIR
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "Tanager_Native_Stack_BuenosAires.h5")

def extract_georeferencing_from_h5(h5_path):
    """
    Parses internal StructMetadata.0 ODL string from source HDF5 to extract
    exact native geospatial bounding coordinates.
    """
    with h5py.File(h5_path, 'r') as f:
        if "HDFEOS INFORMATION/StructMetadata.0" not in f:
            raise KeyError(f"Missing HDFEOS StructMetadata in {os.path.basename(h5_path)}")
            
        meta_data = f["HDFEOS INFORMATION/StructMetadata.0"][()]
        if isinstance(meta_data, (np.ndarray, list)): 
            meta_data = meta_data[0]
        odl = meta_data.decode('ascii') if isinstance(meta_data, bytes) else str(meta_data)
        
        ul_match = re.search(r'UpperLeftPointMtrs=\(\s*(-?[\d\.]+)\s*,\s*(-?[\d\.]+)\s*\)', odl)
        lr_match = re.search(r'LowerRightMtrs=\(\s*(-?[\d\.]+)\s*,\s*(-?[\d\.]+)\s*\)', odl)
        x_match = re.search(r'XDim=(\d+)', odl)
        y_match = re.search(r'YDim=(\d+)', odl)
        zone_match = re.search(r'ZoneCode=(-?\d+)', odl)
        
        if not all([ul_match, lr_match, x_match, y_match, zone_match]):
            raise RuntimeError(f"Failed to parse required geometric parameters from ODL in {os.path.basename(h5_path)}")
            
        ul_x, ul_y = float(ul_match.group(1)), float(ul_match.group(2))
        lr_x, lr_y = float(lr_match.group(1)), float(lr_match.group(2))
        x_dim = int(x_match.group(1))
        y_dim = int(y_match.group(1))
        zone = int(zone_match.group(1))
        
        # Returns: (min_x, min_y, max_x, max_y), CRS string, Zone
        bounds = (ul_x, lr_y, lr_x, ul_y) 
        
        # Compute proper EPSG based on hemisphere
        if zone < 0:
            crs_str = f"EPSG:{32700 + abs(zone)}"
        else:
            crs_str = f"EPSG:{32600 + zone}"
            
        return bounds, crs_str, zone, (x_dim, y_dim)

def calculate_global_grid(all_bounds):
    """
    Calculates a unified bounding box spanning all native inputs.
    Assumes all bounds are aligned to the same UTM resolution grid.
    """
    global_min_x = min([b[0] for b in all_bounds])
    global_min_y = min([b[1] for b in all_bounds])
    global_max_x = max([b[2] for b in all_bounds])
    global_max_y = max([b[3] for b in all_bounds])
    
    # Calculate dimensions based strictly on 30m resolution
    width = int(round((global_max_x - global_min_x) / TARGET_RESOLUTION))
    height = int(round((global_max_y - global_min_y) / TARGET_RESOLUTION))
    
    target_transform = Affine.translation(global_min_x, global_max_y) * Affine.scale(TARGET_RESOLUTION, -TARGET_RESOLUTION)
    
    return target_transform, width, height, (global_min_x, global_max_y), (global_max_x, global_min_y)

def generate_struct_metadata(grid_name, width, height, ul_mtrs, lr_mtrs, datasets_info, n_times, n_bands, utm_zone):
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
                DimensionName="RGBBand"
                Size=3
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

def percentile_stretch(band_data, fill_value, lower_pct=0, upper_pct=100):
    """
    Applies strict linear contrast stretch based on data percentiles.
    """
    valid_mask = (band_data != fill_value) & (band_data >= 0)
    valid_data = band_data[valid_mask]
    
    if valid_data.size == 0:
        raise ValueError("Cannot perform percentile stretch: Array contains no valid data.")
        
    p_low, p_high = np.percentile(valid_data, (lower_pct, upper_pct))
    
    if p_high == p_low:
        raise ValueError(f"Cannot perform percentile stretch: Zero variance in valid data (Value: {p_low}).")
        
    stretched = (band_data.astype(np.float32) - p_low) / (p_high - p_low)
    stretched = np.clip(stretched, 0, 1) * 255
    
    return stretched.astype(np.uint8)

def process_native_stack():
    print("Discovering and sorting native Tanager scenes...")
    
    root_path = Path(SOURCE_DIR)
    raw_scenes = []
    
    for subfolder in root_path.iterdir():
        if not subfolder.is_dir(): continue
        json_path = list(subfolder.glob("*.json"))
        h5_path = list(subfolder.glob("*_ortho_sr_hdf5.h5"))
        
        if json_path and h5_path:
            with open(json_path[0], 'r') as f:
                stac_data = json.load(f)
            dt = datetime.fromisoformat(stac_data['properties']['datetime'].replace('Z', '+00:00'))
            raw_scenes.append({'h5_file': str(h5_path[0]), 'time': dt, 'json': stac_data})

    if not raw_scenes: 
        raise FileNotFoundError(f"No valid Tanager JSON/HDF5 pairs found in {SOURCE_DIR}")
        
    raw_scenes.sort(key=lambda x: x['time'])

    # Temporal Grouping (Passes)
    grouped_scenes = []
    current_group = [raw_scenes[0]]
    for i in range(1, len(raw_scenes)):
        if (raw_scenes[i]['time'] - current_group[-1]['time']).total_seconds() <= TIME_THRESHOLD_SECONDS:
            current_group.append(raw_scenes[i])
        else:
            grouped_scenes.append(current_group)
            current_group = [raw_scenes[i]]
    grouped_scenes.append(current_group)

    print(f"Grouped into {len(grouped_scenes)} temporal passes.")

    # 1. Establish the Global Native Grid
    all_bounds = []
    global_crs = None
    global_zone = None
    
    for scene in raw_scenes:
        bounds, crs, zone, dims = extract_georeferencing_from_h5(scene['h5_file'])
        all_bounds.append(bounds)
        
        # Enforce strict CRS alignment across the entire temporal stack
        if global_crs is None:
            global_crs = crs
            global_zone = zone
        elif crs != global_crs:
            raise ValueError(f"CRITICAL CRS MISMATCH: Expected {global_crs}, but {os.path.basename(scene['h5_file'])} is {crs}. Native stitching impossible without interpolation.")

    tf_target, width, height, ul_coords, lr_coords = calculate_global_grid(all_bounds)
    print(f"Global Native Grid Established: {width}x{height} pixels (UTM Zone {global_zone})")

    # 2. Extract Dataset schemas and assert required attributes exist
    first_h5 = grouped_scenes[0][0]['h5_file']
    dataset_info_list = []
    band_count = 0
    
    with h5py.File(first_h5, 'r') as f:
        src_grp = f["HDFEOS/GRIDS/HYP/Data Fields"]
        for name in src_grp.keys():
            dset = src_grp[name]
            
            # Explicit failure on missing FillValue to enforce strict data integrity
            if "_FillValue" not in dset.attrs:
                raise AttributeError(f"Dataset '{name}' is missing '_FillValue' attribute in source HDF5.")
                
            fill_attr = dset.attrs["_FillValue"]
            f_val = fill_attr[0] if isinstance(fill_attr, (np.ndarray, list, tuple)) else fill_attr
            
            dataset_info_list.append({
                'name': name, 'h5_path': f"HDFEOS/GRIDS/HYP/Data Fields/{name}", 
                'dtype': dset.dtype, 'shape': dset.shape, 'fill': f_val
            })
            if name == "surface_reflectance": band_count = dset.shape[0]

    datasets_created_info = []
    acqTime_attr = np.zeros(len(grouped_scenes), dtype='float64')

    with h5py.File(OUTPUT_FILE, 'w') as out_h5:
        grp_tanager = out_h5.create_group("HDFEOS/GRIDS/TANAGER/Data Fields")
        meta_grp = out_h5.create_group("METADATA")
        info_grp = out_h5.create_group("HDFEOS INFORMATION")
        
        # Write datasets
        for d_info in dataset_info_list:
            name = d_info['name']
            print(f"  Mosaicking dataset: {name}")
            is_3d = len(d_info['shape']) == 3
            out_shape = (len(grouped_scenes), d_info['shape'][0], height, width) if is_3d else (len(grouped_scenes), height, width)
            out_dset = grp_tanager.create_dataset(name, shape=out_shape, dtype=d_info['dtype'], compression="gzip", fillvalue=d_info['fill'])
            datasets_created_info.append((name, d_info['dtype'], len(out_shape), ["Time", "Band", "YDim", "XDim"] if is_3d else ["Time", "YDim", "XDim"]))

            ds_invalid = None
            if name == "surface_reflectance":
                ds_invalid = grp_tanager.create_dataset("sr_invalid", shape=(len(grouped_scenes), height, width), dtype='uint8', compression="gzip", fillvalue=0)
                datasets_created_info.append(("sr_invalid", np.dtype('uint8'), 3, ["Time", "YDim", "XDim"]))

            per_frame_good_wavelengths = []

            for t_idx, group in enumerate(grouped_scenes):
                pass_canvas = np.full(out_shape[1:], d_info['fill'], dtype=d_info['dtype'])
                
                for scene in group:
                    fpath = scene["h5_file"].replace("\\", "/")
                    src_tf_bounds, src_crs, _, src_dims = extract_georeferencing_from_h5(fpath)
                    
                    # Construct source affine from bounds
                    src_tf = Affine.translation(src_tf_bounds[0], src_tf_bounds[3]) * Affine.scale(TARGET_RESOLUTION, -TARGET_RESOLUTION)

                    with h5py.File(fpath, 'r') as src_h5:
                        src_data = src_h5[d_info["h5_path"]][...]
                        incoming = np.full(out_shape[1:], d_info['fill'], dtype=d_info['dtype'])
                        
                        # CRITICAL: Resampling.nearest enforces zero spectral mixing during spatial translation
                        reproject(source=src_data, destination=incoming, 
                                  src_transform=src_tf, src_crs=src_crs, dst_transform=tf_target, dst_crs=global_crs,
                                  resampling=Resampling.nearest,
                                  src_nodata=d_info['fill'], dst_nodata=d_info['fill'])
                        
                        if "mask" in name.lower() or "nodata" in name.lower():
                            is_fill = (pass_canvas == d_info['fill'])
                            is_valid_overwrite = (pass_canvas == 1) & (incoming == 0)
                            update_mask = (incoming != d_info['fill']) & (is_fill | is_valid_overwrite)
                            pass_canvas[update_mask] = incoming[update_mask]
                        else:
                            mask = (incoming != d_info['fill'])
                            pass_canvas[mask] = incoming[mask]
                
                out_dset[t_idx, ...] = pass_canvas
                
                if name == 'time':
                    valid_times = pass_canvas[pass_canvas != d_info['fill']]
                    if valid_times.size == 0:
                        raise ValueError(f"Frame {t_idx} contains no valid 'time' data. Cannot compute acquisition time.")
                    acqTime_attr[t_idx] = np.median(valid_times)
                
                if name == "surface_reflectance" and ds_invalid is not None:
                    invalid_mask = np.logical_or(np.any(pass_canvas < 0, axis=0), np.any(pass_canvas > 1, axis=0)).astype(np.uint8)
                    ds_invalid[t_idx, ...] = invalid_mask
                
                if name == "surface_reflectance":
                    with h5py.File(group[0]["h5_file"], 'r') as f_attr:
                        if "good_wavelengths" not in f_attr[d_info['h5_path']].attrs:
                            raise AttributeError(f"Missing 'good_wavelengths' in {os.path.basename(group[0]['h5_file'])}")
                        per_frame_good_wavelengths.append(f_attr[d_info['h5_path']].attrs["good_wavelengths"])
                    meta_grp.attrs[f"frame_{t_idx}_json"] = json.dumps(group[0]['json'])

            # Attribute Mapping
            with h5py.File(first_h5, 'r') as f0:
                src_ds = f0[d_info['h5_path']]
                for k, v in src_ds.attrs.items():
                    if k not in ["DIMENSION_LIST", "REFERENCE_LIST", "CLASS", "PALETTE", "good_wavelengths"]:
                        out_dset.attrs[k] = v
                if name == "surface_reflectance":
                    gw_array = np.array(per_frame_good_wavelengths)
                    out_dset.attrs["all_good_wavelengths"] = gw_array
                    out_dset.attrs["good_wavelengths"] = np.logical_and.reduce(gw_array, axis=0).astype(np.int32)

                out_dset.attrs['spatial_ref'] = global_crs
                gdal_transform = [tf_target.c, tf_target.a, tf_target.b, tf_target.f, tf_target.d, tf_target.e]
                out_dset.attrs['GeoTransform'] = np.array(gdal_transform, dtype='float64')

            if name == "surface_reflectance" and ds_invalid is not None:
                ds_invalid.attrs['spatial_ref'] = global_crs
                ds_invalid.attrs['GeoTransform'] = np.array(gdal_transform, dtype='float64')

        if "surface_reflectance" in grp_tanager:
            grp_tanager["surface_reflectance"].attrs["acquisition_time"] = acqTime_attr
            grp_tanager["surface_reflectance"].attrs["spacecraft_id"] = np.array(['Tanager-1']*len(grouped_scenes), dtype='S20')

        # --- Generate native ortho_visual directly from surface_reflectance ---
        print("  Generating strict 'ortho_visual' RGB composite from SR...")
        sr_info = next(d for d in dataset_info_list if d['name'] == 'surface_reflectance')
        sr_dt = sr_info['dtype']
        sr_fill = sr_info['fill']

        ortho_vis_dset = grp_tanager.create_dataset("ortho_visual", shape=(len(grouped_scenes), 3, height, width), dtype=sr_dt, compression="gzip", fillvalue=sr_fill)
        ortho_vis_dset.attrs['spatial_ref'] = global_crs
        ortho_vis_dset.attrs['GeoTransform'] = np.array(gdal_transform, dtype='float64')
        datasets_created_info.append(("ortho_visual", sr_dt, 3, ["Time", "RGBBand", "YDim", "XDim"]))
        
        sr_path_in_first = [d['h5_path'] for d in dataset_info_list if d['name'] == 'surface_reflectance'][0]
        with h5py.File(first_h5, 'r') as f0:
            if 'wavelengths' not in f0[sr_path_in_first].attrs:
                raise AttributeError("Missing 'wavelengths' attribute for RGB extraction.")
            wavelengths = f0[sr_path_in_first].attrs['wavelengths']
            r_idx = int(np.argmin(np.abs(wavelengths - TARGET_RED_NM)))
            g_idx = int(np.argmin(np.abs(wavelengths - TARGET_GREEN_NM)))
            b_idx = int(np.argmin(np.abs(wavelengths - TARGET_BLUE_NM)))
            
        sr_dset_ref = grp_tanager["surface_reflectance"]

        for t_idx in tqdm(range(len(grouped_scenes)), desc=f"  Creating ortho_visual with indices R={r_idx}, G={g_idx}, B={b_idx}"):            
            ortho_vis_dset[t_idx, 0, :, :] = sr_dset_ref[t_idx, r_idx, :, :]
            ortho_vis_dset[t_idx, 1, :, :] = sr_dset_ref[t_idx, g_idx, :, :]
            ortho_vis_dset[t_idx, 2, :, :] = sr_dset_ref[t_idx, b_idx, :, :]

        # Write struct metadata
        struct_meta = generate_struct_metadata("TANAGER", width, height, ul_coords, lr_coords, datasets_created_info, len(grouped_scenes), band_count, global_zone)
        dt_str = h5py.string_dtype(encoding='ascii')
        info_grp.create_dataset("StructMetadata.0", (1,), dtype=dt_str, data=struct_meta)

    print(f"\nProcessing Complete. Saved native temporal stack to: {OUTPUT_FILE}")

if __name__ == "__main__":
    process_native_stack()