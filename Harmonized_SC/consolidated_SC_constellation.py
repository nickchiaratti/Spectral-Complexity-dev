import os
import h5py
import numpy as np
import yaml
from pathlib import Path
from rasterio.transform import Affine
from pyproj import CRS, Transformer
from datetime import datetime

# Define standard configurations
LOCATIONS_YAML = r"f:\Resilio\IMGS 890 Research\Spectral-Complexity-dev\locations_config.yaml"
BASE_DIR = r"C:\satelliteImagery"

def get_sensor_files(location):
    """Finds all native stack files for the given location."""
    return {
        "HLS": list(Path(f"{BASE_DIR}/HLS30").glob(f"HLS_{location}_Native_Stack*.h5")),
        "TANAGER": list(Path(f"{BASE_DIR}/Tanager").glob(f"Tanager_Native_Stack_{location}.h5")),
        "ENMAP": list(Path(f"{BASE_DIR}/enmap").glob(f"EnMAP_Native_Stack_{location}.h5")),
        "DRAGONETTE": list(Path(f"{BASE_DIR}/dragonette").glob(f"Dragonette_Native_Stack_{location}.h5"))
    }

def main(location="SantaBarbara"):
    with open(LOCATIONS_YAML, 'r') as f:
        config = yaml.safe_load(f)['locations'][location]
        
    target_epsg = config['MGRS_EPSG']
    target_crs = CRS.from_epsg(target_epsg)
    target_transform = Affine(config['TARGET_GSD'], 0.0, config['MGRS_UL_X'], 
                              0.0, -config['TARGET_GSD'], config['MGRS_UL_Y'])
    target_w = config['MGRS_WIDTH']
    target_h = config['MGRS_HEIGHT']
    
    # Pre-calculate target grid coordinates
    rows, cols = np.meshgrid(np.arange(target_h), np.arange(target_w), indexing='ij')
    target_x = target_transform.c + cols * target_transform.a
    target_y = target_transform.f + rows * target_transform.e
    
    out_file = f"{BASE_DIR}/consolidatedConstellation/SC_Constellation_{location}.h5"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    
    sensor_files = get_sensor_files(location)
    
    with h5py.File(out_file, 'w') as h5_out:
        # Write target grid metadata
        tg = h5_out.create_group("target_grid")
        tg.attrs["MGRS_EPSG"] = target_epsg
        tg.attrs["GeoTransform"] = np.array(target_transform).astype(np.float64)
        tg.attrs["spatial_ref"] = target_crs.to_wkt()
        tg.attrs["width"] = target_w
        tg.attrs["height"] = target_h
        
        timeline_entries = []
        
        for sensor_type, files in sensor_files.items():
            for fpath in files:
                if not fpath.exists(): continue
                with h5py.File(fpath, 'r') as h5_in:
                    # Iterate groups (for HLS there might be multiple MGRS tiles)
                    for group_name in h5_in['HDFEOS/GRIDS'].keys():
                        # Handle specific sensor linking and indexing
                        sensor_key = f"{sensor_type}_{group_name}"
                        
                        df = h5_in[f'HDFEOS/GRIDS/{group_name}/Data Fields']
                        # Get native CRS and Transform
                        native_wkt = df['surface_reflectance'].attrs.get('spatial_ref', '')
                        native_transform = df['surface_reflectance'].attrs.get('GeoTransform')
                        
                        if not native_wkt or native_transform is None:
                            print(f"Skipping {sensor_key} due to missing spatial metadata")
                            continue
                            
                        if isinstance(native_wkt, bytes):
                            native_wkt = native_wkt.decode('utf-8')
                            
                        if native_wkt.startswith('EPSG:'):
                            native_crs = CRS.from_string(native_wkt)
                        else:
                            native_crs = CRS.from_wkt(native_wkt)
                        native_aff = Affine.from_gdal(*native_transform)
                        native_h, native_w = df['surface_reflectance'].shape[-2:]
                        
                        # Project target coordinates to native
                        if native_crs != target_crs:
                            proj = Transformer.from_crs(target_crs, native_crs, always_xy=True)
                            nx, ny = proj.transform(target_x, target_y)
                        else:
                            nx, ny = target_x, target_y
                            
                        # Apply inverse affine
                        inv = ~native_aff
                        nf_col, nf_row = inv * (nx, ny)
                        row_map = np.round(nf_row).astype(np.int32)
                        col_map = np.round(nf_col).astype(np.int32)
                        
                        valid_mask = (row_map >= 0) & (row_map < native_h) & (col_map >= 0) & (col_map < native_w)
                        
                        # Write mappings
                        sg = h5_out.create_group(f"sensors/{sensor_key}")
                        sg.attrs["source_file"] = str(fpath)
                        sg.attrs["native_CRS"] = native_wkt
                        sg.attrs["native_GeoTransform"] = native_transform
                        
                        sg.create_dataset("row_map", data=row_map, compression="gzip")
                        sg.create_dataset("col_map", data=col_map, compression="gzip")
                        sg.create_dataset("valid_mask", data=valid_mask, compression="gzip")
                        
                        # External Links
                        sg['data_fields'] = h5py.ExternalLink(str(fpath), f'HDFEOS/GRIDS/{group_name}/Data Fields')
                        if f'HDFEOS/GRIDS/{group_name}/Derived Fields' in h5_in:
                            sg['derived_fields'] = h5py.ExternalLink(str(fpath), f'HDFEOS/GRIDS/{group_name}/Derived Fields')
                            
                        # Timeline
                        acq_times = df['surface_reflectance'].attrs.get('acquisition_time', [])
                        spacecrafts = df['surface_reflectance'].attrs.get('spacecraft_id', [])
                        for i, (t, sc) in enumerate(zip(acq_times, spacecrafts)):
                            timeline_entries.append({
                                'time': t,
                                'sensor_key': sensor_key,
                                'frame_index': i,
                                'spacecraft': sc
                            })
                            
        # Consolidate timeline (merge synchronous entries)
        from collections import defaultdict
        time_map = defaultdict(list)
        for e in timeline_entries:
            time_map[e['time']].append(e)
            
        sorted_times = sorted(time_map.keys())
        
        N = len(sorted_times)
        acq_time_arr = np.zeros(N, dtype=np.float64)
        sensor_arr = np.zeros(N, dtype=h5py.string_dtype(encoding='utf-8'))
        fidx_arr = np.zeros(N, dtype=h5py.string_dtype(encoding='utf-8')) # JSON encoded array of indices
        sc_arr = np.zeros(N, dtype=h5py.string_dtype(encoding='utf-8'))
        
        import json
        for i, t in enumerate(sorted_times):
            entries = time_map[t]
            acq_time_arr[i] = t
            sensor_arr[i] = ",".join([e['sensor_key'] for e in entries])
            fidx_arr[i] = json.dumps([e['frame_index'] for e in entries])
            
            sc_list = []
            for e in entries:
                sc = e['spacecraft']
                if isinstance(sc, bytes):
                    sc = sc.decode('utf-8')
                elif isinstance(sc, np.bytes_):
                    sc = sc.decode('utf-8')
                sc_list.append(str(sc))
            sc_arr[i] = ",".join(list(set(sc_list)))
            
        tl = h5_out.create_group("timeline")
        tl.create_dataset("acquisition_time", data=acq_time_arr)
        tl.create_dataset("source_sensor", data=sensor_arr)
        tl.create_dataset("source_frame_index", data=fidx_arr)
        tl.create_dataset("spacecraft_id", data=sc_arr)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--location", default="LakeFire2024")
    args = parser.parse_args()
    main(args.location)
