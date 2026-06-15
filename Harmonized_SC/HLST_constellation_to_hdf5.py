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
import hdfeos_odl
from Harmonized_SC.harmonize_tanager import process_tanager_swaths_to_grid
from Harmonized_SC.harmonize_hls import process_hls_master_stack

import yaml

def main(target_location=None):

    # ==========================================
    # 1. CONFIGURATION & DIRECTORIES
    # ==========================================
    TARGET_RESOLUTION = 30.0
    # --- Pixel Mask Configuration ---
    MIN_ROI_COVERAGE_PERCENT = 25.0 
    SUN_ELEVATION_THRESHOLD = 30
    # HLS Specific Configuration (Unified Fmask for both S30 and L30)
    # Bits 0-5: cirrus, cloud, adj cloud/shadow, cloud shadow, snow/ice, water
    HLS_CLOUD_DILATION =0
    QA_REJECT_MASK = 0b11111 
    AEROSOL_ACCEPT_LEVEL = 'medium' # 'low' (0-1), 'medium' (0-2), 'high' (0-3)

    # TANAGER Specific Configuration
    TANAGER_CLOUD_DILATION = 5
    TANAGER_UNCERTAINTY_THRESHOLD = 0.1
    TANAGER_AEROSOL_THRESHOLD = 0.35

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
        Location = config_data.get("current_run", {}).get("location")
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
    # 3. HDFEOS5 ODL GENERATORS (Imported from hdfeos_odl)
    # ==========================================

    def write_hdf_sensor_group(h5f, group_path, data_dict, wavelengths, crs, transform, tile_mapping_json=None):
        if not data_dict or data_dict['count'] == 0: return
        grp = h5f.create_group(group_path)
        gdal_transform = np.array([transform.c, transform.a, transform.b, transform.f, transform.d, transform.e], dtype='float64')
        dt = h5py.string_dtype(encoding='ascii')
    
        num_frames, bands, h, w = data_dict['sr'].shape
        chunk_h, chunk_w = min(h, 256), min(w, 256)
    
        sr_ds = grp.create_dataset('surface_reflectance', data=data_dict['sr'], compression='gzip', compression_opts=5, chunks=(1, bands, chunk_h, chunk_w))
        sr_ds.attrs['units'] = "Reflectance"; sr_ds.attrs['_FillValue'] = np.nan; sr_ds.attrs['wavelengths'] = wavelengths
        sr_ds.attrs['spatial_ref'] = crs.to_wkt(); sr_ds.attrs['GeoTransform'] = gdal_transform
    
        fmask_ds = grp.create_dataset('Fmask', data=data_dict['fm'][:, 0, :, :], dtype='uint8', compression='gzip', compression_opts=5, chunks=(1, chunk_h, chunk_w))
        fmask_ds.attrs['_FillValue'] = 255
        ang_ds = grp.create_dataset('solar_view_angles', data=data_dict['ag'], compression='gzip', compression_opts=5, chunks=(1, 4, chunk_h, chunk_w))
        ang_ds.attrs['_FillValue'] = np.nan; ang_ds.attrs['band_order'] = ["SZA", "SAA", "VZA", "VAA"]
    
        vis_ds = grp.create_dataset('ortho_visual', data=data_dict['vis'], dtype='uint8', compression='gzip', compression_opts=5, chunks=(1, 4, chunk_h, chunk_w))
        vis_ds.attrs['spatial_ref'] = crs.to_wkt()
        vis_ds.attrs['GeoTransform'] = gdal_transform

        mask_ds = grp.create_dataset('common_mask', data=data_dict['mask'], dtype=bool, compression='gzip', compression_opts=5, chunks=(1, chunk_h, chunk_w))
        mask_ds.attrs['description'] = "True = Invalid/Masked, False = Valid."
        mask_ds.attrs['spatial_ref'] = crs.to_wkt()
        mask_ds.attrs['GeoTransform'] = gdal_transform
        mask_ds.attrs['qa_reject_mask'] = QA_REJECT_MASK
        mask_ds.attrs['cloud_dilation'] = HLS_CLOUD_DILATION
        mask_ds.attrs['aerosol_accept_level'] = AEROSOL_ACCEPT_LEVEL
        mask_ds.attrs['sun_elevation_threshold'] = SUN_ELEVATION_THRESHOLD


            
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


    def process_tanager_master_stack(h5f):
        
        
        res = process_tanager_swaths_to_grid(
            h5f=h5f,
            tanager_source_dir=TANAGER_SOURCE_DIR,
            master_height=master_height,
            master_width=master_width,
            master_crs=master_crs,
            master_transform=master_transform,
            min_roi_coverage=MIN_ROI_COVERAGE_PERCENT,
            sun_elev_thresh=SUN_ELEVATION_THRESHOLD,
            cloud_dil=TANAGER_CLOUD_DILATION,
            uncert_thresh=TANAGER_UNCERTAINTY_THRESHOLD,
            aero_thresh=TANAGER_AEROSOL_THRESHOLD
        )
        
        if res is not None:
            datasets_created_info, total_num_frames, band_count = res
            return hdfeos_odl.generate_dynamic_odl_grid_string("TANAGER", master_width, master_height, master_transform, master_proj, master_zone, master_gctp, datasets_created_info, total_num_frames, band_count)
        else:
            return None

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
        s30_master_data = process_hls_master_stack(
            INPUT_NATIVE_HDF5, 
            s30_daily, 
            10, 
            master_height, 
            master_width, 
            master_transform, 
            master_crs, 
            MIN_ROI_COVERAGE_PERCENT, 
            SUN_ELEVATION_THRESHOLD, 
            HLS_CLOUD_DILATION, 
            QA_REJECT_MASK, 
            AEROSOL_ACCEPT_LEVEL
        )
        if s30_master_data:
            write_hdf_sensor_group(h5f, '/HDFEOS/GRIDS/HLSS30/Data Fields', s30_master_data, S30_WAVELENGTHS, master_crs, master_transform, tile_mapping_json=master_tile_mapping_json)
            odl_blocks.append(hdfeos_odl.generate_hls_odl_grid_string("HLSS30", master_width, master_height, master_transform, master_proj, master_zone, master_gctp, 10, s30_master_data['count']))

        print("Harmonizing HLSL30...")
        l30_master_data = process_hls_master_stack(
            INPUT_NATIVE_HDF5, 
            l30_daily, 
            7, 
            master_height, 
            master_width, 
            master_transform, 
            master_crs, 
            MIN_ROI_COVERAGE_PERCENT, 
            SUN_ELEVATION_THRESHOLD, 
            HLS_CLOUD_DILATION, 
            QA_REJECT_MASK, 
            AEROSOL_ACCEPT_LEVEL
        )
        if l30_master_data:
            write_hdf_sensor_group(h5f, '/HDFEOS/GRIDS/HLSL30/Data Fields', l30_master_data, L30_SR_WAVELENGTHS, master_crs, master_transform, tile_mapping_json=master_tile_mapping_json)
            odl_blocks.append(hdfeos_odl.generate_hls_odl_grid_string("HLSL30", master_width, master_height, master_transform, master_proj, master_zone, master_gctp, 7, l30_master_data['count']))

        # --- 5b. TANAGER Hyperspectral Processing (From Basic SWATH) ---
        tanager_odl = process_tanager_master_stack(h5f)
        if tanager_odl:
            odl_blocks.append(tanager_odl)


if __name__ == '__main__':
    main()
