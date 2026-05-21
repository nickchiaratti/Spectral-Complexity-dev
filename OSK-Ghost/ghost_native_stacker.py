import os
import re
import h5py
import rasterio
import numpy as np
from datetime import datetime
from tqdm import tqdm

# --- Configuration ---
# Target directory based on the user's tree.txt structure
SOURCE_DIR = r"C:\satelliteImagery\OSK-Ghost\SourceData"
OUTPUT_H5 = os.path.join(SOURCE_DIR, "GHOST_Native_Stack_HDFEOS.h5")

def find_ghost_collections(source_dir):
    """
    Scans the source directory for OSK-Ghost .hsi and matching _igm files.
    Extracts the acquisition timestamp to guarantee chronological temporal stacking.
    """
    scene_files = []
    
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.endswith('.hsi'):
                hsi_path = os.path.join(root, file)
                igm_name = file.replace('-l1b.hsi', '-l1b_igm')
                igm_path = os.path.join(root, igm_name)
                
                if os.path.exists(igm_path):
                    match = re.search(r'(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}Z)', file)
                    if match:
                        dt_str = match.group(1)
                        dt = datetime.strptime(dt_str, "%Y-%m-%dT%H-%M-%SZ")
                        scene_files.append({
                            'dt': dt,
                            'dt_str': dt_str,
                            'hsi': hsi_path,
                            'igm': igm_path
                        })
    
    # Sort chronologically to align with the temporal logic of the Gram Matrix pipeline
    scene_files.sort(key=lambda x: x['dt'])
    return scene_files

def main():
    print(f"Scanning for OSK-Ghost Level-1B imagery in: {SOURCE_DIR}")
    scenes = find_ghost_collections(SOURCE_DIR)
    
    T = len(scenes)
    print(f"Found {T} valid acquisitions. Calculating global Geographic (WGS84) bounding box...")
    
    # 1. First Pass: Analyze IGM arrays to define the unified Geographic coordinate grid
    global_min_lon, global_max_lon = np.inf, -np.inf
    global_min_lat, global_max_lat = np.inf, -np.inf
    res_list = []
    num_bands = 0
    
    for scene in scenes:
        if num_bands == 0:
            with rasterio.open(scene['hsi']) as src:
                num_bands = src.count

        with rasterio.open(scene['igm']) as src_igm:
            lat_data = src_igm.read(1)
            lon_data = src_igm.read(2)
            
            # STRICT DATA INTEGRITY:
            # We explicitly allow np.min to fail or propagate NaNs if the telemetry is corrupted.
            # Silent failure masking has been removed per analytical directives.
            global_min_lon = min(global_min_lon, np.min(lon_data))
            global_max_lon = max(global_max_lon, np.max(lon_data))
            global_min_lat = min(global_min_lat, np.min(lat_data))
            global_max_lat = max(global_max_lat, np.max(lat_data))
            
            # Estimate native Geographic resolution (GSD) to establish grid spacing
            dist_col = np.sqrt(np.diff(lat_data, axis=1)**2 + np.diff(lon_data, axis=1)**2)
            dist_row = np.sqrt(np.diff(lat_data, axis=0)**2 + np.diff(lon_data, axis=0)**2)
            res_list.append(np.median(dist_col))
            res_list.append(np.median(dist_row))

    # Determine unified grid resolution and pad the boundary
    res = np.median(res_list)
    global_min_lon -= res * 2
    global_max_lon += res * 2
    global_min_lat -= res * 2
    global_max_lat += res * 2

    grid_width = int(np.ceil((global_max_lon - global_min_lon) / res))
    grid_height = int(np.ceil((global_max_lat - global_min_lat) / res))

    print(f"Unified Geographic Grid Defined:")
    print(f"  CRS: EPSG:4326")
    print(f"  Resolution: {res:.6e} degrees")
    print(f"  Dimensions: {grid_width}x{grid_height} (WxH)")
    print(f"  Tensor Shape: [{T}, {num_bands}, {grid_height}, {grid_width}]")

    # 2. Second Pass: Initialize the HDF-EOS5 structure
    with h5py.File(OUTPUT_H5, 'w') as h5:
        data_grp = h5.create_group('HDFEOS/GRIDS/GHOST/Data Fields')
        
        chunk_h = min(grid_height, 128)
        chunk_w = min(grid_width, 128)
        
        ds_rad = data_grp.create_dataset(
            'radiance', 
            shape=(T, num_bands, grid_height, grid_width),
            dtype='float32',
            chunks=(1, num_bands, chunk_h, chunk_w),
            fillvalue=np.nan,
            compression='lzf'
        )
        
        ds_mask = data_grp.create_dataset(
            'common_mask', shape=(T, grid_height, grid_width), dtype='uint8',
            chunks=(1, chunk_h, chunk_w),
            fillvalue=0, compression='lzf'
        )
        
        dt_strings = [s['dt_str'].encode('utf-8') for s in scenes]
        data_grp.create_dataset('acquisition_time', data=np.array(dt_strings))
        
        ds_rad.attrs['crs'] = "EPSG:4326"
        ds_rad.attrs['spatial_transform'] = [global_min_lon, res, 0.0, global_max_lat, 0.0, -res]
        ds_rad.attrs['description'] = "GHOST-5 L1B Radiance (Direct IGM Coordinate Lookup Table; Strict Nearest Neighbor)"
        ds_rad.attrs['bands'] = num_bands
        
        # 3. Direct Coordinate Look-Up Table (LUT) Mapping
        for t, scene in enumerate(tqdm(scenes, desc="Mapping IGM to Geographic Grid")):
            
            with rasterio.open(scene['igm']) as src_igm:
                lat_data = src_igm.read(1)
                lon_data = src_igm.read(2)

            # Convert explicit geographic coordinates into discrete target tensor indices
            # USING cKDTree for a pull-based nearest-neighbor approach to prevent Moiré gaps
            from scipy.spatial import cKDTree
            
            native_cols = (lon_data.ravel() - global_min_lon) / res
            native_rows = (global_max_lat - lat_data.ravel()) / res
            
            tree = cKDTree(np.c_[native_rows, native_cols])
            
            grid_rows, grid_cols = np.mgrid[0:grid_height, 0:grid_width]
            
            # Query KDTree for the nearest native pixel. distance_upper_bound=1.8 
            # closes intrinsic rotation gaps without extrapolating beyond the footprint bounds.
            dists, indices = tree.query(np.c_[grid_rows.ravel(), grid_cols.ravel()], distance_upper_bound=1.8)
            
            valid_mask = dists <= 1.8
            valid_indices = indices[valid_mask]
            
            valid_mask_2d = valid_mask.reshape((grid_height, grid_width))

            # Construct binary footprint geometry mask via valid spatial queries
            dest_mask = np.zeros((grid_height, grid_width), dtype='uint8')
            dest_mask[valid_mask_2d] = 1
            ds_mask[t, :, :] = dest_mask

            # Build memory-resident target tensor to prevent I/O fragmentation
            frame_tensor = np.full((num_bands, grid_height, grid_width), np.nan, dtype='float32')
            
            with rasterio.open(scene['hsi']) as src_hsi:
                for b in tqdm(range(num_bands), desc=f"  Mapping Bands (Frame {t+1}/{T})", leave=False):
                    band_data = src_hsi.read(b + 1).astype('float32').ravel()
                    
                    # Pull-based nearest neighbor lookup. 
                    # Conceptually preserves the pure local spectral manifold (no mathematical blending)
                    # while completely eliminating pinholes and Moiré gaps.
                    frame_tensor[b, valid_mask_2d] = band_data[valid_indices]

            # Flush strictly georeferenced block to HDF5
            ds_rad[t, :, :, :] = frame_tensor

    print(f"\nProcessing Complete. Native HDFEOS stack saved to:\n{OUTPUT_H5}")

if __name__ == "__main__":
    main()