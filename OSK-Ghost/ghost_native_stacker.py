import os
import re
import glob
import h5py
import rasterio
import numpy as np
from datetime import datetime
from tqdm import tqdm

# --- Configuration ---
# Target directory based on the user's tree.txt structure
SOURCE_DIR = "C:/satelliteImagery/OSK-Ghost/SourceData"
OUTPUT_H5 = os.path.join(SOURCE_DIR, "GHOST_Native_Stack_HDFEOS.h5")

def find_ghost_collections(source_dir):
    """
    Scans the source directory for OSK-Ghost .hsi and matching _igm files.
    Extracts the acquisition timestamp to guarantee chronological temporal stacking.
    """
    scene_files = []
    
    # Walk through the directory tree
    for root, _, files in os.walk(source_dir):
        for file in files:
            # Locate the primary hyperspectral cube (ignore aux.xml)
            if file.endswith('.hsi'):
                hsi_path = os.path.join(root, file)
                
                # The IGM file shares the UUID but replaces -l1b.hsi with -l1b_igm
                igm_name = file.replace('-l1b.hsi', '-l1b_igm')
                igm_path = os.path.join(root, igm_name)
                
                if os.path.exists(igm_path):
                    # Extract ISO 8601 timestamp from filename (e.g., 2025-08-22T20-03-57Z)
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
    
    if not scenes:
        print("No valid paired .hsi and _igm files found. Exiting.")
        return

    print(f"Found {len(scenes)} valid acquisitions. Calculating native tensor boundaries...")
    
    # 1. First Pass: Determine max dimensions for the unprojected tensor
    # Native swaths vary in size. We must find the global max H and W to pad the tensor.
    T = len(scenes)
    max_h = 0
    max_w = 0
    num_bands = 0
    
    for scene in scenes:
        with rasterio.open(scene['hsi']) as src:
            if num_bands == 0:
                num_bands = src.count
            elif num_bands != src.count:
                raise ValueError(f"Band mismatch! Found {src.count} bands, expected {num_bands}.")
            
            if src.height > max_h: max_h = src.height
            if src.width > max_w: max_w = src.width

    print(f"Tensor Shape [T, C, H, W] -> [{T}, {num_bands}, {max_h}, {max_w}]")

    # 2. Second Pass: Initialize the HDF-EOS5 structure and write data natively
    with h5py.File(OUTPUT_H5, 'w') as h5:
        # Construct HDFEOS hierarchy
        data_grp = h5.create_group('HDFEOS/GRIDS/GHOST/Data Fields')
        
        # Determine chunking strategy to prevent memory exhaustion during downstream slicing
        chunk_h = min(max_h, 256)
        chunk_w = min(max_w, 256)
        
        # Create core Datasets
        # User requested explicitly: "radiance" (not surface_reflectance)
        ds_rad = data_grp.create_dataset(
            'radiance', 
            shape=(T, num_bands, max_h, max_w),
            dtype='float32',
            chunks=(1, num_bands, chunk_h, chunk_w),
            fillvalue=np.nan,
            compression='lzf' # lzf is fast and highly effective for padded empty space
        )
        
        # Explicit Coordinate Geometry (IGM)
        ds_lat = data_grp.create_dataset(
            'latitude', shape=(T, max_h, max_w), dtype='float64',
            chunks=(1, min(max_h, 512), min(max_w, 512)),
            fillvalue=np.nan, compression='lzf'
        )
        
        ds_lon = data_grp.create_dataset(
            'longitude', shape=(T, max_h, max_w), dtype='float64',
            chunks=(1, min(max_h, 512), min(max_w, 512)),
            fillvalue=np.nan, compression='lzf'
        )
        
        # Common mask (1 = valid native pixel, 0 = invalid / NaN padded boundary)
        ds_mask = data_grp.create_dataset(
            'common_mask', shape=(T, max_h, max_w), dtype='uint8',
            chunks=(1, min(max_h, 512), min(max_w, 512)),
            fillvalue=0, compression='lzf'
        )
        
        # Store metadata
        dt_strings = [s['dt_str'].encode('utf-8') for s in scenes]
        data_grp.create_dataset('acquisition_time', data=np.array(dt_strings))
        
        ds_rad.attrs['description'] = "GHOST-5 L1B Native Radiance (Padded Tensor)"
        ds_rad.attrs['bands'] = num_bands
        ds_rad.attrs['padding_method'] = "NaN padding to max swath dimensions. Zero interpolation applied."
        
        # 3. Populate Datasets
        for t, scene in enumerate(tqdm(scenes, desc="Ingesting Native Arrays to HDF5")):
            # Read Radiance Cube
            with rasterio.open(scene['hsi']) as src_hsi:
                h, w = src_hsi.height, src_hsi.width
                rad_data = src_hsi.read().astype('float32')
                nodata = src_hsi.nodata
                
                # Apply nodata masking if defined by the ENVI header
                if nodata is not None:
                    rad_data[rad_data == nodata] = np.nan
                
                # Top-left align the native data into the max-dimension tensor
                ds_rad[t, :, :h, :w] = rad_data
            
            # Read IGM (Input Geometry Model)
            with rasterio.open(scene['igm']) as src_igm:
                lat_data = src_igm.read(1).astype('float64')
                lon_data = src_igm.read(2).astype('float64')
                
                ds_lat[t, :h, :w] = lat_data
                ds_lon[t, :h, :w] = lon_data
            
            # Generate the strict boolean common_mask
            mask_array = np.ones((h, w), dtype='uint8')
            if nodata is not None:
                mask_array[rad_data[0] == nodata] = 0
                
            ds_mask[t, :h, :w] = mask_array

    print(f"\nProcessing Complete. Native HDFEOS stack saved to:\n{OUTPUT_H5}")

if __name__ == "__main__":
    main()