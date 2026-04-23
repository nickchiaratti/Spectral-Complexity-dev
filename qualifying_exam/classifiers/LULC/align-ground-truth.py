import os
import h5py
import numpy as np
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.warp import Resampling
from collections import Counter

# ==========================================
# 1. CONFIGURATION & PATHS
# ==========================================
h5_path = "C:/satelliteImagery/LANDSAT/Rochester/LANDSAT_Stack_Rochester_HDFEOS.h5"

# Provide all the years you intend to use for training
RAW_CDLS = {
    2023: "C:/satelliteImagery/ground_truth/cdl/2023_30m_cdls/2023_30m_cdls.tif",
    2024: "C:/satelliteImagery/ground_truth/cdl/2024_30m_cdls/2024_30m_cdls.tif",
    2025: "C:/satelliteImagery/ground_truth/cdl/2025_30m_cdls/2025_30m_cdls.tif"
}

ALIGNED_CDLS = {
    2023: "C:/satelliteImagery/LANDSAT/Rochester/CDL2023_Aligned_Rochester.tif",
    2024: "C:/satelliteImagery/LANDSAT/Rochester/CDL2024_Aligned_Rochester.tif",
    2025: "C:/satelliteImagery/LANDSAT/Rochester/CDL2025_Aligned_Rochester.tif"
}

# Minimum number of GLOBAL pixels required to become a standalone training class. 
MIN_PIXELS = 10000

CDL_NAMES = {
    1: "Corn", 4: "Sorghum", 5: "Soybeans", 24: "Winter Wheat", 36: "Alfalfa", 
    37: "Other Hay/Non Alfalfa", 43: "Potatoes", 59: "Sod/Grass Seed", 61: "Fallow/Idle Cropland",
    68: "Apples", 69: "Grapes", 111: "Open Water", 121: "Developed/Open Space", 
    122: "Developed/Low Intensity", 123: "Developed/Med Intensity", 124: "Developed/High Intensity", 
    131: "Barren", 141: "Deciduous Forest", 142: "Evergreen Forest", 143: "Mixed Forest", 
    152: "Shrubland", 176: "Grassland/Pasture", 190: "Woody Wetlands", 195: "Herbaceous Wetlands"
}

MACRO_GROUPS = {
    122: 121, 123: 121, 124: 121, # Group all developed land
    142: 141, 143: 141,           # Group all forests
    195: 190                      # Group all wetlands
}

# ==========================================
# 2. DATA ALIGNMENT UTILITY
# ==========================================
def main():
    with h5py.File(h5_path, 'r') as f:
        ds = f['/HDFEOS/GRIDS/LANDSAT/Data Fields/surface_reflectance']
        h_30, w_30 = ds.shape[2], ds.shape[3]

        spatial_ref = ds.attrs['spatial_ref']
        if isinstance(spatial_ref, bytes):
            spatial_ref = spatial_ref.decode('utf-8')
        dst_crs = rasterio.crs.CRS.from_user_input(spatial_ref)
        
        tf = ds.attrs['GeoTransform']
        if tf[1] == 30.0:  
            dst_transform = rasterio.Affine.from_gdal(*tf)
        else:
            dst_transform = rasterio.Affine(*tf)

    # Step 1: Reproject all years into memory
    print("Reprojecting raw CDL masks to match HDF5 grid...")
    raw_aligned_masks = {}
    global_counts = Counter()
    
    for year, raw_path in RAW_CDLS.items():
        if not os.path.exists(raw_path):
            print(f"Warning: Raw CDL for {year} not found at {raw_path}. Skipping.")
            continue
            
        with rasterio.open(raw_path) as src:
            source_crs = src.crs or rasterio.crs.CRS.from_epsg(5070)
            with WarpedVRT(src, src_crs=source_crs, crs=dst_crs, transform=dst_transform, width=w_30, height=h_30, resampling=Resampling.nearest) as vrt:
                mask = vrt.read(1)
                
                # Apply Macro Groupings immediately
                for old_val, new_val in MACRO_GROUPS.items():
                    mask[mask == old_val] = new_val
                    
                raw_aligned_masks[year] = mask
                
                # Add to global statistics
                unique_vals, counts = np.unique(mask, return_counts=True)
                for v, c in zip(unique_vals, counts):
                    global_counts[v] += c
                    
    # Step 2: Build the MASTER Global Mapping
    print("\nAnalyzing GLOBAL pixel distributions across all years...")
    dynamic_mapping = {0: 0} # 0 is always background/ignore
    current_class_id = 1
    
    report_lines = []
    def log(msg):
        print(msg)
        report_lines.append(msg + "\n")
    
    log(f"\n{'CDL Code':<10} | {'Category Name':<25} | {'Global Pixels':<15} | {'Action'}")
    log("-" * 80)
    
    for val, count in sorted(global_counts.items(), key=lambda x: x[1], reverse=True):
        if val == 0:
            log(f"{val:<10} | {'No Data / Background':<25} | {count:<15} | Ignored (Class 0)")
            continue
            
        name = CDL_NAMES.get(val, f"Unknown Crop ({val})")
        
        if count >= MIN_PIXELS:
            dynamic_mapping[val] = current_class_id
            log(f"{val:<10} | {name:<25} | {count:<15} | Mapped to Class {current_class_id}")
            current_class_id += 1
        else:
            dynamic_mapping[val] = 0
            log(f"{val:<10} | {name:<25} | {count:<15} | Dropped (< {MIN_PIXELS})")
            
    # Step 3: Apply mapping and Save to disk
    profile = {
        'driver': 'GTiff', 'count': 1, 'crs': dst_crs, 'transform': dst_transform,
        'width': w_30, 'height': h_30, 'dtype': 'uint8', 'nodata': 0, 'compress': 'lzw'
    }
    
    for year, mask in raw_aligned_masks.items():
        final_mask = np.zeros_like(mask, dtype='uint8')
        for cdl_val, contiguous_val in dynamic_mapping.items():
            final_mask[mask == cdl_val] = contiguous_val
            
        out_path = ALIGNED_CDLS[year]
        with rasterio.open(out_path, 'w', **profile) as dst:
            dst.write(final_mask, 1)
        print(f"Saved consistent aligned mask for {year} -> {out_path}")

    # Save the global report
    report_path = "C:/satelliteImagery/LANDSAT/Rochester/CDL_Global_Mapping_Report.txt"
    with open(report_path, "w") as rf:
        rf.writelines(report_lines)
    print(f"\nMaster Summary Report saved to {report_path}")
    print(f"Total valid global training classes: {current_class_id - 1}")

if __name__ == "__main__":
    main()