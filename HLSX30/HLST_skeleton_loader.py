"""
HLST (Harmonized Landsat Sentinel Tanager) ARD Starter Skeleton

This script provides the foundational architecture for interacting with the 
multi-sensor HDF-EOS5 Data Cube. It demonstrates how to safely access the 
chronological HARMONIZED index, translate WGS84 geographic coordinates 
into pixel space, and properly mask analytical data using the strict 
QA common_mask.

Designed for out-of-core memory efficiency: Datasets are accessed by reference 
and sliced into RAM only when needed.
"""

import os
import h5py
import numpy as np
from datetime import datetime, timezone
import rasterio.transform
from pyproj import Transformer, CRS

# ==========================================
# 1. CONFIGURATION
# ==========================================
# Target file path (Update this before handing off)
Location = "Rochesterv2"
ARD_CUBE_PATH = f"C:/satelliteImagery/HLST30/HLST_{Location}_Harmonized_2025_SC_EM-7_Norm-bandCount.h5"

# Example Region of Interest (Lat, Lon) for pixel extraction
TARGET_LAT = 43.142856
TARGET_LON = -77.508451

# ==========================================
# 2. CORE ARD INTERFACE CLASS
# ==========================================
class HLST_ARD_Interface:
    def __init__(self, filepath):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"CRITICAL ERROR: ARD Cube not found at {filepath}")
            
        self.filepath = filepath
        self.h5 = h5py.File(filepath, 'r')
        
        # Verify the unified relational index exists
        self.harm_path = '/HDFEOS/GRIDS/HARMONIZED/Data Fields'
        if self.harm_path not in self.h5:
            raise KeyError(f"CRITICAL ERROR: Unified timeline missing. Expected group: {self.harm_path}")
            
        self.harm_grp = self.h5[self.harm_path]
        
        print(f"[{os.path.basename(filepath)}] Successfully mounted.")
        self._initialize_metrology()
        self._extract_global_timeline()

    def _initialize_metrology(self):
        """
        Extracts the foundational spatial geometry from the ARD attributes.
        Fails loudly if the cube is not geometrically standardized.
        """
        # Reference any dataset in the Harmonized group to extract global attributes
        ref_ds = self.harm_grp['common_mask']
        
        if 'spatial_ref' not in ref_ds.attrs or 'GeoTransform' not in ref_ds.attrs:
            raise KeyError("CRITICAL ERROR: ARD cube is missing 'spatial_ref' or 'GeoTransform' attributes.")
            
        # Handle HDF5 string encoding anomalies
        crs_wkt = ref_ds.attrs['spatial_ref']
        if isinstance(crs_wkt, bytes): 
            crs_wkt = crs_wkt.decode('utf-8')
            
        self.crs = CRS.from_wkt(crs_wkt)
        self.geo_transform = ref_ds.attrs['GeoTransform']
        
        # Build Rasterio/PyProj translation matrices
        self.affine = rasterio.transform.Affine.from_gdal(*self.geo_transform)
        self.inv_affine = ~self.affine
        self.transformer = Transformer.from_crs("EPSG:4326", self.crs, always_xy=True)
        
        # Map Dimensions
        self.num_frames, self.height, self.width = ref_ds.shape
        
        print(f" -> Geometry: {self.width}x{self.height} pixels | CRS: {self.crs.name}")

    def _extract_global_timeline(self):
        """
        Extracts the Relational Provenance Vectors. These 1D arrays allow you 
        to track exactly which sensor generated a specific frame in the timeline.
        """
        ref_ds = self.harm_grp['common_mask']
        
        req_attrs = ['acquisition_time', 'source_grid', 'source_spacecraft', 'source_frame_index']
        for attr in req_attrs:
            if attr not in ref_ds.attrs:
                raise KeyError(f"CRITICAL ERROR: Missing relational provenance attribute '{attr}'")
                
        self.times = ref_ds.attrs['acquisition_time']
        self.indices = ref_ds.attrs['source_frame_index']
        
        # Safely decode byte-strings from HDF5
        raw_grids = ref_ds.attrs['source_grid']
        raw_spacecraft = ref_ds.attrs['source_spacecraft']
        
        self.grids = [g.decode('utf-8') if isinstance(g, bytes) else str(g) for g in raw_grids]
        self.spacecraft = [s.decode('utf-8') if isinstance(s, bytes) else str(s) for s in raw_spacecraft]
        
        print(f" -> Timeline: {self.num_frames} total frames indexed.")

    def latlon_to_pixel(self, lat, lon):
        """Converts WGS84 coordinates to HDF5 array indices (Row/Y, Col/X)."""
        proj_x, proj_y = self.transformer.transform(lon, lat)
        px, py = self.inv_affine * (proj_x, proj_y)
        
        row, col = int(round(py)), int(round(px))
        
        if not (0 <= row < self.height and 0 <= col < self.width):
            raise ValueError(f"Coordinate ({lat}, {lon}) falls outside the ARD Master Grid bounding box.")
            
        return row, col

    def close(self):
        """Closes the HDF5 file handle safely."""
        self.h5.close()

# ==========================================
# 3. EXAMPLE ANALYSIS WORKFLOW
# ==========================================
def run_analysis():
    print("Initializing Data Analysis Skeleton...\n")
    
    # 1. Mount the ARD Cube
    ard = HLST_ARD_Interface(ARD_CUBE_PATH)
    
    # 2. Get pixel indices for the Target Coordinate
    row, col = ard.latlon_to_pixel(TARGET_LAT, TARGET_LON)
    print(f"\nTarget Location ({TARGET_LAT}, {TARGET_LON}) mapped to Array Index: [Y:{row}, X:{col}]")
    
    # 3. Set up memory-efficient pointers to the HDF5 Datasets on disk
    # (These do NOT load the gigabytes of data into RAM yet)
    if 'sliding_volume_z_score' not in ard.harm_grp:
        raise KeyError("Z-Score dataset missing. Ensure Spectral Complexity pipeline was executed.")
        
    ds_zscore = ard.harm_grp['sliding_volume_z_score']
    ds_mask = ard.harm_grp['common_mask']
    
    print("\n--- Initiating Temporal Extraction ---")
    
    valid_observations = 0
    
    # 4. Iterate over the Chronological Timeline
    for t in range(ard.num_frames):
        # Decode the timestamp
        dt = datetime.fromtimestamp(ard.times[t], tz=timezone.utc)
        date_str = dt.strftime('%Y-%m-%d')
        
        # Relational Identity: Which sensor took this image?
        sensor_grid = ard.grids[t]
        local_idx = ard.indices[t]
        
        # Data Extraction: Read ONLY the specific pixel we need from disk
        pixel_mask_value = ds_mask[t, row, col]
        
        # STRICT DATA PURITY GUARDRAIL: Only evaluate pixels where Mask == 1
        if pixel_mask_value == 1:
            pixel_zscore = ds_zscore[t, row, col]
            
            # Additional safety check against algorithm-induced NaNs
            if not np.isnan(pixel_zscore):
                valid_observations += 1
                
                # Optional: Example of pivoting back to the native sensor grid 
                # using the relational keys if you needed raw reflectance data.
                # native_ds = ard.h5[f'/HDFEOS/GRIDS/{sensor_grid}/Data Fields/surface_reflectance']
                # raw_ref = native_ds[local_idx, :, row, col]
                
                print(f"[{date_str}] {sensor_grid.ljust(10)} | Z-Score: {pixel_zscore:>6.3f}")
        else:
            # Pixel was obscured by cloud, shadow, aerosol thickness, or sensor margins
            pass 
            
    print(f"\nExtraction Complete. Recovered {valid_observations} radiometrically valid observations from {ard.num_frames} total epochs.")
    
    # Clean up
    ard.close()

if __name__ == "__main__":
    try:
        run_analysis()
    except Exception as e:
        print(f"\nPIPELINE HALTED: {e}")