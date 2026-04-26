# **Tanager Native Hyperspectral Data Dictionary & Navigation Guide**

## **1\. Overview**

This document describes the structure, contents, and metadata of Tanager-1 (Carbon Mapper) hyperspectral image stacks processed via the native stacking pipeline (tanager\_native\_stacker.py).

The file adheres strictly to the HDF-EOS5 grid specification and aligns with Climate and Forecast (CF) Metadata Conventions. A defining characteristic of this data product is its **Native Coordinate Framing**. To support rigorous doctoral-level sub-pixel target detection and spectral complexity analysis (e.g., Gram matrix hypervolumes), the spatial grid is constructed using a dynamic global union of the input frames.

**Methodological Constraint:** Images are spatially translated using strict nearest-neighbor resampling. Bilinear or cubic interpolation is absolutely forbidden to ensure zero artificial spectral mixing. Mathematical mixing of adjacent pixels corrupts the pure endmember signatures defining the simplex bounding your mixing space (Schowengerdt, *Remote Sensing: Models and Methods for Image Processing*, 2007).

**Note on Failure Handling & Data Integrity:** In alignment with rigorous data science practices, this stacker explicitly rejects the use of synthetic fill values to bypass errors. Missing background models, completely saturated images (zero-variance), or mismatched Coordinate Reference Systems (CRS) intentionally halt processing rather than propagating corrupt or misaligned data into the temporal tensor.

## **2\. HDF5 Hierarchy & Base Path**

All temporal sequences, hyperspectral data cubes, masks, and metadata are stored under a unified HDF-EOS5 group path.

* **Base Target Path:** /HDFEOS/GRIDS/TANAGER/Data Fields  
* **STAC Metadata Path:** /METADATA  
* **HDF-EOS Struct Path:** /HDFEOS INFORMATION/StructMetadata.0

### **Dimensional Conventions & Memory Layout**

* **Temporal Dimension (T):** All datasets share a primary temporal dimension (num\_frames or n\_times). Time alignment is perfectly 1:1 across all datasets.  
* **Spatial Dimensions (H, W):** Spatial datasets share Height (rows) and Width (columns) mapped to a 30.0m native UTM grid.  
* **Spectral Dimensions (B):** The hyperspectral band count (n\_bands), representing the \~400+ contiguous bands of the Tanager Dyson spectrometer.  
* **Memory Layout Constraint:** Data is strictly stored in a **Band Sequential (BSQ)** memory layout (\[T, B, H, W\] for 4D datasets, \[T, 3, H, W\] for RGB composites, and \[T, H, W\] for 2D spatial masks) to optimize temporal-spectral vector extraction.

## **3\. Dataset Specifications**

### **3.1 Hyperspectral Science Datasets**

These datasets represent the core scientific payload copied and mosaicked from the source Level-2 products.

* **surface\_reflectance**  
  * **Shape:** \[T, B, H, W\] (float32 or scaled int, depending on source)  
  * **Description:** Bottom-of-Atmosphere (BOA) hyperspectral reflectance.  
  * **Attributes (Crucial for Spatio-Temporal Metrology):**  
    * acquisition\_time: 1D Array of UNIX timestamps (UTC, float64). Computed dynamically as the spatial median of all valid pixel timestamps in the frame's time dataset, minimizing pushbroom gradient anomalies.  
    * spacecraft\_id: 1D Array denoting the platform (e.g., 'Tanager-1').  
    * wavelengths: 1D Array mapping the B axis to physical center wavelengths (in nm).  
    * GeoTransform: 6-element array defining the GDAL affine transformation matrix for the unified global grid.  
    * spatial\_ref: String containing the Well-Known Text (WKT) Coordinate Reference System (e.g., EPSG:32618).  
    * all\_good\_wavelengths: 2D Array \[T, B\] of boolean flags indicating valid sensor operation per band, per time slice.  
    * good\_wavelengths: 1D Array \[B\] representing the logical AND across all time steps (bands considered universally valid across the entire temporal stack).  
* **Dynamic Source Datasets (e.g., sun\_zenith, time, nodata\_pixels)**  
  * **Shape:** Varies (\[T, H, W\] or \[T, B, H, W\])  
  * **Description:** The stacker dynamically discovers and preserves all ancillary datasets present in the source HYP grid.  
  * **Mosaicking Logic Constraint:** For datasets with "mask" or "nodata" in their nomenclature, a specialized overlap hierarchy is enforced: Valid Data (0) strictly overwrites NoData (1). This prevents data erosion at the stitched seams of adjacent orbital footprints.

### **3.2 Visual & Masking Datasets**

* **ortho\_visual**  
  * **Shape:** \[T, 3, H, W\] (uint8)  
  * **Description:** A true-color RGB composite generated *directly* from the surface\_reflectance dataset (utilizing bands closest to 670nm, 550nm, and 480nm). Applies a robust 2–98% percentile stretch, explicitly excluding NoData and atmospheric artifacts (\< 0). Because it is generated post-reprojection using native nearest-neighbor data, it serves as a geometrically faithful visual reference compliant with CEOS Analysis Ready Data (ARD) visual derivation standards.  
  * **Format Constraints:** Stored in BSQ format.  
* **sr\_invalid**  
  * **Shape:** \[T, H, W\] (uint8)  
  * **Description:** An analytically generated boolean mask identifying corrupted surface reflectance pixels. Evaluates to 1 if *any* band within a pixel's contiguous spectral signature drops below 0 (negative reflectance artifact) or exceeds 1.0 (saturation/specular artifact).

### **3.3 Metadata Datasets**

* **frame\_{t\_idx}\_json** (Located in /METADATA)  
  * **Description:** The raw SpatioTemporal Asset Catalog (STAC) JSON metadata associated with the primary acquisition of that temporal pass, serialized as a string for downstream provenance tracking.

## **4\. AI Agent / Developer Navigation Guide (Python)**

To effectively read and analyze this natively stitched stack, utilize the following h5py access pattern. Given the massive memory footprint of 4D hyperspectral tensors, scripts must utilize direct array slicing along the temporal and spectral axes to prevent RAM exhaustion.

import h5py  
import numpy as np

\# Recommended Access Pattern  
file\_path \= "Tanager\_Native\_Stack\_HDFEOS.h5"  
with h5py.File(file\_path, 'r') as h5:  
    \# 1\. Navigate to the core Data Fields group  
    data\_grp \= h5\['HDFEOS/GRIDS/TANAGER/Data Fields'\]  
      
    \# 2\. Extract datasets by reference (Defers loading to memory)  
    sr\_cube \= data\_grp\['surface\_reflectance'\]  
    sr\_invalid \= data\_grp\['sr\_invalid'\]  
    nodata\_mask \= data\_grp\['nodata\_pixels'\]  
      
    \# 3\. Read metrology and spectral attributes  
    wavelengths \= sr\_cube.attrs\['wavelengths'\]  
    crs\_wkt \= sr\_cube.attrs\['spatial\_ref'\]  
    acq\_times \= sr\_cube.attrs\['acquisition\_time'\]  
      
    num\_frames, num\_bands, height, width \= sr\_cube.shape  
    print(f"Stack Dimensions: {num\_frames} Passes, {num\_bands} Bands, {height}x{width} Spatial Grid")  
      
    \# 4\. Identify a specific band index programmatically (e.g., NIR \~860nm)  
    nir\_target \= 860.0  
    nir\_idx \= int(np.argmin(np.abs(wavelengths \- nir\_target)))  
      
    \# 5\. Chronological Iterator (Memory Efficient)  
    for t in range(num\_frames):  
        \# Load specific band and spatial masks for a single time slice into RAM  
        frame\_nir \= sr\_cube\[t, nir\_idx, :, :\]  
        frame\_invalid \= sr\_invalid\[t, :, :\]  
        frame\_nodata \= nodata\_mask\[t, :, :\]  
          
        \# Combine valid spatial context: Not NoData (1) AND Not Invalid SR (1)  
        valid\_mask \= (frame\_nodata \== 0\) & (frame\_invalid \== 0\)  
          
        \# Extract 1D array of purely valid pixels to prevent synthetic fill bias  
        valid\_nir\_pixels \= frame\_nir\[valid\_mask\]  
          
        if valid\_nir\_pixels.size \> 0:  
            print(f"Pass {t} ({acq\_times\[t\]}): Mean Valid NIR Reflectance \= {np.mean(valid\_nir\_pixels):.4f}")  
        else:  
            print(f"Pass {t}: No radiometrically valid pixels in this temporal frame.")  
