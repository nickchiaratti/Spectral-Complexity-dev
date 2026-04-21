# **HLST (Harmonized Landsat Sentinel & Tanager) ARD Data Dictionary & Navigation Guide**

## **1\. Overview**

This document describes the structure, contents, and metadata of the HLST (Harmonized Landsat Sentinel & Tanager) Analysis Ready Data (ARD) image cubes. This unified Level-3 product mathematically fuses data from NASA's Harmonized Landsat Sentinel-2 (HLS) program and the JPL Tanager-1 hyperspectral mission into a single, continuous spatiotemporal framework.

The file strictly adheres to the HDF-EOS5 grid specification. To preserve absolute data lineage, the architecture employs a **Separation of Concerns (SoC)** model: raw radiometric data and band-dependent physics (Endmembers) are isolated in sensor-specific grids, while mathematically unified indices (Volume, Z-Scores, NDVI) are interleaved into a single chronologically sorted HARMONIZED timeline.

**Note on Failure Handling & Data Integrity:** Following strict data science directives, synthetic fill values (e.g., 0.0) are *not* used to represent missing or masked data in floating-point analytical arrays. Radiometrically invalid pixels, cloud-covered areas, and geometries outside the swath footprint are rigorously populated with NaN (Not a Number) to prevent artificial skewing of downstream models.

## **2\. HDF5 Hierarchy & Base Paths**

The HLST ARD Cube is divided into distinct HDF-EOS5 Grid structures.

* **Sensor-Specific Grids:**  
  * /HDFEOS/GRIDS/HLSS30/Data Fields (Sentinel-2A/B)  
  * /HDFEOS/GRIDS/HLSL30/Data Fields (Landsat 8/9)  
  * /HDFEOS/GRIDS/TANAGER/Data Fields (Tanager-1)  
* **Unified Temporal Index:**  
  * /HDFEOS/GRIDS/HARMONIZED/Data Fields

### **Dimensional Conventions & Memory Layout**

* **Local Temporal Dimension ($T_{L}$):** Sensor-specific time axis.  
* **Global Temporal Dimension ($T_{G}$):** The chronologically fused time axis combining all sensor acquisitions.  
* **Spatial Dimensions ($Y$, $X$):** Uniform across all grids, locked to a Dynamically Centered Albers Equal Area projection (30m GSD) over the chosen ROI.  
* **Spectral/Endmember Dimensions ($B$, $E$):** $B$ varies by sensor (13 for S30, 8 for L30, \~400 for Tanager). $E$ denotes extracted endmembers (default 7).

## **3\. Dataset Specifications: Sensor-Specific Grids**

These groups contain the foundational radiometric data, quality masks, and algorithm physics that rely on specific wavelength counts.

### **3.1 Foundational Radiometry & Provenance**

* **surface\_reflectance**  
  * **Shape:** \[$T_{L}$, $B$, $Y$, $X$\] (float32)  
  * **Description:** Bottom-of-Atmosphere (BOA) reflectance scaled 0.0 to 1.0.  
  * **Attributes:** units, \_FillValue, wavelengths, spatial\_ref, GeoTransform, spacecraft\_id, acquisition\_time, sun\_azimuth, sun\_elevation, cloud\_cover.
* **thermal\_infrared** *(HLSL30 Only)*  
  * **Shape:** \[$T_{L}$, 2, $Y$, $X$\] (float32)  
  * **Description:** Apparent temperature in Kelvin/Celsius.  
  * **Attributes:** units, \_FillValue, wavelengths.
* **ortho\_visual**  
  * **Shape:** \[$T_{L}$, 4, $Y$, $X$\] (uint8)  
  * **Description:** RGBA representation natively derived from surface reflectance via percentile stretch. The Alpha channel (Index 3\) strictly defines the valid swath footprint (255 \= Data, 0 \= NoData/Fill).  
  * **Attributes:** spatial\_ref, GeoTransform.
* **source\_tile\_mask** *(HLS Only)*  
  * **Shape:** \[$T_{L}$, $Y$, $X$\] (uint8)  
  * **Description:** An integer mapping tracing each pixel back to its discrete native MGRS origin tile (e.g., T17TQJ vs T17TQH) to track radiometric overlap provenance.
  * **Attributes:** \_FillValue, description, tile\_mapping (JSON).

### **3.2 Quality Assessment (QA) & Masking**

* **Fmask** *(HLS Only)* / **sr\_invalid** *(Tanager Only)*  
  * **Shape:** \[$T_{L}$, $Y$, $X$\] (uint8)  
  * **Description:** Raw, bit-packed categorical masking flags native to the sensor.  
  * **Attributes:** \_FillValue.
* **solar\_view\_angles**  
  * **Shape:** \[$T_{L}$, 4, $Y$, $X$\] (float32)  
  * **Description:** Per-pixel angular geometry describing sensor-target illumination.
  * **Attributes:** \_FillValue, band\_order (\["SZA", "SAA", "VZA", "VAA"\]).
* **common\_mask**  
  * **Shape:** \[$T_{L}$, $Y$, $X$\] (uint8)  
  * **Description:** A pre-calculated, strict Boolean validity mask. Evaluated via the SpecComplex.py library logic. Consolidates sun angle physics, cloud dilations, aerosol optical depth limits, and spectral uncertainty into a single binary flag (1 \= Valid, 0 \= Invalid).
  * **Attributes:** \_FillValue, description, spatial\_ref, GeoTransform.

### **3.3 Global Endmember Extraction**

* **frame\_endmembers**  
  * **Shape:** \[$T_{L}$, $B$, $E$\] (float32)  
  * **Description:** Spectral signatures of the $E$ most spectrally extreme pixels in the valid frame.  
  * **Attributes:** description, num\_endmembers, Normalization.
* **frame\_endmember\_indices**  
  * **Shape:** \[$T_{L}$, $E$\] (int32)  
  * **Description:** 1D flattened spatial coordinates of the extracted endmembers.  
  * **Attributes:** description.
* **frame\_endmember\_volumes**  
  * **Shape:** \[$T_{L}$, $E$\] (float32)  
  * **Description:** The expanding parallelotope volume curve (Gramian determinant) as endmembers are sequentially added to the simplex.
  * **Attributes:** description, gram\_type, num\_endmembers, Normalization.

## **4\. Dataset Specifications: HARMONIZED Grid**

This group acts as a chronological relational database. It interleaves the derived spatial mathematics of all HLST sensors into continuous time-series arrays.

### **4.1 Relational Provenance Attributes (CRITICAL)**

Every dataset within the HARMONIZED group contains four 1D arrays of length $T_{G}$. These act as Foreign Keys pointing back to the sensor-specific tables:

* **acquisition\_time**: UNIX timestamps (UTC) enforcing the chronological sort.  
* **source\_grid**: String array denoting the origin group (e.g., 'HLSS30').  
* **source\_spacecraft**: String array denoting the specific platform.  
* **source\_frame\_index**: Int32 array mapping $T_{G}$ back to the exact $T_{L}$ index within the native sensor grid.

### **4.2 Harmonized Analytical Arrays**

* **sliding\_volume\_map**  
  * **Shape:** \[$T_{G}$, $Y$, $X$\] (float32)  
  * **Description:** Convex hull spectral volume calculated over a sliding nxn window using the Gramian determinant.  
  * **Attributes:** description, tile\_size, sliding\_stride, gram\_type, num\_endmembers, Normalization.  
* **sliding\_volume\_z\_score**  
  * **Shape:** \[$T_{G}$, $Y$, $X$\] (float32)  
  * **Description:** Global Spectral Complexity Z-score. The background statistical distribution ($\mathcal{N}(\mu, \sigma^2)$) is mathematically shielded from cloud contamination via strict evaluation of the common\_mask before Z-scores are projected.  
  * **Attributes:** description, MASKING\_APPLIED, MASK\_SOURCE.
* **msd\_map**  
  * **Shape:** \[$T_{G}$, $Y$, $X$\] (float32)  
  * **Description:** Mean Spectral Distance calculated over a sliding nxn window.  
  * **Attributes:** description, tile\_size, sliding\_stride.
* **ndvi\_map** / **ndbi\_map**  
  * **Shape:** \[$T_{G}$, $Y$, $X$\] (float32)  
  * **Description:** Band-agnostic spectral indices. Calculated *before* hyperspectral pruning to preserve strict red/nir wavelength physics.  
* **common\_mask**  
  * **Shape:** \[$T_{G}$, $Y$, $X$\] (uint8)  
  * **Description:** A direct temporal interleaving of the sensor-specific common\_mask arrays. Permits $\mathcal{O}(1)$ QA filtering during time-series extraction.

## **5\. AI Agent / Developer Navigation Guide (Python)**

To effectively execute multi-sensor temporal tracking, you must use the HARMONIZED grid as an indexer. This ensures continuous time-series extraction without hardcoding sensor-switching logic.

**Architectural Access Pattern (Relational Indexing):**

import h5py  
import numpy as np  
from datetime import datetime, timezone

file\_path \= "HLST\_Cube\_Rochesterv2\_MasterGrid\_2025.h5"  
with h5py.File(file\_path, 'r') as h5:  
      
    \# 1\. Access the Unified Harmonized Index  
    harm\_grp \= h5\['/HDFEOS/GRIDS/HARMONIZED/Data Fields'\]  
      
    z\_scores \= harm\_grp\['sliding\_volume\_z\_score'\]  
    unified\_masks \= harm\_grp\['common\_mask'\]  
      
    \# 2\. Extract Relational Provenance Keys  
    prov\_grids \= z\_scores.attrs\['source\_grid'\]  
    prov\_times \= z\_scores.attrs\['acquisition\_time'\]  
    prov\_indices \= z\_scores.attrs\['source\_frame\_index'\]  
      
    num\_global\_frames \= len(prov\_times)  
      
    \# 3\. Time-Series Traversal  
    for t\_global in range(num\_global\_frames):  
        dt \= datetime.fromtimestamp(prov\_times\[t\_global\], tz=timezone.utc)  
          
        \# Decode relational pointers  
        source\_grid \= prov\_grids\[t\_global\].decode('utf-8')  
        t\_local \= int(prov\_indices\[t\_global\])  
          
        \# 4\. Strict Late-Binding Mask Application  
        valid\_pixel\_mask \= (unified\_masks\[t\_global, :, :\] \== 1\)  
          
        \# Dynamically apply to data  
        valid\_z\_scores \= np.where(valid\_pixel\_mask, z\_scores\[t\_global, :, :\], np.nan)  
          
        \# 5\. Extract Native Sensor Properties (If needed)  
        \# Because we have the 'source\_grid' and 't\_local' keys, we can instantly pivot   
        \# back to the native grid to extract RGBA visuals or exact Endmember physics.  
        ortho\_rgb \= h5\[f'/HDFEOS/GRIDS/{source\_grid}/Data Fields/ortho\_visual'\]\[t\_local, :, :, :\]  
          
        print(f"\[{dt.strftime('%Y-%m-%d')}\] Processed {source\_grid} (Local Idx: {t\_local}). Mean Z-Score: {np.nanmean(valid\_z\_scores):.3f}")








