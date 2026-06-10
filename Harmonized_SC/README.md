# Harmonized Landsat Sentinel & Tanager (HLST) Spectral Complexity Pipeline

## 🌍 Overview

The **HLST Spectral Complexity Pipeline** is a set of Python integration tools and data processing scripts designed to download, coregister, and analyze satellite imagery from NASA's Harmonized Landsat Sentinel-2 (HLS) program and the JPL Tanager-1 mission. The pipeline fuses hyperspectral and multispectral sources into unified **Analysis-Ready Data (ARD) HDF-EOS5 Image Cubes**. 

Once generated, the pipeline executes high-performance analytical algorithms to calculate temporal spectral complexity (sliding spectral volume, mean spectral distance, and anomaly Z-scores) with robust cloud mask shielding and spatial provenance tracking.

---

## 🚀 Core Pipeline Modules

The primary data extraction and analytics workflow follows three sequential phases:

### Phase 1: Native Ingestion
**`HLS30-earthAccess-to-hdf5.py`**
*   Automates high-throughput geospatial downloading of Sentinel-2 (S30) and Landsat 8/9 (L30) imagery spanning defined temporal boundaries over specific planetary regions of interest (ROI).
*   Built aggressively on NASA Earth Access APIs and STAC concurrent window-reads to stream unprojected (native MGRS footprint) scenes directly into a local foundational Native Data HDF5 structure.

### Phase 2: Master Grid Co-Registration
**`HLST-constellation-to-hdf5.py`**
*   Converts the native unprojected sensor stacks (including Tanager-1) into a structurally unified, Master Grid co-registered stack. 
*   Dynamically centers the projection over the designated ROI using a Deetz & Adams Albers Equal Area projection for zero-distortion spatial tracking. 
*   Synthesizes `Fmask` and hyperspectral algorithmic properties to generate rigorous sensor-agnostic visual overlays and Boolean validity masks.

### Phase 3: Spatial Mathematics
**`HLST_SC_calculations.py`**
*   Spools mathematical analytics across the unified cube leveraging an underlying highly parallelized architecture.
*   Extracts global endmembers iteratively via Simplex expanding algorithms. 
*   Populates the core `HARMONIZED` analytical groups with relational provenance tracking. Implements:
    *   **Sliding Spectral Volume Maps**
    *   **Mean Spectral Distance (MSD)**
    *   **Global Structural Z-scores**
    *   **Generals Indices (NDVI / NDBI)** 

---

## 🛠 Analytics & Auxiliary Scripts

*   **`SpecComplex.py`**: The definitive computational engine. Handles algorithmic abstractions (QA Mask mappings, parallel geometries/gramian determinants calculations, interpolation algorithms, spatial data alignment). 
*   **`HLST_registration_quality_quantification.py`** & **`HLST_registration_quality_multisensor_quantification.py`**: Utilized to enforce accuracy bounds and statistically validate that the raw multispectral and hyperspectral footprints overlay correctly in real-world map-space. 
*   **`HLST-videoGen.py`**: An automated media rendering script designed to translate the complex multidimensional HDF5 frames into animated visual videos or GIFs over the chronologically tracked temporal dimension.
*   **`mgrs_view.py`**: A helper utility to trace native structural boundaries.

---

## 📚 Documentation & Schema

To use the output data, refer directly to the comprehensive **[HLST ARD Data Dictionary.md](./HLST%20ARD%20Data%20Dictionary.md)**. 

The dictionary acts as the developer map/rulebook for parsing the HDF-EOS5 outputs. It structurally outlines all explicit tensor array shapes mapping dimensions like Time ($T_G$), Band Arrays ($B$), and Endmembers ($E$) across the separated multisensor architecture schema.
