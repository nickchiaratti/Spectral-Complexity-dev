# **Hyperspectral Time-Series Analysis Toolkit**

This project provides a Python-based workflow for downloading, processing, and visualizing multi-temporal hyperspectral and multispectral satellite imagery (specifically Landsat 8/9). It creates standardized HDF5 data cubes from raw source data and performs spectral complexity analysis using the maximum distance with Gram matrix algorithm.

## **Overview**

The toolkit addresses the challenge of analyzing time-series raster data where frames are often unaligned, in different coordinate systems, or stored as disparate files. It unifies these into a single 4D HDF5 structure (\[Time, Band, Y, X\]) and provides a dedicated interactive viewer to explore spectral volume changes over time.

## **Workflow**

1. **Download:** Fetch Landsat Level-2 Science Products (L2SP) directly from the USGS M2M API.  
2. **Process:** Align, crop, and stack individual raster scenes into a unified HDF5 data cube. By processing raw Landsat .tar archives or folders.   
3. **Analyze:** Calculate spectral volume maps using the Maximum Distance algorithm (tiled or sliding window).  
4. **Visualize:** Interactively explore the RGB composites, spectral volume maps, and change detection layers.

## **Usage Guide**

### **1\. Data Acquisition**

Use USGS-API-download-data.py to fetch data.

* **Configuration:** Update the username, token, spatialFilter, output directory, and temporalFilter variables in the script.  
* **Run:** python USGS-API-download-data.py

### **2\. Data Processing (Stacking)**

Choose the processor based on your input data type.

Use landsat-to-hdf5-stacker.py to process downloaded .tar files or folders.

* **Config:** Set Location, INCLUDE\_PAN\_BAND, and QA\_MASK\_THRESHOLD\_PERCENT at the top of the file.  
* **Run:** landsat-to-hdf5-stacker.py  
* **Output:** Creates landsat\_source\_aligned.h5.

### **3\. Visualization & Analysis**

Use stacked-hdf5-viewer.py to explore the generated HDF5 file.

* **Config:**  
  * PROCESSING\_METHOD: 'grid' (faster) or 'sliding' (smoother).  
  * TILE\_SIZE: Size of the window for volume estimation (e.g., 3).  
  * IMAGE\_NORMALIZATION\_METHOD: 'log', 'linear', or 'percentile'.  
  * DISPLAY\_LEFT\_PANEL: 'rgb' or 'pan'.  
* **Run:** python stacked-hdf5-viewer.py  
* **Controls:** Use the slider and buttons to navigate time. The viewer will automatically export processed views to a processed\_images\_tif folder.
