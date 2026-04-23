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

* **Local Temporal Dimension (![][image1]):** Sensor-specific time axis.  
* **Global Temporal Dimension (![][image2]):** The chronologically fused time axis combining all sensor acquisitions.  
* **Spatial Dimensions (![][image3],** ![][image4]**):** Uniform across all grids, locked to a common USGS CONUS Albers Equal Area projection (30m GSD).  
* **Spectral/Endmember Dimensions (![][image5],** ![][image6]**):** ![][image5] varies by sensor (13 for S30, 8 for L30, \~400 for Tanager). ![][image6] denotes extracted endmembers (default 7).

## **3\. Dataset Specifications: Sensor-Specific Grids**

These groups contain the foundational radiometric data, quality masks, and algorithm physics that rely on specific wavelength counts.

### **3.1 Foundational Radiometry & Provenance**

* **surface\_reflectance**  
  * **Shape:** \[![][image1], ![][image5], ![][image3], ![][image4]\] (float32)  
  * **Description:** Bottom-of-Atmosphere (BOA) reflectance scaled 0.0 to 1.0.  
  * **Attributes:** wavelengths (1D array of µm), acquisition\_time, spacecraft\_id.  
* **thermal\_infrared** *(HLSL30 Only)*  
  * **Shape:** \[![][image1], 2, ![][image3], ![][image4]\] (float32)  
  * **Description:** Apparent temperature in Kelvin/Celsius.  
* **ortho\_visual**  
  * **Shape:** \[![][image1], 4, ![][image3], ![][image4]\] (uint8)  
  * **Description:** RGBA representation natively derived from surface reflectance via percentile stretch. The Alpha channel (Index 3\) strictly defines the valid swath footprint (255 \= Data, 0 \= NoData/Fill).  
* **source\_tile\_mask** *(HLS Only)*  
  * **Shape:** \[![][image1], ![][image3], ![][image4]\] (uint8)  
  * **Description:** An integer mapping tracing each pixel back to its discrete native MGRS origin tile (e.g., T17TQJ vs T17TQH) to track radiometric overlap provenance.

### **3.2 Quality Assessment (QA) & Masking**

* **Fmask** *(HLS Only)* / **sr\_invalid** *(Tanager Only)*  
  * **Shape:** \[![][image1], ![][image3], ![][image4]\] (uint8)  
  * **Description:** Raw, bit-packed categorical masking flags native to the sensor.  
* **common\_mask**  
  * **Shape:** \[![][image1], ![][image3], ![][image4]\] (uint8)  
  * **Description:** A pre-calculated, strict Boolean validity mask. Evaluated via the SpecComplex.py library logic. Consolidates sun angle physics, cloud dilations, aerosol optical depth limits, and spectral uncertainty into a single binary flag (1 \= Valid, 0 \= Invalid).

### **3.3 Global Endmember Extraction**

* **frame\_endmembers**  
  * **Shape:** \[![][image1], ![][image5], ![][image6]\] (float32)  
  * **Description:** Spectral signatures of the ![][image6] most spectrally extreme pixels in the valid frame.  
* **frame\_endmember\_indices**  
  * **Shape:** \[![][image1], ![][image6]\] (int32)  
  * **Description:** 1D flattened spatial coordinates of the extracted endmembers.  
* **frame\_endmember\_volumes**  
  * **Shape:** \[![][image1], ![][image6]\] (float32)  
  * **Description:** The expanding parallelotope volume curve (Gramian determinant) as endmembers are sequentially added to the simplex.

## **4\. Dataset Specifications: HARMONIZED Grid**

This group acts as a chronological relational database. It interleaves the derived spatial mathematics of all HLST sensors into continuous time-series arrays.

### **4.1 Relational Provenance Attributes (CRITICAL)**

Every dataset within the HARMONIZED group contains four 1D arrays of length ![][image2]. These act as Foreign Keys pointing back to the sensor-specific tables:

* **acquisition\_time**: UNIX timestamps (UTC) enforcing the chronological sort.  
* **source\_grid**: String array denoting the origin group (e.g., 'HLSS30').  
* **source\_spacecraft**: String array denoting the specific platform.  
* **source\_frame\_index**: Int32 array mapping ![][image2] back to the exact ![][image1] index within the native sensor grid.

### **4.2 Harmonized Analytical Arrays**

* **sliding\_volume\_map**  
  * **Shape:** \[![][image2], ![][image3], ![][image4]\] (float32)  
  * **Description:** Convex hull spectral volume calculated over a sliding 3x3 window using the Gramian determinant.  
  * **Attributes:** tile\_size (3), gram\_type, Normalization (e.g., bandCount).  
* **sliding\_volume\_z\_score**  
  * **Shape:** \[![][image2], ![][image3], ![][image4]\] (float32)  
  * **Description:** Global Spectral Complexity Z-score. The background statistical distribution (![][image7]) is mathematically shielded from cloud contamination via strict evaluation of the common\_mask before Z-scores are projected.  
* **msd\_map**  
  * **Shape:** \[![][image2], ![][image3], ![][image4]\] (float32)  
  * **Description:** Mean Spectral Distance calculated over a sliding 3x3 window.  
* **ndvi\_map** / **ndbi\_map**  
  * **Shape:** \[![][image2], ![][image3], ![][image4]\] (float32)  
  * **Description:** Band-agnostic spectral indices. Calculated *before* hyperspectral pruning to preserve strict red/nir wavelength physics.  
* **common\_mask**  
  * **Shape:** \[![][image2], ![][image3], ![][image4]\] (uint8)  
  * **Description:** A direct temporal interleaving of the sensor-specific common\_mask arrays. Permits ![][image8] QA filtering during time-series extraction.

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

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACkAAAAYCAYAAABnRtT+AAACoUlEQVR4Xu2Wz4uNURjH39u9I924XO7vH+9786uELESRIlFsLIiUmJVhYWEsiLCzIT+6KIspSUwkWRgr0WwmjbLDSskfYCFWZHy+c88xZ17u3Ku3GVfdb317zvk+533O85zzvPe9ntdFFx2IIAi2+L7/Hn5sk1vDMaYbMZK8zsb3C4VCTXOJaANoPyqVynazLs54E/qHarW69tfTM4FcLpcnmQf5fD5nNeZp+EoJkVjZ6tlsdg76nXK5XLHajEBXx8n0uxpJrCbBz/geMk1Y3SRfz2Qyc53l0w823UNSy1yNBPfDMXwnXZ11C9H6PNMS/xSmH7/BjWFfR6BZP3YUSHAN/BruR4ME/XsR33eK2BXyRYL6nJiDehda/no060eLUqmUwf8CLg/7okIxFVt7hH0uYiR3a6p+lA6H1RZhX1TogJrc4ATa6Uf8h2Hd1biexWhX4GN4ECkuXSfC/Bx84jd+FXrkI/5u+AjtAHaDZ5JSXMWfiPwH+FP3o5CQTxVbgWLWKTnsAs/chL5SxWKRZcEQ46VwM/qbZDJZQrsBj2otxZ22e/mNA9IN/X6D+o3kodc4P2HHLJl/wb5T4nZtuB/T6fQ8xiNsttOuYf1tYp5CH/RNX9dqtdlwPtp6tFF95czaOs8e11gxFbtVP7aEqjTVjvejCiDwW+wiM9dpjMJeo086FSWNfpdhTJ9Yxk/tmqCdfmwHvulHrm+HKjefz+f6Ehn/NuZDqVRqCfYlp7TSeXYF87NKVHP5tEany3if4pr4vZFOkwB7CXoPe8JrvAQ9jK+incceww5wrQWv0W/9zC8rAewlFQZX6bTQjmBvwmfwggqGZ9CvMT+k5yfv/Jcwfy7G314LXZ3oaoLtxZAcd/6gqNBZZhwzerQEu/jf8RPU4LJkTNgZwwAAAABJRU5ErkJggg==>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAC8AAAAYCAYAAABqWKS5AAADFklEQVR4Xu2WX4jMURTHZ9qZIqT5P7Mz+/tNlCIlrX9JKQlP62FbkQcklJZaHqQ8kQcPSmMVtaWlsEu7T8qTJcWDQl5Iqc2r8iCelPX5zty73f1lZn7Ny4zMqdO953vO7/y+93fOvfcXiXSlK/+h+L6/w/O8z+iXkLozmKNdEoX8KIQm8vl8WbZAsDGw36VSaY+J62G+HXy2r69v4/zT7ZRsNpuD5GQul8taDDuBvhZRCBctnslkloLfLRaLJYu1VdQCfMkRF4PcOoh/x/cIM2Zxs6hKOp1e5oS3TyAzBNlVLgbxg+gcvnMuTlwK7HjEtFZHiun3X+i2oK+jpV6//xMC8X70Z7Dfm4n2AQu+r73S6DRS6xHzlPxfGVcH/X8T7UHiP6EvGu63ev0eRkQGnent7U0Hfa6oHdHnqnLQV0+IPYFWgrgrUQJuey32uxYepmJU5kwzIgGJKa/yBx3zoi/hheh3Ytagt4gbhcguYpMGr+gLOaFx/PuJm0ZvmEuwSgS9ho57tdMrrmDdH8Rd5Znr6G7LQZVURRu2mRei30m4iSRTiURiOS9Yy/yDkpqFqxVsxeIiTMwB5lHtA3yTyWRSBN+gg8LB7uAbwF6v58vlct4s4qPNpdHkXthmOuOVDMc3xjmr2D9ETAuysbphwZ75pnxuUi3Ad/rdfPEZPWNi+7Hfge91iFSrAIfzfm2zDyvW5Hql298827Tfm4rd9Uou202qBYlIxFSM+TjYZfus/Fo4lTvNOCbMtMNbYveh7+0ppUpgT5pczfs9jJBkBUleooVIrS3Uy7YK1X5HDxlSY5A4LB+EF4M/BBs0m7V6kumnD+wxX3ilFmbyVn8UFYceSaVS1YqiW9Gj82RaECUe5uUTjPfQWd9UgfGCNhq+Y4pjvoH5Ey3Gq51gQ8ILhYIvG/9JxgfmJy+GfUk5wK4INzEj2lvgU8wvErtlAZtWRSX2a5eN3URRc4G4/z09upQ0OpgkLlIBLMKf7RL5nLnNFTd2a8IpsAiy05A9hRlnvIl9NhjXqaJWGFB5pfTrZmHBoK6ElD9dqN1BQqiMFQAAAABJRU5ErkJggg==>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABIAAAAYCAYAAAD3Va0xAAABFUlEQVR4XmNgGAUkAWlpaUc5ObnX8vLy/2EYyH8HxLFSUlIiQPogEP9FkvsCxCvExcW50c0CA6CiOUAF/4DYBV0OJAaVmw/kMqLLwwFQgSAQnwYa9kBGRkYai3w51EXR6HIoAKhAE6j4LRCvAXJZ0KRZQOIgeZA6NDlUALIJ6v9ydDmQC0EuBbkY5HJ0eRQAVDAJiP/IysoGADVJImOgeCg0fCah60MBsPAB4q9AvAiIZyFjoGH3ByZ8gN4qQpcjKXzkIennNxDboMuBxKByxIUPyFaK0g9QobEcJJAxwkdBQYEDaMBWIP4EzEb6yHJwIAdJ8s+gtsHwK2A4RQI1CQPldgP5v5DkQOylwLzHhW7WKBixAACd3WiJOQPFsQAAAABJRU5ErkJggg==>

[image4]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABUAAAAYCAYAAAAVibZIAAABh0lEQVR4Xu2UL0gDcRTHb7AF0aI4N293N0ERRDC4bHPRIphmtxhsQxENCopJg1hEWNJqlyEYDAa7YDCZjAaL8/PY74537zYtxn3hsff7fve+7/3+cJ43wL/D9/3xarX6SHRUvEdRVKtUKgH5s9bgX+BnpZZ83Wi3pVJpODGHaDpxMyEdwjDccdqq1YIgWIBvl8vlKat5CA3XrWk1ZdowUg5ul5q64btAWJFCMdC8TID22qsh63nihLSg+QSIy8Q30VK0THIId9rDNM/6SIwVlwZijfjUpo47lrN0ponGjpZYb5PmYi4DZXrnbrBAfsZFzNiGxWJxhPyClxMamzSYZJJ4I+5dUZ3YEs02lAsj37AeGcSm/PlJpiO/wmDCaXPwH9KQbU/zey6NrUcGFI2KoTPeJ9ZiTTV8IPZkF7q2L6SzTBJ1X8AN0w7FmjqaDnHt9XtCFnJWcmYUfcnNak01lHOtae1PUNCi+JI0r/nYlGYH3m9PqBdkCvnAWF6A6SJHMmb5AQZI4wesBHZS/gyF9gAAAABJRU5ErkJggg==>

[image5]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAYCAYAAADzoH0MAAABPElEQVR4XmNgGAUoQF5e3kpOTu4RkP4Pw0D+FyB+BmX/BdJbpaWl1dD1ogCgwklA/E1WVtYUWVxGRkYVKH4XaMh1oCEyyHJwICoqygNUcACo8KqUlJQIujxQfCHUNb7ocmAAlFACKngOpOcDuYzIcjDDgfgnEFsiy8EB0Nl+UBvSsciFQ8NhCpDLgi4PBkDJVpANQP96AmlJEAYCeaDGeiD7JdCQSKAyZnR9YIDk/9cgLwDxLBAGis0FhQmQbhESEuJD1wcH+PwvISGhAI2Bc+Li4mLIcnCAz/8gAIsBkPfQ5cBADkf8gwBQEydQbgcQ/wNiF3R5gvEPNNQWFLhA+V0gtejyoNDXBEq+BdJLGVD9zwyNkfdA+SugGEGSg5v8EOQ3qP//AvETIH4Epf8AxR8A6VyQN1A0j4LhAgBMhWANssplWgAAAABJRU5ErkJggg==>

[image6]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABIAAAAYCAYAAAD3Va0xAAABSElEQVR4Xu2TvUoDURCFdyGNCoa7yC7uL5aCYCEWgo1iEbCzjX0aQfAJ0tpooZVWVjaSOvgEPoF5ATsRGxst1G9krt7cDZrU5sBhZ8+cO3NnNgmCKSZClmVbZVk+VlX1ISTu53k+Y/NRFM2j3dq8spckyZxbxyLEfIHhDb7CDd+Atofnxm1SAwYDr+ChdjxHDl1PURRH6G1Xq4HxVilyinERDogfeC45lgbapfgcrQ7phLEjMc+u3urA5tM0XdAbm59TI8ChE0xrEjPCCu/P8M4Y0xSN3Cb62fApD3Y/0lWlBkWu0d5ZbEs9nbH3EzjLlQJSSArKV5p4PxYykoymI26Ps59QZofrfoKD+7r0e+JjPz+EEfv5RhzHCUUGWuz3/ci1YY9Cs35OQJMufMKz7Oe+QHIHvmg3ofwtdn2f/BTw9f/azxT/Ep+y+Fa5aJHwtgAAAABJRU5ErkJggg==>

[image7]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAYCAYAAACbU/80AAAB6UlEQVR4Xu2UMUsDQRCFE1RQVDSaXDhyuQt2giIiFoJ2Nhba+CMEsRK0sBAbsRBB7JRYihbWYmFhLYIIFjYWVlYiWAhBon5z7sa9jXAmhdU9GHb37Zu5mdndS6USJEiQwIDruoHnecMW3ZLNZrstrhG0OY6TD4LANa1UKvVGVLlcrouNS+yIZVrzvu9vwN0WCoV+Qx6LfD7fie8uVsX/0zb4C9HUHCAG2HgqFovLVpALO6k4ZDKZHuV3Q7wp6azEgHtmnMBcKTjihHCOjQqiSc3ppBgXTG0M0qryOwpwNMl6jFivHPGMKa4BwR72IOelud+SigP6QVXpkslLDOwdmzX5EPr82Txl2ap5uE07qTjIB7A3kh+3+AWV2KDJh9BZ47SmOTMpmbO3rVsqa3Up6+6FJIDfo5yz5mh7B/w5XDllFFiDavWnVKw55vM4VSUpbIj1DnRaPsz8FqtgE0aYEEp7L6Oi5E6sio9cxohYw/8+/w9EL4xb2Am2zvqQ8YrxGBsVrerMmdKv2LFAmo8vsn+NHQTfXdynC322MITd6iD6RMKK655MKkx6OrAumgl+NO0SS0Z7L4Lg5/xr7/8vUG2tO4KG0cxToysj+JTlctl7DUMqlwvSyK+WZGfRezbfLOQptdhkgv/CF/+afo35+4fkAAAAAElFTkSuQmCC>

[image8]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACgAAAAYCAYAAACIhL/AAAAC10lEQVR4Xu2WP2gUQRjF70gOFBQ9vT94f/aOYCNaGKJioRYiokUUJKCQxk5r/xKrs0hhIxIEIU20UjCtBmwMRES0E4JCFFFCREXT2ImJv5fshrnvdtc9TUTQBx9z+733zbyZndm5VOo//jLk8/k19Xp9lc3HIJPNZtfZZCKUy+WN1Wr1oOd5fbVabQupDqtxgX472pE2B8xQc43+j1kiCmlM7aPoKUVjRL+C5wfEFNxOWyBgroLuIZqtlhMqlcpqak+yuustp4Wg9j6aXZaz0GyuIH4TYkTcMNws0W24tFaBaLhJBtxAP8fJ3/Tr3hKbXE0AtIe0CNoilgsgAzeIL1Ez0WuG/0x7ncd0kMfENnIv1TrywOBRYgd1d+IMalvAPUZ7wnILgDhNJ3NqLRcAPks8IyZLpVLOyV+k83txhwPNrTiDAtwgulF+djYRzHQzxAyCF4VCodhEOggMugPJlMwxsUtW7yKJQXwclkb7uYmguAExrxk0EQboutC8dwdSq2e4Xqt3kcQgmh5iumn/a1NSNA4xRxxw9C0Q7+smcrncWj/XQ/0H2j1W7yKJwdDJOkltfn3rIoFmCM2855xWf9bv1DrSFrRjkOgPS8YWcyiqDDJFfPKcb91KGOQVn1lK6jRSPPmTYn3nBrR6xFmXWAmDntnPnSRvk/zmRewjfRfRzMIP85hxOXJdxIxOoJu3SGIwsi+Kun0DIyljgPx+4iNxVdeVywnBGyBOWc6FDBLTLZ8QBzq9aF7Xws6CT77yFu/ghfuXGOP5uUymnJvDQK9/hBiyRLFYLJCfIL7WFreHDth3GQ27EPwxx+Ouuw65p4M+opdOSqloY0vQ9aSJEVnLtYFgqzUs8dvQvxE6fqQL33JJodsMg0/UWm5ZwCoeweTdsH2aANoml4kL+m3J5UKaFTiv0G9LxoHJ7cXcaJt/dH8JGQye45XvtkQUWPEyNYN/wty/iR/wHdO5XqrlcgAAAABJRU5ErkJggg==>