import os
import sys
import urllib.request
import h5py
import json
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "HLSX30"))
import SpecComplexQR as scQR
import SpecComplex as sc

# Configurations
THUILLIER_URL = "https://oceancolor.gsfc.nasa.gov/docs/rsr/f0.txt"
THUILLIER_FILE = "thuillier_2003.txt"
DATA_DIR = r"C:\satelliteImagery\Tanager\Rochesterv2_SourceData"
STACKED_H5 = os.path.join(DATA_DIR, "Tanager_Native_Stack_Rochesterv2.h5")

TILE_SIZE = 3
SLIDING_STRIDE = 1

def download_thuillier():
    if not os.path.exists(THUILLIER_FILE):
        print(f"Downloading Thuillier 2003 from {THUILLIER_URL}...")
        urllib.request.urlretrieve(THUILLIER_URL, THUILLIER_FILE)
    
    wavelengths = []
    esun = []
    with open(THUILLIER_FILE, 'r') as f:
        for line in f:
            if line.strip() and not line.startswith(('#', '!', '/')):
                parts = line.split()
                if len(parts) >= 2:
                    wavelengths.append(float(parts[0]))
                    # Convert uW/cm^2/nm to W/m^2/um
                    # 1 uW/cm^2/nm = 10 W/m^2/um
                    esun.append(float(parts[1]) * 10.0)
    
    return np.array(wavelengths), np.array(esun)

def get_earth_sun_distance(dt):
    day_of_year = dt.timetuple().tm_yday
    d = 1 - 0.01672 * np.cos(np.radians(0.9856 * (day_of_year - 4)))
    return d

def main():
    thuill_wvl, thuill_esun = download_thuillier()

    print(f"Opening {STACKED_H5}...")
    with h5py.File(STACKED_H5, 'r+') as h5f:
        grp = h5f['HDFEOS/GRIDS/TANAGER/Data Fields']
        meta_grp = h5f['METADATA']
        
        n_passes = grp['surface_reflectance'].shape[0]
        bands = grp['surface_reflectance'].shape[1]
        height = grp['surface_reflectance'].shape[2]
        width = grp['surface_reflectance'].shape[3]
        
        # Create output datasets if they don't exist
        if 'toa_reflectance' not in grp:
            grp.create_dataset('toa_reflectance', shape=(n_passes, bands, height, width), dtype='float32', compression='gzip', fillvalue=-9999.0)
        if 'sr_zscore' not in grp:
            grp.create_dataset('sr_zscore', shape=(n_passes, height, width), dtype='float32', compression='gzip', fillvalue=np.nan)
        if 'toa_zscore' not in grp:
            grp.create_dataset('toa_zscore', shape=(n_passes, height, width), dtype='float32', compression='gzip', fillvalue=np.nan)

        tanager_wvl = grp['surface_reflectance'].attrs['wavelengths']
        tanager_esun = np.interp(tanager_wvl, thuill_wvl, thuill_esun)
        
        for t_idx in range(n_passes):
            print(f"\n--- Processing Pass {t_idx + 1}/{n_passes} ---")
            
            # 1. Parse JSON to get sun elevation and timestamp
            json_str = meta_grp.attrs[f"frame_{t_idx}_json"]
            stac_data = json.loads(json_str)
            sun_elev = stac_data['properties']['view:sun_elevation']
            dt_str = stac_data['properties']['datetime']
            dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
            d = get_earth_sun_distance(dt)
            
            # 2. Compute TOA Reflectance
            print("Calculating TOA Reflectance...")
            radiance = grp['toa_radiance'][t_idx, ...]
            rad_fill = grp['toa_radiance'].attrs['_FillValue']
            if isinstance(rad_fill, (np.ndarray, list, tuple)): rad_fill = rad_fill[0]
            
            toa_reflectance = np.full_like(radiance, rad_fill)
            valid_mask_rad = (radiance != rad_fill)
            
            denominator = tanager_esun[:, None, None] * np.sin(np.radians(sun_elev))
            denominator_full = np.broadcast_to(denominator, radiance.shape)
            toa_reflectance[valid_mask_rad] = (np.pi * radiance[valid_mask_rad] * (d**2)) / denominator_full[valid_mask_rad]
            
            grp['toa_reflectance'][t_idx, ...] = toa_reflectance
            
            # 3. Read mask and apply good bands
            common_mask = grp['common_mask'][t_idx, ...]  # Boolean mask, True = invalid
            valid_spatial_mask = ~common_mask
            
            gw = grp['surface_reflectance'].attrs['all_good_wavelengths'][t_idx]
            good_bands = (gw == 1)
            
            print(f"Using {np.sum(good_bands)} good bands.")
            
            sr_data = grp['surface_reflectance'][t_idx, good_bands, :, :]
            toa_data = toa_reflectance[good_bands, :, :]
            
            # 4. Spectral Complexity
            print("Processing SR Spectral Complexity...")
            sr_vol = scQR.process_volume_sliding_tile(sr_data, TILE_SIZE, SLIDING_STRIDE, num_endmembers=7, gram_type='minEndmember', norm_type='bandCount')
            
            print("Processing TOA Spectral Complexity...")
            toa_vol = scQR.process_volume_sliding_tile(toa_data, TILE_SIZE, SLIDING_STRIDE, num_endmembers=7, gram_type='minEndmember', norm_type='bandCount')
            
            sr_vol[~valid_spatial_mask] = np.nan
            toa_vol[~valid_spatial_mask] = np.nan
            
            sr_z = sc.calculate_global_z_score(sr_vol, valid_spatial_mask)
            toa_z = sc.calculate_global_z_score(toa_vol, valid_spatial_mask)
            
            grp['sr_zscore'][t_idx, ...] = sr_z
            grp['toa_zscore'][t_idx, ...] = toa_z
            
    print("\nProcessing complete. Data saved to HDF5.")

if __name__ == "__main__":
    main()
