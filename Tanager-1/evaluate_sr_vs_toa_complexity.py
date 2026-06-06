import os
import sys
import urllib.request
import h5py
import json
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

import sys
# Add HLSX30 to path to import SpecComplexQR and SpecComplex
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "HLSX30"))
import SpecComplexQR as maxD
import SpecComplex as nsc

# Configurations
THUILLIER_URL = "https://oceancolor.gsfc.nasa.gov/docs/rsr/f0.txt"
THUILLIER_FILE = "thuillier_2003.txt"
DATA_DIR = r"C:\satelliteImagery\Tanager\Rochesterv2_SourceData\20250704_165204_61_4001"
H5_RAD = os.path.join(DATA_DIR, "20250704_165204_61_4001_ortho_radiance_hdf5.h5")
H5_SR = os.path.join(DATA_DIR, "20250704_165204_61_4001_ortho_sr_hdf5.h5")
JSON_PATH = os.path.join(DATA_DIR, "20250704_165204_61_4001.json")

TILE_SIZE = 3
SLIDING_STRIDE = 1

def download_thuillier():
    if not os.path.exists(THUILLIER_FILE):
        print(f"Downloading Thuillier 2003 from {THUILLIER_URL}...")
        urllib.request.urlretrieve(THUILLIER_URL, THUILLIER_FILE)
    
    # Parse f0.txt (Wavelength in nm, Irradiance in uW/cm^2/nm)
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

def calculate_toa_reflectance():
    # 1. Load Thuillier
    thuill_wvl, thuill_esun = download_thuillier()
    
    # 2. Parse metadata
    with open(JSON_PATH, 'r') as f:
        data = json.load(f)
    dt_str = data['properties']['datetime']
    sun_elev = data['properties']['view:sun_elevation']
    
    dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
    d = get_earth_sun_distance(dt)
    print(f"Earth-Sun Distance: {d:.4f} AU, Sun Elevation: {sun_elev} deg")
    
    # 3. Calculate TOA
    with h5py.File(H5_RAD, 'r') as h5_rad:
        rad_ds = h5_rad['HDFEOS/GRIDS/HYP/Data Fields/toa_radiance']
        # For Tanager, wavelengths are in nanometers (same as Thuillier in f0.txt)
        tanager_wvl = rad_ds.attrs['wavelengths']
        fill_value = rad_ds.attrs['_FillValue']
        
        # Interpolate Thuillier to Tanager wavelengths
        tanager_esun = np.interp(tanager_wvl, thuill_wvl, thuill_esun)
        
        print("Reading radiance data...")
        radiance = rad_ds[:]
        
        toa_reflectance = np.full_like(radiance, fill_value)
        valid_mask = (radiance != fill_value)
        
        print("Computing TOA Reflectance...")
        # (PI * L * d^2) / (ESUN * sin(sun_elev))
        denominator = tanager_esun[:, None, None] * np.sin(np.radians(sun_elev))
        denominator_full = np.broadcast_to(denominator, radiance.shape)
        toa_reflectance[valid_mask] = (np.pi * radiance[valid_mask] * (d**2)) / denominator_full[valid_mask]
        
        return toa_reflectance, tanager_wvl, fill_value

def extract_sr():
    with h5py.File(H5_SR, 'r') as h5_sr:
        sr_ds = h5_sr['HDFEOS/GRIDS/HYP/Data Fields/surface_reflectance']
        sr_data = sr_ds[:] # Shape is (Bands, Y, X)
        
        # Get valid spatial mask
        cloud = h5_sr['HDFEOS/GRIDS/HYP/Data Fields/beta_cloud_mask'][:]
        cirrus = h5_sr['HDFEOS/GRIDS/HYP/Data Fields/beta_cirrus_mask'][:]
        nodata = h5_sr['HDFEOS/GRIDS/HYP/Data Fields/nodata_pixels'][:]
        
        valid_spatial_mask = (cloud == 0) & (cirrus == 0) & (nodata == 0)
        
        # Get good wavelengths mask
        # all_good_wavelengths might be 2D (time, bands) or 1D
        gw = sr_ds.attrs.get('all_good_wavelengths')
        if gw is not None:
            if len(gw.shape) > 1:
                gw = gw[0] # first time pass
            good_bands = (gw == 1)
        else:
            good_bands = np.ones(sr_data.shape[0], dtype=bool)
            
        return sr_data, valid_spatial_mask, good_bands


def main():
    print("Extracting SR and Spatial Mask...")
    sr_data, valid_spatial_mask, good_bands = extract_sr()
    
    print("Calculating TOA Reflectance...")
    toa_data, tanager_wvl, fill_val = calculate_toa_reflectance()
    
    # Filter bands
    sr_data = sr_data[good_bands, :, :]
    toa_data = toa_data[good_bands, :, :]
    
    print(f"Using {np.sum(good_bands)} good bands out of {len(good_bands)}")
    
    sys.path.append(r'f:\Resilio\IMGS 890 Research\Spectral-Complexity-dev\HLSX30')
    import SpecComplexQR as scQR

    print("Processing SR Spectral Complexity...")
    sr_vol = scQR.process_volume_sliding_tile(sr_data, TILE_SIZE, SLIDING_STRIDE, num_endmembers=7, gram_type='minEndmember', norm_type='bandCount')
    
    print("Processing TOA Spectral Complexity...")
    toa_vol = scQR.process_volume_sliding_tile(toa_data, TILE_SIZE, SLIDING_STRIDE, num_endmembers=7, gram_type='minEndmember', norm_type='bandCount')

    # Apply spatial mask
    sr_vol[valid_spatial_mask == 0] = np.nan
    toa_vol[valid_spatial_mask == 0] = np.nan
    
    valid_sr_vol = sr_vol[~np.isnan(sr_vol)]
    valid_toa_vol = toa_vol[~np.isnan(toa_vol)]
    
    if len(valid_sr_vol) == 0 or len(valid_toa_vol) == 0:
        print("No valid pixels found!")
        return
        
    sr_z = np.full_like(sr_vol, np.nan)
    toa_z = np.full_like(toa_vol, np.nan)
    
    sr_mean = np.mean(valid_sr_vol)
    sr_std = np.std(valid_sr_vol)
    sr_z[~np.isnan(sr_vol)] = (valid_sr_vol - sr_mean) / sr_std
    
    toa_mean = np.mean(valid_toa_vol)
    toa_std = np.std(valid_toa_vol)
    toa_z[~np.isnan(toa_vol)] = (valid_toa_vol - toa_mean) / toa_std
    
    diff = sr_z - toa_z
    diff_valid = diff[~np.isnan(diff)]
    
    print("\n--- Z-Score Comparison ---")
    print(f"Mean Difference (SR - TOA): {np.mean(diff_valid):.4f}")
    print(f"Max Difference: {np.max(diff_valid):.4f}")
    print(f"Min Difference: {np.min(diff_valid):.4f}")
    print(f"Std of Difference: {np.std(diff_valid):.4f}")
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.title("SR Z-Score")
    plt.imshow(sr_z, cmap='viridis')
    plt.colorbar()
    
    plt.subplot(1, 3, 2)
    plt.title("TOA Z-Score")
    plt.imshow(toa_z, cmap='viridis')
    plt.colorbar()
    
    plt.subplot(1, 3, 3)
    plt.title("Difference (SR - TOA)")
    plt.imshow(diff, cmap='bwr', vmin=-2, vmax=2)
    plt.colorbar()
    
    plt.tight_layout()
    plot_path = os.path.join(os.path.dirname(__file__), "sr_vs_toa_zscore_diff.png")
    plt.savefig(plot_path)
    print(f"Saved plot to {plot_path}")

if __name__ == "__main__":
    main()
