import os
import sys
import json
import h5py
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from Py6S import SixS, AtmosProfile, AeroProfile, Geometry, Wavelength
from pyproj import Transformer
from tqdm import tqdm
from joblib import Parallel, delayed
from Py6S import AtmosCorr

# --- Configuration ---
SOURCE_DIR = "C:/satelliteImagery/Tanager/Rochesterv2_SourceData"
AERONET_FILE = "C:/satelliteImagery/ground_truth/AERONET_20250101_20251231_ROCX2025/20250101_20251231_ROCX2025.tot_lev20"

def get_aeronet_aod(aeronet_path):
    """Parses AERONET Level 2.0, calculates daily means, interpolates gaps, and derives AOD at 550nm."""
    print("Parsing AERONET data...")
    df = pd.read_csv(aeronet_path, skiprows=6)
    
    # Filter missing values (-999.0)
    df = df[(df['AOD_500nm-AOD'] != -999.0) & (df['AOD_675nm-AOD'] != -999.0)].copy()
    
    # Parse dates
    df['Date'] = pd.to_datetime(df['Date(dd:mm:yyyy)'], format='%d:%m:%Y')
    
    # Calculate Daily Means
    daily = df.groupby('Date')[['AOD_500nm-AOD', 'AOD_675nm-AOD']].mean()
    
    # Reindex to fill missing days and interpolate gaps
    full_range = pd.date_range(start=daily.index.min(), end=daily.index.max(), freq='D')
    daily = daily.reindex(full_range)
    daily = daily.interpolate(method='linear')
    daily = daily.ffill().bfill() # Handle edges if completely missing
    
    # Calculate Angstrom exponent and AOD at 550nm
    daily['Angstrom'] = -np.log(daily['AOD_500nm-AOD'] / daily['AOD_675nm-AOD']) / np.log(500.0 / 675.0)
    daily['AOD_550'] = daily['AOD_500nm-AOD'] * (550.0 / 500.0) ** (-daily['Angstrom'])
    
    return daily['AOD_550']

def copy_h5_structure(grp_in, grp_out):
    """Recursively copies HDF5 structure while omitting 'toa_radiance'."""
    for k, v in grp_in.attrs.items():
        grp_out.attrs[k] = v
        
    for name, item in grp_in.items():
        if isinstance(item, h5py.Group):
            subgrp = grp_out.create_group(name)
            copy_h5_structure(item, subgrp)
        elif isinstance(item, h5py.Dataset):
            if name == 'toa_radiance':
                continue
            grp_in.copy(name, grp_out, name=name)

def run_6s_band(wavelength_um, fwhm_um, sza, saa, vza, vaa, aod, lat, acq_date):
    """
    Runs 6S for a single band to calculate the atmospheric correction coefficients.
    """
    s = SixS()
    s.geometry = Geometry.User()
    s.geometry.solar_z = sza
    s.geometry.solar_a = saa
    s.geometry.view_z = vza
    s.geometry.view_a = vaa
    s.geometry.month = acq_date.month
    s.geometry.day = acq_date.day

    # Set Aerosol model to Continental and AOD at 550nm to the scene median AOD
    s.aero_profile = AeroProfile.PredefinedType(AeroProfile.Continental)
    s.aot550 = float(aod)

    # Use climatology for Ozone and Water Vapor based on latitude and date
    s.atmos_profile = AtmosProfile.FromLatitudeAndDate(lat, acq_date.strftime('%Y-%m-%d'))
    
    # Set wavelength (Py6S expects micrometers)
    min_wl = wavelength_um - (fwhm_um / 2)
    max_wl = wavelength_um + (fwhm_um / 2)
    s.wavelength = Wavelength(float(min_wl), float(max_wl))
    
    s.altitudes.set_sensor_satellite_level()
    s.altitudes.set_target_sea_level()
    
    # Enable atmospheric correction output
    from Py6S import AtmosCorr
    s.atmos_corr = AtmosCorr.AtmosCorrLambertianFromRadiance(100)
    
    s.run()
    
    return s.outputs.coef_xa, s.outputs.coef_xb, s.outputs.coef_xc

def process_py6s(target_file=None):
    aod_series = get_aeronet_aod(AERONET_FILE)
    
    source_dir = Path(SOURCE_DIR)
    # Search for un-stacked basic_radiance scenes
    if target_file:
        h5_files = list(source_dir.rglob(f"*{target_file}*basic_radiance*.h5"))
    else:
        h5_files = list(source_dir.rglob("*basic_radiance*.h5"))
    
    if not h5_files:
        print(f"No basic_radiance HDF5 files found in {SOURCE_DIR}")
        return
        
    for h5_path in h5_files:
        print(f"\nProcessing {h5_path.name}...")
        
        # Locate STAC metadata JSON
        item_id = h5_path.parent.name
        json_path = h5_path.parent / f"{item_id}.json"
        
        acq_dt = None
        if json_path.exists():
            with open(json_path, 'r') as jf:
                stac_item = json.load(jf)
                dt_str = stac_item.get('properties', {}).get('datetime')
                if dt_str:
                    acq_dt = pd.to_datetime(dt_str).to_pydatetime().replace(tzinfo=None)
        
        if not acq_dt:
            raise ValueError(f"Could not determine acquisition date from STAC metadata: {json_path}")
            
        print(f"  Acquisition Date: {acq_dt.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Retrieve AOD via daily lookup
        acq_date_only = pd.Timestamp(acq_dt.date())
        if acq_date_only in aod_series.index:
            aod_val = aod_series[acq_date_only]
        else:
            if len(aod_series) > 0:
                print(f"  WARNING: Date {acq_date_only.strftime('%Y-%m-%d')} not in AERONET data range. Fallback to nearest neighbor.")
                nearest_idx = aod_series.index.get_indexer([acq_date_only], method='nearest')[0]
                aod_val = aod_series.iloc[nearest_idx]
            else:
                raise ValueError("AERONET AOD series is entirely empty.")
                
        print(f"  AERONET Extrapolated AOD (550nm): {aod_val:.4f}")
        
        out_name = h5_path.name.replace("basic_radiance", "basic_6Ssr")
        out_path = h5_path.parent / out_name
        
        with h5py.File(h5_path, 'r') as f_in:
            with h5py.File(out_path, 'w') as f_out:
                print("  Mirroring HDF5 structure...")
                copy_h5_structure(f_in, f_out)
                
                grp_path = 'HDFEOS/SWATHS/HYP/Data Fields'
                if grp_path not in f_in:
                    raise KeyError(f"Required group {grp_path} not found in {h5_path.name}")
                    
                grp_in = f_in[grp_path]
                grp_out = f_out[grp_path]
                
                toa_ds = grp_in['toa_radiance']
                
                # Strict Data Integrity Check
                if '_FillValue' not in toa_ds.attrs:
                    raise KeyError(f"Strict enforcement failed: Missing '_FillValue' attribute in toa_radiance for {h5_path.name}")
                
                fill_val = toa_ds.attrs['_FillValue']
                if isinstance(fill_val, (np.ndarray, list, tuple)):
                    fill_val = fill_val[0]

                if len(toa_ds.shape) == 3:
                    n_bands, height, width = toa_ds.shape
                else:
                    raise ValueError(f"Unexpected shape for basic_radiance swath toa_radiance: {toa_ds.shape}. Expected 3D.")
                
                # Extract band parameters
                wavelengths = toa_ds.attrs['wavelengths']
                wl_units_raw = toa_ds.attrs.get('wavelengths_units', 'nm')
                wl_units = wl_units_raw.decode('utf-8') if isinstance(wl_units_raw, bytes) else str(wl_units_raw)
                fwhm = toa_ds.attrs['fwhm']
                
                if 'nm' in wl_units.lower():
                    wavelengths_um = wavelengths / 1000.0
                    fwhm_um = fwhm / 1000.0
                else:
                    wavelengths_um = wavelengths
                    fwhm_um = fwhm

                # Calculate centroid using Geolocation Fields
                lat_array = f_in['HDFEOS/SWATHS/HYP/Geolocation Fields/Latitude'][...]
                lon_array = f_in['HDFEOS/SWATHS/HYP/Geolocation Fields/Longitude'][...]
                
                valid_lat = lat_array[lat_array != fill_val]
                valid_lon = lon_array[lon_array != fill_val]
                
                center_lat = np.median(valid_lat) if valid_lat.size > 0 else 0.0
                center_lon = np.median(valid_lon) if valid_lon.size > 0 else 0.0
                
                # Create surface reflectance dataset mimicking the source properties
                sr_py6s_ds = grp_out.create_dataset(
                    "surface_reflectance", 
                    shape=toa_ds.shape, 
                    dtype=np.float32, 
                    chunks=toa_ds.chunks,
                    compression="gzip", 
                    fillvalue=fill_val
                )
                
                # Mirror original attributes
                for k, v in toa_ds.attrs.items():
                    if k not in ["DIMENSION_LIST", "CLASS", "REFERENCE_LIST", "good_wavelengths", "all_good_wavelengths"]:
                        sr_py6s_ds.attrs[k] = v
                
                # Assign AOD as requested
                sr_py6s_ds.attrs['aerosol_optical_depth_550nm'] = aod_val
                
                # Determine water absorption exclusions
                good_wavelengths = np.ones(n_bands, dtype=np.int32)
                for i, wl in enumerate(wavelengths):
                    if (1340 <= wl <= 1440) or (1780 <= wl <= 1970):
                        good_wavelengths[i] = 0
                sr_py6s_ds.attrs['good_wavelengths'] = good_wavelengths
                
                # Load auxiliary geometries
                sza_layer = grp_in['sun_zenith'][...]
                saa_layer = grp_in['sun_azimuth'][...]
                vza_layer = grp_in['sensor_zenith'][...]
                vaa_layer = grp_in['sensor_azimuth'][...]
                
                valid_sza = sza_layer[sza_layer != fill_val]
                valid_saa = saa_layer[saa_layer != fill_val]
                valid_vza = vza_layer[vza_layer != fill_val]
                valid_vaa = vaa_layer[vaa_layer != fill_val]
                
                median_sza = np.median(valid_sza) if valid_sza.size > 0 else 30.0
                median_saa = np.median(valid_saa) if valid_saa.size > 0 else 180.0
                median_vza = np.median(valid_vza) if valid_vza.size > 0 else 0.0
                median_vaa = np.median(valid_vaa) if valid_vaa.size > 0 else 0.0
                
                print(f"  Geographic Centroid: Lat={center_lat:.4f}, Lon={center_lon:.4f}")
                print(f"  Median Parameters: SZA={median_sza:.2f}, SAA={median_saa:.2f}, VZA={median_vza:.2f}, VAA={median_vaa:.2f}")

                print("  Running Py6S over all bands...")
                results = Parallel(n_jobs=-1)(
                    delayed(run_6s_band)(
                        wavelengths_um[b], fwhm_um[b], 
                        median_sza, median_saa, median_vza, median_vaa, 
                        aod_val, center_lat, acq_dt
                    ) for b in tqdm(range(n_bands), desc="  6S Band Evaluation")
                )
                
                xa_list, xb_list, xc_list = zip(*results)
                
                xa_arr = np.array(xa_list, dtype=np.float32).reshape(-1, 1, 1)
                xb_arr = np.array(xb_list, dtype=np.float32).reshape(-1, 1, 1)
                xc_arr = np.array(xc_list, dtype=np.float32).reshape(-1, 1, 1)
                
                print("  Applying Atmospheric Correction (Band-by-Band)...")
                toa_frame = toa_ds[...]
                sr_frame = np.full_like(toa_frame, fill_val, dtype=np.float32)
                
                for b in tqdm(range(n_bands), desc="  Applying Coefs"):
                    if good_wavelengths[b] == 0:
                        sr_frame[b, :, :] = 0.0
                        continue
                        
                    band_toa = toa_frame[b, :, :]
                    valid = (band_toa != fill_val)
                    if not np.any(valid):
                        continue
                        
                    xa = xa_arr[b, 0, 0]
                    xb = xb_arr[b, 0, 0]
                    xc = xc_arr[b, 0, 0]
                    
                    y = xa * band_toa[valid] - xb
                    denom = 1.0 + xc * y
                    
                    valid_denom_mask = np.abs(denom) > 1e-6
                    
                    band_sr = np.full_like(band_toa, fill_val, dtype=np.float32)
                    
                    valid_results = np.full(np.sum(valid), fill_val, dtype=np.float32)
                    valid_results[valid_denom_mask] = y[valid_denom_mask] / denom[valid_denom_mask]
                    
                    band_sr[valid] = valid_results
                    sr_frame[b, :, :] = band_sr
                    
                print("  Saving Surface Reflectance...")
                sr_py6s_ds[...] = sr_frame

    print("Py6S processing complete.")

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else None
    process_py6s(target)
