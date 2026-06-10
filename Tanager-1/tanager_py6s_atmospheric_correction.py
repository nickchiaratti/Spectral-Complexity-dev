import os
import h5py
import numpy as np
from datetime import datetime
from Py6S import SixS, AtmosProfile, AeroProfile, Geometry, Wavelength
from pyproj import Transformer
from tqdm import tqdm
from joblib import Parallel, delayed

# --- Configuration ---
SOURCE_FILE = "C:/satelliteImagery/Tanager/Rochesterv2_SourceData/Tanager_Native_Stack_Rochesterv2.h5"

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

    # Use climatology for Ozone and Water Vapor based on latitude and date, as requested
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

def process_py6s():
    with h5py.File(SOURCE_FILE, 'r+') as f:
        grp = f['HDFEOS/GRIDS/TANAGER/Data Fields']
        toa_ds = grp['toa_radiance']
        fill_val = toa_ds.attrs.get('_FillValue', -9999.0)
        if isinstance(fill_val, (np.ndarray, list, tuple)):
            fill_val = fill_val[0]

        n_times, n_bands, height, width = toa_ds.shape
        
        # Read wavelengths and fwhm
        # The units are already W/(m^2 sr um), but what are the wavelength units? Usually nm.
        wavelengths = toa_ds.attrs['wavelengths']
        wl_units_raw = toa_ds.attrs.get('wavelengths_units', 'nm')
        wl_units = wl_units_raw.decode('utf-8') if isinstance(wl_units_raw, bytes) else str(wl_units_raw)
        fwhm = toa_ds.attrs['fwhm']
        
        # Convert wavelengths to micrometers for Py6S if they are in nm
        if 'nm' in wl_units.lower():
            wavelengths_um = wavelengths / 1000.0
            fwhm_um = fwhm / 1000.0
        else:
            wavelengths_um = wavelengths
            fwhm_um = fwhm

        # Extract geometry transformation for latitude calculation
        # GeoTransform: [c, a, b, f, d, e] which maps to [ul_x, x_res, 0, ul_y, 0, y_res]
        geot = toa_ds.attrs['GeoTransform']
        crs_str = toa_ds.attrs['spatial_ref']
        if isinstance(crs_str, bytes):
            crs_str = crs_str.decode('utf-8')
            
        transformer = Transformer.from_crs(crs_str, "EPSG:4326", always_xy=True)
        
        # Calculate centroid in UTM
        center_x_utm = geot[0] + (width / 2.0) * geot[1]
        center_y_utm = geot[3] + (height / 2.0) * geot[5]
        # Transform to lon/lat
        center_lon, center_lat = transformer.transform(center_x_utm, center_y_utm)
        
        # Get acquisition times
        if "surface_reflectance" in grp:
            acq_times = grp['surface_reflectance'].attrs['acquisition_time']
        else:
            acq_times = np.zeros(n_times) # fallback if missing
            
        # Create output dataset
        if "surface_reflectance_py6s" in grp:
            del grp["surface_reflectance_py6s"]
            
        sr_py6s_ds = grp.create_dataset(
            "surface_reflectance_py6s", 
            shape=toa_ds.shape, 
            dtype=np.float32, 
            chunks=toa_ds.chunks,
            compression="gzip", 
            fillvalue=fill_val
        )
        
        # Mirror attributes from surface_reflectance if available
        if "surface_reflectance" in grp:
            sr_mfg = grp["surface_reflectance"]
            for k, v in sr_mfg.attrs.items():
                if k not in ["DIMENSION_LIST", "CLASS", "REFERENCE_LIST", "good_wavelengths", "all_good_wavelengths"]:
                    sr_py6s_ds.attrs[k] = v
        
        # Ensure basic attributes are always present
        sr_py6s_ds.attrs['wavelengths'] = wavelengths
        sr_py6s_ds.attrs['wavelengths_units'] = wl_units_raw
        sr_py6s_ds.attrs['fwhm'] = fwhm
        sr_py6s_ds.attrs['_FillValue'] = fill_val
        sr_py6s_ds.attrs['spatial_ref'] = crs_str
        sr_py6s_ds.attrs['GeoTransform'] = geot
        
        # Generate good_wavelengths list based on water absorption ranges (1340-1440nm and 1780-1970nm)
        good_wavelengths = np.ones(n_bands, dtype=np.int32)
        for i, wl in enumerate(wavelengths):
            if (1340 <= wl <= 1440) or (1780 <= wl <= 1970):
                good_wavelengths[i] = 0
        sr_py6s_ds.attrs['good_wavelengths'] = good_wavelengths
        
        # We need to process time steps independently to avoid massive memory usage
        # Shape: (Time, Bands, Y, X)
        for t_idx in range(n_times):
            print(f"\nProcessing Time Step {t_idx+1}/{n_times}")
            
            # Read atmospheric and geometric layers
            aod_layer = grp['aerosol_optical_depth'][t_idx]
            sza_layer = grp['sun_zenith'][t_idx]
            saa_layer = grp['sun_azimuth'][t_idx]
            vza_layer = grp['sensor_zenith'][t_idx]
            vaa_layer = grp['sensor_azimuth'][t_idx]
            
            # Mask out fill values for median calculation
            valid_aod = aod_layer[aod_layer != fill_val]
            valid_sza = sza_layer[sza_layer != fill_val]
            valid_saa = saa_layer[saa_layer != fill_val]
            valid_vza = vza_layer[vza_layer != fill_val]
            valid_vaa = vaa_layer[vaa_layer != fill_val]
            
            # Calculate medians
            median_aod = np.median(valid_aod) if valid_aod.size > 0 else 0.1
            median_sza = np.median(valid_sza) if valid_sza.size > 0 else 30.0
            median_saa = np.median(valid_saa) if valid_saa.size > 0 else 180.0
            median_vza = np.median(valid_vza) if valid_vza.size > 0 else 0.0
            median_vaa = np.median(valid_vaa) if valid_vaa.size > 0 else 0.0
            
            # Date extraction
            acq_dt = datetime.utcfromtimestamp(acq_times[t_idx])
            
            print(f"  Geographic Centroid: Lat={center_lat:.4f}, Lon={center_lon:.4f}")
            print(f"  Date: {acq_dt.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"  Median Parameters: AOD={median_aod:.3f}, SZA={median_sza:.2f}, SAA={median_saa:.2f}, VZA={median_vza:.2f}, VAA={median_vaa:.2f}")

            # Run Py6S for all bands in parallel
            print("  Running Py6S over all bands...")
            # Using joblib for safe multi-processing since SixS is a separate executable wrapper
            results = Parallel(n_jobs=-1)(
                delayed(run_6s_band)(
                    wavelengths_um[b], fwhm_um[b], 
                    median_sza, median_saa, median_vza, median_vaa, 
                    median_aod, center_lat, acq_dt
                ) for b in tqdm(range(n_bands), desc="  6S Band Evaluation")
            )
            
            xa_list, xb_list, xc_list = zip(*results)
            
            xa_arr = np.array(xa_list, dtype=np.float32).reshape(-1, 1, 1)
            xb_arr = np.array(xb_list, dtype=np.float32).reshape(-1, 1, 1)
            xc_arr = np.array(xc_list, dtype=np.float32).reshape(-1, 1, 1)
            
            # Load TOA Radiance for this time step
            # Shape: (Bands, Y, X)
            print("  Loading TOA Radiance...")
            toa_frame = toa_ds[t_idx, ...]
            print("  Applying Atmospheric Correction (Band-by-Band)...")
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
                
                # Two-step assignment to handle both valid TOA and valid denominator masks
                valid_results = np.full(np.sum(valid), fill_val, dtype=np.float32)
                valid_results[valid_denom_mask] = y[valid_denom_mask] / denom[valid_denom_mask]
                
                band_sr[valid] = valid_results
                sr_frame[b, :, :] = band_sr
                
            print("  Saving Surface Reflectance...")
            sr_py6s_ds[t_idx, ...] = sr_frame

    print("Py6S processing complete.")

if __name__ == "__main__":
    process_py6s()
