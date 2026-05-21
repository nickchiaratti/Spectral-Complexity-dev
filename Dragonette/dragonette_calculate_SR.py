import numpy as np
import h5py
from datetime import datetime

# RGB Wavelength Targets for Ortho Visual generation
TARGET_RED_NM = 680.0
TARGET_GREEN_NM = 534.0
TARGET_BLUE_NM = 480.0

def get_earth_sun_distance(dt):
    """
    Calculates the Earth-Sun distance (d) in AU for a given datetime.
    Reference: Duffie & Beckman (2013), Solar Engineering of Thermal Processes.
    """
    day_of_year = dt.timetuple().tm_yday
    # Simplified calculation for astronomical distance
    d = 1 - 0.01672 * np.cos(np.radians(0.9856 * (day_of_year - 4)))
    return d

def apply_apparent_reflectance(h5_path):
    """
    Computes TOA-to-Apparent Reflectance conversion natively from the HDF5 tensor.
    Relies on explicit 2D arrays (e.g., solar_illumination) and 1D geometry attributes 
    established during the stacking phase, completely decoupling from external STAC JSONs.
    """
    with h5py.File(h5_path, 'r+') as h5:
        # 1. Dataset Resolution
        try:
            rad_ds = h5['HDFEOS/GRIDS/WYVERN/Data Fields/radiance']
            esun_ds = h5['HDFEOS/GRIDS/WYVERN/Data Fields/solar_illumination']
        except KeyError as e:
            raise KeyError(f"CRITICAL: Required dataset missing from HDF5 structure: {e}. Was the stacker run with the updated CF-compliant architecture?")

        num_times, num_bands, height, width = rad_ds.shape
        
        # 2. Metrology Extraction
        try:
            acq_times = rad_ds.attrs['acquisition_time']
            sun_elevations = rad_ds.attrs['sun_elevation']
        except KeyError as e:
            raise KeyError(f"CRITICAL: Missing essential metrology attribute {e} on the radiance dataset.")
        
        # 3. Output Dataset Allocation
        if 'surface_reflectance' in h5['HDFEOS/GRIDS/WYVERN/Data Fields']:
            del h5['HDFEOS/GRIDS/WYVERN/Data Fields/surface_reflectance']
        
        refl_ds = h5['HDFEOS/GRIDS/WYVERN/Data Fields'].create_dataset(
            "surface_reflectance", 
            shape=rad_ds.shape, 
            dtype='float32', 
            compression="gzip",
            fillvalue=rad_ds.fillvalue
        )

        for t in range(num_times):
            # Geometry & Astronomical Parameters
            sun_elev = sun_elevations[t]
            sun_zenith_rad = np.radians(90.0 - sun_elev)
            cos_zenith = np.cos(sun_zenith_rad)
            
            dt = datetime.fromtimestamp(acq_times[t])
            d = get_earth_sun_distance(dt)
            
            # Extract precise ESUN vector for this specific temporal pass
            esun = esun_ds[t, :]
            
            print(f"Calculating Pass {t}: Sun Zenith {90-sun_elev:.2f} deg, Distance {d:.4f} AU")
            
            # --- Fail-Fast Analytical Constraints ---
            if cos_zenith <= 0:
                raise ValueError(f"Geometric failure at Time {t}: Sun zenith is >= 90 degrees (Night/Terminator pass). Division by zero or negative reflectance imminent.")
            if np.any(esun <= 0):
                raise ValueError(f"Calibration failure at Time {t}: 'solar_illumination' contains zero or negative values. Cannot proceed with conversion.")
            
            # Industry Best Practice: Band-Sequential (BSQ) processing for out-of-core memory management.
            # Processing iteratively along the spectral axis aligns with HDF5 contiguous storage 
            # and prevents RAM exhaustion from massive 3D matrix allocations.
            for b in range(num_bands):
                radiance_band = rad_ds[t, b, :, :]
                mask = (radiance_band != rad_ds.fillvalue)
                
                reflectance_band = np.full(radiance_band.shape, rad_ds.fillvalue, dtype='float32')
                
                # --- Exploratory Path Radiance (Lp) Estimation via DOS1 ---
                # Find the dark object radiance to represent the atmospheric scattering baseline.
                # We use the 1st percentile of strictly positive valid data to reject 
                # anomalous sensor noise floors.
                valid_rad = radiance_band[mask]
                positive_rad = valid_rad[valid_rad > 0]
                
                if len(positive_rad) > 0:
                    L_p = np.percentile(positive_rad, 1)
                else:
                    # Fail-fast: If the entire band is 0 or negative, calibration is catastrophically compromised.
                    raise ValueError(f"Radiometric failure at Time {t}, Band {b}: No positive radiance values exist to calculate path radiance.")
                
                # Denominator collapses to a physical scalar for this specific band and time
                denominator = esun[b] * cos_zenith
                
                # Core Equation: rho = (PI * (L - Lp) * d^2) / (ESUN * cos(theta))
                # Executed exclusively on valid pixels. Subtracting Lp removes the severe blue/Rayleigh bias.
                reflectance_band[mask] = (np.pi * (radiance_band[mask] - L_p) * (d**2)) / denominator
                
                refl_ds[t, b, :, :] = reflectance_band

        # Sync attributes to inherit spatial/spectral metadata
        for k, v in rad_ds.attrs.items():
            if k not in ["scale", "offset", "unit", "description"]:
                refl_ds.attrs[k] = v
                
        refl_ds.attrs['unit'] = "unitless"
        refl_ds.attrs['description'] = "Apparent Surface Reflectance (First-order DOS1 atmospheric scattering compensation applied)"

        # 4. Ortho Visual Generation
        print("Extracting 'ortho_visual' specific bands from apparent surface reflectance...")
        
        # Explicitly fetch the 2D wavelength dataset
        try:
            wv_ds = h5['HDFEOS/GRIDS/WYVERN/Data Fields/wavelength']
        except KeyError:
            raise KeyError("CRITICAL: 'wavelength' dataset missing. Ensure the stacker utilized the 2D CF-compliant spectral architecture.")
        
        if 'ortho_visual' in h5['HDFEOS/GRIDS/WYVERN/Data Fields']:
            del h5['HDFEOS/GRIDS/WYVERN/Data Fields/ortho_visual']
            
        vis_ds = h5['HDFEOS/GRIDS/WYVERN/Data Fields'].create_dataset(
            "ortho_visual",
            shape=(num_times, 3, height, width),
            dtype='float32',
            compression="gzip",
            fillvalue=rad_ds.fillvalue
        )
        
        for t in range(num_times):
            # Dynamically calculate RGB indices per temporal pass to account for 
            # minor inter-platform calibration variations (e.g., Dragonette-2 vs Dragonette-3)
            pass_wv = wv_ds[t, :]
            r_idx = int(np.argmin(np.abs(pass_wv - TARGET_RED_NM)))
            g_idx = int(np.argmin(np.abs(pass_wv - TARGET_GREEN_NM)))
            b_idx = int(np.argmin(np.abs(pass_wv - TARGET_BLUE_NM)))
            
            vis_ds[t, 0, :, :] = refl_ds[t, r_idx, :, :]
            vis_ds[t, 1, :, :] = refl_ds[t, g_idx, :, :]
            vis_ds[t, 2, :, :] = refl_ds[t, b_idx, :, :]
            
        vis_ds.attrs['spatial_ref'] = rad_ds.attrs.get('spatial_ref', '')
        vis_ds.attrs['GeoTransform'] = rad_ds.attrs.get('GeoTransform', np.zeros(6))
        vis_ds.attrs['unit'] = "unitless"
        vis_ds.attrs['description'] = "True Color RGB composite derived from Apparent Surface Reflectance"

    print(f"Reflectance estimation and visual derivation complete.")

if __name__ == "__main__":
    # Ensure this points to the newly processed stack
    apply_apparent_reflectance("C:/satelliteImagery/dragonette/ROCX_SourceData/Wyvern_Native_Stack_ROCX.h5")