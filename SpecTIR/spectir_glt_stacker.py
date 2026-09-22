import os
import sys
import numpy as np
import h5py
from pathlib import Path
from PIL import Image

# Add parent folder to sys.path to find SpecComplex
script_dir = Path(__file__).resolve().parent
if str(script_dir.parent) not in sys.path:
    sys.path.insert(0, str(script_dir.parent))
import SpecComplex as sc
import spectral
import matplotlib.pyplot as plt

import estimate_noise

OUTPUT_DIR = r"C:\satelliteImagery\share2012\SpecTIR"

def plot_snr_spectrum(X, noise_covariances, wavelengths, run_id=None):
    """
    Produces a Band-by-Band SNR Spectrum plot to objectively compare 
    different noise covariance estimation methods.
    
    Parameters:
        X (ndarray): Hyperspectral image data, shape (N pixels, p bands)
        noise_covariances (dict): Dictionary mapping method names to their 
                                  estimated noise covariance matrices.
                                  e.g., {'CR Method': K_n_CR, 'SSMR Method': K_n_SSMR}
        wavelengths (ndarray): Array of wavelengths for the x-axis, shape (p,).
    """
    N, p = X.shape
    
    if wavelengths is None:
        wavelengths = np.arange(1, p + 1)
        x_label = "Spectral Band Number"
    else:
        x_label = "Wavelength (nm)"
        
    # 1. Calculate the actual data variance for each spectral band
    # ddof=1 is used for an unbiased sample variance estimate
    data_variances = np.var(X, axis=0, ddof=1)
    
    plt.figure(figsize=(10, 6))
    
    # 2. Calculate and plot the SNR spectrum for each estimation method
    for method_name, cov_matrix in noise_covariances.items():
        
        # Extract the noise variances (the main diagonal of the covariance matrix)
        noise_variances = np.diag(cov_matrix)
        
        # Estimate signal variance: Total Data Variance - Noise Variance
        signal_variances = data_variances - noise_variances
        
        # Robustness safeguard: Prevent negative signal variances from causing 
        # math domain errors in the log10 function.
        epsilon = 1e-10
        signal_variances = np.clip(signal_variances, a_min=epsilon, a_max=None)
        noise_variances = np.clip(noise_variances, a_min=epsilon, a_max=None)
        
        # Calculate SNR (Linear scale)
        snr_linear = signal_variances / noise_variances
        
        # Plot the SNR spectrum
        plt.plot(wavelengths, snr_linear, label=f"{method_name}", linewidth=1.5)
        
    # 3. Format the plot for scientific publication
    plt.title("Band-by-Band Signal-to-Noise Ratio (SNR) Comparison")
    plt.xlabel(x_label)
    plt.ylabel("SNR (Linear)")
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add a horizontal line at 1 (where Noise Power = Signal Power)
    plt.axhline(1, color='black', linewidth=1, linestyle='-')
    
    plt.tight_layout(pad=0.1)
    plt.savefig(os.path.join(OUTPUT_DIR, f"snr_spectrum_{run_id}.png"), dpi=400)


def process_spectir_glt(run_id):
    REF_DAT = rf"C:\satelliteImagery\share2012\SpecTIR\{run_id}\{run_id}_pol_ref.dat"
    REF_HDR = rf"C:\satelliteImagery\share2012\SpecTIR\{run_id}\{run_id}_pol_ref.hdr"
    LOG_FILE = rf"C:\satelliteImagery\share2012\SpecTIR\{run_id}\{run_id}_pol_ref.log"

    GLT_DAT = rf"C:\satelliteImagery\share2012\SpecTIR\IGM_GLTS_AVON_AM\IGM_GLTS\{run_id}_rad_glt.dat"
    GLT_HDR = rf"C:\satelliteImagery\share2012\SpecTIR\IGM_GLTS_AVON_AM\IGM_GLTS\{run_id}_rad_glt.hdr"

    OUTPUT_H5 = os.path.join(OUTPUT_DIR, f"SpecTIR_Ortho_Stack_{run_id}.h5")

    if not os.path.exists(REF_DAT) or not os.path.exists(GLT_DAT):
        print(f"Skipping {run_id}: Missing REF or GLT data files.")
        return

    print(f"\n--- Processing run: {run_id} (SENSOR SPACE) ---")
    print(f"Loading metadata from ENVI headers...")
    ref_img = spectral.envi.open(REF_HDR, REF_DAT)
    glt_img = spectral.envi.open(GLT_HDR, GLT_DAT)
    
    lines, samples, bands = ref_img.shape
    map_lines, map_samples, glt_bands = glt_img.shape
    
    print(f"Ref Image: {lines}L x {samples}S x {bands}B (dtype: {ref_img.dtype})")
    print(f"GLT Image: {map_lines}L x {map_samples}S x {glt_bands}B")
    
    # Parse GeoTransform and Spatial Ref from GLT Map Info
    map_info = glt_img.metadata.get('map info', [])
    gdal_transform = [0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    crs_wkt = ""
    if len(map_info) >= 11:
        x_tie = float(map_info[3])
        y_tie = float(map_info[4])
        pixel_x = float(map_info[5])
        pixel_y = float(map_info[6])
        gdal_transform = [x_tie - pixel_x/2.0, pixel_x, 0.0, y_tie + pixel_y/2.0, 0.0, -pixel_y]
        zone = map_info[7]
        hemisphere = map_info[8]
        crs_wkt = f"PROJCS[\"WGS 84 / UTM zone {zone}{'N' if hemisphere.lower() == 'north' else 'S'}\"]"
    
    wavelengths = np.zeros(bands, dtype='float32')
    if 'wavelength' in ref_img.metadata:
        # ENVI metadata provides SpecTIR wavelengths in micrometers (e.g. 0.4 to 2.5).
        # We multiply by 1000.0 to convert to nanometers for the CIE calculations.
        wavelengths = np.array([float(w) for w in ref_img.metadata['wavelength']], dtype='float32') * 1000.0
        
    fill_value = 0 # Default fill value if not found
    if 'data ignore value' in ref_img.metadata:
        fill_value = float(ref_img.metadata['data ignore value'])
        if ref_img.dtype.char in ['h', 'i', 'b', 'H', 'I', 'B']:
            fill_value = int(fill_value)
            
    print(f"Using Fill Value: {fill_value}")
    
    scale_val = 10000.0
    if 'reflectance scale factor' in ref_img.metadata:
        try:
            val = ref_img.metadata['reflectance scale factor']
            scale_val = float(val[0]) if isinstance(val, list) else float(val)
        except Exception:
            pass
            
    # 1. Load RAW Sensor Data
    print(f"Loading unwarped flightline into memory...")
    raw_data = np.asarray(ref_img.load())
    
    # 2. Extract Noise Matrices in true sensor geometry
    print("Estimating Noise Matrices on unwarped data...")
    raw_float = raw_data.astype(np.float32) / scale_val
    image_cube_float = np.transpose(raw_float, (2, 0, 1))
    
    # Find valid mask (ignoring fill values for true physical noise)
    scaled_fill_val = fill_value / scale_val
    valid_mask = np.all(image_cube_float != scaled_fill_val, axis=0)
    invalid_values = (image_cube_float < -1.5) | (image_cube_float > 1.5)
    valid_mask = valid_mask & ~np.any(invalid_values, axis=0)
    
    print("  -> SSD Noise Covariance...")
    ssd_cov = estimate_noise.estimate_noise_ssd(image_cube_float)
    print("  -> MLR Noise Covariance...")
    mlr_cov = estimate_noise.estimate_noise_mlr(image_cube_float)
    print("  -> DCT Noise Covariance...")
    dct_cov = estimate_noise.estimate_noise_dct(image_cube_float, valid_mask)
    print("  -> MNF Noise Covariance...")
    mnf_cov = estimate_noise.estimate_noise_mnf(image_cube_float)
    
    print("  -> Generating SNR Spectrum Plot...")
    X_valid = np.transpose(image_cube_float, (1, 2, 0))[valid_mask]
    noise_covs = {
        'SSD': ssd_cov,
        'MLR': mlr_cov,
        'DCT': dct_cov,
        'MNF': mnf_cov
    }
    plot_snr_spectrum(X_valid, noise_covs, wavelengths, run_id=run_id)
    
    del raw_float
    del image_cube_float
    del X_valid
    
    # 3. Save to HDF5
    print(f"Creating HDF5 Output at {OUTPUT_H5}...")
    with h5py.File(OUTPUT_H5, 'w') as out_h5:
        grp = out_h5.create_group("HDFEOS/GRIDS/SPECTIR/Data Fields")
        
        # Save Unwarped Surface Reflectance
        print("Saving unwarped surface_reflectance...")
        out_shape = (1, bands, lines, samples)
        chunks_dim = (1, bands, min(lines, 256), min(samples, 256))
        
        ds_ref = grp.create_dataset(
            "surface_reflectance", 
            data=np.expand_dims(np.transpose(raw_data, (2, 0, 1)), axis=0),
            dtype=ref_img.dtype, 
            compression="gzip", 
            compression_opts=6, 
            shuffle=True, 
            fillvalue=fill_value, 
            chunks=chunks_dim
        )
        
        ds_ref.attrs['Scale_Factor'] = np.array([scale_val], dtype='float32')
        ds_ref.attrs['acquisition_time'] = np.array([0.0], dtype='float64')
        dt_str = h5py.string_dtype(encoding='ascii')
        ds_ref.attrs.create('spacecraft_id', data=np.array(['SpecTIR'], dtype=dt_str))
        ds_ref.attrs['wavelengths'] = wavelengths
        ds_ref.attrs['spatial_ref'] = crs_wkt
        ds_ref.attrs['GeoTransform'] = np.array(gdal_transform, dtype='float64')
        
        # Save Noise Covariances
        print("Saving noise covariance matrices...")
        def save_cov(name, data, desc):
            ds = grp.create_dataset(name, data=data, dtype=np.float32, compression="gzip", compression_opts=5)
            ds.attrs['description'] = desc
            ds.attrs['spatial_ref'] = crs_wkt
            ds.attrs['GeoTransform'] = np.array(gdal_transform, dtype='float64')
            ds.attrs['glt_mapping_dataset'] = "GLT"
            
        ds_ref.attrs['glt_mapping_dataset'] = "GLT"
            
        save_cov("ssd_noise_cov", ssd_cov, "Estimated full noise covariance matrix (Bands x Bands) using SSD method")
        save_cov("mnf_noise_cov", mnf_cov, "Estimated full noise covariance matrix (Bands x Bands) using MNF method")
        save_cov("mlr_noise_cov", mlr_cov, "Estimated full noise covariance matrix (Bands x Bands) using MLR method")
        save_cov("dct_noise_cov", dct_cov, "Estimated full noise covariance matrix (Bands x Bands) using DCT method")
        
        # 4. Save GLT
        print("Saving GLT lookup array for downstream mapping...")
        glt_data = np.asarray(glt_img.load())
        ds_glt = grp.create_dataset(
            "GLT",
            data=np.expand_dims(np.transpose(glt_data, (2, 0, 1)), axis=0),
            dtype=glt_img.dtype,
            compression="gzip",
            shuffle=True, 
            fillvalue=0,
            compression_opts=6
        )
        ds_glt.attrs['description'] = "Geographic Lookup Table. GLT[map_y, map_x] = [sensor_x, sensor_y]"
        ds_glt.attrs['spatial_ref'] = crs_wkt
        ds_glt.attrs['GeoTransform'] = np.array(gdal_transform, dtype='float64')
        

        # 5. Generate and Save Flightpath Visual preview
        print("Generating flightpath_vis RGB composite...")
        
        # Generate RGBA in sensor space using the full CIE transform
        raw_data_t = np.transpose(raw_data, (2, 0, 1)) # (bands, lines, samples)
        rgba_img_raw = sc.generate_rgba_from_hsi(frame_data=raw_data_t, wavelengths=wavelengths, nodata=fill_value, scale=scale_val)
        rgba_img_raw_t = np.transpose(rgba_img_raw, (2, 0, 1)) # (4, lines, samples)
        
        ds_vis = grp.create_dataset(
            "flightpath_vis", 
            data=np.expand_dims(rgba_img_raw_t, axis=0),
            dtype='uint8', 
            compression="gzip", 
            shuffle=True, 
            fillvalue=0,
            compression_opts=6
        )
        ds_vis.attrs['spatial_ref'] = crs_wkt
        ds_vis.attrs['GeoTransform'] = np.array(gdal_transform, dtype='float64')
        ds_vis.attrs['glt_mapping_dataset'] = "GLT"
        
        # Save PNG preview for Flightpath (unwarped)
        print("Exporting high-resolution flightpath_vis preview (600 DPI)...")
        png_path_flightpath = os.path.join(OUTPUT_DIR, f"SpecTIR_{run_id}_flightpath_vis_preview.png")
        img_flightpath = Image.fromarray(rgba_img_raw, 'RGBA')
        img_flightpath.save(png_path_flightpath, dpi=(600, 600))
        print(f"Saved: {png_path_flightpath}")
        
        # Warp to map space for the Orthorectified PNG preview
        print("Generating orthorectified RGB composite preview...")
        b1 = glt_data[:, :, 0]
        b2 = glt_data[:, :, 1]
        valid_glt = (b1 != 0) & (b2 != 0)
        
        sample_idx = (np.abs(b1[valid_glt]) - 1).astype(int)
        line_idx = (np.abs(b2[valid_glt]) - 1).astype(int)
        sample_idx = np.clip(sample_idx, 0, samples - 1)
        line_idx = np.clip(line_idx, 0, lines - 1)
        
        vis_buffer_rgba = np.zeros((4, map_lines, map_samples), dtype=np.uint8)
        for c in range(4):
            vis_buffer_rgba[c, valid_glt] = rgba_img_raw_t[c, line_idx, sample_idx]
            
        print("Exporting high-resolution orthorectified_vis preview (600 DPI)...")
        png_path_ortho = os.path.join(OUTPUT_DIR, f"SpecTIR_{run_id}_orthorectified_vis_preview.png")
        ortho_rgba_for_pil = np.transpose(vis_buffer_rgba, (1, 2, 0))
        img_ortho = Image.fromarray(ortho_rgba_for_pil, 'RGBA')
        img_ortho.save(png_path_ortho, dpi=(600, 600))
        print(f"Saved: {png_path_ortho}")

    print(f"Synthesis Complete. Stack Stored at: {OUTPUT_H5}")

if __name__ == "__main__":
    runs_to_process = [
        "0920-1631",
        "0920-1638",
        "0920-1646",
        "0920-1654",
        "0920-1706"
    ]
    for run in runs_to_process:
        process_spectir_glt(run)
