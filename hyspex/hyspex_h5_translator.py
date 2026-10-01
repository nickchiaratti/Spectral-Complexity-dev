import os
import sys
import numpy as np
import h5py
from pathlib import Path
from PIL import Image

# Add parent folder to sys.path to find SpecComplex and noise estimators
script_dir = Path(__file__).resolve().parent
if str(script_dir.parent) not in sys.path:
    sys.path.insert(0, str(script_dir.parent))
import SpecComplex as sc
import spectral
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

HDR_PATH = r"F:\GlobusData\hyspex\20250918_GE18\dual\GE18_55m_02_Mjolnir_float32_geo_VS620.hdr"
OUTPUT_DIR = r"C:\satelliteImagery\hyspex\20250918_GE18\dual"
RUN_ID = "20250918_ge18_55m_02"

WAVELENGTH_CUTOFF= 2440
NOISE_ESTIMATIONS = ['ssd', 'nfd']
    

# Assuming you have an estimate_noise module in the parent or SpecTIR directory
# If it's in SpecTIR, you might need to adjust the import or path.
try:
    import estimate_noise
except ImportError:
    # Attempt to add SpecTIR path if it's there
    spectir_dir = script_dir.parent / "SpecTIR"
    if str(spectir_dir) not in sys.path:
        sys.path.insert(0, str(spectir_dir))
    import estimate_noise

def get_max_valid_rectangle(mask):
    """
    Finds the maximum area rectangle of True values in a 2D boolean array.
    Returns (y, x, height, width) of the largest fully-valid sub-region.
    """
    rows, cols = mask.shape
    max_area = 0
    best_rect = (0, 0, 0, 0)
    
    heights = np.zeros(cols, dtype=int)
    for r in range(rows):
        heights = np.where(mask[r], heights + 1, 0)
        stack = []
        for c in range(cols + 1):
            h = heights[c] if c < cols else 0
            start = c
            while stack and stack[-1][1] > h:
                idx, val = stack.pop()
                area = val * (c - idx)
                if area > max_area:
                    max_area = area
                    best_rect = (r - val + 1, idx, val, c - idx)
                start = idx
            stack.append((start, h))
            
    return best_rect

def plot_snr_spectrum(X, noise_covariances, wavelengths, run_id, output_dir):
    N, p = X.shape
    if wavelengths is None:
        wavelengths = np.arange(1, p + 1)
        x_label = "Spectral Band Number"
    else:
        x_label = "Wavelength (nm)"
        
    data_variances = np.var(X, axis=0, ddof=1)
    plt.figure(figsize=(10, 6))
    
    for method_name, cov_matrix in noise_covariances.items():
        noise_variances = np.diag(cov_matrix)
        signal_variances = data_variances - noise_variances
        
        epsilon = 1e-10
        signal_variances = np.clip(signal_variances, a_min=epsilon, a_max=None)
        noise_variances = np.clip(noise_variances, a_min=epsilon, a_max=None)
        
        snr_linear = signal_variances / noise_variances
        plt.plot(wavelengths, snr_linear, label=f"{method_name}", linewidth=1.5)
        
    plt.title(f"Band-by-Band SNR Comparison - {run_id}")
    plt.xlabel(x_label)
    plt.ylabel("SNR (Linear)")
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axhline(1, color='black', linewidth=1, linestyle='-')
    
    plt.tight_layout(pad=0.1)
    plt.savefig(os.path.join(output_dir, f"snr_spectrum_{run_id}.png"), dpi=400)
    plt.close()


def process_hyspex_to_h5(input_hdr, output_dir, run_id):
    if not os.path.exists(input_hdr):
        print(f"File not found: {input_hdr}")
        return

    output_h5 = os.path.join(output_dir, f"HySpex_Ortho_Stack_{run_id}.h5")
    print(f"\n--- Processing HySpex Orthorectified Data: {run_id} ---")
    
    # Determine expected image file based on header file name
    base_name = os.path.splitext(input_hdr)[0]
    expected_img = base_name + ".bsq" # Common for HySpex/PARGE
    
    if not os.path.exists(expected_img):
        # Fallback to other common extensions if .bsq is not there
        for ext in ['', '.img', '.dat', '.raw']:
            if os.path.exists(base_name + ext):
                expected_img = base_name + ext
                break
                
    if not os.path.exists(expected_img):
        raise FileNotFoundError(f"Missing ENVI data file for header '{input_hdr}'. Expected a file like '{base_name}.bsq'. This missing data problem must be corrected before proceeding.")

    # Load Image
    img = spectral.envi.open(input_hdr, image=expected_img)
    lines, samples, bands = img.shape
    print(f"Image Shape: {lines}L x {samples}S x {bands}B (dtype: {img.dtype})")
    
    # Verify file size to catch incomplete data problems
    expected_bytes = lines * samples * bands * np.dtype(img.dtype).itemsize
    header_offset = int(img.metadata.get('header offset', 0))
    expected_bytes += header_offset
    
    actual_bytes = os.path.getsize(expected_img)
    if actual_bytes < expected_bytes:
        raise ValueError(f"Data integrity problem: ENVI data file '{expected_img}' is incomplete. "
                         f"Expected at least {expected_bytes} bytes based on header dimensions, "
                         f"but found only {actual_bytes} bytes. Please correct this missing data problem before proceeding.")
    
    
    # Parse GeoTransform and CRS
    map_info = img.metadata.get('map info', [])
    gdal_transform = [0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    crs_wkt = img.metadata.get('coordinate system string', "")
    
    if len(map_info) >= 11:
        x_tie = float(map_info[3])
        y_tie = float(map_info[4])
        pixel_x = float(map_info[5])
        pixel_y = float(map_info[6])
        gdal_transform = [x_tie - pixel_x/2.0, pixel_x, 0.0, y_tie + pixel_y/2.0, 0.0, -pixel_y]

    # Wavelengths
    wavelengths = np.zeros(bands, dtype='float32')
    if 'wavelength' in img.metadata:
        wavelengths = np.array([float(w) for w in img.metadata['wavelength']], dtype='float32')
        
    valid_bands_idx = np.arange(bands)
    if WAVELENGTH_CUTOFF is not None:
        valid_bands_mask = wavelengths < WAVELENGTH_CUTOFF
        valid_bands_idx = np.where(valid_bands_mask)[0]
        bands = len(valid_bands_idx)
        wavelengths = wavelengths[valid_bands_idx]
        print(f"Applying WAVELENGTH_CUTOFF of {WAVELENGTH_CUTOFF} nm. Retaining {bands} bands.")
        
    # Fill value
    fill_value = -1.0
    if 'data ignore value' in img.metadata:
        fill_value = float(img.metadata['data ignore value'])
    print(f"Using Fill Value: {fill_value}")
    
    # Optional scaling factor if present
    scale_val = 1.0

    # Use Memory Map instead of loading the entire array
    print("Opening orthorectified image via memmap...")
    img_mm = img.open_memmap()
    
    scaled_fill_val = fill_value / scale_val
    
    # Auto-crop the bounding box of valid data to remove large nodata regions
    print("Determining valid data bounds for auto-cropping...")
    band_data = img_mm[:, :, valid_bands_idx[bands // 2]]
    
    # Create mask where pixel is NOT fill_value (check on raw data to avoid scaling precision errors)
    valid_mask_2d = (band_data != fill_value)
    
    valid_y, valid_x = np.nonzero(valid_mask_2d)
    if len(valid_y) == 0:
        raise ValueError(f"No valid data found in image (all pixels match fill value).")
        
    min_y, max_y = np.min(valid_y), np.max(valid_y)
    min_x, max_x = np.min(valid_x), np.max(valid_x)
    
    cropped_lines = max_y - min_y + 1
    cropped_samples = max_x - min_x + 1
    
    print(f"Original Grid: {lines}L x {samples}S")
    print(f"Cropped Grid:  {cropped_lines}L x {cropped_samples}S (min_y={min_y}, max_y={max_y}, min_x={min_x}, max_x={max_x})")
    
    # Update GeoTransform for the new cropped origin
    gdal_transform[0] = gdal_transform[0] + min_x * gdal_transform[1]
    gdal_transform[3] = gdal_transform[3] + min_y * gdal_transform[5]
    
    # --- Empirical Line Calibration (ELM) ---
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    elm_files = filedialog.askopenfilenames(
        title="Select ELM Calibration .npz file (Cancel to skip)",
        filetypes=[("NPZ files", "*.npz"), ("All files", "*.*")]
    )
    
    elm_gain = None
    elm_offset = None
    
    if elm_files and elm_files[0].endswith('.npz'):
        print(f"\n--- Loading existing ELM Calibration from {elm_files[0]} ---")
        cal_data = np.load(elm_files[0])
        elm_gain = cal_data['gain']
        elm_offset = cal_data['offset']
        if WAVELENGTH_CUTOFF is not None:
            elm_gain = elm_gain[valid_bands_idx]
            elm_offset = elm_offset[valid_bands_idx]
    else:
        print("\n--- No ELM Calibration provided. Proceeding without reflectance conversion ---")
    # --- End ELM ---
    
    # Estimate Noise on the entire valid pixel area
    print("Estimating Noise Matrices on the entire valid spatial area...")
    subset = img_mm[min_y:max_y+1, min_x:max_x+1, valid_bands_idx]
    subset_float = np.transpose(subset.astype(np.float32) / scale_val, (2, 0, 1))
    
    # Check for fill_value on raw data before transpose/scaling
    valid_mask_subset = np.all(subset != fill_value, axis=2)
    
    noise_covs = {}
    if 'ssd' in NOISE_ESTIMATIONS:
        noise_covs['SSD'] = estimate_noise.estimate_noise_ssd(subset_float, valid_mask_subset)
    if 'mlr' in NOISE_ESTIMATIONS:
        noise_covs['MLR'] = estimate_noise.estimate_noise_mlr(subset_float, valid_mask_subset)
    if 'dct' in NOISE_ESTIMATIONS:
        noise_covs['DCT'] = estimate_noise.estimate_noise_dct(subset_float, valid_mask_subset)
    if 'mnf' in NOISE_ESTIMATIONS:
        noise_covs['MNF'] = estimate_noise.estimate_noise_mnf(subset_float, valid_mask_subset)
    if 'nfd' in NOISE_ESTIMATIONS:
        # spectral.noise_from_diffs requires a contiguous unmasked HSI cube (rows, cols, bands)
        # Find the max rectangle that contains NO fill_values to prevent edge-gradient artifacts
        ry, rx, rh, rw = get_max_valid_rectangle(valid_mask_subset)
        if rh > 10 and rw > 10:
            print(f"Executing spectral.noise_from_diffs on inscribed contiguous {rh}x{rw} region...")
            # Extract inscribed rectangle and transpose to SPy format (H, W, B)
            nfd_subset = np.transpose(subset_float[:, ry:ry+rh, rx:rx+rw], (1, 2, 0))
            nfd_stats = spectral.noise_from_diffs(nfd_subset)
            noise_covs['NFD'] = nfd_stats.cov
        else:
            print("Warning: Could not find a large enough contiguous valid area for NFD. Skipping.")
    
    print("Generating SNR Spectrum Plot...")
    X_valid = np.transpose(subset_float, (1, 2, 0))[valid_mask_subset]
    plot_snr_spectrum(X_valid, noise_covs, wavelengths, run_id, output_dir)
    
    del subset_float
    del X_valid
    
    # Create HDF5
    print(f"Creating HDF5 Output at {output_h5}...")
    with h5py.File(output_h5, 'w') as out_h5:
        grp = out_h5.create_group("HDFEOS/GRIDS/HYSPEX/Data Fields")
        
        print("Initializing HDF5 Datasets...")
        chunks_dim = (1, bands, min(cropped_lines, 256), min(cropped_samples, 256))
        ds_ref = grp.create_dataset(
            "surface_reflectance", 
            shape=(1, bands, cropped_lines, cropped_samples),
            dtype=np.float32, 
            compression="gzip", 
            compression_opts=6, 
            shuffle=True, 
            fillvalue=fill_value, 
            chunks=chunks_dim
        )
        
        ds_ref.attrs['Scale_Factor'] = np.array([scale_val], dtype='float32')
        acq_time = img.metadata.get('acquisition time', 'Unknown')
        dt_str = h5py.string_dtype(encoding='ascii')
        ds_ref.attrs.create('acquisition_time', data=np.array([acq_time], dtype=dt_str))
        ds_ref.attrs.create('spacecraft_id', data=np.array(['HySpex'], dtype=dt_str))
        ds_ref.attrs['wavelengths'] = wavelengths
        
        # Parse and save description telemetry
        desc = img.metadata.get('description', '')
        if desc:
            ds_ref.attrs.create('sensor_description', data=np.array([desc], dtype=dt_str))
            
            import re
            matches = re.finditer(r'([A-Za-z0-9_ ]+?)\s*=\s*([A-Za-z0-9_.-]+)', desc)
            for match in matches:
                key = str(match.group(1)).strip().replace(' ', '_').lower()
                if not key:
                    continue
                val_str = match.group(2).strip()
                try:
                    if '.' in val_str:
                        val = float(val_str)
                        ds_ref.attrs[key] = np.array([val], dtype='float32')
                    else:
                        val = int(val_str)
                        ds_ref.attrs[key] = np.array([val], dtype='int32')
                except ValueError:
                    ds_ref.attrs.create(key, data=np.array([val_str], dtype=dt_str))

        if crs_wkt:
            ds_ref.attrs.create('spatial_ref', data=np.array([crs_wkt], dtype=dt_str))
        ds_ref.attrs['GeoTransform'] = np.array(gdal_transform, dtype='float64')
        
        # Save Noise Covariances
        print("Saving noise covariance matrices...")
        def save_cov(name, data, desc):
            ds = grp.create_dataset(name, data=data, dtype=np.float32, compression="gzip", compression_opts=5)
            ds.attrs.create('description', data=np.array([desc], dtype=dt_str))
            if crs_wkt:
                ds.attrs.create('spatial_ref', data=np.array([crs_wkt], dtype=dt_str))
            ds.attrs['GeoTransform'] = np.array(gdal_transform, dtype='float64')
            
        for key, cov in noise_covs.items():
            save_cov(f"{key.lower()}_noise_cov", cov, f"Estimated full noise covariance matrix (Bands x Bands) using {key} method")
        
        # Initialize RGB Preview Dataset
        ds_vis = grp.create_dataset(
            "ortho_vis", 
            shape=(1, 4, cropped_lines, cropped_samples),
            dtype='uint8', 
            compression="gzip", 
            shuffle=True, 
            fillvalue=0,
            compression_opts=6
        )
        if crs_wkt:
            ds_vis.attrs.create('spatial_ref', data=np.array([crs_wkt], dtype=dt_str))
        ds_vis.attrs['GeoTransform'] = np.array(gdal_transform, dtype='float64')
        
        # Identify Water Absorption Bands
        is_water_band = ((wavelengths > 1350) & (wavelengths < 1450)) | ((wavelengths > 1800) & (wavelengths < 1950))
        
        # Chunked Processing to populate HDF5 datasets
        print("Processing surface reflectance and RGBA chunks...")
        vis_bands = np.where(wavelengths < 1000.0)[0]
        vis_wavelengths = wavelengths[vis_bands]
        
        chunk_size = 256
        for y in range(min_y, max_y + 1, chunk_size):
            y_end = min(y + chunk_size, max_y + 1)
            y_out = y - min_y
            y_out_end = y_end - min_y
            
            for x in range(min_x, max_x + 1, chunk_size):
                x_end = min(x + chunk_size, max_x + 1)
                x_out = x - min_x
                x_out_end = x_end - min_x
                
                # Load spatial chunk via memmap and subset wavelengths
                chunk = img_mm[y:y_end, x:x_end, valid_bands_idx]
                chunk_float = chunk.astype(np.float32) / scale_val
                
                # Check for fill_value on raw data to avoid scaling precision errors
                valid_mask_chunk = (chunk != fill_value)
                
                if elm_gain is not None:
                    # Apply ELM using extremely fast vectorized NumPy broadcasting
                    # This replaces the need for slower python for-loops or external libraries like SPy
                    chunk_float = chunk_float * elm_gain + elm_offset
                    
                    # Apply Savitzky-Golay filter to smooth spectral spikes (User requested despite general smoothing rule)
                    # We MUST process contiguous valid bands separately so we don't bleed water-band noise into valid data!
                    valid_band_indices = np.where(~is_water_band)[0]
                    sequences = np.split(valid_band_indices, np.where(np.diff(valid_band_indices) > 1)[0] + 1)
                    
                    for seq in sequences:
                        if len(seq) > 5:
                            window = min(7, len(seq) if len(seq) % 2 != 0 else len(seq) - 1)
                            if window >= 3:
                                chunk_float[:, :, seq] = savgol_filter(chunk_float[:, :, seq], window_length=window, polyorder=2, axis=2)
                    
                    # Mask water bands entirely as nodata
                    chunk_float[:, :, is_water_band] = fill_value
                    valid_mask_chunk[:, :, is_water_band] = False
                    
                    # Clamp to standard reflectance bounds (0-1) safely on valid pixels
                    chunk_float[valid_mask_chunk] = np.clip(chunk_float[valid_mask_chunk], 0.0, 1.0)
                
                # Enforce exact HDF5 fill_value to fix nodata leakage from scales
                chunk_float[~valid_mask_chunk] = fill_value
                
                # Convert to (bands, lines, samples)
                chunk_float_t = np.transpose(chunk_float, (2, 0, 1))
                
                # Write to surface_reflectance
                ds_ref[0, :, y_out:y_out_end, x_out:x_out_end] = chunk_float_t
                
                # Generate RGBA on <1000nm subset
                chunk_vis = chunk_float_t[vis_bands, :, :]
                rgba_chunk = sc.generate_rgba_from_hsi(frame_data=chunk_vis, wavelengths=vis_wavelengths, nodata=scaled_fill_val, scale=1.0)
                rgba_chunk_t = np.transpose(rgba_chunk, (2, 0, 1))
                ds_vis[0, :, y_out:y_out_end, x_out:x_out_end] = rgba_chunk_t

        print("Exporting high-resolution ortho_vis preview (600 DPI)...")
        # Load the assembled RGBA dataset back for PNG export (it is lightweight 8-bit)
        rgba_img_raw_t = ds_vis[0, :, :, :]
        rgba_img_raw = np.transpose(rgba_img_raw_t, (1, 2, 0))
        png_path = os.path.join(output_dir, f"HySpex_{run_id}_ortho_vis_preview.png")
        img_pil = Image.fromarray(rgba_img_raw, 'RGBA')
        img_pil.save(png_path, dpi=(600, 600))
        print(f"Saved: {png_path}")

    print(f"Synthesis Complete. Stack Stored at: {output_h5}")

if __name__ == "__main__":
    
    process_hyspex_to_h5(HDR_PATH, OUTPUT_DIR, RUN_ID)
