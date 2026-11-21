import os
import sys
import numpy as np
import h5py
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import spectral.io.envi as envi

# Add parent directory to sys.path to find SpecComplex and SpecTIR modules
script_dir = Path(__file__).resolve().parent
parent_dir = script_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))
spectir_dir = parent_dir / "SpecTIR"
if str(spectir_dir) not in sys.path:
    sys.path.insert(0, str(spectir_dir))

import SpecComplex as sc
try:
    import estimate_noise
except ImportError:
    estimate_noise = None

# Default Configuration Paths
DEFAULT_HDR_PATH = r"C:\satelliteImagery\headwall-vnir\pfm1-landmine-uav-vnir-hsi-IGARSS-2026\site_with_only_mines.hdr"
DEFAULT_IMG_PATH = r"C:\satelliteImagery\headwall-vnir\pfm1-landmine-uav-vnir-hsi-IGARSS-2026\site_with_only_mines"
DEFAULT_GT_HDR = r"C:\satelliteImagery\headwall-vnir\pfm1-landmine-uav-vnir-hsi-IGARSS-2026\binary_ground_truth_mask_for_pfm1.hdr"
DEFAULT_GT_IMG = r"C:\satelliteImagery\headwall-vnir\pfm1-landmine-uav-vnir-hsi-IGARSS-2026\binary_ground_truth_mask_for_pfm1"
DEFAULT_SVC_PATH = r"C:\satelliteImagery\headwall-vnir\pfm1-landmine-uav-vnir-hsi-IGARSS-2026\target_signature_pfm_1_svc.txt"
OUTPUT_DIR = r"C:\satelliteImagery\headwall-vnir\pfm1-landmine-uav-vnir-hsi-IGARSS-2026"

# Scientific scaling configuration:
# The raw Headwall cube is ELM-calibrated into Percent Reflectance (0 - 100%).
# Downstream algorithms (Gram volumes, OSP 1% critical threshold h_crit = 0.010 * sqrt(B),
# CIE 1931 CMF colorimetry, and SSD noise estimation) assume unit reflectance [0.0, 1.0].
# When NORMALIZE_TO_UNIT_REFLECTANCE is True, values are scaled by 0.01 and Scale_Factor is 1.0.
NORMALIZE_TO_UNIT_REFLECTANCE = True
COMPUTE_NOISE_ESTIMATION = True
CHUNK_SIZE = 256


def plot_snr_spectrum(X_valid, noise_covs, wavelengths, run_id, output_dir, method_signal_vars=None):
    if len(noise_covs) == 0:
        return
    data_vars = np.var(X_valid, axis=0, ddof=1)
    num_methods = len(noise_covs)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    fig, axes = plt.subplots(num_methods, 1, figsize=(11, 4 * num_methods), sharex=True)
    if num_methods == 1:
        axes = [axes]

    for i, (method_name, cov_matrix) in enumerate(noise_covs.items()):
        ax = axes[i]
        c = colors[i % len(colors)]
        noise_vars = np.diag(cov_matrix)
        if method_signal_vars and method_name in method_signal_vars:
            base_var = method_signal_vars[method_name]
        else:
            base_var = data_vars

        signal_vars = np.clip(base_var - noise_vars, a_min=1e-10, a_max=None)
        noise_vars = np.clip(noise_vars, a_min=1e-10, a_max=None)
        snr_linear = signal_vars / noise_vars

        ax.plot(wavelengths, snr_linear, label=f"{method_name}", color=c, linewidth=1.5)
        ax.axhline(1, color='black', linewidth=1, linestyle='--', alpha=0.5, label='SNR = 1')
        ax.set_ylabel(f"{method_name} SNR (Linear)", color=c, fontweight='bold', fontsize=11)
        ax.tick_params(axis='y', labelcolor=c)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='upper right', fontsize=10)
        ax.set_title(f"{method_name} Noise Estimation - Linear SNR", fontsize=12, fontweight='bold')

    axes[-1].set_xlabel("Wavelength (nm)", fontsize=11)
    fig.suptitle(f"Band-by-Band SNR Comparison - {run_id}", fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(pad=1.2)
    out_png = os.path.join(output_dir, f"snr_spectrum_{run_id}.png")
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"Saved SNR plot: {out_png}")


def process_headwall_to_h5(
    hdr_path=DEFAULT_HDR_PATH,
    img_path=DEFAULT_IMG_PATH,
    gt_hdr=DEFAULT_GT_HDR,
    gt_img=DEFAULT_GT_IMG,
    svc_path=DEFAULT_SVC_PATH,
    output_dir=OUTPUT_DIR,
    normalize_to_unit_reflectance=NORMALIZE_TO_UNIT_REFLECTANCE
):
    """
    Translates Headwall Nano-Hyperspec VNIR ENVI imagery into a standardized HDF5 container.
    Constructs:
      - 'surface_reflectance': shape (1, bands, lines, samples), dtype float32
      - 'ortho_vis': shape (1, 4, lines, samples), dtype uint8 true color RGBA via CIE 1931 CMFs
      - 'ground_truth_mask': shape (lines, samples), dtype uint8 (248 PFM-1 mine target pixels)
      - 'target_signature_svc': shape (bands, 2), dtype float32 (reference ground spectrum)
      - 'ssd_noise_cov': shape (bands, bands), dtype float32 (homogeneous noise covariance)
    Creates HDF5 SoftLink from /HDFEOS/GRIDS/HYSPEX to /HDFEOS/GRIDS/HEADWALL for 100% interoperability.
    """
    if not os.path.exists(hdr_path):
        raise FileNotFoundError(f"Headwall header not found: {hdr_path}")
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Headwall image file not found: {img_path}")

    os.makedirs(output_dir, exist_ok=True)
    run_id = Path(hdr_path).stem
    output_h5 = os.path.join(output_dir, f"{run_id}.h5")
    print(f"\n{'='*60}\nTranslating Headwall VNIR Data to HDF5: {run_id}\n{'='*60}")
    print(f"Input ENVI HDR: {hdr_path}")
    print(f"Output HDF5:    {output_h5}")

    # Open ENVI via Spectral Python
    img = envi.open(hdr_path, img_path)
    lines, samples, bands = img.shape
    print(f"Image Shape: {lines} Lines x {samples} Samples x {bands} Bands (dtype: {img.dtype})")

    # Wavelengths
    wavelengths = np.array([float(w) for w in img.metadata.get('wavelength', [])], dtype=np.float32)
    if len(wavelengths) != bands:
        raise ValueError(f"Wavelength count mismatch: header lists {len(wavelengths)}, image has {bands} bands.")

    # Spatial Reference and Coordinates
    map_info = img.metadata.get('map info', [])
    crs_wkt = img.metadata.get('coordinate system string', "")
    gdal_transform = [0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    if len(map_info) >= 7:
        try:
            x_tie = float(map_info[3])
            y_tie = float(map_info[4])
            pixel_x = float(map_info[5])
            pixel_y = float(map_info[6])
            gdal_transform = [x_tie - pixel_x / 2.0, pixel_x, 0.0, y_tie + pixel_y / 2.0, 0.0, -pixel_y]
        except (ValueError, IndexError):
            pass

    # Scaling factor
    if normalize_to_unit_reflectance:
        scale_val = 100.0  # Raw data is percent reflectance [0, 100], divide by 100.0 to get [0, 1]
        out_scale_factor = 1.0
        units_desc = "Unit Reflectance [0.0, 1.0]"
    else:
        scale_val = 1.0
        out_scale_factor = 100.0
        units_desc = "Percent Reflectance [0.0, 100.0]"

    # Open memmap for streaming
    print("Opening ENVI image via memory map...")
    img_mm = img.open_memmap()

    # Load Ground Truth Mask if available
    gt_data = None
    if gt_hdr and os.path.exists(gt_hdr) and os.path.exists(gt_img):
        print(f"Loading Ground Truth Mask: {gt_hdr}...")
        gt_envi = envi.open(gt_hdr, gt_img)
        gt_data = np.squeeze(gt_envi.read_band(0)).astype(np.uint8)
        num_targets = int(np.sum(gt_data == 1))
        print(f"Loaded ground truth mask with {num_targets} target pixels.")

    # Load SVC reference spectrum if available
    svc_data = None
    if svc_path and os.path.exists(svc_path):
        print(f"Loading SVC Reference Ground Spectrum: {svc_path}...")
        try:
            svc_raw = np.loadtxt(svc_path, skiprows=1)
            # svc_raw columns: wavelength, reflectance (%)
            if normalize_to_unit_reflectance:
                svc_data = np.column_stack((svc_raw[:, 0], svc_raw[:, 1] / 100.0)).astype(np.float32)
            else:
                svc_data = svc_raw.astype(np.float32)
            print(f"Loaded SVC reference spectrum ({len(svc_data)} bands).")
        except Exception as e:
            print(f"Warning: Failed to parse SVC spectrum: {e}")

    # Estimate Noise Covariance on a representative spatial crop
    noise_covs = {}
    method_signal_vars = {}
    if COMPUTE_NOISE_ESTIMATION and estimate_noise is not None:
        print("Estimating Noise Covariance Matrix (SSD) on representative spatial crop...")
        try:
            # Crop center 400x400 region
            cy, cx = lines // 2, samples // 2
            h_crop = min(400, lines)
            w_crop = min(400, samples)
            crop_patch = img_mm[cy - h_crop//2 : cy + h_crop//2, cx - w_crop//2 : cx + w_crop//2, :]
            crop_float = np.transpose(crop_patch.astype(np.float32) / scale_val, (2, 0, 1))
            crop_mask = np.ones((h_crop, w_crop), dtype=bool)

            ssd_cov, ssd_sig = estimate_noise.estimate_noise_ssd(
                crop_float, crop_mask, return_signal_variance=True
            )
            noise_covs['SSD'] = ssd_cov
            method_signal_vars['SSD'] = ssd_sig
            print("SSD noise covariance estimation complete.")

            # Generate SNR spectrum plot
            X_valid = np.transpose(crop_float, (1, 2, 0)).reshape(-1, bands)
            plot_snr_spectrum(X_valid, noise_covs, wavelengths, run_id, output_dir, method_signal_vars)
        except Exception as e:
            print(f"Warning: Noise estimation failed: {e}")

    # Create HDF5 File
    print(f"\nCreating HDF5 file: {output_h5}...")
    with h5py.File(output_h5, 'w') as out_h5:
        # Canonical group: HEADWALL
        grp = out_h5.create_group("HDFEOS/GRIDS/HEADWALL/Data Fields")

        # SoftLink to HYSPEX for seamless compatibility with hyspex_viewer and process_metrics
        out_h5["HDFEOS/GRIDS/HYSPEX"] = h5py.SoftLink("/HDFEOS/GRIDS/HEADWALL")

        dt_str = h5py.string_dtype(encoding='ascii')

        # 1. Surface Reflectance Dataset
        print("Initializing dataset 'surface_reflectance'...")
        chunks_dim = (1, bands, min(lines, CHUNK_SIZE), min(samples, CHUNK_SIZE))
        ds_sr = grp.create_dataset(
            "surface_reflectance",
            shape=(1, bands, lines, samples),
            dtype=np.float32,
            compression="gzip",
            compression_opts=4,
            shuffle=True,
            fillvalue=-1.0,
            chunks=chunks_dim
        )
        ds_sr.attrs['Scale_Factor'] = np.array([out_scale_factor], dtype='float32')
        ds_sr.attrs['wavelengths'] = wavelengths
        ds_sr.attrs['units'] = np.array([units_desc], dtype=dt_str)
        ds_sr.attrs['original_units'] = np.array(["Percent Reflectance (0-100%) via ELM Calibration"], dtype=dt_str)
        ds_sr.attrs['spacecraft_id'] = np.array(["Headwall Nano-Hyperspec"], dtype=dt_str)
        ds_sr.attrs['sensor_description'] = np.array([img.metadata.get('description', 'Headwall VNIR UAV')], dtype=dt_str)
        if crs_wkt:
            ds_sr.attrs['spatial_ref'] = np.array([crs_wkt], dtype=dt_str)
        ds_sr.attrs['GeoTransform'] = np.array(gdal_transform, dtype='float64')

        # 2. True Color Ortho Preview Dataset (ortho_vis)
        print("Initializing dataset 'ortho_vis'...")
        ds_vis = grp.create_dataset(
            "ortho_vis",
            shape=(1, 4, lines, samples),
            dtype='uint8',
            compression="gzip",
            compression_opts=4,
            shuffle=True,
            fillvalue=0
        )
        if crs_wkt:
            ds_vis.attrs['spatial_ref'] = np.array([crs_wkt], dtype=dt_str)
        ds_vis.attrs['GeoTransform'] = np.array(gdal_transform, dtype='float64')

        # 3. Ground Truth Mask Dataset
        if gt_data is not None:
            print("Saving dataset 'ground_truth_mask'...")
            ds_gt = grp.create_dataset(
                "ground_truth_mask",
                data=gt_data,
                dtype='uint8',
                compression="gzip",
                compression_opts=4
            )
            ds_gt.attrs['num_targets'] = int(np.sum(gt_data == 1))
            ds_gt.attrs['target_name'] = np.array(["PFM-1 Landmine"], dtype=dt_str)
            if crs_wkt:
                ds_gt.attrs['spatial_ref'] = np.array([crs_wkt], dtype=dt_str)
            ds_gt.attrs['GeoTransform'] = np.array(gdal_transform, dtype='float64')

        # 4. SVC Reference Spectrum Dataset
        if svc_data is not None:
            print("Saving dataset 'target_signature_svc'...")
            ds_svc = grp.create_dataset(
                "target_signature_svc",
                data=svc_data,
                dtype='float32'
            )
            ds_svc.attrs['columns'] = np.array(["Wavelength (nm)", units_desc], dtype=dt_str)

        # 5. Noise Covariance Matrix
        for key, cov in noise_covs.items():
            ds_cov = grp.create_dataset(
                f"{key.lower()}_noise_cov",
                data=cov,
                dtype=np.float32,
                compression="gzip",
                compression_opts=4
            )
            ds_cov.attrs['description'] = np.array([f"Noise covariance matrix using {key}"], dtype=dt_str)

        # Populate Surface Reflectance and RGBA Preview via Chunked Streaming
        print(f"Populating datasets in chunks of {CHUNK_SIZE}x{CHUNK_SIZE}...")
        for y in range(0, lines, CHUNK_SIZE):
            y_end = min(y + CHUNK_SIZE, lines)
            for x in range(0, samples, CHUNK_SIZE):
                x_end = min(x + CHUNK_SIZE, samples)

                # Read raw spatial chunk from ENVI memmap
                chunk = img_mm[y:y_end, x:x_end, :] # (H_chunk, W_chunk, bands)
                chunk_float = chunk.astype(np.float32) / scale_val

                # Write to surface_reflectance (1, bands, lines, samples)
                chunk_float_t = np.transpose(chunk_float, (2, 0, 1))
                ds_sr[0, :, y:y_end, x:x_end] = chunk_float_t

                # Generate true-color RGBA via CIE 1931 CMFs
                rgba_chunk = sc.generate_rgba_from_hsi(
                    frame_data=chunk_float_t,
                    wavelengths=wavelengths,
                    nodata=-1.0
                )
                rgba_chunk_t = np.transpose(rgba_chunk, (2, 0, 1))
                ds_vis[0, :, y:y_end, x:x_end] = rgba_chunk_t

            pct = (y_end / lines) * 100.0
            print(f"  Processed {y_end}/{lines} lines ({pct:.1f}%)...")

        # Export high-resolution ortho visual preview PNG
        print("Exporting high-resolution ortho_vis preview PNG...")
        rgba_img_raw = np.transpose(ds_vis[0, :, :, :], (1, 2, 0))
        png_path = os.path.join(output_dir, f"{run_id}_ortho_vis_preview.png")
        img_pil = Image.fromarray(rgba_img_raw, 'RGBA')
        img_pil.save(png_path, dpi=(300, 300))
        print(f"Saved preview: {png_path}")

    print(f"\nTranslation Complete! Standardized HDF5 file ready at:\n  {output_h5}\n")
    return output_h5


if __name__ == "__main__":
    # Check if header path passed via CLI or environment
    selected_hdr = None
    if len(sys.argv) > 1 and not sys.argv[1].startswith("--"):
        selected_hdr = sys.argv[1]
    elif os.environ.get("HEADLESS") == "1" or "--headless" in sys.argv:
        selected_hdr = DEFAULT_HDR_PATH
    else:
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)

            selected_hdr = filedialog.askopenfilename(
                title="Select Headwall ENVI Header (.hdr)",
                filetypes=[("ENVI Header", "*.hdr"), ("All files", "*.*")],
                initialdir=OUTPUT_DIR
            )
        except Exception:
            selected_hdr = None

    if not selected_hdr:
        print(f"No file selected. Using default: {DEFAULT_HDR_PATH}")
        selected_hdr = DEFAULT_HDR_PATH

    base = os.path.splitext(selected_hdr)[0]
    expected_img = base
    if not os.path.exists(expected_img):
        for ext in ['.img', '.dat', '.raw', '.bsq']:
            if os.path.exists(base + ext):
                expected_img = base + ext
                break

    process_headwall_to_h5(
        hdr_path=selected_hdr,
        img_path=expected_img,
        output_dir=os.path.dirname(selected_hdr)
    )
