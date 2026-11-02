import os
import sys
import argparse
import numpy as np
import spectral.io.envi as envi
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# ==============================================================================
# DEFAULT CONFIGURATION & FILE PATHS
# ==============================================================================
DEFAULT_HDR_PATH = r"F:\GlobusData\hyspex\20250918_GE18\dual\GE18_55m_01_Mjolnir_float32_geo_VS620.hdr"
DEFAULT_OUTPUT_NPZ = r"C:\satelliteImagery\hyspex\calibration_targets\elm_calibration_GE18_55m_01.npz"
DEFAULT_OUTPUT_FIG = r"C:\satelliteImagery\hyspex\calibration_targets\elm_validation_chips_GE18_55m_01.png"

DEFAULT_WHITE_PANEL_TXT = r"C:\satelliteImagery\hyspex\calibration_targets\big_white_panel.txt"
DEFAULT_GRAY_PANEL_TXT = r"C:\satelliteImagery\hyspex\calibration_targets\big_gray_panel.txt"

# Panel Central Coordinates
DEFAULT_MAP_X = 1059  # Big White Panel center X
DEFAULT_MAP_Y = 258   # Big White Panel center Y
DEFAULT_GRAY_X = 679  # Big Gray Panel center X
DEFAULT_GRAY_Y = 558  # Big Gray Panel center Y

WINDOW_SIZE = 15

# Evaluation Chips Configuration
VALIDATION_CHIPS = [
    {"name": "Red Location", "x": 269, "y": 975, "color": "#d62728", "label": "Red Target (x=269, y=975)"},
    {"name": "Green Location", "x": 214, "y": 901, "color": "#2ca02c", "label": "Green Target (x=214, y=901)"},
    {"name": "Blue Location", "x": 163, "y": 842, "color": "#1f77b4", "label": "Blue Target (x=163, y=842)"}
]


def extract_window_chip(img_mm, cx, cy, window_size=15):
    """
    Extracts a spatial chip of dimensions (window_size, window_size, bands)
    centered at (cx, cy) from an ENVI memory-mapped image cube.
    """
    lines, samples, bands = img_mm.shape
    offset = window_size // 2
    y_start = max(0, cy - offset)
    y_end = min(lines, cy + offset + 1)
    x_start = max(0, cx - offset)
    x_end = min(samples, cx + offset + 1)

    chip = img_mm[y_start:y_end, x_start:x_end, :].astype(np.float32)
    return chip


def extract_window_radiance(img_mm, cx, cy, window_size=15, fill_value=-1.0, scale_val=1.0):
    """
    Extracts a (window_size x window_size) spatial tile centered at (cx, cy)
    and computes the mean radiance spectrum across all valid pixels for each band.
    """
    chip = extract_window_chip(img_mm, cx, cy, window_size=window_size) / scale_val
    bands = chip.shape[2]
    
    valid_mask = (chip != fill_value)
    flat_chip = chip.reshape(-1, bands)
    flat_valid = valid_mask.reshape(-1, bands)

    mean_rad = np.zeros(bands, dtype=np.float32)
    std_rad = np.zeros(bands, dtype=np.float32)

    for b in range(bands):
        valid_pix = flat_chip[flat_valid[:, b], b]
        if len(valid_pix) > 0:
            mean_rad[b] = np.mean(valid_pix)
            std_rad[b] = np.std(valid_pix)
        else:
            mean_rad[b] = 0.0
            std_rad[b] = 0.0

    return mean_rad, std_rad, chip


def compute_elm(rad_white, rad_gray, refl_white, refl_gray, wavelengths):
    """
    Computes Empirical Line Method (ELM) gain and offset vectors across all bands
    using the 15x15 mean radiance observations of the white and gray panels.
    Excludes atmospheric water absorption bands (1350-1450 nm and 1800-2050 nm).
    """
    bands = len(wavelengths)
    elm_gain = np.ones(bands, dtype=np.float32)
    elm_offset = np.zeros(bands, dtype=np.float32)

    # Water absorption bands: 1.35-1.45 um and 1.80-2.05 um, plus noisy SWIR > 2.42 um
    is_water_band = ((wavelengths >= 1350.0) & (wavelengths <= 1450.0)) | ((wavelengths >= 1800.0) & (wavelengths <= 2050.0)) | (wavelengths > 2420.0)

    for b in range(bands):
        if is_water_band[b]:
            elm_gain[b] = 1.0
            elm_offset[b] = 0.0
            continue

        x = np.array([rad_white[b], rad_gray[b]], dtype=np.float64)
        y = np.array([refl_white[b], refl_gray[b]], dtype=np.float64)

        if np.ptp(x) < 1e-6 or np.mean(x) < 1e-6:
            with np.errstate(divide='ignore', invalid='ignore'):
                elm_gain[b] = float(y[0] / x[0]) if x[0] > 0 else 1.0
            elm_offset[b] = 0.0
        else:
            m = (y[0] - y[1]) / (x[0] - x[1])
            c = y[0] - m * x[0]
            elm_gain[b] = float(m)
            elm_offset[b] = float(c)

    return elm_gain, elm_offset, is_water_band


def make_gapped_spectral_plot_arrays(wv, data, gap_threshold_nm=20.0):
    """
    Inserts a midpoint wavelength with np.nan data whenever the step between
    adjacent wavelength channels exceeds `gap_threshold_nm`.
    Prevents Matplotlib from drawing artificial lines across omitted water absorption bands.
    """
    wv = np.asarray(wv, dtype=np.float32)
    data = np.asarray(data, dtype=np.float32)
    gap_indices = np.where(np.diff(wv) > gap_threshold_nm)[0]

    if len(gap_indices) == 0:
        return wv, data

    new_wv = list(wv)
    if data.ndim == 1:
        new_data = list(data)
        for offset, idx in enumerate(gap_indices):
            insert_pos = idx + 1 + offset
            mid_wv = (wv[idx] + wv[idx + 1]) / 2.0
            new_wv.insert(insert_pos, mid_wv)
            new_data.insert(insert_pos, np.nan)
        return np.array(new_wv, dtype=np.float32), np.array(new_data, dtype=np.float32)
    elif data.ndim == 2:
        rows = [data[i, :] for i in range(len(wv))]
        nan_row = np.full(data.shape[1], np.nan, dtype=np.float32)
        for offset, idx in enumerate(gap_indices):
            insert_pos = idx + 1 + offset
            mid_wv = (wv[idx] + wv[idx + 1]) / 2.0
            new_wv.insert(insert_pos, mid_wv)
            rows.insert(insert_pos, nan_row)
        return np.array(new_wv, dtype=np.float32), np.array(rows, dtype=np.float32)


def generate_validation_figure(img_mm, elm_gain, elm_offset, wavelengths, is_water_band,
                               chips_config=VALIDATION_CHIPS, window_size=15, scale_val=1.0,
                               output_fig_path=DEFAULT_OUTPUT_FIG, show_plot=True):
    """
    Renders a 2-row validation figure:
      - Top row: 3 subplots displaying true-color / RGB 15x15 image chips for Red, Green, and Blue locations.
      - Bottom row: Single span subplot showing ELM-calibrated surface reflectance spectra for all 225 individual pixels per chip.
    """
    bands = len(wavelengths)
    r_band = int(np.argmin(np.abs(wavelengths - 640.0)))
    g_band = int(np.argmin(np.abs(wavelengths - 550.0)))
    b_band = int(np.argmin(np.abs(wavelengths - 470.0)))

    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.25], hspace=0.32, wspace=0.25)

    # 1. Plot RGB Image Chips (Top Row)
    for i, cfg in enumerate(chips_config):
        ax_chip = fig.add_subplot(gs[0, i])
        cx, cy = cfg['x'], cfg['y']
        raw_chip = extract_window_chip(img_mm, cx, cy, window_size=window_size) / scale_val
        
        # Apply ELM Calibration to chip
        cal_chip = raw_chip * elm_gain + elm_offset
        cal_chip[:, :, is_water_band] = np.nan

        rgb = cal_chip[:, :, [r_band, g_band, b_band]]
        # Robust percentile normalization for crisp visual presentation
        p_low = np.nanpercentile(rgb, 1.0)
        p_high = np.nanpercentile(rgb, 99.0)
        if p_high > p_low:
            rgb_disp = np.clip((rgb - p_low) / (p_high - p_low), 0.0, 1.0)
        else:
            rgb_disp = np.clip(rgb, 0.0, 1.0)

        ax_chip.imshow(rgb_disp)
        ax_chip.set_title(f"{cfg['name']}\n(x={cx}, y={cy})", fontsize=11, fontweight='bold', color=cfg['color'])
        ax_chip.set_xlabel("Local Column (0-14)")
        ax_chip.set_ylabel("Local Row (0-14)")
        
        # Center pixel crosshair
        center_idx = window_size // 2
        ax_chip.scatter(center_idx, center_idx, color='white', s=70, marker='+', linewidth=2.0)
        ax_chip.scatter(center_idx, center_idx, color='black', s=25, marker='.', linewidth=1.0)

    # 2. Plot Calibrated Spectra (Bottom Row) - All individual pixel curves
    ax_spec = fig.add_subplot(gs[1, :])

    # Shaded water vapor absorption intervals and SWIR cutoff
    ax_spec.axvspan(1350, 1450, color='#e0e0e0', alpha=0.55, label='Atmospheric H₂O Absorption (Omitted)')
    ax_spec.axvspan(1800, 2050, color='#e0e0e0', alpha=0.55)
    if np.any(wavelengths > 2420):
        ax_spec.axvspan(2420, np.nanmax(wavelengths) + 10, color='#e0e0e0', alpha=0.55, label='SWIR Noise Cutoff (>2420 nm)')

    wv_valid = wavelengths[~is_water_band]

    for cfg in chips_config:
        cx, cy = cfg['x'], cfg['y']
        raw_chip = extract_window_chip(img_mm, cx, cy, window_size=window_size) / scale_val
        cal_chip = raw_chip * elm_gain + elm_offset

        # Reshape to (pixels, bands) and filter to non-water bands
        pixels = cal_chip.reshape(-1, bands)[:, ~is_water_band]

        # Break lines across telluric water absorption intervals: (num_gapped, 225)
        wv_g, pixels_g = make_gapped_spectral_plot_arrays(wv_valid, pixels.T)

        # Plot all individual pixel spectra with low alpha to expose clustering and outliers
        ax_spec.plot(wv_g, pixels_g[:, 0], color=cfg['color'], alpha=0.25, linewidth=0.7,
                     label=f"{cfg['name']} (All {pixels.shape[0]} Pixels, x={cx}, y={cy})")
        ax_spec.plot(wv_g, pixels_g[:, 1:], color=cfg['color'], alpha=0.25, linewidth=0.7)

        # Overlay mean spectrum with distinct dashed line
        mean_spec = np.nanmean(pixels, axis=0)
        _, mean_g = make_gapped_spectral_plot_arrays(wv_valid, mean_spec)
        ax_spec.plot(wv_g, mean_g, color=cfg['color'], linewidth=2.0, linestyle='--',
                     label=f"{cfg['name']} Mean")

    ax_spec.set_title(r"ELM-Calibrated Surface Reflectance Spectra ($15 \times 15$ Chips, All Overlapping Pixels)",
                      fontsize=12, fontweight='bold')
    ax_spec.set_xlabel("Wavelength (nm)", fontsize=11)
    ax_spec.set_ylabel("Surface Reflectance", fontsize=11)
    ax_spec.set_ylim(-0.02, 1.0)
    ax_spec.grid(True, linestyle='--', alpha=0.6)
    ax_spec.legend(loc='upper right', fontsize=9, ncol=2)

    fig.suptitle(r"HySpex Mjolnir VS-620: $15 \times 15$ Tile ELM Calibration Validation",
                 fontsize=14, fontweight='bold', y=0.98)

    if output_fig_path:
        os.makedirs(os.path.dirname(output_fig_path), exist_ok=True)
        plt.savefig(output_fig_path, dpi=300, bbox_inches='tight')
        print(f"Validation figure successfully saved to: {output_fig_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Compute ELM Calibration using 15x15 tile averages over calibration panels.")
    parser.add_argument("--hdr", type=str, default=DEFAULT_HDR_PATH, help="Path to raw ENVI header file.")
    parser.add_argument("--white_panel_txt", type=str, default=DEFAULT_WHITE_PANEL_TXT, help="Path to white panel ground truth spectrum.")
    parser.add_argument("--gray_panel_txt", type=str, default=DEFAULT_GRAY_PANEL_TXT, help="Path to gray panel ground truth spectrum.")
    parser.add_argument("--white_x", type=int, default=DEFAULT_MAP_X, help="Center X coordinate of white panel (default: 1059).")
    parser.add_argument("--white_y", type=int, default=DEFAULT_MAP_Y, help="Center Y coordinate of white panel (default: 258).")
    parser.add_argument("--gray_x", type=int, default=DEFAULT_GRAY_X, help="Center X coordinate of gray panel (default: 679).")
    parser.add_argument("--gray_y", type=int, default=DEFAULT_GRAY_Y, help="Center Y coordinate of gray panel (default: 558).")
    parser.add_argument("--window_size", type=int, default=WINDOW_SIZE, help="Averaging window size (default: 15).")
    parser.add_argument("--out_npz", type=str, default=DEFAULT_OUTPUT_NPZ, help="Path to save output ELM calibration npz file.")
    parser.add_argument("--out_fig", type=str, default=DEFAULT_OUTPUT_FIG, help="Path to save output validation figure.")
    parser.add_argument("--no_show", action="store_true", help="Do not display interactive plot window.")
    args = parser.parse_args()

    print("================================================================================")
    print("HYSPEX MJOLNIR VS-620: 15x15 TILE EMPIRICAL LINE METHOD (ELM) CALIBRATION")
    print("================================================================================")
    print(f"Header:            {args.hdr}")
    print(f"White Panel Target: {args.white_panel_txt} at (x={args.white_x}, y={args.white_y})")
    print(f"Gray Panel Target:  {args.gray_panel_txt} at (x={args.gray_x}, y={args.gray_y})")
    print(f"Window Size:       {args.window_size}x{args.window_size} ({args.window_size**2} pixels)")
    print(f"Output NPZ:        {args.out_npz}")
    print(f"Output Figure:     {args.out_fig}")
    print("--------------------------------------------------------------------------------")

    if not os.path.exists(args.hdr):
        raise FileNotFoundError(f"Raw ENVI header file not found: {args.hdr}")
    if not os.path.exists(args.white_panel_txt):
        raise FileNotFoundError(f"Ground truth white panel file not found: {args.white_panel_txt}")
    if not os.path.exists(args.gray_panel_txt):
        raise FileNotFoundError(f"Ground truth gray panel file not found: {args.gray_panel_txt}")

    # 1. Open ENVI Image
    img = envi.open(args.hdr)
    base_name = os.path.splitext(args.hdr)[0]
    expected_img = base_name + ".bsq"
    if not os.path.exists(expected_img):
        for ext in ['', '.img', '.dat', '.raw']:
            if os.path.exists(base_name + ext):
                expected_img = base_name + ext
                break

    img_mm = img.open_memmap(interleave='bip')
    lines, samples, bands = img_mm.shape
    print(f"Loaded image: {lines} lines x {samples} samples x {bands} bands")

    scale_val = 1.0
    if 'reflectance scale factor' in img.metadata:
        try:
            val = img.metadata['reflectance scale factor']
            scale_val = float(val[0]) if isinstance(val, list) else float(val)
        except Exception:
            pass

    fill_value = -1.0
    if 'data ignore value' in img.metadata:
        fill_value = float(img.metadata['data ignore value'])

    wavelengths = np.array([float(w) for w in img.metadata['wavelength']], dtype=np.float32)

    # 2. Extract 15x15 Tile Radiance for White and Gray Panels
    print(f"\nExtracting {args.window_size}x{args.window_size} radiance over Big White Panel at ({args.white_x}, {args.white_y})...")
    rad_white, std_white, _ = extract_window_radiance(
        img_mm, args.white_x, args.white_y, window_size=args.window_size, fill_value=fill_value, scale_val=scale_val
    )
    b700 = int(np.argmin(np.abs(wavelengths - 700.0)))
    print(f"  White Panel Mean Radiance at 700 nm: {rad_white[b700]:.4f} +/- {std_white[b700]:.4f}")

    print(f"\nExtracting {args.window_size}x{args.window_size} radiance over Big Gray Panel at ({args.gray_x}, {args.gray_y})...")
    rad_gray, std_gray, _ = extract_window_radiance(
        img_mm, args.gray_x, args.gray_y, window_size=args.window_size, fill_value=fill_value, scale_val=scale_val
    )
    print(f"  Gray Panel Mean Radiance at 700 nm:  {rad_gray[b700]:.4f} +/- {std_gray[b700]:.4f}")

    # 3. Load and Resample Ground Truth Spectra
    d_white = np.loadtxt(args.white_panel_txt, skiprows=1)
    d_gray = np.loadtxt(args.gray_panel_txt, skiprows=1)

    f_white = interp1d(d_white[:, 0], d_white[:, 1], kind='linear', bounds_error=False, fill_value="extrapolate")
    f_gray = interp1d(d_gray[:, 0], d_gray[:, 1], kind='linear', bounds_error=False, fill_value="extrapolate")

    refl_white = f_white(wavelengths)
    refl_gray = f_gray(wavelengths)
    print(f"  Ground Truth Reflectance at 700 nm:  White = {refl_white[b700]:.4f}, Gray = {refl_gray[b700]:.4f}")

    # 4. Compute ELM Calibration Coefficients
    print("\nComputing 2-point ELM Gain and Offset vectors (excluding 1350-1450 nm & 1800-2050 nm)...")
    elm_gain, elm_offset, is_water_band = compute_elm(rad_white, rad_gray, refl_white, refl_gray, wavelengths)

    valid_bands = ~is_water_band
    print(f"  Gain min/max/mean (valid bands):   {np.min(elm_gain[valid_bands]):.3f} / {np.max(elm_gain[valid_bands]):.3f} / {np.mean(elm_gain[valid_bands]):.3f}")
    print(f"  Offset min/max/mean (valid bands): {np.min(elm_offset[valid_bands]):.4f} / {np.max(elm_offset[valid_bands]):.4f} / {np.mean(elm_offset[valid_bands]):.4f}")

    # 5. Save ELM Calibration NPZ
    os.makedirs(os.path.dirname(args.out_npz), exist_ok=True)
    np.savez(args.out_npz, gain=elm_gain, offset=elm_offset, wavelengths=wavelengths)
    print(f"\nSaved ELM Calibration file to: {args.out_npz}")

    # 6. Generate Validation Figure with 3 Chips & Calibrated Spectra
    print("\nGenerating validation figure for Red, Green, and Blue 15x15 chips...")
    generate_validation_figure(
        img_mm=img_mm,
        elm_gain=elm_gain,
        elm_offset=elm_offset,
        wavelengths=wavelengths,
        is_water_band=is_water_band,
        chips_config=VALIDATION_CHIPS,
        window_size=args.window_size,
        scale_val=scale_val,
        output_fig_path=args.out_fig,
        show_plot=(not args.no_show)
    )

    print("\nELM calibration and validation completed successfully!")
    print("================================================================================")


if __name__ == "__main__":
    main()
