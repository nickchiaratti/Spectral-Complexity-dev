#!/usr/bin/env python3
"""
compare_real_tile_vs_simulated.py

Compares the spectral complexity metrics displayed in `hyspex_viewer.py` for a real
HySpex Ortho Stack 15x15 tile centered at (1059, 258) with a simulated calibration target
("CALTARGET:big_white_panel") at SNR=100 generated using methods from `simulate_inset_squares.py`.

Dataset:
  Real HDF5: C:\\satelliteImagery\\hyspex\\20250918_GE18\\dual\\HySpex_Ortho_Stack_20250918_ge18_55m_01.h5
  Center Coords: (map_x=1059, map_y=258)
  Tile Size: 15x15
  Extracted Endmembers: 10
  Sensor: HySpex Mjolnir (475 bands)
  Simulated Material: CALTARGET:big_white_panel
  Simulated SNR: 100
"""

import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import h5py

# Add simulated-data-test to path for library methods
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SIM_DIR = os.path.normpath(os.path.join(CURRENT_DIR, "..", "simulated-data-test"))
if SIM_DIR not in sys.path:
    sys.path.append(SIM_DIR)

try:
    from simulate_inset_squares import (
        maximumDistance_parallelotopeVolumes,
        maximumDistance_parallelotopeVolumes_noise,
        add_wavelength_scaled_noise,
        generate_rgba_from_hsi,
        load_asd_txt,
        CALIBRATION_TARGETS_DIR,
        cvHySpexMjolnir_WV_FILE,
        cvHySpexMjolnir_FWHM_FILE
    )
except ImportError:
    # Direct fallbacks if imported outside workspace root
    CALIBRATION_TARGETS_DIR = r"C:\satelliteImagery\ground_truth\calibration_targets"
    cvHySpexMjolnir_WV_FILE = os.path.normpath(
        os.path.join(CURRENT_DIR, "..", "wavelengths", "hyspex_mjolnir_vs620_wavelengths_microns.txt")
    )
    cvHySpexMjolnir_FWHM_FILE = os.path.normpath(
        os.path.join(CURRENT_DIR, "..", "wavelengths", "hyspex_mjolnir_vs620_fwhm_microns.txt")
    )

try:
    from spectral_complexity_vs_pca import resample_asd_to_sensor, cvASD_WV_FILE
except ImportError:
    cvASD_WV_FILE = r"C:\satelliteImagery\ground_truth\splib07\ASCIIdata\ASCIIdata_splib07b_cvASD\s07_ASD_Wavelengths_ASD_0.35-2.5_microns_2151_ch.txt"
    def resample_asd_to_sensor(asd_spectrum, asd_wavelengths_nm, sensor_cw_nm, sensor_fwhm_nm):
        valid = ~np.isnan(asd_spectrum) & ~np.isnan(asd_wavelengths_nm)
        asd_spectrum = asd_spectrum[valid]
        asd_wavelengths_nm = asd_wavelengths_nm[valid]
        diff = sensor_cw_nm[:, None] - asd_wavelengths_nm[None, :]
        srf = np.exp(-4.0 * np.log(2.0) * (diff / sensor_fwhm_nm[:, None])**2)
        norm = np.sum(srf, axis=1)
        norm[norm == 0] = 1.0
        return np.sum(srf * asd_spectrum[None, :], axis=1) / norm

SPLIB07_ASD_DIR = r"C:\satelliteImagery\ground_truth\splib07\ASCIIdata\ASCIIdata_splib07b_cvASD"
NICKS_LEAVES_DIR = r"C:\satelliteImagery\ground_truth\NickLeaves_20261001"

# --- Configuration Defaults ---
DEFAULT_H5_PATH = "C:/satelliteImagery/hyspex/20250918_GE18/dual/GE18_55m_01_Mjolnir_float32_geo_VS620.h5"
DEFAULT_MAP_X = 1059
DEFAULT_MAP_Y = 258
DEFAULT_TILE_SIZE = 15
DEFAULT_NUM_ENDMEMBERS = 10
DEFAULT_MATERIAL = "CALTARGET:big_white_panel"
DEFAULT_MATERIAL_2 = None
DEFAULT_LAYOUT = "uniform"
DEFAULT_SNR = 200
DEFAULT_SHOT_NOISE_FRACTION = 0.70
DEFAULT_SHADE_VARIABILITY = 0.03
DEFAULT_SENSOR = "cvHySpexMjolnir"
DEFAULT_OUTPUT_DIR = "C:/satelliteImagery/simulated/hyspex_comparison"
DEFAULT_SEED = 42

# Accessible Okabe-Ito color cycle
try:
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.colormaps['okabe_ito'].colors)
except Exception:
    pass


def make_gapped_spectral_plot_arrays(wv, data, gap_threshold_nm=20.0):
    """
    Inserts a NaN entry at atmospheric water absorption gaps (> 20 nm channel spacing)
    so matplotlib breaks lines cleanly without connecting across the omitted intervals.
    Supports 1D (bands,) and 2D (bands, channels/pixels/endmembers) data arrays.
    """
    diffs = np.diff(wv)
    gap_indices = np.where(diffs > gap_threshold_nm)[0]
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


def retrieve_real_tile(h5_path, map_x, map_y, tile_size=15, num_endmembers=10, context_size=150):
    """
    Retrieves the tile, context visual, and complexity metrics from the real HySpex Ortho Stack HDF5 file.
    
    Returns:
        dict containing:
            'tile_data': (bands, H, W) normalized surface reflectance
            'ortho_vis': (H, W, C) RGB visual chunk for 15x15 tile
            'context_vis': (H_c, W_c, C) RGB visual chunk for context_size x context_size window
            'context_extent': [x_min, x_max, y_max, y_min] map coordinates for context imshow
            'tile_bbox': (x_start, y_start, tile_size, tile_size)
            'wavelengths': (bands,) wavelengths in nm
            'wavelengths_microns': (bands,) wavelengths in microns
            'precomputed_heights': (num_endmembers,) from H5 if available
            'precomputed_noise_heights': dict of noise sources ('ssd', 'mnf')
            'computed_heights': (num_endmembers,) freshly computed from tile
            'computed_volumes_linear': (num_endmembers,) linear volume
            'computed_volumes_log': (num_endmembers,) ln(volume)
            'computed_adj_volumes': (num_endmembers,) adjusted volume
            'computed_derivative': (num_endmembers,) -dh
            'endmembers': (bands, num_endmembers)
            'indices': (num_endmembers,) 1D pixel indices
            'coords_2d': (r_tile, c_tile) coordinates of selected endmembers
    """
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"HySpex HDF5 file not found: {h5_path}")

    print(f"\n--- Loading Real HySpex Data ---")
    print(f"File: {h5_path}")
    print(f"Center Coordinate: (map_x={map_x}, map_y={map_y})")
    print(f"Tile Size: {tile_size}x{tile_size}")
    print(f"Context Window Size: {context_size}x{context_size}")

    offset = tile_size // 2
    y_start = map_y - offset
    y_end = map_y + offset + 1
    x_start = map_x - offset
    x_end = map_x + offset + 1

    c_offset = context_size // 2
    y_c_start = map_y - c_offset
    y_c_end = map_y + c_offset + 1
    x_c_start = map_x - c_offset
    x_c_end = map_x + c_offset + 1

    with h5py.File(h5_path, 'r') as f:
        grp_path = "HDFEOS/GRIDS/HYSPEX/Data Fields"
        if grp_path not in f:
            raise KeyError(f"Group '{grp_path}' not found in HDF5 file.")
        grp = f[grp_path]

        # 1. Surface Reflectance
        sr_ds = grp['surface_reflectance']
        scale_factor = float(sr_ds.attrs.get('Scale_Factor', [1.0])[0])
        fill_val = sr_ds.fillvalue
        wavelengths_nm = np.array(sr_ds.attrs['wavelengths'], dtype=np.float32)
        wavelengths_microns = wavelengths_nm / 1000.0 if np.nanmax(wavelengths_nm) > 100.0 else wavelengths_nm.copy()

        lines = sr_ds.shape[2]
        samples = sr_ds.shape[3]

        # Extract spatial tile (1, bands, lines, samples)
        raw_tile = sr_ds[0, :, y_start:y_end, x_start:x_end].astype(np.float64)
        bands, h, w = raw_tile.shape
        if h != tile_size or w != tile_size:
            raise ValueError(f"Extracted tile shape ({h}, {w}) does not match requested tile size ({tile_size}, {tile_size}).")

        # 2. Ortho Vis Chunk & Context
        vis_chunk = None
        context_vis = None
        y_c0 = max(0, y_c_start)
        y_c1 = min(lines, y_c_end)
        x_c0 = max(0, x_c_start)
        x_c1 = min(samples, x_c_end)

        if 'ortho_vis' in grp:
            ov_ds = grp['ortho_vis']
            ov_sub = ov_ds[0, :, y_c0:y_c1, x_c0:x_c1]
            if ov_sub.ndim == 3 and ov_sub.shape[0] in [3, 4]:
                context_vis = np.transpose(ov_sub, (1, 2, 0))
            y_r0 = y_start - y_c0
            y_r1 = y_end - y_c0
            x_r0 = x_start - x_c0
            x_r1 = x_end - x_c0
            vis_chunk = context_vis[y_r0:y_r1, x_r0:x_r1]

        # 3. Precomputed Metrics from H5 (from process_metrics.py)
        num_slots = num_endmembers + 1
        precomputed_heights = None
        h_name = f'heights_map_{tile_size}x{tile_size}'
        if h_name in grp:
            h_data = grp[h_name][0, map_y, map_x, :]
            precomputed_heights = h_data[:num_slots]

        precomputed_noise_heights = {}
        for src in ['ssd', 'mnf', 'mlr', 'dct']:
            n_name = f'noise_heights_map_{src}_{tile_size}x{tile_size}'
            if n_name in grp:
                ndata = grp[n_name][0, map_y, map_x, :]
                precomputed_noise_heights[src] = ndata[:num_slots]

    # Data integrity: Completely exclude telluric water absorption bands (1.35-1.45 um and 1.80-2.00 um)
    # and any unmeasured / fill value (-1.0) bands so they are NEVER used in calculations.
    missing_bands_mask = np.all(raw_tile == fill_val, axis=(1, 2))
    if fill_val is not None:
        missing_bands_mask |= np.all(raw_tile < 0, axis=(1, 2))
    is_water_band = ((wavelengths_nm >= 1350.0) & (wavelengths_nm <= 1450.0)) | ((wavelengths_nm >= 1800.0) & (wavelengths_nm <= 2000.0)) | (wavelengths_nm > 2400.0)
    invalid_bands_mask = missing_bands_mask | is_water_band
    valid_bands_mask = ~invalid_bands_mask

    # Slicing out invalid water absorption bands entirely so they are NEVER used in calculations
    tile_normalized = (raw_tile / scale_factor)[valid_bands_mask, :, :].astype(np.float64)
    wavelengths_nm = wavelengths_nm[valid_bands_mask]
    wavelengths_microns = wavelengths_microns[valid_bands_mask]
    bands = len(wavelengths_nm)

    # 4. Compute MaxD Parallelotope metrics on the valid real tile bands
    pixels = tile_normalized.reshape(bands, -1)
    em, vol_log, heights, indices = maximumDistance_parallelotopeVolumes(pixels, num_slots)

    # Linear volume: safe cumprod matching hyspex_viewer.py
    safe_h = np.where(heights <= 0, 1.0, heights)
    volumes_linear = np.cumprod(safe_h)
    
    # Adjusted volume: (V^(1/k)) / sqrt(k)
    k_indices = np.arange(1, num_slots + 1)
    adj_volumes = (volumes_linear ** (1.0 / k_indices)) / np.sqrt(k_indices)
    
    # Orthogonal Subspace Distance decay rate (derivative of actual heights h_1..h_k)
    derivative = np.full_like(heights, np.nan)
    if len(heights) > 2:
        derivative[1:] = -np.gradient(heights[1:])

    r_tile, c_tile = np.unravel_index(indices, (tile_size, tile_size))

    return {
        'tile_data': tile_normalized,
        'tile_clean': tile_normalized,
        'ortho_vis': vis_chunk,
        'context_vis': context_vis,
        'context_extent': [x_c0, x_c1, y_c1, y_c0],
        'tile_bbox': (x_start, y_start, tile_size, tile_size),
        'context_size': context_size,
        'wavelengths_nm': wavelengths_nm,
        'wavelengths_microns': wavelengths_microns,
        'precomputed_heights': precomputed_heights,
        'precomputed_noise_heights': precomputed_noise_heights,
        'computed_heights': heights,
        'computed_volumes_log': vol_log,
        'computed_volumes_linear': volumes_linear,
        'computed_adj_volumes': adj_volumes,
        'computed_derivative': derivative,
        'endmembers': em,
        'indices': indices,
        'coords_2d': (r_tile, c_tile),
        'scale_factor': scale_factor,
        'fill_val': fill_val
    }


def load_material_spectrum(material_name, sensor_name=DEFAULT_SENSOR):
    """
    Loads and resamples a material spectrum to match the sensor bandpass,
    omitting atmospheric water absorption intervals (1.35-1.45, 1.80-2.00, >2.42 um).
    Supports:
      - 'CALTARGET:<name>' or 'CALIBRATION:<name>'
      - 'NICKLEAVES:<name>'
      - 'splib07:<name>' or 'USGS:<name>'
    """
    if material_name.startswith(("CALTARGET:", "CALIBRATION:")):
        filename_str = material_name.split(":", 1)[1]
        csv_path = os.path.join(CALIBRATION_TARGETS_DIR, f"{sensor_name}_{filename_str}.csv")
        if not os.path.exists(csv_path):
            csv_path = os.path.join(CALIBRATION_TARGETS_DIR, f"cvHySpex_{filename_str}.csv")
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"Calibration target CSV not found: {csv_path}")
        print(f"Loaded Calibration Spectrum from: {csv_path}")
        df = pd.read_csv(csv_path)
        spec = df['DHR_mean'].values.astype(np.float32)
        if 'wavelength' in df.columns:
            wv = df['wavelength'].values.astype(np.float32)
            wavelengths_nm = wv if np.nanmax(wv) > 100.0 else wv * 1000.0
        else:
            wv_microns = load_asd_txt(cvHySpexMjolnir_WV_FILE)
            wavelengths_nm = wv_microns * 1000.0
    elif material_name.startswith("NICKLEAVES:"):
        filename_str = material_name.split(":", 1)[1]
        csv_path = os.path.join(NICKS_LEAVES_DIR, f"{sensor_name}_{filename_str}.csv")
        if not os.path.exists(csv_path):
            csv_path = os.path.join(NICKS_LEAVES_DIR, f"cvHySpex_{filename_str}.csv")
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"Leaf CSV not found: {csv_path}")
        print(f"Loaded Leaf Spectrum from: {csv_path}")
        df = pd.read_csv(csv_path)
        spec = df['DHR_mean'].values.astype(np.float32)
        wv = df['wavelength'].values.astype(np.float32)
        wavelengths_nm = wv if np.nanmax(wv) > 100.0 else wv * 1000.0
    elif material_name.startswith(("splib07:", "USGS:")):
        query = material_name.split(":", 1)[1]
        clean_q = query.lower().replace(" ", "_")
        matches = glob.glob(os.path.join(SPLIB07_ASD_DIR, "**", f"*{clean_q}*.txt"), recursive=True)
        if not matches:
            matches = glob.glob(os.path.join(SPLIB07_ASD_DIR, "ChapterA_ArtificialMaterials", f"*{clean_q}*.txt"))
        if not matches:
            raise FileNotFoundError(f"USGS splib07 material matching '{query}' not found in {SPLIB07_ASD_DIR}")
        target_file = matches[0]
        print(f"Loaded USGS splib07 Spectrum from: {target_file}")
        spectrum_asd = load_asd_txt(target_file)
        asd_wv_nm = load_asd_txt(cvASD_WV_FILE) * 1000.0
        sensor_cw_nm = load_asd_txt(cvHySpexMjolnir_WV_FILE) * 1000.0
        sensor_fwhm_nm = load_asd_txt(cvHySpexMjolnir_FWHM_FILE) * 1000.0
        spec = resample_asd_to_sensor(spectrum_asd, asd_wv_nm, sensor_cw_nm, sensor_fwhm_nm).astype(np.float32)
        wavelengths_nm = sensor_cw_nm
    else:
        raise ValueError(f"Unsupported material format: {material_name}. Expected 'CALTARGET:...', 'NICKLEAVES:...', or 'splib07:...'")

    # Ensure water absorption bands (1.35-1.45 um, 1.80-2.00 um) and noisy SWIR (>2.42 um) are NEVER used
    is_water_band = ((wavelengths_nm >= 1350.0) & (wavelengths_nm <= 1450.0)) | ((wavelengths_nm >= 1800.0) & (wavelengths_nm <= 2000.0)) | (wavelengths_nm > 2420.0)
    if np.any(is_water_band):
        valid_bands = ~is_water_band
        spec = spec[valid_bands]
        wavelengths_nm = wavelengths_nm[valid_bands]

    wavelengths_microns = wavelengths_nm / 1000.0
    return wavelengths_nm, wavelengths_microns, spec


def generate_simulated_tile(sensor_name=DEFAULT_SENSOR,
                            material_name=DEFAULT_MATERIAL,
                            material_2_name=DEFAULT_MATERIAL_2,
                            material_3_name=None,
                            layout=DEFAULT_LAYOUT,
                            tile_size=15,
                            snr=DEFAULT_SNR,
                            num_endmembers=10,
                            shot_noise_fraction=0.70,
                            shade_variability=DEFAULT_SHADE_VARIABILITY,
                            seed=42):
    """
    Generates a simulated tile for the HySpex Mjolnir sensor using methods from
    `simulate_inset_squares.py`.
    Supports single material ('uniform') and multi-material layouts ('split_v', 'split_h', 'edge_with_border').
    """
    print(f"\n--- Generating Simulated Sensor Data ---")
    print(f"Sensor: {sensor_name}")
    print(f"Layout: {layout}")
    print(f"Primary Material: {material_name}")
    if material_2_name:
        print(f"Secondary Material: {material_2_name}")
    if material_3_name:
        print(f"Tertiary Material: {material_3_name}")
    print(f"Tile Size: {tile_size}x{tile_size}")
    print(f"Target Linear SNR: {snr} (Shot noise fraction: {shot_noise_fraction})")
    if shade_variability > 0:
        print(f"Magnitude Variability: {shade_variability * 100:.1f}%")

    # 1. Load Material Spectra
    wavelengths_nm, wavelengths_microns, spec1 = load_material_spectrum(material_name, sensor_name)
    spec2 = None
    if material_2_name:
        _, _, spec2 = load_material_spectrum(material_2_name, sensor_name)
    spec3 = None
    if material_3_name:
        _, _, spec3 = load_material_spectrum(material_3_name, sensor_name)

    num_bands = len(spec1)

    # 2. Build Spatial Tile
    clean_tile = np.zeros((num_bands, tile_size, tile_size), dtype=np.float32)
    if layout == "split_v" and spec2 is not None:
        half = tile_size // 2
        clean_tile[:, :, :half] = spec1[:, None, None]
        # 1-pixel mixed boundary to model PSF mixing
        clean_tile[:, :, half] = 0.5 * spec1[:, None] + 0.5 * spec2[:, None]
        clean_tile[:, :, half+1:] = spec2[:, None, None]
    elif layout == "split_h" and spec2 is not None:
        half = tile_size // 2
        clean_tile[:, :half, :] = spec1[:, None, None]
        clean_tile[:, half, :] = 0.5 * spec1[:, None] + 0.5 * spec2[:, None]
        clean_tile[:, half+1:, :] = spec2[:, None, None]
    elif layout == "split_diag" and spec2 is not None:
        # Diagonal split matching real panel edge orientation at (1106, 326)
        # Top-left = spec1 (White Panel), Bottom-right = spec2 (Asphalt)
        for r in range(tile_size):
            for c in range(tile_size):
                val = (r + c) - (tile_size - 1)
                if val < -1:
                    clean_tile[:, r, c] = spec1
                elif val > 1:
                    clean_tile[:, r, c] = spec2
                else:
                    clean_tile[:, r, c] = 0.5 * spec1 + 0.5 * spec2
    elif layout == "edge_with_border" and spec2 is not None and spec3 is not None:
        half = tile_size // 2
        clean_tile[:, :, :half-1] = spec1[:, None, None]
        clean_tile[:, :, half-1:half+1] = spec3[:, None, None]
        clean_tile[:, :, half+1:] = spec2[:, None, None]
    else:
        clean_tile[:, :, :] = spec1[:, None, None]

    shade_map = None
    if shade_variability > 0:
        if seed is not None:
            np.random.seed(seed)
        # Match method used in simulated-data-test\spectral_complexity_vs_pca.py
        # shading_map = np.random.uniform(0.97, 1.0, (tile_size, tile_size)) for 3% variability
        shade_map = np.random.uniform(1.0 - shade_variability, 1.0, (tile_size, tile_size)).astype(np.float32)
        clean_tile = clean_tile * shade_map[None, :, :]

    # 3. Inject wavelength-scaled noise (Poisson-Gaussian model)
    noisy_tile, noise_array = add_wavelength_scaled_noise(
        clean_tile,
        snr=snr,
        shot_noise_fraction=shot_noise_fraction,
        seed=seed
    )

    # 4. Generate True Color RGBA Image
    try:
        rgba = generate_rgba_from_hsi(noisy_tile, wavelengths_nm)
    except Exception as e:
        print(f"Warning: generate_rgba_from_hsi failed ({e}), falling back to 3-band composite.")
        b_idx = [min(50, num_bands - 1), min(100, num_bands - 1), min(150, num_bands - 1)]
        rgb = noisy_tile[b_idx, :, :].transpose((1, 2, 0))
        rgb = np.clip(rgb / np.percentile(rgb, 98), 0, 1)
        rgba = (rgb * 255).astype(np.uint8)
        alpha = np.full((tile_size, tile_size, 1), 255, dtype=np.uint8)
        rgba = np.concatenate([rgba, alpha], axis=-1)

    # 5. Extract Endmembers and Spectral Complexity Metrics with Noise
    num_slots = num_endmembers + 1
    pixels = noisy_tile.reshape(num_bands, -1)
    noise_pixels = noise_array.reshape(num_bands, -1)

    em, vol_log, heights, indices, noise_heights, noise_norms_mean = maximumDistance_parallelotopeVolumes_noise(
        pixels,
        num_slots,
        noise_pixels=noise_pixels
    )

    safe_h = np.where(heights <= 0, 1.0, heights)
    volumes_linear = np.cumprod(safe_h)
    k_indices = np.arange(1, num_slots + 1)
    adj_volumes = (volumes_linear ** (1.0 / k_indices)) / np.sqrt(k_indices)

    noise_volumes_log = np.full_like(noise_heights, -np.inf)
    if noise_heights[1] > 0:
        noise_volumes_log[1] = np.log(noise_heights[1])
        for j in range(2, len(noise_heights)):
            if noise_heights[j] > 0:
                noise_volumes_log[j] = noise_volumes_log[j - 1] + np.log(noise_heights[j])
    noise_volumes_linear = np.exp(np.where(np.isinf(noise_volumes_log), np.nan, noise_volumes_log))
    noise_volumes_linear[0] = 0.0

    r_tile, c_tile = np.unravel_index(indices, (tile_size, tile_size))

    # Orthogonal Subspace Distance decay rate (derivative of actual heights h_1..h_k)
    height_derivative = np.full_like(heights, np.nan)
    if len(heights) > 2:
        height_derivative[1:] = -np.gradient(heights[1:])

    return {
        'clean_tile': clean_tile,
        'noisy_tile': noisy_tile,
        'noise_array': noise_array,
        'rgba_image': rgba,
        'wavelengths_microns': wavelengths_microns,
        'wavelengths_nm': wavelengths_nm,
        'material_spectrum': spec1,
        'material_2_spectrum': spec2,
        'material_3_spectrum': spec3,
        'material_name': material_name,
        'material_2_name': material_2_name,
        'material_3_name': material_3_name,
        'layout': layout,
        'computed_heights': heights,
        'computed_height_derivative': height_derivative,
        'computed_derivative': height_derivative,
        'computed_volumes_log': vol_log,
        'computed_volumes_linear': volumes_linear,
        'computed_adj_volumes': adj_volumes,
        'noise_heights': noise_heights,
        'noise_norms_mean': noise_norms_mean,
        'noise_volumes_linear': noise_volumes_linear,
        'noise_volumes_log': noise_volumes_log,
        'endmembers': em,
        'indices': indices,
        'coords_2d': (r_tile, c_tile),
        'snr': snr,
        'shade_map': shade_map,
        'shade_variability': shade_variability
    }


def display_comparison_table(real_res, sim_res, num_endmembers=10):
    """
    Prints a detailed side-by-side comparison table of spectral complexity
    Orthogonal Subspace Distance and volume metrics for both real and simulated data.
    """
    snr = sim_res.get('snr', 100)
    snr_str = f"{int(snr) if snr == int(snr) else snr}"
    shade_var = sim_res.get('shade_variability', 0.0)
    shade_str = f", Shade={int(round(shade_var * 100))}%" if shade_var > 0 else ""
    sim_header = f"SIMULATED DATA (SNR={snr_str}{shade_str})"

    print("\n" + "=" * 115)
    print("SPECTRAL COMPLEXITY COMPARISON: REAL HYSPEX TILE vs. SIMULATED CALIBRATION TARGET")
    print("=" * 115)
    print(f"{'EM':<10} | {'REAL DATA':^40} | {sim_header:^60}")
    print(f"{'k':<10} | {'Height':^10} {'Volume (Lin)':^14} {'ln(Vol)':^8} {'H5 Noise':^8} | "
          f"{'Height':^10} {'Sim Noise H':^12} {'Mean Noise':^11} {'Volume (Lin)':^14} {'ln(Vol)':^8}")
    print("-" * 115)

    real_h = real_res['computed_heights']
    real_v_lin = real_res['computed_volumes_linear']
    real_v_log = real_res['computed_volumes_log']
    
    # Check if SSD or MNF noise is available from H5
    real_n = real_res['precomputed_noise_heights'].get('ssd', real_res['precomputed_noise_heights'].get('mnf', None))

    sim_h = sim_res['computed_heights']
    sim_nh = sim_res['noise_heights']
    sim_nm = sim_res['noise_norms_mean']
    sim_v_lin = sim_res['computed_volumes_linear']
    sim_v_log = sim_res['computed_volumes_log']

    num_rows = min(len(real_h), len(sim_h))
    for k in range(num_rows):
        idx_str = f"EM {k}" if k > 0 else "0 (origin)"
        
        # Real strings
        rh_str = f"{real_h[k]:.6f}"
        rv_lin_str = f"{real_v_lin[k]:.3e}" if k > 0 else "1.000"
        rv_log_str = f"{real_v_log[k]:.2f}" if not np.isinf(real_v_log[k]) else "-inf"
        rn_str = f"{real_n[k]:.4f}" if (real_n is not None and k < len(real_n) and real_n[k] > 0) else "N/A"

        # Sim strings
        sh_str = f"{sim_h[k]:.6f}"
        snh_str = f"{sim_nh[k]:.6f}" if sim_nh[k] > 0 else "0.000000"
        snm_str = f"{sim_nm[k]:.6f}" if sim_nm[k] > 0 else "0.000000"
        sv_lin_str = f"{sim_v_lin[k]:.3e}" if k > 0 else "0.000"
        sv_log_str = f"{sim_v_log[k]:.2f}" if not np.isinf(sim_v_log[k]) else "-inf"

        print(f"{idx_str:<10} | {rh_str:<10} {rv_lin_str:<14} {rv_log_str:<8} {rn_str:<8} | "
              f"{sh_str:<10} {snh_str:<12} {snm_str:<11} {sv_lin_str:<14} {sv_log_str:<8}")

    print("-" * 115)
    print("Note: Index 0 is the localized origin (min-norm pixel). Endmembers 1 to 10 are the extracted vertices.")
    print("=" * 115 + "\n")


def plot_comparison(real_res, sim_res, map_x, map_y, tile_size=15, num_endmembers=10, output_dir=DEFAULT_OUTPUT_DIR, show_plot=True):
    """
    Plots a multi-panel visual and analytical comparison between real and simulated data.
    Organized into a 7-panel layout:
      Top Row (4 panels):
        - Panel 1: Flightline Context Visual (150x150) with Red Bounding Box around 15x15 Tile
        - Panel 2: Real Data Tile Zoom (15x15) with Extracted Endmembers
        - Panel 3: Simulated Tile RGBA (15x15) with Extracted Endmembers
        - Panel 4: Spectral Reflectance Comparison (Water Absorption Bands Omitted)
      Bottom Row (3 panels):
        - Panel 5: Orthogonal Heights (h_k) Comparison
        - Panel 6: Complexity Volume (cumprod h_k) & Adjusted Volume Comparison
        - Panel 7: Extracted Endmember Spectra (k=1..4)
    """
    os.makedirs(output_dir, exist_ok=True)
    fig = plt.figure(figsize=(22, 11))
    fig.canvas.manager.set_window_title("HySpex Real vs. Simulated Complexity Comparison")

    snr = sim_res.get('snr', 100)
    snr_str = f"{int(snr) if snr == int(snr) else snr}"
    shade_var = sim_res.get('shade_variability', 0.0)
    shade_suffix = f", Magnitude Variability={int(round(shade_var * 100))}%" if shade_var > 0 else ""

    # Nested GridSpec with independent horizontal spacing per row:
    # Row 1: 4 panels with customized width_ratios (shrink context visual, expand spectral plot)
    # Row 2: 3 panels with generous wspace=0.30 to prevent y-axis and twin-axis label overlap
    gs_main = fig.add_gridspec(2, 1, height_ratios=[1, 1.1], hspace=0.3)
    gs_top = gs_main[0].subgridspec(1, 4, width_ratios=[0.65, 1.0, 1.0, 1.85], wspace=0.22)
    gs_bot = gs_main[1].subgridspec(1, 3, wspace=0.3)

    # ---------------------------------------------------------
    # Panel 1: Flightline Context Visual (150x150) with Red Bounding Box around 15x15 Tile
    # ---------------------------------------------------------
    ax_context = fig.add_subplot(gs_top[0, 0])
    c_size = real_res.get('context_size', 150)
    if real_res.get('context_vis') is not None:
        ext = real_res.get('context_extent')
        ax_context.imshow(real_res['context_vis'], extent=ext)
        x_start, y_start, tile_w, tile_h = real_res['tile_bbox']
        rect = patches.Rectangle((x_start, y_start), tile_w, tile_h, linewidth=2.0, edgecolor='red', facecolor='none')
        ax_context.add_patch(rect)
        #ax_context.plot(map_x, map_y, marker='+', color='red', markersize=8, markeredgewidth=1.5)
        #ax_context.text(x_start + tile_w / 2.0, y_start - 2, f"{tile_w}x{tile_h} Tile",
        #                color='red', fontsize=8, fontweight='bold', ha='center', va='bottom')
        ax_context.set_title(f"HySpex Source", fontsize=11, fontweight='bold')
        ax_context.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
        #ax_context.set_xlabel("Source Line")
        #ax_context.set_ylabel("Source Column")
    else:
        ax_context.text(0.5, 0.5, "Context Visual\nNot Available", ha='center', va='center', transform=ax_context.transAxes)
        ax_context.set_title(f"HySpex Source", fontsize=11, fontweight='bold')

    # ---------------------------------------------------------
    # Panel 2: Real Tile Visual (15x15 Zoom with Extracted EMs)
    # ---------------------------------------------------------
    ax_real_img = fig.add_subplot(gs_top[0, 1])
    if real_res['ortho_vis'] is not None:
        ax_real_img.imshow(real_res['ortho_vis'])
    else:
        # Pseudo-RGB from clean tile
        rgb = real_res['tile_clean'][[50, 100, 150], :, :].transpose((1, 2, 0))
        rgb = np.clip(rgb / np.percentile(rgb, 98), 0, 1)
        ax_real_img.imshow(rgb)

    r_real, c_real = real_res['coords_2d']
    ax_real_img.scatter(c_real, r_real, facecolors='none', edgecolors='red', s=70, linewidth=1.5, label='Extracted EMs')
    for i, (r, c) in enumerate(zip(r_real, c_real)):
        ax_real_img.text(c + 0.3, r - 0.3, str(i + 1), color='yellow', fontsize=9, fontweight='bold')
    ax_real_img.set_title(f"Real Data: {tile_size}x{tile_size} Tile Zoom", fontsize=11, fontweight='bold')
    ax_real_img.set_xlabel("Local Column")
    ax_real_img.set_ylabel("Local Row")
    ax_real_img.legend(loc='lower right', fontsize='small')

    # ---------------------------------------------------------
    # Panel 3: Simulated Tile RGBA
    # ---------------------------------------------------------
    ax_sim_img = fig.add_subplot(gs_top[0, 2])
    ax_sim_img.imshow(sim_res['rgba_image'])
    r_sim, c_sim = sim_res['coords_2d']
    ax_sim_img.scatter(c_sim, r_sim, facecolors='none', edgecolors='cyan', s=70, linewidth=1.5, label='Extracted EMs')
    for i, (r, c) in enumerate(zip(r_sim, c_sim)):
        ax_sim_img.text(c + 0.3, r - 0.3, str(i + 1), color='black', fontsize=9, fontweight='bold')

    mat1_disp = sim_res.get('material_name', 'Material 1').split(':', 1)[-1].replace('_', ' ')
    mat2_disp = sim_res.get('material_2_name', '').split(':', 1)[-1].replace('_', ' ') if sim_res.get('material_2_name') else None
    layout = sim_res.get('layout', 'uniform')

    if layout in ['split_v', 'split_h', 'split_diag', 'edge_with_border'] and mat2_disp:
        sim_title = f"Simulated Tile \nMaterials: {mat1_disp} / {mat2_disp}"
    else:
        sim_title = f"Simulated Tile\nMaterial: {mat1_disp}"
    ax_sim_img.set_title(sim_title, fontsize=11, fontweight='bold')
    ax_sim_img.set_xlabel("Local Column")
    ax_sim_img.set_ylabel("Local Row")
    ax_sim_img.legend(loc='lower right', fontsize='small')

    # ---------------------------------------------------------
    # Panel 4: Spectral Profiles Comparison (Water Absorption Bands Omitted)
    # ---------------------------------------------------------
    ax_spec = fig.add_subplot(gs_top[0, 3])
    wv_real = real_res['wavelengths_nm']
    real_pix = real_res['tile_clean'].reshape(len(wv_real), -1)

    wv_sim = sim_res['wavelengths_nm']
    mat_spec = sim_res['material_spectrum']
    sim_noisy_pix = sim_res['noisy_tile'].reshape(len(wv_sim), -1)

    wv_real_g, real_pix_g = make_gapped_spectral_plot_arrays(wv_real, real_pix)
    wv_sim_g, mat_spec_g = make_gapped_spectral_plot_arrays(wv_sim, mat_spec)
    _, sim_noisy_pix_g = make_gapped_spectral_plot_arrays(wv_sim, sim_noisy_pix)

    # Highlight water absorption regions with shaded spans (do not plot any values in these windows)
    ax_spec.axvspan(1350, 1450, color='#e0e0e0', alpha=0.55)#, label='Atmospheric H₂O')
    ax_spec.axvspan(1800, 2000, color='#e0e0e0', alpha=0.55)
    ax_spec.axvspan(2420, 2500, color='#e0e0e0', alpha=0.55)

    # Plot sample of real pixels (gapped arrays prevent drawing lines across water absorption intervals)
    #ax_spec.plot(wv_real_g, real_pix_g[:, ::15], color='royalblue', alpha=0.35, linewidth=0.8)
    
    center_idx = (tile_size // 2) * tile_size + (tile_size // 2)
    if layout in ['split_v', 'split_h', 'split_diag', 'edge_with_border'] and mat2_disp:
        if layout == "split_diag":
            left_idx = 2 * tile_size + 2  # Top-left (White Panel)
            right_idx = (tile_size - 3) * tile_size + (tile_size - 3)  # Bottom-right (Asphalt)
            p1_lbl = "Real Top-Left Material"
            p2_lbl = "Real Bottom-Right Material"
        else:
            left_idx = (tile_size // 2) * tile_size + 2
            right_idx = (tile_size // 2) * tile_size + (tile_size - 3)
            p1_lbl = "Real Material 1"
        corner_h = max(1, tile_size // 3)
        corner_w = max(1, tile_size // 3)
        ul_indices = [r * tile_size + c for r in range(corner_h) for c in range(corner_w)]
        lr_indices = [r * tile_size + c for r in range(tile_size - corner_h, tile_size) for c in range(tile_size - corner_w, tile_size)]

        # Diagonal boundary corridor from lower-left to upper-right (+/- 1 pixel)
        if layout == "split_diag":
            diag_indices = [
                r * tile_size + c
                for r in range(tile_size)
                for c in range(tile_size)
                if abs((r + c) - (tile_size - 1)) <= 1
            ]
        elif layout == "split_v":
            half = tile_size // 2
            diag_indices = [
                r * tile_size + c
                for r in range(tile_size)
                for c in range(tile_size)
                if abs(c - half) <= 1
            ]
        elif layout == "split_h":
            half = tile_size // 2
            diag_indices = [
                r * tile_size + c
                for r in range(tile_size)
                for c in range(tile_size)
                if abs(r - half) <= 1
            ]
        else:
            diag_indices = [
                r * tile_size + c
                for r in range(tile_size)
                for c in range(tile_size)
                if abs((r + c) - (tile_size - 1)) <= 1
            ]

        ax_spec.plot(wv_real_g, real_pix_g[:, left_idx], color='xkcd:sky blue', linewidth=1, label=p1_lbl)
        #ax_spec.plot(wv_real_g, real_pix_g[:, ul_indices], color='xkcd:sky blue', linewidth=1, alpha=0.25)
        ax_spec.plot(wv_real_g, real_pix_g[:, right_idx], color='navy', linewidth=1, label=p2_lbl)
        #ax_spec.plot(wv_real_g, real_pix_g[:, lr_indices], color='navy', linewidth=1, alpha=0.25)
        ax_spec.plot(wv_real_g, real_pix_g[:, center_idx], color='cornflowerblue', linewidth=1, label='Real Center Pixel')
        #ax_spec.plot(wv_real_g, real_pix_g[:, diag_indices], color='cornflowerblue', linewidth=1, alpha=0.25)

        ax_spec.plot(wv_sim_g, sim_noisy_pix_g[:, left_idx], color='indianred', linewidth=1.2, label='Simulated Top-Left Pixel')
        ax_spec.plot(wv_sim_g, sim_noisy_pix_g[:, right_idx], color='darkred', linewidth=1.2, label='Simulated Bottom-Right Pixel')
        ax_spec.plot(wv_sim_g, sim_noisy_pix_g[:, center_idx], color='red', linewidth=1.2, label='Simulated Center Pixel')
        #ax_spec.plot(wv_sim_g, mat_spec_g, color='green', linewidth=1.5, label=f'Ground Truth: {mat1_disp}')
        #if sim_res.get('material_2_spectrum') is not None:
        #    _, mat_spec_2_g = make_gapped_spectral_plot_arrays(wv_sim, sim_res['material_2_spectrum'])
        #    ax_spec.plot(wv_sim_g, mat_spec_2_g, color='darkorange', linewidth=1.5, label=f'Ground Truth: {mat2_disp}')
        ax_spec.set_ylim(0, 0.6)
        ax_spec.legend(loc='lower right', fontsize='small',ncol=2)
    else:
        ax_spec.plot(wv_real_g, real_pix_g[:, 0], color='blue', linewidth=1, label='Real Spectra')
        ax_spec.plot(wv_sim_g, sim_noisy_pix_g[:, center_idx], color='red', linewidth=1, label='Simulated Spectra')
        ax_spec.plot(wv_sim_g, sim_noisy_pix_g[:, 1:], color='red', linewidth=0.5, alpha=0.05)
        ax_spec.plot(wv_real_g, real_pix_g[:, 1:], color='blue', linewidth=0.5, alpha=0.05)
        ax_spec.plot(wv_sim_g, mat_spec_g, color='green', linewidth=2.0, label=f'Ground Truth Spectrum')
        ax_spec.legend(loc='lower left', fontsize='small')

    ax_spec.set_title("Real vs Simulated Spectra", fontsize=11, fontweight='bold')
    ax_spec.set_xlabel("Wavelength (nm)")
    ax_spec.set_ylabel("Reflectance")
    ax_spec.grid(True, linestyle='--', alpha=0.6)
    

    # ---------------------------------------------------------
    # Panel 5: Spectral Complexity Heights Comparison
    # ---------------------------------------------------------
    ax_h = fig.add_subplot(gs_bot[0, 0])
    n_pts = len(real_res['computed_heights'])
    em_idx = np.arange(1, n_pts + 1)

    # Real Heights
    ax_h.plot(em_idx, real_res['computed_heights'], marker='o', linestyle='-', color='blue', linewidth=1.8, label=r'Real Distance ($h_k$)')
    ax_h.plot(em_idx[1:], real_res['computed_derivative'][1:], marker='^', linestyle=':', color='blue', alpha=0.8, label=r'Real $-\Delta h_k$')

    # Real Noise from H5 (if available)
    #if 'ssd' in real_res['precomputed_noise_heights'] and real_res['precomputed_noise_heights']['ssd'] is not None:
    #    ax_h.plot(em_idx[:len(real_res['precomputed_noise_heights']['ssd'])], real_res['precomputed_noise_heights']['ssd'], marker='s', linestyle='--', color='purple', alpha=0.7, label='Real Noise (SSD)')
    #if 'mnf' in real_res['precomputed_noise_heights'] and real_res['precomputed_noise_heights']['mnf'] is not None:
    #    ax_h.plot(em_idx[:len(real_res['precomputed_noise_heights']['mnf'])], real_res['precomputed_noise_heights']['mnf'], marker='x', linestyle='--', color='darkcyan', alpha=0.7, label='Real Noise (MNF)')

    # Simulated Heights & Noise
    ax_h.plot(em_idx, sim_res['computed_heights'], marker='s', linestyle='-', color='crimson', linewidth=1.8, label=r'Simulated Distance ($h_k$)')
    if 'computed_height_derivative' in sim_res and sim_res['computed_height_derivative'] is not None:
        ax_h.plot(em_idx[1:], sim_res['computed_height_derivative'][1:], marker='v', linestyle=':', color='crimson', alpha=0.8, label=r'Simulated $-\Delta h_k$')
    elif 'computed_derivative' in sim_res and sim_res['computed_derivative'] is not None:
        ax_h.plot(em_idx[1:], sim_res['computed_derivative'][1:], marker='v', linestyle=':', color='crimson', alpha=0.8, label=r'Simulated $-\Delta h_k$')
    #ax_h.plot(em_idx, sim_res['noise_heights'], marker='x', linestyle='--', color='darkmagenta', linewidth=1.5, label='Simulated Noise Distance')
    ax_h.plot(em_idx[1:], sim_res['noise_norms_mean'][1:], marker='x', linestyle='--', color='#F76902', linewidth=1.2, label='Simulated Mean Noise')

    eps32 = float(np.finfo(np.float32).eps)
    ax_h.set_yscale('symlog', linthresh=1e-3)
    ax_h.set_ylim(bottom=0)
    ax_h.set_title("Orthogonal Subspace Distance ($h_k$)", fontsize=11, fontweight='bold')
    ax_h.set_xlabel("Endmember Index (k)")
    ax_h.set_ylabel(r"Orthogonal Subspace Distance ($h_k$)")
    ax_h.set_xticks(em_idx)
    ax_h.grid(True, linestyle='--', alpha=0.6)
    ax_h.legend(loc='best', fontsize='x-small')

    #secax_h = ax_h.secondary_xaxis('top', functions=(lambda x: x - 1, lambda x: x + 1))
    #secax_h.set_xlabel('Subspace Dimensionality')
    #secax_h.set_xticks(em_idx - 1)

    # ---------------------------------------------------------
    # Panel 6: Complexity Volume Comparison (SymLog / Log)
    # ---------------------------------------------------------
    ax_v = fig.add_subplot(gs_bot[0, 1])
    ax_v_twin = ax_v.twinx()

    # Volumes (Linear plotted on log scale)
    real_vol = real_res['computed_volumes_linear']
    sim_vol = sim_res['computed_volumes_linear']
    sim_noise_vol = sim_res['noise_volumes_linear']

    p1 = ax_v.plot(em_idx[1:], real_vol[1:], marker='o', linestyle='-', color='blue', linewidth=1.8, label='Real Volume')
    p2 = ax_v.plot(em_idx[1:], sim_vol[1:], marker='s', linestyle='-', color='crimson', linewidth=1.8, label=r'Simulated Volume ($\prod h_k$)')
    p3 = ax_v.plot(em_idx[1:], sim_noise_vol[1:], marker='x', linestyle='--', color='#F76902', linewidth=1.5, label='Simulated Noise Vol')

    ax_v.set_yscale('log')
    ax_v.set_title(r"Spectral Complexity Volume ($\prod h_k$) Comparison", fontsize=11, fontweight='bold')
    ax_v.set_xlabel("Endmember Index (k)")
    ax_v.set_ylabel(r"Spectral Complexity Volume ($\prod h_k$)")
    ax_v.set_xticks(em_idx[1:])
    ax_v.grid(True, linestyle='--', alpha=0.6)

    # Adjusted volume on twin axis: (V^(1/k)) / sqrt(k)
    #p4 = ax_v_twin.plot(em_idx[1:], real_res['computed_adj_volumes'][1:], marker='d', linestyle='-.', color='cyan', alpha=0.7, label='Real Adj Vol')
    #p5 = ax_v_twin.plot(em_idx[1:], sim_res['computed_adj_volumes'][1:], marker='v', linestyle='-.', color='coral', alpha=0.7, label='Sim Adj Vol')
    #ax_v_twin.set_ylabel(r"Adjusted Volume $(V^{1/k})/\sqrt{k}$")

    #secax_v = ax_v.secondary_xaxis('top', functions=(lambda x: x - 1, lambda x: x + 1))
    #secax_v.set_xlabel('Subspace Dimensionality')
    #secax_v.set_xticks(em_idx[1:] - 1)

    # Combine legends
    lines_v = p1 + p2 + p3# + p4 + p5
    labels_v = [l.get_label() for l in lines_v]
    ax_v.legend(lines_v, labels_v, loc='lower left', fontsize='x-small')

    # ---------------------------------------------------------
    # Panel 7: Extracted Endmembers Comparison (Water Absorption Bands Omitted)
    # ---------------------------------------------------------
    ax_em = fig.add_subplot(gs_bot[0, 2])
    real_em = real_res['endmembers']
    sim_em = sim_res['endmembers']

    wv_real_g, real_em_g = make_gapped_spectral_plot_arrays(wv_real, real_em)
    wv_sim_g, sim_em_g = make_gapped_spectral_plot_arrays(wv_sim, sim_em)

    # Highlight water absorption regions with shaded spans (do not plot any values in these windows)
    ax_em.axvspan(1350, 1450, color='#e0e0e0', alpha=0.55)#, label='Atmospheric H₂O')
    ax_em.axvspan(1800, 2000, color='#e0e0e0', alpha=0.55)
    ax_em.axvspan(2420, 2500, color='#e0e0e0', alpha=0.55)

    # Show first 4 endmembers to keep readable (gapped arrays prevent connecting lines across water gaps)
    # Discriminable blue-dominated palette for Real EMs (Navy, Royal Blue, Cerulean, Sky Blue)
    real_colors = ['#1D65AF', '#08306B',  '#0098D8', '#5C9DED']
    for j in range(min(4, num_endmembers)):
        ax_em.plot(wv_real_g, real_em_g[:, j], color=real_colors[j], linewidth=1.2, label=f'Real EM {j+1}')
    # Discriminable red-dominated palette for Simulated EMs (Maroon, Crimson, Scarlet, Coral)
    sim_colors = ['#B71C1C','#67000D',  '#E53935', '#EE6055']
    for j in range(min(4, num_endmembers)):
        ax_em.plot(wv_sim_g, sim_em_g[:, j], color=sim_colors[j], linewidth=1.2, label=f'Sim EM {j+1}')

    ax_em.set_title("Extracted Endmember Spectra (k=1..4)", fontsize=11, fontweight='bold')
    ax_em.set_xlabel("Wavelength (nm)")
    ax_em.set_ylabel("Reflectance")
    ax_em.grid(True, linestyle='--', alpha=0.6)
    ax_em.legend(loc='best', fontsize='x-small', ncol=2)

    # Overall Supertitle
    if layout in ['split_v', 'split_h', 'split_diag', 'edge_with_border'] and mat2_disp:
        suptitle_text = f"HySpex Mjolnir Comparison: Real {tile_size}x{tile_size} Tile at ({map_x}, {map_y}) vs. Simulated Edge ({mat1_disp} + {mat2_disp}, SNR={snr_str}, Shade={shade_suffix})"
    else:
        suptitle_text = f"HySpex Mjolnir Comparison: Real {tile_size}x{tile_size} Tile at ({map_x}, {map_y}) vs. Simulated {mat1_disp} (SNR={snr_str}, Shade={shade_suffix})"
    fig.suptitle(suptitle_text, fontsize=14, fontweight='bold', y=0.98)

    save_path = os.path.join(output_dir, f"hyspex_comparison_real_vs_simulated_{map_x}_{map_y}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved comparison figure to: {save_path}")

    # Show interactive window for IDE workflow
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Compare HySpex real data tile with simulated calibration target.")
    parser.add_argument("--preset", type=str, choices=["interior", "edge", "edge_3mat"], default=None,
                        help="Preset configuration: 'interior' (1059, 258 single panel) or 'edge' (1106, 326 panel/asphalt boundary).")
    parser.add_argument("--h5_path", type=str, default=DEFAULT_H5_PATH, help="Path to HySpex Ortho Stack HDF5 file.")
    parser.add_argument("--map_x", type=int, default=DEFAULT_MAP_X, help="Center X coordinate (samples / columns).")
    parser.add_argument("--map_y", type=int, default=DEFAULT_MAP_Y, help="Center Y coordinate (lines / rows).")
    parser.add_argument("--tile_size", type=int, default=DEFAULT_TILE_SIZE, help="Square tile size (default: 15).")
    parser.add_argument("--context_size", type=int, default=150, help="Square context window size for flightline overview (default: 150).")
    parser.add_argument("--num_endmembers", type=int, default=DEFAULT_NUM_ENDMEMBERS, help="Number of endmembers to extract.")
    parser.add_argument("--material", type=str, default=DEFAULT_MATERIAL, help="Primary simulated material name.")
    parser.add_argument("--material_2", type=str, default=DEFAULT_MATERIAL_2, help="Secondary simulated material name (e.g. 'splib07:Asphalt_GDS376').")
    parser.add_argument("--material_3", type=str, default=None, help="Tertiary simulated material name.")
    parser.add_argument("--layout", type=str, choices=["uniform", "split_v", "split_h", "split_diag", "edge_with_border"], default=DEFAULT_LAYOUT,
                        help="Spatial layout for simulated tile ('uniform', 'split_v', 'split_h', 'split_diag', 'edge_with_border').")
    parser.add_argument("--snr", type=float, default=DEFAULT_SNR, help="Target linear SNR for simulated tile.")
    parser.add_argument("--shot_fraction", type=float, default=DEFAULT_SHOT_NOISE_FRACTION, help="Shot noise fraction.")
    parser.add_argument("--shade_variability", type=float, default=DEFAULT_SHADE_VARIABILITY, help="Fractional shade/illumination variability (default: 0.03 for 3%%).")
    parser.add_argument("--sensor", type=str, default=DEFAULT_SENSOR, help="Simulated sensor name.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for noise generation.")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Output directory for plots.")
    parser.add_argument("--no_show", action="store_true", help="Skip interactive window display.")
    args = parser.parse_args()

    # Apply presets if selected
    if args.preset == "interior":
        args.map_x = 1059
        args.map_y = 258
        args.material = "CALTARGET:big_white_panel"
        args.material_2 = None
        args.layout = "uniform"
        args.shade_variability = 0.03
    elif args.preset == "edge":
        args.map_x = 1106
        args.map_y = 326
        args.material = "CALTARGET:big_white_panel"
        args.material_2 = "splib07:Asphalt_GDS376"
        args.layout = "split_diag"
        args.shade_variability = 0.03
    elif args.preset == "edge_3mat":
        args.map_x = 1121
        args.map_y = 258
        args.material = "CALTARGET:big_white_panel"
        args.material_2 = "splib07:Asphalt_GDS376"
        args.material_3 = "splib07:Asphalt_Tar_GDS346"
        args.layout = "edge_with_border"
        args.shade_variability = 0.03

    # 1. Retrieve Real Tile Metrics
    real_results = retrieve_real_tile(
        h5_path=args.h5_path,
        map_x=args.map_x,
        map_y=args.map_y,
        tile_size=args.tile_size,
        num_endmembers=args.num_endmembers,
        context_size=args.context_size
    )

    # 2. Generate Simulated Tile Metrics
    sim_results = generate_simulated_tile(
        sensor_name=args.sensor,
        material_name=args.material,
        material_2_name=args.material_2,
        material_3_name=args.material_3,
        layout=args.layout,
        tile_size=args.tile_size,
        snr=args.snr,
        num_endmembers=args.num_endmembers,
        shot_noise_fraction=args.shot_fraction,
        shade_variability=args.shade_variability,
        seed=args.seed
    )

    # 3. Display Comparison Table in Console
    display_comparison_table(
        real_res=real_results,
        sim_res=sim_results,
        num_endmembers=args.num_endmembers
    )

    # 4. Generate Publication-Quality Comparison Plots
    plot_comparison(
        real_res=real_results,
        sim_res=sim_results,
        map_x=args.map_x,
        map_y=args.map_y,
        tile_size=args.tile_size,
        num_endmembers=args.num_endmembers,
        output_dir=args.output_dir,
        show_plot=not args.no_show
    )


if __name__ == "__main__":
    main()
