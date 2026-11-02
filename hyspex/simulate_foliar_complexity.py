#!/usr/bin/env python3
"""
simulate_foliar_complexity.py

Pure-Pixel Foliar Differentiability Simulation:
Contrasting Spectral Complexity Metrics and PCA Dimensionality Across Healthy and Diseased Maple Foliage.

Dataset:
  Empirical leaf contact-probe ASD measurements from NICKLEAVES_20261001 resampled to the HySpex Mjolnir sensor:
    1. HealthyLeaf1: Norway Maple (Acer platanoides) - Healthy
    2. HealthyLeaf2: Silver Maple (Acer saccharinum) - Healthy
    3. DiseaseLeaf1: Norway Maple (Acer platanoides) - Rhytisma acerinum Tar Spot
    4. DiseaseLeaf2: Silver Maple (Acer saccharinum) - Rhytisma acerinum Tar Spot

Scenarios:
  1. Two-Material Contrast (HealthyLeaf1 vs. HealthyLeaf2):
     Isolates inter-species separation; orthogonal subspace distances plateau at k* = 2 endmembers.
  2. Four-Material Pathological Contrast (50% of healthy foliage replaced with tar spot equivalents):
     Introducing fungal necrotic lesions sustains elevated orthogonal subspace distances through k* = 4 endmembers
     before plunging into the sensor noise floor.
  3. Direct Contrast against Principal Component Analysis (PCA):
     Demonstrates how variance-based dimensionality reduction (PCA) suppresses biologically critical foliar differences
     into discarded tail variance, whereas MaxD Parallelotope Spectral Complexity recovers true physical material dimensionality.

Simulation Parameters (matching white panel with asphalt):
  - Tile Size: 15x15 (225 pure pixels)
  - Sensor: cvHySpexMjolnir (413 channels, telluric water absorption bands omitted)
  - SNR: 200 (Heteroscedastic Poisson-Gaussian noise, 70% shot noise fraction)
  - Shading Variability: 3% uniform illumination variability U(0.97, 1.0)
  - Seed: 42
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.decomposition import PCA

# Add simulated-data-test to sys.path for sensor simulation methods
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SIM_DIR = os.path.normpath(os.path.join(CURRENT_DIR, "..", "simulated-data-test"))
if SIM_DIR not in sys.path:
    sys.path.append(SIM_DIR)

try:
    from simulate_inset_squares import (
        maximumDistance_parallelotopeVolumes_noise,
        add_wavelength_scaled_noise,
        generate_rgba_from_hsi
    )
except ImportError:
    raise ImportError("Could not import simulate_inset_squares. Ensure simulated-data-test is in sys.path.")

# --- Default Configurations ---
NICKS_LEAVES_DIR = r"C:\satelliteImagery\ground_truth\NickLeaves_20261001"
DEFAULT_OUTPUT_DIR = r"C:\satelliteImagery\simulated\foliar_comparison"
DEFAULT_TILE_SIZE = 15
DEFAULT_SNR = 200
DEFAULT_SHOT_NOISE_FRACTION = 0.70
DEFAULT_SHADE_VARIABILITY = 0.06
DEFAULT_SEED = 42
DEFAULT_NUM_ENDMEMBERS = 10
DEFAULT_SENSOR = "cvHySpexMjolnir"

# Accessible Okabe-Ito color cycle
try:
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.colormaps['okabe_ito'].colors)
except Exception:
    pass


def make_gapped_spectral_plot_arrays(wv, data, gap_threshold_nm=20.0):
    """
    Inserts a NaN entry at atmospheric water absorption gaps (> 20 nm channel spacing)
    so matplotlib breaks lines cleanly without connecting across omitted intervals.
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


def load_foliar_spectrum(material_key, sensor_name=DEFAULT_SENSOR):
    """
    Loads empirical foliar reflectance spectrum resampled to sensor response,
    omitting atmospheric water absorption bands.
    """
    csv_name = f"{sensor_name}_{material_key}.csv"
    csv_path = os.path.join(NICKS_LEAVES_DIR, csv_name)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Leaf CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    spec = df['DHR_mean'].values.astype(np.float32)
    wv = df['wavelength'].values.astype(np.float32)
    wavelengths_nm = wv if np.nanmax(wv) > 100.0 else wv * 1000.0

    # Guarantee water absorption bands are omitted
    is_water_band = ((wavelengths_nm >= 1350.0) & (wavelengths_nm <= 1450.0)) | \
                    ((wavelengths_nm >= 1800.0) & (wavelengths_nm <= 2000.0)) | \
                    (wavelengths_nm > 2420.0)
    if np.any(is_water_band):
        valid_bands = ~is_water_band
        spec = spec[valid_bands]
        wavelengths_nm = wavelengths_nm[valid_bands]

    return wavelengths_nm, spec


def simulate_foliar_tile(scenario="healthy_2mat",
                         tile_size=DEFAULT_TILE_SIZE,
                         snr=DEFAULT_SNR,
                         shot_noise_fraction=DEFAULT_SHOT_NOISE_FRACTION,
                         shade_variability=DEFAULT_SHADE_VARIABILITY,
                         num_endmembers=DEFAULT_NUM_ENDMEMBERS,
                         seed=DEFAULT_SEED,
                         sensor_name=DEFAULT_SENSOR):
    """
    Builds a synthetic pure-pixel foliar tile and computes MaxD Parallelotope Complexity and PCA metrics.
    
    Scenarios:
      - 'Single_Mat': 100% HealthyLeaf1 (Norway Maple).
      - 'healthy_2mat': 50% HealthyLeaf1 (Norway Maple), 50% HealthyLeaf2 (Silver Maple).
      - 'pathological_4mat': 50% of each healthy variant replaced with tar spot equivalent:
                            Top-Left: HealthyLeaf1 (Norway Maple)
                            Bottom-Left: DiseaseLeaf1 (Norway Maple Tar Spot)
                            Top-Right: HealthyLeaf2 (Silver Maple)
                            Bottom-Right: DiseaseLeaf2 (Silver Maple Tar Spot)
    """
    print(f"\n--- Simulating Foliar Tile: {scenario.upper()} ---")
    print(f"Sensor: {sensor_name} | Tile Size: {tile_size}x{tile_size} | Linear SNR: {snr}")
    print(f"Shading Variability: {shade_variability * 100:.1f}% | Random Seed: {seed}")

    # 1. Load Spectra
    wv, spec_hl1 = load_foliar_spectrum("HealthyLeaf1", sensor_name)
    spec_hl2 = None
    spec_dl1 = None
    spec_dl2 = None
    if scenario in ["healthy_2mat", "pathological_4mat"]:
        _, spec_hl2 = load_foliar_spectrum("HealthyLeaf2", sensor_name)
    if scenario == "pathological_4mat":
        _, spec_dl1 = load_foliar_spectrum("DiseaseLeaf1", sensor_name)
        _, spec_dl2 = load_foliar_spectrum("DiseaseLeaf2", sensor_name)

    num_bands = len(spec_hl1)
    clean_tile = np.zeros((num_bands, tile_size, tile_size), dtype=np.float32)
    material_map = np.zeros((tile_size, tile_size), dtype=np.int32)
    class_names = []

    half_c = tile_size // 2 + 1  # 8 columns for left half, 7 for right half
    half_r = tile_size // 2 + 1  # 8 rows for top half, 7 for bottom half

    if scenario in ["Single_Mat", "single_mat"]:
        # 100% Healthy Norway Maple
        clean_tile[:, :, :] = spec_hl1[:, None, None]
        material_map[:, :] = 0
        class_names = [
            "Norway Maple (Healthy)"
        ]
        material_spectra = {
            "HealthyLeaf1": spec_hl1,
        }
    elif scenario == "healthy_2mat":
        # 50% Healthy Norway Maple, 50% Healthy Silver Maple
        clean_tile[:, :, :half_c] = spec_hl1[:, None, None]
        clean_tile[:, :, half_c:] = spec_hl2[:, None, None]
        material_map[:, :half_c] = 0
        material_map[:, half_c:] = 1
        class_names = [
            "Norway Maple (Healthy)",
            "Silver Maple (Healthy)"
        ]
        material_spectra = {
            "HealthyLeaf1": spec_hl1,
            "HealthyLeaf2": spec_hl2
        }
    elif scenario == "pathological_4mat":
        # 50% of each species replaced with diseased counterpart
        # Top-Left: Healthy Norway Maple
        clean_tile[:, :half_r, :half_c] = spec_hl1[:, None, None]
        material_map[:half_r, :half_c] = 0
        # Bottom-Left: Tar Spot Norway Maple
        clean_tile[:, half_r:, :half_c] = spec_dl1[:, None, None]
        material_map[half_r:, :half_c] = 1
        # Top-Right: Healthy Silver Maple
        clean_tile[:, :half_r, half_c:] = spec_hl2[:, None, None]
        material_map[:half_r, half_c:] = 2
        # Bottom-Right: Tar Spot Silver Maple
        clean_tile[:, half_r:, half_c:] = spec_dl2[:, None, None]
        material_map[half_r:, half_c:] = 3
        class_names = [
            "Norway Maple (Healthy)",
            "Norway Maple (Tar Spot)",
            "Silver Maple (Healthy)",
            "Silver Maple (Tar Spot)"
        ]
        material_spectra = {
            "HealthyLeaf1": spec_hl1,
            "DiseaseLeaf1": spec_dl1,
            "HealthyLeaf2": spec_hl2,
            "DiseaseLeaf2": spec_dl2
        }
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    # 2. Apply 3% Uniform Shading Map matching spectral_complexity_vs_pca.py
    shade_map = None
    if shade_variability > 0:
        if seed is not None:
            np.random.seed(seed)
        shade_map = np.random.uniform(1.0 - shade_variability, 1.0, (tile_size, tile_size)).astype(np.float32)
        clean_tile = clean_tile * shade_map[None, :, :]

    # 3. Add Wavelength-Scaled Heteroscedastic Noise
    noisy_tile, noise_array = add_wavelength_scaled_noise(
        clean_tile,
        snr=snr,
        shot_noise_fraction=shot_noise_fraction,
        seed=seed
    )

    # 4. Generate True Color RGBA Image
    try:
        rgba = generate_rgba_from_hsi(noisy_tile, wv)
    except Exception as e:
        print(f"Warning: generate_rgba_from_hsi fallback ({e})")
        b_idx = [min(50, num_bands - 1), min(100, num_bands - 1), min(150, num_bands - 1)]
        rgb = noisy_tile[b_idx, :, :].transpose((1, 2, 0))
        rgb = np.clip(rgb / np.percentile(rgb, 98), 0, 1)
        rgba = (rgb * 255).astype(np.uint8)
        alpha = np.full((tile_size, tile_size, 1), 255, dtype=np.uint8)
        rgba = np.concatenate([rgba, alpha], axis=-1)

    # 5. Extract MaxD Endmembers and Parallelotope Metrics
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

    derivative = np.full_like(heights, np.nan)
    if len(heights) > 2:
        derivative[1:] = -np.gradient(heights[1:])

    r_tile, c_tile = np.unravel_index(indices, (tile_size, tile_size))

    # 6. Compute Principal Component Analysis (PCA)
    pca = PCA().fit(pixels.T)
    pca_evr = pca.explained_variance_ratio_
    pca_cum_evr = np.cumsum(pca_evr)
    pca_singular_vals = pca.singular_values_

    return {
        'scenario': scenario,
        'clean_tile': clean_tile,
        'noisy_tile': noisy_tile,
        'noise_array': noise_array,
        'rgba_image': rgba,
        'material_map': material_map,
        'class_names': class_names,
        'material_spectra': material_spectra,
        'wavelengths_nm': wv,
        'computed_heights': heights,
        'computed_volumes_log': vol_log,
        'computed_volumes_linear': volumes_linear,
        'computed_adj_volumes': adj_volumes,
        'computed_derivative': derivative,
        'noise_heights': noise_heights,
        'noise_norms_mean': noise_norms_mean,
        'endmembers': em,
        'indices': indices,
        'coords_2d': (r_tile, c_tile),
        'pca_evr': pca_evr,
        'pca_cum_evr': pca_cum_evr,
        'pca_singular_vals': pca_singular_vals,
        'snr': snr,
        'tile_size': tile_size,
        'shade_variability': shade_variability
    }


def display_comparison_table(res_2mat, res_4mat, num_endmembers=10):
    """
    Prints a detailed side-by-side console comparison table contrasting the 2-material healthy case
    with the 4-material pathological case, plus PCA cumulative variance.
    """
    print("\n" + "=" * 125)
    print("FOLIAR DIFFERENTIABILITY: 2-MATERIAL HEALTHY MAPLE vs. 4-MATERIAL PATHOLOGICAL (TAR SPOT) CONTRAST")
    print("=" * 125)
    print(f"{'EM':<10} | {'2-MATERIAL HEALTHY (Norway vs. Silver)':^52} | {'4-MATERIAL PATHOLOGICAL (50% Diseased)':^55}")
    print(f"{'k':<10} | {'Distance (h_k)':^15} {'Noise h':^10} {'Volume (Lin)':^14} {'ln(Vol)':^8} | "
          f"{'Distance (h_k)':^15} {'Noise h':^10} {'Volume (Lin)':^14} {'ln(Vol)':^8}")
    print("-" * 125)

    h_2 = res_2mat['computed_heights']
    nh_2 = res_2mat['noise_heights']
    v_2 = res_2mat['computed_volumes_linear']
    lnv_2 = res_2mat['computed_volumes_log']

    h_4 = res_4mat['computed_heights']
    nh_4 = res_4mat['noise_heights']
    v_4 = res_4mat['computed_volumes_linear']
    lnv_4 = res_4mat['computed_volumes_log']

    num_rows = min(len(h_2), len(h_4))
    for k in range(num_rows):
        idx_str = f"EM {k}" if k > 0 else "0 (origin)"
        
        # 2-material strings
        h2_str = f"{h_2[k]:.6f}"
        nh2_str = f"{nh_2[k]:.6f}" if nh_2[k] > 0 else "0.000000"
        v2_str = f"{v_2[k]:.3e}" if k > 0 else "1.000"
        lnv2_str = f"{lnv_2[k]:.2f}" if not np.isinf(lnv_2[k]) else "-inf"

        # 4-material strings
        h4_str = f"{h_4[k]:.6f}"
        nh4_str = f"{nh_4[k]:.6f}" if nh_4[k] > 0 else "0.000000"
        v4_str = f"{v_4[k]:.3e}" if k > 0 else "1.000"
        lnv4_str = f"{lnv_4[k]:.2f}" if not np.isinf(lnv_4[k]) else "-inf"

        print(f"{idx_str:<10} | {h2_str:<15} {nh2_str:<10} {v2_str:<14} {lnv2_str:<8} | "
              f"{h4_str:<15} {nh4_str:<10} {v4_str:<14} {lnv4_str:<8}")

    print("-" * 125)
    print("\nPCA EXPLAINED VARIANCE RATIO COMPARISON:")
    print(f"{'PC':<6} | {'2-Material Healthy':^30} | {'4-Material Pathological':^30}")
    print(f"{'#':<6} | {'Individual EVR':^15} {'Cumulative EVR':^15} | {'Individual EVR':^15} {'Cumulative EVR':^15}")
    print("-" * 72)
    evr_2 = res_2mat['pca_evr']
    cum_2 = res_2mat['pca_cum_evr']
    evr_4 = res_4mat['pca_evr']
    cum_4 = res_4mat['pca_cum_evr']
    for p in range(min(6, len(evr_2))):
        print(f"PC {p+1:<3} | {evr_2[p]*100:>13.2f}% {cum_2[p]*100:>13.2f}% | {evr_4[p]*100:>13.2f}% {cum_4[p]*100:>13.2f}%")
    print("=" * 125 + "\n")


def plot_foliar_scenario(res, output_path, show_plot=True):
    """
    Plots a multi-panel visual and analytical summary for an individual foliar scenario.
    Uses a clean 2-row, 3-column layout where the Ground Truth legend is incorporated
    directly into the Simulated Tile RGBA plot matching the simulated pixel colors.
    """
    fig = plt.figure(figsize=(20, 11))
    fig.canvas.manager.set_window_title(f"Foliar Complexity: {res['scenario']}")

    # 2 rows, 3 columns:
    # Row 1: Simulated Tile RGBA with Ground Truth Swatches, Pure Foliar Spectra, Noisy Reflectance Sample
    # Row 2: Orthogonal Subspace Distance (SymLog), Complexity Volume (Log), Extracted Endmember Spectra
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.1], hspace=0.25, wspace=0.2)

    tile_size = res['tile_size']
    snr = res['snr']
    wv = res['wavelengths_nm']
    em_idx = np.arange(1, len(res['computed_heights']) + 1)

    # ---------------------------------------------------------
    # Panel 1: Simulated Tile RGBA with Ground Truth Swatches
    # ---------------------------------------------------------
    ax_img = fig.add_subplot(gs[0, 0])
    ax_img.imshow(res['rgba_image'])
    r_em, c_em = res['coords_2d']
    #ax_img.scatter(c_em, r_em, facecolors='none', edgecolors='cyan', s=80, linewidth=1.8)
    #for i, (r, c) in enumerate(zip(r_em, c_em)):
    #    ax_img.text(c + 0.35, r - 0.35, str(i + 1), color='yellow', fontsize=9, fontweight='bold')
    if res['scenario'] == "pathological_4mat":
        ax_img.axvline(7.5, color='white', linestyle=':', linewidth=1.0, alpha=0.6)
        ax_img.axhline(7.5, color='white', linestyle=':', linewidth=1.0, alpha=0.6)
        tile_title = f"Simulated {tile_size}x{tile_size} Tile: 50% Tar Spot Replacement\nTop: Healthy Leaves | Bottom: Diseased Leaves\n(Left: Norway Maple | Right: Silver Maple)"
    elif res['scenario'] in ["Single_Mat", "single_mat"]:
        tile_title = f"Simulated {tile_size}x{tile_size} Tile: Single Material\n(Norway Maple - Healthy)"
    else:
        ax_img.axvline(7.5, color='white', linestyle=':', linewidth=1.0, alpha=0.6)
        tile_title = f"Simulated {tile_size}x{tile_size} Tile: Healthy Maple Species\nLeft: Norway Maple | Right: Silver Maple"

    ax_img.set_title(tile_title, fontsize=10, fontweight='bold')
    ax_img.set_xlabel("Local Column")
    ax_img.set_ylabel("Local Row")

    # ---------------------------------------------------------
    # Panel 2: Pure Foliar Reflectance Spectra
    # ---------------------------------------------------------
    ax_pure = fig.add_subplot(gs[0, 1])
    ax_pure.axvspan(1350, 1450, color='#e0e0e0', alpha=0.55)#, label='Atmospheric H₂O')
    ax_pure.axvspan(1800, 2000, color='#e0e0e0', alpha=0.55)
    ax_pure.axvspan(2420, 2500, color='#e0e0e0', alpha=0.55)

    colors = ['#2ca02c', '#d62728', '#1f77b4', '#8c564b']
    for idx, (k_name, spec) in enumerate(res['material_spectra'].items()):
        wv_g, spec_g = make_gapped_spectral_plot_arrays(wv, spec)
        lbl = res['class_names'][idx]
        ax_pure.plot(wv_g, spec_g, color=colors[idx % len(colors)], linewidth=2.0, label=lbl)

    ax_pure.set_title("Pure Foliar Reflectance Spectra", fontsize=11, fontweight='bold')
    ax_pure.set_xlabel("Wavelength (nm)")
    ax_pure.set_ylabel("Reflectance")
    ax_pure.set_ylim(-0.02, 0.70)
    ax_pure.grid(True, linestyle='--', alpha=0.6)
    ax_pure.legend(loc='upper right', fontsize='x-small')

    # ---------------------------------------------------------
    # Panel 3: Noisy Pixel Reflectance Sample
    # ---------------------------------------------------------
    ax_noisy = fig.add_subplot(gs[0, 2])
    ax_noisy.axvspan(1350, 1450, color='#e0e0e0', alpha=0.55)
    ax_noisy.axvspan(1800, 2000, color='#e0e0e0', alpha=0.55)
    ax_noisy.axvspan(2420, 2500, color='#e0e0e0', alpha=0.55)

    pix_noisy = res['noisy_tile'].reshape(len(wv), -1)
    wv_g, pix_noisy_g = make_gapped_spectral_plot_arrays(wv, pix_noisy)
    ax_noisy.plot(wv_g, pix_noisy_g[:, ::25], color='slategray', alpha=0.45, linewidth=0.8)
    
    # Overplot the ground truth curves for reference
    for idx, (k_name, spec) in enumerate(res['material_spectra'].items()):
        _, spec_g = make_gapped_spectral_plot_arrays(wv, spec)
        ax_noisy.plot(wv_g, spec_g, color=colors[idx % len(colors)], linestyle='--', linewidth=1.5, label=f"GT: {res['class_names'][idx].split()[0]}")

    ax_noisy.set_title(f"Tile Reflectance + Noise (SNR={snr})", fontsize=11, fontweight='bold')
    ax_noisy.set_xlabel("Wavelength (nm)")
    ax_noisy.set_ylabel("Reflectance")
    ax_noisy.set_ylim(-0.02, 0.70)
    ax_noisy.grid(True, linestyle='--', alpha=0.6)
    ax_noisy.legend(loc='upper right', fontsize='x-small')

    # ---------------------------------------------------------
    # Panel 4: Orthogonal Subspace Distance (h_k) on SymLog Scale
    # ---------------------------------------------------------
    ax_h = fig.add_subplot(gs[1, 0])
    ax_h.plot(em_idx, res['computed_heights'], marker='o', linestyle='-', color='blue', linewidth=1.8, label='Orthogonal Subspace Distance ($h_k$)')
    ax_h.plot(em_idx[1:], res['computed_derivative'][1:], marker='^', linestyle=':', color='green', alpha=0.8, label=r'Decay Rate ($-\Delta h$)')
    ax_h.plot(em_idx, res['noise_heights'], marker='s', linestyle='--', color='purple', linewidth=1.5, label='Simulated Noise Distance')
    ax_h.plot(em_idx[1:], res['noise_norms_mean'][1:], marker='d', linestyle=':', color='chocolate', linewidth=1.2, label='Mean Noise Floor')

    # 1% Reflectance Contrast Threshold: h_crit = 0.010 * sqrt(B)
    B = len(wv)
    h_crit = 0.010 * np.sqrt(B)
    ax_h.axhline(h_crit, color='black', linestyle='--', linewidth=1.5, label=f'1% Threshold ($h_{{crit}} = {h_crit:.3f}$)')

    ax_h.set_yscale('symlog', linthresh=1e-3)
    ax_h.set_ylim(bottom=0)
    ax_h.set_title(r"Orthogonal Subspace Distance ($h_k$)", fontsize=11, fontweight='bold')
    ax_h.set_xlabel("Vertex Index (k)")
    ax_h.set_ylabel(r"Orthogonal Subspace Distance ($h_k$)")
    ax_h.set_xticks(em_idx)
    ax_h.grid(True, linestyle='--', alpha=0.6)
    ax_h.legend(loc='upper right', fontsize='x-small')

    #secax_h = ax_h.secondary_xaxis('top', functions=(lambda x: x - 1, lambda x: x + 1))
    #secax_h.set_xlabel('Subspace Dimensionality')
    #secax_h.set_xticks(em_idx - 1)

    # ---------------------------------------------------------
    # Panel 5: Complexity Volume & Adjusted Volume Comparison
    # ---------------------------------------------------------
    ax_v = fig.add_subplot(gs[1, 1])
    ax_v_twin = ax_v.twinx()

    v_lin = res['computed_heights'][1] - res['computed_volumes_linear']
    p1 = ax_v.plot(em_idx[1:],  v_lin[1:], marker='s', linestyle='-', color='crimson', linewidth=1.8, label='Spectral Complexity Vol')
    #ax_v.set_yscale('log')
    ax_v.set_title(r"Spectral Complexity ($\prod h_k$)", fontsize=11, fontweight='bold')
    ax_v.set_xlabel("Vertex Index (k)")
    ax_v.set_ylabel("Linear Parallelotope Volume")
    ax_v.set_xticks(em_idx[1:])
    ax_v.grid(True, linestyle='--', alpha=0.6)

    #p2 = ax_v_twin.plot(em_idx[1:], res['computed_adj_volumes'][1:], marker='v', linestyle='-.', color='coral', alpha=0.8, label='Adjusted Volume')
    #ax_v_twin.set_ylabel(r"Adjusted Volume $(V^{1/k})/\sqrt{k}$")

    #secax_v = ax_v.secondary_xaxis('top', functions=(lambda x: x - 1, lambda x: x + 1))
    #secax_v.set_xlabel('Subspace Dimensionality')
    #secax_v.set_xticks(em_idx[1:] - 1)

    lines_v = p1# + p2
    labels_v = [l.get_label() for l in lines_v]
    ax_v.legend(lines_v, labels_v, loc='upper right', fontsize='x-small')

    # ---------------------------------------------------------
    # Panel 6: Extracted Endmember Spectra
    # ---------------------------------------------------------

    if res['scenario'] in ["Single_Mat", "single_mat"]:
        title_prefix = "Single-Material Scenario"
    elif res['scenario'] == "healthy_2mat":
        title_prefix = "Two-Material Scenario"
    else:
        title_prefix = "Four-Material Scenario"

    ax_em = fig.add_subplot(gs[1, 2])
    ax_em.axvspan(1350, 1450, color='#e0e0e0', alpha=0.55, label='Atmospheric H₂O')
    ax_em.axvspan(1800, 2000, color='#e0e0e0', alpha=0.55)
    ax_em.axvspan(2420, 2500, color='#e0e0e0', alpha=0.55)

    em_spectra = res['endmembers']
    wv_g, em_spectra_g = make_gapped_spectral_plot_arrays(wv, em_spectra)
    for j in range(min(4, em_spectra.shape[1])):
        ax_em.plot(wv_g, em_spectra_g[:, j], color=colors[j % len(colors)], linestyle='-', linewidth=1.5, label=f"Extracted EM {j+1}")

    ax_em.set_title(f"{title_prefix} Extracted Endmember Spectra (k=1..4)", fontsize=11, fontweight='bold')
    ax_em.set_xlabel("Wavelength (nm)")
    ax_em.set_ylabel("Reflectance")
    ax_em.grid(True, linestyle='--', alpha=0.6)
    ax_em.legend(loc='upper right', fontsize='x-small')

    # Supertitle
    if res['scenario'] in ["Single_Mat", "single_mat"]:
        scen_title = "Single-Material Healthy Foliage (Norway Maple)"
    elif res['scenario'] == "healthy_2mat":
        scen_title = "Two-Material Healthy Foliage (Norway Maple vs. Silver Maple)"
    else:
        scen_title = "Four-Material Pathological Contrast (50% Diseased Tar Spot Replacement)"
    fig.suptitle(f"HySpex Mjolnir Simulation: {scen_title} (SNR={snr}, Shading={res['shade_variability']:.2%})", fontsize=14, fontweight='bold', y=0.98)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved scenario figure to: {output_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def plot_contrast_figure(res_2mat, res_4mat, output_path, show_plot=True):
    """
    Generates a high-level executive contrast figure directly comparing the 2-material healthy case
    with the 4-material pathological case, contrasting Orthogonal Subspace Distances, Volumes, and PCA Scree plots.
    Designed specifically for report synthesis and slide presentations.
    """
    fig = plt.figure(figsize=(22, 11))
    fig.canvas.manager.set_window_title("Healthy vs. Infected Complexity")

    gs_main = fig.add_gridspec(2, 1, height_ratios=[1, 1.1], hspace=0.25)
    gs_top = gs_main[0].subgridspec(1, 3, wspace=0.32)
    gs_bot = gs_main[1].subgridspec(1, 3, wspace=0.3)

    wv = res_2mat['wavelengths_nm']
    em_idx = np.arange(1, len(res_2mat['computed_heights']) + 1)

    # ---------------------------------------------------------
    # Top Row - Panel 1: 2-Material Healthy Tile RGBA
    # ---------------------------------------------------------
    ax_t1 = fig.add_subplot(gs_top[0, 0])
    ax_t1.imshow(res_2mat['rgba_image'])
    r_em1, c_em1 = res_2mat['coords_2d']
    ax_t1.scatter(c_em1, r_em1, facecolors='none', edgecolors='cyan', s=80, linewidth=1.8, label='Extracted EMs')
    for i, (r, c) in enumerate(zip(r_em1, c_em1)):
        ax_t1.text(c + 0.35, r - 0.35, str(i + 1), color='yellow', fontsize=9, fontweight='bold')
    ax_t1.axvline(7.5, color='white', linestyle=':', linewidth=1.0, alpha=0.6)
    ax_t1.set_title("2-Material Tile: Healthy Maple Species\nLeft: Norway Maple | Right: Silver Maple", fontsize=10, fontweight='bold')
    ax_t1.set_xlabel("Local Column")
    ax_t1.set_ylabel("Local Row")
    ax_t1.legend(loc='lower right', fontsize='small')

    # ---------------------------------------------------------
    # Top Row - Panel 2: 4-Material Pathological Tile RGBA
    # ---------------------------------------------------------
    ax_t2 = fig.add_subplot(gs_top[0, 1])
    ax_t2.imshow(res_4mat['rgba_image'])
    r_em2, c_em2 = res_4mat['coords_2d']
    ax_t2.scatter(c_em2, r_em2, facecolors='none', edgecolors='cyan', s=80, linewidth=1.8, label='Extracted EMs')
    for i, (r, c) in enumerate(zip(r_em2, c_em2)):
        ax_t2.text(c + 0.35, r - 0.35, str(i + 1), color='yellow', fontsize=9, fontweight='bold')
    ax_t2.axvline(7.5, color='white', linestyle=':', linewidth=1.0, alpha=0.6)
    ax_t2.axhline(7.5, color='white', linestyle=':', linewidth=1.0, alpha=0.6)
    ax_t2.set_title("4-Material Tile: 50% Tar Spot Diseased Replacement\nTop: Healthy Leaves | Bottom: Diseased Leaves\n(Left: Norway Maple | Right: Silver Maple)", fontsize=9, fontweight='bold')
    ax_t2.set_xlabel("Local Column")
    ax_t2.set_ylabel("Local Row")
    ax_t2.legend(loc='lower right', fontsize='small')

    # ---------------------------------------------------------
    # Top Row - Panel 3: Foliar Reflectance Profiles Comparison
    # ---------------------------------------------------------
    ax_t3 = fig.add_subplot(gs_top[0, 2])
    ax_t3.axvspan(1350, 1450, color='#e0e0e0', alpha=0.55)
    ax_t3.axvspan(1800, 2000, color='#e0e0e0', alpha=0.55)
    ax_t3.axvspan(2420, 2500, color='#e0e0e0', alpha=0.55)

    spec_colors = ['#2ca02c', '#8c564b', '#1f77b4', '#d62728']
    tile_size = res_4mat['tile_size']
    half_r = tile_size // 2 + 1
    half_c = tile_size // 2 + 1

    # Define 4 square corner regions (quadrants) of the tile
    quadrant_slices = [
        (slice(0, half_r), slice(0, half_c)),       # Top-Left: Norway Maple (Healthy)
        (slice(half_r, None), slice(0, half_c)),    # Bottom-Left: Norway Maple (Tar Spot)
        (slice(0, half_r), slice(half_c, None)),    # Top-Right: Silver Maple (Healthy)
        (slice(half_r, None), slice(half_c, None)), # Bottom-Right: Silver Maple (Tar Spot)
    ]
    noisy_tile = res_4mat['noisy_tile']
    for q, (r_slice, c_slice) in enumerate(quadrant_slices):
        q_pixels = noisy_tile[:, r_slice, c_slice].reshape(len(wv), -1)
        wv_g, q_pix_g = make_gapped_spectral_plot_arrays(wv, q_pixels)
        ax_t3.plot(wv_g, q_pix_g, color=spec_colors[q], alpha=0.25, linewidth=0.7)
        ax_t3.plot([], [], color=spec_colors[q], linewidth=2.0, label=res_4mat['class_names'][q])

    ax_t3.set_title("Simulated Pixel Reflectance Profiles", fontsize=11, fontweight='bold')
    ax_t3.set_xlabel("Wavelength (nm)")
    ax_t3.set_ylabel("Reflectance")
    ax_t3.set_ylim(-0.01, 0.60)
    ax_t3.grid(True, linestyle='--', alpha=0.6)
    ax_t3.legend(loc='upper right', fontsize='x-small')

    # ---------------------------------------------------------
    # Bottom Row - Panel 4: Orthogonal Subspace Distance Comparison (SymLog)
    # ---------------------------------------------------------
    ax_b1 = fig.add_subplot(gs_bot[0, 0])
    ax_b1.plot(em_idx, res_2mat['computed_heights'], marker='o', linestyle='-', color='blue', linewidth=2.0, label=r'2-Material Healthy $h_k$')
    ax_b1.plot(em_idx[1:], res_2mat['computed_derivative'][1:], marker='^', linestyle=':', color='blue', linewidth=2.0, label=r'Healthy $-\Delta h_k$')
    ax_b1.plot(em_idx, res_4mat['computed_heights'], marker='s', linestyle='-', color='red', linewidth=2.0, label=r'4-Material Infected $h_k$')
    ax_b1.plot(em_idx[1:], res_4mat['computed_derivative'][1:], marker='^', linestyle=':', color='red', linewidth=2.0, label=r'Infected $-\Delta h_k$')
    ax_b1.plot(em_idx[1:], res_4mat['noise_norms_mean'][1:], marker='x', linestyle='--', color='#F76902', linewidth=1.4, label='Simulated Mean Noise')

    # 1% Reflectance Contrast Threshold: h_crit = 0.010 * sqrt(B)
    B = len(wv)
    h_crit = 0.010 * np.sqrt(B)
    ax_b1.axhline(h_crit, color='black', linestyle='--', linewidth=1.5, label=f'1% Threshold ($h_{{crit}} = {h_crit:.3f}$)')

    # Annotate plateau points
    #ax_b1.annotate('Plateau after k*=2\n(Species distinction)', xy=(2, res_2mat['computed_heights'][1]), xytext=(2.2, 0.3),
    #               arrowprops=dict(facecolor='forestgreen', arrowstyle='->', lw=1.5), fontsize=8, fontweight='bold', color='forestgreen')
    #ax_b1.annotate('Sustained through k*=4\n(Pathology dimensions)', xy=(4, res_4mat['computed_heights'][3]), xytext=(4.3, 0.2),
    #               arrowprops=dict(facecolor='crimson', arrowstyle='->', lw=1.5), fontsize=8, fontweight='bold', color='crimson')

    ax_b1.set_yscale('symlog', linthresh=1e-2)
    ax_b1.set_ylim(bottom=0)
    ax_b1.set_title("Orthogonal Subspace Distance ($h_k$) Comparison", fontsize=11, fontweight='bold')
    ax_b1.set_xlabel("Endmember Index (k)")
    ax_b1.set_ylabel(r"Orthogonal Subspace Distance ($h_k$)")
    ax_b1.set_xticks(em_idx)
    ax_b1.grid(True, linestyle='--', alpha=0.6)
    ax_b1.legend(loc='upper right', fontsize='small')

    #secax_b1 = ax_b1.secondary_xaxis('top', functions=(lambda x: x - 1, lambda x: x + 1))
    #secax_b1.set_xlabel('Subspace Dimensionality')
    #secax_b1.set_xticks(em_idx - 1)

    # ---------------------------------------------------------
    # Bottom Row - Panel 5: Complexity Volume Comparison (Log)
    # ---------------------------------------------------------
    ax_b2 = fig.add_subplot(gs_bot[0, 1])
    ax_b2.plot(em_idx[1:], res_2mat['computed_heights'][1] - res_2mat['computed_volumes_linear'][1:], marker='o', linestyle='-', color='blue', linewidth=2.0, label=r'2-Material Healthy $\prod h_k$')
    ax_b2.plot(em_idx[1:], res_2mat['computed_heights'][1] - res_4mat['computed_volumes_linear'][1:], marker='s', linestyle='-', color='red', linewidth=2.0, label=r'4-Material Infected $\prod h_k$')
    
    #ax_b2.set_yscale('log')
    ax_b2.set_title(r"Spectral Complexity Volume ($\prod h_k$) Contrast", fontsize=11, fontweight='bold')
    ax_b2.set_xlabel("Endmember Index (k)")
    ax_b2.set_ylabel("Linear Parallelotope Volume")
    ax_b2.set_xticks(em_idx[1:])
    ax_b2.grid(True, linestyle='--', alpha=0.6)
    ax_b2.legend(loc='upper right', fontsize='small')

    #secax_b2 = ax_b2.secondary_xaxis('top', functions=(lambda x: x - 1, lambda x: x + 1))
    #secax_b2.set_xlabel('Subspace Dimensionality')
    #secax_b2.set_xticks(em_idx[1:] - 1)

    # ---------------------------------------------------------
    # Bottom Row - Panel 6: PCA Scree & Cumulative Variance Contrast
    # ---------------------------------------------------------
    ax_b3 = fig.add_subplot(gs_bot[0, 2])
    ax_b3_twin = ax_b3.twinx()

    pc_idx = np.arange(1, 7)
    evr2_pct = res_2mat['pca_evr'][:6] * 100.0
    evr4_pct = res_4mat['pca_evr'][:6] * 100.0
    cum2_pct = res_2mat['pca_cum_evr'][:6] * 100.0
    cum4_pct = res_4mat['pca_cum_evr'][:6] * 100.0

    # Bar chart for individual variance
    w = 0.35
    ax_b3.bar(pc_idx - w/2, evr2_pct, width=w, color='forestgreen', alpha=0.65, label='2-Mat PC Variance (%)')
    ax_b3.bar(pc_idx + w/2, evr4_pct, width=w, color='crimson', alpha=0.65, label='4-Mat PC Variance (%)')
    ax_b3.set_ylabel("Individual Explained Variance (%)")
    ax_b3.set_xlabel("Principal Component Index")
    ax_b3.set_xticks(pc_idx)
    ax_b3.set_ylim(0, 105)
    ax_b3.grid(True, linestyle='--', alpha=0.4)

    # Cumulative lines on twin axis
    p_c1 = ax_b3_twin.plot(pc_idx, cum2_pct, marker='o', linestyle='-', color='darkgreen', linewidth=1.8, label='2-Mat Cumulative (%)')
    p_c2 = ax_b3_twin.plot(pc_idx, cum4_pct, marker='s', linestyle='-', color='darkred', linewidth=1.8, label='4-Mat Cumulative (%)')
    ax_b3_twin.axhline(99.0, color='gray', linestyle=':', label='99% Variance Threshold')
    ax_b3_twin.set_ylabel("Cumulative Explained Variance (%)")
    ax_b3_twin.set_ylim(85, 101)

    ax_b3.set_title("PCA Dimensionality Results", fontsize=11, fontweight='bold')
    
    # Combined legend
    handles_b3, labels_b3 = ax_b3.get_legend_handles_labels()
    handles_twin, labels_twin = ax_b3_twin.get_legend_handles_labels()
    ax_b3.legend(handles_b3 + handles_twin, labels_b3 + labels_twin, loc='center right', fontsize='x-small')

    # Executive Supertitle
    fig.suptitle(f"Pure-Pixel Foliar Differentiability: Spectral Complexity vs. PCA Material Dimensionality Contrast\n" + \
                 f"Linear SNR: {res_2mat['snr']} | Shading Variability: {res_2mat['shade_variability']:.2%}",
                 fontsize=14, fontweight='bold', y=0.98)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved executive contrast figure to: {output_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Simulate pure-pixel foliar differentiability comparing healthy maple species with tar spot pathology.")
    parser.add_argument("--tile_size", type=int, default=DEFAULT_TILE_SIZE, help="Square tile dimension (default: 15).")
    parser.add_argument("--snr", type=float, default=DEFAULT_SNR, help="Target linear SNR (default: 200).")
    parser.add_argument("--shot_fraction", type=float, default=DEFAULT_SHOT_NOISE_FRACTION, help="Shot noise fraction (default: 0.70).")
    parser.add_argument("--shade_variability", type=float, default=DEFAULT_SHADE_VARIABILITY, help="Shading variability (default: 0.03 for 3%%).")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for noise generation.")
    parser.add_argument("--sensor", type=str, default=DEFAULT_SENSOR, help="Sensor response name.")
    parser.add_argument("--num_endmembers", type=int, default=DEFAULT_NUM_ENDMEMBERS, help="Number of endmembers to extract.")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Output directory for generated plots.")
    parser.add_argument("--no_show", action="store_true", help="Skip interactive window display (headless execution).")
    args = parser.parse_args()

    # 1. Run 2-Material Healthy Maple Simulation
    res_2mat = simulate_foliar_tile(
        scenario="healthy_2mat",
        tile_size=args.tile_size,
        snr=args.snr,
        shot_noise_fraction=args.shot_fraction,
        shade_variability=args.shade_variability,
        num_endmembers=args.num_endmembers,
        seed=args.seed,
        sensor_name=args.sensor
    )

    # 2. Run 4-Material Pathological (Tar Spot Replacement) Simulation
    res_4mat = simulate_foliar_tile(
        scenario="pathological_4mat",
        tile_size=args.tile_size,
        snr=args.snr,
        shot_noise_fraction=args.shot_fraction,
        shade_variability=args.shade_variability,
        num_endmembers=args.num_endmembers,
        seed=args.seed,
        sensor_name=args.sensor
    )

    # 3. Print Structured Comparison Table in Console
    display_comparison_table(
        res_2mat=res_2mat,
        res_4mat=res_4mat,
        num_endmembers=args.num_endmembers
    )

    # 4. Generate Publication-Quality Plots
    p1 = os.path.join(args.output_dir, "foliar_complexity_2mat_healthy.png")
    plot_foliar_scenario(res_2mat, output_path=p1, show_plot=not args.no_show)

    p2 = os.path.join(args.output_dir, "foliar_complexity_4mat_pathological.png")
    plot_foliar_scenario(res_4mat, output_path=p2, show_plot=not args.no_show)

    p3 = os.path.join(args.output_dir, "foliar_complexity_healthy_vs_pathological_contrast.png")
    plot_contrast_figure(res_2mat, res_4mat, output_path=p3, show_plot=not args.no_show)


if __name__ == "__main__":
    main()
