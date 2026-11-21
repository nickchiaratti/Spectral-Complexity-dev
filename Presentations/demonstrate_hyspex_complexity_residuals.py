#!/usr/bin/env python3
"""
demonstrate_hyspex_complexity_residuals.py
Evaluation of Spectral Complexity Cholesky MaxD with Residual Tracking (RSS) and Endmember Indices

Demonstrates real and simulated HySpex results across all core scenarios:
  1. Homogeneous Calibration Panel: Real Airborne Tile (1059, 258) vs. Physical Simulation (CALTARGET)
  2. Structural Boundary Edge: Real Airborne Tile (1108, 325) vs. Physical Simulation (Panel / Asphalt)
  3. Foliar Complexity Contrasts:
     - Single Leaf (1-Material Shading)
     - Healthy Tile (2-Material Species Boundary)
     - Pathological Tile (4-Material Tar Spot Infection)
  4. Multi-Target Foliar Complexity (5-Material Tile: 4 Unique Biological Species + 1 Replicate Scan)

Plots orthogonal heights h_k alongside:
  - indices_arr: Spatial coordinates of extracted endmember pivots on the tile imagery
  - rss_arr: Total residual sum of squares ||P_k^perp Y||_F^2 and pooled residual noise standard deviation sigma_k
  - Sensor noise floor baselines and drop ratios rho_k = sigma_{k-1} / sigma_k

Designed for IDE execution:
  - Run directly in VS Code / PyCharm (Run / F5).
  - Configurable constants below.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import h5py

# Ensure repository packages are in Python path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))
HYSPEX_DIR = os.path.join(REPO_ROOT, "hyspex")
SIM_DIR = os.path.join(REPO_ROOT, "simulated-data-test")
PRES_DIR = os.path.join(REPO_ROOT, "Presentations", "2026-10-27")

for d in [REPO_ROOT, HYSPEX_DIR, SIM_DIR, PRES_DIR]:
    if d not in sys.path:
        sys.path.insert(0, d)

from SpecComplex import maximumDistance_cholesky
from compare_real_tile_vs_simulated import (
    retrieve_real_tile,
    generate_simulated_tile,
    DEFAULT_H5_PATH
)
from simulate_foliar_complexity import simulate_foliar_tile
import generate_slide5_foliar_complexity_multi_sensor as s5
from simulate_inset_squares import add_wavelength_scaled_noise

# =============================================================================
# USER CONFIGURATION PARAMETERS (Adjust for fine-tuning in IDE)
# =============================================================================
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output_hyspex_residuals")
DEFAULT_DPI = 300
NUM_ENDMEMBERS = 8
TILE_SIZE = 15
SNR = 200
SHOT_NOISE_FRACTION = 0.70
SHADE_VARIABILITY = 0.06
SEED = 42

# Palette colors for endmembers
EM_COLORS = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']


def extract_rgb_thumbnail(tile_cube, wavelengths_nm=None):
    """
    Constructs an RGB visualization (H, W, 3) from a hyperspectral cube (B, H, W).
    Uses true-color wavelengths (650, 550, 480 nm) with 1st-99th percentile stretching.
    """
    bands, height, width = tile_cube.shape
    if wavelengths_nm is not None and len(wavelengths_nm) == bands:
        wv = np.array(wavelengths_nm)
        idx_r = int(np.argmin(np.abs(wv - 650.0)))
        idx_g = int(np.argmin(np.abs(wv - 550.0)))
        idx_b = int(np.argmin(np.abs(wv - 480.0)))
    else:
        idx_r = int(bands * 0.6)
        idx_g = int(bands * 0.4)
        idx_b = int(bands * 0.2)

    r_band = tile_cube[idx_r].copy()
    g_band = tile_cube[idx_g].copy()
    b_band = tile_cube[idx_b].copy()

    rgb = np.stack([r_band, g_band, b_band], axis=-1)
    for c in range(3):
        v = rgb[..., c]
        p_low, p_high = np.nanpercentile(v, 1.0), np.nanpercentile(v, 99.0)
        if p_high > p_low:
            rgb[..., c] = np.clip((v - p_low) / (p_high - p_low), 0.0, 1.0)
        else:
            rgb[..., c] = np.clip(v, 0.0, 1.0)

    return rgb


def compute_residual_metrics(rss_arr, heights_arr, B, N):
    """
    Computes degree-of-freedom pooled residual standard deviation,
    variance drop ratios, and analytical extreme-value noise floors.
    """
    k_len = len(rss_arr)
    sigmas = np.zeros(k_len, dtype=np.float64)
    ratios = np.ones(k_len, dtype=np.float64)
    h_noise = np.zeros(k_len, dtype=np.float64)

    for k in range(1, k_len):
        dof = (B - k) * (N - k)
        s = np.sqrt(rss_arr[k] / dof) if dof > 0 else 0.0
        sigmas[k] = s

        if k >= 2 and sigmas[k] > 0:
            ratios[k] = sigmas[k - 1] / sigmas[k]

        N_cand = max(N - k, 1)
        ev_term = (B - k) + 2.0 * np.sqrt((B - k) * np.log(N_cand)) + 2.0 * np.log(N_cand)
        h_noise[k] = s * np.sqrt(ev_term)

    return sigmas, ratios, h_noise


def plot_scenario_analysis(tile_data, wavelengths, scenario_name, expected_k, out_path, num_endmembers=NUM_ENDMEMBERS):
    """
    Generates a 4-panel figure:
      Panel 1: Spatial tile with numbered markers showing indices_arr positions.
      Panel 2: Endmember reflectance spectra corresponding to indices_arr.
      Panel 3: Dual-axis plot of orthogonal heights h_k vs. residual sum of squares (RSS_k) and sigma_k.
      Panel 4: Residual variance drop ratio rho_k = sigma_{k-1} / sigma_k with noise floor baseline.
    """
    B, H, W = tile_data.shape
    N = H * W
    pixels_flat = tile_data.reshape(B, N)

    # Execute updated Cholesky maxD
    em, vols, heights, indices, rss = maximumDistance_cholesky(pixels_flat, num_endmembers=num_endmembers)
    k_len = len(heights)

    sigmas, ratios, h_noise = compute_residual_metrics(rss, heights, B, N)

    # Convert flat indices to (row, col) coordinates
    rows = indices // W
    cols = indices % W

    fig = plt.figure(figsize=(21, 5.2), dpi=DEFAULT_DPI)
    gs = GridSpec(1, 4, width_ratios=[1.0, 1.25, 1.35, 1.15], wspace=0.34)

    # Panel 1: Spatial Tile with Endmember Overlay
    ax_map = fig.add_subplot(gs[0])
    rgb_thumb = extract_rgb_thumbnail(tile_data, wavelengths)
    ax_map.imshow(rgb_thumb)
    ax_map.set_title(f"Spatial Tile ({H}x{W})\nEndmember Pivot Locations", fontsize=10.5, fontweight='bold')

    for k in range(k_len):
        color = EM_COLORS[k % len(EM_COLORS)]
        marker_label = f"v_{k}" if k > 0 else "v_0"
        ax_map.scatter(cols[k], rows[k], color=color, s=110, edgecolor='black', linewidth=1.2, zorder=5)
        ax_map.text(cols[k] + 0.4, rows[k] - 0.4, marker_label, color='white', fontsize=8, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.15', facecolor=color, alpha=0.85, edgecolor='black', lw=0.6),
                    zorder=6)

    ax_map.set_xticks(range(0, W, max(1, W // 5)))
    ax_map.set_yticks(range(0, H, max(1, H // 5)))
    ax_map.set_xlabel("Pixel Column", fontsize=9, fontweight='bold')
    ax_map.set_ylabel("Pixel Row", fontsize=9, fontweight='bold')

    # Panel 2: Endmember Spectra
    ax_spec = fig.add_subplot(gs[1])
    wv = wavelengths if wavelengths is not None else np.arange(B)
    for k in range(k_len):
        color = EM_COLORS[k % len(EM_COLORS)]
        lbl = f"v_{k} (idx={indices[k]}, r={rows[k]}, c={cols[k]})" if k > 0 else f"v_0 (origin: idx={indices[0]})"
        ax_spec.plot(wv, em[:, k], label=lbl, color=color, linewidth=1.6)

    ax_spec.set_title(f"Extracted Endmember Spectra\n(k = 0..{k_len-1})", fontsize=10.5, fontweight='bold')
    ax_spec.set_xlabel("Wavelength (nm)", fontsize=9, fontweight='bold')
    ax_spec.set_ylabel("Surface Reflectance", fontsize=9, fontweight='bold')
    ax_spec.grid(True, linestyle=':', alpha=0.5)
    ax_spec.legend(loc='upper right', fontsize=7.0, framealpha=0.9)

    # Panel 3: Dual-Axis h_k vs. RSS_k and sigma_k
    ax_h = fig.add_subplot(gs[2])
    k_axis = np.arange(1, k_len)

    line1 = ax_h.plot(k_axis, heights[1:], 'o-', color='#1f77b4', linewidth=2.0, markersize=6.5, label='Height $h_k$ [Left]')
    ax_h.set_yscale('log')
    ax_h.set_xlabel("Endmember Index ($k$)", fontsize=9, fontweight='bold')
    ax_h.set_ylabel("Orthogonal Height $h_k$ [Log Scale]", fontsize=9, fontweight='bold', color='#1f77b4')
    ax_h.tick_params(axis='y', labelcolor='#1f77b4')
    ax_h.set_xticks(k_axis)
    ax_h.grid(True, which='both', linestyle=':', alpha=0.4)

    # Secondary Axis for Pooled Residual Noise (sigma_k) and RSS
    ax_r = ax_h.twinx()
    line2 = ax_r.plot(k_axis, sigmas[1:], 's--', color='#d62728', linewidth=1.8, markersize=5.5, label=r'Residual $\hat{\sigma}_k$ [Right]')
    ax_r.set_ylabel(r"Pooled Noise $\hat{\sigma}_k = \sqrt{\mathrm{RSS}_k / \mathrm{dof}}$", fontsize=9, fontweight='bold', color='#d62728')
    ax_r.tick_params(axis='y', labelcolor='#d62728')
    ax_r.set_yscale('log')

    # Add vertical indicator for expected physical transition
    vl = None
    if 1 <= expected_k < k_len:
        vl = ax_h.axvline(expected_k, color='green', linestyle=':', linewidth=1.5, alpha=0.8,
                          label=f'Intrinsic Rank $k^*={expected_k}$')

    # Combined Legend
    lines = line1 + line2
    if vl is not None:
        lines.append(vl)
    labels = [l.get_label() for l in lines]
    ax_h.legend(lines, labels, loc='upper right', fontsize=7.5, framealpha=0.92)
    ax_h.set_title(f"{scenario_name}\nOrthogonal Heights $h_k$ & Residual Noise Floor", fontsize=10.5, fontweight='bold')

    # Panel 4: Residual Variance Drop Ratio rho_k
    ax_drop = fig.add_subplot(gs[3])
    k_ratio_axis = np.arange(2, k_len)
    ratio_vals = ratios[2:]
    bar_colors = ['#2ca02c' if r > 1.5 else '#7f7f7f' for r in ratio_vals]
    bars = ax_drop.bar(k_ratio_axis, ratio_vals, color=bar_colors, width=0.55, edgecolor='black', linewidth=0.8)
    ax_drop.axhline(1.0, color='red', linestyle='--', linewidth=1.2, label='Noise Floor (1.0x)')

    for b, r in zip(bars, ratio_vals):
        y_text = b.get_height() + (max(ratio_vals) * 0.025 if max(ratio_vals) > 0 else 0.1)
        ax_drop.text(b.get_x() + b.get_width() / 2.0, y_text, f"{r:.2f}x",
                     ha='center', va='bottom', fontsize=7.5, fontweight='bold')

    ax_drop.set_xticks(k_ratio_axis)
    ax_drop.set_xlabel("Endmember Pivot ($k$)", fontsize=9, fontweight='bold')
    ax_drop.set_ylabel(r"Drop Ratio $\rho_k = \hat{\sigma}_{k-1}/\hat{\sigma}_k$", fontsize=9, fontweight='bold')
    ax_drop.set_title(r"Variance Drop Ratio $\rho_k$" + "\n" + r"($\rho_k \gg 1$: Signal | $\rho_k \approx 1$: Noise)",
                      fontsize=10.0, fontweight='bold')
    max_rv = max(ratio_vals) if len(ratio_vals) > 0 and max(ratio_vals) > 0 else 2.0
    ax_drop.set_ylim(0, max_rv * 1.30)
    ax_drop.grid(True, linestyle=':', alpha=0.4)
    ax_drop.legend(loc='upper right', fontsize=7.5, framealpha=0.9)

    plt.subplots_adjust(left=0.05, right=0.96, top=0.88, bottom=0.12, wspace=0.34)
    plt.savefig(out_path, dpi=DEFAULT_DPI, bbox_inches='tight')
    plt.close()

    # Print summary table to console
    print(f"\n================================================================================")
    print(f"Scenario: {scenario_name} (B={B}, N={N}, Expected k*={expected_k})")
    print(f"Saved: {out_path}")
    print(f"{'k':<3} | {'Index':<6} {'(Row,Col)':<9} | {'Height h_k':<12} {'RSS_k':<12} {'sigma_k':<11} {'Drop Ratio':<11} | {'Regime'}")
    print(f"--------------------------------------------------------------------------------")
    for k in range(k_len):
        if k == 0:
            print(f"{0:<3} | {indices[0]:<6} ({rows[0]},{cols[0]}):<9 | {'--':<12} {rss[0]:<12.5f} {'--':<11} {'--':<11} | Origin (v_0)")
        else:
            regime = "SIGNAL" if (k <= expected_k) else "NOISE FLOOR"
            ratio_str = f"{ratios[k]:.2f}x" if k >= 2 else "--"
            print(f"{k:<3} | {indices[k]:<6} ({rows[k]},{cols[k]}):<9 | {heights[k]:<12.5f} {rss[k]:<12.5f} {sigmas[k]:<11.5f} {ratio_str:<11} | {regime}")

    return {
        'scenario_name': scenario_name,
        'expected_k': expected_k,
        'heights': heights,
        'rss': rss,
        'sigmas': sigmas,
        'ratios': ratios,
        'indices': indices
    }


def generate_executive_overview_figure(scenario_records, out_path):
    """
    Constructs a 2x3 executive summary visual for slides comparing
    h_k with sigma_k and drop ratios rho_k across all core scenarios.
    Each of the 6 scenario cells contains:
      - Top sub-panel: Orthogonal height h_k (left) & pooled noise sigma_k (right)
      - Bottom sub-panel: Residual variance drop ratio rho_k = sigma_{k-1} / sigma_k
    """
    fig = plt.figure(figsize=(19, 10.5), dpi=DEFAULT_DPI)
    outer_gs = GridSpec(2, 3, wspace=0.34, hspace=0.38, left=0.05, right=0.95, top=0.93, bottom=0.07)

    for i, rec in enumerate(scenario_records[:6]):
        inner_gs = outer_gs[i].subgridspec(2, 1, height_ratios=[2.2, 1.0], hspace=0.10)
        ax_top = fig.add_subplot(inner_gs[0])
        ax_bot = fig.add_subplot(inner_gs[1], sharex=ax_top)

        k_len = len(rec['heights'])
        k_axis = np.arange(1, k_len)

        # Top Primary Y-axis: Height h_k
        l1 = ax_top.plot(k_axis, rec['heights'][1:], 'o-', color='#1f77b4', linewidth=2.0, markersize=5.5,
                         label=r'Height $h_k$ [Left]')
        ax_top.set_yscale('log')
        ax_top.tick_params(axis='x', labelbottom=False)
        ax_top.set_ylabel(r"Height $h_k$ [Log]", fontsize=8.5, fontweight='bold', color='#1f77b4')
        ax_top.tick_params(axis='y', labelcolor='#1f77b4')
        ax_top.grid(True, which='both', linestyle=':', alpha=0.35)

        # Top Secondary Y-axis: Pooled Noise sigma_k
        ax_top_r = ax_top.twinx()
        l2 = ax_top_r.plot(k_axis, rec['sigmas'][1:], 's--', color='#d62728', linewidth=1.8, markersize=4.8,
                           label=r'Noise $\hat{\sigma}_k$ [Right]')
        ax_top_r.set_yscale('log')
        ax_top_r.set_ylabel(r"$\hat{\sigma}_k$", fontsize=8.5, fontweight='bold', color='#d62728')
        ax_top_r.tick_params(axis='y', labelcolor='#d62728')

        # Highlight transition
        exp_k = rec['expected_k']
        vl = None
        if 1 <= exp_k < k_len:
            vl = ax_top.axvline(exp_k, color='green', linestyle=':', linewidth=1.4, alpha=0.85,
                                label=f'Rank $k^*={exp_k}$')

        lines = l1 + l2
        if vl is not None:
            lines.append(vl)
        labels = [l.get_label() for l in lines]
        ax_top.legend(lines, labels, loc='upper right', fontsize=6.5, framealpha=0.9)
        ax_top.set_title(f"({chr(97+i)}) {rec['scenario_name']}", fontsize=9.5, fontweight='bold', pad=6)

        # Bottom Sub-panel: Residual Drop Ratio rho_k
        k_r = np.arange(2, k_len)
        ratios_plot = rec['ratios'][2:]
        colors_bar = ['#2ca02c' if r > 1.5 else '#7f7f7f' for r in ratios_plot]
        bars = ax_bot.bar(k_r, ratios_plot, color=colors_bar, width=0.55, edgecolor='black', linewidth=0.6)
        ax_bot.axhline(1.0, color='red', linestyle='--', linewidth=1.0, label=r'Noise Floor ($1.0x$)')

        for b, r in zip(bars, ratios_plot):
            y_pos = b.get_height() + (max(ratios_plot) * 0.04 if max(ratios_plot) > 0 else 0.1)
            ax_bot.text(b.get_x() + b.get_width() / 2.0, y_pos, f"{r:.1f}x",
                        ha='center', va='bottom', fontsize=7.0, fontweight='bold')

        ax_bot.set_ylabel(r"$\rho_k$", fontsize=8.5, fontweight='bold')
        ax_bot.set_xticks(k_axis)
        ax_bot.set_xlabel("Endmember Pivot ($k$)", fontsize=8.5, fontweight='bold')
        max_r = max(ratios_plot) if len(ratios_plot) > 0 and max(ratios_plot) > 0 else 2.0
        ax_bot.set_ylim(0, max_r * 1.35)
        ax_bot.grid(True, linestyle=':', alpha=0.35)
        if i == 0:
            ax_bot.legend(loc='upper right', fontsize=6.0, framealpha=0.9)

    plt.savefig(out_path, dpi=DEFAULT_DPI, bbox_inches='tight')
    plt.close()
    print(f"\n[Done] Executive overview figure saved: {out_path}")


def run_all_scenarios():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("================================================================================")
    print("DEMONSTRATING HYSPEX COMPLEXITY RESIDUALS AND ENDMEMBER INDICES")
    print(f"Output Directory: {OUTPUT_DIR}")
    print("================================================================================")

    records = []

    # 1. Slide 1 Scenario A: Real White Panel vs. Simulated White Panel
    print("\n--- 1. Homogeneous Calibration Panel ---")
    r_panel = retrieve_real_tile(DEFAULT_H5_PATH, map_x=1059, map_y=258, tile_size=TILE_SIZE, num_endmembers=NUM_ENDMEMBERS)
    rec_r_panel = plot_scenario_analysis(
        tile_data=r_panel['tile_data'],
        wavelengths=r_panel['wavelengths_nm'],
        scenario_name="Real Airborne HySpex: White Panel (1059, 258)",
        expected_k=1,
        out_path=os.path.join(OUTPUT_DIR, "scenario1_real_panel_residuals.png")
    )
    records.append(rec_r_panel)

    s_panel = generate_simulated_tile(
        sensor_name="HySpexMjolnir",
        material_name="CALTARGET:big_white_panel",
        layout="single",
        tile_size=TILE_SIZE,
        snr=SNR,
        num_endmembers=NUM_ENDMEMBERS,
        shot_noise_fraction=SHOT_NOISE_FRACTION,
        shade_variability=0.055,
        seed=SEED
    )
    rec_s_panel = plot_scenario_analysis(
        tile_data=s_panel['noisy_tile'],
        wavelengths=s_panel['wavelengths_nm'],
        scenario_name="Simulated HySpex: White Panel",
        expected_k=1,
        out_path=os.path.join(OUTPUT_DIR, "scenario1_sim_panel_residuals.png")
    )
    records.append(rec_s_panel)

    # 2. Slide 1 Scenario B: Real Boundary Edge vs. Simulated Boundary Edge
    print("\n--- 2. Structural Boundary Edge ---")
    r_boundary = retrieve_real_tile(DEFAULT_H5_PATH, map_x=1108, map_y=325, tile_size=TILE_SIZE, num_endmembers=NUM_ENDMEMBERS)
    rec_r_bnd = plot_scenario_analysis(
        tile_data=r_boundary['tile_data'],
        wavelengths=r_boundary['wavelengths_nm'],
        scenario_name="Real Airborne HySpex: Boundary Edge (1108, 325)",
        expected_k=2,
        out_path=os.path.join(OUTPUT_DIR, "scenario2_real_boundary_residuals.png")
    )
    records.append(rec_r_bnd)

    s_boundary = generate_simulated_tile(
        sensor_name="HySpexMjolnir",
        material_name="CALTARGET:big_white_panel",
        material_2_name="USGS:Asphalt",
        layout="split_diag",
        tile_size=TILE_SIZE,
        snr=SNR,
        num_endmembers=NUM_ENDMEMBERS,
        shot_noise_fraction=SHOT_NOISE_FRACTION,
        shade_variability=0.055,
        seed=SEED
    )
    rec_s_bnd = plot_scenario_analysis(
        tile_data=s_boundary['noisy_tile'],
        wavelengths=s_boundary['wavelengths_nm'],
        scenario_name="Simulated HySpex: Panel / Asphalt Boundary",
        expected_k=2,
        out_path=os.path.join(OUTPUT_DIR, "scenario2_sim_boundary_residuals.png")
    )
    records.append(rec_s_bnd)

    # 3. Slide 3 & 4: Foliar Contrasts (Single Leaf, Healthy 2-Mat, Pathological 4-Mat)
    print("\n--- 3. Foliar Complexity Contrasts ---")
    res_1mat = simulate_foliar_tile(scenario="Single_Mat", tile_size=TILE_SIZE, snr=SNR,
                                    shot_noise_fraction=SHOT_NOISE_FRACTION, shade_variability=SHADE_VARIABILITY,
                                    num_endmembers=NUM_ENDMEMBERS, seed=SEED, sensor_name="cvHySpexMjolnir")
    rec_fol_1 = plot_scenario_analysis(
        tile_data=res_1mat['noisy_tile'],
        wavelengths=res_1mat['wavelengths_nm'],
        scenario_name="Foliar: Single Leaf (1-Material Shading)",
        expected_k=1,
        out_path=os.path.join(OUTPUT_DIR, "scenario3_foliar_single_leaf_residuals.png")
    )
    records.append(rec_fol_1)

    res_2mat = simulate_foliar_tile(scenario="healthy_2mat", tile_size=TILE_SIZE, snr=SNR,
                                    shot_noise_fraction=SHOT_NOISE_FRACTION, shade_variability=SHADE_VARIABILITY,
                                    num_endmembers=NUM_ENDMEMBERS, seed=SEED, sensor_name="cvHySpexMjolnir")
    rec_fol_2 = plot_scenario_analysis(
        tile_data=res_2mat['noisy_tile'],
        wavelengths=res_2mat['wavelengths_nm'],
        scenario_name="Foliar: Healthy 2-Material (Norway vs. Silver Maple)",
        expected_k=2,
        out_path=os.path.join(OUTPUT_DIR, "scenario3_foliar_healthy_2mat_residuals.png")
    )
    records.append(rec_fol_2)

    res_4mat = simulate_foliar_tile(scenario="pathological_4mat", tile_size=TILE_SIZE, snr=SNR,
                                    shot_noise_fraction=SHOT_NOISE_FRACTION, shade_variability=SHADE_VARIABILITY,
                                    num_endmembers=NUM_ENDMEMBERS, seed=SEED, sensor_name="cvHySpexMjolnir")
    rec_fol_4 = plot_scenario_analysis(
        tile_data=res_4mat['noisy_tile'],
        wavelengths=res_4mat['wavelengths_nm'],
        scenario_name="Foliar: Pathological 4-Material (Tar Spot Infection)",
        expected_k=4,
        out_path=os.path.join(OUTPUT_DIR, "scenario3_foliar_pathological_4mat_residuals.png")
    )
    records.append(rec_fol_4)

    # 4. Slide 5: Multi-Target 5-Material Tile on HySpex
    print("\n--- 4. Multi-Target 5-Material Foliar Tile ---")
    clean_wv, clean_specs, _ = s5.load_sensor_spectra('cvHySpexMjolnir')
    clean_tile, _ = s5.build_5material_tile(clean_specs, tile_size=TILE_SIZE)
    np.random.seed(SEED)
    shade_map = np.random.uniform(1.0 - SHADE_VARIABILITY, 1.0, (TILE_SIZE, TILE_SIZE)).astype(np.float32)
    tile_shaded = clean_tile * shade_map[None, :, :]
    noisy_tile, _ = add_wavelength_scaled_noise(tile_shaded, snr=SNR, shot_noise_fraction=SHOT_NOISE_FRACTION, seed=SEED)
    rec_fol_5 = plot_scenario_analysis(
        tile_data=noisy_tile,
        wavelengths=clean_wv,
        scenario_name="Foliar: 5-Material Tile (4 Species + 1 Replicate)",
        expected_k=4,
        out_path=os.path.join(OUTPUT_DIR, "scenario4_foliar_5mat_residuals.png")
    )
    records.append(rec_fol_5)

    # Generate Executive Summary Figure (Slide-ready 2x3 grid)
    exec_out = os.path.join(OUTPUT_DIR, "hyspex_complexity_residuals_executive_overview.png")
    # Select the 6 most diverse representative scenarios
    selected_for_exec = [rec_r_panel, rec_r_bnd, rec_fol_1, rec_fol_2, rec_fol_4, rec_fol_5]
    generate_executive_overview_figure(selected_for_exec, exec_out)

    print("\nAll scenario demonstrations completed successfully.")
    print(f"Generated figures saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    run_all_scenarios()
