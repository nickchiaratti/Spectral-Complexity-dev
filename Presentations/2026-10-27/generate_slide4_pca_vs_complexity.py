#!/usr/bin/env python3
"""
generate_slide4_pca_vs_complexity.py
Slide 4 Embedded Visual: Dimensionality Metric Comparison (MaxD OSP vs. PCA Scree)

This script generates the high-resolution comparison figure for Slide 4 of the
October 27, 2026 Spectral Complexity presentation:
  - Output: C:\\satelliteImagery\\simulated\\foliar_comparison\\spectral_complexity_vs_pca_scree.png

Plot Architecture:
  - Subplot (a): Dual-Axis PCA Scree (Individual & Cumulative Variance):
      • Primary (Left) Y-Axis [Log Scale %]: Individual explained variance ratio for each PC.
        Demonstrates that PC 1 absorbs >88–97% of variance, leaving PC 2 (species) and
        PC 4 (disease necrosis) below or near heuristic cutoffs.
      • Secondary (Right) Y-Axis [Linear Scale %]: Cumulative explained variance trajectory.
        Demonstrates that standard stopping rules (e.g., 95% or 99% cumulative variance)
        truncate the dimensionality prematurely at k=1 or k=2, squashing physical foliar
        dimensions before material boundaries are resolved.
  - Subplot (b): MaxD Residual Decays: Physical Scale Contrast [SymLog Scale]:
      • Plots the orthogonal subspace distance trajectories (h_r) for:
          1. Single Leaf (1-Mat): Intra-material shading spike from 1 to 2 endmembers (h_1 ≈ 0.40)
             collapsing immediately to sensor noise floor at r=3 (h_2 = 0.032 ≈ h_noise).
          2. Healthy Tile (2-Mat): Inter-species boundary (h_1 = 1.55) sustaining orthogonal distance
             at 3.8x higher scale before dropping to noise floor at r=4.
          3. Pathological Tile (4-Mat): Tar spot necrosis (h_1 = 3.89, 9.6x higher scale) sustaining
             three non-zero orthogonal directions before dropping to noise floor at r=5.
      • Sensor Noise Floor baseline (h_noise ≈ 0.026).

Designed for IDE execution:
  - Simply open in VS Code / PyCharm and run (Run / F5).
  - Adjust configuration constants below for fine tuning.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Add repository root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))
HYSPEX_DIR = os.path.join(REPO_ROOT, "hyspex")
SIM_DIR = os.path.join(REPO_ROOT, "simulated-data-test")

for d in [HYSPEX_DIR, SIM_DIR, REPO_ROOT]:
    if d not in sys.path:
        sys.path.insert(0, d)

from simulate_foliar_complexity import simulate_foliar_tile

# =============================================================================
# USER CONFIGURATION PARAMETERS (Adjust for fine-tuning in IDE)
# =============================================================================
DATA_DIR = r"C:\satelliteImagery\ground_truth\NickLeaves_20261001"
OUTPUT_DIR = r"C:\satelliteImagery\simulated\foliar_comparison"
OUTPUT_FILENAME = "spectral_complexity_vs_pca_scree.png"

# Layout Mode: 'dual_panel' (PCA dual-axis + MaxD) or 'three_panel' (Indiv PCA + Cumul PCA + MaxD)
LAYOUT_MODE = "dual_panel"

# Figure Dimensions & Quality
FIG_SIZE_DUAL = (15.2, 5.3)     # Width, Height in inches for dual-panel mode
FIG_SIZE_TRI = (16.8, 5.2)      # Width, Height in inches for three-panel mode
DPI = 300                       # Output resolution
BAR_WIDTH = 0.24                # Width of individual PCA scree bars

# Styling Colors (Okabe-Ito / Publication standard)
COLOR_SINGLE = "#f39c12"        # Amber / Orange for Single Leaf (1-Material)
COLOR_HEALTHY = "#27ae60"       # Green for Healthy (2-Material Species)
COLOR_PATHOLOGICAL = "#c0392b"  # Red for Pathological (4-Material Tar Spot)
COLOR_CUTOFF_INDIV = "black"    # Individual PCA 1% cutoff line
COLOR_CUTOFF_CUMUL = "dimgray"  # Cumulative PCA 99% threshold line
COLOR_NOISE = "gray"            # Sensor noise floor baseline

# Simulation Parameters
TILE_SIZE = 15                  # 15x15 pixels (225 pure pixels)
SNR = 200                       # Sensor signal-to-noise ratio
SHOT_NOISE_FRACTION = 0.70      # Shot noise fraction
SHADE_VARIABILITY = 0.06        # 6% shading variability: np.random.uniform(1.0 - 0.06, 1.0)
NUM_ENDMEMBERS = 11              # Number of MaxD endmembers to extract
RANDOM_SEED = 42                # Reproducible simulation seed
SENSOR_PREFIX = "cvHySpexMjolnir" # Empirical leaf spectra sensor model

# Thresholds & Limits
PCA_CUTOFF_INDIV_PCT = 1.0      # Standard 1% Kaiser individual variance cutoff
PCA_CUTOFF_CUMUL_PCT = 99.0     # Standard 99% cumulative variance threshold
NOISE_FLOOR = 0.026             # Sensor noise floor baseline (h_noise)
TAU_CRIT = 0.010                # 1.0% RMS reflectance contrast per band threshold
SYMLOG_THRESHOLD = 1e-2         # Symlog threshold for orthogonal heights
Y_LIM_INDIV = (0.005, 180)      # Y-axis bounds for individual PCA scree log plot
Y_LIM_CUMUL = (80.0, 102.0)     # Y-axis bounds for cumulative PCA linear plot

# Typography & Sizing
TITLE_FONT_SIZE = 11.5          # Subplot title font size
LABEL_FONT_SIZE = 10.5          # Axis label font size
TICK_FONT_SIZE = 9.5            # Tick label font size
LEGEND_FONT_SIZE = 6.0          # Legend font size
ANNOTATION_FONT_SIZE = 8.0      # Text annotation font size

# Telluric water vapor bands to mask
TELLURIC_RANGES = [(1350, 1450), (1800, 2000),(2420,2500)]


def generate_pca_vs_complexity_figure():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    print(f"\n=======================================================")
    print(f"Generating Slide 4 Visual: PCA vs. MaxD Scree & Residuals")
    print(f"Layout Mode: {LAYOUT_MODE.upper()}")
    print(f"=======================================================")

    # 1. Simulate foliar spatial tiles using simulate_foliar_tile
    print("Simulating foliar tiles using simulate_foliar_tile...")
    res_1mat = simulate_foliar_tile(
        scenario="Single_Mat",
        tile_size=TILE_SIZE,
        snr=SNR,
        shot_noise_fraction=SHOT_NOISE_FRACTION,
        shade_variability=SHADE_VARIABILITY,
        num_endmembers=NUM_ENDMEMBERS,
        seed=RANDOM_SEED,
        sensor_name=SENSOR_PREFIX
    )
    res_2mat = simulate_foliar_tile(
        scenario="healthy_2mat",
        tile_size=TILE_SIZE,
        snr=SNR,
        shot_noise_fraction=SHOT_NOISE_FRACTION,
        shade_variability=SHADE_VARIABILITY,
        num_endmembers=NUM_ENDMEMBERS,
        seed=RANDOM_SEED,
        sensor_name=SENSOR_PREFIX
    )
    res_4mat = simulate_foliar_tile(
        scenario="pathological_4mat",
        tile_size=TILE_SIZE,
        snr=SNR,
        shot_noise_fraction=SHOT_NOISE_FRACTION,
        shade_variability=SHADE_VARIABILITY,
        num_endmembers=NUM_ENDMEMBERS,
        seed=RANDOM_SEED,
        sensor_name=SENSOR_PREFIX
    )

    # Retain references to simulated tiles
    tile_1 = res_1mat['noisy_tile']
    tile_2 = res_2mat['noisy_tile']
    tile_4 = res_4mat['noisy_tile']

    # 2. Extract PCA Explained Variance Ratios (Individual and Cumulative)
    var_1 = res_1mat['pca_evr'][:5] * 100
    cum_1 = res_1mat['pca_cum_evr'][:5] * 100

    var_2 = res_2mat['pca_evr'][:5] * 100
    cum_2 = res_2mat['pca_cum_evr'][:5] * 100

    var_4 = res_4mat['pca_evr'][:5] * 100
    cum_4 = res_4mat['pca_cum_evr'][:5] * 100

    print(f"Single Leaf PCA Indiv (%): {[round(float(v), 2) for v in var_1]} | Cumul (%): {[round(float(v), 2) for v in cum_1]}")
    print(f"Healthy 2-Mat PCA Indiv (%): {[round(float(v), 2) for v in var_2]} | Cumul (%): {[round(float(v), 2) for v in cum_2]}")
    print(f"Infected 4-Mat PCA Indiv (%): {[round(float(v), 2) for v in var_4]} | Cumul (%): {[round(float(v), 2) for v in cum_4]}")

    # 3. Extract MaxD Residual Subspace Distances
    r_idx = np.arange(1, 10) # Endmember Count r = 1 to 5
    h1_curve = res_1mat['computed_heights'][:9]
    h2_curve = res_2mat['computed_heights'][:9]
    h4_curve = res_4mat['computed_heights'][:9]

    print(fr"Single Leaf ($h_k$ (r=1..5): {[round(float(v), 4) for v in h1_curve]}")
    print(fr"Healthy 2-Mat $h_k$ (r=1..5): {[round(float(v), 4) for v in h2_curve]}")
    print(fr"Infected 4-Mat $h_k$ (r=1..5): {[round(float(v), 4) for v in h4_curve]}")

    x_pca = np.arange(1, 6)

    # 5. Build Matplotlib Visual
    if LAYOUT_MODE == "dual_panel":
        fig, (ax_pca, ax_maxd) = plt.subplots(1, 2, figsize=FIG_SIZE_DUAL, dpi=DPI)

        # ---------------------------------------------------------------------
        # Panel (a): Dual-Axis PCA (Individual Bars + Cumulative Lines)
        # ---------------------------------------------------------------------
        # Primary (Left) Axis: Individual Explained Variance (Log Scale)
        b1 = ax_pca.bar(x_pca - BAR_WIDTH, var_1, BAR_WIDTH, label='Single Leaf (Indiv)', 
                        color=COLOR_SINGLE, alpha=0.55, edgecolor=COLOR_SINGLE, linewidth=1.2)
        b2 = ax_pca.bar(x_pca, var_2, BAR_WIDTH, label='Healthy 2-Mat (Indiv)', 
                        color=COLOR_HEALTHY, alpha=0.55, edgecolor=COLOR_HEALTHY, linewidth=1.2)
        b3 = ax_pca.bar(x_pca + BAR_WIDTH, var_4, BAR_WIDTH, label='Infected 4-Mat (Indiv)', 
                        color=COLOR_PATHOLOGICAL, alpha=0.55, edgecolor=COLOR_PATHOLOGICAL, linewidth=1.2)
        #l_cut = ax_pca.axhline(PCA_CUTOFF_INDIV_PCT, color=COLOR_CUTOFF_INDIV, linestyle=':', 
        #                       linewidth=1.4, label=f'Kaiser Cutoff ({PCA_CUTOFF_INDIV_PCT}% Indiv)')

        ax_pca.set_yscale('log')
        ax_pca.set_ylim(Y_LIM_INDIV[0], Y_LIM_INDIV[1])
        ax_pca.set_xlabel('Principal Component Index', fontsize=LABEL_FONT_SIZE, fontweight='bold')
        ax_pca.set_ylabel('Individual Explained Variance (%) [Log Scale]', fontsize=LABEL_FONT_SIZE, fontweight='bold')
        ax_pca.set_title('PCA Scree: Individual Bars vs. Cumulative Lines', fontsize=TITLE_FONT_SIZE, fontweight='bold', pad=10)
        ax_pca.set_xticks(x_pca)
        ax_pca.set_xticklabels([f'PC {i}' for i in x_pca], fontsize=TICK_FONT_SIZE)
        ax_pca.tick_params(axis='both', labelsize=TICK_FONT_SIZE)
        ax_pca.grid(True, which='both', linestyle=':', alpha=0.35)

        # Annotations on Individual Bars
        #ax_pca.text(2, float(var_2[1]) * 1.35, f'{float(var_2[1]):.2f}%\n(Lost)', 
        #            ha='center', fontsize=ANNOTATION_FONT_SIZE, color=COLOR_HEALTHY, fontweight='bold')
        #ax_pca.text(4 + BAR_WIDTH, 0.012, '<0.05%\n(Lost)', 
        #            ha='center', fontsize=ANNOTATION_FONT_SIZE, color=COLOR_PATHOLOGICAL, fontweight='bold')

        # Secondary (Right) Axis: Cumulative Explained Variance (Linear Scale)
        ax_pca_r = ax_pca.twinx()
        l1, = ax_pca_r.plot(x_pca, cum_1, 'o-', color=COLOR_SINGLE, linewidth=2.2, markersize=6.5, 
                            label='Single Leaf (Cumul)')
        l2, = ax_pca_r.plot(x_pca, cum_2, 's-', color=COLOR_HEALTHY, linewidth=2.2, markersize=6.5, 
                            label='Healthy 2-Mat (Cumul)')
        l3, = ax_pca_r.plot(x_pca, cum_4, '^-', color=COLOR_PATHOLOGICAL, linewidth=2.2, markersize=6.5, 
                            label='Infected 4-Mat (Cumul)')
        l_cumcut = ax_pca_r.axhline(PCA_CUTOFF_CUMUL_PCT, color=COLOR_CUTOFF_CUMUL, linestyle='--', 
                                    linewidth=1.3, label=f'Threshold ({PCA_CUTOFF_CUMUL_PCT}% Cumul)')

        ax_pca_r.set_ylabel('Cumulative Explained Variance (%) [Linear]', fontsize=LABEL_FONT_SIZE, 
                            fontweight='bold', color=COLOR_CUTOFF_CUMUL)
        ax_pca_r.set_ylim(Y_LIM_CUMUL[0], Y_LIM_CUMUL[1])
        ax_pca_r.tick_params(axis='y', labelcolor=COLOR_CUTOFF_CUMUL, labelsize=TICK_FONT_SIZE)

        # Combined Legend (neatly framed in lower center)
        handles = [b1, l1, b2, l2, b3, l3, l_cumcut]
        labels = [h.get_label() for h in handles]
        ax_pca.legend(handles, labels, loc='right', ncol=2, fontsize=LEGEND_FONT_SIZE - 0.5, 
                      framealpha=0.92, facecolor='white', edgecolor='#cccccc')

        # ---------------------------------------------------------------------
        # Panel (b): MaxD Orthogonal Residual Decays (Physical Scale Contrast)
        # ---------------------------------------------------------------------
        B = len(res_1mat['wavelengths_nm'])
        h_crit = TAU_CRIT * np.sqrt(B)
        ax_maxd.plot(r_idx, h1_curve, 'o-', color=COLOR_SINGLE, linewidth=2.0, markersize=7, alpha=0.9, 
                     label='Single Leaf (1-Mat Shading)')
        ax_maxd.plot(r_idx, h2_curve, 'o-', color=COLOR_HEALTHY, linewidth=2.4, markersize=8, 
                     label='Healthy Tile (2-Mat Species)')
        ax_maxd.plot(r_idx, h4_curve, 's-', color=COLOR_PATHOLOGICAL, linewidth=2.4, markersize=8, 
                     label='Infected Tile (4-Mat Disease)')
        ax_maxd.axhline(h_crit, color='black', linestyle='--', linewidth=1.5,
                        label=f'1% Threshold ($h_{{crit}} = {h_crit:.3f}$)')
        ax_maxd.axhline(NOISE_FLOOR, color=COLOR_NOISE, linestyle=':', linewidth=1.5, 
                        label=f'Sensor Noise Floor (≈{NOISE_FLOOR})')

        ax_maxd.set_yscale('symlog', linthresh=SYMLOG_THRESHOLD)
        ax_maxd.set_xlabel('Endmember Index ($k$)', fontsize=LABEL_FONT_SIZE, fontweight='bold')
        ax_maxd.set_ylabel('Orthogonal Residual Distance $h_k$', fontsize=LABEL_FONT_SIZE, fontweight='bold')
        ax_maxd.set_title('Spectral Complexity Orthogonal Subspace Distances', fontsize=TITLE_FONT_SIZE, fontweight='bold', pad=10)
        ax_maxd.set_xticks(r_idx)
        #ax_maxd.set_xticklabels(['r=1\n(Origin)', 'r=2\n(Shade/Species)', 'r=3\n(Noise/Species)', 'r=4\n(Noise/Disease)', 'r=5\n(Noise Floor)'], 
        #                        fontsize=TICK_FONT_SIZE - 0.8)
        ax_maxd.set_ylim(bottom=0)
        ax_maxd.tick_params(axis='both', labelsize=TICK_FONT_SIZE)
        ax_maxd.grid(True, which='both', linestyle=':', alpha=0.4)
        ax_maxd.legend(loc='upper right', fontsize=LEGEND_FONT_SIZE, framealpha=0.9)

        # Scale Callouts on MaxD
       #ax_maxd.annotate(f'Species: $h_1={h2_curve[1]:.2f}$\n(3.8x Shading Scale)', 
       #                 xy=(2, h2_curve[1]), xytext=(2.25, 1.8),
       #                 arrowprops=dict(arrowstyle='->', color=COLOR_HEALTHY, lw=1.4),
       #                 fontsize=ANNOTATION_FONT_SIZE + 0.5, color=COLOR_HEALTHY, fontweight='bold')
       #ax_maxd.annotate(f'Disease: $h_1={h4_curve[1]:.2f}$\n(9.6x Shading Scale)', 
       #                 xy=(2, h4_curve[1]), xytext=(2.25, 4.3),
       #                 arrowprops=dict(arrowstyle='->', color=COLOR_PATHOLOGICAL, lw=1.4),
       #                 fontsize=ANNOTATION_FONT_SIZE + 0.5, color=COLOR_PATHOLOGICAL, fontweight='bold')

    else:
        # 3-Panel Layout Mode (Separate Subplots)
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=FIG_SIZE_TRI, dpi=DPI)

        # (a) Individual PCA
        ax1.bar(x_pca - BAR_WIDTH, var_1, BAR_WIDTH, label='Single Leaf (1-Mat)', color=COLOR_SINGLE, alpha=0.85)
        ax1.bar(x_pca, var_2, BAR_WIDTH, label='Healthy (2-Mat)', color=COLOR_HEALTHY, alpha=0.85)
        ax1.bar(x_pca + BAR_WIDTH, var_4, BAR_WIDTH, label='Infected (4-Mat)', color=COLOR_PATHOLOGICAL, alpha=0.85)
        ax1.axhline(PCA_CUTOFF_INDIV_PCT, color=COLOR_CUTOFF_INDIV, linestyle='--', linewidth=1.3, label='1% Kaiser Cutoff')
        ax1.set_yscale('log')
        ax1.set_ylim(Y_LIM_INDIV[0], Y_LIM_INDIV[1])
        ax1.set_xlabel('Principal Component Index', fontweight='bold', fontsize=LABEL_FONT_SIZE)
        ax1.set_ylabel('Individual Explained Variance (%) [Log]', fontweight='bold', fontsize=LABEL_FONT_SIZE)
        ax1.set_title('(a) PCA Individual Variance Scree', fontweight='bold', fontsize=TITLE_FONT_SIZE, pad=10)
        ax1.set_xticks(x_pca)
        ax1.set_xticklabels([f'PC {i}' for i in x_pca], fontsize=TICK_FONT_SIZE)
        ax1.grid(True, which='both', linestyle=':', alpha=0.4)
        ax1.legend(loc='upper right', fontsize=LEGEND_FONT_SIZE, framealpha=0.9)

        # (b) Cumulative PCA
        ax2.plot(x_pca, cum_1, 'o-', color=COLOR_SINGLE, linewidth=2.2, markersize=7, label='Single Leaf (1-Mat)')
        ax2.plot(x_pca, cum_2, 's-', color=COLOR_HEALTHY, linewidth=2.2, markersize=7, label='Healthy (2-Mat)')
        ax2.plot(x_pca, cum_4, '^-', color=COLOR_PATHOLOGICAL, linewidth=2.2, markersize=7, label='Infected (4-Mat)')
        ax2.axhline(PCA_CUTOFF_CUMUL_PCT, color=COLOR_CUTOFF_CUMUL, linestyle='--', linewidth=1.3, label='99% Cumulative Threshold')
        ax2.set_xlabel('Number of Components ($k$)', fontweight='bold', fontsize=LABEL_FONT_SIZE)
        ax2.set_ylabel('Cumulative Explained Variance (%)', fontweight='bold', fontsize=LABEL_FONT_SIZE)
        ax2.set_title('(b) PCA Cumulative Variance Trajectory', fontweight='bold', fontsize=TITLE_FONT_SIZE, pad=10)
        ax2.set_xticks(x_pca)
        ax2.set_xticklabels([f'k={i}' for i in x_pca], fontsize=TICK_FONT_SIZE)
        ax2.set_ylim(Y_LIM_CUMUL[0], Y_LIM_CUMUL[1])
        ax2.grid(True, linestyle=':', alpha=0.5)
        ax2.legend(loc='lower right', fontsize=LEGEND_FONT_SIZE, framealpha=0.9)

        # (c) MaxD
        B = len(res_1mat['wavelengths_nm'])
        h_crit = TAU_CRIT * np.sqrt(B)
        ax3.plot(r_idx, h1_curve, 'o-', color=COLOR_SINGLE, linewidth=1.8, markersize=6, alpha=0.85, label='Single Leaf (1-Mat)')
        ax3.plot(r_idx, h2_curve, 'o-', color=COLOR_HEALTHY, linewidth=2.2, markersize=7, label='Healthy (2-Mat)')
        ax3.plot(r_idx, h4_curve, 's-', color=COLOR_PATHOLOGICAL, linewidth=2.2, markersize=7, label='Infected (4-Mat)')
        ax3.axhline(h_crit, color='black', linestyle='--', linewidth=1.5, label=f'1% Threshold ($h_{{crit}} = {h_crit:.3f}$)')
        ax3.axhline(NOISE_FLOOR, color=COLOR_NOISE, linestyle=':', linewidth=1.5, label=f'Noise Floor (≈{NOISE_FLOOR})')
        ax3.set_yscale('symlog', linthresh=SYMLOG_THRESHOLD)
        ax3.set_xlabel('Endmember Count ($r$)', fontweight='bold', fontsize=LABEL_FONT_SIZE)
        ax3.set_ylabel('Residual Distance $h_r$ [Symlog]', fontweight='bold', fontsize=LABEL_FONT_SIZE)
        ax3.set_title('(c) MaxD Residuals: Scale Contrast', fontweight='bold', fontsize=TITLE_FONT_SIZE, pad=10)
        ax3.set_xticks(r_idx)
        ax3.set_xticklabels(['r=1', 'r=2', 'r=3', 'r=4', 'r=5'], fontsize=TICK_FONT_SIZE)
        ax3.set_ylim(bottom=0)
        ax3.grid(True, which='both', linestyle=':', alpha=0.4)
        ax3.legend(loc='upper right', fontsize=LEGEND_FONT_SIZE, framealpha=0.9)

    plt.tight_layout()
    print(f"Saving high-resolution visual to: {out_path}")
    plt.savefig(out_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"[Done] Slide 4 visual regenerated successfully: {out_path}\n")
    return out_path


if __name__ == '__main__':
    generate_pca_vs_complexity_figure()
