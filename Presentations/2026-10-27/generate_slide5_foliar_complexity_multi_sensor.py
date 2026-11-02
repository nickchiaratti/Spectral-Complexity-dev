#!/usr/bin/env python3
"""
generate_slide5_foliar_complexity_multi_sensor.py
Slide 5 Visual: Multi-Sensor Foliar Complexity Simulation (5-Material Tile)

Simulates a 5-material foliar tile (15x15 pixels, 6% shading variability, SNR=200)
across all sensor configurations for which empirical leaf spectra have been resampled:
  1. rsLandsat8 (B = 7 bands)
  2. rsSentinel2 (B = 13 bands)
  3. rsWorldView3 (B = 16 bands)
  4. cvHYMAP2014 (B = 126 bands)
  5. cvAVIRISc2014 (B = 224 bands)
  6. cvHySpexMjolnir (B = 413 bands)
  7. cvASD (B = 2151 bands)

The 5 simulated materials are:
  1. HealthyLeaf1: Norway Maple (Acer platanoides) - Healthy
  2. HealthyLeaf2: Silver Maple (Acer saccharinum) - Healthy
  3. DiseaseLeaf1: Norway Maple (Acer platanoides) - Rhytisma acerinum Tar Spot
  4. DiseaseLeaf2: Silver Maple (Acer saccharinum) - Rhytisma acerinum Tar Spot
  5. DiseaseLeaf1b: Re-measurement / replicate scan of DiseaseLeaf1

Produces an executive multi-panel publication figure demonstrating how the universal
1% discriminability threshold:
    h_crit(B) = 0.010 * sqrt(B)
consistently identifies the 4 distinct biological dimensions across all sensors, while
correctly rejecting the redundant 5th endmember (DiseaseLeaf1b) as sub-threshold scree.

Designed for IDE execution:
  - Simply open in VS Code / PyCharm and run (Run / F5).
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches

# Ensure repository packages are in Python path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))
HYSPEX_DIR = os.path.join(REPO_ROOT, "hyspex")
SIM_DIR = os.path.join(REPO_ROOT, "simulated-data-test")

for d in [HYSPEX_DIR, SIM_DIR, REPO_ROOT]:
    if d not in sys.path:
        sys.path.insert(0, d)

from simulate_inset_squares import (
    add_wavelength_scaled_noise,
    maximumDistance_parallelotopeVolumes
)

# =============================================================================
# USER CONFIGURATION PARAMETERS (Adjust for fine-tuning in IDE)
# =============================================================================
DATA_DIR = r"C:\satelliteImagery\ground_truth\NickLeaves_20261001"
OUTPUT_DIR = r"C:\satelliteImagery\simulated\foliar_comparison"
OUTPUT_FILENAME = "foliar_complexity_multi_sensor_5mat.png"

# Simulation Parameters
TILE_SIZE = 15                 # 15x15 pixels (225 pure pixels)
SNR = 200                      # Sensor signal-to-noise ratio
SHOT_NOISE_FRACTION = 0.70     # 70% Poisson shot noise, 30% Gaussian read noise
SHADE_VARIABILITY = 0.06       # 6% uniform shading variability: U(0.94, 1.0)
NUM_ENDMEMBERS = 8             # Number of MaxD endmembers to extract
SEED = 42                      # Random seed for reproducible simulation
TAU_CRIT = 0.010               # 1.0% RMS reflectance contrast per band

# 5 Foliar Target Definitions
MATERIALS = [
    {
        'key': 'HealthyLeaf1',
        'name': 'Norway Maple (Healthy)',
        'species': 'Acer platanoides',
        'color': '#2ca02c'      # Green
    },
    {
        'key': 'HealthyLeaf2',
        'name': 'Silver Maple (Healthy)',
        'species': 'Acer saccharinum',
        'color': '#1f77b4'      # Blue
    },
    {
        'key': 'DiseaseLeaf1',
        'name': 'Norway Maple (Tar Spot)',
        'species': 'Rhytisma acerinum',
        'color': '#8c564b'      # Brown
    },
    {
        'key': 'DiseaseLeaf2',
        'name': 'Silver Maple (Tar Spot)',
        'species': 'Rhytisma acerinum',
        'color': '#d62728'      # Red
    },
    {
        'key': 'DiseaseLeaf1b',
        'name': 'Norway Maple (Tar Spot 1b)',
        'species': 'Re-measurement scan',
        'color': '#e6550d'      # Amber / Rust
    }
]

# Sensors to Evaluate (Resampled in leaf_data_conversion.py)
SENSOR_SPECS = [
    {
        'key': 'rsLandsat8',
        'name': 'Landsat-8 OLI',
        'type': 'Multispectral',
        'platform': 'Spaceborne (7 bands)'
    },
    {
        'key': 'rsSentinel2',
        'name': 'Sentinel-2 MSI',
        'type': 'Multispectral',
        'platform': 'Spaceborne (13 bands)'
    },
    {
        'key': 'rsWorldView3',
        'name': 'WorldView-3',
        'type': 'Multispectral',
        'platform': 'Spaceborne (16 bands)'
    },
    {
        'key': 'cvHYMAP2014',
        'name': 'HyMap 2014',
        'type': 'Hyperspectral',
        'platform': 'Airborne (126 bands)'
    },
    {
        'key': 'cvAVIRISc2014',
        'name': 'AVIRIS Classic',
        'type': 'Hyperspectral',
        'platform': 'Airborne (224 bands)'
    },
    {
        'key': 'cvHySpexMjolnir',
        'name': 'HySpex Mjolnir',
        'type': 'Hyperspectral',
        'platform': 'Airborne (413 bands)'
    },
    {
        'key': 'cvASD',
        'name': 'ASD FieldSpec',
        'type': 'Full-Range Spectroradiometer',
        'platform': 'Laboratory / Contact (2151 bands)'
    }
]


def load_sensor_spectra(sensor_key: str):
    """
    Loads the 5 resampled leaf spectra for a given sensor and extracts valid non-NaN bands.
    """
    raw_specs = []
    wavelengths = None
    for mat in MATERIALS:
        fpath = os.path.join(DATA_DIR, f"{sensor_key}_{mat['key']}.csv")
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Missing resampled spectrum: {fpath}")
        df = pd.read_csv(fpath)
        if wavelengths is None:
            wavelengths = df['wavelength'].values
        raw_specs.append(df['DHR_mean'].values.copy())

    # Determine valid band mask across all 5 spectra
    valid_mask = np.ones(len(wavelengths), dtype=bool)
    for s in raw_specs:
        valid_mask &= ~np.isnan(s)

    # For HySpex and full-range ASD, exclude telluric water absorption intervals
    if sensor_key in ['cvHySpexMjolnir', 'cvHySpex', 'cvASD']:
        is_water = ((wavelengths >= 1350.0) & (wavelengths <= 1450.0)) | \
                   ((wavelengths >= 1800.0) & (wavelengths <= 2000.0)) | \
                   (wavelengths > 2420.0)
        valid_mask &= ~is_water

    clean_specs = [s[valid_mask].astype(np.float32) for s in raw_specs]
    clean_wv = wavelengths[valid_mask]
    return clean_wv, clean_specs, valid_mask


def build_5material_tile(clean_specs, tile_size=TILE_SIZE):
    """
    Constructs a 15x15 pure-pixel tile containing all 5 materials:
      - Top-Left Quadrant:     Material 0 (Norway Maple Healthy)
      - Top-Right Quadrant:    Material 1 (Silver Maple Healthy)
      - Bottom-Left Quadrant:  Material 2 (Norway Maple Tar Spot)
      - Bottom-Right Quadrant: Material 3 (Silver Maple Tar Spot)
      - Center 5x5 Inset:      Material 4 (Norway Maple Tar Spot Replicate 1b)
    """
    num_bands = len(clean_specs[0])
    tile = np.zeros((num_bands, tile_size, tile_size), dtype=np.float32)
    material_map = np.zeros((tile_size, tile_size), dtype=np.int32)

    half_r = tile_size // 2 + 1  # 8
    half_c = tile_size // 2 + 1  # 8

    # 4 Quadrants
    tile[:, :half_r, :half_c] = clean_specs[0][:, None, None]
    material_map[:half_r, :half_c] = 0

    tile[:, :half_r, half_c:] = clean_specs[1][:, None, None]
    material_map[:half_r, half_c:] = 1

    tile[:, half_r:, :half_c] = clean_specs[2][:, None, None]
    material_map[half_r:, :half_c] = 2

    tile[:, half_r:, half_c:] = clean_specs[3][:, None, None]
    material_map[half_r:, half_c:] = 3

    # Center 5x5 Inset (rows 5..9, cols 5..9)
    c_start = (tile_size - 5) // 2  # 5
    c_end = c_start + 5             # 10
    tile[:, c_start:c_end, c_start:c_end] = clean_specs[4][:, None, None]
    material_map[c_start:c_end, c_start:c_end] = 4

    return tile, material_map


def simulate_sensor_complexity(sensor_meta):
    """
    Simulates the 5-material tile for a specific sensor and extracts MaxD orthogonal heights.
    """
    s_key = sensor_meta['key']
    clean_wv, clean_specs, _ = load_sensor_spectra(s_key)
    B = len(clean_wv)
    sqrt_B = np.sqrt(B)
    h_crit = TAU_CRIT * sqrt_B

    # Build pure-pixel 5-material tile
    clean_tile, mat_map = build_5material_tile(clean_specs, tile_size=TILE_SIZE)

    # Apply 6% uniform shading variability: U(0.94, 1.0)
    np.random.seed(SEED)
    shade_map = np.random.uniform(1.0 - SHADE_VARIABILITY, 1.0, (TILE_SIZE, TILE_SIZE)).astype(np.float32)
    tile_shaded = clean_tile * shade_map[None, :, :]

    # Add heteroscedastic Poisson-Gaussian noise at SNR=200
    noisy_tile, _ = add_wavelength_scaled_noise(
        tile_shaded,
        snr=SNR,
        shot_noise_fraction=SHOT_NOISE_FRACTION,
        seed=SEED
    )

    # Compute MaxD Parallelotope Orthogonal Heights
    pixels = noisy_tile.reshape(B, -1)
    em, vol_log, heights, indices = maximumDistance_parallelotopeVolumes(pixels, NUM_ENDMEMBERS)

    # Extract non-zero orthogonal residual heights (k = 1..NUM_ENDMEMBERS - 1)
    h_steps = heights[1:NUM_ENDMEMBERS]
    rms_steps = h_steps / sqrt_B

    # Decay rate / negative gradient
    decay_rate = np.full_like(h_steps, np.nan)
    if len(h_steps) > 2:
        decay_rate[:-1] = -(h_steps[1:] - h_steps[:-1])

    return {
        'sensor': sensor_meta,
        'B': B,
        'sqrt_B': sqrt_B,
        'h_crit': h_crit,
        'heights': h_steps,
        'rms_steps': rms_steps,
        'decay_rate': decay_rate,
        'clean_tile': clean_tile,
        'noisy_tile': noisy_tile,
        'material_map': mat_map,
        'wavelengths': clean_wv
    }


def generate_slide5_figure():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)

    print("\n=======================================================")
    print("Generating Slide 5 Visual: Multi-Sensor Foliar Complexity")
    print(f"5 Materials: 4 Distinct Maples + Replicate Scan 1b")
    print(f"Shading: {SHADE_VARIABILITY*100:.1f}% | SNR: {SNR} | Tau Crit: {TAU_CRIT*100:.1f}%")
    print("=======================================================")

    results = []
    for s_meta in SENSOR_SPECS:
        print(f"Simulating sensor: {s_meta['name']} ({s_meta['platform']})...")
        res = simulate_sensor_complexity(s_meta)
        results.append(res)
        h = res['heights']
        hc = res['h_crit']
        print(f"  B = {res['B']}, h_crit = {hc:.4f}")
        print(f"  h_k (k=1..{len(h)}): {[round(float(x), 4) for x in h]}")
        print(f"  h_k > h_crit:       {[bool(x > hc) for x in h]}")

    # Set up 2x4 composite figure (1 Tile Layout panel + 7 Sensor Orthogonal Subspace Distance panels)
    fig = plt.figure(figsize=(24, 11), dpi=300)
    gs = fig.add_gridspec(2, 4, wspace=0.30, hspace=0.48, top=0.92, bottom=0.08, left=0.05, right=0.97)

    fig.suptitle(
        r"Multi-Sensor Foliar Complexity Simulation: 5-Material Tile with $1\%$ Discriminability Threshold ($h_{\mathrm{crit}} = 0.010 \cdot \sqrt{B}$)",
        fontsize=15, fontweight='bold'
    )

    # -------------------------------------------------------------------------
    # Panel (a): 5-Material Tile Spatial Layout & Class Definition
    # -------------------------------------------------------------------------
    ax_map = fig.add_subplot(gs[0, 0])
    mat_map = results[0]['material_map']
    
    # Custom discrete colormap for the 5 materials
    cmap_5 = mcolors.ListedColormap([m['color'] for m in MATERIALS])
    norm_5 = mcolors.BoundaryNorm(np.arange(-0.5, 5.5, 1), cmap_5.N)

    im_map = ax_map.imshow(mat_map, cmap=cmap_5, norm=norm_5, interpolation='nearest')
    ax_map.set_title("(a) Simulated 5-Material Tile (15x15)", fontsize=11, fontweight='bold')
    ax_map.set_xlabel("Local Column", fontsize=9.5)
    ax_map.set_ylabel("Local Row", fontsize=9.5)
    ax_map.set_xticks(np.arange(0, TILE_SIZE, 3))
    ax_map.set_yticks(np.arange(0, TILE_SIZE, 3))

    # Add dividing lines demarcating the 4 quadrants and central inset
    ax_map.axvline(7.5, color='white', linestyle='--', linewidth=1.2, alpha=0.8)
    ax_map.axhline(7.5, color='white', linestyle='--', linewidth=1.2, alpha=0.8)
    rect_inset = patches.Rectangle((4.5, 4.5), 5.0, 5.0, linewidth=2.0, edgecolor='yellow', facecolor='none', linestyle='-')
    ax_map.add_patch(rect_inset)

    # Legend for the 5 materials cleanly positioned beneath panel (a)
    legend_patches = [
        patches.Patch(facecolor=m['color'], edgecolor='black', label=f"{m['name']}")
        for m in MATERIALS
    ]
    ax_map.legend(handles=legend_patches, loc='upper center', bbox_to_anchor=(0.5, -0.22), fontsize=7.8, framealpha=0.95, ncol=2)

    # -------------------------------------------------------------------------
    # Panels (b)-(h): Orthogonal Subspace Distance Curves for Each Sensor
    # Row 0: Multispectral (Landsat-8, Sentinel-2, WorldView-3)
    # Row 1: Hyperspectral & Laboratory (HyMap, AVIRIS, HySpex, ASD)
    # -------------------------------------------------------------------------
    sensor_axes = [
        fig.add_subplot(gs[0, 1]),  # (b) Landsat-8
        fig.add_subplot(gs[0, 2]),  # (c) Sentinel-2
        fig.add_subplot(gs[0, 3]),  # (d) WorldView-3
        fig.add_subplot(gs[1, 0]),  # (e) HyMap 2014
        fig.add_subplot(gs[1, 1]),  # (f) AVIRIS Classic
        fig.add_subplot(gs[1, 2]),  # (g) HySpex Mjolnir
        fig.add_subplot(gs[1, 3])   # (h) ASD FieldSpec
    ]

    panel_letters = ['b', 'c', 'd', 'e', 'f', 'g', 'h']

    for idx, (res, ax, p_let) in enumerate(zip(results, sensor_axes, panel_letters)):
        s_meta = res['sensor']
        B = res['B']
        h_crit = res['h_crit']
        h_vals = res['heights']
        k_steps = np.arange(1, len(h_vals) + 1)

        # Cumulative 3D-Normalized Volume:
        # V_cum^(3D)(k) = V_3 + sum_{j=4}^k (prod_{m=1}^j h_m)^(3/j) for k >= 4
        # with base V_cum^(3D)(3) = V_3 = h_1 * h_2 * h_3
        P = np.cumprod(h_vals)
        K = len(h_vals)
        v_3d_terms = P ** (3.0 / k_steps)
        v_cum_3d = np.full(K, np.nan)
        if K >= 3:
            v_cum_3d[2] = P[2]  # V_3 = h1 * h2 * h3
            for k_idx in range(3, K):
                v_cum_3d[k_idx] = v_cum_3d[k_idx - 1] + v_3d_terms[k_idx]

        # Plot Orthogonal Distance (h_k)
        p1 = ax.plot(k_steps, h_vals, 'o-', color='#1f77b4', linewidth=2.0, markersize=5.5,label=r'Orthogonal Distance $h_k$')

        # Plot 1% Critical Threshold: h_crit = 0.010 * sqrt(B)
        p2 = ax.axhline(h_crit, color='blue', linestyle='--', linewidth=1.8,label=rf'1% Threshold ($h_{{\mathrm{{crit}}}} = {h_crit:.3f}$)')
        ax.set_yscale('symlog', linthresh=1e-2)
        ax.set_ylim(bottom=0, top=max(h_vals[0] * 1.5, h_crit * 3.0))
        ax.set_title(f"({p_let}) {s_meta['name']} ($B={B}$)", fontsize=11, fontweight='bold')
        ax.set_xlabel("Endmember Index (k)", fontsize=9.5)
        ax.set_ylabel(r"Distance $h_k$", fontsize=9.5, color='blue')
        ax.set_xticks(k_steps)
        ax.grid(True, linestyle=':', alpha=0.5)

        av = ax.twinx()
        k_vol = k_steps[2:]
        v_plot = v_cum_3d[2:]
        p3 = av.plot(k_vol, v_plot, 's--', color='crimson', linewidth=1.8, markersize=5.0,label=r'Cumul 3D Vol $V_{\mathrm{cum}}^{(3\mathrm{D})}$')
        av.set_ylabel(r"Cumulative 3D Volume $V_{\mathrm{cum}}^{(3\mathrm{D})}$", fontsize=8.5, color='crimson')
        av.tick_params(axis='y', labelcolor='crimson')
        #if len(v_plot) > 0 and not np.all(np.isnan(v_plot)):
        #    av.set_ylim(bottom=0, top=np.nanmax(v_plot) * 1.35)
        av.grid(False)

        # Annotate material boundary (k=4 plunges below threshold)
        #if len(h_vals) >= 4:
        #    ax.axvline(3.5, color='gray', linestyle=':', linewidth=1.2, alpha=0.7)
        #    # Add callout text
        #    y_pos = max(h_vals[0] * 0.45, h_crit * 1.8)
        #    ax.text(2.0, y_pos, "3 Signal Steps\n(4 Physical Materials)", color='#1f77b4',
        #            fontsize=8.0, fontweight='bold', ha='center',
        #            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.85, edgecolor='#1f77b4'))
#
        #    ax.text(5.5, h_crit * 1.5, "Sub-1% Scree\n(Replicate 1b Rejected)", color='#d62728',
        #            fontsize=8.0, fontweight='bold', ha='center',
        #            bbox=dict(boxstyle='round,pad=0.2', facecolor='#fff2f2', alpha=0.9, edgecolor='#d62728'))
#
        

        all_lines = p1 + [p2] + p3
        ax.legend(all_lines, [l.get_label() for l in all_lines], loc='upper right', fontsize=7.5, framealpha=0.9)

    print(f"\nSaving multi-sensor foliar complexity figure to: {out_path}")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Done] Slide 5 figure generated successfully: {out_path}\n")
    return out_path


if __name__ == "__main__":
    generate_slide5_figure()
