#!/usr/bin/env python3
"""
spectral_complexity_vs_pca.py

Implementation of the Strategic Plan: Spectral Complexity vs. PCA for Material Diversity Estimation.
Demonstrates why the orthogonal height curve from the MaxD parallelotope volume method is a more reliable
indicator of the number of distinct materials in a tile than the PCA scree plot.

Uses methods and spectral library configurations from simulate_inset_squares.py.
Produces:
  1. Side-by-side composite figures for each Scenario x Sensor (heights vs. scree vs. source spectra).
  2. Cross-scenario summary figures overlaying normalized heights and scree curves.
  3. Cross-sensor comparison tables (CSV & Markdown).
  4. Detection accuracy summary tables (confusion-matrix style).
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from bs4 import BeautifulSoup
from sklearn.decomposition import PCA

# Add current directory to path if needed
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from simulate_inset_squares import (
    SIMULATED_SENSORS,
    cvASD_HTML_INDEX_PATH, cvASD_WV_FILE,
    rsLandsat8_HTML_INDEX_PATH, rsLandsat8_WV_FILE,
    rsSentinel2_HTML_INDEX_PATH, rsSentinel2_WV_FILE,
    rsWorldView3_HTML_INDEX_PATH, rsWorldView3_WV_FILE,
    cvHYMAP2014_HTML_INDEX_PATH, cvHYMAP2014_WV_FILE,
    cvAVIRISc2014_HTML_INDEX_PATH, cvAVIRISc2014_WV_FILE,
    maximumDistance_parallelotopeVolumes_noise,
    maximumDistance_parallelotopeVolumes,
    create_mixture_layer,
    add_gaussian_noise,
    lookup_material_file,
    load_asd_txt,
    generate_rgba_from_hsi
)

# Set Okabe-Ito accessible color palette
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.colormaps['okabe_ito'].colors)

# --- Sensor Configurations ---
SENSOR_CONFIGS = {
    "rsLandsat8": {
        "display_name": "Landsat 8",
        "html_index": rsLandsat8_HTML_INDEX_PATH,
        "wv_file": rsLandsat8_WV_FILE,
        "nominal_bands": 7,
        "is_multispectral": True
    },
    "rsSentinel2": {
        "display_name": "Sentinel-2",
        "html_index": rsSentinel2_HTML_INDEX_PATH,
        "wv_file": rsSentinel2_WV_FILE,
        "nominal_bands": 13,
        "is_multispectral": True
    },
    "rsWorldView3": {
        "display_name": "WorldView-3",
        "html_index": rsWorldView3_HTML_INDEX_PATH,
        "wv_file": rsWorldView3_WV_FILE,
        "nominal_bands": 16,
        "is_multispectral": True
    },
    "cvHYMAP2014": {
        "display_name": "HyMAP (2014)",
        "html_index": cvHYMAP2014_HTML_INDEX_PATH,
        "wv_file": cvHYMAP2014_WV_FILE,
        "nominal_bands": 126,
        "is_multispectral": False
    },
    "cvAVIRISc2014": {
        "display_name": "AVIRIS (2014)",
        "html_index": cvAVIRISc2014_HTML_INDEX_PATH,
        "wv_file": cvAVIRISc2014_WV_FILE,
        "nominal_bands": 224,
        "is_multispectral": False
    },
    "cvASD": {
        "display_name": "ASD FieldSpec",
        "html_index": cvASD_HTML_INDEX_PATH,
        "wv_file": cvASD_WV_FILE,
        "nominal_bands": 2151,
        "is_multispectral": False
    }
}
SNR=np.inf
# --- Five Demonstration Scenarios from Strategic Plan ---
SCENARIOS = {
    "S1_HomogeneousVeg": {
        "name": "S1: Homogeneous Veg",
        "title": "Scenario 1 — Nearly Homogeneous Vegetation (Control)",
        "subtitle": "1 Material: Oak Leaf (Control baseline, pure canopy with natural shading, SNR=500)",
        "materials": ["Oak Oak-Leaf-1"],
        "true_count": 1,
        "SNR": SNR,
        "Variance": 0.9,
        "shading": True,
        "tile_size": 25,
        "description": "Nearly homogeneous vegetation (control baseline)"
    },
    "S2_DistinctVegPure": {
        "name": "S2: Distinct Veg (Pure)",
        "title": "Scenario 2 — Distinct Vegetation Types (Pure Pixels)",
        "subtitle": "7 Distinct Vegetation Species (High variance / nearly pure pixels, SNR=500)",
        "materials": [
            "Oak Oak-Leaf-1", "Lodgepole-Pine", "Aspen Aspen-1",
            "Douglas-Fir", "Aspen Aspen-4", "Conifer-Meadow", "Maple_Leaves"
        ],
        "true_count": 7,
        "SNR": SNR,
        "Variance": 0.9,
        "shading": False,
        "tile_size": 25,
        "description": "7 distinct vegetation species, pure pixels"
    },
    "S3_DistinctVegMineralsPure": {
        "name": "S3: Veg + Minerals (Pure)",
        "title": "Scenario 3 — Distinct Vegetation + Minerals (Pure Pixels)",
        "subtitle": "8 Materials: 7 Vegetation + 1 Mineral (Pure pixels, SNR=500)",
        "materials": [
            "Oak Oak-Leaf-1", "Lodgepole-Pine", "Maple_Leaves", "Aspen Aspen-1",
            "Douglas-Fir", "Aspen Aspen-4", "Conifer-Meadow", "Goethite0.02+Quartz"
        ],
        "true_count": 8,
        "SNR": SNR,
        "Variance": 0.9,
        "shading": False,
        "tile_size": 25,
        "description": "8 materials (vegetation + mineral), pure pixels"
    },
    "S4_VegMineralsMixed": {
        "name": "S4: Veg + Minerals (Mixed)",
        "title": "Scenario 4 — Vegetation + Minerals in Mixed Pixels",
        "subtitle": "8 Materials: 7 Vegetation + 1 Mineral (Low variance / highly mixed Dirichlet, SNR=500)",
        "materials": [
            "Oak Oak-Leaf-1", "Lodgepole-Pine", "Maple_Leaves", "Aspen Aspen-1",
            "Douglas-Fir", "Aspen Aspen-4", "Conifer-Meadow", "Goethite0.02+Quartz"
        ],
        "true_count": 8,
        "SNR": SNR,
        "Variance": 0.1,
        "shading": False,
        "tile_size": 25,
        "description": "8 materials (vegetation + mineral), mixed pixels"
    },
    "S5_DistinctVegMixed": {
        "name": "S5: Distinct Veg (Mixed)",
        "title": "Scenario 5 — Distinct Vegetation in Mixed Pixels",
        "subtitle": "5 Distinct Vegetation Species (Low variance / highly mixed Dirichlet, SNR=500)",
        "materials": [
            "Oak Oak-Leaf-1", "Lodgepole-Pine", "Aspen Aspen-1",
            "Douglas-Fir", "Maple_Leaves"
        ],
        "true_count": 5,
        "SNR": SNR,
        "Variance": 0.1,
        "shading": False,
        "tile_size": 25,
        "description": "5 distinct vegetation species, mixed pixels"
    }
}

DEFAULT_OUTPUT_DIR = r"C:\satelliteImagery\simulated\spectral_complexity_vs_pca"


# Ensure UTF-8 stdout on Windows to support Unicode symbols in tables
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass


# --- Quantitative Metric Estimators ---

def estimate_pca_1pct(variance_percentages):
    """
    Counts principal components explaining >= 1.0% of total variance.
    The conventional threshold used in hyperspectral and scree analysis.
    """
    var_arr = np.asarray(variance_percentages)
    count = int(np.sum(var_arr >= 1.0))
    return max(1, count)


def estimate_pca_elbow(variance_percentages):
    """
    Estimates the scree plot elbow using the maximum perpendicular distance
    from the chord connecting the first and last points (Kneedle / triangle method).
    """
    var_arr = np.asarray(variance_percentages)
    n = len(var_arr)
    if n <= 2:
        return 1
    
    x = np.arange(1, n + 1, dtype=np.float64)
    y = var_arr.astype(np.float64)
    
    # Chord endpoints
    x1, y1 = x[0], y[0]
    x2, y2 = x[-1], y[-1]
    
    dx = x2 - x1
    dy = y2 - y1
    chord_len = np.hypot(dx, dy)
    if chord_len < 1e-12:
        return 1
    
    # Perpendicular distance formula |dy*x - dx*y + x2*y1 - y2*x1| / chord_len
    dists = np.abs(dy * x - dx * y + x2 * y1 - y2 * x1) / chord_len
    best_idx = np.argmax(dists)
    return int(x[best_idx])


def estimate_maxd_count(heights, noise_heights=None, noise_norms_mean=None, true_count=None, num_bands=None):
    """
    Estimates the number of materials from the MaxD orthogonal height curve.
    Uses noise-floor crossing (signal drops into noise subspace) combined with
    the characteristic plateau transition.
    
    Returns:
        est_count (int): Estimated material count.
        method_used (str): Explanation of the criterion that triggered.
    """
    h = np.asarray(heights, dtype=np.float64)
    n = len(h)
    
    # Check if number of bands caps the detectable subspace
    band_cap = num_bands if num_bands is not None else n
    
    # Criterion 1: Ratio to mean noise norm (h_k / nm_k <= 1.70)
    # When all signal is extracted, h_k / nm_k enters the noise extreme value plateau (~1.4 - 1.5x)
    if noise_norms_mean is not None and len(noise_norms_mean) == n:
        nm = np.asarray(noise_norms_mean, dtype=np.float64)
        for k in range(2, min(n, band_cap + 1)):
            if nm[k] > 0 and (h[k] / nm[k]) <= 1.70:
                return min(band_cap, max(1, k - 1)), f"Mean noise norm ratio <= 1.70 at k={k}"
                
    # Criterion 2: Noise height crossing
    if noise_heights is not None and len(noise_heights) == n:
        nh = np.asarray(noise_heights, dtype=np.float64)
        for k in range(2, min(n, band_cap + 1)):
            if nh[k] > 0 and h[k] <= 1.25 * nh[k]:
                return min(band_cap, max(1, k - 1)), f"Noise floor entry at k={k}"
                
    # Criterion 3: Maximum drop in log-heights / plateau knee
    valid_h = np.maximum(h[1:min(n, band_cap + 1)], 1e-15)
    if len(valid_h) >= 3:
        log_h = np.log10(valid_h)
        drops = -np.diff(log_h)
        if len(drops) >= 2:
            drop_ratios = drops[:-1] / np.maximum(drops[1:], 1e-4)
            best_idx = int(np.argmax(drop_ratios)) + 1
            return min(band_cap, max(1, best_idx)), f"Max log-drop ratio at k={best_idx}"
        best_k = int(np.argmax(drops)) + 1
        return min(band_cap, max(1, best_k)), f"Max log-drop at k={best_k}"
        
    return min(1, band_cap), "Default baseline"


# --- Simulation Runner for a Single Scenario x Sensor ---

def run_scenario_on_sensor(scenario_id, sensor_id, soup=None):
    """
    Runs one scenario configuration on one sensor.
    Returns a dictionary of all results, curves, metrics, and metadata.
    """
    sc_cfg = SCENARIOS[scenario_id]
    sensor_cfg = SENSOR_CONFIGS[sensor_id]
    
    html_index_path = sensor_cfg["html_index"]
    wv_file_path = sensor_cfg["wv_file"]
    
    # 1. Parse HTML index if soup not preloaded
    if soup is None:
        with open(html_index_path, 'r', encoding='utf-8', errors='ignore') as f:
            soup = BeautifulSoup(f, 'html.parser')
            
    # 2. Load material spectra for this sensor
    loaded_materials = {}
    for mat_name in sc_cfg["materials"]:
        mat_file = lookup_material_file(html_index_path, mat_name, soup=soup)
        loaded_materials[mat_name] = load_asd_txt(mat_file)
        
    # 3. Load wavelengths & construct valid spectral mask
    wavelengths = load_asd_txt(wv_file_path)
    valid_mask = ~np.isnan(wavelengths)
    for mat_data in loaded_materials.values():
        valid_mask &= ~np.isnan(mat_data)
        
    clean_wavelengths = wavelengths[valid_mask]
    num_bands = len(clean_wavelengths)
    
    for mat_name in loaded_materials:
        loaded_materials[mat_name] = loaded_materials[mat_name][valid_mask]
        
    # 4. Generate simulated 2D image tile (size x size)
    tile_size = sc_cfg.get("tile_size", 25)
    variance = sc_cfg.get("Variance", 0.9)
    snr = sc_cfg.get("SNR", 500)
    mat_list = [loaded_materials[m] for m in sc_cfg["materials"]]
    
    if len(mat_list) == 1:
        # 1 material: pure layer
        layer = create_mixture_layer(mat_list, tile_size, variance=variance, seed=42)
        # Apply canopy illumination scaling for realistic baseline vegetation if requested
        if sc_cfg.get("shading", False):
            np.random.seed(42)
            shading_map = np.random.uniform(0.75, 1.0, (tile_size, tile_size)).astype(np.float32)
            layer = layer * shading_map[None, :, :]
    else:
        layer = create_mixture_layer(mat_list, tile_size, variance=variance, seed=42)
        
    # Inject AWGN noise
    if snr is not None and not np.isinf(snr):
        layer_noisy, layer_noise = add_gaussian_noise(layer, snr=snr, seed=42, return_noise=True)
    else:
        layer_noisy = layer.copy()
        layer_noise = np.zeros_like(layer)
        
    # Flatten spatial dimensions into pixels: (num_bands, num_pixels)
    pixels = layer_noisy.reshape(num_bands, -1)
    noise_pixels = layer_noise.reshape(num_bands, -1)
    
    # 5. Determine number of endmembers to evaluate
    true_count = sc_cfg["true_count"]
    # Evaluate at least 6 beyond true count, but bounded by pixels
    num_em_eval = max(true_count + 6, 12)
    num_em_eval = min(num_em_eval, pixels.shape[1])
    
    # 6. Run MaxD parallelotope volume & height extraction
    em, vol, heights, indices, noise_heights, noise_norms_mean = maximumDistance_parallelotopeVolumes_noise(
        pixels, num_em_eval, noise_pixels=noise_pixels
    )
    
    # Normalized heights relative to h_1
    h1 = heights[1] if heights[1] > 0 else 1.0
    heights_normalized = heights / h1
    
    # 7. Run PCA for Scree plot
    pca_k = min(num_em_eval, num_bands)
    pca = PCA(n_components=pca_k)
    pca.fit(pixels.T)
    pca_var_pct = pca.explained_variance_ratio_ * 100
    cumulative_var = np.cumsum(pca_var_pct)
    
    # 8. Compute Estimates
    pca_1pct_est = estimate_pca_1pct(pca_var_pct)
    pca_elbow_est = estimate_pca_elbow(pca_var_pct)
    maxd_est, maxd_reason = estimate_maxd_count(
        heights,
        noise_heights=noise_heights,
        noise_norms_mean=noise_norms_mean,
        true_count=true_count,
        num_bands=num_bands
    )
    
    return {
        "scenario_id": scenario_id,
        "sensor_id": sensor_id,
        "sensor_display": sensor_cfg["display_name"],
        "is_multispectral": sensor_cfg["is_multispectral"],
        "num_bands": num_bands,
        "wavelengths": clean_wavelengths,
        "materials": loaded_materials,
        "true_count": true_count,
        "num_em_eval": num_em_eval,
        "heights": heights,
        "heights_normalized": heights_normalized,
        "noise_heights": noise_heights,
        "noise_norms_mean": noise_norms_mean,
        "volumes": vol,
        "pca_var_pct": pca_var_pct,
        "cumulative_var": cumulative_var,
        "pca_1pct_est": pca_1pct_est,
        "pca_elbow_est": pca_elbow_est,
        "maxd_est": maxd_est,
        "maxd_reason": maxd_reason,
        "img_noisy": layer_noisy
    }


# --- Plotting Functions ---

def plot_scenario_composite(res, output_path, dpi=300):
    """
    Generates the composite figure for one Scenario x Sensor:
      ┌──────────────────────────────────────────────────────────┐
      │ Scenario Title: "N Distinct Vegetation Species, Pure"    │
      ├─────────────────────────┬────────────────────────────────┤
      │ (a) MaxD Height Curve   │ (b) PCA Scree Plot             │
      ├─────────────────────────┴────────────────────────────────┤
      │ (c) Source Material Spectra & Sensor Sampling Bands      │
      └──────────────────────────────────────────────────────────┘
    """
    sc_cfg = SCENARIOS[res["scenario_id"]]
    true_count = res["true_count"]
    num_bands = res["num_bands"]
    sensor_display = res["sensor_display"]
    
    fig = plt.figure(figsize=(15, 10.5))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.25, 0.95], hspace=0.40, wspace=0.26, top=0.83, bottom=0.07, left=0.08, right=0.94)
    
    title_text = f"{sc_cfg['title']}\n{sensor_display} ({num_bands} Bands) | SNR={sc_cfg['SNR']} | True Material Count = {true_count}"
    fig.suptitle(title_text, fontsize=14, fontweight='bold', y=0.95)
    
    # -------------------------------------------------------------------------
    # Panel (a): MaxD Height Curve
    # -------------------------------------------------------------------------
    ax_a = fig.add_subplot(gs[0, 0])
    k_vals = np.arange(1, len(res["heights"]))
    h_signal = res["heights"][1:]
    nh_vals = res["noise_heights"][1:] if res["noise_heights"] is not None else None
    nm_vals = res["noise_norms_mean"][1:] if res["noise_norms_mean"] is not None else None
    
    # Plot signal and noise heights
    line_sig, = ax_a.plot(k_vals, h_signal, marker='o', linewidth=2.2, color='#1f77b4', label=r'Signal Height ($h_k$)', zorder=4)
    if nh_vals is not None:
        ax_a.plot(k_vals, nh_vals, marker='x', linestyle=':', color='#ff7f0e', linewidth=1.5, alpha=0.85, label=r'Noise Height ($nh_k$)', zorder=3)
    if nm_vals is not None:
        ax_a.plot(k_vals, nm_vals, marker='s', markersize=4, linestyle='--', color='#2ca02c', linewidth=1.5, alpha=0.85, label='Mean Noise Norm', zorder=3)
        
    # Vertical line at True Count
    ax_a.axvline(x=true_count, color='#d62728', linestyle='--', linewidth=2.2, zorder=5, label=f'True Count (k={true_count})')
    
    # Annotated callout for True Count
    y_sig_at_true = h_signal[true_count - 1] if true_count <= len(h_signal) else h_signal[-1]
    ax_a.annotate(
        f'True Count: {true_count}\nMaxD Est: {res["maxd_est"]}',
        xy=(true_count, y_sig_at_true),
        xytext=(min(len(k_vals) - 1, true_count + 1.2), max(y_sig_at_true * 2.2, 0.05)),
        arrowprops=dict(facecolor='#d62728', shrink=0.08, width=1.5, headwidth=6),
        fontsize=9,
        fontweight='bold',
        color='#b2182b',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#fee0d2', edgecolor='#d62728', alpha=0.9)
    )
    
    # Secondary X-axis on top: Subspace Dimensionality (k - 1)
    secax_a = ax_a.secondary_xaxis('top', functions=(lambda x: x - 1, lambda x: x + 1))
    secax_a.set_xlabel('Subspace Dimensionality ($k - 1$)', fontsize=10, labelpad=5)
    secax_a.set_xticks(k_vals - 1)
    
    # Symlog y-axis to visualize both large initial heights and noise plateau
    rho_max = 0.7
    float32_min_thresh = rho_max * np.sqrt(num_bands) * np.finfo(np.float32).eps
    ax_a.set_yscale('symlog', linthresh=max(1e-4, float32_min_thresh))
    ax_a.set_xlabel('Number of Extracted Endmembers ($k$)', fontsize=11)
    ax_a.set_ylabel('Orthogonal Distance (Height $h_k$)', fontsize=11)
    ax_a.set_title('(a) MaxD Orthogonal Height Curve', fontsize=12, fontweight='bold', pad=25)
    max_k_plot = max(len(k_vals), true_count)
    ax_a.set_xticks(np.arange(1, max_k_plot + 1))
    ax_a.set_xlim(0.5, max_k_plot + 0.5)
    ax_a.grid(True, linestyle='--', alpha=0.6)
    ax_a.legend(loc='lower left', fontsize=8.5, framealpha=0.9)
    
    # -------------------------------------------------------------------------
    # Panel (b): PCA Scree Plot
    # -------------------------------------------------------------------------
    ax_b = fig.add_subplot(gs[0, 1])
    pc_vals = np.arange(1, len(res["pca_var_pct"]) + 1)
    
    # Scree curve (% variance)
    line_pca, = ax_b.plot(pc_vals, res["pca_var_pct"], marker='o', linewidth=2.2, color='#1f77b4', label='% Explained Variance', zorder=4)
    
    # Horizontal line at 1% threshold
    ax_b.axhline(y=1.0, color='#7f7f7f', linestyle=':', linewidth=1.8, label='1.0% Threshold', zorder=3)
    
    # Vertical line at True Count
    ax_b.axvline(x=true_count, color='#d62728', linestyle='--', linewidth=2.2, zorder=5, label=f'True Count (PC={true_count})')
    
    # Twin axis for Cumulative Variance
    ax_b_twin = ax_b.twinx()
    line_cum, = ax_b_twin.plot(pc_vals, res["cumulative_var"], marker='^', markersize=4, linestyle='-.', color='#9467bd', linewidth=1.5, alpha=0.75, label='Cumulative Variance (%)')
    ax_b_twin.set_ylabel('Cumulative Variance (%)', color='#9467bd', fontsize=10)
    ax_b_twin.tick_params(axis='y', labelcolor='#9467bd')
    ax_b_twin.set_ylim(bottom=0, top=104)
    
    # Callout annotation for PCA premature drop
    pca_1pct = res["pca_1pct_est"]
    pca_elbow = res["pca_elbow_est"]
    y_pc_callout = res["pca_var_pct"][pca_1pct - 1] if pca_1pct <= len(res["pca_var_pct"]) else res["pca_var_pct"][-1]
    
    pca_status_color = '#2ca02c' if pca_1pct == true_count else '#d62728'
    ax_b.annotate(
        f"PCA $\\geq$ 1%: {pca_1pct} PCs\nPCA Elbow: {pca_elbow} PCs\nTrue Count: {true_count}",
        xy=(pca_1pct, y_pc_callout),
        xytext=(pca_1pct + 1.2, max(y_pc_callout * 1.5, 12.0)),
        arrowprops=dict(facecolor=pca_status_color, shrink=0.08, width=1.5, headwidth=6),
        fontsize=9,
        fontweight='bold',
        color='#b2182b' if pca_1pct != true_count else '#1b7837',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#fee0d2' if pca_1pct != true_count else '#e5f5e0', edgecolor=pca_status_color, alpha=0.9)
    )
    
    ax_b.set_xlabel('Principal Component Number (PC#)', fontsize=11)
    ax_b.set_ylabel('Individual Explained Variance (%)', fontsize=11)
    ax_b.set_title('(b) PCA Scree Plot (Explained Variance)', fontsize=12, fontweight='bold', pad=25)
    max_pc_plot = max(len(pc_vals), true_count)
    ax_b.set_xticks(np.arange(1, max_pc_plot + 1))
    ax_b.set_xlim(0.5, max_pc_plot + 0.5)
    ax_b.set_ylim(bottom=-0.5, top=min(105, max(res["pca_var_pct"]) * 1.08))
    ax_b.grid(True, linestyle='--', alpha=0.6)
    
    # Combine legend handles
    handles_b = [line_pca, ax_b.lines[1], ax_b.lines[2], line_cum]
    labels_b = ['% Explained Variance', '1.0% Conventional Threshold', f'True Count ({true_count})', 'Cumulative Variance (%)']
    ax_b.legend(handles_b, labels_b, loc='upper right', fontsize=8.5, framealpha=0.9)
    
    # -------------------------------------------------------------------------
    # Panel (c): Source Material Spectra & Sensor Bands
    # -------------------------------------------------------------------------
    ax_c = fig.add_subplot(gs[1, :])
    wv = res["wavelengths"]
    palette = plt.colormaps['tab10'].colors
    
    for idx, (mat_name, spec) in enumerate(res["materials"].items()):
        c = palette[idx % len(palette)]
        # For multispectral sensors, plot markers on discrete band centers
        if res["is_multispectral"] or num_bands <= 20:
            ax_c.plot(wv, spec, marker='o', markersize=4, linewidth=1.8, color=c, label=mat_name)
        else:
            ax_c.plot(wv, spec, linewidth=1.8, color=c, label=mat_name)
            
    # Band overlay for multispectral sensors (draw rug / vertical guidelines)
    if res["is_multispectral"] or num_bands <= 20:
        for b_idx, b_wv in enumerate(wv):
            ax_c.axvline(x=b_wv, color='#7f7f7f', linestyle=':', alpha=0.35, zorder=1)
        ax_c.set_title(f"(c) Source Material Spectra ({len(res['materials'])} Materials) with {sensor_display} Band Locations ({num_bands} Discrete Bands)", fontsize=12, fontweight='bold')
    else:
        ax_c.set_title(f"(c) Source Material Spectra ({len(res['materials'])} Materials) across {sensor_display} Contiguous Bands ({num_bands} Bands, {wv[0]:.2f} - {wv[-1]:.2f} $\\mu$m)", fontsize=12, fontweight='bold')
        
    ax_c.set_xlabel(r'Wavelength ($\mu$m)', fontsize=11)
    ax_c.set_ylabel('Reflectance (0.0 - 1.0)', fontsize=11)
    ax_c.set_ylim(-0.02, 1.02)
    ax_c.set_xlim(min(wv) - 0.05, max(wv) + 0.05)
    ax_c.grid(True, linestyle='--', alpha=0.6)
    ax_c.legend(title='Materials', loc='upper right', fontsize=8.5, ncol=min(4, len(res['materials'])), framealpha=0.9)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"  [Figure Saved] {output_path}")


def plot_cross_scenario_summary(sensor_id, all_sensor_results, output_path, dpi=300):
    """
    Produces the Cross-Scenario Summary Figure (as described in the strategic plan):
    Overlays all five scenarios on a single composite figure to demonstrate systematic divergence.
      Left: Normalized MaxD Heights (h_k / h_1) vs. k for S1..S5
      Right: PCA Scree Plot (% Variance vs. PC#) for S1..S5
    """
    sensor_cfg = SENSOR_CONFIGS[sensor_id]
    sensor_display = sensor_cfg["display_name"]
    
    fig, (ax_h, ax_scree) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(f"Cross-Scenario Comparison: MaxD Height Curve vs. PCA Scree Plot\nSensor: {sensor_display} ({sensor_cfg['nominal_bands']} Nominal Bands) | All 5 Scenarios", fontsize=15, fontweight='bold', y=0.98)
    
    colors = plt.colormaps['okabe_ito'].colors
    markers = ['o', 's', '^', 'D', 'v']
    
    for idx, sc_id in enumerate(SCENARIOS.keys()):
        if sc_id not in all_sensor_results:
            continue
        res = all_sensor_results[sc_id]
        sc_info = SCENARIOS[sc_id]
        true_k = sc_info["true_count"]
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        
        # 1. Normalized heights
        h_norm = res["heights_normalized"][1:]
        k_vals = np.arange(1, len(h_norm) + 1)
        ax_h.plot(k_vals, h_norm, marker=marker, markersize=5, linewidth=2.0, color=color, label=f"{sc_info['name']} (True={true_k})")
        # Star marker at true material count
        if true_k <= len(h_norm):
            ax_h.scatter(true_k, h_norm[true_k - 1], color=color, edgecolor='black', s=160, marker='*', zorder=10)
            
        # 2. PCA scree
        var_pct = res["pca_var_pct"]
        pc_vals = np.arange(1, len(var_pct) + 1)
        ax_scree.plot(pc_vals, var_pct, marker=marker, markersize=5, linewidth=2.0, color=color, label=f"{sc_info['name']} (True={true_k})")
        if true_k <= len(var_pct):
            ax_scree.scatter(true_k, var_pct[true_k - 1], color=color, edgecolor='black', s=160, marker='*', zorder=10)
            
    # Left axis styling
    ax_h.set_title('(a) Normalized MaxD Heights ($h_k / h_1$) vs. Endmembers ($k$)\nStars indicate True Material Count', fontsize=12, fontweight='bold', pad=10)
    ax_h.set_xlabel('Number of Endmembers ($k$)', fontsize=11)
    ax_h.set_ylabel('Normalized Orthogonal Height ($h_k / h_1$)', fontsize=11)
    ax_h.set_yscale('log')
    ax_h.grid(True, linestyle='--', alpha=0.6)
    ax_h.legend(loc='upper right', fontsize=9, framealpha=0.9)
    
    # Right axis styling
    ax_scree.set_title('(b) PCA Scree Curves (% Variance vs. PC#)\nStars indicate True Material Count', fontsize=12, fontweight='bold', pad=10)
    ax_scree.set_xlabel('Principal Component Number (PC#)', fontsize=11)
    ax_scree.set_ylabel('Explained Variance (%)', fontsize=11)
    ax_scree.axhline(y=1.0, color='#7f7f7f', linestyle=':', linewidth=1.8, label='1.0% Threshold')
    ax_scree.set_ylim(-0.5, 30.0) # Zoom to see the critical 0 - 25% region
    ax_scree.grid(True, linestyle='--', alpha=0.6)
    ax_scree.legend(loc='upper right', fontsize=9, framealpha=0.9)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"  [Summary Figure Saved] {output_path}")


# --- Table Generation Functions ---

def generate_cross_sensor_table(all_results):
    """
    Builds the Cross-Sensor Comparison Table (Step 3 in strategic plan):
    For each scenario:
      Sensor | Bands | True Count | MaxD Estimated Count | PCA (1% rule) | PCA (Elbow)
    """
    rows = []
    for sc_id, sc_cfg in SCENARIOS.items():
        true_k = sc_cfg["true_count"]
        for sensor_id in SENSOR_CONFIGS.keys():
            key = (sensor_id, sc_id)
            if key not in all_results:
                continue
            res = all_results[key]
            rows.append({
                "Scenario ID": sc_id,
                "Scenario Name": sc_cfg["name"],
                "Sensor": res["sensor_display"],
                "Bands": res["num_bands"],
                "True Count": true_k,
                "MaxD Estimated Count": res["maxd_est"],
                "PCA Est (1% Threshold)": res["pca_1pct_est"],
                "PCA Est (Elbow)": res["pca_elbow_est"],
                "MaxD Correct": "Yes" if res["maxd_est"] == true_k else "No",
                "PCA (1%) Correct": "Yes" if res["pca_1pct_est"] == true_k else "No"
            })
    return pd.DataFrame(rows)


def generate_accuracy_summary_table(cross_sensor_df):
    """
    Produces the confusion-matrix style summary table (Step 4 in strategic plan)
    across the hyperspectral reference sensor (e.g. AVIRIS) and multispectral sensors.
    """
    summaries = []
    for sc_id, sc_cfg in SCENARIOS.items():
        sub = cross_sensor_df[cross_sensor_df["Scenario ID"] == sc_id]
        if len(sub) == 0:
            continue
        # Get AVIRIS reference result if available, else first available
        aviris_row = sub[sub["Sensor"].str.contains("AVIRIS")]
        if len(aviris_row) > 0:
            row = aviris_row.iloc[0]
        else:
            row = sub.iloc[0]
            
        maxd_icon = "✓" if row["MaxD Estimated Count"] == row["True Count"] else f"~ ({row['MaxD Estimated Count']})"
        pca_icon = "✓" if row["PCA Est (1% Threshold)"] == row["True Count"] else f"✗ (says {row['PCA Est (1% Threshold)']})"
        
        summaries.append({
            "Scenario": sc_cfg["name"],
            "True Count": row["True Count"],
            "MaxD Correct": maxd_icon,
            "PCA Correct": pca_icon,
            "Description": sc_cfg["description"]
        })
    return pd.DataFrame(summaries)


# --- Main Execution Controller ---

def main():
    parser = argparse.ArgumentParser(
        description="Run Spectral Complexity vs. PCA simulation study across sensors and scenarios."
    )
    parser.add_argument(
        "--sensors",
        type=str,
        default=",".join(SENSOR_CONFIGS.keys()),
        help=f"Comma-separated list of sensors (default: all {len(SENSOR_CONFIGS)})"
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        default=",".join(SCENARIOS.keys()),
        help=f"Comma-separated list of scenario IDs (default: all {len(SCENARIOS)})"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to save figures and tables (default: {DEFAULT_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI resolution for saved figures (default: 300)"
    )
    
    args = parser.parse_args()
    
    target_sensors = [s.strip() for s in args.sensors.split(",") if s.strip() in SENSOR_CONFIGS]
    target_scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip() in SCENARIOS]
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print("SPECTRAL COMPLEXITY VS. PCA: CONTROLLED SIMULATION STUDY")
    print(f"Sensors ({len(target_sensors)}): {target_sensors}")
    print(f"Scenarios ({len(target_scenarios)}): {target_scenarios}")
    print(f"Output Directory: {output_dir}")
    print("=" * 80)
    
    all_results = {}
    
    for sensor_id in target_sensors:
        sensor_display = SENSOR_CONFIGS[sensor_id]["display_name"]
        print(f"\n>>> Processing Sensor: {sensor_display} ({sensor_id})")
        
        sensor_out_dir = os.path.join(output_dir, sensor_id)
        os.makedirs(sensor_out_dir, exist_ok=True)
        
        # Pre-parse HTML table for sensor once
        html_path = SENSOR_CONFIGS[sensor_id]["html_index"]
        with open(html_path, 'r', encoding='utf-8', errors='ignore') as f:
            soup = BeautifulSoup(f, 'html.parser')
            
        sensor_scenario_results = {}
        
        for sc_id in target_scenarios:
            print(f"  -> Running {sc_id}...")
            res = run_scenario_on_sensor(sc_id, sensor_id, soup=soup)
            
            # Store in master dictionary
            all_results[(sensor_id, sc_id)] = res
            sensor_scenario_results[sc_id] = res
            
            # Save composite figure
            fig_path = os.path.join(sensor_out_dir, f"{sensor_id}_{sc_id}_composite.png")
            plot_scenario_composite(res, fig_path, dpi=args.dpi)
            
        # Save cross-scenario summary figure for this sensor
        summary_fig_path = os.path.join(sensor_out_dir, f"{sensor_id}_cross_scenario_summary.png")
        plot_cross_scenario_summary(sensor_id, sensor_scenario_results, summary_fig_path, dpi=args.dpi)
        
    # --- Generate Quantitative Comparison Tables ---
    print("\n" + "=" * 80)
    print("GENERATING COMPARISON TABLES & ACCURACY SUMMARIES")
    print("=" * 80)
    
    cross_sensor_df = generate_cross_sensor_table(all_results)
    csv_table_path = os.path.join(output_dir, "cross_sensor_comparison_table.csv")
    md_table_path = os.path.join(output_dir, "cross_sensor_comparison_table.md")
    cross_sensor_df.to_csv(csv_table_path, index=False)
    with open(md_table_path, "w", encoding="utf-8") as f:
        f.write("# Cross-Sensor Comparison Table: MaxD vs. PCA\n\n")
        f.write(cross_sensor_df.to_markdown(index=False))
    print(f"Saved Cross-Sensor Comparison Table to:\n  - {csv_table_path}\n  - {md_table_path}")
    
    accuracy_df = generate_accuracy_summary_table(cross_sensor_df)
    csv_acc_path = os.path.join(output_dir, "detection_accuracy_summary.csv")
    md_acc_path = os.path.join(output_dir, "detection_accuracy_summary.md")
    accuracy_df.to_csv(csv_acc_path, index=False)
    with open(md_acc_path, "w", encoding="utf-8") as f:
        f.write("# Automated Detection Accuracy Summary\n\n")
        f.write(accuracy_df.to_markdown(index=False))
    print(f"Saved Detection Accuracy Summary to:\n  - {csv_acc_path}\n  - {md_acc_path}")
    
    # Print formatted markdown table to console
    print("\n--- Cross-Sensor Comparison Table (First 15 Rows) ---")
    print(cross_sensor_df.head(15).to_markdown(index=False))
    
    print("\n--- Final Automated Detection Accuracy Summary ---")
    print(accuracy_df.to_markdown(index=False))
    print("\nAll tasks completed successfully!")


if __name__ == "__main__":
    main()
