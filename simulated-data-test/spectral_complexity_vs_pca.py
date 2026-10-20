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
import matplotlib.lines as mlines
from bs4 import BeautifulSoup
from sklearn.decomposition import PCA
from kneed import KneeLocator
from skimage.filters import threshold_triangle
# Set Okabe-Ito accessible color palette
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.colormaps['okabe_ito'].colors)



DEFAULT_OUTPUT_DIR = r"C:\satelliteImagery\simulated\spectral_complexity_vs_pca"
SNR=np.inf
TILE_SIZE = 3

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
    NICKS_LEAVES_DIR,
    maximumDistance_parallelotopeVolumes_noise,
    maximumDistance_parallelotopeVolumes,
    create_mixture_layer,
    add_gaussian_noise,
    lookup_material_file,
    load_asd_txt,
    generate_rgba_from_hsi
)

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

# --- Demonstration Scenarios ---
SCENARIOS = {
    "S0_HomogeneousVeg": {
        "name": "S0: Homogeneous Veg",
        "title": "Scenario 0 — Homogeneous Vegetation (Control)",
        "subtitle": "1 Material: Oak Leaf (Control baseline, single spectra)",
        "materials": ["Oak Oak-Leaf-1"],
        "true_count": 1,
        "SNR": SNR,
        "Variance": 1,
        "shading": False,
        "tile_size": TILE_SIZE,
        "description": "Homogeneous single spectrum vegetation (control baseline)"
    },
    "S1_HomogeneousVeg": {
        "name": "S1: Homogeneous Veg",
        "title": "Scenario 1 — Nearly Homogeneous Vegetation",
        "subtitle": "1 Material: Oak Leaf (Control baseline, 100-75% scaling)",
        "materials": ["Oak Oak-Leaf-1"],
        "true_count": 1,
        "SNR": SNR,
        "Variance": 1,
        "shading": True,
        "tile_size": TILE_SIZE,
        "description": "Nearly homogeneous vegetation"
    },
    "S2_DistinctVegPure": {
        "name": "S2: Veg (Pure)",
        "title": "Scenario 2 — Vegetation Types (Pure Pixels)",
        "subtitle": "7 Vegetation Species (High variance / nearly pure pixels)",
        "materials": [
            "Oak Oak-Leaf-1", "Lodgepole-Pine", "Aspen Aspen-1",
            "Douglas-Fir", "Aspen Aspen-4", "Conifer-Meadow", "NICKLEAVES:HealthyLeaf1"
        ],
        "true_count": 7,
        "SNR": SNR,
        "Variance": 1,
        "shading": False,
        "tile_size": TILE_SIZE,
        "description": "7 distinct vegetation species, pure pixels"
    },
    "S3_DistinctVegMineralsPure": {
        "name": "S3: Veg + Minerals (Pure)",
        "title": "Scenario 3 — Vegetation + Minerals (Pure Pixels)",
        "subtitle": "8 Materials: 7 Vegetation + 1 Mineral (Pure pixels)",
        "materials": [
            "Oak Oak-Leaf-1", "Lodgepole-Pine", "NICKLEAVES:HealthyLeaf1", "Aspen Aspen-1",
            "Douglas-Fir", "Aspen Aspen-4", "Conifer-Meadow", "Goethite0.02+Quartz"
        ],
        "true_count": 8,
        "SNR": SNR,
        "Variance": 1,
        "shading": False,
        "tile_size": TILE_SIZE,
        "description": "8 materials (vegetation + mineral), pure pixels"
    },
    "S4_VegMineralsMixed": {
        "name": "S4: Veg + Minerals (Mixed)",
        "title": "Scenario 4 — Vegetation + Minerals in Mixed Pixels",
        "subtitle": "8 Materials: 7 Vegetation + 1 Mineral (Low variance / highly mixed Dirichlet)",
        "materials": [
            "Oak Oak-Leaf-1", "Lodgepole-Pine", "NICKLEAVES:HealthyLeaf1", "Aspen Aspen-1", "Douglas-Fir", "Aspen Aspen-4", "Conifer-Meadow", "Goethite0.02+Quartz"
        ],
        "true_count": 8,
        "SNR": SNR,
        "Variance": 0.25,
        "shading": False,
        "tile_size": TILE_SIZE,
        "description": "8 materials (vegetation + mineral), mixed pixels"
    },
    "S5_DistinctVegMixed": {
        "name": "S5: Veg (Mixed)",
        "title": "Scenario 5 — Vegetation in Mixed Pixels",
        "subtitle": "5 Vegetation Species (Low variance / highly mixed Dirichlet)",
        "materials": [
            "Oak Oak-Leaf-1", "Lodgepole-Pine", "Aspen Aspen-1",
            "Douglas-Fir", "NICKLEAVES:HealthyLeaf1"
        ],
        "true_count": 5,
        "SNR": SNR,
        "Variance": 0.25,
        "shading": False,
        "tile_size": TILE_SIZE,
        "description": "5 distinct vegetation species, mixed pixels"
    },
    "S6_VegModifiedPure": {
        "name": "S6: Veg Single Modification (Pure)",
        "title": "Scenario 6 — Modified Vegetation (Pure Pixels)",
        "subtitle": "5 Vegetation Species incl. Tar Spot Diseased Maple (Pure)",
        "materials": [
            "Oak Oak-Leaf-1", "Lodgepole-Pine", "Aspen Aspen-1",
            "Douglas-Fir", "NICKLEAVES:DiseaseLeaf1"
        ],
        "true_count": 5,
        "SNR": SNR,
        "Variance": 1,
        "shading": False,
        "tile_size": TILE_SIZE,
        "description": "5 vegetation species, one with tar spot fungal modification (Norway Maple), pure pixels"
    },
    "S7_VegModifiedMixed": {
        "name": "S7: Veg Single Modification (Mixed)",
        "title": "Scenario 7 — Modified Vegetation in Mixed Pixels",
        "subtitle": "5 Vegetation Species incl. Tar Spot Diseased Maple (Mixed)",
        "materials": [
            "Oak Oak-Leaf-1", "Lodgepole-Pine", "Aspen Aspen-1",
            "Douglas-Fir", "NICKLEAVES:DiseaseLeaf1"
        ],
        "true_count": 5,
        "SNR": SNR,
        "Variance": 0.25,
        "shading": False,
        "tile_size": TILE_SIZE,
        "description": "5 vegetation species, one with tar spot fungal modification (Norway Maple), mixed pixels"
    }
}




# Ensure UTF-8 stdout on Windows to support Unicode symbols in tables
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
    Estimates the scree plot elbow using the Kneedle algorithm (kneed KneeLocator).
    The Scree curve (% explained variance) is convex and decreasing.
    """
    var_arr = np.asarray(variance_percentages, dtype=np.float64)
    n = len(var_arr)
    if n <= 2 or np.ptp(var_arr) < 1e-6:
        return 1
    
    x = np.arange(1, n + 1)
    y = var_arr

    try:
        kl = KneeLocator(x, y, curve='convex', direction='decreasing')
        if kl.knee is not None and kl.knee > 0:
            return int(kl.knee)
    except Exception:
        pass
    return 1


def estimate_optimal_coordinates(curve_values):
    """
    Optimal Coordinates (n_oc) method from Raîche et al. (2013), Equations (1)-(4).
    https://doi.org/10.1027/1614-2241/a000051

    Pure published scree test: evaluates whether each point k falls on or
    above the extrapolated linear chord formed between subsequent coordinate (k + 1)
    and the last coordinate p of the scree tail:
        slope = (y_p - y_{k+1}) / (p - (k + 1))
        y_hat_k = y_{k+1} - slope
        Retain k if y_k >= y_hat_k; stop on first failure.
    """
    y = np.asarray(curve_values, dtype=np.float64)
    p = len(y)
    best_k = 1
    for i in range(p - 2):
        y_next = y[i + 1]
        y_last = y[-1]
        dx = (p - 1) - (i + 1)
        slope = (y_last - y_next) / dx
        y_hat = y_next - slope  # extrapolated back 1 unit to i
        if y[i] >= y_hat:
            best_k = i + 1
        else:
            break
    return best_k


def estimate_optimal_coordinate_count(curve_values, num_bands=None):
    """
    Unified function for both PCA scree and MaxD orthogonal projection heights.
    Incorporates the float32 precision threshold by default and evaluates
    alternate stopping conditions prior to entering estimate_optimal_coordinates.
    """
    y = np.asarray(curve_values, dtype=np.float64)
    bands = num_bands if num_bands is not None else len(y)

    # Subspace dimension bound
    if num_bands is not None and len(y) > num_bands:
        y = y[:num_bands]

    # Incorporate threshold based on float32 precision (PCA not impacted as var % >> 1e-7)
    rho_max = 0.7
    float32_thresh = rho_max * np.sqrt(bands) * np.finfo(np.float32).eps

    # Alternate stopping condition: all points below float32 threshold
    valid = np.where(y > float32_thresh)[0]
    if len(valid) == 0:
        return 1

    # Trim trailing saturated subspace zeros to valid floor + 1 baseline tail coordinate
    tail_idx = min(len(y), valid[-1] + 2)
    y = y[:tail_idx]

    # Alternate stopping conditions: minimal coordinate conditions
    p = len(y)
    if p <= 1 or np.ptp(y) < 1e-12:
        return 1
    if p == 2:
        return 1

    # Enter the published Raîche optimal coordinates function
    best_k = estimate_optimal_coordinates(y)
    return min(bands, best_k)


def estimate_pca_oc(variance_percentages, num_bands=None):
    """
    Convenience wrapper for PCA Scree Optimal Coordinates using estimate_optimal_coordinate_count.
    """
    return estimate_optimal_coordinate_count(variance_percentages, num_bands=num_bands)




def estimate_maxd_elbow(manhattan_sum):
    """
    Estimates the MaxD elbow from the cumulative Manhattan distance curve
    (cumulative sum of orthogonal heights) using the Kneedle algorithm (kneed KneeLocator).
    The cumulative Manhattan curve is concave and increasing towards a plateau.
    """
    y = np.asarray(manhattan_sum, dtype=np.float64)
    n = len(y)
    if n <= 1 or np.ptp(y) < 1e-8:
        return 1
        
    # Anchor at origin (0 endmembers = 0 distance) to detect initial step
    y_full = np.concatenate(([0.0], y))
    x_full = np.arange(0, len(y_full))
    
    try:
        kl = KneeLocator(x_full, y_full, curve='concave', direction='increasing')
        if kl.knee is not None and kl.knee > 0:
            return int(kl.knee)
    except Exception:
        pass
    return 1


def estimate_heights_knee_count(heights_signal, num_bands=None):
    """
    Estimates the material count from the MaxD orthogonal projection height curve
    using the kneed KneeLocator to select the last height before the values plateau.
    
    When SNR is finite, heights decay geometrically across dimensions and flatten
    into a noise plateau (rather than dropping to 0). In logarithmic space,
    this transition forms a sharp convex knee that KneeLocator reliably detects.
    When heights drop to numerical zero (below float32 precision, e.g. when bands
    are exhausted), the last valid height before the drop is selected.
    """
    height_array = np.asarray(heights_signal, dtype=np.float64)
    n = len(height_array)
    if n <= 1 or np.ptp(height_array) < 1e-8:
        return 1, "Single spectrum / zero heights"
        
    bands = num_bands if num_bands is not None else n
    rho_max = 0.7
    float32_thresh = rho_max * np.sqrt(bands) * np.finfo(np.float32).eps
    
    # Check if first signal height is already numerical zero
    if height_array[0] <= float32_thresh:
        return 1, "Below numerical threshold at k=1"
        
    # Criterion 1: If heights drop to numerical zero / below float32 precision (subspace saturated)
    for k in range(1, n):
        if height_array[k] <= float32_thresh:
            return min(bands, k), f"Drop below float32 threshold at k={k+1}"
            
    # Criterion 2: KneeLocator in logarithmic space to find the last height before noise plateau
    safe_h = np.maximum(height_array, float32_thresh)
    log_h = np.log10(safe_h)
    x = np.arange(1, n + 1)
    
    try:
        kl = KneeLocator(x, log_h, curve='convex', direction='decreasing', S=1.0)
        if kl.knee is not None and kl.knee > 0:
            return min(bands, int(kl.knee)), f"Kneedle log plateau knee at k={int(kl.knee)}"
    except Exception:
        pass
        
    # Fallback to skimage threshold_triangle in log space
    try:
        tri_thresh = threshold_triangle(log_h)
        above_thresh = np.where(log_h >= tri_thresh)[0]
        if len(above_thresh) > 0:
            best_k = int(above_thresh[-1] + 1)
            return min(bands, best_k), f"Threshold triangle log knee at k={best_k}"
    except Exception:
        pass
        
    return min(bands, 1), "Baseline"


def estimate_heights_optimal_coordinate_count(heights_signal, num_bands=None):
    """
    Estimates the material count from the MaxD orthogonal projection height curve
    using estimate_optimal_coordinate_count.
    """
    k = estimate_optimal_coordinate_count(heights_signal, num_bands=num_bands)
    return k, f"Optimal coordinates knee at k={k}"


# Backward compatibility alias
estimate_maxd_count = estimate_heights_knee_count

# --- Simulation Runner for a Single Scenario x Sensor ---

def run_scenario_on_sensor(scenario_id, sensor_id, soup=None, snr=None, tile_size=None):
    """
    Runs one scenario configuration on one sensor.
    Returns a dictionary of all results, curves, metrics, and metadata.
    """
    sc_cfg = SCENARIOS[scenario_id].copy()
    if snr is not None:
        sc_cfg["SNR"] = snr
    if tile_size is not None:
        sc_cfg["tile_size"] = tile_size
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
        if mat_name.startswith("NICKLEAVES:"):
            # Load user-collected leaf spectra from sensor-resampled CSV
            leaf_id = mat_name.split(":", 1)[1]
            csv_path = os.path.join(NICKS_LEAVES_DIR, f"{sensor_id}_{leaf_id}.csv")
            df_leaf = pd.read_csv(csv_path)
            loaded_materials[mat_name] = df_leaf['DHR_mean'].values.astype(np.float32)
        else:
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
    variance = sc_cfg.get("Variance")
    snr = sc_cfg.get("SNR")
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
    # Evaluate at least 6 beyond true count (do not bound by number of pixels)
    num_em_eval = max(true_count + 6, 12)
    
    pca_k = min(num_em_eval, num_bands, pixels.shape[1])
    rho_max = 0.7
    float32_minimum_threshold = rho_max * np.sqrt(num_bands) * np.finfo(np.float32).eps
    
    # 6. Run MaxD parallelotope volume & height extraction
    if np.ptp(pixels, axis=1).max() < float32_minimum_threshold:
        em = np.repeat(pixels[:, [0]], num_em_eval, axis=1)
        vol = np.full(num_em_eval, -np.inf, dtype=np.float64)
        heights = np.zeros(num_em_eval, dtype=np.float64)
        indices = np.zeros(num_em_eval, dtype=int)
        noise_heights = np.zeros(num_em_eval, dtype=np.float64)
        noise_norms_mean = np.zeros(num_em_eval, dtype=np.float64)
        
        # Zero variance dataset (single spectrum): 1 component explains 100% of signal
        pca_var_pct = np.zeros(pca_k, dtype=np.float64)
        pca_var_pct[0] = 100.0
        cumulative_var = np.full(pca_k, 100.0, dtype=np.float64)
    else:
        em, vol, heights, indices, noise_heights, noise_norms_mean = maximumDistance_parallelotopeVolumes_noise(
            pixels, num_em_eval, noise_pixels=noise_pixels
        )
        # 7. Run PCA for Scree plot
        pca = PCA(n_components=pca_k)
        pca.fit(pixels.T)
        pca_var_pct = pca.explained_variance_ratio_ * 100
        cumulative_var = np.cumsum(pca_var_pct)
    
    # Normalized heights relative to h_1
    h1 = heights[1] if heights[1] > 0 else 1.0
    heights_normalized = heights / h1
    
    # Manhattan distance summation (cumulative sum of orthogonal heights)
    h_signal = heights[1:]
    manhattan_sum = np.cumsum(h_signal)
    
    # 8. Compute Estimates
    pca_1pct_est = estimate_pca_1pct(pca_var_pct)
    pca_elbow_est = estimate_pca_elbow(pca_var_pct)
    pca_oc_est = estimate_optimal_coordinate_count(pca_var_pct, num_bands=num_bands)
    maxd_elbow_est = estimate_maxd_elbow(manhattan_sum)
    heights_knee_est, heights_knee_reason = estimate_heights_knee_count(h_signal, num_bands=num_bands)
    heights_oc_est = estimate_optimal_coordinate_count(h_signal, num_bands=num_bands)
    heights_oc_reason = f"Optimal coordinates knee at k={heights_oc_est}"
    maxd_est = heights_knee_est
    
    return {
        "scenario_id": scenario_id,
        "sensor_id": sensor_id,
        "sensor_display": sensor_cfg["display_name"],
        "is_multispectral": sensor_cfg["is_multispectral"],
        "num_bands": num_bands,
        "wavelengths": clean_wavelengths,
        "materials": loaded_materials,
        "true_count": true_count,
        "SNR": sc_cfg["SNR"],
        "num_em_eval": num_em_eval,
        "heights": heights,
        "heights_normalized": heights_normalized,
        "manhattan_sum": manhattan_sum,
        "noise_heights": noise_heights,
        "noise_norms_mean": noise_norms_mean,
        "volumes": vol,
        "pca_var_pct": pca_var_pct,
        "cumulative_var": cumulative_var,
        "pca_1pct_est": pca_1pct_est,
        "pca_elbow_est": pca_elbow_est,
        "pca_oc_est": pca_oc_est,
        "heights_knee_est": heights_knee_est,
        "heights_knee_reason": heights_knee_reason,
        "heights_oc_est": heights_oc_est,
        "heights_oc_reason": heights_oc_reason,
        "maxd_est": heights_knee_est,
        "maxd_elbow_est": maxd_elbow_est,
        "maxd_reason": heights_knee_reason,
        "tile_size": tile_size,
        "img_noisy": layer_noisy,
        "indices": indices
    }


# --- Plotting Functions ---

def format_material_name(mat_name):
    """Formats material name for readable, concise plot legends."""
    return (mat_name
            .replace("Goethite0.02+Quartz", "Goethite+Qtz")
            .replace("Oak Oak-Leaf-1", "Oak Leaf")
            .replace("Maple_Leaves", "Maple")
            .replace("NICKLEAVES:HealthyLeaf1", "Maple (Healthy)")
            .replace("NICKLEAVES:DiseaseLeaf1", "Maple (Tar Spot)")
            .replace("Lodgepole-Pine", "Pine")
            .replace("Conifer-Meadow", "Meadow")
            .replace("Aspen Aspen-", "Aspen-"))


def generate_rgb_from_multispectral(img_data, wavelengths, low=5.0, high=95.0, gamma=1):
    """
    Directly extracts Red, Green, and Blue bands from multispectral imagery
    (Landsat 8, Sentinel-2, WorldView-3) and applies percentile contrast scaling.
    Preserves relative spectral color balance across channels.
    
    Parameters:
        img_data (np.ndarray): Shape (bands, height, width).
        wavelengths (np.ndarray): Band center wavelengths (in micrometers or nm).
        low (float): Lower percentile for contrast stretching.
        high (float): Higher percentile for contrast stretching.
        gamma (float): Gamma correction parameter.
        
    Returns:
        rgb_image (np.ndarray): Shape (height, width, 3), float32 in [0, 1].
    """
    wv = np.asarray(wavelengths, dtype=np.float64)
    if np.nanmean(wv) > 100.0:
        wv_um = wv / 1000.0
    else:
        wv_um = wv
        
    # Find bands closest to standard RGB wavelengths (microns):
    # Red: ~0.66 um, Green: ~0.55 um, Blue: ~0.48 um
    idx_r = int(np.argmin(np.abs(wv_um - 0.66)))
    idx_g = int(np.argmin(np.abs(wv_um - 0.55)))
    idx_b = int(np.argmin(np.abs(wv_um - 0.48)))
    
    r_band = img_data[idx_r]
    g_band = img_data[idx_g]
    b_band = img_data[idx_b]
    
    valid_mask = ~(np.isnan(r_band) | np.isnan(g_band) | np.isnan(b_band))
    all_valid = np.concatenate([r_band[valid_mask], g_band[valid_mask], b_band[valid_mask]])
    
    if all_valid.size == 0:
        return np.zeros((*r_band.shape, 3), dtype=np.float32)
        
    p_low, p_high = np.percentile(all_valid, (low, high))
    
    rgb = np.zeros((*r_band.shape, 3), dtype=np.float32)
    for i, band in enumerate([r_band, g_band, b_band]):
        if p_high > p_low + 1e-5:
            stretched = np.clip((band - p_low) / (p_high - p_low), 0.0, 1.0)
        else:
            stretched = np.clip(band / max(p_high, 0.25), 0.0, 1.0)
        if gamma != 1.0 and gamma > 0:
            stretched = np.power(stretched, 1.0 / gamma)
        rgb[:, :, i] = stretched
        
    return rgb


def plot_scenario_composite(res, output_path, dpi=300):
    """
    Generates the composite figure for one Scenario x Sensor:
      ┌──────────────────────────────────────────────────────────┐
      │ Scenario Title: "N Distinct Vegetation Species, Pure"    │
      ├─────────────────────────┬────────────────────────────────┤
      │ (a) MaxD Height Curve   │ (b) PCA Scree Plot             │
      ├─────────────────────────┼────────────────────────────────┤
      │ (c) Source Spectra      │ (d) Simulated Tile (RGB)       │
      └─────────────────────────┴────────────────────────────────┘
    """
    sc_cfg = SCENARIOS[res["scenario_id"]]
    true_count = res["true_count"]
    num_bands = res["num_bands"]
    sensor_display = res["sensor_display"]
    
    fig = plt.figure(figsize=(15, 11))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.15, 1.0], hspace=0.38, wspace=0.26, top=0.83, bottom=0.07, left=0.08, right=0.94)
    
    actual_snr = res.get("SNR", sc_cfg.get("SNR"))
    snr_disp = "inf" if np.isinf(actual_snr) else f"{actual_snr}"
    t_size = res.get("tile_size", sc_cfg.get("tile_size", 3))
    title_text = f"{sc_cfg['title']}\n{sensor_display} ({num_bands} Bands) | Tile: {t_size}x{t_size} | SNR={snr_disp} | True Material Count = {true_count}"
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
    if nh_vals is not None and np.any(nh_vals > 0):
        ax_a.plot(k_vals, nh_vals, marker='x', linestyle=':', color='#ff7f0e', linewidth=1.5, alpha=0.85, label=r'Noise Height ($nh_k$)', zorder=3)
    if nm_vals is not None and np.any(nm_vals > 0):
        ax_a.plot(k_vals, nm_vals, marker='s', markersize=4, linestyle='--', color='#2ca02c', linewidth=1.5, alpha=0.85, label='Mean Noise Norm', zorder=3)
        
    # Vertical line at True Count
    ax_a.axvline(x=true_count, color='#d62728', linestyle='--', linewidth=2.2, zorder=5, label=f'True Count (k={true_count})')
    
    # Twin axis for Manhattan Distance (Cumulative Sum of Heights)
    ax_a_twin = ax_a.twinx()
    manhattan_sum = res.get("manhattan_sum", np.cumsum(h_signal))
    line_manhattan, = ax_a_twin.plot(
        k_vals, manhattan_sum,
        marker='^', markersize=4, linestyle='-.', color='#9467bd',
        linewidth=1.5, alpha=0.75, label=r'Cumulative Manhattan Dist ($\sum h_k$)', zorder=2
    )
    ax_a_twin.set_ylabel(r'Cumulative Manhattan Distance ($\sum h_k$)', color='#9467bd', fontsize=10)
    ax_a_twin.tick_params(axis='y', labelcolor='#9467bd')
    ax_a_twin.set_ylim(bottom=0, top=max(manhattan_sum) * 1.08 if max(manhattan_sum) > 0 else 1.0)
    ax_a_twin.grid(False)
    
    # Plot prediction markers on source data
    hSum_elbow = res.get("maxd_elbow_est", "N/A")
    knee_est = res.get("heights_knee_est", res.get("maxd_est", "N/A"))
    oc_est = res.get("heights_oc_est", "N/A")
    
    if isinstance(knee_est, (int, np.integer)) and 1 <= knee_est <= len(h_signal):
        ax_a.scatter(
            knee_est, h_signal[knee_est - 1],
            marker='D', s=90, color='#e7298a', edgecolor='black', linewidth=1.5,
            zorder=7, label=f'Heights Knee (k={knee_est})'
        )
    if isinstance(oc_est, (int, np.integer)) and 1 <= oc_est <= len(h_signal):
        ax_a.scatter(
            oc_est, h_signal[oc_est - 1],
            marker='o', s=180, facecolors='none', edgecolors='#d62728', linewidth=2.5,
            zorder=8, label=f'Heights OC (k={oc_est})'
        )
    if isinstance(hSum_elbow, (int, np.integer)) and 1 <= hSum_elbow <= len(manhattan_sum):
        ax_a_twin.scatter(
            hSum_elbow, manhattan_sum[hSum_elbow - 1],
            marker='s', s=100, color='#9467bd', edgecolor='black', linewidth=1.5,
            zorder=7, label=f'Manhattan Elbow (k={hSum_elbow})'
        )
    
    # Secondary X-axis on top: Subspace Dimensionality (k - 1)
    secax_a = ax_a.secondary_xaxis('top', functions=(lambda x: x - 1, lambda x: x + 1))
    secax_a.set_xlabel('Subspace Dimensionality ($k - 1$)', fontsize=10, labelpad=5)
    secax_a.set_xticks(k_vals - 1)
    
    # Symlog y-axis to visualize both large initial heights and noise plateau
    rho_max = 0.7
    float32_minimum_threshold = rho_max * np.sqrt(num_bands) * np.finfo(np.float32).eps
    ax_a.set_yscale('symlog', linthresh=float32_minimum_threshold)

    # Dynamically update the bottom value of the Orthogonal Distance y scale
    plotted_heights = [h_signal[h_signal > 0] if np.any(h_signal > 0) else h_signal]
    if nh_vals is not None and np.any(nh_vals > 0):
        plotted_heights.append(nh_vals[nh_vals > 0])
    if nm_vals is not None and np.any(nm_vals > 0):
        plotted_heights.append(nm_vals[nm_vals > 0])
    min_plotted = float(np.min(np.concatenate(plotted_heights))) if plotted_heights else 1e-4

    ax_a.set_ylim(bottom=min_plotted * 0.8)
    ax_a.set_xlabel('Number of Extracted Endmembers ($k$)', fontsize=11)
    ax_a.set_ylabel('Orthogonal Distance (Height $h_k$)', fontsize=11)
    ax_a.set_title('(a) MaxD Orthogonal Height Curve', fontsize=12, fontweight='bold', pad=25)
    max_k_plot = max(len(k_vals), true_count)
    ax_a.set_xticks(np.arange(1, max_k_plot + 1))
    #ax_a.set_xlim(0.5, max_k_plot + 3.5) # Do not constrain maximum distance results to PCA limits
    ax_a.grid(True, linestyle='--', alpha=0.6)
    
    # Combined legend for ax_a and ax_a_twin
    handles_a, labels_a = ax_a.get_legend_handles_labels()
    handles_a_twin, labels_a_twin = ax_a_twin.get_legend_handles_labels()
    ax_a.legend(handles_a + handles_a_twin, labels_a + labels_a_twin, loc='best', fontsize=8, framealpha=0.9)
    
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
    
    # Plot prediction markers on source data
    pca_1pct = res["pca_1pct_est"]
    pca_elbow = res["pca_elbow_est"]
    pca_oc = res.get("pca_oc_est", "N/A")
    var_arr = res["pca_var_pct"]
    
    if isinstance(pca_elbow, (int, np.integer)) and 1 <= pca_elbow <= len(var_arr):
        ax_b.scatter(
            pca_elbow, var_arr[pca_elbow - 1],
            marker='D', s=90, color='#ff7f0e', edgecolor='black', linewidth=1.5,
            zorder=7, label=f'PCA Elbow (PC={pca_elbow})'
        )
    if isinstance(pca_1pct, (int, np.integer)) and 1 <= pca_1pct <= len(var_arr):
        ax_b.scatter(
            pca_1pct, var_arr[pca_1pct - 1],
            marker='s', s=150, facecolors='none', edgecolors='#e7298a', linewidth=2.2,
            zorder=8, label=f'PCA $\\geq$ 1% (PC={pca_1pct})'
        )
    if isinstance(pca_oc, (int, np.integer)) and 1 <= pca_oc <= len(var_arr):
        ax_b.scatter(
            pca_oc, var_arr[pca_oc - 1],
            marker='o', s=220, facecolors='none', edgecolors='#d62728', linewidth=2.2,
            zorder=9, label=f'PCA OC (PC={pca_oc})'
        )
    
    ax_b.set_xlabel('Principal Component Number (PC#)', fontsize=11)
    ax_b.set_ylabel('Individual Explained Variance (%)', fontsize=11)
    ax_b.set_title('(b) PCA Scree Plot (Explained Variance)', fontsize=12, fontweight='bold', pad=25)
    max_pc_plot = max(len(pc_vals), true_count)
    ax_b.set_xticks(np.arange(1, max_pc_plot + 1))
    ax_b.set_xlim(0.5, max_pc_plot + 0.5)
    ax_b.grid(True, linestyle='--', alpha=0.6)
    ax_b.set_yscale('symlog', linthresh=0.01)
    ax_b.set_ylim(bottom=0, top=min(105, max(res["pca_var_pct"]) * 1.08))
    
    # Combine legend handles
    handles_b, labels_b = ax_b.get_legend_handles_labels()
    handles_b_twin, labels_b_twin = ax_b_twin.get_legend_handles_labels()
    ax_b.legend(handles_b + handles_b_twin, labels_b + labels_b_twin, loc='best', fontsize=8.0, framealpha=0.9)
    
    # -------------------------------------------------------------------------
    # Panel (c): Input Pixel Spectra (with Noise/SNR & Mixing) & Sensor Bands
    # -------------------------------------------------------------------------
    ax_c = fig.add_subplot(gs[1, 0])
    wv = res["wavelengths"]
    img_data = res["img_noisy"]
    pixels = img_data.reshape(img_data.shape[0], -1)
    num_pixels = pixels.shape[1]
    
    alpha = max(0.06, min(0.6, 3.5 / np.sqrt(num_pixels)))
    if res["is_multispectral"] or num_bands <= 20:
        ax_c.plot(wv, pixels, marker='o', markersize=2.5, linewidth=0.9, color='#1f77b4', alpha=alpha)
    else:
        ax_c.plot(wv, pixels, linewidth=0.9, color='#1f77b4', alpha=alpha)
        
    # Band overlay for multispectral sensors (draw rug / vertical guidelines)
    if res["is_multispectral"] or num_bands <= 20:
        for b_idx, b_wv in enumerate(wv):
            ax_c.axvline(x=b_wv, color='#7f7f7f', linestyle=':', alpha=0.35, zorder=1)
            
    snr_val = res.get("SNR", "inf")
    snr_str = "inf" if (snr_val is not None and np.isinf(snr_val)) else str(snr_val)
    ax_c.set_title(f"(c) Input Pixel Spectra ({num_pixels} Pixels, SNR={snr_str}, {num_bands} Bands)", fontsize=12, fontweight='bold', pad=10)
    ax_c.set_xlabel(r'Wavelength ($\mu$m)', fontsize=11)
    ax_c.set_ylabel('Reflectance (0.0 - 1.0)', fontsize=11)
    y_min = min(-0.05, float(np.percentile(pixels, 0.1)) - 0.05)
    y_max = max(1.05, float(np.percentile(pixels, 99.9)) + 0.05)
    ax_c.set_ylim(y_min, y_max)
    ax_c.set_xlim(min(wv) - 0.05, max(wv) + 0.05)
    ax_c.grid(True, linestyle='--', alpha=0.6)
    ax_c.plot([], [], color='#1f77b4', linewidth=1.5, label=f'{num_pixels} Pixels (SNR={snr_str})')
    ax_c.legend(loc='upper right', fontsize=8.5, framealpha=0.85)
    
    # -------------------------------------------------------------------------
    # Panel (d): Simulated Tile True Color RGB Image
    # -------------------------------------------------------------------------
    ax_d = fig.add_subplot(gs[1, 1])
    img_data = res["img_noisy"]
    
    # Low spectral resolution multispectral sensors (Landsat 8, Sentinel-2, WorldView-3)
    # do not work accurately with CIE 1931 CMF integration; use direct RGB band extraction
    # with contrast scaling instead.
    if res.get("is_multispectral", False) or res["sensor_id"] in ["rsLandsat8", "rsSentinel2", "rsWorldView3"]:
        rgb_disp = generate_rgb_from_multispectral(img_data, res["wavelengths"])
    else:
        wv_nm = res["wavelengths"] * 1e3
        rgb_disp = generate_rgba_from_hsi(img_data, wv_nm)
    
    ax_d.imshow(rgb_disp)
    h_pix, w_pix = img_data.shape[1], img_data.shape[2]
    ax_d.set_title(f"(d) Simulated {h_pix}x{w_pix} Tile (True Color RGB)", fontsize=12, fontweight='bold', pad=10)
    ax_d.set_xlabel('Pixel Column (X)', fontsize=10)
    ax_d.set_ylabel('Pixel Row (Y)', fontsize=10)
    ax_d.set_xticks(np.arange(0, w_pix, max(1, w_pix // 5)))
    ax_d.set_yticks(np.arange(0, h_pix, max(1, h_pix // 5)))
    
    # Overlay extracted endmember locations (first true_count endmembers)
    if "indices" in res and res["indices"] is not None:
        idx_to_plot = res["indices"][:true_count]
        r_coords, c_coords = np.unravel_index(idx_to_plot, (h_pix, w_pix))
        ax_d.scatter(
            c_coords, r_coords,
            facecolors='none', edgecolors='#ffff00', marker='o',
            s=70, linewidth=1.8, label=f'First {true_count} MaxD Endmembers', zorder=5
        )
        for i_em, (r_pos, c_pos) in enumerate(zip(r_coords, c_coords)):
            ax_d.text(
                c_pos + 0.6, r_pos - 0.6, str(i_em + 1),
                color='#ffff00', fontsize=8, fontweight='bold',
                zorder=6
            )
        ax_d.legend(loc='lower right', fontsize=8, framealpha=0.85)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"  [Figure Saved] {output_path}")


def plot_cross_scenario_summary(sensor_id, all_sensor_results, output_path, dpi=300):
    """
    Produces the Cross-Scenario Summary Figure:
      Row 1 (Top):
        Left: Normalized MaxD Heights (h_k / h_1) vs. k for S1..S5
        Right: PCA Scree Plot (% Variance vs. PC#) for S1..S5
      Row 2 (Middle):
        Visual representation of simulated tiles used for scenarios S1..S5
      Row 3 (Bottom):
        Input pixel spectra (img_noisy) in small plots below scenarios
    """
    sensor_cfg = SENSOR_CONFIGS[sensor_id]
    sensor_display = sensor_cfg["display_name"]
    
    # Identify active scenarios to display (skipping control S0)
    sc_list = [sc_id for sc_id in SCENARIOS.keys() if not sc_id.startswith("S0") and sc_id in all_sensor_results]
    num_sc = len(sc_list)
    if num_sc == 0:
        return
        
    active_res = all_sensor_results.get(sc_list[0]) if sc_list else None
    snr_val = active_res.get("SNR") if active_res else None
    snr_suffix = f" | SNR={'inf' if (snr_val is not None and np.isinf(snr_val)) else snr_val}" if snr_val is not None else ""
    
    fig = plt.figure(figsize=(max(20, 3.8 * num_sc), 15.5))
    gs = gridspec.GridSpec(
        3, 2 * num_sc,
        figure=fig,
        height_ratios=[1.7, 1.0, 1.1],
        hspace=0.42,
        wspace=0.36,
        top=0.89,
        bottom=0.07,
        left=0.06,
        right=0.97
    )
    
    sc_nums = [sc_id.split('_')[0] for sc_id in sc_list]
    sc_range_label = f"{sc_nums[0]}\u2013{sc_nums[-1]}" if len(sc_nums) > 1 else sc_nums[0] if sc_nums else ""
    t_size = active_res.get("tile_size") if active_res else None
    tile_suffix = f" | Tile: {t_size}x{t_size}" if t_size is not None else ""
    title_text = (
        f"Cross-Scenario Evaluation: MaxD Complexity Curves, Simulated Scenes & Source Spectra\n"
        f"Sensor: {sensor_display} ({sensor_cfg['nominal_bands']} Nominal Bands){tile_suffix}{snr_suffix} | Scenarios ({sc_range_label})"
    )
    fig.suptitle(title_text, fontsize=15, fontweight='bold', y=0.965)
    
    colors = plt.colormaps['tab10'].colors
    markers = ['o', 's', '^', 'D', 'v', 'p', 'P', '*']
    sc_colors = {sc_id: colors[i % len(colors)] for i, sc_id in enumerate(sc_list)}
    sc_markers = {sc_id: markers[i % len(markers)] for i, sc_id in enumerate(sc_list)}
    
    # -------------------------------------------------------------------------
    # Row 1: (a) Normalized MaxD Heights & (b) PCA Scree Plots
    # -------------------------------------------------------------------------
    ax_h = fig.add_subplot(gs[0, :num_sc])
    ax_scree = fig.add_subplot(gs[0, num_sc:])
    
    for sc_id in sc_list:
        res = all_sensor_results[sc_id]
        sc_info = SCENARIOS[sc_id]
        true_k = sc_info["true_count"]
        color = sc_colors[sc_id]
        marker = sc_markers[sc_id]
        
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
    num_bands = sensor_cfg["nominal_bands"]
    rho_max = 0.7
    float32_minimum_threshold = rho_max * np.sqrt(num_bands) * np.finfo(np.float32).eps
    ax_h.set_title('(a) Normalized MaxD Heights ($h_k / h_1$) vs. Endmembers ($k$)\nStars indicate True Material Count', fontsize=12, fontweight='bold', pad=10)
    ax_h.set_xlabel('Number of Endmembers ($k$)', fontsize=11)
    ax_h.set_ylabel('Normalized Orthogonal Height ($h_k / h_1$)', fontsize=11)
    ax_h.set_yscale('symlog', linthresh=float32_minimum_threshold)
    ax_h.grid(True, linestyle='--', alpha=0.6)
    ax_h.legend(loc='best', fontsize=9, framealpha=0.9)
    
    # Right axis styling
    ax_scree.set_title('(b) PCA Scree Curves (% Variance vs. PC#)\nStars indicate True Material Count', fontsize=12, fontweight='bold', pad=10)
    ax_scree.set_xlabel('Principal Component Number (PC#)', fontsize=11)
    ax_scree.set_ylabel('Explained Variance (%)', fontsize=11)
    ax_scree.axhline(y=1.0, color='#7f7f7f', linestyle=':', linewidth=1.8, label='1.0% Threshold')
    ax_scree.set_yscale('symlog', linthresh=0.01)
    ax_scree.set_ylim(-0.1, 100.0)
    ax_scree.grid(True, linestyle='--', alpha=0.6)
    ax_scree.legend(loc='best', fontsize=9, framealpha=0.9)
    
    # -------------------------------------------------------------------------
    # Row 2: Simulated Tiles for Scenarios 1 through 5
    # -------------------------------------------------------------------------
    for col_idx, sc_id in enumerate(sc_list):
        ax_tile = fig.add_subplot(gs[1, 2 * col_idx : 2 * col_idx + 2])
        res = all_sensor_results[sc_id]
        sc_info = SCENARIOS[sc_id]
        true_k = sc_info["true_count"]
        sc_color = sc_colors[sc_id]
        img_data = res["img_noisy"]
        
        if res.get("is_multispectral", False) or res["sensor_id"] in ["rsLandsat8", "rsSentinel2", "rsWorldView3"]:
            rgb_disp = generate_rgb_from_multispectral(img_data, res["wavelengths"])
        else:
            wv_nm = res["wavelengths"] * 1e3
            rgb_disp = generate_rgba_from_hsi(img_data, wv_nm)
            
        ax_tile.imshow(rgb_disp)
        ax_tile.set_title(f"{sc_info['name']}\n(True Count = {true_k})", fontsize=10.5, fontweight='bold', color=sc_color, pad=6)
        ax_tile.set_xticks([])
        ax_tile.set_yticks([])
        ax_tile.set_xlabel(f"{img_data.shape[1]}×{img_data.shape[2]} pixels", fontsize=8.5, labelpad=4)
        
        # Color-coded border matching scenario
        for spine in ax_tile.spines.values():
            spine.set_edgecolor(sc_color)
            spine.set_linewidth(2.0)
            
        if col_idx == 0:
            ax_tile.set_ylabel("Simulated Scenes\n(RGB Tiles)", fontsize=10, fontweight='bold', labelpad=8)
            
    # -------------------------------------------------------------------------
    # Row 3: Input Pixel Spectra in Plots Below Scenarios
    # -------------------------------------------------------------------------
    for col_idx, sc_id in enumerate(sc_list):
        ax_spec = fig.add_subplot(gs[2, 2 * col_idx : 2 * col_idx + 2])
        res = all_sensor_results[sc_id]
        sc_info = SCENARIOS[sc_id]
        sc_color = sc_colors[sc_id]
        wv = res["wavelengths"]
        img_data = res["img_noisy"]
        pixels = img_data.reshape(img_data.shape[0], -1)
        num_pixels = pixels.shape[1]
        
        alpha = max(0.06, min(0.6, 3.5 / np.sqrt(num_pixels)))
        if res.get("is_multispectral", False) or sensor_cfg["nominal_bands"] <= 20:
            ax_spec.plot(wv, pixels, marker='o', markersize=2.0, linewidth=0.9, color=sc_color, alpha=alpha)
        else:
            ax_spec.plot(wv, pixels, linewidth=0.9, color=sc_color, alpha=alpha)
            
        y_min = min(-0.05, float(np.percentile(pixels, 0.1)) - 0.05)
        y_max = max(1.05, float(np.percentile(pixels, 99.9)) + 0.05)
        ax_spec.set_ylim(y_min, y_max)
        ax_spec.set_xlim(min(wv) - 0.05, max(wv) + 0.05)
        ax_spec.grid(True, linestyle='--', alpha=0.5)
        ax_spec.tick_params(labelsize=8)
        ax_spec.set_xlabel(r'Wavelength ($\mu$m)', fontsize=9)
        ax_spec.set_title(f"Input Spectra ({num_pixels} Pixels)", fontsize=9.5, fontweight='bold', color=sc_color, pad=4)
        
        # Color-coded border matching scenario
        for spine in ax_spec.spines.values():
            spine.set_edgecolor(sc_color)
            spine.set_linewidth(1.5)
            
        if col_idx == 0:
            ax_spec.set_ylabel("Reflectance\n(0.0 - 1.0)", fontsize=10, fontweight='bold', labelpad=8)
            
        ax_spec.plot([], [], color=sc_color, linewidth=1.5, label=f"{num_pixels} Pixels")
        ax_spec.legend(loc='upper right', fontsize=7, framealpha=0.75, handlelength=1.0, borderaxespad=0.2)
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"  [Summary Figure Saved] {output_path}")


def plot_prediction_error_vs_snr(df, output_path, dpi=300):
    """
    Plots material count prediction error (Estimated Count - True Count) vs. SNR.
    For each SNR value, creates a scatter plot of differences across evaluated scenarios.
    Markers uniquely identify prediction methods; colors indicate sensor source.
    """
    if df is None or len(df) == 0:
        print("  [Warning] Dataframe is empty, skipping prediction error vs. SNR plot.")
        return
        
    # Exclude S0 if other scenarios are present
    if "Scenario ID" in df.columns and len(df["Scenario ID"].unique()) > 1:
        df_plot = df[df["Scenario ID"] != "S0_HomogeneousVeg"].copy()
    else:
        df_plot = df.copy()
        
    METHOD_COLS = {
        "Heights Est (Knee)": {"name": "Heights (Knee)", "color": "#1f77b4", "aliases": ["Heights Est (Knee)", "Heights (Knee)", "MaxD Estimated Count"]},
        "Heights Est (OC)": {"name": "Heights (OC)", "color": "#2ca02c", "aliases": ["Heights Est (OC)", "Heights (OC)"]},
        #"Manhattan Est (Elbow)": {"name": "MaxD (Elbow)", "color": "#17becf", "aliases": ["MaxD Est (Elbow)", "MaxD (Elbow)"]},
        #"PCA Est (1% Threshold)": {"name": "PCA (1%)", "color": "#ff7f0e", "aliases": ["PCA Est (1% Threshold)", "PCA (1%)"]},
        "PCA Est (Elbow)": {"name": "PCA (Elbow)", "color": "#8c564b", "aliases": ["PCA Est (Elbow)", "PCA (Elbow)"]},
        "PCA Est (OC)": {"name": "PCA (OC)", "color": "#d62728", "aliases": ["PCA Est (OC)", "PCA (OC)"]},
    }

    SENSOR_MARKERS = {
        "Landsat 8": {"marker": "o", "size": 65},
        "Sentinel-2": {"marker": "s", "size": 60},
        "WorldView-3": {"marker": "D", "size": 60},
        "HyMAP (2014)": {"marker": "^", "size": 65},
        "AVIRIS (2014)": {"marker": "v", "size": 65},
        "ASD FieldSpec": {"marker": "*", "size": 95},
    }
    
    # Sort SNR values logically
    raw_snrs = df_plot["SNR"].unique()
    def snr_sort_key(s):
        try:
            val = float(s)
            return (1, 1e9) if np.isinf(val) else (0, val)
        except:
            return (1, 1e9) if str(s).lower() in ('inf', 'np.inf') else (0, float(s))
            
    sorted_snrs = sorted(raw_snrs, key=snr_sort_key)
    snr_labels = [
        r'$\infty$' if (str(s).lower() in ('inf', 'np.inf') or np.isinf(float(s))) 
        else str(int(float(s)) if float(s).is_integer() else float(s)) 
        for s in sorted_snrs
    ]
    snr_to_x = {s: i for i, s in enumerate(sorted_snrs)}
    
    fig_w = max(11.0, 1.6 * len(sorted_snrs))
    fig, ax = plt.subplots(figsize=(fig_w, 9.5))
    
    method_keys = list(METHOD_COLS.keys())
    num_methods = len(method_keys)
    method_offsets = np.linspace(-0.28, 0.28, num_methods)
    
    np.random.seed(42)
    
    min_err = 0
    max_err = 0
    
    for _, row in df_plot.iterrows():
        snr_val = row["SNR"]
        try:
            f_snr = float(snr_val)
        except:
            f_snr = np.inf
            
        matched_key = None
        for k in sorted_snrs:
            try:
                if np.isinf(f_snr) and (np.isinf(float(k)) or str(k).lower() in ('inf', 'np.inf')):
                    matched_key = k
                    break
                elif abs(f_snr - float(k)) < 1e-5:
                    matched_key = k
                    break
            except:
                if str(k).lower() == str(snr_val).lower():
                    matched_key = k
                    break
                    
        if matched_key is None:
            continue
        x_base = snr_to_x[matched_key]
        
        sensor = row.get("Sensor", "Unknown")
        s_info = SENSOR_MARKERS.get(sensor, {"marker": "o", "size": 65})
        marker = s_info["marker"]
        size = s_info["size"]
        
        try:
            true_k = float(row["True Count"])
        except (ValueError, TypeError, KeyError):
            continue
            
        for m_idx, (col_key, m_info) in enumerate(METHOD_COLS.items()):
            val = None
            for alias in m_info["aliases"]:
                if alias in row and not pd.isna(row[alias]) and row[alias] != "N/A":
                    val = row[alias]
                    break
            if val is None:
                continue
            try:
                pred_k = float(val)
            except (ValueError, TypeError):
                continue
                
            error = pred_k - true_k
            min_err = min(min_err, error)
            max_err = max(max_err, error)
            
            jitter_x = x_base + method_offsets[m_idx] + np.random.uniform(-0.025, 0.025)
            
            ax.scatter(
                jitter_x, error,
                marker=marker,
                s=size,
                color=m_info["color"],
                edgecolors='black',
                linewidths=0.6,
                alpha=0.82,
                zorder=4
            )
            
    # Horizontal zero error reference
    ax.axhline(0, color='black', linestyle='-', linewidth=1.8, zorder=3, alpha=0.9)
    ax.text(
        len(sorted_snrs) - 0.45, 0.25, 'Zero Error (Exact Match)',
        color='black', fontsize=9.5, fontweight='bold', ha='right', va='bottom', zorder=5,
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='none')
    )
    
    # Vertical dividers and alternating background shading
    if len(sorted_snrs) > 1:
        for div_x in np.arange(0.5, len(sorted_snrs) - 0.5, 1.0):
            ax.axvline(div_x, color='#d0d0d0', linestyle='--', linewidth=0.9, zorder=1)
        for i in range(len(sorted_snrs)):
            if i % 2 == 1:
                ax.axvspan(i - 0.5, i + 0.5, color='#f8f9fa', zorder=0)
                
    ax.set_xlim(-0.5, len(sorted_snrs) - 0.5)
    y_floor = min(-7, int(min_err - 1))
    y_ceil = max(13, int(max_err + 2))
    ax.set_ylim(y_floor - 0.5, y_ceil + 0.5)
    ax.set_yticks(np.arange(y_floor, y_ceil + 1, 1))
    
    ax.set_xticks(range(len(sorted_snrs)))
    ax.set_xticklabels(snr_labels, fontsize=12, fontweight='bold')
    ax.set_xlabel("Signal-to-Noise Ratio (SNR)", fontsize=13, fontweight='bold', labelpad=10)
    ax.set_ylabel(
        r"Prediction Error: Material Count Difference ($\hat{k} - k_{\mathrm{true}}$)" + "\n[ + Overestimation | 0 Exact Match | − Underestimation ]",
        fontsize=12, fontweight='bold', labelpad=10
    )
    ax.set_title(
        "Material Count Prediction Error vs. SNR Across Sensors & Scenarios\n(Colored by Prediction Method | Shaped by Sensor Source)",
        fontsize=14.5, fontweight='bold', pad=15
    )
    
    ax.grid(True, axis='y', linestyle=':', alpha=0.6, zorder=1)
    ax.set_axisbelow(True)
    
    # Legends: Prediction Method (Color) on left, Sensor Source (Marker) on right
    method_handles = [
        mlines.Line2D([], [], color=info["color"], marker='o', linestyle='None',
                      markersize=9, markeredgecolor='black', label=info["name"])
        for col_name, info in METHOD_COLS.items()
    ]
    sensor_handles = [
        mlines.Line2D([], [], color='#444444', marker=info["marker"], linestyle='None',
                      markersize=8, markeredgecolor='black', label=sensor)
        for sensor, info in SENSOR_MARKERS.items()
        if sensor in df_plot["Sensor"].values or len(df_plot) == 0
    ]
    if not sensor_handles:
        sensor_handles = [
            mlines.Line2D([], [], color='#444444', marker=info["marker"], linestyle='None',
                          markersize=8, markeredgecolor='black', label=sensor)
            for sensor, info in SENSOR_MARKERS.items()
        ]
        
    leg1 = ax.legend(
        handles=method_handles, title="Prediction Method (Color)",
        loc='upper left', bbox_to_anchor=(0.015, 0.98),
        fontsize=9.5, title_fontsize=10.5, framealpha=0.92, facecolor='white', edgecolor='#cccccc'
    )
    ax.add_artist(leg1)
    leg2 = ax.legend(
        handles=sensor_handles, title="Sensor Source (Marker)",
        loc='upper right', bbox_to_anchor=(0.985, 0.98),
        fontsize=9.5, title_fontsize=10.5, framealpha=0.92, facecolor='white', edgecolor='#cccccc'
    )
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"  [Prediction Error Figure Saved] {output_path}")


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
            t_size = res.get("tile_size", sc_cfg.get("tile_size", 3))
            rows.append({
                "Tile Size": f"{t_size}x{t_size}",
                "SNR": res.get("SNR", "N/A"),
                "Scenario ID": sc_id,
                "Scenario Name": sc_cfg["name"],
                "Sensor": res["sensor_display"],
                "Bands": res["num_bands"],
                "True Count": true_k,
                "Heights Est (Knee)": res.get("heights_knee_est", res.get("maxd_est")),
                "Heights Est (OC)": res.get("heights_oc_est", "N/A"),
                "MaxD Est (Elbow)": res.get("maxd_elbow_est", 1),
                "PCA Est (1% Threshold)": res["pca_1pct_est"],
                "PCA Est (Elbow)": res["pca_elbow_est"],
                "PCA Est (OC)": res.get("pca_oc_est", "N/A"),
                "Knee Correct": "Yes" if res.get("heights_knee_est", res.get("maxd_est")) == true_k else "No",
                "OC Correct": "Yes" if res.get("heights_oc_est") == true_k else "No",
                "PCA (1%) Correct": "Yes" if res["pca_1pct_est"] == true_k else "No",
                "PCA (OC) Correct": "Yes" if res.get("pca_oc_est") == true_k else "No"
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
            
        knee_val = row.get("Heights Est (Knee)", row.get("MaxD Estimated Count"))
        oc_val = row.get("Heights Est (OC)", "N/A")
        pca_1pct_val = row.get("PCA Est (1% Threshold)")
        pca_oc_val = row.get("PCA Est (OC)", "N/A")
        true_k = row["True Count"]
        
        knee_icon = "✓" if knee_val == true_k else f"~ ({knee_val})"
        oc_icon = "✓" if oc_val == true_k else f"~ ({oc_val})"
        pca_1pct_icon = "✓" if pca_1pct_val == true_k else f"✗ (says {pca_1pct_val})"
        pca_oc_icon = "✓" if pca_oc_val == true_k else f"✗ (says {pca_oc_val})"
        
        summaries.append({
            "Scenario": sc_cfg["name"],
            "True Count": true_k,
            "Heights (Knee)": knee_icon,
            "Heights (OC)": oc_icon,
            "PCA (1%)": pca_1pct_icon,
            "PCA (OC)": pca_oc_icon,
            "Description": sc_cfg["description"]
        })
    return pd.DataFrame(summaries)


# --- Main Execution Controller ---

def run_study(target_sensors, target_scenarios, output_dir, snr=8, tile_size=3, dpi=300):
    """
    Executes the simulation study for a specific SNR value and tile size across target sensors and scenarios.
    Saves composite figures, cross-scenario summaries, and comparison tables to output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    snr_disp = "inf" if np.isinf(snr) else str(snr)
    
    print("\n" + "=" * 80)
    print("SPECTRAL COMPLEXITY VS. PCA: CONTROLLED SIMULATION STUDY")
    print(f"Tile Size: {tile_size}x{tile_size}")
    print(f"SNR: {snr_disp}")
    print(f"Sensors ({len(target_sensors)}): {target_sensors}")
    print(f"Scenarios ({len(target_scenarios)}): {target_scenarios}")
    print(f"Output Directory: {output_dir}")
    print("=" * 80)
    
    all_results = {}
    
    for sensor_id in target_sensors:
        sensor_display = SENSOR_CONFIGS[sensor_id]["display_name"]
        print(f"\n>>> Processing Sensor: {sensor_display} ({sensor_id}) [Tile={tile_size}x{tile_size}, SNR={snr_disp}]")
        
        sensor_out_dir = os.path.join(output_dir, sensor_id)
        os.makedirs(sensor_out_dir, exist_ok=True)
        
        # Pre-parse HTML table for sensor once
        html_path = SENSOR_CONFIGS[sensor_id]["html_index"]
        with open(html_path, 'r', encoding='utf-8', errors='ignore') as f:
            soup = BeautifulSoup(f, 'html.parser')
            
        sensor_scenario_results = {}
        
        for sc_id in target_scenarios:
            print(f"  -> Running {sc_id} (Tile={tile_size}x{tile_size}, SNR={snr_disp})...")
            res = run_scenario_on_sensor(sc_id, sensor_id, soup=soup, snr=snr, tile_size=tile_size)
            
            # Store in master dictionary
            all_results[(sensor_id, sc_id)] = res
            sensor_scenario_results[sc_id] = res
            
            # Save composite figure
            fig_path = os.path.join(sensor_out_dir, f"{sensor_id}_{sc_id}_composite.png")
            plot_scenario_composite(res, fig_path, dpi=dpi)
            
        # Save cross-scenario summary figure for this sensor
        summary_fig_path = os.path.join(sensor_out_dir, f"{sensor_id}_cross_scenario_summary.png")
        plot_cross_scenario_summary(sensor_id, sensor_scenario_results, summary_fig_path, dpi=dpi)
        
    # --- Generate Quantitative Comparison Tables ---
    print("\n" + "=" * 80)
    print(f"GENERATING COMPARISON TABLES & ACCURACY SUMMARIES [Tile={tile_size}x{tile_size}, SNR={snr_disp}]")
    print("=" * 80)
    
    cross_sensor_df = generate_cross_sensor_table(all_results)
    csv_table_path = os.path.join(output_dir, "cross_sensor_comparison_table.csv")
    md_table_path = os.path.join(output_dir, "cross_sensor_comparison_table.md")
    cross_sensor_df.to_csv(csv_table_path, index=False)
    with open(md_table_path, "w", encoding="utf-8") as f:
        f.write(f"# Cross-Sensor Comparison Table: MaxD vs. PCA (Tile={tile_size}x{tile_size}, SNR={snr_disp})\n\n")
        f.write(cross_sensor_df.to_markdown(index=False))
    print(f"Saved Cross-Sensor Comparison Table to:\n  - {csv_table_path}\n  - {md_table_path}")
    
    accuracy_df = generate_accuracy_summary_table(cross_sensor_df)
    csv_acc_path = os.path.join(output_dir, "detection_accuracy_summary.csv")
    md_acc_path = os.path.join(output_dir, "detection_accuracy_summary.md")
    accuracy_df.to_csv(csv_acc_path, index=False)
    with open(md_acc_path, "w", encoding="utf-8") as f:
        f.write(f"# Automated Detection Accuracy Summary (Tile={tile_size}x{tile_size}, SNR={snr_disp})\n\n")
        f.write(cross_sensor_df.to_markdown(index=False))
    print(f"Saved Cross-Sensor Comparison Table to:\n  - {csv_table_path}\n  - {md_table_path}")
    
    accuracy_df = generate_accuracy_summary_table(cross_sensor_df)
    csv_acc_path = os.path.join(output_dir, "detection_accuracy_summary.csv")
    md_acc_path = os.path.join(output_dir, "detection_accuracy_summary.md")
    accuracy_df.to_csv(csv_acc_path, index=False)
    with open(md_acc_path, "w", encoding="utf-8") as f:
        f.write(f"# Automated Detection Accuracy Summary (SNR={snr_disp})\n\n")
        f.write(accuracy_df.to_markdown(index=False))
    print(f"Saved Detection Accuracy Summary to:\n  - {csv_acc_path}\n  - {md_acc_path}")
    
    # Save Prediction Error vs. SNR figure for this study
    pred_error_fig_path = os.path.join(output_dir, "prediction_error_vs_snr.png")
    plot_prediction_error_vs_snr(cross_sensor_df, pred_error_fig_path, dpi=dpi)
    
    # Print formatted markdown table to console
    print("\n--- Cross-Sensor Comparison Table (First 15 Rows) ---")
    print(cross_sensor_df.head(15).to_markdown(index=False))
    
    print("\n--- Final Automated Detection Accuracy Summary ---")
    print(accuracy_df.to_markdown(index=False))
    
    return all_results, cross_sensor_df, accuracy_df


def main(snr_set=None, tile_size_set=None):
    parser = argparse.ArgumentParser(
        description="Run Spectral Complexity vs. PCA simulation study across sensors, scenarios, SNRs, and tile sizes."
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
        help=f"Base directory to save figures and tables (default: {DEFAULT_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--snrs",
        type=str,
        default=None,
        help="Comma-separated list of SNR values (e.g. '5,10,15,30,50,75,100,200,500,inf')"
    )
    parser.add_argument(
        "--snr",
        type=str,
        default=None,
        help="Single SNR value to evaluate (e.g. '8' or 'inf')"
    )
    parser.add_argument(
        "--tile-sizes",
        type=str,
        default=None,
        help="Comma-separated list of tile sizes (e.g. '3,5,15,35,55,101')"
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=None,
        help="Single tile size to evaluate (e.g. 3 or 101)"
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
    
    def parse_snr_val(val_str):
        val_str = str(val_str).strip()
        if val_str.lower() in ("inf", "np.inf", "infinite"):
            return np.inf
        return float(val_str) if '.' in val_str else int(val_str)
        
    if args.snr is not None:
        snr_list = [parse_snr_val(args.snr)]
    elif args.snrs is not None:
        snr_list = [parse_snr_val(s) for s in args.snrs.split(",")]
    elif snr_set is not None:
        snr_list = snr_set
    else:
        snr_list = [5, 10, 15, 30, 50, 75, 100, 200, 500, np.inf]
        
    if args.tile_size is not None:
        tile_size_list = [args.tile_size]
    elif args.tile_sizes is not None:
        tile_size_list = [int(t.strip()) for t in args.tile_sizes.split(",")]
    elif tile_size_set is not None:
        tile_size_list = tile_size_set
    else:
        tile_size_list = [3, 5, 15, 35, 55, 101]
        
    print("=" * 80)
    print("STARTING MULTI-TILE & MULTI-SNR SIMULATION SUITE")
    print(f"Tile Sizes ({len(tile_size_list)}): {tile_size_list}")
    print(f"SNR Values ({len(snr_list)}): {snr_list}")
    print("=" * 80)
    
    base_output_dir = args.output_dir
    parent_dir = os.path.dirname(base_output_dir) if base_output_dir == DEFAULT_OUTPUT_DIR else base_output_dir
    
    all_grand_comparison_dfs = []
    
    for tile_size in tile_size_list:
        tile_dir_name = f"Spectral Complexity vs PCA {tile_size}x{tile_size}"
        tile_output_dir = os.path.join(parent_dir, tile_dir_name)
        os.makedirs(tile_output_dir, exist_ok=True)
        
        all_snr_comparison_dfs = []
        
        for snr in snr_list:
            snr_dir_name = f"spectral_complexity_vs_pca_SNR-{snr}"
            snr_output_dir = os.path.join(tile_output_dir, snr_dir_name)
            
            all_results, cross_sensor_df, accuracy_df = run_study(
                target_sensors=target_sensors,
                target_scenarios=target_scenarios,
                output_dir=snr_output_dir,
                snr=snr,
                tile_size=tile_size,
                dpi=args.dpi
            )
            
            all_snr_comparison_dfs.append(cross_sensor_df)
            all_grand_comparison_dfs.append(cross_sensor_df)
            
        if len(snr_list) > 1:
            master_df = pd.concat(all_snr_comparison_dfs, ignore_index=True)
            summary_dir = os.path.join(tile_output_dir, "spectral_complexity_vs_pca_cross_SNR_summary")
            os.makedirs(summary_dir, exist_ok=True)
            master_csv = os.path.join(summary_dir, "master_cross_snr_comparison.csv")
            master_md = os.path.join(summary_dir, "master_cross_snr_comparison.md")
            master_df.to_csv(master_csv, index=False)
            with open(master_md, "w", encoding="utf-8") as f:
                f.write(f"# Master Cross-SNR Comparison Table: MaxD vs. PCA ({tile_size}x{tile_size} Tiles)\n\n")
                f.write(master_df.to_markdown(index=False))
            print("\n" + "=" * 80)
            print(f"MASTER CROSS-SNR SUMMARY SAVED FOR TILE SIZE {tile_size}x{tile_size}")
            print(f"  - {master_csv}")
            print(f"  - {master_md}")
            
            # Save Master Prediction Error vs. SNR figure across all SNRs for this tile size
            master_fig_path = os.path.join(summary_dir, "prediction_error_vs_snr.png")
            plot_prediction_error_vs_snr(master_df, master_fig_path, dpi=args.dpi)
            print(f"  - {master_fig_path}")
            print("=" * 80)
            
    if len(tile_size_list) > 1 and len(all_grand_comparison_dfs) > 0:
        grand_df = pd.concat(all_grand_comparison_dfs, ignore_index=True)
        grand_summary_dir = os.path.join(parent_dir, "Spectral Complexity vs PCA Cross-Tile-Size Summary")
        os.makedirs(grand_summary_dir, exist_ok=True)
        grand_csv = os.path.join(grand_summary_dir, "master_cross_tile_size_comparison.csv")
        grand_md = os.path.join(grand_summary_dir, "master_cross_tile_size_comparison.md")
        grand_df.to_csv(grand_csv, index=False)
        with open(grand_md, "w", encoding="utf-8") as f:
            f.write("# Master Cross-Tile-Size Comparison Table: MaxD vs. PCA Across All Tile Sizes & SNRs\n\n")
            f.write(grand_df.to_markdown(index=False))
        print("\n" + "=" * 80)
        print("GRAND MASTER CROSS-TILE-SIZE SUMMARY SAVED")
        print(f"  - {grand_csv}")
        print(f"  - {grand_md}")
        
        grand_fig_path = os.path.join(grand_summary_dir, "prediction_error_vs_snr_all_tiles.png")
        plot_prediction_error_vs_snr(grand_df, grand_fig_path, dpi=args.dpi)
        print(f"  - {grand_fig_path}")
        print("=" * 80)
        
    print("\nAll tasks completed successfully!")


if __name__ == "__main__":
    SNR_set = [5,10,15,30,50,75,100,200,500,np.inf]
    TILE_SIZE_set = [3,5,9,11,15,35,55,101]
    main(snr_set=SNR_set, tile_size_set=TILE_SIZE_set)

