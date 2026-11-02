#!/usr/bin/env python3
"""
evaluate_height_scree.py

Evaluates the orthogonal residual distance (height) decay curves across spatial pixels
to identify the endmember index at which each curve plateaus into the "scree".

Scientific Motivation:
When extracting endmembers using Orthogonal Subspace Projection (MaxD / Gram-Schmidt simplex
expansion), extracting more endmembers than the number of distinct constituent materials
causes the orthogonal projection distances (h_k) to plunge from macroscopic material
separation distances into a flat, noise- and texture-dominated baseline ("the scree").

Methods Implemented:
1. L-Method (Default / Recommended; Salvador & Chan, IEEE TKDE 2004):
   Fits a two-line piecewise regression (Signal Cliff Model vs. Scree Plateau Model)
   in log10-space across candidate split points k* in [2, K-2]. The optimal split
   minimizes the total weighted root-mean-squared error (RMSE). Directly formalizes
   Cattell's (1966) Scree Test.
2. Kneedle Algorithm (Satopaa et al., IEEE ICDCSW 2011):
   Finds the point of maximum curvature / perpendicular distance from the diagonal chord
   connecting the initial material height to the final scree height.

Output:
- Publication-quality standalone heatmap of the identified plateau index k*.
- Comprehensive multi-panel diagnostic figure (True Color RGB, Heatmap, Histogram,
  and exemplar curve fits for low, medium, and high complexity pixels).
- Optional HDF5 dataset creation (saving scree_plateau_index_15x15 into the file).
"""

import os
import sys
import argparse
import time
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# =============================================================================
# DEFAULT CONFIGURATION (Configured for IDE Execution)
# =============================================================================
DEFAULT_H5_PATH = r"C:\satelliteImagery\hyspex\20250918_GE18\dual\GE18_55m_01_Mjolnir_float32_geo_VS620.h5"
DEFAULT_DATASET = "/HDFEOS/GRIDS/HYSPEX/Data Fields/heights_map_ssd_15x15"
FALLBACK_DATASET = "/HDFEOS/GRIDS/HYSPEX/Data Fields/heights_map_15x15"
NOISE_DATASET = "/HDFEOS/GRIDS/HYSPEX/Data Fields/noise_heights_map_ssd_15x15"
RGB_DATASET = "/HDFEOS/GRIDS/HYSPEX/Data Fields/ortho_vis"

DEFAULT_METHOD = "l_method"       # 'l_method' (recommended) or 'kneedle'
LOG_SPACE = True                  # Evaluate curves in log10 space (recommended for physical scale decay)
DEFAULT_TAU_CRIT = 0.010          # Critical RMS reflectance contrast threshold (1.0% separation)
DEFAULT_NUM_BANDS = None          # Auto-detected from surface_reflectance if None
OUTPUT_DIR = None                 # Defaults to directory of H5 file if None
SAVE_TO_H5 = False                # Set True to write the scree map back into the HDF5 file
DPI = 300


# =============================================================================
# CORE MATHEMATICAL METHODS
# =============================================================================
def detect_scree_l_method_vectorized(height_curves: np.ndarray, log_space: bool = True) -> np.ndarray:
    """
    Vectorized L-Method (Salvador & Chan, 2004) piecewise two-line regression.
    
    Fits two straight lines to each height curve:
      Line 1 (Signal Cliff): k in [1, s]
      Line 2 (Scree Plateau): k in [s, K]
    and determines the split index s* that minimizes the total weighted sum of squared errors.

    Args:
        height_curves: Array of shape (M, K) containing orthogonal heights h_1..h_K.
        log_space: If True, evaluates in log10 space (preserves order-of-magnitude physical decay).

    Returns:
        plateau_index: 1D array of length M containing the 1-based index k* (1..K)
                       at which the curve transitions to the scree plateau.
    """
    M, K = height_curves.shape
    if log_space:
        Y = np.log10(np.maximum(height_curves, 1e-12)).astype(np.float32)
    else:
        Y = height_curves.astype(np.float32)

    x = np.arange(1, K + 1, dtype=np.float32)
    best_err = np.full(M, np.inf, dtype=np.float32)
    best_split = np.zeros(M, dtype=np.int32)

    # Test each candidate split index s (0-based in range 1..K-2)
    for s in range(1, K - 1):
        # Segment 1: indices 0..s (inclusive)
        x1 = x[:s + 1]
        X1 = np.vstack([x1, np.ones_like(x1)]).T
        pinv1 = np.linalg.pinv(X1)  # (2, s+1)
        y1 = Y[:, :s + 1]
        beta1 = y1 @ pinv1.T
        pred1 = beta1 @ X1.T
        err1 = np.sum((y1 - pred1) ** 2, axis=1)

        # Segment 2: indices s..K-1 (inclusive)
        x2 = x[s:]
        X2 = np.vstack([x2, np.ones_like(x2)]).T
        pinv2 = np.linalg.pinv(X2)  # (2, K-s)
        y2 = Y[:, s:]
        beta2 = y2 @ pinv2.T
        pred2 = beta2 @ X2.T
        err2 = np.sum((y2 - pred2) ** 2, axis=1)

        # Weighted total error
        total_err = err1 + err2
        mask = total_err < best_err
        best_err[mask] = total_err[mask]
        best_split[mask] = s + 1  # 1-based endmember index

    return best_split


def detect_scree_kneedle_vectorized(height_curves: np.ndarray, log_space: bool = True) -> np.ndarray:
    """
    Vectorized Kneedle algorithm (Satopaa et al., 2011).
    
    Normalizes each curve to a unit square and finds the point k* of maximum
    perpendicular distance from the diagonal chord connecting (1, y_1) to (K, y_K).

    Args:
        height_curves: Array of shape (M, K) containing orthogonal heights h_1..h_K.
        log_space: If True, evaluates in log10 space.

    Returns:
        plateau_index: 1D array of length M containing the 1-based knee index k* (1..K).
    """
    M, K = height_curves.shape
    if log_space:
        Y = np.log10(np.maximum(height_curves, 1e-12))
    else:
        Y = height_curves.copy()

    y_min = Y[:, -1:]
    y_max = Y[:, 0:1]
    denom = np.maximum(y_max - y_min, 1e-9)
    y_norm = (Y - y_min) / denom  # Decreasing: 1 at k=0, 0 at k=K-1

    x_norm = np.linspace(0, 1, K).reshape(1, K)
    # Distance to chord line (1 - x_norm)
    dists = (1.0 - x_norm) - y_norm
    knee_idx = np.argmax(dists, axis=1) + 1  # 1-based index (1..K)
    return knee_idx


def detect_scree_ratio_k2_vectorized(height_curves: np.ndarray, return_plateau_start: bool = True) -> np.ndarray:
    """
    Vectorized Consecutive Drop Ratio evaluated strictly for indices k >= 2.

    Computes drop ratio R_k = h_k / h_{k+1} for k in [2, K-1], explicitly bypassing
    the macroscopic illumination dynamic range drop (h_1 / h_2) between the min-norm
    local origin and max-norm seed endmember.

    Args:
        height_curves: Array of shape (M, K) containing orthogonal heights h_1..h_K.
        return_plateau_start: If True, returns k* = k_drop + 1 (the first endmember on the
                              scree plateau). If False, returns k_drop (the last signal endmember).

    Returns:
        index: 1D array of length M containing the 1-based index (3..K if return_plateau_start=True).
    """
    # height_curves columns are h_1, h_2, ..., h_K
    # ratios are h_k / h_{k+1}: column 0 is h_1/h_2, column 1 is h_2/h_3, ...
    ratios = height_curves[:, :-1] / np.maximum(height_curves[:, 1:], 1e-12)
    # Force evaluation only for k >= 2: ignore column 0 (h_1/h_2)
    ratios_k2 = ratios[:, 1:]  # columns correspond to k = 2, 3, ..., K-1
    best_offset = np.argmax(ratios_k2, axis=1)  # 0 means k=2
    k_drop = best_offset + 2  # 1-based index of last endmember before drop
    if return_plateau_start:
        return k_drop + 1
    return k_drop


def detect_scree_gradient_ratio_vectorized(
    height_curves: np.ndarray,
    tail_steps: int = 4,
    ratio_threshold: float = 2.0,
    log_space: bool = False
) -> np.ndarray:
    """
    Vectorized Gradient Ratio evaluation to identify where the nearly constant
    scree gradient begins.

    Addresses gradual PSF spatial mixing by computing the ratio of the local
    orthogonal height gradient to the asymptotic scree baseline gradient:
        Gamma_k = s_k / g_scree
    where g_scree is the average downward gradient over the last tail_steps endmembers.
    Identifies the first index k >= 2 where Gamma_k <= ratio_threshold (i.e., where
    the local rate of decrease stabilizes to within ratio_threshold of the constant scree gradient).

    Args:
        height_curves: Array of shape (M, K) containing orthogonal heights h_1..h_K.
        tail_steps: Number of final endmember steps to average for baseline scree gradient.
        ratio_threshold: Gradient ratio tolerance (e.g., 2.0 means local slope <= 2x scree slope).
        log_space: If True, computes gradients in log10 space.

    Returns:
        plateau_index: 1D array of length M containing the 1-based index k* (2..K).
    """
    M, K = height_curves.shape
    if log_space:
        Y = np.log10(np.maximum(height_curves, 1e-12)).astype(np.float32)
    else:
        Y = height_curves.astype(np.float32)

    # Downward rate of decrease: s_k = Y_k - Y_{k+1}
    slopes = Y[:, :-1] - Y[:, 1:]  # shape (M, K-1), k=1..K-1
    slopes = np.maximum(slopes, 1e-6)

    # Baseline scree gradient from the tail
    g_scree = np.mean(slopes[:, -tail_steps:], axis=1, keepdims=True)
    g_scree = np.maximum(g_scree, 1e-6)

    # Ratio of local gradient to scree baseline: Gamma_k = s_k / g_scree
    # Force evaluation for k >= 2: ignore column 0 (s_1)
    gamma_k2 = slopes[:, 1:] / g_scree  # columns correspond to k=2..K-1

    # First index where slope is within ratio_threshold of scree baseline
    below_thresh = gamma_k2 <= ratio_threshold
    has_match = np.any(below_thresh, axis=1)
    k_scree = np.where(has_match, np.argmax(below_thresh, axis=1) + 2, K)

    return k_scree


# =============================================================================
# DATASET RESOLUTION & VALIDATION
# =============================================================================
def resolve_dataset(f: h5py.File, requested_path: str, fallback_path: str) -> h5py.Dataset:
    """
    Locates the requested dataset in the HDF5 file, falling back cleanly
    to the secondary path if the primary name differs, or raising a KeyError.
    """
    if requested_path in f:
        print(f"Loading dataset: '{requested_path}'")
        return f[requested_path]
    elif fallback_path in f:
        print(f"Requested dataset '{requested_path}' not found directly.")
        print(f"Falling back to matching dataset: '{fallback_path}'")
        return f[fallback_path]
    else:
        # Check if dataset name without leading slash exists in group
        grp_path = "/HDFEOS/GRIDS/HYSPEX/Data Fields"
        if grp_path in f:
            available = list(f[grp_path].keys())
            raise KeyError(
                f"Neither '{requested_path}' nor '{fallback_path}' were found in HDF5 file.\n"
                f"Available fields under '{grp_path}': {available}"
            )
        raise KeyError(f"Could not locate heights dataset '{requested_path}' in HDF5 file.")


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================
def save_scree_heatmap(
    scree_map: np.ndarray,
    out_path: str,
    method_name: str,
    title_suffix: str = "",
    dpi: int = 300
) -> None:
    """
    Renders and saves a standalone high-resolution heatmap of the scree plateau index map.
    Preserves exact spatial proportions and renders invalid pixels cleanly.
    """
    H, W = scree_map.shape
    valid_data = scree_map[~np.isnan(scree_map)]
    if len(valid_data) == 0:
        raise ValueError("No valid pixels found in scree_map to render heatmap.")

    min_k = int(np.floor(np.percentile(valid_data, 1)))
    max_k = int(np.ceil(np.percentile(valid_data, 99)))
    min_k = max(min_k, 2)
    max_k = max(max_k, min_k + 1)
    num_bins = max_k - min_k + 1

    # Discrete colormap matching integer indices
    base_cmap = plt.colormaps['plasma'].resampled(num_bins)
    base_cmap.set_bad(color='#e0e0e0', alpha=0.5)
    norm = mcolors.Normalize(vmin=min_k - 0.5, vmax=max_k + 0.5)

    # Compute figure aspect ratio matching image dimensions
    aspect_ratio = W / H
    fig_height = 8.5
    fig_width = max(fig_height * aspect_ratio + 1.8, 10.0)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
    im = ax.imshow(scree_map, cmap=base_cmap, norm=norm, interpolation='nearest')

    if method_name == 'l_method':
        method_label = "L-Method Piecewise Two-Line Fit"
    elif method_name in ['ratio_k2', 'drop_ratio']:
        method_label = "Consecutive Drop Ratio (k >= 2)"
    elif method_name in ['gradient_ratio', 'slope_ratio']:
        method_label = "Gradient Ratio to Scree Baseline (k >= 2)"
    else:
        method_label = "Kneedle Curvature"
    ax.set_title(
        f"Orthogonal Subspace Distance Scree Plateau Index ($k^*$) — {method_label}{title_suffix}",
        fontsize=13, fontweight='bold', pad=12
    )
    ax.set_xlabel("Cross-Track Spatial Coordinate (Pixels)", fontsize=10, fontweight='bold')
    ax.set_ylabel("Along-Track Spatial Coordinate (Pixels)", fontsize=10, fontweight='bold')
    ax.tick_params(labelsize=9)

    cbar = plt.colorbar(im, ax=ax, fraction=0.038, pad=0.03, ticks=np.arange(min_k, max_k + 1))
    cbar.set_label(r"Identified Scree Plateau Index ($k^*$)", fontsize=11, fontweight='bold')
    cbar.ax.tick_params(labelsize=10)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight', dpi=dpi)
    plt.close()
    print(f"Saved standalone scree heatmap to: {out_path}")


def save_comprehensive_evaluation_figure(
    scree_map: np.ndarray,
    h_curves: np.ndarray,
    n_curves: np.ndarray,
    valid_mask: np.ndarray,
    rgb_img: np.ndarray,
    out_path: str,
    method_name: str,
    dpi: int = 300
) -> None:
    """
    Generates a publication-grade multi-panel figure containing:
    1. RGB True Color Reference (ortho_vis)
    2. Spatial Scree Plateau Index Heatmap
    3. Global Empirical Distribution Histogram
    4. Exemplar Height Curves with fitted piecewise cliff/scree models
    """
    valid_indices = np.where(valid_mask)
    flat_indices = scree_map[valid_mask].astype(np.int32)
    M, K = h_curves.shape
    x_steps = np.arange(1, K + 1, dtype=np.float64)
    log_H = np.log10(np.maximum(h_curves, 1e-12))

    fig = plt.figure(figsize=(19, 11), dpi=dpi)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.5, 1.0], hspace=0.28, wspace=0.22)

    # -------------------------------------------------------------------------
    # Panel (a): RGB Context
    # -------------------------------------------------------------------------
    ax_rgb = fig.add_subplot(gs[0, 0])
    if rgb_img is not None:
        ax_rgb.imshow(rgb_img)
        ax_rgb.set_title("(a) Flightline True Color (ortho_vis)", fontsize=11, fontweight='bold')
    else:
        ax_rgb.text(0.5, 0.5, "RGB Not Available", ha='center', va='center', fontsize=11)
        ax_rgb.set_title("(a) True Color Reference", fontsize=11, fontweight='bold')
    ax_rgb.set_xlabel("Cross-Track (Pixels)", fontsize=9, fontweight='bold')
    ax_rgb.set_ylabel("Along-Track (Pixels)", fontsize=9, fontweight='bold')

    # -------------------------------------------------------------------------
    # Panel (b): Spatial Scree Plateau Heatmap
    # -------------------------------------------------------------------------
    ax_map = fig.add_subplot(gs[0, 1])
    min_k = 2
    max_k = min(int(np.percentile(flat_indices, 99.5)), 12)
    max_k = max(max_k, min_k + 1)
    num_bins = max_k - min_k + 1

    cmap = plt.colormaps['plasma'].resampled(num_bins)
    cmap.set_bad(color='#dcdcdc', alpha=0.6)
    norm = mcolors.Normalize(vmin=min_k - 0.5, vmax=max_k + 0.5)

    im = ax_map.imshow(scree_map, cmap=cmap, norm=norm, interpolation='nearest')
    if method_name == 'l_method':
        method_title = "L-Method Two-Line Fit"
    elif method_name in ['ratio_k2', 'drop_ratio']:
        method_title = r"Drop Ratio ($k \geq 2$)"
    elif method_name in ['gradient_ratio', 'slope_ratio']:
        method_title = r"Gradient Ratio ($k \geq 2$)"
    else:
        method_title = "Kneedle"
    ax_map.set_title(f"(b) Scree Plateau Index $k^*$ ({method_title})", fontsize=11, fontweight='bold')
    ax_map.set_xlabel("Cross-Track (Pixels)", fontsize=9, fontweight='bold')
    ax_map.set_ylabel("Along-Track (Pixels)", fontsize=9, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax_map, fraction=0.045, pad=0.03, ticks=np.arange(min_k, max_k + 1))
    cbar.set_label(r"Plateau Index $k^*$", fontsize=10, fontweight='bold')
    cbar.ax.tick_params(labelsize=9)

    # -------------------------------------------------------------------------
    # Panel (c): Empirical Frequency Distribution
    # -------------------------------------------------------------------------
    ax_hist = fig.add_subplot(gs[0, 2])
    unique_k, counts = np.unique(flat_indices, return_counts=True)
    pcts = (counts / len(flat_indices)) * 100.0
    bars = ax_hist.bar(unique_k, pcts, color='#3b528b', edgecolor='black', alpha=0.88, width=0.65)
    ax_hist.set_title(r"(c) Scree Plateau Index Distribution", fontsize=11, fontweight='bold')
    ax_hist.set_xlabel(r"Identified Plateau Index ($k^*$)", fontsize=10, fontweight='bold')
    ax_hist.set_ylabel("Scene Percentage (%)", fontsize=10, fontweight='bold')
    ax_hist.set_xticks(np.arange(min_k, max_k + 1))
    ax_hist.set_xlim(min_k - 0.6, max_k + 0.6)
    ax_hist.grid(axis='y', linestyle=':', alpha=0.5)
    ax_hist.set_ylim(0, max(pcts) * 1.18)
    for b, p in zip(bars, pcts):
        if p > 0.4 and b.get_x() + b.get_width() / 2 <= max_k:
            ax_hist.text(
                b.get_x() + b.get_width() / 2, p + 0.8,
                f"{p:.1f}%", ha='center', fontsize=8.5, fontweight='bold'
            )

    # -------------------------------------------------------------------------
    # Panels (d, e, f): Exemplar Decay Curves with Fitted Cliff vs Scree Models
    # -------------------------------------------------------------------------
    # Select 3 representative complexity tiers
    avail_indices = np.unique(flat_indices)
    target_tiers = [2, 4, 6] if 2 in avail_indices else [3, 5, 7]
    selected_tiers = [t if t in avail_indices else avail_indices[min(i, len(avail_indices)-1)]
                      for i, t in enumerate(target_tiers)]

    tier_labels = ["Low Complexity (1-2 Mat)", "Moderate Complexity (3-4 Mat)", "High Complexity (5+ Mat)"]

    for i, (k_target, label) in enumerate(zip(selected_tiers, tier_labels)):
        ax_ex = fig.add_subplot(gs[1, i])
        match_pts = np.where(flat_indices == k_target)[0]
        p_idx = match_pts[len(match_pts) // 2]  # Median exemplar

        curve = h_curves[p_idx]
        k_star = flat_indices[p_idx]
        s = k_star - 1

        # Plot raw orthogonal heights
        ax_ex.plot(
            x_steps, curve, 'o-', color='#1f77b4', linewidth=2.0, markersize=5.5,
            label=r'Empirical Height $h_k$'
        )

        # Plot noise threshold if available
        if n_curves is not None:
            ax_ex.plot(
                x_steps, n_curves[p_idx], ':', color='#7f7f7f', linewidth=1.6,
                label='SSD Noise Threshold'
            )

        # Plot two-line fit models
        log_curve = log_H[p_idx]
        p1 = np.polyfit(x_steps[:s + 1], log_curve[:s + 1], 1)
        fit1 = 10 ** np.polyval(p1, x_steps[:s + 1])
        p2 = np.polyfit(x_steps[s:], log_curve[s:], 1)
        fit2 = 10 ** np.polyval(p2, x_steps[s:])

        ax_ex.plot(x_steps[:s + 1], fit1, '--', color='#d62728', linewidth=2.0, label='Signal Cliff Model')
        ax_ex.plot(x_steps[s:], fit2, '--', color='#2ca02c', linewidth=2.0, label='Scree Plateau Model')
        ax_ex.axvline(
            k_star, color='purple', linestyle='-.', linewidth=1.8,
            label=f'Plateau Index ($k^*={k_star}$)'
        )

        ax_ex.set_yscale('log')
        panel_letter = chr(ord('d') + i)
        ax_ex.set_title(f"({panel_letter}) {label} ($k^*={k_star}$)", fontsize=10.5, fontweight='bold')
        ax_ex.set_xlabel('Endmember Step ($k$)', fontsize=9.5, fontweight='bold')
        ax_ex.set_ylabel(r'Orthogonal Distance $h_k$', fontsize=9.5, fontweight='bold')
        ax_ex.set_xticks(np.arange(1, K + 1, 2))
        ax_ex.grid(True, which='both', linestyle=':', alpha=0.4)
        ax_ex.legend(fontsize=7.5, loc='upper right')

    plt.savefig(out_path, bbox_inches='tight', dpi=dpi)
    plt.close()
    print(f"Saved comprehensive evaluation figure to: {out_path}")


# =============================================================================
# MAIN EVALUATION PIPELINE
# =============================================================================
def evaluate_height_scree(
    h5_path: str = DEFAULT_H5_PATH,
    dataset_name: str = DEFAULT_DATASET,
    method: str = DEFAULT_METHOD,
    log_space: bool = LOG_SPACE,
    tau_crit: float = DEFAULT_TAU_CRIT,
    num_bands: int = DEFAULT_NUM_BANDS,
    out_dir: str = OUTPUT_DIR,
    save_to_h5: bool = SAVE_TO_H5,
    dpi: int = DPI
) -> dict:
    """
    Main evaluation pipeline:
    1. Loads height curves from the specified HDF5 dataset.
    2. Identifies band count B and critical height threshold h_crit = tau_crit * sqrt(B).
    3. Evaluates 1-material decision rule (h_2 < h_crit) separating homogeneous pixels.
    4. Evaluates multi-material pixels using the specified method (L-Method, Kneedle, etc.).
    5. Generates and saves publication-quality heatmaps and evaluation figures.
    6. Optionally writes the resulting index map into the HDF5 file.
    """
    if not os.path.isfile(h5_path):
        raise FileNotFoundError(f"HDF5 file does not exist: {h5_path}")

    if out_dir is None:
        out_dir = os.path.dirname(h5_path)
    os.makedirs(out_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(h5_path))[0]
    print(f"\n===================================================================")
    print(f"Orthogonal Subspace Distance Scree Plateau Evaluation")
    print(f"File:       {h5_path}")
    print(f"Dataset:    {dataset_name}")
    print(f"Method:     {method.upper()} (Log Space = {log_space})")
    print(f"Decision Rule: tau_crit = {tau_crit} (1% reflectance contrast per band)")
    print(f"===================================================================")

    t0 = time.time()
    with h5py.File(h5_path, 'r') as f:
        ds = resolve_dataset(f, dataset_name, FALLBACK_DATASET)
        raw_shape = ds.shape
        raw_dtype = ds.dtype
        print(f"Dataset shape: {raw_shape}, dtype: {raw_dtype}")

        # Squeeze leading singleton dimensions if present (e.g. shape (1, H, W, K) -> (H, W, K))
        h_data = np.squeeze(ds[:])
        if h_data.ndim != 3:
            raise ValueError(f"Expected 3D heights array (H, W, K), got shape {h_data.shape}")

        # Auto-detect band count B from surface_reflectance if available
        detected_bands = None
        sr_path = "/HDFEOS/GRIDS/HYSPEX/Data Fields/surface_reflectance"
        if sr_path in f:
            sr_shape = f[sr_path].shape
            if len(sr_shape) == 4:
                detected_bands = sr_shape[1]
            elif len(sr_shape) == 3:
                detected_bands = sr_shape[0] if sr_shape[0] < min(sr_shape[1], sr_shape[2]) else sr_shape[2]
            print(f"Auto-detected band count from '{sr_path}': B = {detected_bands}")

        # Load optional noise heights if available
        n_data = None
        if NOISE_DATASET in f:
            print(f"Found noise heights dataset: '{NOISE_DATASET}'")
            n_data = np.squeeze(f[NOISE_DATASET][:])

        # Load RGB image for visual reference if available
        rgb_img = None
        if RGB_DATASET in f:
            rgb_raw = np.squeeze(f[RGB_DATASET][:])
            if rgb_raw.ndim == 3 and rgb_raw.shape[0] in [3, 4]:
                rgb_img = np.transpose(rgb_raw[:3], (1, 2, 0))
            elif rgb_raw.ndim == 3 and rgb_raw.shape[2] in [3, 4]:
                rgb_img = rgb_raw[:, :, :3]

    H, W, num_em = h_data.shape
    print(f"Image grid dimensions: {H} rows x {W} cols, {num_em} endmember heights per pixel")

    # Determine band count B and critical height h_crit
    B = num_bands if num_bands is not None else (detected_bands if detected_bands is not None else 475)
    sqrt_B = np.sqrt(B)

    # Strict Data Integrity: Valid pixels have non-NaN and positive orthogonal heights
    # Index 0 is origin (0.0), so test index 1
    valid_mask = ~np.isnan(h_data[:, :, 1]) & (h_data[:, :, 1] > 0)
    num_valid = int(np.count_nonzero(valid_mask))
    total_pixels = H * W
    print(f"Valid pixels: {num_valid:,} / {total_pixels:,} ({num_valid / total_pixels * 100:.1f}%)")

    if num_valid == 0:
        raise ValueError("No valid pixels with non-zero heights found in the dataset.")

    # Slice out height curves for valid pixels (excluding origin at index 0)
    # Curves correspond to k = 1..num_em - 1
    h_curves = h_data[valid_mask][:, 1:]
    n_curves = n_data[valid_mask][:, 1:] if n_data is not None else None
    M, K = h_curves.shape

    # -------------------------------------------------------------------------
    # Decision Rule: 1-Material vs. Multi-Material Separation
    # If the 2nd endmember is within tau_crit (1.0%) RMS reflectance of the 1st
    # endmember (h_2 / sqrt(B) < tau_crit), the two are not worth separating into
    # distinct materials. The pixel is definitively 1-material -> assign k* = 2.
    # -------------------------------------------------------------------------
    flat_scree_indices = np.full(M, 2, dtype=np.int32)

    if tau_crit is not None and tau_crit > 0:
        h_crit = tau_crit * sqrt_B
        h2 = h_curves[:, 1]  # Column 1 corresponds to h_2 (index k=2)
        is_1mat = h2 < h_crit
        multi_mask = ~is_1mat
        num_1mat = int(np.count_nonzero(is_1mat))
        num_multi = int(np.count_nonzero(multi_mask))
        print(f"\n--- Physical Decision Rule (tau_crit = {tau_crit:.3f}, B = {B}, h_crit = {h_crit:.4f}) ---")
        print(f"  1-Material Pixels (h_2 < h_crit):      {num_1mat:,} / {M:,} ({num_1mat / M * 100:.2f}%) -> Assigned k* = 2 (Scree starts at k=2)")
        print(f"  Multi-Material Pixels (h_2 >= h_crit): {num_multi:,} / {M:,} ({num_multi / M * 100:.2f}%) -> Evaluating via {method.upper()}")
    else:
        multi_mask = np.ones(M, dtype=bool)
        num_multi = M
        print(f"\nDecision rule disabled (tau_crit = {tau_crit}). Evaluating all {M:,} pixels via {method.upper()}...")

    t_eval_start = time.time()

    if num_multi > 0:
        h_multi = h_curves[multi_mask]
        if method == "l_method":
            flat_scree_indices[multi_mask] = detect_scree_l_method_vectorized(h_multi, log_space=log_space)
        elif method == "kneedle":
            flat_scree_indices[multi_mask] = detect_scree_kneedle_vectorized(h_multi, log_space=log_space)
        elif method in ["ratio_k2", "drop_ratio"]:
            flat_scree_indices[multi_mask] = detect_scree_ratio_k2_vectorized(h_multi, return_plateau_start=True)
        elif method in ["gradient_ratio", "slope_ratio"]:
            flat_scree_indices[multi_mask] = detect_scree_gradient_ratio_vectorized(h_multi, ratio_threshold=2.0, log_space=log_space)
        else:
            raise ValueError(f"Unknown method '{method}'. Supported methods: 'l_method', 'kneedle', 'ratio_k2', 'gradient_ratio'")

    t_eval = time.time() - t_eval_start
    print(f"Completed scree detection in {t_eval:.2f} seconds ({M / t_eval:,.0f} pixels/sec)")

    # Build 2D spatial map preserving exact NaNs outside valid mask (matching raw float32 dtype)
    scree_map = np.full((H, W), np.nan, dtype=np.float32)
    scree_map[valid_mask] = flat_scree_indices.astype(np.float32)

    # Statistical summary
    print("\n--- Scree Plateau Index Statistics ---")
    print(f"  Min k*:    {np.min(flat_scree_indices)}")
    print(f"  Median k*: {np.median(flat_scree_indices)}")
    print(f"  Mean k*:   {np.mean(flat_scree_indices):.2f}")
    print(f"  Max k*:    {np.max(flat_scree_indices)}")
    print(f"  Std Dev:   {np.std(flat_scree_indices):.2f}")

    unique_vals, counts = np.unique(flat_scree_indices, return_counts=True)
    print("\nFrequency Distribution:")
    for v, c in zip(unique_vals, counts):
        pct = c / num_valid * 100.0
        print(f"  k* = {v:2d}: {c:7,d} pixels ({pct:5.2f}%)")

    # Generate Visuals
    heatmap_path = os.path.join(out_dir, f"{base_name}_scree_plateau_heatmap.png")
    eval_fig_path = os.path.join(out_dir, f"{base_name}_scree_evaluation_report.png")

    print(f"\nGenerating publication visuals (DPI = {dpi})...")
    save_scree_heatmap(
        scree_map=scree_map,
        out_path=heatmap_path,
        method_name=method,
        dpi=dpi
    )
    save_comprehensive_evaluation_figure(
        scree_map=scree_map,
        h_curves=h_curves,
        n_curves=n_curves,
        valid_mask=valid_mask,
        rgb_img=rgb_img,
        out_path=eval_fig_path,
        method_name=method,
        dpi=dpi
    )

    # Optionally write dataset into HDF5 file
    if save_to_h5:
        target_field = f"scree_plateau_index_{raw_shape[-1]}x{raw_shape[-1]}" if "x" in dataset_name else "scree_plateau_index"
        full_field_path = f"/HDFEOS/GRIDS/HYSPEX/Data Fields/{target_field}"
        print(f"\nWriting dataset '{full_field_path}' into {h5_path}...")
        with h5py.File(h5_path, 'r+') as f:
            grp = f["/HDFEOS/GRIDS/HYSPEX/Data Fields"]
            if target_field in grp:
                print(f"Updating existing dataset '{target_field}'...")
                grp[target_field][0] = scree_map
            else:
                # Store matching original dimensions (1, H, W)
                out_to_write = np.expand_dims(scree_map, axis=0)
                ds_new = grp.create_dataset(target_field, data=out_to_write, dtype=np.float32)
                ds_new.attrs["Description"] = f"Endmember index k* where orthogonal height curve reaches scree ({method})"
                ds_new.attrs["Method"] = method
                ds_new.attrs["Log_Space"] = log_space
        print("HDF5 write complete.")

    total_time = time.time() - t0
    print(f"\nEvaluation pipeline completed successfully in {total_time:.2f} seconds.")
    print(f"Standalone Heatmap:        {heatmap_path}")
    print(f"Comprehensive Evaluation:  {eval_fig_path}")

    return {
        "scree_map": scree_map,
        "valid_mask": valid_mask,
        "heatmap_path": heatmap_path,
        "eval_fig_path": eval_fig_path,
        "distribution": dict(zip(unique_vals.tolist(), counts.tolist()))
    }


# =============================================================================
# CLI / SCRIPT RUNNER
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate HySpex orthogonal height curves to identify the scree plateau index map."
    )
    parser.add_argument(
        "--file", "-f", default=DEFAULT_H5_PATH,
        help=f"Path to HySpex HDF5 file (default: {DEFAULT_H5_PATH})"
    )
    parser.add_argument(
        "--dataset", "-d", default=DEFAULT_DATASET,
        help=f"Heights dataset path (default: {DEFAULT_DATASET})"
    )
    parser.add_argument(
        "--method", "-m", choices=["l_method", "kneedle", "ratio_k2", "gradient_ratio"], default=DEFAULT_METHOD,
        help="Detection method: 'l_method' (default), 'kneedle', 'ratio_k2' (drop ratio k>=2), or 'gradient_ratio' (ratio to scree slope)"
    )
    parser.add_argument(
        "--outdir", "-o", default=OUTPUT_DIR,
        help="Output directory for saved figures (default: directory of H5 file)"
    )
    parser.add_argument(
        "--save-h5", action="store_true", default=SAVE_TO_H5,
        help="Write computed scree plateau index map into the HDF5 file"
    )
    parser.add_argument(
        "--tau-crit", type=float, default=DEFAULT_TAU_CRIT,
        help=f"Critical RMS reflectance contrast threshold for 1-material separation (default: {DEFAULT_TAU_CRIT}; 0 to disable)"
    )
    parser.add_argument(
        "--num-bands", "-B", type=int, default=DEFAULT_NUM_BANDS,
        help="Number of spectral channels B (default: auto-detected from surface_reflectance, or 475)"
    )
    parser.add_argument(
        "--dpi", type=int, default=DPI,
        help=f"Resolution for output figures (default: {DPI})"
    )

    args = parser.parse_args()

    evaluate_height_scree(
        h5_path=args.file,
        dataset_name=args.dataset,
        method=args.method,
        log_space=LOG_SPACE,
        tau_crit=args.tau_crit,
        num_bands=args.num_bands,
        out_dir=args.outdir,
        save_to_h5=args.save_h5,
        dpi=args.dpi
    )
