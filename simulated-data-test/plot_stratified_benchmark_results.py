"""
plot_stratified_benchmark_results.py
====================================
Generates publication-quality figures (300 DPI) from the Land-Cover Stratified Patches
benchmark results stored under /STRATIFIED_CHIP_BENCHMARK/ in HLST_Rochesterv2.h5.

Figures Generated:
  1. Figure 16: Subpixel Change Detection Sensitivity Curves (P_CD vs. Mixture Fraction f)
     Stratified across Agricultural, Forest, Wetland, and Developed biomes.
     Evaluates SC-CCM (1x1 Center Pixel) vs. USGS Landsat PyCCD.
  2. Figure 17: Material Category Sensitivity Curves (Artificial, Vegetation, Charred Soil, Turbid Water).
  3. Figure: Individual Target Material Sensitivity Curves (SC-CCM Cluster & Grand Mean Pixel).
  4. Figure 1c: Executive Composite Summary (2-panel synthesis figure for rapid manuscript review).
  5. Figure 2: Operational ROC Space Comparison (TPR vs. FPR on un-injected controls).
  6. Figure 3: Stratified Detection Latency Distributions (Paired boxplots: SC-CCM vs. USGS CCD).
  7. Figure 4: Material Contrast Sensitivity Matrix (f_min Heatmap at P_CD >= 75%).
  8. Figure: Strata Material Classes 2x2 Sensitivity (P_CD,tgt vs. f with McNemar paired significance).

Configured for direct execution in the IDE.
"""

import os
import sys
import shutil
import datetime
import argparse
from pathlib import Path
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors

# Patch matplotlib.style.core deprecation for scienceplots compatibility in matplotlib >= 3.9
import matplotlib.style as mstyle
if not hasattr(mstyle, 'core'):
    mstyle.core = mstyle

import scienceplots
plt.style.use(['science', 'ieee'])

# Ensure file locking is disabled on Windows
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
CONFIDENCE_INTERVAL = 0.99
MCNEMAR_ALPHA = 0.01

# Dynamically ensure generate_stratified_patch_benchmark can be imported
_script_dir = Path(__file__).resolve().parent
_candidates = [
    _script_dir,
    _script_dir.parent,
    _script_dir.parent / "spectral-complexity-public",
    _script_dir.parent.parent / "spectral-complexity-public",
    Path("F:/Resilio/IMGS 890 Research/spectral-complexity-public"),
]
for _p in _candidates:
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    import generate_stratified_patch_benchmark as gsb
except Exception:
    gsb = None

# ==============================================================================
# CONFIGURATION & CONSTANTS
# ==============================================================================
H5_PATH = "C:/satelliteImagery/HLST30/HLST_Rochesterv2.h5"
MEDIA_DIR = "C:/satelliteImagery/HLST30/SC-CCM"

# Select comparison methods: "volume_map_3d_zscore", "sliding_volume_z_score", or "pyccd"
METHOD_1 = "sliding_volume_z_score"
METHOD_2 = "pyccd"


def normalize_method_name(name):
    """
    Normalizes user-specified method strings to standard canonical identifiers:
    'volume_map_3d_zscore', 'sliding_volume_z_score', or 'pyccd'.
    """
    n = str(name).strip().lower().replace("-", "_")
    if any(k in n for k in ["3d", "volume_map_3d"]):
        return "volume_map_3d_zscore"
    elif any(k in n for k in ["maxvol", "max_vol", "sliding", "gram", "base"]):
        return "sliding_volume_z_score"
    elif any(k in n for k in ["ccd", "pyccd", "usgs"]):
        return "pyccd"
    return name


def get_method_descriptor(method_key, data_base=None, data_3d=None):
    """
    Returns metadata, styling, and data table references for the chosen method.
    """
    key = normalize_method_name(method_key)
    if key == "volume_map_3d_zscore":
        if data_3d is None:
            raise ValueError("3D volume benchmark data (/STRATIFIED_CHIP_BENCHMARK_3D) is not available in container.")
        base_fps = data_3d.get("baseline_fps", {})
        tot_pix = base_fps.get("TOTAL_PIXELS", 1)
        fp_cnt = base_fps.get("SC_CCM", 0)
        return {
            "key": "volume_map_3d_zscore",
            "full_name": "SC-CCM 3D",
            "short_name": "SC-CCM 3D",
            "label": "SC-CCM 3D",
            "tag": "SC-CCM 3D",
            "table": data_3d["sc_table"],
            "cluster_table": data_3d.get("sc_table_cluster", data_3d["sc_table"]),
            "segments": data_3d.get("sc_segments"),
            "color": "#0072B2",      # Okabe-Ito Blue
            "marker": "o",
            "linestyle": "-",
            "fpr_key": "SC_CCM",
            "fp_count": fp_cnt,
            "total_pixels": tot_pix,
            "fpr": fp_cnt / max(tot_pix, 1),
        }
    elif key == "sliding_volume_z_score":
        if data_base is None:
            raise ValueError("Baseline benchmark data (/STRATIFIED_CHIP_BENCHMARK) is not available in container.")
        base_fps = data_base.get("baseline_fps", {})
        tot_pix = base_fps.get("TOTAL_PIXELS", 1)
        fp_cnt = base_fps.get("SC_CCM", 0)
        return {
            "key": "sliding_volume_z_score",
            "full_name": "SC-CCM maxVol",
            "short_name": "SC-CCM maxVol",
            "label": "SC-CCM maxVol",
            "tag": "SC-CCM maxVol",
            "table": data_base["sc_table"],
            "cluster_table": data_base.get("sc_table_cluster", data_base["sc_table"]),
            "segments": data_base.get("sc_segments"),
            "color": "#E69F00",      # Okabe-Ito Amber / Gold
            "marker": "^",
            "linestyle": "--",
            "fpr_key": "SC_CCM",
            "fp_count": fp_cnt,
            "total_pixels": tot_pix,
            "fpr": fp_cnt / max(tot_pix, 1),
        }
    elif key == "pyccd":
        ref_data = data_3d if data_3d is not None else data_base
        if ref_data is None:
            raise ValueError("Benchmark data with PyCCD results is not available.")
        base_fps = ref_data.get("baseline_fps", {})
        tot_pix = base_fps.get("TOTAL_PIXELS", 1)
        fp_cnt = base_fps.get("PYCCD", 0)
        return {
            "key": "pyccd",
            "full_name": "USGS Landsat PyCCD",
            "short_name": "USGS PyCCD",
            "label": "USGS PyCCD",
            "tag": "USGS PyCCD",
            "table": ref_data["py_table"],
            "cluster_table": ref_data["py_table"],
            "segments": ref_data.get("py_segments"),
            "color": "#D55E00",      # Okabe-Ito Vermillion
            "marker": "s",
            "linestyle": ":",
            "fpr_key": "PYCCD",
            "fp_count": fp_cnt,
            "total_pixels": tot_pix,
            "fpr": fp_cnt / max(tot_pix, 1),
        }
    else:
        raise ValueError(f"Unknown comparison method: '{method_key}'. Must be 'volume_map_3d_zscore', 'sliding_volume_z_score', or 'pyccd'.")


# Aesthetic color palette for IEEE transactions publications (Okabe-Ito colorblind-safe)
COLOR_SCCCD = "#0072B2"      # Okabe-Ito Blue (SC-CCM virtual constellation)
COLOR_PYCCD = "#D55E00"      # Okabe-Ito Vermillion (USGS Landsat PyCCD)
COLOR_NEUTRAL = "#4d4d4d"    # Charcoal Gray
COLOR_GRID = "#e0e0e0"       # Light Grid

STRATA_PALETTE = {
    "CROPLAND": "#E69F00",    # Okabe-Ito Orange (Cultivated)
    "TREE COVER": "#009E73",  # Okabe-Ito Bluish Green (Forest)
    "WETLAND": "#0072B2",     # Okabe-Ito Blue (Wetland)
    "DEVELOPED": "#CC79A7",   # Okabe-Ito Reddish Purple (Developed Land)
}

# Annual NLCD Level-1 land-cover classification mapping
STRATA_DISPLAY = {
    "CROPLAND": "Cultivated",
    "TREE COVER": "Forest",
    "WETLAND": "Wetland",
    "DEVELOPED": "Developed Land",
}

CATEGORY_PALETTE = {
    "Artificial":   "#d62728",  # Crimson / Red
    "Vegetation":   "#2ca02c",  # Forest Green
    "Charred Soil": "black",  # Burnt Orange
    "Turbid Water": "#1f77b4",  # Blue
    "Other":        "#7f7f7f",  # Medium Gray
}

def compute_binomial_ci(k, n, confidence=CONFIDENCE_INTERVAL, method="wilson"):
    """
    Computes confidence interval for binomial proportion k/n in percentage [0, 100].

    Parameters
    ----------
    k : int
        Number of successes (detected breaks).
    n : int
        Total number of trials.
    confidence : float, default=0.95
        Confidence level (e.g. 0.95 for 95% two-sided CI).
    method : str, default='wilson'
        Method to compute CI:
        - 'wilson': Wilson score interval. Strictly bounded within [0, 100]%, avoiding
                    degenerate zero width at boundary proportions (k=0 or k=n).
        - 'se': 1 standard error (+/- 1 SE).
        - 'normal': Wald normal approximation (+/- z * SE).

    Returns
    -------
    p_pct : float
        Empirical percentage [0, 100].
    ci_low : float
        Lower confidence limit [0, 100].
    ci_high : float
        Upper confidence limit [0, 100].
    """
    if n <= 0:
        return 0.0, 0.0, 0.0
    p = float(k) / float(n)
    if method == "se":
        se = float(np.sqrt(max(0.0, p * (1.0 - p)) / n))
        ci_low = max(0.0, (p - se) * 100.0)
        ci_high = min(100.0, (p + se) * 100.0)
        return p * 100.0, ci_low, ci_high

    try:
        from scipy.stats import norm
        z = float(norm.ppf(1.0 - (1.0 - confidence) / 2.0))
    except Exception:
        z = 1.959963984540054

    if method == "normal":
        se = float(np.sqrt(max(0.0, p * (1.0 - p)) / n))
        ci_low = max(0.0, (p - z * se) * 100.0)
        ci_high = min(100.0, (p + z * se) * 100.0)
        return p * 100.0, ci_low, ci_high
    else:  # 'wilson'
        denom = 1.0 + (z ** 2) / n
        center = (p + (z ** 2) / (2.0 * n)) / denom
        margin = (z * np.sqrt(max(0.0, (p * (1.0 - p)) / n + (z ** 2) / (4.0 * (n ** 2))))) / denom
        ci_low = max(0.0, (center - margin) * 100.0)
        ci_high = min(100.0, (center + margin) * 100.0)
        return p * 100.0, ci_low, ci_high

def categorize_material_role(m_id, m_name, role=""):
    """
    Categorizes a target material into one of 4 primary thematic groups:
    Artificial, Vegetation, Charred Soil, Turbid Water (or Other).
    """
    r_low = (role or "").lower()
    id_low = m_id.lower()
    name_low = m_name.lower()

    if any(k in r_low for k in ["synthetic", "artificial", "manmade", "construction", "debris"]) or \
       any(k in id_low or k in name_low for k in ["plastic", "concrete", "plywood", "shingle", "metal", "aluminum", "fabric", "nylon", "asphalt"]):
        return "Artificial"
    elif any(k in r_low for k in ["vegetation", "foliage", "canopy", "crop"]) or \
         any(k in id_low or k in name_low for k in ["leaf", "leaves", "willow", "grass", "pine", "shrub", "aspen", "tree", "crop"]):
        return "Vegetation"
    elif any(k in r_low for k in ["burn", "disturbance", "pyrogenic", "soil"]) or \
         any(k in id_low or k in name_low for k in ["burn", "fire", "char", "soil"]):
        return "Charred Soil"
    elif any(k in r_low for k in ["water", "flood", "inundation", "liquid"]) or \
         any(k in id_low or k in name_low for k in ["water", "flood", "montmor"]):
        return "Turbid Water"
    return "Other"


def extract_material_groups(sc_table=None):
    """
    Dynamically builds MATERIAL_GROUPS from generate_stratified_patch_benchmark.TARGET_MATERIALS
    if available, otherwise dynamically discovers them from the HDF5 results table.
    Ensures full flexibility to any number of implanted materials.
    """
    cat_order = ["Artificial", "Vegetation", "Charred Soil", "Turbid Water", "Other"]
    cats = {c: [] for c in cat_order}

    if gsb is not None and hasattr(gsb, "TARGET_MATERIALS"):
        for item in gsb.TARGET_MATERIALS:
            m_id = item["id"]
            m_name = item["name"]
            role = item.get("role", "")
            cat = categorize_material_role(m_id, m_name, role)
            cats[cat].append((m_id, m_name))
    elif sc_table is not None:
        seen = set()
        for row in sc_table:
            m_id = row["material_id"].decode("utf-8") if isinstance(row["material_id"], bytes) else str(row["material_id"])
            m_name = row["material_name"].decode("utf-8") if isinstance(row["material_name"], bytes) else str(row["material_name"])
            pair = (m_id, m_name)
            if pair not in seen:
                seen.add(pair)
                cat = categorize_material_role(m_id, m_name, "")
                cats[cat].append(pair)

    groups = []
    for c in cat_order:
        if cats[c]:
            groups.append({"group": c, "materials": cats[c]})
    return groups


# Dynamic material grouping (defaults to dynamically loaded list or empty until data loaded)
MATERIAL_GROUPS = extract_material_groups()


# ==============================================================================
# DATA LOADER
# ==============================================================================
def load_benchmark_data(h5_path, group_name="/STRATIFIED_CHIP_BENCHMARK"):
    """
    Loads benchmark metadata, results tables, and patch baseline statistics from HDF5.
    Supports either /STRATIFIED_CHIP_BENCHMARK (baseline) or /STRATIFIED_CHIP_BENCHMARK_3D (3D volume).
    """
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"Master HDF5 file not found: {h5_path}")

    with h5py.File(h5_path, "r", locking=False) as h5:
        if group_name not in h5:
            raise KeyError(f"Group '{group_name}' not found in master container '{h5_path}'.")

        bench_grp = h5[group_name]
        meta_grp = bench_grp["METADATA"]
        res_grp = bench_grp["RESULTS"]
        target_metric = bench_grp.attrs.get("target_metric", "sliding_volume_z_score" if group_name == "/STRATIFIED_CHIP_BENCHMARK" else "volume_map_3d_zscore")

        # 1. Parse Metadata
        patch_ids = [p.decode() if isinstance(p, bytes) else str(p) for p in meta_grp.attrs["patch_ids"]]
        patch_names = [p.decode() if isinstance(p, bytes) else str(p) for p in meta_grp.attrs["patch_names"]]
        fractions = np.array(meta_grp.attrs["mixture_fractions"], dtype=np.float32)
        patch_bounds = np.array(meta_grp.attrs["patch_bounds"], dtype=np.int32)
        injected_offsets = np.array(meta_grp.attrs["injected_offsets"], dtype=np.int32)
        onset_date_str = str(meta_grp.attrs["injection_onset_date"])

        # 2. Parse Structured Results Tables
        sc_table = res_grp["scccd_break_table"][:]
        py_table = res_grp["pyccd_break_table"][:]
        sc_segments = res_grp["scccd_break_segments"][:] if "scccd_break_segments" in res_grp else None
        py_segments = res_grp["pyccd_break_segments"][:] if "pyccd_break_segments" in res_grp else None

        # 3. Compute 3x3 Spatial Cluster Detection Table for SC-CCM
        counts = res_grp["scccd_change_count"][:] if "scccd_change_count" in res_grp else None
        dates = res_grp["scccd_change_date"][:] if "scccd_change_date" in res_grp else None

        sc_table_cluster = sc_table.copy()
        if counts is not None and dates is not None:
            pid_to_idx = {pid.encode(): i for i, pid in enumerate(patch_ids)}
            from datetime import datetime, timezone
            try:
                onset_dt = datetime.strptime(str(onset_date_str), "%Y-%m-%d").replace(tzinfo=timezone.utc)
                onset_epoch = onset_dt.timestamp()
            except Exception:
                onset_epoch = 0.0

            for i, row in enumerate(sc_table):
                cond_idx = row["condition_idx"]
                p_idx = pid_to_idx[row["patch_id"]]
                p_num = row["pixel_num"] - 1
                r0, c0 = injected_offsets[p_num]
                cnt_w = counts[cond_idx, p_idx, r0-1:r0+2, c0-1:c0+2]
                dt_w = dates[cond_idx, p_idx, r0-1:r0+2, c0-1:c0+2]

                valid_breaks = (cnt_w > 0) & (~np.isnan(dt_w)) & (dt_w >= onset_epoch)
                is_detected = np.any(valid_breaks)
                sc_table_cluster["break_detected"][i] = is_detected
                if is_detected:
                    earliest_dt = float(np.min(dt_w[valid_breaks]))
                    sc_table_cluster["break_timestamp"][i] = earliest_dt
                    sc_table_cluster["latency_days"][i] = float((earliest_dt - onset_epoch) / 86400.0)

        # 4. Read Baseline Change Counts to Evaluate False Positive Rate
        baseline_fps = {"SC_CCM": 0, "PYCCD": 0, "TOTAL_PIXELS": 0}
        if "BASELINE_CHIPS" in bench_grp and "scccd_baseline_change_count" in bench_grp["BASELINE_CHIPS"]:
            sc_base = bench_grp["BASELINE_CHIPS/scccd_baseline_change_count"][:]
            py_base = bench_grp["BASELINE_CHIPS/pyccd_baseline_change_count"][:]
            baseline_fps["SC_CCM"] = int(np.sum(sc_base > 0))
            baseline_fps["PYCCD"] = int(np.sum(py_base > 0))
            baseline_fps["TOTAL_PIXELS"] = int(sc_base.size)
        elif "SCCCD_Results/change_count" in h5 and "PYCCD_Results/change_count" in h5:
            sc_cc = h5["SCCCD_Results/change_count"]
            py_cc = h5["PYCCD_Results/change_count"]

            total_uninjected = 0
            sc_fp_count = 0
            py_fp_count = 0

            for (r0, c0, h_p, w_p) in patch_bounds:
                sc_patch = sc_cc[r0:r0+h_p, c0:c0+w_p]
                py_patch = py_cc[r0:r0+h_p, c0:c0+w_p]

                # Create mask of the 4 injected positions
                inj_mask = np.zeros((h_p, w_p), dtype=bool)
                for (roff, coff) in injected_offsets:
                    inj_mask[roff, coff] = True

                # Evaluate un-injected background pixels only
                uninjected_mask = ~inj_mask
                sc_fp_count += int(np.sum(sc_patch[uninjected_mask] > 0))
                py_fp_count += int(np.sum(py_patch[uninjected_mask] > 0))
                total_uninjected += int(np.sum(uninjected_mask))

            baseline_fps["SC_CCM"] = sc_fp_count
            baseline_fps["PYCCD"] = py_fp_count
            baseline_fps["TOTAL_PIXELS"] = total_uninjected

    # Ensure MATERIAL_GROUPS reflects materials present in the loaded dataset
    material_groups = extract_material_groups(sc_table)

    data = {
        "group_name": group_name,
        "target_metric": target_metric,
        "patch_ids": patch_ids,
        "patch_names": patch_names,
        "fractions": fractions,
        "patch_bounds": patch_bounds,
        "injected_offsets": injected_offsets,
        "onset_date_str": onset_date_str,
        "sc_table": sc_table,
        "sc_table_cluster": sc_table_cluster,
        "py_table": py_table,
        "sc_segments": sc_segments,
        "py_segments": py_segments,
        "baseline_fps": baseline_fps,
        "material_groups": material_groups,
    }
    return data


def load_dual_benchmark_data(h5_path, base_group="/STRATIFIED_CHIP_BENCHMARK", group_3d="/STRATIFIED_CHIP_BENCHMARK_3D"):
    """
    Loads both baseline and 3D volume benchmark datasets from master container if present.
    """
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"Master HDF5 file not found: {h5_path}")

    with h5py.File(h5_path, "r", locking=False) as h5:
        has_base = base_group in h5
        has_3d = group_3d in h5

    data_base = load_benchmark_data(h5_path, group_name=base_group) if has_base else None
    data_3d = load_benchmark_data(h5_path, group_name=group_3d) if has_3d else None
    return data_base, data_3d


# ==============================================================================
# FIGURE 1: LAND-COVER STRATA SUBPIXEL SENSITIVITY (METHOD 1 vs. METHOD 2)
# ==============================================================================
def plot_figure1_recall_curves(data, save_path):
    """
    Plots subpixel change detection sensitivity curves comparing Method 1
    vs. Method 2 across all 4 land-cover strata on a single unified coordinate system.
    """
    m1 = data.get("method_1")
    m2 = data.get("method_2")
    m_ref = data.get("method_ref")

    sc_tab_pixel = m1["table"] if m1 else data["sc_table"]
    py_tab = m2["table"] if m2 else data["py_table"]
    ref_tab = m_ref["table"] if m_ref else None

    m1_lbl = m1.get("short_name", "SC-CCM 3D") if m1 else "SC-CCM 3D"
    m2_lbl = m2.get("short_name", "SC-CCM maxVol") if m2 else "SC-CCM maxVol"
    m_ref_lbl = m_ref.get("short_name", "Ref") if m_ref else "Ref"

    m1_ls = m1.get("linestyle", "-") if m1 else "-"
    m2_ls = m2.get("linestyle", "--") if m2 else "--"
    m_ref_ls = m_ref.get("linestyle", ":") if m_ref else ":"

    m1_marker = m1.get("marker", "o") if m1 else "o"
    m2_marker = m2.get("marker", "s") if m2 else "s"
    m_ref_marker = m_ref.get("marker", "^") if m_ref else "^"

    fractions = data["fractions"]
    patch_ids = data["patch_ids"]
    patch_names = data["patch_names"]
    f_pcts = fractions * 100.0

    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    fig.patch.set_facecolor("white")

    handles_m1 = []
    handles_m2 = []
    handles_ref = []

    # Horizontal x-axis biases (dodging) to prevent overlapping error bars
    n_strata = len(patch_ids)
    n_curves = 3 if ref_tab is not None else 2
    x_offsets = np.linspace(-0.6, 0.6, n_curves * n_strata)

    for idx, (pid, pname) in enumerate(zip(patch_ids, patch_names)):
        color = STRATA_PALETTE.get(pid, "#333333")
        display_name = STRATA_DISPLAY.get(pid, pid.title())

        pix_recs, pix_l, pix_h = [], [], []
        py_recs, py_l, py_h = [], [], []
        ref_recs, ref_l, ref_h = [], [], []

        for f in fractions:
            sc_p_sub = sc_tab_pixel[(np.isclose(sc_tab_pixel["mixture_fraction"], f)) & (sc_tab_pixel["patch_id"] == pid.encode())]
            py_sub = py_tab[(np.isclose(py_tab["mixture_fraction"], f)) & (py_tab["patch_id"] == pid.encode())]

            # Method 1
            n_p = len(sc_p_sub)
            k_p = int(np.sum(sc_p_sub["break_detected"])) if n_p > 0 else 0
            p_p, pi_l, pi_h = compute_binomial_ci(k_p, n_p, confidence=CONFIDENCE_INTERVAL, method="wilson")
            pix_recs.append(p_p)
            pix_l.append(pi_l)
            pix_h.append(pi_h)

            # Method 2
            n_py = len(py_sub)
            k_py = int(np.sum(py_sub["break_detected"])) if n_py > 0 else 0
            p_py, py_l_val, py_h_val = compute_binomial_ci(k_py, n_py, confidence=CONFIDENCE_INTERVAL, method="wilson")
            py_recs.append(p_py)
            py_l.append(py_l_val)
            py_h.append(py_h_val)

            # Optional Reference Method
            if ref_tab is not None:
                r_sub = ref_tab[(np.isclose(ref_tab["mixture_fraction"], f)) & (ref_tab["patch_id"] == pid.encode())]
                n_r = len(r_sub)
                k_r = int(np.sum(r_sub["break_detected"])) if n_r > 0 else 0
                p_r, rl_val, rh_val = compute_binomial_ci(k_r, n_r, confidence=CONFIDENCE_INTERVAL, method="wilson")
                ref_recs.append(p_r)
                ref_l.append(rl_val)
                ref_h.append(rh_val)

        pix_recs = np.array(pix_recs)
        pix_l = np.array(pix_l)
        pix_h = np.array(pix_h)
        py_recs = np.array(py_recs)
        py_l = np.array(py_l)
        py_h = np.array(py_h)

        if ref_tab is not None:
            x_sc = f_pcts + x_offsets[3 * idx]
            x_py = f_pcts + x_offsets[3 * idx + 1]
            x_rf = f_pcts + x_offsets[3 * idx + 2]
        else:
            x_sc = f_pcts + x_offsets[2 * idx]
            x_py = f_pcts + x_offsets[2 * idx + 1]

        # Vertical bar error bars for 95% Wilson score confidence intervals
        ax.errorbar(
            x_sc, pix_recs,
            yerr=[np.maximum(0.0, pix_recs - pix_l), np.maximum(0.0, pix_h - pix_recs)],
            fmt="none", ecolor=color, elinewidth=0.7, capsize=1.5, capthick=0.6, alpha=0.75, zorder=3
        )
        ax.errorbar(
            x_py, py_recs,
            yerr=[np.maximum(0.0, py_recs - py_l), np.maximum(0.0, py_h - py_recs)],
            fmt="none", ecolor=color, elinewidth=0.7, capsize=1.5, capthick=0.6, alpha=0.75, zorder=2
        )
        if ref_tab is not None and len(ref_recs) > 0:
            ref_recs = np.array(ref_recs)
            ref_l = np.array(ref_l)
            ref_h = np.array(ref_h)
            ax.errorbar(
                x_rf, ref_recs,
                yerr=[np.maximum(0.0, ref_recs - ref_l), np.maximum(0.0, ref_h - ref_recs)],
                fmt="none", ecolor=color, elinewidth=0.5, capsize=1.2, capthick=0.5, alpha=0.55, zorder=1
            )
            h_rf, = ax.plot(x_rf, ref_recs, color=color, marker=m_ref_marker, markersize=2.4, linewidth=0.9,
                            linestyle=m_ref_ls, label=f"{display_name} ({m_ref_lbl})", alpha=0.7, zorder=3)
            handles_ref.append(h_rf)

        # Plot Method 1 curve
        h_px, = ax.plot(x_sc, pix_recs, color=color, marker=m1_marker, markersize=3.0, linewidth=1.2,
                        linestyle=m1_ls, label=f"{display_name} ({m1_lbl})", zorder=5)
        handles_m1.append(h_px)

        # Plot Method 2 curve
        h_py, = ax.plot(x_py, py_recs, color=color, marker=m2_marker, markersize=2.8, linewidth=1.1,
                        linestyle=m2_ls, label=f"{display_name} ({m2_lbl})", zorder=4)
        handles_m2.append(h_py)

    ax.set_xlabel(r"Target Mixture Fraction $f$ (\%)", fontsize=7.5)
    ax.set_ylabel(r"Detection Sensitivity $P_{\mathrm{CD}}$ (\%)", fontsize=7.5)
    ax.set_xlim(-2.0, max(f_pcts) + 3.0)
    ax.set_ylim(-3.0, 105.0)
    ax.set_xticks(np.arange(0, int(max(f_pcts)) + 1, 10))
    ax.set_xticklabels([f"{x:.0f}\\%" for x in np.arange(0, int(max(f_pcts)) + 1, 10)], fontsize=6.2)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_yticklabels([f"{y:.0f}\\%" for y in np.arange(0, 101, 20)], fontsize=6.2)
    ax.grid(True, linestyle="--", alpha=0.45, color=COLOR_GRID)

    # Order legend
    all_handles = handles_m1 + handles_m2 + handles_ref
    all_labels = [h.get_label() for h in all_handles]
    ncols = 3 if len(handles_ref) > 0 else 2
    ax.legend(handles=all_handles, labels=all_labels, loc="lower right", fontsize=4.6 if ncols == 3 else 5.0, ncol=ncols,
              framealpha=0.92, columnspacing=0.4 if ncols == 3 else 0.5, handlelength=1.0, handletextpad=0.25,
              borderpad=0.25, labelspacing=0.25)

    plt.subplots_adjust(top=0.96, bottom=0.13, left=0.13, right=0.97)

    save_base = str(Path(save_path).with_suffix(""))
    plt.savefig(f"{save_base}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{save_base}.pdf", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Land-Cover Strata Figure to: {save_base}.png / .pdf")


# ==============================================================================
# FIGURE: MATERIAL CATEGORY AGGREGATED SENSITIVITY (METHOD 1 vs. METHOD 2)
# ==============================================================================
def plot_figure_material_categories_8curves(data, save_path):
    """
    Plots subpixel change detection sensitivity curves across the material categories
    (Artificial, Turbid Water, Vegetation, Charred Soil) comparing Method 1
    vs. Method 2 on a single unified coordinate system.
    """
    m1 = data.get("method_1")
    m2 = data.get("method_2")
    m_ref = data.get("method_ref")

    sc_tab_pixel = m1["table"] if m1 else data["sc_table"]
    py_tab = m2["table"] if m2 else data["py_table"]
    ref_tab = m_ref["table"] if m_ref else None

    m1_lbl = m1.get("short_name", "SC-CCM 3D") if m1 else "SC-CCM 3D"
    m2_lbl = m2.get("short_name", "SC-CCM maxVol") if m2 else "SC-CCM maxVol"
    m_ref_lbl = m_ref.get("short_name", "Ref") if m_ref else "Ref"

    m1_ls = m1.get("linestyle", "-") if m1 else "-"
    m2_ls = m2.get("linestyle", "--") if m2 else "--"
    m_ref_ls = m_ref.get("linestyle", ":") if m_ref else ":"

    m1_marker = m1.get("marker", "o") if m1 else "o"
    m2_marker = m2.get("marker", "s") if m2 else "s"
    m_ref_marker = m_ref.get("marker", "^") if m_ref else "^"

    fractions = data["fractions"]
    material_groups = data.get("material_groups", MATERIAL_GROUPS)
    f_pcts = fractions * 100.0

    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    fig.patch.set_facecolor("white")

    handles_m1 = []
    handles_m2 = []
    handles_ref = []

    # Horizontal x-axis biases (dodging) to prevent overlapping error bars
    n_groups = len(material_groups)
    n_curves = 3 if ref_tab is not None else 2
    x_offsets = np.linspace(-0.6, 0.6, n_curves * n_groups)

    for idx, grp in enumerate(material_groups):
        grp_name = grp["group"]
        color = CATEGORY_PALETTE.get(grp_name, "#333333")
        mat_pairs = [(m[0].encode(), m[1].encode()) for m in grp["materials"]]
        
        sc_p_mask = np.zeros(len(sc_tab_pixel), dtype=bool)
        py_mask = np.zeros(len(py_tab), dtype=bool)
        ref_mask = np.zeros(len(ref_tab), dtype=bool) if ref_tab is not None else None

        for mid_b, mname_b in mat_pairs:
            sc_p_mask |= ((sc_tab_pixel["material_id"] == mid_b) & (sc_tab_pixel["material_name"] == mname_b))
            py_mask |= ((py_tab["material_id"] == mid_b) & (py_tab["material_name"] == mname_b))
            if ref_mask is not None:
                ref_mask |= ((ref_tab["material_id"] == mid_b) & (ref_tab["material_name"] == mname_b))

        sc_p_grp = sc_tab_pixel[sc_p_mask]
        py_grp = py_tab[py_mask]
        ref_grp = ref_tab[ref_mask] if ref_tab is not None else None

        pix_recs, pix_l, pix_h = [], [], []
        py_recs, py_l, py_h = [], [], []
        ref_recs, ref_l, ref_h = [], [], []

        for f in fractions:
            sc_p_f = sc_p_grp[np.isclose(sc_p_grp["mixture_fraction"], f)]
            py_f = py_grp[np.isclose(py_grp["mixture_fraction"], f)]

            # Method 1 with Wilson CI
            n_p = len(sc_p_f)
            k_p = int(np.sum(sc_p_f["break_detected"])) if n_p > 0 else 0
            p_p, pi_l, pi_h = compute_binomial_ci(k_p, n_p, confidence=CONFIDENCE_INTERVAL, method="wilson")
            pix_recs.append(p_p)
            pix_l.append(pi_l)
            pix_h.append(pi_h)

            # Method 2 with Wilson CI
            n_py = len(py_f)
            k_py = int(np.sum(py_f["break_detected"])) if n_py > 0 else 0
            p_py, py_l_val, py_h_val = compute_binomial_ci(k_py, n_py, confidence=CONFIDENCE_INTERVAL, method="wilson")
            py_recs.append(p_py)
            py_l.append(py_l_val)
            py_h.append(py_h_val)

            # Optional Reference Method
            if ref_grp is not None:
                rf_f = ref_grp[np.isclose(ref_grp["mixture_fraction"], f)]
                n_rf = len(rf_f)
                k_rf = int(np.sum(rf_f["break_detected"])) if n_rf > 0 else 0
                p_rf, rf_l_val, rf_h_val = compute_binomial_ci(k_rf, n_rf, confidence=CONFIDENCE_INTERVAL, method="wilson")
                ref_recs.append(p_rf)
                ref_l.append(rf_l_val)
                ref_h.append(rf_h_val)

        pix_recs = np.array(pix_recs)
        pix_l = np.array(pix_l)
        pix_h = np.array(pix_h)
        py_recs = np.array(py_recs)
        py_l = np.array(py_l)
        py_h = np.array(py_h)

        if ref_tab is not None:
            x_sc = f_pcts + x_offsets[3 * idx]
            x_py = f_pcts + x_offsets[3 * idx + 1]
            x_rf = f_pcts + x_offsets[3 * idx + 2]
        else:
            x_sc = f_pcts + x_offsets[2 * idx]
            x_py = f_pcts + x_offsets[2 * idx + 1]

        # Vertical bar error bars for 95% Wilson score confidence intervals
        ax.errorbar(
            x_sc, pix_recs,
            yerr=[np.maximum(0.0, pix_recs - pix_l), np.maximum(0.0, pix_h - pix_recs)],
            fmt="none", ecolor=color, elinewidth=0.7, capsize=1.5, capthick=0.6, alpha=0.75, zorder=3
        )
        ax.errorbar(
            x_py, py_recs,
            yerr=[np.maximum(0.0, py_recs - py_l), np.maximum(0.0, py_h - py_recs)],
            fmt="none", ecolor=color, elinewidth=0.7, capsize=1.5, capthick=0.6, alpha=0.75, zorder=2
        )
        if ref_tab is not None and len(ref_recs) > 0:
            ref_recs = np.array(ref_recs)
            ref_l = np.array(ref_l)
            ref_h = np.array(ref_h)
            ax.errorbar(
                x_rf, ref_recs,
                yerr=[np.maximum(0.0, ref_recs - ref_l), np.maximum(0.0, ref_h - ref_recs)],
                fmt="none", ecolor=color, elinewidth=0.5, capsize=1.2, capthick=0.5, alpha=0.55, zorder=1
            )
            h_rf, = ax.plot(x_rf, ref_recs, color=color, marker=m_ref_marker, markersize=2.4, linewidth=0.9,
                            linestyle=m_ref_ls, label=f"{grp_name} ({m_ref_lbl})", alpha=0.7, zorder=3)
            handles_ref.append(h_rf)

        # Plot Method 1 curve
        h_px, = ax.plot(x_sc, pix_recs, color=color, marker=m1_marker, markersize=3.0, linewidth=1.2,
                        linestyle=m1_ls, label=f"{grp_name} ({m1_lbl})", zorder=5)
        handles_m1.append(h_px)

        # Plot Method 2 curve
        h_py, = ax.plot(x_py, py_recs, color=color, marker=m2_marker, markersize=2.8, linewidth=1.1,
                        linestyle=m2_ls, label=f"{grp_name} ({m2_lbl})", zorder=4)
        handles_m2.append(h_py)

    ax.set_xlabel(r"Target Mixture Fraction $f$ (\%)", fontsize=7.5)
    ax.set_ylabel(r"Detection Sensitivity $P_{\mathrm{CD}}$ (\%)", fontsize=7.5)
    ax.set_xlim(-2.0, max(f_pcts) + 3.0)
    ax.set_ylim(-3.0, 105.0)
    ax.set_xticks(np.arange(0, int(max(f_pcts)) + 1, 10))
    ax.set_xticklabels([f"{x:.0f}\\%" for x in np.arange(0, int(max(f_pcts)) + 1, 10)], fontsize=6.2)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_yticklabels([f"{y:.0f}\\%" for y in np.arange(0, 101, 20)], fontsize=6.2)
    ax.grid(True, linestyle="--", alpha=0.45, color=COLOR_GRID)

    # Order legend
    all_handles = handles_m1 + handles_m2 + handles_ref
    all_labels = [h.get_label() for h in all_handles]
    ncols = 3 if len(handles_ref) > 0 else 2
    ax.legend(handles=all_handles, labels=all_labels, loc="lower right", fontsize=4.6 if ncols == 3 else 5.0, ncol=ncols,
              framealpha=0.92, columnspacing=0.4 if ncols == 3 else 0.5, handlelength=1.0, handletextpad=0.25,
              borderpad=0.25, labelspacing=0.25)

    plt.subplots_adjust(top=0.96, bottom=0.13, left=0.13, right=0.97)

    save_base = str(Path(save_path).with_suffix(""))
    plt.savefig(f"{save_base}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{save_base}.pdf", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Material Category Figure to: {save_base}.png / .pdf")


# ==============================================================================
# FIGURE: INDIVIDUAL MATERIAL SENSITIVITY (SC-CCM ONLY)
# ==============================================================================
def plot_figure_individual_materials_scccd(data, save_path):
    """
    Plots subpixel change detection sensitivity curves for each implanted target material
    for SC-CCM (3x3 Cluster), aggregating the 4 pixels across the 4 land-cover strata,
    along with Grand Mean curves for both 3x3 Cluster and 1x1 Center Pixel.
    """
    sc_tab_pixel = data["sc_table"]
    sc_tab = data.get("sc_table_cluster", sc_tab_pixel)
    fractions = data["fractions"]
    material_groups = data.get("material_groups", MATERIAL_GROUPS)
    f_pcts = fractions * 100.0

    # Collect flat list of all materials across all groups
    flat_materials = []
    for grp in material_groups:
        for m_id, m_name in grp["materials"]:
            flat_materials.append((m_id, m_name))

    # Built-in high-contrast aesthetic styles for canonical USGS materials
    CANONICAL_STYLES = {
        # Artificial Materials
        "white_plastic":    {"color": "#e41a1c", "marker": "o", "label": "White Vinyl Plastic"},
        "concrete":         {"color": "#ff7f00", "marker": "s", "label": "Concrete (WTC01-37A)"},
        "plywood":          {"color": "#a65628", "marker": "^", "label": "Plywood Pine (Fresh)"},
        "asphalt shingle":  {"color": "#984ea3", "marker": "v", "label": "Asphalt Shingle"},
        "painted_aluminum": {"color": "#b15928", "marker": "P", "label": "Painted Aluminum"},
        "galvanized steel": {"color": "#666666", "marker": "X", "label": "Galvanized Steel"},
        "Nylon_Fabric":     {"color": "#4d4d4d", "marker": "d", "label": "Nylon Fabric"},
        # Charred Soil
        "burn_area":        {"color": "#e6550d", "marker": "D", "label": "Burn Area (WRF00-01)"},
        "burn_area2":       {"color": "#fdae6b", "marker": "D", "label": "Burn Area (WRF00-02)"},
        # Vegetation Dynamics
        "oak_leaves":       {"color": "#2ca02c", "marker": "o", "label": "Oak Leaves (Fresh)"},
        "dry_willow":       {"color": "#74c476", "marker": "s", "label": "Willow Dry (Senescent)"},
        "healthy_willow":   {"color": "#31a354", "marker": "s", "label": "Willow Healthy"},
        "dry_grass":        {"color": "#41ab5d", "marker": "^", "label": "Grass Golden Dry"},
        "Pinon_Pine":       {"color": "#006d2c", "marker": "p", "label": "Pinon Pine"},
        "oak_shrub":        {"color": "#005a32", "marker": "v", "label": "Oak Shrub Foliage"},
        "aspen_YellowGreenTop": {"color": "#a1d99b", "marker": "h", "label": "Aspen Foliage"},
        # Hydrologic Inundation
        "turbid_water":     {"color": "#1f77b4", "marker": "p", "label": "Turbid Water"},
    }

    # Distinct palette and markers for dynamic expansion
    fallback_palette = [
        "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c", "#98df8a",
        "#d62728", "#ff9896", "#9467bd", "#c5b0d5", "#8c564b", "#c49c94",
        "#e377c2", "#f7b6d2", "#7f7f7f", "#c7c7c7", "#bcbd22", "#dbdb8d",
        "#17becf", "#9edae5"
    ]
    fallback_markers = ["o", "s", "^", "v", "D", "<", ">", "p", "*", "h", "H", "8", "P", "X", "d"]

    material_styles = {}
    for idx, (m_id, m_name) in enumerate(flat_materials):
        if (m_id in CANONICAL_STYLES) and (m_id != "turbid_water"):
            st = dict(CANONICAL_STYLES[m_id])
            st["label"] = m_name
        else:
            c = fallback_palette[idx % len(fallback_palette)]
            m = fallback_markers[idx % len(fallback_markers)]
            st = {"color": c, "marker": m, "label": m_name}
        material_styles[(m_id, m_name)] = st

    fig = plt.figure(figsize=(7.16, 3.4))
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.8, 1.0], wspace=0.15)
    ax = fig.add_subplot(gs[0])
    ax_leg = fig.add_subplot(gs[1])
    ax_leg.axis("off")


    # Milestone indicator lines at f = 10%, 25%, 45%
    milestones = [10.0, 25.0, 45.0]

    # Plot each individual material curve (Cluster detection)
    handles = []
    for grp in material_groups:
        for m_id, m_name in grp["materials"]:
            style = material_styles.get((m_id, m_name), {"color": "#333333", "marker": "o", "label": m_name})
            sc_m = sc_tab[(sc_tab["material_id"] == m_id.encode()) & (sc_tab["material_name"] == m_name.encode())]

            sc_recs = []
            sc_se = []
            for f in fractions:
                sc_f = sc_m[np.isclose(sc_m["mixture_fraction"], f)]
                n_m = len(sc_f)
                if n_m > 0:
                    p_m = float(np.mean(sc_f["break_detected"]))
                    se_m = float(np.sqrt(max(0.0, p_m * (1.0 - p_m)) / n_m)) * 100.0
                else:
                    p_m = 0.0
                    se_m = 0.0
                sc_recs.append(p_m * 100.0)
                sc_se.append(se_m)

            sc_recs = np.array(sc_recs)
            sc_se = np.array(sc_se)
            sc_lower = np.clip(sc_recs - sc_se, 0.0, 100.0)
            sc_upper = np.clip(sc_recs + sc_se, 0.0, 100.0)

            # Shaded standard error band
            #ax.fill_between(f_pcts, sc_lower, sc_upper, color=style["color"], alpha=0.05, zorder=3)

            h, = ax.plot(f_pcts, sc_recs, color=style["color"], marker=style["marker"], markersize=3.0,
                         linewidth=1.1, linestyle="-", label=style["label"], zorder=5)
            handles.append(h)

    # Plot Grand Mean across all materials (Cluster)
    sc_comb_recs = []
    sc_comb_se = []
    # Also Grand Mean for Pixel
    pix_comb_recs = []
    pix_comb_se = []

    for f in fractions:
        sc_f_all = sc_tab[np.isclose(sc_tab["mixture_fraction"], f)]
        pix_f_all = sc_tab_pixel[np.isclose(sc_tab_pixel["mixture_fraction"], f)]

        n_all = len(sc_f_all)
        if n_all > 0:
            p_all = float(np.mean(sc_f_all["break_detected"]))
            se_all = float(np.sqrt(max(0.0, p_all * (1.0 - p_all)) / n_all)) * 100.0
        else:
            p_all = 0.0
            se_all = 0.0
        sc_comb_recs.append(p_all * 100.0)
        sc_comb_se.append(se_all)

        n_pix = len(pix_f_all)
        if n_pix > 0:
            p_pix = float(np.mean(pix_f_all["break_detected"]))
            se_pix = float(np.sqrt(max(0.0, p_pix * (1.0 - p_pix)) / n_pix)) * 100.0
        else:
            p_pix = 0.0
            se_pix = 0.0
        pix_comb_recs.append(p_pix * 100.0)
        pix_comb_se.append(se_pix)

    sc_comb_recs = np.array(sc_comb_recs)
    sc_comb_se = np.array(sc_comb_se)
    comb_lower = np.clip(sc_comb_recs - sc_comb_se, 0.0, 100.0)
    comb_upper = np.clip(sc_comb_recs + sc_comb_se, 0.0, 100.0)

    pix_comb_recs = np.array(pix_comb_recs)
    pix_comb_se = np.array(pix_comb_se)

    # Shaded standard error band for Grand Mean Cluster
    #ax.fill_between(f_pcts, comb_lower, comb_upper, color="#111111", alpha=0.16, zorder=4)

    grand_n = len(sc_tab[np.isclose(sc_tab["mixture_fraction"], fractions[0])]) if len(fractions) > 0 else 0
    m1_lbl_ind = data.get("method_1", {}).get("short_name", "SC-CCM 3D") if data.get("method_1") else "SC-CCM 3D"
    h_g_cl, = ax.plot(f_pcts, sc_comb_recs, color="#111111", marker="X", markersize=4.0, linewidth=1.6,
                      linestyle="--", label=rf"{m1_lbl_ind} Grand Mean $3\times 3$ ($N={grand_n}$)", zorder=6)
    h_g_px, = ax.plot(f_pcts, pix_comb_recs, color="#555555", marker="o", fillstyle="none", markersize=3.5, linewidth=1.3,
                      linestyle=":", label=rf"{m1_lbl_ind} Grand Mean $1\times 1$ ($N={grand_n}$)", zorder=6)
    handles.extend([h_g_cl, h_g_px])

    ax.set_xlabel(r"Target Mixture Fraction $f$ (\%)", fontsize=7.5)
    ax.set_ylabel(r"Detection Sensitivity $P_{\mathrm{CD}}$ (\%)", fontsize=7.5)
    ax.set_xlim(-2.0, max(f_pcts) + 3.0)
    ax.set_ylim(-3.0, 105.0)
    ax.set_xticks(np.arange(0, int(max(f_pcts)) + 1, 10))
    ax.set_xticklabels([f"{x:.0f}\\%" for x in np.arange(0, int(max(f_pcts)) + 1, 10)], fontsize=6.2)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_yticklabels([f"{y:.0f}\\%" for y in np.arange(0, 101, 20)], fontsize=6.2)
    ax.grid(True, linestyle="--", alpha=0.45, color=COLOR_GRID)

    labels = [h.get_label() for h in handles]
    ax_leg.legend(handles=handles, labels=labels, loc="center left", fontsize=5.0, ncol=1,
                  framealpha=0.92, labelspacing=0.22, handlelength=1.1, handletextpad=0.25, borderpad=0.3)

    plt.subplots_adjust(top=0.96, bottom=0.12, left=0.08, right=0.98)

    save_base = str(Path(save_path).with_suffix(""))
    plt.savefig(f"{save_base}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{save_base}.pdf", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved SC-CCM Individual Materials Figure to: {save_base}.png / .pdf")


# ==============================================================================
# FIGURE: TWO-PANEL COMPOSITE BENCHMARK SENSITIVITY (STRATA + MATERIAL CATEGORIES)
# ==============================================================================
def plot_figure_composite_sensitivity(data, save_path):
    """
    Combines the Land-Cover Strata and Material Category sensitivity comparisons
    into a space-efficient 2-panel side-by-side publication figure for IEEE JSTARS.
    Shows Method 1 3x3 Neighborhood Cluster, Method 1 1x1 On-Target Center, and Method 2.
    """
    m1 = data.get("method_1")
    m2 = data.get("method_2")

    sc_tab_pixel = m1["table"] if m1 else data["sc_table"]
    sc_tab_clust = m1.get("cluster_table", data.get("sc_table_cluster", sc_tab_pixel)) if m1 else data.get("sc_table_cluster", sc_tab_pixel)
    py_tab = m2["table"] if m2 else data["py_table"]

    m1_lbl = m1.get("short_name", "SC-CCM 3D") if m1 else "SC-CCM 3D"
    m2_lbl = m2.get("short_name", "USGS PyCCD") if m2 else "USGS PyCCD"

    m2_ls = m2.get("linestyle", "--") if m2 else "--"
    m2_marker = m2.get("marker", "s") if m2 else "s"

    fractions = data["fractions"]
    patch_ids = data["patch_ids"]
    f_pcts = fractions * 100.0
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 2.9), sharey=True)
    fig.patch.set_facecolor("white")

    # --------------------------------------------------------------------------
    # Panel (a): Land-Cover Strata
    # --------------------------------------------------------------------------
    class_handles_a = []

    for pid in patch_ids:
        color = STRATA_PALETTE.get(pid, "#333333")
        display_name = STRATA_DISPLAY.get(pid, pid.title())

        clust_recs, pix_recs, py_recs = [], [], []

        for f in fractions:
            sc_c_sub = sc_tab_clust[(np.isclose(sc_tab_clust["mixture_fraction"], f)) & (sc_tab_clust["patch_id"] == pid.encode())]
            sc_p_sub = sc_tab_pixel[(np.isclose(sc_tab_pixel["mixture_fraction"], f)) & (sc_tab_pixel["patch_id"] == pid.encode())]
            py_sub = py_tab[(np.isclose(py_tab["mixture_fraction"], f)) & (py_tab["patch_id"] == pid.encode())]

            n_c = len(sc_c_sub)
            k_c = int(np.sum(sc_c_sub["break_detected"])) if n_c > 0 else 0
            p_c, _, _ = compute_binomial_ci(k_c, n_c, confidence=CONFIDENCE_INTERVAL, method="wilson")
            clust_recs.append(p_c)

            n_p = len(sc_p_sub)
            k_p = int(np.sum(sc_p_sub["break_detected"])) if n_p > 0 else 0
            p_p, _, _ = compute_binomial_ci(k_p, n_p, confidence=CONFIDENCE_INTERVAL, method="wilson")
            pix_recs.append(p_p)

            n_py = len(py_sub)
            k_py = int(np.sum(py_sub["break_detected"])) if n_py > 0 else 0
            p_py, _, _ = compute_binomial_ci(k_py, n_py, confidence=CONFIDENCE_INTERVAL, method="wilson")
            py_recs.append(p_py)

        ax1.plot(f_pcts, np.array(clust_recs), color=color, marker="o", markersize=2.8, linewidth=1.2,
                 linestyle="-", zorder=6)
        ax1.plot(f_pcts, np.array(pix_recs), color=color, marker="o", fillstyle="none", markersize=2.5, linewidth=1.0,
                 linestyle=":", zorder=5)
        ax1.plot(f_pcts, np.array(py_recs), color=color, marker=m2_marker, markersize=2.6, linewidth=1.1,
                 linestyle=m2_ls, zorder=4)

        class_handles_a.append(mlines.Line2D([], [], color=color, marker="s", markersize=3.2,
                                             linewidth=0, label=display_name))

    method_handles = [
        mlines.Line2D([], [], color="#222222", linestyle="-", marker="o", markersize=2.8,
                      linewidth=1.2, label=rf"{m1_lbl} $3\times 3$ Cluster"),
        mlines.Line2D([], [], color="#222222", linestyle=":", marker="o", fillstyle="none",
                      markersize=2.5, linewidth=1.0, label=rf"{m1_lbl} $1\times 1$ Center"),
        mlines.Line2D([], [], color="#222222", linestyle=m2_ls, marker=m2_marker, markersize=2.6,
                      linewidth=1.1, label=rf"{m2_lbl}")
    ]

    ax1.set_title(r"(a) Land-Cover Strata Sensitivity", fontsize=8.0, fontweight="bold", pad=6)
    ax1.set_xlabel(r"Target Mixture Fraction $f$ (\%)", fontsize=7.5)
    ax1.set_ylabel(r"Detection Sensitivity $P_{\mathrm{CD}}$ (\%)", fontsize=7.5)
    ax1.set_xlim(-2.0, max(f_pcts) + 3.0)
    ax1.set_ylim(-3.0, 105.0)
    ax1.set_xticks(np.arange(0, int(max(f_pcts)) + 1, 10))
    ax1.set_xticklabels([f"{x:.0f}\\%" for x in np.arange(0, int(max(f_pcts)) + 1, 10)], fontsize=6.2)
    ax1.set_yticks(np.arange(0, 101, 20))
    ax1.set_yticklabels([f"{y:.0f}\\%" for y in np.arange(0, 101, 20)], fontsize=6.2)
    ax1.grid(True, linestyle="--", alpha=0.45, color=COLOR_GRID)

    all_h_a = class_handles_a + method_handles
    ax1.legend(handles=all_h_a, loc="lower right", fontsize=4.1, ncol=2,
               framealpha=0.92, columnspacing=1.2, handlelength=1.1, handletextpad=0.30,
               borderpad=0.25, labelspacing=0.22)

    # --------------------------------------------------------------------------
    # Panel (b): Target Material Categories
    # --------------------------------------------------------------------------
    material_groups = data.get("material_groups", MATERIAL_GROUPS)
    class_handles_b = []

    for grp in material_groups:
        grp_name = grp["group"]
        color = CATEGORY_PALETTE.get(grp_name, "#333333")
        mat_pairs = [(m[0].encode(), m[1].encode()) for m in grp["materials"]]

        sc_c_mask = np.zeros(len(sc_tab_clust), dtype=bool)
        sc_p_mask = np.zeros(len(sc_tab_pixel), dtype=bool)
        py_mask = np.zeros(len(py_tab), dtype=bool)
        for mid_b, mname_b in mat_pairs:
            sc_c_mask |= ((sc_tab_clust["material_id"] == mid_b) & (sc_tab_clust["material_name"] == mname_b))
            sc_p_mask |= ((sc_tab_pixel["material_id"] == mid_b) & (sc_tab_pixel["material_name"] == mname_b))
            py_mask |= ((py_tab["material_id"] == mid_b) & (py_tab["material_name"] == mname_b))

        sc_c_grp = sc_tab_clust[sc_c_mask]
        sc_p_grp = sc_tab_pixel[sc_p_mask]
        py_grp = py_tab[py_mask]

        clust_recs, clust_l, clust_h = [], [], []
        pix_recs, py_recs, py_l, py_h = [], [], [], []

        for f in fractions:
            sc_c_f = sc_c_grp[np.isclose(sc_c_grp["mixture_fraction"], f)]
            sc_p_f = sc_p_grp[np.isclose(sc_p_grp["mixture_fraction"], f)]
            py_f = py_grp[np.isclose(py_grp["mixture_fraction"], f)]

            n_c = len(sc_c_f)
            k_c = int(np.sum(sc_c_f["break_detected"])) if n_c > 0 else 0
            p_c, cl_l, cl_h = compute_binomial_ci(k_c, n_c, confidence=CONFIDENCE_INTERVAL, method="wilson")
            clust_recs.append(p_c)
            clust_l.append(cl_l)
            clust_h.append(cl_h)

            n_p = len(sc_p_f)
            k_p = int(np.sum(sc_p_f["break_detected"])) if n_p > 0 else 0
            p_p, _, _ = compute_binomial_ci(k_p, n_p, confidence=CONFIDENCE_INTERVAL, method="wilson")
            pix_recs.append(p_p)

            n_py = len(py_f)
            k_py = int(np.sum(py_f["break_detected"])) if n_py > 0 else 0
            p_py, py_l_val, py_h_val = compute_binomial_ci(k_py, n_py, confidence=CONFIDENCE_INTERVAL, method="wilson")
            py_recs.append(p_py)
            py_l.append(py_l_val)
            py_h.append(py_h_val)

        clust_recs = np.array(clust_recs)
        clust_l = np.array(clust_l)
        clust_h = np.array(clust_h)
        pix_recs = np.array(pix_recs)
        py_recs = np.array(py_recs)
        py_l = np.array(py_l)
        py_h = np.array(py_h)

        ax2.fill_between(f_pcts, clust_l, clust_h, color=color, alpha=0.10, zorder=3)
        ax2.fill_between(f_pcts, py_l, py_h, color=color, alpha=0.06, zorder=2)

        ax2.plot(f_pcts, clust_recs, color=color, marker="o", markersize=2.8, linewidth=1.2,
                 linestyle="-", zorder=6)
        ax2.plot(f_pcts, pix_recs, color=color, marker="o", fillstyle="none", markersize=2.5, linewidth=1.0,
                 linestyle=":", zorder=5)
        ax2.plot(f_pcts, py_recs, color=color, marker=m2_marker, markersize=2.6, linewidth=1.1,
                 linestyle=m2_ls, zorder=4)

        class_handles_b.append(mlines.Line2D([], [], color=color, marker="s", markersize=3.2,
                                             linewidth=0, label=grp_name))

    ax2.set_title(r"(b) Target Material Category Sensitivity", fontsize=8.0, fontweight="bold", pad=6)
    ax2.set_xlabel(r"Target Mixture Fraction $f$ (\%)", fontsize=7.5)
    ax2.set_xlim(-2.0, max(f_pcts) + 3.0)
    ax2.set_ylim(-3.0, 105.0)
    ax2.set_xticks(np.arange(0, int(max(f_pcts)) + 1, 10))
    ax2.set_xticklabels([f"{x:.0f}\\%" for x in np.arange(0, int(max(f_pcts)) + 1, 10)], fontsize=6.2)
    ax2.grid(True, linestyle="--", alpha=0.45, color=COLOR_GRID)

    all_h_b = class_handles_b + method_handles
    ax2.legend(handles=all_h_b, loc="lower right", fontsize=4.1, ncol=2,
               framealpha=0.92, columnspacing=1.2, handlelength=1.1, handletextpad=0.30,
               borderpad=0.25, labelspacing=0.22)

    plt.subplots_adjust(top=0.92, bottom=0.13, left=0.07, right=0.98, wspace=0.10)

    save_base = str(Path(save_path).with_suffix(""))
    plt.savefig(f"{save_base}.png", dpi=300, pad_inches=0, bbox_inches="tight")
    plt.savefig(f"{save_base}.pdf", dpi=600, pad_inches=0, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved 2-Panel Composite Sensitivity Figure to: {save_base}.png / .pdf")


# ==============================================================================
# FIGURE 2: OPERATIONAL ROC SPACE COMPARISON
# ==============================================================================
def plot_figure2_roc_space(data, save_path):
    """
    Plots operational ROC points comparing Method 1 vs. Method 2.
    False positive rate is calculated from un-injected spatial control pixels.
    """
    m1 = data.get("method_1")
    m2 = data.get("method_2")

    sc_tab = m1["cluster_table"] if m1 else data.get("sc_table_cluster", data["sc_table"])
    py_tab = m2["table"] if m2 else data["py_table"]
    fractions = data["fractions"]
    patch_ids = data["patch_ids"]
    baseline_fps = data["baseline_fps"]

    m1_name = m1.get("short_name", "SC-CCM 3D") if m1 else "SC-CCM 3D"
    m2_name = m2.get("short_name", "USGS PyCCD") if m2 else "USGS PyCCD"
    m1_color = m1.get("color", COLOR_SCCCD) if m1 else COLOR_SCCCD
    m2_color = m2.get("color", COLOR_PYCCD) if m2 else COLOR_PYCCD
    m1_ls = m1.get("linestyle", "-") if m1 else "-"
    m2_ls = m2.get("linestyle", "--") if m2 else "--"
    m1_fpr_key = m1.get("fpr_key", "SC_CCM") if m1 else "SC_CCM"
    m2_fpr_key = m2.get("fpr_key", "SC_CCM") if m2 else "SC_CCM"

    total_controls = m1.get("total_pixels", baseline_fps.get("TOTAL_PIXELS", 1)) if m1 else baseline_fps.get("TOTAL_PIXELS", 1)
    sc_fpr = m1.get("fpr", baseline_fps.get(m1_fpr_key, 0) / max(total_controls, 1)) if m1 else (baseline_fps.get("SC_CCM", 0) / max(total_controls, 1))
    py_fpr = m2.get("fpr", baseline_fps.get(m2_fpr_key, 0) / max(total_controls, 1)) if m2 else (baseline_fps.get("PYCCD", 0) / max(total_controls, 1))
    m1_fp = m1.get("fp_count", baseline_fps.get(m1_fpr_key, 0)) if m1 else baseline_fps.get(m1_fpr_key, 0)
    m2_fp = m2.get("fp_count", baseline_fps.get(m2_fpr_key, 0)) if m2 else baseline_fps.get(m2_fpr_key, 0)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    fig.patch.set_facecolor("white")

    # Diagonal random classifier reference line
    ax.plot([0, 1], [0, 1], color="gray", linestyle=":", linewidth=1.2, label="Random Guess (AUC = 0.50)")

    # Compute TPR trajectory across mixture fractions
    sc_tprs = []
    py_tprs = []

    for f in fractions:
        sc_sub = sc_tab[np.isclose(sc_tab["mixture_fraction"], f)]
        py_sub = py_tab[np.isclose(py_tab["mixture_fraction"], f)]
        sc_tprs.append(float(np.mean(sc_sub["break_detected"])) if len(sc_sub) > 0 else 0.0)
        py_tprs.append(float(np.mean(py_sub["break_detected"])) if len(py_sub) > 0 else 0.0)

    annotate_milestones = [0.0, 0.10, 0.25, 0.50, 0.75, 1.00] if len(fractions) > 8 else fractions
    # Place lower-FPR trajectory labels on the left (with margin) and higher-FPR labels on the right
    m1_on_left = sc_fpr < py_fpr
    x_min_lim = min(-0.011, min(sc_fpr, py_fpr) - 0.010)
    x_max_lim = max(0.036, max(sc_fpr, py_fpr) + 0.022)

    # Plot Method 1 trajectory
    sc_fprs = [sc_fpr] * len(fractions)
    ax.plot(sc_fprs, sc_tprs, color=m1_color, linestyle=m1_ls, linewidth=2.0, zorder=3)
    for i, (fpr_val, tpr_val, f_val) in enumerate(zip(sc_fprs, sc_tprs, fractions)):
        marker_size = 45 + (i / max(len(fractions) - 1, 1)) * 100
        ax.scatter(fpr_val, tpr_val, color=m1_color, s=marker_size, edgecolors="black", linewidths=0.9, zorder=5)
        if any(np.isclose(f_val, m) for m in annotate_milestones):
            dx = -10 if m1_on_left else 10
            ha = "right" if m1_on_left else "left"
            ax.annotate(rf"$f={f_val:.2f}$ ({tpr_val*100:.1f}\%)", (fpr_val, tpr_val),
                        textcoords="offset points", xytext=(dx, -3), ha=ha, fontsize=7.5, color=m1_color)

    # Plot Method 2 trajectory
    py_fprs = [py_fpr] * len(fractions)
    ax.plot(py_fprs, py_tprs, color=m2_color, linestyle=m2_ls, linewidth=2.0, zorder=3)
    for i, (fpr_val, tpr_val, f_val) in enumerate(zip(py_fprs, py_tprs, fractions)):
        marker_size = 45 + (i / max(len(fractions) - 1, 1)) * 100
        ax.scatter(fpr_val, tpr_val, color=m2_color, s=marker_size, edgecolors="black", linewidths=0.9, zorder=5)
        if any(np.isclose(f_val, m) for m in annotate_milestones):
            dx = 10 if m1_on_left else -10
            ha = "left" if m1_on_left else "right"
            ax.annotate(rf"$f={f_val:.2f}$ ({tpr_val*100:.1f}\%)", (fpr_val, tpr_val),
                        textcoords="offset points", xytext=(dx, -3), ha=ha, fontsize=7.5, color=m2_color)

    # Plot stratified markers at milestone fraction (prefer f = 0.45 if available)
    milestone_f = 0.45 if np.any(np.isclose(fractions, 0.45)) else (float(fractions[-1]) if len(fractions) > 0 else 0.45)
    for pid in patch_ids:
        sc_sub = sc_tab[np.isclose(sc_tab["mixture_fraction"], milestone_f) & (sc_tab["patch_id"] == pid.encode())]
        py_sub = py_tab[np.isclose(py_tab["mixture_fraction"], milestone_f) & (py_tab["patch_id"] == pid.encode())]
        sc_strat_tpr = float(np.mean(sc_sub["break_detected"])) if len(sc_sub) > 0 else 0.0
        py_strat_tpr = float(np.mean(py_sub["break_detected"])) if len(py_sub) > 0 else 0.0
        color = STRATA_PALETTE.get(pid, "black")

        ax.scatter(sc_fpr, sc_strat_tpr, color=color, marker="^", s=70, edgecolors="blue", linewidths=1.0, zorder=6)
        ax.scatter(py_fpr, py_strat_tpr, color=color, marker="v", s=70, edgecolors="red", linewidths=1.0, zorder=6)

    # Styling and Limits (Zoomed Operational View)
    ax.set_xlim(x_min_lim, x_max_lim)
    ax.set_ylim(-0.04, 1.04)
    ax.set_xlabel("False Positive Rate (FPR) on Un-Injected Control Pixels", fontsize=9.5)
    ax.set_ylabel(r"True Positive Rate ($P_{\mathrm{CD}}$ / Sensitivity) on Injected Disturbance", fontsize=9.5)
    ax.set_title(f"Operational ROC Space: {m1_name} vs. {m2_name}", fontsize=11.0, fontweight="bold", pad=10)
    ax.tick_params(axis="both", labelsize=8.0)
    ax.grid(True, linestyle="--", alpha=0.5, color=COLOR_GRID)

    # Custom Legend
    legend_elements = [
        mpatches.Patch(facecolor=m1_color, edgecolor="black", label=rf"{m1_name} Trajectory (FPR = {sc_fpr*100:.2f}\%, N={total_controls})"),
        mpatches.Patch(facecolor=m2_color, edgecolor="black", label=rf"{m2_name} Trajectory (FPR = {py_fpr*100:.2f}\%, N={total_controls})"),
        plt.Line2D([0], [0], marker="^", color="w", markeredgecolor="blue", markerfacecolor="gray", markersize=7.5, label=rf"Stratified {m1_name} at $f = {milestone_f:.2f}$"),
        plt.Line2D([0], [0], marker="v", color="w", markeredgecolor="red", markerfacecolor="gray", markersize=7.5, label=rf"Stratified {m2_name} at $f = {milestone_f:.2f}$"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=7.5, framealpha=0.92)

    # Context Text Box
    max_f = float(fractions[-1]) if len(fractions) > 0 else 1.0
    stats_text = (
        f"Control Background Statistics:\n"
        f"  Total Un-Injected Pixels: {total_controls}\n"
        rf"  {m1_name} False Alarms:   {m1_fp} ({sc_fpr*100:.2f}\%)" + "\n"
        rf"  {m2_name} False Alarms:   {m2_fp} ({py_fpr*100:.2f}\%)" + "\n"
        f"\nOperational Separation:\n"
        rf"  {m1_name} reaches TPR = {sc_tprs[-1]*100:.1f}\% at $f = {max_f:.2f}$." + "\n"
        rf"  {m2_name} reaches TPR = {py_tprs[-1]*100:.1f}\% at $f = {max_f:.2f}$." + "\n"
        f"  Demonstrates subpixel recovery without\n"
        f"  background false alarm inflation."
    )
    ax.text(0.98, 0.04, stats_text, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7.2, family="monospace",
            bbox=dict(boxstyle="round,pad=0.6", facecolor="#ffffff", edgecolor="#adb5bd", linewidth=1.0, alpha=0.94))

    save_base = str(Path(save_path).with_suffix(""))
    plt.savefig(f"{save_base}.png", dpi=300, pad_inches=0.04, bbox_inches="tight")
    plt.savefig(f"{save_base}.pdf", dpi=600, pad_inches=0.04, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Figure 2 to: {save_base}.png / .pdf")


# ==============================================================================
# FIGURE 3: STRATIFIED DETECTION LATENCY DISTRIBUTIONS
# ==============================================================================
def plot_figure3_latency_distributions(data, save_path):
    """
    Plots paired boxplots comparing detection latency (days from injection onset)
    between Method 1 and Method 2 across land-cover strata.
    """
    m1 = data.get("method_1")
    m2 = data.get("method_2")

    sc_tab = m1["cluster_table"] if m1 else data.get("sc_table_cluster", data["sc_table"])
    py_tab = m2["table"] if m2 else data["py_table"]
    patch_ids = data["patch_ids"]

    m1_name = m1.get("short_name", "SC-CCM 3D") if m1 else "SC-CCM 3D"
    m2_name = m2.get("short_name", "USGS PyCCD") if m2 else "USGS PyCCD"
    m1_color = m1.get("color", COLOR_SCCCD) if m1 else COLOR_SCCCD
    m2_color = m2.get("color", COLOR_PYCCD) if m2 else COLOR_PYCCD

    fig, ax = plt.subplots(figsize=(7.6, 4.5))
    fig.patch.set_facecolor("white")

    categories = patch_ids + ["OVERALL"]
    cat_labels = [STRATA_DISPLAY.get(p, p).replace(" ", "\n") for p in patch_ids] + ["Overall Scene\nAggregate"]

    sc_lat_data = []
    py_lat_data = []
    sc_counts = []
    py_counts = []

    for cat in categories:
        if cat == "OVERALL":
            sc_lats = sc_tab["latency_days"][sc_tab["break_detected"] & ~np.isnan(sc_tab["latency_days"])]
            py_lats = py_tab["latency_days"][py_tab["break_detected"] & ~np.isnan(py_tab["latency_days"])]
        else:
            sc_lats = sc_tab["latency_days"][(sc_tab["patch_id"] == cat.encode()) & sc_tab["break_detected"] & ~np.isnan(sc_tab["latency_days"])]
            py_lats = py_tab["latency_days"][(py_tab["patch_id"] == cat.encode()) & py_tab["break_detected"] & ~np.isnan(py_tab["latency_days"])]

        sc_lat_data.append(sc_lats)
        py_lat_data.append(py_lats)
        sc_counts.append(len(sc_lats))
        py_counts.append(len(py_lats))

    centers = np.arange(len(categories)) * 2.5
    positions_sc = centers - 0.45
    positions_py = centers + 0.45

    # Render Method 1 Boxplots
    ax.boxplot(
        sc_lat_data, positions=positions_sc, widths=0.65, patch_artist=True,
        boxprops=dict(facecolor=m1_color, alpha=0.75, edgecolor="black", linewidth=1.0),
        medianprops=dict(color="yellow", linewidth=1.8),
        whiskerprops=dict(color="black", linewidth=1.0),
        capprops=dict(color="black", linewidth=1.0),
        flierprops=dict(marker="o", markerfacecolor=m1_color, markeredgecolor="none", markersize=3.5, alpha=0.5),
    )

    # Render Method 2 Boxplots
    ax.boxplot(
        py_lat_data, positions=positions_py, widths=0.65, patch_artist=True,
        boxprops=dict(facecolor=m2_color, alpha=0.75, edgecolor="black", linewidth=1.0),
        medianprops=dict(color="yellow", linewidth=1.8),
        whiskerprops=dict(color="black", linewidth=1.0),
        capprops=dict(color="black", linewidth=1.0),
        flierprops=dict(marker="s", markerfacecolor=m2_color, markeredgecolor="none", markersize=3.5, alpha=0.5),
    )

    max_vals = [float(np.max(x)) for x in (sc_lat_data + py_lat_data) if len(x) > 0]
    y_data_max = max(max_vals) if max_vals else 120.0
    y_ann_base = max(145.0, y_data_max + 12.0)
    y_top_lim = y_ann_base + 52.0

    # Annotate medians and sample counts cleanly above boxes without horizontal overlap
    for i, (sc_lats, py_lats, sc_n, py_n) in enumerate(zip(sc_lat_data, py_lat_data, sc_counts, py_counts)):
        sc_med = np.median(sc_lats) if len(sc_lats) > 0 else np.nan
        py_med = np.median(py_lats) if len(py_lats) > 0 else np.nan

        if not np.isnan(sc_med):
            ax.annotate(f"Med: {sc_med:.1f}d\n(N={sc_n})", (positions_sc[i] - 0.08, y_ann_base),
                        ha="center", va="bottom", fontsize=6.5, color=m1_color)

        if not np.isnan(py_med):
            ax.annotate(f"Med: {py_med:.1f}d\n(N={py_n})", (positions_py[i] + 0.08, y_ann_base),
                        ha="center", va="bottom", fontsize=6.5, color=m2_color)
        else:
            ax.annotate("N=0", (positions_py[i] + 0.08, y_ann_base),
                        ha="center", va="bottom", fontsize=6.5, color="gray")

    ax.set_xticks(centers)
    ax.set_xticklabels(cat_labels, fontsize=8.5)
    ax.set_ylabel(r"Detection Latency $\Delta t$ (Days Post-Onset)", fontsize=9.5)
    ax.set_ylim(-5, y_top_lim)
    ax.set_xlim(centers[0] - 1.3, centers[-1] + 1.3)
    ax.set_title(f"Detection Confirmation Latency: {m1_name} vs. {m2_name}", fontsize=11.0, fontweight="bold", pad=26)
    ax.tick_params(axis="y", labelsize=8.0)
    ax.grid(True, linestyle="--", alpha=0.5, color=COLOR_GRID, axis="y")

    # Legend above plot area so it never overlaps median annotations
    legend_elements = [
        mpatches.Patch(facecolor=m1_color, edgecolor="black", label=f"{m1_name} [Total N = {sc_counts[-1]}]"),
        mpatches.Patch(facecolor=m2_color, edgecolor="black", label=f"{m2_name} [Total N = {py_counts[-1]}]"),
    ]
    ax.legend(handles=legend_elements, loc="lower center", bbox_to_anchor=(0.5, 1.01),
              ncol=2, fontsize=8.5, framealpha=0.92)

    save_base = str(Path(save_path).with_suffix(""))
    plt.savefig(f"{save_base}.png", dpi=300, pad_inches=0.04, bbox_inches="tight")
    plt.savefig(f"{save_base}.pdf", dpi=600, pad_inches=0.04, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Figure 3 to: {save_base}.png / .pdf")


# ==============================================================================
# FIGURE 4: MATERIAL CONTRAST SENSITIVITY MATRIX (f_min HEATMAP)
# ==============================================================================
def plot_figure4_material_sensitivity_heatmap(data, save_path):
    """
    Plots dual heatmaps comparing the minimum detectable mixture fraction (f_min)
    between Method 1 and Method 2 across target materials and land-cover strata.
    """
    m1 = data.get("method_1")
    m2 = data.get("method_2")

    sc_tab = m1["cluster_table"] if m1 else data.get("sc_table_cluster", data["sc_table"])
    py_tab = m2["table"] if m2 else data["py_table"]
    fractions = data["fractions"]
    patch_ids = data["patch_ids"]
    material_groups = data.get("material_groups", MATERIAL_GROUPS)

    m1_name = m1.get("short_name", "SC-CCM 3D") if m1 else "SC-CCM 3D"
    m2_name = m2.get("short_name", "SC-CCM maxVol") if m2 else "SC-CCM maxVol"
    m1_full = m1.get("full_name", m1_name) if m1 else "SC-CCM 3D"
    m2_full = m2.get("full_name", m2_name) if m2 else "SC-CCM maxVol"

    # Flat list of materials in structured order
    flat_materials = []
    group_labels = []
    group_boundaries = []
    curr_idx = 0

    for g in material_groups:
        group_labels.append((curr_idx + len(g["materials"]) / 2.0 - 0.5, g["group"]))
        for m_id, m_name in g["materials"]:
            flat_materials.append((m_id, m_name))
            curr_idx += 1
        group_boundaries.append(curr_idx - 0.5)

    n_mats = len(flat_materials)
    n_cols = len(patch_ids) + 1  # 4 strata + overall

    col_names = [STRATA_DISPLAY.get(p, p).replace(" ", "\n") for p in patch_ids] + ["Overall\nScene"]

    # Maximum fraction and undetected sentinel value dynamically derived from fractions
    f_max = float(np.max(fractions)) if len(fractions) > 0 else 1.0
    f_step = float(np.min(np.diff(fractions))) if len(fractions) > 1 else 0.05
    undetected_val = round(f_max + f_step, 4)

    # Matrices of f_min initialized to undetected_val
    matrix_sc = np.full((n_mats, n_cols), undetected_val, dtype=np.float32)
    matrix_py = np.full((n_mats, n_cols), undetected_val, dtype=np.float32)

    matrix_sc_rec = np.zeros((n_mats, n_cols), dtype=np.float32)
    matrix_py_rec = np.zeros((n_mats, n_cols), dtype=np.float32)

    for m_idx, (m_id, m_name) in enumerate(flat_materials):
        for c_idx in range(n_cols):
            if c_idx < len(patch_ids):
                pid = patch_ids[c_idx]
                sc_sub = sc_tab[(sc_tab["material_id"] == m_id.encode()) & 
                                (sc_tab["material_name"] == m_name.encode()) & 
                                (sc_tab["patch_id"] == pid.encode())]
                py_sub = py_tab[(py_tab["material_id"] == m_id.encode()) & 
                                (py_tab["material_name"] == m_name.encode()) & 
                                (py_tab["patch_id"] == pid.encode())]
            else:
                sc_sub = sc_tab[(sc_tab["material_id"] == m_id.encode()) & 
                                (sc_tab["material_name"] == m_name.encode())]
                py_sub = py_tab[(py_tab["material_id"] == m_id.encode()) & 
                                (py_tab["material_name"] == m_name.encode())]

            # Determine f_min for P_CD >= 0.75
            for f in fractions:
                rec_sc = np.mean(sc_sub["break_detected"][sc_sub["mixture_fraction"] == f]) if len(sc_sub) > 0 else 0.0
                if rec_sc >= 0.75 and matrix_sc[m_idx, c_idx] >= undetected_val:
                    matrix_sc[m_idx, c_idx] = f
                    matrix_sc_rec[m_idx, c_idx] = rec_sc

                rec_py = np.mean(py_sub["break_detected"][py_sub["mixture_fraction"] == f]) if len(py_sub) > 0 else 0.0
                if rec_py >= 0.75 and matrix_py[m_idx, c_idx] >= undetected_val:
                    matrix_py[m_idx, c_idx] = f
                    matrix_py_rec[m_idx, c_idx] = rec_py

    # Colormap and boundaries spanning the entire range of fractions (e.g. 0.05 to 1.00) + undetected
    bounds_cm = [0.0]
    for f in fractions:
        bounds_cm.append(float(f + f_step / 2.0))
    bounds_cm.append(float(undetected_val + f_step / 2.0))

    base_cmap = plt.colormaps["RdYlGn_r"]
    norm_colors = base_cmap(np.linspace(0.02, 0.98, len(fractions)))
    undetected_color = np.array([[0.93, 0.93, 0.93, 1.0]])  # Light neutral gray for undetected
    all_colors = np.vstack([norm_colors, undetected_color])

    cmap = mcolors.ListedColormap(all_colors)
    norm = mcolors.BoundaryNorm(bounds_cm, cmap.N)

    # Dynamically scale figure height according to the number of materials (IEEE 2-column width: 7.16 inches)
    fig_h = max(5.4, 2.5 + n_mats * 0.17)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, fig_h), sharey=True)
    fig.patch.set_facecolor("white")

    # Left: Method 1
    im1 = ax1.imshow(matrix_sc, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
    ax1.set_title(rf"(a) {m1_full} Limit of Detection $f_{{\min}}$ ($P_{{\mathrm{{CD}}}} \ge 75\%$)",
                  fontsize=7.2, fontweight="bold", pad=7)
    ax1.set_xticks(np.arange(n_cols))
    ax1.set_xticklabels(col_names, fontsize=6.2)
    ax1.set_yticks(np.arange(n_mats))
    ax1.set_yticklabels([m[1] for m in flat_materials], fontsize=5.8)

    # Right: Method 2
    im2 = ax2.imshow(matrix_py, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
    ax2.set_title(rf"(b) {m2_full} Limit of Detection $f_{{\min}}$ ($P_{{\mathrm{{CD}}}} \ge 75\%$)",
                  fontsize=7.2, fontweight="bold", pad=7)
    ax2.set_xticks(np.arange(n_cols))
    ax2.set_xticklabels(col_names, fontsize=6.2)

    # Annotate cell values with luminance-aware contrast
    for ax, mat, mat_rec in [(ax1, matrix_sc, matrix_sc_rec), (ax2, matrix_py, matrix_py_rec)]:
        for r in range(n_mats):
            for c in range(n_cols):
                val = mat[r, c]
                if val <= f_max:
                    txt = f"{val:.2f}"
                    c_idx = norm(val)
                    rgba = cmap(c_idx)
                    lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                    text_color = "black" if lum > 0.52 else "white"
                    font_weight = "bold"
                else:
                    txt = rf"$>{f_max:.2f}$" if f_max < 1.0 else "Undet."
                    text_color = "black"
                    font_weight = "normal"
                ax.text(c, r, txt, ha="center", va="center", fontsize=5.2, color=text_color, fontweight=font_weight)

        # Draw group division lines
        for b in group_boundaries[:-1]:
            ax.axhline(b, color="black", linewidth=1.0, linestyle="-")

    # Colorbar
    cbar_ax = fig.add_axes([0.22, 0.05, 0.65, 0.025])
    if len(fractions) > 10:
        cbar_ticks = [f for f in fractions if np.isclose(f % 0.10, 0.0) or np.isclose(f, fractions[0])]
    else:
        cbar_ticks = list(fractions)
    cbar_ticks.append(undetected_val)

    cbar = fig.colorbar(im1, cax=cbar_ax, orientation="horizontal", ticks=cbar_ticks)
    tick_labels = []
    for val in cbar_ticks:
        if np.isclose(val, fractions[0]):
            tick_labels.append(f"$f = {val:.2f}$\n(High Sens.)")
        elif val <= f_max:
            tick_labels.append(f"{val:.2f}")
        else:
            tick_labels.append(f"Undet.\n($>{f_max:.2f}$)")
    cbar.ax.set_xticklabels(tick_labels, fontsize=5.2)

    fig.subplots_adjust(left=0.21, right=0.98, top=0.92, bottom=0.15, wspace=0.08)
    fig.suptitle(rf"Material Contrast Sensitivity Matrix: {m1_name} vs. {m2_name} ($f_{{\min}}$ at $P_{{\mathrm{{CD}}}} \ge 75\%$)",
                 fontsize=8.0, fontweight="bold", y=0.98)

    save_base = str(Path(save_path).with_suffix(""))
    plt.savefig(f"{save_base}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{save_base}.pdf", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Figure 4 to: {save_base}.png / .pdf")


# ==============================================================================
# FIGURE: STATISTICALLY SIGNIFICANT PAIRED SENSITIVITY (MCNEMAR'S TEST)
# ==============================================================================
def compute_mcnemar_test(x, y):
    """
    Computes McNemar's test with continuity correction for paired binary classifications.

    Parameters
    ----------
    x : np.ndarray
        Binary array of detections for Model 1 (SC-CCM on-target).
    y : np.ndarray
        Binary array of detections for Model 2 (USGS PyCCD on-target).

    Returns
    -------
    b : int
        Discordant count: detected by SC-CCM, missed by PyCCD.
    c : int
        Discordant count: missed by SC-CCM, detected by PyCCD.
    chi2_stat : float
        McNemar chi-square test statistic with Edwards continuity correction.
    p_val : float
        Two-sided asymptotic p-value from chi-square distribution with 1 DOF.
    """
    from scipy.stats import chi2
    b = int(np.sum((x == 1) & (y == 0)))
    c = int(np.sum((x == 0) & (y == 1)))
    bc = b + c
    if bc == 0:
        return 0, 0, 0.0, 1.0
    if abs(b - c) < 1.0:
        stat = 0.0
    else:
        stat = ((abs(b - c) - 1.0) ** 2) / float(bc)
    p_val = float(chi2.sf(stat, df=1))
    return b, c, stat, p_val


def plot_land_class_mcnemar_paired_significance(data, save_path):
    """
    Plots stratified change detection sensitivity (Method 1 vs. Method 2)
    on a single upper plot across Cultivated, Forest, Wetland, and Developed strata,
    and below it at reduced height, the paired sensitivity differences (Delta P_CD).
    Shaded bands illustrate confidence intervals with coverage CONFIDENCE_INTERVAL for P_CD.
    Discrete markers are plotted strictly at fractions where the paired difference
    is statistically significant (McNemar test with Edwards continuity correction, p < MCNEMAR_ALPHA).
    Excludes the overall combined series in both plots.
    """
    from scipy.stats import norm
    m1 = data.get("method_1")
    m2 = data.get("method_2")
    sc_tab_pixel = m1["table"] if m1 else data["sc_table"]
    py_tab = m2["table"] if m2 else data["py_table"]
    m1_lbl = m1.get("short_name", "SC-CCM 3D") if m1 else "SC-CCM 3D"
    m2_lbl = m2.get("short_name", "SC-CCM maxVol") if m2 else "SC-CCM maxVol"
    m1_ls = m1.get("linestyle", "-") if m1 else "-"
    m2_ls = m2.get("linestyle", "--") if m2 else "--"
    m1_marker = m1.get("marker", "o") if m1 else "o"
    m2_marker = m2.get("marker", "s") if m2 else "s"

    fractions = data["fractions"]
    patch_ids = data["patch_ids"]
    patch_names = data["patch_names"]
    f_pcts = fractions * 100.0

    fig = plt.figure(figsize=(3.5, 3.8))
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(2, 1, height_ratios=[2.0, 1.2], hspace=0.22)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)

    handles_m1 = []
    handles_m2 = []

    ci_pct = int(round(CONFIDENCE_INTERVAL * 100))
    z_ci = float(norm.ppf(1.0 - (1.0 - CONFIDENCE_INTERVAL) / 2.0))
    all_deltas = []

    for pid in patch_ids:
        color = STRATA_PALETTE.get(pid, "#333333")
        display_name = STRATA_DISPLAY.get(pid, pid.title())

        sc_pts, sc_ci_l, sc_ci_h = [], [], []
        py_pts, py_ci_l, py_ci_h = [], [], []
        deltas, delta_ci_l, delta_ci_h = [], [], []
        sig_f, sig_sc, sig_py = [], [], []
        sig_f_d, sig_d = [], []

        for f in fractions:
            m = (sc_tab_pixel["patch_id"] == pid.encode()) & np.isclose(sc_tab_pixel["mixture_fraction"], f)
            x = sc_tab_pixel["break_detected"][m].astype(int)
            y = py_tab["break_detected"][m].astype(int)

            n = len(x)
            k_sc = int(np.sum(x))
            k_py = int(np.sum(y))

            p_sc, sl, sh = compute_binomial_ci(k_sc, n, confidence=CONFIDENCE_INTERVAL, method="wilson")
            p_py, pl, ph = compute_binomial_ci(k_py, n, confidence=CONFIDENCE_INTERVAL, method="wilson")

            b, c, stat, pval = compute_mcnemar_test(x, y)
            d = p_sc - p_py
            var_d = (float(b + c) - float((b - c) ** 2) / float(n)) / float(n ** 2) if n > 0 else 0.0
            se_d = np.sqrt(max(0.0, var_d)) * 100.0
            dl = d - z_ci * se_d
            dh = d + z_ci * se_d

            sc_pts.append(p_sc)
            sc_ci_l.append(sl)
            sc_ci_h.append(sh)
            py_pts.append(p_py)
            py_ci_l.append(pl)
            py_ci_h.append(ph)
            deltas.append(d)
            all_deltas.append(d)
            delta_ci_l.append(dl)
            delta_ci_h.append(dh)

            if pval < MCNEMAR_ALPHA:
                sig_f.append(f * 100.0)
                sig_sc.append(p_sc)
                sig_py.append(p_py)
                sig_f_d.append(f * 100.0)
                sig_d.append(d)

        sc_pts = np.array(sc_pts)
        sc_ci_l = np.array(sc_ci_l)
        sc_ci_h = np.array(sc_ci_h)
        py_pts = np.array(py_pts)
        py_ci_l = np.array(py_ci_l)
        py_ci_h = np.array(py_ci_h)
        deltas = np.array(deltas)
        delta_ci_l = np.array(delta_ci_l)
        delta_ci_h = np.array(delta_ci_h)

        # Top plot: Sensitivity curves
        h_sc, = ax1.plot(f_pcts, sc_pts, color=color, linestyle=m1_ls, linewidth=1.4, zorder=4,
                         label=f"{display_name} ({m1_lbl})")
        h_py, = ax1.plot(f_pcts, py_pts, color=color, linestyle=m2_ls, linewidth=1.2, zorder=3,
                         label=f"{display_name} ({m2_lbl})")
        handles_m1.append(h_sc)
        handles_m2.append(h_py)

        # Discrete markers plotted strictly where pval < MCNEMAR_ALPHA
        if len(sig_f) > 0:
            ax1.scatter(sig_f, sig_sc, color=color, marker=m1_marker, s=18, edgecolors="black", linewidths=0.5, zorder=5)
            ax1.scatter(sig_f, sig_py, color=color, marker=m2_marker, s=16, edgecolors="black", linewidths=0.5, zorder=5)

        # Bottom plot: Paired difference curve and confidence band
        ax2.plot(f_pcts, deltas, color=color, linestyle="-", linewidth=1.3, alpha=0.85, zorder=3, label=display_name)
        if len(sig_f_d) > 0:
            ax2.scatter(sig_f_d, sig_d, color=color, s=16, edgecolors="black", linewidths=0.5, zorder=5)

    # Top plot styling
    ax1.set_ylabel(r"$P_{\mathrm{CD, tgt}}$ (\%)", fontsize=8.0)
    ax1.set_xlim(-2.0, max(f_pcts) + 3.0)
    ax1.set_ylim(-3.0, 105.0)
    ax1.set_yticks(np.arange(0, 101, 20))
    ax1.set_yticklabels([f"{y:.0f}\\%" for y in np.arange(0, 101, 20)], fontsize=6.5)
    ax1.grid(True, linestyle="--", alpha=0.5, color=COLOR_GRID)

    # Order legend: Col 1 = Method 1, Col 2 = Method 2, with entries for McNemar statistically significant marks
    h_sig_sc = plt.Line2D([0], [0], marker=m1_marker, color="w", markeredgecolor="black", markerfacecolor=None,
                          markersize=4.0, label=rf"{m1_lbl} Adv. ($\alpha = {MCNEMAR_ALPHA:.2f}$)")
    h_sig_py = plt.Line2D([0], [0], marker=m2_marker, color="w", markeredgecolor="black", markerfacecolor=None,
                          markersize=3.5, label=rf"{m2_lbl} Adv. ($\alpha = {MCNEMAR_ALPHA:.2f}$)")
    all_handles = handles_m1 + [h_sig_sc] + handles_m2 + [h_sig_py]
    all_labels = [h.get_label() for h in all_handles]
    ax1.legend(handles=all_handles, labels=all_labels, loc="lower right", fontsize=4.8, ncol=2, framealpha=0.92,
               borderpad=0.25, labelspacing=0.2, handlelength=1.0, handletextpad=0.3, columnspacing=0.6)

    # Bottom plot styling
    d_min = min(all_deltas) if all_deltas else -30.0
    d_max = max(all_deltas) if all_deltas else 30.0
    y_low = min(-35.0, np.floor((d_min - 5.0) / 10.0) * 10.0)
    y_high = max(30.0, np.ceil((d_max + 5.0) / 10.0) * 10.0)

    ax2.axhline(0.0, color="black", linestyle="-", linewidth=0.8, zorder=2)
    ax2.axhspan(0.0, max(y_high, 35.0), color="#e6f2ff", alpha=0.45, zorder=0)
    ax2.axhspan(min(y_low, -35.0), 0.0, color="#fff0e6", alpha=0.45, zorder=0)
    ax2.text(0.02, y_high * 0.80, rf"{m1_lbl} Advantage", transform=ax2.get_yaxis_transform(), fontsize=5.8, color="#004080", fontweight="bold", ha="left", va="center")
    ax2.text(0.02, y_low * 0.75, rf"{m2_lbl} Advantage", transform=ax2.get_yaxis_transform(), fontsize=5.8, color="#993300", fontweight="bold", ha="left", va="center")

    ax2.set_xlabel(r"Target Mixture Fraction $f$ (\%)", fontsize=8.0)
    ax2.set_ylabel(r"$\Delta P_{\mathrm{CD, tgt}}$ (\%)", fontsize=8.0)
    ax2.set_ylim(y_low, y_high)
    ax2.set_xticks(np.arange(0, int(max(f_pcts)) + 1, 10))
    ax2.set_xticklabels([f"{x:.0f}\\%" for x in np.arange(0, int(max(f_pcts)) + 1, 10)], fontsize=6.5)
    step = 20 if (y_high - y_low) >= 60 else 10
    yticks = np.arange(int(y_low), int(y_high) + 1, step)
    ax2.set_yticks(yticks)
    ax2.set_yticklabels([f"{y:+d}\\%" for y in yticks], fontsize=6.5)
    ax2.grid(True, linestyle="--", alpha=0.5, color=COLOR_GRID)

    # Legend for bottom plot with entry for McNemar significant marks
    h_sig_d = plt.Line2D([0], [0], marker="o", color="w", markeredgecolor="black", markerfacecolor="white",
                         markersize=4.0, label=rf"McNemar $\alpha = {MCNEMAR_ALPHA:.2f}$")
    handles_ax2, labels_ax2 = ax2.get_legend_handles_labels()
    ax2.legend(handles=handles_ax2 + [h_sig_d], labels=labels_ax2 + [h_sig_d.get_label()],
               loc="lower right", bbox_to_anchor=(0.99, 0.04), fontsize=4.8, ncol=3, framealpha=0.92,
               borderpad=0.25, labelspacing=0.2, handlelength=0.9, handletextpad=0.25, columnspacing=0.5)

    plt.subplots_adjust(top=0.97, bottom=0.10, left=0.12, right=0.97)

    save_base = str(Path(save_path).with_suffix(""))
    plt.savefig(f"{save_base}.png", dpi=300, pad_inches=0, bbox_inches="tight")
    plt.savefig(f"{save_base}.pdf", dpi=600, pad_inches=0, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Land Class McNemar Paired Significance Figure to: {save_base}.png / .pdf")


def plot_material_category_mcnemar_paired_significance(data, save_path):
    """
    Plots stratified change detection sensitivity (Method 1 vs. Method 2)
    on a single upper plot across aggregated material classes (Artificial, Vegetation,
    Charred Soil, Turbid Water), and below it at reduced height, the paired sensitivity
    differences (Delta P_CD).
    Discrete markers are plotted strictly at fractions where the paired difference
    is statistically significant (McNemar test with continuity correction, p < 1 - CONFIDENCE_INTERVAL).
    """
    m1 = data.get("method_1")
    m2 = data.get("method_2")
    sc_tab_pixel = m1["table"] if m1 else data["sc_table"]
    py_tab = m2["table"] if m2 else data["py_table"]
    m1_lbl = m1.get("short_name", "SC-CCM 3D") if m1 else "SC-CCM 3D"
    m2_lbl = m2.get("short_name", "SC-CCM maxVol") if m2 else "SC-CCM maxVol"
    m1_ls = m1.get("linestyle", "-") if m1 else "-"
    m2_ls = m2.get("linestyle", "--") if m2 else "--"
    m1_marker = m1.get("marker", "o") if m1 else "o"
    m2_marker = m2.get("marker", "s") if m2 else "s"

    fractions = data["fractions"]
    material_groups = data.get("material_groups", MATERIAL_GROUPS)
    f_pcts = fractions * 100.0

    fig = plt.figure(figsize=(3.5, 3.8))
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(2, 1, height_ratios=[2.0, 1.2], hspace=0.22)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)

    handles_m1 = []
    handles_m2 = []
    all_deltas = []

    for grp in material_groups:
        grp_name = grp["group"]
        color = CATEGORY_PALETTE.get(grp_name, "#333333")
        display_name = grp_name
        mat_pairs = [(m[0].encode(), m[1].encode()) for m in grp["materials"]]

        m_cat = np.zeros(len(sc_tab_pixel), dtype=bool)
        for mid_b, mname_b in mat_pairs:
            m_cat |= ((sc_tab_pixel["material_id"] == mid_b) & (sc_tab_pixel["material_name"] == mname_b))

        sc_pts, py_pts = [], []
        sig_f, sig_sc, sig_py = [], [], []
        deltas, sig_f_d, sig_d = [], [], []

        for f in fractions:
            m = m_cat & np.isclose(sc_tab_pixel["mixture_fraction"], f)
            x = sc_tab_pixel["break_detected"][m].astype(int)
            y = py_tab["break_detected"][m].astype(int)

            p_sc = float(np.mean(x)) * 100.0 if len(x) > 0 else 0.0
            p_py = float(np.mean(y)) * 100.0 if len(y) > 0 else 0.0
            sc_pts.append(p_sc)
            py_pts.append(p_py)
            d = p_sc - p_py
            deltas.append(d)
            all_deltas.append(d)

            b, c, stat, pval = compute_mcnemar_test(x, y)
            if pval < (1 - CONFIDENCE_INTERVAL):
                sig_f.append(f * 100.0)
                sig_sc.append(p_sc)
                sig_py.append(p_py)
                sig_f_d.append(f * 100.0)
                sig_d.append(d)

        # Top plot: Sensitivity curves
        h_sc, = ax1.plot(f_pcts, sc_pts, color=color, linestyle=m1_ls, linewidth=1.4, zorder=3,
                         label=f"{display_name} ({m1_lbl})")
        h_py, = ax1.plot(f_pcts, py_pts, color=color, linestyle=m2_ls, linewidth=1.2, zorder=3,
                         label=f"{display_name} ({m2_lbl})")
        handles_m1.append(h_sc)
        handles_m2.append(h_py)

        # Discrete markers plotted strictly where p < (1 - CONFIDENCE_INTERVAL)
        if len(sig_f) > 0:
            ax1.scatter(sig_f, sig_sc, color=color, marker=m1_marker, s=18, edgecolors="black", linewidths=0.5, zorder=5)
            ax1.scatter(sig_f, sig_py, color=color, marker=m2_marker, s=16, edgecolors="black", linewidths=0.5, zorder=5)

        # Bottom plot: Paired difference curve
        ax2.plot(f_pcts, deltas, color=color, linestyle="-", linewidth=1.3, alpha=0.85, zorder=3, label=display_name)
        if len(sig_f_d) > 0:
            ax2.scatter(sig_f_d, sig_d, color=color, s=16, edgecolors="black", linewidths=0.5, zorder=5)

    # Top plot styling
    ax1.set_ylabel(r"$P_{\mathrm{CD, tgt}}$ (\%)", fontsize=8.0)
    ax1.set_xlim(-2.0, max(f_pcts) + 3.0)
    ax1.set_ylim(-3.0, 105.0)
    ax1.set_yticks(np.arange(0, 101, 20))
    ax1.set_yticklabels([f"{y:.0f}\\%" for y in np.arange(0, 101, 20)], fontsize=6.5)
    ax1.grid(True, linestyle="--", alpha=0.5, color=COLOR_GRID)

    # Order legend: Col 1 = Method 1, Col 2 = Method 2, with entries for McNemar statistically significant marks
    alpha_sig = 1 - CONFIDENCE_INTERVAL
    h_sig_sc = plt.Line2D([0], [0], marker=m1_marker, color="w", markeredgecolor="black", markerfacecolor="white",
                          markersize=4.0, label=rf"{m1_lbl} Adv. ($\alpha = {alpha_sig:.2f}$)")
    h_sig_py = plt.Line2D([0], [0], marker=m2_marker, color="w", markeredgecolor="black", markerfacecolor="white",
                          markersize=3.5, label=rf"{m2_lbl} Adv. ($\alpha = {alpha_sig:.2f}$)")
    all_handles = handles_m1 + [h_sig_sc] + handles_m2 + [h_sig_py]
    all_labels = [h.get_label() for h in all_handles]
    ax1.legend(handles=all_handles, labels=all_labels, loc="lower right", fontsize=4.8, ncol=2, framealpha=0.92,
               borderpad=0.25, labelspacing=0.2, handlelength=1.0, handletextpad=0.3, columnspacing=0.6)

    # Bottom plot styling
    d_min = min(all_deltas) if all_deltas else -30.0
    d_max = max(all_deltas) if all_deltas else 30.0
    y_low = min(-35.0, np.floor((d_min - 5.0) / 10.0) * 10.0)
    y_high = max(30.0, np.ceil((d_max + 5.0) / 10.0) * 10.0)

    ax2.axhline(0.0, color="black", linestyle="-", linewidth=0.8, zorder=2)
    ax2.axhspan(0.0, max(y_high, 35.0), color="#e6f2ff", alpha=0.45, zorder=0)
    ax2.axhspan(min(y_low, -35.0), 0.0, color="#fff0e6", alpha=0.45, zorder=0)
    ax2.text(0.02, y_high * 0.80, rf"{m1_lbl} Advantage ($\alpha = {alpha_sig:.2f}$)", transform=ax2.get_yaxis_transform(), fontsize=5.8, color="#004080", fontweight="bold", ha="left", va="center")
    ax2.text(0.02, y_low * 0.75, rf"{m2_lbl} Advantage ($\alpha = {alpha_sig:.2f}$)", transform=ax2.get_yaxis_transform(), fontsize=5.8, color="#993300", fontweight="bold", ha="left", va="center")

    ax2.set_xlabel(r"Target Mixture Fraction $f$ (\%)", fontsize=8.0)
    ax2.set_ylabel(r"$\Delta P_{\mathrm{CD, tgt}}$ (\%)", fontsize=8.0)
    ax2.set_ylim(y_low, y_high)
    ax2.set_xticks(np.arange(0, int(max(f_pcts)) + 1, 10))
    ax2.set_xticklabels([f"{x:.0f}\\%" for x in np.arange(0, int(max(f_pcts)) + 1, 10)], fontsize=6.5)
    step = 20 if (y_high - y_low) >= 60 else 10
    yticks = np.arange(int(y_low), int(y_high) + 1, step)
    ax2.set_yticks(yticks)
    ax2.set_yticklabels([f"{y:+d}\\%" for y in yticks], fontsize=6.5)
    ax2.grid(True, linestyle="--", alpha=0.5, color=COLOR_GRID)

    # Legend for bottom plot with entry for McNemar significant marks
    h_sig_d = plt.Line2D([0], [0], marker="o", color="w", markeredgecolor="black", markerfacecolor="white",
                         markersize=4.0, label=rf"McNemar $\alpha = {alpha_sig:.2f}$")
    handles_ax2, labels_ax2 = ax2.get_legend_handles_labels()
    ax2.legend(handles=handles_ax2 + [h_sig_d], labels=labels_ax2 + [h_sig_d.get_label()],
               loc="lower right", bbox_to_anchor=(0.99, 0.04), fontsize=4.8, ncol=3, framealpha=0.92,
               borderpad=0.25, labelspacing=0.2, handlelength=0.9, handletextpad=0.25, columnspacing=0.5)

    plt.subplots_adjust(top=0.97, bottom=0.10, left=0.12, right=0.97)

    save_base = str(Path(save_path).with_suffix(""))
    plt.savefig(f"{save_base}.png", dpi=300, pad_inches=0, bbox_inches="tight")
    plt.savefig(f"{save_base}.pdf", dpi=600, pad_inches=0, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Material Category McNemar Paired Significance Figure to: {save_base}.png / .pdf")


# Maintain backward compatibility alias
plot_material_class_mcnemar_paired_significance = plot_material_category_mcnemar_paired_significance


# ==============================================================================
# FIGURE: STRATA-MATERIAL CLASS SENSITIVITY (2x2 GRID, MCNEMAR SIGNIFICANCE ONLY)
# ==============================================================================
def plot_strata_material_classes_mcnemar_2x2(data, save_path):
    """
    Plots subpixel change detection sensitivity curves (P_CD,tgt vs. mixture fraction f)
    for each material class across the four land-cover strata in a 2x2 small-multiples grid.
    Compares Method 1 (1x1 on-target center pixel) against Method 2.
    Discrete markers are plotted strictly at mixture fractions where the paired difference
    is statistically significant according to the McNemar test with Edwards continuity correction
    (p < MCNEMAR_ALPHA).
    """
    m1 = data.get("method_1")
    m2 = data.get("method_2")
    sc_tab_pixel = m1["table"] if m1 else data["sc_table"]
    py_tab = m2["table"] if m2 else data["py_table"]
    m1_lbl = m1.get("short_name", "SC-CCM 3D") if m1 else "SC-CCM 3D"
    m2_lbl = m2.get("short_name", "SC-CCM maxVol") if m2 else "SC-CCM maxVol"
    m1_ls = m1.get("linestyle", "-") if m1 else "-"
    m2_ls = m2.get("linestyle", "--") if m2 else "--"
    m1_marker = m1.get("marker", "o") if m1 else "o"
    m2_marker = m2.get("marker", "s") if m2 else "s"

    fractions = data["fractions"]
    patch_ids = data["patch_ids"]
    material_groups = data.get("material_groups", MATERIAL_GROUPS)
    f_pcts = fractions * 100.0

    fig, axes = plt.subplots(2, 2, figsize=(3.5, 3.65), sharex=True, sharey=True)
    fig.patch.set_facecolor("white")

    panel_tags = ["(a) Cultivated", "(b) Forest", "(c) Wetland", "(d) Developed Land"]
    axes_flat = axes.flatten()

    for idx, (ax, pid) in enumerate(zip(axes_flat, patch_ids)):
        for grp in material_groups:
            grp_name = grp["group"]
            color = CATEGORY_PALETTE.get(grp_name, "#333333")
            mat_pairs = [(m[0].encode(), m[1].encode()) for m in grp["materials"]]

            m_cat = np.zeros(len(sc_tab_pixel), dtype=bool)
            for mid_b, mname_b in mat_pairs:
                m_cat |= ((sc_tab_pixel["material_id"] == mid_b) & (sc_tab_pixel["material_name"] == mname_b))
            m_stratum = m_cat & (sc_tab_pixel["patch_id"] == pid.encode())

            sc_pts, py_pts = [], []
            sig_f, sig_sc, sig_py = [], [], []

            for f in fractions:
                m = m_stratum & np.isclose(sc_tab_pixel["mixture_fraction"], f)
                x = sc_tab_pixel["break_detected"][m].astype(int)
                y = py_tab["break_detected"][m].astype(int)

                p_sc = float(np.mean(x)) * 100.0 if len(x) > 0 else 0.0
                p_py = float(np.mean(y)) * 100.0 if len(y) > 0 else 0.0
                sc_pts.append(p_sc)
                py_pts.append(p_py)

                b, c, stat, pval = compute_mcnemar_test(x, y)
                if pval < MCNEMAR_ALPHA:
                    sig_f.append(f * 100.0)
                    sig_sc.append(p_sc)
                    sig_py.append(p_py)

            ax.plot(f_pcts, sc_pts, color=color, linestyle=m1_ls, linewidth=1.1, zorder=3)
            ax.plot(f_pcts, py_pts, color=color, linestyle=m2_ls, linewidth=0.9, zorder=3)

            if len(sig_f) > 0:
                ax.scatter(sig_f, sig_sc, color=color, marker=m1_marker, s=13, edgecolors="black", linewidths=0.5, zorder=5)
                ax.scatter(sig_f, sig_py, color=color, marker=m2_marker, s=11, edgecolors="black", linewidths=0.5, zorder=5)

        ax.text(0.04, 0.91, panel_tags[idx], transform=ax.transAxes, fontsize=6.2, fontweight="bold",
                zorder=5, bbox=dict(boxstyle="square,pad=0.15", facecolor="white", edgecolor="none", alpha=0.85))
        max_f_pct = float(max(f_pcts)) if len(f_pcts) > 0 else 100.0
        x_tick_step = 10 if max_f_pct <= 55 else 20
        x_ticks = np.arange(0, int(max_f_pct) + 1, x_tick_step)
        ax.set_xlim(-2, max_f_pct + 3)
        ax.set_ylim(-2, 115)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([f"{x:.0f}\\%" for x in x_ticks], fontsize=5.5)
        ax.set_yticks(np.arange(0, 101, 20))
        ax.set_yticklabels([f"{y:.0f}\\%" for y in np.arange(0, 101, 20)], fontsize=5.5)
        ax.grid(True, linestyle="--", alpha=0.45, color=COLOR_GRID, zorder=0)

    # Left column y-labels
    axes[0, 0].set_ylabel(r"$P_{\mathrm{CD, tgt}}$ (\%)", fontsize=7.2)
    axes[1, 0].set_ylabel(r"$P_{\mathrm{CD, tgt}}$ (\%)", fontsize=7.2)

    # Bottom row x-labels
    axes[1, 0].set_xlabel(r"Target Abundance $f$ (\%)", fontsize=7.0)
    axes[1, 1].set_xlabel(r"Target Abundance $f$ (\%)", fontsize=7.0)

    # Build structured legend handles: 4 columns x 2 rows
    alpha_str = rf"{MCNEMAR_ALPHA:.2f}"
    mat_handles = [
        mlines.Line2D([0], [0], color=CATEGORY_PALETTE.get(g["group"], "#333333"),
                      linestyle="-", linewidth=1.2, label=g["group"])
        for g in material_groups
    ]
    style_handles = [
        mlines.Line2D([0], [0], color="#222222", linestyle=m1_ls, linewidth=1.1, label=m1_lbl),
        mlines.Line2D([0], [0], color="#222222", linestyle=m2_ls, linewidth=1.0, label=m2_lbl),
        mlines.Line2D([0], [0], marker=m1_marker, color="w", markerfacecolor="white", markeredgecolor="black",
                      markeredgewidth=0.5, markersize=3.4, label=rf"{m1_lbl} Adv. ($p < {alpha_str}$)"),
        mlines.Line2D([0], [0], marker=m2_marker, color="w", markerfacecolor="white", markeredgecolor="black",
                      markeredgewidth=0.5, markersize=3.2, label=rf"{m2_lbl} Adv. ($p < {alpha_str}$)")
    ]

    n_cols = max(len(mat_handles), len(style_handles))
    interleaved_handles = []
    for c_idx in range(n_cols):
        if c_idx < len(mat_handles):
            interleaved_handles.append(mat_handles[c_idx])
        if c_idx < len(style_handles):
            interleaved_handles.append(style_handles[c_idx])

    fig.legend(handles=interleaved_handles, loc="lower center", bbox_to_anchor=(0.50, 0.03),
               fontsize=4.4, ncol=n_cols, framealpha=0.92)

    plt.subplots_adjust(top=0.97, bottom=0.17, left=0.13, right=0.98, hspace=0.05, wspace=0.09)

    save_base = str(Path(save_path).with_suffix(""))
    plt.savefig(f"{save_base}.png", dpi=300, pad_inches=0, bbox_inches="tight")
    plt.savefig(f"{save_base}.pdf", dpi=600, pad_inches=0, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Strata Material Classes McNemar (2x2) Figure to: {save_base}.png / .pdf")


# Aliases
plot_material_classes_strata_2x2 = plot_strata_material_classes_mcnemar_2x2
plot_stratified_material_mcnemar_2x2 = plot_strata_material_classes_mcnemar_2x2


# ==============================================================================
# FIGURES: SC-CCM SPATIAL SUPPORT EVALUATION (TARGET 1x1 vs. NEIGHBORHOOD 3x3)
# ==============================================================================
def plot_sccm_spatial_support_facets(data, save_path):
    """
    Plots SC-CCM change detection sensitivity comparing 3x3 spatial cluster (P_CD,nbh)
    vs. 1x1 on-target center pixel (P_CD,tgt) across the four land-cover strata
    as a compact 2x2 small-multiples grid formatted for single-column IEEE manuscript placement.
    """
    sc_pixel = data["sc_table"]
    sc_clust = data.get("sc_table_cluster", sc_pixel)
    fractions = data["fractions"]
    patch_ids = data["patch_ids"]
    f_pcts = fractions * 100.0

    fig, axes = plt.subplots(2, 2, figsize=(3.5, 3.4), sharex=True, sharey=True)
    fig.patch.set_facecolor("white")

    panel_tags = ["(a) Cultivated", "(b) Forest", "(c) Wetland", "(d) Developed Land"]
    axes_flat = axes.flatten()

    for idx, (ax, pid) in enumerate(zip(axes_flat, patch_ids)):
        color = STRATA_PALETTE.get(pid, "#333333")

        p_tgt, p_nbh = [], []
        for f in fractions:
            m = (sc_pixel["patch_id"] == pid.encode()) & np.isclose(sc_pixel["mixture_fraction"], f)
            p_tgt.append(float(np.mean(sc_pixel["break_detected"][m])) * 100.0)
            p_nbh.append(float(np.mean(sc_clust["break_detected"][m])) * 100.0)

        p_tgt = np.array(p_tgt)
        p_nbh = np.array(p_nbh)

        # Plot curves
        ax.plot(f_pcts, p_nbh, color=color, linestyle="-", linewidth=1.3,
                markersize=2.8, zorder=3)
        ax.plot(f_pcts, p_tgt, color=color, linestyle="--", linewidth=1.1,
                markerfacecolor="white", markeredgecolor=color, markeredgewidth=0.8,
                markersize=2.8, zorder=3)

        # Shaded spatial dispersion gain
        ax.fill_between(f_pcts, p_tgt, p_nbh, color=color, alpha=0.20, zorder=2)


        # Internal panel label to eliminate external margin
        ax.text(0.04, 0.91, panel_tags[idx], transform=ax.transAxes, fontsize=6.2, fontweight="bold",
                zorder=5, bbox=dict(boxstyle="square,pad=0.15", facecolor="white", edgecolor="none", alpha=0.85))

        max_f_pct = float(max(f_pcts)) if len(f_pcts) > 0 else 100.0
        x_tick_step = 10 if max_f_pct <= 55 else 20
        x_ticks = np.arange(0, int(max_f_pct) + 1, x_tick_step)
        ax.set_xlim(-2, max_f_pct + 3)
        ax.set_ylim(-2, 105)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([f"{x:.0f}\\%" for x in x_ticks], fontsize=5.5)
        ax.set_yticks(np.arange(0, 101, 20))
        ax.set_yticklabels([f"{y:.0f}\\%" for y in np.arange(0, 101, 20)], fontsize=5.5)
        ax.grid(True, linestyle="--", alpha=0.45, color=COLOR_GRID, zorder=0)

    # Left column y-labels
    axes[0, 0].set_ylabel(r"$P_{\mathrm{CD}}$ (\%)", fontsize=7.2)
    axes[1, 0].set_ylabel(r"$P_{\mathrm{CD}}$ (\%)", fontsize=7.2)

    # Bottom row x-labels
    axes[1, 0].set_xlabel(r"Target Abundance $f$ (\%)", fontsize=7.0)
    axes[1, 1].set_xlabel(r"Target Abundance $f$ (\%)", fontsize=7.0)

    # Global neutral legend handles (neutral dark charcoal so it unambiguously applies to all 4 strata)
    neutral_handles = [
        plt.Line2D([0], [0], color="black", linestyle="-", linewidth=1.2,
                   label=r"$3\times 3$ Neighborhood ($P_{\mathrm{CD, nbh}}$)"),
        plt.Line2D([0], [0], color="black", linestyle="--", linewidth=1.1, dashes=(3.0, 1.5),
                   label=r"$1\times 1$ Center Target ($P_{\mathrm{CD, tgt}}$)"),
        mpatches.Patch(facecolor="#666666", alpha=0.25, edgecolor="none",
                       label=r"Spatial Gain ($\Delta P_{\mathrm{CD}}$)")
    ]

    fig.legend(handles=neutral_handles, loc="lower center", bbox_to_anchor=(0.51, 0.008),
               ncol=3, fontsize=5.0, columnspacing=0.55, handlelength=2.4, handletextpad=0.22,
               borderpad=0.2, framealpha=0.92)

    plt.subplots_adjust(top=0.97, bottom=0.15, left=0.13, right=0.98, hspace=0.14, wspace=0.11)

    save_base = str(Path(save_path).with_suffix(""))
    plt.savefig(f"{save_base}.png", dpi=300, pad_inches=0, bbox_inches="tight")
    plt.savefig(f"{save_base}.pdf", dpi=600, pad_inches=0, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved SC-CCM Spatial Support Facets (2x2) Figure to: {save_base}.png / .pdf")


# Maintain alias
plot_sccm_spatial_support_2x2 = plot_sccm_spatial_support_facets


def plot_sccm_spatial_support_stacked(data, save_path):
    """
    Plots SC-CCM change detection sensitivity comparing 3x3 spatial cluster (P_CD,nbh)
    vs. 1x1 on-target center pixel (P_CD,tgt) on a two-panel stacked figure:
      Upper panel (a): Absolute sensitivity curves with Wilson score confidence intervals.
      Lower panel (b): Spatial gain (Delta P_spatial = P_CD,nbh - P_CD,tgt) across target mixture fraction f
    """
    sc_pixel = data["sc_table"]
    sc_clust = data.get("sc_table_cluster", sc_pixel)
    fractions = data["fractions"]
    patch_ids = data["patch_ids"]
    f_pcts = fractions * 100.0

    m1 = data.get("method_1")
    m1_lbl = m1.get("short_name", "SC-CCM") if m1 else "SC-CCM"

    fig = plt.figure(figsize=(7.16, 4.9))
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(2, 1, height_ratios=[2.2, 1.15], hspace=0.16)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)

    handles_nbh = []
    handles_tgt = []
    all_deltas = []

    n_strata = len(patch_ids)
    x_offsets = np.linspace(-0.6, 0.6, 2 * n_strata)

    for idx, pid in enumerate(patch_ids):
        color = STRATA_PALETTE.get(pid, "#333333")
        display_name = STRATA_DISPLAY.get(pid, pid.title())

        nbh_pts, nbh_l, nbh_h = [], [], []
        tgt_pts, tgt_l, tgt_h = [], [], []
        deltas = []

        for f in fractions:
            m = (sc_pixel["patch_id"] == pid.encode()) & np.isclose(sc_pixel["mixture_fraction"], f)

            # Target 1x1
            k_t = int(np.sum(sc_pixel["break_detected"][m]))
            n_t = len(sc_pixel["break_detected"][m])
            p_t, tl, th = compute_binomial_ci(k_t, n_t, confidence=CONFIDENCE_INTERVAL, method="wilson")
            tgt_pts.append(p_t)
            tgt_l.append(tl)
            tgt_h.append(th)

            # Neighborhood 3x3
            k_n = int(np.sum(sc_clust["break_detected"][m]))
            n_n = len(sc_clust["break_detected"][m])
            p_n, nl, nh = compute_binomial_ci(k_n, n_n, confidence=CONFIDENCE_INTERVAL, method="wilson")
            nbh_pts.append(p_n)
            nbh_l.append(nl)
            nbh_h.append(nh)

            deltas.append(p_n - p_t)
            all_deltas.append(p_n - p_t)

        nbh_pts = np.array(nbh_pts)
        tgt_pts = np.array(tgt_pts)
        deltas = np.array(deltas)

        x_nbh = f_pcts + x_offsets[2 * idx]
        x_tgt = f_pcts + x_offsets[2 * idx + 1]

        # Error bars for upper panel
        ax1.errorbar(x_nbh, nbh_pts,
                     yerr=[np.maximum(0.0, nbh_pts - np.array(nbh_l)), np.maximum(0.0, np.array(nbh_h) - nbh_pts)],
                     fmt="none", ecolor=color, elinewidth=1.0, capsize=2.0, capthick=0.8, alpha=0.75, zorder=3)
        ax1.errorbar(x_tgt, tgt_pts,
                     yerr=[np.maximum(0.0, tgt_pts - np.array(tgt_l)), np.maximum(0.0, np.array(tgt_h) - tgt_pts)],
                     fmt="none", ecolor=color, elinewidth=1.0, capsize=2.0, capthick=0.8, alpha=0.75, zorder=2)

        # Sensitivity curves
        h_n, = ax1.plot(x_nbh, nbh_pts, color=color, linestyle="-", linewidth=1.8,
                        markersize=4.5, zorder=5, label=rf"{display_name} ({m1_lbl} $3\times 3$ Cluster)")
        h_t, = ax1.plot(x_tgt, tgt_pts, color=color, linestyle="--", linewidth=1.5,
                        markerfacecolor="white", markeredgecolor=color, markeredgewidth=1.1,
                        markersize=4.0, zorder=4, label=rf"{display_name} ({m1_lbl} $1\times 1$ Center)")
        handles_nbh.append(h_n)
        handles_tgt.append(h_t)

        # Bottom plot: spatial gain delta curve
        ax2.plot(f_pcts, deltas, color=color, linestyle="-", linewidth=1.8, marker="o", markersize=4.0,
                 alpha=0.9, zorder=3, label=display_name)

    max_f_pct = float(max(f_pcts)) if len(f_pcts) > 0 else 100.0
    x_ticks = np.arange(0, int(max_f_pct) + 1, 10)

    # Upper plot styling
    ax1.text(0.015, 0.92, "(a)", transform=ax1.transAxes, fontsize=9.5, fontweight="bold")
    ax1.set_ylabel(r"Detection Sensitivity $P_{\mathrm{CD}}$ (\%)", fontsize=9.0)
    ax1.set_xlim(-2.0, max_f_pct + 3.0)
    ax1.set_ylim(-3.0, 105.0)
    ax1.set_yticks(np.arange(0, 101, 20))
    ax1.set_yticklabels([f"{y:.0f}\\%" for y in np.arange(0, 101, 20)], fontsize=7.5)
    ax1.tick_params(axis="x", labelbottom=False)
    ax1.grid(True, linestyle="--", alpha=0.5, color=COLOR_GRID)

    all_handles = handles_nbh + handles_tgt
    all_labels = [h.get_label() for h in all_handles]
    ax1.legend(handles=all_handles, labels=all_labels, loc="lower right", fontsize=6.5, ncol=2, framealpha=0.92)

    # Lower plot styling
    d_max = max(all_deltas) if all_deltas else 25.0
    d_min = min(all_deltas) if all_deltas else 0.0
    y2_top = max(25.0, float(np.ceil((d_max + 14.0) / 5.0) * 5.0))
    y2_bot = min(0.0, float(np.floor((d_min - 2.0) / 5.0) * 5.0))
    y2_step = 5 if (y2_top - y2_bot) <= 35 else 10
    y2_ticks = np.arange(int(y2_bot), int(y2_top) + 1, y2_step)

    ax2.text(0.015, 0.85, "(b)", transform=ax2.transAxes, fontsize=9.5, fontweight="bold")
    ax2.axhline(0.0, color="black", linestyle="-", linewidth=1.0, zorder=2)

    ax2.set_xlabel(r"Target Mixture Fraction $f$ (\%)", fontsize=9.0)
    ax2.set_ylabel(r"$\Delta P_{\mathrm{spatial}}$ (\%)", fontsize=9.0)
    ax2.set_xlim(-2.0, max_f_pct + 3.0)
    ax2.set_ylim(y2_bot, y2_top)
    ax2.set_xticks(x_ticks)
    ax2.set_xticklabels([f"{x:.0f}\\%" for x in x_ticks], fontsize=7.5)
    ax2.set_yticks(y2_ticks)
    ax2.set_yticklabels([f"{y:.0f}\\%" for y in y2_ticks], fontsize=7.5)
    ax2.grid(True, linestyle="--", alpha=0.5, color=COLOR_GRID)

    ax2.legend(loc="upper right", fontsize=6.5, ncol=4, framealpha=0.92)

    plt.subplots_adjust(top=0.97, bottom=0.08, left=0.09, right=0.97)

    save_base = str(Path(save_path).with_suffix(""))
    plt.savefig(f"{save_base}.png", dpi=300, pad_inches=0, bbox_inches="tight")
    plt.savefig(f"{save_base}.pdf", dpi=600, pad_inches=0, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved SC-CCM Spatial Support Stacked Figure to: {save_base}.png / .pdf")


# ==============================================================================
# REPORTING: SUBPIXEL CHANGE DETECTION SENSITIVITY TABLES
# ==============================================================================
def print_and_export_subpixel_sensitivity_tables(data, out_dir, alpha=(1-CONFIDENCE_INTERVAL), fractions=(0.2, 0.4, 0.6, 0.8)):
    """
    Computes and formats Subpixel Change Detection Sensitivity P_CD(f) at reference abundance fractions:
      e.g. f in [0.20, 0.40, 0.60, 0.80] (20%, 40%, 60%, 80% target abundance)
    Evaluates on-target Method 1 (P_CD,tgt) vs. Method 2 on identical center pixels.
    Identifies statistically significant differences via paired McNemar tests with Edwards continuity correction.
    Bolds percentages where differences are statistically significant (p < alpha).
    Exports structured tables for publication in IEEE JSTARS without multi-row confidence interval brackets.
    """
    m1 = data.get("method_1")
    m2 = data.get("method_2")
    m1_name = m1.get("short_name", "SC-CCM 3D") if m1 else "SC-CCM 3D"
    m2_name = m2.get("short_name", "SC-CCM maxVol") if m2 else "SC-CCM maxVol"
    sc_tab_pixel = m1["table"] if m1 else data["sc_table"]
    sc_tab_clust = m1.get("cluster_table", sc_tab_pixel) if m1 else data.get("sc_table_cluster", sc_tab_pixel)
    py_tab = m2["table"] if m2 else data["py_table"]
    patch_ids = data["patch_ids"]
    patch_names = data["patch_names"]
    avail_fractions = list(data.get("fractions", []))
    if fractions is None:
        fractions = [0.2, 0.4, 0.6, 0.8]
    else:
        fractions = list(fractions)
    if len(avail_fractions) > 0:
        matched = [f for f in fractions if any(np.isclose(f, af) for af in avail_fractions)]
        fractions = matched if len(matched) > 0 else avail_fractions

    def compute_reference_sensitivities(sc_c_sub, sc_p_sub, py_sub):
        res = {}
        for f in fractions:
            sc_c_f = sc_c_sub[np.isclose(sc_c_sub["mixture_fraction"], f)]
            sc_p_f = sc_p_sub[np.isclose(sc_p_sub["mixture_fraction"], f)]
            py_f = py_sub[np.isclose(py_sub["mixture_fraction"], f)]

            xc = sc_c_f["break_detected"].astype(int)
            xp = sc_p_f["break_detected"].astype(int)
            yp = py_f["break_detected"].astype(int)

            n_c = len(xc)
            k_c = int(np.sum(xc)) if n_c > 0 else 0
            p_c, c_l, c_h = compute_binomial_ci(k_c, n_c, confidence=CONFIDENCE_INTERVAL, method="wilson")

            n_p = len(xp)
            k_p = int(np.sum(xp)) if n_p > 0 else 0
            p_p, p_l, p_h = compute_binomial_ci(k_p, n_p, confidence=CONFIDENCE_INTERVAL, method="wilson")

            n_py = len(yp)
            k_py = int(np.sum(yp)) if n_py > 0 else 0
            p_py, py_l, py_h = compute_binomial_ci(k_py, n_py, confidence=CONFIDENCE_INTERVAL, method="wilson")

            # Paired McNemar tests with Edwards continuity correction against Method 2
            _, _, _, pval_c_py = compute_mcnemar_test(xc, yp)
            _, _, _, pval_p_py = compute_mcnemar_test(xp, yp)

            bold_c = bool((pval_c_py < alpha) and (p_c > p_py))
            bold_p = bool((pval_p_py < alpha) and (p_p > p_py))
            bold_py = bool((pval_p_py < alpha) and (p_py > p_p))

            diff = p_p - p_py
            res[f] = {
                "c": (p_c, c_l, c_h, bold_c),
                "p": (p_p, p_l, p_h, bold_p),
                "py": (p_py, py_l, py_h, bold_py),
                "diff": diff,
                "pval_c": pval_c_py,
                "pval_p": pval_p_py,
            }
        return res

    print("\n" + "=" * 125)
    print(f"SUBPIXEL CHANGE DETECTION SENSITIVITY REPORTING: {m1_name} vs. {m2_name}")
    print(f"Confidence Intervals: {int(CONFIDENCE_INTERVAL * 100)}% Wilson Score Interval (stored in res dict)")
    print(f"Statistical Significance: Paired McNemar test (alpha = {alpha:.2f}, Edwards continuity correction)")
    print("Reference Target Abundance Fractions:")
    for f in fractions:
        print(f"  f = {f:.2f} ({int(round(f * 100))}% target abundance)")
    print("=" * 125)

    def _fmt_term(item):
        p = item[0]
        sig = "*" if (len(item) > 3 and item[3]) else " "
        return f"{p:5.1f}%{sig}"

    # 1. Land-Cover Strata
    print("\n[TABLE 1: Land Cover Class SUBPIXEL SENSITIVITY]")
    col_width = 22
    header_cols = [f"{'P_CD,tgt(f=' + f'{f:.2f}' + ')':<{col_width}}" for f in fractions]
    header = f"{'Land-Cover Class':<18} | " + " | ".join(header_cols)
    subhead_cols = [f"{m1_name:<10} {m2_name:<10}" for _ in fractions]
    subhead = f"{'':<18} | " + " | ".join(subhead_cols)
    print(header)
    print(subhead)
    print("-" * len(subhead))

    strata_labels = STRATA_DISPLAY

    strata_rows = []
    for pid in patch_ids:
        label = strata_labels.get(pid, pid.title())
        sc_c_p = sc_tab_clust[sc_tab_clust["patch_id"] == pid.encode()]
        sc_p_p = sc_tab_pixel[sc_tab_pixel["patch_id"] == pid.encode()]
        py_p = py_tab[py_tab["patch_id"] == pid.encode()]
        st = compute_reference_sensitivities(sc_c_p, sc_p_p, py_p)
        strata_rows.append((label, st))
        row_str = f"{label:<18} | " + "  | ".join([f"{_fmt_term(st[f]['p'])}  {_fmt_term(st[f]['py'])}" for f in fractions])
        print(row_str)

    st_all = compute_reference_sensitivities(sc_tab_clust, sc_tab_pixel, py_tab)
    print("-" * len(subhead))
    first_f = fractions[0] if len(fractions) > 0 else 0.2
    grand_n = len(sc_tab_pixel[np.isclose(sc_tab_pixel["mixture_fraction"], first_f)]) if np.any(np.isclose(sc_tab_pixel["mixture_fraction"], first_f)) else len(sc_tab_pixel) // max(len(data["fractions"]), 1)
    overall_str = f"{'Overall (N=' + str(grand_n) + ')':<18} | " + "  | ".join([f"{_fmt_term(st_all[f]['p'])}  {_fmt_term(st_all[f]['py'])}" for f in fractions])
    print(overall_str)

    # 2. Material Categories
    print("\n[TABLE 2: TARGET MATERIAL CATEGORIES SUBPIXEL SENSITIVITY]")
    header_mat = f"{'Thematic Category':<18} | " + " | ".join(header_cols)
    print(header_mat)
    print(subhead)
    print("-" * len(subhead))

    material_groups = data.get("material_groups", MATERIAL_GROUPS)

    cat_rows = []
    for grp in material_groups:
        gname = grp["group"]
        mat_pairs = [(m[0].encode(), m[1].encode()) for m in grp["materials"]]
        sc_c_mask = np.zeros(len(sc_tab_clust), dtype=bool)
        sc_p_mask = np.zeros(len(sc_tab_pixel), dtype=bool)
        py_mask = np.zeros(len(py_tab), dtype=bool)
        for mid_b, mname_b in mat_pairs:
            sc_c_mask |= ((sc_tab_clust["material_id"] == mid_b) & (sc_tab_clust["material_name"] == mname_b))
            sc_p_mask |= ((sc_tab_pixel["material_id"] == mid_b) & (sc_tab_pixel["material_name"] == mname_b))
            py_mask |= ((py_tab["material_id"] == mid_b) & (py_tab["material_name"] == mname_b))

        sc_c_m = sc_tab_clust[sc_c_mask]
        sc_p_m = sc_tab_pixel[sc_p_mask]
        py_m = py_tab[py_mask]
        st = compute_reference_sensitivities(sc_c_m, sc_p_m, py_m)
        cat_rows.append((gname, st))
        row_str = f"{gname:<18} | " + "  | ".join([f"{_fmt_term(st[f]['p'])}  {_fmt_term(st[f]['py'])}" for f in fractions])
        print(row_str)

    # 3. Export LaTeX Table
    export_filenames = [
        "table_subpixel_change_detection_sensitivity.tex",
        "table_milestone_subpixel_recall.tex",
    ]

    def _format_cell(val_tuple, is_bold=None):
        p = val_tuple[0]
        bold = val_tuple[3] if (len(val_tuple) > 3 and is_bold is None) else (is_bold if is_bold is not None else False)
        if bold:
            return f"\\textbf{{{p:.1f}\\%}}"
        else:
            return f"{p:.1f}\\%"

    col_spec = "l " + " ".join(["cc"] * len(fractions))
    multicol_headers = " & ".join([f"\\multicolumn{{2}}{{c}}{{$P_{{\\mathrm{{CD, tgt}}}}(f = {f:.2f})$}}" for f in fractions])
    cline_strs = " ".join([f"\\cline{{{2*i+2}-{2*i+3}}}" for i in range(len(fractions))])
    method_headers = " & ".join([f"{m1_name} & {m2_name}"] * len(fractions))
    empty_cells = " & ".join([""] * (2 * len(fractions)))

    latex_content = [
        "\\begin{table*}[!t]\n",
        "\\centering\n",
        f"\\caption{{Subpixel Change Detection Sensitivity ($P_{{\\mathrm{{CD, tgt}}}}$) at Selected Abundance Fractions Across Land Cover Class and Material Categories for {m1_name} vs. {m2_name}. Bold values indicate statistically significant differences ($\\alpha = {alpha:.2f}$).}}\n",
        "\\label{tab_subpixel_sensitivity}\n",
        "\\setlength{\\tabcolsep}{6pt}\n",
        "\\renewcommand{\\arraystretch}{1.2}\n",
        f"\\begin{{tabular}}{{{col_spec}}}\n",
        "\\hline\\hline\n",
        f"& {multicol_headers} \\\\\n",
        f"{cline_strs}\n",
        f"Class Aggregated & {method_headers} \\\\\n",
        "\\hline\n",
        f"\\textit{{Land Cover Class}} & {empty_cells} \\\\\n" if empty_cells else "\\textit{Land Cover Class} \\\\\n",
    ]

    for gname, st in cat_rows:
        gname_tex = gname.replace("&", r"\&")
        cells = " & ".join([f"{_format_cell(st[f]['p'])} & {_format_cell(st[f]['py'])}" for f in fractions])
        row_str = f"\\quad {gname_tex} & {cells} \\\\\n"
        latex_content.append(row_str)
    
    latex_content.append("\\hline\n")
    latex_content.append(f"\\textit{{Target Material Class}} & {empty_cells} \\\\\n" if empty_cells else "\\textit{Target Material Class} \\\\\n")

    for pname, st in strata_rows:
        cells = " & ".join([f"{_format_cell(st[f]['p'])} & {_format_cell(st[f]['py'])}" for f in fractions])
        row_str = f"\\quad {pname} & {cells} \\\\\n"
        latex_content.append(row_str)

    overall_cells = " & ".join([f"{_format_cell(st_all[f]['p'])} & {_format_cell(st_all[f]['py'])}" for f in fractions])
    overall_row = f"\\quad \\textbf{{Overall Combined ($N={grand_n}$)}} & {overall_cells} \\\\\n"
    latex_content.append("\\hline\n")
    latex_content.append(overall_row)

    latex_content.append("\\hline\\hline\n")
    latex_content.append("\\end{tabular}\n")
    latex_content.append("\\end{table*}\n")

    full_tex = "".join(latex_content)
    for fname in export_filenames:
        target_path = os.path.join(out_dir, fname)
        with open(target_path, "w", encoding="utf-8") as f:
            f.write(full_tex)
        print(f"Saved LaTeX Subpixel Change Detection Sensitivity Table to: {target_path}")

# Maintain backward compatibility alias
print_and_export_milestone_tables = print_and_export_subpixel_sensitivity_tables


def print_and_export_target_materials_table(data, out_dir):
    """
    Prints the target materials and their thematic categories to the terminal console
    and exports publication-quality LaTeX tables (.tex) to the media directory.
    Exports:
      - table_target_materials.tex (double-column description IEEE table* format)
      - table_target_materials_and_categories.tex (alias for convenient reference)
      - table_target_materials_wide.tex (alias for convenient reference)
    """
    material_groups = data.get("material_groups", MATERIAL_GROUPS)

    # 1. Print formatted ASCII table to terminal console (omitting raw identifier)
    print("\n" + "=" * 80)
    print("TARGET ENDMEMBER MATERIALS (USGS splib07a)")
    print("=" * 80)
    print(f"{'Thematic Category':<22} | {'USGS splib07 Description':<50}")
    print("-" * 80)
    total_materials = 0
    for grp in material_groups:
        cat_name = grp["group"]
        for idx, (m_id, m_name) in enumerate(grp["materials"]):
            cat_display = cat_name if idx == 0 else ""
            print(f"{cat_display:<22} | {m_name:<50}")
            total_materials += 1
        print("-" * 80)
    print(f"Total Target Materials: {total_materials} across {len(material_groups)} thematic categories.")
    print("=" * 80)

    # 2. Build Single IEEE Column Width Table with 2 Internal Description Columns
    latex_2col_desc = [
        "% IEEE JSTARS Table: Target Materials and Thematic Categories (1 IEEE Column Width, 2 Columns Internally)\n",
        "\\begin{table}[!t]\n",
        "\\centering\n",
        "\\caption{Endmember Materials Selected from the USGS Spectral Library (splib07a) for Subpixel Implanted Target Benchmarking.}\n",
        "\\label{tab_target_materials}\n",
        "\\setlength{\\tabcolsep}{2.5pt}\n",
        "\\renewcommand{\\arraystretch}{1.1}\n",
        "\\footnotesize\n",
        "\\begin{tabular}{ll}\n",
        "\\hline\\hline\n",
    ]

    for grp in material_groups:
        cat_name = grp["group"]
        mats = grp["materials"]
        n_mats = len(mats)
        cat_header = f"{cat_name} Materials ($n={n_mats}$)" if cat_name == "Artificial" else f"{cat_name} ($n={n_mats}$)"
        latex_2col_desc.append(f"\\multicolumn{{2}}{{l}}{{\\textit{{{cat_header}}}}} \\\\\n")
        latex_2col_desc.append("\\hline\n")

        # Pair materials into two columns
        for idx in range(0, n_mats, 2):
            m1_name = mats[idx][1].replace("_", r"\_")
            m2_name = mats[idx + 1][1].replace("_", r"\_") if (idx + 1 < n_mats) else ""
            latex_2col_desc.append(f"{m1_name} & {m2_name} \\\\\n")
        latex_2col_desc.append("\\hline\\hline\n")

    latex_2col_desc.append("\\end{tabular}\n")
    latex_2col_desc.append("\\end{table}\n")

    # Write files
    desc_tex_str = "".join(latex_2col_desc)

    primary_path = os.path.join(out_dir, "table_target_materials.tex")
    with open(primary_path, "w", encoding="utf-8") as f:
        f.write(desc_tex_str)
    print(f"Saved LaTeX Target Materials Table (Double-Column Description) to: {primary_path}")

    alias_path = os.path.join(out_dir, "table_target_materials_and_categories.tex")
    with open(alias_path, "w", encoding="utf-8") as f:
        f.write(desc_tex_str)
    print(f"Saved LaTeX Target Materials Table Alias to: {alias_path}")

    wide_path = os.path.join(out_dir, "table_target_materials_wide.tex")
    with open(wide_path, "w", encoding="utf-8") as f:
        f.write(desc_tex_str)
    print(f"Saved LaTeX Target Materials Table Wide to: {wide_path}")


def print_and_export_multiple_breaks_summary(data, out_dir):
    """
    Summarizes single vs. multiple post-implant break segment behavior for Method 1 and Method 2,
    printing quantitative metrics to the terminal and exporting a LaTeX table.
    """
    m1 = data.get("method_1")
    m2 = data.get("method_2")
    m1_name = m1.get("short_name", "SC-CCM 3D") if m1 else "SC-CCM 3D"
    m2_name = m2.get("short_name", "SC-CCM maxVol") if m2 else "SC-CCM maxVol"
    m1_full = m1.get("full_name", m1_name) if m1 else "SC-CCM 3D"
    m2_full = m2.get("full_name", m2_name) if m2 else "SC-CCM maxVol"

    sc_table = m1["table"] if m1 else data["sc_table"]
    py_table = m2["table"] if m2 else data["py_table"]
    sc_segments = m1.get("segments") if m1 else data.get("sc_segments")
    py_segments = m2.get("segments") if m2 else data.get("py_segments")

    patch_ids = data["patch_ids"]
    patch_names = data["patch_names"]

    has_multi_col = ("has_multiple_post_breaks" in sc_table.dtype.names)
    if not has_multi_col:
        print("\nNote: Multi-break columns not present in results table. Skipping multi-break summary.")
        return

    print("\n" + "=" * 115)
    print("TEMPORAL BREAK SEGMENTS AND POST-IMPLANT MULTIPLICITY ANALYSIS")
    print("=" * 115)
    print(f"{'Stratum / Class':<22} | {'Algorithm':<12} | {'P_CD (%)':<10} | {'1 Break (%)':<12} | {'2+ Breaks (%)':<14} | {'Med Lat (d)':<12} | {'Med Conf Lat (d)'}")
    print("-" * 115)

    summary_rows = []

    def _summarize_slice(sc_sub, py_sub, label):
        n_tot = len(sc_sub)
        # Method 1
        sc_det = np.sum(sc_sub["break_detected"])
        sc_det_pct = (sc_det / n_tot) * 100.0 if n_tot > 0 else 0.0
        sc_single = np.sum(sc_sub["break_detected"] & (sc_sub["post_onset_breaks"] == 1))
        sc_single_pct = (sc_single / n_tot) * 100.0 if n_tot > 0 else 0.0
        sc_multi = np.sum(sc_sub["has_multiple_post_breaks"])
        sc_multi_pct = (sc_multi / n_tot) * 100.0 if n_tot > 0 else 0.0
        sc_lats = sc_sub["latency_days"][sc_sub["break_detected"]]
        sc_med_lat = float(np.median(sc_lats)) if len(sc_lats) > 0 else np.nan
        sc_clats = sc_sub["confirmation_latency_days"][sc_sub["break_detected"]]
        sc_med_clat = float(np.median(sc_clats)) if len(sc_clats) > 0 else np.nan

        # Method 2
        py_det = np.sum(py_sub["break_detected"])
        py_det_pct = (py_det / n_tot) * 100.0 if n_tot > 0 else 0.0
        py_single = np.sum(py_sub["break_detected"] & (py_sub["post_onset_breaks"] == 1))
        py_single_pct = (py_single / n_tot) * 100.0 if n_tot > 0 else 0.0
        py_multi = np.sum(py_sub["has_multiple_post_breaks"])
        py_multi_pct = (py_multi / n_tot) * 100.0 if n_tot > 0 else 0.0
        py_lats = py_sub["latency_days"][py_sub["break_detected"]]
        py_med_lat = float(np.median(py_lats)) if len(py_lats) > 0 else np.nan
        py_clats = py_sub["confirmation_latency_days"][py_sub["break_detected"]]
        py_med_clat = float(np.median(py_clats)) if len(py_clats) > 0 else np.nan

        print(f"{label:<22} | {m1_name:<12} | {sc_det_pct:6.1f}%    | {sc_single_pct:6.1f}%      | {sc_multi_pct:6.1f}%        | {sc_med_lat:6.1f} d      | {sc_med_clat:6.1f} d")
        print(f"{'':<22} | {m2_name:<12} | {py_det_pct:6.1f}%    | {py_single_pct:6.1f}%      | {py_multi_pct:6.1f}%        | {py_med_lat:6.1f} d      | {py_med_clat:6.1f} d")
        print("-" * 115)

        return {
            "label": label,
            "sc_det": sc_det_pct, "sc_single": sc_single_pct, "sc_multi": sc_multi_pct, "sc_lat": sc_med_lat, "sc_clat": sc_med_clat,
            "py_det": py_det_pct, "py_single": py_single_pct, "py_multi": py_multi_pct, "py_lat": py_med_lat, "py_clat": py_med_clat,
        }

    for p_id in patch_ids:
        disp_name = STRATA_DISPLAY.get(p_id, p_id)
        sc_sub = sc_table[sc_table["patch_id"] == p_id.encode()]
        py_sub = py_table[py_table["patch_id"] == p_id.encode()]
        summary_rows.append(_summarize_slice(sc_sub, py_sub, disp_name))

    # Overall Aggregate
    all_summary = _summarize_slice(sc_table, py_table, "Overall Combined")
    summary_rows.append(all_summary)

    # Segment model statistics if available
    if sc_segments is not None and len(sc_segments) > 0:
        print(f"\nRecorded Segment Models in HDF5:")
        print(f"  {m1_name} Segments: {len(sc_segments)} total models (mean {len(sc_segments)/len(sc_table):.2f} segments/instance)")
        if py_segments is not None and len(py_segments) > 0:
            print(f"  {m2_name} Segments: {len(py_segments)} total models (mean {len(py_segments)/len(py_table):.2f} segments/instance)")

    # Build LaTeX Table
    tex_lines = [
        "% IEEE JSTARS Table: Temporal Break Multiplicity and Latency\n",
        "\\begin{table*}[!t]\n",
        "\\centering\n",
        f"\\caption{{Post-Implantation Break Multiplicity and Latency Comparison Across Land-Cover Strata ({m1_name} vs. {m2_name}).}}\n",
        "\\label{tab_break_multiplicity}\n",
        "\\setlength{\\tabcolsep}{5pt}\n",
        "\\renewcommand{\\arraystretch}{1.15}\n",
        "\\footnotesize\n",
        "\\begin{tabular}{l cccc cccc}\n",
        "\\hline\\hline\n",
        f"& \\multicolumn{{4}}{{c}}{{\\textbf{{{m1_name}}}}} & \\multicolumn{{4}}{{c}}{{\\textbf{{{m2_name}}}}} \\\\\n",
        "\\cline{2-5} \\cline{6-9}\n",
        "Land-Cover Stratum & $P_{\\mathrm{CD}}$ (\\%) & 1 Break (\\%) & $\\ge 2$ Breaks (\\%) & Latency (d) & $P_{\\mathrm{CD}}$ (\\%) & 1 Break (\\%) & $\\ge 2$ Breaks (\\%) & Latency (d) \\\\\n",
        "\\hline\n",
    ]

    for row in summary_rows:
        lbl = row["label"]
        prefix = "\\textbf{" if "Overall" in lbl else ""
        suffix = "}" if "Overall" in lbl else ""
        if "Overall" in lbl:
            tex_lines.append("\\hline\n")
        sc_lat_str = f"{row['sc_lat']:.1f}" if not np.isnan(row['sc_lat']) else "--"
        py_lat_str = f"{row['py_lat']:.1f}" if not np.isnan(row['py_lat']) else "--"
        tex_lines.append(
            f"{prefix}{lbl}{suffix} & "
            f"{row['sc_det']:.1f} & {row['sc_single']:.1f} & {row['sc_multi']:.1f} & {sc_lat_str} & "
            f"{row['py_det']:.1f} & {row['py_single']:.1f} & {row['py_multi']:.1f} & {py_lat_str} \\\\\n"
        )

    tex_lines.append("\\hline\\hline\n")
    tex_lines.append("\\end{tabular}\n")
    tex_lines.append("\\end{table*}\n")

    out_tex_path = os.path.join(out_dir, "table_break_multiplicity.tex")
    with open(out_tex_path, "w", encoding="utf-8") as f:
        f.write("".join(tex_lines))
    print(f"Saved LaTeX Break Multiplicity Table to: {out_tex_path}")


# ==============================================================================
# FIGURE: COMPARISON OF SC-CCM 3D VS. SC-CCM maxVol ACROSS STRATA
# ==============================================================================
def plot_figure_comparison_recall_curves(data_base, data_3d, save_path, m1_desc=None, m2_desc=None):
    """
    Plots subpixel change detection sensitivity curves comparing:
      1. SC-CCM 3D (3D Equivalent Volume)
      2. SC-CCM maxVol (Baseline Sliding Volume)
    across all 4 land-cover strata in a 2x2 small-multiples grid layout.
    """
    if m1_desc is not None:
        sc_3d = m1_desc["table"]
        m1_lbl = m1_desc.get("short_name", "SC-CCM 3D")
        c_3d = m1_desc.get("color", "#0072B2")
        m1_marker = m1_desc.get("marker", "o")
        m1_ls = m1_desc.get("linestyle", "-")
    else:
        sc_3d = data_3d["sc_table"]
        m1_lbl = "SC-CCM 3D"
        c_3d = "#0072B2"
        m1_marker = "o"
        m1_ls = "-"

    if m2_desc is not None:
        sc_base = m2_desc["table"]
        m2_lbl = m2_desc.get("short_name", "SC-CCM maxVol")
        c_base = m2_desc.get("color", "#E69F00")
        m2_marker = m2_desc.get("marker", "^")
        m2_ls = m2_desc.get("linestyle", "--")
    else:
        sc_base = data_base["sc_table"]
        m2_lbl = "SC-CCM maxVol"
        c_base = "#E69F00"
        m2_marker = "^"
        m2_ls = "--"

    fractions = data_3d["fractions"] if data_3d is not None else data_base["fractions"]
    patch_ids = data_base["patch_ids"] if data_base is not None else data_3d["patch_ids"]
    f_pcts = fractions * 100.0

    fig, axes = plt.subplots(2, 2, figsize=(7.16, 5.5), sharex=True, sharey=True)
    fig.patch.set_facecolor("white")
    axes_flat = axes.flatten()

    # Dodging offsets for error bars
    x_off = np.array([-0.3, 0.3])
    panel_letters = ["(a)", "(b)", "(c)", "(d)"]

    for idx, pid in enumerate(patch_ids):
        ax = axes_flat[idx]
        display_name = STRATA_DISPLAY.get(pid, pid.title())

        r_3d, r_3d_l, r_3d_h = [], [], []
        r_base, r_base_l, r_base_h = [], [], []

        for f in fractions:
            s_3d_f = sc_3d[(np.isclose(sc_3d["mixture_fraction"], f)) & (sc_3d["patch_id"] == pid.encode())]
            s_base_f = sc_base[(np.isclose(sc_base["mixture_fraction"], f)) & (sc_base["patch_id"] == pid.encode())]

            # Method 1
            n_3d = len(s_3d_f)
            k_3d = int(np.sum(s_3d_f["break_detected"])) if n_3d > 0 else 0
            p_3d, l_3d, h_3d = compute_binomial_ci(k_3d, n_3d, confidence=CONFIDENCE_INTERVAL, method="wilson")
            r_3d.append(p_3d); r_3d_l.append(l_3d); r_3d_h.append(h_3d)

            # Method 2
            n_b = len(s_base_f)
            k_b = int(np.sum(s_base_f["break_detected"])) if n_b > 0 else 0
            p_b, l_b, h_b = compute_binomial_ci(k_b, n_b, confidence=CONFIDENCE_INTERVAL, method="wilson")
            r_base.append(p_b); r_base_l.append(l_b); r_base_h.append(h_b)

        r_3d = np.array(r_3d); r_3d_l = np.array(r_3d_l); r_3d_h = np.array(r_3d_h)
        r_base = np.array(r_base); r_base_l = np.array(r_base_l); r_base_h = np.array(r_base_h)

        # Error bars
        ax.errorbar(f_pcts + x_off[0], r_3d, yerr=[np.maximum(0.0, r_3d - r_3d_l), np.maximum(0.0, r_3d_h - r_3d)],
                    fmt="none", ecolor=c_3d, elinewidth=0.7, capsize=1.5, capthick=0.6, alpha=0.75, zorder=3)
        ax.errorbar(f_pcts + x_off[1], r_base, yerr=[np.maximum(0.0, r_base - r_base_l), np.maximum(0.0, r_base_h - r_base)],
                    fmt="none", ecolor=c_base, elinewidth=0.7, capsize=1.5, capthick=0.6, alpha=0.75, zorder=3)

        # Plot lines
        ax.plot(f_pcts + x_off[0], r_3d, color=c_3d, marker=m1_marker, markersize=3.2, linewidth=1.2,
                linestyle=m1_ls, label=m1_lbl, zorder=6)
        ax.plot(f_pcts + x_off[1], r_base, color=c_base, marker=m2_marker, markersize=3.2, linewidth=1.1,
                linestyle=m2_ls, label=m2_lbl, zorder=5)

        ax.set_title(f"{panel_letters[idx]} {display_name}", fontsize=8.0, fontweight="bold", pad=5)
        ax.grid(True, linestyle="--", alpha=0.45, color=COLOR_GRID)
        ax.set_xlim(-2.0, max(f_pcts) + 3.0)
        ax.set_ylim(-3.0, 105.0)
        ax.set_xticks(np.arange(0, int(max(f_pcts)) + 1, 10))
        ax.set_xticklabels([f"{x:.0f}\\%" for x in np.arange(0, int(max(f_pcts)) + 1, 10)], fontsize=6.2)
        ax.set_yticks(np.arange(0, 101, 20))
        ax.set_yticklabels([f"{y:.0f}\\%" for y in np.arange(0, 101, 20)], fontsize=6.2)

        if idx in [2, 3]:
            ax.set_xlabel(r"Target Mixture Fraction $f$ (\%)", fontsize=7.5)
        if idx in [0, 2]:
            ax.set_ylabel(r"Detection Sensitivity $P_{\mathrm{CD}}$ (\%)", fontsize=7.5)

        if idx == 0:
            ax.legend(loc="lower right", fontsize=5.5, framealpha=0.92, handlelength=1.1,
                      handletextpad=0.25, borderpad=0.25, labelspacing=0.25)

    plt.subplots_adjust(top=0.94, bottom=0.08, left=0.08, right=0.98, hspace=0.25, wspace=0.15)
    save_base = str(Path(save_path).with_suffix(""))
    plt.savefig(f"{save_base}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{save_base}.pdf", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Land-Cover Strata Comparison Figure to: {save_base}.png / .pdf")


# ==============================================================================
# FIGURE: COMPARISON OF SC-CCM 3D VS. SC-CCM maxVol ACROSS MATERIAL CATEGORIES
# ==============================================================================
def plot_figure_comparison_material_categories(data_base, data_3d, save_path, m1_desc=None, m2_desc=None):
    """
    Plots subpixel change detection sensitivity curves comparing:
      1. SC-CCM 3D (3D Equivalent Volume)
      2. SC-CCM maxVol (Baseline Sliding Volume)
    across the 4 material categories in a 2x2 small-multiples grid layout.
    """
    if m1_desc is not None:
        sc_3d = m1_desc["table"]
        m1_lbl = m1_desc.get("short_name", "SC-CCM 3D")
        c_3d = m1_desc.get("color", "#0072B2")
        m1_marker = m1_desc.get("marker", "o")
        m1_ls = m1_desc.get("linestyle", "-")
    else:
        sc_3d = data_3d["sc_table"]
        m1_lbl = "SC-CCM 3D"
        c_3d = "#0072B2"
        m1_marker = "o"
        m1_ls = "-"

    if m2_desc is not None:
        sc_base = m2_desc["table"]
        m2_lbl = m2_desc.get("short_name", "SC-CCM maxVol")
        c_base = m2_desc.get("color", "#E69F00")
        m2_marker = m2_desc.get("marker", "^")
        m2_ls = m2_desc.get("linestyle", "--")
    else:
        sc_base = data_base["sc_table"]
        m2_lbl = "SC-CCM maxVol"
        c_base = "#E69F00"
        m2_marker = "^"
        m2_ls = "--"

    fractions = data_3d["fractions"] if data_3d is not None else data_base["fractions"]
    material_groups = data_3d.get("material_groups", data_base.get("material_groups", MATERIAL_GROUPS))
    f_pcts = fractions * 100.0

    fig, axes = plt.subplots(2, 2, figsize=(7.16, 5.5), sharex=True, sharey=True)
    fig.patch.set_facecolor("white")
    axes_flat = axes.flatten()

    x_off = np.array([-0.3, 0.3])
    panel_letters = ["(a)", "(b)", "(c)", "(d)"]

    for idx, grp in enumerate(material_groups[:4]):
        ax = axes_flat[idx]
        grp_name = grp["group"]
        mat_pairs = [(m[0].encode(), m[1].encode()) for m in grp["materials"]]

        sc_3d_mask = np.zeros(len(sc_3d), dtype=bool)
        sc_base_mask = np.zeros(len(sc_base), dtype=bool)
        for mid_b, mname_b in mat_pairs:
            sc_3d_mask |= ((sc_3d["material_id"] == mid_b) & (sc_3d["material_name"] == mname_b))
            sc_base_mask |= ((sc_base["material_id"] == mid_b) & (sc_base["material_name"] == mname_b))

        s_3d_grp = sc_3d[sc_3d_mask]
        s_base_grp = sc_base[sc_base_mask]

        r_3d, r_3d_l, r_3d_h = [], [], []
        r_base, r_base_l, r_base_h = [], [], []

        for f in fractions:
            s_3d_f = s_3d_grp[np.isclose(s_3d_grp["mixture_fraction"], f)]
            s_base_f = s_base_grp[np.isclose(s_base_grp["mixture_fraction"], f)]

            n_3d = len(s_3d_f)
            k_3d = int(np.sum(s_3d_f["break_detected"])) if n_3d > 0 else 0
            p_3d, l_3d, h_3d = compute_binomial_ci(k_3d, n_3d, confidence=CONFIDENCE_INTERVAL, method="wilson")
            r_3d.append(p_3d); r_3d_l.append(l_3d); r_3d_h.append(h_3d)

            n_b = len(s_base_f)
            k_b = int(np.sum(s_base_f["break_detected"])) if n_b > 0 else 0
            p_b, l_b, h_b = compute_binomial_ci(k_b, n_b, confidence=CONFIDENCE_INTERVAL, method="wilson")
            r_base.append(p_b); r_base_l.append(l_b); r_base_h.append(h_b)

        r_3d = np.array(r_3d); r_3d_l = np.array(r_3d_l); r_3d_h = np.array(r_3d_h)
        r_base = np.array(r_base); r_base_l = np.array(r_base_l); r_base_h = np.array(r_base_h)

        # Error bars
        ax.errorbar(f_pcts + x_off[0], r_3d, yerr=[np.maximum(0.0, r_3d - r_3d_l), np.maximum(0.0, r_3d_h - r_3d)],
                    fmt="none", ecolor=c_3d, elinewidth=0.7, capsize=1.5, capthick=0.6, alpha=0.75, zorder=3)
        ax.errorbar(f_pcts + x_off[1], r_base, yerr=[np.maximum(0.0, r_base - r_base_l), np.maximum(0.0, r_base_h - r_base)],
                    fmt="none", ecolor=c_base, elinewidth=0.7, capsize=1.5, capthick=0.6, alpha=0.75, zorder=3)

        # Plot lines
        ax.plot(f_pcts + x_off[0], r_3d, color=c_3d, marker=m1_marker, markersize=3.2, linewidth=1.2,
                linestyle=m1_ls, label=m1_lbl, zorder=6)
        ax.plot(f_pcts + x_off[1], r_base, color=c_base, marker=m2_marker, markersize=3.2, linewidth=1.1,
                linestyle=m2_ls, label=m2_lbl, zorder=5)

        ax.set_title(f"{panel_letters[idx]} {grp_name}", fontsize=8.0, fontweight="bold", pad=5)
        ax.grid(True, linestyle="--", alpha=0.45, color=COLOR_GRID)
        ax.set_xlim(-2.0, max(f_pcts) + 3.0)
        ax.set_ylim(-3.0, 105.0)
        ax.set_xticks(np.arange(0, int(max(f_pcts)) + 1, 10))
        ax.set_xticklabels([f"{x:.0f}\\%" for x in np.arange(0, int(max(f_pcts)) + 1, 10)], fontsize=6.2)
        ax.set_yticks(np.arange(0, 101, 20))
        ax.set_yticklabels([f"{y:.0f}\\%" for y in np.arange(0, 101, 20)], fontsize=6.2)

        if idx in [2, 3]:
            ax.set_xlabel(r"Target Mixture Fraction $f$ (\%)", fontsize=7.5)
        if idx in [0, 2]:
            ax.set_ylabel(r"Detection Sensitivity $P_{\mathrm{CD}}$ (\%)", fontsize=7.5)

        if idx == 0:
            ax.legend(loc="lower right", fontsize=5.5, framealpha=0.92, handlelength=1.1,
                      handletextpad=0.25, borderpad=0.25, labelspacing=0.25)

    plt.subplots_adjust(top=0.94, bottom=0.08, left=0.08, right=0.98, hspace=0.25, wspace=0.15)
    save_base = str(Path(save_path).with_suffix(""))
    plt.savefig(f"{save_base}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{save_base}.pdf", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Material Category Comparison Figure to: {save_base}.png / .pdf")


# ==============================================================================
# FIGURE: TWO-PANEL COMPOSITE METHOD COMPARISON (SC-CCM 3D vs. SC-CCM maxVol)
# ==============================================================================
def plot_figure_comparison_composite(data_base, data_3d, save_path, m1_desc=None, m2_desc=None):
    """
    Combines Overall Strata Aggregate and Sensitivity Advantage (Delta P_CD) into
    a space-efficient 2-panel publication figure comparing SC-CCM 3D vs. SC-CCM maxVol.
    """
    if m1_desc is not None:
        sc_3d = m1_desc["table"]
        m1_lbl = m1_desc.get("short_name", "SC-CCM 3D")
        c_3d = m1_desc.get("color", "#0072B2")
        m1_marker = m1_desc.get("marker", "o")
        m1_ls = m1_desc.get("linestyle", "-")
    else:
        sc_3d = data_3d["sc_table"]
        m1_lbl = "SC-CCM 3D"
        c_3d = "#0072B2"
        m1_marker = "o"
        m1_ls = "-"

    if m2_desc is not None:
        sc_base = m2_desc["table"]
        m2_lbl = m2_desc.get("short_name", "SC-CCM maxVol")
        c_base = m2_desc.get("color", "#E69F00")
        m2_marker = m2_desc.get("marker", "^")
        m2_ls = m2_desc.get("linestyle", "--")
    else:
        sc_base = data_base["sc_table"]
        m2_lbl = "SC-CCM maxVol"
        c_base = "#E69F00"
        m2_marker = "^"
        m2_ls = "--"

    fractions = data_3d["fractions"] if data_3d is not None else data_base["fractions"]
    f_pcts = fractions * 100.0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 2.9), sharey=False)
    fig.patch.set_facecolor("white")

    # --------------------------------------------------------------------------
    # Panel (a): Overall Strata Aggregate
    # --------------------------------------------------------------------------
    r_3d_all, r_base_all = [], []
    for f in fractions:
        s3 = sc_3d[np.isclose(sc_3d["mixture_fraction"], f)]
        sb = sc_base[np.isclose(sc_base["mixture_fraction"], f)]

        r_3d_all.append(float(np.mean(s3["break_detected"])) * 100.0 if len(s3) > 0 else 0.0)
        r_base_all.append(float(np.mean(sb["break_detected"])) * 100.0 if len(sb) > 0 else 0.0)

    grand_n = len(sc_3d[np.isclose(sc_3d["mixture_fraction"], fractions[0])]) if len(fractions) > 0 else len(sc_3d)

    ax1.plot(f_pcts, r_3d_all, color=c_3d, marker=m1_marker, markersize=3.5, linewidth=1.4,
             linestyle=m1_ls, label=rf"{m1_lbl} ($N={grand_n}$)", zorder=6)
    ax1.plot(f_pcts, r_base_all, color=c_base, marker=m2_marker, markersize=3.5, linewidth=1.2,
             linestyle=m2_ls, label=rf"{m2_lbl} ($N={grand_n}$)", zorder=5)

    ax1.set_title(r"(a) Grand Mean Sensitivity", fontsize=8.0, fontweight="bold", pad=6)
    ax1.set_xlabel(r"Target Mixture Fraction $f$ (\%)", fontsize=7.5)
    ax1.set_ylabel(r"Detection Sensitivity $P_{\mathrm{CD}}$ (\%)", fontsize=7.5)
    ax1.set_xlim(-2.0, max(f_pcts) + 3.0)
    ax1.set_ylim(-3.0, 105.0)
    ax1.set_xticks(np.arange(0, int(max(f_pcts)) + 1, 10))
    ax1.set_xticklabels([f"{x:.0f}\\%" for x in np.arange(0, int(max(f_pcts)) + 1, 10)], fontsize=6.2)
    ax1.set_yticks(np.arange(0, 101, 20))
    ax1.set_yticklabels([f"{y:.0f}\\%" for y in np.arange(0, 101, 20)], fontsize=6.2)
    ax1.grid(True, linestyle="--", alpha=0.45, color=COLOR_GRID)
    ax1.legend(loc="lower right", fontsize=5.5, framealpha=0.92, handlelength=1.1,
               handletextpad=0.25, borderpad=0.25, labelspacing=0.25)

    # --------------------------------------------------------------------------
    # Panel (b): Delta Gain of SC-CCM 3D over SC-CCM maxVol
    # --------------------------------------------------------------------------
    diff_gain = np.array(r_3d_all) - np.array(r_base_all)

    ax2.plot(f_pcts, diff_gain, color="#0072B2", marker="D", markersize=3.5, linewidth=1.3,
             linestyle="-", label=rf"Gain: {m1_lbl} vs. {m2_lbl} ($\Delta P_{{\mathrm{{CD}}}}$)", zorder=6)
    ax2.axhline(0, color="gray", linestyle=":", linewidth=0.8, alpha=0.7)

    ax2.set_title(r"(b) Sensitivity Difference ($\Delta P_{\mathrm{CD}}$)", fontsize=8.0, fontweight="bold", pad=6)
    ax2.set_xlabel(r"Target Mixture Fraction $f$ (\%)", fontsize=7.5)
    ax2.set_ylabel(r"$\Delta P_{\mathrm{CD}}$ (percentage points)", fontsize=7.5)
    ax2.set_xlim(-2.0, max(f_pcts) + 3.0)
    ax2.set_xticks(np.arange(0, int(max(f_pcts)) + 1, 10))
    ax2.set_xticklabels([f"{x:.0f}\\%" for x in np.arange(0, int(max(f_pcts)) + 1, 10)], fontsize=6.2)
    ax2.grid(True, linestyle="--", alpha=0.45, color=COLOR_GRID)
    ax2.legend(loc="upper left", fontsize=5.5, framealpha=0.92, handlelength=1.1,
               handletextpad=0.25, borderpad=0.25, labelspacing=0.25)

    plt.subplots_adjust(top=0.92, bottom=0.13, left=0.07, right=0.98, wspace=0.18)
    save_base = str(Path(save_path).with_suffix(""))
    plt.savefig(f"{save_base}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{save_base}.pdf", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved 2-Panel Method Comparison Composite Figure to: {save_base}.png / .pdf")


# ==============================================================================
# REPORTING: SC-CCM 3D VS. SC-CCM maxVol COMPARISON TABLE
# ==============================================================================
def print_and_export_method_comparison_table(data_base, data_3d, out_dir, alpha=(1 - CONFIDENCE_INTERVAL), m1_desc=None, m2_desc=None):
    """
    Computes and formats a direct side-by-side performance comparison between
    SC-CCM 3D (volume_map_3d_zscore) and SC-CCM maxVol (sliding_volume_z_score).
    Evaluates sensitivity P_CD across mixture fractions, latency, multi-break rates,
    and paired McNemar significance between the two spectral complexity representations.
    Exports table_baseline_vs_3d_comparison.tex.
    """
    m1_lbl = m1_desc.get("short_name", "SC-CCM 3D") if m1_desc else "SC-CCM 3D"
    m2_lbl = m2_desc.get("short_name", "SC-CCM maxVol") if m2_desc else "SC-CCM maxVol"
    sc_3d = m1_desc["table"] if m1_desc else data_3d["sc_table"]
    sc_base = m2_desc["table"] if m2_desc else data_base["sc_table"]
    fractions = data_3d["fractions"] if data_3d is not None else data_base["fractions"]
    patch_ids = data_base["patch_ids"] if data_base is not None else data_3d["patch_ids"]

    print("\n" + "=" * 130)
    print(f"SPECTRAL VOLUME METHOD COMPARISON: {m1_lbl} vs. {m2_lbl}")
    print("=" * 130)
    print(f"{'Stratum / Category':<22} | {'Representation':<16} | " + " | ".join([f"f={f*100:.0f}%" for f in fractions]) + f" | {'Med Lat (d)':<12} | {'Multi-Break %'}")
    print("-" * 130)

    rows = []

    def _eval_method_sub(sub, label, repr_name):
        n_tot = len(sub)
        recs = []
        for f in fractions:
            f_sub = sub[np.isclose(sub["mixture_fraction"], f)]
            n_f = len(f_sub)
            k_f = int(np.sum(f_sub["break_detected"])) if n_f > 0 else 0
            p_f, _, _ = compute_binomial_ci(k_f, n_f, confidence=CONFIDENCE_INTERVAL, method="wilson")
            recs.append(p_f)
        lats = sub["latency_days"][sub["break_detected"] & ~np.isnan(sub["latency_days"])]
        med_lat = float(np.median(lats)) if len(lats) > 0 else np.nan
        has_multi = "has_multiple_post_breaks" in sub.dtype.names
        multi_pct = float(np.mean(sub["has_multiple_post_breaks"])) * 100.0 if (has_multi and n_tot > 0) else np.nan
        rec_str = " | ".join([f"{r:5.1f}%" for r in recs])
        lat_str = f"{med_lat:5.1f} d" if not np.isnan(med_lat) else "  N/A "
        mul_str = f"{multi_pct:5.1f}%" if not np.isnan(multi_pct) else " N/A "
        print(f"{label:<22} | {repr_name:<16} | {rec_str} | {lat_str:<12} | {mul_str}")
        return {"label": label, "repr": repr_name, "recs": recs, "med_lat": med_lat, "multi_pct": multi_pct}

    # Strata slices
    for pid in patch_ids:
        disp_name = STRATA_DISPLAY.get(pid, pid)
        s3 = sc_3d[sc_3d["patch_id"] == pid.encode()]
        sb = sc_base[sc_base["patch_id"] == pid.encode()]
        r3 = _eval_method_sub(s3, disp_name, m1_lbl)
        rb = _eval_method_sub(sb, "", m2_lbl)
        rows.extend([r3, rb])
        print("-" * 130)

    # Overall Combined
    r3_all = _eval_method_sub(sc_3d, "Overall Aggregate", m1_lbl)
    rb_all = _eval_method_sub(sc_base, "", m2_lbl)
    rows.extend([r3_all, rb_all])
    print("=" * 130)

    # McNemar test between SC-CCM 3D and SC-CCM maxVol overall
    x_3d = sc_3d["break_detected"].astype(int)
    x_base = sc_base["break_detected"].astype(int)
    b, c, stat, pval = compute_mcnemar_test(x_3d, x_base)
    print(f"\nPaired McNemar Significance Test ({m1_lbl} vs. {m2_lbl}, N={len(x_3d)}):")
    print(f"  {m1_lbl} detected, {m2_lbl} missed (b) = {b}")
    print(f"  {m2_lbl} detected, {m1_lbl} missed (c) = {c}")
    print(f"  McNemar Chi2 Statistic       = {stat:.4f}")
    sig_str = "(Statistically Significant)" if pval < alpha else "(Not Significant)"
    print(f"  p-value                      = {pval:.4e} {sig_str}")
    print("=" * 130 + "\n")

    # Export LaTeX table
    tex_cols = "l l " + "c " * len(fractions) + "c c"
    tex_lines = [
        f"% IEEE JSTARS Table: {m1_lbl} vs {m2_lbl} Comparison\n",
        "\\begin{table*}[!t]\n",
        "\\centering\n",
        f"\\caption{{Subpixel Change Detection Sensitivity Comparison: {m1_lbl} vs. {m2_lbl}.}}\n",
        "\\label{tab_baseline_vs_3d_comparison}\n",
        "\\setlength{\\tabcolsep}{5pt}\n",
        "\\renewcommand{\\arraystretch}{1.15}\n",
        "\\footnotesize\n",
        f"\\begin{{tabular}}{{{tex_cols}}}\n",
        "\\hline\\hline\n",
        "& & \\multicolumn{" + str(len(fractions)) + "}{c}{\\textbf{Detection Sensitivity $P_{\\mathrm{CD}}$ (\\%)}} & \\textbf{Latency} & \\textbf{Multi-Break} \\\\\n",
        "\\cline{3-" + str(2 + len(fractions)) + "}\n",
        "Land-Cover Stratum & Method & " + " & ".join([f"$f={f*100:.0f}\\%$" for f in fractions]) + " & (Median Days) & (\\%) \\\\\n",
        "\\hline\n",
    ]

    for row in rows:
        lbl = row["label"]
        prefix = "\\textbf{" if "Overall" in lbl else ""
        suffix = "}" if "Overall" in lbl else ""
        if "Overall" in lbl and row["repr"] == m1_lbl:
            tex_lines.append("\\hline\n")
        rec_tex = " & ".join([f"{r:.1f}" for r in row["recs"]])
        lat_tex = f"{row['med_lat']:.1f}" if not np.isnan(row['med_lat']) else "--"
        mul_tex = f"{row['multi_pct']:.1f}" if not np.isnan(row['multi_pct']) else "--"
        tex_lines.append(f"{prefix}{lbl}{suffix} & {row['repr']} & {rec_tex} & {lat_tex} & {mul_tex} \\\\\n")

    tex_lines.append("\\hline\\hline\n")
    tex_lines.append("\\end{tabular}\n")
    tex_lines.append("\\end{table*}\n")

    out_tex_path = os.path.join(out_dir, "table_baseline_vs_3d_comparison.tex")
    with open(out_tex_path, "w", encoding="utf-8") as f:
        f.write("".join(tex_lines))
    print(f"Saved LaTeX Method Comparison Table to: {out_tex_path}")


# ==============================================================================
# MAIN EXECUTION ENTRY POINT
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Plot stratified benchmark results.")
    parser.add_argument("--file", type=str, default=H5_PATH, help="Path to master HDF5 container")
    parser.add_argument("--media-dir", type=str, default=MEDIA_DIR, help="Output directory for generated figures")
    parser.add_argument("--group", type=str, default=None, help="Specific benchmark group to plot, or None for dual mode")
    parser.add_argument("--method-1", type=str, default=METHOD_1, help=f"Primary comparison method (default: {METHOD_1})")
    parser.add_argument("--method-2", type=str, default=METHOD_2, help=f"Secondary comparison method (default: {METHOD_2})")
    args = parser.parse_args()

    h5_path = args.file
    media_dir = args.media_dir
    os.makedirs(media_dir, exist_ok=True)

    print("=" * 80)
    print("PLOTTING STRATIFIED BENCHMARK RESULTS")
    print(f"Source: {h5_path}")
    print(f"Target Directory: {media_dir}")
    print(f"Comparison: Method 1 = '{args.method_1}', Method 2 = '{args.method_2}'")
    print("=" * 80)

    # Determine whether dual comparison is possible
    with h5py.File(h5_path, "r", locking=False) as h5:
        has_base = "/STRATIFIED_CHIP_BENCHMARK" in h5
        has_3d = "/STRATIFIED_CHIP_BENCHMARK_3D" in h5

    if args.group is not None:
        target_group = args.group
        data_loaded = load_benchmark_data(h5_path, group_name=target_group)
        data_base, data_3d = (data_loaded, None) if "3D" not in target_group else (None, data_loaded)
    elif has_3d and has_base:
        print("Detected both Baseline (/STRATIFIED_CHIP_BENCHMARK) and 3D Volume (/STRATIFIED_CHIP_BENCHMARK_3D).")
        data_base, data_3d = load_dual_benchmark_data(h5_path)
    elif has_3d:
        print("Detected 3D Volume benchmark (/STRATIFIED_CHIP_BENCHMARK_3D).")
        data_3d = load_benchmark_data(h5_path, group_name="/STRATIFIED_CHIP_BENCHMARK_3D")
        data_base = None
    else:
        print("Detected Baseline benchmark (/STRATIFIED_CHIP_BENCHMARK).")
        data_base = load_benchmark_data(h5_path, group_name="/STRATIFIED_CHIP_BENCHMARK")
        data_3d = None

    # Resolve comparison method descriptors
    m1_desc = get_method_descriptor(args.method_1, data_base, data_3d)
    m2_desc = get_method_descriptor(args.method_2, data_base, data_3d)

    # Construct primary active data dictionary
    primary_source = data_3d if data_3d is not None else data_base
    data = dict(primary_source)
    data["method_1"] = m1_desc
    data["method_2"] = m2_desc
    data["sc_table"] = m1_desc["table"]
    data["py_table"] = m2_desc["table"]
    data["sc_table_cluster"] = m1_desc.get("cluster_table", m1_desc["table"])
    data["sc_segments"] = m1_desc.get("segments")
    data["py_segments"] = m2_desc.get("segments")

    print(f"Loaded active benchmark data:")
    print(f"  Method 1: {m1_desc['full_name']} ({len(m1_desc['table'])} records)")
    print(f"  Method 2: {m2_desc['full_name']} ({len(m2_desc['table'])} records)")

    # Output Paths
    fig1_path = os.path.join(media_dir, "Figure_stratified_recall_curves.png")
    fig_mat_path = os.path.join(media_dir, "Figure_stratified_materials_sensitivity.png")
    fig_ind_mat_path = os.path.join(media_dir, "Figure_scccd_individual_materials.png")
    fig_comp_path = os.path.join(media_dir, "Figure_stratified_benchmark_composite.png")
    fig2_path = os.path.join(media_dir, "Figure_stratified_roc_space.png")
    fig3_path = os.path.join(media_dir, "Figure_stratified_latency_distributions.png")
    fig4_path = os.path.join(media_dir, "Figure_stratified_material_heatmap.png")
    fig_mcnemar_path = os.path.join(media_dir, "Figure_mcnemar_paired_significance.png")
    fig_mat_mcnemar_path = os.path.join(media_dir, "Figure_material_category_mcnemar_paired_significance.png")
    fig_strata_mat_mcnemar_path = os.path.join(media_dir, "Figure_strata_material_classes_mcnemar_2x2.png")
    fig_spatial_facets_path = os.path.join(media_dir, "Figure_sccm_spatial_support_facets.png")
    fig_spatial_stacked_path = os.path.join(media_dir, "Figure_sccm_spatial_support_stacked.png")

    # Generate Standard Figures comparing Method 1 vs. Method 2
    print(f"\nGenerating Figure 16: Land-Cover Strata Sensitivity ({m1_desc['short_name']} vs. {m2_desc['short_name']})...")
    plot_figure1_recall_curves(data, fig1_path)

    print(f"\nGenerating Figure 17: Material Category Sensitivity ({m1_desc['short_name']} vs. {m2_desc['short_name']})...")
    plot_figure_material_categories_8curves(data, fig_mat_path)

    n_total_mats = sum(len(g["materials"]) for g in data.get("material_groups", MATERIAL_GROUPS))
    print(f"\nGenerating Figure: Individual Material Performance ({m1_desc['short_name']}, {n_total_mats} Curves)...")
    plot_figure_individual_materials_scccd(data, fig_ind_mat_path)

    print(f"\nGenerating Figure 1c: Two-Panel Composite Sensitivity Figure ({m1_desc['short_name']} vs. {m2_desc['short_name']})...")
    plot_figure_composite_sensitivity(data, fig_comp_path)

    print(f"\nGenerating Figure: Land Class McNemar Paired Statistical Significance ({m1_desc['short_name']} vs. {m2_desc['short_name']})...")
    plot_land_class_mcnemar_paired_significance(data, fig_mcnemar_path)

    print(f"\nGenerating Figure: Material Category McNemar Paired Statistical Significance ({m1_desc['short_name']} vs. {m2_desc['short_name']})...")
    plot_material_category_mcnemar_paired_significance(data, fig_mat_mcnemar_path)

    print(f"\nGenerating Figure: Strata Material Classes McNemar Significance ({m1_desc['short_name']} vs. {m2_desc['short_name']})...")
    plot_strata_material_classes_mcnemar_2x2(data, fig_strata_mat_mcnemar_path)
    shutil.copyfile(os.path.join(media_dir, "Figure_strata_material_classes_mcnemar_2x2.png"), os.path.join(media_dir, "Figure_strata_material_classes_2x2.png"))
    shutil.copyfile(os.path.join(media_dir, "Figure_strata_material_classes_mcnemar_2x2.pdf"), os.path.join(media_dir, "Figure_strata_material_classes_2x2.pdf"))

    print("\nGenerating Figure: SC-CCM Spatial Support Facets (2x2 Small Multiples)...")
    plot_sccm_spatial_support_facets(data, fig_spatial_facets_path)
    shutil.copyfile(os.path.join(media_dir, "Figure_sccm_spatial_support_facets.png"), os.path.join(media_dir, "Figure_sccm_spatial_support_2x2.png"))
    shutil.copyfile(os.path.join(media_dir, "Figure_sccm_spatial_support_facets.pdf"), os.path.join(media_dir, "Figure_sccm_spatial_support_2x2.pdf"))

    print("\nGenerating Figure: SC-CCM Spatial Support Stacked (Sensitivity + Spatial Gain)...")
    plot_sccm_spatial_support_stacked(data, fig_spatial_stacked_path)

    print(f"\nGenerating Figure 2: Operational ROC Space Comparison ({m1_desc['short_name']} vs. {m2_desc['short_name']})...")
    plot_figure2_roc_space(data, fig2_path)

    print(f"\nGenerating Figure 3: Stratified Detection Latency Distributions ({m1_desc['short_name']} vs. {m2_desc['short_name']})...")
    plot_figure3_latency_distributions(data, fig3_path)

    print(f"\nGenerating Figure 4: Material Contrast Sensitivity Matrix ({m1_desc['short_name']} vs. {m2_desc['short_name']})...")
    plot_figure4_material_sensitivity_heatmap(data, fig4_path)

    # Maintain descriptive filename copies for reference
    shutil.copyfile(os.path.join(media_dir, "Figure_mcnemar_paired_significance.pdf"), os.path.join(media_dir, "fig20.pdf"))
    shutil.copyfile(os.path.join(media_dir, "Figure_mcnemar_paired_significance.png"), os.path.join(media_dir, "fig20.png"))
    shutil.copyfile(os.path.join(media_dir, "Figure_sccm_spatial_support_facets.pdf"), os.path.join(media_dir, "fig21.pdf"))
    shutil.copyfile(os.path.join(media_dir, "Figure_sccm_spatial_support_facets.png"), os.path.join(media_dir, "fig21.png"))

    # Export sensitivity & multiplicity tables
    print_and_export_subpixel_sensitivity_tables(data, media_dir)
    print_and_export_target_materials_table(data, media_dir)
    print_and_export_multiple_breaks_summary(data, media_dir)

    # Dual Comparison Figures & Tables if both datasets are available
    if data_base is not None and data_3d is not None:
        print("\n" + "=" * 80)
        print(f"GENERATING COMPARATIVE FIGURES & TABLES: {m1_desc['short_name']} vs. {m2_desc['short_name']}")
        print("=" * 80)

        fig_comp_recall = os.path.join(media_dir, "Figure_comparison_recall_curves.png")
        fig_comp_mat = os.path.join(media_dir, "Figure_comparison_material_categories.png")
        fig_comp_summary = os.path.join(media_dir, "Figure_comparison_composite.png")

        print(f"Generating Comparison Figure: Land-Cover Strata ({m1_desc['short_name']} vs. {m2_desc['short_name']})...")
        plot_figure_comparison_recall_curves(data_base, data_3d, fig_comp_recall, m1_desc=m1_desc, m2_desc=m2_desc)

        print(f"Generating Comparison Figure: Material Categories ({m1_desc['short_name']} vs. {m2_desc['short_name']})...")
        plot_figure_comparison_material_categories(data_base, data_3d, fig_comp_mat, m1_desc=m1_desc, m2_desc=m2_desc)

        print(f"Generating Comparison Figure: 2-Panel Composite ({m1_desc['short_name']} vs. {m2_desc['short_name']})...")
        plot_figure_comparison_composite(data_base, data_3d, fig_comp_summary, m1_desc=m1_desc, m2_desc=m2_desc)

        print("Generating Comparison Table: Method Comparison Summary...")
        print_and_export_method_comparison_table(data_base, data_3d, media_dir, m1_desc=m1_desc, m2_desc=m2_desc)

    print("\n" + "=" * 80)
    print("BENCHMARK FIGURES AND TABLES GENERATED SUCCESSFULLY AT 300 DPI.")
    print("=" * 80)


if __name__ == "__main__":
    main()

