"""
patch_benchmark_vis.py
======================
Interactive Diagnostic Viewer for Land-Cover Stratified Patch Benchmark Results.
Visualizes results produced by generate_patch_benchmark.py:
  - Left panel: 13x13 spatial patch grid (inner 9x9 evaluated region) overlaid with
    dates of change detection, break counts, target injection markers (P1-P4),
    and interactive pixel selection.
  - Right top panel: Spectral Complexity (Z-score) Satellite Image Time Series (SITS)
    with Harmonized SC-CCD harmonic fits, confidence bounds, anomalies, and all detected breaks.
  - Right bottom panel: Landsat NIR Band (0.865 um) Surface Reflectance SITS with
    USGS LCMAP PyCCD harmonic fits, confidence bounds, and all detected breaks.
  - Controls:
      * Mouse click on spatial grid to select any pixel.
      * Keyboard interaction:
          [M] / [m] or [Right] / [Left] : Cycle Target Material
          [F] / [f] or [Up] / [Down] / [+] / [-] : Cycle Pixel Fill Fraction
          [P] / [p] or [1]-[4] : Cycle Land-Cover Stratum (Patch Location)
          [T] / [t] : Cycle through the 4 injected target pixels (P1-P4)
          [I] / [i] : Toggle Probe Injection Mode (Any Selected Pixel vs. Fixed Benchmark Targets)
          [C] / [c] : Toggle Grid Overlay Mode (Change Date vs. Total Break Count)
          [D] / [d] : Print Diagnostic Report of All Segments and Breaks for Selected Pixel
          [S] / [s] : Save current view to PNG
          [H] / [h] : Print help and shortcut guide
          [Q] / [Esc] : Exit viewer
      * Interactive clickable buttons along bottom for easy navigation.
"""


import sys
from pathlib import Path
_script_dir = Path(__file__).resolve().parent
_repo_root = _script_dir.parent
for _p in [
    _repo_root,
    _repo_root / "change_detection" / "SC-CCM",
    _repo_root / "change_detection" / "pyccd",
    _script_dir,
]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import os
import sys
import math
import argparse
import datetime
from datetime import timezone
from pathlib import Path
import h5py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
from matplotlib.widgets import Button
from scipy.stats import chi2

# Ensure single-threaded BLAS/OpenMP execution for workers
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

# Ensure script directory and parent directory are on sys.path
script_dir = Path(__file__).resolve().parent
candidates = [
    script_dir,
    script_dir.parent / "spectral-complexity-public",
    script_dir.parent.parent / "spectral-complexity-public",
    Path("F:/Resilio/IMGS 890 Research/spectral-complexity-public"),
]
for p in candidates:
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Import core modules
try:
    import SC_CCM_main as scccm
except ImportError:
    import harmonized_CCD_main as scccm

import pyccd_landsat_main as pccd
import ccd

# Apply compatibility patch for PyCCD coordinate descent and mode array
pccd.patch_pyccd()

# ==============================================================================
# CONFIGURATION DEFAULTS
# ==============================================================================
DEFAULT_H5_PATH = "C:/satelliteImagery/HLST30/HLST_Rochesterv2.h5"
PATCH_SIZE = 13
BORDER_MARGIN = 2
INJ_OFFSETS = [
    (4, 4),  # P1
    (4, 8),  # P2
    (8, 4),  # P3
    (8, 8),  # P4
]
INJECTION_ONSET_DATE = "2022-06-01"

LAND_COVER_PATCHES = [
    {
        "id": "CROPLAND",
        "class": "Cropland",
        "name": "Rush Open Farmland",
        "r0": 1266,
        "c0": 225,
        "color": "#e6ab02",
    },
    {
        "id": "TREE COVER",
        "class": "Forest",
        "name": "Mendon Ponds Park Forest",
        "r0": 1144,
        "c0": 528,
        "color": "#2ca02c",
    },
    {
        "id": "WETLAND",
        "class": "Wetland",
        "name": "Thousand Acre Swamp (Penfield)",
        "r0": 608,
        "c0": 858,
        "color": "#17becf",
    },
    {
        "id": "DEVELOPED",
        "class": "Developed Land",
        "name": "Rochester Airport",
        "r0": 803,
        "c0": 269,
        "color": "#d62728",
    },
]


# ==============================================================================
# DATA LOADING & PREPARATION
# ==============================================================================
def load_benchmark_container(h5_path=DEFAULT_H5_PATH):
    """
    Loads benchmark dataset hierarchy, preloads metadata, baseline chips,
    and returns a structured dictionary along with an open HDF5 file handle.
    """
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"Master HDF5 container not found at: {h5_path}")

    f = h5py.File(h5_path, "r")
    bm_group = "/STRATIFIED_CHIP_BENCHMARK"
    if bm_group not in f:
        f.close()
        raise KeyError(f"Benchmark group '{bm_group}' not found in container '{h5_path}'. "
                       f"Please run generate_patch_benchmark.py first.")

    bm = f[bm_group]
    meta_grp = bm["METADATA"]

    # Extract metadata attributes
    harm_acq_times = meta_grp.attrs["harm_acq_times"][:]
    mixture_fractions = meta_grp.attrs["mixture_fractions"][:].astype(np.float32)

    injected_offsets = meta_grp.attrs.get("injected_offsets", np.array(INJ_OFFSETS, dtype=np.int32))
    onset_date_str = meta_grp.attrs.get("injection_onset_date", INJECTION_ONSET_DATE)
    if isinstance(onset_date_str, bytes):
        onset_date_str = onset_date_str.decode("utf-8")

    onset_dt = datetime.datetime.strptime(onset_date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    onset_ts = onset_dt.timestamp()

    # Preload target spectra
    spec_l8 = meta_grp["target_spectra_l8"][:]
    spec_s2 = meta_grp["target_spectra_s2"][:] if "target_spectra_s2" in meta_grp else None

    # Load baseline chips
    base_grp = bm["BASELINE_CHIPS"]
    chips = {
        "HARMONIZED_z_score": base_grp["HARMONIZED_z_score"][:],
        "HARMONIZED_common_mask": base_grp["HARMONIZED_common_mask"][:],
        "HLSL30": base_grp["HLSL30"][:],
        "HLSL30_Fmask": base_grp["HLSL30_Fmask"][:],
        "HLSL30_common_mask": base_grp["HLSL30_common_mask"][:],
        "HLSS30": base_grp["HLSS30"][:] if "HLSS30" in base_grp else None,
    }

    # Extract scale factors and fill values
    l30_scale = float(base_grp["HLSL30"].attrs.get("scale_factor", 10000.0))
    l30_fill = int(base_grp["HLSL30"].attrs.get("_FillValue", -9999))

    # Resolve acquisition timestamps for Landsat-8/9 observations
    l30_times = None
    if "/HDFEOS/GRIDS/HLSL30/Data Fields/surface_reflectance" in f:
        l30_times = f["/HDFEOS/GRIDS/HLSL30/Data Fields/surface_reflectance"].attrs.get("acquisition_time")
    if l30_times is None and "HLSL30" in base_grp and "acquisition_time" in base_grp["HLSL30"].attrs:
        l30_times = base_grp["HLSL30"].attrs["acquisition_time"]
    if l30_times is None:
        l30_times = harm_acq_times[:chips["HLSL30"].shape[1]]
    else:
        l30_times = l30_times[:]

    # Resolve spacecraft source tags for harmonized observations
    spacecrafts = None
    if "/HDFEOS/GRIDS/HARMONIZED/Data Fields/sliding_volume_z_score" in f:
        harm_ds = f["/HDFEOS/GRIDS/HARMONIZED/Data Fields/sliding_volume_z_score"]
        if "source_spacecraft" in harm_ds.attrs:
            sc_raw = harm_ds.attrs["source_spacecraft"][:]
            spacecrafts = [s.decode("utf-8") if isinstance(s, bytes) else str(s) for s in sc_raw]
        elif "source_grid" in harm_ds.attrs:
            sg_raw = harm_ds.attrs["source_grid"][:]
            spacecrafts = [s.decode("utf-8") if isinstance(s, bytes) else str(s) for s in sg_raw]
    if spacecrafts is None and "/HDFEOS/GRIDS/HARMONIZED/Data Fields/common_mask" in f:
        mask_ds = f["/HDFEOS/GRIDS/HARMONIZED/Data Fields/common_mask"]
        if "source_spacecraft" in mask_ds.attrs:
            sc_raw = mask_ds.attrs["source_spacecraft"][:]
            spacecrafts = [s.decode("utf-8") if isinstance(s, bytes) else str(s) for s in sc_raw]
    if spacecrafts is None:
        spacecrafts = ["Harmonized"] * len(harm_acq_times)

    # Load results tables and segments
    res_grp = bm["RESULTS"]
    scccd_table = res_grp["scccd_break_table"][:]
    pyccd_table = res_grp["pyccd_break_table"][:]
    scccd_segments = res_grp["scccd_break_segments"][:] if "scccd_break_segments" in res_grp else None
    pyccd_segments = res_grp["pyccd_break_segments"][:] if "pyccd_break_segments" in res_grp else None

    # Load precomputed spatial results if available
    scccd_spatial_dates = res_grp["scccd_change_date"][:] if "scccd_change_date" in res_grp else None
    scccd_spatial_counts = res_grp["scccd_change_count"][:] if "scccd_change_count" in res_grp else None
    pyccd_spatial_dates = res_grp["pyccd_change_date"][:] if "pyccd_change_date" in res_grp else None
    pyccd_spatial_counts = res_grp["pyccd_change_count"][:] if "pyccd_change_count" in res_grp else None

    # Parse distinct target materials from break table
    materials = []
    num_fractions = len(mixture_fractions)
    num_materials = len(spec_l8)

    for m_idx in range(num_materials):
        c_idx = m_idx * num_fractions
        rows = scccd_table[scccd_table["condition_idx"] == c_idx]
        if len(rows) > 0:
            m_id = rows[0]["material_id"].decode("utf-8") if isinstance(rows[0]["material_id"], bytes) else str(rows[0]["material_id"])
            m_name = rows[0]["material_name"].decode("utf-8") if isinstance(rows[0]["material_name"], bytes) else str(rows[0]["material_name"])
        else:
            m_id = f"mat_{m_idx}"
            m_name = f"Material {m_idx + 1}"
        materials.append({
            "idx": m_idx,
            "id": m_id,
            "name": m_name,
            "spec_l8": spec_l8[m_idx],
            "spec_s2": spec_s2[m_idx] if spec_s2 is not None else None,
        })

    # Prepare harmonic design matrices
    frac_years_harm = scccm.extract_fractional_years(harm_acq_times)
    X_full_scccd = scccm.build_harmonic_matrix(frac_years_harm)

    l30_dates = [datetime.datetime.fromtimestamp(t, tz=timezone.utc).date() for t in l30_times]
    ordinal_dates_l30 = np.array([d.toordinal() for d in l30_dates], dtype=np.int64)
    X_full_pyccd = pccd.build_harmonic_basis(ordinal_dates_l30)
    nominal_thermals = np.full(len(l30_times), pccd.NOMINAL_THERMAL_KELVIN_SCALED, dtype=np.int16)

    # Reference to simulated cubes dataset (lazily sliced on demand)
    sim_z_cubes_ds = bm["SIMULATED_CUBES/sliding_volume_z_score"]

    data_bundle = {
        "h5_file": f,
        "sim_z_cubes_ds": sim_z_cubes_ds,
        "chips": chips,
        "harm_acq_times": harm_acq_times,
        "l30_times": l30_times,
        "l30_scale": l30_scale,
        "l30_fill": l30_fill,
        "spacecrafts": spacecrafts,
        "mixture_fractions": mixture_fractions,
        "materials": materials,
        "injected_offsets": [tuple(off) for off in injected_offsets],
        "onset_date_str": onset_date_str,
        "onset_dt": onset_dt,
        "onset_ts": onset_ts,
        "scccd_table": scccd_table,
        "pyccd_table": pyccd_table,
        "scccd_segments": scccd_segments,
        "pyccd_segments": pyccd_segments,
        "scccd_spatial_dates": scccd_spatial_dates,
        "scccd_spatial_counts": scccd_spatial_counts,
        "pyccd_spatial_dates": pyccd_spatial_dates,
        "pyccd_spatial_counts": pyccd_spatial_counts,
        "frac_years_harm": frac_years_harm,
        "X_full_scccd": X_full_scccd,
        "ordinal_dates_l30": ordinal_dates_l30,
        "X_full_pyccd": X_full_pyccd,
        "nominal_thermals": nominal_thermals,
    }
    return data_bundle


def generate_patch_rgb(l30_patch, cm_patch):
    """
    Generates a natural true-color RGB composite (Red=B4, Green=B3, Blue=B2)
    using the clearest available Landsat observation for background rendering.
    """
    valid_counts = np.sum((cm_patch == 0), axis=(1, 2))
    best_t = int(np.argmax(valid_counts))

    r = l30_patch[best_t, 3].astype(np.float32) / 10000.0
    g = l30_patch[best_t, 2].astype(np.float32) / 10000.0
    b = l30_patch[best_t, 1].astype(np.float32) / 10000.0

    rgb = np.stack([r, g, b], axis=-1)
    rgb = np.clip(rgb / 0.25, 0.0, 1.0)
    return rgb, best_t


# ==============================================================================
# ALGORITHMIC TIME SERIES & BREAK EVALUATION
# ==============================================================================
class BenchmarkEngine:
    """
    Encapsulates SC-CCD and USGS PyCCD evaluation with LRU caching
    to guarantee fast interactive responsiveness during navigation.
    """
    def __init__(self, data):
        self.data = data
        self.scccd_cache = {}
        self.pyccd_cache = {}

    def get_scccd_results(self, mat_idx, frac_idx, p_idx):
        """
        Executes Harmonized SC-CCD regression across the inner 9x9 patch region.
        Operates on patch-specific simulated complexity cubes across all conditions (including fraction 0.00).
        """
        cond_idx = mat_idx * len(self.data["mixture_fractions"]) + frac_idx
        cache_key = (cond_idx, p_idx)
        if cache_key in self.scccd_cache:
            return self.scccd_cache[cache_key]

        z_patch = self.data["sim_z_cubes_ds"][cond_idx, p_idx]
        mask_patch = self.data["chips"]["HARMONIZED_common_mask"][p_idx]
        valid_patch = (mask_patch == 0)

        # Only evaluate the inner 9x9 region (rows 2:11, cols 2:11)
        # The outer 2-pixel margin is un-evaluated by SC-CCD
        y_inner = z_patch[:, 2:11, 2:11]
        valid_inner = valid_patch[:, 2:11, 2:11]

        chunk_args = (
            0, 9, 9,
            y_inner, valid_inner,
            self.data["frac_years_harm"],
            self.data["X_full_scccd"],
            self.data["harm_acq_times"],
            scccm.MIN_SAMPLES,
            scccm.CHANGE_PROBABILITY,
            scccm.CONSECUTIVE_ANOMALIES
        )
        res = scccm._process_row_chunk(chunk_args)

        pred_full = np.full(z_patch.shape, np.nan, dtype=np.float32)
        rmse_full = np.full(z_patch.shape, np.nan, dtype=np.float32)
        anom_full = np.zeros(z_patch.shape, dtype=np.uint8)
        date_full = np.full((PATCH_SIZE, PATCH_SIZE), np.nan, dtype=np.float64)
        count_full = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.int32)

        pred_full[:, 2:11, 2:11] = res[2]
        rmse_full[:, 2:11, 2:11] = res[3]
        anom_full[:, 2:11, 2:11] = res[5]
        chunk_segments = res[8] if len(res) > 8 else {}

        # Parse spatial change dates and counts from chunk_segments (identically to generate_patch_benchmark.py)
        onset_ts = self.data["onset_ts"]
        harm_acq_times = self.data["harm_acq_times"]
        for r_in in range(9):
            for c_in in range(9):
                p_segs = chunk_segments.get((r_in, c_in), [])
                p_cnt = len([s for s in p_segs if s.get("has_break", False)])
                count_full[2 + r_in, 2 + c_in] = p_cnt

                eff_ts_p = np.nan
                for s in p_segs:
                    if s.get("has_break", False):
                        b_ts_cand = s.get("break_date", np.nan)
                        c_ts_cand = s.get("confirm_date", np.nan)
                        if b_ts_cand >= onset_ts:
                            eff_ts_p = b_ts_cand
                            break
                        elif c_ts_cand >= onset_ts:
                            run_idx = [k for k in range(s["break_idx"], s["confirm_idx"] + 1)
                                       if valid_inner[k, r_in, c_in] and harm_acq_times[k] >= onset_ts]
                            eff_ts_p = float(harm_acq_times[run_idx[0]]) if run_idx else c_ts_cand
                            break
                date_full[2 + r_in, 2 + c_in] = eff_ts_p

        result = {
            "z_data": z_patch,
            "valid_mask": valid_patch,
            "predicted": pred_full,
            "rmse": rmse_full,
            "anomaly": anom_full,
            "change_date": date_full,
            "change_count": count_full,
            "segments": chunk_segments,
        }
        self.scccd_cache[cache_key] = result
        return result

    def get_pyccd_results(self, mat_idx, frac_idx, p_idx, r, c, inject_probe=True):
        """
        Executes USGS PyCCD 7-band detection on the selected pixel.
        Operates on patch-specific surface reflectance and injected targets across all conditions.
        If inject_probe is True or (r, c) is an injected offset, target mixture is applied.
        """
        fraction = float(self.data["mixture_fractions"][frac_idx])
        cond_idx = mat_idx * len(self.data["mixture_fractions"]) + frac_idx
        is_target = (r, c) in self.data["injected_offsets"]
        is_injected = is_target or inject_probe
        f_weight = fraction if is_injected else 0.0

        cache_key = (cond_idx, p_idx, r, c, is_injected, f_weight > 0.0)
        if cache_key in self.pyccd_cache:
            return self.pyccd_cache[cache_key]

        l30_patch = self.data["chips"]["HLSL30"][p_idx]
        fm_patch = self.data["chips"]["HLSL30_Fmask"][p_idx]
        cm_patch = self.data["chips"]["HLSL30_common_mask"][p_idx]

        blues = l30_patch[:, 1, r, c].copy()
        greens = l30_patch[:, 2, r, c].copy()
        reds = l30_patch[:, 3, r, c].copy()
        nirs = l30_patch[:, 4, r, c].copy()
        swir1s = l30_patch[:, 5, r, c].copy()
        swir2s = l30_patch[:, 6, r, c].copy()

        if is_injected and f_weight > 0.0:
            spec_l8 = self.data["materials"][mat_idx]["spec_l8"]
            l30_times = self.data["l30_times"]
            onset_ts = self.data["onset_ts"]
            post_onset_idx = np.where(l30_times >= onset_ts)[0]
            scale = self.data["l30_scale"]
            fill_val = self.data["l30_fill"]

            for t in post_onset_idx:
                obs = np.array([blues[t], greens[t], reds[t], nirs[t], swir1s[t], swir2s[t]])
                if not np.any(obs == fill_val):
                    blues[t] = int(round((1.0 - f_weight) * blues[t] + f_weight * spec_l8[1] * scale))
                    greens[t] = int(round((1.0 - f_weight) * greens[t] + f_weight * spec_l8[2] * scale))
                    reds[t] = int(round((1.0 - f_weight) * reds[t] + f_weight * spec_l8[3] * scale))
                    nirs[t] = int(round((1.0 - f_weight) * nirs[t] + f_weight * spec_l8[4] * scale))
                    swir1s[t] = int(round((1.0 - f_weight) * swir1s[t] + f_weight * spec_l8[5] * scale))
                    swir2s[t] = int(round((1.0 - f_weight) * swir2s[t] + f_weight * spec_l8[6] * scale))

        fm_pixel = fm_patch[:, r, c]
        cm_pixel = cm_patch[:, r, c]
        sr_pixel_bands = np.column_stack([blues, greens, reds, nirs, swir1s, swir2s])
        qas = pccd.decode_categorical_qa(fm_pixel, cm_pixel, sr_pixel_bands, fill_val=self.data["l30_fill"])

        proc_params = {
            "QA_BITPACKED": False,
            "QA_FILL": 0,
            "QA_CLEAR": 1,
            "QA_WATER": 2,
            "QA_SHADOW": 3,
            "QA_SNOW": 4,
            "QA_CLOUD": 5,
            "LASSO_MAX_ITER": 2000,
        }

        ordinal_dates = self.data["ordinal_dates_l30"]
        nominal_thermals = self.data["nominal_thermals"]

        try:
            ccd_res = ccd.detect(
                ordinal_dates, blues, greens, reds, nirs, swir1s, swir2s,
                nominal_thermals, qas, params=proc_params
            )
            models = ccd_res.get("change_models", [])
        except Exception:
            models = []

        breaks = [m for m in models if m.get("change_probability", 0) > 0 and m.get("break_day", 0) > 0]

        actual_nir = nirs.astype(np.float32) / float(self.data["l30_scale"])
        valid_refl = (nirs != self.data["l30_fill"]) & (nirs != -9999) & (nirs != -32768) & (actual_nir > 0.0)
        valid_mask = ~cm_pixel.astype(bool) & (fm_pixel != 255) & valid_refl

        result = {
            "actual_nir": actual_nir,
            "valid_mask": valid_mask,
            "models": models,
            "breaks": breaks,
            "is_target": is_target,
            "is_injected": is_injected,
            "f_weight": f_weight,
        }
        self.pyccd_cache[cache_key] = result
        return result


# ==============================================================================
# INTERACTIVE BENCHMARK VIEWER CLASS
# ==============================================================================
class PatchBenchmarkViewer:
    """
    Main interactive visualization controller linking:
      1. Left: 13x13 Spatial change detection date/count grid with pixel selection.
      2. Top Right: Harmonized SC-CCD Z-score time series with all detected breaks.
      3. Bottom Right: USGS LCMAP Landsat NIR time series with all detected breaks.
      4. Keyboard & mouse event bindings for exploration.
    """
    def __init__(self, data, initial_patch=0, initial_mat=0, initial_frac=0, initial_pixel=(4, 4)):
        self.data = data
        self.engine = BenchmarkEngine(data)

        self.num_patches = len(LAND_COVER_PATCHES)
        self.num_materials = len(data["materials"])
        self.num_fractions = len(data["mixture_fractions"])

        self.p_idx = max(0, min(self.num_patches - 1, initial_patch))
        self.mat_idx = max(0, min(self.num_materials - 1, initial_mat))
        self.frac_idx = max(0, min(self.num_fractions - 1, initial_frac))
        self.sel_r, self.sel_c = initial_pixel
        self.grid_mode = "date"  # 'date' or 'count'
        self.inject_probe = True  # Enable on-the-fly target injection simulation on selected pixel

        # Precompute RGB composites for all 4 strata
        self.patch_rgbs = []
        for p in range(self.num_patches):
            rgb, _ = generate_patch_rgb(
                self.data["chips"]["HLSL30"][p],
                self.data["chips"]["HLSL30_common_mask"][p]
            )
            self.patch_rgbs.append(rgb)

        # Setup figure and layout
        plt.style.use("default")
        self.fig = plt.figure(figsize=(19, 10))
        self.fig.canvas.manager.set_window_title(
            "Stratified Patch Benchmark Viewer: Harmonized SC-CCD vs. USGS LCMAP PyCCD"
        )

        # GridSpec: Left (Col 0) for spatial grid & colorbar; Right (Col 1) for stacked SITS plots
        self.gs = self.fig.add_gridspec(
            2, 2,
            width_ratios=[1.0, 1.85],
            height_ratios=[1.0, 1.0],
            left=0.05, right=0.96, top=0.91, bottom=0.10,
            hspace=0.28, wspace=0.18
        )

        gs_left = self.gs[:, 0].subgridspec(2, 1, height_ratios=[1.0, 0.045], hspace=0.10)
        self.ax_grid = self.fig.add_subplot(gs_left[0])
        self.ax_cbar = self.fig.add_subplot(gs_left[1])
        self.ax_sc = self.fig.add_subplot(self.gs[0, 1])
        self.ax_nir = self.fig.add_subplot(self.gs[1, 1], sharex=self.ax_sc)

        # Interactive state elements
        self.cbar = None
        self.rect_patch = None
        self.buttons = []

        self._setup_buttons()
        self._connect_events()

    def _setup_buttons(self):
        """Creates interactive navigation buttons along the bottom."""
        p_offs = self.data["injected_offsets"]
        btn_specs = [
            ("◀ Mat", 0.035, 0.025, 0.048, 0.035, self.prev_material),
            ("Mat ▶", 0.086, 0.025, 0.048, 0.035, self.next_material),
            ("◀ Frac", 0.138, 0.025, 0.048, 0.035, self.prev_fraction),
            ("Frac ▶", 0.189, 0.025, 0.048, 0.035, self.next_fraction),
            ("◀ Patch", 0.241, 0.025, 0.052, 0.035, self.prev_patch),
            ("Patch ▶", 0.296, 0.025, 0.052, 0.035, self.next_patch),
            ("Target P1", 0.352, 0.025, 0.058, 0.035, lambda e: self.select_pixel(p_offs[0][0], p_offs[0][1])),
            ("Target P2", 0.413, 0.025, 0.058, 0.035, lambda e: self.select_pixel(p_offs[1][0], p_offs[1][1])),
            ("Target P3", 0.474, 0.025, 0.058, 0.035, lambda e: self.select_pixel(p_offs[2][0], p_offs[2][1])),
            ("Target P4", 0.535, 0.025, 0.058, 0.035, lambda e: self.select_pixel(p_offs[3][0], p_offs[3][1])),
            ("Grid: Date", 0.598, 0.025, 0.062, 0.035, self.toggle_grid_mode),
            ("Probe: Any", 0.663, 0.025, 0.065, 0.035, self.toggle_probe_mode),
            ("Details", 0.732, 0.025, 0.052, 0.035, lambda e: self.print_pixel_details()),
            ("Save PNG", 0.788, 0.025, 0.058, 0.035, lambda e: self.save_png()),
        ]

        for label, left, bottom, width, height, callback in btn_specs:
            ax_btn = self.fig.add_axes([left, bottom, width, height])
            btn = Button(ax_btn, label, color="#f0f0f0", hovercolor="#d0e0ff")
            btn.on_clicked(callback)
            btn.label.set_fontsize(8.5)
            btn.label.set_fontweight("bold")
            self.buttons.append((btn, ax_btn))

    def _connect_events(self):
        """Connects mouse click and keypress handlers to matplotlib canvas."""
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)

    # ==========================================================================
    # NAVIGATION ACTIONS
    # ==========================================================================
    def toggle_grid_mode(self, event=None):
        self.grid_mode = "count" if self.grid_mode == "date" else "date"
        for btn, ax in self.buttons:
            if "Grid:" in btn.label.get_text():
                btn.label.set_text(f"Grid: {'Date' if self.grid_mode == 'date' else 'Count'}")
        self.render(refresh_grid=True)

    def toggle_probe_mode(self, event=None):
        self.inject_probe = not self.inject_probe
        for btn, ax in self.buttons:
            if "Probe:" in btn.label.get_text():
                btn.label.set_text(f"Probe: {'Any' if self.inject_probe else 'Fixed'}")
        self.render(refresh_grid=False)

    def next_material(self, event=None):
        self.mat_idx = (self.mat_idx + 1) % self.num_materials
        self.render()

    def prev_material(self, event=None):
        self.mat_idx = (self.mat_idx - 1) % self.num_materials
        self.render()

    def next_fraction(self, event=None):
        self.frac_idx = min(self.num_fractions - 1, self.frac_idx + 1)
        self.render()

    def prev_fraction(self, event=None):
        self.frac_idx = max(0, self.frac_idx - 1)
        self.render()

    def next_patch(self, event=None):
        self.p_idx = (self.p_idx + 1) % self.num_patches
        self.render()

    def prev_patch(self, event=None):
        self.p_idx = (self.p_idx - 1) % self.num_patches
        self.render()

    def cycle_target_pixel(self, event=None):
        targets = self.data["injected_offsets"]
        curr = (self.sel_r, self.sel_c)
        if curr in targets:
            idx = (targets.index(curr) + 1) % len(targets)
        else:
            idx = 0
        self.select_pixel(targets[idx][0], targets[idx][1])

    def select_pixel(self, r, c):
        if 0 <= r < PATCH_SIZE and 0 <= c < PATCH_SIZE:
            self.sel_r = r
            self.sel_c = c
            self.render(refresh_grid=False)

    def on_click(self, event):
        if event.inaxes != self.ax_grid or event.xdata is None or event.ydata is None:
            return
        c = int(round(event.xdata))
        r = int(round(event.ydata))
        if 0 <= r < PATCH_SIZE and 0 <= c < PATCH_SIZE:
            self.select_pixel(r, c)

    def on_key_press(self, event):
        key = event.key
        if key in ("m", "right"):
            self.next_material()
        elif key in ("M", "left"):
            self.prev_material()
        elif key in ("f", "up", "+", "="):
            self.next_fraction()
        elif key in ("F", "down", "-", "_"):
            self.prev_fraction()
        elif key in ("p", "]"):
            self.next_patch()
        elif key in ("P", "["):
            self.prev_patch()
        elif key in ("1", "2", "3", "4"):
            self.p_idx = int(key) - 1
            self.render()
        elif key in ("t", "T"):
            self.cycle_target_pixel()
        elif key in ("i", "I"):
            self.toggle_probe_mode()
        elif key in ("c", "C"):
            self.toggle_grid_mode()
        elif key in ("d", "D"):
            self.print_pixel_details()
        elif key in ("s", "S"):
            self.save_png()
        elif key in ("h", "H"):
            self.print_help()
        elif key in ("q", "escape"):
            plt.close(self.fig)

    def print_help(self):
        print("\n" + "=" * 70)
        print("STRATIFIED PATCH BENCHMARK VIEWER KEYBOARD SHORTCUTS:")
        print("  [M] / [m] or [Right] / [Left] : Cycle Target Material")
        print("  [F] / [f] or [Up] / [Down] / [+] / [-] : Cycle Pixel Fill Fraction")
        print("  [P] / [p] or [1]-[4] : Cycle Land-Cover Stratum (Patch)")
        print("  [T] / [t] : Cycle through the 4 injected target pixels (P1-P4)")
        print("  [I] / [i] : Toggle Probe Injection Mode (Any Pixel vs. Fixed Targets)")
        print("  [C] / [c] : Toggle Grid Overlay (Change Date vs. Total Break Count)")
        print("  [D] / [d] : Print Diagnostic Report of All Segments for Selected Pixel")
        print("  [Mouse Click] on 13x13 grid : Select pixel to inspect SITS")
        print("  [S] / [s] : Save current figure to PNG")
        print("  [Q] / [Esc] : Exit viewer")
        print("=" * 70 + "\n")

    def print_pixel_details(self):
        """Prints a comprehensive diagnostic breakdown of all segments and breaks for the selected pixel."""
        patch_info = LAND_COVER_PATCHES[self.p_idx]
        mat_info = self.data["materials"][self.mat_idx]
        fraction = float(self.data["mixture_fractions"][self.frac_idx])
        cond_idx = self.mat_idx * self.num_fractions + self.frac_idx
        is_inj_target = (self.sel_r, self.sel_c) in self.data["injected_offsets"]
        target_tag = f"Target P{self.data['injected_offsets'].index((self.sel_r, self.sel_c)) + 1}" if is_inj_target else "Background / Neighbor"

        print("\n" + "=" * 80)
        print(f"PIXEL TIME SERIES & BREAK DIAGNOSTIC REPORT")
        print(f"  Patch    : [{self.p_idx + 1}/{self.num_patches}] {patch_info['class']} ({patch_info['name']})")
        print(f"  Pixel    : Row {self.sel_r}, Col {self.sel_c} [{target_tag}]")
        print(f"  Target   : [{self.mat_idx + 1}/{self.num_materials}] {mat_info['name']} (Condition ID: {cond_idx})")
        print(f"  Fraction : [{self.frac_idx + 1}/{self.num_fractions}] {fraction:.2f}")
        print("-" * 80)

        # SC-CCD Segments
        scccd_res = self.engine.get_scccd_results(self.mat_idx, self.frac_idx, self.p_idx)
        if 2 <= self.sel_r < 11 and 2 <= self.sel_c < 11:
            sc_segs = scccd_res.get("segments", {}).get((self.sel_r - 2, self.sel_c - 2), [])
        else:
            sc_segs = []

        print(f"[Harmonized SC-CCD] Total Segments: {len(sc_segs)}")
        for s in sc_segs:
            s_idx = s.get("segment_idx", "?")
            s_start = datetime.datetime.fromtimestamp(s["start_date"], timezone.utc).strftime("%Y-%m-%d")
            s_end = datetime.datetime.fromtimestamp(s["end_date"], timezone.utc).strftime("%Y-%m-%d")
            has_b = s.get("has_break", False)
            if has_b:
                b_dt = datetime.datetime.fromtimestamp(s["break_date"], timezone.utc).strftime("%Y-%m-%d")
                c_dt = datetime.datetime.fromtimestamp(s["confirm_date"], timezone.utc).strftime("%Y-%m-%d") if not np.isnan(s.get("confirm_date", np.nan)) else "N/A"
                is_post = (s["break_date"] >= self.data["onset_ts"])
                lat = (s["break_date"] - self.data["onset_ts"]) / 86400.0 if is_post else np.nan
                lat_str = f"+{lat:.1f}d" if is_post else "pre-onset"
                print(f"  Seg {s_idx}: [{s_start} to {s_end}] | BREAK: {b_dt} (confirm: {c_dt}, {lat_str}) | RMSE={s.get('rmse', 0):.4f} | n={s.get('n_points', 0)}")
            else:
                print(f"  Seg {s_idx}: [{s_start} to {s_end}] | ACTIVE (no break) | RMSE={s.get('rmse', 0):.4f} | n={s.get('n_points', 0)}")

        # PyCCD Segments
        pyccd_res = self.engine.get_pyccd_results(
            self.mat_idx, self.frac_idx, self.p_idx, self.sel_r, self.sel_c,
            inject_probe=self.inject_probe
        )
        py_models = pyccd_res.get("models", [])
        print(f"\n[USGS LCMAP PyCCD] Total Model Segments: {len(py_models)}")
        for m_idx, m in enumerate(py_models, 1):
            s_dt = datetime.date.fromordinal(m["start_day"]).strftime("%Y-%m-%d")
            e_dt = datetime.date.fromordinal(m["end_day"]).strftime("%Y-%m-%d")
            b_day = m.get("break_day", 0)
            has_b = (b_day is not None and b_day > 0 and m.get("change_probability", 0) > 0)
            if has_b:
                b_date = datetime.date.fromordinal(b_day)
                b_ts = datetime.datetime(b_date.year, b_date.month, b_date.day, tzinfo=timezone.utc).timestamp()
                is_post = (b_ts >= self.data["onset_ts"])
                lat = (b_ts - self.data["onset_ts"]) / 86400.0 if is_post else np.nan
                lat_str = f"+{lat:.1f}d" if is_post else "pre-onset"
                b_str = b_date.strftime("%Y-%m-%d")
                print(f"  Seg {m_idx}: [{s_dt} to {e_dt}] | BREAK: {b_str} ({lat_str}, prob={m.get('change_probability', 0):.2f}) | RMSE={m['nir']['rmse']:.1f}")
            else:
                print(f"  Seg {m_idx}: [{s_dt} to {e_dt}] | ACTIVE (no break) | RMSE={m['nir']['rmse']:.1f}")
        print("=" * 80 + "\n")

    def save_png(self, output_path=None):
        if output_path is None:
            mat_id = self.data["materials"][self.mat_idx]["id"]
            p_id = LAND_COVER_PATCHES[self.p_idx]["id"].replace(" ", "_")
            frac = self.data["mixture_fractions"][self.frac_idx]
            output_path = f"benchmark_vis_{p_id}_{mat_id}_f{frac:.2f}_r{self.sel_r}c{self.sel_c}.png"

        self.fig.savefig(output_path, dpi=200, bbox_inches="tight")
        print(f"Screenshot successfully saved to: {os.path.abspath(output_path)}")

    # ==========================================================================
    # RENDERING ENGINE
    # ==========================================================================
    def render(self, refresh_grid=True):
        """
        Renders the entire 3-panel display:
          1. ax_grid: 13x13 patch with true-color background and change overlay.
          2. ax_sc: Harmonized SC-CCD Z-score time series with all detected breaks.
          3. ax_nir: Landsat NIR PyCCD surface reflectance time series with all detected breaks.
        """
        fraction = float(self.data["mixture_fractions"][self.frac_idx])
        cond_idx = self.mat_idx * self.num_fractions + self.frac_idx
        mat_info = self.data["materials"][self.mat_idx]
        patch_info = LAND_COVER_PATCHES[self.p_idx]
        onset_ts = self.data["onset_ts"]
        onset_dt = self.data["onset_dt"]

        # Compute / fetch SC-CCD results across patch
        scccd_res = self.engine.get_scccd_results(self.mat_idx, self.frac_idx, self.p_idx)
        # Compute / fetch PyCCD results for currently selected pixel
        pyccd_res = self.engine.get_pyccd_results(
            self.mat_idx, self.frac_idx, self.p_idx, self.sel_r, self.sel_c,
            inject_probe=self.inject_probe
        )

        # ----------------------------------------------------------------------
        # 1. RENDER 13x13 SPATIAL GRID (ax_grid)
        # ----------------------------------------------------------------------
        if refresh_grid:
            self.ax_grid.clear()
            self.ax_cbar.clear()
            self.cbar = None

            # Underlay clear-sky natural true-color RGB composite
            rgb_base = self.patch_rgbs[self.p_idx]
            grid_extent = [-0.5, PATCH_SIZE - 0.5, PATCH_SIZE - 0.5, -0.5]
            self.ax_grid.imshow(rgb_base, interpolation="nearest", extent=grid_extent)

            change_dates = scccd_res["change_date"].copy()
            change_counts = scccd_res["change_count"]

            if self.grid_mode == "count":
                # Render total break count overlay
                self.ax_cbar.set_visible(True)
                counts_eval = change_counts.astype(np.float32)
                counts_eval[:2, :] = np.nan
                counts_eval[11:, :] = np.nan
                counts_eval[:, :2] = np.nan
                counts_eval[:, 11:] = np.nan
                masked_counts = np.ma.masked_invalid(counts_eval)

                max_c = max(1, int(np.nanmax(counts_eval))) if not np.all(np.isnan(counts_eval)) else 1
                cmap = plt.cm.plasma
                cmap.set_bad(color="white", alpha=0.0)
                norm = mcolors.Normalize(vmin=0, vmax=max_c)
                im = self.ax_grid.imshow(
                    masked_counts, cmap=cmap, norm=norm, alpha=0.72,
                    interpolation="nearest", extent=grid_extent
                )
                self.cbar = self.fig.colorbar(im, cax=self.ax_cbar, orientation="horizontal")
                self.cbar.set_label("Total Detected Breaks Across Time Series", fontsize=9, fontweight="bold")
                self.cbar.set_ticks(np.arange(0, max_c + 1))
            else:
                # Render post-onset change date overlay
                change_dates_disp = change_dates.copy()
                change_dates_disp[change_counts == 0] = np.nan

                if not np.all(np.isnan(change_dates_disp)):
                    self.ax_cbar.set_visible(True)
                    cmap = plt.cm.gist_rainbow
                    cmap.set_bad(color="white", alpha=0.0)
                    masked_dates = np.ma.masked_invalid(change_dates_disp)
                    im = self.ax_grid.imshow(
                        masked_dates, cmap=cmap, alpha=0.72,
                        interpolation="nearest", extent=grid_extent
                    )
                    self.cbar = self.fig.colorbar(im, cax=self.ax_cbar, orientation="horizontal")
                    self.cbar.set_label("SC-CCD Detected Change Date", fontsize=9, fontweight="bold")
                    ticks = self.cbar.get_ticks()
                    min_t, max_t = np.nanmin(change_dates_disp), np.nanmax(change_dates_disp)
                    if not np.isnan(min_t) and not np.isnan(max_t):
                        valid_ticks = [t for t in ticks if min_t <= t <= max_t]
                        if len(valid_ticks) > 0:
                            self.cbar.set_ticks(valid_ticks)
                            self.cbar.set_ticklabels([
                                datetime.datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%d")
                                for t in valid_ticks
                            ], fontsize=8)
                else:
                    self.ax_cbar.set_visible(False)

            # Display badge on pixels with multiple breaks (> 1)
            for r_in in range(9):
                for c_in in range(9):
                    r_p, c_p = 2 + r_in, 2 + c_in
                    cnt = change_counts[r_p, c_p]
                    if cnt > 1:
                        self.ax_grid.text(
                            c_p, r_p + 0.30, f"\u00d7{cnt}",
                            color="yellow", fontsize=7.5, fontweight="bold",
                            ha="center", va="center",
                            bbox=dict(boxstyle="round,pad=0.1", facecolor="black", alpha=0.7, edgecolor="none")
                        )

            # Mark inner 9x9 evaluated boundary with dashed yellow box
            eval_box = mpatches.Rectangle(
                (1.5, 1.5), 9, 9,
                linewidth=1.8, edgecolor="yellow", facecolor="none", linestyle="--",
                label="Evaluated Region (9x9)"
            )
            self.ax_grid.add_patch(eval_box)

            # Mark 4 designated target injection pixels (P1-P4)
            for p_num, (roff, coff) in enumerate(self.data["injected_offsets"], 1):
                self.ax_grid.plot(
                    coff, roff, marker="o", markersize=9,
                    markeredgecolor="white", markeredgewidth=1.8, markerfacecolor="red"
                )
                self.ax_grid.text(
                    coff + 0.22, roff - 0.22, f"P{p_num}",
                    color="white", fontweight="bold", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.18", facecolor="#1a1a1a", alpha=0.75, edgecolor="white", linewidth=0.8)
                )

            # Grid coordinates & labels
            self.ax_grid.set_xticks(np.arange(PATCH_SIZE))
            self.ax_grid.set_yticks(np.arange(PATCH_SIZE))
            self.ax_grid.set_xticks(np.arange(-0.5, PATCH_SIZE, 1.0), minor=True)
            self.ax_grid.set_yticks(np.arange(-0.5, PATCH_SIZE, 1.0), minor=True)
            self.ax_grid.grid(False, which="major")
            self.ax_grid.grid(True, which="minor", color="white", linestyle=":", linewidth=0.8, alpha=0.6)
            self.ax_grid.tick_params(which="minor", length=0)
            self.ax_grid.set_xlim(-0.5, PATCH_SIZE - 0.5)
            self.ax_grid.set_ylim(PATCH_SIZE - 0.5, -0.5)
            self.ax_grid.set_xlabel(f"Patch Column (0 to {PATCH_SIZE - 1})", fontsize=10, fontweight="bold")
            self.ax_grid.set_ylabel(f"Patch Row (0 to {PATCH_SIZE - 1})", fontsize=10, fontweight="bold")

            frac_lbl = f"{fraction:.2f} (Unperturbed Patch)" if fraction == 0.0 else f"{fraction:.2f}"
            grid_title = (
                f"{PATCH_SIZE}x{PATCH_SIZE} Spatial Grid: {patch_info['class']} ({patch_info['name']})\n"
                f"Target: {mat_info['name']} (Fraction: {frac_lbl}) | Evaluated: Inner 9x9"
            )
            self.ax_grid.set_title(grid_title, fontsize=11, fontweight="bold", pad=8)

            # Selection bounding box
            self.rect_patch = mpatches.Rectangle(
                (self.sel_c - 0.5, self.sel_r - 0.5), 1, 1,
                linewidth=2.8, edgecolor="cyan", facecolor="none", linestyle="-"
            )
            self.ax_grid.add_patch(self.rect_patch)
        else:
            if self.rect_patch is not None:
                self.rect_patch.set_xy((self.sel_c - 0.5, self.sel_r - 0.5))

        # ----------------------------------------------------------------------
        # 2. RENDER HARMONIZED SC-CCD TIME SERIES (ax_sc)
        # ----------------------------------------------------------------------
        self.ax_sc.clear()

        harm_times = self.data["harm_acq_times"]
        harm_dates = np.array([datetime.datetime.fromtimestamp(t, timezone.utc) for t in harm_times])
        actual_z = scccd_res["z_data"][:, self.sel_r, self.sel_c]
        valid_mask_z = scccd_res["valid_mask"][:, self.sel_r, self.sel_c]
        pred_z = scccd_res["predicted"][:, self.sel_r, self.sel_c]
        rmse_z = scccd_res["rmse"][:, self.sel_r, self.sel_c]
        anom_z = (scccd_res["anomaly"][:, self.sel_r, self.sel_c] == 1)

        spacecrafts = np.array(self.data["spacecrafts"])
        markers = [
            ("s", "Sentinel", ["sentinel", "s30", "hlss30"]),
            ("o", "Landsat", ["landsat", "l30", "hlsl30"]),
            ("D", "Tanager", ["tanager"])
        ]

        # Plot observations differentiated by sensor platform
        matched_any = np.zeros(len(spacecrafts), dtype=bool)
        for marker_type, sc_keyword, sc_aliases in markers:
            sc_mask = np.array([any(alias in str(s).lower() for alias in sc_aliases) for s in spacecrafts])
            matched_any |= sc_mask

            idx_valid_norm = valid_mask_z & sc_mask & ~anom_z
            if np.any(idx_valid_norm):
                self.ax_sc.plot(
                    harm_dates[idx_valid_norm], actual_z[idx_valid_norm],
                    color="black", marker=marker_type, markersize=3.5, linestyle="None",
                    label=f"Valid ({sc_keyword})"
                )

            idx_valid_anom = valid_mask_z & sc_mask & anom_z
            if np.any(idx_valid_anom):
                self.ax_sc.plot(
                    harm_dates[idx_valid_anom], actual_z[idx_valid_anom],
                    color="crimson", marker=marker_type, markersize=5.5, linestyle="None",
                    label=f"Anomaly ({sc_keyword})"
                )

            idx_invalid = ~valid_mask_z & sc_mask
            if np.any(idx_invalid):
                self.ax_sc.plot(
                    harm_dates[idx_invalid], actual_z[idx_invalid],
                    color="gray", marker=marker_type, markersize=2.5, linestyle="None",
                    markerfacecolor="none", alpha=0.35, label=f"Masked ({sc_keyword})"
                )

        if not np.all(matched_any):
            sc_mask = ~matched_any
            idx_valid_norm = valid_mask_z & sc_mask & ~anom_z
            if np.any(idx_valid_norm):
                self.ax_sc.plot(
                    harm_dates[idx_valid_norm], actual_z[idx_valid_norm],
                    color="black", marker="x", markersize=3.5, linestyle="None",
                    label="Valid (Other)"
                )
            idx_valid_anom = valid_mask_z & sc_mask & anom_z
            if np.any(idx_valid_anom):
                self.ax_sc.plot(
                    harm_dates[idx_valid_anom], actual_z[idx_valid_anom],
                    color="crimson", marker="x", markersize=5.5, linestyle="None",
                    label="Anomaly (Other)"
                )
            idx_invalid = ~valid_mask_z & sc_mask
            if np.any(idx_invalid):
                self.ax_sc.plot(
                    harm_dates[idx_invalid], actual_z[idx_invalid],
                    color="gray", marker="x", markersize=2.5, linestyle="None",
                    markerfacecolor="none", alpha=0.35, label="Masked (Other)"
                )

        # Plot SC-CCD Harmonic Regression Segments with confidence bands
        p_val = scccm.CHANGE_PROBABILITY
        df = 3
        rmse_mult = float(np.sqrt(chi2.ppf(p_val, df=df)))
        bound_label = f"Prediction \u00b1{rmse_mult:.2f}\u03c3 (p={p_val})"

        pred_mask = ~np.isnan(pred_z)
        if np.any(pred_mask):
            pred_dates = harm_dates[pred_mask]
            preds = pred_z[pred_mask]
            rmses = rmse_z[pred_mask]

            rmse_diff = np.diff(rmses)
            break_indices = np.where(rmse_diff != 0)[0] + 1

            date_segs = np.split(pred_dates, break_indices)
            pred_segs = np.split(preds, break_indices)
            rmse_segs = np.split(rmses, break_indices)

            seg_colors = ["#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]
            for i, (s_dates, s_preds, s_rmses) in enumerate(zip(date_segs, pred_segs, rmse_segs)):
                if len(s_dates) == 0:
                    continue
                c = seg_colors[i % len(seg_colors)]
                upper = s_preds + rmse_mult * s_rmses
                lower = s_preds - rmse_mult * s_rmses

                seg_lbl = "Harmonic Fit" if i == 0 else f"Harmonic Fit (Seg {i+1})"
                self.ax_sc.plot(s_dates, s_preds, color=c, linestyle="--", linewidth=1.8, label=seg_lbl)
                self.ax_sc.fill_between(
                    s_dates, lower, upper, color=c, alpha=0.15,
                    label=bound_label if i == 0 else None
                )

        # Mark injection onset date
        self.ax_sc.axvline(
            x=self.data["onset_dt"], color="darkviolet", linestyle="--", linewidth=1.8,
            label=f"Onset ({self.data['onset_date_str']})"
        )

        # Extract and mark ALL SC-CCD breaks for this pixel
        if 2 <= self.sel_r < 11 and 2 <= self.sel_c < 11:
            pixel_segs = scccd_res.get("segments", {}).get((self.sel_r - 2, self.sel_c - 2), [])
        else:
            pixel_segs = []

        break_segs = [s for s in pixel_segs if s.get("has_break", False)]
        tot_sc_breaks = len(break_segs)
        post_sc_breaks = 0

        for b_idx, s in enumerate(break_segs, 1):
            b_ts = s.get("break_date", np.nan)
            c_ts = s.get("confirm_date", np.nan)
            if np.isnan(b_ts):
                continue

            dt_break = datetime.datetime.fromtimestamp(b_ts, timezone.utc)
            is_post = (b_ts >= onset_ts) or (not np.isnan(c_ts) and c_ts >= onset_ts)

            if is_post:
                post_sc_breaks += 1
                lat = (b_ts - onset_ts) / 86400.0
                sc_lbl = f"SC Break #{b_idx}: {dt_break.strftime('%Y-%m-%d')} (+{lat:.1f}d)"
                self.ax_sc.axvline(
                    x=dt_break, color="crimson", linestyle=":", linewidth=2.2,
                    label=sc_lbl
                )
                if not np.isnan(c_ts) and c_ts != b_ts:
                    dt_confirm = datetime.datetime.fromtimestamp(c_ts, timezone.utc)
                    c_lat = (c_ts - onset_ts) / 86400.0
                    self.ax_sc.axvline(
                        x=dt_confirm, color="#9467bd", linestyle="-.", linewidth=1.5,
                        label=f"SC Confirm #{b_idx}: {dt_confirm.strftime('%Y-%m-%d')} (+{c_lat:.1f}d)"
                    )
            else:
                sc_lbl = f"Pre-Onset Break #{b_idx}: {dt_break.strftime('%Y-%m-%d')}"
                self.ax_sc.axvline(
                    x=dt_break, color="#e6550d", linestyle="--", linewidth=1.8,
                    label=sc_lbl
                )

        # Pixel identification string
        is_inj_target = (self.sel_r, self.sel_c) in self.data["injected_offsets"]
        target_tag = f"Target P{self.data['injected_offsets'].index((self.sel_r, self.sel_c)) + 1}" if is_inj_target else "Background / Neighbor"

        sc_title = (
            f"Harmonized SC-CCD: Spectral Complexity (Z-Score) | Pixel ({self.sel_r}, {self.sel_c}) [{target_tag}] | "
            f"Total Breaks: {tot_sc_breaks} (Post-Onset: {post_sc_breaks})"
        )
        self.ax_sc.set_title(sc_title, fontsize=11, fontweight="bold", pad=6)
        self.ax_sc.set_ylabel("Spectral Complexity (Z-Score)", fontsize=10, fontweight="bold")
        self.ax_sc.grid(True, linestyle=":", alpha=0.6)

        handles, labels = self.ax_sc.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        self.ax_sc.legend(
            by_label.values(), by_label.keys(),
            loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0, fontsize=8
        )

        # ----------------------------------------------------------------------
        # 3. RENDER LANDSAT NIR PYCCD TIME SERIES (ax_nir)
        # ----------------------------------------------------------------------
        self.ax_nir.clear()

        l30_times = self.data["l30_times"]
        l30_dates = np.array([datetime.datetime.fromtimestamp(t, timezone.utc) for t in l30_times])
        actual_nir = pyccd_res["actual_nir"]
        valid_mask_nir = pyccd_res["valid_mask"]
        models_py = pyccd_res["models"]
        breaks_py = pyccd_res["breaks"]

        is_valid_refl_val = (actual_nir > 0.0) & ~np.isclose(actual_nir, -1.0, atol=0.05) & (actual_nir != self.data["l30_fill"] / float(self.data["l30_scale"]))

        idx_norm_nir = valid_mask_nir & is_valid_refl_val
        if np.any(idx_norm_nir):
            self.ax_nir.plot(
                l30_dates[idx_norm_nir], actual_nir[idx_norm_nir],
                color="black", marker="o", markersize=3.8, linestyle="None",
                label="Valid (Landsat NIR)"
            )

        idx_mask_nir = ~valid_mask_nir & is_valid_refl_val
        if np.any(idx_mask_nir):
            self.ax_nir.plot(
                l30_dates[idx_mask_nir], actual_nir[idx_mask_nir],
                color="silver", marker="o", markersize=2.5, linestyle="None",
                markerfacecolor="none", alpha=0.4, label="Masked (Cloud/Shadow)"
            )

        # Plot PyCCD Harmonic Segments directly from models_py
        seg_colors = ["#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]
        X_full = self.data["X_full_pyccd"]
        ordinal_dates = self.data["ordinal_dates_l30"]
        scale = float(self.data["l30_scale"])

        for i, m in enumerate(models_py):
            c = seg_colors[i % len(seg_colors)]
            s_day = m["start_day"]
            e_day = m["end_day"]
            b_day = m.get("break_day", 0)
            has_b = (b_day is not None and b_day > 0 and m.get("change_probability", 0) > 0)
            seg_end = b_day if has_b else e_day

            seg_indices = np.where((ordinal_dates >= s_day) & (ordinal_dates <= seg_end))[0]
            if len(seg_indices) == 0:
                continue

            b_info = m["nir"]
            theta = np.array([float(b_info["intercept"])] + list(b_info["coefficients"]), dtype=np.float64)
            preds_sub = (X_full[seg_indices] @ theta) / scale
            rmse_sub = float(b_info["rmse"]) / scale
            dates_sub = l30_dates[seg_indices]

            lbl = "PyCCD Fit" if i == 0 else f"PyCCD Fit (Seg {i+1})"
            self.ax_nir.plot(dates_sub, preds_sub, color=c, linestyle="-", linewidth=2.0, label=lbl)
            self.ax_nir.fill_between(
                dates_sub, preds_sub - 3.0 * rmse_sub, preds_sub + 3.0 * rmse_sub,
                color=c, alpha=0.15, label="PyCCD \u00b13\u03c3" if i == 0 else None
            )

        # Mark injection onset date
        self.ax_nir.axvline(
            x=self.data["onset_dt"], color="darkviolet", linestyle="--", linewidth=1.8,
            label=f"Onset ({self.data['onset_date_str']})"
        )

        # Mark ALL detected PyCCD breaks across time series
        tot_py_breaks = len(breaks_py)
        post_py_breaks = 0

        for b_idx, b in enumerate(breaks_py, 1):
            b_day = b.get("break_day", 0)
            if b_day is not None and b_day > 0 and b.get("change_probability", 0) > 0:
                dt_py_break = datetime.date.fromordinal(b_day)
                dt_py_ts = datetime.datetime(dt_py_break.year, dt_py_break.month, dt_py_break.day, tzinfo=timezone.utc).timestamp()
                dt_obj = datetime.datetime(dt_py_break.year, dt_py_break.month, dt_py_break.day, tzinfo=timezone.utc)
                is_post = (dt_py_ts >= onset_ts)
                if is_post:
                    post_py_breaks += 1
                    latency_py = (dt_py_ts - onset_ts) / 86400.0
                    lbl = f"PyCCD Break #{b_idx}: {dt_py_break.strftime('%Y-%m-%d')} (+{latency_py:.1f}d)"
                    self.ax_nir.axvline(
                        x=dt_obj, color="crimson", linestyle=":", linewidth=2.2, label=lbl
                    )
                else:
                    lbl = f"Pre-Onset Break #{b_idx}: {dt_py_break.strftime('%Y-%m-%d')}"
                    self.ax_nir.axvline(
                        x=dt_obj, color="#e6550d", linestyle="--", linewidth=1.8, label=lbl
                    )

        # Dynamic Y-axis scaling based on valid observations and fits
        valid_vals = actual_nir[idx_norm_nir]
        if len(valid_vals) > 5:
            q1, q99 = np.percentile(valid_vals, [1, 99])
            pad = max(0.04, 0.15 * (q99 - q1))
            ymin = max(0.0, float(q1 - pad))
            ymax = min(1.0, float(q99 + pad))
            if ymax > ymin + 0.05:
                self.ax_nir.set_ylim(ymin, ymax)

        frac_str = f"{fraction:.2f} (Unperturbed Baseline)" if fraction == 0.0 else f"{fraction:.2f}"
        if pyccd_res.get("is_target", False):
            p_num = self.data["injected_offsets"].index((self.sel_r, self.sel_c)) + 1
            pixel_tag = f"Target P{p_num} (Benchmark Instance)"
        else:
            pixel_tag = "Interactive Probe Pixel" if (self.inject_probe and fraction > 0.0) else "Background Pixel"

        nir_title = (
            f"USGS LCMAP Landsat PyCCD (NIR Band 0.865 \u03bcm) | Target: {mat_info['name']} (Fraction: {frac_str})\n"
            f"Selected Pixel: ({self.sel_r}, {self.sel_c}) [{pixel_tag}] | Total Breaks: {tot_py_breaks} (Post-Onset: {post_py_breaks})"
        )
        self.ax_nir.set_title(nir_title, fontsize=10.5, fontweight="bold", pad=6)
        self.ax_nir.set_ylabel("NIR Surface Reflectance", fontsize=10, fontweight="bold")
        self.ax_nir.set_xlabel("Acquisition Date", fontsize=10, fontweight="bold")
        self.ax_nir.grid(True, linestyle=":", alpha=0.6)

        self.ax_nir.xaxis.set_major_locator(mdates.YearLocator())
        self.ax_nir.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        self.ax_nir.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))

        handles, labels = self.ax_nir.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        self.ax_nir.legend(
            by_label.values(), by_label.keys(),
            loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0, fontsize=8
        )

        # ----------------------------------------------------------------------
        # 4. SUPER-TITLE & BENCHMARK QUANTITATIVE METRIC BANNER
        # ----------------------------------------------------------------------
        if is_inj_target:
            p_num = self.data["injected_offsets"].index((self.sel_r, self.sel_c)) + 1
            sc_tbl = self.data["scccd_table"]
            py_tbl = self.data["pyccd_table"]

            sc_match = sc_tbl[
                (sc_tbl["condition_idx"] == cond_idx) &
                (sc_tbl["patch_id"] == patch_info["id"].encode("utf-8")) &
                (sc_tbl["pixel_num"] == p_num)
            ]
            if len(sc_match) > 0:
                row = sc_match[0]
                det = bool(row["break_detected"])
                lat = float(row["latency_days"])
                c_lat = float(row["confirmation_latency_days"])
                mag = float(row["break_magnitude"])
                n_tot = int(row["total_breaks"])
                n_post = int(row["post_onset_breaks"])
                has_mult = bool(row["has_multiple_post_breaks"])
                mult_tag = " [Multiple]" if has_mult else ""
                if det:
                    bench_sc_info = f"Detected: True (+{lat:.1f}d, conf=+{c_lat:.1f}d, mag={mag:.2f}) | Breaks: {n_tot} (Post: {n_post}){mult_tag}"
                else:
                    bench_sc_info = f"Detected: False | Breaks: {n_tot} (Post: {n_post})"
            else:
                bench_sc_info = f"Breaks: {tot_sc_breaks} (Post: {post_sc_breaks})"

            py_match = py_tbl[
                (py_tbl["condition_idx"] == cond_idx) &
                (py_tbl["patch_id"] == patch_info["id"].encode("utf-8")) &
                (py_tbl["pixel_num"] == p_num)
            ]
            if len(py_match) > 0:
                row = py_match[0]
                det = bool(row["break_detected"])
                lat = float(row["latency_days"])
                mag = float(row["break_magnitude"])
                n_tot = int(row["total_breaks"])
                n_post = int(row["post_onset_breaks"])
                has_mult = bool(row["has_multiple_post_breaks"])
                mult_tag = " [Multiple]" if has_mult else ""
                if det:
                    bench_py_info = f"Detected: True (+{lat:.1f}d, mag={mag:.2f}) | Breaks: {n_tot} (Post: {n_post}){mult_tag}"
                else:
                    bench_py_info = f"Detected: False | Breaks: {n_tot} (Post: {n_post})"
            else:
                bench_py_info = f"Breaks: {tot_py_breaks} (Post: {post_py_breaks})"
        else:
            bench_sc_info = f"Neighbor (Breaks: {tot_sc_breaks}, Post: {post_sc_breaks})"
            if self.inject_probe and fraction > 0.0:
                bench_py_info = f"Probe (Breaks: {tot_py_breaks}, Post: {post_py_breaks})"
            else:
                bench_py_info = f"Background (Breaks: {tot_py_breaks}, Post: {post_py_breaks})"

        frac_str = f"{fraction:.2f} (Unperturbed Patch)" if fraction == 0.0 else f"{fraction:.2f}"
        super_str = (
            f"STRATIFIED PATCH BENCHMARK | Stratum: [{self.p_idx + 1}/{self.num_patches}] {patch_info['class']} ({patch_info['name']}) | "
            f"Target: [{self.mat_idx + 1}/{self.num_materials}] {mat_info['name']} | "
            f"Fraction: [{self.frac_idx + 1}/{self.num_fractions}] {frac_str}\n"
            f"Selected Pixel: Row {self.sel_r}, Col {self.sel_c} [{target_tag}]  ⟶  "
            f"SC-CCD: {bench_sc_info}  |  PyCCD: {bench_py_info}"
        )
        self.fig.suptitle(super_str, fontsize=12, fontweight="bold", y=0.975)

        self.fig.canvas.draw()


# ==============================================================================
# MAIN COMMAND LINE INTERFACE & ENTRY POINT
# ==============================================================================
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Interactive Diagnostic Viewer for Stratified Patch Target Injection Benchmark Results."
    )
    parser.add_argument(
        "--h5-path", type=str, default=DEFAULT_H5_PATH,
        help=f"Path to master HDF5 container (default: {DEFAULT_H5_PATH})"
    )
    parser.add_argument(
        "--patch", type=str, default="0",
        help="Initial patch index (0-3) or stratum ID (e.g. CROPLAND, TREE COVER, WETLAND, DEVELOPED)"
    )
    parser.add_argument(
        "--material", type=str, default="0",
        help="Initial material index (0-21) or ID (e.g. concrete, asphalt shingle, etc.)"
    )
    parser.add_argument(
        "--fraction", type=float, default=0.0,
        help="Initial mixture fill fraction (default: 0.0 for unperturbed patch)"
    )
    parser.add_argument(
        "--pixel", type=int, nargs=2, default=[4, 4],
        help="Initial pixel row and col (default: 4 4 for target P1)"
    )
    parser.add_argument(
        "--save-png", type=str, default=None,
        help="Filename to export initial rendering to PNG and exit (non-interactive mode)"
    )
    parser.add_argument(
        "--no-gui", action="store_true",
        help="Disable interactive GUI display (useful for batch or headless environments)"
    )
    return parser.parse_args()


def resolve_indices(args, data):
    """Maps CLI string or index arguments to valid numeric indices."""
    # Resolve patch index
    p_arg = str(args.patch).strip().upper()
    patch_idx = 0
    if p_arg.isdigit():
        patch_idx = int(p_arg)
    else:
        for idx, p in enumerate(LAND_COVER_PATCHES):
            if p_arg in (p["id"].upper(), p["class"].upper(), p["name"].upper()):
                patch_idx = idx
                break

    # Resolve material index
    m_arg = str(args.material).strip().lower()
    mat_idx = 0
    if m_arg.isdigit():
        mat_idx = int(m_arg)
    else:
        for idx, m in enumerate(data["materials"]):
            if m_arg in (m["id"].lower(), m["name"].lower()):
                mat_idx = idx
                break

    # Resolve fraction index
    frac_val = float(args.fraction)
    fractions = data["mixture_fractions"]
    frac_idx = int(np.argmin(np.abs(fractions - frac_val)))

    # Resolve pixel coordinates
    sel_r, sel_c = args.pixel[0], args.pixel[1]
    return patch_idx, mat_idx, frac_idx, (sel_r, sel_c)


def main():
    args = parse_arguments()

    if args.save_png or args.no_gui:
        matplotlib.use("Agg")

    print(f"\n[Patch Benchmark Viewer] Loading benchmark container: {args.h5_path}")
    data = load_benchmark_container(args.h5_path)

    p_idx, mat_idx, frac_idx, (sel_r, sel_c) = resolve_indices(args, data)

    print(f"Loaded {len(data['materials'])} materials across {len(LAND_COVER_PATCHES)} strata.")
    print(f"Initializing viewer at:")
    print(f"  Stratum : [{p_idx}] {LAND_COVER_PATCHES[p_idx]['class']} ({LAND_COVER_PATCHES[p_idx]['name']})")
    print(f"  Material: [{mat_idx}] {data['materials'][mat_idx]['name']}")
    print(f"  Fraction: [{frac_idx}] {data['mixture_fractions'][frac_idx]:.2f}")
    print(f"  Pixel   : Row {sel_r}, Col {sel_c}")

    viewer = PatchBenchmarkViewer(
        data,
        initial_patch=p_idx,
        initial_mat=mat_idx,
        initial_frac=frac_idx,
        initial_pixel=(sel_r, sel_c)
    )
    viewer.render(refresh_grid=True)

    if args.save_png:
        viewer.save_png(args.save_png)
        print(f"Export complete. Exiting...")
        data["h5_file"].close()
        return

    if not args.no_gui:
        viewer.print_help()
        plt.show()

    data["h5_file"].close()


if __name__ == "__main__":
    main()
