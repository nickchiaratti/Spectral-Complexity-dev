"""
plot_injection_spatial_and_timeseries.py
========================================
Visualizes multi-sensor synthetic target injection experiments across Rochester, NY:

Figure 1: Spatial comparison of detected change date overlays without injection (baseline)
          alongside with injection, spatially subset to Site A and Site B.
Figure 2: Multi-panel time series plots of before and after injection responses
          (Spectral Complexity Z-Score or NDVI) for each target material across all mixture levels.

Turn-key executable in the IDE or via CLI:
    python plot_injection_spatial_and_timeseries.py
    python plot_injection_spatial_and_timeseries.py --metric NDVI
    python plot_injection_spatial_and_timeseries.py --metric SC --save
"""

import os
import sys
import argparse
import datetime
from datetime import timezone
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.dates as mdates
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib as mpl

# ==============================================================================
# DEFAULT CONFIGURATION
# ==============================================================================
DEFAULT_H5_PATH = "C:/satelliteImagery/HLST30/HLST_Rochesterv2.h5"
DEFAULT_METRIC = "SC"  # 'SC' (Spectral Complexity Z-Score) or 'NDVI'
SAVE_DIR = "F:/Resilio/IMGS 890 Research/JSTARS-Spectral-Complexity/plot-scripts"


def resolve_metric_groups(metric_name="SC"):
    """Maps metric abbreviation to baseline/injected group names and dataset keys."""
    m_str = str(metric_name).strip().upper()
    if m_str in ("SC", "SPECTRAL_COMPLEXITY", "Z_SCORE", "SLIDING_VOLUME_Z_SCORE"):
        return {
            "display": "Spectral Complexity (Z-Score)",
            "short": "SC",
            "baseline_group": "SCCCD_Results",
            "injected_group": "INJECTED_SCCCD_Results",
            "baseline_ds": "sliding_volume_z_score",
            "injected_ds": "injected_sliding_volume_z_score",
            "ylabel": "Spectral Complexity Z-Score",
            "unit": "σ",
        }
    elif m_str in ("NDVI", "NDVI_INDEX"):
        return {
            "display": "Harmonized NDVI",
            "short": "NDVI",
            "baseline_group": "NDVICCD_Results",
            "injected_group": "INJECTED_NDVICCD_Results",
            "baseline_ds": "ndvi",
            "injected_ds": "injected_ndvi",
            "ylabel": "Normalized Difference Vegetation Index",
            "unit": "NDVI",
        }
    else:
        raise ValueError(f"Unsupported metric '{metric_name}'. Choose 'SC' or 'NDVI'.")


def load_injection_data(h5_path, metric_cfg):
    """Loads injection metadata, change detection dates, true-color frames, and time series."""
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"Master HDF5 container not found at: {h5_path}")

    with h5py.File(h5_path, "r") as f:
        # Load injection metadata
        if "INJECTION_METADATA" not in f:
            raise KeyError("INJECTION_METADATA group missing from container. Run create_injected_dataset.py first.")
        meta_attrs = f["INJECTION_METADATA"].attrs
        materials = [
            m.decode("utf-8") if isinstance(m, bytes) else str(m)
            for m in meta_attrs["materials"]
        ]
        site_a_rows = np.array(meta_attrs["site_a_rows"], dtype=int)
        site_a_cols = np.array(meta_attrs["site_a_cols"], dtype=int)
        site_b_rows = np.array(meta_attrs["site_b_rows"], dtype=int)
        site_b_cols = np.array(meta_attrs["site_b_cols"], dtype=int)
        site_a_weights = np.array(meta_attrs["site_a_weights"], dtype=float)
        site_b_weights = np.array(meta_attrs["site_b_weights"], dtype=float)
        onset_ts = float(meta_attrs.get("injection_onset_timestamp", 1609459200.0))

        # Harmonized grids
        harm_grp = f["/HDFEOS/GRIDS/HARMONIZED/Data Fields"]
        acq_times = np.array(harm_grp[metric_cfg["baseline_ds"]].attrs["acquisition_time"][:], dtype=float)
        dates = [datetime.datetime.fromtimestamp(ts, timezone.utc) for ts in acq_times]

        # Select clear summer ortho frame for true-color underlay
        target_ortho_date = datetime.date(2025, 9, 12)
        diffs = [abs((d.date() - target_ortho_date).days) for d in dates]
        ortho_idx = int(np.argmin(diffs))

        ortho_raw = harm_grp["ortho_visual"][ortho_idx]
        if ortho_raw.shape[0] in (3, 4):
            ortho_img = np.transpose(ortho_raw[:3], (1, 2, 0)).astype(np.float32) / 255.0
        else:
            ortho_img = ortho_raw.astype(np.float32) / 255.0

        # Change detection results
        b_grp_name = metric_cfg["baseline_group"]
        i_grp_name = metric_cfg["injected_group"]

        if b_grp_name not in f:
            raise KeyError(f"Baseline CCD results '{b_grp_name}' not found. Run Harmonized CCD baseline first.")
        if i_grp_name not in f:
            raise KeyError(f"Injected CCD results '{i_grp_name}' not found. Run Harmonized CCD on injected datasets first.")

        base_change_dates = f[b_grp_name]["change_date_timestamp"][:]
        base_change_counts = f[b_grp_name]["change_count"][:]
        inj_change_dates = f[i_grp_name]["change_date_timestamp"][:]
        inj_change_counts = f[i_grp_name]["change_count"][:]

        # Spatial bounding boxes for sites with context padding (2 pixels)
        pad = 2
        bbox_a = (
            int(min(site_a_rows) - pad),
            int(max(site_a_rows) + pad + 1),
            int(min(site_a_cols) - pad),
            int(max(site_a_cols) + pad + 1),
        )
        bbox_b = (
            int(min(site_b_rows) - pad),
            int(max(site_b_rows) + pad + 1),
            int(min(site_b_cols) - pad),
            int(max(site_b_cols) + pad + 1),
        )

        # Slice spatial ortho and change patches
        ortho_a = ortho_img[bbox_a[0]:bbox_a[1], bbox_a[2]:bbox_a[3]]
        ortho_b = ortho_img[bbox_b[0]:bbox_b[1], bbox_b[2]:bbox_b[3]]

        base_cd_a = base_change_dates[bbox_a[0]:bbox_a[1], bbox_a[2]:bbox_a[3]]
        base_cc_a = base_change_counts[bbox_a[0]:bbox_a[1], bbox_a[2]:bbox_a[3]]
        inj_cd_a = inj_change_dates[bbox_a[0]:bbox_a[1], bbox_a[2]:bbox_a[3]]
        inj_cc_a = inj_change_counts[bbox_a[0]:bbox_a[1], bbox_a[2]:bbox_a[3]]

        base_cd_b = base_change_dates[bbox_b[0]:bbox_b[1], bbox_b[2]:bbox_b[3]]
        base_cc_b = base_change_counts[bbox_b[0]:bbox_b[1], bbox_b[2]:bbox_b[3]]
        inj_cd_b = inj_change_dates[bbox_b[0]:bbox_b[1], bbox_b[2]:bbox_b[3]]
        inj_cc_b = inj_change_counts[bbox_b[0]:bbox_b[1], bbox_b[2]:bbox_b[3]]

        # Load time series for all 100 injection sites
        # Site A: 10 materials x 5 weights
        # Site B: 10 materials x 5 weights
        all_r = np.concatenate([site_a_rows, site_b_rows])
        all_c = np.concatenate([site_a_cols, site_b_cols])
        min_r, max_r = int(min(all_r)), int(max(all_r) + 1)
        min_c, max_c = int(min(all_c)), int(max(all_c) + 1)

        b_ds = harm_grp[metric_cfg["baseline_ds"]]
        i_ds = harm_grp[metric_cfg["injected_ds"]]
        mask_ds = harm_grp["common_mask"]

        # Read only bounding box slice for speed
        base_sub = b_ds[:, min_r:max_r, min_c:max_c]
        inj_sub = i_ds[:, min_r:max_r, min_c:max_c]
        mask_sub = mask_ds[:, min_r:max_r, min_c:max_c]

        if np.issubdtype(base_sub.dtype, np.integer):
            base_sub = base_sub.astype(np.float32) / 10000.0
            base_sub[base_sub < -3.0] = np.nan
        if np.issubdtype(inj_sub.dtype, np.integer):
            inj_sub = inj_sub.astype(np.float32) / 10000.0
            inj_sub[inj_sub < -3.0] = np.nan

    return {
        "materials": materials,
        "site_a_rows": site_a_rows,
        "site_a_cols": site_a_cols,
        "site_b_rows": site_b_rows,
        "site_b_cols": site_b_cols,
        "site_a_weights": site_a_weights,
        "site_b_weights": site_b_weights,
        "onset_ts": onset_ts,
        "onset_date": datetime.datetime.fromtimestamp(onset_ts, timezone.utc),
        "acq_times": acq_times,
        "dates": dates,
        "bbox_a": bbox_a,
        "bbox_b": bbox_b,
        "ortho_a": ortho_a,
        "ortho_b": ortho_b,
        "base_cd_a": base_cd_a,
        "base_cc_a": base_cc_a,
        "inj_cd_a": inj_cd_a,
        "inj_cc_a": inj_cc_a,
        "base_cd_b": base_cd_b,
        "base_cc_b": base_cc_b,
        "inj_cd_b": inj_cd_b,
        "inj_cc_b": inj_cc_b,
        "base_sub": base_sub,
        "inj_sub": inj_sub,
        "mask_sub": mask_sub,
        "min_r": min_r,
        "min_c": min_c,
    }


# ==============================================================================
# FIGURE 1: SPATIAL OVERLAY OF DETECTED CHANGE DATES
# ==============================================================================
def plot_figure_1_spatial_overlay(data, metric_cfg, save_path=None):
    """
    Renders Figure 1:
    A 2x2 multi-panel figure displaying detected change date overlays without injection
    alongside with injection, spatially subset to Site A and Site B.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle(
        f"Spatial Change Detection Overlays: Baseline vs. Synthetic Target Injection\n"
        f"Target Metric: {metric_cfg['display']}",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    # Date colormap mapping: from onset (2021-01-01) to end (2026-06-01)
    min_ts = data["onset_ts"]  # 2021-01-01
    max_ts = max(data["acq_times"])  # ~2026-06-01
    norm = Normalize(vmin=min_ts, vmax=max_ts)
    cmap = plt.get_cmap("turbo")

    panels = [
        {
            "ax": axes[0, 0],
            "title": "Site A (Suburban Background) - Baseline (Without Injection)",
            "ortho": data["ortho_a"],
            "bbox": data["bbox_a"],
            "rows": data["site_a_rows"],
            "cols": data["site_a_cols"],
            "weights": data["site_a_weights"],
            "cd": data["base_cd_a"],
            "cc": data["base_cc_a"],
            "is_injected": False,
        },
        {
            "ax": axes[0, 1],
            "title": "Site A (Suburban Background) - Target Injection (With Injection)",
            "ortho": data["ortho_a"],
            "bbox": data["bbox_a"],
            "rows": data["site_a_rows"],
            "cols": data["site_a_cols"],
            "weights": data["site_a_weights"],
            "cd": data["inj_cd_a"],
            "cc": data["inj_cc_a"],
            "is_injected": True,
        },
        {
            "ax": axes[1, 0],
            "title": "Site B (Agricultural Background) - Baseline (Without Injection)",
            "ortho": data["ortho_b"],
            "bbox": data["bbox_b"],
            "rows": data["site_b_rows"],
            "cols": data["site_b_cols"],
            "weights": data["site_b_weights"],
            "cd": data["base_cd_b"],
            "cc": data["base_cc_b"],
            "is_injected": False,
        },
        {
            "ax": axes[1, 1],
            "title": "Site B (Agricultural Background) - Target Injection (With Injection)",
            "ortho": data["ortho_b"],
            "bbox": data["bbox_b"],
            "rows": data["site_b_rows"],
            "cols": data["site_b_cols"],
            "weights": data["site_b_weights"],
            "cd": data["inj_cd_b"],
            "cc": data["inj_cc_b"],
            "is_injected": True,
        },
    ]

    total_detected = {}

    for p in panels:
        ax = p["ax"]
        ortho = p["ortho"]
        b_r0, b_r1, b_c0, b_c1 = p["bbox"]
        rows = p["rows"]
        cols = p["cols"]
        weights = p["weights"]
        cd = p["cd"]
        cc = p["cc"]

        # Display true color background
        ax.imshow(ortho, extent=[-0.5, ortho.shape[1] - 0.5, ortho.shape[0] - 0.5, -0.5])

        # Plot injection grid pixels
        detected_count = 0
        for i, r in enumerate(rows):
            loc_y = r - b_r0
            for j, c in enumerate(cols):
                loc_x = c - b_c0
                has_break = cc[loc_y, loc_x] > 0
                ts = cd[loc_y, loc_x]

                if has_break and not np.isnan(ts):
                    detected_count += 1
                    marker_color = cmap(norm(ts))
                    # Draw filled detected marker with black border
                    rect = patches.Rectangle(
                        (loc_x - 0.45, loc_y - 0.45),
                        0.9,
                        0.9,
                        facecolor=marker_color,
                        edgecolor="black",
                        linewidth=1.2,
                        zorder=4,
                    )
                    ax.add_patch(rect)
                else:
                    # Draw open square for un-broken pixel
                    rect = patches.Rectangle(
                        (loc_x - 0.45, loc_y - 0.45),
                        0.9,
                        0.9,
                        facecolor="none",
                        edgecolor="cyan" if not p["is_injected"] else "white",
                        linestyle="--",
                        linewidth=1.0,
                        alpha=0.85,
                        zorder=3,
                    )
                    ax.add_patch(rect)

        # Set axes labels and ticks matching injection matrix
        col_locs = [c - b_c0 for c in cols]
        row_locs = [r - b_r0 for r in rows]

        ax.set_xticks(col_locs)
        ax.set_xticklabels([f"f={w:.2f}" for w in weights], fontsize=9, rotation=0)

        ax.set_yticks(row_locs)
        ax.set_yticklabels(
            [f"{data['materials'][i][:16]}" for i in range(len(rows))],
            fontsize=8,
        )

        ax.set_title(
            f"{p['title']}\nDetected Breaks: {detected_count}/50 ({detected_count * 2.0:.1f}%)",
            fontsize=11,
            fontweight="bold",
        )
        ax.grid(False)

    # Common colorbar for change detection dates
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=0.025, pad=0.03)
    cbar.set_label("Detected Break Date", fontsize=11, fontweight="bold")

    # Format colorbar ticks as YYYY-MM-DD
    tick_years = [2021, 2022, 2023, 2024, 2025, 2026]
    tick_ts = [datetime.datetime(y, 1, 1, tzinfo=timezone.utc).timestamp() for y in tick_years]
    cbar.set_ticks(tick_ts)
    cbar.set_ticklabels([f"{y}-01-01" for y in tick_years])

    # Annotation legend
    legend_elements = [
        patches.Patch(facecolor="none", edgecolor="cyan", linestyle="--", label="Clean Background (0 Breaks)"),
        patches.Patch(facecolor="none", edgecolor="white", linestyle="--", label="Undetected / Sub-threshold"),
        patches.Patch(facecolor="orange", edgecolor="black", label="Break Detected (Color = Date)"),
    ]
    axes[1, 0].legend(handles=legend_elements, loc="lower left", fontsize=8, framealpha=0.85)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved Figure 1 to: {save_path}")

    return fig


# ==============================================================================
# FIGURE 2: BEFORE AND AFTER INJECTION TIME SERIES
# ==============================================================================
def plot_figure_2_timeseries_overview(data, metric_cfg, save_path=None):
    """
    Renders Figure 2:
    A 5x2 faceted multi-panel figure displaying before and after injection time series
    for all 10 target materials across multiple mixture levels.
    """
    fig, axes = plt.subplots(5, 2, figsize=(18, 16), sharex=True, sharey=True)
    fig.suptitle(
        f"Multi-Sensor Temporal Response Before and After Target Injection (2015-2026)\n"
        f"Target Metric: {metric_cfg['display']}",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    dates = data["dates"]
    onset_date = data["onset_date"]
    min_r, min_c = data["min_r"], data["min_c"]

    # Representative weights to show in overview: Baseline + 3 mixture levels
    # Low (f=0.10, Site B), Medium (f=0.25, Site A), High (f=0.50, Site B)
    mats = data["materials"]

    for m_idx in range(10):
        ax = axes[m_idx % 5, m_idx // 5]
        mat_name = mats[m_idx]

        # Pixel coordinates for representative mixture levels:
        # Site A: col index 2 corresponds to w=0.25
        r_a = data["site_a_rows"][m_idx] - min_r
        c_a_med = data["site_a_cols"][2] - min_c  # w = 0.25

        # Site B: col index 0 is w=0.10, col index 4 is w=0.50
        r_b = data["site_b_rows"][m_idx] - min_r
        c_b_low = data["site_b_cols"][0] - min_c  # w = 0.10
        c_b_high = data["site_b_cols"][4] - min_c  # w = 0.50

        # Baseline series at Site A and Site B
        ts_base_a = data["base_sub"][:, r_a, c_a_med]
        mask_a = data["mask_sub"][:, r_a, c_a_med] == 0
        valid_base_a = mask_a & ~np.isnan(ts_base_a)

        ts_inj_low = data["inj_sub"][:, r_b, c_b_low]
        ts_inj_med = data["inj_sub"][:, r_a, c_a_med]
        ts_inj_high = data["inj_sub"][:, r_b, c_b_high]

        # Plot baseline trajectory
        valid_dates_a = [dates[k] for k in range(len(dates)) if valid_base_a[k]]
        valid_vals_a = ts_base_a[valid_base_a]

        ax.plot(
            valid_dates_a,
            valid_vals_a,
            color="black",
            marker="o",
            markersize=2.5,
            linestyle="-",
            linewidth=0.7,
            alpha=0.6,
            label="Baseline (f=0.00)",
        )

        # Plot injected responses for post-2021 observations
        onset_idx = [k for k, d in enumerate(dates) if d >= onset_date][0]
        post_dates = dates[onset_idx:]

        # High weight
        mask_b_high = data["mask_sub"][onset_idx:, r_b, c_b_high] == 0
        v_h = mask_b_high & ~np.isnan(ts_inj_high[onset_idx:])
        ax.plot(
            [post_dates[k] for k in range(len(post_dates)) if v_h[k]],
            ts_inj_high[onset_idx:][v_h],
            color="#d73027",  # Red
            marker="s",
            markersize=3.0,
            linestyle="-",
            linewidth=1.2,
            alpha=0.85,
            label="Injected f=0.50",
        )

        # Medium weight
        mask_a_med = data["mask_sub"][onset_idx:, r_a, c_a_med] == 0
        v_m = mask_a_med & ~np.isnan(ts_inj_med[onset_idx:])
        ax.plot(
            [post_dates[k] for k in range(len(post_dates)) if v_m[k]],
            ts_inj_med[onset_idx:][v_m],
            color="#fdae61",  # Orange
            marker="^",
            markersize=3.0,
            linestyle="-",
            linewidth=1.1,
            alpha=0.85,
            label="Injected f=0.25",
        )

        # Low weight
        mask_b_low = data["mask_sub"][onset_idx:, r_b, c_b_low] == 0
        v_l = mask_b_low & ~np.isnan(ts_inj_low[onset_idx:])
        ax.plot(
            [post_dates[k] for k in range(len(post_dates)) if v_l[k]],
            ts_inj_low[onset_idx:][v_l],
            color="#4575b4",  # Blue
            marker="x",
            markersize=3.0,
            linestyle="--",
            linewidth=0.9,
            alpha=0.80,
            label="Injected f=0.10",
        )

        # Disturbance onset vertical line
        ax.axvline(
            onset_date,
            color="crimson",
            linestyle="--",
            linewidth=1.3,
            label="Onset (2021-01-01)" if m_idx == 0 else None,
        )

        ax.set_title(f"{m_idx + 1}. {mat_name}", fontsize=11, fontweight="bold", pad=4)
        ax.grid(True, linestyle=":", alpha=0.5)

        if m_idx % 5 == 2 and m_idx // 5 == 0:
            ax.set_ylabel(metric_cfg["ylabel"], fontsize=12, fontweight="bold")

    axes[4, 0].set_xlabel("Observation Date", fontsize=11, fontweight="bold")
    axes[4, 1].set_xlabel("Observation Date", fontsize=11, fontweight="bold")

    # Format x-axis dates
    axes[4, 0].xaxis.set_major_locator(mdates.YearLocator(2))
    axes[4, 0].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[4, 1].xaxis.set_major_locator(mdates.YearLocator(2))
    axes[4, 1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # Single comprehensive legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=5,
        fontsize=10,
        bbox_to_anchor=(0.5, 0.94),
        framealpha=0.9,
    )

    plt.tight_layout(rect=[0.02, 0.03, 0.98, 0.91])

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved Figure 2 to: {save_path}")

    return fig


# ==============================================================================
# FIGURE 3: INTERACTIVE DETAIL INSPECTOR (MATERIAL AND MIXTURE COMPARISON)
# ==============================================================================
def plot_figure_3_interactive_detail(data, metric_cfg, initial_material=0, initial_weight_idx=4):
    """
    Renders an interactive detail figure focusing on a single material across all 10 mixture weights,
    with upper time series and lower delta deviation panel.
    """
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(14, 9), sharex=True, gridspec_kw={"height_ratios": [2.5, 1.2]})

    dates = data["dates"]
    onset_date = data["onset_date"]
    min_r, min_c = data["min_r"], data["min_c"]
    all_weights = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

    state = {
        "mat_idx": initial_material,
        "w_idx": initial_weight_idx,
    }

    def render():
        ax_top.clear()
        ax_bot.clear()

        m_idx = state["mat_idx"]
        w_val = all_weights[state["w_idx"]]
        mat_name = data["materials"][m_idx]

        # Determine if pixel is in Site A (odd weights 0.05, 0.15...) or Site B (even weights 0.10, 0.20...)
        if state["w_idx"] % 2 == 0:
            # Site A
            col_pos = state["w_idx"] // 2
            r_coord = data["site_a_rows"][m_idx]
            c_coord = data["site_a_cols"][col_pos]
            site_label = f"Site A (Suburban), Pixel ({r_coord}, {c_coord})"
        else:
            # Site B
            col_pos = state["w_idx"] // 2
            r_coord = data["site_b_rows"][m_idx]
            c_coord = data["site_b_cols"][col_pos]
            site_label = f"Site B (Agricultural), Pixel ({r_coord}, {c_coord})"

        loc_r = r_coord - min_r
        loc_c = c_coord - min_c

        ts_base = data["base_sub"][:, loc_r, loc_c]
        ts_inj = data["inj_sub"][:, loc_r, loc_c]
        valid_mask = (data["mask_sub"][:, loc_r, loc_c] == 0) & ~np.isnan(ts_base)

        valid_dates = [dates[k] for k in range(len(dates)) if valid_mask[k]]
        valid_base = ts_base[valid_mask]
        valid_inj = ts_inj[valid_mask]

        # Upper plot: Time series trajectories
        ax_top.plot(
            valid_dates,
            valid_base,
            color="#2c3e50",
            marker="o",
            markersize=3.5,
            linestyle="-",
            linewidth=1.0,
            alpha=0.75,
            label="Baseline (Without Injection)",
        )

        onset_idx = [k for k, d in enumerate(valid_dates) if d >= onset_date]
        if onset_idx:
            first_post = onset_idx[0]
            ax_top.plot(
                valid_dates[first_post:],
                valid_inj[first_post:],
                color="#e74c3c",
                marker="s",
                markersize=4.0,
                linestyle="-",
                linewidth=1.5,
                alpha=0.9,
                label=f"Injected (f = {w_val:.2f})",
            )

        ax_top.axvline(
            onset_date,
            color="red",
            linestyle="--",
            linewidth=1.5,
            label="Injection Onset (2021-01-01)",
        )

        ax_top.set_title(
            f"Material: {mat_name} | Mixture Fraction: f = {w_val:.2f} | Location: {site_label}\n"
            f"Metric: {metric_cfg['display']}",
            fontsize=13,
            fontweight="bold",
        )
        ax_top.set_ylabel(metric_cfg["ylabel"], fontsize=11, fontweight="bold")
        ax_top.grid(True, linestyle=":", alpha=0.6)
        ax_top.legend(loc="upper left", fontsize=10, framealpha=0.9)

        # Lower plot: Residual divergence Delta = Injected - Baseline
        delta = valid_inj - valid_base
        ax_bot.axhline(0, color="gray", linestyle="-", linewidth=0.8)
        ax_bot.plot(
            valid_dates,
            delta,
            color="#8e44ad",
            marker="d",
            markersize=3.5,
            linestyle="-",
            linewidth=1.0,
            alpha=0.85,
            label="Delta (Injected - Baseline)",
        )
        ax_bot.axvline(onset_date, color="red", linestyle="--", linewidth=1.5)

        ax_bot.set_ylabel(f"Δ {metric_cfg['unit']}", fontsize=11, fontweight="bold")
        ax_bot.set_xlabel("Observation Date", fontsize=11, fontweight="bold")
        ax_bot.grid(True, linestyle=":", alpha=0.6)
        ax_bot.legend(loc="upper left", fontsize=9, framealpha=0.9)

        ax_bot.xaxis.set_major_locator(mdates.YearLocator(2))
        ax_bot.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

        fig.canvas.draw_idle()

    def on_key(event):
        if event.key in ("up", "right"):
            state["mat_idx"] = (state["mat_idx"] + 1) % 10
            render()
        elif event.key in ("down", "left"):
            state["mat_idx"] = (state["mat_idx"] - 1) % 10
            render()
        elif event.key in ("+", "="):
            state["w_idx"] = min(len(all_weights) - 1, state["w_idx"] + 1)
            render()
        elif event.key in ("-", "_"):
            state["w_idx"] = max(0, state["w_idx"] - 1)
            render()

    fig.canvas.mpl_connect("key_press_event", on_key)
    render()
    return fig


# ==============================================================================
# MAIN DRIVER (IDE READY)
# ==============================================================================
def main(h5_path=DEFAULT_H5_PATH, metric=DEFAULT_METRIC, save=False, save_dir=SAVE_DIR):
    """
    Main entry point for generating spatial overlay and time series comparison figures.
    """
    metric_cfg = resolve_metric_groups(metric)
    print("=" * 70)
    print(" TARGET INJECTION SPATIAL & TIME SERIES VISUALIZATION")
    print("=" * 70)
    print(f" Container: {h5_path}")
    print(f" Target Metric: {metric_cfg['display']}")
    print(f" Baseline Group: {metric_cfg['baseline_group']}")
    print(f" Injected Group: {metric_cfg['injected_group']}")
    print("=" * 70)

    print("Loading injection spatial patches and time series data...")
    data = load_injection_data(h5_path, metric_cfg)

    # Output paths
    os.makedirs(save_dir, exist_ok=True)
    out_f1 = os.path.join(save_dir, f"Figure1_injection_spatial_overlay_{metric_cfg['short']}.png") if save else None
    out_f2 = os.path.join(save_dir, f"Figure2_injection_timeseries_all_materials_{metric_cfg['short']}.png") if save else None

    print("\nRendering Figure 1: Spatial Change Date Overlays...")
    fig1 = plot_figure_1_spatial_overlay(data, metric_cfg, save_path=out_f1)

    print("Rendering Figure 2: Multi-Material Time Series Overview...")
    fig2 = plot_figure_2_timeseries_overview(data, metric_cfg, save_path=out_f2)

    print("Rendering Figure 3: Interactive Detail Inspector...")
    fig3 = plot_figure_3_interactive_detail(data, metric_cfg)

    print("\nDisplaying figures (Use arrow keys in Figure 3 to cycle materials, +/- for weights)...")
    plt.show()

    return fig1, fig2, fig3


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Target Injection Spatial & Time Series Plots")
    parser.add_argument("--h5-path", type=str, default=DEFAULT_H5_PATH, help="Path to master HDF5 container")
    parser.add_argument("--metric", type=str, default=DEFAULT_METRIC, help="Metric: 'SC' or 'NDVI'")
    parser.add_argument("--save", action="store_true", help="Save figures to high-resolution PNG")
    parser.add_argument("--save-dir", type=str, default=SAVE_DIR, help="Directory to save PNG files")
    args = parser.parse_args()

    main(h5_path=args.h5_path, metric=args.metric, save=args.save, save_dir=args.save_dir)
