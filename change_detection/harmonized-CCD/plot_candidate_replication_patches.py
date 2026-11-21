"""
plot_candidate_replication_patches.py
======================================
Identifies, evaluates, and visualizes candidate replication patches across Rochester, NY
for triplicating the multi-sensor synthetic target injection experiment (N = 3 Replicate Design).

Evaluates:
1. Zero false-alarm break verification across PYCCD, SCCCD, and NDVICCD (0/50 clean grid pixels).
2. Window break purity (< 3% breaks in surrounding neighborhood).
3. Periodic seasonal harmonic strength (Annual Amplitude and Periodic SNR = Amp / RMSE) to resolve
   weak background periodicity observed at Site A for materials such as Red Paving Brick.
4. Geographic and land-cover diversity (Suburban Park vs. Agricultural Cropland).
"""

import os
import sys
import datetime
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

H5_PATH = "C:/satelliteImagery/HLST30/HLST_Rochesterv2.h5"
SAVE_PATH = "F:/Resilio/IMGS 890 Research/JSTARS-Spectral-Complexity/media/candidate_replication_patches.png"

# Proposed N = 3 Replicate Design:
# 3 Suburban Sites (Array Type A: weights [0.05, 0.15, 0.25, 0.35, 0.45])
# 3 Agricultural Sites (Array Type B: weights [0.10, 0.20, 0.30, 0.40, 0.50])
PROPOSED_SITES = [
    {
        "id": "A1",
        "role": "Replicate 1 (Suburban)",
        "name": "Site A1: Penfield Suburban (Current Site A)",
        "r": 618,
        "c": 583,
        "type": "Suburban / Park Lawn",
        "weights": [0.05, 0.15, 0.25, 0.35, 0.45],
        "color": "#1f77b4",
    },
    {
        "id": "A2",
        "role": "Replicate 2 (Suburban)",
        "name": "Site A2: Greece Northwest Park",
        "r": 343,
        "c": 351,
        "type": "Suburban Park / Meadow",
        "weights": [0.05, 0.15, 0.25, 0.35, 0.45],
        "color": "#2ca02c",
    },
    {
        "id": "A3",
        "role": "Replicate 3 (Suburban)",
        "name": "Site A3: Mendon/Victor Meadow (Lower Right)",
        "r": 971,
        "c": 937,
        "type": "Lower Right Parkland / Meadow",
        "weights": [0.05, 0.15, 0.25, 0.35, 0.45],
        "color": "#17becf",
    },
    {
        "id": "B1",
        "role": "Replicate 1 (Agricultural)",
        "name": "Site B1: Macedon/Perinton Rural (Current Site B)",
        "r": 709,
        "c": 834,
        "type": "Cropland / Pasture",
        "weights": [0.10, 0.20, 0.30, 0.40, 0.50],
        "color": "#d62728",
    },
    {
        "id": "B2",
        "role": "Replicate 2 (Agricultural)",
        "name": "Site B2: Henrietta Agricultural Field",
        "r": 980,
        "c": 403,
        "type": "Active Cropland",
        "weights": [0.10, 0.20, 0.30, 0.40, 0.50],
        "color": "#ff7f0e",
    },
    {
        "id": "B3",
        "role": "Replicate 3 (Agricultural)",
        "name": "Site B3: Rush Open Farm Parcel",
        "r": 993,
        "c": 485,
        "type": "Extensive Farmland",
        "weights": [0.10, 0.20, 0.30, 0.40, 0.50],
        "color": "#9467bd",
    },
]


def evaluate_site(f, site):
    """Calculates zero-break verification, window break rate, and harmonic metrics."""
    r, c = site["r"], site["c"]
    grid_r = [r + i * 3 for i in range(10)]
    grid_c = [c + j * 3 for j in range(5)]

    py_cc = f["PYCCD_Results"]["change_count"]
    sc_cc = f["SCCCD_Results"]["change_count"]
    nd_cc = f["NDVICCD_Results"]["change_count"]

    # 1. Injection Grid Verification
    grid_breaks = sum(
        (py_cc[gr, gc] > 0) or (sc_cc[gr, gc] > 0) or (nd_cc[gr, gc] > 0)
        for gr in grid_r
        for gc in grid_c
    )

    # 2. Window Break Rate
    w_r0, w_r1 = max(0, r - 2), min(1409, r + 28 + 3)
    w_c0, w_c1 = max(0, c - 2), min(1070, c + 13 + 3)
    win_any = (
        (py_cc[w_r0:w_r1, w_c0:w_c1] > 0)
        | (sc_cc[w_r0:w_r1, w_c0:w_c1] > 0)
        | (nd_cc[w_r0:w_r1, w_c0:w_c1] > 0)
    )
    win_breaks = int(np.sum(win_any))
    win_total = win_any.size
    win_pct = (win_breaks / win_total) * 100.0

    # 3. Harmonic Strength across all 10 material rows
    scccd = f["SCCCD_Results"]
    amps = []
    rmses = []
    for gr in grid_r:
        for gc in grid_c:
            cf = scccd["coef_series"][-1, gr, gc, :]
            amps.append(np.sqrt(cf[1] ** 2 + cf[2] ** 2))
            rmses.append(np.nanmean(scccd["rmse_series"][:, gr, gc]))

    mean_amp = float(np.mean(amps))
    mean_rmse = float(np.mean(rmses))
    periodic_snr = mean_amp / mean_rmse if mean_rmse > 0 else 0.0

    # 4. Red Paving Brick (row 7, col 3) periodicity
    r_brick = grid_r[6]
    c_brick = grid_c[2]
    cf_brick = scccd["coef_series"][-1, r_brick, c_brick, :]
    amp_brick = float(np.sqrt(cf_brick[1] ** 2 + cf_brick[2] ** 2))
    rmse_brick = float(np.nanmean(scccd["rmse_series"][:, r_brick, c_brick]))
    snr_brick = amp_brick / rmse_brick if rmse_brick > 0 else 0.0

    return {
        "grid_breaks": grid_breaks,
        "win_breaks": win_breaks,
        "win_total": win_total,
        "win_pct": win_pct,
        "mean_amp": mean_amp,
        "mean_rmse": mean_rmse,
        "periodic_snr": periodic_snr,
        "brick_amp": amp_brick,
        "brick_rmse": rmse_brick,
        "brick_snr": snr_brick,
        "grid_r": grid_r,
        "grid_c": grid_c,
    }


def main(h5_path=H5_PATH, save_path=SAVE_PATH):
    print("=" * 80)
    print("EVALUATING CANDIDATE REPLICATION PATCHES (N = 3 REPLICATE DESIGN)")
    print("=" * 80)

    with h5py.File(h5_path, "r") as f:
        harm = f["/HDFEOS/GRIDS/HARMONIZED/Data Fields"]
        acq = harm["sliding_volume_z_score"].attrs["acquisition_time"][:]
        dates = [datetime.datetime.fromtimestamp(t, datetime.timezone.utc) for t in acq]

        # Select summer true-color base frame
        target_date = datetime.date(2025, 9, 12)
        diffs = [abs((d.date() - target_date).days) for d in dates]
        ortho_idx = int(np.argmin(diffs))
        ortho_full = np.transpose(harm["ortho_visual"][ortho_idx][:3], (1, 2, 0)).astype(np.float32) / 255.0

        results = []
        for s in PROPOSED_SITES:
            metrics = evaluate_site(f, s)
            results.append({**s, **metrics})

    # Print summary table
    print(f"{'ID':<4} | {'Site Name':<36} | Top-Left | Clean Grid | Win Breaks | Mean SNR | Brick Amp | Brick SNR")
    print("-" * 105)
    for r in results:
        clean_str = f"0/50 (100%)" if r["grid_breaks"] == 0 else f"{r['grid_breaks']} breaks"
        win_str = f"{r['win_breaks']}/{r['win_total']} ({r['win_pct']:.1f}%)"
        print(
            f"{r['id']:<4} | {r['name']:<36} | ({r['r']:4d},{r['c']:4d}) | {clean_str:<10} | {win_str:<10} | "
            f"{r['periodic_snr']:8.3f} | {r['brick_amp']:9.4f} | {r['brick_snr']:9.3f}"
        )

    # --------------------------------------------------------------------------
    # Render Visualization Figure
    # --------------------------------------------------------------------------
    fig = plt.figure(figsize=(20, 14))
    fig.suptitle(
        "Candidate Replication Patches for Triplicated Target Injection Experiment (N = 3 Replicate Design)\n"
        "Resolving Background Scene Variability with 100% Clean Baseline Grids Across 3 Remote Sensing Algorithms",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )

    # Layout:
    # Left: Scene Context Map showing patch distributions across Rochester
    # Right: 6 subplots showing zoomed true-color ortho imagery with injection grids
    gs = fig.add_gridspec(3, 4, width_ratios=[1.6, 1.0, 1.0, 1.0], hspace=0.32, wspace=0.28)

    # Left: Full Scene Overview
    ax_map = fig.add_subplot(gs[:, 0])
    ax_map.imshow(ortho_full)
    ax_map.set_title("Rochester, NY Study Area (30m HLS Grid)\nSpatial Distribution of 6 Replicate Sites", fontsize=12, fontweight="bold")
    ax_map.set_xlabel("UTM / Albers Grid Column", fontsize=10)
    ax_map.set_ylabel("UTM / Albers Grid Row", fontsize=10)

    # Overlay bounding boxes on context map
    for r in results:
        rect = patches.Rectangle(
            (r["c"] - 10, r["r"] - 10),
            35,
            50,
            linewidth=2.0,
            edgecolor=r["color"],
            facecolor="none",
            label=f"{r['id']}: {r['name'][:22]}",
        )
        ax_map.add_patch(rect)
        ax_map.text(
            r["c"] + 25,
            r["r"] + 15,
            r["id"],
            color="white",
            fontsize=11,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor=r["color"], alpha=0.85),
        )

    ax_map.legend(loc="lower left", fontsize=8, framealpha=0.9)

    # Right: 6 Zooms (2 columns x 3 rows)
    # Positions in gridspec: (0, 1), (0, 2), (1, 1), (1, 2), (2, 1), (2, 2)
    zoom_coords = [(0, 1), (0, 2), (1, 1), (1, 2), (2, 1), (2, 2)]

    for idx, (r, (zr, zc)) in enumerate(zip(results, zoom_coords)):
        ax_z = fig.add_subplot(gs[zr, zc])

        # Crop patch with 2-pixel margin
        p_r0, p_r1 = max(0, r["r"] - 2), min(1409, r["r"] + 28 + 3)
        p_c0, p_c1 = max(0, r["c"] - 2), min(1070, r["c"] + 13 + 3)
        crop = ortho_full[p_r0:p_r1, p_c0:p_c1]

        ax_z.imshow(crop, extent=[-0.5, crop.shape[1] - 0.5, crop.shape[0] - 0.5, -0.5])

        # Plot 50 injection pixels
        for gr in r["grid_r"]:
            loc_y = gr - p_r0
            for gc in r["grid_c"]:
                loc_x = gc - p_c0
                rect = patches.Rectangle(
                    (loc_x - 0.45, loc_y - 0.45),
                    0.9,
                    0.9,
                    facecolor="none",
                    edgecolor="cyan",
                    linestyle="--",
                    linewidth=1.0,
                )
                ax_z.add_patch(rect)

        # Highlight Red Paving Brick (row 7, col 3)
        loc_brick_y = r["grid_r"][6] - p_r0
        loc_brick_x = r["grid_c"][2] - p_c0
        brick_rect = patches.Rectangle(
            (loc_brick_x - 0.45, loc_brick_y - 0.45),
            0.9,
            0.9,
            facecolor="yellow",
            edgecolor="black",
            linewidth=1.2,
            alpha=0.7,
        )
        ax_z.add_patch(brick_rect)

        # Subplot title with periodic metrics
        ax_z.set_title(
            f"{r['id']}: {r['name'][:24]}\n"
            f"Periodic SNR: {r['periodic_snr']:.3f} | Brick Amp: {r['brick_amp']:.3f}",
            fontsize=9.5,
            fontweight="bold",
            color=r["color"],
        )
        ax_z.set_xticks([gc - p_c0 for gc in r["grid_c"]])
        ax_z.set_xticklabels([f"{w:.2f}" for w in r["weights"]], fontsize=7)
        ax_z.set_yticks([])
        ax_z.grid(False)

    # Right column: Comparison Metrics Table / Annotation Box
    ax_table = fig.add_subplot(gs[:, 3])
    ax_table.axis("off")

    table_text = (
        "TRIPLICATED EXPERIMENTAL DESIGN SUMMARY\n"
        "====================================\n\n"
        "Goal: Duplicate the synthetic target injection\n"
        "strategy 3 times to average out background\n"
        "scene variability across land cover types.\n\n"
        "Suburban Background Replicates (f = 0.05 - 0.45):\n"
        "-----------------------------------------------\n"
        "  * Site A1 (Current): Penfield Lawn / Park\n"
        "    - Clean Grid: 0 / 50 breaks (100% clean)\n"
        "    - Brick Periodicity: Amp = 0.034, SNR = 0.197\n"
        "  * Site A2 (Replicate 2): Greece Park / Meadow\n"
        "    - Clean Grid: 0 / 50 breaks (100% clean)\n"
        "    - Window Purity: 99.2% clean (5 / 594 breaks)\n"
        "    - Brick Periodicity: Amp = 0.099, SNR = 0.279\n"
        "      (3x higher seasonal amplitude!)\n"
        "  * Site A3 (Replicate 3): Mendon/Victor Meadow\n"
        "    - Location: Lower Right Quadrant (r=971, c=937)\n"
        "    - Clean Grid: 0 / 50 breaks (100% clean)\n"
        "    - Brick Periodicity: Amp = 0.081, SNR = 0.311\n"
        "      (2.4x higher seasonal amplitude than A1!)\n\n"
        "Agricultural Background Replicates (f = 0.10 - 0.50):\n"
        "--------------------------------------------------\n"
        "  * Site B1 (Current): Perinton Farmland\n"
        "    - Clean Grid: 0 / 50 breaks (100% clean)\n"
        "    - Brick Periodicity: Amp = 0.161, SNR = 0.459\n"
        "  * Site B2 (Replicate 2): Henrietta Cropland\n"
        "    - Clean Grid: 0 / 50 breaks (100% clean)\n"
        "    - Mean Periodic SNR: 0.284\n"
        "  * Site B3 (Replicate 3): Rush Open Farm Parcel\n"
        "    - Clean Grid: 0 / 50 breaks (100% clean)\n"
        "    - High Periodic SNR: 0.313, Amp = 0.083\n\n"
        "Total Test Sites: 6 patches (300 pixels total)\n"
        "Every (Material, Weight) tested N = 3 times.\n"
        "Enables empirical detection probability P_d\n"
        "and eliminates single-site background bias."
    )

    ax_table.text(
        0.02,
        0.98,
        table_text,
        transform=ax_table.transAxes,
        fontsize=9.5,
        fontfamily="monospace",
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa", edgecolor="#ced4da"),
    )

    fig.subplots_adjust(left=0.03, right=0.98, top=0.93, bottom=0.04)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved candidate patches figure to: {save_path}")

        # Also copy/save to media/
        media_path = "F:/Resilio/IMGS 890 Research/JSTARS-Spectral-Complexity/media/candidate_replication_patches.png"
        try:
            os.makedirs(os.path.dirname(media_path), exist_ok=True)
            fig.savefig(media_path, dpi=300, bbox_inches="tight")
            print(f"Saved candidate patches figure to media: {media_path}")
        except Exception as e:
            print(f"Warning: Could not save to media path: {e}")

    return fig


if __name__ == "__main__":
    main()
