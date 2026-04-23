"""
analyze_gram_qr_differences.py
================================
Deep comparison of process_volume_sliding_tile_parallel (Gram/det)
vs process_volume_sliding_tile_parallel_QR (QR-decomposition).

Answers three questions:
  1. How large are the differences on finite pixels? (0.1% or orders of magnitude?)
  2. Is there a spatial pattern to differences?
  3. Are differences only in NaN handling, or do valid-pixel values diverge?

Outputs:
  - Console summary tables
  - PNG plots saved to:  ./gram_qr_analysis/
"""

import os
import time
import multiprocessing
import numpy as np
import matplotlib
matplotlib.use('Agg')   # no display needed; saves PNG files
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from SpecComplexQR import (
    process_volume_sliding_tile_parallel,
    process_volume_sliding_tile_parallel_QR,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ARD_H5_PATH     = r"C:\satelliteImagery\HLST30\HLST_Tait_Harmonized.h5"
ARD_GRIDS       = ['HLSL30', 'HLSS30']
MAX_REAL_FRAMES = 10

TILE_SIZE       = 3
STRIDE          = 1
NUM_ENDMEMBERS  = 7
GRAM_TYPE       = 'minEndmember'
NORM_TYPE       = 'bandCount'
N_JOBS          = max(2, multiprocessing.cpu_count())

OUT_DIR         = os.path.join(os.path.dirname(__file__), "gram_qr_analysis")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _timed(fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, time.perf_counter() - t0


def _load_real_frames(h5_path, grids, max_frames):
    """Load surface_reflectance + common_mask, apply mask -> NaN."""
    import h5py
    frames = []
    with h5py.File(h5_path, 'r') as h5:
        for grid in grids:
            dset_path = f"/HDFEOS/GRIDS/{grid}/Data Fields"
            if dset_path not in h5:
                continue
            dg   = h5[dset_path]
            sr   = dg['surface_reflectance']
            mask = dg['common_mask']
            acq_times = sr.attrs.get('acquisition_time', np.zeros(sr.shape[0]))
            for t in range(sr.shape[0]):
                if len(frames) >= max_frames:
                    break
                frame_sr   = sr[t, ...].astype(np.float32)
                frame_mask = mask[t, ...]
                invalid    = frame_mask != 1
                frame_sr[:, invalid] = np.nan
                valid_pct  = 100.0 * float((~invalid).sum()) / float(invalid.size)
                frames.append({
                    'frame_data': frame_sr,
                    'label'     : f"{grid}[{t}] ts={acq_times[t]:.0f}",
                    'grid'      : grid,
                    'local_idx' : t,
                    'valid_pct' : valid_pct,
                    'height'    : sr.shape[2],
                    'width'     : sr.shape[3],
                    'bands'     : sr.shape[1],
                })
            if len(frames) >= max_frames:
                break
    return frames


def _analyse_pair(label, gram_map, qr_map, height, width, valid_pct):
    """
    Compute and return a comprehensive difference report dict.
    Separates analysis into three pixel categories:
      A) Pixels where BOTH maps are NaN (masked/boundary)
      B) Pixels where ONLY ONE map is NaN (structural disagreement)
      C) Pixels where BOTH maps are finite (actual value comparison)
    """
    gm = gram_map.astype(np.float64)
    qm = qr_map.astype(np.float64)

    nan_gram = np.isnan(gm)
    nan_qr   = np.isnan(qm)

    both_nan  = nan_gram & nan_qr       # category A
    one_nan   = nan_gram ^ nan_qr       # category B  (XOR)
    both_fin  = ~nan_gram & ~nan_qr     # category C

    total = height * width

    # --- Category C analysis ------------------------------------------------
    diff_c    = np.where(both_fin, np.abs(gm - qm), np.nan)
    rel_c     = np.where(both_fin & (np.abs(gm) > 0),
                         np.abs(gm - qm) / np.abs(gm), np.nan)
    val_gram  = gm[both_fin]
    val_qr    = qm[both_fin]
    val_diff  = diff_c[both_fin]

    n_fin   = int(both_fin.sum())
    n_bnan  = int(both_nan.sum())
    n_onan  = int(one_nan.sum())

    report = {
        'label'             : label,
        'height'            : height,
        'width'             : width,
        'valid_pct'         : valid_pct,
        'n_total'           : total,
        'n_both_nan'        : n_bnan,
        'n_one_nan'         : n_onan,
        'n_finite'          : n_fin,
        'gram_map'          : gram_map,
        'qr_map'            : qr_map,
        'both_fin_mask'     : both_fin,
        'one_nan_mask'      : one_nan,
        'both_nan_mask'     : both_nan,
        'diff_map'          : diff_c,   # NaN where not both_finite
        'rel_diff_map'      : rel_c,
    }

    if n_fin > 0:
        report.update({
            'fin_max_abs'   : float(val_diff.max()),
            'fin_mean_abs'  : float(val_diff.mean()),
            'fin_median_abs': float(np.median(val_diff)),
            'fin_p95_abs'   : float(np.percentile(val_diff, 95)),
            'fin_p99_abs'   : float(np.percentile(val_diff, 99)),
            'fin_max_rel'   : float(np.nanmax(rel_c)),
            'fin_mean_rel'  : float(np.nanmean(rel_c)),
            'n_nonzero'     : int(np.sum(val_diff > 0)),
            'gram_range'    : (float(val_gram.min()), float(val_gram.max())),
            'gram_mean'     : float(val_gram.mean()),
        })
        # Spatial zone breakdown (border, interior)
        b = TILE_SIZE
        zone_border   = np.zeros((height, width), bool)
        zone_border[:b, :] = True; zone_border[-b:, :] = True
        zone_border[:, :b] = True; zone_border[:, -b:] = True
        zone_interior = ~zone_border

        for zone_name, zone_mask in [('border', zone_border), ('interior', zone_interior)]:
            combined = both_fin & zone_mask
            n_zone   = int(combined.sum())
            if n_zone > 0:
                d_zone = diff_c[combined]
                report[f'zone_{zone_name}_n']        = n_zone
                report[f'zone_{zone_name}_max_abs']  = float(d_zone.max())
                report[f'zone_{zone_name}_mean_abs'] = float(d_zone.mean())
                report[f'zone_{zone_name}_n_nonzero']= int((d_zone > 0).sum())
            else:
                report[f'zone_{zone_name}_n']        = 0
                report[f'zone_{zone_name}_max_abs']  = float('nan')
                report[f'zone_{zone_name}_mean_abs'] = float('nan')
                report[f'zone_{zone_name}_n_nonzero']= 0
    else:
        report.update({
            'fin_max_abs'   : float('nan'),
            'fin_mean_abs'  : float('nan'),
            'fin_median_abs': float('nan'),
            'fin_p95_abs'   : float('nan'),
            'fin_p99_abs'   : float('nan'),
            'fin_max_rel'   : float('nan'),
            'fin_mean_rel'  : float('nan'),
            'n_nonzero'     : 0,
            'gram_range'    : (float('nan'), float('nan')),
            'gram_mean'     : float('nan'),
        })
        for zone_name in ('border', 'interior'):
            report[f'zone_{zone_name}_n']        = 0
            report[f'zone_{zone_name}_max_abs']  = float('nan')
            report[f'zone_{zone_name}_mean_abs'] = float('nan')
            report[f'zone_{zone_name}_n_nonzero']= 0

    return report


def _print_report(r):
    tot  = r['n_total']
    print(f"\n  {'='*60}")
    print(f"  Frame : {r['label']}")
    print(f"  Size  : {r['height']} x {r['width']} = {tot:,} pixels   "
          f"valid input = {r['valid_pct']:.1f}%")
    print(f"  {'='*60}")

    print(f"\n  Pixel category breakdown:")
    print(f"    A) Both maps NaN  (masked/boundary)  : {r['n_both_nan']:>8,}  "
          f"({100.0*r['n_both_nan']/tot:5.1f}%)")
    print(f"    B) Only one map NaN (structural diff) : {r['n_one_nan']:>8,}  "
          f"({'**** !!!! ****' if r['n_one_nan'] > 0 else 'OK - zero'})")
    print(f"    C) Both maps finite (value comparison): {r['n_finite']:>8,}  "
          f"({100.0*r['n_finite']/tot:5.1f}%)")

    if r['n_finite'] == 0:
        print(f"\n  [!] No finite pixels to compare (all masked or boundary).")
        return

    n_nz = r['n_nonzero']
    n_c  = r['n_finite']
    print(f"\n  Finite pixel value statistics (Gram map):")
    print(f"    Range : [{r['gram_range'][0]:.6f},  {r['gram_range'][1]:.6f}]")
    print(f"    Mean  :  {r['gram_mean']:.6f}")

    print(f"\n  Absolute difference  |Gram - QR|  on finite pixels ({n_c:,} pixels):")
    print(f"    Pixels with ANY difference: {n_nz:>8,} / {n_c:,}  ({100.0*n_nz/n_c:.4f}%)")
    if n_nz == 0:
        print(f"    >>> ALL FINITE PIXELS ARE BIT-FOR-BIT IDENTICAL <<<")
    else:
        print(f"    Max absolute diff   : {r['fin_max_abs']:.6e}")
        print(f"    Mean absolute diff  : {r['fin_mean_abs']:.6e}")
        print(f"    Median abs diff     : {r['fin_median_abs']:.6e}")
        print(f"    95th pct abs diff   : {r['fin_p95_abs']:.6e}")
        print(f"    99th pct abs diff   : {r['fin_p99_abs']:.6e}")
        print(f"    Max relative diff   : {r['fin_max_rel']:.4%}")
        print(f"    Mean relative diff  : {r['fin_mean_rel']:.4%}")

    print(f"\n  Spatial zone breakdown (border = outer {TILE_SIZE} pixels):")
    for zone_name in ('border', 'interior'):
        n_z  = r[f'zone_{zone_name}_n']
        n_nz_z = r[f'zone_{zone_name}_n_nonzero']
        mx   = r[f'zone_{zone_name}_max_abs']
        mn   = r[f'zone_{zone_name}_mean_abs']
        if n_z > 0:
            print(f"    {zone_name.capitalize():<10}: {n_z:>6,} finite px  "
                  f"differing={n_nz_z:>6,}  max|diff|={mx:.3e}  mean|diff|={mn:.3e}")
        else:
            print(f"    {zone_name.capitalize():<10}: 0 finite pixels")


def _plot_frame(r, prefix):
    """Save a 4-panel diagnostic figure for one frame."""
    h, w = r['height'], r['width']
    gm = r['gram_map'].astype(np.float64)
    qm = r['qr_map'].astype(np.float64)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f"Gram vs QR  |  {r['label']}", fontsize=12, fontweight='bold')

    # -- Panel 1: Gram map
    ax = axes[0, 0]
    img = ax.imshow(gm, cmap='viridis', aspect='auto')
    ax.set_title("Gram/det output map")
    plt.colorbar(img, ax=ax, shrink=0.8)

    # -- Panel 2: QR map
    ax = axes[0, 1]
    img = ax.imshow(qm, cmap='viridis', aspect='auto')
    ax.set_title("QR output map")
    plt.colorbar(img, ax=ax, shrink=0.8)

    # -- Panel 3: NaN category map
    cat = np.zeros((h, w), dtype=np.float32)
    cat[r['both_nan_mask']]  = 0.0   # both NaN  → gray
    cat[r['both_fin_mask']]  = 1.0   # both finite → blue
    cat[r['one_nan_mask']]   = 2.0   # structural mismatch → red
    cmap_cat = mcolors.ListedColormap(['#888888', '#2196F3', '#F44336'])
    ax = axes[0, 2]
    img = ax.imshow(cat, cmap=cmap_cat, vmin=0, vmax=2, aspect='auto')
    ax.set_title("Pixel categories\n0=both NaN(grey)  1=finite(blue)  2=mismatch(red)")
    cbar = plt.colorbar(img, ax=ax, shrink=0.8, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(['Both NaN', 'Both finite', 'Structural diff'])

    # -- Panel 4: Absolute difference map (finite pixels)
    ax = axes[1, 0]
    diff_map = r['diff_map']
    if r['n_finite'] > 0 and r['n_nonzero'] > 0:
        img = ax.imshow(diff_map, cmap='hot_r', aspect='auto')
        ax.set_title(f"|Gram - QR| (finite pixels)\nmax={r['fin_max_abs']:.3e}")
        plt.colorbar(img, ax=ax, shrink=0.8)
    else:
        ax.imshow(np.zeros((h, w)), cmap='gray', aspect='auto')
        ax.set_title("|Gram - QR| = 0 everywhere\n(bit-for-bit identical)")
        ax.text(w//2, h//2, "IDENTICAL", ha='center', va='center',
                fontsize=16, color='green', fontweight='bold')

    # -- Panel 5: Histogram of finite values
    ax = axes[1, 1]
    if r['n_finite'] > 0:
        val_g = gm[r['both_fin_mask']]
        val_q = qm[r['both_fin_mask']]
        bins = np.linspace(min(val_g.min(), val_q.min()),
                            max(val_g.max(), val_q.max()), 60)
        ax.hist(val_g, bins=bins, alpha=0.6, label='Gram', color='steelblue', density=True)
        ax.hist(val_q, bins=bins, alpha=0.6, label='QR',   color='tomato',    density=True)
        ax.set_title("Value distribution (finite pixels)")
        ax.set_xlabel("Volume map value")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "No finite pixels", ha='center', va='center',
                transform=ax.transAxes, fontsize=14)
        ax.set_title("Value distribution")

    # -- Panel 6: Scatter Gram vs QR (finite pixels)
    ax = axes[1, 2]
    if r['n_finite'] > 0:
        val_g = gm[r['both_fin_mask']]
        val_q = qm[r['both_fin_mask']]
        # subsample for performance
        idx = np.random.choice(len(val_g), min(5000, len(val_g)), replace=False)
        ax.scatter(val_g[idx], val_q[idx], s=3, alpha=0.3, color='steelblue')
        lo = min(val_g.min(), val_q.min())
        hi = max(val_g.max(), val_q.max())
        ax.plot([lo, hi], [lo, hi], 'r--', lw=1.5, label='y=x (perfect)')
        ax.set_xlabel("Gram value")
        ax.set_ylabel("QR value")
        ax.set_title("QR vs Gram scatter (finite pixels)")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "No finite pixels", ha='center', va='center',
                transform=ax.transAxes, fontsize=14)
        ax.set_title("Scatter plot")

    plt.tight_layout()
    fname = os.path.join(OUT_DIR, f"{prefix}.png")
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  [plot] Saved -> {fname}")


def _aggregate_summary(reports):
    """Print a cross-frame aggregate table."""
    print(f"\n{'='*72}")
    print(f"  Cross-frame aggregate summary  ({len(reports)} frames)")
    print(f"{'='*72}")
    header = (f"  {'Frame':<28}  {'Fin%':>5}  {'Struc':>5}  "
              f"{'MaxAbs':>10}  {'MaxRel%':>8}  {'Nonzero':>8}")
    print(header)
    print(f"  {'-'*28}  {'-'*5}  {'-'*5}  {'-'*10}  {'-'*8}  {'-'*8}")

    any_nonzero = False
    for r in reports:
        n_tot  = r['n_total']
        fin_pct = 100.0 * r['n_finite'] / n_tot if n_tot > 0 else 0.0
        struc  = r['n_one_nan']
        max_a  = r.get('fin_max_abs', float('nan'))
        max_r  = r.get('fin_max_rel', float('nan'))
        n_nz   = r.get('n_nonzero', 0)
        if n_nz > 0:
            any_nonzero = True
        label  = r['label'][:28]
        max_a_s = f"{max_a:.3e}" if not np.isnan(max_a) else "   n/a"
        max_r_s = f"{100*max_r:.4f}%" if not np.isnan(max_r) else "   n/a"
        print(f"  {label:<28}  {fin_pct:>5.1f}  {struc:>5}  "
              f"{max_a_s:>10}  {max_r_s:>8}  {n_nz:>8}")

    print()
    if not any_nonzero:
        print("  *** ALL FINITE PIXELS ACROSS ALL FRAMES ARE BIT-FOR-BIT IDENTICAL ***")
        print("  *** Differences are ONLY in NaN pattern (boundary/masked pixels)  ***")
    else:
        tot_nz   = sum(r.get('n_nonzero', 0) for r in reports)
        tot_fin  = sum(r['n_finite'] for r in reports)
        all_max  = max((r.get('fin_max_abs', 0) for r in reports
                         if not np.isnan(r.get('fin_max_abs', float('nan')))), default=0)
        all_maxr = max((r.get('fin_max_rel', 0) for r in reports
                         if not np.isnan(r.get('fin_max_rel', float('nan')))), default=0)
        print(f"  Total non-zero-diff finite pixels : {tot_nz:,} / {tot_fin:,}")
        print(f"  Global max absolute difference    : {all_max:.6e}")
        print(f"  Global max relative difference    : {all_maxr:.4%}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    np.random.seed(0)

    print(f"\n{'='*72}")
    print(f"  Gram/det vs QR  --  Deep Difference Analysis")
    print(f"  Config : tile={TILE_SIZE}  stride={STRIDE}  em={NUM_ENDMEMBERS}  "
          f"{GRAM_TYPE}/{NORM_TYPE}  n_jobs={N_JOBS}")
    print(f"  Output : {OUT_DIR}")
    print(f"{'='*72}")

    # ------------------------------------------------------------------
    # 1. Synthetic test (clean data, all values in [0,1], no NaN)
    # ------------------------------------------------------------------
    print(f"\n{'--- Synthetic Data ---':^72}")
    syn_data = np.random.rand(10, 200, 200).astype(np.float32)
    gm_syn, t_g = _timed(process_volume_sliding_tile_parallel,
                          syn_data, TILE_SIZE, STRIDE, NUM_ENDMEMBERS,
                          GRAM_TYPE, NORM_TYPE, N_JOBS)
    qm_syn, t_q = _timed(process_volume_sliding_tile_parallel_QR,
                          syn_data, TILE_SIZE, STRIDE, NUM_ENDMEMBERS,
                          GRAM_TYPE, NORM_TYPE, N_JOBS)
    print(f"  Gram: {t_g:.2f}s    QR: {t_q:.2f}s    speed-up: {t_g/t_q:.2f}x")
    r_syn = _analyse_pair("Synthetic 10b 200x200", gm_syn, qm_syn, 200, 200, 100.0)
    _print_report(r_syn)
    _plot_frame(r_syn, "00_synthetic")

    # ------------------------------------------------------------------
    # 2. Real data frames
    # ------------------------------------------------------------------
    if not os.path.isfile(ARD_H5_PATH):
        print(f"\n[skip] Real-data file not found: {ARD_H5_PATH}")
        return

    print(f"\n{'--- Real HLS Surface Reflectance Data ---':^72}")
    frames = _load_real_frames(ARD_H5_PATH, ARD_GRIDS, MAX_REAL_FRAMES)
    print(f"  Loaded {len(frames)} frames.\n")

    all_reports = [r_syn]
    for i, fr in enumerate(frames):
        fd = fr['frame_data']
        h, w = fr['height'], fr['width']
        label = fr['label']

        print(f"  Processing frame {i+1}/{len(frames)} : {label} ...", flush=True)

        gm, t_g = _timed(process_volume_sliding_tile_parallel,
                          fd, TILE_SIZE, STRIDE, NUM_ENDMEMBERS,
                          GRAM_TYPE, NORM_TYPE, N_JOBS)
        qm, t_q = _timed(process_volume_sliding_tile_parallel_QR,
                          fd, TILE_SIZE, STRIDE, NUM_ENDMEMBERS,
                          GRAM_TYPE, NORM_TYPE, N_JOBS)
        print(f"    Gram:{t_g:.2f}s   QR:{t_q:.2f}s   speed-up:{t_g/t_q:.2f}x")

        r = _analyse_pair(label, gm, qm, h, w, fr['valid_pct'])
        _print_report(r)
        _plot_frame(r, f"{i+1:02d}_{fr['grid']}_{fr['local_idx']:03d}")
        all_reports.append(r)

    # ------------------------------------------------------------------
    # 3. Cross-frame aggregate
    # ------------------------------------------------------------------
    _aggregate_summary(all_reports)
    print(f"\n  All plots saved to: {OUT_DIR}\n")


if __name__ == '__main__':
    main()
