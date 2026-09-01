import os
import glob
import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timezone
from scipy import stats
import scienceplots
import sys
from pathlib import Path
script_dir = Path(__file__).resolve().parent
if str(script_dir.parent) not in sys.path:
    sys.path.insert(0, str(script_dir.parent))
import SpecComplex as sc

plt.style.use(['science', 'ieee', 'no-latex'])

# --- Configuration ---
ADD_SEASONAL_UNDERLAY = True  # Configuration toggle for meteorological seasonal background spans
LOCATIONS = ['Tait', 'Hurlingham', 'Malibu', 'Rochesterv2']


def apply_seasonal_underlay(axes, dates):
    if not ADD_SEASONAL_UNDERLAY or not dates:
        return
    if not isinstance(axes, (list, tuple, np.ndarray)):
        axes = [axes]
    min_year = min(d.year for d in dates)
    max_year = max(d.year for d in dates)
    
    # Scientifically curated desaturated hex codes at light opacity (alpha=0.15)
    # Preserves high luminance to maintain WCAG contrast against foreground satellite traces
    season_config = [
        (12, 3, '#D9D9D9', 'Winter'),  # Light gray
        (3,  6, '#A8E6CF', 'Spring'),  # Light green
        (6,  9, '#FFF275', 'Summer'),  # Yellow
        (9, 12, '#FFB74D', 'Fall')     # Orange
    ]
    for ax in axes:
        xlim = ax.get_xlim()
        for y in range(min_year - 1, max_year + 2):
            for start_m, end_m, color, label in season_config:
                if start_m == 12:
                    t0 = datetime(y - 1, 12, 1, tzinfo=timezone.utc)
                    t1 = datetime(y, 3, 1, tzinfo=timezone.utc)
                else:
                    t0 = datetime(y, start_m, 1, tzinfo=timezone.utc)
                    t1 = datetime(y, end_m, 1, tzinfo=timezone.utc)
                t0_num = mdates.date2num(t0)
                t1_num = mdates.date2num(t1)
                if t1_num < xlim[0] or t0_num > xlim[1]:
                    continue
                ax.axvspan(t0, t1, color=color, alpha=0.15, zorder=0, label='_nolegend_')
        ax.set_xlim(xlim)

def get_file_path(location):
    matches = glob.glob(os.path.join("C:/satelliteImagery/HLST30", f"HLST_{location}_Harmonized*SC_EM*.h5"))
    if matches:
        matches.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        return matches[0]
    return f"C:/satelliteImagery/HLST30/HLST_{location}_Harmonized_SC_EM-7_Norm-bandCount.h5"

def plot_global_stats(target_location=None, h5_path=None, location=None, metric='zscore'):
    if target_location is None and location is not None:
        target_location = location
    if h5_path is None:
        if target_location is None:
            raise ValueError("Either target_location or h5_path must be specified.")
        h5_path = get_file_path(target_location)

    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"Critical Data Integrity Error: Target HDF5 file not found at {h5_path}")

    with h5py.File(h5_path, 'r') as h5_file:
        if metric == 'zscore':
            dset_path = '/HDFEOS/GRIDS/HARMONIZED/Data Fields/sliding_volume_z_score'
        elif metric =="box_cox":
            dset_path = '/HDFEOS/GRIDS/HARMONIZED/Data Fields/sliding_volume_box_cox'
        else:
            dset_path = '/HDFEOS/GRIDS/HARMONIZED/Data Fields/sliding_volume_robust_scale'
            
        if dset_path not in h5_file:
            raise KeyError(f"Critical Data Integrity Error: {dset_path} not found in HDF5 file.")
        
        dset = h5_file[dset_path]
        attrs = dset.attrs
        
        times = attrs['acquisition_time']
        if metric == 'zscore':
            lambdas = None
            means = attrs['frame_global_means']
            stds = attrs['frame_global_stds']
        elif metric == "box_cox":
            lambdas = attrs['frame_global_lambdas']
            means = attrs['frame_global_means']
            stds = attrs['frame_global_stds']
        else:
            lambdas = attrs['frame_global_lambdas']
            means = attrs['frame_global_medians']
            stds = attrs['frame_global_iqrs']
            
        grids = attrs['source_grid']
        
        mask_dset_path = '/HDFEOS/GRIDS/HARMONIZED/Data Fields/common_mask'
        common_mask = h5_file[mask_dset_path] if mask_dset_path in h5_file else None

        total_frames = dset.shape[0]
        grids_str = [g if isinstance(g, str) else g.decode('utf-8') for g in grids]
        sensors = {'Landsat (HLSL30)': [], 'Sentinel (HLSS30)': [], 'Tanager': [], 'EnMAP': []}
        for i, g in enumerate(grids_str):
            gu = g.upper()
            if 'HLSL30' in gu:
                sensors['Landsat (HLSL30)'].append(i)
            elif 'HLSS30' in gu:
                sensors['Sentinel (HLSS30)'].append(i)
            elif 'ENMAP' in gu:
                sensors['EnMAP'].append(i)
            elif 'TANAGER' in gu:
                sensors['Tanager'].append(i)
            elif 'DRAGONETTE' in gu or 'WYVERN' in gu:
                sensors.setdefault('Dragonette', []).append(i)
            else:
                sensors.setdefault(g, []).append(i)

        # =========================================================================
        # EXACT POPULATION STREAMING (100% OF DATASET VOXELS ACROSS ALL FRAMES)
        # =========================================================================
        raw_dset_path = '/HDFEOS/GRIDS/HARMONIZED/Data Fields/sliding_volume_map'
        has_raw = raw_dset_path in h5_file
        raw_dset = h5_file[raw_dset_path] if has_raw else None

        # --- PASS 1: Exact Counts, Means, and Global Range over 100% of Pixels ---
        counts = np.zeros(total_frames, dtype=np.int64)
        z_sum = {s: 0.0 for s in sensors}
        z_count = {s: 0 for s in sensors}
        log_sum = {s: 0.0 for s in sensors}
        log_count = {s: 0 for s in sensors}
        global_min_log = np.inf
        global_max_log = -np.inf

        for f_idx in range(total_frames):
            grid_name = None
            for s_name, s_idx in sensors.items():
                if f_idx in s_idx:
                    grid_name = s_name
                    break
            
            m_arr = common_mask[f_idx, :, :] if common_mask is not None else None
            # Standardized score frame
            z_arr = sc.read_scaled_int16(dset, np.s_[f_idx, :, :])
            if m_arr is not None:
                valid_z = z_arr[(m_arr == 0) & ~np.isnan(z_arr)]
            else:
                valid_z = z_arr[~np.isnan(z_arr)]
            counts[f_idx] = len(valid_z)
            if len(valid_z) > 0 and grid_name:
                z_sum[grid_name] += np.sum(valid_z.astype(np.float64))
                z_count[grid_name] += len(valid_z)
                
            # Raw volume frame (log-transformed)
            if raw_dset is not None:
                raw_arr = sc.read_scaled_int16(raw_dset, np.s_[f_idx, :, :])
                if m_arr is not None:
                    valid_raw = raw_arr[(m_arr == 0) & ~np.isnan(raw_arr) & (raw_arr > 0)]
                else:
                    valid_raw = raw_arr[~np.isnan(raw_arr) & (raw_arr > 0)]
                if len(valid_raw) > 0 and grid_name:
                    log_v = np.log(valid_raw.astype(np.float64))
                    log_sum[grid_name] += np.sum(log_v)
                    log_count[grid_name] += len(log_v)
                    f_min, f_max = np.min(log_v), np.max(log_v)
                    if f_min < global_min_log: global_min_log = f_min
                    if f_max > global_max_log: global_max_log = f_max

        z_mean = {s: (z_sum[s] / z_count[s] if z_count[s] > 0 else np.nan) for s in sensors}
        log_mean = {s: (log_sum[s] / log_count[s] if log_count[s] > 0 else np.nan) for s in sensors}

        # Setup exact shared 90-bin histogram grids
        num_bins = 90
        bins_z = np.linspace(-4, 4, num_bins + 1)
        bin_centers_z = 0.5 * (bins_z[:-1] + bins_z[1:])
        bin_width_z = bins_z[1] - bins_z[0]

        if not np.isfinite(global_min_log): global_min_log = -25.0
        if not np.isfinite(global_max_log): global_max_log = 0.0
        bins_log = np.linspace(global_min_log, global_max_log, num_bins + 1)
        bin_centers_log = 0.5 * (bins_log[:-1] + bins_log[1:])
        bin_width_log = bins_log[1] - bins_log[0]

        # --- PASS 2: Exact Central Moments & Exact Histogram Bin Accumulation ---
        hist_z = {s: np.zeros(num_bins, dtype=np.int64) for s in sensors}
        hist_log = {s: np.zeros(num_bins, dtype=np.int64) for s in sensors}
        z_dev2 = {s: 0.0 for s in sensors}
        z_dev3 = {s: 0.0 for s in sensors}
        z_dev4 = {s: 0.0 for s in sensors}
        log_dev2 = {s: 0.0 for s in sensors}
        log_dev3 = {s: 0.0 for s in sensors}
        log_dev4 = {s: 0.0 for s in sensors}

        for f_idx in range(total_frames):
            grid_name = None
            for s_name, s_idx in sensors.items():
                if f_idx in s_idx:
                    grid_name = s_name
                    break
            
            m_arr = common_mask[f_idx, :, :] if common_mask is not None else None
            # Standardized score
            z_arr = sc.read_scaled_int16(dset, np.s_[f_idx, :, :])
            if m_arr is not None:
                valid_z = z_arr[(m_arr == 0) & ~np.isnan(z_arr)]
            else:
                valid_z = z_arr[~np.isnan(z_arr)]
            if len(valid_z) > 0 and grid_name:
                hz, _ = np.histogram(valid_z, bins=bins_z)
                hist_z[grid_name] += hz
                dev_z = valid_z.astype(np.float64) - z_mean[grid_name]
                z_dev2[grid_name] += np.sum(dev_z**2)
                z_dev3[grid_name] += np.sum(dev_z**3)
                z_dev4[grid_name] += np.sum(dev_z**4)

            # Raw volume
            if raw_dset is not None:
                raw_arr = sc.read_scaled_int16(raw_dset, np.s_[f_idx, :, :])
                if m_arr is not None:
                    valid_raw = raw_arr[(m_arr == 0) & ~np.isnan(raw_arr) & (raw_arr > 0)]
                else:
                    valid_raw = raw_arr[~np.isnan(raw_arr) & (raw_arr > 0)]
                if len(valid_raw) > 0 and grid_name:
                    log_v = np.log(valid_raw.astype(np.float64))
                    hl, _ = np.histogram(log_v, bins=bins_log)
                    hist_log[grid_name] += hl
                    dev_l = log_v - log_mean[grid_name]
                    log_dev2[grid_name] += np.sum(dev_l**2)
                    log_dev3[grid_name] += np.sum(dev_l**3)
                    log_dev4[grid_name] += np.sum(dev_l**4)

        # Compute exact population distribution statistics
        sensor_stats_z = {}
        for s in sensors:
            N = z_count[s]
            if N >= 4:
                var_z = z_dev2[s] / (N - 1)
                std_z = np.sqrt(max(0.0, var_z))
                skew_z = (z_dev3[s] / N) / (var_z ** 1.5) if var_z > 0 else 0.0
                kurt_z = ((z_dev4[s] / N) / (var_z ** 2) - 3.0) if var_z > 0 else 0.0

                # Exact Wasserstein distance to standard normal N(0, 1) using empirical CDF
                cdf_emp_z = np.cumsum(hist_z[s]) / N
                cdf_theo_z = stats.norm.cdf(bin_centers_z)
                wd_z = np.sum(np.abs(cdf_emp_z - cdf_theo_z)) * bin_width_z

                # Probability density histogram over 100% of data
                pdf_z = hist_z[s] / (N * bin_width_z)
                sensor_stats_z[s] = {
                    'N': N, 'mean': z_mean[s], 'std': std_z, 'skew': skew_z, 'kurt': kurt_z,
                    'wd': wd_z, 'pdf': pdf_z
                }

        sensor_stats_log = {}
        for s in sensors:
            N = log_count[s]
            if N >= 4:
                var_l = log_dev2[s] / (N - 1)
                std_l = np.sqrt(max(0.0, var_l))
                skew_l = (log_dev3[s] / N) / (var_l ** 1.5) if var_l > 0 else 0.0
                kurt_l = ((log_dev4[s] / N) / (var_l ** 2) - 3.0) if var_l > 0 else 0.0

                # Probability density histogram over 100% of data
                pdf_l = hist_log[s] / (N * bin_width_log)
                sensor_stats_log[s] = {
                    'N': N, 'mean': log_mean[s], 'std': std_l, 'skew': skew_l, 'kurt': kurt_l,
                    'pdf': pdf_l
                }

    # Parse timelines
    dates = [datetime.fromtimestamp(ts, tz=timezone.utc) for ts in times]

    # Style configuration
    clean_fname = os.path.basename(h5_path).replace('_', r'\_')
    if metric == 'zscore':
        fig, (ax_mean, ax_std, ax_count) = plt.subplots(3, 1, figsize=(14, 10.5), sharex=True, gridspec_kw={'height_ratios': [1.2, 1.0, 0.45]})
        ax_lambda = None
    else:
        fig, (ax_lambda, ax_mean, ax_std, ax_count) = plt.subplots(4, 1, figsize=(14, 13.5), sharex=True, gridspec_kw={'height_ratios': [1.0, 1.2, 1.0, 0.45]})
    
    title1 = r"Global Spatio-Temporal Scene Complexity Profile (Standardized Log-Volume $Z$-Score)" if metric == 'zscore' else (r"Global Spatio-Temporal Scene Complexity Profile (Box-Cox Transformed Volume)" if metric == 'box_cox' else r"Global Spatio-Temporal Scene Complexity Profile (Robust Scaled Volume)")
    fig.suptitle(
        title1 + "\n" +
        f"Source: {clean_fname}",
        fontsize=13, fontweight='bold', y=0.96
    )

    colors = {
        'Landsat (HLSL30)': '#1f77b4',
        'Sentinel (HLSS30)': '#2ca02c',
        'Tanager': '#d62728',
        'EnMAP': '#9467bd',
        'Dragonette': '#ff7f0e'
    }
    markers = {
        'Landsat (HLSL30)': '^',
        'Sentinel (HLSS30)': 'o',
        'Tanager': 's',
        'EnMAP': 'D',
        'Dragonette': 'p'
    }

    global_sort_idx = np.argsort(dates)
    global_dates = [dates[i] for i in global_sort_idx]
    global_means = [means[i] for i in global_sort_idx]
    global_stds = [stds[i] for i in global_sort_idx]
    
    # Plot continuous global time series as underlay
    ax_mean.plot(global_dates, global_means, color='gray', linestyle='-', linewidth=1.5, alpha=0.5, zorder=0, label='Global Temporal Trend')
    ax_std.plot(global_dates, global_stds, color='gray', linestyle='-', linewidth=1.5, alpha=0.5, zorder=0, label='Global Temporal Trend')
    if ax_lambda is not None:
        global_lambdas = [lambdas[i] for i in global_sort_idx]
        ax_lambda.plot(global_dates, global_lambdas, color='gray', linestyle='-', linewidth=1.5, alpha=0.5, zorder=0, label='Global Temporal Trend')

    for name, idxs in sensors.items():
        if not idxs: continue
        s_dates = [dates[i] for i in idxs]
        s_means = [means[i] for i in idxs]
        s_stds = [stds[i] for i in idxs]
        s_counts = [counts[i] for i in idxs]

        if ax_lambda is not None:
            s_lambdas = [lambdas[i] for i in idxs]
            ax_lambda.plot(s_dates, s_lambdas, marker=markers.get(name, 'X'), color=colors.get(name, '#555555'), label=f"{name} ($n={len(idxs)}$)",
                           linestyle='', markersize=4, alpha=0.7)

        ax_mean.plot(s_dates, s_means, marker=markers.get(name, 'X'), color=colors.get(name, '#555555'), label=f"{name} ($n={len(idxs)}$)",
                     linestyle='', markersize=4, alpha=0.7)
        ax_std.plot(s_dates, s_stds, marker=markers.get(name, 'X'), color=colors.get(name, '#555555'), label=f"{name} ($n={len(idxs)}$)",
                    linestyle='', markersize=4, alpha=0.7)
        ax_count.plot(s_dates, s_counts, marker=markers.get(name, 'X'), color=colors.get(name, '#555555'), label=f"{name} ($n={len(idxs)}$)",
                      linestyle='', markersize=3, alpha=0.7)

    axes_to_shade = [ax_mean, ax_std, ax_count]
    if ax_lambda is not None:
        axes_to_shade.append(ax_lambda)
    apply_seasonal_underlay(axes_to_shade, dates)

    if ax_lambda is not None:
        # Styling Lambda Panel
        ax_lambda.set_ylabel(r"Optimal Box-Cox ($\lambda$)", fontsize=11, fontweight='bold')
        ax_lambda.set_title("Variance Stabilization & Skewness Correction Parameter", fontsize=11, loc='left', pad=8)
        ax_lambda.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.95, fontsize=9)
        ax_lambda.grid(True, linestyle='--', alpha=0.4)

    # Styling Means Panel
    y_label_mean = r"Scene Mean Log Volume ($\mu_{\mathrm{global}}$)" if metric == 'zscore' else (r"Scene Mean Box-Cox Volume ($\mu_{\mathrm{global}}$)" if metric == 'box_cox' else r"Scene Median Transformed Volume ($\tilde{x}_{\mathrm{global}}$)")
    ax_mean.set_ylabel(y_label_mean, fontsize=11, fontweight='bold')
    ax_mean.set_title("Empirical Spatial Scene Mean Complexity across Multi-Sensor Timeline", fontsize=11, loc='left', pad=8)
    ax_mean.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.95, fontsize=9)
    ax_mean.grid(True, linestyle='--', alpha=0.4)

    # Styling Stds Panel
    y_label_std = r"Scene Std Dev Log Volume ($\sigma_{\mathrm{global}}$)" if metric == 'zscore' else (r"Scene Std Dev Box-Cox Volume ($\sigma_{\mathrm{global}}$)" if metric == 'box_cox' else r"Scene IQR Transformed Volume ($\mathrm{IQR}_{\mathrm{global}}$)")
    ax_std.set_ylabel(y_label_std, fontsize=11, fontweight='bold')
    ax_std.set_title(r"Empirical Spatial Scene Heterogeneity ($\sigma_{\mathrm{global}}$)", fontsize=11, loc='left', pad=8)
    ax_std.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.95, fontsize=9)
    ax_std.grid(True, linestyle='--', alpha=0.4)

    # Styling Counts Panel (Compact Bottom Panel)
    ax_count.set_ylabel(r"Valid Samples ($N$)", fontsize=10, fontweight='bold')
    ax_count.set_title("Valid Spatial Sample Coverage per Acquisition Frame", fontsize=10, loc='left', pad=6)
    ax_count.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.95, fontsize=8)
    ax_count.grid(True, linestyle='--', alpha=0.4)

    # X-axis formatting (applied to bottom shared panel)
    ax_count.xaxis.set_major_locator(mdates.YearLocator())
    ax_count.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax_count.tick_params(axis='x', rotation=0, labelsize=10)
    ax_count.set_xlabel("Acquisition Date", fontsize=11, fontweight='bold', labelpad=10)

    plt.tight_layout()
    plt.subplots_adjust(top=0.88, hspace=0.22)

    # --- Plot Window 2: Connected Multi-Sensor Series with Bounding Bars ---
    fig2, (ax_series, ax_count2) = plt.subplots(2, 1, figsize=(14, 8.5), sharex=True, gridspec_kw={'height_ratios': [1.2, 0.35]})
    title2 = r"Global Scene Mean Log Spectral Complexity (90\% of Values Interval)" if metric == 'zscore' else (r"Global Scene Mean Box-Cox Transformed Complexity (90\% of Values Interval)" if metric == 'box_cox' else r"Global Scene Median Transformed Complexity (90\% of Values Interval)")
    fig2.suptitle(
        title2 + "\n" +
        f"Source: {clean_fname}",
        fontsize=13, fontweight='bold', y=0.96
    )

    for name, idxs in sensors.items():
        if not idxs: continue
        # Sort indices chronologically to ensure valid line connections across temporal timeline
        sorted_idxs = sorted(idxs, key=lambda i: dates[i])
        s_dates = [dates[i] for i in sorted_idxs]
        s_means = [means[i] for i in sorted_idxs]
        s_stds = [stds[i] for i in sorted_idxs]
        s_counts = [counts[i] for i in sorted_idxs]

        # Bounding bars for ± 1.645 * sigma (parametric 90% spatial bounding range at frame)
        y_err = 1.645 * np.array(s_stds)

        # 1. Plot subdued background error bars (decoupled from main trajectory alpha)
        ax_series.errorbar(
            s_dates, s_means, yerr=y_err,
            fmt='none', ecolor=colors[name], elinewidth=0.6,
            capsize=0, alpha=0.22, zorder=1
        )
        ax_series.plot(
            s_dates, s_means,
            marker=markers[name], color=colors[name],
            label=f"{name} ({len(sorted_idxs)} frames)",
            linestyle='-', linewidth=1.5, markersize=4,
            alpha=0.9, zorder=2
        )
        # 2. Plot valid sample counts on bottom panel
        ax_count2.plot(
            s_dates, s_counts,
            marker=markers[name], color=colors[name],
            label=f"{name} ({len(sorted_idxs)} frames)",
            linestyle='', markersize=3, alpha=0.7
        )

    apply_seasonal_underlay([ax_series, ax_count2], dates)

    # Styling Plot Window 2 Main Panel
    y_label2_mean = r"Frame Mean Log(Spectral Complexity) (90\% of Values)" if metric == 'zscore' else (r"Frame Mean Box-Cox Complexity (90\% of Values)" if metric == 'box_cox' else r"Frame Median Transformed Complexity (90\% of Values)")
    ax_series.set_ylabel(y_label2_mean, fontsize=11, fontweight='bold')
    ax_series.set_title("Cross-Sensor Scene Complexity Dynamics across Multi-Sensor Timeline", fontsize=11, loc='left', pad=8)
    ax_series.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.95, fontsize=9)
    ax_series.grid(True, linestyle='--', alpha=0.4)

    # Styling Plot Window 2 Counts Panel
    ax_count2.set_ylabel(r"Frame Valid Pixel Count", fontsize=10, fontweight='bold')
    ax_count2.set_title("Valid Spatial Sample Coverage per Acquisition Frame", fontsize=10, loc='left', pad=6)
    ax_count2.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.95, fontsize=8)
    ax_count2.grid(True, linestyle='--', alpha=0.4)

    # X-axis formatting (applied to bottom shared panel)
    ax_count2.xaxis.set_major_locator(mdates.YearLocator())
    ax_count2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax_count2.tick_params(axis='x', rotation=0, labelsize=10)
    ax_count2.set_xlabel("Acquisition Date", fontsize=11, fontweight='bold', labelpad=10)

    fig2.tight_layout()
    fig2.subplots_adjust(top=0.88, hspace=0.22)

    # --- Plot Window 3: Log-Transformed Distribution Validation ---
    fig3 = plt.figure(figsize=(16, 9.5))
    fig3.suptitle(
        r"Empirical Log-Normal \& 90\% Parametric Interval Distribution Validation" + "\n" +
        f"Source: {clean_fname}",
        fontsize=13, fontweight='bold', y=0.96
    )

    # Right Panel: Dynamically stacked subplots for active sensors
    known_order = ['Landsat (HLSL30)', 'Sentinel (HLSS30)', 'Tanager', 'EnMAP', 'Dragonette']
    active_sensor_names = [s for s in known_order if s in sensors and len(sensors[s]) > 0]
    for s in sensors.keys():
        if s not in active_sensor_names and len(sensors[s]) > 0:
            active_sensor_names.append(s)
    
    if not active_sensor_names:
        active_sensor_names = [s for s in sensors.keys() if len(sensors[s]) > 0]
    
    num_active_sensors = max(1, len(active_sensor_names))
    gs3 = fig3.add_gridspec(num_active_sensors, 2, width_ratios=[1.1, 1.0], hspace=0.28, wspace=0.18)
    
    # Left Panel: Standardized Log Spectral Complexity (Z-Score) Density across sensors (Spans all rows)
    ax_dist_z = fig3.add_subplot(gs3[:, 0])
    z_grid = np.linspace(-4, 4, 200)
    norm_pdf = (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * z_grid**2)
    ax_dist_z.plot(z_grid, norm_pdf, 'k--', linewidth=2, label=r"Theoretical Normal $\mathcal{N}(0, 1)$", zorder=4)

    for s_name in active_sensor_names:
        if s_name not in sensor_stats_z: continue
        st = sensor_stats_z[s_name]
        label_str = fr"{s_name} (Skew: {st['skew']:.2f}, Kurt: {st['kurt']:.2f}, WD: {st['wd']:.3f})"
        s_color = colors.get(s_name, '#333333')
        ax_dist_z.plot(bin_centers_z, st['pdf'], color=s_color, linewidth=1.8, label=label_str, alpha=0.85)

    dist_title = "Standardized Log Spectral Complexity ($Z$-Score) Density" if metric == 'zscore' else ("Standardized Box-Cox Transformed Complexity Density" if metric == 'box_cox' else "Standardized Robust Scaled Complexity Density")
    ax_dist_z.set_title(dist_title, fontsize=11, fontweight='bold', pad=8)
    dist_xlabel = r"Standardized $Z$-Score ($\frac{\ln V - \mu}{\sigma}$)" if metric == 'zscore' else (r"Standardized Box-Cox $Z$-Score" if metric == 'box_cox' else r"Robust Scaled Volume")
    ax_dist_z.set_xlabel(dist_xlabel, fontsize=11, fontweight='bold')
    ax_dist_z.set_ylabel("Probability Density", fontsize=11, fontweight='bold')
    ax_dist_z.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.95, fontsize=9)
    ax_dist_z.grid(True, linestyle='--', alpha=0.4)
    ax_dist_z.set_xlim(-4, 4)

    axes_log = []
    x_min_log, x_max_log = bins_log[0], bins_log[-1]

    for idx, s_name in enumerate(active_sensor_names):
        ax = fig3.add_subplot(gs3[idx, 1], sharex=axes_log[0] if idx > 0 else None)
        axes_log.append(ax)

        s_color = colors.get(s_name, '#333333')
        if s_name in sensor_stats_log:
            stl = sensor_stats_log[s_name]
            mu_s = stl['mean']
            std_s = stl['std']
            skew_s = stl['skew']
            kurt_s = stl['kurt']
            n_px = stl['N']
            n_frames = len(sensors.get(s_name, []))

            ax.plot(bin_centers_log, stl['pdf'], color=s_color, linewidth=2.0, label=r"Empirical $\ln V$", alpha=0.9)

            grid_s = np.linspace(x_min_log, x_max_log, 200)
            fit_s = (1.0 / (std_s * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((grid_s - mu_s) / std_s)**2)
            ax.plot(grid_s, fit_s, 'k--', linewidth=1.5, label=r"Gaussian Fit $\mathcal{N}(\mu, \sigma^2)$", alpha=0.8)

            ax.axvline(mu_s - 1.645*std_s, color='gray', linestyle=':', linewidth=1.2)
            ax.axvline(mu_s + 1.645*std_s, color='gray', linestyle=':', linewidth=1.2)
            ax.axvspan(mu_s - 1.645*std_s, mu_s + 1.645*std_s, color='gray', alpha=0.15, label=r"90\% Interval")

            stats_text = (
                fr"$N = {n_px:,}$ pixels ($n = {n_frames}$ frames)" + "\n" +
                fr"$\mu = {mu_s:.2f}, \sigma = {std_s:.2f}$" + "\n" +
                fr"Skewness $= {skew_s:.2f}$" + "\n" +
                fr"Ex. Kurtosis $= {kurt_s:.2f}$"
            )
            ax.text(0.02, 0.92, stats_text, transform=ax.transAxes,
                    fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9, edgecolor='gray'))
            ax.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9, fontsize=8)
            ax.set_title(fr"{s_name} $\ln V$ Distribution ($N = {n_px:,}$ pixels)", fontsize=10, fontweight='bold', loc='left', pad=4)
        else:
            ax.text(0.5, 0.5, f"No Valid Acquisitions for {s_name} (N = 0)", transform=ax.transAxes,
                    ha='center', va='center', fontsize=10, fontstyle='italic', color='gray')
            ax.set_title(fr"{s_name} $\ln V$ Distribution ($N = 0$ pixels)", fontsize=10, fontweight='bold', loc='left', pad=4)
        ax.set_ylabel("Density", fontsize=9, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.4)
        if idx < num_active_sensors - 1:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel(r"Natural Log of Scene Complexity ($\ln V$)", fontsize=11, fontweight='bold')

    fig3.subplots_adjust(top=0.90, bottom=0.08, left=0.06, right=0.98, hspace=0.32, wspace=0.18)

    # Ensure output directories exist
    output_dir = os.path.dirname(h5_path)
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(h5_path))[0]

    out_local_path1 = os.path.join(output_dir, f"{base_name}_{metric}_global_stats.png")
    fig.savefig(out_local_path1, dpi=500, bbox_inches='tight')

    out_local_path2 = os.path.join(output_dir, f"{base_name}_{metric}_series_bounding_bars.png")
    fig2.savefig(out_local_path2, dpi=500, bbox_inches='tight')

    out_local_path3 = os.path.join(output_dir, f"{base_name}_{metric}_distributions.png")
    fig3.savefig(out_local_path3, dpi=500, bbox_inches='tight')

    plt.close(fig)
    plt.close(fig2)
    plt.close(fig3)

    print(
        f"Successfully created global statistics plots for '{base_name}' at:\n"
        f" [Plot 1 - Global Stats Panel]: {out_local_path1}\n"
        f" [Plot 2 - Series & Bounding Bars]: {out_local_path2}\n"
        f" [Plot 3 - Log-Normal Validation]: {out_local_path3}\n"
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot sliding volume global stats")
    parser.add_argument('--file', '-f', type=str, default=None, help="Path to specific HDF5 file")
    parser.add_argument('--location', '-l', type=str, default=None, help="Target location name")
    parser.add_argument('--metric', '-m', type=str, default=None, help="Metric to plot ('zscore', 'robust', 'box_cox', or 'all')")
    args = parser.parse_args()

    if args.metric:
        if args.metric.lower() == 'all':
            metrics_to_plot = ['zscore', 'robust', 'box_cox']
        else:
            metrics_to_plot = [args.metric]
    else:
        metrics_to_plot = ['zscore', 'robust', 'box_cox']

    if args.file or args.location:
        for m in metrics_to_plot:
            plot_global_stats(target_location=args.location, h5_path=args.file, metric=m)
    else:
        print(f"Starting batch processing across {len(LOCATIONS)} locations: {LOCATIONS}")
        for loc in LOCATIONS:
            print(f"\n========================================\nProcessing location: {loc} ...")
            for m in metrics_to_plot:
                plot_global_stats(target_location=loc, metric=m)
        print("\n========================================\nBatch processing completed successfully.")
