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
plt.style.use(['science', 'ieee'])

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

def plot_global_stats(target_location=None, h5_path=None, location=None):
    if target_location is None and location is not None:
        target_location = location
    if h5_path is None:
        if target_location is None:
            raise ValueError("Either target_location or h5_path must be specified.")
        h5_path = get_file_path(target_location)

    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"Critical Data Integrity Error: Target HDF5 file not found at {h5_path}")

    with h5py.File(h5_path, 'r') as h5_file:
        dset_path = '/HDFEOS/GRIDS/HARMONIZED/Data Fields/sliding_volume_z_score'
        if dset_path not in h5_file:
            raise KeyError(f"Critical Data Integrity Error: {dset_path} not found in HDF5 file.")
        
        dset = h5_file[dset_path]
        attrs = dset.attrs
        
        times = attrs['acquisition_time']
        means = attrs['frame_global_means']
        stds = attrs['frame_global_stds']
        grids = attrs['source_grid']
        
        # Calculate valid (unmasked / non-NaN) pixel counts per frame
        counts = np.sum(~np.isnan(dset[:]), axis=(1, 2))

        grids_str = [g if isinstance(g, str) else g.decode('utf-8') for g in grids]
        sensors = {'Landsat (HLSL30)': [], 'Sentinel (HLSS30)': [], 'Tanager': []}
        for i, g in enumerate(grids_str):
            gu = g.upper()
            if 'HLSL30' in gu:
                sensors['Landsat (HLSL30)'].append(i)
            elif 'HLSS30' in gu:
                sensors['Sentinel (HLSS30)'].append(i)
            else:
                sensors['Tanager'].append(i)

        # Extract strided voxel samples per sensor for log-normal distribution validation
        sensor_z_samples = {}
        for s_name, s_idx in sensors.items():
            if not s_idx: continue
            step = 1#max(1, len(s_idx))
            sampled_frames = s_idx[::step]
            arr = dset[sampled_frames]
            valid_z = arr[~np.isnan(arr)]
            if len(valid_z) > 0:
                sensor_z_samples[s_name] = valid_z

        # Also extract raw sliding_volume_map samples per sensor (masked by common_mask) to validate log-transformation
        raw_dset_path = '/HDFEOS/GRIDS/HARMONIZED/Data Fields/sliding_volume_map'
        mask_dset_path = '/HDFEOS/GRIDS/HARMONIZED/Data Fields/common_mask'
        sensor_raw_log_samples = {}
        all_raw_log_list = []
        if raw_dset_path in h5_file and mask_dset_path in h5_file:
            raw_dset = h5_file[raw_dset_path]
            mask_dset = h5_file[mask_dset_path]
            for s_name, s_idx in sensors.items():
                if not s_idx: continue
                step = 1#max(1, len(s_idx))
                sampled_frames = s_idx[::step]
                raw_arr = raw_dset[sampled_frames]
                mask_arr = mask_dset[sampled_frames]
                valid_raw = raw_arr[(mask_arr == 0) & ~np.isnan(raw_arr) & (raw_arr > 0)]
                if len(valid_raw) > 0:
                    log_v = np.log(valid_raw)
                    sensor_raw_log_samples[s_name] = log_v
                    all_raw_log_list.append(log_v)
        raw_log_samples = np.concatenate(all_raw_log_list) if all_raw_log_list else np.array([])

    # Parse timelines
    dates = [datetime.fromtimestamp(ts, tz=timezone.utc) for ts in times]

    # Style configuration
    clean_fname = os.path.basename(h5_path).replace('_', r'\_')
    fig, (ax_mean, ax_std, ax_count) = plt.subplots(3, 1, figsize=(14, 10.5), sharex=True, gridspec_kw={'height_ratios': [1.2, 1.0, 0.45]})
    
    fig.suptitle(
        r"Global Spatio-Temporal Scene Complexity Profile (Standardized Log-Volume $Z$-Score)" + "\n" +
        f"Source: {clean_fname}",
        fontsize=13, fontweight='bold', y=0.96
    )

    colors = {
        'Landsat (HLSL30)': '#1f77b4',
        'Sentinel (HLSS30)': '#2ca02c',
        'Tanager': '#d62728'
    }
    markers = {
        'Landsat (HLSL30)': '^',
        'Sentinel (HLSS30)': 'o',
        'Tanager': 's'
    }

    for name, idxs in sensors.items():
        if not idxs: continue
        s_dates = [dates[i] for i in idxs]
        s_means = [means[i] for i in idxs]
        s_stds = [stds[i] for i in idxs]
        s_counts = [counts[i] for i in idxs]

        ax_mean.plot(s_dates, s_means, marker=markers[name], color=colors[name], label=f"{name} ($n={len(idxs)}$)",
                     linestyle='', markersize=4, alpha=0.7)
        ax_std.plot(s_dates, s_stds, marker=markers[name], color=colors[name], label=f"{name} ($n={len(idxs)}$)",
                    linestyle='', markersize=4, alpha=0.7)
        ax_count.plot(s_dates, s_counts, marker=markers[name], color=colors[name], label=f"{name} ($n={len(idxs)}$)",
                      linestyle='', markersize=3, alpha=0.7)

    apply_seasonal_underlay([ax_mean, ax_std, ax_count], dates)

    # Styling Means Panel
    ax_mean.set_ylabel(r"Scene Mean Log Volume ($\mu_{\mathrm{global}}$)", fontsize=11, fontweight='bold')
    ax_mean.set_title("Empirical Spatial Scene Mean Complexity across Multi-Sensor Timeline", fontsize=11, loc='left', pad=8)
    ax_mean.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.95, fontsize=9)
    ax_mean.grid(True, linestyle='--', alpha=0.4)

    # Styling Stds Panel
    ax_std.set_ylabel(r"Scene Std Dev Log Volume ($\sigma_{\mathrm{global}}$)", fontsize=11, fontweight='bold')
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
    fig2.suptitle(
        r"Global Scene Mean Log Spectral Complexity (90\% of Values Interval)" + "\n" +
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
    ax_series.set_ylabel(r"Frame Mean Log(Spectral Complexity) (90\% of Values)", fontsize=11, fontweight='bold')
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
    gs3 = fig3.add_gridspec(3, 2, width_ratios=[1.1, 1.0], hspace=0.28, wspace=0.18)
    fig3.suptitle(
        r"Empirical Log-Normal \& 90\% Parametric Interval Distribution Validation" + "\n" +
        f"Source: {clean_fname}",
        fontsize=13, fontweight='bold', y=0.96
    )

    # Left Panel: Standardized Log Spectral Complexity (Z-Score) Density across sensors (Spans all 3 rows)
    ax_dist_z = fig3.add_subplot(gs3[:, 0])
    z_grid = np.linspace(-4, 4, 200)
    norm_pdf = (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * z_grid**2)
    ax_dist_z.plot(z_grid, norm_pdf, 'k--', linewidth=2, label=r"Theoretical Normal $\mathcal{N}(0, 1)$", zorder=4)

    # Shaded 90% parametric interval [-1.645, +1.645]
    ax_dist_z.axvline(-1.645, color='gray', linestyle=':', linewidth=1.5)
    ax_dist_z.axvline(1.645, color='gray', linestyle=':', linewidth=1.5)
    ax_dist_z.axvspan(-1.645, 1.645, color='gray', alpha=0.15, label=r"Parametric 90\% Interval ($\pm 1.645\sigma$)")

    for s_name, s_vals in sensor_z_samples.items():
        if len(s_vals) == 0: continue
        emp_cov = np.mean((s_vals >= -1.645) & (s_vals <= 1.645)) * 100.0
        counts_hist, bins = np.histogram(s_vals, bins=80, range=(-4, 4), density=True)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        ax_dist_z.plot(bin_centers, counts_hist, color=colors[s_name], linewidth=1.8, label=fr"{s_name} (Cov: {emp_cov:.1f}\%)", alpha=0.85)

    ax_dist_z.set_title("Standardized Log Spectral Complexity ($Z$-Score) Density", fontsize=11, fontweight='bold', pad=8)
    ax_dist_z.set_xlabel(r"Standardized $Z$-Score ($\frac{\ln V - \mu}{\sigma}$)", fontsize=11, fontweight='bold')
    ax_dist_z.set_ylabel("Probability Density", fontsize=11, fontweight='bold')
    ax_dist_z.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.95, fontsize=9)
    ax_dist_z.grid(True, linestyle='--', alpha=0.4)
    ax_dist_z.set_xlim(-4, 4)

    # Right Panel: 3 Vertically Stacked Subplots separated by sensor source
    sensor_names_order = ['Landsat (HLSL30)', 'Sentinel (HLSS30)', 'Tanager']
    axes_log = []

    # Determine global x-range across all log samples for consistent bin alignment
    if len(raw_log_samples) > 0:
        _, global_bins = np.histogram(raw_log_samples, bins=80)
        x_min_log, x_max_log = global_bins[0], global_bins[-1]
    else:
        x_min_log, x_max_log = -15, 15

    for idx, s_name in enumerate(sensor_names_order):
        ax = fig3.add_subplot(gs3[idx, 1], sharex=axes_log[0] if idx > 0 else None)
        axes_log.append(ax)

        if s_name in sensor_raw_log_samples and len(sensor_raw_log_samples[s_name]) > 0:
            s_vals = sensor_raw_log_samples[s_name]
            mu_s = np.mean(s_vals)
            std_s = np.std(s_vals)
            skew_s = stats.skew(s_vals)
            kurt_s = stats.kurtosis(s_vals)

            counts_s, bins_s = np.histogram(s_vals, bins=60, range=(x_min_log, x_max_log), density=True)
            bin_c_s = 0.5 * (bins_s[:-1] + bins_s[1:])
            ax.plot(bin_c_s, counts_s, color=colors[s_name], linewidth=2.0, label=r"Empirical $\ln V$", alpha=0.9)

            grid_s = np.linspace(x_min_log, x_max_log, 200)
            fit_s = (1.0 / (std_s * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((grid_s - mu_s) / std_s)**2)
            ax.plot(grid_s, fit_s, 'k--', linewidth=1.5, label=r"Gaussian Fit $\mathcal{N}(\mu, \sigma^2)$", alpha=0.8)

            ax.axvline(mu_s - 1.645*std_s, color='gray', linestyle=':', linewidth=1.2)
            ax.axvline(mu_s + 1.645*std_s, color='gray', linestyle=':', linewidth=1.2)
            ax.axvspan(mu_s - 1.645*std_s, mu_s + 1.645*std_s, color='gray', alpha=0.15, label=r"90\% Interval")

            stats_text = (
                fr"$\mu = {mu_s:.2f}, \sigma = {std_s:.2f}$" + "\n" +
                fr"Skewness $= {skew_s:.2f}$" + "\n" +
                fr"Ex. Kurtosis $= {kurt_s:.2f}$"
            )
            ax.text(0.02, 0.92, stats_text, transform=ax.transAxes,
                    fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9, edgecolor='gray'))
            ax.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9, fontsize=8)
        else:
            ax.text(0.5, 0.5, f"No Valid Acquisitions for {s_name}", transform=ax.transAxes,
                    ha='center', va='center', fontsize=10, fontstyle='italic', color='gray')

        ax.set_title(fr"{s_name} $\ln V$ Distribution", fontsize=10, fontweight='bold', loc='left', pad=4)
        ax.set_ylabel("Density", fontsize=9, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.4)
        if idx < 2:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel(r"Natural Log of Scene Complexity ($\ln V$)", fontsize=11, fontweight='bold')

    fig3.subplots_adjust(top=0.90, bottom=0.08, left=0.06, right=0.98, hspace=0.32, wspace=0.18)

    # Ensure output directories exist
    output_dir = os.path.dirname(h5_path)
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(h5_path))[0]

    out_local_path1 = os.path.join(output_dir, f"{base_name}_zscore_global_stats.png")
    fig.savefig(out_local_path1, dpi=500, bbox_inches='tight')

    out_local_path2 = os.path.join(output_dir, f"{base_name}_zscore_series_bounding_bars.png")
    fig2.savefig(out_local_path2, dpi=500, bbox_inches='tight')

    out_local_path3 = os.path.join(output_dir, f"{base_name}_lognormal_distributions.png")
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
    args = parser.parse_args()

    if args.file or args.location:
        plot_global_stats(target_location=args.location, h5_path=args.file)
    else:
        print(f"Starting batch processing across {len(LOCATIONS)} locations: {LOCATIONS}")
        for loc in LOCATIONS:
            print(f"\n========================================\nProcessing location: {loc} ...")
            plot_global_stats(target_location=loc)
        print("\n========================================\nBatch processing completed successfully.")
