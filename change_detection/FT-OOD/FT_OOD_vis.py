import h5py
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timezone
import os
import glob
import pyproj
import matplotlib.patches as patches
import scienceplots
import warnings
import copy
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates

plt.style.use(['science', 'no-latex'])

LOCATION = "Tait"
H5_PATH = f"C:/satelliteImagery/HLST30/HLST_{LOCATION}_Harmonized_SC_EM-7_Norm-bandCount.h5"

# Colormap for window attribution (dynamically sized to WINDOWS length)
WINDOW_COLORS = ['#1f77b4', '#2ca02c', '#d62728', '#ff7f0e', '#9467bd',
                 '#8c564b', '#e377c2', '#7f7f7f']


def get_inference_h5(location):
    search_pattern = f"C:/satelliteImagery/HLST30/FT-OOD/FT-OOD_{location}_results_*.h5"
    files = glob.glob(search_pattern)
    if not files:
        return None
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]


def plot_pixel_sits(pixel_y, pixel_x, source_h5_path, inference_results_h5,
                    ax_ts_z, ax_ts_s, ax_ts_attr, ax_ts_f, ax_ts_a,
                    current_date=None):
    """Plot time series for a selected pixel across all subplot axes."""
    ax_ts_z.clear()
    ax_ts_s.clear()
    ax_ts_attr.clear()
    ax_ts_f.clear()
    if ax_ts_a is not None:
        ax_ts_a.clear()

    lat, lon = None, None
    with h5py.File(source_h5_path, 'r') as f:
        harm_grp = f['/HDFEOS/GRIDS/HARMONIZED/Data Fields']
        target_metric = 'sliding_volume_z_score'

        has_common_mask = 'common_mask' in harm_grp

        acq_time = harm_grp[target_metric].attrs['acquisition_time'][:]
        z_score = harm_grp[target_metric][:, pixel_y, pixel_x]

        if has_common_mask:
            unified_masks = harm_grp['common_mask'][:, pixel_y, pixel_x]
            is_invalid = unified_masks.astype(bool) | np.isnan(z_score)
        else:
            is_invalid = np.isnan(z_score)

        spacecraft_bytes = harm_grp[target_metric].attrs['source_spacecraft'][:]
        spacecrafts = [s.decode('utf-8') if isinstance(s, bytes) else str(s) for s in spacecraft_bytes]

        geo_transform = harm_grp[target_metric].attrs.get('GeoTransform')
        spatial_ref = harm_grp[target_metric].attrs.get('spatial_ref')
        if geo_transform is not None and spatial_ref is not None:
            try:
                gt = geo_transform
                x_geo = gt[0] + (pixel_x + 0.5) * gt[1] + (pixel_y + 0.5) * gt[2]
                y_geo = gt[3] + (pixel_x + 0.5) * gt[4] + (pixel_y + 0.5) * gt[5]
                spatial_ref_str = spatial_ref.decode('utf-8') if isinstance(spatial_ref, bytes) else str(spatial_ref)
                crs = pyproj.CRS.from_wkt(spatial_ref_str)
                transformer = pyproj.Transformer.from_crs(crs, "epsg:4326", always_xy=True)
                lon, lat = transformer.transform(x_geo, y_geo)
            except Exception:
                pass

    dates = [datetime.fromtimestamp(ts, timezone.utc) for ts in acq_time]

    # ── Load FT-OOD Inference Data ──
    with h5py.File(inference_results_h5, 'r') as f:
        anomaly_scores = f['anomaly_scores'][:, pixel_y, pixel_x]
        drift_status = f['drift_status'][:, pixel_y, pixel_x]
        change_ts = f['first_drift_timestamp'][pixel_y, pixel_x]

        windows = list(f.attrs.get('WINDOWS', [0.5, 1.0, 3.0]))
        k_freqs = int(f.attrs.get('K_FREQUENCIES', 2))
        warning_sigma = float(f.attrs.get('WARNING_SIGMA', 2.0))

        alft_feats = f['alft_features'][:, pixel_y, pixel_x, :]   # (N, ALFT_DIM)
        alft_freqs = f['alft_frequencies'][:, pixel_y, pixel_x, :, :]  # (N, W, K)

        has_xai = 'window_attribution' in f
        if has_xai:
            win_attr = f['window_attribution'][:, pixel_y, pixel_x, :]  # (N, num_windows)
            xai_comp = f['xai_complexity'][:, pixel_y, pixel_x]        # (N,)
        else:
            win_attr = None
            xai_comp = None

    num_windows = len(windows)
    dates_arr = np.array(dates)
    spacecrafts_arr = np.array(spacecrafts)
    frac_years = np.array([d.year + (d.timetuple().tm_yday - 1) / 365.25 for d in dates])

    # ─────────────────────────────────────────
    # Plot 1: Z-Score & Reconstructed Baseline
    # ─────────────────────────────────────────
    for marker_type, sc_keyword in [('s', 'Sentinel'), ('o', 'Landsat'), ('D', 'Tanager')]:
        sc_mask = np.array([sc_keyword.lower() in str(sc).lower() for sc in spacecrafts_arr])

        idx_valid = (~is_invalid) & sc_mask
        if np.any(idx_valid):
            ax_ts_z.plot(dates_arr[idx_valid], z_score[idx_valid], color='k', marker=marker_type,
                         linestyle='None', label=f'Valid ({sc_keyword})')

        idx_inv = is_invalid & sc_mask
        if np.any(idx_inv):
            ax_ts_z.plot(dates_arr[idx_inv], z_score[idx_inv], color='gray', marker=marker_type,
                         linestyle='None', markerfacecolor='none', label=f'Invalid ({sc_keyword})')

    # Reconstruct Harmonic Baseline for longest window
    w_idx = num_windows - 1
    fpw = 2 * k_freqs + 2
    offset = w_idx * fpw

    pred_mask = ~np.isnan(alft_feats[:, 0])
    if np.any(pred_mask):
        pd_dates = dates_arr[pred_mask]
        pd_fracs = frac_years[pred_mask]

        beta_0 = alft_feats[pred_mask, offset]
        beta_cos = alft_feats[pred_mask, offset + 1: offset + 1 + k_freqs]
        beta_sin = alft_feats[pred_mask, offset + 1 + k_freqs: offset + 1 + 2 * k_freqs]
        sigma = alft_feats[pred_mask, offset + 1 + 2 * k_freqs]
        omegas = alft_freqs[pred_mask, w_idx, :]

        y_pred = beta_0 + np.sum(beta_cos * np.cos(omegas * pd_fracs[:, None]) +
                                 beta_sin * np.sin(omegas * pd_fracs[:, None]), axis=1)

        upper_bound = y_pred + warning_sigma * sigma
        lower_bound = y_pred - warning_sigma * sigma

        ax_ts_z.plot(pd_dates, y_pred, 'b--', label=f'Baseline (W={windows[-1]}yr)')
        ax_ts_z.fill_between(pd_dates, lower_bound, upper_bound, color='blue', alpha=0.15,
                             label=f'±{warning_sigma}σ Robust Bound')

        warning_mask = (drift_status == 1)
        if np.any(warning_mask):
            ax_ts_z.scatter(dates_arr[warning_mask], z_score[warning_mask], color='orange',
                            s=40, zorder=4, label='Warning')

        drift_mask = (drift_status == 2)
        if np.any(drift_mask):
            ax_ts_z.scatter(dates_arr[drift_mask], z_score[drift_mask], color='darkred',
                            marker='*', s=150, zorder=5, label='Confirmed Drift')

        if not np.isnan(change_ts):
            c_dt = datetime.fromtimestamp(change_ts, timezone.utc)
            ax_ts_z.axvline(x=c_dt, color='red', linestyle='-.', alpha=0.6, label='First Drift Event')

    if current_date is not None:
        ax_ts_z.axvline(x=current_date, color='magenta', linestyle='--', alpha=0.5, label='Displayed Frame')

    title_str = f"Pixel: ({pixel_x}, {pixel_y})"
    if lat is not None and lon is not None:
        title_str += f" | Lat: {lat:.5f}, Lon: {lon:.5f}"
    ax_ts_z.set_title(title_str)
    ax_ts_z.set_ylabel('Sliding Volume Z-Score')
    ax_ts_z.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    ax_ts_z.grid(True)
    ax_ts_z.set_ylim([-4, 4])

    # ─────────────────────────────────────────
    # Plot 2: Deep SVDD Anomaly Score
    # ─────────────────────────────────────────
    ax_ts_s.plot(dates_arr, anomaly_scores, color='purple', marker='.', linestyle='-', label='OOD Score')
    ax_ts_s.set_ylabel('Deep SVDD Score')
    ax_ts_s.grid(True)
    ax_ts_s.legend(bbox_to_anchor=(1.01, 1), loc='upper left')

    for i in range(len(dates_arr) - 1):
        stat = drift_status[i]
        color = 'white'
        if stat == 3: color = 'lightgray'
        elif stat == 1: color = '#ffeb9c'
        elif stat == 2: color = '#ffc7ce'
        if color != 'white':
            ax_ts_s.axvspan(dates_arr[i], dates_arr[i + 1], color=color, alpha=0.3, lw=0)

    if current_date is not None:
        ax_ts_s.axvline(x=current_date, color='magenta', linestyle='--', alpha=0.5)

    # ─────────────────────────────────────────
    # Plot 3: Window Attribution Stacked Bars
    # ─────────────────────────────────────────
    if has_xai and win_attr is not None:
        # Only plot at timesteps with attribution data
        attr_mask = ~np.all(np.isnan(win_attr), axis=1)
        if np.any(attr_mask):
            attr_dates = dates_arr[attr_mask]
            attr_vals = win_attr[attr_mask, :]  # (n_anom, num_windows)

            # Normalize to fractional attribution per timestep
            attr_totals = np.nansum(attr_vals, axis=1, keepdims=True)
            attr_totals[attr_totals == 0] = 1.0  # avoid division by zero
            attr_frac = attr_vals / attr_totals

            # Stacked bar chart
            bar_width = 2.0  # days
            bottom = np.zeros(len(attr_dates))
            for w in range(num_windows):
                c = WINDOW_COLORS[w % len(WINDOW_COLORS)]
                ax_ts_attr.bar(attr_dates, attr_frac[:, w], width=bar_width,
                               bottom=bottom, color=c, label=f'{windows[w]}yr',
                               edgecolor='none')
                bottom += attr_frac[:, w]

            ax_ts_attr.set_ylabel('Window\nAttribution')
            ax_ts_attr.set_ylim([0, 1])
            ax_ts_attr.legend(bbox_to_anchor=(1.01, 1), loc='upper left', title='Window')
            ax_ts_attr.grid(True, axis='y')

            # Overlay Complexity as line on twin axis
            if xai_comp is not None:
                comp_mask = ~np.isnan(xai_comp) & attr_mask
                if np.any(comp_mask):
                    ax_comp_twin = ax_ts_attr.twinx()
                    ax_comp_twin.plot(dates_arr[comp_mask], xai_comp[comp_mask],
                                     color='black', marker='x', linestyle=':', alpha=0.6,
                                     label='Complexity')
                    ax_comp_twin.set_ylabel('Complexity', color='black')
                    ax_comp_twin.tick_params(axis='y', labelcolor='black')
        else:
            ax_ts_attr.text(0.5, 0.5, 'No attributions for this pixel',
                            ha='center', va='center', transform=ax_ts_attr.transAxes)
            ax_ts_attr.set_ylabel('Window\nAttribution')
    else:
        ax_ts_attr.text(0.5, 0.5, 'XAI not enabled for this run',
                        ha='center', va='center', transform=ax_ts_attr.transAxes)
        ax_ts_attr.set_ylabel('Window\nAttribution')

    # ─────────────────────────────────────────
    # Plot 4: Dominant Frequencies (Periods)
    # ─────────────────────────────────────────
    if np.any(pred_mask):
        omegas_all = alft_freqs[pred_mask, w_idx, :]
        periods = (2.0 * np.pi) / omegas_all

        beta_cos_p = alft_feats[pred_mask, offset + 1: offset + 1 + k_freqs]
        beta_sin_p = alft_feats[pred_mask, offset + 1 + k_freqs: offset + 1 + 2 * k_freqs]
        amps = np.sqrt(beta_cos_p ** 2 + beta_sin_p ** 2)

        colors = ['tab:blue', 'tab:orange', 'tab:green']
        for k in range(k_freqs):
            c = colors[k] if k < len(colors) else 'tab:red'
            valid_p = periods[:, k]
            med_days = np.nanmedian(valid_p) * 365.25

            ax_ts_f.plot(pd_dates, valid_p * 365.25, marker='.', linestyle='-', color=c,
                         label=f'Top-{k + 1} Period (~{med_days:.0f} d)')
            if ax_ts_a is not None:
                ax_ts_a.plot(pd_dates, amps[:, k], marker='x', linestyle=':', color=c, alpha=0.4)

    if current_date is not None:
        ax_ts_f.axvline(x=current_date, color='magenta', linestyle='--', alpha=0.5)

    ax_ts_f.set_ylabel('Dominant Period (Days)')
    ax_ts_f.set_xlabel('Date')
    ax_ts_f.grid(True)
    ax_ts_f.legend(bbox_to_anchor=(1.01, 1), loc='upper left')

    if ax_ts_a is not None:
        ax_ts_a.yaxis.tick_right()
        ax_ts_a.yaxis.set_label_position("right")
        ax_ts_a.set_ylabel('Amplitude (Z-Score)', color='gray')
        ax_ts_a.tick_params(axis='y', labelcolor='gray')


def plot_spatial_anomaly_overlay(source_h5_path, inference_results_h5):
    with h5py.File(source_h5_path, 'r') as f:
        harm_grp = f['/HDFEOS/GRIDS/HARMONIZED/Data Fields']
        target_metric = 'sliding_volume_z_score'
        acq_time = harm_grp[target_metric].attrs['acquisition_time'][:]

        has_common_mask = 'common_mask' in harm_grp
        if has_common_mask:
            unified_masks = harm_grp['common_mask'][:]
            full_valid_mask = ~unified_masks.astype(bool)
        else:
            full_valid_mask = ~np.isnan(harm_grp[target_metric][:])

    def get_ortho(idx):
        with h5py.File(source_h5_path, 'r') as f:
            harm_grp = f['/HDFEOS/GRIDS/HARMONIZED/Data Fields']
            target_metric = 'sliding_volume_z_score'
            spc = harm_grp[target_metric].attrs['source_spacecraft'][idx]
            spc = spc.decode('utf-8') if isinstance(spc, bytes) else str(spc)
            o = harm_grp['ortho_visual'][idx]
            o = np.transpose(o, (1, 2, 0)).astype(np.float32) / 255.0
            valid_mask = np.all(o > 0, axis=-1)
            o[~valid_mask] = 0.0
            return o, spc

    dates = [datetime.fromtimestamp(ts, timezone.utc) for ts in acq_time]
    target_date = datetime(2025, 9, 12, tzinfo=timezone.utc).date()
    diffs = [abs((d.date() - target_date).days) for d in dates]
    base_idx = np.argmin(diffs)
    base_frame, base_sg = get_ortho(base_idx)
    base_date = datetime.fromtimestamp(acq_time[base_idx], timezone.utc)

    with h5py.File(inference_results_h5, 'r') as f:
        anomaly_map = f['first_drift_timestamp'][:]
        change_count_map = f['drift_count'][:]
        anomaly_scores = f['anomaly_scores'][:]
        drift_status = f['drift_status'][:]

        windows = list(f.attrs.get('WINDOWS', [0.5, 1.0, 3.0]))
        k_freqs = int(f.attrs.get('K_FREQUENCIES', 2))
        alft_freqs = f['alft_frequencies'][:]
        alft_feats = f['alft_features'][:]

        has_xai = 'window_attribution' in f
        if has_xai:
            win_attr_all = f['window_attribution'][:]
            xai_comp_all = f['xai_complexity'][:]
        else:
            win_attr_all = None
            xai_comp_all = None

    num_windows = len(windows)
    H, W = full_valid_mask.shape[1], full_valid_mask.shape[2]
    anomaly_map[change_count_map == 0] = np.nan

    # Pre-calculate maps
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        max_score_map = np.nanmax(anomaly_scores, axis=0)

        w_idx = num_windows - 1
        fpw = 2 * k_freqs + 2
        offset = w_idx * fpw

        top1_freq = alft_freqs[:, :, :, w_idx, 0]
        top1_period = (2.0 * np.pi) / top1_freq
        mean_period_map = np.nanmean(top1_period, axis=0)

        sigma_series = alft_feats[:, :, :, offset + 1 + 2 * k_freqs]
        mean_uncertainty = np.nanmean(sigma_series, axis=0)

        # Dominant Window Map: which window has highest mean attribution
        if has_xai and win_attr_all is not None:
            mean_win_attr = np.nanmean(win_attr_all, axis=0)  # (H, W, num_windows)
            dominant_window_map = np.full((H, W), np.nan, dtype=np.float32)
            any_attr = np.any(~np.isnan(mean_win_attr), axis=-1)
            dominant_window_map[any_attr] = np.nanargmax(mean_win_attr[any_attr], axis=-1).astype(np.float32)

            # Mean Complexity map
            mean_complexity_map = np.nanmean(xai_comp_all, axis=0)
        else:
            dominant_window_map = None
            mean_complexity_map = None

    # ─────────────────────────────────────────
    # Window 1: Main Analysis (4 rows of time series)
    # ─────────────────────────────────────────
    fig1 = plt.figure(figsize=(20, 12))
    fig1.canvas.manager.set_window_title(f'FT-OOD Main Analysis: {os.path.basename(inference_results_h5)}')

    gs1 = gridspec.GridSpec(4, 2, width_ratios=[1, 1.5], wspace=0.2, hspace=0.35)
    ax_img = fig1.add_subplot(gs1[:, 0])
    ax_ts_z = fig1.add_subplot(gs1[0, 1])
    ax_ts_s = fig1.add_subplot(gs1[1, 1], sharex=ax_ts_z)
    ax_ts_attr = fig1.add_subplot(gs1[2, 1], sharex=ax_ts_z)
    ax_ts_f = fig1.add_subplot(gs1[3, 1], sharex=ax_ts_z)
    ax_ts_a = ax_ts_f.twinx()

    # ─────────────────────────────────────────
    # Window 2: Parameter Maps (3x2 grid if XAI available)
    # ─────────────────────────────────────────
    if has_xai:
        fig2 = plt.figure(figsize=(18, 12))
        fig2.canvas.manager.set_window_title('FT-OOD Parameter & XAI Maps')
        gs2 = gridspec.GridSpec(2, 3, wspace=0.35, hspace=0.3)
        ax_max_score = fig2.add_subplot(gs2[0, 0])
        ax_drift_cnt = fig2.add_subplot(gs2[0, 1])
        ax_dom_win = fig2.add_subplot(gs2[0, 2])
        ax_per = fig2.add_subplot(gs2[1, 0])
        ax_unc = fig2.add_subplot(gs2[1, 1])
        ax_comp_map = fig2.add_subplot(gs2[1, 2])
    else:
        fig2 = plt.figure(figsize=(16, 12))
        fig2.canvas.manager.set_window_title('FT-OOD Parameter Maps')
        gs2 = gridspec.GridSpec(2, 2, wspace=0.3, hspace=0.3)
        ax_max_score = fig2.add_subplot(gs2[0, 0])
        ax_drift_cnt = fig2.add_subplot(gs2[0, 1])
        ax_per = fig2.add_subplot(gs2[1, 0])
        ax_unc = fig2.add_subplot(gs2[1, 1])

    # ── Base Ortho + Anomaly ──
    ax_img.imshow(base_frame)
    ax_img.set_title(f"Structural Anomalies (First Drift)\n{base_sg} Acquisition: {base_date.strftime('%Y-%m-%d')} UTC")

    from matplotlib.cm import viridis, plasma, inferno, YlOrRd
    from matplotlib.colors import ListedColormap, BoundaryNorm

    if not np.all(np.isnan(anomaly_map)):
        masked_anom = np.ma.masked_invalid(anomaly_map)
        cmap = copy.copy(viridis)
        cmap.set_bad(color='white', alpha=0)
        im1 = ax_img.imshow(masked_anom, cmap=cmap, alpha=0.7)
        cbar = plt.colorbar(im1, ax=ax_img)
        ticks = cbar.get_ticks()
        min_anom, max_anom = np.nanmin(anomaly_map), np.nanmax(anomaly_map)
        if not np.isnan(min_anom):
            ticks = ticks[(ticks >= min_anom) & (ticks <= max_anom)]
            cbar.set_ticks(ticks)
            cbar.set_ticklabels([datetime.fromtimestamp(t, timezone.utc).strftime('%Y-%m-%d') for t in ticks])

    # ── Spatial Maps ──
    im_s = ax_max_score.imshow(max_score_map, cmap=inferno)
    ax_max_score.set_title("Max OOD Anomaly Score")
    plt.colorbar(im_s, ax=ax_max_score)

    im_c = ax_drift_cnt.imshow(change_count_map, cmap=YlOrRd)
    ax_drift_cnt.set_title("Total Drift Events")
    plt.colorbar(im_c, ax=ax_drift_cnt)

    masked_per = np.ma.masked_invalid(mean_period_map)
    cmap_per = copy.copy(viridis)
    cmap_per.set_bad(color='gray', alpha=1.0)
    im3 = ax_per.imshow(masked_per, cmap=cmap_per, vmin=0.5, vmax=3.0)
    ax_per.set_title(f"Mean Dominant Period (W={windows[-1]}yr)")
    plt.colorbar(im3, ax=ax_per, label="Period (Years)")

    masked_unc = np.ma.masked_invalid(mean_uncertainty)
    cmap_unc = copy.copy(plasma)
    cmap_unc.set_bad(color='gray', alpha=1.0)
    im2 = ax_unc.imshow(masked_unc, cmap=cmap_unc)
    ax_unc.set_title(f"Mean Predictive Uncertainty (W={windows[-1]}yr)")
    plt.colorbar(im2, ax=ax_unc, label="Sigma")

    # ── XAI-specific maps ──
    if has_xai:
        # Dominant Window Map (categorical colormap)
        if dominant_window_map is not None:
            win_colors = [WINDOW_COLORS[i % len(WINDOW_COLORS)] for i in range(num_windows)]
            win_colors.insert(0, '#808080')  # gray for no-data
            cmap_win = ListedColormap(win_colors)
            bounds = np.arange(-0.5, num_windows + 0.5, 1)
            norm_win = BoundaryNorm(bounds, cmap_win.N)

            masked_dom = np.ma.masked_invalid(dominant_window_map)
            im_dw = ax_dom_win.imshow(masked_dom, cmap=cmap_win, norm=norm_win)
            cbar_dw = plt.colorbar(im_dw, ax=ax_dom_win, ticks=range(num_windows))
            cbar_dw.set_ticklabels([f'{w}yr' for w in windows])
            ax_dom_win.set_title("Dominant Attribution Window")

        # Mean Complexity Map
        if mean_complexity_map is not None:
            masked_comp = np.ma.masked_invalid(mean_complexity_map)
            cmap_comp = copy.copy(inferno)
            cmap_comp.set_bad(color='gray', alpha=1.0)
            im_comp = ax_comp_map.imshow(masked_comp, cmap=cmap_comp)
            ax_comp_map.set_title("Mean Attribution Complexity")
            plt.colorbar(im_comp, ax=ax_comp_map, label="Complexity (entropy)")

    # ── Initial state for time series ──
    for ax in [ax_ts_z, ax_ts_s, ax_ts_attr, ax_ts_f]:
        ax.text(0.5, 0.5, 'Click a pixel on any map to view data',
                ha='center', va='center', transform=ax.transAxes)

    rects = []
    maps_axes = [ax_img, ax_max_score, ax_drift_cnt, ax_per, ax_unc]
    if has_xai:
        maps_axes.extend([ax_dom_win, ax_comp_map])
    for ax in maps_axes:
        rect = patches.Rectangle((-1, -1), 1, 1, linewidth=2, edgecolor='cyan', facecolor='none', visible=False)
        ax.add_patch(rect)
        rects.append(rect)

    def update_pixel(x, y):
        print(f"Selecting pixel {x}, {y}")
        for rect in rects:
            rect.set_xy((x - 0.5, y - 0.5))
            rect.set_visible(True)

        current_date_ts = None
        current_sg = None
        if not np.isnan(anomaly_map[y, x]):
            anom_ts = anomaly_map[y, x]
            idx = np.argmin(np.abs(acq_time - anom_ts))
            new_base, current_sg = get_ortho(idx)
            ax_img.images[0].set_array(new_base)
            current_date_ts = acq_time[idx]
        else:
            ax_img.images[0].set_array(base_frame)
            current_date_ts = acq_time[base_idx]
            current_sg = base_sg

        current_date = datetime.fromtimestamp(current_date_ts, timezone.utc)
        ax_img.set_title(f"Structural Anomalies (First Drift)\n{current_sg} Acquisition: {current_date.strftime('%Y-%m-%d')} UTC")

        plot_pixel_sits(y, x, source_h5_path, inference_results_h5,
                        ax_ts_z, ax_ts_s, ax_ts_attr, ax_ts_f, ax_ts_a,
                        current_date=current_date)
        fig1.canvas.draw()
        fig2.canvas.draw()

    def onclick(event):
        if event.inaxes not in maps_axes:
            return
        x, y = int(event.xdata), int(event.ydata)
        if x < 0 or x >= W or y < 0 or y >= H:
            return
        update_pixel(x, y)

    fig1.canvas.mpl_connect('button_press_event', onclick)
    fig2.canvas.mpl_connect('button_press_event', onclick)


if __name__ == "__main__":
    inference_h5 = get_inference_h5(LOCATION)
    if inference_h5 and os.path.exists(inference_h5):
        print(f"Loading latest inference results: {inference_h5}")
        plot_spatial_anomaly_overlay(H5_PATH, inference_h5)
        plt.show()
    else:
        print("Run FT_OOD_main.py first to create output h5")
