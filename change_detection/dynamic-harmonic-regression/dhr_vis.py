import h5py
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timezone
import os
import pyproj
import matplotlib.patches as patches
import scienceplots
import warnings
import copy
import matplotlib.gridspec as gridspec
import matplotlib.widgets as widgets
import sys
# Add the project root to sys.path so we can import pixel_extractor
_current_dir = os.path.dirname(os.path.abspath(__file__))
_root_dir = os.path.abspath(os.path.join(_current_dir, '..', '..'))
if _root_dir not in sys.path:
    sys.path.insert(0, _root_dir)
from pixel_extractor import append_single_pixel
plt.style.use(['science','no-latex'])

LOCATION = "Tait"
IGNORE_COMMON_MASK = False
H5_PATH = f"C:/satelliteImagery/HLST30/HLST_{LOCATION}_Harmonized_SC_EM-7_Norm-bandCount.h5"

import glob

def get_inference_h5(location, ignore_common_mask=True):
    search_pattern = f"C:/satelliteImagery/HLST30/DHR/{location}_DHR_Change_Detection_*.h5"
    files = glob.glob(search_pattern)
    if not files:
        return None
        
    filtered_files = []
    for f in files:
        is_unmasked = f.endswith("_unmasked.h5")
        if ignore_common_mask and is_unmasked:
            filtered_files.append(f)
        elif not ignore_common_mask and not is_unmasked:
            filtered_files.append(f)
            
    if not filtered_files:
        return None
        
    filtered_files.sort(key=os.path.getmtime, reverse=True)
    return filtered_files[0]

def plot_pixel_sits(pixel_y, pixel_x, source_h5_path, inference_results_h5, ax_ts_z, ax_ts_f, ax_ts_a=None, current_date=None):
    ax_ts_z.clear()
    ax_ts_f.clear()
    if ax_ts_a is not None:
        ax_ts_a.clear()

    lat, lon = None, None
    with h5py.File(source_h5_path, 'r') as f:
        harm_grp = f['/HDFEOS/GRIDS/HARMONIZED/Data Fields']
        acq_time = harm_grp['sliding_volume_z_score'].attrs['acquisition_time'][:]
        z_score = harm_grp['sliding_volume_z_score'][:, pixel_y, pixel_x]
        
        unified_masks = harm_grp['common_mask'][:, pixel_y, pixel_x]
        is_invalid = unified_masks.astype(bool)
        
        spacecraft_bytes = harm_grp['sliding_volume_z_score'].attrs['source_spacecraft'][:]
        spacecrafts = [s.decode('utf-8') if isinstance(s, bytes) else str(s) for s in spacecraft_bytes]
        
        geo_transform = harm_grp['sliding_volume_z_score'].attrs.get('GeoTransform')
        spatial_ref = harm_grp['sliding_volume_z_score'].attrs.get('spatial_ref')
        if geo_transform is not None and spatial_ref is not None:
            try:
                gt = geo_transform
                x_geo = gt[0] + (pixel_x + 0.5) * gt[1] + (pixel_y + 0.5) * gt[2]
                y_geo = gt[3] + (pixel_x + 0.5) * gt[4] + (pixel_y + 0.5) * gt[5]
                if isinstance(spatial_ref, bytes):
                    spatial_ref_str = spatial_ref.decode('utf-8')
                else:
                    spatial_ref_str = str(spatial_ref)
                crs = pyproj.CRS.from_wkt(spatial_ref_str)
                transformer = pyproj.Transformer.from_crs(crs, "epsg:4326", always_xy=True)
                lon, lat = transformer.transform(x_geo, y_geo)
            except Exception as e:
                pass
        
    dates = [datetime.fromtimestamp(ts, timezone.utc) for ts in acq_time]
    
    with h5py.File(inference_results_h5, 'r') as f:
        predicted = f['predicted_series'][:, pixel_y, pixel_x]
        rmse = f['rmse_series'][:, pixel_y, pixel_x]
        anomalies = f['anomaly_flags'][:, pixel_y, pixel_x]
        change_date_ts = f['change_date_timestamp'][pixel_y, pixel_x]
        dom_freq = f['dominant_frequencies_series'][:, :, pixel_y, pixel_x] # [N, K]
        rmse_multiplier = f.attrs.get('RMSE_MULTIPLIER', 3.0)
        max_window_years = f.attrs.get('MAX_WINDOW_YEARS', 5.0)
        if 'amplitude_series' in f:
            amp_series_pixel = f['amplitude_series'][:, :, pixel_y, pixel_x]
            has_amp = True
        else:
            has_amp = False
    
    valid_mask = ~is_invalid
    dates_arr = np.array(dates)
    spacecrafts_arr = np.array(spacecrafts)
    
    # -----------------------
    # PLOT 1: Z-SCORE & PREDS
    # -----------------------
    for marker_type, sc_keyword in [('s', 'Sentinel'), ('o', 'Landsat'), ('D', 'Tanager')]:
        sc_mask = np.array([sc_keyword.lower() in str(sc).lower() for sc in spacecrafts_arr])
        
        idx_valid = valid_mask & sc_mask
        if np.any(idx_valid):
            ax_ts_z.plot(dates_arr[idx_valid], z_score[idx_valid], color='k', marker=marker_type, linestyle='None', label=f'Valid ({sc_keyword})')
            
        idx_invalid = is_invalid & sc_mask
        if np.any(idx_invalid):
            ax_ts_z.plot(dates_arr[idx_invalid], z_score[idx_invalid], color='gray', marker=marker_type, linestyle='None', markerfacecolor='none', label=f'Invalid ({sc_keyword})')
            
    pred_mask = ~np.isnan(predicted)
    if np.any(pred_mask):
        pred_dates = dates_arr[pred_mask]
        preds = predicted[pred_mask]
        rmses = rmse[pred_mask]
        
        upper_bound = preds + rmse_multiplier * rmses
        lower_bound = preds - rmse_multiplier * rmses
        
        ax_ts_z.plot(pred_dates, preds, 'b--', label='Harmonic Prediction')
        ax_ts_z.fill_between(pred_dates, lower_bound, upper_bound, color='blue', alpha=0.15, label=f'±{rmse_multiplier}σ Bound')
        
        anom_mask = anomalies[pred_mask] == 1
        if np.any(anom_mask):
            anom_dates = pred_dates[anom_mask]
            anom_vals = z_score[pred_mask][anom_mask]
            ax_ts_z.scatter(anom_dates, anom_vals, color='red', s=30, zorder=4, label='Anomaly (Unconfirmed)')
            
            if not np.isnan(change_date_ts):
                change_dt = datetime.fromtimestamp(change_date_ts, timezone.utc)
                conf_mask = np.array([d >= change_dt for d in anom_dates])
                if np.any(conf_mask):
                    ax_ts_z.scatter(anom_dates[conf_mask], anom_vals[conf_mask], color='darkred', marker='*', s=150, zorder=6, label='Confirmed Structural Change')
            
    if current_date is not None:
        ax_ts_z.axvline(x=current_date, color='orange', linestyle='--', label='Displayed Frame')

    title_str = f"Pixel: ({pixel_x}, {pixel_y})"
    if lat is not None and lon is not None:
        title_str += f" | Lat: {lat:.5f}, Lon: {lon:.5f}"
    ax_ts_z.set_title(title_str)
    ax_ts_z.set_ylabel('Z-Score')
    ax_ts_z.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    ax_ts_z.grid(True)
    ax_ts_z.set_ylim([-4, 4])
    
    # -----------------------
    # PLOT 2: FREQUENCIES
    # -----------------------
    if np.any(pred_mask):
        dom_freq_valid = dom_freq[pred_mask] # [N_valid, K]
        periods = (2.0 * np.pi) / dom_freq_valid
        if has_amp:
            amp_valid = amp_series_pixel[pred_mask]
        
        # Calculate actual data span and pseudo-Nyquist limit for each prediction date
        valid_ts = acq_time[~is_invalid]
        pred_spans = np.zeros(len(pred_dates))
        pred_samples = np.zeros(len(pred_dates))
        for i, pd in enumerate(pred_dates):
            pd_ts = pd.timestamp()
            win_start = pd_ts - max_window_years * 365.25 * 86400
            in_win = valid_ts[(valid_ts >= win_start) & (valid_ts < pd_ts)]
            if len(in_win) > 0:
                pred_spans[i] = (pd_ts - in_win.min()) / (365.25 * 86400)
                pred_samples[i] = len(in_win)
            else:
                pred_spans[i] = 0.0
                pred_samples[i] = 0
                
        nyquist_periods = np.full_like(pred_spans, np.nan)
        valid_n_idx = pred_samples > 0
        nyquist_periods[valid_n_idx] = 2.0 * pred_spans[valid_n_idx] / pred_samples[valid_n_idx]
        
        colors = ['tab:blue', 'tab:orange', 'tab:green']
        for k in range(min(3, periods.shape[1])):
            color = colors[k] if k < len(colors) else 'tab:red'
            periods_k = periods[:, k].copy()
            
            # When the period is equal to or greater than the actual data span, 
            # or the maximum window capacity, it indicates a pattern was not found.
            invalid_mask = (periods_k >= (pred_spans * 0.95)) | (periods_k >= (max_window_years * 0.98))
            periods_k[invalid_mask] = np.nan
            
            valid_p = periods_k[~np.isnan(periods_k)]
            if len(valid_p) > 0:
                med_days = np.median(valid_p) * 365.25
                label_str = f'Top-{k+1} Period (~{med_days:.0f} days)'
            else:
                label_str = f'Top-{k+1} Period (Not Found)'
                
            ax_ts_f.plot(pred_dates, periods_k * 365.25, marker='.', linestyle='-', color=color, label=label_str)
            
            if has_amp:
                amp_k = amp_valid[:, k].copy()
                amp_k[invalid_mask] = np.nan
                ax_ts_a.plot(pred_dates, amp_k, marker='x', linestyle=':', color=color, alpha=0.4)
                
        # Plot Nyquist Limit Bounding Line
        ax_ts_f.plot(pred_dates, nyquist_periods * 365.25, color='black', linestyle='--', linewidth=1.5, label='Nyquist Limit (Period)')
            
    if current_date is not None:
        ax_ts_f.axvline(x=current_date, color='orange', linestyle='--')
        
    ax_ts_f.set_ylabel('Dominant Period (Days)')
    ax_ts_f.set_xlabel('Date')
    import matplotlib.dates as mdates
    ax_ts_f.xaxis.set_major_locator(mdates.YearLocator())
    ax_ts_f.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    ax_ts_f.grid(True)
    if ax_ts_a is not None:
        ax_ts_a.yaxis.tick_right()
        ax_ts_a.yaxis.set_label_position("right")
        ax_ts_a.set_ylabel('Amplitude (Z-Score)', color='gray')
        ax_ts_a.tick_params(axis='y', labelcolor='gray')
    #ax_ts_f.set_ylim([0.2, 3.0]) # Cap period display


def plot_spatial_anomaly_overlay(source_h5_path, inference_results_h5):
    with h5py.File(source_h5_path, 'r') as f:
        harm_grp = f['/HDFEOS/GRIDS/HARMONIZED/Data Fields']
        acq_time = harm_grp['sliding_volume_z_score'].attrs['acquisition_time'][:]
        unified_masks = harm_grp['common_mask'][:]
        full_valid_mask = ~unified_masks.astype(bool)
        
    def get_ortho(idx):
        with h5py.File(source_h5_path, 'r') as f:
            harm_grp = f['/HDFEOS/GRIDS/HARMONIZED/Data Fields']
            spc = harm_grp['sliding_volume_z_score'].attrs['source_spacecraft'][idx]
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
        anomaly_map = f['change_date_timestamp'][:]
        change_count_map = f['change_count'][:]
        min_samples = f.attrs.get('MIN_SAMPLES', 8)
        rmse_series = f['rmse_series'][:]
        dom_freq = f['dominant_frequencies_series'][:] # [N, K, H, W]
        if 'amplitude_series' in f:
            amp_series = f['amplitude_series'][:]
            has_amp = True
        else:
            has_amp = False
        
    H, W = full_valid_mask.shape[1], full_valid_mask.shape[2]
    
    anomaly_map[change_count_map == 0] = np.nan
    
    valid_initial_counts = np.sum(full_valid_mask, axis=0)
    insufficient_data = valid_initial_counts < min_samples
    
    # Pre-calculate maps
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean_uncertainty = np.nanmean(rmse_series, axis=0)
        top1_freq = dom_freq[:, 0, :, :]
        top1_period = (2.0 * np.pi) / top1_freq
        mean_period_map = np.nanmean(top1_period, axis=0)
        f_hz = top1_freq / (2.0 * np.pi)
        freq_var_map = np.nanvar(f_hz, axis=0)
        if has_amp:
            mean_amp = np.nanmean(amp_series[:, 0, :, :], axis=0)
        
    mean_uncertainty[insufficient_data] = np.nan
    mean_period_map[insufficient_data] = np.nan
    freq_var_map[insufficient_data] = np.nan
    if has_amp:
        mean_amp[insufficient_data] = np.nan
    
    # Setup GridSpec
    # Window 1: Main Analysis (Ortho + Time Series)
    fig1 = plt.figure(figsize=(18, 9))
    window1_title = f'DHR Main Analysis: {os.path.basename(inference_results_h5)}'
    fig1.canvas.manager.set_window_title(window1_title)
    
    gs1 = gridspec.GridSpec(2, 2, width_ratios=[1, 1.5], wspace=0.2, hspace=0.3)
    ax_img = fig1.add_subplot(gs1[:, 0])
    
    ax_ts_z = fig1.add_subplot(gs1[0, 1])
    ax_ts_f = fig1.add_subplot(gs1[1, 1], sharex=ax_ts_z)
    ax_ts_a = ax_ts_f.twinx() if has_amp else None
    
    # Window 2: Maps
    fig2 = plt.figure(figsize=(16, 12))
    window2_title = f'DHR Parameter Maps: {os.path.basename(inference_results_h5)}'
    fig2.canvas.manager.set_window_title(window2_title)
    
    gs2 = gridspec.GridSpec(2, 2, wspace=0.3, hspace=0.3)
    ax_unc = fig2.add_subplot(gs2[0, 0])
    ax_amp = fig2.add_subplot(gs2[0, 1])
    ax_amp.axis('off') # default to off if no amp
    ax_per = fig2.add_subplot(gs2[1, 0])
    ax_var = fig2.add_subplot(gs2[1, 1])
    
    # 1. Base Ortho + Anomaly
    ax_img.imshow(base_frame)
    ax_img.set_title(f"Structural Anomalies\n{base_sg} Acquisition: {base_date.strftime('%Y-%m-%d')} UTC")
    
    gray = np.zeros((H, W, 4))
    gray[insufficient_data] = [0.5, 0.5, 0.5, 0.5]
    ax_img.imshow(gray)
    
    if not np.all(np.isnan(anomaly_map)):
        from matplotlib.cm import viridis
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
            
    # 2. Uncertainty to Amplitude Ratio
    from matplotlib.cm import plasma, inferno
    if has_amp:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            unc_amp_ratio = mean_uncertainty / mean_amp
            
        masked_unc = np.ma.masked_invalid(unc_amp_ratio)
        cmap_unc = copy.copy(plasma)
        cmap_unc.set_bad(color='gray', alpha=1.0)
        
        valid_ratio = unc_amp_ratio[~np.isnan(unc_amp_ratio) & ~np.isinf(unc_amp_ratio)]
        vmax = np.percentile(valid_ratio, 95) if len(valid_ratio) > 0 else 1.0
        
        im2 = ax_unc.imshow(masked_unc, cmap=cmap_unc, vmax=vmax)
        ax_unc.set_title("Uncertainty-to-Amplitude Ratio")
        plt.colorbar(im2, ax=ax_unc, label="Ratio (S / Amp)")
    else:
        masked_unc = np.ma.masked_invalid(mean_uncertainty)
        cmap_unc = copy.copy(plasma)
        cmap_unc.set_bad(color='gray', alpha=1.0)
        im2 = ax_unc.imshow(masked_unc, cmap=cmap_unc)
        ax_unc.set_title("Mean Predictive Uncertainty (S)")
        plt.colorbar(im2, ax=ax_unc, label="Mean S")
    
    # 3. Mean Dominant Period
    masked_per = np.ma.masked_invalid(mean_period_map)
    cmap_per = copy.copy(viridis)
    cmap_per.set_bad(color='gray', alpha=1.0)
    im3 = ax_per.imshow(masked_per, cmap=cmap_per, vmin=0.5, vmax=3.0)
    ax_per.set_title("Mean Dominant Period (Mode)")
    plt.colorbar(im3, ax=ax_per, label="Period (Years)")
    
    # 4. Frequency Variance
    masked_var = np.ma.masked_invalid(freq_var_map)
    cmap_var = copy.copy(inferno)
    cmap_var.set_bad(color='gray', alpha=1.0)
    im4 = ax_var.imshow(masked_var, cmap=cmap_var)
    ax_var.set_title("Dominant Frequency Instability (Variance)")
    plt.colorbar(im4, ax=ax_var, label="Variance (cycles/year)^2")
    
    # 5. Amplitude (If Available)
    if has_amp:
        ax_amp.axis('on')
        masked_amp = np.ma.masked_invalid(mean_amp)
        cmap_amp = copy.copy(plasma)
        cmap_amp.set_bad(color='gray', alpha=1.0)
        im5 = ax_amp.imshow(masked_amp, cmap=cmap_amp)
        ax_amp.set_title("Mean Top-1 Amplitude")
        plt.colorbar(im5, ax=ax_amp, label="Amplitude (Z-Score)")
    
    # Initial state for time series axes
    ax_ts_z.text(0.5, 0.5, 'Click a pixel on any map to view data', horizontalalignment='center', verticalalignment='center', transform=ax_ts_z.transAxes)
    ax_ts_f.text(0.5, 0.5, 'Click a pixel on any map to view data', horizontalalignment='center', verticalalignment='center', transform=ax_ts_f.transAxes)

    # Window 3: Data Extraction
    fig3 = plt.figure(figsize=(4, 4))
    fig3.canvas.manager.set_window_title("Pixel Extraction")
    
    current_pixel = {'x': None, 'y': None}
    
    ax_radio = fig3.add_axes([0.1, 0.4, 0.8, 0.5])
    ax_radio.set_title("Select Category")
    categories = ('structural change', 'transient event', 'stable periodic patterns', 'noisy data', 'indeterminate data')
    radio = widgets.RadioButtons(ax_radio, categories)
    
    # Custom visual feedback since scienceplots style hides the default radio circles
    def update_radio_style(label):
        for text in radio.labels:
            if text.get_text() == label:
                text.set_color('red')
                text.set_fontweight('bold')
            else:
                text.set_color('black')
                text.set_fontweight('normal')
        fig3.canvas.draw_idle()
        
    radio.on_clicked(update_radio_style)
    update_radio_style(categories[0]) # initialize first option
    
    # Store reference to prevent garbage collection of event listeners
    fig3.radio = radio
    
    ax_btn = fig3.add_axes([0.3, 0.1, 0.4, 0.15])
    btn = widgets.Button(ax_btn, 'Extract Pixel')
    
    # Store reference to prevent garbage collection of event listeners
    fig3.btn = btn
    
    def on_extract_clicked(event):
        if current_pixel['x'] is None or current_pixel['y'] is None:
            print("No pixel selected yet. Please click on a map first.")
            return
            
        category = radio.value_selected
        x, y = current_pixel['x'], current_pixel['y']
        print(f"Extracting pixel ({x}, {y}) as '{category}'...")
        
        try:
            append_single_pixel(
                h5_path=source_h5_path,
                location=LOCATION,
                x=x,
                y=y,
                category=category,
                json_path="z-score-samples.json"
            )
            print("Extraction successful.")
        except Exception as e:
            print(f"Error extracting pixel: {e}")
            
    btn.on_clicked(on_extract_clicked)

    # Add selection rectangles to all axes
    rects = []
    maps_axes = [ax_img, ax_unc, ax_per, ax_var]
    if has_amp:
        maps_axes.append(ax_amp)
        
    for ax in maps_axes:
        rect = patches.Rectangle((-1, -1), 1, 1, linewidth=2, edgecolor='cyan', facecolor='none', visible=False)
        ax.add_patch(rect)
        rects.append(rect)

    def update_pixel(x, y):
        print(f"Selecting pixel {x}, {y}")
        current_pixel['x'] = x
        current_pixel['y'] = y
        for rect in rects:
            rect.set_xy((x-0.5, y-0.5))
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
        ax_img.set_title(f"Structural Anomalies\n{current_sg} Acquisition: {current_date.strftime('%Y-%m-%d')} UTC")
        
        plot_pixel_sits(y, x, source_h5_path, inference_results_h5, ax_ts_z, ax_ts_f, ax_ts_a=ax_ts_a, current_date=current_date)
        fig1.canvas.draw()
        fig2.canvas.draw()

    def onclick(event):
        maps_axes = [ax_img, ax_unc, ax_per, ax_var]
        if has_amp:
            maps_axes.append(ax_amp)
        if event.inaxes not in maps_axes: return
        x, y = int(event.xdata), int(event.ydata)
        if x < 0 or x >= W or y < 0 or y >= H: return
        update_pixel(x, y)

    fig1.canvas.mpl_connect('button_press_event', onclick)
    fig2.canvas.mpl_connect('button_press_event', onclick)

if __name__ == "__main__":
    inference_h5 = get_inference_h5(LOCATION, IGNORE_COMMON_MASK)
    if inference_h5 and os.path.exists(inference_h5):
        print(f"Loading latest inference results: {inference_h5}")
        plot_spatial_anomaly_overlay(H5_PATH, inference_h5)
        plt.show()
    else:
        print("Run dhr_main.py first to create output h5")
