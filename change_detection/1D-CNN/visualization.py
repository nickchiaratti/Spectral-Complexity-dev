import h5py
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timezone
import os
import scienceplots
import pyproj
plt.style.use(['science','no-latex'])

LOCATION = "Tait"
TRAIN_END_YEAR = "2023"
OUTPUT_DIR = f"C:/satelliteImagery/HLST30/1D-CNN-{LOCATION}-TrainEnd{TRAIN_END_YEAR}"
H5_PATH = f"C:/satelliteImagery/HLST30/HLST_{LOCATION}_Harmonized_SC_EM-7_Norm-bandCount.h5"
INFERENCE_H5 = os.path.join(OUTPUT_DIR, f'CNN_{LOCATION}_baseline_weights_pre{TRAIN_END_YEAR}.h5')


def validate_inference_results_schema(f, file_path):
    if 'inference_results' not in f:
        raise ValueError(f"No inference_results dataset found in {file_path}")
    ds = f['inference_results']
    if 'train_end_date' not in ds.attrs or 'confidence_multiplier' not in ds.attrs:
        raise ValueError("Inference results are missing required metadata (train_end_date or confidence_multiplier). Please re-run the CNN pipeline, as the results were generated with an out-of-date codebase.")
    
    names = ds.dtype.names
    required_cols = ['Pixel_X', 'Pixel_Y', 'Timestamp_T21', 'Timestamp_T_Last', 'Pred_1', 'Std_1', 'Anomaly_Flag', 'Confirmed_Change']
    missing_cols = [c for c in required_cols if c not in names]
    if missing_cols:
        raise ValueError(f"Inference results dataset is missing required columns: {missing_cols}. Please re-run the CNN pipeline, as the results were generated with an out-of-date codebase.")
        
    if 'Attr_DoY' in names:
        expected_attrs = ['Attr_DoY', 'Attr_ToD', 'Attr_dt', 'Attr_ZScore', 'Attr_Spatial']
        missing_attrs = [c for c in expected_attrs if c not in names]
        if missing_attrs:
            raise ValueError(f"Inference results dataset has partial attribution data but is missing required attribution columns: {missing_attrs}. Please re-run the CNN pipeline (`change_detection/1D-CNN/train_evaluate.py`), as the results were generated with an out-of-date codebase.")


def plot_pixel_sits(pixel_y, pixel_x, source_h5_path, inference_results_h5, ax=None, ax_rob=None, current_date=None):
    # This visualizes the 1D time series for a pixel
    lat, lon = None, None
    with h5py.File(source_h5_path, 'r') as f:
        harm_grp = f['/HDFEOS/GRIDS/HARMONIZED/Data Fields']
        acq_time = harm_grp['sliding_volume_z_score'].attrs['acquisition_time'][:]
        z_score = harm_grp['sliding_volume_z_score'][:, pixel_y, pixel_x]
        z_score = np.clip(z_score, -5.0, 5.0)
        
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
                print(f"Warning: Could not compute lat/lon: {e}")
            
    dates = [datetime.fromtimestamp(ts, timezone.utc) for ts in acq_time]
    
    # Load inference results for this pixel
    with h5py.File(inference_results_h5, 'r') as f:
        validate_inference_results_schema(f, inference_results_h5)
            
        res = f['inference_results'][:]
        
        aleatoric_val = 0
        if 'baseline_aleatoric_map' in f:
            aleatoric_val = float(f['baseline_aleatoric_map'][pixel_y, pixel_x])
            
        train_end_date = f['inference_results'].attrs['train_end_date']
        conf_mult = float(f['inference_results'].attrs['confidence_multiplier'])
        
    mask = (res['Pixel_X'] == pixel_x) & (res['Pixel_Y'] == pixel_y)
    pixel_res = res[mask]
    
    if ax is None:
        fig, (ax, ax_rob) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={'height_ratios': [2.5, 1]})
        show_plot = True
    else:
        show_plot = False
        if ax_rob is None:
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider = make_axes_locatable(ax)
            ax_rob = divider.append_axes("bottom", size="35%", pad=0.3, sharex=ax)
    
    # Separate valid vs interpolated (invalid)
    valid_mask = ~is_invalid
    
    dt = datetime.strptime(train_end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    split_date = dt
    
    # Historical <= 2024
    hist_mask = np.array([d < split_date for d in dates])
    
    mon_mask = ~hist_mask
    
    dates_arr = np.array(dates)
    spacecrafts_arr = np.array(spacecrafts)
    
    for marker_type, sc_keyword in [('s', 'Sentinel'), ('o', 'Landsat'), ('D', 'Tanager')]:
        sc_mask = np.array([sc_keyword.lower() in str(sc).lower() for sc in spacecrafts_arr])
        
        idx_hist = hist_mask & valid_mask & sc_mask
        if np.any(idx_hist):
            ax.plot(dates_arr[idx_hist], z_score[idx_hist], color='k', marker=marker_type, linestyle='None', label=f'Valid Historical ({sc_keyword})')
            
        idx_mon = mon_mask & valid_mask & sc_mask
        if np.any(idx_mon):
            ax.plot(dates_arr[idx_mon], z_score[idx_mon], color='b', marker=marker_type, linestyle='None', label=f'Valid Actuals ({sc_keyword})')
            
        idx_invalid = is_invalid & sc_mask
        if np.any(idx_invalid):
            ax.plot(dates_arr[idx_invalid], z_score[idx_invalid], color='gray', marker=marker_type, linestyle='None', markerfacecolor='none', label=f'Cloud Masked ({sc_keyword})')
    
    # Predictions
    if len(pixel_res) > 0:
        pred_dates = []
        anomaly_flags = []
        confirmed_flags = []
        attr_doy = []
        attr_tod = []
        attr_dt = []
        attr_zscore = []
        attr_spatial = []
        explainer_sens = []
        explainer_comp = []
        
        # Determine number of predictions
        pred_cols = [c for c in pixel_res.dtype.names if c.startswith('Pred_')]
        num_preds = len(pred_cols)
        
        preds = {k: [] for k in range(1, num_preds + 1)}
        stds = {k: [] for k in range(1, num_preds + 1)}
        
        valid_idx = np.where(~is_invalid)[0]
        valid_acq_time = acq_time[valid_idx]
        
        for row in pixel_res:
            # The first prediction target is Timestamp_T21
            target_ts = row['Timestamp_T21']
            d_target = datetime.fromtimestamp(target_ts, timezone.utc)
            
            pred_dates.append(d_target)
            anomaly_flags.append(row['Anomaly_Flag'])
            confirmed_flags.append(row['Confirmed_Change'])
                
            if 'Attr_DoY' in pixel_res.dtype.names:
                attr_doy.append(row['Attr_DoY'])
                attr_tod.append(row['Attr_ToD'])
                attr_dt.append(row['Attr_dt'])
                attr_zscore.append(row['Attr_ZScore'])
                attr_spatial.append(row['Attr_Spatial'])
            if 'Explainer_Sensitivity' in pixel_res.dtype.names and 'Explainer_Complexity' in pixel_res.dtype.names:
                explainer_sens.append(row['Explainer_Sensitivity'])
                explainer_comp.append(row['Explainer_Complexity'])
                
            for k in range(1, num_preds + 1):
                preds[k].append(row[f'Pred_{k}'])
                stds[k].append(row[f'Std_{k}'])
                
        srt = np.argsort(pred_dates)
        pred_dates = np.array(pred_dates)[srt]
        anomaly_flags = np.array(anomaly_flags)[srt]
        confirmed_flags = np.array(confirmed_flags)[srt]
        if len(explainer_sens) > 0:
            explainer_sens = np.array(explainer_sens)[srt]
            explainer_comp = np.array(explainer_comp)[srt]
        
        for k in range(1, num_preds + 1):
            preds[k] = np.array(preds[k])[srt]
            stds[k] = np.array(stds[k])[srt]
            
        tot_upper = preds[1] + conf_mult * stds[1]
        tot_lower = preds[1] - conf_mult * stds[1]
        
        al_upper = preds[1] + conf_mult * aleatoric_val
        al_lower = preds[1] - conf_mult * aleatoric_val
        
        colors = ['r--', 'g--', 'b--', 'm--', 'c--', 'y--']
        for k in range(1, num_preds + 1):
            color = colors[(k-1) % len(colors)]
            ax.plot(pred_dates, preds[k], color, label=f'Prediction t+{k}')
            
        # Draw Aleatoric Base
        ax.fill_between(pred_dates, al_lower, al_upper, color='gray', alpha=0.3, label=f'Aleatoric Uncertainty ±{conf_mult}σ')
        
        # Draw Epistemic Additive Regions
        ax.fill_between(pred_dates, tot_upper, al_upper, color='purple', alpha=0.3, label=f'+ Epistemic Uncertainty')
        ax.fill_between(pred_dates, al_lower, tot_lower, color='purple', alpha=0.3)
        
        # Anomalies
        anom_dates = pred_dates[anomaly_flags == 1]
        anom_vals = preds[1][anomaly_flags == 1]
        ax.scatter(anom_dates, anom_vals, color='red', s=30, zorder=4, label='Anomaly (Unconfirmed)')
        
        # Confirmed Anomalies
        conf_dates = pred_dates[confirmed_flags == 1]
        conf_vals = preds[1][confirmed_flags == 1]
        if len(conf_dates) > 0:
            ax.scatter(conf_dates, conf_vals, color='darkred', marker='*', s=150, zorder=6, label='Confirmed Structural Change')
        
        # Attribution Bars
        if 'Attr_DoY' in pixel_res.dtype.names:
            anom_mask = (anomaly_flags == 1)
            a_doy = np.array(attr_doy)[srt][anom_mask]
            
            if len(anom_dates) > 0 and not np.isnan(a_doy[0]):
                a_tod = np.array(attr_tod)[srt][anom_mask]
                a_dt = np.array(attr_dt)[srt][anom_mask]
                a_zscore = np.array(attr_zscore)[srt][anom_mask]
                a_spat = np.array(attr_spatial)[srt][anom_mask]
                
                tot = a_doy + a_tod + a_dt + a_zscore + a_spat
                tot[tot == 0] = 1 # prevent zero division
                
                p_doy = a_doy / tot * 100
                p_tod = a_tod / tot * 100
                p_dt = a_dt / tot * 100
                p_z = a_zscore / tot * 100
                p_s = a_spat / tot * 100
                
                if hasattr(ax, '_twin_attr'):
                    ax2 = ax._twin_attr
                    ax2.clear()
                else:
                    ax2 = ax.twinx()
                    ax._twin_attr = ax2
                    
                ax2.set_ylim(0, 400) # Max 100%, squishes bars to bottom 25% of chart
                ax2.set_ylabel('Attribution %', color='purple')
                ax2.tick_params(axis='y', labelcolor='purple')
                ax2.set_yticks([0, 25, 50, 75, 100])
                
                w = 15 # bar width in days
                ax2.bar(anom_dates, p_doy, w, label='DoY', color='tab:blue', alpha=0.6)
                ax2.bar(anom_dates, p_tod, w, bottom=p_doy, label='ToD', color='tab:orange', alpha=0.6)
                ax2.bar(anom_dates, p_dt, w, bottom=p_doy+p_tod, label='dt (Multi-year)', color='tab:green', alpha=0.6)
                ax2.bar(anom_dates, p_z, w, bottom=p_doy+p_tod+p_dt, label='Z-Score', color='tab:red', alpha=0.6)
                ax2.bar(anom_dates, p_s, w, bottom=p_doy+p_tod+p_dt+p_z, label='Spatial', color='tab:purple', alpha=0.6)
                ax2.legend(loc='lower left', fontsize='small')

        # Explainer Robustness Linked Time Series (Bottom Panel)
        if ax_rob is not None and len(explainer_sens) > 0:
            valid_mask = ~np.isnan(explainer_sens) & ~np.isnan(explainer_comp)
            if np.any(valid_mask):
                v_dates = pred_dates[valid_mask]
                v_sens = explainer_sens[valid_mask]
                v_comp = explainer_comp[valid_mask]
                
                v_srt = np.argsort(v_dates)
                v_dates = v_dates[v_srt]
                v_sens = v_sens[v_srt]
                v_comp = v_comp[v_srt]
                
                ax_rob.plot(v_dates, v_sens, color='tab:red', marker='o', markersize=4, linestyle='-', linewidth=1.5, label='Sensitivity')
                if hasattr(ax_rob, '_twin_comp'):
                    ax_rob_twin = ax_rob._twin_comp
                    ax_rob_twin.clear()
                else:
                    ax_rob_twin = ax_rob.twinx()
                    ax_rob._twin_comp = ax_rob_twin
                    
                ax_rob_twin.plot(v_dates, v_comp, color='tab:brown', marker='s', markersize=4, linestyle='--', linewidth=1.5, label='Complexity')
                
                ax_rob.set_ylabel('Sensitivity', color='tab:red', fontsize=9)
                ax_rob.tick_params(axis='y', labelcolor='tab:red', labelsize=8)
                ax_rob_twin.set_ylabel('Complexity', color='tab:brown', fontsize=9)
                ax_rob_twin.tick_params(axis='y', labelcolor='tab:brown', labelsize=8)
                
                lines1, labels1 = ax_rob.get_legend_handles_labels()
                lines2, labels2 = ax_rob_twin.get_legend_handles_labels()
                ax_rob.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize='x-small')
            ax_rob.set_title("XAI Explanation Robustness (Sensitivity & Complexity)", fontsize=9)
            ax_rob.grid(True, linestyle=':', alpha=0.6)
            plt.setp(ax.get_xticklabels(), visible=False)
        
        # Missing/Insufficient Baseline
        missing_dates = pred_dates[anomaly_flags == 255]
        missing_vals = preds[1][anomaly_flags == 255]
        if len(missing_dates) > 0:
            ax.scatter(missing_dates, missing_vals, color='gray', s=30, marker='x', zorder=4, label='Missing Baseline')
        
    ax.axvline(x=split_date, color='grey', linestyle='--', label='Train-Test Split')
    if current_date is not None:
        ax.axvline(x=current_date, color='orange', linestyle='--', label='Displayed Frame')

    if ax_rob is not None:
        ax_rob.axvline(x=split_date, color='grey', linestyle='--')
        if current_date is not None:
            ax_rob.axvline(x=current_date, color='orange', linestyle='--')
        ax_rob.set_xlabel('Date')
        import matplotlib.dates as mdates
        ax_rob.xaxis.set_major_locator(mdates.YearLocator())
    else:
        ax.set_xlabel('Date')
        import matplotlib.dates as mdates
        ax.xaxis.set_major_locator(mdates.YearLocator())

    if lat is not None and lon is not None:
        ax.set_title(f"Pixel Location: ({pixel_x}, {pixel_y}) | Lat: {lat:.5f}, Lon: {lon:.5f}")
    else:
        ax.set_title(f"Pixel Location: ({pixel_x}, {pixel_y})")
    ax.set_ylabel('Spectral Complexity (Z-Score)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    ax.grid(True)
    if show_plot:
        plt.show()

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
            o[~valid_mask] = 0.0 # Set NoData to black to avoid white backgrounds
            
            return o, spc
        
    # Find frame for 2025-09-12
    dates = [datetime.fromtimestamp(ts, timezone.utc) for ts in acq_time]
    target_date = datetime(2025, 9, 12, tzinfo=timezone.utc).date()
    # Find closest date if exact not match
    diffs = [abs((d.date() - target_date).days) for d in dates]
    base_idx = np.argmin(diffs)
    
    # base frame shape (H, W, 3)
    base_frame, base_sg = get_ortho(base_idx)
    base_date = datetime.fromtimestamp(acq_time[base_idx], timezone.utc)
        
    # Load inference results
    with h5py.File(inference_results_h5, 'r') as f:
        validate_inference_results_schema(f, inference_results_h5)
        res = f['inference_results'][:]
        
    H, W = full_valid_mask.shape[1], full_valid_mask.shape[2]
    
    # Overlay Arrays
    anomaly_map = np.zeros((H, W)) # To store timestamp of first anomaly
    anomaly_map[:] = np.nan
    anomaly_counts = np.zeros((H, W)) # To store number of anomalies
    
    valid_initial_counts = np.sum(full_valid_mask, axis=0)
    insufficient_data = valid_initial_counts < 23
                
    for row in res:
        if row['Confirmed_Change'] == 1:
            x, y = row['Pixel_X'], row['Pixel_Y']
            anomaly_counts[y, x] += 1
            if np.isnan(anomaly_map[y, x]) or row['Timestamp_T_Last'] < anomaly_map[y, x]:
                anomaly_map[y, x] = row['Timestamp_T_Last']
                
    import matplotlib.patches as patches
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(18, 9))
    gs = GridSpec(2, 2, width_ratios=[1, 1.3], height_ratios=[2.5, 1], hspace=0.15)
    ax_img = fig.add_subplot(gs[:, 0])
    ax_ts = fig.add_subplot(gs[0, 1])
    ax_rob = fig.add_subplot(gs[1, 1], sharex=ax_ts)
    ax_img.imshow(base_frame)
    ax_img.set_title(f"{base_sg} Acquisition: {base_date.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    
    # Overlay insufficient data
    # Create RGBA for gray
    gray = np.zeros((H, W, 4))
    gray[insufficient_data, 0] = 0.5
    gray[insufficient_data, 1] = 0.5
    gray[insufficient_data, 2] = 0.5
    gray[insufficient_data, 3] = 0.5
    ax_img.imshow(gray)
    
    # Overlay anomalies
    if not np.all(np.isnan(anomaly_map)):
        from matplotlib.cm import viridis
        # mask anomaly map
        masked_anom = np.ma.masked_invalid(anomaly_map)
        cmap = viridis
        cmap.set_bad(color='white', alpha=0)
        im = ax_img.imshow(masked_anom, cmap=cmap, alpha=0.7)
        
        cbar = plt.colorbar(im, ax=ax_img)
        # Fix colorbar ticks to dates
        ticks = cbar.get_ticks()
        # limit ticks to valid range
        min_anom, max_anom = np.nanmin(anomaly_map), np.nanmax(anomaly_map)
        if not np.isnan(min_anom):
            ticks = ticks[(ticks >= min_anom) & (ticks <= max_anom)]
            cbar.set_ticks(ticks)
            cbar.set_ticklabels([datetime.fromtimestamp(t, timezone.utc).strftime('%Y-%m-%d') for t in ticks])
            
    # Load per-pixel baselines
    has_baselines = False
    with h5py.File(inference_results_h5, 'r') as f:
        if 'baseline_aleatoric_map' in f and 'baseline_epistemic_map' in f:
            aleatoric_map = f['baseline_aleatoric_map'][:]
            epistemic_map = f['baseline_epistemic_map'][:]
            has_baselines = True
            
    # Optional Maps Window
    ax_al = None
    ax_ep = None
    rect_al = None
    rect_ep = None
    if has_baselines:
        fig_maps, (ax_al, ax_ep) = plt.subplots(1, 2, figsize=(14, 6))
        fig_maps.canvas.manager.set_window_title('Uncertainty Heatmaps')
        
        al_plot = np.where(insufficient_data, np.nan, aleatoric_map)
        ep_plot = np.where(insufficient_data, np.nan, epistemic_map)
        
        im_al = ax_al.imshow(al_plot, cmap='magma')
        ax_al.set_title("Aleatoric Uncertainty (Natural Noise RMSE)")
        fig_maps.colorbar(im_al, ax=ax_al)
        rect_al = plt.Rectangle((0,0), 1, 1, fill=False, edgecolor='cyan', linewidth=2, visible=False)
        ax_al.add_patch(rect_al)
        
        im_ep = ax_ep.imshow(ep_plot, cmap='viridis')
        ax_ep.set_title("Epistemic Uncertainty (Model Ignorance Std)")
        fig_maps.colorbar(im_ep, ax=ax_ep)
        rect_ep = plt.Rectangle((0,0), 1, 1, fill=False, edgecolor='cyan', linewidth=2, visible=False)
        ax_ep.add_patch(rect_ep)
        
    ax_ts.text(0.5, 0.5, 'Click a pixel on the map to view data', horizontalalignment='center', verticalalignment='center', transform=ax_ts.transAxes)

    rect = patches.Rectangle((-1, -1), 1, 1, linewidth=2, edgecolor='orange', facecolor='none', visible=False)
    ax_img.add_patch(rect)

    def onclick(event):
        if event.inaxes not in [ax_img, ax_al, ax_ep]: return
        x, y = int(event.xdata), int(event.ydata)
        if x < 0 or x >= W or y < 0 or y >= H: return
        print(f"Clicked on {x}, {y}")
        
        rect.set_xy((x - 0.5, y - 0.5))
        rect.set_visible(True)
        if rect_al: rect_al.set_xy((x - 0.5, y - 0.5)); rect_al.set_visible(True)
        if rect_ep: rect_ep.set_xy((x - 0.5, y - 0.5)); rect_ep.set_visible(True)
        
        current_date_ts = None
        current_sg = None
        if not np.isnan(anomaly_map[y, x]):
            anom_ts = anomaly_map[y, x]
            # find closest acq_time
            idx = np.argmin(np.abs(acq_time - anom_ts))
            new_base, current_sg = get_ortho(idx)
            ax_img.images[0].set_array(new_base)
            current_date_ts = acq_time[idx]
        else:
            # Revert to base 2025-09-12
            ax_img.images[0].set_array(base_frame)
            current_date_ts = acq_time[base_idx]
            current_sg = base_sg
            
        current_date = datetime.fromtimestamp(current_date_ts, timezone.utc)
        ax_img.set_title(f"{current_sg} Acquisition: {current_date.strftime('%Y-%m-%d %H:%M:%S')} UTC")
        ax_ts.clear()
        ax_rob.clear()
        plot_pixel_sits(y, x, source_h5_path, inference_results_h5, ax=ax_ts, ax_rob=ax_rob, current_date=current_date)
        ax_ts.set_ylim([-7, 7])
        fig.canvas.draw()

    fig.canvas.mpl_connect('button_press_event', onclick)
    if has_baselines:
        fig_maps.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

if __name__ == "__main__":
    if os.path.exists(INFERENCE_H5):
        plot_spatial_anomaly_overlay(H5_PATH, INFERENCE_H5)
    else:
        print("Run inference first to create inference_results.h5")
