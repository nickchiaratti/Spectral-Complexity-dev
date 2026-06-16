import h5py
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timezone
import os
import pyproj
import matplotlib.patches as patches
import scienceplots
plt.style.use(['science','no-latex'])

LOCATION = "Malibu"
H5_PATH = f"C:/satelliteImagery/HLST30/HLST_{LOCATION}_Harmonized_SC_EM-7_Norm-bandCount.h5"
CONFIG = 'C0L1Q1_P4_E1'

import glob

def get_inference_h5(location, config):
    search_pattern = f"C:/satelliteImagery/HLST30/CCD/{location}_CCD_Harmonized_Change_Detection_{config}.h5"
    files = glob.glob(search_pattern)
    if not files:
        return None
    # Sort by modification time, descending
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]

def plot_pixel_sits(pixel_y, pixel_x, source_h5_path, inference_results_h5, ax=None, current_date=None):
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
                print(f"Warning: Could not compute lat/lon: {e}")
        
    dates = [datetime.fromtimestamp(ts, timezone.utc) for ts in acq_time]
    
    with h5py.File(inference_results_h5, 'r') as f:
        predicted = f['predicted_series'][:, pixel_y, pixel_x]
        rmse = f['rmse_series'][:, pixel_y, pixel_x]
        anomalies = f['anomaly_flags'][:, pixel_y, pixel_x]
        rmse_multiplier = f.attrs.get('RMSE_MULTIPLIER', 3.0)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
        show_plot = True
    else:
        show_plot = False
        
    valid_mask = ~is_invalid
    dates_arr = np.array(dates)
    spacecrafts_arr = np.array(spacecrafts)
    
    # Plot Actuals
    for marker_type, sc_keyword in [('s', 'Sentinel'), ('o', 'Landsat'), ('D', 'Tanager')]:
        sc_mask = np.array([sc_keyword.lower() in str(sc).lower() for sc in spacecrafts_arr])
        
        idx_valid = valid_mask & sc_mask
        if np.any(idx_valid):
            ax.plot(dates_arr[idx_valid], z_score[idx_valid], color='k', marker=marker_type, linestyle='None', label=f'Valid ({sc_keyword})')
            
        idx_invalid = is_invalid & sc_mask
        if np.any(idx_invalid):
            ax.plot(dates_arr[idx_invalid], z_score[idx_invalid], color='gray', marker=marker_type, linestyle='None', markerfacecolor='none', label=f'Invalid ({sc_keyword})')
            
    # Plot Predictions
    pred_mask = ~np.isnan(predicted)
    if np.any(pred_mask):
        pred_dates = dates_arr[pred_mask]
        preds = predicted[pred_mask]
        rmses = rmse[pred_mask]
        
        upper_bound = preds + rmse_multiplier * rmses
        lower_bound = preds - rmse_multiplier * rmses
        
        ax.plot(pred_dates, preds, 'b--', label='Harmonic Prediction (t+1)')
        ax.fill_between(pred_dates, lower_bound, upper_bound, color='blue', alpha=0.15, label=f'Prediction ±{rmse_multiplier}σ')
        
        anom_mask = anomalies[pred_mask] == 1
        if np.any(anom_mask):
            anom_dates = pred_dates[anom_mask]
            anom_vals = z_score[pred_mask][anom_mask]  # Highlight the actuals that are anomalous
            ax.plot(anom_dates, anom_vals, 'rx', markersize=10, mew=2, label='Anomalies')
            
    if current_date is not None:
        ax.axvline(x=current_date, color='orange', linestyle='--', label='Displayed Frame')

    if lat is not None and lon is not None:
        ax.set_title(f"Pixel Location: ({pixel_x}, {pixel_y}) | Lat: {lat:.5f}, Lon: {lon:.5f}")
    else:
        ax.set_title(f"Pixel Location: ({pixel_x}, {pixel_y})")
    ax.set_xlabel('Date')
    import matplotlib.dates as mdates
    ax.xaxis.set_major_locator(mdates.YearLocator())
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
        min_samples = f.attrs.get('MIN_SAMPLES', 20)
        
    H, W = full_valid_mask.shape[1], full_valid_mask.shape[2]
    
    anomaly_map[change_count_map == 0] = np.nan
    
    fig, (ax_img, ax_ts) = plt.subplots(1, 2, figsize=(18, 8))
    ax_img.imshow(base_frame)
    ax_img.set_title(f"{base_sg} Acquisition: {base_date.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    
    valid_initial_counts = np.sum(full_valid_mask, axis=0)
    insufficient_data = valid_initial_counts < min_samples
    
    gray = np.zeros((H, W, 4))
    gray[insufficient_data, 0] = 0.5
    gray[insufficient_data, 1] = 0.5
    gray[insufficient_data, 2] = 0.5
    gray[insufficient_data, 3] = 0.5
    ax_img.imshow(gray)
    
    if not np.all(np.isnan(anomaly_map)):
        from matplotlib.cm import viridis
        masked_anom = np.ma.masked_invalid(anomaly_map)
        cmap = viridis
        cmap.set_bad(color='white', alpha=0)
        im = ax_img.imshow(masked_anom, cmap=cmap, alpha=0.7)
        cbar = plt.colorbar(im, ax=ax_img)
        ticks = cbar.get_ticks()
        min_anom, max_anom = np.nanmin(anomaly_map), np.nanmax(anomaly_map)
        if not np.isnan(min_anom):
            ticks = ticks[(ticks >= min_anom) & (ticks <= max_anom)]
            cbar.set_ticks(ticks)
            cbar.set_ticklabels([datetime.fromtimestamp(t, timezone.utc).strftime('%Y-%m-%d') for t in ticks])
            
    rect = patches.Rectangle((-1, -1), 1, 1, linewidth=2, edgecolor='orange', facecolor='none', visible=False)
    ax_img.add_patch(rect)
    
    ax_ts.text(0.5, 0.5, 'Click a pixel on the map to view data', horizontalalignment='center', verticalalignment='center', transform=ax_ts.transAxes)

    def onclick(event):
        if event.inaxes != ax_img: return
        x, y = int(event.xdata), int(event.ydata)
        if x < 0 or x >= W or y < 0 or y >= H: return
        print(f"Clicked on {x}, {y}")
        
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
        ax_img.set_title(f"{current_sg} Acquisition: {current_date.strftime('%Y-%m-%d %H:%M:%S')} UTC")
        ax_ts.clear()
        plot_pixel_sits(y, x, source_h5_path, inference_results_h5, ax=ax_ts, current_date=current_date)
        ax_ts.set_ylim([-4, 4])
        fig.canvas.draw()

    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

if __name__ == "__main__":
    inference_h5 = get_inference_h5(LOCATION, CONFIG)
    if inference_h5 and os.path.exists(inference_h5):
        print(f"Loading latest inference results: {inference_h5}")
        plot_spatial_anomaly_overlay(H5_PATH, inference_h5)
    else:
        print("Run harmonized_CCD_main.py first to create output h5")
