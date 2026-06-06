import h5py
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timezone
import os
import scienceplots
import pyproj
plt.style.use(['science','no-latex'])


def plot_pixel_sits(pixel_y, pixel_x, source_h5_path, inference_results_h5, ax=None, current_date=None, train_end_date="2024-01-01"):
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
        res = f['inference_results'][:]
        if 'train_end_date' in f['inference_results'].attrs:
            val = f['inference_results'].attrs['train_end_date']
            train_end_date = val.decode('utf-8') if isinstance(val, bytes) else str(val)
        
    mask = (res['Pixel_X'] == pixel_x) & (res['Pixel_Y'] == pixel_y)
    pixel_res = res[mask]
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
        show_plot = True
    else:
        show_plot = False
    
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
        mean_residual = []
        preds1 = []
        preds2 = []
        preds3 = []
        anomaly_flags = []
        for row in pixel_res:
            d23 = datetime.fromtimestamp(row['Timestamp_T23'], timezone.utc)
            pred_dates.append(d23)
            mean_residual.append(row['Mean_Residual'])
            preds1.append(row['Pred_1'])
            preds2.append(row['Pred_2'])
            preds3.append(row['Pred_3'])
            anomaly_flags.append(row['Anomaly_Flag'])
            
        

        srt = np.argsort(pred_dates)
        pred_dates = np.array(pred_dates)[srt]
        preds1 = np.array(preds1)[srt]
        preds2 = np.array(preds2)[srt]
        preds3 = np.array(preds3)[srt]
        anomaly_flags = np.array(anomaly_flags)[srt]
        upper_bound = preds1 + 3 * np.std(mean_residual)
        lower_bound = preds1 - 3 * np.std(mean_residual)
        
        ax.plot(pred_dates, preds1, 'r--', label='Prediction t+1')
        ax.plot(pred_dates, preds2, 'g--', label='Prediction t+2')
        ax.plot(pred_dates, preds3, 'b--', label='Prediction t+3')
        ax.fill_between(pred_dates, lower_bound, upper_bound, alpha=0.1, label='Prediction t+1±3σ')
        
        # Anomalies
        anom_dates = pred_dates[anomaly_flags > 0]
        anom_preds = preds3[anomaly_flags > 0]
        ax.plot(anom_dates, anom_preds, 'rx', markersize=10, mew=2, label='Anomalies')
        
    ax.axvline(x=split_date, color='grey', linestyle='--', label='Train-Test Split')
    if current_date is not None:
        ax.axvline(x=current_date, color='orange', linestyle='--', label='Displayed Frame')

    if lat is not None and lon is not None:
        ax.set_title(f"Pixel Location: ({pixel_x}, {pixel_y}) | Lat: {lat:.5f}, Lon: {lon:.5f}")
    else:
        ax.set_title(f"Pixel Location: ({pixel_x}, {pixel_y})")
    ax.set_xlabel('Date')
    ax.set_ylabel('Spectral Complexity (Z-Score)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    ax.grid(True)
    if show_plot:
        plt.show()

def plot_spatial_anomaly_overlay(source_h5_path, inference_results_h5, train_end_date="2024-01-01"):
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
        res = f['inference_results'][:]
        if 'train_end_date' in f['inference_results'].attrs:
            val = f['inference_results'].attrs['train_end_date']
            train_end_date = val.decode('utf-8') if isinstance(val, bytes) else str(val)
        
    H, W = full_valid_mask.shape[1], full_valid_mask.shape[2]
    
    # Overlay Arrays
    anomaly_map = np.zeros((H, W)) # To store timestamp of first anomaly
    anomaly_map[:] = np.nan
    anomaly_counts = np.zeros((H, W)) # To store number of anomalies
    
    valid_initial_counts = np.sum(full_valid_mask, axis=0)
    insufficient_data = valid_initial_counts < 23
                
    for row in res:
        if row['Anomaly_Flag']:
            x, y = row['Pixel_X'], row['Pixel_Y']
            anomaly_counts[y, x] += 1
            if np.isnan(anomaly_map[y, x]) or row['Timestamp_T23'] < anomaly_map[y, x]:
                anomaly_map[y, x] = row['Timestamp_T23']
                
    # Filter out pixels with <= 2 anomalies
    anomaly_map[anomaly_counts <= 2] = np.nan
                
    import matplotlib.patches as patches
    fig, (ax_img, ax_ts) = plt.subplots(1, 2, figsize=(18, 8))
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
        
    rect = patches.Rectangle((-1, -1), 1, 1, linewidth=2, edgecolor='orange', facecolor='none', visible=False)
    ax_img.add_patch(rect)
    
    # Add initial placeholder text for the time series subplot
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
            # Change basemap to first date of anomaly
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
        plot_pixel_sits(y, x, source_h5_path, inference_results_h5, ax=ax_ts, current_date=current_date, train_end_date=train_end_date)
        ax_ts.set_ylim([-4, 4])
        fig.canvas.draw()

    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

if __name__ == "__main__":
    h5_path = "C:/satelliteImagery/HLST30/HLST_Malibu_Harmonized_SC_EM-7_Norm-bandCount.h5"
    inference_h5 = "c:/satelliteImagery/HLST30/1D-CNN-Malibu/inference_results.h5"
    if h5py.is_hdf5(inference_h5):
        plot_spatial_anomaly_overlay(h5_path, inference_h5, train_end_date="2024-01-01")
    else:
        print("Run inference first to create inference_results.h5")
