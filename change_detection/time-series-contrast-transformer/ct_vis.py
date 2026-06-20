import h5py
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timezone
import os
import scienceplots
import matplotlib.patches as patches
import matplotlib.dates as mdates
from matplotlib.cm import viridis

plt.style.use(['science', 'no-latex'])

LOCATION = "Tait"
H5_PATH = f"C:/satelliteImagery/HLST30/HLST_{LOCATION}_Harmonized_SC_EM-7_Norm-bandCount.h5"
OUTPUT_DIR = "C:/satelliteImagery/HLST30/ContrastiveTransformer"
INFERENCE_H5 = os.path.join(OUTPUT_DIR, f"{LOCATION}_ContrastiveTransformer_RegimeShifts.h5")

def plot_pixel_sits(pixel_y, pixel_x, source_h5_path, inference_h5_path, ax=None, current_date=None):
    """Visualizes the 1D time series and Contrastive Transformer representations for a pixel."""
    
    with h5py.File(source_h5_path, 'r') as f:
        harm_grp = f['/HDFEOS/GRIDS/HARMONIZED/Data Fields']
        acq_time = harm_grp['sliding_volume_z_score'].attrs['acquisition_time'][:]
        z_score = harm_grp['sliding_volume_z_score'][:, pixel_y, pixel_x]
        z_score = np.clip(z_score, -5.0, 5.0)
        
        unified_masks = harm_grp['common_mask'][:, pixel_y, pixel_x]
        is_invalid = unified_masks.astype(bool)
        
        spacecraft_bytes = harm_grp['sliding_volume_z_score'].attrs['source_spacecraft'][:]
        spacecrafts = [s.decode('utf-8') if isinstance(s, bytes) else str(s) for s in spacecraft_bytes]
            
    dates = [datetime.fromtimestamp(ts, timezone.utc) for ts in acq_time]
    dates_arr = np.array(dates)
    
    # Load inference results for this pixel
    with h5py.File(inference_h5_path, 'r') as f:
        window_times = f['embeddings/window_center_times'][:]
        win_dates_arr = np.array([datetime.fromtimestamp(ts, timezone.utc) for ts in window_times])
        
        drift_flags = f['temporal_drift/flags'][:, pixel_y, pixel_x]
        cosine_dist = f['temporal_drift/cosine_distance'][:, pixel_y, pixel_x]
        
        coher_flags = f['spatial_coherence/flags'][:, pixel_y, pixel_x]
        coher_sim = f['spatial_coherence/neighbor_similarity'][:, pixel_y, pixel_x]
        
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
        show_plot = True
    else:
        show_plot = False
    
    valid_mask = ~is_invalid
    spacecrafts_arr = np.array(spacecrafts)
    
    # Plot Observations
    for marker_type, sc_keyword in [('s', 'Sentinel'), ('o', 'Landsat'), ('D', 'Tanager')]:
        sc_mask = np.array([sc_keyword.lower() in str(sc).lower() for sc in spacecrafts_arr])
        
        idx_valid = valid_mask & sc_mask
        if np.any(idx_valid):
            ax.plot(dates_arr[idx_valid], z_score[idx_valid], color='k', marker=marker_type, 
                    linestyle='None', label=f'Valid Actuals ({sc_keyword})')
            
        idx_invalid = is_invalid & sc_mask
        if np.any(idx_invalid):
            ax.plot(dates_arr[idx_invalid], z_score[idx_invalid], color='gray', marker=marker_type, 
                    linestyle='None', markerfacecolor='none', label=f'Cloud Masked ({sc_keyword})')

    # Plot Anomalies at actual observation timestamps
    # For window-based anomalies, find the closest valid observation to the anomalous window center
    def get_closest_valid_idx(target_ts):
        valid_idx = np.where(valid_mask)[0]
        if len(valid_idx) == 0:
            return None
        closest = valid_idx[np.argmin(np.abs(acq_time[valid_idx] - target_ts))]
        return closest

    drift_win_idx = np.where(drift_flags == 1)[0]
    coher_win_idx = np.where(coher_flags == 1)[0]
    
    plotted_drift = []
    plotted_coher = []
    
    for wi in drift_win_idx:
        ts = window_times[wi]
        idx = get_closest_valid_idx(ts)
        if idx is not None and idx not in plotted_drift:
            plotted_drift.append(idx)
            
    for wi in coher_win_idx:
        ts = window_times[wi]
        idx = get_closest_valid_idx(ts)
        if idx is not None and idx not in plotted_coher:
            plotted_coher.append(idx)
    
    if len(plotted_drift) > 0:
        ax.scatter(dates_arr[plotted_drift], z_score[plotted_drift], color='red', marker='X', s=80, 
                   zorder=4, label='Temporal Drift Anomaly')
        
    if len(plotted_coher) > 0:
        ax.scatter(dates_arr[plotted_coher], z_score[plotted_coher], color='darkorange', marker='*', s=250, 
                   zorder=5, label='Spatial Coherence Anomaly')

    # Add secondary axis for the contrastive metrics
    ax2 = ax.twinx()
    ax2.set_ylabel('Contrastive Metrics', color='purple')
    ax2.plot(win_dates_arr, cosine_dist, color='purple', linestyle='-', marker='.', alpha=0.6, label='Temporal Cosine Distance')
    ax2.plot(win_dates_arr, 1.0 - coher_sim, color='green', linestyle='--', marker='+', alpha=0.6, label='Spatial Divergence (1 - Sim)')
    ax2.set_ylim(-0.1, 2.0)
    ax2.tick_params(axis='y', labelcolor='purple')

    if current_date is not None:
        ax.axvline(x=current_date, color='orange', linestyle='--', label='Displayed Frame')

    ax.set_title(f"Pixel Location: ({pixel_x}, {pixel_y}) | Contrastive Transformer Regime Shifts")
    ax.set_xlabel('Date')
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.set_ylabel('Spectral Complexity (Z-Score)')
    
    # Combine legends
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    ax.grid(True)
    if show_plot:
        plt.show()

def get_ortho(idx, source_h5_path):
    with h5py.File(source_h5_path, 'r') as f:
        harm_grp = f['/HDFEOS/GRIDS/HARMONIZED/Data Fields']
        spc = harm_grp['sliding_volume_z_score'].attrs['source_spacecraft'][idx]
        spc = spc.decode('utf-8') if isinstance(spc, bytes) else str(spc)
        
        o = harm_grp['ortho_visual'][idx]
        o = np.transpose(o, (1, 2, 0)).astype(np.float32) / 255.0
            
        valid_mask = np.all(o > 0, axis=-1)
        o[~valid_mask] = 0.0 # Set NoData to black
        
        return o, spc

def plot_spatial_anomaly_overlay(source_h5_path, inference_results_h5):
    with h5py.File(source_h5_path, 'r') as f:
        harm_grp = f['/HDFEOS/GRIDS/HARMONIZED/Data Fields']
        acq_time = harm_grp['sliding_volume_z_score'].attrs['acquisition_time'][:]
        unified_masks = harm_grp['common_mask'][:]
        full_valid_mask = ~unified_masks.astype(bool)
        
    # Find base frame (approx 2025-09-12 for consistency)
    dates = [datetime.fromtimestamp(ts, timezone.utc) for ts in acq_time]
    target_date = datetime(2025, 9, 12, tzinfo=timezone.utc).date()
    diffs = [abs((d.date() - target_date).days) for d in dates]
    base_idx = np.argmin(diffs)
    
    base_frame, base_sg = get_ortho(base_idx, source_h5_path)
    base_date = datetime.fromtimestamp(acq_time[base_idx], timezone.utc)
        
    # Load inference results
    with h5py.File(inference_results_h5, 'r') as f:
        drift_map = f['temporal_drift/change_date_timestamp'][:]
        coher_map = f['spatial_coherence/change_date_timestamp'][:]
        
    H, W = full_valid_mask.shape[1], full_valid_mask.shape[2]
    
    # Create combined anomaly map (take earliest trigger if both trigger)
    combined_map = np.full((H, W), np.nan)
    
    for y in range(H):
        for x in range(W):
            r = drift_map[y, x]
            d = coher_map[y, x]
            
            if not np.isnan(r) and not np.isnan(d):
                combined_map[y, x] = min(r, d)
            elif not np.isnan(r):
                combined_map[y, x] = r
            elif not np.isnan(d):
                combined_map[y, x] = d
                
    valid_initial_counts = np.sum(full_valid_mask, axis=0)
    insufficient_data = valid_initial_counts < 10
                
    fig, (ax_img, ax_ts) = plt.subplots(1, 2, figsize=(18, 8))
    ax_img.imshow(base_frame)
    ax_img.set_title(f"{base_sg} Acquisition: {base_date.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    
    # Overlay insufficient data
    gray = np.zeros((H, W, 4))
    gray[insufficient_data, 0] = 0.5
    gray[insufficient_data, 1] = 0.5
    gray[insufficient_data, 2] = 0.5
    gray[insufficient_data, 3] = 0.5
    ax_img.imshow(gray)
    
    # Overlay anomalies
    if not np.all(np.isnan(combined_map)):
        masked_anom = np.ma.masked_invalid(combined_map)
        cmap = viridis
        cmap.set_bad(color='white', alpha=0)
        im = ax_img.imshow(masked_anom, cmap=cmap, alpha=0.7)
        
        cbar = plt.colorbar(im, ax=ax_img)
        ticks = cbar.get_ticks()
        min_anom, max_anom = np.nanmin(combined_map), np.nanmax(combined_map)
        if not np.isnan(min_anom):
            ticks = ticks[(ticks >= min_anom) & (ticks <= max_anom)]
            cbar.set_ticks(ticks)
            cbar.set_ticklabels([datetime.fromtimestamp(t, timezone.utc).strftime('%Y-%m-%d') for t in ticks])
            
    ax_ts.text(0.5, 0.5, 'Click a pixel on the map to view data', horizontalalignment='center', verticalalignment='center', transform=ax_ts.transAxes)

    rect = patches.Rectangle((-1, -1), 1, 1, linewidth=2, edgecolor='orange', facecolor='none', visible=False)
    ax_img.add_patch(rect)

    def onclick(event):
        if event.inaxes != ax_img: return
        x, y = int(event.xdata), int(event.ydata)
        if x < 0 or x >= W or y < 0 or y >= H: return
        print(f"Clicked on {x}, {y}")
        
        rect.set_xy((x - 0.5, y - 0.5))
        rect.set_visible(True)
        
        current_date_ts = None
        current_sg = None
        if not np.isnan(combined_map[y, x]):
            anom_ts = combined_map[y, x]
            idx = np.argmin(np.abs(acq_time - anom_ts))
            new_base, current_sg = get_ortho(idx, source_h5_path)
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
        ax_ts.set_ylim([-5, 5])
        fig.canvas.draw()

    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

if __name__ == "__main__":
    if os.path.exists(INFERENCE_H5):
        plot_spatial_anomaly_overlay(H5_PATH, INFERENCE_H5)
    else:
        print(f"Run inference first to create {INFERENCE_H5}")
