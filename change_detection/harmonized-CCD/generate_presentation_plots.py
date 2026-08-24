import os
import sys
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
from datetime import datetime, timezone
import scienceplots
import re
import glob

plt.style.use(['science','no-latex'])
# Set presentation quality parameters
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.dpi': 300
})

N_ENDMEMBERS = 4
# Target inputs
loc = "Tait"
px_y = 16
px_x = 76

# Bring in custom module
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if root_dir not in sys.path:
    sys.path.append(root_dir)
import SpecComplex as sc

def get_inference_h5(location, config, target_metric):
    search_pattern = f"C:/satelliteImagery/HLST30/CCD/{location}_CCD_Harmonized_Change_Detection_{target_metric}_{config}.h5"
    files = glob.glob(search_pattern)
    if not files: return None
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]

def get_season(dt):
    m = dt.month
    if m in [12, 1, 2]: return 'Winter'
    if m in [3, 4, 5]: return 'Spring'
    if m in [6, 7, 8]: return 'Summer'
    return 'Fall'

def save_ortho_visual(pixel_y, pixel_x, source_h5_path, inference_results_h5, out_path):
    print("Generating Ortho Visual Overlay...")
    with h5py.File(inference_results_h5, 'r') as f:
        target_metric = f.attrs.get('TARGET_METRIC', 'sliding_volume_z_score')
        anomaly_map = f['change_date_timestamp'][:]
        change_count_map = f['change_count'][:]
        
    with h5py.File(source_h5_path, 'r') as f:
        harm_grp = f['/HDFEOS/GRIDS/HARMONIZED/Data Fields']
        acq_time = harm_grp[target_metric].attrs['acquisition_time'][:]
        
        geo_str = ""
        geo_transform = harm_grp['common_mask'].attrs.get('GeoTransform')
        spatial_ref = harm_grp['common_mask'].attrs.get('spatial_ref')
        if geo_transform is not None and spatial_ref is not None:
            import pyproj
            try:
                if isinstance(spatial_ref, bytes):
                    spatial_ref = spatial_ref.decode('utf-8')
                transformer = pyproj.Transformer.from_crs(spatial_ref, "EPSG:4326", always_xy=True)
                X = geo_transform[0] + (pixel_x + 0.5) * geo_transform[1] + (pixel_y + 0.5) * geo_transform[2]
                Y = geo_transform[3] + (pixel_x + 0.5) * geo_transform[4] + (pixel_y + 0.5) * geo_transform[5]
                lon, lat = transformer.transform(X, Y)
                geo_str = f" ({lat:.4f}°, {lon:.4f}°)"
            except Exception as e:
                print(f"Error calculating geo coordinates: {e}")
        
    def get_ortho(idx):
        with h5py.File(source_h5_path, 'r') as f:
            harm_grp = f['/HDFEOS/GRIDS/HARMONIZED/Data Fields']
            o = harm_grp['ortho_visual'][idx]
            o = np.transpose(o, (1, 2, 0)).astype(np.float32) / 255.0
            valid_mask = np.all(o > 0, axis=-1)
            o[~valid_mask] = 0.0
            return o
            
    dates = [datetime.fromtimestamp(ts, timezone.utc) for ts in acq_time]
    pixel_change_ts = anomaly_map[pixel_y, pixel_x]
    if change_count_map[pixel_y, pixel_x] > 0 and pixel_change_ts > 0:
        target_date = datetime.fromtimestamp(pixel_change_ts, timezone.utc).date()
    else:
        target_date = datetime(2025, 9, 12, tzinfo=timezone.utc).date()
        
    diffs = [abs((d.date() - target_date).days) for d in dates]
    base_idx = np.argmin(diffs)
    base_frame = get_ortho(base_idx)
    base_date = datetime.fromtimestamp(acq_time[base_idx], timezone.utc)
    
    anomaly_map[change_count_map == 0] = np.nan
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(base_frame)
    ax.set_title(f"Pixel: x={pixel_x}, y={pixel_y}{geo_str}\nIdentified Date of Change: {target_date.strftime('%Y-%m-%d')} UTC")
    ax.set_xticks([])
    ax.set_yticks([])
    
    cmap = plt.cm.gist_rainbow
    cmap.set_bad('none')
    im = ax.imshow(anomaly_map, cmap=cmap, alpha=0.7)
    
    import matplotlib.patheffects as pe
    # High-contrast bounding box for target pixel (widened to 3x3 to keep the center pixel visible)
    rect = patches.Rectangle((pixel_x - 1.5, pixel_y - 1.5), 3, 3, 
                             linewidth=2.5, edgecolor='white', facecolor='none',
                             path_effects=[pe.withStroke(linewidth=4.5, foreground='black')])
    ax.add_patch(rect)
    
    # Add an arrow pointing to the pixel
    ax.annotate(
        'Target Pixel', 
        xy=(pixel_x, pixel_y), 
        xytext=(pixel_x - 15, pixel_y - 10), 
        arrowprops=dict(facecolor='white', edgecolor='black', shrink=0.15, width=2, headwidth=8),
        ha='right', va='bottom', color='white', fontweight='bold', fontsize=12,
        path_effects=[pe.withStroke(linewidth=3, foreground='black')]
    )
    
    import matplotlib.ticker as ticker
    def format_unix_timestamp(x, pos):
        if np.isnan(x) or x <= 0:
            return ""
        try:
            return datetime.fromtimestamp(x, timezone.utc).strftime('%Y-%m')
        except:
            return ""
            
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.15)
    
    cbar = plt.colorbar(im, cax=cax, format=ticker.FuncFormatter(format_unix_timestamp))
    cbar.set_label('Date of Detected Change', rotation=270, labelpad=25)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

def save_time_series(pixel_y, pixel_x, source_h5_path, inference_results_h5, out_path):
    print("Generating Time Series Plot...")
    with h5py.File(inference_results_h5, 'r') as f:
        target_metric = f.attrs.get('TARGET_METRIC', 'sliding_volume_z_score')
        
    # Read acquisition time, masks, spacecrafts and target metric values from source
    with h5py.File(source_h5_path, 'r') as f_sc:
        harm_grp = f_sc['/HDFEOS/GRIDS/HARMONIZED/Data Fields']
        if target_metric not in harm_grp:
            target_metric = list(harm_grp.keys())[0]
            
        attrs = harm_grp[target_metric].attrs
        acq_time = attrs['acquisition_time'][:]
        spacecrafts = [s.decode('utf-8') if isinstance(s, bytes) else str(s) for s in attrs['source_spacecraft'][:]]
        unified_masks = harm_grp['common_mask'][:, pixel_y, pixel_x]
        val_series = harm_grp[target_metric][:, pixel_y, pixel_x]

    # Read inference data
    with h5py.File(inference_results_h5, 'r') as f_inf:
        # Use anomaly flags (which corresponds to detected change)
        change_indices = f_inf['anomaly_flags'][:, pixel_y, pixel_x]
        rmse_series = f_inf['rmse_series'][:, pixel_y, pixel_x]
        predicted = f_inf['predicted_series'][:, pixel_y, pixel_x]
        if 'CHANGE_PROBABILITY' in f_inf.attrs:
            from scipy.stats import chi2
            p = f_inf.attrs['CHANGE_PROBABILITY']
            df = f_inf.attrs.get('CHI2_DEGREES_OF_FREEDOM', 1)
            rmse_multiplier = np.sqrt(chi2.ppf(p, df=df))
        else:
            rmse_multiplier = f_inf.attrs.get('RMSE_MULTIPLIER', 3.0)
        
    dates = [datetime.fromtimestamp(ts, timezone.utc) for ts in acq_time]
    
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_title(f"Target Metric Time Series: {target_metric}\nPixel: x={pixel_x}, y={pixel_y}")
    ax.set_ylabel("Spectral Complexity")
    
    valid_mask = ~unified_masks.astype(bool)
    is_invalid = ~valid_mask
    dates_arr = np.array(dates)
    spacecrafts_arr = np.array(spacecrafts)
    
    # Plot Actuals
    is_anomaly = (change_indices == 1)
    for marker_type, sc_keyword in [('s', 'Sentinel'), ('o', 'Landsat'), ('D', 'Tanager')]:
        sc_mask = np.array([sc_keyword.lower() in str(sc).lower() for sc in spacecrafts_arr])
        
        idx_valid_normal = valid_mask & sc_mask & ~is_anomaly
        if np.any(idx_valid_normal):
            ax.plot(dates_arr[idx_valid_normal], val_series[idx_valid_normal], color='k', marker=marker_type, linestyle='None', label=f'Valid ({sc_keyword})')
            
        idx_valid_anomaly = valid_mask & sc_mask & is_anomaly
        if np.any(idx_valid_anomaly):
            ax.plot(dates_arr[idx_valid_anomaly], val_series[idx_valid_anomaly], color='r', marker=marker_type, linestyle='None', label=f'Anomaly ({sc_keyword})')
            
        idx_invalid = is_invalid & sc_mask
        if np.any(idx_invalid):
            ax.plot(dates_arr[idx_invalid], val_series[idx_invalid], color='gray', marker=marker_type, linestyle='None', markerfacecolor='none', label=f'Invalid ({sc_keyword})')
            
    # Reconstruct segments
    segments = []
    current_seg = []
    current_rmse = None
    for i in range(len(acq_time)):
        if unified_masks[i]: continue
        r = rmse_series[i]
        if np.isnan(r): continue
        if current_rmse is None:
            current_rmse = r
            current_seg.append(i)
        elif abs(r - current_rmse) < 1e-6:
            current_seg.append(i)
        else:
            segments.append(current_seg)
            current_seg = [i]
            current_rmse = r
    if current_seg:
        segments.append(current_seg)
        
    segment_colors = ['blue', 'green', 'purple', 'orange', 'cyan', 'magenta', 'brown']
    for i, seg in enumerate(segments):
        if not seg: continue
        seg_times = acq_time[seg]
        seg_dates = [dates[idx] for idx in seg]
        
        c = segment_colors[i % len(segment_colors)]
        
        t0 = seg_times[0]
        x_norm = (seg_times - t0) / (365.25 * 24 * 3600)
        
        y_pred = predicted[seg]
        r = rmse_series[seg[0]] * rmse_multiplier
        
        bound_label = f'Segment {i+1} Anomaly Bounds (1±RMSE)'
        pred_label = f'Segment {i+1} Harmonic Fit'
        
        ax.plot(seg_dates, y_pred, color=c, linewidth=2, label=pred_label, zorder=2)
        ax.fill_between(seg_dates, y_pred - r, y_pred + r, color=c, alpha=0.15, label=bound_label, zorder=1)

    # Anomalies are now plotted with their respective sensor markers in red above.
    
    valid_vals = val_series[~np.isnan(val_series) & ~unified_masks.astype(bool)]
    if len(valid_vals) > 0:
        min_val = np.min(valid_vals)
        max_val = np.max(valid_vals)
        y_min = min_val * 1.1 if min_val < 0 else min_val * 0.9
        y_max = max_val * 1.1 if max_val > 0 else max_val * 0.9
        if y_min == y_max:
            y_min, y_max = y_min - 1, y_max + 1
        ax.set_ylim(y_min, y_max)
        
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    ax.grid(True, which='major', linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

def save_separability_plot(pixel_y, pixel_x, source_h5_path, inference_results_h5, out_path):
    print("Generating Separability Plot...")
    with h5py.File(inference_results_h5, 'r') as f_inf:
        rmse = f_inf['rmse_series'][:, pixel_y, pixel_x]
        
    with h5py.File(source_h5_path, 'r') as f_sc:
        harm_grp = f_sc['/HDFEOS/GRIDS/HARMONIZED/Data Fields']
        target_metric = f_sc.attrs.get('TARGET_METRIC', 'sliding_volume_z_score')
        if target_metric not in harm_grp:
            target_metric = list(harm_grp.keys())[0]
            
        attrs = harm_grp[target_metric].attrs
        acq_time = attrs['acquisition_time'][:]
        source_grids = [s.decode('utf-8') if isinstance(s, bytes) else str(s) for s in attrs['source_grid'][:]]
        source_frames = attrs['source_frame_index'][:]
        spacecrafts = [s.decode('utf-8') if isinstance(s, bytes) else str(s) for s in attrs['source_spacecraft'][:]]
        unified_masks = harm_grp['common_mask'][:, pixel_y, pixel_x]
        
    raw_h5_path = re.sub(r'_SC_.*?\.h5$', '.h5', source_h5_path)
    
    segments = []
    current_seg = []
    current_rmse = None
    for i in range(len(acq_time)):
        if unified_masks[i]: continue
        r = rmse[i]
        if np.isnan(r): continue
        if current_rmse is None:
            current_rmse = r
            current_seg.append(i)
        elif abs(r - current_rmse) < 1e-6:
            current_seg.append(i)
        else:
            segments.append(current_seg)
            current_seg = [i]
            current_rmse = r
    if current_seg:
        segments.append(current_seg)
        
    target_seasons = ['Spring', 'Summer', 'Fall']
    target_sensors = ['Landsat', 'Sentinel-2', 'Tanager']
    n_sensors = len(target_sensors)
    segment_colors = ['blue', 'green', 'purple', 'orange', 'cyan', 'magenta', 'brown']
    
    fig, axes = plt.subplots(n_sensors, 3, figsize=(18, 5 * n_sensors), squeeze=False)
    fig.suptitle(f"Seasonal Separability Across Segments\nPixel: x={pixel_x}, y={pixel_y}", fontsize=18)
    
    global_y_max = 0
    global_wl_min = float('inf')
    global_wl_max = float('-inf')
    
    def get_sensor_cat(s_str):
        s_str = str(s_str).lower()
        if 'landsat' in s_str: return 'Landsat'
        if 'sentinel' in s_str: return 'Sentinel-2'
        if 'tanager' in s_str: return 'Tanager'
        return None
    
    with h5py.File(raw_h5_path, 'r') as f_raw:
        for row_idx, sensor_cat in enumerate(target_sensors):
            for sea_idx, season in enumerate(target_seasons):
                ax = axes[row_idx, sea_idx]
                ax.set_title(f"{sensor_cat} - {season}")
                ax.set_xlabel("Wavelength (nm)")
                if sea_idx == 0:
                    ax.set_ylabel("Surface Reflectance")
                    
                for seg_idx, seg in enumerate(segments):
                    seg_color = segment_colors[seg_idx % len(segment_colors)]
                    
                    season_spectra = []
                    
                    for i in seg:
                        if get_sensor_cat(spacecrafts[i]) != sensor_cat:
                            continue
                            
                        dt = datetime.fromtimestamp(acq_time[i], timezone.utc)
                        if get_season(dt) != season: continue
                            
                        grid = source_grids[i]
                        frame_idx = source_frames[i]
                        
                        sr_ds = f_raw[f"/HDFEOS/GRIDS/{grid}/Data Fields/surface_reflectance"]
                        patch = sr_ds[frame_idx, :, max(0, pixel_y-1):pixel_y+2, max(0, pixel_x-1):pixel_x+2]
                        
                        w_attr = sr_ds.attrs.get('wavelengths', sr_ds.attrs.get('wavelength'))
                        wavelengths = w_attr[:] if w_attr is not None else np.arange(1, patch.shape[0]+1, dtype=float)
                        
                        if grid == "TANAGER":
                            gw_mask = sr_ds.attrs.get("all_good_wavelengths")[frame_idx].astype(bool)
                            patch = patch[gw_mask, :, :]
                            wavelengths = wavelengths[gw_mask]
                            
                        if np.max(wavelengths) < 10: wavelengths *= 1000
                        
                        patch = np.transpose(patch, (1, 2, 0))
                        em, _ = sc.maximumDistance(patch, N_ENDMEMBERS, strict_nan=False)
                        if not np.isnan(em).all():
                            season_spectra.append((wavelengths, em))
                            
                    if not season_spectra: continue
                    
                    lines_by_len = {}
                    for wavelengths, em in season_spectra:
                        l = len(wavelengths)
                        if l not in lines_by_len:
                            lines_by_len[l] = []
                        for j in range(em.shape[1]):
                            if not np.isnan(em[:, j]).all():
                                ax.plot(wavelengths, em[:, j], color=seg_color, alpha=0.15)
                                lines_by_len[l].append((wavelengths, em[:, j]))
                                
                    for l, items in lines_by_len.items():
                        wl_arr = items[0][0]
                        arr = np.array([itm[1] for itm in items])
                        centroid = np.nanmean(arr, axis=0)
                        ax.plot(wl_arr, centroid, color=seg_color, linewidth=3.0, label=f"Seg {seg_idx+1} Centroid")
                        max_val = np.nanmax(arr)
                        if not np.isnan(max_val):
                            global_y_max = max(global_y_max, max_val * 1.1)
                        if len(wl_arr) > 0:
                            global_wl_min = min(global_wl_min, np.nanmin(wl_arr))
                            global_wl_max = max(global_wl_max, np.nanmax(wl_arr))
                            
                handles, labels = ax.get_legend_handles_labels()
                if handles:
                    ax.legend(loc='upper left')
            
    for ax in axes.flat:
        ax.set_ylim(0, global_y_max if global_y_max > 0 else 1.0)
        if global_wl_min < float('inf') and global_wl_max > float('-inf'):
            ax.set_xlim(global_wl_min, global_wl_max)
        
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

def save_lcmap_plot(pixel_y, pixel_x, source_h5_path, inference_results_h5, out_path):
    print("Generating LCMAP Classification Plot...")
    import os
    import sys
    
    # Add classification module to path
    class_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'classification'))
    if class_dir not in sys.path:
        sys.path.append(class_dir)
        
    try:
        from lcmap_classification import extract_lcmap_time_series
    except ImportError:
        print('Failed to import lcmap_classification. Ensure it is in the classification/ directory.')
        return
        
    lcmap_classes = ['Developed', 'Cropland', 'Grass/Shrub', 'Tree Cover', 'Water', 'Wetland', 'Ice/Snow', 'Barren']
    colors_map = {
        'Developed': 'gray', 'Cropland': 'yellowgreen', 'Grass/Shrub': 'wheat', 
        'Tree Cover': 'forestgreen', 'Water': 'blue', 'Wetland': 'teal', 
        'Ice/Snow': 'cyan', 'Barren': 'sandybrown'
    }

    segments, sensors, extracted_data = extract_lcmap_time_series(
        pixel_y, pixel_x, source_h5_path, inference_results_h5, lcmap_classes=lcmap_classes
    )
    
    if not segments or not sensors:
        print('No valid data found for LCMAP classification.')
        return
        
    fig, axes = plt.subplots(len(sensors), len(segments), figsize=(7*len(segments), 4*len(sensors)), squeeze=False)
    fig.suptitle(f'LCMAP Level 1 Class Percentages (Hard SAM over Full Library)\nPixel: x={pixel_x}, y={pixel_y}', fontsize=14)
    
    for row_idx, sensor in enumerate(sensors):
        for col_idx, _ in enumerate(segments):
            ax = axes[row_idx, col_idx]
            
            segment_colors = ['blue', 'green', 'purple', 'orange', 'cyan', 'magenta', 'brown']
            c_bg = segment_colors[col_idx % len(segment_colors)]
            import matplotlib.colors as mcolors
            
            data = extracted_data.get(row_idx, {}).get(col_idx, None)
            
            if not data or not data['dates']:
                if row_idx == 0 and data:
                    start_date = datetime.fromtimestamp(data['start_acq_time'], timezone.utc).strftime('%Y-%m-%d')
                    end_date = datetime.fromtimestamp(data['end_acq_time'], timezone.utc).strftime('%Y-%m-%d')
                    ax.set_title(f'Seg {col_idx+1}\n{start_date} to {end_date}', bbox=dict(facecolor=mcolors.to_rgba(c_bg, alpha=0.3), edgecolor='none'))
                if col_idx == 0:
                    ax.set_ylabel(f'{sensor}\nPercentage')
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
                continue
                
            dates = data['dates']
            class_counts = data['class_counts']
            
            if row_idx == 0:
                start_date = datetime.fromtimestamp(data['start_acq_time'], timezone.utc).strftime('%Y-%m-%d')
                end_date = datetime.fromtimestamp(data['end_acq_time'], timezone.utc).strftime('%Y-%m-%d')
                ax.set_title(f'Seg {col_idx+1}\n{start_date} to {end_date}', bbox=dict(facecolor=mcolors.to_rgba(c_bg, alpha=0.3), edgecolor='none'))
            if col_idx == 0:
                ax.set_ylabel(f'{sensor}\nPercentage')
                
            y = np.vstack([class_counts[c] for c in lcmap_classes])
            x = dates
            
            colors = [colors_map[c] for c in lcmap_classes]
            ax.stackplot(x, y, labels=lcmap_classes, colors=colors, alpha=0.8)
            
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            if len(x) > 0:
                start_year = x[0].year
                end_year = x[-1].year
                for year in range(start_year, end_year + 2):
                    dt = datetime(year, 1, 1, tzinfo=timezone.utc)
                    if x[0] <= dt <= x[-1]:
                        ax.axvline(dt, color='black', linestyle='--', alpha=0.5, linewidth=1)
            
            ax.set_ylim(0, 1)
            
            if col_idx == len(segments)-1 and row_idx == 0:
                from matplotlib.patches import Patch
                legend_elements = [Patch(facecolor=colors_map[c], label=c) for c in lcmap_classes]
                ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.subplots_adjust(top=0.90, right=0.85)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()


def save_all_seasons_separability_plot(pixel_y, pixel_x, source_h5_path, inference_results_h5, out_path):
    print("Generating All-Seasons Separability Plot...")
    with h5py.File(inference_results_h5, 'r') as f_inf:
        rmse = f_inf['rmse_series'][:, pixel_y, pixel_x]
        
    with h5py.File(source_h5_path, 'r') as f_sc:
        harm_grp = f_sc['/HDFEOS/GRIDS/HARMONIZED/Data Fields']
        target_metric = f_sc.attrs.get('TARGET_METRIC', 'sliding_volume_z_score')
        if target_metric not in harm_grp:
            target_metric = list(harm_grp.keys())[0]
            
        attrs = harm_grp[target_metric].attrs
        acq_time = attrs['acquisition_time'][:]
        source_grids = [s.decode('utf-8') if isinstance(s, bytes) else str(s) for s in attrs['source_grid'][:]]
        source_frames = attrs['source_frame_index'][:]
        spacecrafts = [s.decode('utf-8') if isinstance(s, bytes) else str(s) for s in attrs['source_spacecraft'][:]]
        unified_masks = harm_grp['common_mask'][:, pixel_y, pixel_x]
        
    raw_h5_path = re.sub(r'_SC_.*?\.h5$', '.h5', source_h5_path)
    
    segments = []
    current_seg = []
    current_rmse = None
    for i in range(len(acq_time)):
        if unified_masks[i]: continue
        r = rmse[i]
        if np.isnan(r): continue
        if current_rmse is None:
            current_rmse = r
            current_seg.append(i)
        elif abs(r - current_rmse) < 1e-6:
            current_seg.append(i)
        else:
            segments.append(current_seg)
            current_seg = [i]
            current_rmse = r
    if current_seg:
        segments.append(current_seg)
        
    target_sensors = ['Landsat', 'Sentinel-2', 'Tanager']
    n_sensors = len(target_sensors)
    segment_colors = ['blue', 'green', 'purple', 'orange', 'cyan', 'magenta', 'brown']
    
    fig, axes = plt.subplots(n_sensors, 1, figsize=(10, 5 * n_sensors), squeeze=False)
    fig.suptitle(f"Aggregated All-Seasons Separability Across Segments\nPixel: x={pixel_x}, y={pixel_y}", fontsize=16)
    
    global_y_max = 0
    global_wl_min = float('inf')
    global_wl_max = float('-inf')
    
    def get_sensor_cat(s_str):
        s_str = str(s_str).lower()
        if 'landsat' in s_str: return 'Landsat'
        if 'sentinel' in s_str: return 'Sentinel-2'
        if 'tanager' in s_str: return 'Tanager'
        return None
    
    with h5py.File(raw_h5_path, 'r') as f_raw:
        for row_idx, sensor_cat in enumerate(target_sensors):
            ax = axes[row_idx, 0]
            ax.set_title(f"{sensor_cat} - All Seasons")
            ax.set_xlabel("Wavelength (nm)")
            ax.set_ylabel("Surface Reflectance")
            
            for seg_idx, seg in enumerate(segments):
                seg_color = segment_colors[seg_idx % len(segment_colors)]
                
                season_spectra = []
                
                for i in seg:
                    if get_sensor_cat(spacecrafts[i]) != sensor_cat:
                        continue
                        
                    grid = source_grids[i]
                    frame_idx = source_frames[i]
                    
                    sr_ds = f_raw[f"/HDFEOS/GRIDS/{grid}/Data Fields/surface_reflectance"]
                    patch = sr_ds[frame_idx, :, max(0, pixel_y-1):pixel_y+2, max(0, pixel_x-1):pixel_x+2]
                    
                    w_attr = sr_ds.attrs.get('wavelengths', sr_ds.attrs.get('wavelength'))
                    wavelengths = w_attr[:] if w_attr is not None else np.arange(1, patch.shape[0]+1, dtype=float)
                    
                    if grid == "TANAGER":
                        gw_mask = sr_ds.attrs.get("all_good_wavelengths")[frame_idx].astype(bool)
                        patch = patch[gw_mask, :, :]
                        wavelengths = wavelengths[gw_mask]
                        
                    if np.max(wavelengths) < 10: wavelengths *= 1000
                    
                    patch = np.transpose(patch, (1, 2, 0))
                    em, _ = sc.maximumDistance(patch, N_ENDMEMBERS, strict_nan=False)
                    if not np.isnan(em).all():
                        season_spectra.append((wavelengths, em))
                        
                if not season_spectra: continue
                
                lines_by_len = {}
                for wavelengths, em in season_spectra:
                    l = len(wavelengths)
                    if l not in lines_by_len:
                        lines_by_len[l] = []
                    for j in range(em.shape[1]):
                        if not np.isnan(em[:, j]).all():
                            ax.plot(wavelengths, em[:, j], color=seg_color, alpha=0.15)
                            lines_by_len[l].append((wavelengths, em[:, j]))
                            
                for l, items in lines_by_len.items():
                    wl_arr = items[0][0]
                    arr = np.array([itm[1] for itm in items])
                    centroid = np.nanmean(arr, axis=0)
                    ax.plot(wl_arr, centroid, color=seg_color, linewidth=3.0, label=f"Seg {seg_idx+1} Centroid")
                    max_val = np.nanmax(arr)
                    if not np.isnan(max_val):
                        global_y_max = max(global_y_max, max_val * 1.1)
                    if len(wl_arr) > 0:
                        global_wl_min = min(global_wl_min, np.nanmin(wl_arr))
                        global_wl_max = max(global_wl_max, np.nanmax(wl_arr))
                        
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(loc='upper left')
                
    for ax in axes.flat:
        ax.set_ylim(0, global_y_max if global_y_max > 0 else 1.0)
        if global_wl_min < float('inf') and global_wl_max > float('-inf'):
            ax.set_xlim(global_wl_min, global_wl_max)
        
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    from harmonized_CCD_main import LOCATION, H5_PATH, ENABLE_CONSTANT, ENABLE_LINEAR, ENABLE_QUADRATIC, TEMPORAL_PERIODS, TARGET_METRIC
    
    
    
    # Auto-resolve configurations
    term_str = f"C{int(ENABLE_CONSTANT)}L{int(ENABLE_LINEAR)}Q{int(ENABLE_QUADRATIC)}"
    period_str = f"P{len(TEMPORAL_PERIODS)}"
    config = f"{term_str}_{period_str}"
    
    inf_h5 = get_inference_h5(loc, config, TARGET_METRIC)
    
    if not inf_h5:
        print("Could not find inference H5 for Tait.")
        sys.exit(1)
        
    # Construct base H5 path (assuming standard naming convention)
    # The source H5 is usually {loc}_Harmonized.h5 or similar.
    # Looking at harmonized_CCD_vis.py, it uses H5_PATH from harmonized_CCD_main, but we need Tait specifically.
    source_h5 = H5_PATH
        
    # Output directory
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'presentation_plots')
    os.makedirs(out_dir, exist_ok=True)
    
    p1 = os.path.join(out_dir, f'ortho_overlay_y{px_y}_x{px_x}.png')
    p2 = os.path.join(out_dir, f'time_series_y{px_y}_x{px_x}.png')
    p3 = os.path.join(out_dir, f'separability_y{px_y}_x{px_x}.png')
    p4 = os.path.join(out_dir, f'lcmap_classes_y{px_y}_x{px_x}.png')
    p5 = os.path.join(out_dir, f'all_seasons_separability_y{px_y}_x{px_x}.png')
    
    save_ortho_visual(px_y, px_x, source_h5, inf_h5, p1)
    save_time_series(px_y, px_x, source_h5, inf_h5, p2)
    save_separability_plot(px_y, px_x, source_h5, inf_h5, p3)
    save_lcmap_plot(px_y, px_x, source_h5, inf_h5, p4)
    save_all_seasons_separability_plot(px_y, px_x, source_h5, inf_h5, p5)
    
    print(f"Presentation plots successfully saved to: {out_dir}")
