import h5py
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timezone
import os
import pyproj
import matplotlib.patches as patches
import scienceplots
import matplotlib.dates as mdates
import glob
import sys
from pathlib import Path

script_dir = Path(__file__).resolve().parent
repo_dir = str(script_dir.parent.parent)
if repo_dir not in sys.path:
    sys.path.insert(0, repo_dir)
import SpecComplex as sc

plt.style.use(['science','no-latex'])

# --- CONFIGURATION ---
N_ENDMEMBERS = 4  # Number of endmembers extracted via MaxD per patch
PLOT_INVALID_SAMPLES = False # Whether to plot invalid samples (gray markers) in the time series
# ---------------------

from harmonized_CCD_main import LOCATION, H5_PATH, ENABLE_CONSTANT, ENABLE_LINEAR, ENABLE_QUADRATIC, TEMPORAL_PERIODS, TARGET_METRIC, TARGET_NAME, get_source_h5_path
_term_str = f"C{int(ENABLE_CONSTANT)}L{int(ENABLE_LINEAR)}Q{int(ENABLE_QUADRATIC)}"
_period_str = f"P{len(TEMPORAL_PERIODS)}"
CONFIG = f"{_term_str}_{_period_str}"

def get_inference_h5(location, config, target_metric):
    candidates = [
        #f"C:/satelliteImagery/HLST30/CCD/{location}_CCD_Harmonized_Change_Detection_{target_metric}_{config}.h5",
        f"C:/satelliteImagery/MGRS30mConstellation/CCD/{location}_CCD_Harmonized_Change_Detection_{target_metric}_{config}.h5",
    ]
    for pattern in candidates:
        files = glob.glob(pattern)
        if files:
            files.sort(key=os.path.getmtime, reverse=True)
            return files[0]
    return None

def plot_pixel_sits(pixel_y, pixel_x, source_h5_path, inference_results_h5, ax=None, current_date=None):
    lat, lon = None, None
    with h5py.File(inference_results_h5, 'r') as f:
        target_metric = f.attrs.get('TARGET_METRIC', TARGET_METRIC)

    with h5py.File(source_h5_path, 'r') as f:
        harm_grp = f['/HDFEOS/GRIDS/HARMONIZED/Data Fields']
        acq_time = harm_grp[target_metric].attrs['acquisition_time'][:]
        z_score = sc.read_scaled_int16(harm_grp[target_metric], np.s_[:, pixel_y, pixel_x])
        
        unified_masks = harm_grp['common_mask'][:, pixel_y, pixel_x]
        is_invalid = unified_masks.astype(bool)
        
        spacecraft_bytes = harm_grp[target_metric].attrs['source_spacecraft'][:]
        spacecrafts = [s.decode('utf-8') if isinstance(s, bytes) else str(s) for s in spacecraft_bytes]
        
        geo_transform = harm_grp[target_metric].attrs.get('GeoTransform')
        spatial_ref = harm_grp[target_metric].attrs.get('spatial_ref')
        if geo_transform is not None and spatial_ref is not None:
            try:
                gt = geo_transform
                x_geo = gt[0] + (pixel_x + 0.5) * gt[1] + (pixel_y + 0.5) * gt[2]
                y_geo = gt[3] + (pixel_x + 0.5) * gt[4] + (pixel_y + 0.5) * gt[5]
                
                if isinstance(spatial_ref, bytes):
                    spatial_ref_str = spatial_ref.decode('utf-8')
                else:
                    spatial_ref_str = str(spatial_ref)
                
                crs = pyproj.CRS.from_user_input(spatial_ref_str)
                transformer = pyproj.Transformer.from_crs(crs, "epsg:4326", always_xy=True)
                lon, lat = transformer.transform(x_geo, y_geo)
            except Exception as e:
                print(f"Warning: Could not compute lat/lon: {e}")
        
    dates = [datetime.fromtimestamp(ts, timezone.utc) for ts in acq_time]
    
    with h5py.File(inference_results_h5, 'r') as f:
        predicted = f['predicted_series'][:, pixel_y, pixel_x]
        rmse = f['rmse_series'][:, pixel_y, pixel_x]
        anomalies = f['anomaly_flags'][:, pixel_y, pixel_x]
        if 'CHANGE_PROBABILITY' in f.attrs:
            from scipy.stats import chi2
            p = f.attrs['CHANGE_PROBABILITY']
            
            # Read degrees of freedom from attributes, or default to 1 for backwards compatibility with older runs
            df = f.attrs.get('CHI2_DEGREES_OF_FREEDOM', 1)
            
            # For older runs where it wasn't saved, try to import from main script if available
            if 'CHI2_DEGREES_OF_FREEDOM' not in f.attrs:
                import sys, os
                try:
                    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
                    import harmonized_CCD_main
                    df = harmonized_CCD_main.CHI2_DEGREES_OF_FREEDOM
                except Exception:
                    pass

            rmse_multiplier = np.sqrt(chi2.ppf(p, df=df))
            bound_label = f"Prediction ±{rmse_multiplier:.2f}σ (p={p}, df={df})"
        else:
            rmse_multiplier = f.attrs.get('RMSE_MULTIPLIER', 3.0)
            bound_label = f"Prediction ±{rmse_multiplier}σ"
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
        show_plot = True
    else:
        show_plot = False
        
    valid_mask = ~is_invalid
    dates_arr = np.array(dates)
    spacecrafts_arr = np.array(spacecrafts)
    
    # Plot Actuals
    is_anomaly = (anomalies == 1)
    for marker_type, sc_keyword in [('s', 'Sentinel'), ('o', 'Landsat'), ('D', 'Tanager'), ('*', 'EnMAP'), ('^', 'Dragonette')]:
        sc_mask = np.array([sc_keyword.lower() in str(sc).lower() for sc in spacecrafts_arr])
        
        idx_valid_normal = valid_mask & sc_mask & ~is_anomaly
        if np.any(idx_valid_normal):
            ax.plot(dates_arr[idx_valid_normal], z_score[idx_valid_normal], color='k', marker=marker_type, linestyle='None', label=f'Valid ({sc_keyword})')
            
        idx_valid_anomaly = valid_mask & sc_mask & is_anomaly
        if np.any(idx_valid_anomaly):
            ax.plot(dates_arr[idx_valid_anomaly], z_score[idx_valid_anomaly], color='r', marker=marker_type, linestyle='None', label=f'Anomaly ({sc_keyword})')
            
        if PLOT_INVALID_SAMPLES:
            idx_invalid = is_invalid & sc_mask
            if np.any(idx_invalid):
                ax.plot(dates_arr[idx_invalid], z_score[idx_invalid], color='gray', marker=marker_type, linestyle='None', markerfacecolor='none', label=f'Invalid ({sc_keyword})')
            
    # Plot Predictions
    pred_mask = ~np.isnan(predicted)
    if np.any(pred_mask):
        pred_dates = dates_arr[pred_mask]
        preds = predicted[pred_mask]
        rmses = rmse[pred_mask]
        
        # Detect segments based on changes in the frozen RMSE
        rmse_diff = np.diff(rmses)
        # Add 1 because diff shifts indices by 1
        break_indices = np.where(rmse_diff != 0)[0] + 1
        
        date_segments = np.split(pred_dates, break_indices)
        pred_segments = np.split(preds, break_indices)
        rmse_segments = np.split(rmses, break_indices)
        
        segment_colors = ['blue', 'green', 'purple', 'orange', 'cyan', 'magenta', 'brown']
        
        for i, (seg_dates, seg_preds, seg_rmses) in enumerate(zip(date_segments, pred_segments, rmse_segments)):
            if len(seg_dates) == 0:
                continue
                
            c = segment_colors[i % len(segment_colors)]
            upper_bound = seg_preds + rmse_multiplier * seg_rmses
            lower_bound = seg_preds - rmse_multiplier * seg_rmses
            
            label = 'Harmonic Prediction' if i == 0 else f'Prediction (Seg {i+1})'
            ax.plot(seg_dates, seg_preds, color=c, linestyle='--', label=label)
            
            fill_label = bound_label if i == 0 else None
            ax.fill_between(seg_dates, lower_bound, upper_bound, color=c, alpha=0.15, label=fill_label)
        
        # Anomalies are now plotted with their respective sensor markers in red above.
            
    if current_date is not None:
        ax.axvline(x=current_date, color='orange', linestyle='--', label='Displayed Frame')

    if lat is not None and lon is not None:
        ax.set_title(f"{target_metric} | Pixel Location: ({pixel_x}, {pixel_y}) | Lat: {lat:.5f}, Lon: {lon:.5f}")
    else:
        ax.set_title(f"{target_metric} | Pixel Location: ({pixel_x}, {pixel_y})")
    ax.set_xlabel('Date')
    import matplotlib.dates as mdates
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.set_ylabel(TARGET_NAME)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    ax.grid(True)
    if show_plot:
        plt.show()


import re
from matplotlib.animation import FuncAnimation

def animate_pixel_endmembers(pixel_y, pixel_x, source_h5_path, inference_results_h5):
    import sys, os, re
    import numpy as np
    import h5py
    from datetime import datetime, timezone
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    import matplotlib.colors as mcolors
    
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if root_dir not in sys.path:
        sys.path.append(root_dir)
    import SpecComplex as sc
    
    raw_h5_path = re.sub(r'_SC_.*?\.h5$', '.h5', source_h5_path)
    if not os.path.exists(raw_h5_path):
        print(f"Error: Raw H5 file not found at {raw_h5_path}")
        return
        
    def get_sensor_group(sc_str):
        s = str(sc_str).upper()
        if 'LANDSAT' in s: return 'Landsat'
        if 'SENTINEL' in s: return 'Sentinel'
        if 'TANAGER' in s: return 'Tanager'
        if 'ENMAP' in s: return 'EnMAP'
        if 'DRAGONETTE' in s: return 'Dragonette'
        return s

    endmembers_over_time = []
    dates_over_time = []
    sensors_over_time = []
    wavelengths_over_time = []
    segment_indices_over_time = []
    
    with h5py.File(inference_results_h5, 'r') as f_inf:
        rmse = f_inf['rmse_series'][:, pixel_y, pixel_x]
    
    with h5py.File(source_h5_path, 'r') as f_sc, h5py.File(raw_h5_path, 'r') as f_raw:
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
        
        _, H, W = harm_grp['common_mask'].shape
        y_start = max(0, pixel_y - 1)
        y_end = min(H, pixel_y + 2)
        x_start = max(0, pixel_x - 1)
        x_end = min(W, pixel_x + 2)
        
        # Segment logic
        segments = []
        current_segment = []
        current_rmse = None
        for i in range(len(acq_time)):
            if unified_masks[i]: continue
            r = rmse[i]
            
            if len(current_segment) == 0:
                current_rmse = r
                current_segment.append(i)
            elif (np.isnan(r) and np.isnan(current_rmse)):
                current_segment.append(i)
            elif (not np.isnan(r) and not np.isnan(current_rmse) and abs(r - current_rmse) < 1e-6):
                current_segment.append(i)
            else:
                segments.append(current_segment)
                current_segment = [i]
                current_rmse = r
        if current_segment:
            segments.append(current_segment)
            
        frame_to_segment = {}
        for seg_idx, seg_list in enumerate(segments):
            for idx in seg_list:
                frame_to_segment[idx] = seg_idx
        
        for i in range(len(acq_time)):
            if unified_masks[i]: 
                continue
                
            grid = source_grids[i]
            frame_idx = source_frames[i]
            
            sr_ds = f_raw[f"/HDFEOS/GRIDS/{grid}/Data Fields/surface_reflectance"]
            patch = sr_ds[frame_idx, :, y_start:y_end, x_start:x_end]
            
            w_attr = sr_ds.attrs.get('wavelengths')
            if w_attr is None: w_attr = sr_ds.attrs.get('wavelength')
            if w_attr is not None:
                wavelengths = w_attr[:]
            else:
                wavelengths = np.arange(1, patch.shape[0] + 1, dtype=float)
                
            if grid == "TANAGER":
                gw_mask = sr_ds.attrs.get("all_good_wavelengths")[frame_idx].astype(bool)
                patch = patch[gw_mask, :, :]
                wavelengths = wavelengths[gw_mask]
                
            if np.max(wavelengths) < 10:
                wavelengths = wavelengths * 1000
                
            patch = np.transpose(patch, (1, 2, 0))
            
            em, _ = sc.maximumDistance(patch, 7, strict_nan=False)
            
            if not np.isnan(em).all():
                endmembers_over_time.append(em)
                dates_over_time.append(datetime.fromtimestamp(acq_time[i], timezone.utc))
                sensors_over_time.append(get_sensor_group(spacecrafts[i]))
                wavelengths_over_time.append(wavelengths)
                segment_indices_over_time.append(frame_to_segment.get(i, 0))
                
    if not endmembers_over_time:
        print("No valid data to animate for this pixel neighborhood.")
        return
        
    unique_sensors = sorted(list(set(sensors_over_time)))
    fig, axes = plt.subplots(len(unique_sensors), 1, figsize=(10, 3*len(unique_sensors)), squeeze=False)
    
    lines_by_sensor = {}
    titles_by_sensor = {}
    colors = plt.cm.turbo(np.linspace(0, 1, 7))
    
    global_wl_min = np.min([np.nanmin(wl) for wl in wavelengths_over_time if len(wl) > 0]) * 0.95
    global_wl_max = np.max([np.nanmax(wl) for wl in wavelengths_over_time if len(wl) > 0]) * 1.05
    
    for row_idx, sensor in enumerate(unique_sensors):
        ax = axes[row_idx, 0]
        sensor_lines = []
        for j in range(7):
            line, = ax.plot([], [], marker='o', markersize=4, color=colors[j], label=f'EM {j+1}')
            sensor_lines.append(line)
        ax.set_ylabel(f"{sensor}\nReflectance")
        ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1.0))
        lines_by_sensor[sensor] = sensor_lines
        titles_by_sensor[sensor] = ax.set_title(f"{sensor} - Waiting for data...")
        
    axes[-1, 0].set_xlabel("Wavelength (nm)")
    fig.subplots_adjust(right=0.75, hspace=0.4)
    
    frame_text = fig.text(0.02, 0.98, '', fontsize=12, verticalalignment='top')
    
    def init():
        for sensor_lines in lines_by_sensor.values():
            for line in sensor_lines:
                line.set_data([], [])
        for t in titles_by_sensor.values():
            t.set_text("")
        frame_text.set_text("")
        all_artists = []
        for lines in lines_by_sensor.values():
            all_artists.extend(lines)
        all_artists.extend(titles_by_sensor.values())
        all_artists.append(frame_text)
        return all_artists
        
    segment_colors = ['blue', 'green', 'purple', 'orange', 'cyan', 'magenta', 'brown']
    
    from matplotlib.colors import LinearSegmentedColormap
    season_cmap = LinearSegmentedColormap.from_list('seasons', ['lightblue', 'lightgreen', 'lightyellow', 'navajowhite', 'lightblue'])
    
    # Add colorbar for season underlay
    cax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap=season_cmap, norm=plt.Normalize(vmin=1, vmax=365))
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_ticks([1, 91, 182, 274, 365])
    cbar.set_ticklabels(['Jan 1', 'Apr 1', 'Jul 1', 'Oct 1', 'Dec 31'])
    cbar.set_label('Season of Image Acquisition (Day of Year)')
    
    def update(frame):
        em = endmembers_over_time[frame]
        dt = dates_over_time[frame]
        sensor = sensors_over_time[frame]
        wl = wavelengths_over_time[frame]
        seg_idx = segment_indices_over_time[frame]
        
        sensor_lines = lines_by_sensor[sensor]
        for j in range(7):
            if j < em.shape[1]:
                sensor_lines[j].set_data(wl, em[:, j])
            else:
                sensor_lines[j].set_data([], [])
                
        c_bg = segment_colors[seg_idx % len(segment_colors)]
        title = titles_by_sensor[sensor]
        title.set_text(f"{sensor} 3x3 Endmembers | Seg {seg_idx+1} | {dt.strftime('%Y-%m-%d')}")
        title.set_bbox(dict(facecolor=mcolors.to_rgba(c_bg, alpha=0.3), edgecolor='none'))
        
        ax = title.axes
        doy = dt.timetuple().tm_yday
        fraction = doy / 366.0
        ax.set_facecolor(season_cmap(fraction))
        
        frame_text.set_text(f"Frame: {frame+1}/{len(endmembers_over_time)}")
        
        ax.set_xlim(global_wl_min, global_wl_max)
        max_r = np.nanmax(em)
        if np.isnan(max_r) or max_r < 0.1: max_r = 1.0
        ax.set_ylim(0, max_r * 1.1)
        
        all_artists = []
        for lines in lines_by_sensor.values():
            all_artists.extend(lines)
        all_artists.extend(titles_by_sensor.values())
        all_artists.append(frame_text)
        return all_artists
        
    ani = FuncAnimation(fig, update, frames=len(endmembers_over_time), init_func=init, blit=False, interval=500, repeat=True)
    fig._endmember_animation = ani
    
    fig.subplots_adjust(bottom=0.15)
    from matplotlib.widgets import Button
    ax_save_btn = fig.add_axes([0.8, 0.02, 0.15, 0.06])
    btn_save = Button(ax_save_btn, 'Export Animation')
    
    def save_animation(event):
        print(f"FFMpeg not available, falling back to GIF export...")
        out_name = f"Endmembers_x{pixel_x}_y{pixel_y}.gif"
        print(f"Exporting animation (GIF) to {out_name}...")
        ani.save(out_name, writer='pillow', fps=5)
        print("Export complete.")
            
    btn_save.on_clicked(save_animation)
    fig._btn_save = btn_save
    
    plt.show(block=False)


def plot_segment_spectra(pixel_y, pixel_x, source_h5_path, inference_results_h5):
    import sys, os, re
    import numpy as np
    import h5py
    from datetime import datetime, timezone
    import matplotlib.pyplot as plt
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if root_dir not in sys.path:
        sys.path.append(root_dir)
    import SpecComplex as sc
    
    raw_h5_path = re.sub(r'_SC_.*?\.h5$', '.h5', source_h5_path)
    if not os.path.exists(raw_h5_path):
        print(f"Error: Raw H5 file not found at {raw_h5_path}")
        return

    def get_sensor_group(sc_str):
        s = str(sc_str).upper()
        if 'LANDSAT' in s: return 'Landsat'
        if 'SENTINEL' in s: return 'Sentinel'
        if 'TANAGER' in s: return 'Tanager'
        if 'ENMAP' in s: return 'EnMAP'
        if 'DRAGONETTE' in s: return 'Dragonette'
        return s
        
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
        
        _, H, W = harm_grp['common_mask'].shape

    y_start = max(0, pixel_y - 1)
    y_end = min(H, pixel_y + 2)
    x_start = max(0, pixel_x - 1)
    x_end = min(W, pixel_x + 2)
    
    # Identify Segments based on stable RMSE predictions
    segments = []
    current_segment = []
    current_rmse = None
    for i in range(len(acq_time)):
        if unified_masks[i]: continue
        r = rmse[i]
        
        if len(current_segment) == 0:
            current_rmse = r
            current_segment.append(i)
        elif (np.isnan(r) and np.isnan(current_rmse)):
            current_segment.append(i)
        elif (not np.isnan(r) and not np.isnan(current_rmse) and abs(r - current_rmse) < 1e-6):
            current_segment.append(i)
        else:
            segments.append(current_segment)
            current_segment = [i]
            current_rmse = r
    if current_segment:
        segments.append(current_segment)
        
    if not segments:
        print("No valid structural segments found for this pixel.")
        return

    # Discover present sensors
    present_sensors = set()
    for seg in segments:
        for i in seg:
            present_sensors.add(get_sensor_group(spacecrafts[i]))
    
    sensors = sorted(list(present_sensors))
    
    fig, axes = plt.subplots(len(sensors), len(segments), figsize=(6*len(segments), 4*len(sensors)), squeeze=False)
    fig.suptitle(f"Segment Spectra Overview (All Frames)\nPixel: x={pixel_x}, y={pixel_y}", fontsize=14)
    
    with h5py.File(raw_h5_path, 'r') as f_raw:
        for col_idx, seg in enumerate(segments):
            seg_data = {s: [] for s in sensors}
            seg_wl = {s: None for s in sensors}
            
            for i in seg:
                grid = source_grids[i]
                frame_idx = source_frames[i]
                sensor = get_sensor_group(spacecrafts[i])
                
                sr_ds = f_raw[f"/HDFEOS/GRIDS/{grid}/Data Fields/surface_reflectance"]
                patch = sr_ds[frame_idx, :, y_start:y_end, x_start:x_end]
                
                w_attr = sr_ds.attrs.get('wavelengths')
                if w_attr is None: w_attr = sr_ds.attrs.get('wavelength')
                if w_attr is not None:
                    wavelengths = w_attr[:]
                else:
                    wavelengths = np.arange(1, patch.shape[0] + 1, dtype=float)
                    
                if grid == "TANAGER":
                    gw_mask = sr_ds.attrs.get("all_good_wavelengths")[frame_idx].astype(bool)
                    patch = patch[gw_mask, :, :]
                    wavelengths = wavelengths[gw_mask]
                    
                if np.max(wavelengths) < 10:
                    wavelengths = wavelengths * 1000
                    
                if seg_wl[sensor] is None:
                    seg_wl[sensor] = wavelengths
                    
                patch = np.transpose(patch, (1, 2, 0))
                em, _ = sc.maximumDistance(patch, N_ENDMEMBERS, strict_nan=False)
                if not np.isnan(em).all():
                    seg_data[sensor].append(em)
                    
            start_date = datetime.fromtimestamp(acq_time[seg[0]], timezone.utc).strftime('%Y-%m-%d')
            end_date = datetime.fromtimestamp(acq_time[seg[-1]], timezone.utc).strftime('%Y-%m-%d')
            
            for row_idx, sensor in enumerate(sensors):
                ax = axes[row_idx, col_idx]
                
                segment_colors = ['blue', 'green', 'purple', 'orange', 'cyan', 'magenta', 'brown']
                c_bg = segment_colors[col_idx % len(segment_colors)]
                import matplotlib.colors as mcolors
                
                data = seg_data[sensor]
                
                if row_idx == 0:
                    ax.set_title(f"Seg {col_idx+1}\n{start_date} to {end_date}", bbox=dict(facecolor=mcolors.to_rgba(c_bg, alpha=0.3), edgecolor='none'))
                if col_idx == 0:
                    ax.set_ylabel(f"{sensor}\nReflectance")
                
                if not data:
                    ax.text(0.5, 0.5, "No Data for Sensor", ha='center', va='center', transform=ax.transAxes)
                    continue
                    
                # -- Spectral Alignment using Hungarian Algorithm & SAM --
                from scipy.optimize import linear_sum_assignment
                aligned_data = []
                ref_em = data[0]  # First frame as the reference for this segment/sensor
                aligned_data.append(ref_em)
                
                for em in data[1:]:
                    # Compute norms (add epsilon to prevent div by zero)
                    norm_ref = np.linalg.norm(ref_em, axis=0) + 1e-8
                    norm_curr = np.linalg.norm(em, axis=0) + 1e-8
                    
                    # Cosine similarity
                    cos_sim = np.dot(ref_em.T, em) / np.outer(norm_ref, norm_curr)
                    cos_sim = np.clip(cos_sim, -1.0, 1.0)
                    
                    # Cost is the Spectral Angle (we want to minimize it)
                    cost_matrix = np.arccos(cos_sim)
                    
                    # Hungarian assignment
                    # row_ind corresponds to reference indices
                    # col_ind corresponds to current frame indices
                    row_ind, col_ind = linear_sum_assignment(cost_matrix)
                    
                    aligned_em = np.zeros_like(em)
                    # Align current frame's endmembers to match the reference slots
                    aligned_em[:, row_ind] = em[:, col_ind]
                    aligned_data.append(aligned_em)
                    
                cube = np.stack(aligned_data, axis=0) # (time, bands, 7)
                wl = seg_wl[sensor]
                colors = plt.cm.turbo(np.linspace(0, 1, N_ENDMEMBERS))
                
                for j in range(N_ENDMEMBERS):
                    em_series = cube[:, :, j]
                    
                    if np.isnan(em_series).all():
                        continue
                        
                    label = f'EM {j+1} (Rank {j+1})' if (row_idx==0 and col_idx==len(segments)-1) else ""
                    if label:
                        ax.plot([], [], color=colors[j], alpha=1.0, label=label)
                        
                    for t in range(em_series.shape[0]):
                        if not np.isnan(em_series[t]).all():
                            ax.plot(wl, em_series[t], color=colors[j], alpha=0.15)
                
                ax.set_xlabel("Wavelength (nm)")
                max_val = np.nanmax(cube)
                ax.set_ylim(0, max_val * 1.1 if not np.isnan(max_val) else 1.0)
                
                if col_idx == len(segments)-1 and row_idx == 0:
                    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    global_y_max = 0
    for ax in axes.flat:
        global_y_max = max(global_y_max, ax.get_ylim()[1])
    for ax in axes.flat:
        ax.set_ylim(0, global_y_max)

    plt.tight_layout()
    plt.subplots_adjust(top=0.90, right=0.85)
    plt.show(block=False)



def plot_season_spectra(pixel_y, pixel_x, source_h5_path, inference_results_h5):
    import sys, os, re
    import numpy as np
    import h5py
    from datetime import datetime, timezone
    import matplotlib.pyplot as plt
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if root_dir not in sys.path:
        sys.path.append(root_dir)
    import SpecComplex as sc
    
    raw_h5_path = re.sub(r'_SC_.*?\.h5$', '.h5', source_h5_path)
    if not os.path.exists(raw_h5_path):
        print(f"Error: Raw H5 file not found at {raw_h5_path}")
        return

    def get_season(dt):
        m = dt.month
        if m in [12, 1, 2]: return 'Winter'
        if m in [3, 4, 5]: return 'Spring'
        if m in [6, 7, 8]: return 'Summer'
        return 'Fall'
        
    seasons_order = ['Winter', 'Spring', 'Summer', 'Fall']

    def get_sensor_group(sc_str):
        s = str(sc_str).upper()
        if 'LANDSAT' in s: return 'Landsat'
        if 'SENTINEL' in s: return 'Sentinel'
        if 'TANAGER' in s: return 'Tanager'
        if 'ENMAP' in s: return 'EnMAP'
        if 'DRAGONETTE' in s: return 'Dragonette'
        return s
        
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
        
        _, H, W = harm_grp['common_mask'].shape

    y_start = max(0, pixel_y - 1)
    y_end = min(H, pixel_y + 2)
    x_start = max(0, pixel_x - 1)
    x_end = min(W, pixel_x + 2)
    
    # Identify Segments based on stable RMSE predictions
    segments = []
    current_segment = []
    current_rmse = None
    for i in range(len(acq_time)):
        if unified_masks[i]: continue
        r = rmse[i]
        
        if len(current_segment) == 0:
            current_rmse = r
            current_segment.append(i)
        elif (np.isnan(r) and np.isnan(current_rmse)):
            current_segment.append(i)
        elif (not np.isnan(r) and not np.isnan(current_rmse) and abs(r - current_rmse) < 1e-6):
            current_segment.append(i)
        else:
            segments.append(current_segment)
            current_segment = [i]
            current_rmse = r
    if current_segment:
        segments.append(current_segment)
        
    if not segments:
        print("No valid structural segments found for this pixel.")
        return

    # Discover present sensors
    present_sensors = set()
    for seg in segments:
        for i in seg:
            present_sensors.add(get_sensor_group(spacecrafts[i]))
    
    sensors = sorted(list(present_sensors))
    
    fig, axes = plt.subplots(len(sensors), len(segments) * 4, figsize=(6*len(segments)*4, 4*len(sensors)), squeeze=False)
    fig.suptitle(f"Season Spectra Overview (All Frames)\nPixel: x={pixel_x}, y={pixel_y}", fontsize=14)
    
    with h5py.File(raw_h5_path, 'r') as f_raw:
        for seg_idx, seg in enumerate(segments):
            seg_data = {s: {sea: [] for sea in seasons_order} for s in sensors}
            seg_wl = {s: None for s in sensors}
            
            for i in seg:
                grid = source_grids[i]
                frame_idx = source_frames[i]
                sensor = get_sensor_group(spacecrafts[i])
                dt = datetime.fromtimestamp(acq_time[i], timezone.utc)
                season = get_season(dt)
                
                sr_ds = f_raw[f"/HDFEOS/GRIDS/{grid}/Data Fields/surface_reflectance"]
                patch = sr_ds[frame_idx, :, y_start:y_end, x_start:x_end]
                
                w_attr = sr_ds.attrs.get('wavelengths')
                if w_attr is None: w_attr = sr_ds.attrs.get('wavelength')
                if w_attr is not None:
                    wavelengths = w_attr[:]
                else:
                    wavelengths = np.arange(1, patch.shape[0] + 1, dtype=float)
                    
                if grid == "TANAGER":
                    gw_mask = sr_ds.attrs.get("all_good_wavelengths")[frame_idx].astype(bool)
                    patch = patch[gw_mask, :, :]
                    wavelengths = wavelengths[gw_mask]
                    
                if np.max(wavelengths) < 10:
                    wavelengths = wavelengths * 1000
                    
                if seg_wl[sensor] is None:
                    seg_wl[sensor] = wavelengths
                    
                patch = np.transpose(patch, (1, 2, 0))
                em, _ = sc.maximumDistance(patch, N_ENDMEMBERS, strict_nan=False)
                if not np.isnan(em).all():
                    seg_data[sensor][season].append(em)
                    
            start_date = datetime.fromtimestamp(acq_time[seg[0]], timezone.utc).strftime('%Y-%m-%d')
            end_date = datetime.fromtimestamp(acq_time[seg[-1]], timezone.utc).strftime('%Y-%m-%d')
            
            for row_idx, sensor in enumerate(sensors):
                for sea_idx, season in enumerate(seasons_order):
                    col_idx = seg_idx * 4 + sea_idx
                    ax = axes[row_idx, col_idx]
                    
                    segment_colors = ['blue', 'green', 'purple', 'orange', 'cyan', 'magenta', 'brown']
                    c_bg = segment_colors[seg_idx % len(segment_colors)]
                    import matplotlib.colors as mcolors
                    
                    data = seg_data[sensor][season]
                    
                    if row_idx == 0:
                        ax.set_title(f"Seg {seg_idx+1} {season}\n{start_date} to {end_date}", bbox=dict(facecolor=mcolors.to_rgba(c_bg, alpha=0.3), edgecolor='none'))
                    if col_idx == 0:
                        ax.set_ylabel(f"{sensor}\nReflectance")
                    
                    if not data:
                        ax.text(0.5, 0.5, "No Data", ha='center', va='center', transform=ax.transAxes)
                        continue
                        
                    # -- Spectral Alignment using Hungarian Algorithm & SAM --
                    from scipy.optimize import linear_sum_assignment
                    aligned_data = []
                    ref_em = data[0]  # First frame as the reference for this segment/sensor/season
                    aligned_data.append(ref_em)
                    
                    for em in data[1:]:
                        # Compute norms (add epsilon to prevent div by zero)
                        norm_ref = np.linalg.norm(ref_em, axis=0) + 1e-8
                        norm_curr = np.linalg.norm(em, axis=0) + 1e-8
                        
                        # Cosine similarity
                        cos_sim = np.dot(ref_em.T, em) / np.outer(norm_ref, norm_curr)
                        cos_sim = np.clip(cos_sim, -1.0, 1.0)
                        
                        # Cost is the Spectral Angle (we want to minimize it)
                        cost_matrix = np.arccos(cos_sim)
                        
                        # Hungarian assignment
                        row_ind, col_ind = linear_sum_assignment(cost_matrix)
                        
                        aligned_em = np.zeros_like(em)
                        # Align current frame's endmembers to match the reference slots
                        aligned_em[:, row_ind] = em[:, col_ind]
                        aligned_data.append(aligned_em)
                        
                    cube = np.stack(aligned_data, axis=0) # (time, bands, 7)
                    wl = seg_wl[sensor]
                    colors = plt.cm.turbo(np.linspace(0, 1, N_ENDMEMBERS))
                    
                    for j in range(N_ENDMEMBERS):
                        em_series = cube[:, :, j]
                        
                        if np.isnan(em_series).all():
                            continue
                            
                        label = f'EM {j+1} (Rank {j+1})' if (row_idx==0 and col_idx==len(segments)*4 - 1) else ""
                        if label:
                            ax.plot([], [], color=colors[j], alpha=1.0, label=label)
                            
                        for t in range(em_series.shape[0]):
                            if not np.isnan(em_series[t]).all():
                                ax.plot(wl, em_series[t], color=colors[j], alpha=0.15)
                    
                    ax.set_xlabel("Wavelength (nm)")
                    max_val = np.nanmax(cube)
                    ax.set_ylim(0, max_val * 1.1 if not np.isnan(max_val) else 1.0)
                    
                    if col_idx == len(segments)*4 - 1 and row_idx == 0:
                        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    global_y_max = 0
    for ax in axes.flat:
        global_y_max = max(global_y_max, ax.get_ylim()[1])
    for ax in axes.flat:
        ax.set_ylim(0, global_y_max)

    plt.tight_layout()
    plt.subplots_adjust(top=0.90, right=0.85)
    plt.show(block=False)

def plot_segment_endmembers(pixel_y, pixel_x, source_h5_path, inference_results_h5, strict_exclusion=True):
    import sys, os, re
    import numpy as np
    import h5py
    from datetime import datetime, timezone
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if root_dir not in sys.path:
        sys.path.append(root_dir)
    import SpecComplex as sc
    
    raw_h5_path = re.sub(r'_SC_.*?\.h5$', '.h5', source_h5_path)
    if not os.path.exists(raw_h5_path):
        print(f"Error: Raw H5 file not found at {raw_h5_path}")
        return

    def get_sensor_group(sc_str):
        s = str(sc_str).upper()
        if 'LANDSAT' in s: return 'Landsat'
        if 'SENTINEL' in s: return 'Sentinel'
        if 'TANAGER' in s: return 'Tanager'
        if 'ENMAP' in s: return 'EnMAP'
        if 'DRAGONETTE' in s: return 'Dragonette'
        return s

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
        
        _, H, W = harm_grp['common_mask'].shape

    y_start = max(0, pixel_y - 1)
    y_end = min(H, pixel_y + 2)
    x_start = max(0, pixel_x - 1)
    x_end = min(W, pixel_x + 2)
    
    # Identify Segments based on stable RMSE predictions
    segments = []
    current_segment = []
    current_rmse = None
    for i in range(len(acq_time)):
        if unified_masks[i]: continue
        r = rmse[i]
        
        if len(current_segment) == 0:
            current_rmse = r
            current_segment.append(i)
        elif (np.isnan(r) and np.isnan(current_rmse)):
            current_segment.append(i)
        elif (not np.isnan(r) and not np.isnan(current_rmse) and abs(r - current_rmse) < 1e-6):
            current_segment.append(i)
        else:
            segments.append(current_segment)
            current_segment = [i]
            current_rmse = r
    if current_segment:
        segments.append(current_segment)
        
    if not segments:
        print("No valid structural segments found for this pixel.")
        return

    # Discover present sensors
    present_sensors = set()
    for seg in segments:
        for i in seg:
            present_sensors.add(get_sensor_group(spacecrafts[i]))
    
    sensors = sorted(list(present_sensors))
    
    fig, axes = plt.subplots(len(sensors), len(segments), figsize=(6*len(segments), 4*len(sensors)), squeeze=False)
    fig.suptitle(f"Segment Endmembers (MaxD on Valid Pixels)\nPixel: x={pixel_x}, y={pixel_y}", fontsize=14)
    
    with h5py.File(raw_h5_path, 'r') as f_raw:
        for col_idx, seg in enumerate(segments):
            seg_data = {s: [] for s in sensors}
            seg_wl = {s: None for s in sensors}
            
            for i in seg:
                grid = source_grids[i]
                frame_idx = source_frames[i]
                sensor = get_sensor_group(spacecrafts[i])
                
                sr_ds = f_raw[f"/HDFEOS/GRIDS/{grid}/Data Fields/surface_reflectance"]
                patch = sr_ds[frame_idx, :, y_start:y_end, x_start:x_end]
                
                w_attr = sr_ds.attrs.get('wavelengths')
                if w_attr is None: w_attr = sr_ds.attrs.get('wavelength')
                if w_attr is not None:
                    wavelengths = w_attr[:]
                else:
                    wavelengths = np.arange(1, patch.shape[0] + 1, dtype=float)
                    
                if grid == "TANAGER":
                    gw_mask = sr_ds.attrs.get("all_good_wavelengths")[frame_idx].astype(bool)
                    patch = patch[gw_mask, :, :]
                    wavelengths = wavelengths[gw_mask]
                    
                if np.max(wavelengths) < 10:
                    wavelengths = wavelengths * 1000
                    
                if seg_wl[sensor] is None:
                    seg_wl[sensor] = wavelengths
                    
                patch = np.transpose(patch, (1, 2, 0)) # shape is (H, W, Bands) e.g., (3, 3, Bands)
                
                # Check for masked/invalid values
                is_invalid = np.isnan(patch).any() or np.all(patch == 0)
                if strict_exclusion and is_invalid:
                    continue # Skip this frame entirely
                    
                seg_data[sensor].append(patch)
                    
            start_date = datetime.fromtimestamp(acq_time[seg[0]], timezone.utc).strftime('%Y-%m-%d')
            end_date = datetime.fromtimestamp(acq_time[seg[-1]], timezone.utc).strftime('%Y-%m-%d')
            
            for row_idx, sensor in enumerate(sensors):
                ax = axes[row_idx, col_idx]
                
                segment_colors = ['blue', 'green', 'purple', 'orange', 'cyan', 'magenta', 'brown']
                c_bg = segment_colors[col_idx % len(segment_colors)]
                import matplotlib.colors as mcolors
                
                patches_list = seg_data[sensor]
                
                if row_idx == 0:
                    ax.set_title(f"Seg {col_idx+1}\n{start_date} to {end_date}", bbox=dict(facecolor=mcolors.to_rgba(c_bg, alpha=0.3), edgecolor='none'))
                if col_idx == 0:
                    ax.set_ylabel(f"{sensor}\nReflectance")
                
                if not patches_list:
                    ax.text(0.5, 0.5, "No Data for Sensor", ha='center', va='center', transform=ax.transAxes)
                    continue
                    
                # Combine patches spatially: from list of (3,3,B) to (3, 3*V, B)
                combined_patch = np.concatenate(patches_list, axis=1)
                
                em, _ = sc.maximumDistance(combined_patch, N_ENDMEMBERS, strict_nan=False)
                if np.isnan(em).all():
                    ax.text(0.5, 0.5, "No Valid Endmembers", ha='center', va='center', transform=ax.transAxes)
                    continue
                    
                wl = seg_wl[sensor]
                colors = plt.cm.turbo(np.linspace(0, 1, N_ENDMEMBERS))
                
                for j in range(N_ENDMEMBERS):
                    em_signature = em[:, j]
                    label = f'EM {j+1} (Rank {j+1})' if (row_idx==0 and col_idx==len(segments)-1) else ""
                    ax.plot(wl, em_signature, color=colors[j], label=label)
                
                ax.set_xlabel("Wavelength (nm)")
                max_val = np.nanmax(em)
                ax.set_ylim(0, max_val * 1.1 if not np.isnan(max_val) else 1.0)
                
                if col_idx == len(segments)-1 and row_idx == 0:
                    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    global_y_max = 0
    for ax in axes.flat:
        global_y_max = max(global_y_max, ax.get_ylim()[1])
    for ax in axes.flat:
        ax.set_ylim(0, global_y_max)

    plt.tight_layout()
    plt.subplots_adjust(top=0.90, right=0.85)
    plt.show(block=False)




def plot_seasonal_separability(pixel_y, pixel_x, source_h5_path, inference_results_h5):
    """
    Plots the spectral density clouds for each season, overlaid by segment.
    This provides a direct visual representation of intra-season variance vs inter-segment change.
    """
    import sys, os, re
    import numpy as np
    import h5py
    from datetime import datetime, timezone
    import matplotlib.pyplot as plt
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if root_dir not in sys.path:
        sys.path.append(root_dir)
    import SpecComplex as sc

    def get_season(dt):
        m = dt.month
        if m in [12, 1, 2]: return 'Winter'
        if m in [3, 4, 5]: return 'Spring'
        if m in [6, 7, 8]: return 'Summer'
        return 'Fall'

    def get_sensor_group(sc_str):
        s = str(sc_str).upper()
        if 'LANDSAT' in s: return 'Landsat'
        if 'SENTINEL' in s: return 'Sentinel'
        if 'TANAGER' in s: return 'Tanager'
        if 'ENMAP' in s: return 'EnMAP'
        if 'DRAGONETTE' in s: return 'Dragonette'
        return s

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
        
        _, H, W = harm_grp['common_mask'].shape

    y_start = max(0, pixel_y - 1)
    y_end = min(H, pixel_y + 2)
    x_start = max(0, pixel_x - 1)
    x_end = min(W, pixel_x + 2)
    
    segments = []
    current_segment = []
    current_rmse = None
    for i in range(len(acq_time)):
        if unified_masks[i]: continue
        r = rmse[i]
        if np.isnan(r): continue
        if current_rmse is None:
            current_rmse = r
            current_segment.append(i)
        elif abs(r - current_rmse) < 1e-6:
            current_segment.append(i)
        else:
            segments.append(current_segment)
            current_segment = [i]
            current_rmse = r
    if current_segment:
        segments.append(current_segment)
        
    if not segments:
        print("No valid structural segments found for this pixel.")
        return

    raw_h5_path = re.sub(r'_SC_.*?\.h5$', '.h5', source_h5_path)
    target_seasons = ['Spring', 'Summer', 'Fall']
    target_sensors = ['Landsat', 'Sentinel-2', 'Tanager']
    n_sensors = len(target_sensors)
    
    fig, axes = plt.subplots(n_sensors, 3, figsize=(18, 5 * n_sensors), squeeze=False)
    fig.suptitle(f"Seasonal Separability Across Segments (Centroid vs Variance)\nPixel: x={pixel_x}, y={pixel_y}", fontsize=14)
    
    segment_colors = ['blue', 'green', 'purple', 'orange', 'cyan', 'magenta', 'brown']
    
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
                        if get_season(dt) != season:
                            continue
                            
                        grid = source_grids[i]
                        frame_idx = source_frames[i]
                        
                        sr_ds = f_raw[f"/HDFEOS/GRIDS/{grid}/Data Fields/surface_reflectance"]
                        patch = sr_ds[frame_idx, :, y_start:y_end, x_start:x_end]
                        
                        w_attr = sr_ds.attrs.get('wavelengths')
                        if w_attr is None: w_attr = sr_ds.attrs.get('wavelength')
                        if w_attr is not None:
                            wavelengths = w_attr[:]
                        else:
                            wavelengths = np.arange(1, patch.shape[0] + 1, dtype=float)
                            
                        if np.max(wavelengths) < 10:
                            wavelengths = wavelengths * 1000
                            
                        patch = np.transpose(patch, (1, 2, 0))
                        em, _ = sc.maximumDistance(patch, N_ENDMEMBERS, strict_nan=False)
                        if not np.isnan(em).all():
                            season_spectra.append((wavelengths, em))
                    
                    if not season_spectra:
                        continue
                        
                    # Plot the density cloud
                    lines_by_len = {}
                    for wavelengths, em in season_spectra:
                        l = len(wavelengths)
                        if l not in lines_by_len:
                            lines_by_len[l] = []
                        for j in range(em.shape[1]):
                            if not np.isnan(em[:, j]).all():
                                ax.plot(wavelengths, em[:, j], color=seg_color, alpha=0.1)
                                lines_by_len[l].append((wavelengths, em[:, j]))
                                
                    # Plot the segment centroid for this season
                    for l, items in lines_by_len.items():
                        wl_arr = items[0][0]
                        arr = np.array([itm[1] for itm in items])
                        centroid = np.nanmean(arr, axis=0)
                        ax.plot(wl_arr, centroid, color=seg_color, linewidth=2.5, label=f"Seg {seg_idx+1} Centroid")
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
    plt.show(block=False)

def plot_lcmap_classes(pixel_y, pixel_x, source_h5_path, inference_results_h5):
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
    plt.show(block=False)

def plot_spatial_anomaly_overlay(source_h5_path, inference_results_h5):

    with h5py.File(inference_results_h5, 'r') as f:
        target_metric = f.attrs.get('TARGET_METRIC', 'sliding_volume_z_score')

    with h5py.File(source_h5_path, 'r') as f:
        harm_grp = f['/HDFEOS/GRIDS/HARMONIZED/Data Fields']
        acq_time = harm_grp[target_metric].attrs['acquisition_time'][:]
        unified_masks = harm_grp['common_mask'][:]
        full_valid_mask = ~unified_masks.astype(bool)
        
    def get_ortho(idx):
        with h5py.File(source_h5_path, 'r') as f:
            harm_grp = f['/HDFEOS/GRIDS/HARMONIZED/Data Fields']
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
        anomaly_map = f['change_date_timestamp'][:]
        change_count_map = f['change_count'][:]
        min_samples = f.attrs.get('MIN_SAMPLES', 20)
        
    H, W = full_valid_mask.shape[1], full_valid_mask.shape[2]
    
    anomaly_map[change_count_map == 0] = np.nan
    
    fig, (ax_img, ax_ts) = plt.subplots(1, 2, figsize=(18, 8))
    fig.subplots_adjust(bottom=0.15)
    
    from matplotlib.widgets import Button
    ax_anim_btn = fig.add_axes([0.31, 0.02, 0.15, 0.06])
    btn_anim = Button(ax_anim_btn, 'Animate 3x3 Endmembers')
    fig._btn_anim = btn_anim
    
    current_selected = {'x': None, 'y': None}
    
    def on_animate_click(event):
        x = current_selected['x']
        y = current_selected['y']
        if x is not None and y is not None:
            animate_pixel_endmembers(y, x, source_h5_path, inference_results_h5)
            
    btn_anim.on_clicked(on_animate_click)
    
    ax_spectra_btn = fig.add_axes([0.47, 0.02, 0.15, 0.06])
    btn_spectra = Button(ax_spectra_btn, 'Plot Segment Spectra')
    fig._btn_spectra = btn_spectra
    
    def on_spectra_click(event):
        x = current_selected['x']
        y = current_selected['y']
        if x is not None and y is not None:
            plot_segment_spectra(y, x, source_h5_path, inference_results_h5)
            
    btn_spectra.on_clicked(on_spectra_click)
    
    
    
    
    ax_season_btn = fig.add_axes([0.63, 0.09, 0.15, 0.06])
    btn_season = Button(ax_season_btn, 'Plot Season Spectra')
    fig._btn_season = btn_season
    
    def on_season_click(event):
        x = current_selected['x']
        y = current_selected['y']
        if x is not None and y is not None:
            plot_season_spectra(y, x, source_h5_path, inference_results_h5)
            
    btn_season.on_clicked(on_season_click)

    ax_endmembers_btn = fig.add_axes([0.63, 0.02, 0.15, 0.06])
    btn_endmembers = Button(ax_endmembers_btn, 'Plot Segment Endmembers')
    fig._btn_endmembers = btn_endmembers
    
    def on_endmembers_click(event):
        x = current_selected['x']
        y = current_selected['y']
        if x is not None and y is not None:
            plot_segment_endmembers(y, x, source_h5_path, inference_results_h5, strict_exclusion=True)
            
    btn_endmembers.on_clicked(on_endmembers_click)
    
    ax_lcmap_btn = fig.add_axes([0.79, 0.02, 0.10, 0.06])
    btn_lcmap = Button(ax_lcmap_btn, 'LCMAP')
    fig._btn_lcmap = btn_lcmap
    
    def on_lcmap_click(event):
        x = current_selected['x']
        y = current_selected['y']
        if x is not None and y is not None:
            plot_lcmap_classes(y, x, source_h5_path, inference_results_h5)
            
    btn_lcmap.on_clicked(on_lcmap_click)
    
    ax_sep_btn = fig.add_axes([0.90, 0.02, 0.08, 0.06])
    btn_sep = Button(ax_sep_btn, 'Separability')
    fig._btn_sep = btn_sep
    
    def on_sep_click(event):
        x = current_selected['x']
        y = current_selected['y']
        if x is not None and y is not None:
            plot_seasonal_separability(y, x, source_h5_path, inference_results_h5)
            
    btn_sep.on_clicked(on_sep_click)

    extent = None
    geo_transform = None
    with h5py.File(inference_results_h5, 'r') as f:
        geo_transform = f.attrs.get('GeoTransform')
    if geo_transform is None:
        with h5py.File(source_h5_path, 'r') as f:
            harm_grp = f['/HDFEOS/GRIDS/HARMONIZED/Data Fields']
            geo_transform = harm_grp[target_metric].attrs.get('GeoTransform')

    if geo_transform is not None:
        gt = geo_transform
        left = gt[0]
        right = gt[0] + W * gt[1]
        top = gt[3]
        bottom = gt[3] + H * gt[5]
        extent = [left, right, bottom, top]

    ax_img.imshow(base_frame, extent=extent)
    ax_img.set_title(f"{base_sg} Acquisition: {base_date.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    
    valid_initial_counts = np.sum(full_valid_mask, axis=0)
    insufficient_data = valid_initial_counts < min_samples
    
    gray = np.zeros((H, W, 4))
    gray[insufficient_data, 0] = 0.5
    gray[insufficient_data, 1] = 0.5
    gray[insufficient_data, 2] = 0.5
    gray[insufficient_data, 3] = 0.5
    ax_img.imshow(gray, extent=extent)
    
    if not np.all(np.isnan(anomaly_map)):
        from matplotlib.cm import gist_rainbow
        masked_anom = np.ma.masked_invalid(anomaly_map)
        cmap = gist_rainbow
        cmap.set_bad(color='white', alpha=0)
        im = ax_img.imshow(masked_anom, cmap=cmap, alpha=0.7, extent=extent)
        cbar = plt.colorbar(im, ax=ax_img)
        ticks = cbar.get_ticks()
        min_anom, max_anom = np.nanmin(anomaly_map), np.nanmax(anomaly_map)
        if not np.isnan(min_anom):
            ticks = ticks[(ticks >= min_anom) & (ticks <= max_anom)]
            cbar.set_ticks(ticks)
            cbar.set_ticklabels([datetime.fromtimestamp(t, timezone.utc).strftime('%Y-%m-%d') for t in ticks])
            cbar.set_label('First Date of Detected Change')
            
    if extent is not None:
        rect = patches.Rectangle((0, 0), 1, 1, linewidth=2, edgecolor='orange', facecolor='none', visible=False)
    else:
        rect = patches.Rectangle((-1, -1), 1, 1, linewidth=2, edgecolor='orange', facecolor='none', visible=False)
    ax_img.add_patch(rect)
    
    ax_ts.text(0.5, 0.5, 'Click a pixel on the map to view data', horizontalalignment='center', verticalalignment='center', transform=ax_ts.transAxes)

    def onclick(event):
        if event.inaxes != ax_img: return
        if extent is not None:
            x = int((event.xdata - gt[0]) / gt[1])
            y = int((event.ydata - gt[3]) / gt[5])
        else:
            x, y = int(event.xdata), int(event.ydata)
            
        if x < 0 or x >= W or y < 0 or y >= H: return
        print(f"Clicked on {x}, {y}")
        
        if extent is not None:
            rect.set_xy((gt[0] + x * gt[1], gt[3] + (y + 1) * gt[5]))
            rect.set_width(gt[1])
            rect.set_height(abs(gt[5]))
        else:
            rect.set_xy((x - 0.5, y - 0.5))
        rect.set_visible(True)
        
        current_selected['x'] = x
        current_selected['y'] = y
        
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
        #ax_ts.set_ylim([-4, 4])
        fig.canvas.draw()

    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

if __name__ == "__main__":
    inference_h5 = get_inference_h5(LOCATION, CONFIG, TARGET_METRIC)
    if inference_h5 and os.path.exists(inference_h5):
        print(f"Loading latest inference results: {inference_h5}")
        
        # Check source H5 file corresponding to inference
        try:
            with h5py.File(inference_h5, 'r') as f:
                loc_name = f.attrs.get('LOCATION', LOCATION)
        except Exception:
            loc_name = LOCATION
        source_h5 = get_source_h5_path(loc_name)
        if not os.path.exists(source_h5):
            source_h5 = H5_PATH
            
        plot_spatial_anomaly_overlay(source_h5, inference_h5)
    else:
        print(f"No inference results found for {LOCATION}. Run harmonized_CCD_main.py first.")
