import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import datetime
import math
import rasterio
from pyproj import Transformer
from tqdm import tqdm
from zoneinfo import ZoneInfo
import SpecComplex as sc
import scienceplots
plt.style.use(['grid','sans'])

# ==========================================
# 1. INITIAL PATH RESOLUTION
# ==========================================
Location = "Tait"
multisensor = True

TARGET_METRIC_ESTIMATE = 'sliding_volume_z_score_masked'


suffix = "_WRS16"
if TARGET_METRIC_ESTIMATE == 'sliding_volume_z_score':
    suffix += '_zscore'
if TARGET_METRIC_ESTIMATE == 'sliding_volume_z_score_masked':
    suffix += '_maskedZscore'
elif TARGET_METRIC_ESTIMATE == 'sliding_volume_map':
    suffix += '_SC'
elif TARGET_METRIC_ESTIMATE == 'evi_map':
    suffix += '_EVI'

if multisensor:
    H5_CCDC_OUTPUT_PATH = f"C:/satelliteImagery/AnomalyDetector/CCD/{Location}/CCD_Multisensor_Change_Detection_{Location}{suffix}.h5"
else:
    H5_CCDC_OUTPUT_PATH = f"C:/satelliteImagery/AnomalyDetector/CCD/{Location}/CCD_Landsat_Change_Detection_{Location}{suffix}.h5"

BASEMAP_TARGET_DATE = "2025-09-19" 

COMPLEXITY_DICT = {
    'sliding_volume_map': 'Spectral Complexity',
    'sliding_volume_z_score': 'Spectral Complexity Z-Score',
    'sliding_volume_z_score_masked': 'Spectral Complexity (Z-Score)',
    'sliding_volume_local_z_score': 'Spectral Complexity Local Z-Score'
}

# ==========================================
# 2. UTILITIES
# ==========================================
def extract_fractional_years(acq_times):
    frac_years = []
    timestamps = []
    for dt in acq_times:
        try:
            dt_obj = datetime.datetime.fromtimestamp(float(dt), tz=datetime.timezone.utc)
        except ValueError:
            dt_str = dt.decode('utf-8') if isinstance(dt, bytes) else str(dt)
            dt_obj = datetime.datetime.strptime(dt_str[:10], "%Y-%m-%d").replace(tzinfo=datetime.timezone.utc)
            
        timestamps.append(dt_obj)
        year = dt_obj.year
        start_of_year = datetime.datetime(year, 1, 1, tzinfo=datetime.timezone.utc)
        start_of_next = datetime.datetime(year + 1, 1, 1, tzinfo=datetime.timezone.utc)
        year_duration = (start_of_next - start_of_year).total_seconds()
        elapsed = (dt_obj - start_of_year).total_seconds()
        frac_years.append(year + (elapsed / year_duration))
        
    return np.array(frac_years), timestamps

def build_harmonic_matrix(t, num_harmonics):
    """
    Constructs a Fourier basis matrix incorporating a linear trend.
    Dynamically scales based on the number of harmonics.
    Columns: [Intercept, Slope, Cos(1x), Sin(1x), Cos(2x), Sin(2x), ..., Cos(ux), Sin(ux)]
    """
    w = 2.0 * math.pi
    cols = [
        np.ones_like(t),  # a0 (Intercept)
        t                 # c1 (Linear Trend)
    ]
    for u in range(1, num_harmonics + 1):
        cols.append(np.cos(u * w * t))
        cols.append(np.sin(u * w * t))
    return np.column_stack(cols)

def frac_year_to_year_month(frac_year):
    if np.isnan(frac_year):
        return "Unknown"
    frac_year = float(frac_year)
    year = int(frac_year)
    remainder = frac_year - year
    is_leap = year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
    days_in_year = 366 if is_leap else 365
    dt = datetime.datetime(year, 1, 1, tzinfo=datetime.timezone.utc) + datetime.timedelta(days=remainder * days_in_year)
    return dt.strftime('%Y-%m-%d')

# ==========================================
# 3. INTERACTIVE GUI CLASS
# ==========================================
class CCDCTrajectoryViewer:
    def __init__(self):
        print(f"Loading trained CCD parameters and models from: {H5_CCDC_OUTPUT_PATH}")
        self.f_ccdc = h5py.File(H5_CCDC_OUTPUT_PATH, 'r')
        
        self.multisensor = bool(self._get_required_attr('multisensor'))
        self.landsat_path = str(self._get_required_attr('landsat_path'))
        self.tanager_path = str(self._get_required_attr('tanager_path'))
        self.train_end_year = float(self._get_required_attr('train_end_year'))
        self.sun_elevation_threshold = float(self._get_required_attr('sun_elevation_threshold'))
        self.cloud_dilation = int(self._get_required_attr('cloud_dilation'))
        self.qa_reject_mask = int(self._get_required_attr('qa_reject_mask'))
        self.radsat_accept_value = int(self._get_required_attr('radsat_accept_value'))
        self.aerosol_accept_level = str(self._get_required_attr('aerosol_accept_level'))
        self.target_metric = str(self._get_required_attr('target_metric'))

        # DOCTORAL UPGRADE: Explicit Metadata Provenance
        # Reading architectural constraints directly from the trained HDF5 file
        self.num_harmonics = int(self._get_required_attr('num_harmonics'))
        self.min_training_observations = int(self._get_required_attr('min_training_observations'))
        self.rmse_multiplier = float(self._get_required_attr('rmse_multiplier'))
        
        print(f"\n--- Extracted Model Architecture ---")
        print(f"Harmonics: {self.num_harmonics} | Min Obs Required: {self.min_training_observations}")

        self.coeffs = self.f_ccdc['coefficients'][...]
        self.rmse = self.f_ccdc['rmse'][...]
        self.change_mask = self.f_ccdc['change_mask'][...]
        self.change_date = self.f_ccdc['change_date_frac_year'][...]
        
        # DOCTORAL UPGRADE: Dynamically infer harmonic depth from the trained coefficients matrix
        # Shape is [num_coeffs, height, width]. num_coeffs = 2 (Intercept + Slope) + (2 * num_harmonics)
        num_coeffs = self.coeffs.shape[0]
        self.num_harmonics = (num_coeffs - 2) // 2
        print(f"Dynamically detected {self.num_harmonics} harmonics ({num_coeffs} parameters) from trained baseline.")
        
        print("\nLoading Virtual Constellation datasets into memory...")
        if self.multisensor:
            l_frac, l_time, l_y, l_mask, l_base, l_sens, l_h, l_w, l_gt, l_sr = self._load_and_preprocess_sensor(self.landsat_path, 'LANDSAT')
            t_frac, t_time, t_y, t_mask, t_base, t_sens, t_h, t_w, t_gt, t_sr = self._load_and_preprocess_sensor(self.tanager_path, 'TANAGER')
            
            if l_h != t_h or l_w != t_w:
                raise ValueError(f"CRITICAL ERROR: Spatial dimension mismatch. Landsat: {l_h}x{l_w}, Tanager: {t_h}x{t_w}")
                
            all_frac_years = np.concatenate([l_frac, t_frac])
            all_timestamps = l_time + t_time
            all_y_data = np.concatenate([l_y, t_y], axis=0)
            all_valid_mask = np.concatenate([l_mask, t_mask], axis=0)
            all_basemaps = l_base + t_base
            all_sensor_flags = np.concatenate([l_sens, t_sens])
            
            self.height, self.width = l_h, l_w
            geo_transform, spatial_ref = l_gt, l_sr
        else:
            all_frac_years, all_timestamps, all_y_data, all_valid_mask, all_basemaps, all_sensor_flags, self.height, self.width, geo_transform, spatial_ref = self._load_and_preprocess_sensor(self.landsat_path, 'LANDSAT')

        print("\nSorting merged time-series chronologically...")
        sort_idx = np.argsort(all_frac_years)
        
        self.frac_years = all_frac_years[sort_idx]
        self.timestamps = [all_timestamps[i] for i in sort_idx]
        self.raw_data = all_y_data[sort_idx, ...]
        self.valid_mask = all_valid_mask[sort_idx, ...]
        self.cached_basemaps = [all_basemaps[i] for i in sort_idx]
        self.sensor_flags = all_sensor_flags[sort_idx]

        self.train_mask = self.frac_years <= (self.train_end_year + 1.0)
        self.test_mask = self.frac_years > (self.train_end_year + 1.0)
        
        self.dense_t = np.linspace(np.min(self.frac_years), np.max(self.frac_years), 1000)
        self.dense_X = build_harmonic_matrix(self.dense_t, self.num_harmonics)
        
        self.affine_transform = None
        self.proj_transformer = None
        
        if geo_transform is not None and spatial_ref is not None:
            if isinstance(spatial_ref, bytes):
                spatial_ref = spatial_ref.decode('utf-8')
            crs = rasterio.crs.CRS.from_wkt(spatial_ref)
            self.affine_transform = rasterio.transform.Affine.from_gdal(*geo_transform)
            self.proj_transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
            
        target_date = datetime.datetime.strptime(BASEMAP_TARGET_DATE, "%Y-%m-%d").replace(tzinfo=datetime.timezone.utc)
        time_diffs = [abs((dt - target_date).total_seconds()) for dt in self.timestamps]
        self.default_basemap_idx = np.argmin(time_diffs)
        
        self.active_x = None
        self.active_y = None
        self.seq_idx = 0 # Now directly maps to the global chronological index
            
        self._init_ui()

    def _get_required_attr(self, attr_name):
        if attr_name not in self.f_ccdc.attrs:
            raise KeyError(f"CRITICAL ERROR: Configuration attribute '{attr_name}' is missing.")
        val = self.f_ccdc.attrs[attr_name]
        if isinstance(val, bytes):
            return val.decode('utf-8')
        elif isinstance(val, np.ndarray) and val.size == 1:
            item = val.item()
            return item.decode('utf-8') if isinstance(item, bytes) else item
        return val

    def _generate_3d_mask(self, grid_name, data_grp, indices, height, width):
        mask_3d = np.zeros((len(indices), height, width), dtype=bool)
        for i, original_idx in enumerate(tqdm(indices, desc=f"Generating {grid_name} QA Masks", leave=False)):
            if grid_name == 'LANDSAT':
                mask_3d[i] = sc.get_landsat_mask(
                    data_grp, original_idx, (height, width),
                    sun_elevation_threshold=self.sun_elevation_threshold,
                    cloud_dilation=self.cloud_dilation,
                    qa_reject_mask=self.qa_reject_mask,
                    radsat_accept_value=self.radsat_accept_value,
                    aerosol_accept_level=self.aerosol_accept_level
                )
            elif grid_name == 'TANAGER':
                mask_3d[i] = sc.get_tanager_mask(
                    data_grp, original_idx, (height, width),
                    sun_elevation_threshold=self.sun_elevation_threshold,
                    cloud_dilation=self.cloud_dilation
                )
        return mask_3d

    def _load_and_preprocess_sensor(self, h5_path, grid_expected):
        with h5py.File(h5_path, 'r') as f:
            data_grp = f[f'/HDFEOS/GRIDS/{grid_expected}/Data Fields']
            sr_ds = data_grp['surface_reflectance']
            vis_ds = data_grp['ortho_visual']
            height, width = sr_ds.shape[2], sr_ds.shape[3]
            
            acq_times = sr_ds.attrs.get('acquisition_time')
            frac_years, timestamps = extract_fractional_years(acq_times)
            num_frames = len(frac_years)
            
            y_data = data_grp[self.target_metric][...]
            indices = list(range(num_frames))
            valid_mask = self._generate_3d_mask(grid_expected, data_grp, indices, height, width)
            valid_mask &= ~np.isnan(y_data)
            
            basemaps = []
            for i in tqdm(range(num_frames), desc=f"Caching {grid_expected} ARD Basemaps"):
                raw_vis = vis_ds[i, ...]
                bip_vis = np.transpose(raw_vis, (1, 2, 0))
                rgba = bip_vis.astype(np.float32) / 255.0
                #rgba[..., 3] = np.where(bip_vis[..., 3] > 0, 1.0, 0.0)
                basemaps.append(rgba)
            
            geo_transform = sr_ds.attrs.get('GeoTransform')
            spatial_ref = sr_ds.attrs.get('spatial_ref')
            sensor_flags = np.array([grid_expected] * num_frames)
            
            return frac_years, timestamps, y_data, valid_mask, basemaps, sensor_flags, height, width, geo_transform, spatial_ref

    def _init_control_ui(self):
        self.fig_controls = plt.figure(figsize=(6, 2.5))
        self.fig_controls.canvas.manager.set_window_title("Basemap Navigation")
        
        self.ax_info = self.fig_controls.add_axes([0.05, 0.5, 0.9, 0.4])
        self.ax_info.axis('off')
        self.info_text = self.ax_info.text(0.5, 0.5, "Select a pixel on the map to begin.", 
                                           ha='center', va='center', fontsize=11, family='monospace',
                                           bbox=dict(facecolor='#f4f4f4', edgecolor='gray', boxstyle='round,pad=0.5'))
        
        ax_prev = self.fig_controls.add_axes([0.1, 0.1, 0.35, 0.3])
        ax_next = self.fig_controls.add_axes([0.55, 0.1, 0.35, 0.3])
        
        self.btn_prev = Button(ax_prev, '<< Prev Frame')
        self.btn_next = Button(ax_next, 'Next Frame >>')
        
        self.btn_prev.on_clicked(self._on_prev)
        self.btn_next.on_clicked(self._on_next)

    def _init_ui(self):
        self._init_control_ui()
        self.fig, (self.ax_map, self.ax_plot) = plt.subplots(1, 2, figsize=(16, 7))
        self.fig.canvas.manager.set_window_title("CCD Static Baseline Diagnostic Viewer")
        plt.subplots_adjust(wspace=0.2)

        rgba_basemap = self.cached_basemaps[self.default_basemap_idx]
        dt_utc = self.timestamps[self.default_basemap_idx]
        dt_est = dt_utc.astimezone(ZoneInfo("America/New_York"))
        basemap_date = dt_est.strftime("%Y-%m-%d %H:%M %Z")
        
        self.basemap_img = self.ax_map.imshow(rgba_basemap)
        
        change_overlay = np.ma.masked_where(self.change_mask == 0, self.change_date)
        im = self.ax_map.imshow(change_overlay, cmap='coolwarm', alpha=0.9, interpolation='none')
        self.fig.colorbar(im, ax=self.ax_map, orientation='horizontal', pad=0.05, label="Year of Structural Change")
        
        self.ax_map.set_title(f"Detected Anomalies\nBasemap: {basemap_date} ({self.sensor_flags[self.default_basemap_idx]})")
        self.ax_map.axis('off')
        
        self.marker_circle, = self.ax_map.plot([], [], marker='o',color='orange', markersize=12, fillstyle='none', mew=1.5)
        
        self.ax_plot.set_title("Temporal Trajectory & Harmonic Fit")
        self.ax_plot.grid(True, linestyle='--', alpha=0.6)
        
        self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self._load_pixel_state(self.width // 2, self.height // 2)

    def onclick(self, event):
        if event.inaxes == self.ax_map:
            x, y = int(round(event.xdata)), int(round(event.ydata))
            if 0 <= x < self.width and 0 <= y < self.height:
                self._load_pixel_state(x, y)

    def _on_prev(self, event):
        if self.active_x is not None and self.seq_idx > 0:
            self.seq_idx -= 1
            self._render_current_state()

    def _on_next(self, event):
        if self.active_x is not None and self.seq_idx < len(self.timestamps) - 1:
            self.seq_idx += 1
            self._render_current_state()

    def _display_error(self, x, y, msg):
        self.ax_plot.clear()
        self.ax_plot.text(0.5, 0.5, msg, ha='center', va='center', color='red', 
                          fontsize=11, fontweight='bold', wrap=True,
                          bbox=dict(facecolor='#ffebee', alpha=0.9, edgecolor='darkred'))
        
        coord_str = f"Pixel ({x}, {y})"
        self.ax_plot.set_title(f"ALGORITHMIC VALIDATION HALTED | {coord_str}", color='darkred')
        self.ax_plot.axis('off')
        self.info_text.set_text("Validation Halted. Select a new pixel.")
        self.fig_controls.canvas.draw_idle()
        self.fig.canvas.draw_idle()

    def _load_pixel_state(self, x, y):
        self.active_x = x
        self.active_y = y
            
        is_anomalous = self.change_mask[y, x]
        anomaly_date = self.change_date[y, x]
        pixel_data = self.raw_data[:, y, x]
        pixel_coeffs = self.coeffs[:, y, x]
        pixel_rmse = self.rmse[y, x]
        pixel_valid_mask = self.valid_mask[:, y, x]

        try:
            if is_anomalous:
                valid_indices_after = np.where(pixel_valid_mask & (self.frac_years >= anomaly_date - 1e-5))[0]
                if len(valid_indices_after) < 1:
                    raise ValueError(f"CRITICAL ERROR: Failed to locate the first valid frame in the anomaly sequence.")
                
                first_global_idx = valid_indices_after[0]
                
                X_all = build_harmonic_matrix(self.frac_years, self.num_harmonics)
                y_pred_all = X_all @ pixel_coeffs
                residual = np.abs(pixel_data[first_global_idx] - y_pred_all[first_global_idx])
                
                if residual <= 3.0 * pixel_rmse:
                    raise ValueError(f"CRITICAL ERROR: The first anomaly frame ({self.frac_years[first_global_idx]:.3f}) is NOT mathematically anomalous.\nResidual ({residual:.3f}) <= 3*RMSE ({3.0*pixel_rmse:.3f}).")
                
                # Snap to the global chronological index of the 1st anomaly
                self.seq_idx = first_global_idx
            else:
                self.seq_idx = self.default_basemap_idx
                
        except ValueError as e:
            self._display_error(x, y, str(e))
            return

        self._render_current_state()

    def _render_current_state(self):
        x, y = self.active_x, self.active_y
        current_global_idx = self.seq_idx # Now operates on the dense timeline
        
        self.marker_circle.set_data([x], [y])
        
        if self.affine_transform and self.proj_transformer:
            east, north = self.affine_transform * (x + 0.5, y + 0.5)
            lon, lat = self.proj_transformer.transform(east, north)
            coord_str = f"Lat: {lat:.5f}, Lon: {lon:.5f}"
        else:
            coord_str = f"Pixel ({x}, {y})"
            
        is_anomalous = self.change_mask[y, x]
        anomaly_date = self.change_date[y, x]
        pixel_data = self.raw_data[:, y, x]
        pixel_coeffs = self.coeffs[:, y, x]
        pixel_rmse = self.rmse[y, x]

        self.basemap_img.set_data(self.cached_basemaps[current_global_idx])
        dt_utc = self.timestamps[current_global_idx]
        dt_est = dt_utc.astimezone(ZoneInfo("America/New_York"))
        matched_date = dt_est.strftime("%Y-%m-%d %H:%M %Z")
        matched_sensor = self.sensor_flags[current_global_idx]
        
        # --- DOCTORAL ENHANCEMENT: Explicit Dynamic QA Feedback ---
        is_frame_valid = self.valid_mask[current_global_idx, y, x]
        qa_status_text = "[VALID]" if is_frame_valid else "[QA REJECTED]"
        title_qa_color = "darkgreen" if is_frame_valid else "darkred"
        
        title_suffix = ""
        if is_anomalous:
            if abs(self.frac_years[current_global_idx] - anomaly_date) < 1e-5:
                title_suffix = "\n(Break Detected)"
            elif self.frac_years[current_global_idx] > anomaly_date:
                title_suffix = "\n(Post-Anomaly)"
            else:
                title_suffix = "\n(Pre-Anomaly)"
        
        self.ax_map.set_title(f"Detected Anomalies\nBasemap: {matched_date} ({matched_sensor}){title_suffix}")
        
        ctrl_str = f"Frame {self.seq_idx + 1} of {len(self.timestamps)} Total Obs\n{matched_date} | Sensor: {matched_sensor}\nQA Status: {qa_status_text}"
        self.info_text.set_text(ctrl_str)
        # Update the color of the Info Text box to reflect QA status instantly
        self.info_text.set_color('black')
        self.info_text.set_bbox(dict(facecolor='#e8f5e9' if is_frame_valid else '#ffebee', 
                                     edgecolor=title_qa_color, boxstyle='round,pad=0.5'))

        self.ax_plot.clear()
        
        pixel_valid_mask = self.valid_mask[:, y, x]
        
        # 1. Valid Points (Used by the Algorithm)
        l_train_v = self.train_mask & (self.sensor_flags == 'LANDSAT') & pixel_valid_mask
        t_train_v = self.train_mask & (self.sensor_flags == 'TANAGER') & pixel_valid_mask
        l_test_v  = self.test_mask & (self.sensor_flags == 'LANDSAT') & pixel_valid_mask
        t_test_v  = self.test_mask & (self.sensor_flags == 'TANAGER') & pixel_valid_mask
        
        # 2. Rejected Points (Ignored due to Clouds/Shadows/Sun Angle)
        l_train_i = self.train_mask & (self.sensor_flags == 'LANDSAT') & ~pixel_valid_mask
        t_train_i = self.train_mask & (self.sensor_flags == 'TANAGER') & ~pixel_valid_mask
        l_test_i  = self.test_mask & (self.sensor_flags == 'LANDSAT') & ~pixel_valid_mask
        t_test_i  = self.test_mask & (self.sensor_flags == 'TANAGER') & ~pixel_valid_mask

        if not np.isnan(pixel_coeffs[0]):
            X_all = build_harmonic_matrix(self.frac_years, self.num_harmonics)
            y_pred_all = X_all @ pixel_coeffs
            upper_bound = y_pred_all + (self.rmse_multiplier * pixel_rmse)
            lower_bound = y_pred_all - (self.rmse_multiplier * pixel_rmse)
            out_of_bounds = (pixel_data > upper_bound) | (pixel_data < lower_bound)
            
            l_test_v_in = l_test_v & ~out_of_bounds
            l_test_v_out = l_test_v & out_of_bounds
            t_test_v_in = t_test_v & ~out_of_bounds
            t_test_v_out = t_test_v & out_of_bounds
        else:
            l_test_v_in = l_test_v
            l_test_v_out = np.zeros_like(l_test_v)
            t_test_v_in = t_test_v
            t_test_v_out = np.zeros_like(t_test_v)

        # Plot Valid Observations
        if np.any(l_train_v): self.ax_plot.scatter(self.frac_years[l_train_v], pixel_data[l_train_v], c='black', marker='o', s=20, label='Landsat Train', zorder=4)
        if np.any(t_train_v): self.ax_plot.scatter(self.frac_years[t_train_v], pixel_data[t_train_v], c='black', marker='x', s=30, label='Tanager Train', zorder=4)
        if np.any(l_test_v_in): self.ax_plot.scatter(self.frac_years[l_test_v_in], pixel_data[l_test_v_in], c='gray', marker='o', s=20, label='Landsat Test', zorder=4)
        if np.any(l_test_v_out): self.ax_plot.scatter(self.frac_years[l_test_v_out], pixel_data[l_test_v_out], c='red', marker='o', s=20, label='Landsat Test (Anomaly)', zorder=5)
        if np.any(t_test_v_in): self.ax_plot.scatter(self.frac_years[t_test_v_in], pixel_data[t_test_v_in], c='gray', marker='x', s=30, label='Tanager Test', zorder=4)
        if np.any(t_test_v_out): self.ax_plot.scatter(self.frac_years[t_test_v_out], pixel_data[t_test_v_out], c='red', marker='x', s=30, label='Tanager Test (Anomaly)', zorder=5)

        # Plot Masked/Rejected Observations
        if np.any(l_train_i): self.ax_plot.scatter(self.frac_years[l_train_i], pixel_data[l_train_i], facecolors='none', edgecolors='black', marker='o', s=20, alpha=0.3, label='Landsat (Cloud Masked)', zorder=3)
        if np.any(t_train_i): self.ax_plot.scatter(self.frac_years[t_train_i], pixel_data[t_train_i], c='black', marker='x', s=30, alpha=0.3, label='Tanager (Cloud Masked)', zorder=3)
        if np.any(l_test_i): self.ax_plot.scatter(self.frac_years[l_test_i], pixel_data[l_test_i], facecolors='none', edgecolors='gray', marker='o', s=20, alpha=0.3, label='Landsat (Cloud Masked)', zorder=3)
        if np.any(t_test_i): self.ax_plot.scatter(self.frac_years[t_test_i], pixel_data[t_test_i], c='gray', marker='x', s=30, alpha=0.3, label='Tanager (Cloud Masked)', zorder=3)

        # --- Base Line Drawing Logic ---
        if not np.isnan(pixel_coeffs[0]):
            smooth_curve = self.dense_X @ pixel_coeffs
            self.ax_plot.plot(self.dense_t, smooth_curve, 'b-', linewidth=2, label='CCD Baseline Fit')
            RMSE_upper_bound = smooth_curve + (self.rmse_multiplier * pixel_rmse)
            RMSE_lower_bound = smooth_curve - (self.rmse_multiplier * pixel_rmse)
            self.ax_plot.fill_between(self.dense_t, RMSE_lower_bound, RMSE_upper_bound, color='blue', alpha=0.15, label=f'±{self.rmse_multiplier} RMSE Boundary')
        else:
            self.ax_plot.text(0.5, 0.5, f"INSUFFICIENT CLEAR-SKY DATA\nBaseline could not be fit (n < {self.min_training_observations})", 
                               transform=self.ax_plot.transAxes, ha='center', va='center', 
                               color='darkorange', fontsize=12, fontweight='bold',
                               bbox=dict(facecolor='white', alpha=0.9, edgecolor='darkorange'))

        current_frac_year = self.frac_years[current_global_idx]
        
        # Change the style of the crosshair depending on the mathematical validity of the frame
        #if is_frame_valid:
        #    self.ax_plot.axvline(x=current_frac_year, color='orange', linestyle='-', linewidth=2.5, label='Currently Viewed Frame [Valid]')
        #else:
        #    self.ax_plot.axvline(x=current_frac_year, color='darkorange', linestyle=':', linewidth=2, label='Currently Viewed Frame [Rejected]')

        if is_anomalous:
            anomaly_date_str = frac_year_to_year_month(anomaly_date)
            self.ax_plot.axvline(x=anomaly_date, color='red', linestyle='--', linewidth=2, label=f'Change Detected ({anomaly_date_str})')
            self.ax_plot.set_title(f"Observations For: {coord_str}")
        else:
            self.ax_plot.set_title(f"Observations For: {coord_str}")

        self.ax_plot.axvline(x=self.train_end_year + 1.0, color='green', linestyle='-.', linewidth=2, label='End of Training')
        self.ax_plot.set_xlabel("Year")
        self.ax_plot.set_ylabel(f"{COMPLEXITY_DICT[self.target_metric]}")
        self.ax_plot.grid(True, linestyle='--', alpha=0.6)
        
        self.ax_plot.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
        plt.subplots_adjust(right=0.85) 
        self.ax_plot.set_ylim(-3.5,3.5)
        
        self.fig_controls.canvas.draw_idle()
        self.fig.canvas.draw_idle()

if __name__ == "__main__":
    viewer = CCDCTrajectoryViewer()
    plt.show()