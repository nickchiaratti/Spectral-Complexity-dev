import h5py
import numpy as np
import matplotlib.pyplot as plt
import datetime
import math
import rasterio
from pyproj import Transformer
from tqdm import tqdm
from zoneinfo import ZoneInfo
import SpecComplex as sc

# ==========================================
# 1. CONFIGURATION
# ==========================================
Location = "Tait"
Frame_Reg = "WRS16" #"CoReg"
H5_RAW_PATH = f"C:/satelliteImagery/LANDSAT/{Location}/LANDSAT_Stack_{Location}_GEE_2015_2025_{Frame_Reg}_SC_EM-7_Gram-minEndmember_Norm-bandCount.h5"

TARGET_METRIC = 'sliding_volume_z_score_masked'
suffix = f'_{Frame_Reg}'
if TARGET_METRIC == 'sliding_volume_z_score':
    suffix += '_zscore'
if TARGET_METRIC == 'sliding_volume_z_score_masked':
    suffix += '_maskedZscore'
elif TARGET_METRIC == 'sliding_volume_map':
    suffix += '_SC'
elif TARGET_METRIC == 'evi_map':
    suffix += '_EVI'

H5_CCDC_OUTPUT_PATH = f"C:/satelliteImagery/LANDSAT/{Location}/SlidingBaselineCCD_Change_Detection_{Location}{suffix}.h5"

# --- Basemap Configuration ---
BASEMAP_TARGET_DATE = "2025-09-19" 

# --- Centralized Masking Configuration ---
SUN_ELEVATION_THRESHOLD = 30
CLOUD_DILATION = 1
QA_REJECT_MASK = 0b111111
AEROSOL_ACCEPT_LEVEL = 'medium' 

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

def build_harmonic_matrix(t):
    w = 2.0 * math.pi
    return np.column_stack([
        np.ones_like(t), t, np.cos(w * t), np.sin(w * t), np.cos(2 * w * t), np.sin(2 * w * t)
    ])

def frac_year_to_year_month(frac_year):
    if np.isnan(frac_year):
        return "Unknown"
    frac_year = float(frac_year)
    year = int(frac_year)
    remainder = frac_year - year
    is_leap = year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
    days_in_year = 366 if is_leap else 365
    dt = datetime.datetime(year, 1, 1, tzinfo=datetime.timezone.utc) + datetime.timedelta(days=remainder * days_in_year)
    return dt.strftime('%Y-%m')

# ==========================================
# 3. INTERACTIVE GUI CLASS
# ==========================================
class CCDCTrajectoryViewer:
    def __init__(self):
        print("Loading datasets into memory...")
        
        self.f_raw = h5py.File(H5_RAW_PATH, 'r')
        
        # Dynamic Sensor Routing
        grids = list(self.f_raw['/HDFEOS/GRIDS'].keys())
        if 'LANDSAT' in grids:
            self.grid_name = 'LANDSAT'
        elif 'TANAGER' in grids:
            self.grid_name = 'TANAGER'
        else:
            raise KeyError("Neither LANDSAT nor TANAGER grid found in HDF5 file.")
            
        self.data_grp = self.f_raw[f'/HDFEOS/GRIDS/{self.grid_name}/Data Fields']
        self.sr_ds = self.data_grp['surface_reflectance']
        self.raw_data = np.nan_to_num(self.data_grp[TARGET_METRIC][...], nan=0.0)
        
        if 'ortho_visual' in self.data_grp:
            self.vis_ds = self.data_grp['ortho_visual']
        else:
            raise KeyError("CRITICAL: 'ortho_visual' dataset not found in HDF5 Data Fields.")
        
        self.frac_years, self.timestamps = extract_fractional_years(self.sr_ds.attrs.get('acquisition_time'))
        
        print(f"Loading generated CCD data from: {H5_CCDC_OUTPUT_PATH}")
        self.f_ccdc = h5py.File(H5_CCDC_OUTPUT_PATH, 'r')
        self.coeffs = self.f_ccdc['coefficients'][...]
        self.rmse = self.f_ccdc['rmse'][...]
        self.change_mask = self.f_ccdc['change_mask'][...]
        self.change_date = self.f_ccdc['change_date_frac_year'][...]
        
        self.height, self.width = self.change_mask.shape
        
        self.dense_t = np.linspace(np.min(self.frac_years), np.max(self.frac_years), 1000)
        self.dense_X = build_harmonic_matrix(self.dense_t)
        
        geo_transform = self.sr_ds.attrs.get('GeoTransform')
        spatial_ref = self.sr_ds.attrs.get('spatial_ref')
        
        self.affine_transform = None
        self.proj_transformer = None
        
        if geo_transform is not None and spatial_ref is not None:
            if isinstance(spatial_ref, bytes):
                spatial_ref = spatial_ref.decode('utf-8')
            crs = rasterio.crs.CRS.from_wkt(spatial_ref)
            self.affine_transform = rasterio.transform.Affine.from_gdal(*geo_transform)
            self.proj_transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
            
        self._precalculate_basemaps()
        self._precalculate_masks()
        self._init_ui()

    def _precalculate_basemaps(self):
        print("Extracting and caching true-color basemaps...")
        self.cached_basemaps = []
        
        for i in tqdm(range(len(self.timestamps)), desc="Caching ARD Frames"):
            raw_vis = self.vis_ds[i, ...]
            bip_vis = np.transpose(raw_vis, (1, 2, 0))
            rgba = bip_vis.astype(np.float32) / 255.0
            rgba[..., 3] = np.where(bip_vis[..., 3] > 0, 1.0, 0.0)
            self.cached_basemaps.append(rgba)
            
        target_date = datetime.datetime.strptime(BASEMAP_TARGET_DATE, "%Y-%m-%d").replace(tzinfo=datetime.timezone.utc)
        time_diffs = [abs((dt - target_date).total_seconds()) for dt in self.timestamps]
        self.default_basemap_idx = np.argmin(time_diffs)
        self.current_basemap_idx = None

    def _precalculate_masks(self):
        """Global mask generation ensures absolute parity with the training script."""
        print(f"Applying unified QA, Cloud, and Aerosol masks via SpecComplex for {self.grid_name}...")
        self.valid_mask = np.ones((len(self.timestamps), self.height, self.width), dtype=bool)
        
        for i in tqdm(range(len(self.timestamps)), desc="Caching Masks"):
            if self.grid_name == 'LANDSAT':
                self.valid_mask[i] = sc.get_landsat_mask(
                    self.data_grp, i, (self.height, self.width),
                    sun_elevation_threshold=SUN_ELEVATION_THRESHOLD,
                    cloud_dilation=CLOUD_DILATION,
                    qa_reject_mask=QA_REJECT_MASK,
                    radsat_accept_value=0,
                    aerosol_accept_level=AEROSOL_ACCEPT_LEVEL
                )
            elif self.grid_name == 'TANAGER':
                self.valid_mask[i] = sc.get_tanager_mask(
                    self.data_grp, i, (self.height, self.width),
                    sun_elevation_threshold=SUN_ELEVATION_THRESHOLD,
                    cloud_dilation=CLOUD_DILATION
                )

    def _init_ui(self):
        self.fig, (self.ax_map, self.ax_plot) = plt.subplots(1, 2, figsize=(16, 7))
        self.fig.canvas.manager.set_window_title("CCD Diagnostic Viewer (Fourier Decomposition)")
        plt.subplots_adjust(wspace=0.2)

        self.current_basemap_idx = self.default_basemap_idx
        rgba_basemap = self.cached_basemaps[self.current_basemap_idx]
        
        self.basemap_img = self.ax_map.imshow(rgba_basemap)
        
        change_overlay = np.ma.masked_where(self.change_mask == 0, self.change_date)
        im = self.ax_map.imshow(change_overlay, cmap='coolwarm', alpha=0.9, interpolation='none')
        self.fig.colorbar(im, ax=self.ax_map, orientation='horizontal', pad=0.05, label="Year of Structural Change")
        
        dt_utc = self.timestamps[self.current_basemap_idx]
        dt_est = dt_utc.astimezone(ZoneInfo("America/New_York"))
        matched_date = dt_est.strftime("%Y-%m-%d %H:%M %Z")
        self.ax_map.set_title(f"Detected Anomalies\nBasemap: {matched_date}")
        self.ax_map.axis('off')
        
        self.marker, = self.ax_map.plot([], [], 'kx', markersize=10, mew=2)
        self.marker_circle, = self.ax_map.plot([], [], 'ko', markersize=12, fillstyle='none', mew=1.5)
        
        self.ax_plot.set_xlabel("Year")
        self.ax_plot.set_ylabel(f"Normalized {TARGET_METRIC}")
        self.ax_plot.grid(True, linestyle='--', alpha=0.6)
        
        self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self._draw_trajectory(self.width // 2, self.height // 2)

    def onclick(self, event):
        if event.inaxes == self.ax_map:
            x, y = int(round(event.xdata)), int(round(event.ydata))
            if 0 <= x < self.width and 0 <= y < self.height:
                self._draw_trajectory(x, y)

    def _draw_trajectory(self, x, y):
        self.ax_plot.clear()
        
        self.marker.set_data([x], [y])
        self.marker_circle.set_data([x], [y])
        
        if self.affine_transform and self.proj_transformer:
            east, north = self.affine_transform * (x + 0.5, y + 0.5)
            lon, lat = self.proj_transformer.transform(east, north)
            coord_str = f"Lat: {lat:.5f}, Lon: {lon:.5f}"
        else:
            coord_str = f"Pixel ({x}, {y})"
        
        pixel_data = self.raw_data[:, y, x]
        pixel_coeffs = self.coeffs[:, y, x]
        pixel_rmse = self.rmse[y, x]
        is_anomalous = self.change_mask[y, x]
        anomaly_date = self.change_date[y, x]

        # =====================================================
        # STATE 1: ANOMALOUS PIXEL RENDERING
        # =====================================================
        if is_anomalous:
            # Slicing the precalculated SpecComplex mask instantly replaces the 3x3 patch logic
            pixel_valid_mask = self.valid_mask[:, y, x]
            
            # Seek the specific frames executing the threshold break
            valid_indices_after = np.where(pixel_valid_mask & (self.frac_years >= anomaly_date - 1e-5))[0]
            
            # Explicit Failure Enforcements (No fallback logic permitted)
            if len(valid_indices_after) < 2:
                raise ValueError(f"CRITICAL ERROR: Failed to locate the second valid frame in the anomaly sequence for Pixel ({x}, {y}). Sequence length < 2.")
            
            best_idx = valid_indices_after[1]
            
            # Strict Mathematical Validation
            X_all = build_harmonic_matrix(self.frac_years)
            y_pred_all = X_all @ pixel_coeffs
            residual = np.abs(pixel_data[best_idx] - y_pred_all[best_idx])
            
            if residual <= 3.0 * pixel_rmse:
                raise ValueError(f"CRITICAL ERROR: The second valid frame ({self.frac_years[best_idx]:.3f}) is NOT mathematically anomalous. Residual ({residual:.3f}) <= 3*RMSE ({3.0*pixel_rmse:.3f}).")

            if best_idx != self.current_basemap_idx:
                self.current_basemap_idx = best_idx
                self.basemap_img.set_data(self.cached_basemaps[best_idx])
            
            pre_mask = self.frac_years < anomaly_date
            post_mask = self.frac_years >= anomaly_date
            
            self.ax_plot.scatter(self.frac_years[pre_mask], pixel_data[pre_mask], c='black', s=15, label='Pre-Break Obs', zorder=3)
            self.ax_plot.scatter(self.frac_years[post_mask], pixel_data[post_mask], c='red', s=15, label='Anomalous Obs', zorder=3)

            if not np.isnan(pixel_coeffs[0]):
                smooth_curve = self.dense_X @ pixel_coeffs
                self.ax_plot.plot(self.dense_t, smooth_curve, 'b-', linewidth=2, label='CCD Baseline Fit')
                upper_bound = smooth_curve + (3.0 * pixel_rmse)
                lower_bound = smooth_curve - (3.0 * pixel_rmse)
                self.ax_plot.fill_between(self.dense_t, lower_bound, upper_bound, color='blue', alpha=0.15, label='±3 RMSE Boundary')

            anomaly_date_str = frac_year_to_year_month(anomaly_date)
            self.ax_plot.axvline(x=anomaly_date, color='red', linestyle='--', linewidth=2, label=f'Change Detected ({anomaly_date_str})')
            self.ax_plot.set_title(f"STRUCTURAL CHANGE CONFIRMED | {coord_str}")
            
            title_suffix = "\n(Anomaly Frame #2)"

        # =====================================================
        # STATE 2: STABLE PIXEL RENDERING (FOURIER DECOMPOSITION)
        # =====================================================
        else:
            best_idx = self.default_basemap_idx
            
            if best_idx != self.current_basemap_idx:
                self.current_basemap_idx = best_idx
                self.basemap_img.set_data(self.cached_basemaps[best_idx])
                
            self.ax_plot.scatter(self.frac_years, pixel_data, c='black', s=15, label='Validated Observations', zorder=3)

            if not np.isnan(pixel_coeffs[0]):
                trend_component = pixel_coeffs[0] + (pixel_coeffs[1] * self.dense_t)
                self.ax_plot.plot(self.dense_t, trend_component, color='darkorange', linestyle='--', linewidth=2.5, label='Underlying Linear Trend')
                
                smooth_curve = self.dense_X @ pixel_coeffs
                self.ax_plot.plot(self.dense_t, smooth_curve, 'b-', linewidth=1.5, label='Combined Fourier Harmonic')
                
                upper_bound = smooth_curve + (3.0 * pixel_rmse)
                lower_bound = smooth_curve - (3.0 * pixel_rmse)
                self.ax_plot.fill_between(self.dense_t, lower_bound, upper_bound, color='blue', alpha=0.15, label='±3 RMSE Boundary')

            self.ax_plot.set_title(f"Stable Analytical Baseline | {coord_str}")
            title_suffix = ""

        # --- Global Formatting Updates ---
        dt_utc = self.timestamps[best_idx]
        dt_est = dt_utc.astimezone(ZoneInfo("America/New_York"))
        matched_date = dt_est.strftime("%Y-%m-%d %H:%M %Z")
        self.ax_map.set_title(f"Detected Anomalies\nBasemap: {matched_date}{title_suffix}")

        self.ax_plot.set_xlabel("Year")
        self.ax_plot.set_ylabel(f"Metric: {TARGET_METRIC}")
        self.ax_plot.grid(True, linestyle='--', alpha=0.6)
        self.ax_plot.legend(loc='best', fontsize='small')
        
        self.fig.canvas.draw_idle()

if __name__ == "__main__":
    viewer = CCDCTrajectoryViewer()
    plt.show()