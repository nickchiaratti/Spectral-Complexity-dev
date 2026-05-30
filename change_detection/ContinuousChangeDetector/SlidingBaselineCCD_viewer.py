import h5py
import numpy as np
import matplotlib.pyplot as plt
import datetime
import math
import rasterio
from pyproj import Transformer
from tqdm import tqdm

# ==========================================
# 1. CONFIGURATION
# ==========================================
H5_RAW_PATH = "C:/satelliteImagery/LANDSAT/Tait/LANDSAT_Stack_Tait_GEE_2015_2025_SC_EM-7_Gram-minEndmember_Norm-bandCount.h5"
TARGET_METRIC = 'sliding_volume_z_score'

# Output formatting to match the new Sliding Baseline (SB) script
if TARGET_METRIC == 'sliding_volume_z_score':
    suffix = '_zscore'
elif TARGET_METRIC == 'sliding_volume_map':
    suffix = '_SC'
elif TARGET_METRIC == 'evi_map':
    suffix = '_EVI'
else:
    suffix = ''

H5_SB_OUTPUT_PATH = f"C:/satelliteImagery/LANDSAT/Tait/SB_Change_Detection_Tait{suffix}.h5"

# --- Basemap Configuration ---
BASEMAP_TARGET_DATE = "2025-09-19" 
LANDSAT_RGB_BANDS = (3, 2, 1)      

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
    if np.isnan(frac_year): return "Unknown"
    frac_year = float(frac_year)
    year = int(frac_year)
    remainder = frac_year - year
    is_leap = year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
    days_in_year = 366 if is_leap else 365
    dt = datetime.datetime(year, 1, 1, tzinfo=datetime.timezone.utc) + datetime.timedelta(days=remainder * days_in_year)
    return dt.strftime('%Y-%m')

def percentile_normalize(arr, lower=2.0, upper=98.0):
    arr_valid = arr[~np.isnan(arr)]
    if len(arr_valid) == 0: return np.zeros_like(arr)
    p_low, p_high = np.percentile(arr_valid, (lower, upper))
    if p_low == p_high: return np.zeros_like(arr)
    return np.clip((arr - p_low) / (p_high - p_low), 0, 1)

# ==========================================
# 3. INTERACTIVE GUI CLASS
# ==========================================
class SlidingBaselineViewer:
    def __init__(self):
        print("Loading datasets into memory...")
        self.f_raw = h5py.File(H5_RAW_PATH, 'r')
        self.data_grp = self.f_raw['/HDFEOS/GRIDS/LANDSAT/Data Fields']
        self.sr_ds = self.data_grp['surface_reflectance']
        self.raw_data = np.nan_to_num(self.data_grp[TARGET_METRIC][...], nan=0.0)
        
        self.frac_years, self.timestamps = extract_fractional_years(self.sr_ds.attrs.get('acquisition_time'))
        
        print(f"Loading Continuous SB output from: {H5_SB_OUTPUT_PATH}")
        self.f_sb = h5py.File(H5_SB_OUTPUT_PATH, 'r')
        self.coeffs = self.f_sb['coefficients'][...]
        self.rmse = self.f_sb['rmse'][...]
        self.change_mask = self.f_sb['change_mask'][...]
        self.change_date = self.f_sb['change_date_frac_year'][...]
        
        self.height, self.width = self.change_mask.shape
        
        geo_transform = self.sr_ds.attrs.get('GeoTransform')
        spatial_ref = self.sr_ds.attrs.get('spatial_ref')
        self.affine_transform = None
        self.proj_transformer = None
        
        if geo_transform is not None and spatial_ref is not None:
            self.affine_transform = rasterio.Affine(*geo_transform)
            if isinstance(spatial_ref, bytes): spatial_ref = spatial_ref.decode('utf-8')
            crs = rasterio.crs.CRS.from_wkt(spatial_ref)
            self.proj_transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
            
        self._precalculate_basemaps()
        self._init_ui()

    def _precalculate_basemaps(self):
        print("Pre-calculating and caching true-color basemaps...")
        self.cached_basemaps = []
        for i in tqdm(range(len(self.timestamps)), desc="Caching RGB"):
            sr_frame = self.sr_ds[i, ...]
            rgb_uint8 = (np.stack([
                percentile_normalize(sr_frame[LANDSAT_RGB_BANDS[0], ...]),
                percentile_normalize(sr_frame[LANDSAT_RGB_BANDS[1], ...]),
                percentile_normalize(sr_frame[LANDSAT_RGB_BANDS[2], ...])
            ], axis=-1) * 255).astype(np.uint8)
            self.cached_basemaps.append(rgb_uint8)
            
        target_dt = datetime.datetime.strptime(BASEMAP_TARGET_DATE, "%Y-%m-%d").replace(tzinfo=datetime.timezone.utc)
        self.default_basemap_idx = np.argmin([abs((dt - target_dt).total_seconds()) for dt in self.timestamps])
        self.current_basemap_idx = self.default_basemap_idx

    def _init_ui(self):
        self.fig, (self.ax_map, self.ax_plot) = plt.subplots(1, 2, figsize=(16, 7))
        self.fig.canvas.manager.set_window_title("Sliding Baseline Diagnostic Viewer")
        plt.subplots_adjust(wspace=0.2)

        rgb_basemap = self.cached_basemaps[self.current_basemap_idx]
        basemap_date = self.timestamps[self.current_basemap_idx].strftime("%Y-%m-%d")
        self.basemap_img = self.ax_map.imshow(rgb_basemap)
        
        change_overlay = np.ma.masked_where(self.change_mask == 0, self.change_date)
        im = self.ax_map.imshow(change_overlay, cmap='coolwarm', alpha=0.9, interpolation='none')
        self.fig.colorbar(im, ax=self.ax_map, orientation='horizontal', pad=0.05, label="Year of Structural Change")
        
        self.ax_map.set_title(f"Detected Anomalies\nBasemap: {basemap_date} (True Color)")
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

        if is_anomalous:
            time_diffs = np.abs(self.frac_years - anomaly_date)
            best_idx = np.argmin(time_diffs)
        else:
            best_idx = self.default_basemap_idx
            
        if best_idx != self.current_basemap_idx:
            self.current_basemap_idx = best_idx
            self.basemap_img.set_data(self.cached_basemaps[best_idx])
            matched_date = self.timestamps[best_idx].strftime("%Y-%m-%d")
            self.ax_map.set_title(f"Detected Anomalies\nBasemap: {matched_date} (True Color)")

        # --- DOCTORAL LOGIC: DYNAMIC STATE-MACHINE RENDERING ---
        if is_anomalous:
            # Mask data strictly based on the localized anomaly date
            pre_anomaly_mask = self.frac_years < anomaly_date
            post_anomaly_mask = self.frac_years >= anomaly_date
            
            self.ax_plot.scatter(self.frac_years[pre_anomaly_mask], pixel_data[pre_anomaly_mask], 
                                 c='black', s=15, label='Active Baseline Obs', zorder=3)
            self.ax_plot.scatter(self.frac_years[post_anomaly_mask], pixel_data[post_anomaly_mask], 
                                 c='gray', s=15, alpha=0.7, label='Post-Break Obs', zorder=3)
                                 
            # Truncate the harmonic wave shortly after it breaks
            dense_t = np.linspace(np.min(self.frac_years), anomaly_date + 0.25, 500)
            
            date_str = frac_year_to_year_month(anomaly_date)
            self.ax_plot.axvline(x=anomaly_date, color='red', linestyle='--', linewidth=2, label=f'Break Detected ({date_str})')
            self.ax_plot.set_title(f"STRUCTURAL CHANGE CONFIRMED | {coord_str}")
        else:
            # If stable, all points belong to the continuously updating baseline
            self.ax_plot.scatter(self.frac_years, pixel_data, c='black', s=15, label='Stable Baseline Obs', zorder=3)
            dense_t = np.linspace(np.min(self.frac_years), np.max(self.frac_years), 1000)
            self.ax_plot.set_title(f"Stable Baseline | {coord_str}")

        if not np.isnan(pixel_coeffs[0]):
            dense_X = build_harmonic_matrix(dense_t)
            smooth_curve = dense_X @ pixel_coeffs
            self.ax_plot.plot(dense_t, smooth_curve, 'b-', linewidth=2, label='Final Harmonic Model')
            
            upper_bound = smooth_curve + (3.0 * pixel_rmse)
            lower_bound = smooth_curve - (3.0 * pixel_rmse)
            self.ax_plot.fill_between(dense_t, lower_bound, upper_bound, color='blue', alpha=0.15, label='±3 RMSE Boundary')

        self.ax_plot.set_xlabel("Year")
        self.ax_plot.set_ylabel(f"Metric: {TARGET_METRIC}")
        self.ax_plot.grid(True, linestyle='--', alpha=0.6)
        self.ax_plot.legend(loc='best', fontsize='small')
        self.fig.canvas.draw_idle()

if __name__ == "__main__":
    viewer = SlidingBaselineViewer()
    plt.show()