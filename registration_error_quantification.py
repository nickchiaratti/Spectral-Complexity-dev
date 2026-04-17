"""
Multi-Sensor Co-Registration Temporal Bracketing Quantifier
Interactive Visual Analytics Interface & Summary Dashboard

Iterates across the entire hyperspectral timeline, bracketing 
each epoch with the preceding and subsequent multispectral acquisitions.
Produces a longitudinal summary plot and an interactive UI to visually 
validate the Euclidean optimization vectors.

Author: [Your Name/Lab]
Date: 2026-04-15
"""

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import matplotlib.dates as mdates
from datetime import datetime, timezone
import rasterio.transform
from pyproj import Transformer, CRS
from skimage.registration import phase_cross_correlation
from skimage.transform import EuclideanTransform, warp
from scipy.optimize import minimize
from scipy.stats import pearsonr
import SpecComplex as sc

# ==========================================
# 1. CONFIGURATION
# ==========================================
Location = "Rochester"

TS_LOCATION = {'latlon': (43.092815, -77.621573), 'label': "43.131725, -77.560376"}
SPAN = 100 # 100x100 window required for robust Fourier structural context

landsat_path = f"C:/satelliteImagery/LANDSAT/{Location}/LANDSAT_Stack_{Location}_GEE_2015_2025_WRS16_SC_EM-7_Gram-minEndmember_Norm-bandCount.h5"
tanager_path = f"C:/satelliteImagery/Tanager/{Location}/Tanager_Stack_{Location}_HDFEOS_SC_EM-7_Gram-minEndmember_Norm-bandCount.h5"
# ==========================================
# 2. UTILITY FUNCTIONS
# ==========================================

def extract_luminance_and_mask(grp, f_idx, y_start, y_end, x_start, x_end):
    """
    Extracts the BSQ ortho_visual array, converts RGB to CIE 1931 Luminance,
    and returns it alongside the bounding alpha mask per Interface Spec.
    """
    raw_vis = grp['ortho_visual'][f_idx, :, y_start:y_end, x_start:x_end]
    bip_vis = np.transpose(raw_vis, (1, 2, 0))
    rgb = bip_vis[..., :3].astype(np.float32) / 255.0
    luminance = np.dot(rgb, [0.299, 0.587, 0.114])
    
    # Enforce Interface Spec: Alpha > 0 is valid. 
    if bip_vis.shape[-1] == 4:
        local_mask = bip_vis[..., 3] > 0
    else:
        local_mask = np.ones(luminance.shape, dtype=bool)
        
    return luminance, local_mask, rgb

# ==========================================
# 3. INTERACTIVE VIEWER & DASHBOARD CLASS
# ==========================================

class CoRegistrationViewer:
    def __init__(self, h5_l, h5_t):
        self.h5_l = h5_l
        self.h5_t = h5_t
        self.grp_l = h5_l['/HDFEOS/GRIDS/LANDSAT/Data Fields']
        self.grp_t = h5_t['/HDFEOS/GRIDS/TANAGER/Data Fields']
        
        self.comparisons = []
        self.current_idx = 0
        
        self._setup_metrology()
        self._precalculate_all_transformations()
        
        # Instantiate both the Statistical Overview and the Details-on-Demand UI
        self._plot_longitudinal_summary()
        self._init_ui()
        self.update_display()

    def _setup_metrology(self):
        """Initializes the geographic to pixel coordinate mappings."""
        geo_tf = self.grp_t['surface_reflectance'].attrs['GeoTransform']
        crs_wkt = self.grp_t['surface_reflectance'].attrs['spatial_ref']
        if isinstance(crs_wkt, bytes): crs_wkt = crs_wkt.decode('utf-8')
        
        self.affine = rasterio.transform.Affine.from_gdal(*geo_tf)
        self.pixel_width = abs(self.affine.a)
        self.pixel_height = abs(self.affine.e)
        
        crs = CRS.from_wkt(crs_wkt)
        transformer_to_px = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        
        t_lat, t_lon = TS_LOCATION['latlon']
        proj_x, proj_y = transformer_to_px.transform(t_lon, t_lat)
        px, py = ~self.affine * (proj_x, proj_y)
        self.t_x, self.t_y = int(round(px)), int(round(py))
        
        self.half_span = SPAN // 2
        self.h, self.w = self.grp_t['ortho_visual'].shape[2:] 
        self.x_start, self.x_end = max(0, self.t_x - self.half_span), min(self.w, self.t_x + self.half_span)
        self.y_start, self.y_end = max(0, self.t_y - self.half_span), min(self.h, self.t_y + self.half_span)
        
        self.rel_center_y = self.t_y - self.y_start
        self.rel_center_x = self.t_x - self.x_start

    def _precalculate_all_transformations(self):
        """Iterates through temporal brackets and caches optimization results for rapid UI rendering."""
        l_times = self.grp_l['surface_reflectance'].attrs['acquisition_time']
        t_times = self.grp_t['surface_reflectance'].attrs['acquisition_time']
        
        print(f"Pre-calculating optimizations for {len(t_times)} Tanager epochs...")
        
        for t_idx, t_time in enumerate(t_times):
            t_dt = datetime.fromtimestamp(t_time, tz=timezone.utc)
            
            l_before_indices = np.where(l_times < t_time)[0]
            l_after_indices = np.where(l_times > t_time)[0]
            
            l_prev_idx = l_before_indices[-1] if len(l_before_indices) > 0 else None
            l_next_idx = l_after_indices[0] if len(l_after_indices) > 0 else None
            
            if l_prev_idx is not None:
                self._compute_and_store(l_prev_idx, t_idx, l_times[l_prev_idx], t_dt, "L_prev (Before)")
            
            if l_next_idx is not None:
                self._compute_and_store(l_next_idx, t_idx, l_times[l_next_idx], t_dt, "L_next (After)")

    def _compute_and_store(self, l_idx, t_idx, l_time, t_dt, bracket_label):
        l_dt = datetime.fromtimestamp(l_time, tz=timezone.utc)
        delta_days = (t_dt - l_dt).total_seconds() / (60*60*24)
        
        print(f"  -> Processing {t_dt.strftime('%Y-%m-%d')} vs {l_dt.strftime('%Y-%m-%d')} ({bracket_label})...")
        
        lum_l, mask_l, _ = extract_luminance_and_mask(self.grp_l, l_idx, self.y_start, self.y_end, self.x_start, self.x_end)
        lum_t, mask_t, _ = extract_luminance_and_mask(self.grp_t, t_idx, self.y_start, self.y_end, self.x_start, self.x_end)
        
        overlap = mask_l & mask_t
        
        # Strict fail-fast handling without injecting fill values
        if np.sum(overlap) == 0:
            print("     [FAILED] No valid overlapping pixels.")
            self.comparisons.append({
                'valid': False, 't_idx': t_idx, 'l_idx': l_idx, 't_dt': t_dt, 'l_dt': l_dt, 
                'label': bracket_label, 'delta_days': delta_days, 'error_msg': '0 Valid Overlapping Pixels'
            })
            return

        # 1. Phase Correlation Seed
        shift_vector, error, diffphase = phase_cross_correlation(
            reference_image=lum_l, moving_image=lum_t, 
            reference_mask=mask_l, moving_mask=mask_t, upsample_factor=100
        )
        init_dy, init_dx = shift_vector

        # 2. Euclidean Optimization
        def objective_function(params):
            dy, dx, theta_deg = params
            t1 = EuclideanTransform(translation=(-self.rel_center_x, -self.rel_center_y))
            t2 = EuclideanTransform(rotation=np.deg2rad(theta_deg))
            t3 = EuclideanTransform(translation=(self.rel_center_x + dx, self.rel_center_y + dy))
            tform = t1 + t2 + t3
            
            warped_lum = warp(lum_t, tform.inverse, order=3, mode='constant', cval=np.nan)
            warped_mask = warp(mask_t.astype(float), tform.inverse, order=0, mode='constant', cval=0.0).astype(bool)
            
            current_overlap = mask_l & warped_mask & ~np.isnan(warped_lum)
            if np.sum(current_overlap) < (0.15 * mask_l.size):
                return 1.0 
            r, _ = pearsonr(lum_l[current_overlap], warped_lum[current_overlap])
            return -r 

        initial_guess = [init_dy, init_dx, 0.0]
        bounds = [(init_dy - 2.0, init_dy + 2.0), (init_dx - 2.0, init_dx + 2.0), (-2.0, 2.0)]
        
        result = minimize(objective_function, initial_guess, method='Nelder-Mead', bounds=bounds, options={'xatol': 1e-4, 'fatol': 1e-4})
        opt_dy, opt_dx, opt_theta = result.x
        
        shift_meters_x = opt_dx * self.pixel_width
        shift_meters_y = opt_dy * self.pixel_height
        magnitude_meters = np.sqrt(shift_meters_x**2 + shift_meters_y**2)

        self.comparisons.append({
            'valid': True, 't_idx': t_idx, 'l_idx': l_idx, 't_dt': t_dt, 'l_dt': l_dt, 
            'label': bracket_label, 'delta_days': delta_days,
            'opt_dy': opt_dy, 'opt_dx': opt_dx, 'opt_theta': opt_theta, 
            'corr': -result.fun, 'mag': magnitude_meters,
            'shift_m_x': shift_meters_x, 'shift_m_y': shift_meters_y
        })

    def _plot_longitudinal_summary(self):
        """Builds the 3-panel statistical overview of temporal registration drift."""
        unique_t_dates = sorted(list(set([comp['t_dt'] for comp in self.comparisons])))
        
        mag_prev, rot_prev, corr_prev = [], [], []
        mag_next, rot_next, corr_next = [], [], []
        plot_dates = []
        
        for t_dt in unique_t_dates:
            plot_dates.append(t_dt)
            c_prev = next((c for c in self.comparisons if c['t_dt'] == t_dt and c['label'] == 'L_prev (Before)'), None)
            c_next = next((c for c in self.comparisons if c['t_dt'] == t_dt and c['label'] == 'L_next (After)'), None)
            
            if c_prev and c_prev['valid']:
                mag_prev.append(c_prev['mag'])
                rot_prev.append(c_prev['opt_theta'])
                corr_prev.append(c_prev['corr'])
            else:
                mag_prev.append(np.nan); rot_prev.append(np.nan); corr_prev.append(np.nan)
                
            if c_next and c_next['valid']:
                mag_next.append(c_next['mag'])
                rot_next.append(c_next['opt_theta'])
                corr_next.append(c_next['corr'])
            else:
                mag_next.append(np.nan); rot_next.append(np.nan); corr_next.append(np.nan)

        self.fig_sum, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
        self.fig_sum.canvas.manager.set_window_title("Longitudinal Co-Registration Stability")
        
        # 1. Translation Magnitude Plot
        ax1.plot(plot_dates, mag_prev, marker='^', linestyle='--', color='tab:blue', linewidth=1.5, markersize=8, label='Landsat Before ($L_{-1}$)')
        ax1.plot(plot_dates, mag_next, marker='v', linestyle=':', color='tab:orange', linewidth=1.5, markersize=8, label='Landsat After ($L_{+1}$)')
        ax1.set_ylabel("Absolute Offset Magnitude (Meters)", fontweight='bold')
        ax1.set_title("Geometric Translation Drift over Time", fontweight='bold')
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.legend(loc='upper right')
        
        # 2. Rotation Error Plot
        ax2.plot(plot_dates, rot_prev, marker='^', linestyle='--', color='tab:blue', linewidth=1.5, markersize=8)
        ax2.plot(plot_dates, rot_next, marker='v', linestyle=':', color='tab:orange', linewidth=1.5, markersize=8)
        ax2.axhline(0, color='black', linewidth=1, alpha=0.5) 
        ax2.set_ylabel("Orbital Yaw Rotation (Degrees)", fontweight='bold')
        ax2.set_title("Rotational Misalignment over Time", fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle='--')

        # 3. Correlation (r) Plot
        ax3.plot(plot_dates, corr_prev, marker='^', linestyle='--', color='tab:blue', linewidth=1.5, markersize=8)
        ax3.plot(plot_dates, corr_next, marker='v', linestyle=':', color='tab:orange', linewidth=1.5, markersize=8)
        ax3.set_ylabel("Valid Pearson Correlation (r)", fontweight='bold')
        ax3.set_title("Structural Confidence Score", fontweight='bold')
        ax3.set_xlabel("Tanager Acquisition Date", fontweight='bold')
        ax3.grid(True, alpha=0.3, linestyle='--')
        
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax3.tick_params(axis='x', rotation=45)
        
        for ax in [ax1, ax2, ax3]:
            for dt in plot_dates:
                ax.axvline(dt, color='purple', alpha=0.2, linewidth=10, zorder=0)

        self.fig_sum.suptitle(f"Temporal Bracketing Registration Analysis\nTarget: {TS_LOCATION['label']} | Euclidean Optimization Window: {SPAN}x{SPAN}px", fontsize=16, fontweight='bold')
        self.fig_sum.tight_layout()

    def _init_ui(self):
        """Initializes the Details-on-Demand interactive viewer."""
        # Switched to 1x2 subplots with a wider figure spacing (wspace=0.4) to create a central gutter
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(15, 7))
        self.fig.canvas.manager.set_window_title("Interactive Co-Registration Analytics")
        self.fig.subplots_adjust(bottom=0.2, top=0.85, wspace=0.4)
        
        # Navigation Buttons (Centered around the new layout)
        ax_prev = self.fig.add_axes([0.41, 0.05, 0.08, 0.05])
        ax_next = self.fig.add_axes([0.51, 0.05, 0.08, 0.05])
        self.btn_prev = Button(ax_prev, '<< Prev')
        self.btn_next = Button(ax_next, 'Next >>')
        self.btn_prev.on_clicked(self._on_prev)
        self.btn_next.on_clicked(self._on_next)
        
        # Status Text
        self.status_text = self.fig.text(0.5, 0.12, "", ha='center', va='center', fontsize=12, fontweight='bold')
        
        # Dynamic Statistics Annotation (Centered in the visual bridge between ax1 and ax2)
        self.stats_annotation = self.fig.text(0.5, 0.5, "", ha='center', va='center', 
                                              fontsize=12, bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    def _on_prev(self, event):
        if self.current_idx > 0:
            self.current_idx -= 1
            self.update_display()

    def _on_next(self, event):
        if self.current_idx < len(self.comparisons) - 1:
            self.current_idx += 1
            self.update_display()

    def update_display(self):
        comp = self.comparisons[self.current_idx]
        
        self.ax1.clear()
        self.ax2.clear()
        
        for ax in [self.ax1, self.ax2]:
            ax.axis('off')
            
        t_date_str = comp['t_dt'].strftime('%Y-%m-%d')
        l_date_str = comp['l_dt'].strftime('%Y-%m-%d')
        
        status = f"Pair {self.current_idx + 1} of {len(self.comparisons)} | Target: {TS_LOCATION['label']} ({SPAN}px)\n"
        status += f"Evaluating Tanager {t_date_str} vs Landsat {l_date_str} [{comp['label']}: {comp['delta_days']:+.1f} days]"
        self.status_text.set_text(status)

        # Extract imagery for current frames
        lum_l, mask_l, rgb_l = extract_luminance_and_mask(self.grp_l, comp['l_idx'], self.y_start, self.y_end, self.x_start, self.x_end)
        lum_t, mask_t, rgb_t = extract_luminance_and_mask(self.grp_t, comp['t_idx'], self.y_start, self.y_end, self.x_start, self.x_end)

        # 1. Show Reference (Landsat)
        self.ax1.imshow(rgb_l)
        self.ax1.axhline(self.rel_center_y, color='red', linestyle='--', lw=1, alpha=0.8)
        self.ax1.axvline(self.rel_center_x, color='red', linestyle='--', lw=1, alpha=0.8)
        self.ax1.plot(self.rel_center_x, self.rel_center_y, 'r+', markersize=15, mew=2, label='Target Focus')
        self.ax1.set_title(f"Reference: Landsat ({l_date_str})")
        self.ax1.legend(loc='lower right')

        if not comp['valid']:
            self.ax2.imshow(rgb_t)
            self.ax2.set_title(f"Moving: Tanager ({t_date_str})\n[CALCULATION ABORTED]")
            self.stats_annotation.set_text(f"DATA INTEGRITY FAILURE\n\n{comp['error_msg']}")
            self.stats_annotation.set_color('red')
            self.fig.canvas.draw_idle()
            return
            
        self.stats_annotation.set_color('black')

        # 2. Show Moving (Tanager) with Correction Vectors
        self.ax2.imshow(rgb_t)
        self.ax2.axhline(self.rel_center_y, color='red', linestyle='--', lw=1, alpha=0.8)
        self.ax2.axvline(self.rel_center_x, color='red', linestyle='--', lw=1, alpha=0.8)
        self.ax2.plot(self.rel_center_x, self.rel_center_y, 'r+', markersize=15, mew=2, label='Nominal Focus')
        
        true_x = self.rel_center_x - comp['opt_dx']
        true_y = self.rel_center_y - comp['opt_dy']
        self.ax2.plot(true_x, true_y, 'b+', markersize=15, mew=2, label='Euclidean Correction')
        
        self.ax2.annotate('', xy=(true_x, true_y), xytext=(self.rel_center_x, self.rel_center_y),
                          arrowprops=dict(arrowstyle='->', color='blue', lw=2))
                     
        self.ax2.set_title(f"Moving: Tanager ({t_date_str})")
        self.ax2.legend(loc='lower right')

        stats_text = (f"Optimized Misregistration:\n\n"
                      f"ΔY: {comp['opt_dy']:+.3f} px ({comp['shift_m_y']:+.1f}m)\n"
                      f"ΔX: {comp['opt_dx']:+.3f} px ({comp['shift_m_x']:+.1f}m)\n"
                      f"Rotation: {comp['opt_theta']:+.3f}°\n"
                      f"Total Offset: {comp['mag']:.1f}m\n"
                      f"Valid Correlation (r): {comp['corr']:.3f}")
                      
        self.stats_annotation.set_text(stats_text)

        self.fig.canvas.draw_idle()

# ==========================================
# 4. EXECUTION ENTRY POINT
# ==========================================

def main():
    print("--- Initializing Interactive Co-Registration Analytics ---")
    try:
        h5_l = h5py.File(landsat_path, 'r')
        h5_t = h5py.File(tanager_path, 'r')
    except Exception as e:
        print(f"Error opening files: {e}")
        return
        
    viewer = CoRegistrationViewer(h5_l, h5_t)
    
    # plt.show() will block execution until BOTH the interactive viewer 
    # and the longitudinal summary plot windows are closed by the user.
    plt.show()
    
    h5_l.close()
    h5_t.close()

if __name__ == "__main__":
    main()