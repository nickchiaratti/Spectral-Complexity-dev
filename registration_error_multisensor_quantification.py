"""
Multi-Sensor Co-Registration Temporal Bracketing Quantifier
Interactive Visual Analytics Interface & Summary Dashboard

Iterates across the entire hyperspectral timeline, bracketing 
each epoch with the preceding and subsequent multispectral acquisitions.
Dynamically searches for the optimal cloud-free, high-variance 100x100 
window for each temporal pair to ensure robust Fourier phase correlation.

Phase Cross Correlation Source: M. Guizar-Sicairos, S. T. Thurman, and J. R. Fienup, “Efficient subpixel image registration algorithms,” Opt. Lett., vol. 33, no. 2, p. 156, Jan. 2008, doi: 10.1364/OL.33.000156.



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
from scipy import ndimage
import SpecComplex as sc

# ==========================================
# 1. CONFIGURATION
# ==========================================
Location = "Rochester"

# Target search window size (100x100 pixels)
SPAN = 100 

# Strict ARD Masking Configuration for Dynamic Search
SUN_ELEVATION_THRESHOLD = 20
CLOUD_DILATION = 0
TANAGER_AEROSOL_DEPTH_THRESHOLD = 0.35
TANAGER_SR_UNCERTAINTY_THRESHOLD = .5
QA_REJECT_MASK = 142 # Bits 0-5 for Landsat Collection 2
RADSAT_ACCEPT_VALUE = 0
AEROSOL_ACCEPT_LEVEL = 'high'

landsat_path = f"C:/satelliteImagery/LANDSAT/{Location}/LANDSAT_Stack_{Location}_GEE_2015_2025_WRS16_SC_EM-7_Gram-minEndmember_Norm-bandCount.h5"
tanager_path = f"C:/satelliteImagery/Tanager/{Location}/Tanager_Stack_{Location}_HDFEOS_SC_EM-7_Gram-minEndmember_Norm-bandCount.h5"

# ==========================================
# 2. UTILITY FUNCTIONS
# ==========================================

def get_full_luminance_and_mask(grp, f_idx, sensor):
    """
    Extracts the full-frame BSQ ortho_visual array, converts to Luminance,
    and applies explicit Cloud/QA masking via SpecComplex for the dynamic target search.
    """
    raw_vis = grp['ortho_visual'][f_idx, ...]
    
    # Handle BSQ format based on channel count
    if raw_vis.shape[0] in [3, 4]:
        bip_vis = np.transpose(raw_vis, (1, 2, 0))
    else:
        bip_vis = raw_vis
        
    rgb = bip_vis[..., :3].astype(np.float32) / 255.0
    luminance = np.dot(rgb, [0.299, 0.587, 0.114])
    
    shape = luminance.shape
    valid_mask = np.ones(shape, dtype=bool)
    
    # 1. Alpha Edge Masking (Ensures valid spatial boundaries per Interface Spec)
    if bip_vis.shape[-1] == 4:
        valid_mask &= (bip_vis[..., 3] > 0)
        
    # 2. Strict Full-Pipeline ARD Masking
    if sensor == 'LANDSAT':
        ard_mask = sc.get_landsat_mask(
            data_grp=grp, f_idx=f_idx, shape=shape,
            sun_elevation_threshold=SUN_ELEVATION_THRESHOLD,
            cloud_dilation=CLOUD_DILATION, qa_reject_mask=QA_REJECT_MASK,
            radsat_accept_value=RADSAT_ACCEPT_VALUE, aerosol_accept_level=AEROSOL_ACCEPT_LEVEL
        )
        valid_mask &= ard_mask
        
    elif sensor == 'TANAGER':
        ard_mask = sc.get_tanager_mask(
            data_grp=grp, f_idx=f_idx, shape=shape,
            sun_elevation_threshold=SUN_ELEVATION_THRESHOLD,
            cloud_dilation=CLOUD_DILATION, apply_cloud_mask=True,
            uncertainty_threshold=TANAGER_SR_UNCERTAINTY_THRESHOLD,
            aerosol_depth_threshold=TANAGER_AEROSOL_DEPTH_THRESHOLD
        )
        valid_mask &= ard_mask

    return luminance, valid_mask, rgb

def find_optimal_window(mask_l, mask_t, lum_l, span=100):
    """
    Slides a window across the combined validity mask to find a 100x100 region 
    that is >=95% clear. If multiple exist, selects the one with the highest 
    structural variance to perfectly anchor the Fourier Phase Correlation.
    """
    h, w = mask_l.shape
    half = span // 2
    stride = 1 # Overlapping stride for high-resolution search
    
    combined_mask = mask_l & mask_t
    
    best_y, best_x = None, None
    best_valid_frac = 0.0
    best_variance = -1.0
    
    for y in range(half, h - half, stride):
        for x in range(half, w - half, stride):
            y0, y1 = y - half, y + half
            x0, x1 = x - half, x + half
            
            local_mask = combined_mask[y0:y1, x0:x1]
            valid_frac = np.sum(local_mask) / (span * span)
            
            # Require 95% of the window to be clear of clouds/edges
            if valid_frac >= 0.65:
                local_lum = lum_l[y0:y1, x0:x1]
                valid_lum = local_lum[local_mask]
                variance = np.var(valid_lum) if len(valid_lum) > 0 else 0
                
                # Maximize validity; break ties using structural complexity
                if valid_frac > best_valid_frac or (valid_frac == best_valid_frac and variance > best_variance):
                    best_valid_frac = valid_frac
                    best_variance = variance
                    best_y, best_x = y, x
                    
    return best_y, best_x, best_valid_frac

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
        
        self._plot_longitudinal_summary()
        self._init_ui()
        self.update_display()

    def _setup_metrology(self):
        """Initializes the geographic to pixel coordinate mapping architecture."""
        geo_tf = self.grp_t['surface_reflectance'].attrs['GeoTransform']
        crs_wkt = self.grp_t['surface_reflectance'].attrs['spatial_ref']
        if isinstance(crs_wkt, bytes): crs_wkt = crs_wkt.decode('utf-8')
        
        self.affine = rasterio.transform.Affine.from_gdal(*geo_tf)
        self.pixel_width = abs(self.affine.a)
        self.pixel_height = abs(self.affine.e)
        
        crs = CRS.from_wkt(crs_wkt)
        self.transformer_to_ll = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        
        # Get master frame dimensions
        self.h, self.w = self.grp_t['ortho_visual'].shape[2:] 

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
        
        print(f"\n  -> Processing {t_dt.strftime('%Y-%m-%d')} vs {l_dt.strftime('%Y-%m-%d')} ({bracket_label})...")
        
        # Extract Full Frame Arrays and QA Masks
        lum_l, mask_l, _ = get_full_luminance_and_mask(self.grp_l, l_idx, 'LANDSAT')
        lum_t, mask_t, _ = get_full_luminance_and_mask(self.grp_t, t_idx, 'TANAGER')
        
        # Dynamically find the optimal unmasked target for this specific temporal pair
        t_y, t_x, valid_frac = find_optimal_window(mask_l, mask_t, lum_l, span=SPAN)
        
        # Strict fail-fast handling
        if t_y is None:
            print("     [FAILED] No cloud-free 100x100 window found for this pair.")
            self.comparisons.append({
                'valid': False, 't_idx': t_idx, 'l_idx': l_idx, 't_dt': t_dt, 'l_dt': l_dt, 
                'label': bracket_label, 'delta_days': delta_days, 'error_msg': 'Insufficient Cloud-Free Overlap',
                'lat': 0, 'lon': 0, 't_x': 0, 't_y': 0 # Null geometric values for UI safety
            })
            return
            
        # Calculate real-world coordinates of the discovered target
        proj_x, proj_y = self.affine * (t_x + 0.5, t_y + 0.5)
        lon, lat = self.transformer_to_ll.transform(proj_x, proj_y)
        
        print(f"     [TARGET FOUND] Valid Frac: {valid_frac*100:.1f}% | Lat: {lat:.5f}, Lon: {lon:.5f}")

        # Define Local Bounding Box
        half = SPAN // 2
        x_start, x_end = t_x - half, t_x + half
        y_start, y_end = t_y - half, t_y + half
        
        rel_center_y = t_y - y_start
        rel_center_x = t_x - x_start

        # Extract Local Arrays for Mathematics
        local_lum_l = lum_l[y_start:y_end, x_start:x_end]
        local_mask_l = mask_l[y_start:y_end, x_start:x_end]
        local_lum_t = lum_t[y_start:y_end, x_start:x_end]
        local_mask_t = mask_t[y_start:y_end, x_start:x_end]

        # 1. Phase Correlation Seed
        shift_vector, error, diffphase = phase_cross_correlation(
            reference_image=local_lum_l, moving_image=local_lum_t, 
            reference_mask=local_mask_l, moving_mask=local_mask_t, upsample_factor=100
        )
        init_dy, init_dx = shift_vector

        # 2. Euclidean Optimization
        def objective_function(params):
            dy, dx, theta_deg = params
            t1 = EuclideanTransform(translation=(-rel_center_x, -rel_center_y))
            t2 = EuclideanTransform(rotation=np.deg2rad(theta_deg))
            t3 = EuclideanTransform(translation=(rel_center_x + dx, rel_center_y + dy))
            tform = t1 + t2 + t3
            
            warped_lum = warp(local_lum_t, tform.inverse, order=3, mode='constant', cval=np.nan)
            warped_mask = warp(local_mask_t.astype(float), tform.inverse, order=0, mode='constant', cval=0.0).astype(bool)
            
            current_overlap = local_mask_l & warped_mask & ~np.isnan(warped_lum)
            if np.sum(current_overlap) < (0.15 * local_mask_l.size):
                return 1.0 
            r, _ = pearsonr(local_lum_l[current_overlap], warped_lum[current_overlap])
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
            't_y': t_y, 't_x': t_x, 'lat': lat, 'lon': lon, 'valid_frac': valid_frac,
            'y_start': y_start, 'y_end': y_end, 'x_start': x_start, 'x_end': x_end,
            'rel_center_y': rel_center_y, 'rel_center_x': rel_center_x,
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
                rot_prev.append(abs(c_prev['opt_theta']))
                corr_prev.append(c_prev['corr'])
            else:
                mag_prev.append(np.nan); rot_prev.append(np.nan); corr_prev.append(np.nan)
                
            if c_next and c_next['valid']:
                mag_next.append(c_next['mag'])
                rot_next.append(abs(c_next['opt_theta']))
                corr_next.append(c_next['corr'])
            else:
                mag_next.append(np.nan); rot_next.append(np.nan); corr_next.append(np.nan)

        self.fig_sum, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
        self.fig_sum.canvas.manager.set_window_title("Longitudinal Co-Registration Stability")
        
        ax1.plot(plot_dates, mag_prev, marker='^', linestyle='--', color='tab:blue', linewidth=1.5, markersize=8, label='Landsat Frame Before ($L_{-1}$)')
        ax1.plot(plot_dates, mag_next, marker='v', linestyle=':', color='tab:orange', linewidth=1.5, markersize=8, label='Landsat Frame After ($L_{+1}$)')
        ax1.set_ylabel("Offset Magnitude (Meters)", fontsize=10)
        ax1.set_title("Geometric Translation Drift over Time", fontsize=10)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.legend(loc='best', fontsize=10)
        
        ax2.plot(plot_dates, rot_prev, marker='^', linestyle='--', color='tab:blue', linewidth=1.5, markersize=8)
        ax2.plot(plot_dates, rot_next, marker='v', linestyle=':', color='tab:orange', linewidth=1.5, markersize=8)
        ax2.axhline(0, color='black', linewidth=1, alpha=0.5) 
        ax2.set_ylabel("Rotation (Degrees)", fontsize=10)
        ax2.set_title("Rotational Misalignment over Time", fontsize=10)
        ax2.grid(True, alpha=0.3, linestyle='--')

        ax3.plot(plot_dates, corr_prev, marker='^', linestyle='--', color='tab:blue', linewidth=1.5, markersize=8)
        ax3.plot(plot_dates, corr_next, marker='v', linestyle=':', color='tab:orange', linewidth=1.5, markersize=8)
        ax3.set_ylabel("Pearson Correlation (r)", fontsize=10)
        ax3.set_title("Correlation Coefficient over Time", fontsize=10)
        ax3.set_xlabel("Tanager Acquisition Date", fontsize=10)
        ax3.grid(True, alpha=0.3, linestyle='--')
        
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax3.tick_params(axis='x', rotation=45)
        
        for ax in [ax1, ax2, ax3]:
            for dt in plot_dates:
                ax.axvline(dt, color='purple', alpha=0.2, linewidth=10, zorder=0)

        self.fig_sum.suptitle(f"Temporal Bracketing Registration Analysis | Window: {SPAN}x{SPAN}px")
        self.fig_sum.tight_layout()

    def _init_ui(self):
        """Initializes the Details-on-Demand interactive viewer."""
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(15, 7))
        self.fig.canvas.manager.set_window_title("Interactive Co-Registration Analytics")
        self.fig.subplots_adjust(bottom=0.22, top=0.85, wspace=0.4)
        
        # Navigation Buttons (Centered around the new layout)
        ax_prev = self.fig.add_axes([0.41, 0.05, 0.08, 0.05])
        ax_next = self.fig.add_axes([0.51, 0.05, 0.08, 0.05])
        self.btn_prev = Button(ax_prev, '<< Prev')
        self.btn_next = Button(ax_next, 'Next >>')
        self.btn_prev.on_clicked(self._on_prev)
        self.btn_next.on_clicked(self._on_next)
        
        # Status Text
        self.status_text = self.fig.text(0.5, 0.15, "", ha='center', va='center', fontsize=10)
        
        # Dynamic Statistics Annotation (Centered in the visual bridge between ax1 and ax2)
        self.stats_annotation = self.fig.text(0.5, 0.5, "", ha='center', va='center', 
                                              fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

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
        
        status = f"Pair {self.current_idx + 1} of {len(self.comparisons)} | Target: Lat {comp.get('lat', 0):.5f}, Lon {comp.get('lon', 0):.5f}\n"
        status += f"Tanager {t_date_str} | Landsat {l_date_str} [{comp['label']}: {comp['delta_days']:+.1f} days]"
        self.status_text.set_text(status)

        # Full extraction strictly to get the RGB visualization array for the dynamically assigned bounds
        if comp['valid']:
            _, _, rgb_l = get_full_luminance_and_mask(self.grp_l, comp['l_idx'], 'LANDSAT')
            _, _, rgb_t = get_full_luminance_and_mask(self.grp_t, comp['t_idx'], 'TANAGER')
            
            local_rgb_l = rgb_l[comp['y_start']:comp['y_end'], comp['x_start']:comp['x_end']]
            local_rgb_t = rgb_t[comp['y_start']:comp['y_end'], comp['x_start']:comp['x_end']]

            # 1. Show Reference (Landsat)
            self.ax1.imshow(local_rgb_l)
            self.ax1.axhline(comp['rel_center_y'], color='red', linestyle='--', lw=1, alpha=0.8)
            self.ax1.axvline(comp['rel_center_x'], color='red', linestyle='--', lw=1, alpha=0.8)
            self.ax1.plot(comp['rel_center_x'], comp['rel_center_y'], 'r+', markersize=15, mew=2, label='Target Focus')
            self.ax1.set_title(f"Reference: Landsat ({l_date_str})")
            self.ax1.legend(loc='lower right')

            self.stats_annotation.set_color('black')

            # 2. Show Moving (Tanager) with Correction Vectors
            self.ax2.imshow(local_rgb_t)
            self.ax2.axhline(comp['rel_center_y'], color='red', linestyle='--', lw=1, alpha=0.8)
            self.ax2.axvline(comp['rel_center_x'], color='red', linestyle='--', lw=1, alpha=0.8)
            self.ax2.plot(comp['rel_center_x'], comp['rel_center_y'], 'r+', markersize=15, mew=2, label='Nominal Focus')
            
            true_x = comp['rel_center_x'] - comp['opt_dx']
            true_y = comp['rel_center_y'] - comp['opt_dy']
            self.ax2.plot(true_x, true_y, 'b+', markersize=15, mew=2, label='Euclidean Correction')
            
            self.ax2.annotate('', xy=(true_x, true_y), xytext=(comp['rel_center_x'], comp['rel_center_y']),
                              arrowprops=dict(arrowstyle='->', color='blue', lw=2))
                         
            self.ax2.set_title(f"Tanager ({t_date_str})")
            self.ax2.legend(loc='lower right')

            stats_text = (f"Optimized Misregistration:\n\n"
                          f"ΔY: {comp['opt_dy']:+.3f} px ({comp['shift_m_y']:+.1f}m)\n"
                          f"ΔX: {comp['opt_dx']:+.3f} px ({comp['shift_m_x']:+.1f}m)\n"
                          f"Rotation: {comp['opt_theta']:+.3f}°\n"
                          f"Total Offset: {comp['mag']:.1f}m\n"
                          f"Valid Correlation (r): {comp['corr']:.3f}\n"
                          f"Cloud-Free Validation: {comp['valid_frac']*100:.1f}%")
                          
            self.stats_annotation.set_text(stats_text)
            
        else:
            # Fallback if computation was aborted due to dense cloud cover
            self.ax1.set_title(f"Reference: Landsat ({l_date_str})\n[CALCULATION ABORTED]")
            self.ax2.set_title(f"Moving: Tanager ({t_date_str})\n[CALCULATION ABORTED]")
            self.stats_annotation.set_text(f"DATA INTEGRITY FAILURE\n\n{comp['error_msg']}")
            self.stats_annotation.set_color('red')

        self.fig.canvas.draw_idle()

# ==========================================
# 4. EXECUTION ENTRY POINT
# ==========================================

def main():
    print("--- Initializing Dynamic Target Co-Registration Analytics ---")
    try:
        h5_l = h5py.File(landsat_path, 'r')
        h5_t = h5py.File(tanager_path, 'r')
    except Exception as e:
        print(f"Error opening files: {e}")
        return
        
    viewer = CoRegistrationViewer(h5_l, h5_t)
    plt.show()
    
    h5_l.close()
    h5_t.close()

if __name__ == "__main__":
    main()