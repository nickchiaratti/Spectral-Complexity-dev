"""
Multi-Sensor Co-Registration Temporal Bracketing Dashboard
Interactive Visual Analytics Interface & Summary Quantifier

Designates a configurable 'Baseline Grid' (e.g., HLSL30) as the geometric anchor.
Iterates across all other sensor timelines, bracketing each epoch with the 
preceding and subsequent baseline acquisitions. 
Dynamically searches for optimal cloud-free windows to execute Fourier 
phase correlation and Euclidean alignment quantification.

Phase Cross Correlation Source: M. Guizar-Sicairos, S. T. Thurman, and 
J. R. Fienup, “Efficient subpixel image registration algorithms,” 
Opt. Lett., vol. 33, no. 2, p. 156, Jan. 2008.
"""

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import matplotlib.dates as mdates
from datetime import datetime, timezone, timedelta
import rasterio.transform
from pyproj import Transformer, CRS
from skimage.registration import phase_cross_correlation
from skimage.transform import EuclideanTransform, warp
from scipy.optimize import minimize
from scipy.stats import pearsonr
import warnings

# ==========================================
# 1. CONFIGURATION
# ==========================================
Location = "Tait"

# Point directly to the finalized ARD Master Cube
ARD_CUBE_PATH = f"C:/satelliteImagery/HLST30/HLST_{Location}_Harmonized_2025_SC_EM-7_Norm-bandCount.h5"

# The Absolute Geometric Anchor (Reference Sensor)
BASELINE_GRID = "TANAGER"

# Target search window size (100x100 pixels)
SPAN = 100 

# ==========================================
# 2. UTILITY FUNCTIONS
# ==========================================
def get_luminance_and_mask(grp, f_idx):
    """
    Extracts the full-frame ortho_visual array, converts to Luminance,
    and applies the strictly pre-calculated ARD common_mask.
    """
    if 'ortho_visual' not in grp or 'common_mask' not in grp:
        raise ValueError(f"CRITICAL ERROR: Required datasets missing in grid {grp.name}")

    raw_vis = grp['ortho_visual'][f_idx, ...]
    
    if raw_vis.shape[0] in [3, 4]:
        bip_vis = np.transpose(raw_vis, (1, 2, 0))
    else:
        bip_vis = raw_vis
        
    rgb = bip_vis[..., :3].astype(np.float32) / 255.0
    luminance = np.dot(rgb, [0.299, 0.587, 0.114])
    
    # 1 = Valid, 0 = Invalid/Masked
    valid_mask = grp['common_mask'][f_idx, ...] == 1
    
    if bip_vis.shape[-1] == 4:
        valid_mask &= (bip_vis[..., 3] > 0)

    return luminance, valid_mask, rgb

def find_optimal_window(mask_ref, mask_mov, lum_ref, span=100):
    """
    Slides a window across the combined validity mask to find a 100x100 region 
    that is >=65% clear. Selects the one with the highest structural variance 
    to perfectly anchor the Fourier Phase Correlation.
    """
    h, w = mask_ref.shape
    half = span // 2
    stride = span // 4 # Overlapping stride for high-resolution search
    
    combined_mask = mask_ref & mask_mov
    
    best_y, best_x = None, None
    best_valid_frac = 0.0
    best_variance = -1.0
    
    for y in range(half, h - half, stride):
        for x in range(half, w - half, stride):
            y0, y1 = y - half, y + half
            x0, x1 = x - half, x + half
            
            local_mask = combined_mask[y0:y1, x0:x1]
            valid_frac = np.sum(local_mask) / (span * span)
            
            if valid_frac >= 0.65:
                local_lum = lum_ref[y0:y1, x0:x1]
                valid_lum = local_lum[local_mask]
                variance = np.var(valid_lum) if len(valid_lum) > 0 else 0
                
                if valid_frac > best_valid_frac or (valid_frac == best_valid_frac and variance > best_variance):
                    best_valid_frac = valid_frac
                    best_variance = variance
                    best_y, best_x = y, x
                    
    return best_y, best_x, best_valid_frac

# ==========================================
# 3. INTERACTIVE VIEWER & DASHBOARD CLASS
# ==========================================
class MultiSensorCoRegistrationViewer:
    def __init__(self, h5_ard):
        self.h5_ard = h5_ard
        self.baseline_name = BASELINE_GRID
        
        if f'/HDFEOS/GRIDS/{self.baseline_name}' not in self.h5_ard:
            raise ValueError(f"CRITICAL ERROR: Configured Baseline Grid '{self.baseline_name}' not found in ARD Cube.")
            
        self.b_grp = self.h5_ard[f'/HDFEOS/GRIDS/{self.baseline_name}/Data Fields']
        
        self.comparisons = []
        self.current_idx = 0
        
        self._setup_metrology()
        self._precalculate_all_transformations()
        
        if len(self.comparisons) == 0:
            print("No valid temporal brackets found for geometric analysis.")
            return
            
        self._plot_longitudinal_summary()
        self._init_ui()
        self.update_display()

    def _setup_metrology(self):
        geo_tf = self.b_grp['ortho_visual'].attrs['GeoTransform']
        crs_wkt = self.b_grp['ortho_visual'].attrs['spatial_ref']
        if isinstance(crs_wkt, bytes): crs_wkt = crs_wkt.decode('utf-8')
        
        self.affine = rasterio.transform.Affine.from_gdal(*geo_tf)
        self.pixel_width = abs(self.affine.a)
        self.pixel_height = abs(self.affine.e)
        
        crs = CRS.from_wkt(crs_wkt)
        self.transformer_to_ll = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

    def _precalculate_all_transformations(self):
        """Iterates all target grids, bracketing them against the Baseline grid."""
        b_times = self.b_grp['surface_reflectance'].attrs['acquisition_time']
        
        target_grids = [g for g in self.h5_ard['/HDFEOS/GRIDS'].keys() if g not in [self.baseline_name, 'HARMONIZED']]
        print(f"Anchoring analysis to {self.baseline_name}. Evaluating Target Grids: {target_grids}")

        for t_grid in target_grids:
            print(f"\n{'='*40}\nProcessing Base: {self.baseline_name} <-> Target: {t_grid}\n{'='*40}")
            t_grp = self.h5_ard[f'/HDFEOS/GRIDS/{t_grid}/Data Fields']
            t_times = t_grp['surface_reflectance'].attrs['acquisition_time']
            
            for t_idx, t_time in enumerate(t_times):
                t_dt = datetime.fromtimestamp(t_time, tz=timezone.utc)
                
                b_before_indices = np.where(b_times < t_time)[0]
                b_after_indices = np.where(b_times > t_time)[0]
                
                b_prev_idx = b_before_indices[-1] if len(b_before_indices) > 0 else None
                b_next_idx = b_after_indices[0] if len(b_after_indices) > 0 else None
                
                if b_prev_idx is not None:
                    self._compute_and_store(t_grp, t_grid, b_prev_idx, t_idx, b_times[b_prev_idx], t_dt, f"{self.baseline_name}_prev (Before)")
                
                if b_next_idx is not None:
                    self._compute_and_store(t_grp, t_grid, b_next_idx, t_idx, b_times[b_next_idx], t_dt, f"{self.baseline_name}_next (After)")

    def _compute_and_store(self, t_grp, t_grid, b_idx, t_idx, b_time, t_dt, bracket_label):
        b_dt = datetime.fromtimestamp(b_time, tz=timezone.utc)
        delta_days = (t_dt - b_dt).total_seconds() / (60*60*24)
        
        print(f"  -> {t_grid} [{t_dt.strftime('%Y-%m-%d')}] vs Anchor [{b_dt.strftime('%Y-%m-%d')}] ({bracket_label})...")
        
        lum_b, mask_b, rgb_b = get_luminance_and_mask(self.b_grp, b_idx)
        lum_t, mask_t, rgb_t = get_luminance_and_mask(t_grp, t_idx)
        
        t_y, t_x, valid_frac = find_optimal_window(mask_b, mask_t, lum_b, span=SPAN)
        
        if t_y is None:
            print("     [FAILED] No cloud-free 100x100 window overlap found.")
            self.comparisons.append({
                'valid': False, 't_grid': t_grid, 't_idx': t_idx, 'b_idx': b_idx, 
                't_dt': t_dt, 'b_dt': b_dt, 'label': bracket_label, 'delta_days': delta_days, 
                'error_msg': 'Insufficient Cloud-Free Overlap',
                'lat': 0, 'lon': 0, 't_x': 0, 't_y': 0
            })
            return
            
        proj_x, proj_y = self.affine * (t_x + 0.5, t_y + 0.5)
        lon, lat = self.transformer_to_ll.transform(proj_x, proj_y)

        half = SPAN // 2
        x_start, x_end = t_x - half, t_x + half
        y_start, y_end = t_y - half, t_y + half
        
        rel_center_y = t_y - y_start
        rel_center_x = t_x - x_start

        local_lum_b = lum_b[y_start:y_end, x_start:x_end]
        local_mask_b = mask_b[y_start:y_end, x_start:x_end]
        local_lum_t = lum_t[y_start:y_end, x_start:x_end]
        local_mask_t = mask_t[y_start:y_end, x_start:x_end]

        shift_vector, error, diffphase = phase_cross_correlation(
            reference_image=local_lum_b, moving_image=local_lum_t, 
            reference_mask=local_mask_b, moving_mask=local_mask_t, upsample_factor=100
        )
        init_dy, init_dx = shift_vector

        def objective_function(params):
            dy, dx, theta_deg = params
            t1 = EuclideanTransform(translation=(-rel_center_x, -rel_center_y))
            t2 = EuclideanTransform(rotation=np.deg2rad(theta_deg))
            t3 = EuclideanTransform(translation=(rel_center_x + dx, rel_center_y + dy))
            tform = t1 + t2 + t3
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                warped_lum = warp(local_lum_t, tform.inverse, order=3, mode='constant', cval=np.nan)
                warped_mask = warp(local_mask_t.astype(float), tform.inverse, order=0, mode='constant', cval=0.0).astype(bool)
            
            current_overlap = local_mask_b & warped_mask & ~np.isnan(warped_lum)
            if np.sum(current_overlap) < (0.15 * local_mask_b.size):
                return 1.0 
            r, _ = pearsonr(local_lum_b[current_overlap], warped_lum[current_overlap])
            return -r 

        initial_guess = [init_dy, init_dx, 0.0]
        bounds = [(init_dy - 2.0, init_dy + 2.0), (init_dx - 2.0, init_dx + 2.0), (-2.0, 2.0)]
        
        result = minimize(objective_function, initial_guess, method='Nelder-Mead', bounds=bounds, options={'xatol': 1e-4, 'fatol': 1e-4})
        opt_dy, opt_dx, opt_theta = result.x
        
        shift_meters_x = opt_dx * self.pixel_width
        shift_meters_y = opt_dy * self.pixel_height
        magnitude_meters = np.sqrt(shift_meters_x**2 + shift_meters_y**2)

        if magnitude_meters > 250.0:
            print(f"     [FAILED] Offset {magnitude_meters:.1f}m > 250m physical limit. Tracking failure.")
            self.comparisons.append({
                'valid': False, 't_grid': t_grid, 't_idx': t_idx, 'b_idx': b_idx, 
                't_dt': t_dt, 'b_dt': b_dt, 'label': bracket_label, 'delta_days': delta_days, 
                'error_msg': f'Cloud lock failure (Calculated drift: {magnitude_meters:.1f}m)',
                'lat': 0, 'lon': 0, 't_x': 0, 't_y': 0
            })
            return

        print(f"     [SUCCESS] Offset={magnitude_meters:.2f}m, r={-result.fun:.3f} (Valid Overlap: {valid_frac*100:.0f}%)")

        self.comparisons.append({
            'valid': True, 't_grid': t_grid, 't_idx': t_idx, 'b_idx': b_idx, 
            't_dt': t_dt, 'b_dt': b_dt, 'label': bracket_label, 'delta_days': delta_days,
            't_y': t_y, 't_x': t_x, 'lat': lat, 'lon': lon, 'valid_frac': valid_frac,
            'y_start': y_start, 'y_end': y_end, 'x_start': x_start, 'x_end': x_end,
            'rel_center_y': rel_center_y, 'rel_center_x': rel_center_x,
            'opt_dy': opt_dy, 'opt_dx': opt_dx, 'opt_theta': opt_theta, 
            'corr': -result.fun, 'mag': magnitude_meters,
            'shift_m_x': shift_meters_x, 'shift_m_y': shift_meters_y
        })

    def _plot_longitudinal_summary(self):
        """Builds a unified 3-panel statistical overview of temporal registration drift across all target sensors."""
        valid_comps = [c for c in self.comparisons if c['valid']]
        if not valid_comps: return
        
        target_grids = list(set([c['t_grid'] for c in valid_comps]))
        
        self.fig_sum, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
        self.fig_sum.canvas.manager.set_window_title("Multi-Sensor Baseline Stability")
        
        # 1. Establish Vertical Epoch Bands aligned to the Anchor Grid
        unique_b_dates = sorted(list(set(c['b_dt'] for c in valid_comps)))
        for ax in [ax1, ax2, ax3]:
            for b_date in unique_b_dates:
                # Draws a 4-day wide visual band centered on the Anchor acquisition
                ax.axvspan(b_date - timedelta(days=2), b_date + timedelta(days=2), 
                           color='purple', alpha=0.2, zorder=0, lw=0)
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(target_grids)))
        
        for i, t_grid in enumerate(target_grids):
            grid_comps = [c for c in valid_comps if c['t_grid'] == t_grid]
            
            # 2. Separate into "Before" and "After" temporal brackets
            prev_comps = sorted([c for c in grid_comps if 'prev' in c['label']], key=lambda x: x['b_dt'])
            next_comps = sorted([c for c in grid_comps if 'next' in c['label']], key=lambda x: x['b_dt'])
            
            # 3. Plot Connected Bracketing Vectors vs the Anchor X-Axis
            if prev_comps:
                dates = [c['b_dt'] for c in prev_comps]
                mags = [c['mag'] for c in prev_comps]
                rots = [abs(c['opt_theta']) for c in prev_comps]
                corrs = [c['corr'] for c in prev_comps]
                
                ax1.plot(dates, mags, color=colors[i], marker='^', markersize=10, linestyle='--', label=f'{t_grid} Frame Before ($L_{{-1}}$)', zorder=3)
                ax2.plot(dates, rots, color=colors[i], marker='^', markersize=10, linestyle='--', zorder=3)
                ax3.plot(dates, corrs, color=colors[i], marker='^', markersize=10, linestyle='--', zorder=3)
                
            if next_comps:
                dates = [c['b_dt'] for c in next_comps]
                mags = [c['mag'] for c in next_comps]
                rots = [abs(c['opt_theta']) for c in next_comps]
                corrs = [c['corr'] for c in next_comps]
                
                # Use a dotted line and downward triangle for the trailing bracket
                ax1.plot(dates, mags, color=colors[i], marker='v', markersize=10, linestyle=':', label=f'{t_grid} Frame After ($L_{{+1}}$)', zorder=3)
                ax2.plot(dates, rots, color=colors[i], marker='v', markersize=10, linestyle=':', zorder=3)
                ax3.plot(dates, corrs, color=colors[i], marker='v', markersize=10, linestyle=':', zorder=3)
            
            print(f"\n--- {t_grid} vs {self.baseline_name} Baseline Averages ---")
            mags_all = [c['mag'] for c in grid_comps]
            rots_all = [abs(c['opt_theta']) for c in grid_comps]
            corrs_all = [c['corr'] for c in grid_comps]
            print(f"Valid Evaluations: {len(mags_all)}")
            print(f"Mean Translation:  {np.mean(mags_all):.2f}m")
            print(f"Mean Abs Rotation: {np.mean(rots_all):.3f}°")
            print(f"Mean Correlation:  {np.mean(corrs_all):.3f}")

        # Labeling and Formatting
        ax1.set_ylabel("Offset Magnitude (Meters)", fontsize=10)
        ax1.set_title("Geometric Translation Drift over Time", fontsize=10)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.legend(loc='upper left', fontsize=10)
        
        ax2.axhline(0, color='black', linewidth=1, alpha=0.5) 
        ax2.set_ylabel("Rotation (Degrees)", fontsize=10)
        ax2.set_title("Rotational Misalignment over Time", fontsize=10)
        ax2.grid(True, alpha=0.3, linestyle='--')

        ax3.set_ylabel("Pearson Correlation (r)", fontsize=10)
        ax3.set_title("Correlation Coefficient over Time", fontsize=10)
        ax3.set_xlabel(f"{self.baseline_name} Acquisition Date", fontsize=10)
        ax3.grid(True, alpha=0.3, linestyle='--')
        
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax3.tick_params(axis='x', rotation=45)
        
        self.fig_sum.suptitle(f"Temporal Bracketing Registration Analysis | Window: {SPAN}x{SPAN}px")
        self.fig_sum.tight_layout()

    def _init_ui(self):
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(15, 7))
        self.fig.canvas.manager.set_window_title("Interactive Multi-Sensor Analytics")
        self.fig.subplots_adjust(bottom=0.22, top=0.85, wspace=0.4)
        
        ax_prev = self.fig.add_axes([0.41, 0.05, 0.08, 0.05])
        ax_next = self.fig.add_axes([0.51, 0.05, 0.08, 0.05])
        self.btn_prev = Button(ax_prev, '<< Prev')
        self.btn_next = Button(ax_next, 'Next >>')
        self.btn_prev.on_clicked(self._on_prev)
        self.btn_next.on_clicked(self._on_next)
        
        self.status_text = self.fig.text(0.5, 0.15, "", ha='center', va='center', fontsize=10)
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
        b_date_str = comp['b_dt'].strftime('%Y-%m-%d')
        
        status = f"Pair {self.current_idx + 1} of {len(self.comparisons)} | Target: Lat {comp.get('lat', 0):.5f}, Lon {comp.get('lon', 0):.5f}\n"
        status += f"{comp['t_grid']} {t_date_str} | Anchor: {self.baseline_name} {b_date_str} [{comp['label']}: {comp['delta_days']:+.1f} days]"
        self.status_text.set_text(status)

        if comp['valid']:
            t_grp = self.h5_ard[f'/HDFEOS/GRIDS/{comp["t_grid"]}/Data Fields']
            _, _, rgb_b = get_luminance_and_mask(self.b_grp, comp['b_idx'])
            _, _, rgb_t = get_luminance_and_mask(t_grp, comp['t_idx'])
            
            local_rgb_b = rgb_b[comp['y_start']:comp['y_end'], comp['x_start']:comp['x_end']]
            local_rgb_t = rgb_t[comp['y_start']:comp['y_end'], comp['x_start']:comp['x_end']]

            # Show Baseline (Anchor)
            self.ax1.imshow(local_rgb_b)
            self.ax1.axhline(comp['rel_center_y'], color='red', linestyle='--', lw=1, alpha=0.8)
            self.ax1.axvline(comp['rel_center_x'], color='red', linestyle='--', lw=1, alpha=0.8)
            self.ax1.plot(comp['rel_center_x'], comp['rel_center_y'], 'r+', markersize=15, mew=2, label='Target Focus')
            self.ax1.set_title(f"Anchor: {self.baseline_name} ({b_date_str})")
            self.ax1.legend(loc='lower right')

            self.stats_annotation.set_color('black')

            # Show Moving (Target Sensor) with Vectors
            self.ax2.imshow(local_rgb_t)
            self.ax2.axhline(comp['rel_center_y'], color='red', linestyle='--', lw=1, alpha=0.8)
            self.ax2.axvline(comp['rel_center_x'], color='red', linestyle='--', lw=1, alpha=0.8)
            self.ax2.plot(comp['rel_center_x'], comp['rel_center_y'], 'r+', markersize=15, mew=2, label='Nominal Focus')
            
            true_x = comp['rel_center_x'] - comp['opt_dx']
            true_y = comp['rel_center_y'] - comp['opt_dy']
            self.ax2.plot(true_x, true_y, 'b+', markersize=15, mew=2, label='Euclidean Correction')
            
            self.ax2.annotate('', xy=(true_x, true_y), xytext=(comp['rel_center_x'], comp['rel_center_y']),
                              arrowprops=dict(arrowstyle='->', color='blue', lw=2))
                         
            self.ax2.set_title(f"Evaluated: {comp['t_grid']} ({t_date_str})")
            self.ax2.legend(loc='lower right')

            stats_text = (f"Calculated Misregistration:\n\n"
                          f"ΔY: {comp['opt_dy']:+.3f} px ({comp['shift_m_y']:+.1f}m)\n"
                          f"ΔX: {comp['opt_dx']:+.3f} px ({comp['shift_m_x']:+.1f}m)\n"
                          f"Rotation: {comp['opt_theta']:+.3f}°\n"
                          f"Total Offset: {comp['mag']:.1f}m\n"
                          f"Valid Correlation (r): {comp['corr']:.3f}\n"
                          f"Cloud-Free Validation: {comp['valid_frac']*100:.1f}%")
                          
            self.stats_annotation.set_text(stats_text)
            
        else:
            self.ax1.set_title(f"Anchor: {self.baseline_name} ({b_date_str})\n[CALCULATION ABORTED]")
            self.ax2.set_title(f"Evaluated: {comp['t_grid']} ({t_date_str})\n[CALCULATION ABORTED]")
            self.stats_annotation.set_text(f"DATA INTEGRITY FAILURE\n\n{comp['error_msg']}")
            self.stats_annotation.set_color('red')

        self.fig.canvas.draw_idle()

# ==========================================
# 4. EXECUTION ENTRY POINT
# ==========================================
def main():
    print("--- Initializing Baseline-Anchored Co-Registration Analytics ---")
    try:
        h5_ard = h5py.File(ARD_CUBE_PATH, 'r')
    except Exception as e:
        print(f"CRITICAL ERROR: Could not open ARD Cube: {e}")
        return
        
    viewer = MultiSensorCoRegistrationViewer(h5_ard)
    plt.show()
    
    h5_ard.close()

if __name__ == "__main__":
    main()