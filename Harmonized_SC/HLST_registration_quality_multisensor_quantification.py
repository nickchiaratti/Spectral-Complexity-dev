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
from matplotlib.widgets import Button, TextBox
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
Location = "Rochesterv2"

# Point directly to the finalized ARD Master Cube
ARD_CUBE_PATH = f"C:/satelliteImagery/HLST30/HLST_{Location}_Harmonized_SC_EM-7_Norm-bandCount.h5"

# The Absolute Geometric Anchor (Reference Sensor)
BASELINE_GRID = "HARMONIZED"

# Target search window size (100x100 pixels)
SPAN = 150 

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
    valid_mask = grp['common_mask'][f_idx, ...] != 1
    
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
        """Iterates sequentially frame-to-frame through the HARMONIZED grid."""
        times = self.b_grp['ortho_visual'].attrs['acquisition_time']
        num_frames = len(times)
        
        print(f"Anchoring analysis sequentially (Frame to Frame) within {self.baseline_name}.")

        print(f"\n{'='*40}\nProcessing Frame-to-Frame: {self.baseline_name}\n{'='*40}")
        
        for i in range(num_frames - 1):
            b_idx = i
            t_idx = i + 1
            
            b_time = times[b_idx]
            t_time = times[t_idx]
            
            t_dt = datetime.fromtimestamp(t_time, tz=timezone.utc)
            
            self._compute_and_store(self.b_grp, self.baseline_name, b_idx, t_idx, b_time, t_dt, f"Frame {b_idx} -> {t_idx}")

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
        """Builds a unified 3-panel statistical overview of frame-to-frame registration drift."""
        valid_comps = [c for c in self.comparisons if c['valid']]
        if not valid_comps: return
        
        self.fig_sum, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
        self.fig_sum.canvas.manager.set_window_title("Sequential Frame-to-Frame Stability")
        
        frame_indices = [c['t_idx'] for c in valid_comps]
        
        mags = [c['mag'] for c in valid_comps]
        rots = [abs(c['opt_theta']) for c in valid_comps]
        corrs = [c['corr'] for c in valid_comps]
        
        ax1.plot(frame_indices, mags, color='blue', marker='o', linestyle='-', zorder=3)
        ax2.plot(frame_indices, rots, color='red', marker='o', linestyle='-', zorder=3)
        ax3.plot(frame_indices, corrs, color='green', marker='o', linestyle='-', zorder=3)
        
        print(f"\n--- {self.baseline_name} Frame-to-Frame Averages ---")
        print(f"Valid Evaluations: {len(mags)}")
        print(f"Mean Translation:  {np.mean(mags):.2f}m")
        print(f"Mean Abs Rotation: {np.mean(rots):.3f}°")
        print(f"Mean Correlation:  {np.mean(corrs):.3f}")

        # Labeling and Formatting
        ax1.set_ylabel("Offset Magnitude (Meters)", fontsize=10)
        ax1.set_title("Frame-to-Frame Geometric Translation Drift", fontsize=10)
        ax1.grid(True, alpha=0.3, linestyle='--')
        
        ax2.axhline(0, color='black', linewidth=1, alpha=0.5) 
        ax2.set_ylabel("Rotation (Degrees)", fontsize=10)
        ax2.set_title("Frame-to-Frame Rotational Misalignment", fontsize=10)
        ax2.grid(True, alpha=0.3, linestyle='--')

        ax3.set_ylabel("Pearson Correlation (r)", fontsize=10)
        ax3.set_title("Frame-to-Frame Correlation Coefficient", fontsize=10)
        ax3.set_xlabel("Frame Index (Target Frame)", fontsize=10)
        ax3.grid(True, alpha=0.3, linestyle='--')
        
        self.fig_sum.suptitle(f"Sequential Frame-to-Frame Registration Analysis | Window: {SPAN}x{SPAN}px")
        self.fig_sum.tight_layout()
        
        # --- Option B: 2D Scatter/Density Heatmap of Global Shifts ---
        shift_x = [c['shift_m_x'] for c in valid_comps]
        shift_y = [c['shift_m_y'] for c in valid_comps]
        
        self.fig_heat, self.ax_heat = plt.subplots(figsize=(10, 8))
        self.fig_heat.canvas.manager.set_window_title("2D Registration Error Heatmap")
        
        h = self.ax_heat.hist2d(shift_x, shift_y, bins=30, cmap='inferno')
        self.fig_heat.colorbar(h[3], ax=self.ax_heat, label='Frequency')
        
        self.ax_heat.set_xlabel("Shift X (Meters)")
        self.ax_heat.set_ylabel("Shift Y (Meters)")
        self.ax_heat.set_title(f"2D Scatter/Density Heatmap of Global Shifts: {self.baseline_name}")
        self.ax_heat.grid(True, alpha=0.3, linestyle='--')
        
        stats_str = (f"Summary Statistics:\n"
                     f"Valid Evaluations: {len(mags)}\n"
                     f"Mean Translation: {np.mean(mags):.2f}m\n"
                     f"Mean Abs Rotation: {np.mean(rots):.3f}°\n"
                     f"Mean Correlation: {np.mean(corrs):.3f}")
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.9)
        self.ax_heat.text(0.05, 0.95, stats_str, transform=self.ax_heat.transAxes, fontsize=11,
                          verticalalignment='top', bbox=props)
                     
        self.fig_heat.tight_layout()
        self.fig_heat.savefig(f"registration_error_heatmap_{Location}.png", dpi=300)

    def _init_ui(self):
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(15, 7))
        self.fig.canvas.manager.set_window_title("Interactive Multi-Sensor Analytics")
        self.fig.subplots_adjust(bottom=0.22, top=0.85, wspace=0.4)
        
        ax_prev = self.fig.add_axes([0.35, 0.05, 0.08, 0.05])
        ax_next = self.fig.add_axes([0.45, 0.05, 0.08, 0.05])
        ax_jump = self.fig.add_axes([0.65, 0.05, 0.06, 0.05])
        
        self.btn_prev = Button(ax_prev, '<< Prev')
        self.btn_next = Button(ax_next, 'Next >>')
        self.text_box = TextBox(ax_jump, 'Jump to Frame: ', initial='')
        
        self.btn_prev.on_clicked(self._on_prev)
        self.btn_next.on_clicked(self._on_next)
        self.text_box.on_submit(self._on_jump)
        
        self.status_text = self.fig.text(0.5, 0.15, "", ha='center', va='center', fontsize=10)
        self.stats_annotation = self.fig.text(0.5, 0.5, "", ha='center', va='center', 
                                              fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    def _on_jump(self, text):
        try:
            target_frame = int(text)
            best_idx = 0
            min_diff = float('inf')
            for i, comp in enumerate(self.comparisons):
                diff = min(abs(comp['b_idx'] - target_frame), abs(comp['t_idx'] - target_frame))
                if diff < min_diff:
                    min_diff = diff
                    best_idx = i
            
            self.current_idx = best_idx
            self.update_display()
        except ValueError:
            print(f"Invalid frame number: {text}")

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
        
        b_idx = comp['b_idx']
        t_idx = comp['t_idx']
        
        if 'source_spacecraft' in self.b_grp['ortho_visual'].attrs:
            b_sc_raw = self.b_grp['ortho_visual'].attrs['source_spacecraft'][b_idx]
            t_sc_raw = self.b_grp['ortho_visual'].attrs['source_spacecraft'][t_idx]
            b_sc = b_sc_raw.decode('utf-8') if isinstance(b_sc_raw, bytes) else b_sc_raw
            t_sc = t_sc_raw.decode('utf-8') if isinstance(t_sc_raw, bytes) else t_sc_raw
        else:
            b_sc = self.baseline_name
            t_sc = comp['t_grid']
        
        status = f"Pair {self.current_idx + 1} of {len(self.comparisons)} | Target: Lat {comp.get('lat', 0):.5f}, Lon {comp.get('lon', 0):.5f}\n"
        status += f"Target Frame: {t_idx} [{t_sc}] ({t_date_str}) | Anchor Frame: {b_idx} [{b_sc}] ({b_date_str}) [{comp['label']}]"
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
            self.ax1.set_title(f"Anchor: {b_sc} ({b_date_str})")
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
                         
            self.ax2.set_title(f"Evaluated: {t_sc} ({t_date_str})")
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
            self.ax1.set_title(f"Anchor: {b_sc} ({b_date_str})\n[CALCULATION ABORTED]")
            self.ax2.set_title(f"Evaluated: {t_sc} ({t_date_str})\n[CALCULATION ABORTED]")
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