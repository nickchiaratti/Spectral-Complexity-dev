"""
HLST Multisensor Statistical Cross-Calibration Dashboard

Extracts localized spectral complexity volumes from the HARMONIZED ARD timeline.
Performs 3 distinct statistical evaluations to prove the distribution 
of Gramian hypervolumes and the strict linear relationship between sensors.

Updates: 
- Evaluates 3 distinct comparisons: L8 vs L9, S2A vs S2B, L8/9 vs Tanager.
- Requires at least 5 collections separated by at most 5 days.
- Displays the three comparisons in separate windows.
- Uses common_mask == 0 for valid data.
- Spectral complexity values compared in log scales, Z-scores in linear scales.
"""

import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from scipy import stats
from datetime import datetime, timezone
import tkinter as tk
from tkinter import filedialog
import sys
from pathlib import Path
script_dir = Path(__file__).resolve().parent
if str(script_dir.parent) not in sys.path:
    sys.path.insert(0, str(script_dir.parent))
import SpecComplex as sc

# ==========================================
# 1. CONFIGURATION
# ==========================================
# Maximum allowable time difference to consider frames a "Match"
MATCH_TOLERANCE_DAYS = 5.0 

# ==========================================
# 2. DATA EXTRACTION ENGINE
# ==========================================
class HLST_Statistical_Extractor:
    def __init__(self, filepath, pair_name, sensor1_list, sensor2_list):
        print(f"\nMounting ARD Cube: {filepath}")
        print(f"Preparing Extraction for: {pair_name}")
        self.h5 = h5py.File(filepath, 'r')
        self.pair_name = pair_name
        self.sensor1_list = sensor1_list
        self.sensor2_list = sensor2_list
        
        harm_path = '/HDFEOS/GRIDS/HARMONIZED/Data Fields'
        if harm_path not in self.h5:
            raise KeyError(f"CRITICAL ERROR: {harm_path} missing.")
            
        self.harm_grp = self.h5[harm_path]
        
        # Datasets
        self.ds_vol = self.harm_grp['sliding_volume_map']
        self.ds_zscore = self.harm_grp['sliding_volume_box_cox']#'sliding_volume_z_score']
        self.ds_mask = self.harm_grp['common_mask']
        
        # Relational Vectors
        self.times = self.ds_mask.attrs['acquisition_time']
        raw_spacecraft = self.ds_mask.attrs['source_spacecraft']
        self.spacecrafts = [g.decode('utf-8') if isinstance(g, bytes) else str(g) for g in raw_spacecraft]
        self.num_frames = len(self.times)

    def find_matched_pairs(self):
        """Finds all occurrences where sensors acquired within the threshold."""
        s1_meta = [(t, self.times[t]) for t in range(self.num_frames) if self.spacecrafts[t] in self.sensor1_list]
        s2_meta = [(t, self.times[t]) for t in range(self.num_frames) if self.spacecrafts[t] in self.sensor2_list]
        
        self.matched_pairs = []
        used_s2 = set()
        
        for idx_1, time_1 in s1_meta:
            best_s2 = None
            min_diff = float('inf')
            
            for idx_2, time_2 in s2_meta:
                if idx_2 in used_s2: continue
                diff = abs(time_1 - time_2)
                if diff < min_diff and diff <= (MATCH_TOLERANCE_DAYS * 24 * 3600):
                    min_diff = diff
                    best_s2 = idx_2
                    
            if best_s2 is not None:
                used_s2.add(best_s2)
                dt_1 = datetime.fromtimestamp(time_1, tz=timezone.utc).strftime('%Y-%m-%d')
                dt_2 = datetime.fromtimestamp(self.times[best_s2], tz=timezone.utc).strftime('%Y-%m-%d')
                self.matched_pairs.append({
                    'date_1': dt_1,
                    'date_2': dt_2,
                    'idx_1': idx_1,
                    'idx_2': best_s2,
                    'space_1': self.spacecrafts[idx_1],
                    'space_2': self.spacecrafts[best_s2],
                    'diff_days': min_diff / (24 * 3600)
                })
                
        print(f" -> Discovered {len(self.matched_pairs)} matched temporal brackets for {self.pair_name} (Tolerance: <= {MATCH_TOLERANCE_DAYS} days).")
        if len(self.matched_pairs) < 5:
            print(f" -> Insufficient satellite collections for {self.pair_name}. Need at least 5.")
            return False
        return True

    def extract_matched_distributions(self):
        """Extracts and additively concatenates spatial intersections for all matched dates for both Vol and Z-Score."""
        if not self.find_matched_pairs():
            return False
            
        print("Extracting and additively combining spatial intersections...")
        
        vol_1_global, vol_2_global = [], []
        z_1_global, z_2_global = [], []
        self.pair_data = []
        
        for i, pair in enumerate(self.matched_pairs):
            vol_1 = sc.read_scaled_int16(self.ds_vol, np.s_[pair['idx_1'], :, :])
            vol_2 = sc.read_scaled_int16(self.ds_vol, np.s_[pair['idx_2'], :, :])
            z_1 = sc.read_scaled_int16(self.ds_zscore, np.s_[pair['idx_1'], :, :])
            z_2 = sc.read_scaled_int16(self.ds_zscore, np.s_[pair['idx_2'], :, :])
            
            mask_1 = self.ds_mask[pair['idx_1'], :, :]
            mask_2 = self.ds_mask[pair['idx_2'], :, :]
            
            # STRICT GUARDRAIL: Only intersecting physical pixels that are clear in BOTH sensors (mask == 0)
            # Ensure neither the volume nor the normalized z-score contains NaNs
            joint_mask = (mask_1 == 0) & (mask_2 == 0) & (vol_1 > 0) & (vol_2 > 0) & \
                         (~np.isnan(vol_1)) & (~np.isnan(vol_2)) & \
                         (~np.isnan(z_1)) & (~np.isnan(z_2))
            
            intersect_1_v = vol_1[joint_mask]
            intersect_2_v = vol_2[joint_mask]
            intersect_1_z = z_1[joint_mask]
            intersect_2_z = z_2[joint_mask]
            
            if len(intersect_1_v) > 10:
                vol_1_global.append(intersect_1_v)
                vol_2_global.append(intersect_2_v)
                z_1_global.append(intersect_1_z)
                z_2_global.append(intersect_2_z)
                
                self.pair_data.append({
                    'date_1': pair['date_1'],
                    'date_2': pair['date_2'],
                    'space_1': pair['space_1'],
                    'space_2': pair['space_2'],
                    'diff_days': pair['diff_days'],
                    'vol_1': intersect_1_v,
                    'vol_2': intersect_2_v,
                    'z_1': intersect_1_z,
                    'z_2': intersect_2_z
                })
                
        if not self.pair_data:
            print(f" -> Insufficient intersecting clear pixels for {self.pair_name}.")
            return False
            
        # Sort comparisons in order of time difference between them
        self.pair_data.sort(key=lambda x: x['diff_days'])
            
        # Flatten into massive 1D continuous arrays mapping ALL matched observations
        self.vol_1_global = np.concatenate(vol_1_global) if vol_1_global else np.array([])
        self.vol_2_global = np.concatenate(vol_2_global) if vol_2_global else np.array([])
        self.z_1_global = np.concatenate(z_1_global) if z_1_global else np.array([])
        self.z_2_global = np.concatenate(z_2_global) if z_2_global else np.array([])
        
        print(f" -> Global {self.pair_name} Sensor 1 Pixels Extracted: {len(self.vol_1_global):,}")
        print(f" -> Global {self.pair_name} Sensor 2 Pixels Extracted: {len(self.vol_2_global):,}")
        return True

# ==========================================
# 3. INTERACTIVE DASHBOARD
# ==========================================
class InteractiveDashboard:
    def __init__(self, ard):
        self.ard = ard
        self.current_idx = 0
        
        if len(self.ard.vol_1_global) == 0:
            print("Insufficient overlapping data for plotting.")
            return

        self.fig, self.axes = plt.subplots(2, 3, figsize=(18, 12))
        self.fig.canvas.manager.set_window_title(f"Multisensor Scaling: {self.ard.pair_name}")
        self.fig.subplots_adjust(wspace=0.3, hspace=0.4, bottom=0.12, right=0.88)
        
        # Isolate axes for Top Row (Raw Volume) and Bottom Row (Z-Score)
        self.ax_density_v, self.ax_qq_v, self.ax_spatial_v = self.axes[0]
        self.ax_density_z, self.ax_qq_z, self.ax_spatial_z = self.axes[1]
        
        # Dedicated colorbar axes for Panel 3 to prevent resizing artifacts
        self.cax_v = self.fig.add_axes([0.895, 0.56, 0.01, 0.32]) 
        self.cax_z = self.fig.add_axes([0.895, 0.12, 0.01, 0.32]) 

        # Setup UI Buttons
        axprev = plt.axes([0.42, 0.03, 0.07, 0.04])
        axnext = plt.axes([0.51, 0.03, 0.07, 0.04])
        self.bnext = Button(axnext, 'Next >')
        self.bprev = Button(axprev, '< Prev')
        self.bnext.on_clicked(self.next_pair)
        self.bprev.on_clicked(self.prev_pair)

        self._plot_global_densities()
        self._plot_global_qq_plots()
        self._update_spatial_panels()
        
        plt.suptitle(f"Multisensor Dimensionality Scaling: {self.ard.pair_name}", fontsize=16, y=0.98)

    # --- Reusable Subplot Generators ---
    
    def _plot_density(self, ax, data_1, data_2, title, xlabel, is_log=False):
        samp_1 = np.random.choice(data_1, min(len(data_1), 100000), replace=False)
        samp_2 = np.random.choice(data_2, min(len(data_2), 100000), replace=False)
        
        if is_log:
            # For log scale density, calculate KDE on log10 data to make it look correct
            samp_1_log = np.log10(samp_1[samp_1 > 0])
            samp_2_log = np.log10(samp_2[samp_2 > 0])
            kde_1 = stats.gaussian_kde(samp_1_log)
            kde_2 = stats.gaussian_kde(samp_2_log)
            x_range = np.linspace(min(samp_1_log.min(), samp_2_log.min()), max(samp_1_log.max(), samp_2_log.max()), 500)
            x_plot = 10**x_range
            
            ax.plot(x_plot, kde_1(x_range), color='blue', lw=2, label='Sensor 1')
            ax.fill_between(x_plot, kde_1(x_range), alpha=0.3, color='blue')
            ax.plot(x_plot, kde_2(x_range), color='orange', lw=2, label='Sensor 2')
            ax.fill_between(x_plot, kde_2(x_range), alpha=0.3, color='orange')
            ax.set_xscale('log')
        else:
            kde_1 = stats.gaussian_kde(samp_1)
            kde_2 = stats.gaussian_kde(samp_2)
            x_range = np.linspace(min(samp_1.min(), samp_2.min()), max(samp_1.max(), samp_2.max()), 500)
            
            ax.plot(x_range, kde_1(x_range), color='blue', lw=2, label='Sensor 1')
            ax.fill_between(x_range, kde_1(x_range), alpha=0.3, color='blue')
            ax.plot(x_range, kde_2(x_range), color='orange', lw=2, label='Sensor 2')
            ax.fill_between(x_range, kde_2(x_range), alpha=0.3, color='orange')
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel("Probability Density", fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(loc='upper right')

    def _plot_qq(self, ax, data_1, data_2, title, xlabel, ylabel, is_log=False):
        quantiles = np.linspace(1, 99, 99)
        q_1 = np.percentile(data_1, quantiles)
        q_2 = np.percentile(data_2, quantiles)
        
        if is_log:
            ax.set_xscale('log')
            ax.set_yscale('log')
            # Fit on log-log for Q-Q
            slope, intercept, r_val_qq, _, _ = stats.linregress(np.log10(q_1), np.log10(q_2))
            fit_x = np.array([q_1.min(), q_1.max()])
            fit_y = (10**intercept) * (fit_x**slope)
            sign = '*'
            eq_label = f'Fit (Log): y = {10**intercept:.2f} * x^{slope:.2f}'
            eq_text = f"Power Equation:\n$y = {10**intercept:.3f} \\cdot x^{{{slope:.3f}}}$\n$R^2: {r_val_qq**2:.4f}$"
        else:
            slope, intercept, r_val_qq, _, _ = stats.linregress(q_1, q_2)
            fit_x = np.array([q_1.min(), q_1.max()])
            fit_y = intercept + slope * fit_x
            sign = '+' if intercept >= 0 else '-'
            eq_label = f'Fit: y = {slope:.2f}x {sign} {abs(intercept):.2f}'
            eq_text = f"Linear Equation:\n$y = {slope:.3f}x {sign} {abs(intercept):.3f}$\n$R^2: {r_val_qq**2:.4f}$"
            
        ax.scatter(q_1, q_2, color='purple', s=20, alpha=0.8, edgecolor='k', label='Empirical Quantiles')
        ax.plot(fit_x, fit_y, 'r--', lw=2, label=eq_label)
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(loc='upper left')
        ax.text(0.05, 0.75, eq_text, transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

    def _plot_spatial_scatter(self, ax, cax, data_1, data_2, title, xlabel, ylabel, is_log=False):
        ax.clear()
        cax.clear()
        
        if is_log:
            ax.set_xscale('log')
            ax.set_yscale('log')
            slope_sp, intercept_sp, r_val_sp, _, _ = stats.linregress(np.log10(data_1), np.log10(data_2))
            
            # Using log scaling for hexbin requires explicitly providing bins if we want to mimic log space
            hb = ax.hexbin(data_1, data_2, gridsize=50, cmap='inferno', bins='log', mincnt=1, xscale='log', yscale='log')
            
            x_sp = np.array([data_1.min(), data_1.max()])
            y_sp = (10**intercept_sp) * (x_sp**slope_sp)
            eq_text_sp = f"Power Law: $y = {10**intercept_sp:.3f}x^{{{slope_sp:.3f}}}$\nLog Pearson $r$: {r_val_sp:.3f}\n$R^2$: {r_val_sp**2:.3f}"
            ax.plot(x_sp, y_sp, 'cyan', linestyle='--', lw=2, label=f'Trend (Log)')
        else:
            slope_sp, intercept_sp, r_val_sp, _, _ = stats.linregress(data_1, data_2)
            hb = ax.hexbin(data_1, data_2, gridsize=50, cmap='inferno', bins='log', mincnt=1)
            x_sp = np.array([data_1.min(), data_1.max()])
            y_sp = intercept_sp + slope_sp * x_sp
            sign_sp = '+' if intercept_sp >= 0 else '-'
            eq_text_sp = f"$y = {slope_sp:.3f}x {sign_sp} {abs(intercept_sp):.3f}$\nPearson $r$: {r_val_sp:.3f}\n$R^2$: {r_val_sp**2:.3f}"
            ax.plot(x_sp, y_sp, 'cyan', linestyle='--', lw=2, label=f'Trend: y = {slope_sp:.2f}x')

        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(loc='lower right')
        ax.text(0.05, 0.85, eq_text_sp, transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'), color='black')
                 
        cb = self.fig.colorbar(hb, cax=cax)
        cb.set_label('log10(Pixel Count)', size=9)

    # --- Dashboard Orchestrators ---

    def _plot_global_densities(self):
        """PANEL 1: Additive Density Distributions for Vol and Z-Score."""
        self._plot_density(self.ax_density_v, self.ard.vol_1_global, self.ard.vol_2_global, 
                           "Global Additive Density (Raw Volume)", "Spectral Complexity Volume", is_log=True)
        self._plot_density(self.ax_density_z, self.ard.z_1_global, self.ard.z_2_global, 
                           "Global Additive Density (Z-Score)", "Spectral Complexity Z-Score", is_log=False)

    def _plot_global_qq_plots(self):
        """PANEL 2: Additive Empirical Q-Q Plots for Vol and Z-Score."""
        self._plot_qq(self.ax_qq_v, self.ard.vol_1_global, self.ard.vol_2_global,
                      "Global Empirical Q-Q (Raw Volume)", "Sensor 1 Quantiles (Vol)", "Sensor 2 Quantiles (Vol)", is_log=True)
        self._plot_qq(self.ax_qq_z, self.ard.z_1_global, self.ard.z_2_global,
                      "Global Empirical Q-Q (Z-Score)", "Sensor 1 Quantiles (Z-Score)", "Sensor 2 Quantiles (Z-Score)", is_log=False)

    def _update_spatial_panels(self):
        """PANEL 3: Interactive single-date Spatial Scatter Plot for Vol and Z-Score."""
        data = self.ard.pair_data[self.current_idx]
        dt_1, dt_2, diff_days = data['date_1'], data['date_2'], data['diff_days']
        sp_1, sp_2 = data['space_1'], data['space_2']
        title_str = f"Spatial Match [{self.current_idx + 1}/{len(self.ard.pair_data)}]\n{sp_1}: {dt_1} vs {sp_2}: {dt_2} (Δ {diff_days:.1f}d)"
        
        # Top Row: Raw Volume Update
        self._plot_spatial_scatter(self.ax_spatial_v, self.cax_v, data['vol_1'], data['vol_2'], 
                                   title_str, f"{sp_1} Volume (Local)", f"{sp_2} Volume (Local)", is_log=True)
        
        # Bottom Row: Z-Score Update
        self._plot_spatial_scatter(self.ax_spatial_z, self.cax_z, data['z_1'], data['z_2'], 
                                   title_str, f"{sp_1} Z-Score (Local)", f"{sp_2} Z-Score (Local)", is_log=False)
        
        self.fig.canvas.draw_idle()

    def next_pair(self, event):
        if self.current_idx < len(self.ard.pair_data) - 1:
            self.current_idx += 1
            self._update_spatial_panels()

    def prev_pair(self, event):
        if self.current_idx > 0:
            self.current_idx -= 1
            self._update_spatial_panels()

# ==========================================
# 4. SUMMARY FIGURE GENERATION (Publication / Pipeline)
# ==========================================
def create_summary_figure(filepath, comparisons, output_filename=None, output_dir=None):
    """
    Creates a multi-panel hexbin cross-correlation summary figure.
    """
    valid_comparisons = []
    extractors = []
    
    for comp in comparisons:
        try:
            extractor = HLST_Statistical_Extractor(
                filepath, comp["name"], comp["s1"], comp["s2"]
            )
            if extractor.extract_matched_distributions():
                valid_comparisons.append(comp)
                extractors.append(extractor)
            if hasattr(extractor, 'h5'):
                extractor.h5.close()
        except Exception as e:
            print(f"  -> Skipping {comp['name']}: {e}")

    if not valid_comparisons:
        print(f"  -> No comparison pairs met criteria for summary figure.")
        return None

    num_panels = len(valid_comparisons)
    fig, axes = plt.subplots(1, num_panels, figsize=(3.5 * num_panels, 3.0), layout='constrained')
    if num_panels == 1:
        axes = [axes]
        
    hb_list = []
    first_clim = None

    for ax, comp, extractor in zip(axes, valid_comparisons, extractors):
        data_1 = extractor.z_1_global
        data_2 = extractor.z_2_global

        slope, intercept, r_val, _, _ = stats.linregress(data_1, data_2)
        if first_clim is None:
            hb = ax.hexbin(data_1, data_2, gridsize=50, cmap='cividis', bins='log', mincnt=1)
            first_clim = hb.get_clim()
        else:
            hb = ax.hexbin(data_1, data_2, gridsize=50, cmap='cividis', bins='log', mincnt=1, vmin=first_clim[0], vmax=first_clim[1])
        hb_list.append(hb)

        x_sp = np.array([data_1.min(), data_1.max()])
        y_sp = intercept + slope * x_sp
        sign = '+' if intercept >= 0 else '-'

        n_pixels = len(data_1)
        eq_text = (f"Pixels: {n_pixels:.1e}\n"
                   f"$r$: {r_val:.3f}\n"
                   f"$y = {slope:.2f}x {sign} {abs(intercept):.2f}$")

        ax.plot(x_sp, y_sp, 'cyan', linestyle='--', lw=1.5, label='Trend')

        ax.set_title(comp["name"], fontsize=10, fontweight='bold')
        ax.set_xlabel(comp.get("s1_label", "Sensor 1 $Z$-Score"), fontsize=9)
        ax.set_ylabel(comp.get("s2_label", "Sensor 2 $Z$-Score"), fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.5)

        ax.text(0.05, 0.95, eq_text, transform=ax.transAxes,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', pad=2),
                color='black', fontsize=8, va='top')

    if hb_list:
        cb = fig.colorbar(hb_list[0], ax=axes if isinstance(axes, list) else axes.ravel().tolist(), fraction=0.02, pad=0.02)
        cb.set_label('log10(Pixel Count)', size=8)
        cb.ax.tick_params(labelsize=7)

    if output_dir is None:
        output_dir = os.path.dirname(filepath)
    os.makedirs(output_dir, exist_ok=True)

    if output_filename is None:
        base_name = os.path.splitext(os.path.basename(filepath))[0]
        output_filename = f"{base_name}_cross_sensor_correlation.png"

    out_path = os.path.join(output_dir, output_filename)
    fig.savefig(out_path, dpi=500, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> Saved Cross-Sensor Correlation Summary Plot to: {out_path}")
    return out_path

def plot_cross_sensor_correlations(target_location=None, h5_path=None, location=None):
    """
    Main entry point for pipeline execution.
    """
    if target_location is None and location is not None:
        target_location = location
    if h5_path is None and target_location is not None:
        import glob
        matches = glob.glob(os.path.join("C:/satelliteImagery/HLST30", f"HLST_{target_location}_Harmonized*SC_EM*.h5"))
        if matches:
            matches.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            h5_path = matches[0]
            
    if not h5_path or not os.path.exists(h5_path):
        print(f"Warning: Cannot plot cross-sensor correlation, file not found: {h5_path}")
        return

    cross_comparisons = [
        {
            "name": "Landsat vs Sentinel",
            "s1_label": "Landsat $Z$-Score", "s2_label": "Sentinel $Z$-Score",
            "s1": ["LANDSAT_8", "LANDSAT-8", "LANDSAT_9", "LANDSAT-9"],
            "s2": ["Sentinel-2A", "SENTINEL-2A", "Sentinel-2B", "SENTINEL-2B"]
        },
        {
            "name": "Landsat vs Tanager",
            "s1_label": "Landsat $Z$-Score", "s2_label": "Tanager $Z$-Score",
            "s1": ["LANDSAT_8", "LANDSAT-8", "LANDSAT_9", "LANDSAT-9"],
            "s2": ["Tanager-1", "TANAGER-1"]
        },
        {
            "name": "Sentinel vs Tanager",
            "s1_label": "Sentinel $Z$-Score", "s2_label": "Tanager $Z$-Score",
            "s1": ["Sentinel-2A", "SENTINEL-2A", "Sentinel-2B", "SENTINEL-2B"],
            "s2": ["Tanager-1", "TANAGER-1"]
        },
        {
            "name": "Landsat vs EnMAP",
            "s1_label": "Landsat $Z$-Score", "s2_label": "EnMAP $Z$-Score",
            "s1": ["LANDSAT_8", "LANDSAT-8", "LANDSAT_9", "LANDSAT-9"],
            "s2": ["EnMAP", "ENMAP", "enmap"]
        },
        {
            "name": "Sentinel vs EnMAP",
            "s1_label": "Sentinel $Z$-Score", "s2_label": "EnMAP $Z$-Score",
            "s1": ["Sentinel-2A", "SENTINEL-2A", "Sentinel-2B", "SENTINEL-2B"],
            "s2": ["EnMAP", "ENMAP", "enmap"]
        }
    ]

    create_summary_figure(h5_path, cross_comparisons)

# ==========================================
# 5. EXECUTION POINT
# ==========================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Multisensor Statistical Cross-Calibration Dashboard")
    parser.add_argument('--file', '-f', type=str, default=None, help="Path to specific HDF5 file")
    parser.add_argument('--location', '-l', type=str, default=None, help="Location name (e.g. SanRafael, Tait)")
    parser.add_argument('--save-summary', action='store_true', help="Export summary figure without interactive GUI")
    args = parser.parse_args()

    file_path = args.file
    if not file_path and args.location:
        loc_candidate = f"C:/satelliteImagery/HLST30/HLST_{args.location}_Harmonized_SC_EM-7_Norm-None.h5"
        if os.path.exists(loc_candidate):
            file_path = loc_candidate

    if not file_path:
        root = tk.Tk()
        root.withdraw()
        
        print("Please select the processed HLST ARD HDF5 Cube...")
        file_path = tk.filedialog.askopenfilename(
            title="Select HLST ARD Master Grid HDF5 Cube",
            filetypes=[("HDF5 files", "*.h5")]
        )
    
    if not file_path:
        print("Execution cancelled.")
        sys.exit(0)

    if args.save_summary:
        plot_cross_sensor_correlations(h5_path=file_path)
        sys.exit(0)

    comparisons = [
        {"name": "Landsat-8 vs Landsat-9", "s1": ["LANDSAT_8", "LANDSAT-8"], "s2": ["LANDSAT_9", "LANDSAT-9"]},
        {"name": "Sentinel-2A vs Sentinel-2B", "s1": ["Sentinel-2A", "SENTINEL-2A"], "s2": ["Sentinel-2B", "SENTINEL-2B"]},
        {"name": "Landsat 8/9 vs Tanager", "s1": ["LANDSAT_8", "LANDSAT-8", "LANDSAT_9", "LANDSAT-9"], "s2": ["Tanager-1", "TANAGER-1"]},
        {"name": "Sentinel-2 vs Tanager", "s1": ["Sentinel-2A", "SENTINEL-2A", "Sentinel-2B", "SENTINEL-2B"], "s2": ["Tanager-1", "TANAGER-1"]},
        {"name": "Landsat 8/9 vs EnMAP", "s1": ["LANDSAT_8", "LANDSAT-8", "LANDSAT_9", "LANDSAT-9"], "s2": ["EnMAP", "ENMAP", "enmap"]},
        {"name": "Sentinel-2 vs EnMAP", "s1": ["Sentinel-2A", "SENTINEL-2A", "Sentinel-2B", "SENTINEL-2B"], "s2": ["EnMAP", "ENMAP", "enmap"]}
    ]

    dashboards = []
    
    for comp in comparisons:
        try:
            ard_interface = HLST_Statistical_Extractor(file_path, comp["name"], comp["s1"], comp["s2"])
            if ard_interface.extract_matched_distributions():
                dashboard = InteractiveDashboard(ard_interface)
                dashboards.append(dashboard)
                # Note: We do not close the h5 file until all dashboards are done, or we just let it stay open.
        except Exception as e:
            print(f"Error processing {comp['name']}: {e}")

    if dashboards:
        plt.show()
    else:
        print("No comparisons met the criteria.")
        
    # Optional: ensure file closes. The last interface instance still holds the open file reference, but it's shared.
    if 'ard_interface' in locals() and hasattr(ard_interface, 'h5'):
        ard_interface.h5.close()
