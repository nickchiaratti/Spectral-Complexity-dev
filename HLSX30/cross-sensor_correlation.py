"""
HLST Multisensor Statistical Cross-Calibration Dashboard

Extracts localized spectral complexity volumes from the HARMONIZED ARD timeline.
Performs 3 distinct statistical evaluations to prove the distribution 
of Gramian hypervolumes and the strict linear relationship between the 
8-band HLSL30 (Landsat) and 13-band HLSS30 (Sentinel-2) manifolds.

Updates: 
- Extracts and visualizes BOTH the raw 'sliding_volume_map' and 
  the normalized 'sliding_volume_z_score' simultaneously in a 2x3 grid.
- Evaluates ALL matching collections across the temporal bounds.
- Additively combines intersecting pixels for global density and Q-Q charting.
- Implements an interactive GUI to slide through the single-date spatial comparisons.

1. Overlaid Gaussian Density Distributions (Global Additive)
2. Empirical Quantile-Quantile (Q-Q) Plot (Global Additive)
3. Temporally Bracketed Spatial Scatter Plot (Interactive Slider)
"""

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from scipy import stats
from datetime import datetime, timezone
import tkinter as tk
from tkinter import filedialog

# ==========================================
# 1. CONFIGURATION
# ==========================================
# Maximum allowable time difference to consider frames a "Match"
MATCH_TOLERANCE_DAYS = 2.0 

# ==========================================
# 2. DATA EXTRACTION ENGINE
# ==========================================
class HLST_Statistical_Extractor:
    def __init__(self, filepath):
        print(f"Mounting ARD Cube: {filepath}")
        self.h5 = h5py.File(filepath, 'r')
        
        harm_path = '/HDFEOS/GRIDS/HARMONIZED/Data Fields'
        if harm_path not in self.h5:
            raise KeyError(f"CRITICAL ERROR: {harm_path} missing.")
            
        self.harm_grp = self.h5[harm_path]
        
        # Datasets
        if 'sliding_volume_map' not in self.harm_grp or 'sliding_volume_z_score' not in self.harm_grp or 'common_mask' not in self.harm_grp:
            raise KeyError("CRITICAL ERROR: Required analytical datasets missing. Run SC calculations first.")
            
        self.ds_vol = self.harm_grp['sliding_volume_map']
        self.ds_zscore = self.harm_grp['sliding_volume_z_score']
        self.ds_mask = self.harm_grp['common_mask']
        
        # Relational Vectors
        self.times = self.ds_mask.attrs['acquisition_time']
        raw_grids = self.ds_mask.attrs['source_grid']
        self.grids = [g.decode('utf-8') if isinstance(g, bytes) else str(g) for g in raw_grids]
        self.num_frames = len(self.times)

    def find_matched_pairs(self):
        """Finds all occurrences where HLSL30 and HLSS30 were acquired within the threshold."""
        l30_meta = [(t, self.times[t]) for t in range(self.num_frames) if self.grids[t] == 'HLSL30']
        s30_meta = [(t, self.times[t]) for t in range(self.num_frames) if self.grids[t] == 'HLSS30']
        
        self.matched_pairs = []
        used_s30 = set()
        
        for idx_l, time_l in l30_meta:
            best_s30 = None
            min_diff = float('inf')
            
            for idx_s, time_s in s30_meta:
                if idx_s in used_s30: continue
                diff = abs(time_l - time_s)
                if diff < min_diff and diff <= (MATCH_TOLERANCE_DAYS * 24 * 3600):
                    min_diff = diff
                    best_s30 = idx_s
                    
            if best_s30 is not None:
                used_s30.add(best_s30)
                dt_l30 = datetime.fromtimestamp(time_l, tz=timezone.utc).strftime('%Y-%m-%d')
                dt_s30 = datetime.fromtimestamp(self.times[best_s30], tz=timezone.utc).strftime('%Y-%m-%d')
                self.matched_pairs.append({
                    'date_l30': dt_l30,
                    'date_s30': dt_s30,
                    'l30_idx': idx_l,
                    's30_idx': best_s30,
                    'diff_days': min_diff / (24 * 3600)
                })
                
        print(f" -> Discovered {len(self.matched_pairs)} matched temporal brackets (Tolerance: <= {MATCH_TOLERANCE_DAYS} days).")

    def extract_matched_distributions(self):
        """Extracts and additively concatenates spatial intersections for all matched dates for both Vol and Z-Score."""
        self.find_matched_pairs()
        
        if not self.matched_pairs:
            raise ValueError("No matching L30 and S30 frames found within the temporal tolerance.")
            
        print("Extracting and additively combining spatial intersections...")
        
        l30_vols_global, s30_vols_global = [], []
        l30_z_global, s30_z_global = [], []
        self.pair_data = []
        
        for i, pair in enumerate(self.matched_pairs):
            vol_l30 = self.ds_vol[pair['l30_idx'], :, :]
            vol_s30 = self.ds_vol[pair['s30_idx'], :, :]
            z_l30 = self.ds_zscore[pair['l30_idx'], :, :]
            z_s30 = self.ds_zscore[pair['s30_idx'], :, :]
            
            mask_l30 = self.ds_mask[pair['l30_idx'], :, :]
            mask_s30 = self.ds_mask[pair['s30_idx'], :, :]
            
            # STRICT GUARDRAIL: Only intersecting physical pixels that are clear in BOTH sensors
            # Ensure neither the volume nor the normalized z-score contains NaNs
            joint_mask = (mask_l30 == 1) & (mask_s30 == 1) & (vol_l30 > 0) & (vol_s30 > 0) & \
                         (~np.isnan(vol_l30)) & (~np.isnan(vol_s30)) & \
                         (~np.isnan(z_l30)) & (~np.isnan(z_s30))
            
            intersect_l30_v = vol_l30[joint_mask]
            intersect_s30_v = vol_s30[joint_mask]
            intersect_l30_z = z_l30[joint_mask]
            intersect_s30_z = z_s30[joint_mask]
            
            if len(intersect_l30_v) > 10:
                l30_vols_global.append(intersect_l30_v)
                s30_vols_global.append(intersect_s30_v)
                l30_z_global.append(intersect_l30_z)
                s30_z_global.append(intersect_s30_z)
                
                self.pair_data.append({
                    'date_l30': pair['date_l30'],
                    'date_s30': pair['date_s30'],
                    'diff_days': pair['diff_days'],
                    'l30_v': intersect_l30_v,
                    's30_v': intersect_s30_v,
                    'l30_z': intersect_l30_z,
                    's30_z': intersect_s30_z
                })
                
        # Flatten into massive 1D continuous arrays mapping ALL matched observations
        self.vol_l30_global = np.concatenate(l30_vols_global) if l30_vols_global else np.array([])
        self.vol_s30_global = np.concatenate(s30_vols_global) if s30_vols_global else np.array([])
        self.z_l30_global = np.concatenate(l30_z_global) if l30_z_global else np.array([])
        self.z_s30_global = np.concatenate(s30_z_global) if s30_z_global else np.array([])
        
        print(f" -> Global HLSL30 Matched Pixels Extracted: {len(self.vol_l30_global):,}")
        print(f" -> Global HLSS30 Matched Pixels Extracted: {len(self.vol_s30_global):,}")

# ==========================================
# 3. INTERACTIVE DASHBOARD
# ==========================================
class InteractiveDashboard:
    def __init__(self, ard):
        self.ard = ard
        self.current_idx = 0
        
        if len(self.ard.vol_l30_global) == 0:
            print("Insufficient overlapping data for plotting.")
            return

        self.fig, self.axes = plt.subplots(2, 3, figsize=(18, 12))
        self.fig.canvas.manager.set_window_title("Multisensor Spectral Complexity Scaling")
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
        
        plt.suptitle("Multisensor Dimensionality Scaling: Landsat 8/9 vs Sentinel-2", fontsize=16, y=0.98)
        plt.show()

    # --- Reusable Subplot Generators ---
    
    def _plot_density(self, ax, data_l30, data_s30, title, xlabel):
        samp_l30 = np.random.choice(data_l30, min(len(data_l30), 100000), replace=False)
        samp_s30 = np.random.choice(data_s30, min(len(data_s30), 100000), replace=False)
        
        kde_l30 = stats.gaussian_kde(samp_l30)
        kde_s30 = stats.gaussian_kde(samp_s30)
        
        x_range = np.linspace(min(samp_l30.min(), samp_s30.min()), max(samp_l30.max(), samp_s30.max()), 500)
        
        ax.plot(x_range, kde_l30(x_range), color='blue', lw=2, label='Landsat 8/9 (8 Bands)')
        ax.fill_between(x_range, kde_l30(x_range), alpha=0.3, color='blue')
        
        ax.plot(x_range, kde_s30(x_range), color='orange', lw=2, label='Sentinel-2 (13 Bands)')
        ax.fill_between(x_range, kde_s30(x_range), alpha=0.3, color='orange')
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel("Probability Density", fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(loc='upper right')

    def _plot_qq(self, ax, data_l30, data_s30, title, xlabel, ylabel):
        quantiles = np.linspace(1, 99, 99)
        q_l30 = np.percentile(data_l30, quantiles)
        q_s30 = np.percentile(data_s30, quantiles)
        
        slope, intercept, r_val_qq, _, _ = stats.linregress(q_l30, q_s30)
        
        ax.scatter(q_l30, q_s30, color='purple', s=20, alpha=0.8, edgecolor='k', label='Empirical Quantiles')
        
        line_x = np.array([q_l30.min(), q_l30.max()])
        sign = '+' if intercept >= 0 else '-'
        eq_label = f'Fit: y = {slope:.2f}(Landsat) {sign} {abs(intercept):.2f}'
        ax.plot(line_x, intercept + slope * line_x, 'r--', lw=2, label=eq_label)
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(loc='upper left')
        
        eq_text = f"Linear Equation:\n$y = {slope:.3f}x {sign} {abs(intercept):.3f}$\n$R^2: {r_val_qq**2:.4f}$"
        ax.text(0.05, 0.75, eq_text, transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

    def _plot_spatial_scatter(self, ax, cax, data_l30, data_s30, title, xlabel, ylabel):
        ax.clear()
        cax.clear()
        
        slope_sp, intercept_sp, r_val_sp, _, _ = stats.linregress(data_l30, data_s30)
        
        hb = ax.hexbin(data_l30, data_s30, gridsize=50, cmap='inferno', bins='log', mincnt=1)
        
        x_sp = np.array([data_l30.min(), data_l30.max()])
        sign_sp = '+' if intercept_sp >= 0 else '-'
        ax.plot(x_sp, intercept_sp + slope_sp * x_sp, 'cyan', linestyle='--', lw=2, label=f'Trend: y = {slope_sp:.2f}x {sign_sp} {abs(intercept_sp):.2f}')
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(loc='lower right')
        
        eq_text_sp = f"$y = {slope_sp:.3f}x {sign_sp} {abs(intercept_sp):.3f}$\nPearson $r$: {r_val_sp:.3f}\n$R^2$: {r_val_sp**2:.3f}"
        ax.text(0.05, 0.85, eq_text_sp, transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'), color='black')
                 
        cb = self.fig.colorbar(hb, cax=cax)
        cb.set_label('log10(Pixel Count)', size=9)

    # --- Dashboard Orchestrators ---

    def _plot_global_densities(self):
        """PANEL 1: Additive Density Distributions for Vol and Z-Score."""
        self._plot_density(self.ax_density_v, self.ard.vol_l30_global, self.ard.vol_s30_global, 
                           "Global Additive Density (Raw Volume)", "Spectral Complexity Volume")
        self._plot_density(self.ax_density_z, self.ard.z_l30_global, self.ard.z_s30_global, 
                           "Global Additive Density (Z-Score)", "Spectral Complexity Z-Score")

    def _plot_global_qq_plots(self):
        """PANEL 2: Additive Empirical Q-Q Plots for Vol and Z-Score."""
        self._plot_qq(self.ax_qq_v, self.ard.vol_l30_global, self.ard.vol_s30_global,
                      "Global Empirical Q-Q (Raw Volume)", "Landsat 8/9 Quantiles (Volume)", "Sentinel-2 Quantiles (Volume)")
        self._plot_qq(self.ax_qq_z, self.ard.z_l30_global, self.ard.z_s30_global,
                      "Global Empirical Q-Q (Z-Score)", "Landsat 8/9 Quantiles (Z-Score)", "Sentinel-2 Quantiles (Z-Score)")

    def _update_spatial_panels(self):
        """PANEL 3: Interactive single-date Spatial Scatter Plot for Vol and Z-Score."""
        data = self.ard.pair_data[self.current_idx]
        dt_l30, dt_s30, diff_days = data['date_l30'], data['date_s30'], data['diff_days']
        title_str = f"Spatial Match [{self.current_idx + 1}/{len(self.ard.pair_data)}]\nL30: {dt_l30} vs S30: {dt_s30} (Δ {diff_days:.1f}d)"
        
        # Top Row: Raw Volume Update
        self._plot_spatial_scatter(self.ax_spatial_v, self.cax_v, data['l30_v'], data['s30_v'], 
                                   title_str, "Landsat 8/9 Volume (Local)", "Sentinel-2 Volume (Local)")
        
        # Bottom Row: Z-Score Update
        self._plot_spatial_scatter(self.ax_spatial_z, self.cax_z, data['l30_z'], data['s30_z'], 
                                   title_str, "Landsat 8/9 Z-Score (Local)", "Sentinel-2 Z-Score (Local)")
        
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
# 4. EXECUTION POINT
# ==========================================
if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    
    print("Please select the processed HLST ARD HDF5 Cube...")
    file_path = tk.filedialog.askopenfilename(
        title="Select HLST ARD Master Grid HDF5 Cube",
        filetypes=[("HDF5 files", "*.h5")]
    )
    
    if file_path:
        ard_interface = HLST_Statistical_Extractor(file_path)
        ard_interface.extract_matched_distributions()
        dashboard = InteractiveDashboard(ard_interface)
        ard_interface.h5.close()
    else:
        print("Execution cancelled.")