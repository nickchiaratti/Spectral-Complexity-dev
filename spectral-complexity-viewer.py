import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
from datetime import datetime, timezone
import tkinter as tk
from tkinter import filedialog
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d import Axes3D
from skimage import exposure

# --- Configuration ---
# Standard Landsat 8/9 True Color Indices: [C(0), B(1), G(2), R(3), NIR(4), S1(5), S2(6)]
LANDSAT_RGB_BANDS = (3, 2, 1) 
# Bands for 3D Convex Hull Projection (e.g., NIR, Red, Blue)
HULL_BANDS = (6, 4, 2) 

display_normalization = False

class ComplexityResultViewer:
    def __init__(self, h5_path):
        self.h5_path = h5_path
        self.h5 = h5py.File(h5_path, 'r')
        
        # Setup Save Directory
        h5_dir = os.path.dirname(h5_path)
        h5_name = os.path.splitext(os.path.basename(h5_path))[0]

        self.save_dir = os.path.join(h5_dir, h5_name + "_Analysis")
        os.makedirs(self.save_dir, exist_ok=True)

        self.source_name = list(self.h5['/HDFEOS/GRIDS'].keys())[0]

        # Access the Data Fields group
        self.data_grp = self.h5[f'HDFEOS/GRIDS/{self.source_name}/Data Fields']

        # Required Datasets
        self.sr_dset = self.data_grp['surface_reflectance']
        self.vol_dset = self.data_grp.get('frame_endmember_volumes')
        self.em_dset = self.data_grp['frame_endmembers']
        self.idx_dset = self.data_grp['frame_endmember_indices']
        self.grid_dset = self.data_grp.get('tile_volume_map')
        self.slide_dset = self.data_grp.get('sliding_volume_map')
        
        self.num_frames, self.num_bands, self.height, self.width = self.sr_dset.shape

        
        # Metadata from attributes
        self.acq_times = self.sr_dset.attrs.get('acquisition_time')
        self.spacecraft_ids = self.sr_dset.attrs.get('spacecraft_id')
        self.wavelengths = self.sr_dset.attrs.get('wavelengths', np.arange(self.sr_dset.shape[1]))
        
        if self.source_name == "TANAGER":
            self.visuals = self.data_grp['ortho_visual']


        self.current_frame = 0
        # Persistent image and colorbar objects
        self.im_grid = None
        self.im_slide = None
        self.cbar_grid = None
        self.cbar_slide = None

        # Initialize UI components
        self._init_control_ui()
        self._init_analysis_ui()
        self._init_map_ui()
        self._init_hull_ui()
        
        self.update_display()


    def _init_control_ui(self):
        """Dedicated control window for navigation and metadata."""
        self.fig_controls = plt.figure(figsize=(6, 4))
        self.fig_controls.canvas.manager.set_window_title("Time Series Navigation")
        self.ax_meta = self.fig_controls.add_axes([0, 0, 1, 1]); self.ax_meta.axis('off')
        self.ctrl_text = self.ax_meta.text(0.5, 0.75, "", ha='center', va='center', 
                                         fontsize=11, family='monospace',
                                         bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
        ax_prev = self.fig_controls.add_axes([0.1, 0.35, 0.25, 0.15])
        ax_next = self.fig_controls.add_axes([0.65, 0.35, 0.25, 0.15])
        ax_input = self.fig_controls.add_axes([0.45, 0.35, 0.1, 0.15])
        self.btn_prev = Button(ax_prev, '<< Prev')
        self.btn_next = Button(ax_next, 'Next >>')
        self.txt_input = TextBox(ax_input, 'Go to: ', initial=str(self.current_frame))
        ax_save = self.fig_controls.add_axes([0.3, 0.08, 0.4, 0.12])
        self.btn_save = Button(ax_save, 'Save Images', color='lightgreen', hovercolor='lime')
        self.btn_prev.on_clicked(self._on_prev)
        self.btn_next.on_clicked(self._on_next)
        self.txt_input.on_submit(self._on_submit)
        self.btn_save.on_clicked(self._on_save_images)

    def _init_analysis_ui(self):
        """Data visualization window for profiles and locations."""
        self.fig_analysis = plt.figure(figsize=(18, 7))
        self.fig_analysis.canvas.manager.set_window_title(f"Endmember Analysis: {os.path.basename(self.h5_path)}")
        self.ax_spatial = self.fig_analysis.add_subplot(131); self.ax_spectral = self.fig_analysis.add_subplot(132)
        self.ax_vol_curve = self.fig_analysis.add_subplot(133)
        self.analysis_hud = self.fig_analysis.text(0.5, 0.95, "", ha='center', fontsize=10, 
                                                style='italic', bbox=dict(facecolor='white', alpha=0.5))
        plt.subplots_adjust(top=0.85, bottom=0.15, left=0.05, right=0.95, wspace=0.3)

    def _init_map_ui(self):
        """Visualization window for spatial complexity maps."""
        self.fig_maps = plt.figure(figsize=(18, 6))
        self.fig_maps.canvas.manager.set_window_title(f"Spatial Complexity Maps: {os.path.basename(self.h5_path)}")
        self.ax_rgb_map = self.fig_maps.add_subplot(131); self.ax_grid_map = self.fig_maps.add_subplot(132)
        self.ax_slide_map = self.fig_maps.add_subplot(133)
        self.map_hud = self.fig_maps.text(0.5, 0.95, "", ha='center', fontsize=10, 
                                        style='italic', bbox=dict(facecolor='white', alpha=0.5))
        plt.subplots_adjust(top=0.85, bottom=0.15, left=0.05, right=0.95, wspace=0.3)

    def _init_hull_ui(self):
        """Visualization window for the 3D Convex Hull of endmembers."""
        self.fig_hull = plt.figure(figsize=(10, 8))
        self.fig_hull.canvas.manager.set_window_title("3D Convex Hull Visualization")
        self.ax_hull = self.fig_hull.add_subplot(111, projection='3d')
        self.hull_hud = self.fig_hull.text(0.5, 0.95, "", ha='center', fontsize=10, 
                                        style='italic', bbox=dict(facecolor='white', alpha=0.5))
        plt.subplots_adjust(top=0.9, bottom=0.1)

    def _format_metadata(self, idx):
        ts = self.acq_times[idx]
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        time_str = dt.strftime('%Y-%m-%dT%H:%M:%S.%f')
        sat_id = self.spacecraft_ids[idx]
        if isinstance(sat_id, bytes): sat_id = sat_id.decode('ascii')
        return (f"FRAME INDEX: {idx:03d} / {self.num_frames-1:03d}\n"
                f"ACQUISITION: {time_str}\n"
                f"SPACECRAFT:  {sat_id}")


    def update_display(self):
        idx = self.current_frame
        metadata_str = self._format_metadata(idx)
        hud_flat = metadata_str.replace('\n', ' | ')
        self.ctrl_text.set_text(metadata_str); self.analysis_hud.set_text(hud_flat)
        self.map_hud.set_text(hud_flat); self.hull_hud.set_text(hud_flat)
        
        # --- Image Processing ---
        frame_sr = self.sr_dset[idx, ...]
        if self.source_name == "LANDSAT":
            r = percentile_normalize_array(frame_sr[LANDSAT_RGB_BANDS[0]])
            g = percentile_normalize_array(frame_sr[LANDSAT_RGB_BANDS[1]])
            b = percentile_normalize_array(frame_sr[LANDSAT_RGB_BANDS[2]])
            rgb = np.nan_to_num(np.stack([r, g, b], axis=-1), nan=0.0)
        elif self.source_name == "TANAGER":
            rgb = self.visuals[idx, ...]
            rgb = np.transpose(rgb, (1, 2, 0))
            rgb = rgb[...,:3]

        
        


        # --- Analysis Figure ---
        self.ax_spatial.clear(); self.ax_spatial.imshow(rgb); em_indices = self.idx_dset[idx]
        for i, flat_idx in enumerate(em_indices):
            row, col = flat_idx // self.width, flat_idx % self.width
            self.ax_spatial.plot(col, row, 'r+', markersize=10)
            self.ax_spatial.annotate(f'V{i}', (col, row), xytext=(3, 3), textcoords='offset points', 
                                    color='yellow', fontweight='bold', fontsize=9)
        self.ax_spatial.set_title("Endmember Locations"); self.ax_spatial.axis('off')

        self.ax_spectral.clear(); endmembers = self.em_dset[idx, ...]
        for i in range(endmembers.shape[1]):
            if not np.all(endmembers[:, i] == 0):
                self.ax_spectral.plot(self.wavelengths, endmembers[:, i], label=f'V{i}', lw=1.5)
        self.ax_spectral.set_title("Spectral Signatures"); self.ax_spectral.grid(True, alpha=0.3)

        self.ax_vol_curve.clear()
        if self.vol_dset is not None:
            vols = self.vol_dset[idx, :]
            self.ax_vol_curve.plot(np.arange(1, len(vols)+1), vols, 'o-', color='green', markersize=4)
            self.ax_vol_curve.set_title("Endmember Volume Curve"); self.ax_vol_curve.grid(True, alpha=0.3)

        # --- Map Figure ---
        self.ax_rgb_map.clear(); self.ax_rgb_map.imshow(rgb); self.ax_rgb_map.axis('off')
        if self.grid_dset is not None:
            if display_normalization == True:
                grid_data = percentile_normalize_array(self.grid_dset[idx])
            else:
                grid_data = self.grid_dset[idx]
            if self.im_grid is None:
                self.im_grid = self.ax_grid_map.imshow(grid_data, cmap='viridis')
                self.cbar_grid = self.fig_maps.colorbar(self.im_grid, ax=self.ax_grid_map, fraction=0.046, pad=0.04)
                self.ax_grid_map.set_title("Tiled Volume Map"); self.ax_grid_map.axis('off')
            else:
                self.im_grid.set_data(grid_data); self.im_grid.set_clim(vmin=np.nanmin(grid_data), vmax=np.nanmax(grid_data))
                self.cbar_grid.update_normal(self.im_grid)
        if self.slide_dset is not None:
            if display_normalization == True:
                slide_data = percentile_normalize_array(self.slide_dset[idx])
            else:
                slide_data = self.slide_dset[idx]
            if self.im_slide is None:
                self.im_slide = self.ax_slide_map.imshow(slide_data, cmap='viridis')
                self.cbar_slide = self.fig_maps.colorbar(self.im_slide, ax=self.ax_slide_map, fraction=0.046, pad=0.04)
                self.ax_slide_map.set_title("Sliding Window Volume Map"); self.ax_slide_map.axis('off')
            else:
                self.im_slide.set_data(slide_data); self.im_slide.set_clim(vmin=np.nanmin(slide_data), vmax=np.nanmax(slide_data))
                self.cbar_slide.update_normal(self.im_slide)

        # --- Hull Figure (3D Convex Hull Projection) ---
        self.ax_hull.clear()
        # Subset pixels for performance
        pixel_data = frame_sr.reshape(self.num_bands, -1).T
        valid_indices = ~np.isnan(pixel_data).any(axis=1)
        pixel_data = pixel_data[valid_indices]
        if pixel_data.shape[0] > 2000:
            pixel_data = pixel_data[np.random.choice(pixel_data.shape[0], 2000, replace=False)]
        
        # Plot 3D Pixel Cloud
        self.ax_hull.scatter(pixel_data[:, HULL_BANDS[0]], pixel_data[:, HULL_BANDS[1]], pixel_data[:, HULL_BANDS[2]], 
                            c='gray', alpha=0.1, s=1, label='Image Pixels')
        
        # Plot Endmembers
        em_xyz = endmembers[list(HULL_BANDS), :].T
        self.ax_hull.scatter(em_xyz[:, 0], em_xyz[:, 1], em_xyz[:, 2], c='red', s=50, label='Endmembers')

        # Calculate and Plot Convex Hull Polyhedron
        try:
            hull = ConvexHull(em_xyz)
            for simplex in hull.simplices:
                self.ax_hull.plot(em_xyz[simplex, 0], em_xyz[simplex, 1], em_xyz[simplex, 2], 'r-', lw=1, alpha=0.5)
        except: pass
        
        self.ax_hull.set_title(f"3D Convex Hull (Bands {HULL_BANDS[0]+1}, {HULL_BANDS[1]+1}, {HULL_BANDS[2]+1})")
        self.ax_hull.set_xlabel(f"Band {HULL_BANDS[0]+1}"); self.ax_hull.set_ylabel(f"Band {HULL_BANDS[1]+1}"); self.ax_hull.set_zlabel(f"Band {HULL_BANDS[2]+1}")
        self.ax_hull.legend(fontsize='x-small')

        for fig in [self.fig_controls, self.fig_analysis, self.fig_maps, self.fig_hull]: fig.canvas.draw_idle()

    def _on_prev(self, event):
        if self.current_frame > 0: self.current_frame -= 1; self.update_display(); self.txt_input.set_val(str(self.current_frame))
    def _on_next(self, event):
        if self.current_frame < self.num_frames - 1: self.current_frame += 1; self.update_display(); self.txt_input.set_val(str(self.current_frame))
    def _on_submit(self, text):
        try:
            val = int(text)
            if 0 <= val < self.num_frames: self.current_frame = val; self.update_display()
            else: self.txt_input.set_val(str(self.current_frame))
        except ValueError: self.txt_input.set_val(str(self.current_frame))

    def _on_save_images(self, event):
        idx = self.current_frame; 
        ts = self.acq_times[idx]
        dt = datetime.fromtimestamp(ts, tz=timezone.utc); 
        time_str = dt.strftime('%Y-%m-%dT%H%M%S')
        for f, name in [(self.fig_analysis, "Analysis"), (self.fig_maps, "Maps"), (self.fig_hull, "Hull")]:
            path = os.path.join(self.save_dir, f"{time_str}_{idx:02d}_{name}.png")
            f.savefig(path, dpi=600); print(f"Saved: {path}")

    def run(self): plt.show()

def percentile_normalize_array(arr, lower_percentile=1, upper_percentile=99):
    '''Normalizes a numpy array to the range [0, 1] using percentiles, ignoring NaNs.'''
    p_low, p_high = np.nanpercentile(arr, (lower_percentile, upper_percentile))
    norm_arr = exposure.rescale_intensity(arr, in_range=(p_low, p_high), out_range=(0, 1))
    return norm_arr.clip(0, 1)

if __name__ == "__main__":
    root = tk.Tk(); root.withdraw()
    file_path = filedialog.askopenfilename(title="Select Complexity Analysis HDF5", filetypes=[("HDF5", "*.h5")])
    if file_path: viewer = ComplexityResultViewer(file_path); viewer.run()
    root.destroy()
    
         

        