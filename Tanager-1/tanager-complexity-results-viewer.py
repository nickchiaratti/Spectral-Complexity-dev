import os
import h5py
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
from datetime import datetime
import tkinter as tk
from tkinter import filedialog
from pathlib import Path

class TanagerComplexityResultViewer:
    def __init__(self, h5_path):
        self.h5_path = h5_path
        self.h5 = h5py.File(h5_path, 'r')
        
        # Access the TANAGER Data Fields group
        self.grid_path = '/HDFEOS/GRIDS/TANAGER/Data Fields'
        if self.grid_path not in self.h5:
            raise KeyError(f"Could not find TANAGER group structure: {self.grid_path}")
            
        self.data_grp = self.h5[self.grid_path]
        
        # Required Datasets
        self.sr_dset = self.data_grp['surface_reflectance']
        self.vis_dset = self.data_grp['ortho_visual']
        self.em_dset = self.data_grp['endmembers']
        self.idx_dset = self.data_grp['endmember_indices']
        
        # Mapping Datasets
        self.tile_dset = self.data_grp.get('tile_volume_map')
        self.slide_dset = self.data_grp.get('sliding_volume_map')
        
        # Volume Curve Dataset
        self.vol_curve_dset = self.data_grp.get('endmember_volumes')
        
        # Metadata
        self.wavelengths = self.sr_dset.attrs.get('wavelengths')
        self.num_frames, self.num_bands, self.height, self.width = self.sr_dset.shape
        self.current_frame = 0
        
        # Persistent image objects for colorbar stability
        self.im_tile = None
        self.im_slide = None
        self.cbar_tile = None
        self.cbar_slide = None
        
        # Initialize UI Components
        self._init_control_ui()
        self._init_analysis_ui()
        self._init_map_ui()
        
        self.update_display()

    def _init_control_ui(self):
        """Creates the navigation and metadata control window."""
        self.fig_controls = plt.figure(figsize=(6, 4))
        self.fig_controls.canvas.manager.set_window_title("Tanager Navigation")
        
        self.ax_meta = self.fig_controls.add_axes([0, 0, 1, 1])
        self.ax_meta.axis('off')
        self.ctrl_text = self.ax_meta.text(0.5, 0.75, "", ha='center', va='center', 
                                         fontsize=10, family='monospace',
                                         bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray'))

        # Navigation Widgets
        ax_prev = self.fig_controls.add_axes([0.1, 0.35, 0.25, 0.15])
        ax_next = self.fig_controls.add_axes([0.65, 0.35, 0.25, 0.15])
        ax_input = self.fig_controls.add_axes([0.45, 0.35, 0.1, 0.15])
        
        self.btn_prev = Button(ax_prev, '<< Previous')
        self.btn_next = Button(ax_next, 'Next >>')
        self.txt_input = TextBox(ax_input, 'Frame: ', initial=str(self.current_frame))
        
        self.btn_prev.on_clicked(self._on_prev)
        self.btn_next.on_clicked(self._on_next)
        self.txt_input.on_submit(self._on_submit)

        # Save Button
        ax_save = self.fig_controls.add_axes([0.3, 0.1, 0.4, 0.12])
        self.btn_save = Button(ax_save, 'Save Images', color='lightgreen', hovercolor='lime')
        self.btn_save.on_clicked(self._on_save)

    def _init_analysis_ui(self):
        """Creates the visualizer window with 3 subplots (Spatial, Spectral, Volume)."""
        self.fig_plots = plt.figure(figsize=(18, 7))
        self.fig_plots.canvas.manager.set_window_title(f"Tanager Analysis: {os.path.basename(self.h5_path)}")
        
        self.ax_spatial = self.fig_plots.add_subplot(131)
        self.ax_spectral = self.fig_plots.add_subplot(132)
        self.ax_vol_curve = self.fig_plots.add_subplot(133)
        
        self.plot_hud = self.fig_plots.text(0.5, 0.95, "", ha='center', fontsize=11, 
                                          fontweight='bold', bbox=dict(facecolor='white', alpha=0.6))
        
        plt.subplots_adjust(top=0.88, bottom=0.12, left=0.05, right=0.95, wspace=0.25)

    def _init_map_ui(self):
        """Creates the spatial complexity map window (RGB, Tiled, Sliding)."""
        self.fig_maps = plt.figure(figsize=(18, 6))
        self.fig_maps.canvas.manager.set_window_title(f"Complexity Maps: {os.path.basename(self.h5_path)}")
        
        self.ax_rgb = self.fig_maps.add_subplot(131)
        self.ax_tile_map = self.fig_maps.add_subplot(132)
        self.ax_slide_map = self.fig_maps.add_subplot(133)
        
        self.map_hud = self.fig_maps.text(0.5, 0.95, "", ha='center', fontsize=11, 
                                        fontweight='bold', bbox=dict(facecolor='white', alpha=0.6))
        
        plt.subplots_adjust(top=0.85, bottom=0.15, left=0.05, right=0.95, wspace=0.3)

    def _get_timestamp(self, idx):
        """Parses timestamp from the METADATA group JSON attributes."""
        try:
            if "METADATA" in self.h5:
                meta_attr = f"frame_{idx}_json"
                if meta_attr in self.h5["METADATA"].attrs:
                    meta_json = json.loads(self.h5["METADATA"].attrs[meta_attr])
                    raw_time = meta_json['properties'].get('datetime', 'Unknown')
                    dt = datetime.fromisoformat(raw_time.replace('Z', '+00:00'))
                    return dt.strftime('%Y-%m-%dT%H:%M:%S.%f')
        except Exception:
            pass
        return "Time Unavailable"

    def update_display(self):
        idx = self.current_frame
        time_str = self._get_timestamp(idx)
        
        # 1. Update Labels & HUDs
        header_info = f"FRAME: {idx:03d} / {self.num_frames-1:03d} | ACQUISITION: {time_str}"
        self.ctrl_text.set_text(header_info.replace(' | ', '\n'))
        self.plot_hud.set_text(header_info)
        self.map_hud.set_text(header_info)
        
        # Common Visual Reference
        vis_data = self.vis_dset[idx, ...]
        rgb = np.transpose(vis_data[:3, ...], (1, 2, 0))
        
        # --- ANALYSIS FIGURE UPDATES ---
        self.ax_spatial.clear()
        self.ax_spatial.imshow(rgb)
        em_indices = self.idx_dset[idx]
        for i, flat_idx in enumerate(em_indices):
            row = flat_idx // self.width
            col = flat_idx % self.width
            self.ax_spatial.plot(col, row, 'r+', markersize=12, markeredgewidth=2)
            self.ax_spatial.annotate(f'V{i}', (col, row), xytext=(4, 4), 
                                    textcoords='offset points', color='yellow', 
                                    fontweight='bold', fontsize=10)
        self.ax_spatial.set_title("Endmember Locations (Ortho Visual)")
        self.ax_spatial.axis('off')
        
        self.ax_spectral.clear()
        endmembers = self.em_dset[idx, ...]
        x_axis = self.wavelengths if self.wavelengths is not None else np.arange(self.num_bands)
        for i in range(endmembers.shape[1]):
            sig = endmembers[:, i]
            if not np.all(np.isnan(sig)):
                self.ax_spectral.plot(x_axis, sig, label=f'V{i}', lw=1.5, alpha=0.8)
        self.ax_spectral.set_title("Endmember Spectral Profiles")
        self.ax_spectral.set_ylabel("Surface Reflectance")
        self.ax_spectral.grid(True, linestyle='--', alpha=0.3)
        self.ax_spectral.legend(fontsize='x-small', ncol=2, loc='upper right')
        
        self.ax_vol_curve.clear()
        if self.vol_curve_dset is not None:
            vols = self.vol_curve_dset[idx, :]
            counts = np.arange(1, len(vols) + 1)
            self.ax_vol_curve.plot(counts, vols, 'o-', color='green', markersize=4, lw=1.5)
            self.ax_vol_curve.set_title("Volume vs Endmember Count")
            self.ax_vol_curve.set_xlabel("Number of Endmembers")
            self.ax_vol_curve.grid(True, alpha=0.3)
            if np.nanmax(vols) > 0 and np.nanmax(vols) / (np.nanmin(vols[vols>0]) + 1e-9) > 100:
                self.ax_vol_curve.set_yscale('log')

        # --- MAP FIGURE UPDATES ---
        self.ax_rgb.clear()
        self.ax_rgb.imshow(rgb)
        self.ax_rgb.set_title("True Color Reference")
        self.ax_rgb.axis('off')

        # Tiled Map
        if self.tile_dset is not None:
            tile_data = self.tile_dset[idx]
            if self.im_tile is None:
                self.im_tile = self.ax_tile_map.imshow(tile_data, cmap='viridis')
                self.cbar_tile = self.fig_maps.colorbar(self.im_tile, ax=self.ax_tile_map, fraction=0.046, pad=0.04)
                self.ax_tile_map.set_title("Tiled Volume Map")
                self.ax_tile_map.axis('off')
            else:
                self.im_tile.set_data(tile_data)
                vmin, vmax = np.nanmin(tile_data), np.nanmax(tile_data)
                self.im_tile.set_clim(vmin=vmin, vmax=vmax)
                self.cbar_tile.update_normal(self.im_tile)

        # Sliding Map
        if self.slide_dset is not None:
            slide_data = self.slide_dset[idx]
            if self.im_slide is None:
                self.im_slide = self.ax_slide_map.imshow(slide_data, cmap='viridis')
                self.cbar_slide = self.fig_maps.colorbar(self.im_slide, ax=self.ax_slide_map, fraction=0.046, pad=0.04)
                self.ax_slide_map.set_title("Sliding Window Volume Map")
                self.ax_slide_map.axis('off')
            else:
                self.im_slide.set_data(slide_data)
                vmin, vmax = np.nanmin(slide_data), np.nanmax(slide_data)
                self.im_slide.set_clim(vmin=vmin, vmax=vmax)
                self.cbar_slide.update_normal(self.im_slide)

        # Refresh
        self.fig_controls.canvas.draw_idle()
        self.fig_plots.canvas.draw_idle()
        self.fig_maps.canvas.draw_idle()

    def _on_prev(self, event):
        if self.current_frame > 0:
            self.current_frame -= 1
            self.update_display()
            self.txt_input.set_val(str(self.current_frame))

    def _on_next(self, event):
        if self.current_frame < self.num_frames - 1:
            self.current_frame += 1
            self.update_display()
            self.txt_input.set_val(str(self.current_frame))

    def _on_submit(self, text):
        try:
            val = int(text)
            if 0 <= val < self.num_frames:
                self.current_frame = val
                self.update_display()
            else:
                self.txt_input.set_val(str(self.current_frame))
        except ValueError:
            self.txt_input.set_val(str(self.current_frame))

    def _on_save(self, event):
        """Saves current Analysis and Map figures as 600 DPI PNGs in a subfolder."""
        idx = self.current_frame
        raw_time = self._get_timestamp(idx)
        # Sanitize timestamp for filenames (colons are usually illegal)
        safe_time = raw_time.replace(':', '.')
        
        # Setup Export Directory
        h5_path_obj = Path(self.h5_path)
        base_dir = h5_path_obj.parent
        folder_name = h5_path_obj.stem # Name of the .h5 file without extension
        export_dir = base_dir / folder_name
        export_dir.mkdir(exist_ok=True)
        
        # Construct Filenames
        prefix = f"{safe_time}_{idx:03d}"
        analysis_filename = export_dir / f"{prefix}_endmember_analysis.png"
        map_filename = export_dir / f"{prefix}_spatial_complexity_map.png"
        
        print(f"Exporting figures to {export_dir} at 600 DPI...")
        try:
            self.fig_plots.savefig(analysis_filename, dpi=600, bbox_inches='tight')
            self.fig_maps.savefig(map_filename, dpi=600, bbox_inches='tight')
            print(f"Successfully saved:\n - {analysis_filename.name}\n - {map_filename.name}")
        except Exception as e:
            print(f"Error saving images: {e}")

    def run(self):
        plt.show()

if __name__ == "__main__":
    root = tk.Tk(); root.withdraw()
    print("Please select the Tanager calculated HDF5 file...")
    file_path = filedialog.askopenfilename(
        title="Select Tanager Analysis Results",
        filetypes=[("HDF5 files", "*.h5")]
    )
    if file_path:
        viewer = TanagerComplexityResultViewer(file_path)
        viewer.run()
    root.destroy()