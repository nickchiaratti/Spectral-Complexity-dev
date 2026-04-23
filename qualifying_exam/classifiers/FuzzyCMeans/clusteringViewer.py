import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
from matplotlib.widgets import Button, TextBox
from datetime import datetime, timezone
import tkinter as tk
from tkinter import filedialog
from skimage import exposure

# --- Configuration ---
# Standard Landsat 8/9 True Color Indices: [C(0), B(1), G(2), R(3), NIR(4), S1(5), S2(6)]
LANDSAT_RGB_BANDS = (3, 2, 1) 

TS_LOCATIONS = [
    {'yx': (56, 127),  'label': "Golf Course",                    'color': 'tab:green'},
    {'yx': (87, 128),  'label': "Artificial turf football field", 'color': 'tab:blue'},
    {'yx': (80, 122),  'label': "Recently added artificial turf field",'color': 'tab:cyan'},
    {'yx': (66, 28),   'label': "Tait Parking Lot",               'color': 'tab:red'}
]

class FCMViewer:
    def __init__(self, h5_path):
        self.h5_path = h5_path
        self.h5 = h5py.File(h5_path, 'r')
        
        # Setup Save Directory
        h5_dir = os.path.dirname(h5_path)
        h5_name = os.path.splitext(os.path.basename(h5_path))[0]
        self.save_dir = os.path.join(h5_dir, h5_name + "_FCM_Visuals")
        os.makedirs(self.save_dir, exist_ok=True)

        self.grid_name = list(self.h5['/HDFEOS/GRIDS'].keys())[0]
        self.data_grp = self.h5[f'HDFEOS/GRIDS/{self.grid_name}/Data Fields']

        # Required Datasets
        if 'fcm_hard_clusters' not in self.data_grp:
            raise ValueError("The selected HDF5 file does not contain 'fcm_hard_clusters'. Please run the FCM clustering script first.")

        self.sr_dset = self.data_grp['surface_reflectance']
        self.fcm_dset = self.data_grp['fcm_hard_clusters']
        
        # Load Cluster Centroids and Features for Profiling
        self.centers_dset = self.data_grp['fcm_cluster_centers']
        self.cluster_centers = self.centers_dset[:]
        # Decode byte strings if necessary
        raw_features = self.centers_dset.attrs.get('features', [])
        self.cluster_features = [f.decode('utf-8') if isinstance(f, bytes) else f for f in raw_features]
        
        self.num_frames, self.height, self.width = self.fcm_dset.shape
        self.num_clusters = self.fcm_dset.attrs.get('num_clusters', 5)
        
        # Metadata from attributes
        self.acq_times = self.sr_dset.attrs.get('acquisition_time')
        self.spacecraft_ids = self.sr_dset.attrs.get('spacecraft_id')
        
        if self.grid_name == "TANAGER":
            self.visuals = self.data_grp['ortho_visual']

        self.current_frame = 0

        # Initialize UI components
        self._init_control_ui()
        self._init_map_ui()
        
        self.update_display()

    def _init_control_ui(self):
        """Dedicated control window for navigation and metadata."""
        self.fig_controls = plt.figure(figsize=(6, 4))
        self.fig_controls.canvas.manager.set_window_title("Time Series Navigation")
        self.ax_meta = self.fig_controls.add_axes([0, 0, 1, 1])
        self.ax_meta.axis('off')
        
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
        self.btn_save = Button(ax_save, 'Save Current View', color='lightblue', hovercolor='skyblue')
        
        self.btn_prev.on_clicked(self._on_prev)
        self.btn_next.on_clicked(self._on_next)
        self.txt_input.on_submit(self._on_submit)
        self.btn_save.on_clicked(self._on_save_images)

    def _init_map_ui(self):
        """Visualization window for RGB, Cluster maps, Profiles, and Time Series."""
        self.fig_maps = plt.figure(figsize=(22, 12))
        self.fig_maps.canvas.manager.set_window_title(f"FCM Clusters: {os.path.basename(self.h5_path)}")
        
        # GridSpec for flexible layout: 2 rows, 3 columns
        gs = self.fig_maps.add_gridspec(2, 3, height_ratios=[1, 1])
        
        self.ax_rgb = self.fig_maps.add_subplot(gs[0, 0])
        self.ax_cluster = self.fig_maps.add_subplot(gs[0, 1])
        self.ax_centers = self.fig_maps.add_subplot(gs[0, 2])
        self.ax_ts = self.fig_maps.add_subplot(gs[1, :]) # Time series spans bottom row
        
        self.map_hud = self.fig_maps.text(0.5, 0.96, "", ha='center', fontsize=12, 
                                        style='italic', bbox=dict(facecolor='white', alpha=0.5))
                                        
        # Generate discrete colormap for the clusters
        self.cmap = plt.get_cmap('Set1', self.num_clusters)
        self.cmap.set_bad(color='black') # Masked (NoData) pixels will be black
        
        plt.subplots_adjust(top=0.90, bottom=0.10, left=0.05, right=0.95, wspace=0.2, hspace=0.35)

    def _format_metadata(self, idx):
        ts = self.acq_times[idx]
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        time_str = dt.strftime('%Y-%m-%dT%H:%M:%S UTC')
        sat_id = self.spacecraft_ids[idx]
        if isinstance(sat_id, bytes): 
            sat_id = sat_id.decode('ascii')
            
        return (f"FRAME INDEX: {idx:03d} / {self.num_frames-1:03d}\n"
                f"ACQUISITION: {time_str}\n"
                f"SPACECRAFT:  {sat_id}")

    def update_display(self):
        idx = self.current_frame
        metadata_str = self._format_metadata(idx)
        hud_flat = metadata_str.replace('\n', ' | ')
        
        self.ctrl_text.set_text(metadata_str)
        self.map_hud.set_text(hud_flat)
        
        # --- Image Processing (RGB) ---
        if self.grid_name == "LANDSAT":
            frame_sr = self.sr_dset[idx, ...]
            r = self.percentile_normalize_array(frame_sr[LANDSAT_RGB_BANDS[0]])
            g = self.percentile_normalize_array(frame_sr[LANDSAT_RGB_BANDS[1]])
            b = self.percentile_normalize_array(frame_sr[LANDSAT_RGB_BANDS[2]])
            rgb = np.nan_to_num(np.stack([r, g, b], axis=-1), nan=0.0)
        elif self.grid_name == "TANAGER":
            rgb = self.visuals[idx, ...]
            rgb = np.transpose(rgb, (1, 2, 0))
            rgb = rgb[...,:3]
        else:
            rgb = np.zeros((self.height, self.width, 3))

        # --- Extract Clustering Map ---
        clusters = self.fcm_dset[idx, ...].astype(float)
        
        # Replace -1 (NoData/Masked) with NaN so the colormap treats it as "bad"
        clusters[clusters == -1] = np.nan

        # --- Update Figures ---
        self.ax_rgb.clear()
        self.ax_rgb.imshow(rgb)
        self.ax_rgb.set_title("True Color Context")
        self.ax_rgb.axis('off')

        self.ax_cluster.clear()
        im_cluster = self.ax_cluster.imshow(
            clusters, 
            cmap=self.cmap, 
            vmin=-0.5, 
            vmax=self.num_clusters-0.5,
            interpolation='nearest'
        )
        self.ax_cluster.set_title(f"FCM Hard Clusters (k={self.num_clusters})")
        self.ax_cluster.axis('off')
        
        # Add discrete legend
        patches = [mpatches.Patch(color=self.cmap(i), label=f'Cluster {i}') for i in range(self.num_clusters)]
        patches.append(mpatches.Patch(color='black', label='Masked/NoData'))
        self.ax_cluster.legend(handles=patches, bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)

        # --- Update Cluster Feature Profiles ---
        self.ax_centers.clear()
        x_ticks = np.arange(len(self.cluster_features))
        
        # Plot each cluster's centroid across the feature dimensions
        for i in range(self.num_clusters):
            self.ax_centers.plot(
                x_ticks, self.cluster_centers[i, :], 
                marker='o', label=f'Cluster {i}', 
                color=self.cmap(i), linewidth=2, markersize=8
            )
            
        self.ax_centers.set_xticks(x_ticks)
        self.ax_centers.set_xticklabels(self.cluster_features, rotation=45, ha='right', fontweight='bold')
        self.ax_centers.set_title("Cluster Centroids (Average Feature Values)")
        self.ax_centers.set_ylabel("Feature Value")
        self.ax_centers.grid(True, alpha=0.4, linestyle='--')
        self.ax_centers.legend(loc='best')

        # --- Update Time Series ---
        self.ax_ts.clear()
        dt_times = [datetime.fromtimestamp(ts, tz=timezone.utc) for ts in self.acq_times]
        
        # Draw location markers on the maps and extract time series
        for loc in TS_LOCATIONS:
            y, x = loc['yx']
            if y < self.height and x < self.width:
                # Add bounding boxes on maps to locate exactly where the TS points are
                self.ax_rgb.plot(x, y, marker='s', markersize=10, markeredgecolor=loc['color'], markerfacecolor='none', markeredgewidth=2)
                self.ax_cluster.plot(x, y, marker='s', markersize=10, markeredgecolor=loc['color'], markerfacecolor='none', markeredgewidth=2)
                
                # Plot Time Series line
                vals = self.fcm_dset[:, y, x].astype(float)
                vals[vals == -1] = np.nan # Disregard masked values in line
                self.ax_ts.plot(dt_times, vals, marker='o', linestyle='-', color=loc['color'], 
                                label=loc['label'], markersize=6, alpha=0.8)

        # Plot vertical line indicating current frame
        self.ax_ts.axvline(dt_times[idx], color='black', linestyle='--', linewidth=2, label='Current Frame')
        
        # Add seasonal background spans
        if len(self.ax_ts.lines) > 0:
            xlims = self.ax_ts.get_xlim() # Capture limits generated by the data
            
            min_ts = min(self.acq_times)
            max_ts = max(self.acq_times)
            min_dt = datetime.fromtimestamp(min_ts, tz=timezone.utc)
            max_dt = datetime.fromtimestamp(max_ts, tz=timezone.utc)
            
            for yr in range(min_dt.year - 1, max_dt.year + 2):
                # Winter (Dec 1 prev year - Mar 1 curr year) -> light gray
                self.ax_ts.axvspan(datetime(yr - 1, 12, 1, tzinfo=timezone.utc), 
                                   datetime(yr, 3, 1, tzinfo=timezone.utc), 
                                   color='lightgray', alpha=0.3, zorder=0, lw=0)
                # Spring (Mar 1 - Jun 1) -> light green
                self.ax_ts.axvspan(datetime(yr, 3, 1, tzinfo=timezone.utc), 
                                   datetime(yr, 6, 1, tzinfo=timezone.utc), 
                                   color='lightgreen', alpha=0.2, zorder=0, lw=0)
                # Summer (Jun 1 - Sep 1) -> light yellow
                self.ax_ts.axvspan(datetime(yr, 6, 1, tzinfo=timezone.utc), 
                                   datetime(yr, 9, 1, tzinfo=timezone.utc), 
                                   color='lightyellow', alpha=0.3, zorder=0, lw=0)
                # Fall (Sep 1 - Dec 1) -> light orange
                self.ax_ts.axvspan(datetime(yr, 9, 1, tzinfo=timezone.utc), 
                                   datetime(yr, 12, 1, tzinfo=timezone.utc), 
                                   color='orange', alpha=0.15, zorder=0, lw=0)
            
            self.ax_ts.set_xlim(xlims) # Restore limits so it doesn't zoom out to empty seasons

        self.ax_ts.set_title("Cluster Classification Over Time")
        self.ax_ts.set_ylabel("Cluster Index")
        self.ax_ts.set_yticks(range(self.num_clusters))
        self.ax_ts.grid(True, alpha=0.3, linestyle='--')
        self.ax_ts.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        self.ax_ts.tick_params(axis='x', rotation=45)
        self.ax_ts.legend(loc='center left', bbox_to_anchor=(1.01, 0.5))

        # Redraw canvases
        self.fig_controls.canvas.draw_idle()
        self.fig_maps.canvas.draw_idle()

    def percentile_normalize_array(self, arr, lower_percentile=1, upper_percentile=99):
        '''Normalizes a numpy array to the range [0, 1] using percentiles, ignoring NaNs.'''
        if np.all(np.isnan(arr)):
            return np.zeros_like(arr)
            
        p_low, p_high = np.nanpercentile(arr, (lower_percentile, upper_percentile))
        
        if p_low == p_high:
            return np.zeros_like(arr)
            
        norm_arr = exposure.rescale_intensity(arr, in_range=(p_low, p_high), out_range=(0, 1))
        return norm_arr.clip(0, 1)

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

    def _on_save_images(self, event):
        idx = self.current_frame
        ts = self.acq_times[idx]
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        time_str = dt.strftime('%Y%m%d_%H%M%S')
        
        path = os.path.join(self.save_dir, f"{time_str}_frame_{idx:03d}_FCM_Map.png")
        self.fig_maps.savefig(path, dpi=400, bbox_inches='tight')
        print(f"Saved: {path}")

    def run(self): 
        plt.show()

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    
    print("Please select the HDF5 file resulting from the FCM clustering...")
    file_path = filedialog.askopenfilename(
        title="Select FCM Clustering HDF5", 
        filetypes=[("HDF5 files", "*.h5")]
    )
    
    if file_path: 
        try:
            viewer = FCMViewer(file_path)
            viewer.run()
        except Exception as e:
            print(f"Error loading viewer: {e}")
            
    root.destroy()