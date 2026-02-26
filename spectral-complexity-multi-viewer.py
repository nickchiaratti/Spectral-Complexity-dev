import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.widgets import Button, TextBox
from datetime import datetime, timezone
import tkinter as tk
from tkinter import filedialog
from scipy.spatial import ConvexHull
from skimage import exposure

# --- Configuration ---
# Standard Landsat 8/9 True Color Indices: [C(0), B(1), G(2), R(3), NIR(4), S1(5), S2(6)]
LANDSAT_RGB_BANDS = (3, 2, 1) 
# Default Projection Bands for 3D Hull (Indices)
HULL_BANDS_LANDSAT = (6, 4, 2) 
HULL_BANDS_TANAGER = (100, 50, 20) # Example hyperspectral indices

# Time Series Locations (y, x)
TS_LOCATIONS = [
    {'yx': (47, 31),  'label': "Forested region",                    'color': 'tab:green'},
    {'yx': (87, 128), 'label': "Artificial turf football field",     'color': 'tab:blue'},
    {'yx': (82, 123), 'label': "Recently added artificial turf field",'color': 'tab:cyan'},
    {'yx': (67, 28),  'label': "Tait Parking Lot",          'color': 'tab:red'}
]

LANDSAT_STRICT_QA = True
DISPLAY_NORMALIZATION = False

SAVE_DIR = "C:/satelliteImagery/MultiSensor_Analysis"
if LANDSAT_STRICT_QA:
    SAVE_DIR += "_Strict_QA"

class MultiComplexityViewer:
    def __init__(self, file_paths):
        self.files = []
        self.all_frames = []
        
        # 1. Load and Parse both files
        for path in file_paths:
            h5 = h5py.File(path, 'r')
            source_name = list(h5['/HDFEOS/GRIDS'].keys())[0]
            data_grp = h5[f'HDFEOS/GRIDS/{source_name}/Data Fields']
            
            sr_dset = data_grp['surface_reflectance']
            acq_times = sr_dset.attrs.get('acquisition_time')
            sat_ids = sr_dset.attrs.get('spacecraft_id')
            
            # Retrieve and Scale Wavelengths
            # TANAGER: nm -> um (divide by 1000)
            # LANDSAT: um (keep as is)
            raw_wl = sr_dset.attrs.get('wavelengths')
            if raw_wl is not None:
                if source_name == 'TANAGER':
                    wavelengths = raw_wl[:] / 1000.0
                else:
                    wavelengths = raw_wl[:]
            else:
                wavelengths = np.arange(sr_dset.shape[1])
            
            num_frames = sr_dset.shape[0]
            
            file_info = {
                'path': path,
                'h5': h5,
                'source': source_name,
                'data_grp': data_grp,
                'wavelengths': wavelengths
            }
            self.files.append(file_info)

            # Extract individual frames into a flat list for interleaving
            for i in range(num_frames):
                sat_id = sat_ids[i]
                if isinstance(sat_id, bytes): sat_id = sat_id.decode('ascii')
                
                self.all_frames.append({
                    'timestamp': acq_times[i],
                    'file_idx': len(self.files) - 1,
                    'frame_idx': i,
                    'source': source_name,
                    'sat_id': sat_id
                })

        # 2. Interleave frames by acquisition time
        self.all_frames.sort(key=lambda x: x['timestamp'])
        self.num_total_frames = len(self.all_frames)
        self.current_idx = 0

        # 2b. Pre-load Time Series for Multiple Pixels
        # Structure: Sensor -> Label -> Lists
        self.ts_data = {
            'LANDSAT': {loc['label']: {'t': [], 'v': []} for loc in TS_LOCATIONS},
            'TANAGER': {loc['label']: {'t': [], 'v': []} for loc in TS_LOCATIONS}
        }
        
        print(f"Extracting time series for {len(TS_LOCATIONS)} locations...")
        for frame in self.all_frames:
            file_info = self.files[frame['file_idx']]
            dgrp = file_info['data_grp']
            if 'sliding_volume_map' in dgrp:
                try:
                    dset = dgrp['sliding_volume_map']
                    src = frame['source']
                    key = 'LANDSAT' if 'LANDSAT' in src.upper() else 'TANAGER'
                    dt = datetime.fromtimestamp(frame['timestamp'], tz=timezone.utc)
                    
                    for loc in TS_LOCATIONS:
                        y, x = loc['yx']
                        # Check bounds
                        if dset.shape[1] > y and dset.shape[2] > x:
                            val = dset[frame['frame_idx'], y, x]
                            # Only add if valid number
                            if not np.isnan(val):
                                self.ts_data[key][loc['label']]['t'].append(dt)
                                self.ts_data[key][loc['label']]['v'].append(val)
                except Exception as e:
                    print(f"TS Extraction Error frame {frame['frame_idx']}: {e}")

        self.save_dir = SAVE_DIR

        # Persistent Plotting objects
        self.im_slide = None
        self.cbar_slide = None

        # 3. Initialize UI
        self._init_control_ui()
        self._init_combined_ui() # Combined 2x3 Layout
        self._init_hull_ui()
        
        self.update_display()

    def _init_control_ui(self):
        self.fig_controls = plt.figure(figsize=(6, 5))
        self.fig_controls.canvas.manager.set_window_title("Timeline Navigation")
        self.ax_meta = self.fig_controls.add_axes([0, 0, 1, 1]); self.ax_meta.axis('off')
        self.ctrl_text = self.ax_meta.text(0.5, 0.85, "", ha='center', va='center', 
                                         fontsize=10, family='monospace',
                                         bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
        
        # Navigation Row
        ax_prev = self.fig_controls.add_axes([0.1, 0.55, 0.25, 0.1])
        ax_next = self.fig_controls.add_axes([0.65, 0.55, 0.25, 0.1])
        ax_input = self.fig_controls.add_axes([0.45, 0.55, 0.1, 0.1])
        
        self.btn_prev = Button(ax_prev, '<< Prev')
        self.btn_next = Button(ax_next, 'Next >>')
        self.txt_input = TextBox(ax_input, 'Go: ', initial='0')
        
        # Single Save Row
        ax_save = self.fig_controls.add_axes([0.3, 0.40, 0.4, 0.1])
        self.btn_save = Button(ax_save, 'Save Current', color='lightgreen')

        # Auto Save Row
        self.ax_meta.text(0.5, 0.28, "--- Batch Processing ---", ha='center', va='center', fontsize=9)
        
        ax_start = self.fig_controls.add_axes([0.2, 0.15, 0.15, 0.08])
        ax_end = self.fig_controls.add_axes([0.5, 0.15, 0.15, 0.08])
        ax_auto = self.fig_controls.add_axes([0.3, 0.03, 0.4, 0.1])

        self.txt_start = TextBox(ax_start, 'Start: ', initial='0')
        self.txt_end = TextBox(ax_end, 'End: ', initial=str(self.num_total_frames-1))
        self.btn_auto = Button(ax_auto, 'Auto Save Range', color='lightblue')
        
        self.btn_prev.on_clicked(self._on_prev)
        self.btn_next.on_clicked(self._on_next)
        self.txt_input.on_submit(self._on_submit)
        self.btn_save.on_clicked(self._on_save_images)
        self.btn_auto.on_clicked(self._on_auto_save)

    def _init_combined_ui(self):
        """Combines Fingerprint and Heatmaps into a single 2x3 figure."""
        self.fig_combined = plt.figure(figsize=(18, 10))
        self.fig_combined.canvas.manager.set_window_title("Comprehensive Complexity Analysis")
        
        # Metadata HUD (Heads-Up Display)
        self.combined_hud = self.fig_combined.text(0.5, 0.96, "", ha='center', fontsize=11, 
                                                  bbox=dict(facecolor='white', alpha=0.8, edgecolor='lightgray'))
        
        # Row 1: Analysis (Spatial, Spectral, Curve)
        self.ax_spatial = self.fig_combined.add_subplot(231)
        self.ax_spectral = self.fig_combined.add_subplot(232)
        self.ax_vol_curve = self.fig_combined.add_subplot(233)
        
        # Row 2: Maps (Slide Map, Combined Time Series)
        self.ax_slide_map = self.fig_combined.add_subplot(234)
        
        # Combined TS Plot (Spans 2 columns: 5 and 6)
        self.ax_ts_main = self.fig_combined.add_subplot(2, 3, (5, 6))
        self.ax_ts_twin = self.ax_ts_main.twinx()
        
        plt.subplots_adjust(top=0.9, bottom=0.05, left=0.05, right=0.95, hspace=0.25, wspace=0.2)

    def _init_hull_ui(self):
        self.fig_hull = plt.figure(figsize=(8, 7))
        self.fig_hull.canvas.manager.set_window_title("3D Convex Hull Projection")
        self.ax_hull = self.fig_hull.add_subplot(111, projection='3d')

    def _format_metadata(self, frame_info):
        dt = datetime.fromtimestamp(frame_info['timestamp'], tz=timezone.utc)
        return (f"TIMELINE:   {self.current_idx + 1} / {self.num_total_frames}\n"
                f"SOURCE:     {frame_info['source']}\n"
                f"SPACECRAFT: {frame_info['sat_id']}\n"
                f"ACQUIRED:   {dt.strftime('%Y-%m-%d %H:%M:%S UTC')}")

    def update_display(self):
        frame_info = self.all_frames[self.current_idx]
        file_info = self.files[frame_info['file_idx']]
        f_idx = frame_info['frame_idx']
        data_grp = file_info['data_grp']

        # Update metadata text
        meta_str = self._format_metadata(frame_info)
        self.ctrl_text.set_text(meta_str)
        # Update HUD text (flattened)
        self.combined_hud.set_text(meta_str.replace('\n', ' | '))

        # --- RGB Generation ---
        sr_data = data_grp['surface_reflectance'][f_idx, ...]
        if frame_info['source'] == 'LANDSAT':
            r = percentile_normalize_array(sr_data[LANDSAT_RGB_BANDS[0]])
            g = percentile_normalize_array(sr_data[LANDSAT_RGB_BANDS[1]])
            b = percentile_normalize_array(sr_data[LANDSAT_RGB_BANDS[2]])
            rgb = np.nan_to_num(np.stack([r, g, b], axis=-1), nan=0.0)
            hull_bands = HULL_BANDS_LANDSAT
        else: # TANAGER
            vis = data_grp['ortho_visual'][f_idx, ...]
            rgb = np.transpose(vis[:3, ...], (1, 2, 0)) # Drop alpha
            hull_bands = HULL_BANDS_TANAGER

        # --- Row 1: Spatial & Spectral Analysis ---
        self.ax_spatial.clear()
        self.ax_spatial.imshow(rgb)
        em_indices = data_grp['frame_endmember_indices'][f_idx]
        h, w = sr_data.shape[1:]
        for i, flat_idx in enumerate(em_indices):
            row, col = flat_idx // w, flat_idx % w
            self.ax_spatial.plot(col, row, 'r+', markersize=8)
            self.ax_spatial.annotate(f'V{i}', (col, row), color='yellow', fontsize=8, fontweight='bold')
        
        # Plot Time Series Locations
        for loc in TS_LOCATIONS:
            y, x = loc['yx']
            self.ax_spatial.plot(x, y, marker='s', markersize=10, markeredgecolor=loc['color'], 
                                 markerfacecolor='none', markeredgewidth=1.5, linestyle='None')

        self.ax_spatial.set_title(f"EM Locations ({frame_info['source']})")
        self.ax_spatial.axis('off')

        self.ax_spectral.clear()
        endmembers = data_grp['frame_endmembers'][f_idx, ...]
        wl = file_info['wavelengths']
        for i in range(endmembers.shape[1]):
            if not np.all(np.isnan(endmembers[:, i])) and np.any(endmembers[:, i] != 0):
                self.ax_spectral.plot(wl, endmembers[:, i], label=f'V{i}', lw=1)
        self.ax_spectral.set_title("Spectral Signatures")
        self.ax_spectral.set_xlabel("Wavelength (μm)") # Standardized label
        self.ax_spectral.set_ylabel("Reflectance")
        self.ax_spectral.set_ylim(0, 1)
        self.ax_spectral.legend(loc='upper right')
        self.ax_spectral.grid(True, alpha=0.3)

        self.ax_vol_curve.clear()
        vols = data_grp['frame_endmember_volumes'][f_idx]
        self.ax_vol_curve.plot(np.arange(1, len(vols)+1), np.pad(vols[2:], (2,0), 'constant', constant_values=0), 'o-', markersize=4, color='green')
        self.ax_vol_curve.set_title("Complexity Curve")
        self.ax_vol_curve.set_xlabel("Endmember Count")
        self.ax_vol_curve.set_ylabel("Volume")
        self.ax_vol_curve.grid(True, alpha=0.2)

        # --- Row 2: Maps and Time Series ---
        curr_dt = datetime.fromtimestamp(frame_info['timestamp'], tz=timezone.utc)

        # 1. Sliding Complexity Map (Bottom-Left)
        def update_map(ax, dset, im_attr, cbar_attr, title):
            data = dset[f_idx]
            if DISPLAY_NORMALIZATION:
                data = percentile_normalize_array(data)
            
            # Use current instance attributes for persistent colorbars
            curr_im = getattr(self, im_attr)
            curr_cbar = getattr(self, cbar_attr)

            if curr_im is None:
                new_im = ax.imshow(data, cmap='viridis')
                setattr(self, im_attr, new_im)
                new_cbar = self.fig_combined.colorbar(new_im, ax=ax, fraction=0.046, pad=0.04)
                setattr(self, cbar_attr, new_cbar)
                
                # Draw unfilled squares for TS Locations
                for loc in TS_LOCATIONS:
                    y, x = loc['yx']
                    ax.plot(x, y, marker='s', markersize=10, markeredgecolor=loc['color'], 
                            markerfacecolor='none', markeredgewidth=1.5, linestyle='None')
                
                ax.set_title(title)
                ax.axis('off')
            else:
                curr_im.set_data(data)
                curr_im.set_clim(vmin=np.nanmin(data), vmax=np.nanmax(data))
                curr_cbar.update_normal(curr_im)

        if 'sliding_volume_map' in data_grp:
            update_map(self.ax_slide_map, data_grp['sliding_volume_map'], 'im_slide', 'cbar_slide', "Sliding Complexity")
        
        # 2. Combined Time Series Plot (Bottom-Center/Right)
        self.ax_ts_main.clear()
        self.ax_ts_twin.clear()
        
        # Plot LANDSAT on Main (Left Axis)
        for loc in TS_LOCATIONS:
            label = loc['label']
            data = self.ts_data['LANDSAT'][label]
            if data['t']:
                self.ax_ts_main.plot(data['t'], data['v'], marker='^', color=loc['color'], label=label,
                                      markersize=4, linestyle='--', linewidth=1, alpha=0.6)
        
        # Plot TANAGER on Twin (Right Axis)
        for loc in TS_LOCATIONS:
            label = loc['label']
            data = self.ts_data['TANAGER'][label]
            if data['t']:
                self.ax_ts_twin.plot(data['t'], data['v'], marker='s', color=loc['color'], label=label,
                                      markersize=5, linestyle='-', linewidth=1.5, alpha=0.9)

        self.ax_ts_main.set_title("Time Series")
        
        # Setup Left Axis (LANDSAT)
        self.ax_ts_main.set_ylabel("LANDSAT Volume", color='tab:blue', fontweight='bold')
        self.ax_ts_main.tick_params(axis='y', labelcolor='tab:blue')
        self.ax_ts_main.grid(True, alpha=0.3)
        self.ax_ts_main.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        self.ax_ts_main.tick_params(axis='x', rotation=45, labelsize=8)
        
        # Setup Right Axis (TANAGER)
        self.ax_ts_twin.set_ylabel("TANAGER Volume", color='tab:red', fontweight='bold')
        self.ax_ts_twin.yaxis.set_label_position("right")
        self.ax_ts_twin.tick_params(axis='y', labelcolor='tab:red')

        # Combined Vertical Line
        self.ax_ts_main.axvline(curr_dt, color='black', linestyle='--', alpha=0.8, linewidth=1.5)
        
        self.ax_ts_main.legend(loc='upper left')

        # --- 3D Hull Figure ---
        self.ax_hull.clear()
        pixel_data = sr_data.reshape(sr_data.shape[0], -1).T
        valid_mask = ~np.isnan(pixel_data).any(axis=1)
        pixel_data = pixel_data[valid_mask]
        
        # Subsample for UI speed
        if pixel_data.shape[0] > 1500:
            pixel_data = pixel_data[np.random.choice(pixel_data.shape[0], 1500, replace=False)]
        
        # Ensure hull bands are within range
        b1, b2, b3 = [min(b, sr_data.shape[0]-1) for b in hull_bands]
        
        self.ax_hull.scatter(pixel_data[:, b1], pixel_data[:, b2], pixel_data[:, b3], c='gray', alpha=0.1, s=1)
        
        em_xyz = endmembers[[b1, b2, b3], :4].T
        self.ax_hull.scatter(em_xyz[:, 0], em_xyz[:, 1], em_xyz[:, 2], c='red', s=40, label='Endmembers')
        
        try:
            hull = ConvexHull(em_xyz)
            for s in hull.simplices:
                self.ax_hull.plot(em_xyz[s, 0], em_xyz[s, 1], em_xyz[s, 2], 'r-', alpha=0.3)
        except: pass
        
        self.ax_hull.set_title(f"3D Scatter: Bands {b1}, {b2}, {b3}")
        self.ax_hull.set_xlabel(f"B{b1}"); self.ax_hull.set_ylabel(f"B{b2}"); self.ax_hull.set_zlabel(f"B{b3}")

        # Refresh
        for f in [self.fig_controls, self.fig_combined, self.fig_hull]:
            f.canvas.draw_idle()

    def _on_prev(self, event):
        if self.current_idx > 0:
            self.current_idx -= 1
            self.update_display()
            self.txt_input.set_val(str(self.current_idx))

    def _on_next(self, event):
        if self.current_idx < self.num_total_frames - 1:
            self.current_idx += 1
            self.update_display()
            self.txt_input.set_val(str(self.current_idx))

    def _on_submit(self, text):
        try:
            val = int(text)
            if 0 <= val < self.num_total_frames:
                self.current_idx = val
                self.update_display()
        except: self.txt_input.set_val(str(self.current_idx))

    def _on_auto_save(self, event):
        try:
            start_idx = int(self.txt_start.text)
            end_idx = int(self.txt_end.text)
            
            # Clamp to valid range
            start_idx = max(0, start_idx)
            end_idx = min(self.num_total_frames - 1, end_idx)
            
            if start_idx > end_idx:
                print("Start frame must be less than or equal to End frame.")
                return

            print(f"Starting Batch Save from {start_idx} to {end_idx}...")
            
            for i in range(start_idx, end_idx + 1):
                self.current_idx = i
                self.update_display()
                self.txt_input.set_val(str(self.current_idx))
                # Force update to ensure plots are rendered before saving
                plt.pause(0.2) 
                self._on_save_images(None)
                
            print("Batch Save Complete.")
            
        except ValueError:
            print("Invalid Start or End frame index.")

    def _on_save_images(self, event):
        info = self.all_frames[self.current_idx]
        file_info = self.files[info['file_idx']]
        data_grp = file_info['data_grp']
        
        # Determine output directory based on attributes or default
        save_path = self.save_dir
        if 'frame_endmember_volumes' in data_grp:
            try:
                vol_dset = data_grp['frame_endmember_volumes']
                num_em = vol_dset.attrs.get('num_endmembers', 'X')
                gram = vol_dset.attrs.get('gram_type', 'X')
                norm = vol_dset.attrs.get('Normalization', 'None')
                # Handle possible byte strings or None
                if hasattr(norm, 'decode'): norm = norm.decode('utf-8')
                if norm is None: norm = "None"
                
                # Construct path: C:/satelliteImagery/LANDSAT_TANAGER_EM-{...}
                new_dir = f"{SAVE_DIR}_EM-{num_em}_Gram-{gram}_Norm-{norm}/"
                os.makedirs(new_dir, exist_ok=True)
                save_path = new_dir
            except Exception as e:
                print(f"Error constructing dynamic path, using default: {e}")

        time_str = datetime.fromtimestamp(info['timestamp'], tz=timezone.utc).strftime('%Y%m%d_%H%M%S')
        prefix = f"{time_str}_{info['source']}_{self.current_idx:02d}"
        if LANDSAT_STRICT_QA:
            prefix += "_Strict_QA"
        for fig, name in [(self.fig_combined, "CombinedAnalysis")]: #, (self.fig_hull, "Hull")]:
            path = os.path.join(save_path, f"{prefix}_{name}.png")
            fig.savefig(path, dpi=300)
            print(f"Saved: {path}")

    def run(self): plt.show()

def percentile_normalize_array(arr, low=2, high=98):
    if np.all(np.isnan(arr)): return np.zeros_like(arr)
    p_low, p_high = np.nanpercentile(arr, (low, high))
    if p_low == p_high: return np.zeros_like(arr)
    return exposure.rescale_intensity(arr, in_range=(p_low, p_high), out_range=(0, 1)).clip(0, 1)

if __name__ == "__main__":
    root = tk.Tk(); root.withdraw()
    print("Select LANDSAT HDF5...")
    #l_path = filedialog.askopenfilename(title="Select Landsat HDF5", filetypes=[("HDF5", "*.h5")])
    l_path = "C:/satelliteImagery/LANDSAT/Tait/LANDSAT_Stack_Tait_HDFEOS_SC_EM-7_Gram-corrected_Norm-None_QA-Strict.h5"

    print("Select TANAGER HDF5...")
    #t_path = filedialog.askopenfilename(title="Select Tanager HDF5", filetypes=[("HDF5", "*.h5")])
    t_path = "C:/satelliteImagery/Tanager/Tait/Tanager_Stack_Tait_HDFEOS_SC_EM-7_Gram-corrected_Norm-None_QA-Loose.h5"
    
    if l_path and t_path:
        viewer = MultiComplexityViewer([l_path, t_path])
        viewer.run()
    else:
        print("Selection cancelled.")
    root.destroy()