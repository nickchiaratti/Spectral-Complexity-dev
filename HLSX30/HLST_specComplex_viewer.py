import os
import platform
# Monkeypatch platform._wmi_query to raise OSError immediately, bypassing Windows WMI hangs/KeyErrors in multiprocessing child processes
def _dummy_wmi_query(*args, **kwargs):
    raise OSError("WMI disabled to prevent hangs")
platform._wmi_query = _dummy_wmi_query

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.widgets import Button, TextBox, CheckButtons, RadioButtons
import matplotlib.gridspec as gridspec
from datetime import datetime, timezone
import tkinter as tk
from tkinter import filedialog
from scipy.stats import pearsonr, spearmanr, norm, skew, kurtosis
from zoneinfo import ZoneInfo
import rasterio.transform
import rasterio.transform
from pyproj import Transformer, CRS
import yaml

# Load Configuration
try:
    from pathlib import Path
    script_dir = Path(__file__).resolve().parent
    with open(os.path.join(script_dir, "locations_config.yaml"), "r") as f:
        config_data = yaml.safe_load(f)
    Location = config_data.get("current_run", {}).get("location", "Tait")
except Exception:
    Location = "Tait"

# --- Configuration ---
complexity_type = 'sliding_volume_z_score' # or 'sliding_volume_map'
HULL_BANDS_LANDSAT = (6, 5, 4) 
HULL_BANDS_TANAGER = (100, 50, 20) 

COMPLEXITY_DICT = {
    'sliding_volume_map': 'Spectral Complexity',
    'sliding_volume_z_score': 'Spectral Complexity Z-Score',
    'sliding_volume_z_score_masked': 'Spectral Complexity Z-Score',
    'sliding_volume_local_z_score': 'Spectral Complexity Local Z-Score',
    'sliding_volume_map_5x5': 'Spectral Complexity 5x5 window',
    'sliding_volume_map_7x7': 'Spectral Complexity 7x7 window',
}
LOG_SCALE = ('map' in complexity_type)
START_YEAR = 2022
END_YEAR = 2025
TS_START_DATE = datetime(START_YEAR, 1, 1, tzinfo=timezone.utc)
TS_END_DATE = datetime(END_YEAR, 12, 31, tzinfo=timezone.utc)
TWIN_Y_AXIS_DEFAULT = False
MASKING = True # In Harmonized viewer, masking relies on pre-computed common_mask

# Example default path (can be overridden by file dialog)
default_harmonized_path = f"C:/satelliteImagery/HLST30/HLST_{Location}_Harmonized_SC_EM-7_Norm-bandCount.h5"

suffix = ''
if complexity_type == 'sliding_volume_z_score':
    suffix = '_zscore'
elif complexity_type == 'sliding_volume_map':
    suffix = '_SpecComplex'
if not MASKING:
    suffix += '_unmasked'
suffix += f'_{END_YEAR-START_YEAR}yr' if START_YEAR != 2025 else '_2025'

SAVE_DIR = f"C:/satelliteImagery/MultiSensor_Analysis_{Location}_Harmonized" + suffix

# Predefined Time Series Locations Map (Latitude, Longitude)
TS_LOCATIONS_MAP = {
    "Tait": [
        {'latlon': (43.13927, -77.50340), 'label': "ROCX NITE Tarp",                  'color': 'tab:purple'},
        {'latlon': (43.142856, -77.508451), 'label': "West Tait Forest",                'color': 'tab:green'},
        {'latlon': (43.144861, -77.501176), 'label': "East Tait Forest",                'color': 'tab:olive'},
        {'latlon': (43.151502, -77.485518), 'label': "Shadow Pines Grass Field",         'color': 'tab:red'},
        {'latlon': (43.151219, -77.486637), 'label': "Shadow Pines Pickleball Court",    'color': 'tab:blue'},
        {'latlon': (43.151877, -77.487111), 'label': "Shadow Pines Playground",          'color': 'tab:cyan'},
    ],
    "Rochesterv2": [
        {'latlon': (43.13927, -77.50340), 'label': "ROCX NITE Tarp",                  'color': 'tab:purple'},
        {'latlon': (43.142856, -77.508451), 'label': "West Tait Forest",                'color': 'tab:green'},
        {'latlon': (43.144861, -77.501176), 'label': "East Tait Forest",                'color': 'tab:olive'},
        {'latlon': (43.151502, -77.485518), 'label': "Shadow Pines Grass Field",         'color': 'tab:red'},
        {'latlon': (43.151219, -77.486637), 'label': "Shadow Pines Pickleball Court",    'color': 'tab:blue'},
        {'latlon': (43.151877, -77.487111), 'label': "Shadow Pines Playground",          'color': 'tab:cyan'},
    ],
    "Malibu": [
        {'latlon': (34.059168, -118.573950), 'label': "Parker Mesa Overlook",                      'color': 'tab:purple'},
        {'latlon': (34.058990, -118.613110), 'label': "Tuna Canyon",             'color': 'tab:green'},
        {'latlon': (34.047931, -118.572716), 'label': "Surfwood Rd",            'color': 'tab:olive'},
        {'latlon': (34.053249, -118.557091), 'label': "Paseo Miramar Viewpoint",                        'color': 'tab:cyan'},
    ],
    "Palisades": [
        {'latlon': (34.05, -118.53), 'label': "Pacific Palisades",                 'color': 'tab:purple'},
        {'latlon': (34.01, -118.49), 'label': "Santa Monica Pier",                 'color': 'tab:green'},
        {'latlon': (34.09, -118.59), 'label': "Topanga State Park",                'color': 'tab:olive'},
    ],
    "MtEtna": [
        {'latlon': (37.738, 14.970), 'label': "Etna West",                         'color': 'tab:green'},
        {'latlon': (37.710, 15.000), 'label': "Etna South",                        'color': 'tab:purple'},
        {'latlon': (37.738, 15.040), 'label': "Etna East",                         'color': 'tab:olive'},
        {'latlon': (37.795, 15.005), 'label': "Etna North",                        'color': 'tab:blue'},
    ],
    "MtEtna-Catania": [
        {'latlon': (37.738, 14.970), 'label': "Etna West",                         'color': 'tab:green'},
        {'latlon': (37.710, 15.000), 'label': "Etna South",                        'color': 'tab:purple'},
        {'latlon': (37.738, 15.040), 'label': "Etna East",                         'color': 'tab:olive'},
        {'latlon': (37.795, 15.005), 'label': "Etna North",                        'color': 'tab:blue'},
    ],
    "BuenosAires": [
        {'latlon': (-34.60, -58.38), 'label': "Buenos Aires Central",              'color': 'tab:purple'},
        {'latlon': (-34.57, -58.42), 'label': "Palermo Woods",                     'color': 'tab:green'},
        {'latlon': (-34.81, -58.53), 'label': "Ezeiza Airport",                    'color': 'tab:olive'},
    ]
}

# Time Series Locations (Latitude, Longitude)
TS_LOCATIONS = TS_LOCATIONS_MAP["Tait"]

DISPLAY_NORMALIZATION = True
DISPLAY_REDUNDANT_FIGURE = True 

class HarmonizedComplexityViewer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.h5 = h5py.File(file_path, 'r')
        
        self.harm_grp = self.h5['/HDFEOS/GRIDS/HARMONIZED/Data Fields']
        
        # Determine base array dataset to extract timeline
        if complexity_type in self.harm_grp:
            self.base_dset = self.harm_grp[complexity_type]
        else:
            self.base_dset = self.harm_grp['sliding_volume_map']
            
        self.common_mask_dset = self.harm_grp['common_mask']
            
        self.total_frames = self.base_dset.shape[0]
        self.height = self.base_dset.shape[1]
        self.width = self.base_dset.shape[2]
        
        # Read Provenance Data
        self.prov_grid = [x.decode('utf-8') if isinstance(x, bytes) else str(x) for x in self.base_dset.attrs['source_grid']]
        self.prov_space = [x.decode('utf-8') if isinstance(x, bytes) else str(x) for x in self.base_dset.attrs['source_spacecraft']]
        self.prov_time = self.base_dset.attrs['acquisition_time']
        self.prov_idx = self.base_dset.attrs['source_frame_index']
        
        # Extract sensor-specific wavelengths for plotting
        self.wavelengths = {}
        for g in np.unique(self.prov_grid):
            data_grp = self.h5[f'/HDFEOS/GRIDS/{g}/Data Fields']
            if 'wavelengths' in data_grp.attrs:
                raw_wl = data_grp.attrs['wavelengths'][:]
            else:
                raw_wl = self.h5[f'/HDFEOS/GRIDS/{g}/Data Fields/surface_reflectance'].attrs['wavelengths'][:]
            if 'TANAGER' in g.upper():
                self.wavelengths[g] = raw_wl / 1000.0
            else:
                self.wavelengths[g] = raw_wl

        # Find specific indices for scatter plot reference
        self.l_file_idx = next((i for i, g in enumerate(self.prov_grid) if 'HLS' in g.upper()), None)
        self.t_file_idx = next((i for i, g in enumerate(self.prov_grid) if 'TANAGER' in g.upper()), None)
        
        # Map Geographic Coordinates to Pixel Coordinates
        data_grp0 = self.h5[f'/HDFEOS/GRIDS/{self.prov_grid[0]}/Data Fields']
        if 'GeoTransform' in data_grp0.attrs:
            geo_transform = data_grp0.attrs['GeoTransform']
            spatial_ref = data_grp0.attrs['spatial_ref']
        else:
            sr_attrs = data_grp0['surface_reflectance'].attrs
            geo_transform = sr_attrs['GeoTransform']
            spatial_ref = sr_attrs['spatial_ref']
            
        if isinstance(spatial_ref, bytes):
            spatial_ref = spatial_ref.decode('utf-8')
            
        crs = CRS.from_wkt(spatial_ref)
        transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        
        affine = rasterio.transform.Affine.from_gdal(*geo_transform)
        inv_affine = ~affine
        
        # Determine location dynamically from the opened HDF5 file's name
        filename = os.path.basename(file_path)
        parts = filename.split('_')
        resolved_location = Location
        if len(parts) > 1 and parts[0] == "HLST":
            resolved_location = parts[1]
            
        global TS_LOCATIONS
        if resolved_location in TS_LOCATIONS_MAP:
            TS_LOCATIONS = TS_LOCATIONS_MAP[resolved_location]
        else:
            # Fallback coordinate: Compute center of the image using project crs and affine
            ul_x, ul_y = affine * (0, 0)
            lr_x, lr_y = affine * (self.width, self.height)
            transformer_back = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
            center_x = (ul_x + lr_x) / 2.0
            center_y = (ul_y + lr_y) / 2.0
            center_lon, center_lat = transformer_back.transform(center_x, center_y)
            TS_LOCATIONS = [{'latlon': (center_lat, center_lon), 'label': f"Grid Center ({resolved_location})", 'color': 'tab:purple'}]

        print("\n--- Coordinate Mapping ---")
        for loc in TS_LOCATIONS:
            lat, lon = loc['latlon']
            proj_x, proj_y = transformer.transform(lon, lat)
            px, py = inv_affine * (proj_x, proj_y)
            loc['yx'] = (int(round(py)), int(round(px)))
            print(f"Mapped [{loc['label']}] Lat/Lon ({lat:.4f}, {lon:.4f}) -> Pixel (y={loc['yx'][0]}, x={loc['yx'][1]})")

        self.current_idx = 0
        self.save_dir = SAVE_DIR.replace(Location, resolved_location)

        self.ts_start_date = TS_START_DATE
        self.ts_end_date = TS_END_DATE
        self.use_twin_axis = TWIN_Y_AXIS_DEFAULT
        self.localization_mode = 'general'
        
        self.im_slide = None
        self.cbar_slide = None
        self.fig_scatter = None
        self.im_slide_redundant = None
        self.cbar_slide_redundant = None
        self.ax_ts_twin = None
        self.ax_ts_redundant_twin = None

        self._recompute_time_series()
        self._init_control_ui()
        self._init_combined_ui()
        if DISPLAY_REDUNDANT_FIGURE:
            self._init_redundant_ui()
            self._init_transect_ui()
        self._init_hull_ui()
        
        self.update_display()

    def _recompute_time_series(self):
        print(f"Loading pre-computed spatial masks for time series data...")
        self.ts_data = {
            'LANDSAT': {loc['label']: {'t': [], 'v': []} for loc in TS_LOCATIONS},
            'SENTINEL': {loc['label']: {'t': [], 'v': []} for loc in TS_LOCATIONS},
            'TANAGER': {loc['label']: {'t': [], 'v': []} for loc in TS_LOCATIONS}
        }
        
        # Load entirely into memory for fast extraction
        all_comp = self.base_dset[:]
        all_mask = self.common_mask_dset[:] if MASKING else np.ones_like(all_comp, dtype=bool)
        
        for loc in TS_LOCATIONS:
            y, x = loc['yx']
            if 0 <= y < self.height and 0 <= x < self.width:
                # Extract 1D array over time
                vals = all_comp[:, y, x]
                masks = all_mask[:, y, x] == 0
                
                for i in range(self.total_frames):
                    if masks[i] and not np.isnan(vals[i]):
                        dt = datetime.fromtimestamp(self.prov_time[i], tz=timezone.utc)
                        grid_upper = self.prov_grid[i].upper()
                        if 'HLSL30' in grid_upper: key = 'LANDSAT'
                        elif 'HLSS30' in grid_upper: key = 'SENTINEL'
                        else: key = 'TANAGER'
                        self.ts_data[key][loc['label']]['t'].append(dt)
                        self.ts_data[key][loc['label']]['v'].append(vals[i])
        print("Time series processing complete.")

    def _init_control_ui(self):
        self.fig_controls = plt.figure(figsize=(6, 12.0))
        self.fig_controls.canvas.manager.set_window_title("Timeline Navigation")
        self.ax_meta = self.fig_controls.add_axes([0, 0, 1, 1]); self.ax_meta.axis('off')
        self.ctrl_text = self.ax_meta.text(0.5, 0.96, "", ha='center', va='center', 
                                         fontsize=10, family='monospace',
                                         bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
        
        # Navigation
        ax_prev = self.fig_controls.add_axes([0.1, 0.90, 0.25, 0.035])
        ax_next = self.fig_controls.add_axes([0.65, 0.90, 0.25, 0.035])
        ax_input = self.fig_controls.add_axes([0.45, 0.90, 0.1, 0.035])
        self.btn_prev = Button(ax_prev, '<< Prev')
        self.btn_next = Button(ax_next, 'Next >>')
        self.txt_input = TextBox(ax_input, 'Go: ', initial='0')
        
        # Save Controls
        ax_save = self.fig_controls.add_axes([0.3, 0.85, 0.4, 0.035])
        self.btn_save = Button(ax_save, 'Save Current', color='lightgreen')

        self.ax_meta.text(0.5, 0.81, "--- Batch Processing ---", ha='center', va='center', fontsize=10)
        ax_start = self.fig_controls.add_axes([0.2, 0.76, 0.15, 0.035])
        ax_end = self.fig_controls.add_axes([0.5, 0.76, 0.15, 0.035])
        ax_auto = self.fig_controls.add_axes([0.3, 0.71, 0.4, 0.035])
        self.txt_start = TextBox(ax_start, 'Start: ', initial='0')
        self.txt_end = TextBox(ax_end, 'End: ', initial=str(self.total_frames-1))
        self.btn_auto = Button(ax_auto, 'Auto Save Range', color='lightblue')
        
        # Scatter Plot Controls
        self.ax_meta.text(0.5, 0.67, f"--- {COMPLEXITY_DICT.get(complexity_type, complexity_type)} Scatter ---", ha='center', va='center', fontsize=10)
        ax_l_frame = self.fig_controls.add_axes([0.2, 0.62, 0.15, 0.035])
        ax_t_frame = self.fig_controls.add_axes([0.5, 0.62, 0.15, 0.035])
        ax_scatter_btn = self.fig_controls.add_axes([0.3, 0.57, 0.4, 0.035])
        self.txt_l_frame = TextBox(ax_l_frame, 'L Idx: ', initial=str(self.l_file_idx if self.l_file_idx is not None else 0))
        self.txt_t_frame = TextBox(ax_t_frame, 'T Idx: ', initial=str(self.t_file_idx if self.t_file_idx is not None else 0))
        self.btn_scatter = Button(ax_scatter_btn, 'Update Scatter', color='lightyellow')
        
        # Localization
        self.ax_meta.text(0.5, 0.53, "--- Parallelotope Localization ---", ha='center', va='center', fontsize=10)
        ax_rad_loc = self.fig_controls.add_axes([0.3, 0.43, 0.4, 0.08])
        self.rad_localization = RadioButtons(ax_rad_loc, ('general', 'datasetMean', 'minEndmember'), active=0)

        # Filters Note
        self.ax_meta.text(0.5, 0.35, "Pixel Filters are baked into HDF5", ha='center', va='center', fontsize=10, style='italic')

        # Time Series
        self.ax_meta.text(0.5, 0.16, "--- Time Series Range ---", ha='center', va='center', fontsize=10)
        ax_ts_start = self.fig_controls.add_axes([0.15, 0.11, 0.3, 0.035])
        ax_ts_end = self.fig_controls.add_axes([0.55, 0.11, 0.3, 0.035])
        self.txt_ts_start = TextBox(ax_ts_start, 'Start: ', initial=self.ts_start_date.strftime("%Y-%m-%d"))
        self.txt_ts_end = TextBox(ax_ts_end, 'End: ', initial=self.ts_end_date.strftime("%Y-%m-%d"))
        
        ax_chk_ts = self.fig_controls.add_axes([0.3, 0.06, 0.4, 0.035])
        self.chk_ts_axis = CheckButtons(ax_chk_ts, ['Use Twin Y-Axis'], [self.use_twin_axis])

        ax_update_mask = self.fig_controls.add_axes([0.3, 0.01, 0.4, 0.035])
        self.btn_update_mask = Button(ax_update_mask, 'Update Range', color='lightcoral')

        self.btn_prev.on_clicked(self._on_prev)
        self.btn_next.on_clicked(self._on_next)
        self.txt_input.on_submit(self._on_submit)
        self.btn_save.on_clicked(self._on_save_images)
        self.btn_auto.on_clicked(self._on_auto_save)
        self.btn_scatter.on_clicked(self._on_update_scatter)
        self.btn_update_mask.on_clicked(self._on_update_mask)
        self.rad_localization.on_clicked(self._on_localization_change)
        self.chk_ts_axis.on_clicked(self._on_ts_axis_toggle)

    def _init_combined_ui(self):
        self.fig_combined = plt.figure(figsize=(18, 10))
        self.fig_combined.canvas.manager.set_window_title("Comprehensive Complexity Analysis")
        self.fig_combined.subplots_adjust(top=0.9, bottom=0.05, left=0.05, right=0.95, hspace=0.25, wspace=0.2)
        
        self.combined_hud = self.fig_combined.text(0.5, 0.98, "", ha='center', va='top', fontsize=10, 
                                                  bbox=dict(facecolor='white', alpha=0.8, edgecolor='lightgray'))
        
        self.ax_spatial = self.fig_combined.add_subplot(231)
        self.ax_spectral = self.fig_combined.add_subplot(232)
        self.ax_vol_curve = self.fig_combined.add_subplot(233)
        self.ax_slide_map = self.fig_combined.add_subplot(234)
        self.ax_ts_main = self.fig_combined.add_subplot(2, 3, (5, 6))

    def _init_redundant_ui(self):
        self.fig_redundant = plt.figure(figsize=(14, 10))
        self.fig_redundant.canvas.manager.set_window_title("Spatial and Complexity Details")
        self.fig_redundant.subplots_adjust(top=0.90, bottom=0.08, left=0.05, right=0.95, hspace=0.3, wspace=0.2)
        
        self.ax_spatial_redundant = self.fig_redundant.add_subplot(2, 2, 1)
        self.ax_slide_map_redundant = self.fig_redundant.add_subplot(2, 2, 2)
        self.ax_ts_redundant_main = self.fig_redundant.add_subplot(2, 2, (3, 4))

    def _init_transect_ui(self):
        self.fig_transect = plt.figure(figsize=(16, 10))
        self.fig_transect.canvas.manager.set_window_title("1D Spatial and Temporal Profiles")
        self.fig_transect.subplots_adjust(top=0.90, bottom=0.10, left=0.05, right=0.95, hspace=0.35, wspace=0.3)
        
        gs = gridspec.GridSpec(2, 4, figure=self.fig_transect)
        self.ax_transect_h = self.fig_transect.add_subplot(gs[0, :2])
        self.ax_transect_v = self.fig_transect.add_subplot(gs[0, 2:])
        self.ax_chip_rgb = self.fig_transect.add_subplot(gs[1, 0])
        self.ax_chip_comp = self.fig_transect.add_subplot(gs[1, 1])
        self.ax_transect_t = self.fig_transect.add_subplot(gs[1, 2:])
        self.ax_transect_t_twin = None

    def _init_hull_ui(self):
        self.fig_hull = plt.figure(figsize=(8, 7))
        self.fig_hull.canvas.manager.set_window_title("3D Parallelotope Projection")
        self.ax_hull = self.fig_hull.add_subplot(111, projection='3d')

    def update_display(self):
        idx = self.current_idx
        grid_name = self.prov_grid[idx]
        src_idx = self.prov_idx[idx]
        spacecraft = self.prov_space[idx]
        acq_time = self.prov_time[idx]
        
        curr_dt = datetime.fromtimestamp(acq_time, tz=timezone.utc)
        dt_et = curr_dt.astimezone(ZoneInfo("America/New_York"))
        
        meta_str = (f"TIMELINE:   {idx + 1} / {self.total_frames}\n"
                    f"SOURCE GRID: {grid_name} (Idx: {src_idx})\n"
                    f"SPACECRAFT: {spacecraft}\n"
                    f"ACQUIRED:   {dt_et.strftime('%Y-%m-%d %H:%M:%S ET')}")
        self.ctrl_text.set_text(meta_str)
        
        # Update textboxes automatically
        if 'HLS' in grid_name.upper():
            self.txt_l_frame.set_val(str(idx))
        elif 'TANAGER' in grid_name.upper():
            self.txt_t_frame.set_val(str(idx))

        filter_str = "Masking: Derived from HARMONIZED common_mask"
        self.combined_hud.set_text(meta_str.replace('\n', ' | ') + '\n' + filter_str)

        # Pull native data
        src_grp = self.h5[f'/HDFEOS/GRIDS/{grid_name}/Data Fields']
        
        raw_vis = self.harm_grp['ortho_visual'][idx, ...]
        endmembers = src_grp['frame_endmembers'][src_idx, ...]
        em_indices = src_grp['frame_endmember_indices'][src_idx, ...]
        vols = src_grp['frame_endmember_volumes'][src_idx, ...]
        
        # Harmonized mapped variables
        comp_data = self.base_dset[idx, ...].copy()
        mask_data = self.common_mask_dset[idx, ...]
        if MASKING:
            comp_data[mask_data == 1] = np.nan

        # RGB handling
        if raw_vis.shape[0] in [3, 4]:
            raw_vis = np.transpose(raw_vis, (1, 2, 0))
            
        if raw_vis.dtype == np.uint8:
            rgba = raw_vis.astype(np.float32) / 255.0
        else:
            rgba = raw_vis.astype(np.float32)
            
        if 'TANAGER' in grid_name.upper():
            for c in range(3):
                chan = rgba[..., c]
                valid_pixels = chan[chan > 0]
                if len(valid_pixels) > 0:
                    p1, p99 = np.percentile(valid_pixels, (1, 99))
                    if p99 > p1:
                        rgba[..., c] = np.clip((chan - p1) / (p99 - p1), 0.0, 1.0)
            
        if rgba.shape[-1] == 4:
            rgba[..., 3] = np.where(rgba[..., 3] > 0, 1.0, 0.0)
        rgb = np.clip(rgba, 0.0, 1.0)

        hull_bands = HULL_BANDS_LANDSAT if 'HLS' in grid_name.upper() else HULL_BANDS_TANAGER
        h, w = self.height, self.width

        # Row 1
        self.ax_spatial.clear()
        self.ax_spatial.imshow(rgb, extent=[0, w, h, 0])
        for i, flat_idx in enumerate(em_indices):
            row, col = flat_idx // w, flat_idx % w
            self.ax_spatial.plot(col + 0.5, row + 0.5, 'r+', markersize=8)
            self.ax_spatial.annotate(f'V{i}', (col + 0.5, row + 0.5), color='yellow', fontsize=10, fontweight='bold')
        for loc in TS_LOCATIONS:
            y, x = loc['yx']
            self.ax_spatial.plot(x + 0.5, y + 0.5, marker='s', markersize=10, markeredgecolor=loc['color'], 
                                 markerfacecolor='none', markeredgewidth=1.5, linestyle='None')
        self.ax_spatial.set_title(f"EM Locations ({grid_name})", color='black')
        self.ax_spatial.axis('off')

        if DISPLAY_REDUNDANT_FIGURE:
            self.ax_spatial_redundant.clear()
            self.ax_spatial_redundant.imshow(rgb, extent=[0, w, h, 0])
            for loc in TS_LOCATIONS:
                y, x = loc['yx']
                self.ax_spatial_redundant.plot(x + 0.5, y + 0.5, marker='s', markersize=10, markeredgecolor=loc['color'], 
                                     markerfacecolor='none', markeredgewidth=1.5, linestyle='None')
            self.ax_spatial_redundant.set_title("Time Series Locations")
            self.ax_spatial_redundant.axis('off')

        self.ax_spectral.clear()
        wl = self.wavelengths[grid_name]
        sort_idx = np.argsort(wl)
        sorted_wl = wl[sort_idx]
        for i in range(endmembers.shape[1]):
            if not np.all(np.isnan(endmembers[:, i])) and np.any(endmembers[:, i] != 0):
                sorted_em = endmembers[:, i][sort_idx]
                self.ax_spectral.plot(sorted_wl, sorted_em, label=f'V{i}', lw=1)
        self.ax_spectral.set_title("Spectral Signatures")
        self.ax_spectral.set_xlabel("Wavelength (μm)") 
        self.ax_spectral.set_ylabel("Reflectance")
        self.ax_spectral.set_ylim(0, 1)
        
        all_wl = np.concatenate(list(self.wavelengths.values()))
        self.ax_spectral.set_xlim(np.nanmin(all_wl) - 0.05, np.nanmax(all_wl) + 0.05)
        
        self.ax_spectral.legend(loc='upper right')
        self.ax_spectral.grid(True, alpha=0.3)

        self.ax_vol_curve.clear()
        self.ax_vol_curve.plot(np.arange(1, len(vols)+1), np.pad(vols[2:], (2,0), 'constant', constant_values=0), 'o-', markersize=4, color='green')
        self.ax_vol_curve.set_title("Complexity Curve")
        self.ax_vol_curve.set_xlabel("Endmember Count")
        self.ax_vol_curve.set_ylabel("Spectral Complexity")
        self.ax_vol_curve.grid(True, alpha=0.2)

        def update_map(ax, data, im_attr, cbar_attr, title, draw_crosshair=False):
            mh, mw = data.shape
            with np.errstate(all='ignore'):
                if DISPLAY_NORMALIZATION and not np.all(np.isnan(data)):
                    v_min, v_max = np.nanpercentile(data, (2, 98))
                else:
                    v_min, v_max = np.nanmin(data), np.nanmax(data)
            if np.isnan(v_min) or np.isnan(v_max):
                v_min, v_max = 0, 1
            elif v_min == v_max:
                v_max = v_min + 1e-6
            
            curr_im = getattr(self, im_attr)
            curr_cbar = getattr(self, cbar_attr)

            if curr_im is None:
                new_im = ax.imshow(data, cmap='viridis', extent=[0, mw, mh, 0], vmin=v_min, vmax=v_max)
                setattr(self, im_attr, new_im)
                
                if LOG_SCALE:
                    new_cbar = ax.figure.colorbar(new_im, format='%.1e', ax=ax, fraction=0.046, pad=0.04)
                else:
                    new_cbar = ax.figure.colorbar(new_im, ax=ax, fraction=0.046, pad=0.04)
                setattr(self, cbar_attr, new_cbar)
                
                for loc in TS_LOCATIONS:
                    y, x = loc['yx']
                    ax.plot(x + 0.5, y + 0.5, marker='s', markersize=10, markeredgecolor=loc['color'], 
                            markerfacecolor='none', markeredgewidth=1.5, linestyle='None')
                if draw_crosshair:
                    t_y, t_x = TS_LOCATIONS[0]['yx']
                    c_color = TS_LOCATIONS[0]['color']
                    ax.axhline(t_y + 0.5, color=c_color, linestyle=':', linewidth=1.5, alpha=0.7)
                    ax.axvline(t_x + 0.5, color=c_color, linestyle=':', linewidth=1.5, alpha=0.7)
                ax.set_title(title)
                ax.axis('off')
            else:
                curr_im.set_data(data)
                curr_im.set_clim(vmin=v_min, vmax=v_max)
                curr_cbar.update_normal(curr_im)

        update_map(self.ax_slide_map, comp_data, 'im_slide', 'cbar_slide', COMPLEXITY_DICT.get(complexity_type, complexity_type))
        
        if DISPLAY_REDUNDANT_FIGURE:
            update_map(self.ax_slide_map_redundant, comp_data, 'im_slide_redundant', 'cbar_slide_redundant', COMPLEXITY_DICT.get(complexity_type, complexity_type), draw_crosshair=True)
            
            # Transects
            transect_data = comp_data.copy()
            target_loc = TS_LOCATIONS[0]
            t_y, t_x = target_loc['yx']
            c_color = target_loc['color']
            
            self.ax_transect_h.clear()
            self.ax_transect_v.clear()
            span = 15; half_span = span // 2
            
            s_style = '--' if 'HLS' in grid_name.upper() else '-'
            s_marker = '^' if 'HLS' in grid_name.upper() else 's'
            s_width = 1.5 if 'HLS' in grid_name.upper() else 2.0
            
            x_start, x_end = max(0, t_x - half_span), min(w, t_x + half_span + 1)
            x_indices = np.arange(x_start, x_end) + 0.5
            h_data = transect_data[t_y, x_start:x_end]
            self.ax_transect_h.plot(x_indices, h_data, color=c_color, marker=s_marker, linestyle=s_style, linewidth=s_width, markersize=5, alpha=0.8)
            self.ax_transect_h.axvline(t_x + 0.5, color='red', linestyle='--', linewidth=1.5)
            self.ax_transect_h.set_title(f"Horizontal Spatial Profile (Row/Y = {t_y})", fontsize=10)
            self.ax_transect_h.set_xlim(x_start, x_end)
            if LOG_SCALE: self.ax_transect_h.set_yscale('log')
            
            y_start, y_end = max(0, t_y - half_span), min(h, t_y + half_span + 1)
            y_indices = np.arange(y_start, y_end) + 0.5
            v_data = transect_data[y_start:y_end, t_x]
            self.ax_transect_v.plot(y_indices, v_data, color=c_color, marker=s_marker, linestyle=s_style, linewidth=s_width, markersize=5, alpha=0.8)
            self.ax_transect_v.axvline(t_y + 0.5, color='red', linestyle='--', linewidth=1.5)
            self.ax_transect_v.set_title(f"Vertical Spatial Profile (Col/X = {t_x})", fontsize=10)
            self.ax_transect_v.set_xlim(y_start, y_end)
            if LOG_SCALE: self.ax_transect_v.set_yscale('log')
            
            self.ax_chip_rgb.clear()
            self.ax_chip_comp.clear()
            rgb_chip = rgb[y_start:y_end, x_start:x_end, :]
            comp_chip = transect_data[y_start:y_end, x_start:x_end]
            rel_y, rel_x = t_y - y_start, t_x - x_start
            
            self.ax_chip_rgb.imshow(rgb_chip)
            self.ax_chip_rgb.axhline(rel_y, color='red', linestyle='--', linewidth=1.5, alpha=0.8)
            self.ax_chip_rgb.axvline(rel_x, color='red', linestyle='--', linewidth=1.5, alpha=0.8)
            self.ax_chip_rgb.axis('off')
            
            with np.errstate(all='ignore'):
                valid_chip = comp_chip[~np.isnan(comp_chip)]
                if len(valid_chip) > 0:
                    c_min, c_max = np.nanmin(valid_chip), np.nanmax(valid_chip)
                else:
                    c_min, c_max = 0, 1
            self.ax_chip_comp.imshow(comp_chip, cmap='viridis', vmin=c_min, vmax=c_max)
            self.ax_chip_comp.axhline(rel_y, color='red', linestyle='--', linewidth=1.5, alpha=0.8)
            self.ax_chip_comp.axvline(rel_x, color='red', linestyle='--', linewidth=1.5, alpha=0.8)
            self.ax_chip_comp.axis('off')
            
            self.ax_transect_t.clear()
            if getattr(self, 'ax_transect_t_twin', None) is not None:
                try: self.ax_transect_t_twin.remove()
                except Exception: pass
                self.ax_transect_t_twin = None
                
            if self.use_twin_axis:
                self.ax_transect_t_twin = self.ax_transect_t.twinx()
            t_ax = self.ax_transect_t_twin if self.use_twin_axis else self.ax_transect_t
            
            label = target_loc['label']
            l_data = self.ts_data['LANDSAT'][label]
            s_data = self.ts_data['SENTINEL'][label]
            t_data = self.ts_data['TANAGER'][label]
            
            filt_t_l, filt_v_l = [], []
            if l_data['t']:
                for i in range(len(l_data['t'])):
                    if self.ts_start_date <= l_data['t'][i] <= self.ts_end_date:
                        filt_t_l.append(l_data['t'][i]); filt_v_l.append(l_data['v'][i])
            
            filt_t_s, filt_v_s = [], []
            if s_data['t']:
                for i in range(len(s_data['t'])):
                    if self.ts_start_date <= s_data['t'][i] <= self.ts_end_date:
                        filt_t_s.append(s_data['t'][i]); filt_v_s.append(s_data['v'][i])
                        
            filt_t_t, filt_v_t = [], []
            if t_data['t']:
                for i in range(len(t_data['t'])):
                    if self.ts_start_date <= t_data['t'][i] <= self.ts_end_date:
                        filt_t_t.append(t_data['t'][i]); filt_v_t.append(t_data['v'][i])

            if filt_t_l:
                self.ax_transect_t.plot(filt_t_l, filt_v_l, marker='^', color=c_color, label=f"L: {label}", markersize=5, linestyle='--', linewidth=1.5, alpha=0.7)
            if filt_t_s:
                self.ax_transect_t.plot(filt_t_s, filt_v_s, marker='o', color=c_color, label=f"S: {label}", markersize=4, linestyle=':', linewidth=1.2, alpha=0.7)
            if filt_t_t:
                t_ax.plot(filt_t_t, filt_v_t, marker='s', color=c_color, label=f"T: {label}", markersize=6, linestyle='-', linewidth=2, alpha=0.9)
            
            self.ax_transect_t.grid(True, alpha=0.3, which="both", ls="--")
            self.ax_transect_t.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            self.ax_transect_t.tick_params(axis='x', rotation=45, labelsize=9)
            self.ax_transect_t.axvline(curr_dt, color='red', linestyle=':', alpha=0.8, linewidth=2, label='Current Frame')
            self.ax_transect_t.set_xlim(self.ts_start_date, self.ts_end_date)
            
            lines_1, labels_1 = self.ax_transect_t.get_legend_handles_labels()
            if self.use_twin_axis and self.ax_transect_t_twin is not None:
                self.ax_transect_t.set_ylabel(f"HLS (L/S)", color='black', fontsize=10)
                self.ax_transect_t_twin.set_ylabel(f"Tanager", color='black', fontsize=10)
                if LOG_SCALE:
                    self.ax_transect_t.set_yscale('log')
                    self.ax_transect_t_twin.set_yscale('log')
                lines_2, labels_2 = self.ax_transect_t_twin.get_legend_handles_labels()
                self.ax_transect_t.legend(lines_1 + lines_2, labels_1 + labels_2, loc='best', fontsize=10)
            else:
                self.ax_transect_t.set_ylabel(COMPLEXITY_DICT.get(complexity_type, complexity_type), fontsize=10)
                if LOG_SCALE: self.ax_transect_t.set_yscale('log')
                self.ax_transect_t.legend(loc='best', fontsize=10)

        # Plot Time Series helper
        def plot_time_series(ax_main, ax_twin=None):
            t_ax = ax_twin if ax_twin is not None else ax_main
            for loc in TS_LOCATIONS:
                label = loc['label']
                l_d = self.ts_data['LANDSAT'][label]
                if l_d['t']:
                    filt_t, filt_v = [t for t in l_d['t'] if self.ts_start_date <= t <= self.ts_end_date], [v for i, v in enumerate(l_d['v']) if self.ts_start_date <= l_d['t'][i] <= self.ts_end_date]
                    if filt_t:
                        ax_main.plot(filt_t, filt_v, marker='^', color=loc['color'], label=f"L: {label}", markersize=4, linestyle='--', linewidth=1, alpha=0.6)
                        
                s_d = self.ts_data['SENTINEL'][label]
                if s_d['t']:
                    filt_t, filt_v = [t for t in s_d['t'] if self.ts_start_date <= t <= self.ts_end_date], [v for i, v in enumerate(s_d['v']) if self.ts_start_date <= s_d['t'][i] <= self.ts_end_date]
                    if filt_t:
                        ax_main.plot(filt_t, filt_v, marker='o', color=loc['color'], label=f"S: {label}", markersize=4, linestyle=':', linewidth=1.2, alpha=0.6)
                
                t_d = self.ts_data['TANAGER'][label]
                if t_d['t']:
                    filt_t, filt_v = [t for t in t_d['t'] if self.ts_start_date <= t <= self.ts_end_date], [v for i, v in enumerate(t_d['v']) if self.ts_start_date <= t_d['t'][i] <= self.ts_end_date]
                    if filt_t:
                        t_ax.plot(filt_t, filt_v, marker='s', color=loc['color'], label=f"T: {label}", markersize=5, linestyle='-', linewidth=1.5, alpha=0.9)

            ax_main.grid(True, alpha=0.3, which="both", ls="--")
            ax_main.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax_main.tick_params(axis='x', rotation=45, labelsize=8)
            ax_main.axvline(curr_dt, color='black', linestyle='--', alpha=0.8, linewidth=1.5)
            ax_main.set_xlim(self.ts_start_date, self.ts_end_date)
            
            lines_1, labels_1 = ax_main.get_legend_handles_labels()
            if self.use_twin_axis and ax_twin is not None:
                if LOG_SCALE:
                    ax_main.set_yscale('log')
                    ax_twin.set_yscale('log')
                lines_2, labels_2 = ax_twin.get_legend_handles_labels()
                ax_main.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', fontsize=8, ncol=2)
            else:
                ax_main.legend(loc='upper left', fontsize=8, ncol=2)

        self.ax_ts_main.clear()
        if self.ax_ts_twin is not None:
            try: self.ax_ts_twin.remove()
            except Exception: pass
            self.ax_ts_twin = None
        if self.use_twin_axis: self.ax_ts_twin = self.ax_ts_main.twinx()
        plot_time_series(self.ax_ts_main, self.ax_ts_twin)

        if DISPLAY_REDUNDANT_FIGURE:
            self.ax_ts_redundant_main.clear()
            if self.ax_ts_redundant_twin is not None:
                try: self.ax_ts_redundant_twin.remove()
                except Exception: pass
                self.ax_ts_redundant_twin = None
            if self.use_twin_axis: self.ax_ts_redundant_twin = self.ax_ts_redundant_main.twinx()
            plot_time_series(self.ax_ts_redundant_main, self.ax_ts_redundant_twin)

        # 3D Hull
        self.ax_hull.clear()
        
        self.ax_hull.text2D(0.5, 0.5, "3D Hull Disabled\n(surface_reflectance removed to save storage space)", 
                            ha='center', va='center', transform=self.ax_hull.transAxes, fontsize=12)
        self.ax_hull.set_axis_off()
        
        figs_to_draw = [self.fig_controls, self.fig_combined, self.fig_hull]
        if DISPLAY_REDUNDANT_FIGURE: figs_to_draw.extend([self.fig_redundant, self.fig_transect])
        for f in figs_to_draw: f.canvas.draw_idle()

    def _on_localization_change(self, label):
        self.localization_mode = label
        self.update_display()
            
    def _on_ts_axis_toggle(self, label):
        self.use_twin_axis = not self.use_twin_axis
        self.update_display()

    def _on_update_mask(self, event):
        try:
            self.ts_start_date = datetime.strptime(self.txt_ts_start.text.strip(), "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            self.txt_ts_start.set_val(self.ts_start_date.strftime("%Y-%m-%d"))
        try:
            self.ts_end_date = datetime.strptime(self.txt_ts_end.text.strip(), "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            self.txt_ts_end.set_val(self.ts_end_date.strftime("%Y-%m-%d"))
        self.update_display()
        if self.fig_scatter is not None and plt.fignum_exists(self.fig_scatter.number):
            self._on_update_scatter(None)

    def _on_prev(self, event):
        if self.current_idx > 0:
            self.current_idx -= 1
            self.update_display()
            self.txt_input.set_val(str(self.current_idx))

    def _on_next(self, event):
        if self.current_idx < self.total_frames - 1:
            self.current_idx += 1
            self.update_display()
            self.txt_input.set_val(str(self.current_idx))

    def _on_submit(self, text):
        try:
            val = int(text)
            if 0 <= val < self.total_frames:
                self.current_idx = val
                self.update_display()
        except: self.txt_input.set_val(str(self.current_idx))

    def _on_update_scatter(self, event):
        try:
            l_idx = int(self.txt_l_frame.text)
            t_idx = int(self.txt_t_frame.text)
        except ValueError:
            print("Invalid frame indices provided. Please enter integers.")
            return

        if l_idx < 0 or l_idx >= self.total_frames or t_idx < 0 or t_idx >= self.total_frames:
            print("Error: Selected frame index out of bounds.")
            return

        l_data = self.base_dset[l_idx, ...].copy()
        if MASKING: l_data[self.common_mask_dset[l_idx] == 1] = np.nan
            
        t_data = self.base_dset[t_idx, ...].copy()
        if MASKING: t_data[self.common_mask_dset[t_idx] == 1] = np.nan

        h = min(l_data.shape[0], t_data.shape[0])
        w = min(l_data.shape[1], t_data.shape[1])
        l_flat = l_data[:h, :w].flatten()
        t_flat = t_data[:h, :w].flatten()

        if LOG_SCALE:
            valid_mask = (~np.isnan(l_flat)) & (~np.isnan(t_flat)) & (l_flat > 0) & (t_flat > 0)
        else:
            valid_mask = (~np.isnan(l_flat)) & (~np.isnan(t_flat))
            
        l_valid = l_flat[valid_mask]
        t_valid = t_flat[valid_mask]
        
        if self.fig_scatter is None or not plt.fignum_exists(self.fig_scatter.number):
            self.fig_scatter = plt.figure(figsize=(12, 10))
            self.fig_scatter.canvas.manager.set_window_title(f"Correlation Scatter")
            self.ax_scatter_lin = self.fig_scatter.add_subplot(221)
            self.ax_scatter_log = self.fig_scatter.add_subplot(222)
            self.ax_hist_l = self.fig_scatter.add_subplot(223)
            self.ax_hist_t = self.fig_scatter.add_subplot(224)
        else:
            self.ax_scatter_lin.clear()
            self.ax_scatter_log.clear()
            self.ax_hist_l.clear()
            self.ax_hist_t.clear()

        if len(l_valid) > 0:
            lin_slope, lin_intercept = np.polyfit(l_valid, t_valid, 1)
            self.ax_scatter_lin.scatter(l_valid, t_valid, alpha=0.3, s=10, color='tab:purple')
            l_range_lin = np.array([np.min(l_valid), np.max(l_valid)])
            self.ax_scatter_lin.plot(l_range_lin, lin_slope * l_range_lin + lin_intercept, color='red', linewidth=2)
            self.ax_scatter_lin.set_title("Linear Correlation")

            if LOG_SCALE:
                log_l = np.log10(l_valid)
                log_t = np.log10(t_valid)
                log_slope, log_intercept = np.polyfit(log_l, log_t, 1)
                self.ax_scatter_log.scatter(l_valid, t_valid, alpha=0.3, s=10, color='tab:orange')
                self.ax_scatter_log.plot(l_range_lin, (10**log_intercept) * (l_range_lin ** log_slope), color='red', linewidth=2)
                self.ax_scatter_log.set_xscale('log')
                self.ax_scatter_log.set_yscale('log')
            self.ax_scatter_log.set_title("Log-Log Correlation" if LOG_SCALE else "N/A for Z-Scores")
            
            # Simple histograms
            self.ax_hist_l.hist(l_valid, bins=50, color='tab:purple', alpha=0.7, density=True)
            self.ax_hist_t.hist(t_valid, bins=50, color='tab:orange', alpha=0.7, density=True)

        self.fig_scatter.canvas.draw_idle()
        self.fig_scatter.show()

    def _on_auto_save(self, event):
        try:
            start_idx = int(self.txt_start.text)
            end_idx = int(self.txt_end.text)
        except ValueError:
            print("Invalid range. Please enter valid integers for Start and End.")
            return
            
        start_idx = max(0, start_idx)
        end_idx = min(self.total_frames - 1, end_idx)
        
        if start_idx > end_idx:
            print("Start index must be less than or equal to End index.")
            return
            
        os.makedirs(self.save_dir, exist_ok=True)
        original_idx = self.current_idx
        
        print(f"Starting batch save from frame {start_idx} to {end_idx} in {self.save_dir}...")
        for i in range(start_idx, end_idx + 1):
            self.current_idx = i
            self.update_display()
            dt_str = datetime.fromtimestamp(self.prov_time[i], tz=timezone.utc).strftime('%Y%m%d_%H%M%S')
            grid = self.prov_grid[i]
            filename = f"Frame_{i:03d}_{grid}_{dt_str}.png"
            out_path = os.path.join(self.save_dir, filename)
            self.fig_combined.savefig(out_path, dpi=150, bbox_inches='tight')
            print(f"Saved [{i - start_idx + 1}/{end_idx - start_idx + 1}]: {filename}")
            
        self.current_idx = original_idx
        self.update_display()
        print("Batch processing complete.")

    def _on_save_images(self, event):
        os.makedirs(self.save_dir, exist_ok=True)
        idx = self.current_idx
        dt_str = datetime.fromtimestamp(self.prov_time[idx], tz=timezone.utc).strftime('%Y%m%d_%H%M%S')
        grid = self.prov_grid[idx]
        filename = f"Frame_{idx:03d}_{grid}_{dt_str}.png"
        out_path = os.path.join(self.save_dir, filename)
        
        self.fig_combined.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {out_path}")

    def run(self): plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="View Spectral Complexity Metrics")
    parser.add_argument("--file", type=str, help="Path to HARMONIZED SC HDF5 Output")
    parser.add_argument("--start_year", type=int, help="Start Year for time series")
    parser.add_argument("--end_year", type=int, help="End Year for time series")
    args = parser.parse_args()
    
    file_path = args.file
    
    if args.start_year is not None or args.end_year is not None:
        if args.start_year is not None:
            START_YEAR = args.start_year
            TS_START_DATE = datetime(START_YEAR, 1, 1, tzinfo=timezone.utc)
        if args.end_year is not None:
            END_YEAR = args.end_year
            TS_END_DATE = datetime(END_YEAR, 12, 31, tzinfo=timezone.utc)
            
        suffix = ''
        if complexity_type == 'sliding_volume_z_score':
            suffix = '_zscore'
        elif complexity_type == 'sliding_volume_map':
            suffix = '_SpecComplex'
        if not MASKING:
            suffix += '_unmasked'
        suffix += f'_{END_YEAR-START_YEAR}yr' if START_YEAR != 2025 else '_2025'
        SAVE_DIR = f"C:/satelliteImagery/MultiSensor_Analysis_{Location}_Harmonized" + suffix
    
    if not file_path:
        root = tk.Tk(); root.withdraw()
        file_path = tk.filedialog.askopenfilename(
            title="Select HARMONIZED SC HDF5 Output",
            initialfile=default_harmonized_path,
            filetypes=[("HDF5 files", "*.h5")]
        )
        root.destroy()
        
    if not file_path:
        file_path = default_harmonized_path
    
    if os.path.exists(file_path):
        viewer = HarmonizedComplexityViewer(file_path)
        viewer.run()
    else:
        print(f"File not found: {file_path}")
