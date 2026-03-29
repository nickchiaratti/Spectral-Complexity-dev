import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.widgets import Button, TextBox, CheckButtons, RadioButtons
from datetime import datetime, timezone
import tkinter as tk
from tkinter import filedialog
from scipy.stats import pearsonr, spearmanr
from scipy import ndimage
from skimage import exposure

import rasterio.transform
from pyproj import Transformer, CRS

# --- Configuration ---
complexity_type = 'sliding_volume_z_score' #'sliding_volume_map'  'sliding_volume_local_z_score'  'sliding_volume_z_score'
# Standard Landsat 8/9 True Color Indices: [C(0), B(1), G(2), R(3), NIR(4), S1(5), S2(6)]
LANDSAT_RGB_BANDS = (3, 2, 1) 
# Default Projection Bands for 3D Hull (Indices)
HULL_BANDS_LANDSAT = (6, 4, 2) 
HULL_BANDS_TANAGER = (100, 50, 20) # Example hyperspectral indices

TS_START_DATE = datetime(2015, 1, 1, tzinfo=timezone.utc)
TS_END_DATE = datetime(2025, 12, 31, tzinfo=timezone.utc)
TWIN_Y_AXIS_DEFAULT = False

# Combined Pixel Mask Configuration
MASKING = False
SUN_ELEVATION_THRESHOLD = 30
CLOUD_DILATION = 0

# Tanager Pixel Mask Configuration
TANAGER_AEROSOL_DEPTH_THRESHOLD = 0.3
TANAGER_SR_UNCERTAINTY_THRESHOLD = 0.10

# LANDSAT Pixel Mask Configuration
QA_REJECT_MASK = 0b111111
RADSAT_ACCEPT_VALUE = 0
AEROSOL_ACCEPT_LEVEL = 'medium' #'low' 'medium' 'high'


# Mapped levels for UI Dropdown
AEROSOL_DICT = {
    'low': [2, 4, 32, 66, 68, 96, 100],
    'medium': [2, 4, 32, 66, 68, 96, 100, 130, 132, 160, 164],
    'high': [2, 4, 32, 66, 68, 96, 100, 130, 132, 160, 164, 192, 194, 196, 224, 228] # Aerosol_Optical_Depth > 0.3
}
landsat_path = "C:/satelliteImagery/LANDSAT/Rochester/LANDSAT_Stack_Rochester_GEE_2015_2025_SC_EM-7_Gram-minEndmember_Norm-bandCount.h5"
tanager_path = "C:/satelliteImagery/Tanager/Rochester/Tanager_Stack_Rochester_HDFEOS_SC_EM-7_Gram-minEndmember_Norm-bandCount.h5"

if complexity_type == 'sliding_volume_z_score':
    suffix = '_zscore'
else:
    suffix = ''

SAVE_DIR = "C:/satelliteImagery/MultiSensor_Analysis_Rochester_minEndmember" + suffix

# Time Series Locations (Latitude, Longitude)
TS_LOCATIONS = [
    {'latlon': (43.142856, -77.508451), 'label': "West Tait Forest",     'color': 'tab:green'},
    {'latlon': (43.144861, -77.501176), 'label': "East Tait Forest",             'color': 'tab:olive'},
    {'latlon': (43.136910, -77.469462), 'label': "Artificial turf football field",  'color': 'tab:blue'},
    {'latlon': (43.138241, -77.470873), 'label': "Recently added artificial turf",  'color': 'tab:cyan'},
    {'latlon': (43.141297, -77.506256), 'label': "Tait Parking Lot",                'color': 'tab:red'},
    {'latlon': (43.139411, -77.504005), 'label': "ROCX NITE Tarp",                  'color': 'tab:purple'},
]

DISPLAY_NORMALIZATION = False
DISPLAY_REDUNDANT_FIGURE = True  # Toggle for the 2-subplot spatial/complexity redundant figure

class MultiComplexityViewer:
    def __init__(self, file_paths):
        self.files = []
        self.all_frames = []
        
        # Initialize UI Mask States
        self.mask_qa_enabled = True
        self.mask_radsat_enabled = True
        self.aerosol_level = AEROSOL_ACCEPT_LEVEL
        self.sun_elev_thresh = SUN_ELEVATION_THRESHOLD
        self.cloud_dilation = CLOUD_DILATION
        
        self.t_aerosol_thresh = TANAGER_AEROSOL_DEPTH_THRESHOLD
        self.t_uncertainty_thresh = TANAGER_SR_UNCERTAINTY_THRESHOLD
        
        # Initialize Time Series Display Range
        self.ts_start_date = TS_START_DATE
        self.ts_end_date = TS_END_DATE
        self.use_twin_axis = TWIN_Y_AXIS_DEFAULT
        self.localization_mode = 'general'
        
        # 1. Load and Parse both files (Strict execution; no silent exceptions)
        for path in file_paths:
            h5 = h5py.File(path, 'r')
            source_name = list(h5['/HDFEOS/GRIDS'].keys())[0]
            data_grp = h5[f'HDFEOS/GRIDS/{source_name}/Data Fields']
            
            sr_dset = data_grp['surface_reflectance']
            acq_times = sr_dset.attrs['acquisition_time']
            sat_ids = sr_dset.attrs['spacecraft_id']
            
            # Retrieve and Scale Wavelengths strictly
            raw_wl = sr_dset.attrs['wavelengths']
            if source_name == 'TANAGER':
                wavelengths = raw_wl[:] / 1000.0
            else:
                wavelengths = raw_wl[:]
            
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

        # Find specific indices for scatter plot reference
        self.l_file_idx = next((i for i, f in enumerate(self.files) if f['source'] == 'LANDSAT'), None)
        self.t_file_idx = next((i for i, f in enumerate(self.files) if f['source'] == 'TANAGER'), None)

        # 2. Map Geographic Coordinates to Pixel Coordinates
        if len(self.files) > 0:
            sample_file = self.files[0]
            sr_attrs = sample_file['data_grp']['surface_reflectance'].attrs
            geo_transform = sr_attrs['GeoTransform']
            spatial_ref = sr_attrs['spatial_ref']
            
            if isinstance(spatial_ref, bytes):
                spatial_ref = spatial_ref.decode('utf-8')
            crs = CRS.from_wkt(spatial_ref)
            transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
            
            affine = rasterio.transform.Affine(*geo_transform)
            inv_affine = ~affine
            
            print("\n--- Coordinate Mapping ---")
            for loc in TS_LOCATIONS:
                lat, lon = loc['latlon']
                proj_x, proj_y = transformer.transform(lon, lat)
                px, py = inv_affine * (proj_x, proj_y)
                loc['yx'] = (int(round(py)), int(round(px)))
                print(f"Mapped [{loc['label']}] Lat/Lon ({lat:.4f}, {lon:.4f}) -> Pixel (y={loc['yx'][0]}, x={loc['yx'][1]})")

        # 3. Interleave frames by acquisition time
        self.all_frames.sort(key=lambda x: x['timestamp'])
        self.num_total_frames = len(self.all_frames)
        self.current_idx = 0

        self.save_dir = SAVE_DIR

        # Persistent Plotting objects
        self.im_slide = None
        self.cbar_slide = None
        self.fig_scatter = None
        
        self.im_slide_redundant = None
        self.cbar_slide_redundant = None
        self.ax_ts_twin = None
        self.ax_ts_redundant_twin = None

        # Process Time Series with initial thresholds
        self._recompute_time_series()

        # 4. Initialize UI
        self._init_control_ui()
        self._init_combined_ui()
        if DISPLAY_REDUNDANT_FIGURE:
            self._init_redundant_ui()
        self._init_hull_ui()
        
        self.update_display()

    def _recompute_time_series(self):
        """
        Loops through all frames and fully re-evaluates the 2D spatial masks 
        to ensure the 1D time series data is perfectly synchronized with the UI maps.
        """
        print(f"Applying spatial masks to time series data... (This may take a moment)")
        
        # Reset data dictionaries
        self.ts_data = {
            'LANDSAT': {loc['label']: {'t': [], 'v': []} for loc in TS_LOCATIONS},
            'TANAGER': {loc['label']: {'t': [], 'v': []} for loc in TS_LOCATIONS}
        }

        for frame in self.all_frames:
            file_info = self.files[frame['file_idx']]
            dgrp = file_info['data_grp']

            dset = dgrp[complexity_type]
            src = frame['source']
            key = 'LANDSAT' if 'LANDSAT' in src.upper() else 'TANAGER'
            dt = datetime.fromtimestamp(frame['timestamp'], tz=timezone.utc)
            f_idx = frame['frame_idx']
            
            # Retrieve the full 2D spatial mask exactly as the image views do
            shape = dset[f_idx].shape
            if key == 'LANDSAT':
                mask = self._get_landsat_mask(dgrp, f_idx, shape)
            else:
                mask = self._get_tanager_mask(dgrp, f_idx, shape)
            
            # Check specific subset locations against the generated mask
            for loc in TS_LOCATIONS:
                y, x = loc['yx']
                # Verify array bounds
                if 0 <= y < dset.shape[1] and 0 <= x < dset.shape[2]:
                    # Only accept if the spatial mask permits this pixel
                    if mask[y, x]:
                        val = dset[f_idx, y, x]
                        if not np.isnan(val):
                            self.ts_data[key][loc['label']]['t'].append(dt)
                            self.ts_data[key][loc['label']]['v'].append(val)

        print("Time series processing complete.")

    def _init_control_ui(self):
        # Expanded figure size to accommodate new scatter, mask, range, and localization controls
        self.fig_controls = plt.figure(figsize=(6, 12.0))
        self.fig_controls.canvas.manager.set_window_title("Timeline Navigation")
        self.ax_meta = self.fig_controls.add_axes([0, 0, 1, 1]); self.ax_meta.axis('off')
        self.ctrl_text = self.ax_meta.text(0.5, 0.96, "", ha='center', va='center', 
                                         fontsize=10, family='monospace',
                                         bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
        
        # Navigation Row
        ax_prev = self.fig_controls.add_axes([0.1, 0.90, 0.25, 0.035])
        ax_next = self.fig_controls.add_axes([0.65, 0.90, 0.25, 0.035])
        ax_input = self.fig_controls.add_axes([0.45, 0.90, 0.1, 0.035])
        
        self.btn_prev = Button(ax_prev, '<< Prev')
        self.btn_next = Button(ax_next, 'Next >>')
        self.txt_input = TextBox(ax_input, 'Go: ', initial='0')
        
        # Single Save Row
        ax_save = self.fig_controls.add_axes([0.3, 0.85, 0.4, 0.035])
        self.btn_save = Button(ax_save, 'Save Current', color='lightgreen')

        # Auto Save Row
        self.ax_meta.text(0.5, 0.81, "--- Batch Processing ---", ha='center', va='center', fontsize=9)
        
        ax_start = self.fig_controls.add_axes([0.2, 0.76, 0.15, 0.035])
        ax_end = self.fig_controls.add_axes([0.5, 0.76, 0.15, 0.035])
        ax_auto = self.fig_controls.add_axes([0.3, 0.71, 0.4, 0.035])

        self.txt_start = TextBox(ax_start, 'Start: ', initial='0')
        self.txt_end = TextBox(ax_end, 'End: ', initial=str(self.num_total_frames-1))
        self.btn_auto = Button(ax_auto, 'Auto Save Range', color='lightblue')
        
        # Scatter Plot Controls
        self.ax_meta.text(0.5, 0.67, "--- Sliding Volume Scatter ---", ha='center', va='center', fontsize=9)
        
        ax_l_frame = self.fig_controls.add_axes([0.2, 0.62, 0.15, 0.035])
        ax_t_frame = self.fig_controls.add_axes([0.5, 0.62, 0.15, 0.035])
        ax_scatter_btn = self.fig_controls.add_axes([0.3, 0.57, 0.4, 0.035])

        self.txt_l_frame = TextBox(ax_l_frame, 'L Idx: ', initial='80')
        self.txt_t_frame = TextBox(ax_t_frame, 'T Idx: ', initial='3')
        self.btn_scatter = Button(ax_scatter_btn, 'Update Scatter', color='lightyellow')
        
        # --- Parallelotope Localization Controls ---
        self.ax_meta.text(0.5, 0.53, "--- Parallelotope Localization ---", ha='center', va='center', fontsize=9)
        ax_rad_loc = self.fig_controls.add_axes([0.3, 0.43, 0.4, 0.08])
        self.rad_localization = RadioButtons(ax_rad_loc, ('general', 'datasetMean', 'minEndmember'), active=0)

        # --- Pixel Filters Controls ---
        self.ax_meta.text(0.5, 0.40, "--- Pixel Filters ---", ha='center', va='center', fontsize=9)
        
        ax_chk = self.fig_controls.add_axes([0.1, 0.30, 0.4, 0.08])
        self.chk_masks = CheckButtons(ax_chk, ['L: QA Rej', 'L: RADSAT Acpt'], [self.mask_qa_enabled, self.mask_radsat_enabled])
        
        ax_rad = self.fig_controls.add_axes([0.55, 0.30, 0.35, 0.08])
        ax_rad.set_title("L: Aerosol", fontsize=8)
        active_idx = ['low', 'medium', 'high', 'all'].index(self.aerosol_level)
        self.rad_aerosol = RadioButtons(ax_rad, ('low', 'medium', 'high', 'all'), active=active_idx)
        
        # Pack 4 filters into 2 rows to save UI space
        ax_sun = self.fig_controls.add_axes([0.1, 0.25, 0.35, 0.035])
        self.txt_sun = TextBox(ax_sun, 'Sun Elev > ', initial=str(self.sun_elev_thresh))

        ax_cdil = self.fig_controls.add_axes([0.55, 0.25, 0.35, 0.035])
        self.txt_cdil = TextBox(ax_cdil, 'Cloud Dil: ', initial=str(self.cloud_dilation))

        ax_t_aod = self.fig_controls.add_axes([0.1, 0.20, 0.35, 0.035])
        self.txt_t_aod = TextBox(ax_t_aod, 'T-AOD < ', initial=str(self.t_aerosol_thresh))

        ax_t_unc = self.fig_controls.add_axes([0.55, 0.20, 0.35, 0.035])
        self.txt_t_unc = TextBox(ax_t_unc, 'T-Unc < ', initial=str(self.t_uncertainty_thresh))

        # --- Time Series Display Controls ---
        self.ax_meta.text(0.5, 0.16, "--- Time Series Range ---", ha='center', va='center', fontsize=9)
        
        ax_ts_start = self.fig_controls.add_axes([0.15, 0.11, 0.3, 0.035])
        ax_ts_end = self.fig_controls.add_axes([0.55, 0.11, 0.3, 0.035])
        self.txt_ts_start = TextBox(ax_ts_start, 'Start: ', initial=self.ts_start_date.strftime("%Y-%m-%d"))
        self.txt_ts_end = TextBox(ax_ts_end, 'End: ', initial=self.ts_end_date.strftime("%Y-%m-%d"))
        
        ax_chk_ts = self.fig_controls.add_axes([0.3, 0.06, 0.4, 0.035])
        self.chk_ts_axis = CheckButtons(ax_chk_ts, ['Use Twin Y-Axis'], [self.use_twin_axis])

        # Update Masks / Parameters Execution Button
        ax_update_mask = self.fig_controls.add_axes([0.3, 0.01, 0.4, 0.035])
        self.btn_update_mask = Button(ax_update_mask, 'Update Masks & Range', color='lightcoral')

        # Connect events
        self.btn_prev.on_clicked(self._on_prev)
        self.btn_next.on_clicked(self._on_next)
        self.txt_input.on_submit(self._on_submit)
        self.btn_save.on_clicked(self._on_save_images)
        self.btn_auto.on_clicked(self._on_auto_save)
        self.btn_scatter.on_clicked(self._on_update_scatter)
        self.btn_update_mask.on_clicked(self._on_update_mask)
        self.rad_localization.on_clicked(self._on_localization_change)
        
        # Real-time toggle specifically for Twin Y-axis since it requires no spatial mask recomputation
        self.chk_ts_axis.on_clicked(self._on_ts_axis_toggle)

    def _init_combined_ui(self):
        self.fig_combined = plt.figure(figsize=(18, 10))
        self.fig_combined.canvas.manager.set_window_title("Comprehensive Complexity Analysis")
        self.fig_combined.subplots_adjust(top=0.9, bottom=0.05, left=0.05, right=0.95, hspace=0.25, wspace=0.2)
        
        self.combined_hud = self.fig_combined.text(0.5, 0.98, "", ha='center', va='top', fontsize=11, 
                                                  bbox=dict(facecolor='white', alpha=0.8, edgecolor='lightgray'))
        
        self.ax_spatial = self.fig_combined.add_subplot(231)
        self.ax_spectral = self.fig_combined.add_subplot(232)
        self.ax_vol_curve = self.fig_combined.add_subplot(233)
        self.ax_slide_map = self.fig_combined.add_subplot(234)
        self.ax_ts_main = self.fig_combined.add_subplot(2, 3, (5, 6))

    def _init_redundant_ui(self):
        self.fig_redundant = plt.figure(figsize=(14, 7))
        self.fig_redundant.canvas.manager.set_window_title("Spatial and Complexity Details")
        self.fig_redundant.subplots_adjust(top=0.85, bottom=0.05, left=0.05, right=0.95, wspace=0.2)
        
        self.ax_spatial_redundant = self.fig_redundant.add_subplot(121)
        self.ax_slide_map_redundant = self.fig_redundant.add_subplot(122)
        
        # New standalone Time Series figure
        self.fig_ts_redundant = plt.figure(figsize=(14, 6))
        self.fig_ts_redundant.canvas.manager.set_window_title("Time Series Detail")
        self.fig_ts_redundant.subplots_adjust(top=0.85, bottom=0.15, left=0.05, right=0.95)
        self.ax_ts_redundant_main = self.fig_ts_redundant.add_subplot(111)

    def _init_hull_ui(self):
        self.fig_hull = plt.figure(figsize=(8, 7))
        self.fig_hull.canvas.manager.set_window_title("3D Parallelotope Projection")
        self.ax_hull = self.fig_hull.add_subplot(111, projection='3d')

    def _format_metadata(self, frame_info):
        dt = datetime.fromtimestamp(frame_info['timestamp'], tz=timezone.utc)
        return (f"TIMELINE:   {self.current_idx + 1} / {self.num_total_frames}\n"
                f"FILE IDX:   {frame_info['frame_idx']} ({frame_info['source']})\n"
                f"SPACECRAFT: {frame_info['sat_id']}\n"
                f"ACQUIRED:   {dt.strftime('%Y-%m-%d %H:%M:%S UTC')}")

    def _get_landsat_mask(self, data_grp, f_idx, shape):
        """Generates a boolean mask for LANDSAT data based on active UI filters."""
        valid_mask = np.ones(shape, dtype=bool)
        
        if not MASKING:
            return valid_mask
        
        # Sun Elevation Check
        sun_elev_arr = data_grp['surface_reflectance'].attrs['sun_elevation']
        if sun_elev_arr[f_idx] < self.sun_elev_thresh:
            return np.zeros(shape, dtype=bool)

        # QA Reject Mask
        if self.mask_qa_enabled:
            qa_pixel = data_grp['QUALITY_L1_PIXEL'][f_idx, ...]
            # True represents BAD pixels (Clouds/Shadows)
            bad_qa_mask = (qa_pixel & QA_REJECT_MASK) != 0
            if self.cloud_dilation > 0:
                kernel = np.ones((3, 3), dtype=bool)
                bad_qa_mask = ndimage.binary_dilation(bad_qa_mask, structure=kernel, iterations=self.cloud_dilation)
            # Valid pixels are where bad_qa_mask is False
            valid_mask &= ~bad_qa_mask

        # RADSAT Accept Value
        if self.mask_radsat_enabled:
            bad_radsat = data_grp['RADIOMETRIC_SATURATION'][f_idx, ...] != RADSAT_ACCEPT_VALUE
            kernel = np.ones((3, 3), dtype=bool)
            bad_radsat = ndimage.binary_dilation(bad_radsat, structure=kernel, iterations=1)
            valid_mask &= ~bad_radsat

        # Aerosol Accept Values
        if self.aerosol_level != 'all':
            aerosol = data_grp['QUALITY_L2_AEROSOL'][f_idx, ...]
            invalid_aerosol = ~np.isin(aerosol, AEROSOL_DICT[self.aerosol_level])
            kernel = np.ones((3, 3), dtype=bool)
            invalid_aerosol = ndimage.binary_dilation(invalid_aerosol, structure=kernel, iterations=1)
            valid_mask &= ~invalid_aerosol

        return valid_mask

    def _get_tanager_mask(self, data_grp, f_idx, shape):
        """Generates a boolean mask for TANAGER data based on active UI filters."""
        valid_mask = np.ones(shape, dtype=bool)
        
        if not MASKING:
            return valid_mask

        # Cloud Mask Check
        cloud_mask = (data_grp['beta_cloud_mask'][f_idx, ...]==1)
        cirrus_mask = (data_grp['beta_cirrus_mask'][f_idx, ...]==1)
        combined_cloud = cloud_mask | cirrus_mask
        if self.cloud_dilation > 0:
            kernel = np.ones((3, 3), dtype=bool)
            combined_cloud = ndimage.binary_dilation(combined_cloud, structure=kernel, iterations=self.cloud_dilation)
        valid_mask &= ~combined_cloud
        
        # Sun Elevation Check (Derived from Sun Zenith)
        zenith = data_grp['sun_zenith'][f_idx, ...]
        # Filter out fill value (-9999.0) and enforce elevation threshold
        valid_mask &= (zenith != -9999.0) & ((90.0 - zenith) >= self.sun_elev_thresh)
            
        # Aerosol Optical Depth Check
        aod = data_grp['aerosol_optical_depth'][f_idx, ...]
        # Filter out fill value (-9999.0) and enforce AOD threshold
        bad_aod_mask = (aod == -9999.0) | (aod <= self.t_aerosol_thresh)
        if self.t_aerosol_thresh > 0:
            kernel = np.ones((3, 3), dtype=bool)
            bad_aod_mask = ndimage.binary_dilation(bad_aod_mask, structure=kernel, iterations=1)
        valid_mask &= ~bad_aod_mask
            
        # Surface Reflectance Uncertainty Check
        gw_mask = data_grp['surface_reflectance'].attrs['all_good_wavelengths']
        valid_bands = gw_mask[f_idx].astype(bool)
        unc = np.nanmax(data_grp['surface_reflectance_uncertainty'][f_idx, valid_bands, ...], axis=0)
            
        # Filter out fill value (-9999.0) and enforce uncertainty threshold
        unc_mask = (unc == -9999.0) | (unc >= self.t_uncertainty_thresh)
        if self.t_uncertainty_thresh > 0:
            kernel = np.ones((3, 3), dtype=bool)
            unc_mask = ndimage.binary_dilation(unc_mask, structure=kernel, iterations=1)
        valid_mask &= ~unc_mask
            
        return valid_mask

    def update_display(self):
        frame_info = self.all_frames[self.current_idx]
        file_info = self.files[frame_info['file_idx']]
        f_idx = frame_info['frame_idx']
        data_grp = file_info['data_grp']
        
        curr_dt = datetime.fromtimestamp(frame_info['timestamp'], tz=timezone.utc)

        meta_str = self._format_metadata(frame_info)
        self.ctrl_text.set_text(meta_str)
        
        # Auto-update the scatter plot UI indices to match the currently viewed frames
        if hasattr(self, 'txt_l_frame') and hasattr(self, 'txt_t_frame'):
            if frame_info['source'] == 'LANDSAT':
                self.txt_l_frame.set_val(str(f_idx))
            elif frame_info['source'] == 'TANAGER':
                self.txt_t_frame.set_val(str(f_idx))
        
        # Add an additional line with the current active filters/thresholds or "Unmasked" status
        if MASKING:
            qa_val = bin(QA_REJECT_MASK) if self.mask_qa_enabled else "OFF"
            filter_str = f"Filters: Sun Elev > {self.sun_elev_thresh}° | Cloud Dilation: {self.cloud_dilation} | T-AOD < {self.t_aerosol_thresh} | L-Aerosol: {self.aerosol_level} | T-Unc < {self.t_uncertainty_thresh} | L-QA Rej: {qa_val}"
        else:
            filter_str = "Filters: Unmasked"
            
        self.combined_hud.set_text(meta_str.replace('\n', ' | ') + '\n' + filter_str)

        sr_data = data_grp['surface_reflectance'][f_idx, ...]

        # Strict execution block requires an ARD compliant visual array
        raw_vis = data_grp['ortho_visual'][f_idx, ...]
        
        # 1. Transform Band Sequential (BSQ) to Band Interleaved by Pixel (BIP)
        if raw_vis.shape[0] in [3, 4]:
            raw_vis = np.transpose(raw_vis, (1, 2, 0))
            
        # 2. Standardize array to float32 for Matplotlib RGBA rendering [0.0, 1.0]
        if raw_vis.dtype == np.uint8:
            rgba = raw_vis.astype(np.float32) / 255.0
        else:
            rgba = raw_vis.astype(np.float32)
            
        # any non-zero source alpha (>0) is valid/opaque (1.0).
        # Zeros are background fill and remain fully transparent (0.0).
        # This allows standard Matplotlib white backgrounds to display clearly through the data voids.
        rgba[..., 3] = np.where(rgba[..., 3] > 0, 1.0, 0.0)
        rgb = rgba

        hull_bands = HULL_BANDS_LANDSAT if frame_info['source'] == 'LANDSAT' else HULL_BANDS_TANAGER

        # --- Row 1: Spatial & Spectral Analysis ---
        self.ax_spatial.clear()
        self.ax_spatial.imshow(rgb)
        em_indices = data_grp['frame_endmember_indices'][f_idx]
        h, w = sr_data.shape[1:]
        for i, flat_idx in enumerate(em_indices):
            row, col = flat_idx // w, flat_idx % w
            self.ax_spatial.plot(col, row, 'r+', markersize=8)
            self.ax_spatial.annotate(f'V{i}', (col, row), color='yellow', fontsize=8, fontweight='bold')
        
        for loc in TS_LOCATIONS:
            y, x = loc['yx']
            self.ax_spatial.plot(x, y, marker='s', markersize=10, markeredgecolor=loc['color'], 
                                 markerfacecolor='none', markeredgewidth=1.5, linestyle='None')

        self.ax_spatial.set_title(f"EM Locations ({frame_info['source']})", color='black')
        self.ax_spatial.axis('off')

        if DISPLAY_REDUNDANT_FIGURE:
            self.ax_spatial_redundant.clear()
            self.ax_spatial_redundant.imshow(rgb)
            for i, flat_idx in enumerate(em_indices):
                row, col = flat_idx // w, flat_idx % w
                self.ax_spatial_redundant.plot(col, row, 'r+', markersize=8)
                self.ax_spatial_redundant.annotate(f'V{i}', (col, row), color='yellow', fontsize=8, fontweight='bold')
            for loc in TS_LOCATIONS:
                y, x = loc['yx']
                self.ax_spatial_redundant.plot(x, y, marker='s', markersize=10, markeredgecolor=loc['color'], 
                                     markerfacecolor='none', markeredgewidth=1.5, linestyle='None')
            self.ax_spatial_redundant.set_title("EM Locations")
            self.ax_spatial_redundant.axis('off')
            
            self.fig_redundant.suptitle(f"{frame_info['sat_id']} - {curr_dt.strftime('%Y-%m-%d %H:%M:%S UTC')}\n{filter_str}", fontsize=14)

        self.ax_spectral.clear()
        endmembers = data_grp['frame_endmembers'][f_idx, ...]
        wl = file_info['wavelengths']
        for i in range(endmembers.shape[1]):
            if not np.all(np.isnan(endmembers[:, i])) and np.any(endmembers[:, i] != 0):
                self.ax_spectral.plot(wl, endmembers[:, i], label=f'V{i}', lw=1)
        self.ax_spectral.set_title("Spectral Signatures")
        self.ax_spectral.set_xlabel("Wavelength (μm)") 
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

        def update_map(ax, dset, im_attr, cbar_attr, title):
            data = dset[f_idx].copy()
            
            # Dynamically apply UI masks for both sensors
            if frame_info['source'] == 'LANDSAT':
                mask = self._get_landsat_mask(data_grp, f_idx, data.shape)
                data[~mask] = np.nan
            elif frame_info['source'] == 'TANAGER':
                mask = self._get_tanager_mask(data_grp, f_idx, data.shape)
                data[~mask] = np.nan
            
            if DISPLAY_NORMALIZATION:
                data = percentile_normalize_array(data)
            
            curr_im = getattr(self, im_attr)
            curr_cbar = getattr(self, cbar_attr)

            if curr_im is None:
                new_im = ax.imshow(data, cmap='viridis')
                setattr(self, im_attr, new_im)
                
                new_cbar = ax.figure.colorbar(new_im, ax=ax, fraction=0.046, pad=0.04)
                setattr(self, cbar_attr, new_cbar)
                
                for loc in TS_LOCATIONS:
                    y, x = loc['yx']
                    ax.plot(x, y, marker='s', markersize=10, markeredgecolor=loc['color'], 
                            markerfacecolor='none', markeredgewidth=1.5, linestyle='None')
                
                ax.set_title(title)
                ax.axis('off')
            else:
                curr_im.set_data(data)
                with np.errstate(all='ignore'):
                    v_min, v_max = np.nanmin(data), np.nanmax(data)
                if not np.isnan(v_min) and not np.isnan(v_max):
                    curr_im.set_clim(vmin=v_min, vmax=v_max)
                curr_cbar.update_normal(curr_im)

        update_map(self.ax_slide_map, data_grp[complexity_type], 'im_slide', 'cbar_slide', "Sliding Complexity")
        
        if DISPLAY_REDUNDANT_FIGURE:
            update_map(self.ax_slide_map_redundant, data_grp[complexity_type], 'im_slide_redundant', 'cbar_slide_redundant', "Sliding Complexity")
        
        # --- Time Series Construction and Plotting ---
        self.ax_ts_main.clear()
        if self.ax_ts_twin is not None:
            try:
                self.ax_ts_twin.remove()
            except Exception:
                pass
            self.ax_ts_twin = None
            
        if self.use_twin_axis:
            self.ax_ts_twin = self.ax_ts_main.twinx()
            
        if DISPLAY_REDUNDANT_FIGURE:
            self.ax_ts_redundant_main.clear()
            if self.ax_ts_redundant_twin is not None:
                try:
                    self.ax_ts_redundant_twin.remove()
                except Exception:
                    pass
                self.ax_ts_redundant_twin = None
            if self.use_twin_axis:
                self.ax_ts_redundant_twin = self.ax_ts_redundant_main.twinx()

        def plot_time_series(ax_main, ax_twin=None):
            """Internal helper method to cleanly plot the time series to any given axis."""
            t_ax = ax_twin if ax_twin is not None else ax_main
            
            # Plot LANDSAT Time Series within Date Range
            for loc in TS_LOCATIONS:
                label = loc['label']
                data = self.ts_data['LANDSAT'][label]
                if data['t']:
                    filt_t, filt_v = [], []
                    for i in range(len(data['t'])):
                        if self.ts_start_date <= data['t'][i] <= self.ts_end_date:
                            filt_t.append(data['t'][i])
                            filt_v.append(data['v'][i])
                    
                    if filt_t:
                        ax_main.plot(filt_t, filt_v, marker='^', color=loc['color'], label=f"L: {label}",
                                              markersize=4, linestyle='--', linewidth=1, alpha=0.6)
            
            # Plot TANAGER Time Series within Date Range
            for loc in TS_LOCATIONS:
                label = loc['label']
                data = self.ts_data['TANAGER'][label]
                if data['t']:
                    filt_t, filt_v = [], []
                    for i in range(len(data['t'])):
                        if self.ts_start_date <= data['t'][i] <= self.ts_end_date:
                            filt_t.append(data['t'][i])
                            filt_v.append(data['v'][i])
                            
                    if filt_t:
                        t_ax.plot(filt_t, filt_v, marker='s', color=loc['color'], label=f"T: {label}",
                                              markersize=5, linestyle='-', linewidth=1.5, alpha=0.9)

            if self.all_frames and len(ax_main.lines) > 0:
                xlims = ax_main.get_xlim() # Capture limits generated by the data
                
                # Constrain background shading only to the requested/active viewing years
                for yr in range(self.ts_start_date.year, self.ts_end_date.year + 2):
                    # Winter (Dec 1 prev year - Mar 1 curr year) -> light gray
                    ax_main.axvspan(datetime(yr - 1, 12, 1, tzinfo=timezone.utc), 
                                            datetime(yr, 3, 1, tzinfo=timezone.utc), 
                                            color='lightgray', alpha=0.3, zorder=0, lw=0)
                    # Spring (Mar 1 - Jun 1) -> light green
                    ax_main.axvspan(datetime(yr, 3, 1, tzinfo=timezone.utc), 
                                            datetime(yr, 6, 1, tzinfo=timezone.utc), 
                                            color='lightgreen', alpha=0.2, zorder=0, lw=0)
                    # Summer (Jun 1 - Sep 1) -> light yellow
                    ax_main.axvspan(datetime(yr, 6, 1, tzinfo=timezone.utc), 
                                            datetime(yr, 9, 1, tzinfo=timezone.utc), 
                                            color='lightyellow', alpha=0.3, zorder=0, lw=0)
                    # Fall (Sep 1 - Dec 1) -> light orange
                    ax_main.axvspan(datetime(yr, 9, 1, tzinfo=timezone.utc), 
                                            datetime(yr, 12, 1, tzinfo=timezone.utc), 
                                            color='orange', alpha=0.15, zorder=0, lw=0)
                
                ax_main.set_xlim(xlims) # Restore limits so it doesn't zoom out to empty seasons

            # Style and Layout the Time Series UI
            title_str = "Twin Axis Time Series" if self.use_twin_axis else "Equated Time Series (Shared Scale)"
            ax_main.set_title(f"{title_str} ({self.ts_start_date.strftime('%Y-%m-%d')} to {self.ts_end_date.strftime('%Y-%m-%d')})")
            
            ax_main.grid(True, alpha=0.3, which="both", ls="--")
            ax_main.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax_main.tick_params(axis='x', rotation=45, labelsize=8)
            ax_main.axvline(curr_dt, color='black', linestyle='--', alpha=0.8, linewidth=1.5)
            
            # Format Legends and Y Labels
            lines_1, labels_1 = ax_main.get_legend_handles_labels()
            if self.use_twin_axis and ax_twin is not None:
                ax_main.set_ylabel("Landsat Volume", color='black', fontweight='bold')
                ax_twin.set_ylabel("Tanager Volume", color='black', fontweight='bold')
                lines_2, labels_2 = ax_twin.get_legend_handles_labels()
                ax_main.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', fontsize=8, ncol=2)
            else:
                ax_main.set_ylabel("Volume", fontweight='bold')
                ax_main.legend(loc='upper left', fontsize=8, ncol=2)

        # Plot onto the primary layout
        plot_time_series(self.ax_ts_main, self.ax_ts_twin)
        
        # Plot onto the new standalone redundant layout
        if DISPLAY_REDUNDANT_FIGURE:
            plot_time_series(self.ax_ts_redundant_main, self.ax_ts_redundant_twin)
            self.fig_ts_redundant.suptitle(f"Time Series Extraction | {frame_info['sat_id']} - {curr_dt.strftime('%Y-%m-%d %H:%M:%S UTC')}\n{filter_str}", fontsize=14)

        # --- 3D Parallelotope Figure ---
        self.ax_hull.clear()
        pixel_data = sr_data.reshape(sr_data.shape[0], -1).T
        valid_mask = ~np.isnan(pixel_data).any(axis=1)
        pixel_data = pixel_data[valid_mask]
        
        b1, b2, b3 = [min(b, sr_data.shape[0]-1) for b in hull_bands]
        
        # Compute Dataset Mean using the full valid frame prior to random subsampling
        if pixel_data.shape[0] > 0:
            mean_dataset_full = np.mean(pixel_data[:, [b1, b2, b3]], axis=0)
        else:
            mean_dataset_full = np.array([0.0, 0.0, 0.0])
            
        if pixel_data.shape[0] > 1500:
            pixel_data = pixel_data[np.random.choice(pixel_data.shape[0], 4000, replace=False)]
        
        self.ax_hull.scatter(pixel_data[:, b1], pixel_data[:, b2], pixel_data[:, b3], c='gray', alpha=0.1, s=1)
        
        # Use all valid endmembers for basis vectors
        em_xyz = endmembers[[b1, b2, b3], :].T
        valid_em_mask = ~np.all(em_xyz == 0, axis=1) & ~np.isnan(em_xyz).any(axis=1)
        valid_em_xyz = em_xyz[valid_em_mask]
        
        self.ax_hull.scatter(valid_em_xyz[:, 0], valid_em_xyz[:, 1], valid_em_xyz[:, 2], c='red', s=40, label='Endmembers')
        
        origin = np.array([0.0, 0.0, 0.0])
        v1 = v2 = v3 = None
        
        # Map origin and basis vectors to exactly mirror Gramian localization mathematical shifts
        if self.localization_mode == 'datasetMean':
            origin = mean_dataset_full
            self.ax_hull.scatter(origin[0], origin[1], origin[2], c='blue', s=80, marker='X', label='Dataset Mean')
            if valid_em_xyz.shape[0] >= 3:
                v1 = valid_em_xyz[0] - origin
                v2 = valid_em_xyz[1] - origin
                v3 = valid_em_xyz[2] - origin
        elif self.localization_mode == 'minEndmember':
            if valid_em_xyz.shape[0] >= 4:
                # Based on maximumDistance logic, index 1 is the minimum magnitude endmember
                origin = valid_em_xyz[1]
                self.ax_hull.scatter(origin[0], origin[1], origin[2], c='blue', s=80, marker='X', label='Min EM (Origin)')
                # Use 1st (0), 3rd (2), and 4th (3) endmembers as the basis vectors
                v1 = valid_em_xyz[0] - origin
                v2 = valid_em_xyz[2] - origin
                v3 = valid_em_xyz[3] - origin
        else: # 'general'
            origin = np.array([0.0, 0.0, 0.0])
            self.ax_hull.scatter(origin[0], origin[1], origin[2], c='blue', s=80, marker='X', label='Origin (0,0,0)')
            if valid_em_xyz.shape[0] >= 3:
                v1 = valid_em_xyz[0]
                v2 = valid_em_xyz[1]
                v3 = valid_em_xyz[2]
        
        # Draw the explicit 3D parallelotope defined by the shifted origin and valid basis vectors
        if v1 is not None and v2 is not None and v3 is not None:
            # Draw basis vectors from the computed origin
            self.ax_hull.plot([origin[0], origin[0] + v1[0]], [origin[1], origin[1] + v1[1]], [origin[2], origin[2] + v1[2]], 'r--', alpha=0.8, linewidth=2)
            self.ax_hull.plot([origin[0], origin[0] + v2[0]], [origin[1], origin[1] + v2[1]], [origin[2], origin[2] + v2[2]], 'r--', alpha=0.8, linewidth=2)
            self.ax_hull.plot([origin[0], origin[0] + v3[0]], [origin[1], origin[1] + v3[1]], [origin[2], origin[2] + v3[2]], 'r--', alpha=0.8, linewidth=2)
            
            # Define the 8 vertices of the parallelotope anchored at the origin
            vertices = np.array([
                origin, 
                origin + v1, 
                origin + v2, 
                origin + v3,
                origin + v1 + v2, 
                origin + v1 + v3, 
                origin + v2 + v3,
                origin + v1 + v2 + v3
            ])
            
            # Define the 12 edges
            edges = [
                (0,1), (0,2), (0,3),
                (1,4), (1,5),
                (2,4), (2,6),
                (3,5), (3,6),
                (4,7), (5,7), (6,7)
            ]
            
            for edge in edges:
                p1, p2 = vertices[edge[0]], vertices[edge[1]]
                self.ax_hull.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 'b-', alpha=0.4, linewidth=1.5)
        
        self.ax_hull.set_title(f"3D Parallelotope: Bands {b1}, {b2}, {b3}\nMode: {self.localization_mode}")
        self.ax_hull.set_xlabel(f"B{b1}"); self.ax_hull.set_ylabel(f"B{b2}"); self.ax_hull.set_zlabel(f"B{b3}")
        self.ax_hull.legend()

        # Refresh
        figs_to_draw = [self.fig_controls, self.fig_combined, self.fig_hull]
        if DISPLAY_REDUNDANT_FIGURE:
            figs_to_draw.append(self.fig_redundant)
            figs_to_draw.append(self.fig_ts_redundant)
        
        for f in figs_to_draw:
            f.canvas.draw_idle()

    # --- UI Callbacks ---
    def _refresh_if_scatter_open(self):
        """Forces the scatter plot to redraw if any mask states have changed."""
        if self.fig_scatter is not None and plt.fignum_exists(self.fig_scatter.number):
            self._on_update_scatter(None)
            
    def _on_localization_change(self, label):
        """Dynamically triggers geometric re-rendering of the Parallelotope projection."""
        self.localization_mode = label
        self.update_display()
            
    def _on_ts_axis_toggle(self, label):
        """Real-time rendering toggle. Twin Y-Axis does not require complex re-calculations."""
        self.use_twin_axis = not self.use_twin_axis
        self.update_display()

    def _on_update_mask(self, event):
        """Applies all selected filter values and triggers the full recomputation."""
        # 1. Read Checkboxes
        chk_status = self.chk_masks.get_status()
        self.mask_qa_enabled = chk_status[0]
        self.mask_radsat_enabled = chk_status[1]

        # 2. Read Radio Buttons
        self.aerosol_level = self.rad_aerosol.value_selected

        # 3. Read TextBoxes (with validation fallback)
        try:
            val = float(self.txt_sun.text)
            if 0 <= val <= 90:
                self.sun_elev_thresh = val
            else:
                self.txt_sun.set_val(str(self.sun_elev_thresh))
        except ValueError:
            self.txt_sun.set_val(str(self.sun_elev_thresh))

        try:
            val = int(self.txt_cdil.text)
            if val >= 0:
                self.cloud_dilation = val
            else:
                self.txt_cdil.set_val(str(self.cloud_dilation))
        except ValueError:
            self.txt_cdil.set_val(str(self.cloud_dilation))

        try:
            val = float(self.txt_t_aod.text)
            self.t_aerosol_thresh = val
        except ValueError:
            self.txt_t_aod.set_val(str(self.t_aerosol_thresh))

        try:
            val = float(self.txt_t_unc.text)
            self.t_uncertainty_thresh = val
        except ValueError:
            self.txt_t_unc.set_val(str(self.t_uncertainty_thresh))

        # 4. Read Date Constraints
        try:
            self.ts_start_date = datetime.strptime(self.txt_ts_start.text.strip(), "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            self.txt_ts_start.set_val(self.ts_start_date.strftime("%Y-%m-%d"))
            
        try:
            self.ts_end_date = datetime.strptime(self.txt_ts_end.text.strip(), "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            self.txt_ts_end.set_val(self.ts_end_date.strftime("%Y-%m-%d"))

        # 5. Trigger full recomputation
        self._recompute_time_series()
        self.update_display()
        self._refresh_if_scatter_open()

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

    def _on_update_scatter(self, event):
        if self.l_file_idx is None or self.t_file_idx is None:
            print("Error: Both LANDSAT and TANAGER files must be loaded for the scatter plot.")
            return

        try:
            l_idx = int(self.txt_l_frame.text)
            t_idx = int(self.txt_t_frame.text)
        except ValueError:
            print("Invalid frame indices provided. Please enter integers.")
            return

        l_grp = self.files[self.l_file_idx]['data_grp']
        t_grp = self.files[self.t_file_idx]['data_grp']

        l_slide_dset = l_grp[complexity_type]
        t_slide_dset = t_grp[complexity_type]

        if l_idx < 0 or l_idx >= l_slide_dset.shape[0] or t_idx < 0 or t_idx >= t_slide_dset.shape[0]:
            print("Error: Selected frame index out of bounds.")
            return

        l_time_arr = l_grp['surface_reflectance'].attrs.get('acquisition_time')
        t_time_arr = t_grp['surface_reflectance'].attrs.get('acquisition_time')
        l_date_str = datetime.fromtimestamp(l_time_arr[l_idx], tz=timezone.utc).strftime('%Y-%m-%d') if l_time_arr is not None else "Unknown Date"
        t_date_str = datetime.fromtimestamp(t_time_arr[t_idx], tz=timezone.utc).strftime('%Y-%m-%d') if t_time_arr is not None else "Unknown Date"

        # Pull data and apply identical dynamic masks to the scatter plot data
        l_data = l_slide_dset[l_idx, ...].copy()
        if 'LANDSAT' in self.files[self.l_file_idx]['source'].upper():
            l_mask = self._get_landsat_mask(l_grp, l_idx, l_data.shape)
            l_data[~l_mask] = np.nan
            
        t_data = t_slide_dset[t_idx, ...].copy()
        if 'TANAGER' in self.files[self.t_file_idx]['source'].upper():
            t_mask = self._get_tanager_mask(t_grp, t_idx, t_data.shape)
            t_data[~t_mask] = np.nan

        # Assure spatial dimensions match for 1:1 scatter comparison
        h = min(l_data.shape[0], t_data.shape[0])
        w = min(l_data.shape[1], t_data.shape[1])
        
        l_flat = l_data[:h, :w].flatten()
        t_flat = t_data[:h, :w].flatten()

        # Remove NaNs and strict zeros
        valid_mask = (~np.isnan(l_flat)) & (~np.isnan(t_flat)) & (l_flat > 0) & (t_flat > 0)
        l_valid = l_flat[valid_mask]
        t_valid = t_flat[valid_mask]
        
        if len(l_valid) > 0:
            ratios = t_valid / l_valid
            median_ratio = np.median(ratios)
            mean_ratio = np.mean(ratios)
            pearsonFit, _ = pearsonr(l_valid, t_valid)
            spearmanFit, _ = spearmanr(l_valid, t_valid)
            
            # Calculate Linear Fit
            lin_slope, lin_intercept = np.polyfit(l_valid, t_valid, 1)
            
            stats_text_scatter = (f"Tiles Analyzed: {len(l_valid)}\n"
                                  f"Pearson r: {pearsonFit:.4f}\n"
                                  f"Spearman r: {spearmanFit:.4f}\n"
                                  f"Median Ratio: {median_ratio:.4f}\n"
                                  f"Mean Ratio: {mean_ratio:.4f}\n"
                                  f"Linear Fit: T = {lin_slope:.2f}L + {lin_intercept:.2f}")

            log_l = np.log10(l_valid)
            log_t = np.log10(t_valid)
            logRatios = log_t / log_l
            median_logRatio = np.median(logRatios)
            mean_logRatio = np.mean(logRatios)
            log_pearson, _ = pearsonr(log_l, log_t)
            log_spearman, _ = spearmanr(log_l, log_t)
            
            # Calculate Log-Log Fit (Exponential)
            log_slope, log_intercept = np.polyfit(log_l, log_t, 1)
            
            logStats_text_scatter = (f"Tiles Analyzed: {len(l_valid)}\n"
                                  f"Log Pearson r: {log_pearson:.4f}\n"
                                  f"Log Spearman r: {log_spearman:.4f}\n"
                                  f"Median Log Ratio: {median_logRatio:.4f}\n"
                                  f"Mean Log Ratio: {mean_logRatio:.4f}\n"
                                  f"Exp Fit: T = {10**log_intercept:.2f} * L^{log_slope:.2f}")
        else:
            stats_text_scatter = logStats_text_scatter = "N/A"

        if self.fig_scatter is None or not plt.fignum_exists(self.fig_scatter.number):
            self.fig_scatter = plt.figure(figsize=(12, 10))
            self.fig_scatter.canvas.manager.set_window_title("Sliding Volume Correlation Scatter")
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
            # --- Linear Subplot ---
            self.ax_scatter_lin.scatter(l_valid, t_valid, alpha=0.3, s=10, label='Sliding Window Tiles', color='tab:purple')
            
            # Plot Linear Regression Line
            l_range_lin = np.array([np.min(l_valid), np.max(l_valid)])
            t_fit_lin = lin_slope * l_range_lin + lin_intercept
            self.ax_scatter_lin.plot(l_range_lin, t_fit_lin, color='red', linewidth=2, label='Linear Fit')
            
            self.ax_scatter_lin.set_title(f"Linear Scale\nLANDSAT ({l_date_str}) vs TANAGER ({t_date_str})")
            self.ax_scatter_lin.set_xlabel(f"LANDSAT Volume")
            self.ax_scatter_lin.set_ylabel(f"TANAGER Volume")
            self.ax_scatter_lin.grid(True, alpha=0.3)
            if MASKING:
                qa_val = bin(QA_REJECT_MASK) if self.mask_qa_enabled else "OFF"
                filter_str = f"Filters: Sun Elev > {self.sun_elev_thresh}° | Cloud Dilation: {self.cloud_dilation} | T-AOD < {self.t_aerosol_thresh} | L-Aerosol: {self.aerosol_level} | T-Unc < {self.t_uncertainty_thresh} | L-QA Rej: {qa_val}"
            else:
                filter_str = "Filters: Unmasked"
                
            self.ax_scatter_lin.text(0.95, 0.05, stats_text_scatter, transform=self.ax_scatter_lin.transAxes, 
                                     ha='right', va='bottom', fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
            self.ax_scatter_lin.legend()

            # --- Log-Log Subplot ---
            self.ax_scatter_log.scatter(l_valid, t_valid, alpha=0.3, s=10, label='Sliding Window Tiles', color='tab:orange')
            
            # Plot Log-Log Regression Line
            l_range_log = np.array([np.min(l_valid), np.max(l_valid)])
            t_fit_log = (10**log_intercept) * (l_range_log ** log_slope)
            self.ax_scatter_log.plot(l_range_log, t_fit_log, color='red', linewidth=2, label='Exponential Fit')
            
            self.ax_scatter_log.set_title(f"Log-Log Scale\nLANDSAT ({l_date_str}) vs TANAGER ({t_date_str})")
            self.ax_scatter_log.set_xlabel(f"LANDSAT Volume")
            self.ax_scatter_log.set_ylabel(f"TANAGER Volume")
            self.ax_scatter_log.set_xscale('log')
            self.ax_scatter_log.set_yscale('log')
            self.ax_scatter_log.grid(True, alpha=0.3, which="both", ls="--")
            self.ax_scatter_log.text(0.95, 0.05, logStats_text_scatter, transform=self.ax_scatter_log.transAxes, 
                                     ha='right', va='bottom', fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
            self.ax_scatter_log.legend()
            
            # --- Landsat Histogram Subplot (Log-Scaled Bins) ---
            # Calculate 256 bin edges equally spaced in log10 space
            bins_l = np.logspace(np.log10(np.min(l_valid)), np.log10(np.max(l_valid)), 256)
            self.ax_hist_l.hist(l_valid, bins=bins_l, color='tab:purple', alpha=0.7)
            self.ax_hist_l.set_xscale('log')
            self.ax_hist_l.set_title(f"LANDSAT Volume Distribution\n({l_date_str})")
            self.ax_hist_l.set_xlabel("Volume (Log Scale)")
            self.ax_hist_l.set_ylabel("Frequency")
            self.ax_hist_l.grid(True, alpha=0.3, which="both", ls="--")
            
            l_stats_text = (f"Mean: {np.mean(l_valid):.4e}\n"
                            f"Median: {np.median(l_valid):.4e}\n"
                            f"Variance: {np.var(l_valid):.4e}")
            self.ax_hist_l.text(0.95, 0.95, l_stats_text, transform=self.ax_hist_l.transAxes, 
                                ha='right', va='top', fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            # --- Tanager Histogram Subplot (Log-Scaled Bins) ---
            bins_t = np.logspace(np.log10(np.min(t_valid)), np.log10(np.max(t_valid)), 256)
            self.ax_hist_t.hist(t_valid, bins=bins_t, color='tab:orange', alpha=0.7)
            self.ax_hist_t.set_xscale('log')
            self.ax_hist_t.set_title(f"TANAGER Volume Distribution\n({t_date_str})")
            self.ax_hist_t.set_xlabel("Volume (Log Scale)")
            self.ax_hist_t.set_ylabel("Frequency")
            self.ax_hist_t.grid(True, alpha=0.3, which="both", ls="--")
            
            t_stats_text = (f"Mean: {np.mean(t_valid):.4e}\n"
                            f"Median: {np.median(t_valid):.4e}\n"
                            f"Variance: {np.var(t_valid):.4e}")
            self.ax_hist_t.text(0.95, 0.95, t_stats_text, transform=self.ax_hist_t.transAxes, 
                                ha='right', va='top', fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            self.fig_scatter.suptitle(f"Sliding Volume Correlation | LANDSAT ({l_date_str}) vs TANAGER ({t_date_str})\n{filter_str}", fontsize=14)
            self.fig_scatter.tight_layout(rect=[0, 0.03, 1, 0.95]) # Prevent suptitle overlap
        else:
            self.ax_scatter_lin.set_title(f"Sliding Volume Correlation (Linear)")
            self.ax_scatter_lin.text(0.5, 0.5, "No valid data to plot.", ha='center', va='center', transform=self.ax_scatter_lin.transAxes)
            self.ax_scatter_log.set_title(f"Sliding Volume Correlation (Log-Log)")
            self.ax_scatter_log.text(0.5, 0.5, "No valid data to plot.", ha='center', va='center', transform=self.ax_scatter_log.transAxes)
            self.ax_hist_l.set_title("LANDSAT Volume Distribution")
            self.ax_hist_l.text(0.5, 0.5, "No valid data to plot.", ha='center', va='center', transform=self.ax_hist_l.transAxes)
            self.ax_hist_t.set_title("TANAGER Volume Distribution")
            self.ax_hist_t.text(0.5, 0.5, "No valid data to plot.", ha='center', va='center', transform=self.ax_hist_t.transAxes)

        self.fig_scatter.canvas.draw_idle()
        self.fig_scatter.show()

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
        
        # Determine output directory strictly from parameters (Fail fast if missing)
        vol_dset = data_grp['frame_endmember_volumes']
        num_em = vol_dset.attrs.get('num_endmembers', 'X')
        gram = vol_dset.attrs.get('gram_type', 'X')
        norm = vol_dset.attrs.get('Normalization', 'None')
        # Handle possible byte strings or None
        if hasattr(norm, 'decode'): norm = norm.decode('utf-8')
        if norm is None: norm = "None"
        
        new_dir = f"{SAVE_DIR}_EM-{num_em}_Gram-{gram}_Norm-{norm}/"
        os.makedirs(new_dir, exist_ok=True)
        save_path = new_dir

        time_str = datetime.fromtimestamp(info['timestamp'], tz=timezone.utc).strftime('%Y%m%d_%H%M%S')
        prefix = f"{time_str}_{info['source']}_{self.current_idx:02d}"
        
        figs_to_save = [(self.fig_combined, "CombinedAnalysis")]
        if DISPLAY_REDUNDANT_FIGURE:
            figs_to_save.append((self.fig_redundant, "SpatialComplexityDetails"))
            figs_to_save.append((self.fig_ts_redundant, "TimeSeriesDetail"))
            
        for fig, name in figs_to_save:
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
    viewer = MultiComplexityViewer([landsat_path, tanager_path])
    viewer.run()
    root.destroy()