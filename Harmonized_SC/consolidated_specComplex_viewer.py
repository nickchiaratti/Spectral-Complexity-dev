import os
import sys
import h5py
import numpy as np
import json
from pathlib import Path

script_dir = Path(__file__).resolve().parent
if str(script_dir.parent) not in sys.path:
    sys.path.insert(0, str(script_dir.parent))
    
# Import original viewer
from Harmonized_SC.HLST_specComplex_viewer import *
import Harmonized_SC.HLST_specComplex_viewer as orig

# Override the metric names to match the new Native SC architecture
orig.complexity_type = 'sliding_volume_3x3_zscore'
orig.complexity_type_comparison = 'ndvi'

class MappedDataset:
    def __init__(self, const_h5, target_metric):
        self.h5 = const_h5
        self.metric = target_metric
        self.timeline = self.h5['timeline/acquisition_time'][:]
        self.sensors = self.h5['timeline/source_sensor'][:]
        self.frame_indices = self.h5['timeline/source_frame_index'][:]
        
        self.target_h = self.h5['target_grid'].attrs['height']
        self.target_w = self.h5['target_grid'].attrs['width']
        self.shape = (len(self.timeline), self.target_h, self.target_w)
        self.attrs = {} # Mock attrs for the GUI
        
    def _read_frame(self, idx):
        # Reads a full frame using the index map
        sk = self.sensors[idx].decode('utf-8').split(',')[0]
        fidx = json.loads(self.frame_indices[idx].decode('utf-8'))[0]
        
        sg = self.h5[f'sensors/{sk}']
        row_map = sg['row_map'][:]
        col_map = sg['col_map'][:]
        valid = sg['valid_mask'][:]
        
        try:
            metric_ds = sg['derived_fields'][self.metric]
        except KeyError:
            metric_ds = sg['data_fields'][self.metric]
            
        native_frame = metric_ds[fidx]
        
        # Apply scaling if needed (the viewer expects scaled float arrays)
        if 'scale_to_float' in metric_ds.attrs:
            scale = metric_ds.attrs['scale_to_float']
            fill = metric_ds.attrs.get('_FillValue', -9999)
            valid_nf = (native_frame != fill)
            nf_float = native_frame.astype(np.float32)
            nf_float[valid_nf] *= scale
            nf_float[~valid_nf] = np.nan
            native_frame = nf_float
            
        frame = np.full((self.target_h, self.target_w), np.nan, dtype=np.float32)
        frame[valid] = native_frame[row_map[valid], col_map[valid]]
        return frame
        
    def __getitem__(self, key):
        if isinstance(key, int):
            return self._read_frame(key)
        elif isinstance(key, tuple):
            if len(key) == 3 and key[0] == slice(None) and isinstance(key[1], int) and isinstance(key[2], int):
                # Requesting a timeseries for a specific pixel
                y, x = key[1], key[2]
                ts = np.full(self.shape[0], np.nan, dtype=np.float32)
                for i in range(self.shape[0]):
                    # Check if pixel is valid for this frame before fetching whole native frame
                    sk = self.sensors[i].decode('utf-8').split(',')[0]
                    sg = self.h5[f'sensors/{sk}']
                    valid_mask = sg['valid_mask'][:]
                    if not valid_mask[y, x]:
                        continue
                        
                    fidx = json.loads(self.frame_indices[i].decode('utf-8'))[0]
                    row_map = sg['row_map'][:]
                    col_map = sg['col_map'][:]
                    ny, nx = row_map[y, x], col_map[y, x]
                    
                    try:
                        metric_ds = sg['derived_fields'][self.metric]
                    except KeyError:
                        metric_ds = sg['data_fields'][self.metric]
                        
                    native_val = metric_ds[fidx, ny, nx]
                    
                    if 'scale_to_float' in metric_ds.attrs:
                        scale = metric_ds.attrs['scale_to_float']
                        fill = metric_ds.attrs.get('_FillValue', -9999)
                        if native_val != fill:
                            ts[i] = native_val * scale
                    else:
                        ts[i] = native_val
                return ts
            elif len(key) == 2 and isinstance(key[0], int) and key[1] == Ellipsis:
                return self._read_frame(key[0])
        return np.full((self.target_h, self.target_w), np.nan, dtype=np.float32)

class NativeDataset:
    def __init__(self, const_h5, target_metric, sensor):
        self.h5 = const_h5
        self.metric = target_metric
        self.sensor = sensor
        
        try:
            self.metric_ds = self.h5[f'/HDFEOS/GRIDS/{self.sensor}/Derived Fields/{self.metric}']
        except KeyError:
            self.metric_ds = self.h5[f'/HDFEOS/GRIDS/{self.sensor}/Data Fields/{self.metric}']
            
        n_frames, h, w = self.metric_ds.shape
        self.shape = (n_frames, h, w)
        self.target_h = h
        self.target_w = w
        self.attrs = {} # Mock attrs for the GUI
        
    def __getitem__(self, key):
        if isinstance(key, int):
            native_frame = self.metric_ds[key]
            if 'scale_to_float' in self.metric_ds.attrs:
                scale = self.metric_ds.attrs['scale_to_float']
                fill = self.metric_ds.attrs.get('_FillValue', -9999)
                valid_nf = (native_frame != fill)
                nf_float = native_frame.astype(np.float32)
                nf_float[valid_nf] *= scale
                nf_float[~valid_nf] = np.nan
                native_frame = nf_float
            return native_frame
        elif isinstance(key, tuple):
            if len(key) == 3 and key[0] == slice(None) and isinstance(key[1], int) and isinstance(key[2], int):
                y, x = key[1], key[2]
                ts = np.full(self.shape[0], np.nan, dtype=np.float32)
                for i in range(self.shape[0]):
                    native_val = self.metric_ds[i, y, x]
                    if 'scale_to_float' in self.metric_ds.attrs:
                        scale = self.metric_ds.attrs['scale_to_float']
                        fill = self.metric_ds.attrs.get('_FillValue', -9999)
                        if native_val != fill:
                            ts[i] = native_val * scale
                    else:
                        ts[i] = native_val
                return ts
            elif len(key) == 2 and isinstance(key[0], int) and key[1] == Ellipsis:
                return self.__getitem__(key[0])
        return np.full((self.target_h, self.target_w), np.nan, dtype=np.float32)

old_init = orig.HarmonizedComplexityViewer.__init__

def new_init(self, file_path):
    self.file_path = file_path
    self.h5 = h5py.File(file_path, 'r')
    
    self.is_native = 'timeline' not in self.h5
    if self.is_native:
        grids = list(self.h5['/HDFEOS/GRIDS'].keys())
        sensor = grids[0]
        
        self.base_dset = NativeDataset(self.h5, orig.complexity_type, sensor)
        self.base_dset_comp = NativeDataset(self.h5, orig.complexity_type_comparison, sensor)
        self.common_mask_dset = NativeDataset(self.h5, 'common_mask', sensor)
        
        self.total_frames = self.base_dset.shape[0]
        self.height = self.base_dset.shape[1]
        self.width = self.base_dset.shape[2]
        
        self.prov_grid = [sensor] * self.total_frames
        self.prov_space = [sensor] * self.total_frames
        self.prov_time = [0] * self.total_frames
        self.prov_idx = list(range(self.total_frames))
        
        self.wavelengths = {sensor: np.zeros(10)}
        try:
            wl = self.h5[f'/HDFEOS/GRIDS/{sensor}/Data Fields/surface_reflectance'].attrs['wavelengths'][:]
            self.wavelengths[sensor] = wl
        except:
            pass
            
        self.l_file_idx = 0 if 'HLS' in sensor else None
        self.t_file_idx = 0 if 'TANAGER' in sensor else None
        self.e_file_idx = 0 if 'ENMAP' in sensor else None
        self.d_file_idx = 0 if 'DRAGONETTE' in sensor else None
        
        sr_ds = self.h5[f'/HDFEOS/GRIDS/{sensor}/Data Fields/surface_reflectance']
        geo_transform = sr_ds.attrs['GeoTransform']
        spatial_ref = sr_ds.attrs['spatial_ref']
    else:
        self.base_dset = MappedDataset(self.h5, orig.complexity_type)
        self.base_dset_comp = MappedDataset(self.h5, orig.complexity_type_comparison)
        self.common_mask_dset = MappedDataset(self.h5, 'common_mask')
        
        self.total_frames = self.base_dset.shape[0]
        self.height = self.base_dset.shape[1]
        self.width = self.base_dset.shape[2]
        
        sensors = self.h5['timeline/source_sensor'][:]
        self.prov_grid = [s.decode('utf-8').split(',')[0] for s in sensors]
        scs = self.h5['timeline/spacecraft_id'][:]
        self.prov_space = [s.decode('utf-8').split(',')[0] for s in scs]
        self.prov_time = self.h5['timeline/acquisition_time'][:]
        fidxs = self.h5['timeline/source_frame_index'][:]
        self.prov_idx = [json.loads(f.decode('utf-8'))[0] for f in fidxs]
        
        self.wavelengths = {}
        for sk in self.h5['sensors'].keys():
            sg = self.h5[f'sensors/{sk}']
            try:
                wl = sg['data_fields']['surface_reflectance'].attrs['wavelengths'][:]
                self.wavelengths[sk] = wl
            except:
                self.wavelengths[sk] = np.zeros(10)
                
        self.l_file_idx = next((i for i, g in enumerate(self.prov_grid) if 'HLS' in g.upper()), None)
        self.t_file_idx = next((i for i, g in enumerate(self.prov_grid) if 'TANAGER' in g.upper()), None)
        self.e_file_idx = next((i for i, g in enumerate(self.prov_grid) if 'ENMAP' in g.upper()), None)
        self.d_file_idx = next((i for i, g in enumerate(self.prov_grid) if 'DRAGONETTE' in g.upper()), None)
        
        geo_transform = self.h5['target_grid'].attrs['GeoTransform']
        spatial_ref = self.h5['target_grid'].attrs['spatial_ref']
    if isinstance(spatial_ref, bytes):
        spatial_ref = spatial_ref.decode('utf-8')
        
    crs = CRS.from_wkt(spatial_ref)
    transformer = orig.Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    if len(geo_transform) == 9 or len(geo_transform) == 6 and abs(geo_transform[0]) < 1000:
        # It's an Affine tuple (a, b, c, d, e, f)
        affine = orig.rasterio.transform.Affine(*geo_transform[:6])
    else:
        # It's a GDAL tuple (c, a, b, f, d, e)
        affine = orig.rasterio.transform.Affine.from_gdal(*geo_transform)
    inv_affine = ~affine
    
    filename = os.path.basename(file_path)
    parts = filename.split('_')
    resolved_location = orig.Location
    if len(parts) > 2 and parts[1] == "Constellation":
        resolved_location = parts[2].replace('.h5', '')
        
    if resolved_location in orig.TS_LOCATIONS_MAP:
        orig.TS_LOCATIONS = orig.TS_LOCATIONS_MAP[resolved_location]
    else:
        ul_x, ul_y = affine * (0, 0)
        lr_x, lr_y = affine * (self.width, self.height)
        transformer_back = orig.Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        center_x = (ul_x + lr_x) / 2.0
        center_y = (ul_y + lr_y) / 2.0
        center_lon, center_lat = transformer_back.transform(center_x, center_y)
        orig.TS_LOCATIONS = [{'latlon': (center_lat, center_lon), 'label': f"Grid Center ({resolved_location})", 'color': 'tab:purple'}]

    valid_locations = []
    print("\n--- Coordinate Mapping ---")
    for loc in orig.TS_LOCATIONS:
        lat, lon = loc['latlon']
        proj_x, proj_y = transformer.transform(lon, lat)
        px, py = inv_affine * (proj_x, proj_y)
        r, c = int(round(py)), int(round(px))
        if 0 <= r < self.height and 0 <= c < self.width:
            loc['yx'] = (r, c)
            valid_locations.append(loc)
            print(f"Mapped [{loc['label']}] Lat/Lon ({lat:.4f}, {lon:.4f}) -> Pixel (y={r}, x={c})")
        else:
            print(f"Warning: [{loc['label']}] out of bounds (y={r}, x={c}). Discarding.")
            
    if not valid_locations:
        print("Warning: No predefined locations fell within image bounds. Falling back to center pixel.")
        r, c = self.height // 2, self.width // 2
        ul_x, ul_y = affine * (c, r)
        transformer_back = orig.Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        center_lon, center_lat = transformer_back.transform(ul_x, ul_y)
        valid_locations = [{'latlon': (center_lat, center_lon), 'label': "Grid Center", 'color': 'tab:purple', 'yx': (r, c)}]
        
    orig.TS_LOCATIONS_ACTIVE = valid_locations
    orig.TS_LOCATIONS = valid_locations

    self.current_idx = 0
    self.save_dir = orig.SAVE_DIR.replace(orig.Location, resolved_location)

    self.ts_start_date = orig.TS_START_DATE
    self.ts_end_date = orig.TS_END_DATE
    self.use_twin_axis = orig.TWIN_Y_AXIS_DEFAULT
    
    self.im_slide = None
    self.cbar_slide = None
    self.fig_scatter = None
    self.im_slide_redundant = None
    self.cbar_slide_redundant = None
    self.ax_ts_twin = None
    self.ax_ts_redundant_twin = None
    self.ax_comp_top_twin = None
    self.ax_comp_bot_twin = None

    self._recompute_time_series()
    self._init_control_ui()
    self._init_combined_ui()
    self._init_comparison_ui()
    if orig.DISPLAY_REDUNDANT_FIGURE:
        self._init_redundant_ui()
        self._init_transect_ui()
    
    self.update_display()

def custom_update_display(self):
    idx = self.current_idx
    
    grid_name = self.prov_grid[idx]
    spacecraft = self.prov_space[idx]
    src_idx = self.prov_idx[idx]
    
    acq_time = self.prov_time[idx]
    curr_dt = orig.datetime.fromtimestamp(acq_time, tz=orig.timezone.utc)
    dt_et = curr_dt.astimezone(orig.ZoneInfo("America/New_York"))
    
    meta_str = (f"TIMELINE:   {idx + 1} / {self.total_frames}\n"
                f"SOURCE GRID: {grid_name} (Idx: {src_idx})\n"
                f"SPACECRAFT: {spacecraft}\n"
                f"ACQUIRED:   {dt_et.strftime('%Y-%m-%d %H:%M:%S ET')}")
    self.ctrl_text.set_text(meta_str)
    
    filter_str = "Masking: Derived from HARMONIZED common_mask"
    self.combined_hud.set_text(meta_str.replace('\n', ' | ') + '\n' + filter_str)

    comp_data = self.base_dset[idx].astype(np.float32)
    mask_data = self.common_mask_dset[idx]
    
    comp_data_for_stats = comp_data.copy()
    if orig.MASKING:
        comp_data_for_stats[mask_data == 1] = np.nan

    self.ax_spatial.clear()
    self.ax_spatial.set_title(f"EM Locations (Consolidated)", color='black')
    self.ax_spatial.axis('off')
        
    self.ax_spectral.clear()
    self.ax_spectral.set_title("Spectral Signatures (Consolidated)")
    self.ax_spectral.set_xlabel("Wavelength (μm)")
    self.ax_spectral.set_ylabel("Reflectance")
    
    self.ax_vol_curve.clear()
    self.ax_vol_curve.set_title("Gram Volume Curve (Consolidated)")
    
    def update_map(ax, data, data_for_stats, mask_arr, im_attr, overlay_attr, cbar_attr, title):
        mh, mw = data.shape
        with np.errstate(all='ignore'):
            if np.all(np.isnan(data_for_stats)):
                v_min, v_max = np.nan, np.nan
            elif orig.DISPLAY_NORMALIZATION:
                v_min, v_max = np.nanpercentile(data_for_stats, (2, 98))
            else:
                v_min, v_max = np.nanmin(data_for_stats), np.nanmax(data_for_stats)
        if np.isnan(v_min) or np.isnan(v_max):
            v_min, v_max = 0, 1
        elif v_min == v_max:
            v_max = v_min + 1e-6
        
        curr_im = getattr(self, im_attr, None)
        curr_overlay = getattr(self, overlay_attr, None)
        curr_cbar = getattr(self, cbar_attr, None)

        overlay_rgba = np.zeros((mh, mw, 4), dtype=np.float32)
        if orig.MASKING and mask_arr is not None:
            overlay_rgba[mask_arr == 1] = [1.0, 0.65, 0.0, 0.5]

        if curr_im is None:
            new_im = ax.imshow(data, cmap='viridis', extent=[0, mw, mh, 0], vmin=v_min, vmax=v_max)
            setattr(self, im_attr, new_im)
            
            new_overlay = ax.imshow(overlay_rgba, extent=[0, mw, mh, 0])
            setattr(self, overlay_attr, new_overlay)
            
            if orig.LOG_SCALE:
                new_cbar = ax.figure.colorbar(new_im, format='%.1e', ax=ax, fraction=0.046, pad=0.04)
            else:
                new_cbar = ax.figure.colorbar(new_im, ax=ax, fraction=0.046, pad=0.04)
            setattr(self, cbar_attr, new_cbar)
            
            for loc in orig.TS_LOCATIONS:
                y, x = loc['yx']
                ax.plot(x + 0.5, y + 0.5, marker='s', markersize=10, markeredgecolor=loc['color'], 
                        markerfacecolor='none', markeredgewidth=1.5, linestyle='None')
            ax.set_title(title)
            ax.axis('off')
        else:
            curr_im.set_data(data)
            curr_im.set_clim(vmin=v_min, vmax=v_max)
            curr_overlay.set_data(overlay_rgba)
            curr_cbar.update_normal(curr_im)

    update_map(self.ax_slide_map, comp_data, comp_data_for_stats, mask_data, 'im_slide', 'overlay_slide', 'cbar_slide', orig.COMPLEXITY_DICT.get(orig.complexity_type, orig.complexity_type))
    
    # Comparison map plotting removed because the appropriate axis is undefined.

    if orig.DISPLAY_REDUNDANT_FIGURE:
        update_map(self.ax_slide_map_redundant, comp_data, comp_data_for_stats, mask_data, 'im_slide_redundant', 'overlay_slide_redundant', 'cbar_slide_redundant', orig.COMPLEXITY_DICT.get(orig.complexity_type, orig.complexity_type))
        if hasattr(self, 'update_transects'):
            self.update_transects(comp_data)

    if hasattr(self, 'fig_combined'):
        self.fig_combined.canvas.draw_idle()
    if orig.DISPLAY_REDUNDANT_FIGURE:
        self.fig_redundant.canvas.draw_idle()

orig.HarmonizedComplexityViewer.__init__ = new_init
orig.HarmonizedComplexityViewer.update_display = custom_update_display

if __name__ == '__main__':
    print("Launching Consolidated Spectral Complexity Viewer (Index Map architecture)...")
    DEFAULT_LOCATION = "Tait"
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="Launch Consolidated Spectral Complexity Viewer.")
    parser.add_argument('--location', type=str, default=None, help="Location name from config.")
    parser.add_argument('--file', type=str, default=None, help="Explicit path to a native or consolidated .h5 file.")
    
    # Parse only known args because sys.argv might contain flags meant for matplotlib/IPython if run interactively
    args, unknown = parser.parse_known_args()
    
    if len(sys.argv) == 1:
        import tkinter as tk
        from tkinter import filedialog
        
        root = tk.Tk()
        root.withdraw()
        
        print("Waiting for file selection...")
        file_path = filedialog.askopenfilename(
            title="Select Spectral Complexity HDF5 File",
            filetypes=[("HDF5 Files", "*.h5"), ("All Files", "*.*")],
            initialdir="C:/satelliteImagery"
        )
        
        if not file_path:
            print("No file selected. Exiting.")
            sys.exit(0)
            
        H5_PATH = file_path
    elif args.file:
        H5_PATH = args.file
    elif args.location:
        LOCATION = args.location
        H5_PATH = f"C:/satelliteImagery/consolidatedConstellation/SC_Constellation_{LOCATION}.h5"
    else:
        LOCATION = DEFAULT_LOCATION
        H5_PATH = f"C:/satelliteImagery/consolidatedConstellation/SC_Constellation_{LOCATION}.h5"
    
    viewer = orig.HarmonizedComplexityViewer(H5_PATH)
    orig.plt.show()
