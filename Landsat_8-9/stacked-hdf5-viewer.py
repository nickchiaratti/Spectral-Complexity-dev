"""
This script provides an interactive viewer for HDF5 datasets in a time series, 
specifically designed for raster images with greater than 3 bands. 
It allows the user to visualize true-color composites, 
calculate and display an estimation of spectral complexity or material abundance 
using the Maximum Distance algorithm, and show difference maps between consecutive volume maps.


The viewer pre-processes all frames of the HDF5 stack into memory. It supports
different processing methods for volume map generation (grid-based tiling,
sliding window tiling, or full-frame processing) and various image
normalization techniques for display.

- Loads 4D HDF5 data (frames, bands, height, width).
- Extracts and displays metadata for each frame (acquisition time, sensor, band names).
- Automatically identifies RGB bands based on wavelength information in the header.
- Pre-calculates and caches:
    - True-color (RGB) images for all frames.
    - Volume maps for all frames using the Maximum Distance algorithm.
    - Difference maps between consecutive volume maps.
- Supports 'grid', 'sliding', or 'full' processing methods for volume estimation.
- Implements 'linear', 'log', or 'percentile' normalization for displayed images.
- Provides 'Next' and 'Prev' buttons for frame navigation.
- Displays three synchronized image panels: True Color (or Panchromatic), Volume Map, and Volume Difference Map.
- **Includes option to display high-resolution Panchromatic band in the left panel if available.**
- Exports processed RGB, Volume, and Difference maps as GeoTIFFs if SAVE_IMAGES is True,
  preserving georeferencing information from the HDF5 attributes.
- Includes error handling for file loading and memory allocation.
- Displays spectral complexity value vs frame index with a pixel selection tool.

Dependencies:
- h5py: For reading HDF5 files.
- numpy: For numerical operations.
- matplotlib: For plotting and interactive visualization.
- rasterio: For reading/writing georeferenced raster data.
- MaxD_Gram.py: Custom module for Maximum Distance algorithm (assumed to be in the same directory).
- NSC_toolbox.py: Custom module for image normalization and other utilities (assumed to be in the same directory).

Configuration:
- TILE_SIZE: Defines the width/height of tiles for volume map processing.
- SAVE_IMAGES: Boolean to enable/disable GeoTIFF export.
- PROCESSING_METHOD: 'grid', 'sliding', or 'full' for volume map calculation.
- IMAGE_NORMALIZATION_METHOD: 'linear', 'log', or 'percentile' for image display.
- MAX_DIST_P1, MAX_DIST_P2, MAX_DIST_P3: Parameters for the Maximum Distance algorithm.


TODO:
- save geotiffs prior to normalization 
- parallelize tile processing
- add option to choose volume[3] or sum(volume[3:]) to account for all endmembers
- 

"""
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.widgets import Slider, Button
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import json
import os
import textwrap
import MaxD_Gram as maxD
import NSC_toolbox as nsc
from rasterio.transform import Affine
import rasterio
from datetime import datetime
from dateutil import parser
from skimage import exposure
from pyproj import Transformer, CRS
from matplotlib.ticker import MaxNLocator
import re

# --- Configuration ---
SAVE_IMAGES = True
# --- Processing Options ---
VOLUME_METHOD = 'peak' #'sum', 'peak', or 'third'
PROCESSING_METHOD = 'sliding'  # 'grid', 'sliding', or 'full'
TILE_SIZE = 3
SLIDING_STRIDE = 1
# --- Image Normalization Options ---
IMAGE_NORMALIZATION_METHOD = 'percentile'
PERCENTILE_LOW = 1
PERCENTILE_HIGH = 99
# --- Display Options ---
DISPLAY_LEFT_PANEL = 'rgb' 

# --- Parameters for Maximum-Distance ---
MAX_DIST_P1 = min(TILE_SIZE**2-1, 10)
MAX_DIST_P2 = 0
MAX_DIST_P3 = 'local'

# --- Pixel Selection Colors ---
PIXEL_COLORS = ['red', 'blue', 'orange', 'cyan']

class HDF5Viewer:
    """
    An interactive matplotlib viewer for HDF-EOS5 Landsat Stacks.
    """
    def __init__(self, h5_path):
        print(f"Loading HDF5 file: {h5_path}")
        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"The file {h5_path} was not found.")
            
        self.h5_file = h5py.File(h5_path, 'r')
        self.infolder = os.path.dirname(h5_path)
        self.filename = os.path.basename(h5_path)
        if SAVE_IMAGES:
            if PROCESSING_METHOD == 'full':
                outfile = self.infolder+'/'+self.filename.replace('.h5', f'-{PROCESSING_METHOD}-{MAX_DIST_P3}-{IMAGE_NORMALIZATION_METHOD}')+'/'+self.filename.replace('.h5','')
            elif PROCESSING_METHOD == 'sliding':
                outfile = self.infolder+'/'+self.filename.replace('.h5', f'-{TILE_SIZE}X{TILE_SIZE}-slide{SLIDING_STRIDE}-{PROCESSING_METHOD}-{MAX_DIST_P3}-{IMAGE_NORMALIZATION_METHOD}')+'/'+self.filename.replace('.h5','')
            else:
                outfile = self.infolder+'/'+self.filename.replace('.h5', f'-{TILE_SIZE}X{TILE_SIZE}-{PROCESSING_METHOD}-{MAX_DIST_P3}-{IMAGE_NORMALIZATION_METHOD}')+'/'+self.filename.replace('.h5','')
            self.output_dir = os.path.dirname(outfile)
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
        
        # --- HDF-EOS5 Path ---
        # Data is now located deep in the hierarchy
        try:
            grid_group = self.h5_file['HDFEOS/GRIDS/Landsat_Grid/Data Fields']
            if 'surface_reflectance' in grid_group:
                self.dset = grid_group['surface_reflectance']
            else:
                self.dset = grid_group['image_stack'] # Fallback for older files
                
            self.is_hdfeos = True
            print("Detected HDF-EOS5 format.")
        except KeyError:
            # Fallback for legacy format
            self.dset = self.h5_file['image_stack']
            self.is_hdfeos = False
            print("Detected Legacy format.")

        # --- Multispectral Geotransform & Projection Setup ---
        self.affine_transform = None
        self.transformer = None
        self.crs_wkt = None
        self.geo_transform = None
        
        try:
            if self.is_hdfeos:
                # Parse ODL Metadata for Georeferencing
                struct_meta = self.h5_file['HDFEOS INFORMATION/StructMetadata.0'][0].decode('ascii')
                self.geo_transform, self.crs_wkt = self.parse_hdfeos_metadata(struct_meta)
            else:
                # Legacy Attribute Loading
                self.crs_wkt = self.dset.attrs.get('crs_wkt')
                if isinstance(self.crs_wkt, bytes):
                    self.crs_wkt = self.crs_wkt.decode('utf-8')
                self.geo_transform = self.dset.attrs.get('transform')
            
            # Initialize Projection
            if self.geo_transform is not None and self.crs_wkt:
                self.affine_transform = Affine.from_gdal(*self.geo_transform)
                src_crs = CRS.from_wkt(self.crs_wkt)
                dst_crs = CRS.from_epsg(4326)
                self.transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
                print("Georeferencing initialized successfully.")
            else:
                print("Warning: Missing geotransform or CRS WKT.")

        except Exception as e:
            print(f"Warning: Could not load georeferencing info: {e}")
        
        self.num_frames, self.num_bands, self.height, self.width = self.dset.shape
        print(f"Data shape: (Frames: {self.num_frames}, Bands: {self.num_bands}, Height: {self.height}, Width: {self.width})")
        
        # --- Check for Panchromatic Stack ---
        self.dset_pan = None
        self.pan_geo_transform = None
        self.has_pan = False
        
        pan_path = 'HDFEOS/GRIDS/Landsat_Grid/Data Fields/panchromatic_stack' if self.is_hdfeos else 'panchromatic_stack'
        if pan_path in self.h5_file:
            self.dset_pan = self.h5_file[pan_path]
            _, _, self.pan_height, self.pan_width = self.dset_pan.shape
            self.has_pan = True
            print(f"Found Panchromatic Stack: {self.dset_pan.shape}")
            
            # Pan geotransform is derived from main grid (2x resolution) if HDF-EOS
            if self.is_hdfeos and self.geo_transform:
                 # Standard Grid Logic: 2x dims, same origin, half scale
                 gt = list(self.geo_transform)
                 gt[1] /= 2.0 # Pixel width
                 gt[5] /= 2.0 # Pixel height
                 self.pan_geo_transform = tuple(gt)
            else:
                 try:
                    self.pan_geo_transform = self.dset_pan.attrs.get('transform')
                 except:
                    self.pan_geo_transform = None

        self.display_left = DISPLAY_LEFT_PANEL
        if self.display_left == 'pan' and not self.has_pan:
            print("Warning: Panchromatic requested for left panel but not found. Reverting to RGB.")
            self.display_left = 'rgb'

        self.current_frame = 0
        self.cache_allocated = False
        
        # --- Pixel Selection State ---
        # Initialize with center pixel to avoid out-of-bounds errors on small ROIs
        center_x = self.width // 2
        center_y = self.height // 2
        self.selected_pixels = [(center_x, center_y)] 
        
        self.selection_rects = []
        self.selection_index = 1 # Next click will replace index 1 (since 0 is pre-filled)
        
        # JSON Sidecar loading (HDF-EOS style) or internal attributes (Legacy)
        json_path = h5_path.replace('.h5', '_metadata.json')
        self.headers_json = {}
        if os.path.exists(json_path):
             with open(json_path, 'r') as jf:
                 self.headers_json = json.load(jf)
             self.frame_metadata_cache = []
             for i in range(self.num_frames):
                 key = f"frame_{i:04d}"
                 if key in self.headers_json:
                     header = self.headers_json[key]['header']
                     self.frame_metadata_cache.append(header)
                 else:
                     self.frame_metadata_cache.append({})
        else:
             self.header_group = self.h5_file.get('source_headers')
             if self.header_group:
                 print("Caching all frame metadata from internal attributes...")
                 self.frame_metadata_cache = []
                 for i in range(self.num_frames):
                    frame_key = f"frame_{i:04d}"
                    header_json = self.header_group.attrs.get(frame_key)
                    if header_json:
                        self.frame_metadata_cache.append(json.loads(header_json))
                    else:
                        self.frame_metadata_cache.append({})
             else:
                 print("Warning: No metadata found.")
                 self.frame_metadata_cache = [{}] * self.num_frames

        # Parse dates
        self.frame_dates = []
        self.frame_dates_str = []
        
        for i, header in enumerate(self.frame_metadata_cache):
            acq_time = header.get('acquisition time', 'N/A')
            try:
                dt_obj = parser.parse(acq_time)
                self.frame_dates.append(dt_obj)
                self.frame_dates_str.append(dt_obj.strftime('%A, %Y-%m-%d'))
            except:
                self.frame_dates.append(i)
                self.frame_dates_str.append(str(i))
        
        self.first_header = self.frame_metadata_cache[0] if self.frame_metadata_cache else {}
        self.ignore_value = self.first_header.get('data ignore value', 0)
        self.rgb_indices = self.get_rgb_indices(self.first_header)

        try:
            self.rgb_frames_cache = np.zeros((self.num_frames, self.height, self.width, 3), dtype=np.float32)
            self.volume_map_cache = np.zeros((self.num_frames, self.height, self.width),dtype=np.float32)     
            self.raw_volume_map_cache = np.zeros((self.num_frames, self.height, self.width),dtype=np.float32)
            self.diff_map_cache = np.zeros((self.num_frames, self.height, self.width),dtype=np.float32)
            self.raw_diff_map_cache = np.zeros((self.num_frames, self.height, self.width),dtype=np.float32)
            self.endmembers = np.zeros((self.num_frames, self.num_bands, MAX_DIST_P1),dtype=np.float32)
            self.endmember_indices = np.zeros((self.num_frames, MAX_DIST_P1),dtype=np.int32)
            self.masked_rgb_cache = np.zeros((self.num_frames, self.height, self.width, 3), dtype=np.float32)
            
            if self.has_pan:
                self.pan_frames_cache = np.zeros((self.num_frames, self.pan_height, self.pan_width), dtype=np.float32)

            self.cache_allocated = True
        except MemoryError:
            print("\n" + "="*50)
            print("FATAL ERROR: MemoryError")
            print("Not enough RAM to pre-load all frames into the cache.")
            print("The data stack is too large. Exiting.")
            print("="*50)
            self.h5_file.close()
            return

        for i in range(self.num_frames):
            print(f"\n  - Processing frame {i+1}/{self.num_frames}")
            self.rgb_frames_cache[i] = self.get_frame_rgb(i, IMAGE_NORMALIZATION_METHOD)
            
            if self.has_pan:
                print(f"      - Caching Panchromatic image ({IMAGE_NORMALIZATION_METHOD} scale)...")
                self.pan_frames_cache[i] = self.get_frame_pan(i, IMAGE_NORMALIZATION_METHOD)

            if PROCESSING_METHOD == 'sliding':
                print(f"Processing volume map using sliding window with stride {SLIDING_STRIDE} and tile size {TILE_SIZE}...")
                volume_map = self._process_volume_sliding_tile(i)
            elif PROCESSING_METHOD == 'grid':
                print(f"Processing volume map using grid with tile size {TILE_SIZE}...")
                volume_map = self._process_volume_tiles(i)
            else:
                print(f"Processing volume estimation for the entire frame.")
                volume_map = self._process_volume_map(i)    
            
            self.raw_volume_map_cache[i] = volume_map
            
            if IMAGE_NORMALIZATION_METHOD == 'log':
                print("      - Stretching Volume Map (Logarithmic scale)...")
                self.volume_map_cache[i] = nsc.log_normalize_array(volume_map)
            elif IMAGE_NORMALIZATION_METHOD == 'linear':
                print("      - Stretching Volume Map (Linear scale)...")
                self.volume_map_cache[i] = nsc.linear_normalize_array(volume_map)
            elif IMAGE_NORMALIZATION_METHOD == 'percentile':
                print("      - Stretching Volume Map (Percentile scale)...")
                self.volume_map_cache[i] = nsc.percentile_normalize_array(volume_map,1,99)

            brightness_mask = 0.25 + (0.75 * self.volume_map_cache[i])
            mask_3ch = np.stack([brightness_mask]*3, axis=-1)
            self.masked_rgb_cache[i] = np.clip(self.rgb_frames_cache[i] * mask_3ch, 0, 1)

        print("\n  - Caching Volume Difference Maps...")
        for i in range(self.num_frames):
            current_map = self.raw_volume_map_cache[i]
            if i == 0:
                prev_map = self.raw_volume_map_cache[-1] # Frame 0 - Frame (last)
            else:
                prev_map = self.raw_volume_map_cache[i-1] # Frame i - Frame (i-1)
            diff_map = current_map - prev_map
            self.raw_diff_map_cache[i] = diff_map
            if IMAGE_NORMALIZATION_METHOD == 'log':
                self.diff_map_cache[i] = nsc.log_normalize_array(diff_map)
            elif IMAGE_NORMALIZATION_METHOD == 'linear':
                self.diff_map_cache[i] = nsc.linear_normalize_array(diff_map)
            elif IMAGE_NORMALIZATION_METHOD == 'percentile':
                self.diff_map_cache[i] = nsc.percentile_normalize_array(diff_map,1,99)
        
        print("\nCaching complete.")
        
        if not PROCESSING_METHOD == 'full':
            if SAVE_IMAGES:
                print(f"Saving processed images to: {self.output_dir}")
                for i in range(self.num_frames):
                    # --- Save Composite PNG of the Viewer Layout ---
                    self._save_composite_png(i, os.path.join(self.output_dir, f"frame_{i:04d}_composite.png"))
                print("PNG export complete.")
        
        self.fig = plt.figure(figsize=(18, 8))
        self.fig.canvas.manager.set_window_title(f"Frame{self.current_frame}-{TILE_SIZE}x{TILE_SIZE}-{PROCESSING_METHOD}")
        
        ax_text = plt.axes([0.1, 0.82, 0.8, 0.15]) 
        ax_img_rgb = plt.axes([0.05, 0.2, 0.28, 0.6]) 
        ax_img_volume = plt.axes([0.36, 0.2, 0.28, 0.6])
        ax_img_diff = plt.axes([0.67, 0.2, 0.28, 0.6])
        ax_prev = plt.axes([0.35, 0.025, 0.1, 0.04])
        ax_next = plt.axes([0.55, 0.025, 0.1, 0.04])
        # Removed slider axis
        
        self.ax_img_rgb = ax_img_rgb
        self.ax_img_volume = ax_img_volume
        self.ax_img_diff = ax_img_diff
        self.ax_text = ax_text
        self.ax_text.axis('off')
        
        initial_metadata = self.frame_metadata_cache[0]
        initial_text = self.format_metadata_text(initial_metadata)
        self.info_text = self.ax_text.text(0.0, 1.0, initial_text, va='top', ha='left', fontsize=9, wrap=True)

        # --- Left Image Panel (RGB) ---
        left_frame = self.rgb_frames_cache[self.current_frame]
        self.im_left = self.ax_img_rgb.imshow(left_frame)
        self.ax_img_rgb.set_title(f"True Color\nFrame {self.current_frame} / {self.num_frames - 1}")
        self.ax_img_rgb.axis('off')
        
        volume_frame = self.volume_map_cache[self.current_frame]
        self.im_volume = self.ax_img_volume.imshow(volume_frame, cmap='viridis', vmin=0, vmax=1)
        self.ax_img_volume.set_title(f"Spectral Complexity Map")
        self.ax_img_volume.axis('off')
        
        self.cbar_vol = self.fig.colorbar(self.im_volume, ax=self.ax_img_volume, fraction=0.046, pad=0.04)
        self.cbar_vol.set_label('Normalized Volume', rotation=270, labelpad=20)
        self.cbar_vol.set_ticks([0, 0.5, 1])
        # Initial tick labels
        raw_vol = self.raw_volume_map_cache[0]
        min_v = np.min(raw_vol)
        max_v = np.max(raw_vol)
        mid_v = (min_v + max_v) / 2
        self.cbar_vol.set_ticklabels([f'{min_v:.2E}', f'{mid_v:.2E}', f'{max_v:.2E}'])

        diff_frame = self.diff_map_cache[self.current_frame]
        self.im_diff = self.ax_img_diff.imshow(diff_frame, cmap='coolwarm', vmin=0, vmax=1)
        prev_idx_str = str(self.num_frames - 1)
        self.ax_img_diff.set_title(f"Spectral Complexity Difference\n(Frame {self.current_frame} - Frame {prev_idx_str})")
        self.ax_img_diff.axis('off')
        
        self.cbar_diff = self.fig.colorbar(self.im_diff, ax=self.ax_img_diff, fraction=0.046, pad=0.04)
        self.cbar_diff.set_label('Normalized Difference', rotation=270, labelpad=20)
        self.cbar_diff.set_ticks([0, 0.5, 1])
        # Initial tick labels
        raw_diff = self.raw_diff_map_cache[0]
        v_abs_max = np.percentile(np.abs(raw_diff), 99)
        self.cbar_diff.set_ticklabels([f'-{v_abs_max:.2E}', '0.00', f'+{v_abs_max:.2E}'])
        
        self.prev_button = Button(ax_prev, '< Prev')
        self.next_button = Button(ax_next, 'Next >')
        
        self.prev_button.on_clicked(self.on_prev)
        self.next_button.on_clicked(self.on_next)
        
        # self.slider = Slider(ax_slider, 'Frame', 0, self.num_frames - 1, valinit=0, valstep=1)
        # self.slider.on_changed(self.on_slider_update)
        
        self.fig_2 = plt.figure(figsize=(8, 8))
        self.fig_2.canvas.manager.set_window_title("Pixel Analysis")
        self.ax_masked = self.fig_2.add_subplot(1, 1, 1)
        self.im_masked = self.ax_masked.imshow(self.masked_rgb_cache[self.current_frame])
        self.ax_masked.set_title("Spectral Complexity-Masked True Color (Click to Select)")
        self.ax_masked.axis('off')
        
        self.selection_rects = []
        for i, color in enumerate(PIXEL_COLORS):
            rect = patches.Rectangle((0, 0), 1, 1, linewidth=6, edgecolor=color, facecolor='none', visible=False)
            if i < len(self.selected_pixels):
                px, py = self.selected_pixels[i]
                rect_size = 1
                rect.set_xy((px - rect_size/2, py - rect_size/2))
                rect.set_width(rect_size)
                rect.set_height(rect_size)
                rect.set_visible(True)
            self.ax_masked.add_patch(rect)
            self.selection_rects.append(rect)
        self.fig_2.canvas.mpl_connect('button_press_event', self.on_click)

        # --- Updated Figure 3 for Publication Quality with Broken Axis ---
        self.fig_3 = plt.figure(figsize=(7, 3.5)) 
        self.fig_3.canvas.manager.set_window_title("Time Series")

        # --- REVERT: Single subplot, no broken axis ---
        self.ax_ts = self.fig_3.add_subplot(1, 1, 1)
        self.ax_ts.set_title("Spectral Complexity Time Series", fontsize=12)
        self.ax_ts.set_xlabel("Acquisition Date", fontsize=10)
        self.ax_ts.set_ylabel("Raw Volume", fontsize=10)
        
        # Apply standard styling
        self.ax_ts.tick_params(axis='both', which='major', labelsize=8)
        self.ax_ts.grid(True, linestyle='--', alpha=0.6) 
        self.ax_ts.spines['right'].set_visible(False)
        self.ax_ts.spines['top'].set_visible(False)
        
        if self.selected_pixels:
            self.plot_pixel_timeseries(0, 0)

    def parse_hdfeos_metadata(self, odl_string):
        """
        Parses ODL StructMetadata string to extract UpperLeft, LowerRight coordinates and UTM zone.
        Returns (transform_tuple, crs_wkt)
        """
        try:
            # Simple regex parsing for the specific keys we need
            ul_match = re.search(r'UpperLeftPointMtrs=\((-?[\d\.]+),(-?[\d\.]+)\)', odl_string)
            lr_match = re.search(r'LowerRightMtrs=\((-?[\d\.]+),(-?[\d\.]+)\)', odl_string)
            dims_match = re.search(r'XDim=(\d+)\s+YDim=(\d+)', odl_string)
            zone_match = re.search(r'ZoneCode=(\d+)', odl_string)

            if ul_match and lr_match and dims_match:
                ul_x, ul_y = float(ul_match.group(1)), float(ul_match.group(2))
                lr_x, lr_y = float(lr_match.group(1)), float(lr_match.group(2))
                x_dim, y_dim = int(dims_match.group(1)), int(dims_match.group(2))
                
                # Calculate pixel size (resolution)
                pixel_width = (lr_x - ul_x) / x_dim
                pixel_height = (lr_y - ul_y) / y_dim # Should be negative usually
                
                # Construct GDAL transform tuple: (c, a, b, f, d, e) -> (ul_x, pix_w, 0, ul_y, 0, pix_h)
                # Note: ODL Y coords often need checking. Usually UL Y is top, LR Y is bottom.
                
                # Assume standard north-up image:
                transform = (ul_x, pixel_width, 0.0, ul_y, 0.0, pixel_height)
                
                # Construct CRS
                if zone_match:
                    zone = int(zone_match.group(1))
                    # Assuming WGS84 (SphereCode 12) and Northern Hemisphere for simplicity or extra parsing
                    # A robust parser would check Projection param. For Landsat US, usually North.
                    crs = CRS.from_dict({'proj': 'utm', 'zone': zone, 'datum': 'WGS84'})
                    return transform, crs.to_wkt()
                
                return transform, None
        except Exception as e:
            print(f"Error parsing HDF-EOS metadata: {e}")
        return None, None
        
    def on_click(self, event):
        if event.inaxes == self.ax_masked:
            x = int(round(event.xdata))
            y = int(round(event.ydata))
            
            if 0 <= x < self.width and 0 <= y < self.height:
                print(f"Clicked on pixel ({x}, {y})")
                
                if len(self.selected_pixels) < 4:
                    self.selected_pixels.append((x, y))
                else:
                    self.selected_pixels[self.selection_index] = (x, y)
                
                current_rect = self.selection_rects[self.selection_index]
                rect_size = 1
                rect_x = x - (rect_size / 2)
                rect_y = y - (rect_size / 2)
                
                current_rect.set_xy((rect_x, rect_y))
                current_rect.set_width(rect_size)
                current_rect.set_height(rect_size)
                current_rect.set_visible(True)
                
                self.selection_index = (self.selection_index + 1) % 4
                self.plot_pixel_timeseries(x, y)
                self.fig_2.canvas.draw_idle()

    def plot_pixel_timeseries(self, x, y):
        # volumes = self.raw_volume_map_cache[:, y, x]
        
        if all(isinstance(d, datetime) for d in self.frame_dates):
            x_values = range(self.num_frames)
            is_dates = True
        else:
            x_values = range(self.num_frames)
            is_dates = False

        # --- REVERT: Use single axis clearing ---
        self.ax_ts.clear()
        
        # Plot on SINGLE subplot
        for i, (px, py) in enumerate(self.selected_pixels):
            p_volumes = self.raw_volume_map_cache[:, py, px]
            color = PIXEL_COLORS[i]
            label_text = f"Pixel {i+1} ({px}, {py})"
            self.ax_ts.plot(x_values, p_volumes, marker='o', linestyle='-', markersize=4, color=color, label=label_text, linewidth=1.5)

        # Restore Title and Limits
        self.ax_ts.set_title(f"Spectral Complexity Time Series", fontsize=12)
        
        # Reset labels (cleared by clear())
        self.ax_ts.set_xlabel("Acquisition Date", fontsize=10)
        self.ax_ts.set_ylabel("Raw Volume", fontsize=10)

        # Legend 
        self.ax_ts.legend(fontsize=8, frameon=False, loc='upper right')
        
        # X-Axis Formatting 
        if is_dates:
            self.ax_ts.set_xticks(x_values)
            if len(self.frame_dates_str) == self.num_frames:
                self.ax_ts.set_xticklabels(self.frame_dates_str, rotation=45, ha='right', fontsize=8)
        else:
            self.ax_ts.set_xlabel("Frame Index", fontsize=10)
            from matplotlib.ticker import MaxNLocator
            self.ax_ts.xaxis.set_major_locator(MaxNLocator(integer=True))
            
        # Grid and Spines restoration
        self.ax_ts.grid(True, linestyle='--', alpha=0.6)
        self.ax_ts.spines['right'].set_visible(False)
        self.ax_ts.spines['top'].set_visible(False)
             
        self.fig_3.tight_layout() 
        self.fig_3.canvas.draw_idle()

    def get_rgb_indices(self, header):
        if 'wavelength' not in header or not header['wavelength']:
            return (0, 1, 2)
        wavelengths = np.array(header['wavelength'], dtype=float)
        units = header.get('wavelength units', 'Micrometers').lower()
        targets_um = {'red': 0.65, 'green': 0.56, 'blue': 0.48}
        if 'nano' in units:
            targets_wl = {k: v * 1000 for k, v in targets_um.items()}
        else:
            targets_wl = targets_um
        r_idx = np.argmin(np.abs(wavelengths - targets_wl['red']))
        g_idx = np.argmin(np.abs(wavelengths - targets_wl['green']))
        b_idx = np.argmin(np.abs(wavelengths - targets_wl['blue']))
        return (r_idx, g_idx, b_idx)
        
    def format_metadata_text(self, metadata):
        t = metadata.get('acquisition time', 'N/A')
        s = metadata.get('sensor type', 'N/A')
        bc = metadata.get('bands', 'N/A')
        bn_str = metadata.get('band names', 'N/A')
        
        try:
            dt_obj = parser.parse(t) 
            day_of_week = dt_obj.strftime('%A')
            formatted_time = dt_obj.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        except (ValueError, ImportError):
            try:
                clean_t = t.split('.')[0].replace('Z', '') 
                dt_obj = datetime.strptime(clean_t, '%Y-%m-%dT%H:%M:%S')
                day_of_week = dt_obj.strftime('%A')
                formatted_time = dt_obj.strftime('%Y-%m-%d %H:%M:%S.000')
            except ValueError:
                day_of_week = "Unknown Day"
                formatted_time = t
        
        # --- FIX: Ensure bn_str is a string before passing to textwrap ---
        if isinstance(bn_str, list):
            bn_str = ", ".join(bn_str)
        
        bn_wrapped_list = textwrap.wrap(str(bn_str), width=90)
        bn_wrapped = "\n    ".join(bn_wrapped_list)
        return f"Acquisition Time: {formatted_time}\nDay of Week: {day_of_week}\nSensor Type: {s}\nBands: {bc}\nBand Names:\n    {bn_wrapped}"

    def get_frame_rgb(self, frame_idx, method='percentile'):
        if any(idx >= self.num_bands for idx in self.rgb_indices):
            return np.zeros((self.height, self.width, 3), dtype=np.float32)

        if method == 'log':
            print("      - Caching True Color (RGB) image (Logarithmic scale)...")
            r = nsc.log_normalize_array(self.dset[frame_idx, self.rgb_indices[0], :, :])
            r_ignore_mask = (self.dset[frame_idx, self.rgb_indices[0], :, :] == self.ignore_value)
            r[r_ignore_mask] = 0
            g = nsc.log_normalize_array(self.dset[frame_idx, self.rgb_indices[1], :, :])
            g_ignore_mask = (self.dset[frame_idx, self.rgb_indices[1], :, :] == self.ignore_value)
            g[g_ignore_mask] = 0
            b = nsc.log_normalize_array(self.dset[frame_idx, self.rgb_indices[2], :, :],1,99)
            b_ignore_mask = (self.dset[frame_idx, self.rgb_indices[2], :, :] == self.ignore_value)
            b[b_ignore_mask] = 0
        elif method == 'linear':
            print("      - Caching True Color (RGB) image (Linear scale)...")
            r = nsc.linear_normalize_array(self.dset[frame_idx, self.rgb_indices[0], :, :])
            r_ignore_mask = (self.dset[frame_idx, self.rgb_indices[0], :, :] == self.ignore_value)
            r[r_ignore_mask] = 0
            g = nsc.linear_normalize_array(self.dset[frame_idx, self.rgb_indices[1], :, :])
            g_ignore_mask = (self.dset[frame_idx, self.rgb_indices[1], :, :] == self.ignore_value)
            g[g_ignore_mask] = 0
            b = nsc.linear_normalize_array(self.dset[frame_idx, self.rgb_indices[2], :, :],1,99)
            b_ignore_mask = (self.dset[frame_idx, self.rgb_indices[2], :, :] == self.ignore_value)
            b[b_ignore_mask] = 0
        elif method == 'percentile':
            print("      - Caching True Color (RGB) image (Percentile scale)...")
            r = nsc.percentile_normalize_array(self.dset[frame_idx, self.rgb_indices[0], :, :],PERCENTILE_LOW,PERCENTILE_HIGH)
            r_ignore_mask = (self.dset[frame_idx, self.rgb_indices[0], :, :] == self.ignore_value)
            r[r_ignore_mask] = 0
            g = nsc.percentile_normalize_array(self.dset[frame_idx, self.rgb_indices[1], :, :],PERCENTILE_LOW,PERCENTILE_HIGH)
            g_ignore_mask = (self.dset[frame_idx, self.rgb_indices[1], :, :] == self.ignore_value)
            g[g_ignore_mask] = 0
            b = nsc.percentile_normalize_array(self.dset[frame_idx, self.rgb_indices[2], :, :],PERCENTILE_LOW,PERCENTILE_HIGH)
            b_ignore_mask = (self.dset[frame_idx, self.rgb_indices[2], :, :] == self.ignore_value)
            b[b_ignore_mask] = 0

        rgb = np.stack([r, g, b], axis=-1).astype(np.float32)
        return rgb

    def get_frame_pan(self, frame_idx, method='percentile'):
        pan_data = self.dset_pan[frame_idx, 0, :, :]
        if method == 'log':
            pan_norm = nsc.log_normalize_array(pan_data)
        elif method == 'linear':
            pan_norm = nsc.linear_normalize_array(pan_data)
        else: 
            pan_norm = nsc.percentile_normalize_array(pan_data, PERCENTILE_LOW, PERCENTILE_HIGH)
        if np.isnan(pan_data).any():
            pan_norm[np.isnan(pan_data)] = 0
        return pan_norm

    def _process_volume_map(self, frame_idx):
        frame_data_M = np.transpose(self.dset[frame_idx, ...], (1, 2, 0))
        volume_map = np.zeros((self.height, self.width), dtype=np.float32)
        image2D = np.reshape(frame_data_M, (self.height * self.width, self.num_bands))
        self.endmembers[frame_idx], self.endmember_indices[frame_idx], volume = nsc.maximumDistance(image2D, MAX_DIST_P1, MAX_DIST_P2, MAX_DIST_P3)
        
        if SAVE_IMAGES:
            plt.figure(figsize=(12,9))
            plt.plot(range(MAX_DIST_P1), volume[0:], marker='o', linestyle='-')
            plt.xlabel('Number of endmembers')
            plt.ylabel('Estimated volume')
            plt.title('Grammian Volume Function')
            plt.savefig(os.path.join(self.output_dir, f"frame_{frame_idx:04d}_volumes.png"),dpi = 1200,bbox_inches='tight')
            plt.close()
            plt.figure(figsize=(12,9))
            nsc.plot_spectral_profiles(self.endmembers[frame_idx],self.num_bands)
            plt.savefig(os.path.join(self.output_dir, f"frame_{frame_idx:04d}_spectra.png"),dpi = 1200,bbox_inches='tight')
            plt.close()
            rows, cols, bands = self.rgb_frames_cache[frame_idx].shape
            plt.imshow(self.rgb_frames_cache[frame_idx])
            for i, idx in enumerate(self.endmember_indices[frame_idx,0:4]):
                if np.any(self.endmembers[frame_idx,:,i] != 0):
                    row = idx // cols
                    col = idx % cols
                    plt.plot(col, row, 'r+', markersize=15, markeredgewidth=2,  label=f'V[{i}]' if i < 2 else None) 
                    plt.annotate(f'V[{i}]', (col, row),  textcoords="offset points", xytext=(0, -15),  ha='center', fontsize=12, color='r', fontweight='bold')
            plt.title('Spatial Locations of Found Endmembers')
            plt.xlabel('Pixel Column')
            plt.ylabel('Pixel Row')
            plt.savefig(os.path.join(self.output_dir, f"frame_{frame_idx:04d}_inSceneEndmembers.png"),dpi = 1200,bbox_inches='tight')
            plt.close()

        if VOLUME_METHOD == 'sum':
            volume_map = np.sum(volume[3:])
        elif VOLUME_METHOD == 'peak':
            volume_map = np.max(volume[3:])
        elif VOLUME_METHOD == 'third':
            volume_map = volume[3]
        return volume_map

    def _process_volume_tiles(self, frame_idx):
        frame_data_M = np.transpose(self.dset[frame_idx, ...], (1, 2, 0))
        volume_map = np.zeros((self.height, self.width), dtype=np.float32)
        for y_start in range(0, self.height, TILE_SIZE):
            for x_start in range(0, self.width, TILE_SIZE):
                y_end = min(y_start + TILE_SIZE, self.height)
                x_end = min(x_start + TILE_SIZE, self.width)
                if (y_end - y_start != TILE_SIZE) or (x_end - x_start != TILE_SIZE):
                    continue
                tile_cube = frame_data_M[y_start:y_end, x_start:x_end, :]
                _, _, volume = nsc.maximumDistance(tile_cube, MAX_DIST_P1, MAX_DIST_P2, MAX_DIST_P3)
                if VOLUME_METHOD == 'sum':
                    volume_map[y_start:y_end, x_start:x_end] = np.sum(volume[3:])
                elif VOLUME_METHOD == 'peak':
                    volume_map[y_start:y_end, x_start:x_end] = np.max(volume[3:])
                elif VOLUME_METHOD == 'third':
                    volume_map[y_start:y_end, x_start:x_end] = volume[3]
        return volume_map

    def _process_volume_sliding_tile(self, frame_idx):
        frame_data_M = np.transpose(self.dset[frame_idx, ...], (1, 2, 0))
        sum_map = np.zeros((self.height, self.width), dtype=np.float32)
        count_map = np.zeros((self.height, self.width), dtype=np.float32)
        for y_start in range(0, self.height - TILE_SIZE + 1, SLIDING_STRIDE):
            for x_start in range(0, self.width - TILE_SIZE + 1, SLIDING_STRIDE):
                y_end = y_start + TILE_SIZE
                x_end = x_start + TILE_SIZE
                tile_cube = frame_data_M[y_start:y_end, x_start:x_end, :]
                _, _, volume = nsc.maximumDistance(tile_cube, MAX_DIST_P1, MAX_DIST_P2, MAX_DIST_P3)
                if VOLUME_METHOD == 'sum':
                    sum_map[y_start:y_end, x_start:x_end] += np.sum(volume[3:])
                elif VOLUME_METHOD == 'peak':
                    sum_map[y_start:y_end, x_start:x_end] += np.max(volume[3:])
                elif VOLUME_METHOD == 'third':
                    sum_map[y_start:y_end, x_start:x_end] += volume[3]
                count_map[y_start:y_end, x_start:x_end] += 1.0
        count_map[count_map == 0] = 1.0
        return sum_map / count_map

    def _save_as_geotiff(self, path, data, bands, transform=None):
        target_transform_tuple = transform if transform is not None else self.geo_transform
        if target_transform_tuple is not None:
            final_transform = Affine.from_gdal(*target_transform_tuple)
        else:
            final_transform = None
        if bands == 1:
            height, width = data.shape
            write_data = data[np.newaxis, :, :]
        else:
            height, width, _ = data.shape
            write_data = np.transpose(data, (2, 0, 1))
        profile = {
            'driver': 'GTiff', 'height': height, 'width': width, 'count': bands,
            'dtype': rasterio.float32, 'crs': self.crs_wkt, 'transform': final_transform, 'compress': 'lzw'
        }
        with rasterio.open(path, 'w', **profile) as dst:
            dst.write(write_data.astype(rasterio.float32))

    def _save_composite_png(self, frame_idx, filename):
        fig_temp = plt.figure(figsize=(18, 8))
        ax_img_rgb = plt.axes([0.05, 0.2, 0.28, 0.6]) 
        ax_img_volume = plt.axes([0.36, 0.2, 0.28, 0.6])
        ax_img_diff = plt.axes([0.67, 0.2, 0.28, 0.6])
        if self.display_left == 'pan':
            left_frame = self.pan_frames_cache[frame_idx]
            im_left = ax_img_rgb.imshow(left_frame, cmap='gray')
            ax_img_rgb.set_title(f"Panchromatic")
        else:
            left_frame = self.rgb_frames_cache[frame_idx]
            im_left = ax_img_rgb.imshow(left_frame)
            ax_img_rgb.set_title(f"True Color")
        ax_img_rgb.axis('off')
        volume_frame = self.volume_map_cache[frame_idx]
        im_vol = ax_img_volume.imshow(volume_frame, cmap='viridis', vmin=0, vmax=1)
        ax_img_volume.set_title(f"Spectral Complexity Map")
        ax_img_volume.axis('off')
        cbar_vol = fig_temp.colorbar(im_vol, ax=ax_img_volume, fraction=0.046, pad=0.04)
        cbar_vol.set_label('Normalized Volume', rotation=270, labelpad=20)
        cbar_vol.set_ticks([0, 0.5, 1])
        raw_vol = self.raw_volume_map_cache[frame_idx]
        min_v = np.min(raw_vol)
        max_v = np.max(raw_vol)
        mid_v = (min_v + max_v) / 2
        cbar_vol.set_ticklabels([f'{min_v:.2E}', f'{mid_v:.2E}', f'{max_v:.2E}'])
        diff_frame = self.diff_map_cache[frame_idx]
        im_diff = ax_img_diff.imshow(diff_frame, cmap='coolwarm', vmin=0, vmax=1)
        if frame_idx == 0:
            prev_idx = self.num_frames - 1
        else:
            prev_idx = frame_idx - 1
        ax_img_diff.set_title(f"Spectral Complexity Difference\n(Frame {frame_idx} - {prev_idx})")
        ax_img_diff.axis('off')
        cbar_diff = fig_temp.colorbar(im_diff, ax=ax_img_diff, fraction=0.046, pad=0.04)
        cbar_diff.set_label('Normalized Difference', rotation=270, labelpad=20)
        cbar_diff.set_ticks([0, 0.5, 1])
        raw_diff = self.raw_diff_map_cache[frame_idx]
        v_abs_max = np.percentile(np.abs(raw_diff), 99)
        cbar_diff.set_ticklabels([f'-{v_abs_max:.2E}', '0.00', f'+{v_abs_max:.2E}'])
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig_temp)

    # on_slider_update removed
        
    def on_prev(self, event):
        self.update_frame(self.current_frame - 1)

    def on_next(self, event):
        self.update_frame(self.current_frame + 1)

    def run(self):
        """Shows the plot window."""
        if self.cache_allocated: # Only run if init was successful
            plt.show()

if __name__ == '__main__':
    HDF5_FILE_PATH = 'C:/satelliteImagery/LANDSAT/Rochester/LC08/aligned_image_stack.h5'
    filetypes = [("HDF5 Files", "*.hdf5 *.h5")]
    print("Opening file dialog to select a data file...")
    filepath = nsc.prompt_for_file("Select a stacked HDF5 Data File (.H5)",filetypes,HDF5_FILE_PATH)
    if not os.path.exists(filepath):
        print(f"Error: File not found at '{filepath}'")
    else:
        viewer = HDF5Viewer(filepath)
        if viewer.cache_allocated:
            viewer.run()