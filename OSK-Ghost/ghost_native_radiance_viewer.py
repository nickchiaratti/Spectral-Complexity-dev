import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import tkinter as tk
from tkinter import filedialog
from rasterio.transform import Affine
from rasterio.warp import transform_bounds

class GhostNativeViewer:
    def __init__(self, h5_path):
        self.h5_path = h5_path
        print(f"Loading {self.h5_path}...")
        self.h5 = h5py.File(self.h5_path, 'r')
        
        grid_path = "HDFEOS/GRIDS/GHOST/Data Fields"
        if grid_path not in self.h5:
            raise KeyError(f"Standard GHOST grid path '{grid_path}' not found in the HDF5 file.")
            
        self.grp = self.h5[grid_path]
        
        # Enforce Dataset Existence - Lat/Lon arrays are now obsolete
        for expected in ["radiance", "common_mask", "acquisition_time"]:
            if expected not in self.grp:
                raise KeyError(f"'{expected}' dataset missing. Ensure the native stacker executed successfully.")
            
        self.rad_dset = self.grp["radiance"]
        self.mask_dset = self.grp["common_mask"]
        self.times = self.grp["acquisition_time"][()]
        
        # Extract Affine Transform directly from CF-compliant attributes
        if 'spatial_transform' in self.rad_dset.attrs:
            st = self.rad_dset.attrs['spatial_transform']
            # st format derived from stacker: [min_lon, res, 0.0, max_lat, 0.0, -res]
            # Map to rasterio Affine(a, b, c, d, e, f)
            self.transform = Affine(st[1], st[2], st[0], st[4], st[5], st[3])
        else:
            raise KeyError("Strict Data Integrity Failure: 'spatial_transform' metadata is missing from the radiance dataset.")
            
        self.num_frames = self.rad_dset.shape[0]
        self.current_frame = 0
        
        # Approximate RGB Band Indices for a 472-band (400-2500nm) GHOST-5 sensor
        self.r_idx = 54
        self.g_idx = 34
        self.b_idx = 13
        
        self.setup_ui()

    def get_frame(self, t_idx):
        """Extracts native radiance bands and applies a 2-98% contrast stretch, ignoring NaN padding."""
        mask = self.mask_dset[t_idx, :, :]
        r_band = self.rad_dset[t_idx, self.r_idx, :, :]
        g_band = self.rad_dset[t_idx, self.g_idx, :, :]
        b_band = self.rad_dset[t_idx, self.b_idx, :, :]
        
        rgb = np.stack([r_band, g_band, b_band], axis=-1)
        
        # Apply strict 2-98% contrast stretch based ONLY on valid unpadded pixels
        valid_pixels = rgb[mask == 1]
        
        if valid_pixels.size == 0:
            return np.full_like(rgb, np.nan) # Strict data integrity: pure NaN instead of synthetic 0.0 fill values
            
        p2 = np.nanpercentile(valid_pixels, 2, axis=0)
        p98 = np.nanpercentile(valid_pixels, 98, axis=0)
        
        # Normalize and clip
        stretched_rgb = (rgb - p2) / (p98 - p2)
        stretched_rgb = np.clip(stretched_rgb, 0, 1)
        
        # Restore NaN areas as transparent/white background to avoid visual artifacts
        stretched_rgb[mask == 0] = np.nan
        
        return stretched_rgb

    def setup_ui(self):
        """Initializes the Matplotlib interactive viewing dashboard."""
        self.fig, self.ax = plt.subplots(figsize=(14, 8))
        plt.subplots_adjust(left=0.05, right=0.98, top=0.95, bottom=0.18)
        
        # Calculate the EPSG:4326 spatial extent
        height, width = self.rad_dset.shape[2], self.rad_dset.shape[3]
        left_deg, top_deg = self.transform * (0, 0)
        right_deg, bottom_deg = self.transform * (width, height)
        
        # Dynamically determine the local UTM zone for accurate metric display
        center_lon = (left_deg + right_deg) / 2
        center_lat = (top_deg + bottom_deg) / 2
        utm_zone = int((center_lon + 180) / 6) + 1
        utm_epsg = 32600 + utm_zone if center_lat >= 0 else 32700 + utm_zone
        
        # Project geographic bounds to UTM meters
        left_m, bottom_m, right_m, top_m = transform_bounds(
            "EPSG:4326", f"EPSG:{utm_epsg}", left_deg, bottom_deg, right_deg, top_deg
        )
        self.extent = [left_m, right_m, bottom_m, top_m]
        
        self.im = self.ax.imshow(self.get_frame(self.current_frame), extent=self.extent, aspect='auto')
        
        # Enable metric coordinate axes
        self.ax.set_xlabel(f"Easting (meters, UTM Zone {utm_zone})")
        self.ax.set_ylabel(f"Northing (meters, UTM Zone {utm_zone})")
        self.ax.grid(True, linestyle='--', alpha=0.5, color='gray')
        
        # Best practice: Prevent matplotlib from abstracting coordinates into scientific notation offsets
        self.ax.ticklabel_format(useOffset=False, style='plain')
        
        self.update_title()
        
        # Add interactive buttons
        axprev = plt.axes([0.2, 0.05, 0.1, 0.075])
        axmap = plt.axes([0.325, 0.05, 0.15, 0.075])
        axexport = plt.axes([0.5, 0.05, 0.15, 0.075])
        axnext = plt.axes([0.675, 0.05, 0.1, 0.075])
        
        self.bprev = Button(axprev, 'Previous')
        self.bprev.on_clicked(self.prev_frame)
        
        self.bnext = Button(axnext, 'Next')
        self.bnext.on_clicked(self.next_frame)
        
        self.bmap = Button(axmap, 'Export Map')
        self.bmap.on_clicked(self.generate_html_map)

        self.bexport = Button(axexport, 'Export PNGs')
        self.bexport.on_clicked(self.export_frames)
        plt.show()

    def export_frames(self, event=None):
        """Exports all valid frames to PNGs, capturing the full Matplotlib figure including axes and coordinate grids."""
        export_dir = os.path.join(os.path.dirname(self.h5_path), 'images')
        os.makedirs(export_dir, exist_ok=True)
        print(f"\nExporting {self.num_frames} frames to {export_dir}...")
        
        # Hide interactive UI button axes so they don't appear in the exported maps
        ui_axes = [self.bprev.ax, self.bmap.ax, self.bexport.ax, self.bnext.ax]
        for ax in ui_axes:
            ax.set_visible(False)

        # Store the current state to restore it after exporting
        original_frame = self.current_frame
        
        for t in range(self.num_frames):
            mask = self.mask_dset[t, :, :]
            
            # Explicitly halt and skip if a frame is entirely out of bounds / empty
            if np.max(mask) == 0:
                continue  
                
            # Update the viewer state to the target frame
            self.current_frame = t
            self.im.set_data(self.get_frame(t))
            self.update_title()
            
            # Force a canvas draw to render the updated image and title
            self.fig.canvas.draw()
            
            try:
                time_str = self.times[t].decode('utf-8')
            except AttributeError:
                time_str = str(self.times[t])
                
            # The dt_str from the stacker ("YYYY-MM-DDTHH-MM-SSZ") is already path-safe
            safe_time_str = time_str.replace(':', '-')
            filename = f"{safe_time_str}.png"
            filepath = os.path.join(export_dir, filename)
            
            # Export the entire figure (including coordinate axes, grid, and title)
            # bbox_inches='tight' ensures the axes labels are not cut off
            self.fig.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"  -> Saved full figure: {filename}")
            
        # Restore the viewer to its original state and turn buttons back on
        for ax in ui_axes:
            ax.set_visible(True)
            
        self.current_frame = original_frame
        self.im.set_data(self.get_frame(self.current_frame))
        self.update_title()
        self.fig.canvas.draw()
        
        print("Export complete.\n")

    def update_title(self):
        try:
            time_str = self.times[self.current_frame].decode('utf-8')
        except AttributeError:
            time_str = str(self.times[self.current_frame])
            
        self.ax.set_title(f"GHOST-5 WGS84 Ortho | Frame {self.current_frame + 1} / {self.num_frames}\nAcquisition: {time_str}")
        self.fig.canvas.draw_idle()

    def generate_html_map(self, event=None):
        """Generates a Leaflet map using affine transformation of the valid footprint mask."""
        features = []
        print("Calculating geographic bounding polygons via affine transformations...")
        
        for t in range(self.num_frames):
            mask = self.mask_dset[t, :, :]
            
            # Find the bounding box indices of valid data in the grid
            valid_rows, valid_cols = np.where(mask == 1)
            if len(valid_rows) == 0:
                continue
                
            min_r, max_r = valid_rows.min(), valid_rows.max()
            min_c, max_c = valid_cols.min(), valid_cols.max()
            
            # Map pixel boundary corners to Geographic (WGS84) coordinates using the affine transform
            # Affine multiplication: transform * (column_x, row_y) -> (longitude, latitude)
            corners = [
                self.transform * (min_c, min_r),
                self.transform * (max_c, min_r),
                self.transform * (max_c, max_r),
                self.transform * (min_c, max_r)
            ]
            
            # Close the polygon
            poly_coords = [list(c) for c in corners]
            poly_coords.append(list(corners[0]))
            
            try:
                time_str = self.times[t].decode('utf-8')
            except AttributeError:
                time_str = str(self.times[t])
                
            feature = {
                "type": "Feature",
                "properties": {
                    "frame": t + 1,
                    "time": time_str
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [poly_coords]
                }
            }
            features.append(feature)

        geojson_data = {
            "type": "FeatureCollection",
            "features": features
        }
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>GHOST-5 Ortho Footprints</title>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>#map {{ width: 100%; height: 100vh; margin: 0; padding: 0; }}</style>
</head>
<body>
<div id="map"></div>
<script>
    var map = L.map('map').setView([0, 0], 2);
    L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
        maxZoom: 19,
        attribution: '© OpenStreetMap'
    }}).addTo(map);

    var geojsonData = {geojson_data};

    function style(feature) {{
        return {{ fillColor: "#ff7800", color: "#000", weight: 2, opacity: 1, fillOpacity: 0.3 }};
    }}

    function onEachFeature(feature, layer) {{
        if (feature.properties) {{
            layer.bindPopup("<b>Frame:</b> " + feature.properties.frame + "<br><b>Time:</b> " + feature.properties.time);
        }}
    }}

    var geojson = L.geoJson(geojsonData, {{
        style: style,
        onEachFeature: onEachFeature
    }}).addTo(map);

    if (geojsonData.features.length > 0) {{
        map.fitBounds(geojson.getBounds());
    }}
</script>
</body>
</html>
"""
        html_path = os.path.join(os.path.dirname(self.h5_path), 'ghost_ortho_map.html')
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        print(f"Map successfully saved to: {html_path}")

    def prev_frame(self, event):
        if self.current_frame > 0:
            self.current_frame -= 1
            self.im.set_data(self.get_frame(self.current_frame))
            self.update_title()

    def next_frame(self, event):
        if self.current_frame < self.num_frames - 1:
            self.current_frame += 1
            self.im.set_data(self.get_frame(self.current_frame))
            self.update_title()

if __name__ == "__main__":
    # Hide the main tkinter window
    root = tk.Tk()
    root.withdraw()
    
    #print("Please select the GHOST_Native_Stack_HDFEOS.h5 file...")
    #file_path = filedialog.askopenfilename(
    #    title="Select GHOST Ortho Stack",
    #    filetypes=[("HDF5 files", "*.h5"), ("All files", "*.*")]
    #)

    file_path = "C:/satelliteImagery/OSK-Ghost/SourceData/GHOST_Native_Stack_HDFEOS.h5"
    
    if file_path:
        viewer = GhostNativeViewer(file_path)
    else:
        print("No file selected. Exiting.")