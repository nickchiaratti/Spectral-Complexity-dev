import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import tkinter as tk
from tkinter import filedialog

class GhostNativeViewer:
    def __init__(self, h5_path):
        self.h5_path = h5_path
        print(f"Loading {self.h5_path}...")
        self.h5 = h5py.File(self.h5_path, 'r')
        
        grid_path = "HDFEOS/GRIDS/GHOST/Data Fields"
        if grid_path not in self.h5:
            raise KeyError(f"Standard GHOST grid path '{grid_path}' not found in the HDF5 file.")
            
        self.grp = self.h5[grid_path]
        
        # Enforce Dataset Existence
        for expected in ["radiance", "latitude", "longitude", "common_mask", "acquisition_time"]:
            if expected not in self.grp:
                raise KeyError(f"'{expected}' dataset missing. Ensure the native stacker executed successfully.")
            
        self.rad_dset = self.grp["radiance"]
        self.lat_dset = self.grp["latitude"]
        self.lon_dset = self.grp["longitude"]
        self.mask_dset = self.grp["common_mask"]
        self.times = self.grp["acquisition_time"][()]
        
        self.num_frames = self.rad_dset.shape[0]
        self.current_frame = 0
        
        # Approximate RGB Band Indices for a 472-band (400-2500nm) GHOST-5 sensor
        # Assuming ~4.4nm bandwidth. R~640nm, G~550nm, B~460nm
        self.r_idx = 54
        self.g_idx = 34
        self.b_idx = 13
        
        self.setup_ui()

    def get_frame(self, t_idx):
        """Extracts native radiance bands and applies a 2-98% contrast stretch, ignoring NaN padding."""
        # Read the mask and 3 bands
        mask = self.mask_dset[t_idx, :, :]
        r_band = self.rad_dset[t_idx, self.r_idx, :, :]
        g_band = self.rad_dset[t_idx, self.g_idx, :, :]
        b_band = self.rad_dset[t_idx, self.b_idx, :, :]
        
        rgb = np.stack([r_band, g_band, b_band], axis=-1)
        
        # Apply strict 2-98% contrast stretch based ONLY on valid unpadded pixels
        valid_pixels = rgb[mask == 1]
        
        if valid_pixels.size == 0:
            return np.zeros_like(rgb) # Fallback if totally empty/invalid
            
        p2 = np.nanpercentile(valid_pixels, 2, axis=0)
        p98 = np.nanpercentile(valid_pixels, 98, axis=0)
        
        # Normalize and clip
        stretched_rgb = (rgb - p2) / (p98 - p2)
        stretched_rgb = np.clip(stretched_rgb, 0, 1)
        
        # Restore NaN areas as white/transparent background
        stretched_rgb[mask == 0] = np.nan
        
        return stretched_rgb

    def setup_ui(self):
        """Initializes the Matplotlib interactive viewing dashboard."""
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        plt.subplots_adjust(bottom=0.2)
        
        self.im = self.ax.imshow(self.get_frame(self.current_frame))
        self.ax.axis('off')
        
        self.update_title()
        
        # Add interactive buttons
        axprev = plt.axes([0.3, 0.05, 0.1, 0.075])
        axnext = plt.axes([0.6, 0.05, 0.1, 0.075])
        axmap = plt.axes([0.425, 0.05, 0.15, 0.075])
        
        self.bprev = Button(axprev, 'Previous')
        self.bprev.on_clicked(self.prev_frame)
        
        self.bnext = Button(axnext, 'Next')
        self.bnext.on_clicked(self.next_frame)
        
        self.bmap = Button(axmap, 'Export Map')
        self.bmap.on_clicked(self.generate_html_map)
        
        plt.show()

    def update_title(self):
        try:
            time_str = self.times[self.current_frame].decode('utf-8')
        except AttributeError:
            time_str = str(self.times[self.current_frame])
            
        self.ax.set_title(f"GHOST-5 Native Swath | Frame {self.current_frame + 1} / {self.num_frames}\nAcquisition: {time_str}")
        self.fig.canvas.draw_idle()

    def generate_html_map(self, event=None):
        """Generates a Leaflet map tracing the valid footprint of the unprojected flightlines."""
        features = []
        print("Tracing valid coordinate geometry from IGM arrays...")
        
        for t in range(self.num_frames):
            mask = self.mask_dset[t, :, :]
            
            # Find the bounding box indices of valid data in the padded array
            valid_rows, valid_cols = np.where(mask == 1)
            if len(valid_rows) == 0:
                continue
                
            min_r, max_r = valid_rows.min(), valid_rows.max()
            min_c, max_c = valid_cols.min(), valid_cols.max()
            
            # Extract coordinates from the Explicit Lat/Lon arrays
            # We take the 4 corners of the valid swath to form the footprint polygon
            corners = [
                (min_r, min_c), (min_r, max_c), 
                (max_r, max_c), (max_r, min_c)
            ]
            
            poly_coords = []
            for r, c in corners:
                lat = float(self.lat_dset[t, r, c])
                lon = float(self.lon_dset[t, r, c])
                if not np.isnan(lat) and not np.isnan(lon):
                    poly_coords.append([lon, lat])
            
            if len(poly_coords) == 4:
                # Close the polygon
                poly_coords.append(poly_coords[0])
                
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
    <title>GHOST-5 Flightlines</title>
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
        html_path = os.path.join(os.path.dirname(self.h5_path), 'ghost_frames_map.html')
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
    
    print("Please select the GHOST_Native_Stack_HDFEOS.h5 file...")
    file_path = filedialog.askopenfilename(
        title="Select GHOST Native Stack",
        filetypes=[("HDF5 files", "*.h5"), ("All files", "*.*")]
    )
    
    if file_path:
        viewer = GhostNativeViewer(file_path)
    else:
        print("No file selected. Exiting.")