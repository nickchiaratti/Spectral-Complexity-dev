import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import tkinter as tk
from tkinter import filedialog
from datetime import datetime, timezone
import os
import json
import rasterio
from rasterio.warp import transform_bounds, transform_geom, transform
from rasterio.features import shapes
from rasterio.transform import Affine

class DragonetteNativeViewer:
    def __init__(self, h5_path):
        self.h5_path = h5_path
        
        print(f"Loading {self.h5_path}...")
        self.h5 = h5py.File(self.h5_path, 'r')
        
        grid_path = "HDFEOS/GRIDS/WYVERN/Data Fields"
        if grid_path not in self.h5:
            raise KeyError(f"Standard WYVERN grid path '{grid_path}' not found in the provided HDF5 file.")
            
        self.grp = self.h5[grid_path]
        
        # Enforce Dataset Existence
        if "ortho_visual" not in self.grp:
            raise KeyError("'ortho_visual' dataset missing. Ensure the native stacker executed successfully.")
        if "radiance" not in self.grp:
            raise KeyError("'radiance' dataset missing.")
            
        self.vis_dset = self.grp["ortho_visual"]
        self.rad_dset = self.grp["radiance"]
        
        # Enforce Explicit Metadata Integrity
        if "GeoTransform" not in self.vis_dset.attrs:
            raise AttributeError("CRITICAL: 'GeoTransform' attribute missing from ortho_visual. Cannot establish absolute spatial extent.")
        if "acquisition_time" not in self.rad_dset.attrs:
            raise AttributeError("CRITICAL: 'acquisition_time' attribute missing from radiance.")

        self.num_frames, self.bands, self.height, self.width = self.vis_dset.shape
        self.times = self.rad_dset.attrs["acquisition_time"]
        self.current_frame = 0
        
        # Decode GDAL Affine Transform into Matplotlib Extent [left, right, bottom, top]
        # Reference: GDAL Data Model (Transform[0]=UL_X, Transform[1]=W_Res, Transform[3]=UL_Y, Transform[5]=H_Res)
        gt = self.vis_dset.attrs["GeoTransform"]
        left = gt[0]
        top = gt[3]
        pixel_width = gt[1]
        pixel_height = gt[5]
        
        right = left + (self.width * pixel_width)
        bottom = top + (self.height * pixel_height)
        self.extent = [left, right, bottom, top]
        
        print(f"Spatial Grid Confirmed: {self.width}x{self.height} pixels.")
        print(f"Extent (Degrees): X[{left:.6f}, {right:.6f}], Y[{bottom:.6f}, {top:.6f}]")

        self.setup_ui()

    def get_frame(self, idx):
        """
        Extracts the requested frame, applies dynamic contrast stretch for viewing,
        and injects NaN for transparency to prevent empty background regions
        from occluding the geographic grid.
        """
        # Read array: (3, Height, Width) -> Transpose to (Height, Width, 3)
        raw_img = self.vis_dset[idx, ...]
        img = np.transpose(raw_img, (1, 2, 0)).astype(np.float32)
        
        # Identify absolute background using fillvalue
        fill_val = self.vis_dset.fillvalue if hasattr(self.vis_dset, 'fillvalue') and self.vis_dset.fillvalue is not None else 0
        background_mask = np.all(img == fill_val, axis=-1)
        
        # Apply stretch to valid data
        valid_mask = ~background_mask
        if np.any(valid_mask):
            p_low, p_high = np.percentile(img[valid_mask], (0.5, 99.5))
            img = (img - p_low) / (p_high - p_low + 1e-5)
            img = np.clip(img, 0, 1)
        
        # Inject NaN
        img[background_mask] = np.nan
        
        return img

    def setup_ui(self):
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        self.fig.canvas.manager.set_window_title("Dragonette Native Geospatial Viewer")
        plt.subplots_adjust(bottom=0.15)
        
        # Render first frame
        self.im = self.ax.imshow(self.get_frame(self.current_frame), extent=self.extent, origin='upper')
        
        # Format axes for rigorous geospatial inspection (Dragonette is EPSG:4326)
        self.ax.set_xlabel("Longitude (Degrees)", fontweight='bold')
        self.ax.set_ylabel("Latitude (Degrees)", fontweight='bold')
        self.ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.4f'))
        self.ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.4f'))
        self.ax.grid(True, linestyle='--', alpha=0.5, color='gray')
        
        self.update_title()
        
        print("Saving all frames to PNG...")
        original_frame = self.current_frame
        for i in range(self.num_frames):
            self.current_frame = i
            self.im.set_data(self.get_frame(self.current_frame))
            self.update_title()
            self.fig.savefig(os.path.join(os.path.dirname(self.h5_path), f'ortho_visual_{i:03d}.png'))
        self.current_frame = original_frame
        self.im.set_data(self.get_frame(self.current_frame))
        self.update_title()
        print("Finished saving frames.")
        
        self.generate_html_map()
        
        # Interaction Panel
        ax_prev = plt.axes([0.4, 0.05, 0.08, 0.06])
        ax_next = plt.axes([0.52, 0.05, 0.08, 0.06])
        self.btn_prev = Button(ax_prev, 'Previous')
        self.btn_next = Button(ax_next, 'Next')
        
        self.btn_prev.on_clicked(self.prev_frame)
        self.btn_next.on_clicked(self.next_frame)
        
        plt.show()

    def update_title(self):
        unix_time = self.times[self.current_frame]
        dt = datetime.fromtimestamp(unix_time, tz=timezone.utc)
        
        # Explicit formatting showing precise temporal resolution
        time_str = dt.strftime('%Y-%m-%d %H:%M:%S UTC')
        title_text = (f"Dragonette Native Stack (Wyvern)\n"
                      f"Temporal Pass {self.current_frame + 1} of {self.num_frames} | {time_str}")
        self.ax.set_title(title_text, fontsize=13, pad=15)
        self.fig.canvas.draw_idle()

    def generate_html_map(self):
        print("Generating HTML map of actual imagery footprints...")
        features = []
        
        gt = self.vis_dset.attrs["GeoTransform"]
        spatial_ref = self.vis_dset.attrs.get("spatial_ref", None)
        if isinstance(spatial_ref, bytes):
            spatial_ref = spatial_ref.decode('utf-8')
            
        if not spatial_ref:
            print("Warning: No spatial_ref found. Cannot generate map.")
            return

        tf = Affine.from_gdal(*gt)

        for i in range(self.num_frames):
            raw_img = self.vis_dset[i, ...]
            fill_val = self.vis_dset.fillvalue if hasattr(self.vis_dset, 'fillvalue') and self.vis_dset.fillvalue is not None else 0
            
            if raw_img.ndim == 3:
                valid_mask = ~np.all(raw_img == fill_val, axis=0)
            else:
                valid_mask = (raw_img != fill_val)

            if not np.any(valid_mask):
                continue
                
            unix_time = self.times[i]
            dt = datetime.fromtimestamp(unix_time, tz=timezone.utc)
            time_str = dt.strftime('%Y-%m-%d %H:%M:%S UTC')
            color_str = f"hsl({(i * 137.5) % 360:.1f}, 70%, 50%)"

            try:
                geoms = list(shapes(valid_mask.astype('uint8'), mask=valid_mask, transform=tf))
                for geom, val in geoms:
                    if spatial_ref.upper() == "EPSG:4326":
                        geom_4326 = geom
                    else:
                        geom_4326 = transform_geom(spatial_ref, "EPSG:4326", geom)
                    feature = {
                        "type": "Feature",
                        "properties": {
                            "is_corner": False,
                            "frame": i,
                            "time": time_str,
                            "color": color_str
                        },
                        "geometry": geom_4326
                    }
                    features.append(feature)
            except Exception as e:
                print(f"Failed to extract footprint for frame {i}: {e}")

            valid_coords = np.argwhere(valid_mask)
            if len(valid_coords) > 0:
                sum_c = valid_coords[:, 0] + valid_coords[:, 1]
                diff_c = valid_coords[:, 0] - valid_coords[:, 1]
                
                corners = {
                    "Top Left": valid_coords[np.argmin(sum_c)],
                    "Bottom Right": valid_coords[np.argmax(sum_c)],
                    "Top Right": valid_coords[np.argmin(diff_c)],
                    "Bottom Left": valid_coords[np.argmax(diff_c)]
                }
                
                for name, (row, col) in corners.items():
                    utm_x, utm_y = tf * (col, row)
                    
                    try:
                        if spatial_ref.upper() == "EPSG:4326":
                            lon, lat = utm_x, utm_y
                        else:
                            lons, lats = transform(spatial_ref, "EPSG:4326", [utm_x], [utm_y])
                            lon, lat = lons[0], lats[0]
                        
                        pt_feature = {
                            "type": "Feature",
                            "properties": {
                                "is_corner": True,
                                "frame": i,
                                "corner_name": name,
                                "lat": f"{lat:.6f}",
                                "lon": f"{lon:.6f}",
                                "color": color_str
                            },
                            "geometry": {
                                "type": "Point",
                                "coordinates": [lon, lat]
                            }
                        }
                        features.append(pt_feature)
                    except Exception as e:
                        pass

        if not features:
            print("No valid frames found to map.")
            return

        geojson_data = {
            "type": "FeatureCollection",
            "features": features
        }

        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Dragonette Frames Map</title>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        body {{ margin: 0; padding: 0; }}
        #map {{ width: 100vw; height: 100vh; }}
        .info-box {{
            padding: 6px 8px;
            font: 14px/16px Arial, Helvetica, sans-serif;
            background: white;
            background: rgba(255,255,255,0.8);
            box-shadow: 0 0 15px rgba(0,0,0,0.2);
            border-radius: 5px;
        }}
        .info-box h4 {{ margin: 0 0 5px; color: #777; }}
    </style>
</head>
<body>
<div id="map"></div>
<script>
    var map = L.map('map').setView([0, 0], 2);

    L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
        maxZoom: 19,
        attribution: '&copy; OpenStreetMap contributors'
    }}).addTo(map);

    var geojsonData = {json.dumps(geojson_data)};

    var info = L.control();

    info.onAdd = function (map) {{
        this._div = L.DomUtil.create('div', 'info-box');
        this.update();
        return this._div;
    }};

    info.update = function (props) {{
        this._div.innerHTML = '<h4>Frame Info</h4>' +  (props ?
            '<b>Frame: ' + props.frame + '</b><br />' + props.time
            : 'Hover over a bounding box');
    }};

    info.addTo(map);

    function style(feature) {{
        if (feature.properties.is_corner) {{
            return {{
                radius: 6,
                fillColor: feature.properties.color,
                color: "#000",
                weight: 1,
                opacity: 1,
                fillOpacity: 0.8
            }};
        }}
        return {{
            weight: 2,
            opacity: 1,
            color: feature.properties.color,
            fillOpacity: 0.2
        }};
    }}

    function highlightFeature(e) {{
        var layer = e.target;
        if (layer.feature.properties.is_corner) return;
        layer.setStyle({{
            weight: 5,
            fillOpacity: 0.5
        }});
        layer.bringToFront();
        info.update(layer.feature.properties);
    }}

    function resetHighlight(e) {{
        var layer = e.target;
        if (layer.feature.properties.is_corner) return;
        geojson.resetStyle(layer);
        info.update();
    }}

    function zoomToFeature(e) {{
        var layer = e.target;
        if (layer.feature.properties.is_corner) return;
        map.fitBounds(layer.getBounds());
    }}

    function onEachFeature(feature, layer) {{
        if (feature.properties.is_corner) {{
            var tooltipContent = "<b>" + feature.properties.corner_name + "</b><br><b>Lat:</b> " + feature.properties.lat + "<br><b>Lon:</b> " + feature.properties.lon;
            layer.bindTooltip(tooltipContent, {{
                direction: 'top',
                className: 'corner-tooltip'
            }});
        }} else {{
            layer.on({{
                mouseover: highlightFeature,
                mouseout: resetHighlight,
                click: zoomToFeature
            }});
            layer.bindPopup("<b>Frame:</b> " + feature.properties.frame + "<br><b>Time:</b> " + feature.properties.time);
        }}
    }}

    var geojson = L.geoJson(geojsonData, {{
        pointToLayer: function (feature, latlng) {{
            return L.circleMarker(latlng, style(feature));
        }},
        style: style,
        onEachFeature: onEachFeature
    }}).addTo(map);

    map.fitBounds(geojson.getBounds());
</script>
</body>
</html>
"""
        html_path = os.path.join(os.path.dirname(self.h5_path), 'frames_map.html')
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        print(f"Map saved to: {html_path}")

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
    root = tk.Tk()
    root.withdraw()
    
    print("Please select the Dragonette Native Stack HDF5 file...")
    file_path = filedialog.askopenfilename(
        title="Select Dragonette Native Stack",
        filetypes=[("HDF5 files", "*.h5")]
    )
    
    if file_path:
        viewer = DragonetteNativeViewer(file_path)
    else:
        print("No file selected. Exiting.")
    root.destroy()
