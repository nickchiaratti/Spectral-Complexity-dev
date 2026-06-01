import os
import glob
import json
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons, Button
from matplotlib.patches import Polygon
from pathlib import Path

# --- Configuration ---
SOURCE_DIR = r"C:\satelliteImagery\Tanager\SourceData"

def load_geotiff_data(source_dir):
    """
    Crawls the source directory to find ortho_visual.tif files and their metadata.
    """
    root_path = Path(source_dir)
    frames = []
    
    # Search for all subfolders
    for subfolder in root_path.iterdir():
        if not subfolder.is_dir():
            continue
            
        tif_files = list(subfolder.glob("*_ortho_visual.tif"))
        json_files = list(subfolder.glob("*.json"))
        
        if tif_files and json_files:
            frames.append({
                'id': subfolder.name,
                'tif_path': tif_files[0],
                'json_path': json_files[0]
            })
            
    return sorted(frames, key=lambda x: x['id'])

class TanagerMosaicViewer:
    def __init__(self, frames):
        self.frames = frames
        self.visible_flags = [True] * len(frames)
        self.borders_visible = True
        self.images = []
        self.extents = []
        self.border_patches = []
        
        print(f"Loading {len(frames)} GeoTIFFs...")
        self.preload_data()
        self.setup_ui()

    def preload_data(self):
        """Reads the imagery and spatial footprint (corners) for each file."""
        for frame in self.frames:
            with rasterio.open(frame['tif_path']) as src:
                # Read RGB (first 3 bands) and Alpha (4th band)
                img = src.read([1, 2, 3, 4])
                img = np.moveaxis(img, 0, -1).astype(np.float32) / 255.0
                
                # Rasterio bounds: (left, bottom, right, top)
                # Matplotlib imshow extent: [left, right, bottom, top]
                b = src.bounds
                extent = [b.left, b.right, b.bottom, b.top]
                
                # Get actual corner coordinates for rotated imagery
                # (0,0), (0, width), (height, width), (height, 0) in pixel space
                # transformed to UTM coordinates
                corners = [
                    src.xy(0, 0),
                    src.xy(0, src.width),
                    src.xy(src.height, src.width),
                    src.xy(src.height, 0)
                ]
                
                self.images.append(img)
                self.extents.append(extent)
                self.border_patches.append(np.array(corners))

    def setup_ui(self):
        # Create Main Figure for Map
        self.fig_map, self.ax_map = plt.subplots(figsize=(12, 10))
        self.fig_map.canvas.manager.set_window_title('Tanager Mosaic Viewer')
        
        self.plot_artists = []
        self.ui_patches = []
        
        for i, (img, ext, corners) in enumerate(zip(self.images, self.extents, self.border_patches)):
            # Display Image
            artist = self.ax_map.imshow(img, extent=ext, interpolation='nearest', zorder=1)
            self.plot_artists.append(artist)
            
            # Create Polygon Patch for rotated boundary
            poly = Polygon(corners, closed=True, linewidth=2, edgecolor='yellow', 
                           facecolor='none', zorder=10, alpha=0.8)
            self.ax_map.add_patch(poly)
            self.ui_patches.append(poly)

        self.ax_map.set_xlabel('UTM Easting (m)')
        self.ax_map.set_ylabel('UTM Northing (m)')
        self.ax_map.set_title('Tanager Ortho Visual Mosaic (EPSG:32618)')

        # Create Control Figure for Selection
        self.fig_ctrl, (self.ax_ctrl, self.ax_btn) = plt.subplots(2, 1, figsize=(4, 8), 
                                                                 gridspec_kw={'height_ratios': [8, 1]})
        self.fig_ctrl.canvas.manager.set_window_title('Frame Selector')
        
        # Frame Checkboxes
        labels = [f["id"] for f in self.frames]
        self.check = CheckButtons(self.ax_ctrl, labels, self.visible_flags)
        self.check.on_clicked(self.toggle_visibility)
        
        # Toggle Borders Button
        self.btn_border = Button(self.ax_btn, 'Toggle Borders')
        self.btn_border.on_clicked(self.toggle_borders)
        
        plt.show()

    def toggle_visibility(self, label):
        # Find index of the label clicked
        index = [f["id"] for f in self.frames].index(label)
        self.visible_flags[index] = not self.visible_flags[index]
        
        # Update visibility in map (Image and its specific border)
        self.plot_artists[index].set_visible(self.visible_flags[index])
        
        # Border should only be visible if both the frame is checked AND global borders are ON
        should_show_border = self.visible_flags[index] and self.borders_visible
        self.ui_patches[index].set_visible(should_show_border)
        
        self.fig_map.canvas.draw_idle()

    def toggle_borders(self, event):
        self.borders_visible = not self.borders_visible
        for i, patch in enumerate(self.ui_patches):
            # Only show border if the frame itself is currently visible
            patch.set_visible(self.borders_visible and self.visible_flags[i])
        
        self.fig_map.canvas.draw_idle()

def main():
    print("Initializing Tanager Mosaic Viewer...")
    frames = load_geotiff_data(SOURCE_DIR)
    
    if not frames:
        print(f"No valid frame folders found in {SOURCE_DIR}")
        return
        
    viewer = TanagerMosaicViewer(frames)

if __name__ == "__main__":
    main()