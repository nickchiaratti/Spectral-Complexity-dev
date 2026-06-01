import h5py
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from tkinter import filedialog, Tk

# --- Configuration ---
TARGET_RGB = {'R': 0.655, 'G': 0.561, 'B': 0.482}

class TanagerStackViewer:
    def __init__(self, h5_path=None):
        if h5_path:
            self.h5_path = h5_path
        else:
            self.h5_path = self.prompt_file()
            if not self.h5_path:
                print("No file selected. Exiting.")
                return

        self.current_frame = 0
        
        # Open file
        try:
            self.h5 = h5py.File(self.h5_path, 'r')
        except Exception as e:
            print(f"Error opening file {self.h5_path}: {e}")
            return

        # 1. Locate Surface Reflectance (for dimensions/metadata)
        standard_path = "/HDFEOS/GRIDS/TANAGER/Data Fields/surface_reflectance"
        if standard_path in self.h5:
            self.sr_path = standard_path
        else:
            print(f"Standard path '{standard_path}' not found. Searching...")
            self.sr_path = self.find_dataset("reflectance")
            
        if not self.sr_path:
             print("Error: Could not find 'surface_reflectance' dataset.")
             return
        
        self.reflectance = self.h5[self.sr_path]
        self.num_frames = self.reflectance.shape[0]
        
        # 2. Locate Ortho Visual
        vis_std_path = "/HDFEOS/GRIDS/TANAGER/Data Fields/ortho_visual"
        if vis_std_path in self.h5:
            self.vis_path = vis_std_path
            self.visuals = self.h5[self.vis_path]
        else:
            print("Warning: 'ortho_visual' dataset not found. Left panel will be empty.")
            self.visuals = None

        # 3. Locate Sliding Volume Map (Calculated)
        vol_path = "/HDFEOS/GRIDS/HYP/Data Fields/sliding_volume_map"
        self.vol_attrs = {}
        if vol_path in self.h5:
            self.vol_path = vol_path
            self.volume_map = self.h5[self.vol_path]
            print("Found Sliding Volume Map.")
            # Read attributes for display
            if 'tile_size' in self.volume_map.attrs:
                self.vol_attrs['tile_size'] = self.volume_map.attrs['tile_size']
            if 'sliding_stride' in self.volume_map.attrs:
                self.vol_attrs['stride'] = self.volume_map.attrs['sliding_stride']
        else:
            print("Warning: 'sliding_volume_map' not found. Right panel will be empty.")
            self.volume_map = None

        # Read Fill Value for SR
        self.fill_value = None
        if "_FillValue" in self.reflectance.attrs:
            fv = self.reflectance.attrs["_FillValue"]
            self.fill_value = fv[0] if isinstance(fv, (np.ndarray, list, tuple)) else fv
            
        self.setup_ui()

    def prompt_file(self):
        print("Opening file dialog to select a data file...")
        root = Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(
            title="Select Tanager HDF5 Stack (Calculated)",
            filetypes=[("HDF5 files", "*.h5"), ("All files", "*.*")]
        )
        root.destroy()
        return file_path

    def find_dataset(self, keyword):
        candidates = []
        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                if keyword in name.lower() and len(obj.shape) >= 3:
                    candidates.append(name)
        self.h5.visititems(visitor)
        return candidates[0] if candidates else None

    def get_current_timestamp(self):
        try:
            if "METADATA" in self.h5:
                key = f"frame_{self.current_frame}_json"
                if key in self.h5["METADATA"].attrs:
                    data = json.loads(self.h5["METADATA"].attrs[key])
                    return data['properties'].get('datetime', f'Frame {self.current_frame}')
        except Exception:
            pass
        return f"Frame {self.current_frame}"

    def get_visual_rgb(self, frame_idx):
        if self.visuals is None:
            return np.zeros((100, 100, 3), dtype=np.uint8)
            
        data = self.visuals[frame_idx]
        if data.shape[0] == 4: # (4, H, W) -> Transpose to (H, W, 4)
            data = np.transpose(data, (1, 2, 0))
            
        if data.shape[-1] == 4:
            rgb = data[..., :3]
            return rgb
        return data

    def get_volume_data(self, frame_idx):
        if self.volume_map is None:
            return np.zeros((100, 100))
        
        # Volume map is usually (Time, Y, X)
        vol = self.volume_map[frame_idx, ...]
        
        # Handle Fill Value if present
        #mask = (vol == -9999.0) | (vol == 0)
        #vol_masked = np.ma.masked_where(mask, vol)
        
        return vol

    def get_nan_mask(self, frame_idx):
        """Creates a boolean mask where True (White) indicates NaN or FillValue in ANY band."""
        # Read full spectral cube for this frame: (Bands, Y, X)
        cube_data = self.reflectance[frame_idx, ...]
        
        mask = np.zeros(cube_data.shape[1:], dtype=bool)
        
        # Check for NaN (Standard)
        if np.issubdtype(cube_data.dtype, np.floating):
            mask |= np.any(np.isnan(cube_data), axis=0)
            
        # Check for Fill Value (Standard Tanager is -9999.0)
        if self.fill_value is not None:
            mask |= np.any(cube_data == self.fill_value, axis=0)
            
        return mask

    def setup_ui(self):
        # Create 1 row, 2 columns (Removed 3rd column)
        self.fig, (self.ax_vis, self.ax_vol) = plt.subplots(1, 2, figsize=(14, 7))
        plt.subplots_adjust(bottom=0.2)
        
        # -- Left: Ortho Visual --
        self.img_vis = self.ax_vis.imshow(self.get_visual_rgb(self.current_frame))
        self.ax_vis.set_title("Ortho Visual (True Color)")
        self.ax_vis.axis('off')
        
        # -- Right: Sliding Volume Map --
        vol_data = self.get_volume_data(self.current_frame)
        self.img_vol = self.ax_vol.imshow(vol_data, cmap='magma') 
        
        # Construct dynamic title from attributes
        vol_title = "Sliding Volume Map"
        if self.vol_attrs:
            ts = self.vol_attrs.get('tile_size', '?')
            st = self.vol_attrs.get('stride', '?')
            vol_title += f"\n(Tile Size: {ts}, Stride: {st})"
            
        self.ax_vol.set_title(vol_title)
        self.ax_vol.axis('off')
        
        # Colorbar for volume
        self.cbar = self.fig.colorbar(self.img_vol, ax=self.ax_vol, fraction=0.046, pad=0.04)
        self.cbar.set_label('Spectral Complexity')

        # Global Title
        time_str = self.get_current_timestamp()
        self.fig.suptitle(f"Pass {self.current_frame+1}/{self.num_frames} : {time_str}", fontsize=14)
        
        # Buttons
        ax_prev = plt.axes([0.4, 0.05, 0.08, 0.075])
        ax_next = plt.axes([0.52, 0.05, 0.08, 0.075])
        
        self.btn_prev = Button(ax_prev, 'Previous')
        self.btn_next = Button(ax_next, 'Next')
        
        self.btn_prev.on_clicked(lambda e: self.update(-1))
        self.btn_next.on_clicked(lambda e: self.update(1))
        
        plt.show()

    def update(self, delta):
        new_f = self.current_frame + delta
        if 0 <= new_f < self.num_frames:
            self.current_frame = new_f
            
            # Update Images
            self.img_vis.set_data(self.get_visual_rgb(self.current_frame))
            
            vol_data = self.get_volume_data(self.current_frame)
            self.img_vol.set_data(vol_data)
            self.img_vol.set_clim(vmin=vol_data.min(), vmax=vol_data.max())
            
            # Update Title
            time_str = self.get_current_timestamp()
            self.fig.suptitle(f"Pass {self.current_frame+1}/{self.num_frames} : {time_str}", fontsize=14)
            
            self.fig.canvas.draw_idle()

    def close(self):
        if hasattr(self, 'h5') and self.h5:
            self.h5.close()

if __name__ == "__main__":
    viewer = TanagerStackViewer()