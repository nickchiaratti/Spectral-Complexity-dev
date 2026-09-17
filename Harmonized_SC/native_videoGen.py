import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tkinter as tk
from tkinter import filedialog
from datetime import datetime, timezone

# ==========================================
# GLOBAL CONFIGURATION
# ==========================================
FPS = 2
DPI = 100

def find_dataset(h5_group, dataset_name):
    """Recursively search for a dataset by name in the HDF5 group."""
    found = []
    def visit_func(name, node):
        if isinstance(node, h5py.Dataset) and name.endswith(dataset_name):
            found.append(node.name)
    h5_group.visititems(visit_func)
    return found[0] if found else None

def generate_video(h5_filepath):
    """Generate an mp4 video from the ortho_visual dataset in the HDF5 file."""
    output_dir = os.path.dirname(h5_filepath)
    basename = os.path.splitext(os.path.basename(h5_filepath))[0]
    output_mp4 = os.path.join(output_dir, f"{basename}_ortho_visual.mp4")

    print(f"Opening HDF5 file: {h5_filepath}")
    
    with h5py.File(h5_filepath, 'r') as f:
        # Find ortho_visual dataset dynamically
        ortho_path = find_dataset(f, 'ortho_visual')
        if not ortho_path:
            raise ValueError(f"CRITICAL ERROR: 'ortho_visual' dataset not found in {h5_filepath}.")
        
        ortho_ds = f[ortho_path]
        num_frames = ortho_ds.shape[0]
        
        # Determine height and width from the dataset shape.
        # EnMAP MGRS stack ortho_visual shape is typically (Time, RGBABand, YDim, XDim)
        if ortho_ds.shape[1] in [3, 4]:
            height, width = ortho_ds.shape[2], ortho_ds.shape[3]
        else:
            height, width = ortho_ds.shape[1], ortho_ds.shape[2]
            
        print(f"Found ortho_visual: {num_frames} frames, {width}x{height} pixels.")
        
        # Try to find acquisition_time. Usually stored as an attribute on surface_reflectance.
        parent_group_path = os.path.dirname(ortho_path)
        sr_path = os.path.join(parent_group_path, 'surface_reflectance')
        
        acq_times = None
        if sr_path in f:
            sr_ds = f[sr_path]
            if 'acquisition_time' in sr_ds.attrs:
                acq_times = sr_ds.attrs['acquisition_time']
                
        if acq_times is None:
            # Fallback to searching the whole file for surface_reflectance
            sr_path_found = find_dataset(f, 'surface_reflectance')
            if sr_path_found:
                sr_ds = f[sr_path_found]
                if 'acquisition_time' in sr_ds.attrs:
                    acq_times = sr_ds.attrs['acquisition_time']
                    
        if acq_times is None or len(acq_times) != num_frames:
            print("Warning: acquisition_time not found or mismatch. Using frame index instead.")
            acq_times = [None] * num_frames
            
        # Determine appropriate figure sizes to match data aspect ratio
        base_width = 8.0
        aspect = height / width
        extra_h = 0.6 # extra vertical space for the date text
        
        fig, ax = plt.subplots(figsize=(base_width, base_width * aspect + extra_h), facecolor='w')
        ax.axis('off')
        
        # Initialize imshow object with correct shape
        im = ax.imshow(np.zeros((height, width, 4), dtype=np.float32))
        
        # Text overlay for date
        txt_date = ax.text(0.5, -0.05, "", transform=ax.transAxes, ha='center', va='top', 
                           color='black', fontsize=14, fontweight='bold', 
                           bbox=dict(facecolor='white', alpha=0.6, pad=3))

        fig.tight_layout(pad=0.1, rect=[0, 0.07, 1, 1])

        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=FPS, metadata=dict(artist='Antigravity'), bitrate=2500)
        
        print(f"Generating video: {output_mp4}")
        with writer.saving(fig, output_mp4, DPI):
            for i in range(num_frames):
                raw_ortho = ortho_ds[i, ...]
                
                # Adjust dimension order if (Bands, H, W) -> (H, W, Bands)
                if raw_ortho.shape[0] in [3, 4]:
                    raw_ortho = np.transpose(raw_ortho, (1, 2, 0))
                    
                rgba = np.zeros((raw_ortho.shape[0], raw_ortho.shape[1], 4), dtype=np.float32)
                rgba[..., :3] = raw_ortho[..., :3] / 255.0
                
                if raw_ortho.shape[-1] == 4:
                    user_alpha = raw_ortho[..., 3]
                    rgba[..., 3] = np.where(user_alpha > 0, 1.0, 0.0)
                else:
                    rgba[..., 3] = 1.0
                
                # Mask out NaN or invalid rgb pixels
                invalid_rgb_mask = np.isnan(rgba[..., 0])
                rgba[invalid_rgb_mask, 3] = 0.0
                rgba = np.nan_to_num(rgba, nan=0.0)
                    
                im.set_data(rgba)
                
                # Format acquisition date
                time_val = acq_times[i]
                if time_val is not None:
                    dt = datetime.fromtimestamp(time_val, tz=timezone.utc)
                    txt_date.set_text(dt.strftime('%Y-%m-%d'))
                else:
                    txt_date.set_text(f"Frame {i}")
                    
                writer.grab_frame()
                if i % 10 == 0 or i == num_frames - 1:
                    print(f"  Processed frame {i+1}/{num_frames}")

        plt.close(fig)
        print(f"Success! Video saved to {output_mp4}")

def main():
    # Hide the root tkinter window
    root = tk.Tk()
    root.withdraw()
    
    file_path = filedialog.askopenfilename(
        title="Select Native Stacked HDF5 File",
        filetypes=[("HDF5 files", "*.h5"), ("All files", "*.*")]
    )
    
    if not file_path:
        print("No file selected. Exiting.")
        return
        
    generate_video(file_path)

if __name__ == "__main__":
    main()
