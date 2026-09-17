import h5py
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tkinter as tk
from tkinter import filedialog
import numpy as np

def main():
    root = tk.Tk()
    root.withdraw()
    
    file_path = filedialog.askopenfilename(
        title="Select SpecTIR HDF5 File",
        filetypes=[("HDF5 files", "*.h5"), ("All files", "*.*")]
    )
    
    if not file_path:
        print("No file selected. Exiting.")
        return
        
    print(f"Opening {file_path}...")
    try:
        with h5py.File(file_path, 'r') as f:
            grp_path = "HDFEOS/GRIDS/SPECTIR/Data Fields"
            if grp_path not in f:
                print(f"Error: Group '{grp_path}' not found in the file.")
                return
                
            grp = f[grp_path]
            
            if 'ortho_visual' not in grp:
                print("Error: 'ortho_visual' dataset not found.")
                return
            if 'neighborhood_map_3x3' not in grp:
                print("Error: 'neighborhood_map_3x3' dataset not found.")
                return
            if 'heights_map_3x3' not in grp:
                print("Error: 'heights_map_3x3' dataset not found.")
                return
                
            print("Reading datasets...")
            ortho = grp['ortho_visual'][:]
            nm = grp['neighborhood_map_3x3'][:]
            heights_map = grp['heights_map_3x3'][:]
            
            # Squeeze extra dimensions (like the frame dimension)
            ortho = np.squeeze(ortho)
            nm = np.squeeze(nm)
            heights_map = np.squeeze(heights_map)
            
            # Mask the neighborhood map if nodata_mask is present
            if 'nodata_mask' in grp:
                nodata_mask = np.squeeze(grp['nodata_mask'][:])
                nm = np.ma.masked_where(nodata_mask, nm)
            
            # Ensure NaNs and Infs are unconditionally masked out
            nm = np.ma.masked_invalid(nm)
            
            # If ortho is channels-first, transpose it for matplotlib (e.g. 4, H, W -> H, W, 4)
            if ortho.ndim == 3 and ortho.shape[0] in [3, 4]:
                ortho = np.transpose(ortho, (1, 2, 0))
                
            print(f"Ortho visual shape for display: {ortho.shape}")
            print(f"Neighborhood map shape for display: {nm.shape}")
                
            fig, axes = plt.subplots(1, 2, figsize=(14, 7))
            
            # Plot ortho_visual
            axes[0].imshow(ortho)
            axes[0].set_title("Ortho Visual")
            axes[0].axis('off')
            
            # Plot neighborhood map
            # Calculate 1st and 99th percentiles for contrast stretch
            valid_nm = nm.compressed()
            vmin_val, vmax_val = np.percentile(valid_nm, [1, 99]) if valid_nm.size > 0 else (None, None)
            
            # Handle potential NaNs in the neighborhood map visually with cividis colormap
            im = axes[1].imshow(nm, cmap='cividis', vmin=vmin_val, vmax=vmax_val)
            axes[1].set_title("Neighborhood Map (3x3)")
            axes[1].axis('off')
            
            # Add a colorbar to the neighborhood map
            cbar = plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
            cbar.set_label("Spectral Complexity Volume")
            
            # State for interactivity
            state = {
                'patches': [],
                'fig_hist': None,
                'ax_hist': None
            }

            def onclick(event):
                # Ensure click is inside one of the axes
                if event.inaxes not in axes:
                    return
                
                # Get integer coordinates
                ix, iy = int(round(event.xdata)), int(round(event.ydata))
                
                # Ensure within bounds
                height, width = nm.shape
                if ix < 0 or ix >= width or iy < 0 or iy >= height:
                    return
                
                # Remove old patches
                for p in state['patches']:
                    p.remove()
                state['patches'].clear()
                
                # Draw 3x3 square (centered on ix, iy implies bottom-left is ix-1.5, iy-1.5)
                for ax in axes:
                    rect = patches.Rectangle((ix - 1.5, iy - 1.5), 3, 3, linewidth=1.5, edgecolor='red', facecolor='none')
                    ax.add_patch(rect)
                    state['patches'].append(rect)
                
                # Update main figure
                fig.canvas.draw_idle()
                
                # Extract heights
                heights = heights_map[iy, ix, :]
                
                # Plot in secondary window
                if state['fig_hist'] is None or not plt.fignum_exists(state['fig_hist'].number):
                    state['fig_hist'], state['ax_hist'] = plt.subplots(figsize=(6, 4))
                    state['fig_hist'].canvas.manager.set_window_title('Spectral Complexity Analysis')
                
                ax_h = state['ax_hist']
                ax_h.clear()
                
                # Plot heights vs endmember index
                endmembers = np.arange(1, len(heights) + 1)
                ax_h.plot(endmembers, heights, marker='o', linestyle='-', color='blue')
                ax_h.set_title(f"Heights at Pixel ({ix}, {iy})")
                ax_h.set_xlabel("Endmember Index")
                ax_h.set_ylabel("Orthogonal Projection Height")
                ax_h.grid(True)
                
                # Force draw
                state['fig_hist'].canvas.draw_idle()
                state['fig_hist'].show()
            
            fig.canvas.mpl_connect('button_press_event', onclick)
            
            plt.tight_layout()
            plt.show()
            
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()
