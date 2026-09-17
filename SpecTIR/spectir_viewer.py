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
                
            print("Reading datasets...")
            ortho = grp['ortho_visual'][:]
            ortho = np.squeeze(ortho)
            
            tile_sizes = [3, 5, 7, 9, 11, 13, 15]
            noise_sources = ['ssd', 'mlr', 'dct', 'mnf']
            
            nms = {}
            heights_maps = {}
            noise_heights_maps = {}
            
            nodata_mask = None
            if 'nodata_mask' in grp:
                nodata_mask = np.squeeze(grp['nodata_mask'][:])
            
            for t in tile_sizes:
                nm_name = f'neighborhood_map_{t}x{t}'
                h_name = f'heights_map_{t}x{t}'
                
                if nm_name in grp and h_name in grp:
                    nm_data = np.squeeze(grp[nm_name][:])
                    h_data = np.squeeze(grp[h_name][:])
                    
                    if nodata_mask is not None:
                        nm_data = np.ma.masked_where(nodata_mask, nm_data)
                    nm_data = np.ma.masked_invalid(nm_data)
                    
                    nms[t] = nm_data
                    heights_maps[t] = h_data
                    
                    noise_heights_maps[t] = {}
                    for src in noise_sources:
                        n_name = f'noise_heights_map_{src}_{t}x{t}'
                        if n_name in grp:
                            noise_heights_maps[t][src] = np.squeeze(grp[n_name][:])
                        else:
                            noise_heights_maps[t][src] = None
                else:
                    print(f"Warning: Data for tile size {t}x{t} not found.")
            
            if ortho.ndim == 3 and ortho.shape[0] in [3, 4]:
                ortho = np.transpose(ortho, (1, 2, 0))
                
            fig, axes = plt.subplots(1, 8, figsize=(18, 10))
            axes = axes.flatten()
            
            # Plot ortho_visual
            axes[0].imshow(ortho)
            axes[0].set_title("Ortho Visual")
            axes[0].axis('off')
            
            # Keep track of which axes have which tile size for drawing boxes
            ax_tile_map = {}
            
            # Plot neighborhood maps
            for idx, t in enumerate(tile_sizes):
                ax = axes[idx + 1]
                if t in nms:
                    nm = nms[t]
                    valid_nm = nm.compressed()
                    vmin_val, vmax_val = np.percentile(valid_nm, [1, 99]) if valid_nm.size > 0 else (None, None)
                    
                    im = ax.imshow(nm, cmap='cividis', vmin=vmin_val, vmax=vmax_val)
                    ax.set_title(f"Neighborhood Map ({t}x{t})")
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("Spectral Complexity Volume")
                    ax_tile_map[ax] = t
                else:
                    ax.set_title(f"Missing {t}x{t}")
                ax.axis('off')
            
            state = {
                'patches': [],
                'fig_hist': None,
                'ax_hist': None
            }

            def onclick(event):
                if event.inaxes not in axes:
                    return
                
                ix, iy = int(round(event.xdata)), int(round(event.ydata))
                
                # Assume all images share height/width from ortho
                height, width = ortho.shape[:2]
                if ix < 0 or ix >= width or iy < 0 or iy >= height:
                    return
                
                for p in state['patches']:
                    p.remove()
                state['patches'].clear()
                
                # Draw crosshair on ortho
                crosshair = axes[0].plot(ix, iy, marker='+', color='red', markersize=12, markeredgewidth=2)[0]
                state['patches'].append(crosshair)
                
                # Draw nxn square on corresponding neighborhood maps
                for ax in axes[1:]:
                    if ax in ax_tile_map:
                        t = ax_tile_map[ax]
                        offset = t / 2.0
                        rect = patches.Rectangle((ix - offset, iy - offset), t, t, linewidth=1.5, edgecolor='red', facecolor='none')
                        ax.add_patch(rect)
                        state['patches'].append(rect)
                
                fig.canvas.draw_idle()
                
                # Setup secondary window
                if state['fig_hist'] is None or not plt.fignum_exists(state['fig_hist'].number):
                    state['fig_hist'], state['ax_hist'] = plt.subplots(1, len(tile_sizes), figsize=(16, 8), sharey=True)
                    state['fig_hist'].canvas.manager.set_window_title('Spectral Complexity Analysis')
                    plt.tight_layout(pad=3.0)
                
                ax_hists = state['ax_hist']
                for idx, t in enumerate(tile_sizes):
                    ax_h = ax_hists[idx]
                    ax_h.clear()
                    
                    if t in heights_maps:
                        heights = heights_maps[t][iy, ix, :]
                        endmembers = np.arange(1, len(heights) + 1)
                        
                        ax_h.plot(endmembers, heights, marker='o', linestyle='-', color='blue', label='Heights')
                        
                        # Plot noise threshold lines
                        colors = {'ssd': 'orange', 'mlr': 'green', 'dct': 'red', 'mnf': 'purple'}
                        for src in noise_sources:
                            if src in noise_heights_maps[t] and noise_heights_maps[t][src] is not None:
                                nh = noise_heights_maps[t][src][iy, ix, :]
                                ax_h.plot(endmembers, nh, linestyle='--', color=colors[src], label=f'{src.upper()} Noise')
                        
                        ax_h.set_title(f"Tile {t}x{t} at ({ix}, {iy})")
                        ax_h.set_ylabel("Orthogonal Height")
                        #ax_h.set_yscale("log")
                        ax_h.grid(True)
                        if idx == 0:
                            ax_h.legend(loc='upper right', fontsize='small')
                        if idx == len(tile_sizes) - 1:
                            ax_h.set_xlabel("Endmember Index")
                
                state['fig_hist'].tight_layout()
                state['fig_hist'].canvas.draw_idle()
                state['fig_hist'].show()
            
            fig.canvas.mpl_connect('button_press_event', onclick)
            
            plt.tight_layout()
            plt.show()
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
