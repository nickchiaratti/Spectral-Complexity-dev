import h5py
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tkinter as tk
from tkinter import filedialog
import numpy as np
#import scipy.ndimage as ndimage
import cupy as cp
from cupyx.scipy import ndimage
import pandas as pd

# Configuration for False Color Complexity Composite

TILE_SIZES = [5,35,101]# 19,31,41,61,81,101]
NOISE_SOURCES = ['ssd', 'mlr', 'dct', 'mnf']

FALSE_COLOR_R_TILE = TILE_SIZES[0]
FALSE_COLOR_G_TILE = TILE_SIZES[len(TILE_SIZES)//2]
FALSE_COLOR_B_TILE = TILE_SIZES[-1] # Adjusted for HySpex default tiles (3, 5, 7, 9)

OVERLAY_GROUND_TRUTH=True
GROUND_TRUTH_FILE=r"C:\satelliteImagery\ground_truth\GPS_GE18_Landmine_UXO_locations.xlsx"
SPATIAL_AVERAGE=True
COMMON_HEATMAP_RANGE=False
COLOR_PALETTE = ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7', '#000000']

def main():
    root = tk.Tk()
    root.withdraw()
    
    file_path = filedialog.askopenfilename(
        title="Select HySpex HDF5 File",
        filetypes=[("HDF5 files", "*.h5"), ("All files", "*.*")]
    )
    
    if not file_path:
        print("No file selected. Exiting.")
        return
        
    print(f"Opening {file_path}...")
    try:
        with h5py.File(file_path, 'r') as f:
            grp_path = "HDFEOS/GRIDS/HYSPEX/Data Fields"
            if grp_path not in f:
                print(f"Error: Group '{grp_path}' not found in the file.")
                return
                
            grp = f[grp_path]
            
            if 'ortho_vis' not in grp:
                print("Error: 'ortho_vis' dataset not found.")
                return
                
            print("Reading datasets...")
            ortho_vis = grp['ortho_vis'][:]
            ortho_vis = np.squeeze(ortho_vis)
            
            surface_reflectance_ds = grp['surface_reflectance']
            wavelengths = surface_reflectance_ds.attrs['wavelengths']
            scale_factor = surface_reflectance_ds.attrs.get('Scale_Factor', [10000.0])[0]
            
            # ortho_vis from HySpex translator is (1, 4, L, S) or (4, L, S)
            if ortho_vis.ndim == 3 and ortho_vis.shape[0] in [3, 4]:
                ortho_vis = np.transpose(ortho_vis, (1, 2, 0))
            
            tile_sizes = TILE_SIZES
            # Use globally configured NOISE_SOURCES
            noise_sources = NOISE_SOURCES
            
            ortho_nms = {}
            ortho_heights = {}
            ortho_noise_heights = {}
            
            print("Loading metrics...")
            for t in tile_sizes:
                nm_name = f'neighborhood_map_{t}x{t}'
                h_name = f'heights_map_{t}x{t}'
                
                if nm_name in grp and h_name in grp:
                    nm_data = np.squeeze(grp[nm_name][:])
                    h_data = np.squeeze(grp[h_name][:])
                    
                    if SPATIAL_AVERAGE:
                        
                        valid_mask = np.isfinite(nm_data)
                        
                        # Use uniform filtering ignoring invalid pixels
                        d_data = cp.array(np.where(valid_mask, nm_data, 0.0))
                        d_weights = cp.array(valid_mask.astype(float))
                        sum_vals = ndimage.uniform_filter(d_data, size=t, mode='constant', cval=0.0).get()
                        sum_weights = ndimage.uniform_filter(d_weights, size=t, mode='constant', cval=0.0).get()
                        
                        with np.errstate(invalid='ignore', divide='ignore'):
                            # Only calculate average for originally valid pixels to prevent 
                            # the spatial filter from dilating the edges into the no-data background!
                            nm_data = np.where(valid_mask & (sum_weights > 0), sum_vals / sum_weights, np.nan)
                    
                    nm_data = np.ma.masked_invalid(nm_data)
                    
                    ortho_nms[t] = nm_data
                    ortho_heights[t] = h_data
                    
                    ortho_noise_heights[t] = {}
                    for src in noise_sources:
                        n_name = f'noise_heights_map_{src}_{t}x{t}'
                        if n_name in grp:
                            ndata = np.squeeze(grp[n_name][:])
                            ortho_noise_heights[t][src] = ndata
                        else:
                            ortho_noise_heights[t][src] = None
                else:
                    print(f"Warning: Data for tile size {t}x{t} not found.")
            
            loaded_tile_sizes = [t for t in tile_sizes if t in ortho_nms]
            
            if len(loaded_tile_sizes) == 0:
                print("No complexity metrics found in this file! Please run process_metrics.py on this file first.")
                return
                
            num_plots = len(loaded_tile_sizes) + 1
                
            # --- Figure 1: Orthorectified View ---
            fig_ortho, axes_ortho = plt.subplots(1, num_plots, figsize=(18, 10))
            if num_plots == 1:
                axes_ortho = [axes_ortho]
            fig_ortho.canvas.manager.set_window_title('Map Geometry (Orthorectified)')
            fig_ortho.tight_layout(pad=0.1)
            axes_ortho = np.array(axes_ortho).flatten()
            
            axes_ortho[0].imshow(ortho_vis)
            axes_ortho[0].set_title("Ortho Visual")
            axes_ortho[0].axis('off')
            
            # Calculate global vmin, vmax for orthorectified complexity maps if requested
            global_vmin_ortho, global_vmax_ortho = np.inf, -np.inf
            if COMMON_HEATMAP_RANGE:
                for t in loaded_tile_sizes:
                    nm = ortho_nms[t]
                    valid_nm = nm.compressed() if np.ma.is_masked(nm) else nm[~np.isnan(nm)]
                    if valid_nm.size > 0:
                        vmin_val, vmax_val = np.percentile(valid_nm, [1, 99])
                        global_vmin_ortho = min(global_vmin_ortho, vmin_val)
                        global_vmax_ortho = max(global_vmax_ortho, vmax_val)
                if np.isinf(global_vmin_ortho): global_vmin_ortho = None
                if np.isinf(global_vmax_ortho): global_vmax_ortho = None

            ax_tile_map_ortho = {}
            for idx, t in enumerate(loaded_tile_sizes):
                ax = axes_ortho[idx + 1]
                nm = ortho_nms[t]
                
                if COMMON_HEATMAP_RANGE:
                    plot_vmin = global_vmin_ortho
                    plot_vmax = global_vmax_ortho
                else:
                    valid_nm = nm.compressed() if np.ma.is_masked(nm) else nm[~np.isnan(nm)]
                    if valid_nm.size > 0:
                        plot_vmin, plot_vmax = np.percentile(valid_nm, [1, 99])
                    else:
                        plot_vmin, plot_vmax = None, None
                
                im = ax.imshow(nm, cmap='gist_rainbow', vmin=plot_vmin, vmax=plot_vmax)
                ax.set_title(f"Ortho Comp ({t}x{t})")
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("ln(Complexity Volume)")
                ax_tile_map_ortho[ax] = t
                ax.axis('off')
            
            fig_hist, ax_hist = plt.subplots(3, len(loaded_tile_sizes), figsize=(16, 16))
            if len(loaded_tile_sizes) == 1:
                ax_hist = ax_hist.reshape(3, 1)
            fig_hist.canvas.manager.set_window_title('Spectral Complexity Analysis')
            fig_hist.tight_layout(pad=0.1)

            ax_h_twins = [ax.twinx() for ax in ax_hist[1, :]]
            ax_h_twins2 = [ax.twinx() for ax in ax_hist[1, :]]
            state = {
                'patches_all': [],
                'fig_hist': fig_hist,
                'ax_hist': ax_hist,
                'ax_h_twins': ax_h_twins,
                'ax_h_twins2': ax_h_twins2
            }

            def onclick(event):
                if event.inaxes is None:
                    return
                
                is_ortho_main = event.inaxes in axes_ortho
                has_fc = 'axes_fc' in locals()
                is_fc_ortho = has_fc and event.inaxes == axes_fc
                has_gt = 'axes_gt' in locals()
                is_gt_ortho = has_gt and event.inaxes == axes_gt
                
                if not (is_ortho_main or is_fc_ortho or is_gt_ortho):
                    return
                    
                ix, iy = int(round(event.xdata)), int(round(event.ydata))
                map_x, map_y = ix, iy
                
                # Clear all previous markers
                for p in state['patches_all']:
                    p.remove()
                state['patches_all'].clear()
                
                # Draw Orthorectified Markers
                if 0 <= map_x < ortho_vis.shape[1] and 0 <= map_y < ortho_vis.shape[0]:
                    main_rect = patches.Rectangle((map_x - 7.5, map_y - 7.5), 15, 15, linewidth=1.5, edgecolor='red', facecolor='none')
                    axes_ortho[0].add_patch(main_rect)
                    state['patches_all'].append(main_rect)
                    for ax in axes_ortho[1:]:
                        if ax in ax_tile_map_ortho:
                            t = ax_tile_map_ortho[ax]
                            offset = t / 2.0
                            rect = patches.Rectangle((map_x - offset, map_y - offset), t, t, linewidth=1.5, edgecolor='red', facecolor='none')
                            ax.add_patch(rect)
                            state['patches_all'].append(rect)
                            
                    if has_fc:
                        crosshair_fc = axes_fc.plot(map_x, map_y, marker='+', color='white', markersize=12, markeredgewidth=2)[0]
                        state['patches_all'].append(crosshair_fc)
                    if has_gt:
                        crosshair_gt = axes_gt.plot(map_x, map_y, marker='+', color='white', markersize=12, markeredgewidth=2)[0]
                        state['patches_all'].append(crosshair_gt)
                
                fig_ortho.canvas.draw_idle()
                if has_fc:
                    fig_fc.canvas.draw_idle()
                if has_gt:
                    fig_gt.canvas.draw_idle()
                
                # Update Histograms
                if state['fig_hist'] is None or not plt.fignum_exists(state['fig_hist'].number):
                    state['fig_hist'], state['ax_hist'] = plt.subplots(3, len(loaded_tile_sizes), figsize=(16, 16))
                    if len(loaded_tile_sizes) == 1:
                        state['ax_hist'] = state['ax_hist'].reshape(3, 1)
                    state['ax_h_twins'] = [ax.twinx() for ax in state['ax_hist'][1, :]]
                    state['ax_h_twins2'] = [ax.twinx() for ax in state['ax_hist'][1, :]]
                    state['fig_hist'].canvas.manager.set_window_title('Spectral Complexity Analysis')
                    plt.tight_layout(pad=0.1)
                    state['fig_hist'].show()
                
                ax_img_row = state['ax_hist'][0, :]
                ax_hists_top = state['ax_hist'][1, :]
                ax_hists_bot = state['ax_hist'][2, :]
                
                global_min_h, global_max_h = np.inf, -np.inf
                global_min_v, global_max_v = np.inf, -np.inf
                global_min_v2, global_max_v2 = np.inf, -np.inf
                for t in loaded_tile_sizes:
                    if t in ortho_heights:
                        h_vals = ortho_heights[t][map_y, map_x, :]
                        if len(h_vals) > 3:
                            rel_h = h_vals[3:]
                            global_min_h = min(global_min_h, np.min(rel_h))
                            global_max_h = max(global_max_h, np.max(rel_h))
                            
                        if len(h_vals) > 0:
                            safe_h = np.where(h_vals <= 0, 1.0, h_vals)
                            v_vals = np.cumprod(safe_h)
                            global_min_v = min(global_min_v, np.min(v_vals))
                            global_max_v = max(global_max_v, np.max(v_vals))
                            
                            e_vals = np.arange(1, len(v_vals) + 1)
                            v_vals_adj = (v_vals ** (1.0 / e_vals)) / np.sqrt(e_vals)
                            global_min_v2 = min(global_min_v2, np.min(v_vals_adj))
                            global_max_v2 = max(global_max_v2, np.max(v_vals_adj))
                                    
                if not np.isinf(global_min_h) and not np.isinf(global_max_h):
                    pad_h = (global_max_h - global_min_h) * 0.05
                    if pad_h == 0: pad_h = 0.1
                    global_min_h -= pad_h
                    global_max_h += pad_h
                    
                if not np.isinf(global_min_v):
                    global_min_v = max(1e-50, global_min_v) * 0.5
                    global_max_v = max(1e-50, global_max_v) * 2.0
                    
                if not np.isinf(global_min_v2):
                    pad_v2 = (global_max_v2 - global_min_v2) * 0.05
                    if pad_v2 == 0: pad_v2 = 0.1
                    global_min_v2 -= pad_v2
                    global_max_v2 += pad_v2
                
                for idx, t in enumerate(loaded_tile_sizes):
                    ax_img = ax_img_row[idx]
                    ax_h = ax_hists_top[idx]
                    ax_s = ax_hists_bot[idx]
                    ax_h_twin = state['ax_h_twins'][idx]
                    ax_h_twin2 = state['ax_h_twins2'][idx]
                    
                    ax_img.clear()
                    ax_h.clear()
                    ax_s.clear()
                    ax_h_twin.clear()
                    ax_h_twin2.clear()
                    
                    if t in ortho_heights:
                        offset = t // 2
                        y_start = max(0, map_y - offset)
                        y_end = min(ortho_vis.shape[0], map_y + offset + 1)
                        x_start = max(0, map_x - offset)
                        x_end = min(ortho_vis.shape[1], map_x + offset + 1)
                        
                        # Note: plotting visual chunk from ortho_vis directly
                        img_block = ortho_vis[y_start:y_end, x_start:x_end]
                        ax_img.imshow(img_block)
                        ax_img.set_title(f"Visual ({t}x{t})")
                        ax_img.axis('off')
                        
                        heights = ortho_heights[t][map_y, map_x, :]
                        endmembers = np.arange(1, len(heights) + 1)
                        
                        safe_heights = np.where(heights <= 0, 1.0, heights)
                        volumes = np.cumprod(safe_heights)
                        deriv = np.gradient(heights)
                        
                        ax_h.plot(endmembers, heights, marker='o', linestyle='-', color='blue', label='Heights')
                        ax_h.plot(endmembers, -deriv, marker='^', linestyle=':', color='green', label='Derivative')
                        
                        # Plot Noise Heights based on the configured sources
                        colors = ['purple', 'orange', 'brown', 'cyan']
                        for idx_src, src in enumerate(NOISE_SOURCES):
                            if src in ortho_noise_heights[t] and ortho_noise_heights[t][src] is not None:
                                n_heights = ortho_noise_heights[t][src][map_y, map_x, :]
                                ax_h.plot(endmembers, n_heights, marker='s', linestyle='--', color=colors[idx_src % len(colors)], label=f'Noise ({src.upper()})')
                                
                        ax_h_twin.plot(endmembers, volumes, marker='s', linestyle='--', color='red', label='Volume')
                        
                        adj_volumes = (volumes ** (1.0 / endmembers)) / np.sqrt(endmembers)
                        ax_h_twin2.plot(endmembers, adj_volumes, marker='d', linestyle='-.', color='magenta', label='Adj Volume')
                        
                        ax_h_twin.set_ylabel("Volume")
                        ax_h_twin.yaxis.set_label_position("right")
                        ax_h_twin.yaxis.tick_right()
                        ax_h_twin.set_yscale('log')
                        
                        ax_h_twin2.set_ylabel("Adj Volume")
                        ax_h_twin2.yaxis.set_label_position("right")
                        ax_h_twin2.spines["right"].set_position(("axes", 1.15))
                        ax_h_twin2.yaxis.tick_right()
                        ax_h_twin2.set_yscale('linear')
                        
                        title_str = f"Tile {t}x{t} at Map ({map_x}, {map_y})"
                        ax_h.set_title(title_str)
                        ax_h.set_ylabel("Orthogonal Height")
                        ax_h.grid(True)
                        if not np.isinf(global_min_h):
                            ax_h.set_ylim(global_min_h, global_max_h)
                        if not np.isinf(global_min_v):
                            ax_h_twin.set_ylim(global_min_v, global_max_v)
                        if not np.isinf(global_min_v2):
                            ax_h_twin2.set_ylim(global_min_v2, global_max_v2)
                        if idx == 0:
                            h1, l1 = ax_h.get_legend_handles_labels()
                            h2, l2 = ax_h_twin.get_legend_handles_labels()
                            h3, l3 = ax_h_twin2.get_legend_handles_labels()
                            ax_h.legend(h1+h2+h3, l1+l2+l3, loc='upper right', fontsize='small')
                        if idx == len(loaded_tile_sizes) - 1:
                            ax_h.set_xlabel("Endmember Index")
                            
                        # Plot Spectra for the txt neighborhood
                        spectra_block = surface_reflectance_ds[0, :, y_start:y_end, x_start:x_end]
                        spectra_pixels = spectra_block.reshape(spectra_block.shape[0], -1)
                        spectra_pixels = spectra_pixels.astype(np.float32)
                        
                        fill_val = surface_reflectance_ds.fillvalue
                        if fill_val is not None:
                            spectra_pixels[spectra_pixels == fill_val] = np.nan
                            
                        spectra_pixels = spectra_pixels / scale_factor
                        
                        center_y_local = map_y - y_start
                        center_x_local = map_x - x_start
                        
                        max_d = max(
                            max(center_y_local, spectra_block.shape[1] - 1 - center_y_local),
                            max(center_x_local, spectra_block.shape[2] - 1 - center_x_local)
                        )
                        
                        for d in range(max_d, -1, -1):
                            pixels_in_ring = []
                            for y_idx in range(spectra_block.shape[1]):
                                for x_idx in range(spectra_block.shape[2]):
                                    if max(abs(y_idx - center_y_local), abs(x_idx - center_x_local)) == d:
                                        pixels_in_ring.append(y_idx * spectra_block.shape[2] + x_idx)
                            
                            if not pixels_in_ring:
                                continue
                                
                            if d == 0:
                                color = 'red'
                                lw = 2
                                alpha = 1.0
                                label = 'Center Pixel'
                            elif (2*d + 1) <= tile_sizes[0]:
                                color, lw, alpha, label = 'blue', 1, 0.7, f'{tile_sizes[0]}x{tile_sizes[0]} Neighborhood'
                            elif len(tile_sizes) > 1 and (2*d + 1) <= tile_sizes[1]:
                                color, lw, alpha, label = 'green', 1, 0.7, f'{tile_sizes[1]}x{tile_sizes[1]} Neighborhood'
                            else:
                                last_t = tile_sizes[-1] if len(tile_sizes) > 0 else 35
                                color, lw, alpha, label = 'orange', 1, 0.7, f'{last_t}x{last_t} Neighborhood'
                                
                            lines = ax_s.plot(wavelengths, spectra_pixels[:, pixels_in_ring], color=color, alpha=alpha, linewidth=lw)
                            lines[0].set_label(label)
                            
                        ax_s.set_title("Spectra")
                        ax_s.set_ylabel("Reflectance")
                        ax_s.set_xlabel("Wavelength (nm)")
                        ax_s.set_ylim(-0.05, 1.05)
                        ax_s.grid(True)
                        handles, labels = ax_s.get_legend_handles_labels()
                        by_label = dict(zip(labels, handles))
                        ax_s.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize='small')
                
                state['fig_hist'].tight_layout()
                state['fig_hist'].canvas.draw_idle()
            
            # --- Figure 2: False Color Composite ---
            if FALSE_COLOR_R_TILE in ortho_nms and FALSE_COLOR_G_TILE in ortho_nms and FALSE_COLOR_B_TILE in ortho_nms:
                def normalize_map(m):
                    m_valid = m.compressed() if np.ma.is_masked(m) else m[~np.isnan(m)]
                    vmin, vmax = np.percentile(m_valid, [2, 98]) if m_valid.size > 0 else (0, 1)
                    if vmax == vmin:
                        vmax = vmin + 1e-5
                    norm_m = np.clip((m - vmin) / (vmax - vmin), 0, 1)
                    if np.ma.is_masked(norm_m):
                        norm_m = norm_m.filled(0)
                    else:
                        norm_m[np.isnan(norm_m)] = 0
                    return norm_m
                    
                rgb_ortho = np.stack([
                    normalize_map(ortho_nms[FALSE_COLOR_R_TILE]), 
                    normalize_map(ortho_nms[FALSE_COLOR_G_TILE]), 
                    normalize_map(ortho_nms[FALSE_COLOR_B_TILE])
                ], axis=-1)
                
                fig_fc, axes_fc = plt.subplots(1, 1, figsize=(10, 8))
                fig_fc.canvas.manager.set_window_title('False Color Complexity View')
                fig_fc.tight_layout(pad=0.1)
                
                axes_fc.imshow(rgb_ortho)
                axes_fc.set_title(f"Orthorectified False Color (R={FALSE_COLOR_R_TILE}, G={FALSE_COLOR_G_TILE}, B={FALSE_COLOR_B_TILE})")
                axes_fc.axis('off')
                
                fig_fc.tight_layout()
                
                # --- Figure 3: Ground Truth Overlay ---
                if OVERLAY_GROUND_TRUTH:
                    geo_transform = surface_reflectance_ds.attrs.get('GeoTransform', None)
                    if geo_transform is not None:
                        try:
                            df = pd.read_excel(GROUND_TRUTH_FILE, header=None)
                            easting = pd.to_numeric(df.iloc[:, 2], errors='coerce').values
                            northing = pd.to_numeric(df.iloc[:, 1], errors='coerce').values
                            
                            valid_coords = ~np.isnan(easting) & ~np.isnan(northing)
                            easting = easting[valid_coords]
                            northing = northing[valid_coords]
                            
                            x_pixels = (easting - geo_transform[0]) / geo_transform[1]
                            y_pixels = (northing - geo_transform[3]) / geo_transform[5]
                            
                            valid = (x_pixels >= 0) & (x_pixels < rgb_ortho.shape[1]) & (y_pixels >= 0) & (y_pixels < rgb_ortho.shape[0])
                            x_pixels_valid = x_pixels[valid]
                            y_pixels_valid = y_pixels[valid]
                            
                            fig_gt, axes_gt = plt.subplots(1, 1, figsize=(10, 8))
                            fig_gt.canvas.manager.set_window_title('Ground Truth Overlay')
                            fig_gt.tight_layout(pad=0.1)
                            
                            axes_gt.imshow(rgb_ortho)
                            axes_gt.scatter(x_pixels_valid, y_pixels_valid, color='red', marker='x', s=50, label='Ground Truth')
                            axes_gt.set_title(f"Ground Truth Overlay (Found {np.sum(valid)} targets)")
                            axes_gt.axis('off')
                            axes_gt.legend()
                            fig_gt.tight_layout()
                            
                        except Exception as e:
                            print(f"Failed to overlay ground truth: {e}")
                    else:
                        print("GeoTransform not found in surface_reflectance, cannot overlay ground truth.")
            
            fig_ortho.canvas.mpl_connect('button_press_event', onclick)
            
            if 'fig_fc' in locals():
                fig_fc.canvas.mpl_connect('button_press_event', onclick)
            else:
                print(f"False color composite skipped: Could not find all of tiles {FALSE_COLOR_R_TILE}, {FALSE_COLOR_G_TILE}, {FALSE_COLOR_B_TILE}")
            
            if 'fig_gt' in locals():
                fig_gt.canvas.mpl_connect('button_press_event', onclick)
                
            plt.show()
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
