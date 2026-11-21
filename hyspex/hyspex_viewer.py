import os
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
import time
import spectral

# Complexity Volume calculation method:
# '3D': Threshold-Truncated 3D-Normalized Volume:
#       V_n^(3D) = V_3 + sum_{j=4}^n (prod_{m=1}^j h_m)^(3/j)
#       where n is the index of the last h_k prior to h_k < h_crit (stopping criteria)
# '3EM': Volume produced by 3 endmembers V_3 = P[2] = h_1 * h_2 * h_3
# 'maxVol': Original method choosing maximum value of the running products P
VOLUME_METHOD = '3D'  # Options: '3D', '3EM', 'maxVol'
SEPARABILITY_THRESHOLD = 0.01  # Minimum h_k threshold fraction: h_crit = SEPARABILITY_THRESHOLD * sqrt(B)
TILE_SIZES = [9,55,65]#[3,5,7,9,15,25,35,55,65,101,201]# 19,31,41,61,81,101]
NOISE_SOURCES = ['ssd', 'mlr', 'dct', 'mnf']

# Orientation configuration:
# When True, flightlines where height > width * ASPECT_RATIO_ROTATION_THRESHOLD (North/South flightlines)
# are rotated 90 degrees counterclockwise so East is at the top of the screen.
ROTATE_NS_FLIGHTLINES = True
ASPECT_RATIO_ROTATION_THRESHOLD = 1.2  # Trigger rotation when height / width > threshold

FALSE_COLOR_R_TILE = TILE_SIZES[0]
FALSE_COLOR_G_TILE = TILE_SIZES[len(TILE_SIZES)//2]
FALSE_COLOR_B_TILE = TILE_SIZES[-1] # Adjusted for HySpex default tiles (3, 5, 7, 9)
FALSE_COLOR_POWER = 2.5  # Nonlinear scaling exponent to amplify high values and make white pixels distinct (>1 enhances contrast of high values)

if VOLUME_METHOD =='3D':
    folder_name = f"V_k-{VOLUME_METHOD}-{int(SEPARABILITY_THRESHOLD*100)}-_tiles-R{TILE_SIZES[0]}_G{TILE_SIZES[len(TILE_SIZES)//2]}_B{TILE_SIZES[-1]}"
else:
    folder_name = f"V_k-{VOLUME_METHOD}_tiles-R{TILE_SIZES[0]}_G{TILE_SIZES[len(TILE_SIZES)//2]}_B{TILE_SIZES[-1]}"

SAVE_INITIAL_FIGURES = True
SAVE_FIGURES_DPI = 400

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
            B = len(wavelengths)
            h_crit = SEPARABILITY_THRESHOLD * np.sqrt(B)
            print(f"Spectral Bands B = {B}, Critical Threshold h_crit = {h_crit:.4f} (separability_threshold = {SEPARABILITY_THRESHOLD})")
            
            # ortho_vis from HySpex translator is (1, 4, L, S) or (4, L, S)
            if ortho_vis.ndim == 3 and ortho_vis.shape[0] in [3, 4]:
                ortho_vis = np.transpose(ortho_vis, (1, 2, 0))
            
            orig_H, orig_W = ortho_vis.shape[0], ortho_vis.shape[1]
            rotate_ccw = ROTATE_NS_FLIGHTLINES and (orig_H > orig_W * ASPECT_RATIO_ROTATION_THRESHOLD)
            if rotate_ccw:
                print(f"Detected North-South flightline ({orig_H} lines x {orig_W} samples, aspect ratio {orig_H/orig_W:.2f} > {ASPECT_RATIO_ROTATION_THRESHOLD}).")
                print("Rotating display 90 degrees counterclockwise (East at top of screen).")
                ortho_vis = np.rot90(ortho_vis, k=1, axes=(0, 1))
            else:
                print(f"Flightline dimensions: {orig_H} lines x {orig_W} samples (aspect ratio {orig_H/orig_W:.2f}). Standard display orientation.")
            
            tile_sizes = TILE_SIZES
            # Use globally configured NOISE_SOURCES
            noise_sources = NOISE_SOURCES
            
            ortho_nms = {}
            ortho_heights = {}
            ortho_noise_heights = {}
            
            print("Loading metrics...")
            for t in tile_sizes:
                h_name = f'heights_map_{t}x{t}'
                
                if h_name in grp:
                    h_data = np.squeeze(grp[h_name][:])
                    
                    if VOLUME_METHOD == '3D':
                        h_nz = np.maximum(h_data[..., 1:], 1e-12)
                        P_map = np.cumprod(h_nz, axis=-1)
                        if h_nz.shape[-1] >= 3:
                            k_steps = np.arange(1, h_nz.shape[-1] + 1, dtype=np.float32)
                            v_terms = P_map ** (3.0 / k_steps)
                            # Stopping criteria: accumulate higher dimensions j >= 4 only while h_k >= h_crit
                            is_above = (h_data[..., 1:] >= h_crit)
                            valid_signal = np.cumprod(is_above.astype(bool), axis=-1).astype(bool)
                            term_mask = valid_signal[..., 3:]
                            v_sum = np.sum(np.where(term_mask, v_terms[..., 3:], 0.0), axis=-1)
                            v_3d = P_map[..., 2] + v_sum
                            nm_data = np.log(np.maximum(v_3d, 1e-12))
                        else:
                            nm_data = np.full(h_data.shape[:-1], np.nan, dtype=np.float32)
                    elif VOLUME_METHOD == '3EM':
                        h_nz = np.maximum(h_data[..., 1:], 1e-12)
                        v_3em = h_nz[..., 0] * h_nz[..., 1] * h_nz[..., 2]
                        nm_data = np.log(np.maximum(v_3em, 1e-12))
                    elif VOLUME_METHOD == 'maxVol':
                        h_nz = np.maximum(h_data[..., 1:], 1e-12)
                        log_P = np.cumsum(np.log(h_nz), axis=-1)
                        if log_P.shape[-1] > 1:
                            nm_data = np.max(log_P[..., 1:], axis=-1)
                        else:
                            nm_data = np.full(h_data.shape[:-1], np.nan, dtype=np.float32)
                    else:
                        raise ValueError(f"Unknown VOLUME_METHOD: '{VOLUME_METHOD}'. Expected '3D', '3EM', or 'maxVol'.")
                    
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
                    
                    if rotate_ccw:
                        nm_data = np.rot90(nm_data, k=1, axes=(0, 1))
                        h_data = np.rot90(h_data, k=1, axes=(0, 1))
                    
                    ortho_nms[t] = nm_data
                    ortho_heights[t] = h_data
                    
                    ortho_noise_heights[t] = {}
                    for src in noise_sources:
                        n_name = f'noise_heights_map_{src}_{t}x{t}'
                        if n_name in grp:
                            ndata = np.squeeze(grp[n_name][:])
                            if rotate_ccw:
                                ndata = np.rot90(ndata, k=1, axes=(0, 1))
                            ortho_noise_heights[t][src] = ndata
                        else:
                            ortho_noise_heights[t][src] = None
                else:
                    print(f"Warning: Heights map for tile size {t}x{t} ('{h_name}') not found.")
            
            loaded_tile_sizes = [t for t in tile_sizes if t in ortho_nms]
            
            if len(loaded_tile_sizes) == 0:
                print("No complexity metrics found in this file! Please run process_metrics.py on this file first.")
                return
                
            num_plots = len(loaded_tile_sizes) + 1
                
            # --- Figure 1: Orthorectified View ---
            fig_ortho, axes_ortho = plt.subplots(num_plots, 1, figsize=(16, max(6, 2.5 * num_plots)))
            if num_plots == 1:
                axes_ortho = [axes_ortho]
            orientation_tag = " [East Top]" if rotate_ccw else ""
            fig_ortho.canvas.manager.set_window_title(f'Orthorectified - {VOLUME_METHOD}{orientation_tag}')
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
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label(f"ln(Complexity Volume - {VOLUME_METHOD})")
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
                'ax_h_twins2': ax_h_twins2,
                'fig_analysis': None,
                'axes_analysis': None,
                'fig_rx': None,
                'axes_rx': None
            }

            def onclick(event):
                if event.inaxes is None:
                    return
                
                is_ortho_main = event.inaxes in axes_ortho
                has_analysis = state.get('axes_analysis') is not None
                is_analysis = has_analysis and (event.inaxes in state['axes_analysis'])
                has_rx = state.get('axes_rx') is not None
                is_rx_ortho = has_rx and (event.inaxes in state['axes_rx'])
                has_fc = 'axes_fc' in locals()
                is_fc_ortho = has_fc and (event.inaxes in axes_fc)
                has_ratio = 'axes_ratio' in locals()
                is_ratio_ortho = has_ratio and (event.inaxes in axes_ratio)
                has_gt = 'axes_gt' in locals()
                is_gt_ortho = has_gt and (event.inaxes in axes_gt)
                
                if not (is_ortho_main or is_analysis or is_rx_ortho or is_fc_ortho or is_ratio_ortho or is_gt_ortho):
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
                            
                    square_width = 50 + 1
                    if has_analysis:
                        for ax in state['axes_analysis']:
                            rect_a = patches.Rectangle((map_x - ((square_width-1)//2), map_y - ((square_width-1)//2)), square_width, square_width, linewidth=1.5, edgecolor='white', facecolor='none')
                            ax.add_patch(rect_a)
                            state['patches_all'].append(rect_a)
                    if has_rx:
                        for ax in state['axes_rx']:
                            rect_rx = patches.Rectangle((map_x - ((square_width-1)//2), map_y - ((square_width-1)//2)), square_width, square_width, linewidth=1.5, edgecolor='white', facecolor='none')
                            ax.add_patch(rect_rx)
                            state['patches_all'].append(rect_rx)
                    if has_fc:
                        for ax in axes_fc:
                            rect_fc = patches.Rectangle((map_x - ((square_width-1)//2), map_y - ((square_width-1)//2)), square_width, square_width, linewidth=1.5, edgecolor='white', facecolor='none')
                            ax.add_patch(rect_fc)
                            state['patches_all'].append(rect_fc)
                    if has_ratio:
                        for ax in axes_ratio:
                            rect_ratio = patches.Rectangle((map_x - ((square_width-1)//2), map_y - ((square_width-1)//2)), square_width, square_width, linewidth=1.5, edgecolor='white', facecolor='none')
                            ax.add_patch(rect_ratio)
                            state['patches_all'].append(rect_ratio)
                    if has_gt:
                        for ax in axes_gt:
                            rect_gt = patches.Rectangle((map_x - ((square_width-1)//2), map_y - ((square_width-1)//2)), square_width, square_width, linewidth=1.5, edgecolor='white', facecolor='none')
                            ax.add_patch(rect_gt)
                            state['patches_all'].append(rect_gt)
                
                fig_ortho.canvas.draw_idle()
                if has_analysis and state.get('fig_analysis') is not None:
                    state['fig_analysis'].canvas.draw_idle()
                if has_rx and state.get('fig_rx') is not None:
                    state['fig_rx'].canvas.draw_idle()
                if has_fc:
                    fig_fc.canvas.draw_idle()
                if has_ratio:
                    fig_ratio.canvas.draw_idle()
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
                for t in loaded_tile_sizes:
                    if t in ortho_heights:
                        h_vals = ortho_heights[t][map_y, map_x, :]
                        if len(h_vals) > 3:
                            rel_h = h_vals[3:]
                            global_min_h = min(global_min_h, np.min(rel_h))
                            global_max_h = max(global_max_h, np.max(rel_h))
                            
                        if len(h_vals) > 3:
                            h_nz = h_vals[1:]
                            safe_hnz = np.maximum(h_nz, 1e-12)
                            P = np.cumprod(safe_hnz)
                            
                            if VOLUME_METHOD == '3D':
                                below_thresh = np.where(h_nz < h_crit)[0]
                                m_stop = int(below_thresh[0]) if len(below_thresh) > 0 else len(h_nz)
                                k_steps = np.arange(1, len(h_nz) + 1)
                                v_3d_terms = P ** (3.0 / k_steps)
                                v_cum_3d = np.full(len(h_vals), np.nan)
                                v_cum_3d[3] = P[2]
                                for idx_k in range(3, min(len(h_nz), m_stop)):
                                    v_cum_3d[idx_k + 1] = v_cum_3d[idx_k] + v_3d_terms[idx_k]
                                valid_v = v_cum_3d[~np.isnan(v_cum_3d)]
                                if len(valid_v) > 0:
                                    global_min_v = min(global_min_v, np.min(valid_v))
                                    global_max_v = max(global_max_v, np.max(valid_v))
                            elif VOLUME_METHOD == '3EM':
                                v3 = P[2]
                                global_min_v = min(global_min_v, v3 * 0.8)
                                global_max_v = max(global_max_v, v3 * 1.2)
                            elif VOLUME_METHOD == 'maxVol':
                                global_min_v = min(global_min_v, np.min(P[2:]))
                                global_max_v = max(global_max_v, np.max(P[2:]))
                                    
                if not np.isinf(global_min_h) and not np.isinf(global_max_h):
                    global_min_h = min(global_min_h, 0.0, h_crit * 0.8)
                    global_max_h = max(global_max_h, h_crit * 1.5)
                    pad_h = (global_max_h - global_min_h) * 0.05
                    if pad_h == 0: pad_h = 0.1
                    global_min_h -= pad_h
                    global_max_h += pad_h
                    
                if not np.isinf(global_min_v) and not np.isinf(global_max_v):
                    if VOLUME_METHOD in ['3D', '3EM']:
                        pad_v = (global_max_v - global_min_v) * 0.05
                        if pad_v == 0: pad_v = 0.1
                        global_min_v = max(0.0, global_min_v - pad_v)
                        global_max_v += pad_v
                    else:  # maxVol (log scale)
                        global_min_v = max(1e-50, global_min_v) * 0.5
                        global_max_v = max(1e-50, global_max_v) * 2.0
                
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
                        
                        # 1% Critical Threshold stopping rule: h_crit = 0.010 * sqrt(B)
                        # Non-zero orthogonal heights are at indices 1..len(heights)-1
                        h_nz = heights[1:]
                        below_thresh = np.where(h_nz < h_crit)[0]
                        if len(below_thresh) > 0:
                            m_stop = int(below_thresh[0])  # number of signal dimensions above threshold
                        else:
                            m_stop = len(h_nz)
                        stop_k = max(1, m_stop + 1)  # endmember index (material count)
                        
                        K = len(heights)
                        volumes = np.full(K, np.nan)
                        safe_hnz = np.maximum(h_nz, 1e-12)
                        P = np.cumprod(safe_hnz)
                        max_p_k = None
                        max_p_val = None

                        if VOLUME_METHOD == '3D':
                            # Threshold-Truncated 3D-Normalized Volume:
                            # Vol^(3D)_n = Vol_3 + sum_{j=4}^n (prod_{m=1}^j h_m)^(3/j)
                            # where n is the index of the last h_k prior to h_k < h_crit
                            if K > 3:
                                k_steps = np.arange(1, len(h_nz) + 1)
                                v_3d_terms = P ** (3.0 / k_steps)
                                volumes[3] = P[2]  # V_3 = h1 * h2 * h3
                                for idx_k in range(3, min(len(h_nz), m_stop)):
                                    volumes[idx_k + 1] = volumes[idx_k] + v_3d_terms[idx_k]
                            y_scale = 'linear'
                            vol_label = r'3D Vol $V_n^{(3\mathrm{D})}$'
                            vol_ylabel = r"3D Volume $V_n^{(3\mathrm{D})}$"
                            val_stop = volumes[stop_k - 1] if (stop_k <= len(volumes) and not np.isnan(volumes[stop_k - 1])) else (volumes[3] if len(volumes) > 3 else np.nan)
                            metric_str = f"V_3D={val_stop:.3f} (k*={stop_k})" if not np.isnan(val_stop) else "V_3D=N/A"

                        elif VOLUME_METHOD == '3EM':
                            # Volume produced by 3 endmembers: V_3 = P[2] = h1 * h2 * h3
                            if K > 3:
                                volumes[3:] = P[2]
                            y_scale = 'linear'
                            v3_val = P[2] if len(P) > 2 else np.nan
                            vol_label = rf'3EM Vol ($V_3 = {v3_val:.3f}$)' if not np.isnan(v3_val) else '3EM Vol'
                            vol_ylabel = r"3-Endmember Volume $V_3$"
                            metric_str = f"V_3EM={v3_val:.3f}" if not np.isnan(v3_val) else "V_3EM=N/A"

                        elif VOLUME_METHOD == 'maxVol':
                            # Original method choosing the maximum value of the products P
                            volumes[1:] = P
                            if len(P) > 2:
                                max_p_val = np.max(P[2:])
                                max_p_k = int(np.argmax(P[2:]) + 4)
                                vol_label = rf'Product $P$ (max={max_p_val:.2e})'
                                metric_str = f"maxVol={max_p_val:.3f} (k={max_p_k})"
                            else:
                                vol_label = 'Product P'
                                metric_str = "maxVol=N/A"
                            y_scale = 'log'
                            vol_ylabel = "Volume (Products P)"
                        
                        deriv = np.gradient(heights)
                        
                        ax_h.plot(endmembers, heights, marker='o', linestyle='-', color='blue', label='Heights')
                        ax_h.plot(endmembers, -deriv, marker='^', linestyle=':', color='green', label='Derivative')
                        ax_h.axhline(h_crit, color='black', linestyle='--', linewidth=1.5, label=f'1% Threshold ({h_crit:.3f})')
                        ax_h.axvline(stop_k, color='dimgray', linestyle=':', linewidth=1.5, label=f'Stop (k={stop_k})')
                        
                        # Plot Noise Heights based on the configured sources
                        colors = ['purple', 'orange', 'brown', 'cyan']
                        for idx_src, src in enumerate(NOISE_SOURCES):
                            if src in ortho_noise_heights[t] and ortho_noise_heights[t][src] is not None:
                                n_heights = ortho_noise_heights[t][src][map_y, map_x, :]
                                ax_h.plot(endmembers, n_heights, marker='s', linestyle='--', color=colors[idx_src % len(colors)], label=f'Noise ({src.upper()})')
                                
                        ax_h_twin.plot(endmembers, volumes, marker='s', linestyle='--', color='red', label=vol_label)
                        if VOLUME_METHOD == 'maxVol' and max_p_val is not None:
                            ax_h_twin.plot(max_p_k, max_p_val, marker='*', markersize=12, color='darkred', label=f'maxVol ({max_p_val:.2e})')
                        elif VOLUME_METHOD == '3EM' and len(P) > 2:
                            ax_h_twin.plot(4, P[2], marker='*', markersize=12, color='darkred', label=f'3EM ($V_3={P[2]:.3f}$)')
                        elif VOLUME_METHOD == '3D' and not np.isnan(val_stop):
                            ax_h_twin.plot(stop_k, val_stop, marker='*', markersize=12, color='darkred', label=rf'$V_n^{{(3\mathrm{{D}})}}={val_stop:.3f}$ ($k^*={stop_k}$)')
                        
                        ax_h_twin.set_ylabel(vol_ylabel, color='red')
                        ax_h_twin.tick_params(axis='y', labelcolor='red')
                        ax_h_twin.yaxis.set_label_position("right")
                        ax_h_twin.yaxis.tick_right()
                        ax_h_twin.set_yscale(y_scale)
                        
                        ax_h_twin2.set_visible(False)
                        
                        if rotate_ccw:
                            orig_center_y = map_x
                            orig_center_x = orig_W - 1 - map_y
                            title_str = f"Tile {t}x{t} at Rotated ({map_x}, {map_y}) [Orig ({orig_center_x}, {orig_center_y})] [Stop k={stop_k}, {metric_str}]"
                        else:
                            orig_center_y = map_y
                            orig_center_x = map_x
                            title_str = f"Tile {t}x{t} at Map ({map_x}, {map_y}) [Stop k={stop_k}, {metric_str}]"
                            
                        ax_h.set_title(title_str)
                        ax_h.set_ylabel("Orthogonal Height")
                        ax_h.grid(True)
                        if not np.isinf(global_min_h):
                            ax_h.set_ylim(global_min_h, global_max_h)
                        if not np.isinf(global_min_v):
                            ax_h_twin.set_ylim(global_min_v, global_max_v)
                        if idx == 0:
                            h1, l1 = ax_h.get_legend_handles_labels()
                            h2, l2 = ax_h_twin.get_legend_handles_labels()
                            ax_h.legend(h1+h2, l1+l2, loc='upper right', fontsize='small')
                        if idx == len(loaded_tile_sizes) - 1:
                            ax_h.set_xlabel("Endmember Index")
                            
                        # Plot Spectra for the txt neighborhood from original surface_reflectance
                        if rotate_ccw:
                            orig_y_start = max(0, orig_center_y - offset)
                            orig_y_end = min(orig_H, orig_center_y + offset + 1)
                            orig_x_start = max(0, orig_center_x - offset)
                            orig_x_end = min(orig_W, orig_center_x + offset + 1)
                            center_y_local = orig_center_y - orig_y_start
                            center_x_local = orig_center_x - orig_x_start
                        else:
                            orig_y_start = y_start
                            orig_y_end = y_end
                            orig_x_start = x_start
                            orig_x_end = x_end
                            center_y_local = map_y - y_start
                            center_x_local = map_x - x_start
                            
                        if surface_reflectance_ds.ndim == 4:
                            spectra_block = surface_reflectance_ds[0, :, orig_y_start:orig_y_end, orig_x_start:orig_x_end]
                        else:
                            spectra_block = surface_reflectance_ds[:, orig_y_start:orig_y_end, orig_x_start:orig_x_end]
                        spectra_pixels = spectra_block.reshape(spectra_block.shape[0], -1)
                        spectra_pixels = spectra_pixels.astype(np.float32)
                        
                        fill_val = surface_reflectance_ds.fillvalue
                        if fill_val is not None:
                            spectra_pixels[spectra_pixels == fill_val] = np.nan
                            
                        spectra_pixels = spectra_pixels / scale_factor
                        
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
            
            def normalize_map(m, power=FALSE_COLOR_POWER):
                if np.ma.is_masked(m):
                    m_valid = np.asarray(m.compressed())
                else:
                    m_valid = np.asarray(m[~np.isnan(m)])
                vmin, vmax = np.percentile(m_valid, [2, 98]) if m_valid.size > 0 else (0, 1)
                if vmax == vmin:
                    vmax = vmin + 1e-5
                norm_m = np.clip((m - vmin) / (vmax - vmin), 0, 1)
                if power != 1.0:
                    norm_m = np.power(norm_m, power)
                if np.ma.is_masked(norm_m):
                    norm_m = norm_m.filled(0)
                else:
                    norm_m[np.isnan(norm_m)] = 0
                return norm_m
                
            # --- Combined Analysis Window: 4 Vertical Subplots ---
            # 1. Ground Truth Extraction
            x_pixels_valid = np.array([])
            y_pixels_valid = np.array([])
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
                        
                        if rotate_ccw:
                            # Rotate ground truth pixel coords 90° CCW:
                            x_gt = y_pixels
                            y_gt = (orig_W - 1.0) - x_pixels
                        else:
                            x_gt = x_pixels
                            y_gt = y_pixels
                            
                        valid_gt = (x_gt >= 0) & (x_gt < ortho_vis.shape[1]) & (y_gt >= 0) & (y_gt < ortho_vis.shape[0])
                        x_pixels_valid = x_gt[valid_gt]
                        y_pixels_valid = y_gt[valid_gt]
                    except Exception as e:
                        print(f"Failed to overlay ground truth: {e}")
                else:
                    print("GeoTransform not found in surface_reflectance, cannot overlay ground truth.")

            # 2. False Color Composite
            rgb_ortho = None
            if FALSE_COLOR_R_TILE in ortho_nms and FALSE_COLOR_G_TILE in ortho_nms and FALSE_COLOR_B_TILE in ortho_nms:
                rgb_ortho = np.stack([
                    normalize_map(ortho_nms[FALSE_COLOR_R_TILE]), 
                    normalize_map(ortho_nms[FALSE_COLOR_G_TILE]), 
                    normalize_map(ortho_nms[FALSE_COLOR_B_TILE])
                ], axis=-1)
            elif len(ortho_nms) > 0:
                t_first = next(iter(ortho_nms))
                rgb_ortho = normalize_map(ortho_nms[t_first])

            # 3. Spectral Diversity Ratio ((h2 * h3) / h1^2)
            def compute_ratio_map(h_arr, t):
                if h_arr is None or h_arr.shape[-1] < 4:
                    return None
                h1 = np.maximum(h_arr[..., 1], 1e-12)
                h2 = np.maximum(h_arr[..., 2], 0.0)
                h3 = np.maximum(h_arr[..., 3], 0.0)
                ratio = (h2 * h3) / (h1 ** 2)
                valid_mask = np.isfinite(ratio) & (h_arr[..., 1] > 1e-6)
                
                if SPATIAL_AVERAGE:
                    try:
                        d_data = cp.array(np.where(valid_mask, ratio, 0.0))
                        d_weights = cp.array(valid_mask.astype(float))
                        sum_vals = ndimage.uniform_filter(d_data, size=t, mode='constant', cval=0.0).get()
                        sum_weights = ndimage.uniform_filter(d_weights, size=t, mode='constant', cval=0.0).get()
                    except Exception:
                        import scipy.ndimage as sc_ndimage
                        d_data = np.where(valid_mask, ratio, 0.0)
                        d_weights = valid_mask.astype(float)
                        sum_vals = sc_ndimage.uniform_filter(d_data, size=t, mode='constant', cval=0.0)
                        sum_weights = sc_ndimage.uniform_filter(d_weights, size=t, mode='constant', cval=0.0)
                        
                    with np.errstate(invalid='ignore', divide='ignore'):
                        ratio = np.where(valid_mask & (sum_weights > 0), sum_vals / sum_weights, np.nan)
                else:
                    ratio = np.where(valid_mask, ratio, np.nan)
                    
                return np.ma.masked_invalid(ratio)

            rgb_ratio = None
            #if FALSE_COLOR_R_TILE in ortho_heights and FALSE_COLOR_G_TILE in ortho_heights and FALSE_COLOR_B_TILE in ortho_heights:
            #    ratio_r = compute_ratio_map(ortho_heights[FALSE_COLOR_R_TILE], FALSE_COLOR_R_TILE)
            #    ratio_g = compute_ratio_map(ortho_heights[FALSE_COLOR_G_TILE], FALSE_COLOR_G_TILE)
            #    ratio_b = compute_ratio_map(ortho_heights[FALSE_COLOR_B_TILE], FALSE_COLOR_B_TILE)
            #    if ratio_r is not None and ratio_g is not None and ratio_b is not None:
            #        rgb_ratio = np.stack([
            #            normalize_map(ratio_r),
            #            normalize_map(ratio_g),
            #            normalize_map(ratio_b)
            #        ], axis=-1)
            #elif len(ortho_heights) > 0:
            #    t_first = next(iter(ortho_heights))
            #    r_single = compute_ratio_map(ortho_heights[t_first], t_first)
            #    if r_single is not None:
            #        rgb_ratio = normalize_map(r_single)
#
            # 4. RX Anomaly Detector (spectral.rx)
            def compute_rx_map(surface_reflectance_ds, orig_H, orig_W, rotate_ccw):
                print("Computing RX anomaly detector scores (spectral.rx)...")
                t0_rx = time.time()
                
                # Sample contiguous lines to estimate clean background covariance
                sample_y = np.linspace(orig_H // 6, orig_H - orig_H // 6, 6, dtype=int)
                train_list = [surface_reflectance_ds[0, :, y, :] for y in sample_y]
                train_arr = np.concatenate(train_list, axis=1) # (B, 6 * W)
                
                valid_bands = np.mean(train_arr > 0.0, axis=1) > 0.1
                if not np.any(valid_bands):
                    valid_bands = np.ones(surface_reflectance_ds.shape[1], dtype=bool)
                    
                train_valid = train_arr[valid_bands].T # (N, B_valid)
                mask = np.all(train_valid > 0.0, axis=-1)
                bg_pixels = train_valid[mask]
                
                if len(bg_pixels) > 10:
                    mean = np.mean(bg_pixels, axis=0)
                    cov = np.cov(bg_pixels, rowvar=False)
                    cov_reg = cov + (1e-4 * np.trace(cov) / cov.shape[0]) * np.eye(cov.shape[0])
                    bg_stats = spectral.GaussianStats(mean, cov_reg)
                else:
                    bg_stats = None
                    
                rx_map = np.full((orig_H, orig_W), np.nan, dtype=np.float32)
                chunk_size = 1000
                for y_start in range(0, orig_H, chunk_size):
                    y_end = min(orig_H, y_start + chunk_size)
                    c_raw = surface_reflectance_ds[0, :, y_start:y_end, :] # (B, chunk_lines, W)
                    c = np.transpose(c_raw[valid_bands], (1, 2, 0)) # (chunk_lines, W, B_valid)
                    valid_mask = np.all(c > 0.0, axis=-1)
                    if np.any(valid_mask):
                        rx_res = spectral.rx(c, background=bg_stats)
                        rx_map[y_start:y_end] = np.where(valid_mask, rx_res, np.nan)
                        
                if rotate_ccw:
                    rx_map = np.rot90(rx_map, k=1, axes=(0, 1))
                    
                print(f"RX map completed in {time.time()-t0_rx:.2f}s.")
                return rx_map
            
            if 'rxanomaly' in grp:
                print("Loading existing rxanomaly dataset from file...")
                rx_ds = grp['rxanomaly']
                if rx_ds.ndim == 3:
                    rx_map = rx_ds[0, :, :].astype(np.float32)
                else:
                    rx_map = rx_ds[:].astype(np.float32)
                if rotate_ccw:
                    rx_map = np.rot90(rx_map, k=1, axes=(0, 1))
            else:
                rx_map = compute_rx_map(surface_reflectance_ds, orig_H, orig_W, rotate_ccw)

            # --- Combined Figure: 3 Subplots Vertically Arranged ---
            fig_analysis, axes_analysis = plt.subplots(2, 1, figsize=(14, 8))
            fig_analysis.canvas.manager.set_window_title(f'Ortho Visual & Spectral Complexity{orientation_tag}')
            fig_analysis.tight_layout(pad=0.1)
            
            # Subplot 0: Ortho Visual with Ground Truth Overlay
            axes_analysis[0].imshow(ortho_vis)
            if len(x_pixels_valid) > 0:
                axes_analysis[0].scatter(x_pixels_valid, y_pixels_valid, color='red', marker='x', s=50, label='Ground Truth')
                axes_analysis[0].legend(loc='upper right')
                axes_analysis[0].set_title(f"1. Ortho Visual with Ground Truth Overlay{orientation_tag}")
            else:
                axes_analysis[0].set_title(f"1. Ortho Visual{orientation_tag}")
            axes_analysis[0].axis('off')
            
            # Subplot 1: False Color Complexity
            if rgb_ortho is not None:
                if rgb_ortho.ndim == 3:
                    axes_analysis[1].imshow(rgb_ortho)
                    axes_analysis[1].set_title(rf"2. False Color Complexity ($Vol_n$={VOLUME_METHOD}) (R={FALSE_COLOR_R_TILE}, G={FALSE_COLOR_G_TILE}, B={FALSE_COLOR_B_TILE}){orientation_tag}")
                else:
                    im_fc = axes_analysis[1].imshow(rgb_ortho, cmap='inferno')
                    fig_analysis.colorbar(im_fc, ax=axes_analysis[1], fraction=0.046, pad=0.04)
                    axes_analysis[1].set_title(f"2. False Color Spectral Complexity ({VOLUME_METHOD}){orientation_tag}")
                if len(x_pixels_valid) > 0:
                    axes_analysis[1].scatter(x_pixels_valid, y_pixels_valid, color='red', marker='x', s=50, linewidth=1.5, label='Ground Truth')
                    axes_analysis[1].legend(loc='upper right')
            else:
                axes_analysis[1].set_title(f"2. False Color Complexity (Not Available){orientation_tag}")
            axes_analysis[1].axis('off')
            
            # Subplot 2: Spectral Diversity Ratio
            if rgb_ratio is not None:
                if rgb_ratio.ndim == 3:
                    axes_analysis[2].imshow(rgb_ratio)
                    axes_analysis[2].set_title(rf"3. Spectral Diversity Ratio $(h_2 \cdot h_3 / h_1^2)$ (R={FALSE_COLOR_R_TILE}, G={FALSE_COLOR_G_TILE}, B={FALSE_COLOR_B_TILE}){orientation_tag}")
                else:
                    im_ratio = axes_analysis[2].imshow(rgb_ratio, cmap='inferno')
                    fig_analysis.colorbar(im_ratio, ax=axes_analysis[2], fraction=0.046, pad=0.04)
                    axes_analysis[2].set_title(rf"3. Spectral Diversity Ratio $(h_2 \cdot h_3 / h_1^2){orientation_tag}")
                if len(x_pixels_valid) > 0:
                    axes_analysis[2].scatter(x_pixels_valid, y_pixels_valid, color='red', marker='x', s=50, linewidth=1.5, label='Ground Truth')
                    axes_analysis[2].legend(loc='upper right')

                axes_analysis[2].axis('off')
            
            fig_analysis.tight_layout()
            state['fig_analysis'] = fig_analysis
            state['axes_analysis'] = axes_analysis
            
            # --- New Figure: RX Anomaly Detector & Ortho Visual ---
            fig_rx, axes_rx = plt.subplots(2, 1, figsize=(14, 8))
            fig_rx.canvas.manager.set_window_title(f'RX Anomaly Detector & Ortho Visual{orientation_tag}')
            fig_rx.tight_layout(pad=0.1)
            
            # Subplot 0: Ortho Visual with Ground Truth Overlay
            axes_rx[0].imshow(ortho_vis)
            if len(x_pixels_valid) > 0:
                axes_rx[0].scatter(x_pixels_valid, y_pixels_valid, color='red', marker='x', s=50, linewidth=1.5, label='Ground Truth')
                axes_rx[0].legend(loc='upper right')
                axes_rx[0].set_title(f"1. Ortho Visual with Ground Truth Overlay {orientation_tag}")
            else:
                axes_rx[0].set_title(f"1. Ortho Visual{orientation_tag}")
            axes_rx[0].axis('off')
            
            # Subplot 1: RX Anomaly Detector (spectral.rx) with black background for the plot area
            cmap_rx = plt.get_cmap('inferno').copy()
            cmap_rx.set_bad(color='black')
            
            axes_rx[1].set_facecolor('black')
            axes_rx[1].set_xticks([])
            axes_rx[1].set_yticks([])
            for s in axes_rx[1].spines.values():
                s.set_visible(False)
                
            rx_valid = rx_map[np.isfinite(rx_map)]
            if rx_valid.size > 0:
                vmin_rx, vmax_rx = np.percentile(rx_valid, [2, 98])
                im_rx = axes_rx[1].imshow(np.clip(rx_map, vmin_rx, vmax_rx), cmap=cmap_rx)
                cb_rx = fig_rx.colorbar(im_rx, ax=axes_rx[1], fraction=0.046, pad=0.04)
                cb_rx.set_label("RX Detector Score")
            else:
                im_rx = axes_rx[1].imshow(rx_map, cmap=cmap_rx)
            if len(x_pixels_valid) > 0:
                axes_rx[1].scatter(x_pixels_valid, y_pixels_valid, color='red', marker='x', s=50, linewidth=1.5, label='Ground Truth')
                axes_rx[1].legend(loc='upper right')
            axes_rx[1].set_title(f"2. RX Anomaly Detector (spectral.rx){orientation_tag}")
            
            fig_rx.tight_layout()
            state['fig_rx'] = fig_rx
            state['axes_rx'] = axes_rx
            
            fig_ortho.canvas.mpl_connect('button_press_event', onclick)
            if 'fig_analysis' in locals():
                fig_analysis.canvas.mpl_connect('button_press_event', onclick)
            if 'fig_rx' in locals():
                fig_rx.canvas.mpl_connect('button_press_event', onclick)
            
            # Save initial display figures
            if SAVE_INITIAL_FIGURES:
                h5_dir = os.path.dirname(os.path.abspath(file_path))
                base_name = os.path.splitext(os.path.basename(file_path))[0]
                save_dir = os.path.join(h5_dir, f"{base_name}_{folder_name}")
                os.makedirs(save_dir, exist_ok=True)
                print(f"\nSaving initial display figures to: {save_dir}")

                if 'fig_ortho' in locals() and fig_ortho is not None:
                    ortho_path = os.path.join(save_dir, f"{base_name}_map_geometry.png")
                    print(f"  Saving Map Geometry figure: {ortho_path}")
                    fig_ortho.savefig(ortho_path, dpi=SAVE_FIGURES_DPI, bbox_inches='tight')

                if 'fig_analysis' in locals() and fig_analysis is not None:
                    analysis_path = os.path.join(save_dir, f"{base_name}_comparative_analysis.png")
                    print(f"  Saving Comparative Analysis figure: {analysis_path}")
                    fig_analysis.savefig(analysis_path, dpi=SAVE_FIGURES_DPI, bbox_inches='tight')

                if 'fig_rx' in locals() and fig_rx is not None:
                    rx_path = os.path.join(save_dir, f"{base_name}_rx_anomaly_detector.png")
                    print(f"  Saving RX Anomaly Detector figure: {rx_path}")
                    fig_rx.savefig(rx_path, dpi=SAVE_FIGURES_DPI, bbox_inches='tight')

                print("Initial display figures saved successfully.\n")

            plt.show()
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
