import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.ndimage import uniform_filter
import argparse
import os

COORDINATES_XY = (1281, 60)

def normalize_map(m):
    m_valid = m.compressed() if np.ma.is_masked(m) else m[~np.isnan(m) & (m != 0)]
    vmin, vmax = np.percentile(m_valid, [1, 99]) if m_valid.size > 0 else (0, 1)
    if vmax == vmin:
        vmax = vmin + 1e-5
    norm_m = np.clip((m - vmin) / (vmax - vmin), 0, 1)
    if np.ma.is_masked(norm_m):
        norm_m = norm_m.filled(0)
    else:
        norm_m[np.isnan(norm_m)] = 0
    return norm_m

def main():
    class Args: pass
    args = Args()
    args.x, args.y = COORDINATES_XY
    args.file = r'C:\satelliteImagery\hyspex\20250918_GE18\dual\HySpex_Ortho_Stack_20250918_ge18_55m_02.h5'
    args.outdir = os.path.join(os.path.dirname(args.file), os.path.splitext(os.path.basename(args.file))[0])

    os.makedirs(args.outdir, exist_ok=True)
    
    print(f"Reading {args.file}...")
    f = h5py.File(args.file, 'r')
    df = f['HDFEOS/GRIDS/HYSPEX/Data Fields']
    
    # 1. 3 separate heatmap images for 3x3, 15x15, 35x35 using spatial average filter
    tile_sizes = [3, 15, 35]
    print("Generating spatial average heatmaps...")
    smoothed_maps = {}
    for t in tile_sizes:
        nm_name = f'neighborhood_map_{t}x{t}'
        if nm_name in df:
            data = np.squeeze(df[nm_name][:])
            
            # Spatial average filter matching tile size (NaN-safe)
            valid_mask_data = ~np.isnan(data)
            data_zeroed = np.where(valid_mask_data, data, 0.0)
            smoothed_data = uniform_filter(data_zeroed, size=t)
            counts = uniform_filter(valid_mask_data.astype(float), size=t)
            with np.errstate(divide='ignore', invalid='ignore'):
                smoothed_data = np.where(counts > 0.0001, smoothed_data / counts, np.nan)
                
            smoothed_maps[t] = smoothed_data
            
            # Masking out 0s/nans for color scaling
            valid_mask = (smoothed_data != 0) & (~np.isnan(smoothed_data))
            if np.any(valid_mask):
                vmin, vmax = np.percentile(smoothed_data[valid_mask], [1, 99])
            else:
                vmin, vmax = 0, 1
                
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(smoothed_data, cmap='gist_rainbow', vmin=vmin, vmax=vmax)
            ax.set_title(f"Spectral Complexity Map ({t}x{t})")
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("ln(Complexity Volume)")
            
            # Add unfilled white circle around coordinate
            circle = patches.Circle((args.x, args.y), radius=7.5, linewidth=2, edgecolor='white', facecolor='none')
            ax.add_patch(circle)
            
            out_name = os.path.join(args.outdir, f"heatmap_{t}x{t}_X{args.x}_Y{args.y}.png")
            plt.savefig(out_name, dpi=300,pad_inches=0, bbox_inches='tight')
            plt.close()
            print(f"  Saved {out_name}")
            
    # 2. Visual chips alongside spectra plots
    print(f"Generating visual chips and spectra for (X={args.x}, Y={args.y})...")
    ortho_vis = np.squeeze(df['ortho_vis'][:]) # Shape: (4, H, W) or (3, H, W)
    if ortho_vis.shape[0] > 3:
        ortho_vis = ortho_vis[:3, :, :]
    
    # Normalize ortho_vis to RGB 0-1 for plotting
    rgb = np.stack([normalize_map(ortho_vis[0]), normalize_map(ortho_vis[1]), normalize_map(ortho_vis[2])], axis=-1)
    
    surface_ref_ds = df['surface_reflectance']
    surface_ref = np.squeeze(surface_ref_ds[:]) # Shape: (Bands, H, W)
    
    scale_factor = surface_ref_ds.attrs.get('Scale_Factor', [1.0])[0]
    fill_val = surface_ref_ds.fillvalue
    
    # Approximate wavelengths from 410 to 2500 nm (assuming linear for plotting, or use band index if actual wavelengths aren't stored)
    if 'wavelengths' in surface_ref_ds.attrs:
        wavelengths = surface_ref_ds.attrs['wavelengths']
    else:
        wavelengths = np.linspace(410, 2500, surface_ref.shape[0])
    
    COLOR_PALETTE = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow', 'brown', 'pink', 'lime', 'teal', 'navy']
    
    for t in tile_sizes:
        offset = t // 2
        y_start = max(0, args.y - offset)
        y_end = min(rgb.shape[0], args.y + offset + 1)
        x_start = max(0, args.x - offset)
        x_end = min(rgb.shape[1], args.x + offset + 1)
        
        chip = rgb[y_start:y_end, x_start:x_end, :]
        
        spectra_block = surface_ref[:, y_start:y_end, x_start:x_end]
        spectra_pixels = spectra_block.reshape(spectra_block.shape[0], -1).astype(np.float32)
        if fill_val is not None:
            spectra_pixels[spectra_pixels == fill_val] = np.nan
        spectra_pixels = spectra_pixels / scale_factor
        
        center_y_local = args.y - y_start
        center_x_local = args.x - x_start
        
        fig, (ax_img, ax_spec) = plt.subplots(1, 2, figsize=(12, 5))
        
        ax_img.imshow(chip)
        ax_img.set_title(f"Visual Chip ({t}x{t})")
        ax_img.axis('off')
        
        # Draw outlines for smaller nested windows
        if t > tile_sizes[0]:
            s = tile_sizes[0]
            rect = patches.Rectangle((center_x_local - s/2.0, center_y_local - s/2.0), s, s, 
                                     linewidth=1.5, edgecolor='blue', facecolor='none', alpha=0.9)
            ax_img.add_patch(rect)
        if len(tile_sizes) > 1 and t > tile_sizes[1]:
            s = tile_sizes[1]
            rect = patches.Rectangle((center_x_local - s/2.0, center_y_local - s/2.0), s, s, 
                                     linewidth=1.5, edgecolor='green', facecolor='none', alpha=0.9)
            ax_img.add_patch(rect)
        
        # Plot center pixel to red, rings to other colors
        max_d = max(max(center_y_local, spectra_block.shape[1] - 1 - center_y_local),
                    max(center_x_local, spectra_block.shape[2] - 1 - center_x_local))
                    
        for d in range(max_d, -1, -1):
            pixels_in_ring = []
            for y_idx in range(spectra_block.shape[1]):
                for x_idx in range(spectra_block.shape[2]):
                    if max(abs(y_idx - center_y_local), abs(x_idx - center_x_local)) == d:
                        pixels_in_ring.append(y_idx * spectra_block.shape[2] + x_idx)
            
            if not pixels_in_ring:
                continue
                
            if d == 0:
                color, lw, alpha, label = 'red', 2, 1.0, 'Center Pixel'
            elif (2*d + 1) <= tile_sizes[0]:
                color, lw, alpha, label = 'blue', 1, 0.7, f'{tile_sizes[0]}x{tile_sizes[0]} Neighborhood'
            elif len(tile_sizes) > 1 and (2*d + 1) <= tile_sizes[1]:
                color, lw, alpha, label = 'green', 1, 0.7, f'{tile_sizes[1]}x{tile_sizes[1]} Neighborhood'
            else:
                last_t = tile_sizes[-1] if len(tile_sizes) > 0 else 35
                color, lw, alpha, label = 'orange', 1, 0.7, f'{last_t}x{last_t} Neighborhood'
                
            lines = ax_spec.plot(wavelengths, spectra_pixels[:, pixels_in_ring], color=color, alpha=alpha, linewidth=lw)
            lines[0].set_label(label)
            
        ax_spec.set_title(f"Spectra ({t}x{t} neighborhood)")
        ax_spec.set_ylabel("Reflectance")
        ax_spec.set_xlabel("Wavelength (nm)")
        ax_spec.set_ylim(-0.05, 1.05)
        ax_spec.grid(True)
        handles, labels = ax_spec.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax_spec.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize='small')
        
        out_name = os.path.join(args.outdir, f"chip_spectra_{t}x{t}_X{args.x}_Y{args.y}.png")
        plt.tight_layout()
        plt.savefig(out_name,pad_inches=0, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved {out_name}")

    # 3. False Color Composite
    if len(tile_sizes) >= 3:
        print("Generating False Color Composite...")
        t_R, t_G, t_B = tile_sizes[0], tile_sizes[len(tile_sizes)//2], tile_sizes[-1]
        
        if t_R in smoothed_maps and t_G in smoothed_maps and t_B in smoothed_maps:
            def normalize_fc_map(m):
                m_valid = m.compressed() if np.ma.is_masked(m) else m[~np.isnan(m) & (m != 0)]
                vmin, vmax = np.percentile(m_valid, [2, 98]) if m_valid.size > 0 else (0, 1)
                if vmax == vmin:
                    vmax = vmin + 1e-5
                norm_m = np.clip((m - vmin) / (vmax - vmin), 0, 1)
                if np.ma.is_masked(norm_m):
                    norm_m = norm_m.filled(0)
                else:
                    norm_m[np.isnan(norm_m)] = 0
                return norm_m

            rgb_fc = np.stack([
                normalize_fc_map(smoothed_maps[t_R]), 
                normalize_fc_map(smoothed_maps[t_G]), 
                normalize_fc_map(smoothed_maps[t_B])
            ], axis=-1)

            fig_fc, axes_fc = plt.subplots(figsize=(10, 8))
            axes_fc.imshow(rgb_fc)
            axes_fc.set_title(f"Orthorectified False Color (R={t_R}, G={t_G}, B={t_B})")
            axes_fc.axis('off')
            
            out_name = os.path.join(args.outdir, f"false_color_composite_X{args.x}_Y{args.y}.png")
            plt.savefig(out_name, dpi=300, pad_inches=0, bbox_inches='tight')
            plt.close()
            print(f"  Saved {out_name}")

    # 4. Twin axis plot: Complexity vs Orthogonal Heights (Combined)
    print(f"Generating Twin-Axis plots for (X={args.x}, Y={args.y})...")
    
    fig, ax1 = plt.subplots(figsize=(10, 8))
    color1 = 'tab:blue'
    ax1.set_xlabel('Endmember Count')
    ax1.set_ylabel('Orthogonal Height', color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_yscale('log')
    ax1.grid(True)
    
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('Spectral Complexity (Volume)', color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_yscale('log')
    
    markers = ['o', 's', '^', 'D', 'v', '*']
    linestyles = ['-', '--', ':', '-.']
    
    for i, t in enumerate(tile_sizes):
        h_name = f'heights_map_{t}x{t}'
        if h_name in df:
            heights = np.squeeze(df[h_name][:])[args.y, args.x, :]
            endmembers = np.arange(1, len(heights) + 1)
            
            # Reconstruct cumulative volume
            volumes = np.cumprod(np.where(heights > 0, heights, 1.0))
            volumes[heights == 0] = 0
            
            m = markers[i % len(markers)]
            ls = linestyles[i % len(linestyles)]
            
            ax1.plot(endmembers[1:], heights[1:], color=color1, marker=m, linestyle=ls, linewidth=2, label=f'Height ({t}x{t})')
            ax2.plot(endmembers[2:], volumes[2:], color=color2, marker=m, linestyle=ls, linewidth=2, label=f'Complexity ({t}x{t})')
            
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='best')
    
    fig.tight_layout()
    out_name = os.path.join(args.outdir, f"twin_axis_combined_X{args.x}_Y{args.y}.png")
    plt.savefig(out_name, pad_inches=0, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved {out_name}")

    

    f.close()
    print("Done! All images saved to the output directory.")

if __name__ == "__main__":
    main()
