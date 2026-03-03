'''
This script correlates volume values between Landsat and Tanager Tiles
Caveats: does not load all the pixel mask data or apply them
'''

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
import SpecComplex as sc
from skimage import exposure
from datetime import datetime, timezone
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm is not installed
    def tqdm(iterable, **kwargs): return iterable

# --- Configuration ---
L_PATH = "C:/satelliteImagery/LANDSAT/Tait/LANDSAT_Stack_Tait_HDFEOS.h5"
T_PATH = "C:/satelliteImagery/Tanager/Tait/Tanager_Stack_Tait_HDFEOS.h5"
localization = 'datasetMean' #'general'
normalization = 'bandCount' #'bandCount'

L_FRAME = 81  # 0-indexed Landsat frame
T_FRAME = 3   # 0-indexed Tanager frame (2025-09-19)
TILE_SIZE = 3
NUM_ENDMEMBERS = 7

def percentile_normalize_array(arr, low=2, high=98):
    """Normalizes array for RGB display."""
    if np.all(np.isnan(arr)): return np.zeros_like(arr)
    p_low, p_high = np.nanpercentile(arr, (low, high))
    if p_low == p_high: return np.zeros_like(arr)
    return exposure.rescale_intensity(arr, in_range=(p_low, p_high), out_range=(0, 1)).clip(0, 1)

def get_volume_curve(valid_data, num_em):
    """
    Calculates the volume curve localized by the dataset mean.
    Returns an array of length `num_em` containing volumes.
    """
    endmembers, _ = sc.maximumDistance(valid_data, num_em)
    mean_dataset = valid_data.mean(axis=0) # Mean across pixels
    if normalization == 'bandCount':
        endmembers = endmembers/np.sqrt(endmembers.shape[0])
    
    vol_curve = np.zeros(num_em)
    for i in range(2, num_em + 1):
        if localization == 'general':
            gram = sc.calcGramLocal(endmembers[:, 0:i], np.zeros(endmembers.shape[0]))
        elif localization == 'datasetMean':
            gram = sc.calcGramLocal(endmembers[:, 0:i], mean_dataset)
        vol_curve[i-1] = np.sqrt(np.abs(np.linalg.det(gram)))
        
    return vol_curve

def main():
    print(f"Loading Landsat Frame {L_FRAME} and Tanager Frame {T_FRAME}...")
    
    try:
        with h5py.File(L_PATH, 'r') as h5_l, h5py.File(T_PATH, 'r') as h5_t:
            sr_l_dset = h5_l['/HDFEOS/GRIDS/LANDSAT/Data Fields/surface_reflectance']
            sr_t_dset = h5_t['/HDFEOS/GRIDS/TANAGER/Data Fields/surface_reflectance']
            
            # Extract Acquisition Times
            acq_times_l = sr_l_dset.attrs.get('acquisition_time')
            acq_times_t = sr_t_dset.attrs.get('acquisition_time')
            
            date_l_str = "Unknown Date"
            date_l_file = "UnknownDate"
            if acq_times_l is not None:
                dt_l = datetime.fromtimestamp(acq_times_l[L_FRAME], tz=timezone.utc)
                date_l_str = dt_l.strftime('%Y-%m-%d %H:%M UTC')
                date_l_file = dt_l.strftime('%Y-%m-%d')
            
            date_t_str = "Unknown Date"
            date_t_file = "UnknownDate"
            if acq_times_t is not None:
                dt_t = datetime.fromtimestamp(acq_times_t[T_FRAME], tz=timezone.utc)
                date_t_str = dt_t.strftime('%Y-%m-%d %H:%M UTC')
                date_t_file = dt_t.strftime('%Y-%m-%d')
            
            # Extract Landsat Frame
            frame_l = sr_l_dset[L_FRAME, ...]
            bands_l, h, w = frame_l.shape
            
            # Extract Tanager Frame (masking invalid wavelengths)
            gw_attr = sr_t_dset.attrs.get("all_good_wavelengths")
            if gw_attr is not None:
                gw_mask = gw_attr.astype(bool)[T_FRAME]
                frame_t = sr_t_dset[T_FRAME, gw_mask, ...]
            else:
                frame_t = sr_t_dset[T_FRAME, ...]
            bands_t = frame_t.shape[0]
            
            # Extract Tanager Visual
            try:
                vis_t_dset = h5_t['/HDFEOS/GRIDS/TANAGER/Data Fields/ortho_visual']
                frame_t_vis = vis_t_dset[T_FRAME, ...]
            except KeyError:
                print("Warning: ortho_visual not found in Tanager HDF5.")
                frame_t_vis = None
            
            # Ensure spatial dimensions match
            if (h, w) != (sr_t_dset.shape[2], sr_t_dset.shape[3]):
                print("Warning: Spatial dimensions do not match exactly! Cropping to minimum overlap.")
                h = min(h, sr_t_dset.shape[2])
                w = min(w, sr_t_dset.shape[3])
                frame_l = frame_l[:, :h, :w]
                frame_t = frame_t[:, :h, :w]
                if frame_t_vis is not None:
                    frame_t_vis = frame_t_vis[:, :h, :w]
                
    except FileNotFoundError as e:
        print(f"Error accessing files: {e}")
        return

    print(f"Frame dimensions: {h}x{w}")
    print(f"Calculating volume curves and Pearson correlation for {TILE_SIZE}x{TILE_SIZE} tiles...")
    
    # Initialize map filled with NaNs
    corr_map = np.full((h, w), np.nan, dtype=np.float32)
    vol_map_l = np.full((h, w), np.nan, dtype=np.float32)
    vol_map_t = np.full((h, w), np.nan, dtype=np.float32)
    
    # Iterate through grid
    for y in tqdm(range(0, h, TILE_SIZE), desc="Processing Rows"):
        for x in range(0, w, TILE_SIZE):
            y_end = min(y + TILE_SIZE, h)
            x_end = min(x + TILE_SIZE, w)
            
            # Extract spatial tiles and flatten spatial dimensions [Pixels, Bands]
            tile_l = frame_l[:, y:y_end, x:x_end].transpose(1, 2, 0).reshape(-1, bands_l)
            tile_t = frame_t[:, y:y_end, x:x_end].transpose(1, 2, 0).reshape(-1, bands_t)
            
            # Filter NaNs
            valid_l = tile_l[~np.isnan(tile_l).any(axis=1)]
            valid_t = tile_t[~np.isnan(tile_t).any(axis=1)]
            
            # Ensure enough valid pixels exist to find the requested number of endmembers
            if valid_l.shape[0] >= NUM_ENDMEMBERS and valid_t.shape[0] >= NUM_ENDMEMBERS:
                try:
                    vol_l = get_volume_curve(valid_l, NUM_ENDMEMBERS)
                    vol_t = get_volume_curve(valid_t, NUM_ENDMEMBERS)
                    
                    # Store max volume for the volume maps
                    vol_map_l[y:y_end, x:x_end] = np.max(vol_l[1:])
                    vol_map_t[y:y_end, x:x_end] = np.max(vol_t[1:])

                    # Create a mask where BOTH curves are greater than 0
                    nonzero_mask = (vol_l[1:] > 1e-12) | (vol_t[1:] > 1e-12)

                    # Only calculate if we have enough non-zero points left (e.g., at least 3)
                    if np.sum(nonzero_mask) >= 3:
                        r, _ = pearsonr(vol_l[1:][nonzero_mask], vol_t[1:][nonzero_mask])
                        # Fill the entire 3x3 block in the map with the calculated correlation
                        corr_map[y:y_end, x:x_end] = r
                except Exception as e:
                    # Catch and skip instances where singular matrices or math domain errors occur
                    pass

    # --- Visualization ---
    print("Generating 2D Correlation Map...")
    plt.figure(figsize=(11, 8))
    
    # Using a diverging colormap where 1 = Blue (High Positive), -1 = Red (High Negative)
    im = plt.imshow(corr_map, cmap='viridis')
    
    cbar = plt.colorbar(im)
    cbar.set_label("Pearson Correlation Coefficient (r)", fontsize=12)
    
    plt.title(f"Volume Curve Pearson Correlation Map\nLandsat ({date_l_str}) vs Tanager ({date_t_str}) | Tile Size: {TILE_SIZE}x{TILE_SIZE}", fontsize=14)
    plt.xlabel("X Pixel Coordinate")
    plt.ylabel("Y Pixel Coordinate")
    
    # Calculate and display general statistics in the corner
    valid_corrs = corr_map[~np.isnan(corr_map)]
    if valid_corrs.size > 0:
        mean_r = np.mean(valid_corrs)
        median_r = np.median(valid_corrs)
        tile_count = valid_corrs.size // (TILE_SIZE**2)
        
        stats_text = (f"Tiles Analyzed: {tile_count}\n"
                      f"Mean r: {mean_r:.5f}\n"
                      f"Median r: {median_r:.5f}")
        plt.figtext(0.13, 0.13, stats_text, ha="left", fontsize=10, 
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
    
    # Setup Output Directory
    out_dir = "C:/satelliteImagery/MultiSensor_Analysis/"
    os.makedirs(out_dir, exist_ok=True)
    
    # --- Save Volume Maps to CSV ---
    print("Saving volume maps to CSV...")
    csv_l_path = os.path.join(out_dir, f"Landsat_{date_l_file}_Loc-{localization}_Norm-{normalization}.csv")
    pd.DataFrame(vol_map_l).to_csv(csv_l_path, index=False, header=False)
    print(f"Saved Landsat volume map to: {csv_l_path}")

    csv_t_path = os.path.join(out_dir, f"Tanager_{date_t_file}_Loc-{localization}_Norm-{normalization}.csv")
    pd.DataFrame(vol_map_t).to_csv(csv_t_path, index=False, header=False)
    print(f"Saved Tanager volume map to: {csv_t_path}")

    # Save the figure
    out_path = os.path.join(out_dir, f"CorrelationMap_L80_T3_EM{NUM_ENDMEMBERS}_gram-{localization}_norm-{normalization}.png")
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved figure to: {out_path}")
    
    # --- RGB Visualization ---
    print("Generating True Color Representation Figure...")
    fig_rgb, axes_rgb = plt.subplots(1, 2, figsize=(16, 8))
    
    # Landsat RGB (Assuming typical indices: R=3, G=2, B=1)
    r = percentile_normalize_array(frame_l[3, ...])
    g = percentile_normalize_array(frame_l[2, ...])
    b = percentile_normalize_array(frame_l[1, ...])
    rgb_l = np.nan_to_num(np.dstack((r, g, b)), nan=0.0)
    
    axes_rgb[0].imshow(rgb_l)
    axes_rgb[0].set_title(f"Landsat True Color (Frame {L_FRAME})", fontsize=14)
    axes_rgb[0].axis('off')
    
    # Tanager RGB
    if frame_t_vis is not None:
        rgb_t = np.transpose(frame_t_vis[:3, ...], (1, 2, 0)) # Drop alpha if present
        axes_rgb[1].imshow(rgb_t)
        axes_rgb[1].set_title(f"Tanager Ortho Visual (Frame {T_FRAME})", fontsize=14)
        axes_rgb[1].axis('off')
    else:
        axes_rgb[1].text(0.5, 0.5, "Ortho Visual Not Available", ha='center', va='center')
        axes_rgb[1].axis('off')
        
    fig_rgb.suptitle(f"True Color Comparison | Landsat ({date_l_str}) vs Tanager ({date_t_str})", fontsize=16)
    
    out_path_rgb = os.path.join(out_dir, f"TrueColor_L{L_FRAME}_T{T_FRAME}.png")
    fig_rgb.tight_layout()
    fig_rgb.savefig(out_path_rgb, dpi=300, bbox_inches='tight')
    print(f"Saved true color figure to: {out_path_rgb}")

    # --- Volume Map Visualization ---
    print("Generating Volume Maps Figure...")
    fig_vol, axes_vol = plt.subplots(1, 2, figsize=(16, 8))
    
    # Landsat Volume
    im_vl = axes_vol[0].imshow(vol_map_l, cmap='viridis', interpolation='nearest')
    axes_vol[0].set_title(f"Landsat Max Volume (Frame {L_FRAME})", fontsize=14)
    axes_vol[0].axis('off')
    cbar_l = fig_vol.colorbar(im_vl, ax=axes_vol[0], fraction=0.046, pad=0.04)
    cbar_l.set_label("Max Volume", fontsize=12)
    
    # Tanager Volume
    im_vt = axes_vol[1].imshow(vol_map_t, cmap='viridis', interpolation='nearest')
    axes_vol[1].set_title(f"Tanager Max Volume (Frame {T_FRAME})", fontsize=14)
    axes_vol[1].axis('off')
    cbar_t = fig_vol.colorbar(im_vt, ax=axes_vol[1], fraction=0.046, pad=0.04)
    cbar_t.set_label("Max Volume", fontsize=12)
    
    fig_vol.suptitle(f"Volume Tiled Maps Comparison | Landsat ({date_l_str}) vs Tanager ({date_t_str}) | Tile Size: {TILE_SIZE}x{TILE_SIZE}", fontsize=16)
    
    out_path_vol = os.path.join(out_dir, f"VolumeMaps_L{L_FRAME}_T{T_FRAME}_EM{NUM_ENDMEMBERS}_gram-{localization}_norm-{normalization}.png")
    fig_vol.tight_layout()
    fig_vol.savefig(out_path_vol, dpi=300, bbox_inches='tight')
    print(f"Saved volume maps figure to: {out_path_vol}")

    # --- Scatter Plot Visualization ---
    print("Generating Volume Scatter Plot Figure...")
    
    l_flat = vol_map_l.flatten()
    t_flat = vol_map_t.flatten()
    
    # Create a mask to remove NaNs (NoData pixels) or exact zeros from BOTH arrays
    valid_scatter_mask = (~np.isnan(l_flat)) & (~np.isnan(t_flat)) & (l_flat > 0) & (t_flat > 0)
    l_valid = l_flat[valid_scatter_mask]
    t_valid = t_flat[valid_scatter_mask]
    
    if len(l_valid) > 0:
        # Calculate Scale Factors
        optimal_scale_factor = np.sum(l_valid * t_valid) / np.sum(l_valid**2)
        ratios = t_valid / l_valid
        median_ratio = np.median(ratios)
        mean_ratio = np.mean(ratios)
        
        
        # Calculate Global Frame Correlations
        global_pearson, _ = pearsonr(l_valid, t_valid)
        global_spearman, _ = spearmanr(l_valid, t_valid)
        log_l = np.log10(l_valid)
        log_t = np.log10(t_valid)
        log_pearson, _ = pearsonr(log_l, log_t)
        print(f"Log-Pearson (Exponential Fit): {log_pearson:.4f}")
        log_spearman, _ = spearmanr(log_l, log_t)
        print(f"Log-Spearman (Exponential Fit): {log_spearman:.4f}")

        slope_b, intercept_loga = np.polyfit(log_l, log_t, 1)
        print(f"Exponential Scaling Factor (b): {slope_b:.4f}")
        print(f"Relationship: Tanager ≈ {10**intercept_loga:.4f} * (Landsat ^ {slope_b:.4f})")
        
        fig_scatter, ax_scatter = plt.subplots(figsize=(9, 8))
        
        # Plot raw data points
        ax_scatter.scatter(l_valid, t_valid, alpha=0.3, s=10, label='Tile Volumes')
        
        # Plot the scaling lines
        max_l = np.max(l_valid)
        line_x = np.array([0, max_l])
        
        ax_scatter.plot(line_x, line_x * optimal_scale_factor, color='red', linewidth=2, 
                 label=f'Least Squares Fit (x{optimal_scale_factor:.2f})')
        ax_scatter.plot(line_x, line_x * median_ratio, color='green', linewidth=2, linestyle='--', 
                 label=f'Median Ratio Fit (x{median_ratio:.2f})')

        ax_scatter.set_title(f"Landsat vs Tanager Volume Map Correlation\nLandsat ({date_l_str}) vs Tanager ({date_t_str})\nLoc: {localization} | Norm: {normalization}", fontsize=14)
        ax_scatter.set_xlabel("Landsat Volume")
        ax_scatter.set_ylabel("Tanager Volume")
        ax_scatter.grid(True, alpha=0.3)
        ax_scatter.legend()
        
        # Add text box with metrics
        stats_text_scatter = (f"Tiles Analyzed: {len(l_valid)}\n"
                              f"Pearson r: {global_pearson:.4f}\n"
                              f"Spearman r: {global_spearman:.4f}\n"
                              f"Log Pearson r: {log_pearson:.4f}\n"
                              f"Log Spearman r: {log_spearman:.4f}\n"
                              f"Least Squares: {optimal_scale_factor:.4f}\n"
                              f"Median Ratio: {median_ratio:.4f}\n"
                              f"Mean Ratio: {mean_ratio:.4f}")
        
        fig_scatter.text(0.15, 0.75, stats_text_scatter, ha="right", fontsize=10, 
                         bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
        
        out_path_scatter = os.path.join(out_dir, f"VolumeScatter_L{L_FRAME}_T{T_FRAME}_EM{NUM_ENDMEMBERS}_gram-{localization}_norm-{normalization}.png")
        fig_scatter.tight_layout()
        fig_scatter.savefig(out_path_scatter, dpi=300, bbox_inches='tight')
        #print(f"Saved volume scatter plot to: {out_path_scatter}")
        print(stats_text_scatter)
    else:
        print("Not enough valid data points to generate scatter plot.")

    plt.show()

if __name__ == "__main__":
    main()