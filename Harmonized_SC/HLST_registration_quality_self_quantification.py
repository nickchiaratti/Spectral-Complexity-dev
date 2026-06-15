"""
Multi-Sensor Co-Registration Sequential Quantifier

Iterates chronologically through all available sensor grids (except HARMONIZED)
in the ARD Master Cube, evaluating the geometric registration error of each 
frame against the immediately subsequent frame (L_i vs L_i+1). 

Produces longitudinal plots of Translation Drift, Rotational Misalignment, 
and Pearson Correlation per sensor.
"""

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timezone
import rasterio.transform
from skimage.registration import phase_cross_correlation
from skimage.transform import EuclideanTransform, warp
from scipy.optimize import minimize
from scipy.stats import pearsonr
import warnings

# ==========================================
# 1. CONFIGURATION
# ==========================================
Location = "Rochesterv2"

# Target search window size (100x100 pixels)
SPAN = 50 

ARD_CUBE_PATH = f"C:/satelliteImagery/HLST30/HLST_{Location}_Harmonized_SC_EM-7_Norm-bandCount.h5"

# ==========================================
# 2. UTILITY FUNCTIONS
# ==========================================
def get_luminance_and_mask(grp, f_idx):
    """
    Extracts the full-frame ortho_visual array, converts to Luminance,
    and applies the strictly pre-calculated ARD common_mask.
    """
    if 'ortho_visual' not in grp or 'common_mask' not in grp:
        raise ValueError(f"CRITICAL ERROR: Required datasets missing in grid {grp.name}")

    raw_vis = grp['ortho_visual'][f_idx, ...]
    
    # Handle BIP vs BSQ geometries
    if raw_vis.shape[0] in [3, 4]:
        bip_vis = np.transpose(raw_vis, (1, 2, 0))
    else:
        bip_vis = raw_vis
        
    rgb = bip_vis[..., :3].astype(np.float32) / 255.0
    # Standard relative luminance calculation
    luminance = np.dot(rgb, [0.299, 0.587, 0.114])
    
    # Ensure correct validity check matching previous ARD processing updates
    valid_mask = grp['common_mask'][f_idx, ...] != 1
    
    # Enforce alpha channel transparency if present
    if bip_vis.shape[-1] == 4:
        valid_mask &= (bip_vis[..., 3] > 0)

    return luminance, valid_mask

def find_optimal_window(mask_ref, mask_mov, lum_ref, span=100):
    """
    Finds a target window that is >=65% valid in BOTH sequential frames.
    Selects the window with the highest structural variance to ensure the 
    correlator has physical edges to lock onto.
    """
    h, w = mask_ref.shape
    half = span // 2
    stride = span // 4 
    
    combined_mask = mask_ref & mask_mov
    
    best_y, best_x = None, None
    best_valid_frac = 0.0
    best_variance = -1.0
    
    for y in range(half, h - half, stride):
        for x in range(half, w - half, stride):
            y0, y1 = y - half, y + half
            x0, x1 = x - half, x + half
            
            local_mask = combined_mask[y0:y1, x0:x1]
            valid_frac = np.sum(local_mask) / (span * span)
            
            if valid_frac >= 0.65:
                local_lum = lum_ref[y0:y1, x0:x1]
                valid_lum = local_lum[local_mask]
                variance = np.var(valid_lum) if len(valid_lum) > 0 else 0
                
                if valid_frac > best_valid_frac or (valid_frac == best_valid_frac and variance > best_variance):
                    best_valid_frac = valid_frac
                    best_variance = variance
                    best_y, best_x = y, x
                    
    return best_y, best_x, best_valid_frac

def plot_sensor_metrics(grid_name, plot_dates, magnitudes, rotations, correlations):
    """Generates the 3-panel registration metrics plot for a specific sensor."""
    valid_pairs_count = np.sum(~np.isnan(magnitudes))
    
    if valid_pairs_count == 0:
        print(f"\n--- {grid_name} Sequential Baseline Averages ---")
        print(f"Total Valid Pairs Evaluated: 0")
        print(f"No valid pairs for {grid_name} to plot.\n")
        return

    avg_mag = np.nanmean(magnitudes)
    avg_rot = np.nanmean(np.abs(rotations))
    avg_corr = np.nanmean(correlations)
    
    print(f"\n--- {grid_name} Sequential Baseline Averages ---")
    print(f"Total Valid Pairs Evaluated: {valid_pairs_count}")
    print(f"Average Sequential Translation: {avg_mag:.2f} meters")
    print(f"Average Absolute Rotation:      {avg_rot:.3f} degrees")
    print(f"Average Structural Correlation: {avg_corr:.3f}\n")

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    fig.canvas.manager.set_window_title(f"{grid_name} Sequential Registration Analysis")
    
    # 1. Magnitude Panel
    ax1.plot(plot_dates, magnitudes, marker='o', linestyle='-', color='tab:blue', linewidth=1.5, markersize=6)
    ax1.axhline(avg_mag, color='red', linestyle='--', linewidth=2, label=f'Average: {avg_mag:.2f}m')
    ax1.set_ylabel("Offset Magnitude (Meters)")
    ax1.set_title("Sequential Geometric Translation Drift")
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(loc='upper left')
    
    # 2. Rotation Panel
    ax2.plot(plot_dates, rotations, marker='o', linestyle='-', color='tab:orange', linewidth=1.5, markersize=6)
    ax2.axhline(0, color='black', linewidth=1, alpha=0.5) 
    ax2.axhline(avg_rot, color='red', linestyle='--', linewidth=2, alpha=0.5, label=f'+Avg Abs: {avg_rot:.3f}°')
    ax2.axhline(-avg_rot, color='red', linestyle='--', linewidth=2, alpha=0.5, label=f'-Avg Abs: -{avg_rot:.3f}°')
    ax2.set_ylabel("Rotation (Degrees)")
    ax2.set_title("Sequential Rotational Misalignment")
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(loc='upper left')

    # 3. Correlation Panel
    ax3.plot(plot_dates, correlations, marker='o', linestyle='-', color='tab:green', linewidth=1.5, markersize=6)
    ax3.axhline(avg_corr, color='red', linestyle='--', linewidth=2, label=f'Average: {avg_corr:.3f}')
    ax3.set_ylabel("Pearson Correlation (r)")
    ax3.set_title("Structural Confidence Score")
    ax3.set_xlabel("Acquisition Date (L_i)")
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.legend(loc='lower left')
    
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax3.tick_params(axis='x', rotation=45)
    
    stats_box = (f"{grid_name} Sequential Averages\n"
                 f"Valid Cloud-Free Pairs: {valid_pairs_count}\n"
                 f"Mean Translation: {avg_mag:.2f} m\n"
                 f"Mean Abs Rotation: {avg_rot:.3f}°\n"
                 f"Mean Correlation: r = {avg_corr:.3f}")
                 
    fig.text(0.98, 0.95, stats_box, fontsize=10, va='top', ha='right',
             bbox=dict(boxstyle='round', facecolor='white'))

    fig.suptitle(f"{grid_name} Registration Stability Analysis | Window: {SPAN}x{SPAN}px", fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

# ==========================================
# 3. MAIN EXECUTION
# ==========================================
def main():
    print("--- Multi-Sensor Intra-Sensor Co-Registration Quantifier ---")
    
    try:
        h5_ard = h5py.File(ARD_CUBE_PATH, 'r')
    except Exception as e:
        print(f"Error opening ARD Cube: {e}")
        return
        
    if '/HDFEOS/GRIDS/HARMONIZED' not in h5_ard:
        raise ValueError("CRITICAL ERROR: Missing /HDFEOS/GRIDS/HARMONIZED group in ARD Cube.")
        
    grp = h5_ard['/HDFEOS/GRIDS/HARMONIZED/Data Fields']
    
    # Validation checks
    l_times = grp['ortho_visual'].attrs['acquisition_time']
    spacecrafts_raw = grp['ortho_visual'].attrs['source_spacecraft']
    spacecrafts = [sc.decode('utf-8') if isinstance(sc, bytes) else sc for sc in spacecrafts_raw]
    
    unique_spacecrafts = sorted(list(set(spacecrafts)))
    print(f"Detected Source Spacecrafts in HARMONIZED for Analysis: {unique_spacecrafts}")

    # Geographic Metrology
    if 'GeoTransform' not in grp['ortho_visual'].attrs:
        raise ValueError("CRITICAL ERROR: GeoTransform missing on 'ortho_visual' for HARMONIZED")
        
    geo_tf = grp['ortho_visual'].attrs['GeoTransform']
    affine = rasterio.transform.Affine.from_gdal(*geo_tf)
    pixel_width = abs(affine.a)
    pixel_height = abs(affine.e)

    for grid_name in unique_spacecrafts:
        print(f"\n{'='*50}\nEvaluating Spacecraft: {grid_name}\n{'='*50}")
        
        frame_indices = [i for i, sc in enumerate(spacecrafts) if sc == grid_name]
        frame_indices = sorted(frame_indices, key=lambda i: l_times[i])
        
        num_frames = len(frame_indices)
        
        if num_frames < 2:
            print(f"Skipping {grid_name}: Insufficient frames ({num_frames}) for sequential analysis.")
            continue

        # Tracking Arrays
        plot_dates = []
        magnitudes = []
        rotations = []
        correlations = []
        
        print(f"Processing {num_frames - 1} sequential pairs for {grid_name}...")

        for idx in range(num_frames - 1):
            ref_idx = frame_indices[idx]
            mov_idx = frame_indices[idx + 1]
            
            ref_dt = datetime.fromtimestamp(l_times[ref_idx], tz=timezone.utc)
            mov_dt = datetime.fromtimestamp(l_times[mov_idx], tz=timezone.utc)
            
            plot_dates.append(mov_dt)
            print(f"[{idx+1}/{num_frames-1}] {ref_dt.strftime('%Y-%m-%d')} -> {mov_dt.strftime('%Y-%m-%d')}: ", end="")
            
            lum_ref, mask_ref = get_luminance_and_mask(grp, ref_idx)
            lum_mov, mask_mov = get_luminance_and_mask(grp, mov_idx)
            
            t_y, t_x, valid_frac = find_optimal_window(mask_ref, mask_mov, lum_ref, span=SPAN)
            
            if t_y is None:
                print("FAILED (Insufficient cloud-free overlap)")
                magnitudes.append(np.nan)
                rotations.append(np.nan)
                correlations.append(np.nan)
                continue
                
            half = SPAN // 2
            x_start, x_end = t_x - half, t_x + half
            y_start, y_end = t_y - half, t_y + half
            
            rel_center_y = t_y - y_start
            rel_center_x = t_x - x_start

            local_lum_ref = lum_ref[y_start:y_end, x_start:x_end]
            local_mask_ref = mask_ref[y_start:y_end, x_start:x_end]
            local_lum_mov = lum_mov[y_start:y_end, x_start:x_end]
            local_mask_mov = mask_mov[y_start:y_end, x_start:x_end]

            # 1. Sub-Pixel Phase Correlation Seed
            shift_vector, error, diffphase = phase_cross_correlation(
                reference_image=local_lum_ref, moving_image=local_lum_mov, 
                reference_mask=local_mask_ref, moving_mask=local_mask_mov, upsample_factor=100
            )
            init_dy, init_dx = shift_vector

            # 2. Euclidean Transform Parametric Optimization
            def objective_function(params):
                dy, dx, theta_deg = params
                t1 = EuclideanTransform(translation=(-rel_center_x, -rel_center_y))
                t2 = EuclideanTransform(rotation=np.deg2rad(theta_deg))
                t3 = EuclideanTransform(translation=(rel_center_x + dx, rel_center_y + dy))
                tform = t1 + t2 + t3
                
                # Suppress mode/cval warnings on invalid bounds during optimization
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    warped_lum = warp(local_lum_mov, tform.inverse, order=3, mode='constant', cval=np.nan)
                    warped_mask = warp(local_mask_mov.astype(float), tform.inverse, order=0, mode='constant', cval=0.0).astype(bool)
                
                current_overlap = local_mask_ref & warped_mask & ~np.isnan(warped_lum)
                if np.sum(current_overlap) < (0.15 * local_mask_ref.size):
                    return 1.0 # Penalize heavily
                    
                r, _ = pearsonr(local_lum_ref[current_overlap], warped_lum[current_overlap])
                return -r 

            initial_guess = [init_dy, init_dx, 0.0]
            bounds = [(init_dy - 2.0, init_dy + 2.0), (init_dx - 2.0, init_dx + 2.0), (-2.0, 2.0)]
            
            result = minimize(objective_function, initial_guess, method='Nelder-Mead', bounds=bounds, options={'xatol': 1e-4, 'fatol': 1e-4})
            
            opt_dy, opt_dx, opt_theta = result.x
            opt_correlation = -result.fun
            
            shift_meters_x = opt_dx * pixel_width
            shift_meters_y = opt_dy * pixel_height
            mag_meters = np.sqrt(shift_meters_x**2 + shift_meters_y**2)
            
            # Physical Outlier Guardrail (Cloud Shadow Lock)
            if mag_meters > 250.0:
                print(f"FAILED (Offset {mag_meters:.1f}m > 250m physical limit. Tracking failure.)")
                magnitudes.append(np.nan)
                rotations.append(np.nan)
                correlations.append(np.nan)
                continue
            
            magnitudes.append(mag_meters)
            rotations.append(opt_theta)
            correlations.append(opt_correlation)
            
            print(f"Offset={mag_meters:.2f}m, Rot={opt_theta:+.3f}°, r={opt_correlation:.3f} (Valid: {valid_frac*100:.0f}%)")

        # Generate plots for this specific sensor grid
        plot_sensor_metrics(grid_name, plot_dates, magnitudes, rotations, correlations)

    h5_ard.close()
    
    # Display all sensor plots simultaneously at the end
    plt.show()

if __name__ == "__main__":
    main()