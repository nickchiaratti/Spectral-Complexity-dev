"""
Intra-Sensor Co-Registration Sequential Quantifier (Landsat Only)

Iterates chronologically through a single-sensor time series, evaluating 
the geometric registration error of each frame against the immediately 
subsequent frame (L_i vs L_i+1). 

Produces longitudinal plots of Translation Drift, Rotational Misalignment, 
and Pearson Correlation, alongside the calculated valid historical averages.

Author: [Your Name/Lab]
Date: 2026-04-17
"""

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timezone
import rasterio.transform
from pyproj import Transformer, CRS
from skimage.registration import phase_cross_correlation
from skimage.transform import EuclideanTransform, warp
from scipy.optimize import minimize
from scipy.stats import pearsonr
import SpecComplex as sc

# ==========================================
# 1. CONFIGURATION
# ==========================================
Location = "Rochester"

# Target search window size (100x100 pixels)
SPAN = 100 

# Strict ARD Masking Configuration
SUN_ELEVATION_THRESHOLD = 25
CLOUD_DILATION = 0
# Updated to 142 (Bits 1, 2, 3, 7) to strictly target Dilated Cloud, Cirrus, Cloud, and Water.
QA_REJECT_MASK = 142 
RADSAT_ACCEPT_VALUE = 0
AEROSOL_ACCEPT_LEVEL = 'high'

landsat_path = f"C:/satelliteImagery/LANDSAT/{Location}/LANDSAT_Stack_{Location}_GEE_2015_2025_WRS16_SC_EM-7_Gram-minEndmember_Norm-bandCount.h5"

# ==========================================
# 2. UTILITY FUNCTIONS
# ==========================================

def get_landsat_luminance_and_mask(grp, f_idx):
    """
    Extracts the full-frame BSQ ortho_visual array, converts to Luminance,
    and applies explicit QA masking via SpecComplex.
    """
    raw_vis = grp['ortho_visual'][f_idx, ...]
    
    if raw_vis.shape[0] in [3, 4]:
        bip_vis = np.transpose(raw_vis, (1, 2, 0))
    else:
        bip_vis = raw_vis
        
    rgb = bip_vis[..., :3].astype(np.float32) / 255.0
    luminance = np.dot(rgb, [0.299, 0.587, 0.114])
    
    shape = luminance.shape
    valid_mask = np.ones(shape, dtype=bool)
    
    if bip_vis.shape[-1] == 4:
        valid_mask &= (bip_vis[..., 3] > 0)
        
    ard_mask = sc.get_landsat_mask(
        data_grp=grp, f_idx=f_idx, shape=shape,
        sun_elevation_threshold=SUN_ELEVATION_THRESHOLD,
        cloud_dilation=CLOUD_DILATION, qa_reject_mask=QA_REJECT_MASK,
        radsat_accept_value=RADSAT_ACCEPT_VALUE, aerosol_accept_level=AEROSOL_ACCEPT_LEVEL
    )
    valid_mask &= ard_mask

    return luminance, valid_mask

def find_optimal_window(mask_ref, mask_mov, lum_ref, span=100):
    """
    Finds a target window that is >=65% valid in BOTH sequential frames.
    Selects the window with the highest structural variance.
    """
    h, w = mask_ref.shape
    half = span // 2
    stride = 1
    
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

# ==========================================
# 3. MAIN EXECUTION
# ==========================================

def main():
    print("--- Intra-Sensor Sequential Co-Registration Quantifier (Landsat) ---")
    
    try:
        h5_l = h5py.File(landsat_path, 'r')
    except Exception as e:
        print(f"Error opening file: {e}")
        return
        
    grp_l = h5_l['/HDFEOS/GRIDS/LANDSAT/Data Fields']
    sr_l = grp_l['surface_reflectance']
    
    l_times = sr_l.attrs['acquisition_time']
    num_frames = sr_l.shape[0]
    
    if num_frames < 2:
        print("Error: Dataset contains insufficient frames for sequential analysis.")
        return

    # Geographic Metrology
    geo_tf = sr_l.attrs['GeoTransform']
    affine = rasterio.transform.Affine.from_gdal(*geo_tf)
    pixel_width = abs(affine.a)
    pixel_height = abs(affine.e)

    # Tracking Arrays (Using NaNs for mathematically void/cloudy pairs)
    plot_dates = []
    magnitudes = []
    rotations = []
    correlations = []
    
    print(f"Evaluating {num_frames - 1} sequential pairs...")

    for idx in range(num_frames - 1):
        ref_idx = idx
        mov_idx = idx + 1
        
        ref_dt = datetime.fromtimestamp(l_times[ref_idx], tz=timezone.utc)
        mov_dt = datetime.fromtimestamp(l_times[mov_idx], tz=timezone.utc)
        
        plot_dates.append(mov_dt) # Anchor the result to the moving frame's date
        print(f"[{idx+1}/{num_frames-1}] {ref_dt.strftime('%Y-%m-%d')} -> {mov_dt.strftime('%Y-%m-%d')}: ", end="")
        
        lum_ref, mask_ref = get_landsat_luminance_and_mask(grp_l, ref_idx)
        lum_mov, mask_mov = get_landsat_luminance_and_mask(grp_l, mov_idx)
        
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
            
            warped_lum = warp(local_lum_mov, tform.inverse, order=3, mode='constant', cval=np.nan)
            warped_mask = warp(local_mask_mov.astype(float), tform.inverse, order=0, mode='constant', cval=0.0).astype(bool)
            
            current_overlap = local_mask_ref & warped_mask & ~np.isnan(warped_lum)
            if np.sum(current_overlap) < (0.15 * local_mask_ref.size):
                return 1.0 # Heavily penalize invalid geometric states
                
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
        
        # Strict Failure Handling for Unmasked Cloud Outliers
        # Landsat 8/9 nominal CE90 is <12m. A >250m shift is a physical impossibility 
        # for L2SP data and indicates the algorithm locked onto moving atmospheric noise.
        if mag_meters > 250.0:
            print(f"FAILED (Offset {mag_meters:.1f}m > 250m physical limit. Tracking failure due to unmasked cloud.)")
            magnitudes.append(np.nan)
            rotations.append(np.nan)
            correlations.append(np.nan)
            continue
        
        magnitudes.append(mag_meters)
        rotations.append(opt_theta)
        correlations.append(opt_correlation)
        
        print(f"Offset={mag_meters:.2f}m, Rot={opt_theta:+.3f}°, r={opt_correlation:.3f} (Valid: {valid_frac*100:.0f}%)")

    # ==========================================
    # 4. STATISTICAL SUMMARY & PLOTTING
    # ==========================================
    
    # Calculate pure structural averages (ignoring NaN gaps)
    avg_mag = np.nanmean(magnitudes)
    avg_rot = np.nanmean(np.abs(rotations)) # Absolute yaw error
    avg_corr = np.nanmean(correlations)
    valid_pairs_count = np.sum(~np.isnan(magnitudes))
    
    print("\n--- Sequential Baseline Averages ---")
    print(f"Total Valid Pairs Evaluated: {valid_pairs_count}")
    print(f"Average Sequential Translation: {avg_mag:.2f} meters")
    print(f"Average Absolute Rotation:      {avg_rot:.3f} degrees")
    print(f"Average Structural Correlation: {avg_corr:.3f}")

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    fig.canvas.manager.set_window_title(f"Landsat Sequential Registration Analysis | Window: {SPAN}x{SPAN}px")
    
    # 1. Magnitude Panel
    ax1.plot(plot_dates, magnitudes, marker='o', linestyle='-', color='tab:blue', linewidth=1.5, markersize=6)
    ax1.axhline(avg_mag, color='red', linestyle='--', linewidth=2, label=f'Mean: {avg_mag:.2f}m')
    ax1.set_ylabel("Offset Magnitude (Meters)")
    ax1.set_title("Sequential Geometric Translation Drift")
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(loc='upper right')
    
    # 2. Rotation Panel
    ax2.plot(plot_dates, rotations, marker='o', linestyle='-', color='tab:orange', linewidth=1.5, markersize=6)
    ax2.axhline(0, color='black', linewidth=1, alpha=0.5) 
    ax2.axhline(avg_rot, color='red', linestyle='--', linewidth=2, alpha=0.5, label=f'|Mean|: {avg_rot:.3f}°')
    ax2.set_ylabel("Rotation (Degrees)")
    ax2.set_title("Sequential Rotational Misalignment")
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(loc='upper right')

    # 3. Correlation Panel
    ax3.plot(plot_dates, correlations, marker='o', linestyle='-', color='tab:green', linewidth=1.5, markersize=6)
    ax3.axhline(avg_corr, color='red', linestyle='--', linewidth=2, label=f'Mean: {avg_corr:.3f}')
    ax3.set_ylabel("Pearson Correlation (r)")
    ax3.set_title("Structural Confidence Score")
    ax3.set_xlabel("Acquisition Date")
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.legend(loc='lower right')
    
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax3.tick_params(axis='x', rotation=45)
    
    stats_box = (f"Landsat 8/9 Sequential Averages\n"
                 f"Valid Cloud-Free Pairs: {valid_pairs_count}\n"
                 f"Mean Translation: {avg_mag:.2f} m\n"
                 f"Mean Abs Rotation: {avg_rot:.3f}°\n"
                 f"Mean Correlation: r = {avg_corr:.3f}")
                 
    fig.text(0.5, 0.85, stats_box, fontsize=10, va='top', ha='center',
             bbox=dict(boxstyle='square', facecolor='white'))

    fig.suptitle(f"Landsat Registration Stability Analysis| Window: {SPAN}x{SPAN}px", fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    h5_l.close()
    plt.show()

if __name__ == "__main__":
    main()