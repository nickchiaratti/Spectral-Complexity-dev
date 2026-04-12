import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from datetime import datetime, timezone
import SpecComplex as sc
from skimage import exposure
try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo

import rasterio.transform
from pyproj import Transformer, CRS

# ==========================================
# 1. CONFIGURATION
# ==========================================
Location = "Tait"
Frame_Reg = "WRS16" 
complexity_type = 'sliding_volume_z_score_masked'
LOG_SCALE = False # Z-Scores cannot be log-scaled

# Strict Target Selection
TARGET_SENSOR = 'LANDSAT' # 'LANDSAT' or 'TANAGER'
TS_DATE = '2025-09-12'
TS_LOCATION = {'latlon': (43.139423, -77.503825), 'label': "ROCX NITE Tarp", 'color': 'tab:purple'}
SPAN = 15 # X and Y pixel span for the 1D profiles

# ARD Mask Configuration
MASKING = True
SUN_ELEVATION_THRESHOLD = 30
CLOUD_DILATION = 2
TANAGER_AEROSOL_DEPTH_THRESHOLD = 0.35
TANAGER_SR_UNCERTAINTY_THRESHOLD = 0.10
QA_REJECT_MASK = 0b111111
RADSAT_ACCEPT_VALUE = 0
AEROSOL_ACCEPT_LEVEL = 'medium'

# File Paths
landsat_path = f"C:/satelliteImagery/LANDSAT/{Location}/LANDSAT_Stack_{Location}_GEE_2015_2025_{Frame_Reg}_SC_EM-7_Gram-minEndmember_Norm-bandCount.h5"
tanager_path = f"C:/satelliteImagery/Tanager/{Location}/Tanager_Stack_{Location}_HDFEOS_SC_EM-7_Gram-minEndmember_Norm-bandCount.h5"

# ==========================================
# 2. UTILITY & MATH FUNCTIONS
# ==========================================

def percentile_normalize_array(arr, low=2, high=98):
    """Normalizes array for True Color RGB display."""
    if np.all(np.isnan(arr)): return np.zeros_like(arr)
    p_low, p_high = np.nanpercentile(arr, (low, high))
    if p_low == p_high: return np.zeros_like(arr)
    return exposure.rescale_intensity(arr, in_range=(p_low, p_high), out_range=(0, 1)).clip(0, 1)

def calculate_subpixel_peak(y_minus1, y_0, y_plus1):
    """
    Fits a 1D quadratic parabola to 3 adjacent pixels and calculates the sub-pixel vertex.
    Returns the offset (dx) relative to the center pixel (x=0).
    Formula: dx = (y[-1] - y[+1]) / (2 * (y[-1] - 2*y[0] + y[+1]))
    """
    if np.isnan([y_minus1, y_0, y_plus1]).any():
        return np.nan, None, None, None # Fail-fast on ARD masked data

    # Quadratic coefficients: y = ax^2 + bx + c
    a = (y_minus1 + y_plus1) / 2.0 - y_0
    b = (y_plus1 - y_minus1) / 2.0
    c = y_0

    if a >= 0:
        # If 'a' is positive, the parabola opens upwards (it's a local minimum, not a peak)
        return np.nan, a, b, c

    dx = -b / (2.0 * a)
    return dx, a, b, c

def find_frame_by_date(h5_file, grid_name, target_date_str):
    """Searches the acquisition_time attribute for a matching YYYY-MM-DD."""
    dset = h5_file[f'/HDFEOS/GRIDS/{grid_name}/Data Fields/surface_reflectance']
    times = dset.attrs.get('acquisition_time')
    if times is None: return None, None
        
    for i, ts in enumerate(times):
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        if dt.strftime('%Y-%m-%d') == target_date_str:
            return i, dt
    return None, None

# ==========================================
# 3. MAIN EXECUTION
# ==========================================

def main():
    print(f"--- Sub-Pixel Anomaly Profiler ---")
    print(f"Target: {TS_LOCATION['label']} at {TS_LOCATION['latlon']}")
    print(f"Spatial Profile Epoch: {TARGET_SENSOR} on {TS_DATE}")
    
    try:
        h5_l = h5py.File(landsat_path, 'r')
        h5_t = h5py.File(tanager_path, 'r')
    except Exception as e:
        print(f"Error opening files: {e}")
        return

    grp_l = h5_l['/HDFEOS/GRIDS/LANDSAT/Data Fields']
    grp_t = h5_t['/HDFEOS/GRIDS/TANAGER/Data Fields']

    # Select the target group dynamically based on configuration
    h5_target = h5_l if TARGET_SENSOR == 'LANDSAT' else h5_t
    grp_target = grp_l if TARGET_SENSOR == 'LANDSAT' else grp_t

    # Find Target Frame
    tgt_idx, tgt_dt = find_frame_by_date(h5_target, TARGET_SENSOR, TS_DATE)
    target_year = datetime.strptime(TS_DATE, '%Y-%m-%d').year
    
    if tgt_idx is None:
        print(f"Error: Target date {TS_DATE} not found in {TARGET_SENSOR} stack.")
        return

    # --- Geometric Mapping (Derived from the targeted sensor's grid) ---
    geo_transform = grp_target['surface_reflectance'].attrs['GeoTransform']
    spatial_ref = grp_target['surface_reflectance'].attrs['spatial_ref']
    if isinstance(spatial_ref, bytes): spatial_ref = spatial_ref.decode('utf-8')
    
    crs = CRS.from_wkt(spatial_ref)
    transformer_to_px = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    transformer_to_ll = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    
    affine = rasterio.transform.Affine.from_gdal(*geo_transform)
    inv_affine = ~affine
    
    t_lat, t_lon = TS_LOCATION['latlon']
    proj_x, proj_y = transformer_to_px.transform(t_lon, t_lat)
    px, py = inv_affine * (proj_x, proj_y)
    t_x, t_y = int(round(px)), int(round(py)) # Center Pixel Indices

    h, w = grp_target['sliding_volume_map'].shape[1:]
    half_span = SPAN // 2
    x_start, x_end = max(0, t_x - half_span), min(w, t_x + half_span + 1)
    y_start, y_end = max(0, t_y - half_span), min(h, t_y + half_span + 1)

    # --- Data Extraction & Strict Masking ---
    comp = grp_target[complexity_type][tgt_idx, ...].copy()
    if MASKING:
        if TARGET_SENSOR == 'LANDSAT':
            mask = sc.get_landsat_mask(grp_target, tgt_idx, comp.shape, SUN_ELEVATION_THRESHOLD, CLOUD_DILATION, QA_REJECT_MASK, RADSAT_ACCEPT_VALUE, AEROSOL_ACCEPT_LEVEL)
        else:
            mask = sc.get_tanager_mask(grp_target, tgt_idx, comp.shape, SUN_ELEVATION_THRESHOLD, CLOUD_DILATION, True, TANAGER_SR_UNCERTAINTY_THRESHOLD, TANAGER_AEROSOL_DEPTH_THRESHOLD)
        comp[~mask] = np.nan
        
    # Extract Local Arrays
    tgt_chip = comp[y_start:y_end, x_start:x_end]
    tgt_h_trans = comp[t_y, x_start:x_end]
    tgt_v_trans = comp[y_start:y_end, t_x]
    
    # Sub-Pixel Mathematical Fit (Extract the 3 pixels centered on the target)
    tgt_dx, tgt_dy = np.nan, np.nan
    tgt_h_coeff, tgt_v_coeff = None, None
    
    # Ensure we have boundary margins to extract a 3x3
    if t_x > 0 and t_x < (w-1):
        tgt_dx, a, b, c = calculate_subpixel_peak(comp[t_y, t_x-1], comp[t_y, t_x], comp[t_y, t_x+1])
        tgt_h_coeff = (a, b, c)
    if t_y > 0 and t_y < (h-1):
        tgt_dy, a, b, c = calculate_subpixel_peak(comp[t_y-1, t_x], comp[t_y, t_x], comp[t_y+1, t_x])
        tgt_v_coeff = (a, b, c)

    # Convert fractional sub-pixel peaks back to EPSG:4326
    def get_peak_latlon(dx, dy):
        if np.isnan(dx) or np.isnan(dy): return None, None
        # Add 0.5 to target the physical geometric centroid of the pixel per GDAL specs
        sub_px = t_x + dx + 0.5
        sub_py = t_y + dy + 0.5
        pr_x, pr_y = affine * (sub_px, sub_py)
        plon, plat = transformer_to_ll.transform(pr_x, pr_y)
        return plat, plon

    tgt_plat, tgt_plon = get_peak_latlon(tgt_dx, tgt_dy)

    print("\n--- Sub-Pixel Peak Predictions ---")
    if tgt_plat: 
        print(f"{TARGET_SENSOR} Peak: {tgt_plat:.5f}, {tgt_plon:.5f}  (Offset: dx={tgt_dx:.3f}, dy={tgt_dy:.3f})")
    else:      
        print(f"{TARGET_SENSOR} Peak: N/A (Failed quadratic fit or masked data)")

    # --- Temporal Extraction (For the bottom right timeline - BOTH SENSORS) ---
    def get_timeline(grp, sensor):
        times, vals = [], []
        acq = grp['surface_reflectance'].attrs['acquisition_time']
        shape = (h, w)
        for i in range(len(acq)):
            dt = datetime.fromtimestamp(acq[i], tz=timezone.utc)
            # Strictly filter data to the target year to preserve intra-annual stationarity
            if dt.year == target_year:
                val = grp[complexity_type][i, t_y, t_x]
                
                # Enforce strict ARD masking across the temporal axis
                if MASKING and not np.isnan(val):
                    if sensor == 'LANDSAT':
                        mask = sc.get_landsat_mask(grp, i, shape, SUN_ELEVATION_THRESHOLD, CLOUD_DILATION, QA_REJECT_MASK, RADSAT_ACCEPT_VALUE, AEROSOL_ACCEPT_LEVEL)
                    else:
                        mask = sc.get_tanager_mask(grp, i, shape, SUN_ELEVATION_THRESHOLD, CLOUD_DILATION, True, TANAGER_SR_UNCERTAINTY_THRESHOLD, TANAGER_AEROSOL_DEPTH_THRESHOLD)
                    
                    # If the ARD module rejects this pixel (e.g., due to cloud dilation), force NaN
                    if not mask[t_y, t_x]:
                        val = np.nan
                        
                if not np.isnan(val):
                    times.append(dt)
                    vals.append(val)
        return times, vals
        
    l_times, l_vals = get_timeline(grp_l, 'LANDSAT')
    t_times, t_vals = get_timeline(grp_t, 'TANAGER')

    # ==========================================
    # 4. VISUALIZATION UI
    # ==========================================
    
    fig = plt.figure(figsize=(16, 10))
    fig.canvas.manager.set_window_title("1D Spatial and Temporal Profiles")
    fig.subplots_adjust(top=0.88, bottom=0.10, left=0.05, right=0.95, hspace=0.35, wspace=0.3)
    gs = gridspec.GridSpec(2, 4, figure=fig)
    
    ax_h = fig.add_subplot(gs[0, :2])
    ax_v = fig.add_subplot(gs[0, 2:])
    ax_chip_rgb = fig.add_subplot(gs[1, 0])
    ax_chip_comp = fig.add_subplot(gs[1, 1])
    ax_t = fig.add_subplot(gs[1, 2:])
    # Removed twin axis to strictly enforce shared comparative scaling

    x_indices = np.arange(x_start, x_end) + 0.5
    y_indices = np.arange(y_start, y_end) + 0.5
    
    # Establish pre-attentive visual encoding (Dashed/Triangle for Landsat, Solid/Square for Tanager)
    tgt_color = 'purple'
    tgt_marker = '^' if TARGET_SENSOR == 'LANDSAT' else 's'
    tgt_ls = '--' if TARGET_SENSOR == 'LANDSAT' else '-'
    tgt_lw = 2 if TARGET_SENSOR == 'LANDSAT' else 1.5
    
    # --- Plot Horizontal Transect ---
    ax_h.plot(x_indices, tgt_h_trans, color=tgt_color, marker=tgt_marker, linestyle=tgt_ls, lw=tgt_lw, label=TARGET_SENSOR)
    ax_h.axvline(t_x + 0.5, color='red', linestyle='--', linewidth=1.5, label='Target Focus')
    
    # Overlay Continuous Parabola Fit
    if tgt_h_coeff and not np.isnan(tgt_dx):
        a, b, c = tgt_h_coeff
        x_dense = np.linspace(-1, 1, 50)
        y_dense = a*(x_dense**2) + b*x_dense + c
        ax_h.plot(x_dense + t_x + 0.5, y_dense, tgt_color, linestyle=':', lw=2, alpha=0.5, label=f'{TARGET_SENSOR} Parabolic Fit')
        ax_h.axvline(t_x + 0.5 + tgt_dx, color='blue', lw=1.5, label='Predicted Center')

    ax_h.set_title(f"Horizontal Spatial Profile (Row/Y = {t_y})")
    ax_h.set_xlabel("X Pixel Coordinate (Longitude)")
    ax_h.set_ylabel('Spectral Complexity Z-Score')
    ax_h.grid(True, alpha=0.3, ls='--')
    ax_h.legend()

    # --- Plot Vertical Transect ---
    ax_v.plot(y_indices, tgt_v_trans, color=tgt_color, marker=tgt_marker, linestyle=tgt_ls, lw=tgt_lw, label=TARGET_SENSOR)
    ax_v.axvline(t_y + 0.5, color='red', linestyle='--', linewidth=1.5, label='Target Focus')
    
    if tgt_v_coeff and not np.isnan(tgt_dy):
        a, b, c = tgt_v_coeff
        y_dense = np.linspace(-1, 1, 50)
        z_dense = a*(y_dense**2) + b*y_dense + c
        ax_v.plot(y_dense + t_y + 0.5, z_dense, tgt_color, linestyle=':', lw=2, alpha=0.5, label=f'{TARGET_SENSOR} Parabolic Fit')
        ax_v.axvline(t_y + 0.5 + tgt_dy, color='blue', lw=1.5, label='Predicted Center')

    ax_v.set_title(f"Vertical Spatial Profile (Col/X = {t_x})")
    ax_v.set_xlabel("Y Pixel Coordinate (Latitude)")
    ax_v.grid(True, alpha=0.3, ls='--')
    ax_v.legend()

    # --- 2D Subsets (Context Chips) ---
    raw_vis = grp_target['ortho_visual'][tgt_idx, :, y_start:y_end, x_start:x_end]
    rgb = np.transpose(raw_vis[:3, ...], (1, 2, 0)).astype(np.float32) / 255.0
    
    rel_y, rel_x = t_y - y_start, t_x - x_start
    
    ax_chip_rgb.imshow(rgb)
    ax_chip_rgb.axhline(rel_y, color='red', linestyle='--', lw=1.5, alpha=0.8)
    ax_chip_rgb.axvline(rel_x, color='red', linestyle='--', lw=1.5, alpha=0.8)
    if not np.isnan(tgt_dx): ax_chip_rgb.plot(rel_x + tgt_dx, rel_y + tgt_dy, color='blue', marker='x', markersize=12, mew=2)
    ax_chip_rgb.set_title(f"{TARGET_SENSOR} True Color Context")
    ax_chip_rgb.axis('off')
    
    with np.errstate(all='ignore'):
        c_min, c_max = np.nanpercentile(tgt_chip, (2, 98)) if not np.all(np.isnan(tgt_chip)) else (0, 1)
    ax_chip_comp.imshow(tgt_chip, cmap='viridis', vmin=c_min, vmax=c_max)
    ax_chip_comp.axhline(rel_y, color='red', linestyle='--', lw=1.5, alpha=0.8)
    ax_chip_comp.axvline(rel_x, color='red', linestyle='--', lw=1.5, alpha=0.8)
    if not np.isnan(tgt_dx): ax_chip_comp.plot(rel_x + tgt_dx, rel_y + tgt_dy, color='blue', marker='x', markersize=12, mew=2)
    ax_chip_comp.set_title(f"{TARGET_SENSOR} Complexity")
    ax_chip_comp.axis('off')

    # --- Temporal Timeline (Shared Axis for True Magnitude Comparison) ---
    ax_t.plot(l_times, l_vals, marker='^', color='purple', ls='--', lw=2, label='Landsat')
    ax_t.plot(t_times, t_vals, marker='s', color='purple', ls='-', lw=1.5, label='Tanager')
    
    # Mark target date matching the target focus semantic
    ax_t.axvline(tgt_dt, color='red', linestyle='--', alpha=0.8, linewidth=2, label='Target Focus')
    
    ax_t.set_title(f"Temporal Profile (Row/Y = {t_y}, Col/X = {t_x})")
    ax_t.grid(True, alpha=0.3, ls='--')
    ax_t.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax_t.tick_params(axis='x', rotation=45)
    ax_t.set_ylabel("Spectral Complexity Z-Score")
    
    # Strictly lock the x-axis to the target year to frame the intra-annual phenology/stationarity
    start_of_year = datetime(target_year, 6, 1, tzinfo=timezone.utc)
    end_of_year = datetime(target_year, 11, 15, 23, 59, 59, tzinfo=timezone.utc)
    ax_t.set_xlim(start_of_year, end_of_year)
    ax_t.legend(loc='best')

    # Formatting and Stats Text
    plat_str = f"{tgt_plat:.5f}" if tgt_plat else "N/A"
    plon_str = f"{tgt_plon:.5f}" if tgt_plon else "N/A"
    
    header_text = (f"Orthogonal Voxel Profiler | Target: {TS_LOCATION['label']}\n"
                   f"Spatial Analysis: {TARGET_SENSOR} ({TS_DATE}) | "
                   f"Sub-Pixel Maxima: Lat {plat_str}, Lon {plon_str}")
    
    fig.suptitle(header_text, fontsize=10)
    
    h5_l.close()
    h5_t.close()
    plt.show()

if __name__ == "__main__":
    main()