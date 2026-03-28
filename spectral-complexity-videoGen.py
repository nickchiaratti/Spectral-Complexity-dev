import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from datetime import datetime, timezone
try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo # For Python < 3.9
from skimage import exposure

import rasterio.transform
from pyproj import Transformer, CRS

# ==========================================
# 1. VIDEO & EXTRACTION CONFIGURATION
# ==========================================

# File Paths
landsat_path = "C:/satelliteImagery/LANDSAT/Rochester/LANDSAT_Stack_Rochester_GEE_2015_2025_SC_EM-7_Gram-minEndmember_Norm-bandCount.h5"
tanager_path = "C:/satelliteImagery/Tanager/Rochester/Tanager_Stack_Rochester_HDFEOS_SC_EM-7_Gram-minEndmember_Norm-bandCount.h5"
OUTPUT_DIR = "C:/satelliteImagery/MultiSensor_Analysis_Videos"
COMPLEXITY_TYPE = 'sliding_volume_z_score'

# Temporal Configuration
START_DATE = datetime(2025, 1, 1, tzinfo=timezone.utc)
END_DATE = datetime(2025, 12, 31, tzinfo=timezone.utc)

# Video Output Configuration
FPS = 1  # Frames Per Second (3 FPS = 0.33 second delay between frames)
DPI = 300
SHOW_PIXEL_INDICATORS = True
GLOBAL_COLOR_SCALE = True # HIGHLY RECOMMENDED: Prevents colormap "flickering"

LANDSAT_RGB_BANDS = (3, 2, 1)

# Time Series Locations (Latitude, Longitude)
TS_LOCATIONS = [
    {'latlon': (43.142856, -77.508451), 'label': "West Tait Forest",     'color': 'white'}, 
    {'latlon': (43.144861, -77.501176), 'label': "East Tait Forest",             'color': 'yellow'},
    {'latlon': (43.136910, -77.469462), 'label': "Artificial turf football field",  'color': 'cyan'},
    {'latlon': (43.138241, -77.470873), 'label': "Recently added artificial turf",  'color': 'magenta'},
    {'latlon': (43.141297, -77.506256), 'label': "Tait Parking Lot",                'color': 'red'},
    {'latlon': (43.139411, -77.504005), 'label': "ROCX NITE Tarp",                  'color': 'lime'},
]

# ==========================================
# 2. UTILITY FUNCTIONS
# ==========================================

def percentile_normalize_array(arr, low=1, high=99):
    if np.all(np.isnan(arr)): return np.zeros_like(arr)
    p_low, p_high = np.nanpercentile(arr, (low, high))
    if p_low == p_high: return np.zeros_like(arr)
    return exposure.rescale_intensity(arr, in_range=(p_low, p_high), out_range=(0, 1)).clip(0, 1)

def map_locations(dset):
    """Maps geographic lat/lon to specific pixel grid coordinates."""
    geo_transform = dset.attrs.get('GeoTransform')
    spatial_ref = dset.attrs.get('spatial_ref')
    mapped = []
    
    if geo_transform is not None and spatial_ref is not None:
        if isinstance(spatial_ref, bytes): spatial_ref = spatial_ref.decode('utf-8')
        crs = CRS.from_wkt(spatial_ref)
        transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        affine = rasterio.transform.Affine(*geo_transform)
        inv_affine = ~affine
        
        for loc in TS_LOCATIONS:
            lat, lon = loc['latlon']
            px, py = inv_affine * transformer.transform(lon, lat)
            mapped.append({'label': loc['label'], 'color': loc['color'], 'y': int(round(py)), 'x': int(round(px))})
    return mapped

# ==========================================
# 3. MAIN VIDEO GENERATOR
# ==========================================

def generate_videos():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Initializing Multi-Sensor Virtual Constellation...")
    
    # 1. Open Both HDF5 Files
    h5_l = h5py.File(landsat_path, 'r')
    h5_t = h5py.File(tanager_path, 'r')
    
    grp_l = h5_l['/HDFEOS/GRIDS/LANDSAT/Data Fields']
    grp_t = h5_t['/HDFEOS/GRIDS/TANAGER/Data Fields']
    
    sr_l = grp_l['surface_reflectance']
    sr_t = grp_t['surface_reflectance']
    
    vol_l = grp_l[COMPLEXITY_TYPE]
    vol_t = grp_t[COMPLEXITY_TYPE]
    
    # Map spatial locations specifically for each sensor's grid
    mapped_locs_l = map_locations(sr_l)
    mapped_locs_t = map_locations(sr_t)
    
    # 2. Extract and Interleave Temporal Metadata
    unified_frames = []
    
    for idx, ts in enumerate(sr_l.attrs.get('acquisition_time', [])):
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        if START_DATE <= dt <= END_DATE:
            unified_frames.append({
                'sensor': 'LANDSAT', 'idx': idx, 'dt': dt, 
                'dt_et': dt.astimezone(ZoneInfo("America/New_York")),
                'vol_dset': vol_l, 'grp': grp_l, 'sr_dset': sr_l
            })
            
    for idx, ts in enumerate(sr_t.attrs.get('acquisition_time', [])):
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        if START_DATE <= dt <= END_DATE:
            unified_frames.append({
                'sensor': 'TANAGER', 'idx': idx, 'dt': dt, 
                'dt_et': dt.astimezone(ZoneInfo("America/New_York")),
                'vol_dset': vol_t, 'grp': grp_t, 'sr_dset': sr_t
            })
            
    # Sort chronologically to create the virtual constellation time series
    unified_frames.sort(key=lambda x: x['dt'])
            
    if not unified_frames:
        print("No frames found in the specified date range.")
        return
        
    print(f"Processing {len(unified_frames)} interleaved frames...")

    # 3. Calculate Global Min/Max for Shared Complexity Colormap
    v_min, v_max = 0, 1
    if GLOBAL_COLOR_SCALE:
        print("Calculating shared global color scale percentiles across both sensors...")
        all_vols = []
        for frame in unified_frames:
            data = frame['vol_dset'][frame['idx'], ...]
            valid_data = data[~np.isnan(data)]
            if len(valid_data) > 0:
                all_vols.append(valid_data)
        
        if all_vols:
            flat_vols = np.concatenate(all_vols)
            v_min, v_max = np.nanpercentile(flat_vols, (2, 98))
            print(f"Global Color Scale Locked: vmin={v_min:.4f}, vmax={v_max:.4f}")

    # 4. Set up matplotlib writers
    Writer = animation.writers['ffmpeg']
    writer_rgb = Writer(fps=FPS, metadata=dict(artist='MultiSensor Virtual Constellation'), bitrate=1800)
    writer_comp = Writer(fps=FPS, metadata=dict(artist='MultiSensor Virtual Constellation'), bitrate=1800)
    writer_side = Writer(fps=FPS, metadata=dict(artist='MultiSensor Virtual Constellation'), bitrate=2500)
    
    prefix = f"MultiSensor_{COMPLEXITY_TYPE}_{START_DATE.strftime('%Y')}-{END_DATE.strftime('%Y')}_Unmasked_{FPS}fps"
    path_rgb = os.path.join(OUTPUT_DIR, f"{prefix}_TrueColor.mp4")
    path_comp = os.path.join(OUTPUT_DIR, f"{prefix}_Complexity.mp4")
    path_side = os.path.join(OUTPUT_DIR, f"{prefix}_SideBySide.mp4")

    # Set up Figures with Black Backgrounds
    fig_rgb, ax_rgb = plt.subplots(figsize=(8, 8), facecolor='black')
    fig_comp, ax_comp = plt.subplots(figsize=(8, 8), facecolor='black')
    fig_side, (ax_s1, ax_s2) = plt.subplots(1, 2, figsize=(16, 8), facecolor='black')
    
    # Aggressively crop out white space/margins
    # fig_side retains a top margin of 0.92 to ensure the subplot titles aren't cut off
    fig_rgb.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
    fig_comp.subplots_adjust(left=0.0, right=0.88, top=1.0, bottom=0.0)
    fig_side.subplots_adjust(left=0.0, right=0.92, top=0.92, bottom=0.0, wspace=0.02)
    
    # Aesthetic background for data voids
    for ax in [ax_rgb, ax_comp, ax_s1, ax_s2]:
        ax.set_facecolor('#2F4F4F') # Dark Slate Gray for masked/void pixels
        ax.axis('off')

    # Initialize Imshow objects
    dummy_shape = (100, 100) # Replaced dynamically on frame 1
    im_rgb = ax_rgb.imshow(np.zeros((*dummy_shape, 3)))
    im_comp = ax_comp.imshow(np.zeros(dummy_shape), cmap='viridis', vmin=v_min, vmax=v_max)
    im_s1 = ax_s1.imshow(np.zeros((*dummy_shape, 3)))
    im_s2 = ax_s2.imshow(np.zeros(dummy_shape), cmap='viridis', vmin=v_min, vmax=v_max)
    
    # Format Colorbars to contrast against the black background
    cbar_comp = fig_comp.colorbar(im_comp, ax=ax_comp, fraction=0.046, pad=0.04)
    cbar_comp.set_label("Complexity Volume", color='white', fontweight='bold')
    cbar_comp.ax.yaxis.set_tick_params(color='white', labelcolor='white')
    cbar_comp.outline.set_edgecolor('white')

    cbar_side = fig_side.colorbar(im_s2, ax=ax_s2, fraction=0.046, pad=0.04)
    cbar_side.set_label("Complexity Volume", color='white', fontweight='bold')
    cbar_side.ax.yaxis.set_tick_params(color='white', labelcolor='white')
    cbar_side.outline.set_edgecolor('white')

    # Initialize Time Annotations
    txt_rgb = ax_rgb.text(0.5, 0.02, "", transform=ax_rgb.transAxes, ha='center', va='bottom', color='white', fontsize=12, fontweight='bold', bbox=dict(facecolor='black', alpha=0.6, pad=3))
    txt_comp = ax_comp.text(0.5, 0.02, "", transform=ax_comp.transAxes, ha='center', va='bottom', color='white', fontsize=12, fontweight='bold', bbox=dict(facecolor='black', alpha=0.6, pad=3))
    # Positioned at 0.02 to ensure it hovers neatly inside the image rather than floating in a cropped margin
    txt_side = fig_side.text(0.5, 0.02, "", ha='center', va='bottom', color='white', fontsize=14, fontweight='bold', bbox=dict(facecolor='black', alpha=0.6, pad=5))
    
    ax_s1.set_title("True Color", color='white', fontweight='bold')
    ax_s2.set_title(f"Complexity ({COMPLEXITY_TYPE})", color='white', fontweight='bold')

    # Initialize Pixel Indicators (Empty lists to be updated per frame)
    inds_rgb, inds_comp, inds_s1, inds_s2 = [], [], [], []
    if SHOW_PIXEL_INDICATORS:
        for _ in TS_LOCATIONS:
            inds_rgb.append(ax_rgb.plot([], [], marker='s', markersize=12, markerfacecolor='none', markeredgewidth=2)[0])
            inds_comp.append(ax_comp.plot([], [], marker='s', markersize=12, markerfacecolor='none', markeredgewidth=2)[0])
            inds_s1.append(ax_s1.plot([], [], marker='s', markersize=12, markerfacecolor='none', markeredgewidth=2)[0])
            inds_s2.append(ax_s2.plot([], [], marker='s', markersize=12, markerfacecolor='none', markeredgewidth=2)[0])

    # 5. Execution Block
    print("Writing video streams...")
    with writer_rgb.saving(fig_rgb, path_rgb, DPI), \
         writer_comp.saving(fig_comp, path_comp, DPI), \
         writer_side.saving(fig_side, path_side, DPI):
        
        for i, frame in enumerate(unified_frames):
            idx = frame['idx']
            sensor = frame['sensor']
            grp = frame['grp']
            
            time_str = f"[{sensor}] Acquired: {frame['dt_et'].strftime('%Y-%m-%d %H:%M:%S ET')}"
            
            if i % 10 == 0:
                print(f"  Rendering frame {i}/{len(unified_frames)}...")
            
            # Extract RGB
            raw_sr = frame['sr_dset'][idx, ...]
            if sensor == 'LANDSAT':
                r = percentile_normalize_array(raw_sr[LANDSAT_RGB_BANDS[0]])
                g = percentile_normalize_array(raw_sr[LANDSAT_RGB_BANDS[1]])
                b = percentile_normalize_array(raw_sr[LANDSAT_RGB_BANDS[2]])
                rgb = np.stack([r, g, b], axis=-1)
            else: # TANAGER
                vis = grp['ortho_visual'][idx, ...]
                rgb = np.transpose(vis[:3, ...], (1, 2, 0))
            
            # Extract Complexity Map
            comp_data = frame['vol_dset'][idx, ...].copy()
            
            if not GLOBAL_COLOR_SCALE:
                with np.errstate(all='ignore'):
                    v_min, v_max = np.nanmin(comp_data), np.nanmax(comp_data)
                im_comp.set_clim(vmin=v_min, vmax=v_max)
                im_s2.set_clim(vmin=v_min, vmax=v_max)

            # Safely adapt axes limits and projection extent dynamically
            h, w = comp_data.shape
            dynamic_extent = [0, w, h, 0] # Mathematical mapping: [left, right, bottom, top]
            
            # Project the images strictly to the new spatial extent
            im_rgb.set_extent(dynamic_extent)
            im_comp.set_extent(dynamic_extent)
            im_s1.set_extent(dynamic_extent)
            im_s2.set_extent(dynamic_extent)

            # Update Images (Replace the underlying pixel array)
            im_rgb.set_data(rgb)
            im_comp.set_data(comp_data)
            im_s1.set_data(rgb)
            im_s2.set_data(comp_data)
            
            # Align the viewer axes to perfectly frame the true spatial extent
            for ax in [ax_rgb, ax_comp, ax_s1, ax_s2]:
                ax.set_xlim(0, w)
                ax.set_ylim(h, 0)
            
            # Update Indicators based on active sensor
            if SHOW_PIXEL_INDICATORS:
                current_locs = mapped_locs_l if sensor == 'LANDSAT' else mapped_locs_t
                for loc_idx, loc in enumerate(current_locs):
                    for ind_list in [inds_rgb, inds_comp, inds_s1, inds_s2]:
                        ind_list[loc_idx].set_data([loc['x']], [loc['y']])
                        ind_list[loc_idx].set_markeredgecolor(loc['color'])

            # Update Text
            txt_rgb.set_text(time_str)
            txt_comp.set_text(time_str)
            txt_side.set_text(time_str)
            
            # Write to video
            writer_rgb.grab_frame()
            writer_comp.grab_frame()
            writer_side.grab_frame()

    plt.close('all')
    h5_l.close()
    h5_t.close()
    print(f"\nSuccess! Virtual Constellation videos saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    generate_videos()