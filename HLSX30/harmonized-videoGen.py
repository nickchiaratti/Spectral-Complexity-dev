import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from datetime import datetime, timezone
from contextlib import ExitStack
from zoneinfo import ZoneInfo
import rasterio.transform
from pyproj import Transformer, CRS

# ==========================================
# 1. VIDEO & EXTRACTION CONFIGURATION
# ==========================================

background_color = 'w' # Dark Slate Gray shows through transparent pixels
text_color = 'black'

# File Paths
Location = "Rochesterv2"
# Point directly to the finalized ARD Master Cube that includes the Spectral Complexity datasets
ARD_CUBE_PATH = f"C:/satelliteImagery/HLSX30/HLS-Tanager_{Location}_Harmonized_2025_SC_EM-7_Norm-bandCount.h5"
OUTPUT_DIR = f"C:/satelliteImagery/MultiSensor_Analysis_{Location}_ARD_Videos"

COMPLEXITY_TYPE = 'sliding_volume_map'

# Temporal Configuration
START_DATE = datetime(2020, 1, 1, tzinfo=timezone.utc)
END_DATE = datetime(2025, 12, 31, tzinfo=timezone.utc)

# QA Filtering Configuration
EXCLUDE_CONTAMINATED_FRAMES = False
# Allowed percentage of cloudy/invalid pixels before dropping the entire frame (0.0 = Strict exclusion)
MAX_CONTAMINATION_FRACTION = 0.0

# Video Output Configuration
FPS = 3  
DPI = 300
EXPORT_GIF = False
GIF_DPI = 100 # Prevents RAM exhaustion during Pillow color quantization
SHOW_PIXEL_INDICATORS = True
GLOBAL_COLOR_SCALE = True # Prevents colormap "flickering" across the time series

# Time Series Locations (Latitude, Longitude)
TS_LOCATIONS = [
    {'latlon': (43.142856, -77.508451), 'label': "West Tait Forest",                'color': 'tab:green'},
    {'latlon': (43.144861, -77.501176), 'label': "East Tait Forest",                'color': 'tab:olive'},
    {'latlon': (43.136910, -77.469462), 'label': "Artificial turf football field",  'color': 'tab:blue'},
    {'latlon': (43.138241, -77.470873), 'label': "Recently added artificial turf",  'color': 'tab:cyan'},
    {'latlon': (43.141297, -77.506256), 'label': "Tait Parking Lot",                'color': 'tab:red'},
    {'latlon': (43.139411, -77.504005), 'label': "ROCX NITE Tarp",                  'color': 'tab:purple'},
]

# ==========================================
# 2. UTILITY FUNCTIONS
# ==========================================

def map_locations(dset):
    """Maps geographic lat/lon to specific pixel grid coordinates utilizing ARD metadata."""
    geo_transform = dset.attrs.get('GeoTransform')
    spatial_ref = dset.attrs.get('spatial_ref')
    mapped = []
    
    if geo_transform is None or spatial_ref is None:
        raise ValueError("CRITICAL ERROR: GeoTransform or spatial_ref missing from ARD dataset.")
        
    if isinstance(spatial_ref, bytes): spatial_ref = spatial_ref.decode('utf-8')
    crs = CRS.from_wkt(spatial_ref)
    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    
    affine = rasterio.transform.Affine.from_gdal(*geo_transform)
    inv_affine = ~affine
    
    for loc in TS_LOCATIONS:
        lat, lon = loc['latlon']
        px, py = inv_affine * transformer.transform(lon, lat)
        mapped.append({'label': loc['label'], 'color': loc['color'], 'y': int(round(py)), 'x': int(round(px))})
        
    return mapped

def decode_h5_string(raw_str):
    """Safely handles h5py byte-string vs native string anomalies."""
    return raw_str.decode('utf-8') if isinstance(raw_str, bytes) else str(raw_str)

# ==========================================
# 3. MAIN VIDEO GENERATOR
# ==========================================

def generate_videos():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Opening ARD Master Cube: {ARD_CUBE_PATH}")
    
    with h5py.File(ARD_CUBE_PATH, 'r') as h5_ard:
        
        # EVIDENCE-BASED FIX: Timeline Indexing via Provenance Attributes
        # The script now explicitly targets the HARMONIZED group as the primary driver,
        # using its embedded metadata to hunt for the required visual arrays in the native grids.
        harm_path = '/HDFEOS/GRIDS/HARMONIZED/Data Fields'
        if harm_path not in h5_ard:
            raise ValueError(f"CRITICAL ERROR: '{harm_path}' missing. Ensure Spectral Complexity script was run.")
            
        harm_grp = h5_ard[harm_path]
        
        if COMPLEXITY_TYPE not in harm_grp:
            raise ValueError(f"CRITICAL ERROR: '{COMPLEXITY_TYPE}' missing in HARMONIZED group.")
            
        vol_ds = harm_grp[COMPLEXITY_TYPE]
        qa_ds = harm_grp.get('common_mask')
        
        if qa_ds is None and EXCLUDE_CONTAMINATED_FRAMES:
            raise ValueError("CRITICAL ERROR: 'common_mask' missing from HARMONIZED group for QA filtering.")
            
        # Establish Unified Geometric Provenance from the HARMONIZED dataset
        mapped_locs = map_locations(vol_ds)
        
        # Extract Absolute Data Provenance Arrays
        try:
            prov_grids = vol_ds.attrs['source_grid']
            prov_spaces = vol_ds.attrs['source_spacecraft']
            prov_times = vol_ds.attrs['acquisition_time']
            prov_indices = vol_ds.attrs['source_frame_index']
        except KeyError as e:
            raise ValueError(f"CRITICAL ERROR: Missing provenance attribute {e} on HARMONIZED dataset.")
            
        unified_frames = []
        
        # Drive the loop using the harmonized timeline
        for global_idx in range(len(prov_times)):
            ts = prov_times[global_idx]
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            
            if START_DATE <= dt <= END_DATE:
                # Strict QA Enforcement using the harmonized mask
                if EXCLUDE_CONTAMINATED_FRAMES:
                    qa_frame = qa_ds[global_idx, ...]
                    bad_pixels = np.sum(qa_frame == 0)
                        
                    if (bad_pixels / qa_frame.size) > MAX_CONTAMINATION_FRACTION:
                        continue # Exclude frame
                        
                # Extract relational pointers
                source_grid = decode_h5_string(prov_grids[global_idx])
                source_spacecraft = decode_h5_string(prov_spaces[global_idx])
                local_idx = int(prov_indices[global_idx])
                
                # Retrieve the sensor-specific ortho visual
                ortho_path = f'/HDFEOS/GRIDS/{source_grid}/Data Fields/ortho_visual'
                if ortho_path not in h5_ard:
                    raise ValueError(f"CRITICAL ERROR: '{ortho_path}' missing for frame {global_idx}.")
                    
                ortho_ds = h5_ard[ortho_path]
                
                unified_frames.append({
                    'sensor': source_spacecraft,
                    'dt': dt, 
                    'dt_et': dt.astimezone(ZoneInfo("America/New_York")),
                    'vol_dset': vol_ds,        # Points to HARMONIZED array
                    'vol_idx': global_idx,     # Index in HARMONIZED array
                    'ortho_dset': ortho_ds,    # Points to NATIVE array
                    'ortho_idx': local_idx     # Index in NATIVE array
                })
                
        if not unified_frames:
            print("No valid frames found in the specified date range after strict QA filtering.")
            return
            
        print(f"Processing {len(unified_frames)} strictly interleaved frames based on HARMONIZED provenance...")

        # 3. Calculate Global Min/Max for Shared Complexity Colormap
        v_min, v_max = 0, 1
        if GLOBAL_COLOR_SCALE:
            print("Calculating mathematically shared global color scale percentiles across the timeline...")
            all_vols = []
            for frame in unified_frames:
                data = frame['vol_dset'][frame['vol_idx'], ...]
                valid_data = data[~np.isnan(data)]
                if len(valid_data) > 0:
                    all_vols.append(valid_data)
            
            if all_vols:
                flat_vols = np.concatenate(all_vols)
                v_min, v_max = np.nanpercentile(flat_vols, (2, 98))
                print(f"Global Color Scale Locked: vmin={v_min:.4f}, vmax={v_max:.4f}")

        # 4. Set up matplotlib writers
        Writer = animation.writers['ffmpeg']
        writer_rgb = Writer(fps=FPS, metadata=dict(artist='MultiSensor ARD Constellation'), bitrate=1800)
        writer_comp = Writer(fps=FPS, metadata=dict(artist='MultiSensor ARD Constellation'), bitrate=1800)
        writer_side = Writer(fps=FPS, metadata=dict(artist='MultiSensor ARD Constellation'), bitrate=2500)
        
        if EXPORT_GIF:
            GifWriter = animation.writers['pillow']
            writer_rgb_gif = GifWriter(fps=FPS)
            writer_comp_gif = GifWriter(fps=FPS)
            writer_side_gif = GifWriter(fps=FPS)
        
        qa_suffix = "_QAFilt" if EXCLUDE_CONTAMINATED_FRAMES else "_Unmasked"
        prefix = f"{Location}_ARD_Constellation_{COMPLEXITY_TYPE}_{START_DATE.strftime('%Y')}-{END_DATE.strftime('%Y')}{qa_suffix}_{FPS}fps"
        
        path_rgb = os.path.join(OUTPUT_DIR, f"{prefix}_TrueColor.mp4")
        path_comp = os.path.join(OUTPUT_DIR, f"{prefix}_Complexity.mp4")
        path_side = os.path.join(OUTPUT_DIR, f"{prefix}_SideBySide.mp4")
        
        if EXPORT_GIF:
            path_rgb_gif = os.path.join(OUTPUT_DIR, f"{prefix}_TrueColor.gif")
            path_comp_gif = os.path.join(OUTPUT_DIR, f"{prefix}_Complexity.gif")
            path_side_gif = os.path.join(OUTPUT_DIR, f"{prefix}_SideBySide.gif")

        # Set up Figures with Black Backgrounds
        fig_rgb, ax_rgb = plt.subplots(figsize=(8, 8), facecolor=background_color)
        fig_comp, ax_comp = plt.subplots(figsize=(8, 8), facecolor=background_color)
        fig_side, (ax_s1, ax_s2) = plt.subplots(1, 2, figsize=(16, 8), facecolor=background_color)
        
        fig_rgb.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
        fig_comp.subplots_adjust(left=0.0, right=0.88, top=1.0, bottom=0.0)
        fig_side.subplots_adjust(left=0.0, right=0.92, top=0.92, bottom=0.0, wspace=0.02)
        
        for ax in [ax_rgb, ax_comp, ax_s1, ax_s2]:
            ax.set_facecolor(background_color) 
            ax.axis('off')

        # Initialize Imshow objects
        dummy_shape = (100, 100)
        im_rgb = ax_rgb.imshow(np.zeros((*dummy_shape, 4))) 
        im_comp = ax_comp.imshow(np.zeros(dummy_shape), cmap='viridis', vmin=v_min, vmax=v_max)
        im_s1 = ax_s1.imshow(np.zeros((*dummy_shape, 4)))
        im_s2 = ax_s2.imshow(np.zeros(dummy_shape), cmap='viridis', vmin=v_min, vmax=v_max)
        
        cbar_comp = fig_comp.colorbar(im_comp, ax=ax_comp, fraction=0.046, pad=0.04)
        cbar_comp.set_label("Complexity Volume", color=text_color, fontweight='bold')
        cbar_comp.ax.yaxis.set_tick_params(color=text_color, labelcolor=text_color)
        cbar_comp.outline.set_edgecolor(text_color)

        cbar_side = fig_side.colorbar(im_s2, ax=ax_s2, fraction=0.046, pad=0.04)
        cbar_side.set_label("Complexity Volume", color=text_color, fontweight='bold')
        cbar_side.ax.yaxis.set_tick_params(color=text_color, labelcolor=text_color)
        cbar_side.outline.set_edgecolor(text_color)

        txt_rgb = ax_rgb.text(0.5, 0.02, "", transform=ax_rgb.transAxes, ha='center', va='bottom', color=text_color, fontsize=12, fontweight='bold', bbox=dict(facecolor=background_color, alpha=0.6, pad=3))
        txt_comp = ax_comp.text(0.5, 0.02, "", transform=ax_comp.transAxes, ha='center', va='bottom', color=text_color, fontsize=12, fontweight='bold', bbox=dict(facecolor=background_color, alpha=0.6, pad=3))
        txt_side = fig_side.text(0.5, 0.02, "", ha='center', va='bottom', color=text_color, fontsize=14, fontweight='bold', bbox=dict(facecolor=background_color, alpha=0.6, pad=5))
        
        ax_s1.set_title("Ortho Visual", color=text_color, fontweight='bold')
        ax_s2.set_title(f"Complexity ({COMPLEXITY_TYPE})", color=text_color, fontweight='bold')

        inds_rgb, inds_comp, inds_s1, inds_s2 = [], [], [], []
        if SHOW_PIXEL_INDICATORS:
            for _ in TS_LOCATIONS:
                inds_rgb.append(ax_rgb.plot([], [], marker='s', markersize=12, markerfacecolor='none', markeredgewidth=2)[0])
                inds_comp.append(ax_comp.plot([], [], marker='s', markersize=12, markerfacecolor='none', markeredgewidth=2)[0])
                inds_s1.append(ax_s1.plot([], [], marker='s', markersize=12, markerfacecolor='none', markeredgewidth=2)[0])
                inds_s2.append(ax_s2.plot([], [], marker='s', markersize=12, markerfacecolor='none', markeredgewidth=2)[0])

        # 5. Execution Block
        print("Writing video streams...")
        with ExitStack() as stack:
            stack.enter_context(writer_rgb.saving(fig_rgb, path_rgb, DPI))
            stack.enter_context(writer_comp.saving(fig_comp, path_comp, DPI))
            stack.enter_context(writer_side.saving(fig_side, path_side, DPI))
            
            if EXPORT_GIF:
                stack.enter_context(writer_rgb_gif.saving(fig_rgb, path_rgb_gif, GIF_DPI))
                stack.enter_context(writer_comp_gif.saving(fig_comp, path_comp_gif, GIF_DPI))
                stack.enter_context(writer_side_gif.saving(fig_side, path_side_gif, GIF_DPI))
            
            for i, frame in enumerate(unified_frames):
                sensor = frame['sensor']
                
                time_str = f"[{sensor}] Acquired: {frame['dt_et'].strftime('%Y-%m-%d %H:%M:%S ET')}"
                
                if i % 10 == 0:
                    print(f"  Rendering frame {i}/{len(unified_frames)}...")
                
                # --- 1. Process Ortho Visual (RGB + Alpha) from NATIVE dataset ---
                raw_ortho = frame['ortho_dset'][frame['ortho_idx'], ...]
                
                if raw_ortho.shape[0] in [3, 4]:
                    raw_ortho = np.transpose(raw_ortho, (1, 2, 0))
                    
                rgba = np.zeros((raw_ortho.shape[0], raw_ortho.shape[1], 4), dtype=np.float32)
                rgba[..., :3] = raw_ortho[..., :3] / 255.0
                    
                if raw_ortho.shape[-1] == 4:
                    user_alpha = raw_ortho[..., 3]
                    rgba[..., 3] = np.where(user_alpha > 0, 1.0, 0.0)
                else:
                    rgba[..., 3] = 1.0 
                    
                invalid_rgb_mask = np.isnan(rgba[..., 0])
                rgba[invalid_rgb_mask, 3] = 0.0
                rgba = np.nan_to_num(rgba, nan=0.0)
                
                # --- 2. Extract Complexity Map from HARMONIZED dataset ---
                comp_data = frame['vol_dset'][frame['vol_idx'], ...].copy()
                
                if raw_ortho.shape[-1] == 4:
                    comp_data[rgba[..., 3] == 0.0] = np.nan
                
                if not GLOBAL_COLOR_SCALE:
                    with np.errstate(all='ignore'):
                        v_min, v_max = np.nanmin(comp_data), np.nanmax(comp_data)
                    im_comp.set_clim(vmin=v_min, vmax=v_max)
                    im_s2.set_clim(vmin=v_min, vmax=v_max)

                h, w = comp_data.shape
                dynamic_extent = [0, w, h, 0] 
                
                im_rgb.set_extent(dynamic_extent)
                im_comp.set_extent(dynamic_extent)
                im_s1.set_extent(dynamic_extent)
                im_s2.set_extent(dynamic_extent)

                im_rgb.set_data(rgba)
                im_comp.set_data(comp_data)
                im_s1.set_data(rgba)
                im_s2.set_data(comp_data)
                
                for ax in [ax_rgb, ax_comp, ax_s1, ax_s2]:
                    ax.set_xlim(0, w)
                    ax.set_ylim(h, 0)
                
                # Coordinate mapping applies universally across the ARD framework
                if SHOW_PIXEL_INDICATORS:
                    for loc_idx, loc in enumerate(mapped_locs):
                        for ind_list in [inds_rgb, inds_comp, inds_s1, inds_s2]:
                            ind_list[loc_idx].set_data([loc['x'] + 0.5], [loc['y'] + 0.5])
                            ind_list[loc_idx].set_markeredgecolor(loc['color'])

                txt_rgb.set_text(time_str)
                txt_comp.set_text(time_str)
                txt_side.set_text(time_str)
                
                writer_rgb.grab_frame()
                writer_comp.grab_frame()
                writer_side.grab_frame()
                
                if EXPORT_GIF:
                    writer_rgb_gif.grab_frame()
                    writer_comp_gif.grab_frame()
                    writer_side_gif.grab_frame()

        plt.close('all')
        print(f"\nSuccess! Virtual Constellation videos saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    generate_videos()