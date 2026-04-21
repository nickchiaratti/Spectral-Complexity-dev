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

background_color = 'w' 
text_color = 'black'
TEXT_OVERLAY = True

Location = "Tait"
ARD_CUBE_PATH = f"C:/satelliteImagery/HLST30/HLST_{Location}_Harmonized_SC_EM-7_Norm-bandCount.h5"
OUTPUT_DIR = f"C:/satelliteImagery/HLST30/HLST_{Location}_Videos"

COMPLEXITY_TYPE = 'sliding_volume_z_score'

START_DATE = datetime(2020, 1, 1, tzinfo=timezone.utc)
END_DATE = datetime(2021, 12, 31, tzinfo=timezone.utc)

if Location == "Tait" or Location == "Rochesterv2":
    TS_LOCATIONS = [
    {'latlon': (43.142856, -77.508451), 'label': "West Tait Forest",                'color': 'tab:green'},
    {'latlon': (43.144861, -77.501176), 'label': "East Tait Forest",                'color': 'tab:olive'},
    {'latlon': (43.136910, -77.469462), 'label': "Artificial turf football field",  'color': 'tab:blue'},
    {'latlon': (43.138241, -77.470873), 'label': "Recently added artificial turf",  'color': 'tab:cyan'},
    {'latlon': (43.141297, -77.506256), 'label': "Tait Parking Lot",                'color': 'tab:red'},
    {'latlon': (43.139411, -77.504005), 'label': "ROCX NITE Tarp",                  'color': 'tab:purple'},
    ]
elif Location == "MtEtna":
    TS_LOCATIONS = [
    {'latlon': (37.738, 14.970), 'label': "left",                'color': 'tab:green'},
    {'latlon': (37.710, 15.000), 'label': "lower",                'color': 'tab:green'},
    {'latlon': (37.738, 15.04), 'label': "right",                'color': 'tab:olive'},
    {'latlon': (37.795, 15.005), 'label': "top",  'color': 'tab:blue'},
    ]

# ==========================================
# --- Coverage & QA Filtering Configuration ---
# Options: 
#   'NONE'       : Renders all frames regardless of cloud/shadow contamination.
#   'POI'        : (Recommended) Frame is rendered ONLY if ALL TS_LOCATIONS are within the frame's valid data coverage.
#   'PERCENTAGE' : Frame is rendered if the overall valid-pixel ratio exceeds MIN_FRAME_VALIDITY_PERCENTAGE.
# ==========================================
COVERAGE_EVALUATION_MODE = 'PERCENTAGE'

# If True, validates pixels against the QA mask (cloud/water filters), meaning data must be both present AND clear.
# If False, only checks that data physically exists (is not NaN / outside the valid satellite swath).
ENFORCE_QA_MASKING = False

# Used strictly if COVERAGE_EVALUATION_MODE = 'PERCENTAGE'
MIN_FRAME_VALIDITY_PERCENTAGE = .25

# Video Output Configuration
FPS = 2
DPI = 300
EXPORT_GIF = False
GIF_DPI = 80 
SHOW_PIXEL_INDICATORS = False

# --- Localized Color Scale Configuration ---
GLOBAL_COLOR_SCALE = True 
COLOR_SCALE_POI_RADIUS = 100 # Pixel radius around TS_LOCATIONS to sample for statistical color limits


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
    
    # Strict Configuration Validation
    valid_modes = ['NONE', 'POI', 'PERCENTAGE']
    if COVERAGE_EVALUATION_MODE not in valid_modes:
        raise ValueError(f"CRITICAL ERROR: '{COVERAGE_EVALUATION_MODE}' is not a recognized COVERAGE_EVALUATION_MODE. Must be one of {valid_modes}.")

    with h5py.File(ARD_CUBE_PATH, 'r') as h5_ard:
        
        harm_path = '/HDFEOS/GRIDS/HARMONIZED/Data Fields'
        if harm_path not in h5_ard:
            raise ValueError(f"CRITICAL ERROR: '{harm_path}' missing. Ensure Spectral Complexity script was run.")
            
        harm_grp = h5_ard[harm_path]
        
        if COMPLEXITY_TYPE not in harm_grp:
            raise ValueError(f"CRITICAL ERROR: '{COMPLEXITY_TYPE}' missing in HARMONIZED group.")
            
        vol_ds = harm_grp[COMPLEXITY_TYPE]
        qa_ds = harm_grp.get('common_mask')
        
        if qa_ds is None and ENFORCE_QA_MASKING:
            raise ValueError(f"CRITICAL ERROR: 'common_mask' missing from HARMONIZED group. Cannot perform QA masking.")
            
        mapped_locs = map_locations(vol_ds)
        height, width = qa_ds.shape[1], qa_ds.shape[2]
        
        # Strict Geographic Bounds Guardrail
        for loc in mapped_locs:
            if not (0 <= loc['y'] < height and 0 <= loc['x'] < width):
                raise ValueError(f"CRITICAL ERROR: TS_LOCATION '{loc['label']}' mathematically falls outside the ARD Cube dimensions ({height}x{width}). Check your ROI or Lat/Lon configurations.")
        
        try:
            prov_grids = vol_ds.attrs['source_grid']
            prov_spaces = vol_ds.attrs['source_spacecraft']
            prov_times = vol_ds.attrs['acquisition_time']
            prov_indices = vol_ds.attrs['source_frame_index']
        except KeyError as e:
            raise ValueError(f"CRITICAL ERROR: Missing provenance attribute {e} on HARMONIZED dataset.")
            
        unified_frames = []
        dropped_due_to_qa = 0
        
        # Pre-extract POI indices for rapid vectorized checking
        poi_y = [loc['y'] for loc in mapped_locs]
        poi_x = [loc['x'] for loc in mapped_locs]
        
        qa_filter_str = "with QA Masking" if ENFORCE_QA_MASKING else "without QA Masking"
        print(f"Applying Strict Coverage Filter: {COVERAGE_EVALUATION_MODE} ({qa_filter_str})")
        
        for global_idx in range(len(prov_times)):
            ts = prov_times[global_idx]
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            
            if START_DATE <= dt <= END_DATE:
                # ----------------------------------------------------
                # CORE QUALITY ASSESSMENT: FRAME COVERAGE EVALUATION
                # ----------------------------------------------------
                if COVERAGE_EVALUATION_MODE != 'NONE':
                    vol_frame = vol_ds[global_idx, ...]
                    
                    if ENFORCE_QA_MASKING:
                        qa_frame = qa_ds[global_idx, ...]
                    
                    if COVERAGE_EVALUATION_MODE == 'PERCENTAGE':
                        if ENFORCE_QA_MASKING:
                            valid_frac = np.sum(qa_frame == 1) / qa_frame.size
                        else:
                            valid_frac = np.sum(~np.isnan(vol_frame)) / vol_frame.size
                            
                        if valid_frac < MIN_FRAME_VALIDITY_PERCENTAGE:
                            dropped_due_to_qa += 1
                            continue 
                            
                    elif COVERAGE_EVALUATION_MODE == 'POI':
                        # Vectorized check: Guarantees every specified POI has data (is not NaN)
                        poi_valid = ~np.isnan(vol_frame[poi_y, poi_x])
                        
                        if ENFORCE_QA_MASKING:
                            qa_valid = (qa_frame[poi_y, poi_x] == 1)
                            poi_valid = poi_valid & qa_valid
                            
                        if not np.all(poi_valid):
                            dropped_due_to_qa += 1
                            continue
                
                source_grid = decode_h5_string(prov_grids[global_idx])
                source_spacecraft = decode_h5_string(prov_spaces[global_idx])
                local_idx = int(prov_indices[global_idx])
                
                ortho_path = f'/HDFEOS/GRIDS/{source_grid}/Data Fields/ortho_visual'
                if ortho_path not in h5_ard:
                    raise ValueError(f"CRITICAL ERROR: '{ortho_path}' missing for frame {global_idx}.")
                    
                ortho_ds = h5_ard[ortho_path]
                
                unified_frames.append({
                    'sensor': source_spacecraft,
                    'dt': dt, 
                    'dt_et': dt.astimezone(ZoneInfo("UTC")),
                    'vol_dset': vol_ds,        
                    'vol_idx': global_idx,     
                    'ortho_dset': ortho_ds,    
                    'ortho_idx': local_idx     
                })
                
        if not unified_frames:
            print(f"Zero valid frames passed the {COVERAGE_EVALUATION_MODE} filter. ({dropped_due_to_qa} frames excluded).")
            return
            
        print(f"Coverage Evaluation Complete. Kept {len(unified_frames)} frames. Excluded {dropped_due_to_qa} frames.")

        # 3. Calculate Global Min/Max for Shared Complexity Colormap
        v_min, v_max = 0, 1
        if GLOBAL_COLOR_SCALE:
            print(f"Calculating localized color scale (Mean ± 1 StdDev) within {COLOR_SCALE_POI_RADIUS}px of TS_LOCATIONS...")
            
            # Generate spatial mask defining the exact POI radii
            Y, X = np.ogrid[:height, :width]
            poi_mask = np.zeros((height, width), dtype=bool)
            for loc in mapped_locs:
                dist_sq = (Y - loc['y'])**2 + (X - loc['x'])**2
                poi_mask |= (dist_sq <= COLOR_SCALE_POI_RADIUS**2)
            
            all_vols = []
            for frame in unified_frames:
                data = frame['vol_dset'][frame['vol_idx'], ...]
                
                # Extract valid pixels STRICTLY within the POI masking radii
                valid_data = data[poi_mask & ~np.isnan(data)]
                if len(valid_data) > 0:
                    all_vols.append(valid_data)
            
            if all_vols:
                flat_vols = np.concatenate(all_vols)
                
                # Mathematical bounds: Mean ± 1 Standard Deviation
                mean_vol = np.mean(flat_vols)
                std_vol = np.std(flat_vols, ddof=1)
                
                v_min = mean_vol - std_vol
                v_max = mean_vol + std_vol
                
                print(f"Global Color Scale Locked (Local POI subset): vmin={v_min:.4f}, vmax={v_max:.4f} (Mean={mean_vol:.4f}, Std={std_vol:.4f})")
            else:
                print("WARNING: No valid data found near TS_LOCATIONS. Defaulting to 0-1.")

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
        
        # Dynamic Filename Suffix based on Filter Configuration
        if COVERAGE_EVALUATION_MODE == 'POI':
            qa_suffix = "_QA-POI-Strict" if ENFORCE_QA_MASKING else "_Coverage-POI"
        elif COVERAGE_EVALUATION_MODE == 'PERCENTAGE':
            qa_suffix = f"_QA-{int(MIN_FRAME_VALIDITY_PERCENTAGE*100)}pct" if ENFORCE_QA_MASKING else f"_Coverage-{int(MIN_FRAME_VALIDITY_PERCENTAGE*100)}pct"
        else:
            qa_suffix = "_Unmasked"
            
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
                
                if SHOW_PIXEL_INDICATORS:
                    for loc_idx, loc in enumerate(mapped_locs):
                        for ind_list in [inds_rgb, inds_comp, inds_s1, inds_s2]:
                            ind_list[loc_idx].set_data([loc['x'] + 0.5], [loc['y'] + 0.5])
                            ind_list[loc_idx].set_markeredgecolor(loc['color'])

                if TEXT_OVERLAY:
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