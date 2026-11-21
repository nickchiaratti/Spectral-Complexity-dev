import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import matplotlib.patheffects as pe
from datetime import datetime, timezone
from contextlib import ExitStack
from zoneinfo import ZoneInfo
import rasterio.transform
from pyproj import Transformer, CRS
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ==========================================
# 1. VIDEO & EXTRACTION CONFIGURATION
# ==========================================

background_color = 'w' 
text_color = 'black'
TEXT_OVERLAY = True

Location = "Rochesterv2"
ARD_CUBE_PATH = f"C:/satelliteImagery/HLST30/HLST_{Location}.h5"
OUTPUT_DIR = f"C:/satelliteImagery/HLST30/"

COMPLEXITY_TYPE = 'sliding_volume_z_score' #'sliding_volume_map'
SOURCE_GRID = 'HLSL30' #None # Set to a specific grid (e.g., 'HLSL30') or None for all

START_DATE = datetime(2015, 4, 1, tzinfo=timezone.utc)
END_DATE = datetime(2026, 8, 1, tzinfo=timezone.utc)

# ==========================================
# RGB / Spectral Band Source Configuration
# Options:
#   'ortho_visual' or None : Uses precomputed '/HDFEOS/GRIDS/HARMONIZED/Data Fields/ortho_visual'
#   Single wavelength in nm (e.g. 665 or [665]) : Grayscale display of closest band
#   3 wavelengths in nm (e.g. [665, 560, 490]) : 3-band composite mapped to [Red, Green, Blue] channels
#     Note on Channel Ordering: Remote sensing standards map list elements to [Red, Green, Blue].
#     True Color (natural color): [665, 560, 490] (Red=665nm, Green=560nm, Blue=490nm).
#     Specifying [490, 560, 665] maps 490nm (Blue) to Red and 665nm (Red) to Blue (BGR display).
# ==========================================
RGB_SOURCE = 'ortho_visual'
# RGB_SOURCE = [490, 560, 665] # 3 wavelengths in nm to use as RGB or 1 wavelength in nm for grayscale
# RGB_SOURCE = [443]

if Location == "Tait" or Location == "Rochesterv2":
    TS_LOCATIONS = [
    #{'latlon': (43.142856, -77.508451), 'label': "West Tait Forest",                'color': 'tab:green'},
    #{'latlon': (43.144861, -77.501176), 'label': "East Tait Forest",                'color': 'tab:olive'},
    #{'latlon': (43.136910, -77.469462), 'label': "Artificial turf football field",  'color': 'tab:blue'},
    #{'latlon': (43.138241, -77.470873), 'label': "Recently added artificial turf",  'color': 'tab:cyan'},
    #{'latlon': (43.141297, -77.506256), 'label': "Tait Parking Lot",                'color': 'tab:red'},
    {'latlon': (43.139411, -77.504005), 'label': "ROCX NITE Tarp",                  'color': 'tab:purple'},
    ]
elif Location == "MtEtna" or Location == "MtEtna-Catania":
    TS_LOCATIONS = [
    {'latlon': (37.738, 14.970), 'label': "left",                'color': 'tab:green'},
    {'latlon': (37.710, 15.000), 'label': "lower",                'color': 'tab:green'},
    {'latlon': (37.738, 15.04), 'label': "right",                'color': 'tab:olive'},
    {'latlon': (37.795, 15.005), 'label': "top",  'color': 'tab:blue'},
    ]
else:
    TS_LOCATIONS = []

# ==========================================
# Coverage & QA Filtering Configuration
# Options: 
#   'NONE'       : Renders all frames regardless of cloud/shadow contamination.
#   'POI'        : Frame is rendered ONLY if ALL TS_LOCATIONS are within the frame's valid data coverage.
#   'PERCENTAGE' : Frame is rendered if the overall valid-pixel ratio exceeds MIN_FRAME_VALIDITY_PERCENTAGE.
# ==========================================
COVERAGE_EVALUATION_MODE = 'PERCENTAGE'

# If True, validates pixels against the QA mask (cloud, shadow, cirrus, aerosol filters).
# Pixels belonging to persistent/per-frame water bodies are excluded from the validity penalty
# so that non-water land coverage is evaluated against aerosols and cloud contamination.
# If False, only checks that data physically exists (is not NaN / outside the valid satellite swath).
ENFORCE_QA_MASKING = True

# Used strictly if COVERAGE_EVALUATION_MODE = 'PERCENTAGE'
# Threshold of valid non-water pixels (0.0 to 1.0 or 0 to 100 percent) required to render the frame.
MIN_FRAME_VALIDITY_PERCENTAGE = .95

# Video Output Configuration
FPS = 2
DPI = 60
EXPORT_GIF = False  
GIF_DPI = 60
SHOW_PIXEL_INDICATORS = False

# Localized Color Scale Configuration
GLOBAL_COLOR_SCALE = False 
COLOR_SCALE_POI_RADIUS = 100 # Pixel radius around TS_LOCATIONS to sample for statistical color limits


# ==========================================
# 2. UTILITY FUNCTIONS & DATA TYPE HANDLING
# ==========================================

def get_dataset_metadata(dataset, require_scale=True):
    """
    Extracts datatype, nodata/fill value, and scale factor from HDF5 dataset attributes.
    Fails explicitly if an integer dataset lacks mandatory fillvalue or scale factor
    (when require_scale=True).
    """
    dtype = dataset.dtype
    attrs = dataset.attrs if hasattr(dataset, 'attrs') else {}

    # 1. Fill value extraction
    fill_val = None
    if "_FillValue" in attrs:
        fill_val = attrs["_FillValue"]
    elif "fillvalue" in attrs:
        fill_val = attrs["fillvalue"]
    elif "fill_value" in attrs:
        fill_val = attrs["fill_value"]

    if fill_val is None and hasattr(dataset, "fillvalue") and dataset.fillvalue is not None:
        if np.issubdtype(dtype, np.floating) or dataset.fillvalue != 0:
            fill_val = dataset.fillvalue

    if np.issubdtype(dtype, np.integer):
        if fill_val is None:
            raise KeyError(
                f"Dataset '{getattr(dataset, 'name', 'dataset')}' has integer datatype ({dtype}) "
                "but lacks mandatory '_FillValue' or 'fillvalue' attribute."
            )
        fill_val = int(fill_val)
    elif np.issubdtype(dtype, np.floating):
        if fill_val is not None:
            fill_val = float(fill_val)

    # 2. Scale factor extraction
    scale = 1.0
    if np.issubdtype(dtype, np.integer):
        if "scale_to_float" in attrs:
            scale = float(attrs["scale_to_float"])
        elif "scale_factor" in attrs:
            sf = float(attrs["scale_factor"])
            scale = (1.0 / sf) if sf > 1.0 else sf
        elif require_scale:
            raise KeyError(
                f"Dataset '{getattr(dataset, 'name', 'dataset')}' has integer datatype ({dtype}) "
                "but lacks mandatory 'scale_factor' or 'scale_to_float' attribute."
            )
    elif np.issubdtype(dtype, np.floating):
        if "scale_to_float" in attrs:
            scale = float(attrs["scale_to_float"])
        elif "scale_factor" in attrs:
            sf = float(attrs["scale_factor"])
            scale = (1.0 / sf) if sf > 1.0 else sf

    return {
        'dtype': dtype,
        'fill_val': fill_val,
        'scale': scale,
        'is_integer': np.issubdtype(dtype, np.integer),
        'is_floating': np.issubdtype(dtype, np.floating)
    }

def load_scaled_frame(dataset, frame_idx, meta=None):
    """
    Loads a single 2D frame from an HDF5 dataset, converting integer data to float32
    and substituting nodata/fill values with np.nan without using assumed defaults.
    """
    if meta is None:
        meta = get_dataset_metadata(dataset)

    raw_frame = dataset[frame_idx, ...]

    if meta['is_integer']:
        nodata = meta['fill_val']
        scale = meta['scale']
        is_nodata = (raw_frame == nodata)
        float_frame = raw_frame.astype(np.float32)
        float_frame[is_nodata] = np.nan
        float_frame *= scale
        return float_frame

    if meta['is_floating']:
        float_frame = raw_frame.astype(np.float32)
        nodata = meta['fill_val']
        if nodata is not None and not np.isnan(nodata):
            float_frame[np.isclose(raw_frame, nodata, equal_nan=True)] = np.nan
        if meta['scale'] != 1.0:
            float_frame *= meta['scale']
        return float_frame

    return raw_frame.astype(np.float32)

def map_locations(dset):
    """Maps geographic lat/lon to specific pixel grid coordinates utilizing ARD metadata."""
    geo_transform = dset.attrs.get('GeoTransform')
    spatial_ref = dset.attrs.get('spatial_ref')
    mapped = []
    
    if geo_transform is None or spatial_ref is None:
        raise ValueError("GeoTransform or spatial_ref missing from ARD dataset.")
        
    if isinstance(spatial_ref, bytes):
        spatial_ref = spatial_ref.decode('utf-8')
    crs = CRS.from_user_input(spatial_ref)
    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    
    affine = rasterio.transform.Affine.from_gdal(*geo_transform)
    inv_affine = ~affine
    
    for loc in TS_LOCATIONS:
        lat, lon = loc['latlon']
        px, py = inv_affine * transformer.transform(lon, lat)
        mapped.append({'label': loc['label'], 'color': loc['color'], 'y': int(round(py)), 'x': int(round(px))})
        
    return mapped

def decode_h5_string(raw_str):
    """Safely handles h5py byte-string vs native string representations."""
    if hasattr(raw_str, 'decode'):
        return raw_str.decode('utf-8')
    return str(raw_str)

def parse_rgb_source(rgb_source):
    """
    Parses and validates RGB_SOURCE configuration.
    Returns:
        mode: 'ortho_visual', 'single_band', or 'rgb_bands'
        wavelengths: None, float (nm), or tuple of 3 floats (wl_r, wl_g, wl_b) in nm
    """
    if rgb_source is None:
        return 'ortho_visual', None
        
    if isinstance(rgb_source, str):
        cleaned = rgb_source.strip().lower()
        if cleaned in ['ortho_visual', 'ortho', 'none']:
            return 'ortho_visual', None
        raise ValueError(
            f"Unrecognized string for RGB_SOURCE: '{rgb_source}'. "
            "Set RGB_SOURCE = 'ortho_visual', RGB_SOURCE = None, a single wavelength number, or a list/tuple of 3 wavelengths."
        )
        
    if isinstance(rgb_source, dict):
        keys = {k.lower(): k for k in rgb_source}
        if 'r' in keys and 'g' in keys and 'b' in keys:
            wl_r = float(rgb_source[keys['r']])
            wl_g = float(rgb_source[keys['g']])
            wl_b = float(rgb_source[keys['b']])
            wl_r = wl_r * 1000.0 if wl_r < 20.0 else wl_r
            wl_g = wl_g * 1000.0 if wl_g < 20.0 else wl_g
            wl_b = wl_b * 1000.0 if wl_b < 20.0 else wl_b
            return 'rgb_bands', (wl_r, wl_g, wl_b)
        raise ValueError(f"Dictionary RGB_SOURCE must contain 'R', 'G', 'B' keys: {rgb_source}")
        
    if isinstance(rgb_source, (list, tuple, np.ndarray)):
        arr = list(rgb_source)
        if len(arr) == 1:
            wl = float(arr[0])
            wl = wl * 1000.0 if wl < 20.0 else wl
            return 'single_band', wl
        elif len(arr) == 3:
            wl_r = float(arr[0]) * 1000.0 if float(arr[0]) < 20.0 else float(arr[0])
            wl_g = float(arr[1]) * 1000.0 if float(arr[1]) < 20.0 else float(arr[1])
            wl_b = float(arr[2]) * 1000.0 if float(arr[2]) < 20.0 else float(arr[2])
            return 'rgb_bands', (wl_r, wl_g, wl_b)
        else:
            raise ValueError(
                f"RGB_SOURCE list/tuple must contain either 1 wavelength (grayscale) or 3 wavelengths [R, G, B], got {len(arr)}: {rgb_source}"
            )
            
    if isinstance(rgb_source, (int, float, np.number)):
        wl = float(rgb_source)
        wl = wl * 1000.0 if wl < 20.0 else wl
        return 'single_band', wl
        
    raise TypeError(
        f"Unsupported type for RGB_SOURCE: {type(rgb_source)}. Must be string ('ortho_visual'), None, number, or list/tuple of 1 or 3 wavelengths."
    )

def get_grid_wavelengths_nm(sr_dset, source_grid):
    """
    Extracts sensor center wavelengths and standardizes them to nanometer (nm) units.
    Wavelengths are stored as a 'wavelengths' attribute on surface_reflectance,
    stored in micrometer units for HLS grids and nanometer units otherwise.
    """
    if 'wavelengths' not in sr_dset.attrs:
        raise KeyError(
            f"Dataset '{sr_dset.name}' missing mandatory 'wavelengths' attribute."
        )
    raw_wl = np.array(sr_dset.attrs['wavelengths'], dtype=np.float32)
    
    if 'HLS' in source_grid.upper():
        # HLS grids (e.g. HLSL30, HLSS30) store center wavelengths in micrometers
        return raw_wl * 1000.0
    else:
        # Non-HLS sensors store wavelengths in nanometers (check explicit units attribute if present)
        attrs = sr_dset.attrs
        unit = decode_h5_string(attrs.get('wavelengths_units', '')).lower()
        if unit in ['um', 'micrometer', 'micrometers', 'micron', 'microns'] or (raw_wl.size > 0 and np.max(raw_wl) < 20.0):
            return raw_wl * 1000.0
        return raw_wl

def find_closest_band(grid_wl_nm, target_wl_nm):
    """Finds index and actual center wavelength of closest spectral band to target wavelength."""
    diffs = np.abs(grid_wl_nm - target_wl_nm)
    idx = int(np.argmin(diffs))
    return idx, float(grid_wl_nm[idx])

def load_scaled_band(sr_dset, time_idx, band_idx, meta=None):
    """
    Loads a single 2D band (Y, X) at time_idx from a 4D surface_reflectance HDF5 dataset,
    converting integer data to float32 and substituting nodata/fill values with np.nan.
    """
    if meta is None:
        meta = get_dataset_metadata(sr_dset)

    raw_band = sr_dset[time_idx, band_idx, :, :]

    if meta['is_integer']:
        nodata = meta['fill_val']
        scale = meta['scale']
        is_nodata = (raw_band == nodata)
        float_band = raw_band.astype(np.float32)
        float_band[is_nodata] = np.nan
        float_band *= scale
        return float_band

    if meta['is_floating']:
        float_band = raw_band.astype(np.float32)
        nodata = meta['fill_val']
        if nodata is not None and not np.isnan(nodata):
            float_band[np.isclose(raw_band, nodata, equal_nan=True)] = np.nan
        if meta['scale'] != 1.0:
            float_band *= meta['scale']
        return float_band

    return raw_band.astype(np.float32)

def generate_single_band_rgba(band_data, low=0.5, high=99.5, gamma=1.1):
    """
    Stretches a single 2D reflectance band into a normalized grayscale RGBA image.
    Invalid / NaN pixels receive alpha = 0.0.
    """
    height, width = band_data.shape
    valid_mask = ~np.isnan(band_data) & (band_data != 0)
    
    rgba = np.zeros((height, width, 4), dtype=np.float32)
    valid_pixels = band_data[valid_mask]
    
    if valid_pixels.size == 0:
        return rgba
        
    p_low, p_high = np.percentile(valid_pixels, (low, high))
    if p_low < p_high:
        stretched = np.clip((band_data - p_low) / (p_high - p_low), 0.0, 1.0)
        if gamma != 1.0:
            with np.errstate(invalid='ignore', divide='ignore'):
                stretched = np.power(stretched, 1.0 / gamma)
                stretched = np.nan_to_num(stretched, nan=0.0, posinf=1.0, neginf=0.0)
    else:
        stretched = np.clip(band_data, 0.0, 1.0)
        
    rgba[..., 0] = stretched
    rgba[..., 1] = stretched
    rgba[..., 2] = stretched
    rgba[..., 3] = np.where(valid_mask, 1.0, 0.0)
    return rgba

def generate_multiband_rgba(r_band, g_band, b_band, low=0.5, high=99.5, gamma=1.1):
    """
    Extracts, stretches, and gamma-corrects 3 reflectance bands into an RGBA composite.
    Calculates joint percentiles across all 3 bands to preserve relative color balance.
    Invalid / NaN pixels receive alpha = 0.0.
    """
    height, width = r_band.shape
    frame_rgb = np.stack([r_band, g_band, b_band], axis=0)
    
    all_zeros = np.all(frame_rgb == 0, axis=0)
    has_nan = np.any(np.isnan(frame_rgb), axis=0)
    valid_mask = ~(all_zeros | has_nan)
    
    rgba = np.zeros((height, width, 4), dtype=np.float32)
    valid_pixels = np.concatenate([
        r_band[valid_mask],
        g_band[valid_mask],
        b_band[valid_mask]
    ])
    
    if valid_pixels.size == 0:
        return rgba
        
    p_low, p_high = np.percentile(valid_pixels, (low, high))
    for i, band in enumerate([r_band, g_band, b_band]):
        if p_low < p_high:
            stretched = np.clip((band - p_low) / (p_high - p_low), 0.0, 1.0)
            if gamma != 1.0:
                with np.errstate(invalid='ignore', divide='ignore'):
                    stretched = np.power(stretched, 1.0 / gamma)
                    stretched = np.nan_to_num(stretched, nan=0.0, posinf=1.0, neginf=0.0)
        else:
            stretched = np.clip(band, 0.0, 1.0)
        rgba[..., i] = stretched
        
    rgba[..., 3] = np.where(valid_mask, 1.0, 0.0)
    return rgba

# ==========================================
# 3. MAIN VIDEO GENERATOR
# ==========================================

def generate_videos():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Opening ARD Master Cube: {ARD_CUBE_PATH}")
    
    # Strict Configuration Validation
    valid_modes = ['NONE', 'POI', 'PERCENTAGE']
    if COVERAGE_EVALUATION_MODE not in valid_modes:
        raise ValueError(f"'{COVERAGE_EVALUATION_MODE}' is not a recognized COVERAGE_EVALUATION_MODE. Must be one of {valid_modes}.")

    try:
        h5_ard = h5py.File(ARD_CUBE_PATH, 'r', locking=False)
    except TypeError:
        h5_ard = h5py.File(ARD_CUBE_PATH, 'r')

    with h5_ard:
        harm_path = '/HDFEOS/GRIDS/HARMONIZED/Data Fields'
        if harm_path not in h5_ard:
            raise ValueError(f"'{harm_path}' missing. Ensure Spectral Complexity script was run.")
            
        harm_grp = h5_ard[harm_path]
        
        target_metric = COMPLEXITY_TYPE
        if target_metric not in harm_grp:
            # Check for known naming aliases across pipeline versions
            if target_metric == 'ndvi' and 'ndvi_map' in harm_grp:
                target_metric = 'ndvi_map'
            elif target_metric == 'ndvi_map' and 'ndvi' in harm_grp:
                target_metric = 'ndvi'
            else:
                raise ValueError(f"Dataset '{target_metric}' missing in HARMONIZED group. Available datasets: {list(harm_grp.keys())}")
            
        vol_ds = harm_grp[target_metric]
        qa_ds = harm_grp.get('common_mask')
        
        if qa_ds is None and ENFORCE_QA_MASKING:
            raise ValueError("'common_mask' missing from HARMONIZED group. Cannot perform QA masking.")
            
        # Locate GeoTransform and spatial_ref
        geo_ds = vol_ds
        if 'GeoTransform' not in geo_ds.attrs or 'spatial_ref' not in geo_ds.attrs:
            found = False
            for ds_name in harm_grp.keys():
                candidate = harm_grp[ds_name]
                if 'GeoTransform' in candidate.attrs and 'spatial_ref' in candidate.attrs:
                    geo_ds = candidate
                    found = True
                    break
            if not found:
                for g in h5_ard['/HDFEOS/GRIDS'].keys():
                    grp_g = h5_ard[f'/HDFEOS/GRIDS/{g}']
                    target_grp = grp_g['Data Fields'] if 'Data Fields' in grp_g else grp_g
                    for ds_name in target_grp.keys():
                        candidate = target_grp[ds_name]
                        if isinstance(candidate, h5py.Dataset) and 'GeoTransform' in candidate.attrs and 'spatial_ref' in candidate.attrs:
                            geo_ds = candidate
                            found = True
                            break
                    if found:
                        break

        mapped_locs = map_locations(geo_ds)
        height, width = vol_ds.shape[1], vol_ds.shape[2]
        
        # Strict Geographic Bounds Guardrail
        for loc in mapped_locs:
            if not (0 <= loc['y'] < height and 0 <= loc['x'] < width):
                raise ValueError(f"TS_LOCATION '{loc['label']}' mathematically falls outside the ARD Cube dimensions ({height}x{width}). Check ROI or Lat/Lon configurations.")
        
        def get_provenance_attr(attr_name):
            if attr_name in vol_ds.attrs:
                return vol_ds.attrs[attr_name]
            if qa_ds is not None and attr_name in qa_ds.attrs:
                return qa_ds.attrs[attr_name]
            for ds_name in harm_grp.keys():
                candidate = harm_grp[ds_name]
                if attr_name in candidate.attrs:
                    return candidate.attrs[attr_name]
            raise KeyError(f"Missing provenance attribute '{attr_name}' on HARMONIZED dataset.")

        prov_grids = get_provenance_attr('source_grid')
        prov_spaces = get_provenance_attr('source_spacecraft')
        prov_times = get_provenance_attr('acquisition_time')
        prov_indices = get_provenance_attr('source_frame_index')
            
        vol_meta = get_dataset_metadata(vol_ds)
        qa_meta = get_dataset_metadata(qa_ds, require_scale=False) if qa_ds is not None else None
        qa_fill = qa_meta['fill_val'] if qa_meta is not None else None

        # Configure Visual / RGB Imagery Source
        rgb_mode, target_wls = parse_rgb_source(RGB_SOURCE)
        if rgb_mode == 'ortho_visual':
            print("Configured RGB_SOURCE: Precomputed ortho_visual (/HDFEOS/GRIDS/HARMONIZED/Data Fields/ortho_visual)")
            ortho_path = '/HDFEOS/GRIDS/HARMONIZED/Data Fields/ortho_visual'
            if ortho_path not in h5_ard:
                raise ValueError(f"Ortho dataset '{ortho_path}' missing in ARD Cube. Ensure ortho_visual was generated or configure RGB_SOURCE with target wavelengths.")
            ortho_ds = h5_ard[ortho_path]
        elif rgb_mode == 'single_band':
            print(f"Configured RGB_SOURCE: Single band grayscale at ~{target_wls:.1f} nm")
            ortho_ds = None
        else:
            r_w, g_w, b_w = target_wls
            print(f"Configured RGB_SOURCE: 3-band composite [Red: ~{r_w:.1f}nm, Green: ~{g_w:.1f}nm, Blue: ~{b_w:.1f}nm]")
            ortho_ds = None

        sr_handles = {}
        sr_meta_cache = {}
        sr_band_map = {}

        unified_frames = []
        dropped_due_to_qa = 0
        
        # Pre-extract POI indices for rapid vectorized checking
        poi_y = [loc['y'] for loc in mapped_locs]
        poi_x = [loc['x'] for loc in mapped_locs]
        
        qa_filter_str = "with QA Masking" if ENFORCE_QA_MASKING else "without QA Masking"
        print(f"Applying Strict Coverage Filter: {COVERAGE_EVALUATION_MODE} ({qa_filter_str})")

        # Locate or precalculate persistent water mask so water bodies do not penalize frame validity
        persistent_water = None
        for candidate_path in [
            '/HDFEOS/GRIDS/HARMONIZED/Data Fields/persistent_water_mask',
            '/HDFEOS/GRIDS/HARMONIZED/Data Fields/water_mask',
            '/PYCCD_Results/water_probability',
        ]:
            if candidate_path in h5_ard:
                d = h5_ard[candidate_path]
                if d.shape == (height, width):
                    persistent_water = (d[:] > 0.5) if 'probability' in candidate_path else (d[:] > 0)
                    break

        if persistent_water is None:
            for g in h5_ard['/HDFEOS/GRIDS'].keys():
                for candidate_name in ['persistent_water_mask', 'water_mask']:
                    candidate_path = f"/HDFEOS/GRIDS/{g}/Data Fields/{candidate_name}"
                    if candidate_path in h5_ard:
                        d = h5_ard[candidate_path]
                        if d.shape == (height, width):
                            persistent_water = (d[:] > 0)
                            break
                if persistent_water is not None:
                    break

        if persistent_water is None:
            water_sum = np.zeros((height, width), dtype=np.int32)
            valid_sum = np.zeros((height, width), dtype=np.int32)
            found_fmask = False
            for g in ['HLSL30', 'HLSS30']:
                fmask_path = f"/HDFEOS/GRIDS/{g}/Data Fields/Fmask"
                if fmask_path in h5_ard:
                    found_fmask = True
                    fm_ds = h5_ard[fmask_path]
                    n_fm = fm_ds.shape[0]
                    step = max(1, n_fm // 50)
                    for idx in range(0, n_fm, step):
                        fm = fm_ds[idx]
                        v = (fm != 255)
                        w_pix = ((fm & 32) > 0)
                        water_sum[v] += w_pix[v]
                        valid_sum[v] += 1
            if found_fmask and np.max(water_sum) > 0:
                water_sum_masked = np.ma.masked_where(valid_sum == 0, water_sum)
                persistent_water = (water_sum_masked >= (1/3 * np.max(water_sum))).filled(False)
            else:
                persistent_water = np.zeros((height, width), dtype=bool)

        min_valid_threshold = MIN_FRAME_VALIDITY_PERCENTAGE / 100.0 if MIN_FRAME_VALIDITY_PERCENTAGE > 1.0 else float(MIN_FRAME_VALIDITY_PERCENTAGE)
        
        for global_idx in range(len(prov_times)):
            ts = float(prov_times[global_idx])
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            
            if START_DATE <= dt <= END_DATE:
                source_grid = decode_h5_string(prov_grids[global_idx])
                if SOURCE_GRID is not None and source_grid != SOURCE_GRID:
                    continue

                # ----------------------------------------------------
                # CORE QUALITY ASSESSMENT: FRAME COVERAGE EVALUATION
                # ----------------------------------------------------
                if COVERAGE_EVALUATION_MODE != 'NONE':
                    vol_frame = load_scaled_frame(vol_ds, global_idx, meta=vol_meta)
                    qa_frame = qa_ds[global_idx, ...] if qa_ds is not None else None
                    
                    if COVERAGE_EVALUATION_MODE == 'PERCENTAGE':
                        if ENFORCE_QA_MASKING:
                            if qa_frame is None:
                                raise ValueError("Cannot enforce QA masking: common_mask is missing.")
                            
                            # Identify per-frame water detection from Fmask if present
                            local_idx = prov_indices[global_idx]
                            fmask_path = f"/HDFEOS/GRIDS/{source_grid}/Data Fields/Fmask"
                            if fmask_path in h5_ard:
                                frame_fmask = h5_ard[fmask_path][local_idx, ...]
                                frame_water = ((frame_fmask & 32) > 0)
                            else:
                                frame_water = False

                            is_water = persistent_water | frame_water
                            non_water = ~is_water
                            
                            # Evaluate validity strictly on non-water pixels (excluding clouds, cirrus, shadows, high aerosols)
                            if np.sum(non_water) > 0:
                                valid_frac = np.sum((qa_frame == 0) & non_water) / np.sum(non_water)
                            else:
                                valid_frac = np.sum(qa_frame == 0) / qa_frame.size
                        else:
                            valid_frac = np.sum(~np.isnan(vol_frame)) / vol_frame.size
                            
                        if valid_frac < min_valid_threshold:
                            dropped_due_to_qa += 1
                            continue 
                            
                    elif COVERAGE_EVALUATION_MODE == 'POI':
                        if len(poi_y) > 0:
                            poi_valid = ~np.isnan(vol_frame[poi_y, poi_x])
                            if ENFORCE_QA_MASKING:
                                if qa_frame is None:
                                    raise ValueError("Cannot enforce QA masking: common_mask is missing.")
                                qa_valid = (qa_frame[poi_y, poi_x] == 0)
                                poi_valid = poi_valid & qa_valid
                                
                            if not np.all(poi_valid):
                                dropped_due_to_qa += 1
                                continue
                
                source_spacecraft = decode_h5_string(prov_spaces[global_idx])
                
                if rgb_mode != 'ortho_visual':
                    if source_grid not in sr_handles:
                        sr_path = f"/HDFEOS/GRIDS/{source_grid}/Data Fields/surface_reflectance"
                        if sr_path not in h5_ard:
                            raise ValueError(f"Dataset '{sr_path}' missing for source grid '{source_grid}'.")
                        sr_ds = h5_ard[sr_path]
                        sr_handles[source_grid] = sr_ds
                        sr_meta_cache[source_grid] = get_dataset_metadata(sr_ds)
                        grid_wl = get_grid_wavelengths_nm(sr_ds, source_grid)
                        if rgb_mode == 'single_band':
                            b_idx, act_wl = find_closest_band(grid_wl, target_wls)
                            sr_band_map[source_grid] = b_idx
                            print(f"  [{source_grid}] Single Band Target {target_wls:.1f}nm -> Band {b_idx} ({act_wl:.1f}nm, error={abs(act_wl - target_wls):.1f}nm)")
                        elif rgb_mode == 'rgb_bands':
                            r_i, r_act = find_closest_band(grid_wl, target_wls[0])
                            g_i, g_act = find_closest_band(grid_wl, target_wls[1])
                            b_i, b_act = find_closest_band(grid_wl, target_wls[2])
                            sr_band_map[source_grid] = (r_i, g_i, b_i)
                            print(f"  [{source_grid}] RGB Targets [{target_wls[0]:.1f}, {target_wls[1]:.1f}, {target_wls[2]:.1f}]nm -> Bands ({r_i}, {g_i}, {b_i}) [{r_act:.1f}, {g_act:.1f}, {b_act:.1f}]nm")
                
                local_idx = int(prov_indices[global_idx])
                unified_frames.append({
                    'sensor': source_spacecraft,
                    'source_grid': source_grid,
                    'dt': dt, 
                    'dt_et': dt.astimezone(ZoneInfo("UTC")),
                    'vol_dset': vol_ds,        
                    'vol_idx': global_idx,     
                    'ortho_dset': ortho_ds,    
                    'ortho_idx': global_idx,
                    'sr_idx': local_idx,
                    'qa_dset': qa_ds,
                    'qa_idx': global_idx
                })
                
        if not unified_frames:
            print(f"Zero valid frames passed the {COVERAGE_EVALUATION_MODE} filter. ({dropped_due_to_qa} frames excluded).")
            return
            
        print(f"Coverage Evaluation Complete. Kept {len(unified_frames)} frames. Excluded {dropped_due_to_qa} frames.")

        # 3. Calculate Global Min/Max for Shared Complexity Colormap
        v_min, v_max = 0.0, 1.0
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
                data = load_scaled_frame(frame['vol_dset'], frame['vol_idx'], meta=vol_meta)
                
                # Extract valid pixels strictly within the POI masking radii
                valid_data = data[poi_mask & ~np.isnan(data)]
                if len(valid_data) > 0:
                    all_vols.append(valid_data)
            
            if all_vols:
                flat_vols = np.concatenate(all_vols)
                mean_vol = float(np.mean(flat_vols))
                std_vol = float(np.std(flat_vols, ddof=1))
                
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
            pct_int = int(round(min_valid_threshold * 100))
            qa_suffix = f"_QA-{pct_int}pct" if ENFORCE_QA_MASKING else f"_Coverage-{pct_int}pct"
        else:
            qa_suffix = "_Unmasked"
            
        prefix = f"{Location}_ARD_Constellation_{target_metric}_{START_DATE.strftime('%Y')}-{END_DATE.strftime('%Y')}{qa_suffix}_{FPS}fps"
        
        if rgb_mode == 'ortho_visual':
            vis_suffix = "TrueColor"
            path_side = os.path.join(OUTPUT_DIR, f"{prefix}_SideBySide.mp4")
            if EXPORT_GIF:
                path_side_gif = os.path.join(OUTPUT_DIR, f"{prefix}_SideBySide.gif")
        elif rgb_mode == 'single_band':
            vis_suffix = f"Band_{int(round(target_wls))}nm"
            path_side = os.path.join(OUTPUT_DIR, f"{prefix}_SideBySide_{vis_suffix}.mp4")
            if EXPORT_GIF:
                path_side_gif = os.path.join(OUTPUT_DIR, f"{prefix}_SideBySide_{vis_suffix}.gif")
        else:
            r_w, g_w, b_w = target_wls
            vis_suffix = f"RGB_{int(round(r_w))}_{int(round(g_w))}_{int(round(b_w))}nm"
            path_side = os.path.join(OUTPUT_DIR, f"{prefix}_SideBySide_{vis_suffix}.mp4")
            if EXPORT_GIF:
                path_side_gif = os.path.join(OUTPUT_DIR, f"{prefix}_SideBySide_{vis_suffix}.gif")

        path_rgb = os.path.join(OUTPUT_DIR, f"{prefix}_{vis_suffix}.mp4")
        path_comp = os.path.join(OUTPUT_DIR, f"{prefix}_Complexity.mp4")
        
        if EXPORT_GIF:
            path_rgb_gif = os.path.join(OUTPUT_DIR, f"{prefix}_{vis_suffix}.gif")
            path_comp_gif = os.path.join(OUTPUT_DIR, f"{prefix}_Complexity.gif")

        # Determine appropriate figure sizes to match data aspect ratio
        base_width = 8.0
        aspect = height / width
        extra_h = 0.6 # extra vertical space for the date text
        
        fig_rgb, ax_rgb = plt.subplots(figsize=(base_width, base_width * aspect + extra_h), facecolor=background_color)
        fig_comp, ax_comp = plt.subplots(figsize=(base_width * 1.15, base_width * aspect + extra_h), facecolor=background_color)
        fig_side, (ax_s1, ax_s2) = plt.subplots(1, 2, figsize=(base_width * 2.15, base_width * aspect + extra_h), facecolor=background_color)

        for ax in [ax_rgb, ax_comp, ax_s1, ax_s2]:
            ax.set_facecolor(background_color) 
            ax.axis('off')

        cmap_comp = plt.get_cmap('cividis').copy()
        cmap_comp.set_bad(color=background_color, alpha=0.0)

        # Initialize Imshow objects with the correct shape to preserve aspect ratio for tight_layout
        dummy_shape = (height, width)
        im_rgb = ax_rgb.imshow(np.zeros((*dummy_shape, 4), dtype=np.float32)) 
        im_comp = ax_comp.imshow(np.zeros(dummy_shape, dtype=np.float32), cmap=cmap_comp, vmin=v_min, vmax=v_max)
        im_s1 = ax_s1.imshow(np.zeros((*dummy_shape, 4), dtype=np.float32))
        im_s2 = ax_s2.imshow(np.zeros(dummy_shape, dtype=np.float32), cmap=cmap_comp, vmin=v_min, vmax=v_max)
        
        divider_comp = make_axes_locatable(ax_comp)
        cax_comp = divider_comp.append_axes("right", size="5%", pad=0.05)
        cbar_comp = fig_comp.colorbar(im_comp, cax=cax_comp)
        cbar_comp.set_label("Spectral Complexity", color=text_color, fontweight='bold')
        cbar_comp.ax.yaxis.set_tick_params(color=text_color, labelcolor=text_color)
        cbar_comp.outline.set_edgecolor(text_color)

        divider_side = make_axes_locatable(ax_s2)
        cax_side = divider_side.append_axes("right", size="5%", pad=0.05)
        cbar_side = fig_side.colorbar(im_s2, cax=cax_side)
        cbar_side.set_label("Spectral Complexity", color=text_color, fontweight='bold')
        cbar_side.ax.yaxis.set_tick_params(color=text_color, labelcolor=text_color)
        cbar_side.outline.set_edgecolor(text_color)

        prog_bars = []
        prog_txts = []
        for f in [fig_rgb, fig_comp, fig_side]:
            ax_prog = f.add_axes([0.05, 0.02, 0.9, 0.04])
            ax_prog.set_xlim(0, 1)
            ax_prog.set_ylim(0, 1)
            ax_prog.axis('off')
            ax_prog.add_patch(patches.Rectangle((0, 0), 1, 1, facecolor='lightgray', edgecolor='black', linewidth=1))
            fg = patches.Rectangle((0, 0), 0, 1, facecolor='dodgerblue', edgecolor='black', linewidth=1)
            ax_prog.add_patch(fg)
            prog_bars.append(fg)
            txt = ax_prog.text(0.5, 0.5, "", ha='center', va='center', color='black', fontsize=12, fontweight='bold')
            prog_txts.append(txt)

        inds_rgb, inds_comp, inds_s1, inds_s2 = [], [], [], []
        if SHOW_PIXEL_INDICATORS:
            for _ in mapped_locs:
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
                time_str = frame['dt_et'].strftime('%Y-%m-%d')
                
                if i % 10 == 0:
                    print(f"  Rendering frame {i}/{len(unified_frames)}...")
                
                # --- 1. Extract Complexity Map with verified datatype and scaling ---
                comp_data = load_scaled_frame(frame['vol_dset'], frame['vol_idx'], meta=vol_meta)
                qa_frame = frame['qa_dset'][frame['qa_idx'], ...] if frame.get('qa_dset') is not None else None

                # --- 2. Process Visual Imagery (RGB / Single Band Grayscale + Alpha) ---
                if rgb_mode == 'ortho_visual':
                    raw_ortho = frame['ortho_dset'][frame['ortho_idx'], ...]
                    
                    if raw_ortho.shape[0] in [3, 4]:
                        raw_ortho = np.transpose(raw_ortho, (1, 2, 0))
                        
                    rgba = np.zeros((raw_ortho.shape[0], raw_ortho.shape[1], 4), dtype=np.float32)
                    rgba[..., :3] = raw_ortho[..., :3].astype(np.float32) / 255.0
                    
                    if raw_ortho.shape[-1] == 4:
                        user_alpha = raw_ortho[..., 3]
                        rgba[..., 3] = np.where(user_alpha > 0, 1.0, 0.0)
                    else:
                        # 3-channel ortho: determine nodata swath boundaries from comp_data NaN or QA fill
                        is_nodata = np.isnan(comp_data)
                        if qa_frame is not None and qa_fill is not None:
                            is_nodata = is_nodata | (qa_frame == qa_fill)
                        rgba[..., 3] = np.where(is_nodata, 0.0, 1.0)
                elif rgb_mode == 'single_band':
                    src_grid = frame['source_grid']
                    sr_ds = sr_handles[src_grid]
                    sr_meta = sr_meta_cache[src_grid]
                    band_idx = sr_band_map[src_grid]
                    
                    band_data = load_scaled_band(sr_ds, frame['sr_idx'], band_idx, meta=sr_meta)
                    rgba = generate_single_band_rgba(band_data)
                    
                    is_nodata = np.isnan(comp_data)
                    if qa_frame is not None and qa_fill is not None:
                        is_nodata = is_nodata | (qa_frame == qa_fill)
                    rgba[is_nodata, 3] = 0.0
                elif rgb_mode == 'rgb_bands':
                    src_grid = frame['source_grid']
                    sr_ds = sr_handles[src_grid]
                    sr_meta = sr_meta_cache[src_grid]
                    r_idx, g_idx, b_idx = sr_band_map[src_grid]
                    
                    r_band = load_scaled_band(sr_ds, frame['sr_idx'], r_idx, meta=sr_meta)
                    g_band = load_scaled_band(sr_ds, frame['sr_idx'], g_idx, meta=sr_meta)
                    b_band = load_scaled_band(sr_ds, frame['sr_idx'], b_idx, meta=sr_meta)
                    
                    rgba = generate_multiband_rgba(r_band, g_band, b_band)
                    
                    is_nodata = np.isnan(comp_data)
                    if qa_frame is not None and qa_fill is not None:
                        is_nodata = is_nodata | (qa_frame == qa_fill)
                    rgba[is_nodata, 3] = 0.0
                    
                invalid_rgb_mask = np.isnan(rgba[..., :3]).any(axis=-1) | (rgba[..., 3] == 0.0)
                rgba[invalid_rgb_mask, 3] = 0.0
                rgba = np.nan_to_num(rgba, nan=0.0)
                
                comp_data[rgba[..., 3] == 0.0] = np.nan
                
                if not GLOBAL_COLOR_SCALE:
                    valid_vals = comp_data[~np.isnan(comp_data)]
                    if len(valid_vals) > 0:
                        v_min = float(np.percentile(valid_vals, 1))
                        v_max = float(np.percentile(valid_vals, 98))
                        if v_min == v_max:
                            v_max = v_min + 1e-6
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
                    total_seconds = (END_DATE - START_DATE).total_seconds()
                    elapsed_seconds = (frame['dt_et'] - START_DATE).total_seconds()
                    progress = max(0.0, min(1.0, elapsed_seconds / total_seconds)) if total_seconds > 0 else 1.0
                    for fg, txt in zip(prog_bars, prog_txts):
                        fg.set_width(progress)
                        txt.set_text(time_str)
                
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
