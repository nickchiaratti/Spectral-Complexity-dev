"""
create_injected_dataset.py
==========================
Multi-Sensor Synthetic Target Injection Pipeline for Change Detection Benchmarking.

Creates in-place injected datasets directly in the master HDF5 container:
  - /HDFEOS/GRIDS/{SENSOR}/Data Fields/injected_surface_reflectance
    (for HLSL30, HLSS30, and TANAGER)
  - /HDFEOS/GRIDS/HARMONIZED/Data Fields/injected_sliding_volume_z_score
  - /HDFEOS/GRIDS/HARMONIZED/Data Fields/injected_ndvi
  - /INJECTION_METADATA

Expanded Replicate Design (N = 3 Replicate Design, 6 Patches Total, 300 Injection Sites):
  - 3 Agricultural Sites (Array Type A: weights [0.05, 0.15, 0.25, 0.35, 0.45]):
      * Site A1: Ogden Western Farmland (r=422, c=155)
      * Site A2: Rush Open Farmland (r=974, c=488)
      * Site A3: Macedon/Perinton Rural (r=700, c=835)
  - 3 Suburban Sites (Array Type B: weights [0.10, 0.20, 0.30, 0.40, 0.50]):
      * Site B1: Greece Northwest Park (r=340, c=352)
      * Site B2: Penfield Suburban Lawn (r=619, c=595)
      * Site B3: Victor/Mendon Meadow (r=845, c=905)

Separation & Decoupling Geometry:
  - 3 un-injected pixels separating each injection site (stride = 4 pixels).
  - Footprint per patch: 37 rows x 17 cols (1,110 m x 510 m).
  - Guarantees 1 full pristine background pixel between adjacent 3x3 sliding windows,
    completely decoupling spatial covariance across injection sites.

Spectral Complexity Computation:
  - For all frames with acquisition time >= 2021-01-01:
    Runs sc.process_volume_sliding_tile for the entire frame (1409 x 1070)
    and computes the background-normalized Z-score using sc.calculate_global_z_score.

Configured for direct execution in the IDE without command line arguments.
"""

import os
import sys
import time
import datetime
from pathlib import Path
import h5py
import numpy as np
import scipy.interpolate
from tqdm import tqdm
import torch

# Ensure repository root is on sys.path
script_dir = Path(__file__).resolve().parent
repo_dir = str(script_dir.parent)
if repo_dir not in sys.path:
    sys.path.insert(0, repo_dir)
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

import SpecComplexTorch as sc

# ==============================================================================
# PIPELINE CONFIGURATION (IDE-READY DEFAULTS)
# ==============================================================================
MASTER_H5_PATH = "C:/satelliteImagery/HLST30/HLST_Rochesterv2.h5"
INJECTION_ONSET_DATE = "2022-06-01"
SPLIB_BASE = r"C:\satelliteImagery\ground_truth\splib07\ASCIIdata"

# Sliding window complexity hyperparameters (matching published pipeline)
TILE_SIZE = 3
SLIDING_STRIDE = 1
NUM_ENDMEMBERS = 7

# Injection grid geometry: 3 separating pixels -> stride = 4
STRIDE = 4
N_ROWS = 10
N_COLS = 5
GRID_H = (N_ROWS - 1) * STRIDE + 1  # 37 rows
GRID_W = (N_COLS - 1) * STRIDE + 1  # 17 cols

# ==============================================================================
# TARGET MATERIALS METADATA (10 USGS splib07 materials across 3 contrast regimes)
# ==============================================================================
TARGET_MATERIALS = [
    {
        "id": "concrete",
        "name": "Concrete (WTC01-37A)",
        "contrast": "Manmade",
        "role": "Synthetic Materials",
        "asd_rel": r"ChapterA_ArtificialMaterials\s07_ASD_Concrete_WTC01-37A_ASDFRa_AREF.txt",
        "l8_rel":  r"ChapterA_ArtificialMaterials\S07LSAT8_Concrete_WTC01-37A_ASDFRa_AREF.txt",
        "s2_rel":  r"ChapterA_ArtificialMaterials\S07SNTL2_Concrete_WTC01-37A_ASDFRa_AREF.txt",
    },
    {
        "id": "asphalt shingle",
        "name": "Asphalt Shingle (GD367)",
        "contrast": "Manmade",
        "role": "Synthetic Materials",
        "asd_rel": r"ChapterA_ArtificialMaterials\s07_ASD_Asphalt_Shingle_GDS367_DkGry_ASDFRa_AREF.txt",
        "l8_rel":  r"ChapterA_ArtificialMaterials\S07LSAT8_Asphalt_Shingle_GDS367_DkGry_ASDFRa_AREF.txt",
        "s2_rel":  r"ChapterA_ArtificialMaterials\S07SNTL2_Asphalt_Shingle_GDS367_DkGry_ASDFRa_AREF.txt",
    },
    {
        "id": "plywood",
        "name": "Plywood Fresh Pine (GDS365)",
        "contrast": "Manmade",
        "role": "Construction / Debris",
        "asd_rel": r"ChapterA_ArtificialMaterials\s07_ASD_Plywood_GDS365_Fresh_Pine_ASDFRa_AREF.txt",
        "l8_rel":  r"ChapterA_ArtificialMaterials\S07LSAT8_Plywood_GDS365_Fresh_Pine_ASDFRa_AREF.txt",
        "s2_rel":  r"ChapterA_ArtificialMaterials\S07SNTL2_Plywood_GDS365_Fresh_Pine_ASDFRa_AREF.txt",
    },
    {
        "id": "white_plastic",
        "name": "White Vinyl Plastic",
        "contrast": "Manmade",
        "role": "Synthetic Materials",
        "asd_rel": r"ChapterA_ArtificialMaterials\s07_ASD_Plastic_Vinyl_GDS372_White_ASDFRa_AREF.txt",
        "l8_rel":  r"ChapterA_ArtificialMaterials\S07LSAT8_Plastic_Vinyl_GDS372_White_ASDFRa_AREF.txt",
        "s2_rel":  r"ChapterA_ArtificialMaterials\S07SNTL2_Plastic_Vinyl_GDS372_White_ASDFRa_AREF.txt",
    },
    {
        "id": "dry_grass",
        "name": "Grass_Golden_Dry (GDS480)",
        "contrast": "senescent vegetation",
        "role": "Vegetation",
        "asd_rel": r"ChapterV_Vegetation\s07_ASD_Grass_Golden_Dry_GDS480_ASDFRa_AREF.txt",
        "l8_rel":  r"ChapterV_Vegetation\S07LSAT8_Grass_Golden_Dry_GDS480_ASDFRa_AREF.txt",
        "s2_rel":  r"ChapterV_Vegetation\S07SNTL2_Grass_Golden_Dry_GDS480_ASDFRa_AREF.txt",
    },
    {
        "id": "dry_willow",
        "name": "Willow Dry",
        "contrast": "senescent vegetation",
        "role": "Vegetation", 
        "asd_rel": r"ChapterV_Vegetation\s07_ASD_Willow_Willow-Leaves-1_dry_ASDFRa_AREF.txt",
        "l8_rel":  r"ChapterV_Vegetation\S07LSAT8_Willow_Willow-Leaves-1_dry_ASDFRa_AREF.txt",
        "s2_rel":  r"ChapterV_Vegetation\S07SNTL2_Willow_Willow-Leaves-1_dry_ASDFRa_AREF.txt",
    },
    {
        "id": "burn_area",
        "name": "BurnArea_TopSurface (WRF00-02)",
        "contrast": "Burn / Pyrogenic",
        "role": "Natural Disturbance",
        "asd_rel": r"ChapterS_SoilsAndMixtures\s07_ASD_BurnArea_TopSurface_WRF00-02_ASDFRb_AREF.txt",
        "l8_rel":  r"ChapterS_SoilsAndMixtures\S07LSAT8_BurnArea_TopSurface_WRF00-02_ASDFRb_AREF.txt",
        "s2_rel":  r"ChapterS_SoilsAndMixtures\S07SNTL2_BurnArea_TopSurface_WRF00-02_ASDFRb_AREF.txt",
    },
    {
        "id": "oak_leaves",
        "name": "Oak Leaves",
        "contrast": "Healthy Vegetation",
        "role": "Vegetation",
        "asd_rel": r"ChapterV_Vegetation\s07_ASD_Oak_Oak-Leaf-1_fresh_ASDFRa_AREF.txt",
        "l8_rel":  r"ChapterV_Vegetation\S07LSAT8_Oak_Oak-Leaf-1_fresh_ASDFRa_AREF.txt",
        "s2_rel":  r"ChapterV_Vegetation\S07SNTL2_Oak_Oak-Leaf-1_fresh_ASDFRa_AREF.txt",
    },
    {
        "id": "oak_shrub",
        "name": "Oak Shrub (QUDU CA01-2)",
        "contrast": "Canopy Foliage",
        "role": "Vegetation",
        "asd_rel": r"ChapterV_Vegetation\s07_ASD_Oak_QUDU_CA01-QUDU-2_bush_2_ASDFRa_AREF.txt",
        "l8_rel":  r"ChapterV_Vegetation\S07LSAT8_Oak_QUDU_CA01-QUDU-2_bush_2_ASDFRa_AREF.txt",
        "s2_rel":  r"ChapterV_Vegetation\S07SNTL2_Oak_QUDU_CA01-QUDU-2_bush_2_ASDFRa_AREF.txt",
    },
    {
        "id": "turbid_water",
        "name": "Water + Montmorillonite (5g/L)",
        "contrast": "Inundation",
        "role": "Flood / Inundation",
        "asd_rel": r"ChapterL_Liquids\s07_ASD_Water+Montmor_SWy-2+5.01g-l_ASDFRa_AREF.txt",
        "l8_rel":  r"ChapterL_Liquids\S07LSAT8_Water+Montmor_SWy-2+5.01g-l_ASDFRa_AREF.txt",
        "s2_rel":  r"ChapterL_Liquids\S07SNTL2_Water+Montmor_SWy-2+5.01g-l_ASDFRa_AREF.txt",
    },
]

# Sentinel-2 band indices matching HLSS30 (10 bands):
S2_SUB_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 11, 12]

# ==============================================================================
# EXPANDED REPLICATE DESIGN (6 PATCHES, STRIDE = 4, 3 SEPARATING PIXELS)
# ==============================================================================
PROPOSED_SITES = [
    {
        "id": "A1",
        "role": "Replicate 1 (Agricultural)",
        "name": "Site A1: Ogden Western Farmland",
        "r": 422,
        "c": 155,
        "type": "Western Agricultural Cropland",
        "weights": [0.05, 0.15, 0.25, 0.35, 0.45],
        "color": "#1f77b4",
    },
    {
        "id": "A2",
        "role": "Replicate 2 (Agricultural)",
        "name": "Site A2: Rush Open Farmland",
        "r": 974,
        "c": 488,
        "type": "Southern Extensive Farmland",
        "weights": [0.05, 0.15, 0.25, 0.35, 0.45],
        "color": "#2ca02c",
    },
    {
        "id": "A3",
        "role": "Replicate 3 (Agricultural)",
        "name": "Site A3: Macedon/Perinton Rural",
        "r": 700,
        "c": 835,
        "type": "Eastern Cropland / Pasture",
        "weights": [0.05, 0.15, 0.25, 0.35, 0.45],
        "color": "#17becf",
    },
    {
        "id": "B1",
        "role": "Replicate 1 (Suburban)",
        "name": "Site B1: Greece Northwest Park",
        "r": 340,
        "c": 352,
        "type": "Suburban Park / Meadow",
        "weights": [0.10, 0.20, 0.30, 0.40, 0.50],
        "color": "#d62728",
    },
    {
        "id": "B2",
        "role": "Replicate 2 (Suburban)",
        "name": "Site B2: Penfield Suburban Lawn",
        "r": 619,
        "c": 595,
        "type": "Suburban / Park Lawn",
        "weights": [0.10, 0.20, 0.30, 0.40, 0.50],
        "color": "#ff7f0e",
    },
    {
        "id": "B3",
        "role": "Replicate 3 (Suburban)",
        "name": "Site B3: Victor/Mendon Meadow",
        "r": 845,
        "c": 905,
        "type": "Lower Right Parkland / Meadow",
        "weights": [0.10, 0.20, 0.30, 0.40, 0.50],
        "color": "#9467bd",
    },
]


# ==============================================================================
# SPECTRAL DATA INGESTION & INTERPOLATION
# ==============================================================================
def load_all_target_spectra(splib_base=SPLIB_BASE, tanager_wavelengths=None):
    """
    Loads and resamples the 10 USGS splib07 target materials for Landsat-8,
    Sentinel-2, and Tanager-1 sensors.
    """
    asd_dir = os.path.join(splib_base, "ASCIIdata_splib07b_cvASD")
    l8_dir  = os.path.join(splib_base, "ASCIIdata_splib07b_rsLandsat8")
    s2_dir  = os.path.join(splib_base, "ASCIIdata_splib07b_rsSentinel2")

    asd_wave_f = os.path.join(asd_dir, "s07_ASD_Wavelengths_ASD_0.35-2.5_microns_2151_ch.txt")
    if not os.path.exists(asd_wave_f):
        raise FileNotFoundError(f"ASD wavelength file not found: {asd_wave_f}")

    asd_wave_nm = np.loadtxt(asd_wave_f, skiprows=1) * 1000.0

    spectra_by_material = []
    for item in TARGET_MATERIALS:
        p_asd = os.path.join(asd_dir, item["asd_rel"])
        p_l8  = os.path.join(l8_dir,  item["l8_rel"])
        p_s2  = os.path.join(s2_dir,  item["s2_rel"])

        r_asd = np.loadtxt(p_asd, skiprows=1)
        r_l8  = np.loadtxt(p_l8,  skiprows=1)
        r_s2  = np.loadtxt(p_s2,  skiprows=1)

        r_asd[r_asd < 0] = np.nan
        r_l8[r_l8 < 0]   = np.nan
        r_s2[r_s2 < 0]   = np.nan

        spec_l8 = r_l8.copy().astype(np.float32)
        spec_s2 = r_s2[S2_SUB_INDICES].astype(np.float32)

        if tanager_wavelengths is not None:
            valid_asd = ~np.isnan(r_asd)
            interp_fn = scipy.interpolate.interp1d(
                asd_wave_nm[valid_asd],
                r_asd[valid_asd],
                kind='linear',
                bounds_error=False,
                fill_value='extrapolate'
            )
            spec_tanager = interp_fn(tanager_wavelengths).astype(np.float32)
            spec_tanager = np.clip(spec_tanager, 0.0, 1.0)
        else:
            spec_tanager = None

        spectra_by_material.append({
            "id": item["id"],
            "name": item["name"],
            "contrast": item["contrast"],
            "role": item["role"],
            "spec_l8": spec_l8,
            "spec_s2": spec_s2,
            "spec_tanager": spec_tanager,
        })

    return spectra_by_material


def build_all_injection_lookups(sites, materials_spectra):
    """
    Builds the injection lookup list across all sites with stride = 4 geometry.
    Returns:
      all_injections: list of 300 injection dicts (site_id, scene_r, scene_c, material_idx, weight, ...)
      site_lookups: dict mapping site_id -> list of injections for that site
    """
    all_injections = []
    site_lookups = {}

    for site in sites:
        s_id = site["id"]
        s_r = site["r"]
        s_c = site["c"]
        weights = site["weights"]
        site_injs = []

        for m_idx in range(len(materials_spectra)):
            scene_r = s_r + m_idx * STRIDE
            for w_idx, f_weight in enumerate(weights):
                scene_c = s_c + w_idx * STRIDE
                inj_info = {
                    "site_id": s_id,
                    "scene_r": scene_r,
                    "scene_c": scene_c,
                    "material_idx": m_idx,
                    "material_name": materials_spectra[m_idx]["name"],
                    "weight": f_weight,
                    "loc_r": m_idx * STRIDE,
                    "loc_c": w_idx * STRIDE,
                }
                site_injs.append(inj_info)
                all_injections.append(inj_info)

        site_lookups[s_id] = site_injs

    return all_injections, site_lookups


def apply_linear_mixture(bg_sr_int16, target_spec_float, f_weight, fill_value=-32768, scale_factor=10000.0):
    """
    Applies linear mixture model strictly preserving nodata/fill flags:
      x_injected = (1 - f) * x_background + f * target_spec
    """
    out = bg_sr_int16.copy()
    num_bands = bg_sr_int16.shape[0]

    for b in range(num_bands):
        val = bg_sr_int16[b]
        if val == fill_value or np.isnan(val):
            continue

        bg_float = float(val) / scale_factor
        t_float = float(target_spec_float[b])

        if np.isnan(t_float):
            continue

        mixed_float = (1.0 - f_weight) * bg_float + f_weight * t_float
        mixed_int16 = int(np.round(mixed_float * scale_factor))
        mixed_int16 = max(0, min(32767, mixed_int16))
        out[b] = mixed_int16

    return out


def read_scaled_int16(dataset, slice_obj=np.s_[...]):
    """
    Reads an HDF5 dataset, checks for scale_factor and _FillValue attributes,
    and returns a float32 array with fill values converted to NaN.
    """
    data = dataset[slice_obj] if hasattr(dataset, '__getitem__') and hasattr(dataset, 'attrs') else dataset
    scale_factor = dataset.attrs.get("scale_factor", 10000.0) if hasattr(dataset, 'attrs') else 10000.0
    fill_value = None
    if hasattr(dataset, 'attrs'):
        fill_value = dataset.attrs.get("_FillValue")
        if fill_value is None:
            fill_value = dataset.attrs.get("fill_value")
    if fill_value is None:
        fill_value = -32768

    if scale_factor is not None and data.dtype == np.int16:
        float_data = data.astype(np.float32)
        if fill_value is not None:
            float_data[data == fill_value] = np.nan
        return float_data / float(scale_factor)

    if np.issubdtype(data.dtype, np.floating) and fill_value is not None:
        data = data.copy()
        data[data == fill_value] = np.nan
        return data

    return data


# ==============================================================================
# IN-PLACE MASTER HDF5 INJECTION (PRIMARY PIPELINE)
# ==============================================================================
def inject_master_h5(
    master_h5_path=MASTER_H5_PATH,
    onset_date_str=INJECTION_ONSET_DATE,
    splib_base=SPLIB_BASE,
):
    """
    Creates injected datasets in-place within the master HDF5 container:
      - /HDFEOS/GRIDS/{SENSOR}/Data Fields/injected_surface_reflectance
      - /HDFEOS/GRIDS/HARMONIZED/Data Fields/injected_sliding_volume_z_score
      - /HDFEOS/GRIDS/HARMONIZED/Data Fields/injected_ndvi
      - /INJECTION_METADATA

    Re-uses common_mask and Fmask without duplication.
    """
    t_start = time.time()
    if not os.path.exists(master_h5_path):
        raise FileNotFoundError(f"Master HDF5 file not found: {master_h5_path}")

    onset_dt = datetime.datetime.fromisoformat(onset_date_str).replace(tzinfo=datetime.timezone.utc)
    onset_ts = onset_dt.timestamp()

    print("\n" + "=" * 80)
    print("IN-PLACE MULTI-SENSOR TARGET INJECTION PIPELINE (N = 3 REPLICATE DESIGN)")
    print("=" * 80)
    print(f"Target container:   {master_h5_path}")
    print(f"Disturbance onset:  {onset_date_str} (timestamp: {onset_ts})")
    print(f"Grid Geometry:      Stride = {STRIDE} (3 separating pixels), Footprint = {GRID_H} rows x {GRID_W} cols")
    print(f"Replicate Patches:  {len(PROPOSED_SITES)} sites ({len(PROPOSED_SITES) * N_ROWS * N_COLS} total injection sites)")

    with h5py.File(master_h5_path, 'r+') as f:
        # Load Tanager wavelengths and target spectra
        tan_sr = f['/HDFEOS/GRIDS/TANAGER/Data Fields/surface_reflectance']
        tan_w = tan_sr.attrs.get('wavelengths', None)
        materials_spectra = load_all_target_spectra(splib_base, tan_w)
        all_injections, site_lookups = build_all_injection_lookups(PROPOSED_SITES, materials_spectra)
        print(f"Loaded {len(materials_spectra)} target materials across {len(all_injections)} injection sites.")

        # Document injection metadata
        if 'INJECTION_METADATA' in f:
            del f['INJECTION_METADATA']
        meta_grp = f.create_group('INJECTION_METADATA')
        meta_grp.attrs['source_dataset'] = master_h5_path
        meta_grp.attrs['injection_onset_date'] = onset_date_str
        meta_grp.attrs['injection_onset_timestamp'] = onset_ts
        meta_grp.attrs['stride'] = STRIDE
        meta_grp.attrs['num_separating_pixels'] = STRIDE - 1
        meta_grp.attrs['grid_height'] = GRID_H
        meta_grp.attrs['grid_width'] = GRID_W
        meta_grp.attrs['num_sites'] = len(PROPOSED_SITES)
        meta_grp.attrs['site_ids'] = [s["id"] for s in PROPOSED_SITES]
        meta_grp.attrs['materials'] = [m["name"] for m in materials_spectra]

        # Backward compatibility attributes for existing visualization scripts (pointing to A1 and B1)
        a1_site = PROPOSED_SITES[0]
        b1_site = PROPOSED_SITES[3]
        meta_grp.attrs['site_a_rows'] = np.array([a1_site["r"] + i * STRIDE for i in range(N_ROWS)], dtype=np.int32)
        meta_grp.attrs['site_a_cols'] = np.array([a1_site["c"] + j * STRIDE for j in range(N_COLS)], dtype=np.int32)
        meta_grp.attrs['site_b_rows'] = np.array([b1_site["r"] + i * STRIDE for i in range(N_ROWS)], dtype=np.int32)
        meta_grp.attrs['site_b_cols'] = np.array([b1_site["c"] + j * STRIDE for j in range(N_COLS)], dtype=np.int32)
        meta_grp.attrs['site_a_weights'] = np.array(a1_site["weights"], dtype=np.float32)
        meta_grp.attrs['site_b_weights'] = np.array(b1_site["weights"], dtype=np.float32)

        # Store metadata for all 6 sites
        for s in PROPOSED_SITES:
            s_id = s["id"]
            meta_grp.attrs[f'site_{s_id}_name'] = s["name"]
            meta_grp.attrs[f'site_{s_id}_role'] = s["role"]
            meta_grp.attrs[f'site_{s_id}_type'] = s["type"]
            meta_grp.attrs[f'site_{s_id}_top_left'] = np.array([s["r"], s["c"]], dtype=np.int32)
            meta_grp.attrs[f'site_{s_id}_extent'] = np.array([s["r"], s["r"] + GRID_H, s["c"], s["c"] + GRID_W], dtype=np.int32)
            meta_grp.attrs[f'site_{s_id}_rows'] = np.array([s["r"] + i * STRIDE for i in range(N_ROWS)], dtype=np.int32)
            meta_grp.attrs[f'site_{s_id}_cols'] = np.array([s["c"] + j * STRIDE for j in range(N_COLS)], dtype=np.int32)
            meta_grp.attrs[f'site_{s_id}_weights'] = np.array(s["weights"], dtype=np.float32)
            meta_grp.attrs[f'site_{s_id}_color'] = s["color"]

        # ----------------------------------------------------------------------
        # Step 1: Create injected_surface_reflectance in each sensor grid
        # ----------------------------------------------------------------------
        for g_name in ['HLSL30', 'HLSS30', 'TANAGER']:
            data_grp = f[f'/HDFEOS/GRIDS/{g_name}/Data Fields']
            print(f"\nProcessing grid '{g_name}'...")

            # Clean re-creation via direct block copy
            if 'injected_surface_reflectance' in data_grp:
                print(f"  Removing existing 'injected_surface_reflectance' in {g_name}...")
                del data_grp['injected_surface_reflectance']

            t_copy = time.time()
            data_grp.copy('surface_reflectance', 'injected_surface_reflectance')
            print(f"  Cloned 'surface_reflectance' to 'injected_surface_reflectance' in {time.time() - t_copy:.2f} s.")

            inj_sr = data_grp['injected_surface_reflectance']
            sensor_times = inj_sr.attrs['acquisition_time'][:]
            n_frames = inj_sr.shape[0]

            if g_name == 'HLSL30':
                spec_key = "spec_l8"
                fill_val = -32768
            elif g_name == 'HLSS30':
                spec_key = "spec_s2"
                fill_val = -32768
            elif g_name == 'TANAGER':
                spec_key = "spec_tanager"
                fill_val = -9999

            injected_count = 0
            for t_idx in range(n_frames):
                if sensor_times[t_idx] < onset_ts:
                    continue

                injected_count += 1

                # Inject targets across each of the 6 sites
                for site in PROPOSED_SITES:
                    s_r0 = site["r"]
                    s_r1 = s_r0 + GRID_H
                    s_c0 = site["c"]
                    s_c1 = s_c0 + GRID_W

                    site_injs = site_lookups[site["id"]]
                    patch = inj_sr[t_idx, :, s_r0:s_r1, s_c0:s_c1]

                    for inj in site_injs:
                        loc_r = inj["loc_r"]
                        loc_c = inj["loc_c"]
                        m_idx = inj["material_idx"]
                        f_w = inj["weight"]
                        t_spec = materials_spectra[m_idx][spec_key]

                        patch[:, loc_r, loc_c] = apply_linear_mixture(
                            patch[:, loc_r, loc_c], t_spec, f_w, fill_value=fill_val, scale_factor=10000.0
                        )

                    inj_sr[t_idx, :, s_r0:s_r1, s_c0:s_c1] = patch

            print(f"  Grid '{g_name}': {injected_count} frames injected across {len(PROPOSED_SITES)} sites ({len(all_injections)} total pixels).")

        # ----------------------------------------------------------------------
        # Step 2: Create injected_sliding_volume_z_score across full frames
        # ----------------------------------------------------------------------
        harm_grp = f['/HDFEOS/GRIDS/HARMONIZED/Data Fields']
        print("\nProcessing Harmonized Spectral Complexity datasets...")

        if 'injected_sliding_volume_map' in harm_grp:
            del harm_grp['injected_sliding_volume_map']

        if 'injected_sliding_volume_z_score' in harm_grp:
            print("  Removing existing 'injected_sliding_volume_z_score'...")
            del harm_grp['injected_sliding_volume_z_score']

        t_zm_copy = time.time()
        harm_grp.copy('sliding_volume_z_score', 'injected_sliding_volume_z_score')
        print(f"  Cloned baseline 'sliding_volume_z_score' in {time.time() - t_zm_copy:.2f} s.")

        inj_zm_ds = harm_grp['injected_sliding_volume_z_score']
        harm_times = inj_zm_ds.attrs['acquisition_time'][:]
        harm_grids = inj_zm_ds.attrs['source_grid'][:]
        harm_indices = inj_zm_ds.attrs['source_frame_index'][:]
        frame_means = inj_zm_ds.attrs['frame_global_means'][:].copy()
        frame_stds = inj_zm_ds.attrs['frame_global_stds'][:].copy()
        num_frames = len(harm_times)
        harm_mask_ds = harm_grp['common_mask']

        injected_frame_indices = [idx for idx, t in enumerate(harm_times) if t >= onset_ts]
        print(f"\nProcessing full-frame Spectral Complexity & Global Z-Score for {len(injected_frame_indices)} injected frames (t >= {onset_date_str})...")

        for f_idx in tqdm(injected_frame_indices, desc="Computing Injected Spectral Complexity"):
            g_code = harm_grids[f_idx]
            g_str = g_code.decode('utf-8') if isinstance(g_code, bytes) else g_code
            loc_idx = int(harm_indices[f_idx])

            inj_sr_ds = f[f'/HDFEOS/GRIDS/{g_str}/Data Fields/injected_surface_reflectance']
            frame_sr = read_scaled_int16(inj_sr_ds, np.s_[loc_idx, ...])

            # Prune defective bands if all_good_wavelengths is present (e.g. TANAGER)
            gw_attr = inj_sr_ds.attrs.get("all_good_wavelengths")
            if gw_attr is not None:
                gw_mask = gw_attr[loc_idx].astype(bool)
                frame_sr_eval = frame_sr[gw_mask, ...]
            else:
                frame_sr_eval = frame_sr

            # 1. GPU-accelerated sliding window volume processing across entire frame (1409 x 1070)
            slide_map = sc.process_volume_sliding_tile(
                frame_sr_eval,
                tile_size=TILE_SIZE,
                stride=SLIDING_STRIDE,
                num_endmembers=NUM_ENDMEMBERS,
            )
            if isinstance(slide_map, tuple):
                slide_map = slide_map[0]
            if isinstance(slide_map, torch.Tensor):
                slide_map = slide_map.detach().cpu().numpy()

            # 2. Extract radiometrically valid pixels from common mask for background statistical model
            frame_mask = harm_mask_ds[f_idx, :, :]
            valid_mask = (frame_mask == 0)

            # 3. Calculate global Z-score using sc.calculate_global_z_score
            z_map, global_mean, global_std = sc.calculate_global_z_score(slide_map, valid_mask)
            if isinstance(z_map, torch.Tensor):
                z_map = z_map.detach().cpu().numpy()

            # 4. Write full-frame Z-scores and store global normalization statistics
            inj_zm_ds[f_idx, :, :] = z_map.astype(np.float32)
            frame_means[f_idx] = global_mean
            frame_stds[f_idx] = global_std

        # Store updated frame statistics back to dataset attributes
        inj_zm_ds.attrs['frame_global_means'] = frame_means
        inj_zm_ds.attrs['frame_global_stds'] = frame_stds
        print(f"Updated full-frame injected_sliding_volume_z_score across {len(injected_frame_indices)} frames.")

        # ----------------------------------------------------------------------
        # Step 3: Create injected_ndvi in HARMONIZED Data Fields
        # ----------------------------------------------------------------------
        if 'ndvi' in harm_grp:
            print("\nProcessing Harmonized Injected NDVI dataset...")
            if 'injected_ndvi' in harm_grp:
                del harm_grp['injected_ndvi']
            harm_grp.copy('ndvi', 'injected_ndvi')
            inj_ndvi_ds = harm_grp['injected_ndvi']
            inj_ndvi_ds.attrs['description'] = "Harmonized Injected Normalized Difference Vegetation Index (NDVI) [int16 scaled by 10000]"

            print("Updating Harmonized NDVI for injected frames (t >= 2021-01-01)...")
            for f_idx in injected_frame_indices:
                g_code = harm_grids[f_idx]
                g_str = g_code.decode('utf-8') if isinstance(g_code, bytes) else g_code
                loc_idx = int(harm_indices[f_idx])

                inj_sr_ds = f[f'/HDFEOS/GRIDS/{g_str}/Data Fields/injected_surface_reflectance']

                if "HLSS30" in g_str:
                    r_idx, n_idx = 3, 7
                elif "HLSL30" in g_str:
                    r_idx, n_idx = 3, 4
                elif "TANAGER" in g_str:
                    r_idx, n_idx = 59, 97
                else:
                    continue

                for site in PROPOSED_SITES:
                    s_r0 = site["r"]
                    s_r1 = s_r0 + GRID_H
                    s_c0 = site["c"]
                    s_c1 = s_c0 + GRID_W

                    sr_inj_patch = inj_sr_ds[loc_idx, :, s_r0:s_r1, s_c0:s_c1]

                    red_raw = sr_inj_patch[r_idx, :, :]
                    nir_raw = sr_inj_patch[n_idx, :, :]

                    red_f = np.full(red_raw.shape, np.nan, dtype=np.float32)
                    nir_f = np.full(nir_raw.shape, np.nan, dtype=np.float32)

                    v_red = (red_raw != -32768) & (red_raw != -9999)
                    v_nir = (nir_raw != -32768) & (nir_raw != -9999)
                    red_f[v_red] = red_raw[v_red].astype(np.float32) / 10000.0
                    nir_f[v_nir] = nir_raw[v_nir].astype(np.float32) / 10000.0

                    denom = nir_f + red_f
                    with np.errstate(divide="ignore", invalid="ignore"):
                        ndvi_patch = (nir_f - red_f) / denom
                    ndvi_patch[np.isinf(ndvi_patch)] = np.nan
                    ndvi_patch = np.clip(ndvi_patch, -1.0, 1.0)

                    int16_ndvi = np.full(ndvi_patch.shape, -32768, dtype=np.int16)
                    valid_m = ~np.isnan(ndvi_patch)
                    scaled = np.round(ndvi_patch[valid_m] * 10000.0)
                    int16_ndvi[valid_m] = np.clip(scaled, -32767, 32767).astype(np.int16)

                    inj_ndvi_ds[f_idx, s_r0:s_r1, s_c0:s_c1] = int16_ndvi

    elapsed = time.time() - t_start
    print("\n" + "=" * 80)
    print(f"SUCCESS: In-place injection complete in {elapsed:.1f} seconds ({elapsed / 60.0:.1f} minutes).")
    print(f"Updated {len(injected_frame_indices)} harmonized frames across {len(PROPOSED_SITES)} sites ({len(all_injections)} total injection sites).")
    print("=" * 80)


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================
def main():
    """Default execution runs the in-place master container injection."""
    inject_master_h5()


if __name__ == '__main__':
    main()
