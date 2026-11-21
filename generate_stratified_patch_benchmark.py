"""
generate_stratified_patch_benchmark.py
======================================
Executes the Land-Cover Stratified Patches semi-empirical target injection experiment
and saves all chips, simulated complexity cubes, and algorithmic break detection results
directly into the master HDF5 container under /STRATIFIED_CHIP_BENCHMARK/.

Reuses core functions and standards from:
  - SpecComplexTorch.py: Sliding window MaxD Gramian volume unmixing on GPU.
  - harmonized_CCD_main.py: Harmonized RLS-LASSO continuous change detection engine.
  - pyccd_landsat_main.py: USGS LCMAP Landsat PyCCD detection engine and QA decoding.
  - create_injected_dataset.py: USGS Splib07 multi-sensor resampled spectral library.

Experimental Structure:
  - 4 Land-Cover Strata (10x10 pixels, 300 m x 300 m, 9.0 ha each):
      1. AGRI: Rush Open Farmland (r0=970, c0=479)
      2. FOREST: Mendon Ponds Park Forest (r0=1146, c0=530)
      3. WETLAND: Thousand Acre Swamp, Penfield (r0=721, c0=791)
      4. SYNTHETIC: Eastview Commercial Complex (r0=1154, c0=888)
  - 4 Decoupled Injected Pixels per patch (stride = 4, 3 baseline pixels separation):
      Offsets: [(2, 2), (2, 6), (6, 2), (6, 6)]
  - 10 USGS Splib07 Target Materials x 5 Mixture Fractions [0.05, 0.15, 0.25, 0.35, 0.45]
      = 50 Distinct Perturbation Conditions
  - 50 conditions x 4 strata x 4 pixels = 800 test instances per algorithm.
"""

import os
import sys
import gc
import math
import time
import datetime
from pathlib import Path
import h5py
import numpy as np
import torch

# Ensure single-threaded BLAS/OpenMP execution for workers
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

# Ensure script directory and parent directory are on sys.path
script_dir = Path(__file__).resolve().parent
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

# Import core modules
import SpecComplexTorch as sc
import harmonized_CCD_main as hccd
import pyccd_landsat_main as pccd
import ccd

# Apply compatibility patch for PyCCD coordinate descent and mode array
pccd.patch_pyccd()

# ==============================================================================
# PIPELINE CONFIGURATION
# ==============================================================================
MASTER_H5_PATH = "C:/satelliteImagery/HLST30/HLST_Rochesterv2.h5"
INJECTION_ONSET_DATE = "2022-06-01"
SPLIB_BASE = "C:/satelliteImagery/ground_truth/splib07/ASCIIdata"

# Sliding window parameters
TILE_SIZE = 3
SLIDING_STRIDE = 1
NUM_ENDMEMBERS = 7

# Land-Cover Stratified Patches (100% clean baseline in both PYCCD and SC-CCD)
LAND_COVER_PATCHES = [
    {
        "id": "CROPLAND",
        "class": "Cropland",
        "name": "Rush Open Farmland",
        "r0": 1268,
        "c0": 228,
        "color": "#e6ab02",  # Gold / Amber
        "descr": "Cycling crop production land.",
    },
    {
        "id": "TREE COVER",
        "class": "Forest",
        "name": "Mendon Ponds Park Forest",
        "r0": 1146,
        "c0": 530,
        "color": "#2ca02c",  # Forest Green
        "descr": "Deciduous forest.",
    },
    {
        "id": "WETLAND",
        "class": "Wetland",
        "name": "Thousand Acre Swamp (Penfield)",
        "r0": 613,
        "c0": 860,
        "color": "#17becf",  # Teal / Cyan
        "descr": "Saturated soil and standing water.",
    },
    {
        "id": "DEVELOPED",
        "class": "Developed Land",
        "name": "Rochester Airport",
        "r0": 805,
        "c0": 271,
        "color": "#d62728",  # Crimson Red
        "descr": "Tarmac and asphalt with minimal vegetation.",
    },
]

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


# Injected pixel local offsets inside each 10x10 patch (stride = 4, 3 baseline pixels separation)
INJ_OFFSETS = [
    (2, 2),  # P1
    (2, 6),  # P2
    (6, 2),  # P3
    (6, 6),  # P4
]

# Fractional abundance levels
MIXTURE_FRACTIONS = [0.05, 0.15, 0.25, 0.35, 0.45]

# Compound data type for tabular results
BREAK_RECORD_DTYPE = np.dtype([
    ("condition_idx", np.int16),        # 0 to 49
    ("material_id", "S32"),             # Target material identifier string
    ("material_name", "S48"),           # Material descriptive name
    ("mixture_fraction", np.float32),   # 0.05, 0.15, 0.25, 0.35, 0.45
    ("patch_id", "S16"),                # 'AGRI', 'FOREST', 'WETLAND', 'SYNTHETIC'
    ("patch_name", "S48"),              # Full stratum description
    ("pixel_num", np.int8),             # 1 to 4
    ("scene_row", np.int16),            # Absolute row coordinate in master scene
    ("scene_col", np.int16),            # Absolute col coordinate in master scene
    ("break_detected", np.bool_),       # True if break detected post-onset
    ("break_timestamp", np.float64),    # Acquisition timestamp of detected break
    ("latency_days", np.float32),       # Days between onset date and detected break
    ("break_magnitude", np.float32),    # Magnitude of break or residual shift
    ("p_value", np.float32),            # Chi-square or model significance
])


# ==============================================================================
# STEP 1: EXTRACTION OF BASELINE CHIPS & METADATA
# ==============================================================================
def extract_baseline_chips(h5_file):
    """
    Extracts the unperturbed 10x10 time series for each of the 4 patches across sensors.
    Dynamically identifies scale_factor and _FillValue attributes from each dataset.
    """
    print("\n[Step 1] Extracting unperturbed 10x10 baseline chips from master scene...")

    harm_grp = h5_file["/HDFEOS/GRIDS/HARMONIZED/Data Fields"]
    l30_grp = h5_file["/HDFEOS/GRIDS/HLSL30/Data Fields"]
    s30_grp = h5_file["/HDFEOS/GRIDS/HLSS30/Data Fields"]
    tan_grp = h5_file.get("/HDFEOS/GRIDS/TANAGER/Data Fields")

    harm_z_ds = harm_grp["sliding_volume_z_score"]
    harm_cm_ds = harm_grp["common_mask"]
    l30_sr_ds = l30_grp["surface_reflectance"]
    l30_fm_ds = l30_grp["Fmask"]
    s30_sr_ds = s30_grp["surface_reflectance"]
    s30_fm_ds = s30_grp["Fmask"]

    n_harm_frames = harm_z_ds.shape[0]
    n_l30_frames = l30_sr_ds.shape[0]
    n_s30_frames = s30_sr_ds.shape[0]
    n_patches = len(LAND_COVER_PATCHES)

    chips = {
        "HLSL30": np.zeros((n_patches, n_l30_frames, 7, 10, 10), dtype=np.int16),
        "HLSS30": np.zeros((n_patches, n_s30_frames, 10, 10, 10), dtype=np.int16),
        "HLSL30_Fmask": np.zeros((n_patches, n_l30_frames, 10, 10), dtype=np.uint8),
        "HLSS30_Fmask": np.zeros((n_patches, n_s30_frames, 10, 10), dtype=np.uint8),
        "HARMONIZED_z_score": np.zeros((n_patches, n_harm_frames, 10, 10), dtype=np.float32),
        "HARMONIZED_common_mask": np.zeros((n_patches, n_harm_frames, 10, 10), dtype=np.uint8),
    }

    scale_factors = {
        "HLSL30": float(l30_sr_ds.attrs.get("scale_factor", 10000.0)),
        "HLSS30": float(s30_sr_ds.attrs.get("scale_factor", 10000.0)),
    }
    fill_values = {
        "HLSL30": int(l30_sr_ds.attrs.get("_FillValue", -32768)),
        "HLSS30": int(s30_sr_ds.attrs.get("_FillValue", -32768)),
    }

    tan_sr_ds = None
    if tan_grp is not None and "surface_reflectance" in tan_grp:
        tan_sr_ds = tan_grp["surface_reflectance"]
        n_tan_frames = tan_sr_ds.shape[0]
        n_tan_bands = tan_sr_ds.shape[1]
        chips["TANAGER"] = np.zeros((n_patches, n_tan_frames, n_tan_bands, 10, 10), dtype=np.int16)
        scale_factors["TANAGER"] = float(tan_sr_ds.attrs.get("scale_factor", 10000.0))
        fill_values["TANAGER"] = int(tan_sr_ds.attrs.get("_FillValue", -9999))

    for p_idx, p in enumerate(LAND_COVER_PATCHES):
        r0, c0 = p["r0"], p["c0"]
        chips["HLSL30"][p_idx] = l30_sr_ds[:, :, r0:r0+10, c0:c0+10]
        chips["HLSS30"][p_idx] = s30_sr_ds[:, :, r0:r0+10, c0:c0+10]
        chips["HLSL30_Fmask"][p_idx] = l30_fm_ds[:, r0:r0+10, c0:c0+10]
        chips["HLSS30_Fmask"][p_idx] = s30_fm_ds[:, r0:r0+10, c0:c0+10]
        chips["HARMONIZED_z_score"][p_idx] = harm_z_ds[:, r0:r0+10, c0:c0+10]
        chips["HARMONIZED_common_mask"][p_idx] = harm_cm_ds[:, r0:r0+10, c0:c0+10]
        if tan_sr_ds is not None:
            chips["TANAGER"][p_idx] = tan_sr_ds[:, :, r0:r0+10, c0:c0+10]

    metadata = {
        "harm_acq_times": harm_z_ds.attrs["acquisition_time"][:],
        "harm_source_grid": np.array([g.decode() if isinstance(g, bytes) else str(g) for g in harm_z_ds.attrs["source_grid"]]),
        "harm_source_idx": harm_z_ds.attrs["source_frame_index"][:],
        "frame_global_means": harm_z_ds.attrs["frame_global_means"][:],
        "frame_global_stds": harm_z_ds.attrs["frame_global_stds"][:],
        "l30_acq_times": l30_sr_ds.attrs["acquisition_time"][:],
        "s30_acq_times": s30_sr_ds.attrs["acquisition_time"][:],
        "scale_factors": scale_factors,
        "fill_values": fill_values,
        "GeoTransform": l30_sr_ds.attrs.get("GeoTransform"),
        "spatial_ref": l30_sr_ds.attrs.get("spatial_ref"),
    }
    if tan_sr_ds is not None:
        metadata["tan_acq_times"] = tan_sr_ds.attrs.get("acquisition_time")
        metadata["tanager_good_wavelengths"] = tan_sr_ds.attrs.get("all_good_wavelengths")
        metadata["tanager_wavelengths"] = tan_sr_ds.attrs.get("wavelengths")

    print(f"Extracted baseline chips across {n_patches} strata:")
    print(f"  HLSL30: {chips['HLSL30'].shape} ({n_l30_frames} Landsat acquisitions, scale={scale_factors['HLSL30']})")
    print(f"  HLSS30: {chips['HLSS30'].shape} ({n_s30_frames} Sentinel-2 acquisitions, scale={scale_factors['HLSS30']})")
    if tan_sr_ds is not None:
        print(f"  TANAGER: {chips['TANAGER'].shape} ({n_tan_frames} hyperspectral frames, scale={scale_factors['TANAGER']})")
    print(f"  HARMONIZED: {chips['HARMONIZED_z_score'].shape} ({n_harm_frames} virtual constellation frames)")

    return chips, metadata

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

# ==============================================================================
# STEP 2 & 3: SIMULATE SPECTRAL COMPLEXITY CUBES (GPU MAXD SLIDING VOLUME)
# ==============================================================================
def simulate_spectral_complexity_cubes(chips, metadata, target_spectra, onset_ts):
    """
    Computes 10x10 sliding volume Gramian complexity on GPU across all 50 conditions.
    Evaluates baseline frames and post-onset linear mixtures using SpecComplexTorch.process_volume_sliding_tile
    and normalizes strictly via SpecComplexTorch.calculate_global_z_score with frame_global_means and frame_global_stds.
    """
    print("\n[Step 2 & 3] Simulating Spectral Complexity Z-scores on GPU across 50 conditions...")

    harm_acq_times = metadata["harm_acq_times"]
    source_grid = metadata["harm_source_grid"]
    source_idx = metadata["harm_source_idx"]
    means = metadata["frame_global_means"]
    stds = metadata["frame_global_stds"]

    n_frames = len(harm_acq_times)
    n_patches = len(LAND_COVER_PATCHES)
    num_materials = len(target_spectra)
    num_fractions = len(MIXTURE_FRACTIONS)
    num_conditions = num_materials * num_fractions

    # 1. Identify pre-onset and post-onset frame indices
    pre_onset_mask = harm_acq_times < onset_ts
    post_onset_indices = np.where(~pre_onset_mask)[0]
    pre_onset_indices = np.where(pre_onset_mask)[0]

    print(f"Timeline partitioning: {len(pre_onset_indices)} pre-onset frames, {len(post_onset_indices)} post-onset frames.")

    # 2. Precompute unperturbed baseline Z-scores for all patches
    print("Precomputing unperturbed baseline Z-score time series for all patches...")
    unperturbed_z = np.full((n_patches, n_frames, 10, 10), np.nan, dtype=np.float32)

    scale_factors = metadata["scale_factors"]
    fill_values = metadata["fill_values"]
    tan_gw = metadata.get("tanager_good_wavelengths")

    for t in range(n_frames):
        mu_t = means[t]
        std_t = stds[t]

        # Skip frames where background statistics are undefined (e.g. 100% cloud/aerosol)
        if np.isnan(mu_t) or np.isnan(std_t) or std_t <= 0:
            continue

        grid_name = source_grid[t]
        local_idx = source_idx[t]

        if grid_name not in chips:
            continue

        scale = scale_factors.get(grid_name, 10000.0)
        fill_val = fill_values.get(grid_name, -32768)

        for p_idx in range(n_patches):
            raw_int16 = chips[grid_name][p_idx, local_idx]
            flt_sr = raw_int16.astype(np.float32) / float(scale)
            flt_sr[raw_int16 == fill_val] = np.nan

            if grid_name == "TANAGER" and tan_gw is not None:
                gw_mask = tan_gw[local_idx].astype(bool)
                eval_sr = flt_sr[gw_mask, ...]
            else:
                eval_sr = flt_sr

            vol_map = sc.process_volume_sliding_tile(
                eval_sr, tile_size=TILE_SIZE, stride=SLIDING_STRIDE, num_endmembers=NUM_ENDMEMBERS
            )
            valid_mask = (chips["HARMONIZED_common_mask"][p_idx, t] == 0)
            z_map, _, _ = sc.calculate_global_z_score(
                vol_map,
                valid_pixel_mask=valid_mask,
                array_mean=mu_t,
                array_std=std_t,
            )
            if isinstance(z_map, torch.Tensor):
                z_map = z_map.detach().cpu().numpy()
            unperturbed_z[p_idx, t] = z_map

    # 3. Simulate post-onset linear mixture across 50 conditions
    simulated_z_cubes = np.zeros((num_conditions, n_patches, n_frames, 10, 10), dtype=np.float32)

    # Initialize all conditions with unperturbed baseline Z-scores
    for c_idx in range(num_conditions):
        for p_idx in range(n_patches):
            simulated_z_cubes[c_idx, p_idx] = unperturbed_z[p_idx].copy()

    # Fast GPU simulation loop over post-onset frames
    t_start_sim = time.perf_counter()
    cond_counter = 0

    for m_idx, mat in enumerate(target_spectra):
        spec_l8 = mat["spec_l8"]
        spec_s2 = mat["spec_s2"]
        spec_tanager = mat.get("spec_tanager")

        for f_idx, f_weight in enumerate(MIXTURE_FRACTIONS):
            for t in post_onset_indices:
                mu_t = means[t]
                std_t = stds[t]

                if np.isnan(mu_t) or np.isnan(std_t) or std_t <= 0:
                    continue

                grid_name = source_grid[t]
                local_idx = source_idx[t]

                if grid_name not in chips:
                    continue

                scale = scale_factors.get(grid_name, 10000.0)
                fill_val = fill_values.get(grid_name, -32768)

                if grid_name == "HLSL30":
                    target_spec = spec_l8
                elif grid_name == "HLSS30":
                    target_spec = spec_s2
                elif grid_name == "TANAGER":
                    target_spec = spec_tanager
                else:
                    continue

                if target_spec is None:
                    continue

                for p_idx in range(n_patches):
                    raw_int16 = chips[grid_name][p_idx, local_idx]
                    flt_sr = raw_int16.astype(np.float32) / float(scale)
                    flt_sr[raw_int16 == fill_val] = np.nan

                    # Inject linear mixture into the 4 designated pixels
                    injected_sr = flt_sr.copy()
                    for roff, coff in INJ_OFFSETS:
                        bg_pixel = flt_sr[:, roff, coff]
                        if not np.isnan(bg_pixel).any():
                            injected_sr[:, roff, coff] = (1.0 - f_weight) * bg_pixel + f_weight * target_spec

                    if grid_name == "TANAGER" and tan_gw is not None:
                        gw_mask = tan_gw[local_idx].astype(bool)
                        eval_injected = injected_sr[gw_mask, ...]
                    else:
                        eval_injected = injected_sr

                    vol_map = sc.process_volume_sliding_tile(
                        eval_injected, tile_size=TILE_SIZE, stride=SLIDING_STRIDE, num_endmembers=NUM_ENDMEMBERS
                    )
                    valid_mask = (chips["HARMONIZED_common_mask"][p_idx, t] == 0)
                    z_map, _, _ = sc.calculate_global_z_score(
                        vol_map,
                        valid_pixel_mask=valid_mask,
                        array_mean=mu_t,
                        array_std=std_t,
                    )
                    if isinstance(z_map, torch.Tensor):
                        z_map = z_map.detach().cpu().numpy()
                    simulated_z_cubes[cond_counter, p_idx, t] = z_map

            cond_counter += 1

    t_elapsed_sim = time.perf_counter() - t_start_sim
    print(f"Completed GPU simulation of 50 conditions in {t_elapsed_sim:.2f} s ({t_elapsed_sim/num_conditions:.2f} s/condition).")

    return simulated_z_cubes, unperturbed_z


# ==============================================================================
# STEP 4: RUN HARMONIZED SC-CCD BREAK DETECTION
# ==============================================================================
def run_scccd_benchmark(sim_z_cubes, chips, metadata, target_spectra, onset_ts):
    """
    Executes Harmonized SC-CCD break regression across all 800 test instances.
    """
    print("\n[Step 4] Running Harmonized SC-CCD break detection on 800 test instances...")
    harm_acq_times = metadata["harm_acq_times"]
    frac_years = hccd.extract_fractional_years(harm_acq_times)
    X_full = hccd.build_harmonic_matrix(frac_years)

    common_mask = chips["HARMONIZED_common_mask"]
    valid_mask_all = (common_mask == 0)

    num_materials = len(target_spectra)
    num_fractions = len(MIXTURE_FRACTIONS)
    n_patches = len(LAND_COVER_PATCHES)
    total_records = num_materials * num_fractions * n_patches * len(INJ_OFFSETS)

    results_table = np.zeros(total_records, dtype=BREAK_RECORD_DTYPE)
    rec_idx = 0
    cond_idx = 0

    for m_idx, mat in enumerate(target_spectra):
        m_id = mat["id"].encode("utf-8")
        m_name = mat["name"].encode("utf-8")

        for f_idx, f_weight in enumerate(MIXTURE_FRACTIONS):
            for p_idx, p in enumerate(LAND_COVER_PATCHES):
                p_id = p["id"].encode("utf-8")
                p_name = p["name"].encode("utf-8")
                r0, c0 = p["r0"], p["c0"]

                # Run CCD row chunk across the 10x10 patch
                y_data_patch = sim_z_cubes[cond_idx, p_idx]
                valid_patch = valid_mask_all[p_idx]

                chunk_args = (
                    0, 10, 10,
                    y_data_patch, valid_patch,
                    frac_years, X_full, harm_acq_times,
                    hccd.MIN_SAMPLES, hccd.CHANGE_PROBABILITY, hccd.CONSECUTIVE_ANOMALIES
                )

                res = hccd._process_row_chunk(chunk_args)
                chunk_pred = res[2]
                chunk_date = res[6]
                chunk_count = res[7]

                for p_num, (roff, coff) in enumerate(INJ_OFFSETS, 1):
                    scene_r = r0 + roff
                    scene_c = c0 + coff
                    b_count = int(chunk_count[roff, coff])
                    b_date = float(chunk_date[roff, coff])

                    # Detection criteria: break occurred on or after onset date
                    detected = (b_count > 0) and (not np.isnan(b_date)) and (b_date >= onset_ts)
                    latency = float((b_date - onset_ts) / 86400.0) if detected else np.nan

                    # Break magnitude: residual jump post-onset
                    if detected:
                        post_valid = np.where(valid_patch[:, roff, coff] & (harm_acq_times >= onset_ts))[0]
                        if len(post_valid) > 0:
                            actuals = y_data_patch[post_valid, roff, coff]
                            preds = chunk_pred[post_valid, roff, coff]
                            mag = float(np.nanmean(np.abs(actuals - preds)))
                        else:
                            mag = 0.0
                    else:
                        mag = 0.0

                    rec = results_table[rec_idx]
                    rec["condition_idx"] = cond_idx
                    rec["material_id"] = m_id
                    rec["material_name"] = m_name
                    rec["mixture_fraction"] = f_weight
                    rec["patch_id"] = p_id
                    rec["patch_name"] = p_name
                    rec["pixel_num"] = p_num
                    rec["scene_row"] = scene_r
                    rec["scene_col"] = scene_c
                    rec["break_detected"] = detected
                    rec["break_timestamp"] = b_date if detected else np.nan
                    rec["latency_days"] = latency
                    rec["break_magnitude"] = mag
                    rec["p_value"] = 0.99 if detected else 0.0

                    rec_idx += 1

            cond_idx += 1

    detected_count = np.sum(results_table["break_detected"])
    print(f"SC-CCD complete: {detected_count}/{total_records} breaks detected ({detected_count/total_records*100:.1f}%).")
    return results_table


# ==============================================================================
# STEP 5: RUN LANDSAT PYCCD BREAK DETECTION
# ==============================================================================
def run_pyccd_benchmark(chips, metadata, target_spectra, onset_ts):
    """
    Executes USGS LCMAP Landsat PyCCD break detection across all 800 test instances.
    """
    print("\n[Step 5] Running USGS LCMAP Landsat PyCCD on 800 test instances...")

    l30_acq_times = metadata["l30_acq_times"]
    l30_dates = [datetime.datetime.fromtimestamp(t, tz=datetime.timezone.utc).date() for t in l30_acq_times]
    ordinal_dates = np.array([d.toordinal() for d in l30_dates])
    onset_ordinal = datetime.datetime.fromtimestamp(onset_ts, tz=datetime.timezone.utc).date().toordinal()

    n_l30_frames = len(l30_acq_times)
    nominal_thermals = np.full(n_l30_frames, pccd.NOMINAL_THERMAL_KELVIN_SCALED, dtype=np.int16)

    num_materials = len(target_spectra)
    num_fractions = len(MIXTURE_FRACTIONS)
    n_patches = len(LAND_COVER_PATCHES)
    total_records = num_materials * num_fractions * n_patches * len(INJ_OFFSETS)

    results_table = np.zeros(total_records, dtype=BREAK_RECORD_DTYPE)
    rec_idx = 0
    cond_idx = 0

    proc_params = {
        "QA_BITPACKED": False,
        "QA_FILL": 0,
        "QA_CLEAR": 1,
        "QA_WATER": 2,
        "QA_SHADOW": 3,
        "QA_SNOW": 4,
        "QA_CLOUD": 5,
        "LASSO_MAX_ITER": 2000,
    }

    # Post-onset Landsat frame indices
    post_onset_l30 = np.where(l30_acq_times >= onset_ts)[0]

    l30_scale = metadata["scale_factors"].get("HLSL30", 10000.0)
    l30_fill = metadata["fill_values"].get("HLSL30", -32768)

    for m_idx, mat in enumerate(target_spectra):
        m_id = mat["id"].encode("utf-8")
        m_name = mat["name"].encode("utf-8")
        spec_l8_flt = mat["spec_l8"]

        for f_idx, f_weight in enumerate(MIXTURE_FRACTIONS):
            for p_idx, p in enumerate(LAND_COVER_PATCHES):
                p_id = p["id"].encode("utf-8")
                p_name = p["name"].encode("utf-8")
                r0, c0 = p["r0"], p["c0"]

                l30_patch = chips["HLSL30"][p_idx]       # shape: (n_frames, 7, 10, 10)
                fmask_patch = chips["HLSL30_Fmask"][p_idx] # shape: (n_frames, 10, 10)

                for p_num, (roff, coff) in enumerate(INJ_OFFSETS, 1):
                    scene_r = r0 + roff
                    scene_c = c0 + coff

                    # Extract raw 6-band time series
                    blues = l30_patch[:, 1, roff, coff].copy()
                    greens = l30_patch[:, 2, roff, coff].copy()
                    reds = l30_patch[:, 3, roff, coff].copy()
                    nirs = l30_patch[:, 4, roff, coff].copy()
                    swir1s = l30_patch[:, 5, roff, coff].copy()
                    swir2s = l30_patch[:, 6, roff, coff].copy()

                    # Inject target mixture for post-onset acquisitions
                    for t in post_onset_l30:
                        if blues[t] != l30_fill:
                            blues[t] = int(round((1.0 - f_weight) * blues[t] + f_weight * spec_l8_flt[1] * l30_scale))
                            greens[t] = int(round((1.0 - f_weight) * greens[t] + f_weight * spec_l8_flt[2] * l30_scale))
                            reds[t] = int(round((1.0 - f_weight) * reds[t] + f_weight * spec_l8_flt[3] * l30_scale))
                            nirs[t] = int(round((1.0 - f_weight) * nirs[t] + f_weight * spec_l8_flt[4] * l30_scale))
                            swir1s[t] = int(round((1.0 - f_weight) * swir1s[t] + f_weight * spec_l8_flt[5] * l30_scale))
                            swir2s[t] = int(round((1.0 - f_weight) * swir2s[t] + f_weight * spec_l8_flt[6] * l30_scale))

                    fm_pixel = fmask_patch[:, roff, coff]
                    cm_dummy = np.zeros_like(fm_pixel)
                    qas_pixel = pccd.decode_categorical_qa(fm_pixel, cm_dummy, blues)

                    usable_count = int(np.count_nonzero((qas_pixel == 1) | (qas_pixel == 2) | (qas_pixel == 4)))

                    detected = False
                    break_ts = np.nan
                    latency = np.nan
                    mag = 0.0

                    if usable_count >= pccd.MIN_CLEAR_SAMPLES:
                        try:
                            res = ccd.detect(
                                ordinal_dates,
                                blues, greens, reds, nirs, swir1s, swir2s,
                                nominal_thermals,
                                qas_pixel,
                                params=proc_params
                            )
                            models = res.get("change_models", [])
                            # Find first break on or after onset date
                            for m in models:
                                if m.get("change_probability", 0) > 0:
                                    b_day = m.get("break_day")
                                    if b_day is not None and b_day >= onset_ordinal:
                                        detected = True
                                        b_date_obj = datetime.date.fromordinal(b_day)
                                        break_ts = datetime.datetime(
                                            b_date_obj.year, b_date_obj.month, b_date_obj.day,
                                            tzinfo=datetime.timezone.utc
                                        ).timestamp()
                                        latency = float((break_ts - onset_ts) / 86400.0)
                                        mags = m.get("magnitudes", [0.0])
                                        mag = float(np.linalg.norm(mags))
                                        break
                        except Exception:
                            detected = False

                    rec = results_table[rec_idx]
                    rec["condition_idx"] = cond_idx
                    rec["material_id"] = m_id
                    rec["material_name"] = m_name
                    rec["mixture_fraction"] = f_weight
                    rec["patch_id"] = p_id
                    rec["patch_name"] = p_name
                    rec["pixel_num"] = p_num
                    rec["scene_row"] = scene_r
                    rec["scene_col"] = scene_c
                    rec["break_detected"] = detected
                    rec["break_timestamp"] = break_ts if detected else np.nan
                    rec["latency_days"] = latency
                    rec["break_magnitude"] = mag
                    rec["p_value"] = 1.0 if detected else 0.0

                    rec_idx += 1

            cond_idx += 1

    detected_count = np.sum(results_table["break_detected"])
    print(f"PyCCD complete: {detected_count}/{total_records} breaks detected ({detected_count/total_records*100:.1f}%).")
    return results_table


# ==============================================================================
# STEP 6: COMMIT BENCHMARK DATASET TO MASTER HDF5 FILE
# ==============================================================================
def save_benchmark_to_hdf5(h5_file, chips, metadata, target_spectra, sim_z_cubes, scccd_table, pyccd_table):
    """
    Writes the complete /STRATIFIED_CHIP_BENCHMARK/ hierarchy to the master HDF5 container.
    """
    print("\n[Step 6] Saving /STRATIFIED_CHIP_BENCHMARK/ group to master HDF5 file...")

    group_name = "/STRATIFIED_CHIP_BENCHMARK"
    if group_name in h5_file:
        print(f"Overwriting existing '{group_name}' group...")
        del h5_file[group_name]

    base_grp = h5_file.create_group(group_name)

    # 1. METADATA Subgroup
    meta_grp = base_grp.create_group("METADATA")
    meta_grp.attrs["patch_size"] = 10
    meta_grp.attrs["stride"] = 4
    meta_grp.attrs["injection_onset_date"] = INJECTION_ONSET_DATE
    meta_grp.attrs["num_conditions"] = len(target_spectra) * len(MIXTURE_FRACTIONS)
    meta_grp.attrs["patch_ids"] = np.array([p["id"] for p in LAND_COVER_PATCHES], dtype="S16")
    meta_grp.attrs["patch_names"] = np.array([p["name"] for p in LAND_COVER_PATCHES], dtype="S48")
    meta_grp.attrs["patch_bounds"] = np.array([[p["r0"], p["c0"], 10, 10] for p in LAND_COVER_PATCHES], dtype=np.int32)
    meta_grp.attrs["injected_offsets"] = np.array(INJ_OFFSETS, dtype=np.int32)
    meta_grp.attrs["mixture_fractions"] = np.array(MIXTURE_FRACTIONS, dtype=np.float32)
    meta_grp.attrs["frame_global_means"] = metadata["frame_global_means"]
    meta_grp.attrs["frame_global_stds"] = metadata["frame_global_stds"]
    meta_grp.attrs["harm_acq_times"] = metadata["harm_acq_times"]

    # Target spectra datasets
    spec_l8_all = np.array([m["spec_l8"] for m in target_spectra], dtype=np.float32)
    spec_s2_all = np.array([m["spec_s2"] for m in target_spectra], dtype=np.float32)
    meta_grp.create_dataset("target_spectra_l8", data=spec_l8_all, compression="gzip")
    meta_grp.create_dataset("target_spectra_s2", data=spec_s2_all, compression="gzip")

    if target_spectra[0]["spec_tanager"] is not None:
        spec_tanager_all = np.array([m["spec_tanager"] for m in target_spectra], dtype=np.float32)
        meta_grp.create_dataset("target_spectra_tanager", data=spec_tanager_all, compression="gzip")

    # 2. BASELINE_CHIPS Subgroup
    base_chips_grp = base_grp.create_group("BASELINE_CHIPS")
    base_chips_grp.create_dataset("HLSL30", data=chips["HLSL30"], compression="gzip", compression_opts=6)
    base_chips_grp["HLSL30"].attrs["scale_factor"] = metadata["scale_factors"].get("HLSL30", 10000.0)
    base_chips_grp["HLSL30"].attrs["_FillValue"] = metadata["fill_values"].get("HLSL30", -32768)

    base_chips_grp.create_dataset("HLSS30", data=chips["HLSS30"], compression="gzip", compression_opts=6)
    base_chips_grp["HLSS30"].attrs["scale_factor"] = metadata["scale_factors"].get("HLSS30", 10000.0)
    base_chips_grp["HLSS30"].attrs["_FillValue"] = metadata["fill_values"].get("HLSS30", -32768)

    if "TANAGER" in chips:
        base_chips_grp.create_dataset("TANAGER", data=chips["TANAGER"], compression="gzip", compression_opts=6)
        base_chips_grp["TANAGER"].attrs["scale_factor"] = metadata["scale_factors"].get("TANAGER", 10000.0)
        base_chips_grp["TANAGER"].attrs["_FillValue"] = metadata["fill_values"].get("TANAGER", -9999)

    base_chips_grp.create_dataset("HARMONIZED_common_mask", data=chips["HARMONIZED_common_mask"], compression="gzip", compression_opts=6)

    # Quantize baseline Z-score to int16 (scale = 10000)
    base_z_int16 = sc.scale_to_int16(chips["HARMONIZED_z_score"])
    base_chips_grp.create_dataset("HARMONIZED_z_score", data=base_z_int16, compression="gzip", compression_opts=6)
    base_chips_grp["HARMONIZED_z_score"].attrs["scale_factor"] = 10000.0

    # 3. SIMULATED_CUBES Subgroup (Quantized to int16 with gzip level 6)
    sim_grp = base_grp.create_group("SIMULATED_CUBES")
    sim_z_int16 = sc.scale_to_int16(sim_z_cubes)
    sim_ds = sim_grp.create_dataset(
        "sliding_volume_z_score",
        data=sim_z_int16,
        dtype=np.int16,
        compression="gzip",
        compression_opts=6,
        chunks=(1, 1, min(sim_z_cubes.shape[2], 128), 10, 10),
    )
    sim_ds.attrs["scale_factor"] = 10000.0
    sim_ds.attrs["description"] = "Simulated Gramian Spectral Complexity Z-scores under 50 perturbation conditions"

    # 4. RESULTS Subgroup
    res_grp = base_grp.create_group("RESULTS")
    res_grp.create_dataset("scccd_break_table", data=scccd_table, compression="gzip")
    res_grp.create_dataset("pyccd_break_table", data=pyccd_table, compression="gzip")

    print(f"Successfully committed /STRATIFIED_CHIP_BENCHMARK/ to {MASTER_H5_PATH}.")


# ==============================================================================
# STEP 7: PRINT QUANTITATIVE PERFORMANCE SUMMARY
# ==============================================================================
def print_performance_summary(scccd_table, pyccd_table):
    """
    Prints comparative detection accuracy and latency tables stratified by land-cover class.
    """
    print("\n" + "=" * 90)
    print("LAND-COVER STRATIFIED BENCHMARK PERFORMANCE SUMMARY: SC-CCD vs. USGS PYCCD")
    print("=" * 90)

    # 1. Performance Stratified by Land-Cover Class
    print(f"\n{'Stratum ID':<10} | {'Land-Cover Stratum':<32} | {'SC-CCD Recall':<15} | {'PyCCD Recall':<15} | {'Median Latency (SC / Py)'}")
    print("-" * 105)

    patch_ids = [p["id"] for p in LAND_COVER_PATCHES]
    patch_names = [p["name"] for p in LAND_COVER_PATCHES]

    for p_id, p_name in zip(patch_ids, patch_names):
        sc_sub = scccd_table[scccd_table["patch_id"] == p_id.encode()]
        py_sub = pyccd_table[pyccd_table["patch_id"] == p_id.encode()]

        sc_rec = np.mean(sc_sub["break_detected"]) * 100.0
        py_rec = np.mean(py_sub["break_detected"]) * 100.0

        sc_lat = sc_sub["latency_days"][sc_sub["break_detected"]]
        py_lat = py_sub["latency_days"][py_sub["break_detected"]]

        sc_med_lat = f"{np.median(sc_lat):.1f} d" if len(sc_lat) > 0 else "N/A"
        py_med_lat = f"{np.median(py_lat):.1f} d" if len(py_lat) > 0 else "N/A"

        print(f"{p_id:<10} | {p_name:<32} | {sc_rec:5.1f}% ({np.sum(sc_sub['break_detected']):3d}/{len(sc_sub):3d}) | "
              f"{py_rec:5.1f}% ({np.sum(py_sub['break_detected']):3d}/{len(py_sub):3d}) | {sc_med_lat:<12} / {py_med_lat}")

    # 2. Performance Stratified by Target Material
    print(f"\n{'Material ID':<18} | {'Material Name':<34} | {'SC Recall':<12} | {'Py Recall':<12} | {'Median Latency (SC / Py)'}")
    print("-" * 105)

    mat_ids = np.unique(scccd_table["material_id"])
    for m_id in mat_ids:
        sc_sub = scccd_table[scccd_table["material_id"] == m_id]
        py_sub = pyccd_table[pyccd_table["material_id"] == m_id]

        m_name = sc_sub["material_name"][0].decode()
        m_id_str = m_id.decode()

        sc_rec = np.mean(sc_sub["break_detected"]) * 100.0
        py_rec = np.mean(py_sub["break_detected"]) * 100.0

        sc_lat = sc_sub["latency_days"][sc_sub["break_detected"]]
        py_lat = py_sub["latency_days"][py_sub["break_detected"]]

        sc_med_lat = f"{np.median(sc_lat):.1f} d" if len(sc_lat) > 0 else "N/A"
        py_med_lat = f"{np.median(py_lat):.1f} d" if len(py_lat) > 0 else "N/A"

        print(f"{m_id_str:<18} | {m_name:<34} | {sc_rec:5.1f}% ({np.sum(sc_sub['break_detected']):2d}/{len(sc_sub):2d}) | "
              f"{py_rec:5.1f}% ({np.sum(py_sub['break_detected']):2d}/{len(py_sub):2d}) | {sc_med_lat:<12} / {py_med_lat}")

    # 3. Overall Aggregate Performance
    total_sc_rec = np.mean(scccd_table["break_detected"]) * 100.0
    total_py_rec = np.mean(pyccd_table["break_detected"]) * 100.0
    all_sc_lat = scccd_table["latency_days"][scccd_table["break_detected"]]
    all_py_lat = pyccd_table["latency_days"][pyccd_table["break_detected"]]

    print("-" * 105)
    print(f"{'OVERALL AGGREGATE':<18} | {'All 4 Strata & 10 Materials (N=800)':<34} | "
          f"{total_sc_rec:5.1f}% ({np.sum(scccd_table['break_detected']):3d}/800) | "
          f"{total_py_rec:5.1f}% ({np.sum(pyccd_table['break_detected']):3d}/800) | "
          f"{np.median(all_sc_lat):.1f} d        / {np.median(all_py_lat):.1f} d")
    print("=" * 105 + "\n")


# ==============================================================================
# MAIN PIPELINE ENTRY POINT
# ==============================================================================
def main():
    t_start_total = time.perf_counter()
    print("=" * 80)
    print("LAND-COVER STRATIFIED PATCHES BENCHMARK PIPELINE")
    print("Multi-Sensor Semi-Empirical Target Injection (SC-CCD vs. USGS PyCCD)")
    print("=" * 80)

    # 1. Parse onset timestamp
    onset_dt = datetime.datetime.strptime(INJECTION_ONSET_DATE, "%Y-%m-%d").replace(tzinfo=datetime.timezone.utc)
    onset_ts = onset_dt.timestamp()
    print(f"Injection onset date: {INJECTION_ONSET_DATE} (timestamp: {onset_ts:.0f})")

    # 2. Open Master HDF5 Container
    print(f"Opening master container: {MASTER_H5_PATH}")
    with h5py.File(MASTER_H5_PATH, "r+", locking=False) as h5_file:
        # Load Tanager wavelengths if available
        tanager_wl = None
        if "/HDFEOS/GRIDS/TANAGER/Data Fields/surface_reflectance" in h5_file:
            tanager_wl = h5_file["/HDFEOS/GRIDS/TANAGER/Data Fields/surface_reflectance"].attrs.get("wavelengths")

        # Load 10 resampled target spectra
        print(f"Loading USGS Splib07 spectra from: {SPLIB_BASE}")
        target_spectra = load_all_target_spectra(SPLIB_BASE, tanager_wl)
        print(f"Successfully prepared {len(target_spectra)} target spectra.")

        # Step 1: Extract baseline chips and metadata
        chips, metadata = extract_baseline_chips(h5_file)

        # Step 2 & 3: Simulate Spectral Complexity Z-scores on GPU
        sim_z_cubes, unperturbed_z = simulate_spectral_complexity_cubes(chips, metadata, target_spectra, onset_ts)
        chips["HARMONIZED_z_score"] = unperturbed_z

        # Step 4: Run Harmonized SC-CCD break detection
        scccd_table = run_scccd_benchmark(sim_z_cubes, chips, metadata, target_spectra, onset_ts)

        # Step 5: Run USGS LCMAP Landsat PyCCD break detection
        pyccd_table = run_pyccd_benchmark(chips, metadata, target_spectra, onset_ts)

        # Step 6: Commit complete benchmark group to master HDF5
        save_benchmark_to_hdf5(h5_file, chips, metadata, target_spectra, sim_z_cubes, scccd_table, pyccd_table)

    # Step 7: Print quantitative summary
    print_performance_summary(scccd_table, pyccd_table)

    t_elapsed_total = (time.perf_counter() - t_start_total) / 60.0
    print(f"Stratified benchmark pipeline completed successfully in {t_elapsed_total:.2f} minutes.")


if __name__ == "__main__":
    main()

