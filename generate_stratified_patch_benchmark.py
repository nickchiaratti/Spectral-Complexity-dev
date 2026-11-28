"""
generate_stratified_patch_benchmark.py
======================================
Executes the Land-Cover Stratified Patches semi-empirical target injection experiment
and saves all chips, simulated complexity cubes, and algorithmic break detection results
directly into the master HDF5 container under /STRATIFIED_CHIP_BENCHMARK/.

Reuses core functions and standards from:
  - SpecComplexTorch.py: Sliding window MaxD Gramian volume (maxVol) and 3D summation equivalent volume (3D) unmixing on GPU.
  - SC_CCM_main_numba.py: Numba-accelerated Harmonized RLS-LASSO continuous change detection engine.
  - pyccd_landsat_main.py: USGS LCMAP Landsat PyCCD detection engine and QA decoding.
  - create_injected_dataset.py: USGS Splib07 multi-sensor resampled spectral library.

Outputs saved to master HDF5 container:
  - /STRATIFIED_CHIP_BENCHMARK/    (Maximum Volume, sliding_volume_z_score)
  - /STRATIFIED_CHIP_BENCHMARK_3D/ (3D Summation Equivalent Volume, volume_map_3d_zscore)

Experimental Structure:
  - 4 Land-Cover Strata (10x10 pixels, 300 m x 300 m, 9.0 ha each):
      1. AGRI: Rush Open Farmland (r0=970, c0=479)
      2. FOREST: Mendon Ponds Park Forest (r0=1146, c0=530)
      3. WETLAND: Thousand Acre Swamp, Penfield (r0=721, c0=791)
      4. SYNTHETIC: Eastview Commercial Complex (r0=1154, c0=888)
  - 4 Decoupled Injected Pixels per patch (stride = 4, 3 baseline pixels separation):
      Offsets: [(2, 2), (2, 6), (6, 2), (6, 6)]
  - 10 USGS Splib07 Target Materials x 21 Mixture Fractions [0.00, 0.05, ..., 1.00]
      = 210 Distinct Perturbation Conditions
  - 210 conditions x 4 strata x 4 pixels = 3,360 test instances per algorithm.
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
import scipy.interpolate
import torch
from joblib import Parallel, delayed
from tqdm import tqdm

# Ensure single-threaded BLAS/OpenMP execution for workers
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

# Ensure repository root and core subdirectories are on sys.path
script_dir = Path(__file__).resolve().parent
for _p in [
    script_dir / "change_detection" / "pyccd",
    script_dir / "change_detection" / "SC-CCM",
    script_dir,
]:
    p_str = str(_p)
    if _p.exists() and p_str not in sys.path:
        sys.path.insert(0, p_str)

# Import core modules
import SpecComplexTorch as sc
import SC_CCM_main_numba as hccd
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
NUM_ENDMEMBERS_3D = 20
SEPARABILITY_THRESHOLD_3D = 0.01

# Spectral complexity benchmark methods evaluated on every run
BENCHMARK_METRICS = [
    {
        "key": "sliding_volume_z_score",
        "label": "Maximum Volume (maxVol)",
        "group_name": "/STRATIFIED_CHIP_BENCHMARK",
        "description": "Simulated Gramian Spectral Complexity Z-scores across full mixture fraction range (0.00 to 1.00)",
    },
    {
        "key": "volume_map_3d_zscore",
        "label": "3D Summation Equivalent Volume (3D)",
        "group_name": "/STRATIFIED_CHIP_BENCHMARK_3D",
        "description": "Simulated 3D Equivalent Spectral Complexity Z-scores across full mixture fraction range (0.00 to 1.00)",
    },
]

# Sentinel-2 band subset indices matching HLSS30 (10 bands from 13-band splib07 resampled S2)
S2_SUB_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 11, 12]

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

# Fractional abundance levels (0% to 100% in 5% increments, matching generate_patch_benchmark.py)
MIXTURE_FRACTIONS = np.round(np.arange(0.00, 1.05, step=0.05), 2)

# Compound data type for tabular results
BREAK_RECORD_DTYPE = np.dtype([
    ("condition_idx", np.int16),        # 0 to num_conditions - 1
    ("material_id", "S32"),             # Target material identifier string
    ("material_name", "S48"),           # Material descriptive name
    ("mixture_fraction", np.float32),   # 0.00 to 1.00 in 0.05 increments
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
def extract_baseline_chips(h5_file, target_metrics=None):
    """
    Extracts the unperturbed 10x10 time series for each of the 4 patches across sensors,
    along with global normalization statistics for each requested spectral complexity metric.
    Dynamically identifies scale_factor and _FillValue attributes from each dataset.
    """
    print("\n[Step 1] Extracting unperturbed 10x10 baseline chips from master scene...")

    if target_metrics is None:
        target_metrics = [m["key"] for m in BENCHMARK_METRICS]
    elif isinstance(target_metrics, str):
        target_metrics = [target_metrics]

    harm_grp = h5_file["/HDFEOS/GRIDS/HARMONIZED/Data Fields"]
    l30_grp = h5_file["/HDFEOS/GRIDS/HLSL30/Data Fields"]
    s30_grp = h5_file["/HDFEOS/GRIDS/HLSS30/Data Fields"]
    tan_grp = h5_file.get("/HDFEOS/GRIDS/TANAGER/Data Fields")

    missing_metrics = [m for m in target_metrics if m not in harm_grp]
    if missing_metrics:
        raise RuntimeError(
            f"Required Harmonized Spectral Complexity dataset(s) {missing_metrics} not found in "
            f"'{harm_grp.name}'. Run HLST_SC_calculations.py with metric='all' first."
        )

    primary_metric = target_metrics[0]
    harm_z_ds = harm_grp[primary_metric]
    harm_cm_ds = harm_grp["common_mask"]
    l30_sr_ds = l30_grp["surface_reflectance"]
    l30_fm_ds = l30_grp["Fmask"]
    l30_cm_ds = l30_grp.get("common_mask")
    s30_sr_ds = s30_grp["surface_reflectance"]
    s30_fm_ds = s30_grp["Fmask"]
    s30_cm_ds = s30_grp.get("common_mask")

    n_harm_frames = harm_z_ds.shape[0]
    n_l30_frames = l30_sr_ds.shape[0]
    n_s30_frames = s30_sr_ds.shape[0]
    n_patches = len(LAND_COVER_PATCHES)

    l30_cm_dtype = l30_cm_ds.dtype if l30_cm_ds is not None else np.bool_
    s30_cm_dtype = s30_cm_ds.dtype if s30_cm_ds is not None else np.bool_

    chips = {
        "HLSL30": np.zeros((n_patches, n_l30_frames, 7, 10, 10), dtype=np.int16),
        "HLSS30": np.zeros((n_patches, n_s30_frames, 10, 10, 10), dtype=np.int16),
        "HLSL30_Fmask": np.zeros((n_patches, n_l30_frames, 10, 10), dtype=np.uint8),
        "HLSS30_Fmask": np.zeros((n_patches, n_s30_frames, 10, 10), dtype=np.uint8),
        "HLSL30_common_mask": np.zeros((n_patches, n_l30_frames, 10, 10), dtype=l30_cm_dtype),
        "HLSS30_common_mask": np.zeros((n_patches, n_s30_frames, 10, 10), dtype=s30_cm_dtype),
        "HARMONIZED_z_score": np.zeros((n_patches, n_harm_frames, 10, 10), dtype=np.float32),
        "HARMONIZED_z_scores": {
            m: np.zeros((n_patches, n_harm_frames, 10, 10), dtype=np.float32)
            for m in target_metrics
        },
        "HARMONIZED_common_mask": np.zeros((n_patches, n_harm_frames, 10, 10), dtype=np.uint8),
    }

    def _read_scale_and_fill(ds, default_fill=-32768):
        if "_FillValue" in ds.attrs:
            fv = int(ds.attrs["_FillValue"])
        elif "fillvalue" in ds.attrs:
            fv = int(ds.attrs["fillvalue"])
        elif "fill_value" in ds.attrs:
            fv = int(ds.attrs["fill_value"])
        elif getattr(ds, "fillvalue", None) is not None and ds.fillvalue != 0:
            fv = int(ds.fillvalue)
        else:
            fv = int(default_fill)

        if "scale_factor" in ds.attrs:
            sc_val = float(ds.attrs["scale_factor"])
        elif "scale_to_float" in ds.attrs:
            sf = float(ds.attrs["scale_to_float"])
            sc_val = (1.0 / sf) if sf < 1.0 else sf
        else:
            sc_val = 10000.0
        return fv, sc_val

    l30_fill, l30_scale = _read_scale_and_fill(l30_sr_ds, default_fill=-32768)
    s30_fill, s30_scale = _read_scale_and_fill(s30_sr_ds, default_fill=-32768)

    scale_factors = {
        "HLSL30": l30_scale,
        "HLSS30": s30_scale,
    }
    fill_values = {
        "HLSL30": l30_fill,
        "HLSS30": s30_fill,
    }

    tan_sr_ds = None
    if tan_grp is not None and "surface_reflectance" in tan_grp:
        tan_sr_ds = tan_grp["surface_reflectance"]
        n_tan_frames = tan_sr_ds.shape[0]
        n_tan_bands = tan_sr_ds.shape[1]
        chips["TANAGER"] = np.zeros((n_patches, n_tan_frames, n_tan_bands, 10, 10), dtype=np.int16)
        tan_fill, tan_scale = _read_scale_and_fill(tan_sr_ds, default_fill=-9999)
        scale_factors["TANAGER"] = tan_scale
        fill_values["TANAGER"] = tan_fill

    for p_idx, p in enumerate(LAND_COVER_PATCHES):
        r0, c0 = p["r0"], p["c0"]
        chips["HLSL30"][p_idx] = l30_sr_ds[:, :, r0:r0+10, c0:c0+10]
        chips["HLSS30"][p_idx] = s30_sr_ds[:, :, r0:r0+10, c0:c0+10]
        chips["HLSL30_Fmask"][p_idx] = l30_fm_ds[:, r0:r0+10, c0:c0+10]
        chips["HLSS30_Fmask"][p_idx] = s30_fm_ds[:, r0:r0+10, c0:c0+10]
        if l30_cm_ds is not None:
            chips["HLSL30_common_mask"][p_idx] = l30_cm_ds[:, r0:r0+10, c0:c0+10]
        if s30_cm_ds is not None:
            chips["HLSS30_common_mask"][p_idx] = s30_cm_ds[:, r0:r0+10, c0:c0+10]
        chips["HARMONIZED_z_score"][p_idx] = harm_z_ds[:, r0:r0+10, c0:c0+10]
        for m in target_metrics:
            chips["HARMONIZED_z_scores"][m][p_idx] = harm_grp[m][:, r0:r0+10, c0:c0+10]
        chips["HARMONIZED_common_mask"][p_idx] = harm_cm_ds[:, r0:r0+10, c0:c0+10]
        if tan_sr_ds is not None:
            chips["TANAGER"][p_idx] = tan_sr_ds[:, :, r0:r0+10, c0:c0+10]

    # Optionally extract baseline PyCCD results if present in master container
    if "PYCCD_Results" in h5_file and "change_date_timestamp" in h5_file["PYCCD_Results"]:
        py_date_ds = h5_file["PYCCD_Results/change_date_timestamp"]
        py_count_ds = h5_file["PYCCD_Results/change_count"]
        baseline_pyccd = {
            "date": np.full((n_patches, 10, 10), np.nan, dtype=np.float64),
            "count": np.zeros((n_patches, 10, 10), dtype=np.int32),
        }
        for p_idx, p in enumerate(LAND_COVER_PATCHES):
            r0, c0 = p["r0"], p["c0"]
            baseline_pyccd["date"][p_idx] = py_date_ds[r0:r0+10, c0:c0+10]
            baseline_pyccd["count"][p_idx] = py_count_ds[r0:r0+10, c0:c0+10]
        chips["BASELINE_PYCCD"] = baseline_pyccd

    metrics_meta = {}
    for m in target_metrics:
        m_ds = harm_grp[m]
        metrics_meta[m] = {
            "frame_global_means": m_ds.attrs["frame_global_means"][:],
            "frame_global_stds": m_ds.attrs["frame_global_stds"][:],
            "spatial_smoothing": bool(m_ds.attrs.get("spatial_smoothing", False)),
        }

    metadata = {
        "harm_acq_times": harm_z_ds.attrs["acquisition_time"][:],
        "harm_source_grid": np.array([g.decode() if isinstance(g, bytes) else str(g) for g in harm_z_ds.attrs["source_grid"]]),
        "harm_source_idx": harm_z_ds.attrs["source_frame_index"][:],
        "frame_global_means": metrics_meta[primary_metric]["frame_global_means"],
        "frame_global_stds": metrics_meta[primary_metric]["frame_global_stds"],
        "spatial_smoothing": metrics_meta[primary_metric]["spatial_smoothing"],
        "metrics": metrics_meta,
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
# STEP 2 & 3: SIMULATE SPECTRAL COMPLEXITY CUBES (GPU MAXVOL & 3D SUMMATION)
# ==============================================================================
def _compute_sliding_volumes_batch(eval_batch, target_metric, apply_spatial_smoothing):
    """Dispatches batched GPU sliding-window complexity for maxVol or 3D summation."""
    if target_metric == "volume_map_3d_zscore":
        return sc.process_volume_3d_sliding_tile_batch(
            eval_batch,
            tile_size=TILE_SIZE,
            stride=SLIDING_STRIDE,
            num_endmembers=NUM_ENDMEMBERS_3D,
            separability_threshold=SEPARABILITY_THRESHOLD_3D,
            apply_spatial_smoothing=apply_spatial_smoothing,
        )
    return sc.process_volume_sliding_tile_batch(
        eval_batch,
        tile_size=TILE_SIZE,
        stride=SLIDING_STRIDE,
        num_endmembers=NUM_ENDMEMBERS,
        apply_spatial_smoothing=apply_spatial_smoothing,
    )


def _extract_sensor_post_onset_sr(chips, grid_name, post_indices, source_idx, scale_factors, fill_values):
    """Extracts float32 surface reflectance with NaN fill masking for post-onset frames of a sensor."""
    if not post_indices or grid_name not in chips:
        return None
    local_indices = [source_idx[t] for t in post_indices]
    raw_arr = chips[grid_name][:, local_indices]
    scale = float(scale_factors.get(grid_name, 10000.0))
    fill_val = fill_values.get(grid_name, -32768)
    flt_arr = raw_arr.astype(np.float32) / scale
    flt_arr[raw_arr == fill_val] = np.nan
    return flt_arr


def _normalize_volume_batch_to_z(vol_batch, valid_mask_batch, means_vec, stds_vec):
    """Applies frame-specific log-normal global Z-score standardization to a (n_patches, n_frames, H, W) batch."""
    m_exp = means_vec[np.newaxis, :, np.newaxis, np.newaxis]
    s_exp = stds_vec[np.newaxis, :, np.newaxis, np.newaxis]
    valid = (vol_batch > 0.0) & valid_mask_batch & (s_exp > 0) & (~np.isnan(m_exp))
    with np.errstate(divide="ignore", invalid="ignore"):
        z_out = np.where(valid, (np.log(vol_batch) - m_exp) / s_exp, np.nan)
    return z_out.astype(np.float32)


def simulate_spectral_complexity_cubes(chips, metadata, target_spectra, onset_ts, target_metric="sliding_volume_z_score"):
    """
    Computes 10x10 sliding volume complexity on GPU across all 50 conditions for either:
      - 'sliding_volume_z_score': Maximum Volume (maxVol, 7 endmembers)
      - 'volume_map_3d_zscore':   3D Summation Equivalent Volume (3D, 20 endmembers, 1% stopping threshold)
    Normalizes via frame_global_means and frame_global_stds from the corresponding master HDF5 metric.
    """
    num_materials = len(target_spectra)
    num_fractions = len(MIXTURE_FRACTIONS)
    num_conditions = num_materials * num_fractions
    print(f"\n[Step 2 & 3] Simulating Spectral Complexity Z-scores on GPU across {num_conditions} conditions (metric: {target_metric})...")

    harm_acq_times = metadata["harm_acq_times"]
    source_grid = metadata["harm_source_grid"]
    source_idx = metadata["harm_source_idx"]

    metric_meta = metadata.get("metrics", {}).get(target_metric, {})
    means = metric_meta.get("frame_global_means", metadata["frame_global_means"])
    stds = metric_meta.get("frame_global_stds", metadata["frame_global_stds"])
    apply_spatial_smoothing = bool(metric_meta.get("spatial_smoothing", metadata.get("spatial_smoothing", False)))

    n_frames = len(harm_acq_times)
    n_patches = len(LAND_COVER_PATCHES)

    # 1. Identify pre-onset and post-onset frame indices
    pre_onset_mask = harm_acq_times < onset_ts
    post_onset_indices = np.where(~pre_onset_mask)[0]
    pre_onset_indices = np.where(pre_onset_mask)[0]

    print(f"Timeline partitioning: {len(pre_onset_indices)} pre-onset frames, {len(post_onset_indices)} post-onset frames (spatial_smoothing={apply_spatial_smoothing}).")

    # 2. Precompute unperturbed baseline Z-scores for all patches
    print(f"Precomputing unperturbed baseline Z-score time series for '{target_metric}'...")
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

        raw_batch = chips[grid_name][:, local_idx]  # (n_patches, bands, 10, 10)
        flt_batch = raw_batch.astype(np.float32) / float(scale)
        flt_batch[raw_batch == fill_val] = np.nan

        if grid_name == "TANAGER" and tan_gw is not None:
            gw_mask = tan_gw[local_idx].astype(bool)
            eval_batch = flt_batch[:, gw_mask, ...]
        else:
            eval_batch = flt_batch

        vol_batch = _compute_sliding_volumes_batch(eval_batch, target_metric, apply_spatial_smoothing)
        valid_batch = (chips["HARMONIZED_common_mask"][:, t] == 0)
        z_batch, _, _ = sc.calculate_global_z_score_batch(
            vol_batch,
            valid_pixel_mask=valid_batch,
            array_mean=mu_t,
            array_std=std_t,
        )
        if isinstance(z_batch, torch.Tensor):
            z_batch = z_batch.detach().cpu().numpy()
        unperturbed_z[:, t] = z_batch

    # 3. Simulate post-onset linear mixture across all conditions using batched GPU execution
    simulated_z_cubes = np.zeros((num_conditions, n_patches, n_frames, 10, 10), dtype=np.float32)
    for c_idx in range(num_conditions):
        simulated_z_cubes[c_idx] = unperturbed_z.copy()

    valid_stats = (~np.isnan(means)) & (~np.isnan(stds)) & (stds > 0)
    l8_post = [t for t in post_onset_indices if source_grid[t] == "HLSL30" and valid_stats[t]]
    s2_post = [t for t in post_onset_indices if source_grid[t] == "HLSS30" and valid_stats[t]]
    tan_post = [t for t in post_onset_indices if source_grid[t] == "TANAGER" and valid_stats[t]]

    flt_l8 = _extract_sensor_post_onset_sr(chips, "HLSL30", l8_post, source_idx, scale_factors, fill_values)
    flt_s2 = _extract_sensor_post_onset_sr(chips, "HLSS30", s2_post, source_idx, scale_factors, fill_values)
    flt_tan = _extract_sensor_post_onset_sr(chips, "TANAGER", tan_post, source_idx, scale_factors, fill_values)

    mask_l8 = (chips["HARMONIZED_common_mask"][:, l8_post] == 0) if l8_post else None
    mask_s2 = (chips["HARMONIZED_common_mask"][:, s2_post] == 0) if s2_post else None

    t_start_sim = time.perf_counter()
    cond_counter = 0

    for mat in target_spectra:
        spec_l8 = mat["spec_l8"]
        spec_s2 = mat["spec_s2"]
        spec_tanager = mat.get("spec_tanager")

        for f_weight in MIXTURE_FRACTIONS:
            # 1. HLSL30 Batched simulation
            if flt_l8 is not None and spec_l8 is not None:
                inj_l8 = flt_l8.copy()
                for roff, coff in INJ_OFFSETS:
                    bg = flt_l8[:, :, :, roff, coff]
                    valid_bg = ~np.isnan(bg).any(axis=-1, keepdims=True)
                    inj_l8[:, :, :, roff, coff] = np.where(valid_bg, (1.0 - f_weight) * bg + f_weight * spec_l8, bg)

                n_p, n_l8, n_b, h, w = inj_l8.shape
                vols_l8 = _compute_sliding_volumes_batch(
                    inj_l8.reshape(n_p * n_l8, n_b, h, w), target_metric, apply_spatial_smoothing
                ).reshape(n_p, n_l8, h, w)
                simulated_z_cubes[cond_counter][:, l8_post] = _normalize_volume_batch_to_z(
                    vols_l8, mask_l8, means[l8_post], stds[l8_post]
                )

            # 2. HLSS30 Batched simulation
            if flt_s2 is not None and spec_s2 is not None:
                inj_s2 = flt_s2.copy()
                for roff, coff in INJ_OFFSETS:
                    bg = flt_s2[:, :, :, roff, coff]
                    valid_bg = ~np.isnan(bg).any(axis=-1, keepdims=True)
                    inj_s2[:, :, :, roff, coff] = np.where(valid_bg, (1.0 - f_weight) * bg + f_weight * spec_s2, bg)

                n_p, n_s2, n_b, h, w = inj_s2.shape
                vols_s2 = _compute_sliding_volumes_batch(
                    inj_s2.reshape(n_p * n_s2, n_b, h, w), target_metric, apply_spatial_smoothing
                ).reshape(n_p, n_s2, h, w)
                simulated_z_cubes[cond_counter][:, s2_post] = _normalize_volume_batch_to_z(
                    vols_s2, mask_s2, means[s2_post], stds[s2_post]
                )

            # 3. TANAGER Batched simulation (per-frame good wavelength mask across all patches)
            if flt_tan is not None and spec_tanager is not None:
                for pos_i, t in enumerate(tan_post):
                    local_idx = source_idx[t]
                    flt_frame = flt_tan[:, pos_i].copy()
                    for roff, coff in INJ_OFFSETS:
                        bg = flt_frame[:, :, roff, coff]
                        valid_bg = ~np.isnan(bg).any(axis=-1, keepdims=True)
                        flt_frame[:, :, roff, coff] = np.where(valid_bg, (1.0 - f_weight) * bg + f_weight * spec_tanager, bg)

                    if tan_gw is not None:
                        gw_mask = tan_gw[local_idx].astype(bool)
                        eval_tan = flt_frame[:, gw_mask, ...]
                    else:
                        eval_tan = flt_frame

                    vols_tan = _compute_sliding_volumes_batch(eval_tan, target_metric, apply_spatial_smoothing)
                    valid_tan = (chips["HARMONIZED_common_mask"][:, t] == 0)
                    z_tan, _, _ = sc.calculate_global_z_score_batch(
                        vols_tan,
                        valid_pixel_mask=valid_tan,
                        array_mean=means[t],
                        array_std=stds[t],
                    )
                    if isinstance(z_tan, torch.Tensor):
                        z_tan = z_tan.detach().cpu().numpy()
                    simulated_z_cubes[cond_counter, :, t] = z_tan

            cond_counter += 1

    t_elapsed_sim = time.perf_counter() - t_start_sim
    print(f"Completed GPU simulation of {num_conditions} conditions for '{target_metric}' in {t_elapsed_sim:.2f} s ({t_elapsed_sim/num_conditions:.3f} s/condition).")

    return simulated_z_cubes, unperturbed_z


# ==============================================================================
# STEP 4: RUN HARMONIZED SC-CCD BREAK DETECTION
# ==============================================================================
def run_scccd_benchmark(sim_z_cubes, chips, metadata, target_spectra, onset_ts, return_spatial=False):
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
    num_conditions = num_materials * num_fractions
    total_records = num_conditions * n_patches * len(INJ_OFFSETS)

    results_table = np.zeros(total_records, dtype=BREAK_RECORD_DTYPE)
    scccd_change_dates = np.full((num_conditions, n_patches, 10, 10), np.nan, dtype=np.float64) if return_spatial else None
    scccd_change_counts = np.zeros((num_conditions, n_patches, 10, 10), dtype=np.int32) if return_spatial else None

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
                valid_patch = valid_mask_all[p_idx] & (~np.isnan(y_data_patch))

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

                if return_spatial:
                    scccd_change_dates[cond_idx, p_idx] = chunk_date
                    scccd_change_counts[cond_idx, p_idx] = chunk_count

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
    if return_spatial:
        return results_table, {"change_date": scccd_change_dates, "change_count": scccd_change_counts}
    return results_table


# ==============================================================================
# STEP 5: RUN LANDSAT PYCCD BREAK DETECTION
# ==============================================================================
def _eval_pyccd_pixel(ordinal_dates, blues, greens, reds, nirs, swir1s, swir2s, nominal_thermals, qas_pixel, proc_params):
    """
    Worker function to execute PyCCD detection on a single pixel time series in a parallel subprocess.
    """
    pccd.patch_pyccd()
    usable_count = int(np.count_nonzero((qas_pixel == 1) | (qas_pixel == 2) | (qas_pixel == 4)))
    if usable_count < pccd.MIN_CLEAR_SAMPLES:
        return []
    try:
        res = ccd.detect(
            ordinal_dates,
            blues, greens, reds, nirs, swir1s, swir2s,
            nominal_thermals,
            qas_pixel,
            params=proc_params,
        )
        return res.get("change_models", [])
    except Exception:
        return []


def run_pyccd_benchmark(chips, metadata, target_spectra, onset_ts, baseline_pyccd=None, return_spatial=False):
    """
    Executes USGS LCMAP Landsat PyCCD break detection across all stratified target conditions
    with full Fmask, HLSL30 common_mask, and 6-band fill-value decoding matching pyccd_landsat_main.py.
    """
    if baseline_pyccd is None and isinstance(chips, dict):
        baseline_pyccd = chips.get("BASELINE_PYCCD")

    l30_acq_times = metadata["l30_acq_times"]
    l30_dates = [datetime.datetime.fromtimestamp(t, tz=datetime.timezone.utc).date() for t in l30_acq_times]
    ordinal_dates = np.array([d.toordinal() for d in l30_dates])
    onset_ordinal = datetime.datetime.fromtimestamp(onset_ts, tz=datetime.timezone.utc).date().toordinal()

    n_l30_frames = len(l30_acq_times)
    nominal_thermals = np.full(n_l30_frames, pccd.NOMINAL_THERMAL_KELVIN_SCALED, dtype=np.int16)

    num_materials = len(target_spectra)
    num_fractions = len(MIXTURE_FRACTIONS)
    n_patches = len(LAND_COVER_PATCHES)
    num_conditions = num_materials * num_fractions
    total_records = num_conditions * n_patches * len(INJ_OFFSETS)
    print(f"\n[Step 5] Running USGS LCMAP Landsat PyCCD across {total_records} test instances...")

    results_table = np.zeros(total_records, dtype=BREAK_RECORD_DTYPE)
    pyccd_change_dates = np.full((num_conditions, n_patches, 10, 10), np.nan, dtype=np.float64) if return_spatial else None
    pyccd_change_counts = np.zeros((num_conditions, n_patches, 10, 10), dtype=np.int32) if return_spatial else None

    if return_spatial and baseline_pyccd is not None:
        for p_idx in range(n_patches):
            pyccd_change_dates[:, p_idx] = baseline_pyccd["date"][p_idx]
            pyccd_change_counts[:, p_idx] = baseline_pyccd["count"][p_idx]

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
    has_l30_cm = "HLSL30_common_mask" in chips

    pyccd_tasks = []
    pyccd_meta = []
    cond_idx = 0

    for m_idx, mat in enumerate(target_spectra):
        m_id = mat["id"].encode("utf-8")
        m_name = mat["name"].encode("utf-8")
        spec_l8_flt = mat["spec_l8"]

        for f_idx, f_weight in enumerate(MIXTURE_FRACTIONS):
            for p_idx, p in enumerate(LAND_COVER_PATCHES):
                p_id = p["id"].encode("utf-8")
                p_name = p["name"].encode("utf-8")
                r0, c0 = p["r0"], p["c0"]

                l30_patch = chips["HLSL30"][p_idx]          # shape: (n_frames, 7, 10, 10)
                fmask_patch = chips["HLSL30_Fmask"][p_idx]  # shape: (n_frames, 10, 10)
                cmask_patch = chips["HLSL30_common_mask"][p_idx] if has_l30_cm else np.zeros_like(fmask_patch, dtype=bool)

                for p_num, (roff, coff) in enumerate(INJ_OFFSETS, 1):
                    scene_r = r0 + roff
                    scene_c = c0 + coff

                    # Extract raw 6-band time series (B, G, R, NIR, SWIR1, SWIR2)
                    blues = l30_patch[:, 1, roff, coff].copy()
                    greens = l30_patch[:, 2, roff, coff].copy()
                    reds = l30_patch[:, 3, roff, coff].copy()
                    nirs = l30_patch[:, 4, roff, coff].copy()
                    swir1s = l30_patch[:, 5, roff, coff].copy()
                    swir2s = l30_patch[:, 6, roff, coff].copy()

                    # Inject target mixture only where all 6 optical bands are valid (not fill)
                    for t in post_onset_l30:
                        obs_pixel = np.array([blues[t], greens[t], reds[t], nirs[t], swir1s[t], swir2s[t]])
                        if not np.any(obs_pixel == l30_fill):
                            blues[t] = int(round((1.0 - f_weight) * blues[t] + f_weight * spec_l8_flt[1] * l30_scale))
                            greens[t] = int(round((1.0 - f_weight) * greens[t] + f_weight * spec_l8_flt[2] * l30_scale))
                            reds[t] = int(round((1.0 - f_weight) * reds[t] + f_weight * spec_l8_flt[3] * l30_scale))
                            nirs[t] = int(round((1.0 - f_weight) * nirs[t] + f_weight * spec_l8_flt[4] * l30_scale))
                            swir1s[t] = int(round((1.0 - f_weight) * swir1s[t] + f_weight * spec_l8_flt[5] * l30_scale))
                            swir2s[t] = int(round((1.0 - f_weight) * swir2s[t] + f_weight * spec_l8_flt[6] * l30_scale))

                    fm_pixel = fmask_patch[:, roff, coff]
                    cm_pixel = np.asarray(cmask_patch[:, roff, coff], dtype=bool)
                    sr_pixel_bands = np.column_stack([blues, greens, reds, nirs, swir1s, swir2s])
                    qas_pixel = pccd.decode_categorical_qa(fm_pixel, cm_pixel, sr_pixel_bands, fill_val=l30_fill)

                    pyccd_tasks.append((
                        ordinal_dates, blues, greens, reds, nirs, swir1s, swir2s,
                        nominal_thermals, qas_pixel, proc_params
                    ))
                    pyccd_meta.append((
                        cond_idx, p_idx, m_id, m_name, f_weight,
                        p_id, p_name, p_num, roff, coff, scene_r, scene_c
                    ))

            cond_idx += 1

    print(f"Parallel dispatching {len(pyccd_tasks)} PyCCD pixel evaluations across CPU cores (batch_size=64)...")
    results_gen = Parallel(n_jobs=-1, batch_size=64, return_as="generator")(
        delayed(_eval_pyccd_pixel)(*task) for task in pyccd_tasks
    )

    for rec_idx, (meta, models) in enumerate(zip(pyccd_meta, tqdm(results_gen, total=len(pyccd_tasks), desc="PyCCD pixels"))):
        cond_idx, p_idx, m_id, m_name, f_weight, p_id, p_name, p_num, roff, coff, scene_r, scene_c = meta

        breaks = [m for m in models if m.get("change_probability", 0) > 0]
        b_count = len(breaks)
        post_onset_breaks = []
        for m in breaks:
            b_day = m.get("break_day")
            if b_day is not None and b_day >= onset_ordinal:
                b_date_obj = datetime.date.fromordinal(b_day)
                b_ts = datetime.datetime(
                    b_date_obj.year, b_date_obj.month, b_date_obj.day,
                    tzinfo=datetime.timezone.utc
                ).timestamp()
                lat = float((b_ts - onset_ts) / 86400.0)
                mags = m.get("magnitudes", [0.0])
                mag_val = float(np.linalg.norm(mags))
                post_onset_breaks.append((b_ts, lat, mag_val))

        if post_onset_breaks:
            pri_b_ts, pri_lat, pri_mag = post_onset_breaks[0]
            if baseline_pyccd is not None:
                base_count = int(baseline_pyccd["count"][p_idx, roff, coff])
                base_date = float(baseline_pyccd["date"][p_idx, roff, coff])
                detected = (
                    (base_count == 0) or
                    np.isnan(base_date) or
                    (b_count > base_count) or
                    (pri_b_ts < base_date)
                )
            else:
                detected = True
            break_ts = pri_b_ts if detected else np.nan
            latency = pri_lat if detected else np.nan
            mag = pri_mag if detected else 0.0
        else:
            detected = False
            break_ts = np.nan
            latency = np.nan
            mag = 0.0

        if return_spatial:
            pyccd_change_dates[cond_idx, p_idx, roff, coff] = break_ts if detected else np.nan
            pyccd_change_counts[cond_idx, p_idx, roff, coff] = b_count

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

    detected_count = np.sum(results_table["break_detected"])
    print(f"PyCCD complete: {detected_count}/{total_records} breaks detected ({detected_count/total_records*100:.1f}%).")
    if return_spatial:
        return results_table, {"change_date": pyccd_change_dates, "change_count": pyccd_change_counts}
    return results_table


# ==============================================================================
# STEP 6: COMMIT BENCHMARK DATASET TO MASTER HDF5 FILE
# ==============================================================================
def save_benchmark_to_hdf5(
    h5_file,
    chips,
    metadata,
    target_spectra,
    sim_z_cubes,
    scccd_table,
    pyccd_table,
    target_metric="sliding_volume_z_score",
    group_name=None,
    scccd_spatial=None,
    description=None,
):
    """
    Writes the complete stratified chip benchmark hierarchy to the master HDF5 container
    for the specified spectral complexity method.
    """
    if group_name is None:
        group_name = "/STRATIFIED_CHIP_BENCHMARK_3D" if target_metric == "volume_map_3d_zscore" else "/STRATIFIED_CHIP_BENCHMARK"

    print(f"\n[Step 6] Saving '{group_name}' group to master HDF5 file (metric: {target_metric})...")

    if group_name in h5_file:
        print(f"Overwriting existing '{group_name}' group...")
        del h5_file[group_name]

    base_grp = h5_file.create_group(group_name)
    base_grp.attrs["target_metric"] = target_metric

    metric_meta = metadata.get("metrics", {}).get(target_metric, {})
    frame_means = metric_meta.get("frame_global_means", metadata["frame_global_means"])
    frame_stds = metric_meta.get("frame_global_stds", metadata["frame_global_stds"])

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
    meta_grp.attrs["frame_global_means"] = frame_means
    meta_grp.attrs["frame_global_stds"] = frame_stds
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
    for mask_key in ("HLSL30_Fmask", "HLSS30_Fmask", "HLSL30_common_mask", "HLSS30_common_mask"):
        if mask_key in chips:
            base_chips_grp.create_dataset(mask_key, data=chips[mask_key], compression="gzip", compression_opts=6)

    # Quantize baseline Z-score to int16 (scale = 10000)
    base_z_arr = chips.get("HARMONIZED_z_scores", {}).get(target_metric, chips["HARMONIZED_z_score"])
    base_z_int16 = sc.scale_to_int16(base_z_arr)
    base_chips_grp.create_dataset("HARMONIZED_z_score", data=base_z_int16, compression="gzip", compression_opts=6)
    base_chips_grp["HARMONIZED_z_score"].attrs["scale_factor"] = 10000.0
    base_chips_grp["HARMONIZED_z_score"].attrs["_FillValue"] = -32768

    # 3. SIMULATED_CUBES Subgroup (Quantized to int16 with gzip level 6)
    sim_grp = base_grp.create_group("SIMULATED_CUBES")
    sim_z_int16 = sc.scale_to_int16(sim_z_cubes)
    sim_ds = sim_grp.create_dataset(
        target_metric,
        data=sim_z_int16,
        dtype=np.int16,
        compression="gzip",
        compression_opts=6,
        chunks=(1, 1, min(sim_z_cubes.shape[2], 128), 10, 10),
    )
    sim_ds.attrs["scale_factor"] = 10000.0
    sim_ds.attrs["_FillValue"] = -32768
    if description is None:
        description = (
            "Simulated 3D Equivalent Spectral Complexity Z-scores under 50 perturbation conditions"
            if target_metric == "volume_map_3d_zscore"
            else "Simulated Gramian Spectral Complexity Z-scores under 50 perturbation conditions"
        )
    sim_ds.attrs["description"] = description

    # 4. RESULTS Subgroup
    res_grp = base_grp.create_group("RESULTS")
    res_grp.create_dataset("scccd_break_table", data=scccd_table, compression="gzip")
    res_grp.create_dataset("pyccd_break_table", data=pyccd_table, compression="gzip")
    if scccd_spatial is not None:
        for k, arr in scccd_spatial.items():
            res_grp.create_dataset(
                f"scccd_{k}",
                data=arr,
                compression="gzip",
                compression_opts=1,
                chunks=(1, 1, 10, 10),
            )

    print(f"Successfully committed {group_name} to {getattr(h5_file, 'filename', MASTER_H5_PATH)}.")


# ==============================================================================
# STEP 7: PRINT QUANTITATIVE PERFORMANCE SUMMARY
# ==============================================================================
def print_performance_summary(scccd_table, pyccd_table, method_label="Maximum Volume (maxVol)"):
    """
    Prints comparative detection accuracy and latency tables stratified by land-cover class.
    """
    print("\n" + "=" * 90)
    print(f"LAND-COVER STRATIFIED BENCHMARK PERFORMANCE SUMMARY: SC-CCD [{method_label}] vs. USGS PYCCD")
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
    n_total = len(scccd_table)
    total_sc_rec = np.mean(scccd_table["break_detected"]) * 100.0
    total_py_rec = np.mean(pyccd_table["break_detected"]) * 100.0
    all_sc_lat = scccd_table["latency_days"][scccd_table["break_detected"]]
    all_py_lat = pyccd_table["latency_days"][pyccd_table["break_detected"]]
    all_sc_med = f"{np.median(all_sc_lat):.1f} d" if len(all_sc_lat) > 0 else "N/A"
    all_py_med = f"{np.median(all_py_lat):.1f} d" if len(all_py_lat) > 0 else "N/A"
    n_mats = len(mat_ids)

    print("-" * 105)
    print(f"{'OVERALL AGGREGATE':<18} | {f'All {len(LAND_COVER_PATCHES)} Strata & {n_mats} Materials (N={n_total})':<34} | "
          f"{total_sc_rec:5.1f}% ({np.sum(scccd_table['break_detected']):4d}/{n_total}) | "
          f"{total_py_rec:5.1f}% ({np.sum(pyccd_table['break_detected']):4d}/{n_total}) | "
          f"{all_sc_med:<12} / {all_py_med}")
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

        # Step 1: Extract baseline chips and metadata for both maxVol and 3D summation metrics
        target_metrics = [m["key"] for m in BENCHMARK_METRICS]
        chips, metadata = extract_baseline_chips(h5_file, target_metrics=target_metrics)

        # Step 5: Run USGS LCMAP Landsat PyCCD break detection once (shared reference across SC methods)
        pyccd_table = run_pyccd_benchmark(chips, metadata, target_spectra, onset_ts)

        # Steps 2-4 & 6: Simulate and evaluate both maxVol and 3D summation spectral complexity methods
        method_results = []
        for method_cfg in BENCHMARK_METRICS:
            metric_key = method_cfg["key"]
            print("\n" + "-" * 80)
            print(f"BENCHMARK METHOD: {method_cfg['label']} ({metric_key})")
            print("-" * 80)

            sim_z_cubes, unperturbed_z = simulate_spectral_complexity_cubes(
                chips, metadata, target_spectra, onset_ts, target_metric=metric_key
            )
            chips["HARMONIZED_z_scores"][metric_key] = unperturbed_z
            if metric_key == BENCHMARK_METRICS[0]["key"]:
                chips["HARMONIZED_z_score"] = unperturbed_z

            scccd_table, scccd_spatial = run_scccd_benchmark(
                sim_z_cubes, chips, metadata, target_spectra, onset_ts, return_spatial=True
            )

            save_benchmark_to_hdf5(
                h5_file,
                chips,
                metadata,
                target_spectra,
                sim_z_cubes,
                scccd_table,
                pyccd_table,
                target_metric=metric_key,
                group_name=method_cfg["group_name"],
                scccd_spatial=scccd_spatial,
                description=method_cfg["description"],
            )
            method_results.append((method_cfg["label"], scccd_table))

    # Step 7: Print quantitative summary for each method
    for method_label, scccd_table in method_results:
        print_performance_summary(scccd_table, pyccd_table, method_label=method_label)

    t_elapsed_total = (time.perf_counter() - t_start_total) / 60.0
    print(f"Stratified benchmark pipeline completed successfully in {t_elapsed_total:.2f} minutes.")


if __name__ == "__main__":
    main()

