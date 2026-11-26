"""
generate_stratified_patch_benchmark.py
======================================
Executes the Land-Cover Stratified Patches semi-empirical target injection experiment
and saves all chips, simulated complexity cubes, and algorithmic break detection results
directly into the master HDF5 container under /STRATIFIED_CHIP_BENCHMARK/.

Reuses core functions and standards from:
  - SpecComplexTorch.py: Sliding window MaxD Gramian volume unmixing on GPU.
  - SC_CCM_main.py: SC-CCM detection engine.
  - pyccd_landsat_main.py: USGS LCMAP Landsat PyCCD detection engine and QA decoding.
  - USGS Splib07 multi-sensor resampled spectral library (Landsat-8, Sentinel-2, Tanager-1).

Experimental Structure:
  - 4 Land-Cover Strata (13x13 pixels, 390 m x 390 m):
      1. AGRI: Rush Open Farmland (r0=1266, c0=225)
      2. FOREST: Mendon Ponds Park Forest (r0=1144, c0=528)
      3. WETLAND: Thousand Acre Swamp, Penfield (r0=608, c0=858)
      4. SYNTHETIC: Rochester Airport (r0=803, c0=269)
  - 4 Decoupled Injected Pixels per patch (stride = 4, 3 baseline pixels separation):
      Offsets: [(4, 4), (4, 8), (8, 4), (8, 8)]
      Inner evaluated region: rows 2..10, cols 2..10 (9x9 pixels)
      Outer 2-pixel margin: rows 0..1, 11..12, cols 0..1, 11..12 (un-evaluated by SC-CCD)
  - 10 USGS Splib07 Target Materials x 5 Mixture Fractions [0.05, 0.15, 0.25, 0.35, 0.45]
      = 50 Distinct Perturbation Conditions
  - 50 conditions x 4 strata x 4 pixels = 800 test instances per algorithm.
"""


import sys
from pathlib import Path
_script_dir = Path(__file__).resolve().parent
_repo_root = _script_dir.parent
for _p in [
    _repo_root / "change_detection" / "pyccd",
    _repo_root / "change_detection" / "SC-CCM",
    _repo_root,
    _script_dir,
]:
    p_str = str(_p)
    while p_str in sys.path:
        sys.path.remove(p_str)
    sys.path.insert(0, p_str)

import os
import sys
import gc
import math
import time
import datetime
import argparse
from pathlib import Path
import h5py
import numpy as np
import scipy.interpolate
from joblib import Parallel, delayed
from tqdm import tqdm

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
import SC_CCM_main as scccm
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

# Pipeline and patch parameters
PATCH_SIZE = 13
BORDER_MARGIN = 2
TILE_SIZE = 3
SLIDING_STRIDE = 1
NUM_ENDMEMBERS = 7
NUM_ENDMEMBERS_3D = 20
DEFAULT_TARGET_METRIC = "volume_map_3d_zscore"

# Land-Cover Stratified Patches (13x13 chips with 100% clean baseline in inner 9x9 region)
LAND_COVER_PATCHES = [
    {
        "id": "CROPLAND",
        "class": "Cropland",
        "name": "Rush Open Farmland",
        "r0": 1266,
        "c0": 225,
        "color": "#e6ab02",  # Gold / Amber
        "descr": "Cycling crop production land.",
    },
    {
        "id": "TREE COVER",
        "class": "Forest",
        "name": "Mendon Ponds Park Forest",
        "r0": 1144,
        "c0": 528,
        "color": "#2ca02c",  # Forest Green
        "descr": "Deciduous forest.",
    },
    {
        "id": "WETLAND",
        "class": "Wetland",
        "name": "Thousand Acre Swamp (Penfield)",
        "r0": 608,
        "c0": 858,
        "color": "#17becf",  # Teal / Cyan
        "descr": "Saturated soil and standing water.",
    },
    {
        "id": "DEVELOPED",
        "class": "Developed Land",
        "name": "Rochester Airport",
        "r0": 803,
        "c0": 269,
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
        "id": "painted_aluminum",
        "name": "Painted Aluminum (GDS333)",
        "contrast": "Manmade",
        "role": "Synthetic Materials",
        "asd_rel": r"ChapterA_ArtificialMaterials\s07_ASD_Painted_Aluminum_GDS333_LgGr_ASDFRa_AREF.txt",
        "l8_rel":  r"ChapterA_ArtificialMaterials\S07LSAT8_Painted_Aluminum_GDS333_LgGr_ASDFRa_AREF.txt",
        "s2_rel":  r"ChapterA_ArtificialMaterials\S07SNTL2_Painted_Aluminum_GDS333_LgGr_ASDFRa_AREF.txt",
    },
    {
        "id": "galvanized steel",
        "name": "Galvanized Sheet Metal (GDS334)",
        "contrast": "Manmade",
        "role": "Synthetic Materials",
        "asd_rel": r"ChapterA_ArtificialMaterials\s07_ASD_GalvanizedSheetMetal_GDS334_ASDFRa_AREF.txt",
        "l8_rel":  r"ChapterA_ArtificialMaterials\S07LSAT8_GalvanizedSheetMetal_GDS334_ASDFRa_AREF.txt",
        "s2_rel":  r"ChapterA_ArtificialMaterials\S07SNTL2_GalvanizedSheetMetal_GDS334_ASDFRa_AREF.txt",
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
        "id": "green_tarp",
        "name": "Green Plastic Tarp (GDS339)",
        "contrast": "Manmade",
        "role": "Construction / Debris",
        "asd_rel": r"ChapterA_ArtificialMaterials\s07_ASD_Plastic_Tarp_GDS339_Green_ASDFRa_AREF.txt",
        "l8_rel":  r"ChapterA_ArtificialMaterials\S07LSAT8_Plastic_Tarp_GDS339_Green_ASDFRa_AREF.txt",
        "s2_rel":  r"ChapterA_ArtificialMaterials\S07SNTL2_Plastic_Tarp_GDS339_Green_ASDFRa_AREF.txt",
    },
    {
        "id": "white_plastic",
        "name": "White Vinyl Plastic (GDS372)",
        "contrast": "Manmade",
        "role": "Synthetic Materials",
        "asd_rel": r"ChapterA_ArtificialMaterials\s07_ASD_Plastic_Vinyl_GDS372_White_ASDFRa_AREF.txt",
        "l8_rel":  r"ChapterA_ArtificialMaterials\S07LSAT8_Plastic_Vinyl_GDS372_White_ASDFRa_AREF.txt",
        "s2_rel":  r"ChapterA_ArtificialMaterials\S07SNTL2_Plastic_Vinyl_GDS372_White_ASDFRa_AREF.txt",
    },
    {
        "id": "Nylon_Fabric",
        "name": "Black Nylon Fabric (GDS436)",
        "contrast": "Manmade",
        "role": "Synthetic Materials",
        "asd_rel": r"ChapterA_ArtificialMaterials\s07_ASD_Nylon_Fabric_GDS436_BlkCoatd_ASDFRa_AREF.txt",
        "l8_rel":  r"ChapterA_ArtificialMaterials\S07LSAT8_Nylon_Fabric_GDS436_BlkCoatd_ASDFRa_AREF.txt",
        "s2_rel":  r"ChapterA_ArtificialMaterials\S07SNTL2_Nylon_Fabric_GDS436_BlkCoatd_ASDFRa_AREF.txt",
    },
    {
        "id": "dry_grass",
        "name": "Dry Golden Grass (GDS480)",
        "contrast": "senescent vegetation",
        "role": "Vegetation",
        "asd_rel": r"ChapterV_Vegetation\s07_ASD_Grass_Golden_Dry_GDS480_ASDFRa_AREF.txt",
        "l8_rel":  r"ChapterV_Vegetation\S07LSAT8_Grass_Golden_Dry_GDS480_ASDFRa_AREF.txt",
        "s2_rel":  r"ChapterV_Vegetation\S07SNTL2_Grass_Golden_Dry_GDS480_ASDFRa_AREF.txt",
    },
    {
        "id": "dry_willow",
        "name": "Dried Willow Leaves (Willow-Leaves-1)",
        "contrast": "senescent vegetation",
        "role": "Vegetation", 
        "asd_rel": r"ChapterV_Vegetation\s07_ASD_Willow_Willow-Leaves-1_dry_ASDFRa_AREF.txt",
        "l8_rel":  r"ChapterV_Vegetation\S07LSAT8_Willow_Willow-Leaves-1_dry_ASDFRa_AREF.txt",
        "s2_rel":  r"ChapterV_Vegetation\S07SNTL2_Willow_Willow-Leaves-1_dry_ASDFRa_AREF.txt",
    },
    {
        "id": "healthy_maple",
        "name": "Healthy Maple Leaves (DW92-1)",
        "contrast": "healthy vegetation",
        "role": "Vegetation", 
        "asd_rel": r"ChapterV_Vegetation\s07_ASD_Maple_Leaves_DW92-1_BECKa_AREF.txt",
        "l8_rel":  r"ChapterV_Vegetation\S07LSAT8_Maple_Leaves_DW92-1_BECKa_AREF.txt",
        "s2_rel":  r"ChapterV_Vegetation\S07SNTL2_Maple_Leaves_DW92-1_BECKa_AREF.txt",
    },
    {
        "id": "Pinon_Pine",
        "name": "Pinon Pine (ANP92-14A)",
        "contrast": "healthy vegetation",
        "role": "Vegetation", 
        "asd_rel": r"ChapterV_Vegetation\s07_ASD_Pinon_Pine_ANP92-14A_needle_BECKa_AREF.txt",
        "l8_rel":  r"ChapterV_Vegetation\S07LSAT8_Pinon_Pine_ANP92-14A_needle_BECKa_AREF.txt",
        "s2_rel":  r"ChapterV_Vegetation\S07SNTL2_Pinon_Pine_ANP92-14A_needle_BECKa_AREF.txt",
    },
    {
        "id": "burn_area",
        "name": "Burn Area Traverse (WRF00-01)",
        "contrast": "Burn / Pyrogenic",
        "role": "Natural Disturbance",
        "asd_rel": r"ChapterS_SoilsAndMixtures\s07_ASD_BurnArea_Traverse_WRF00-01_ASDFRb_AREF.txt",
        "l8_rel":  r"ChapterS_SoilsAndMixtures\S07LSAT8_BurnArea_Traverse_WRF00-01_ASDFRb_AREF.txt",
        "s2_rel":  r"ChapterS_SoilsAndMixtures\S07SNTL2_BurnArea_Traverse_WRF00-01_ASDFRb_AREF.txt",
    },
    {
        "id": "burn_area2",
        "name": "Burn Area Surface (WRF00-02)",
        "contrast": "Burn / Pyrogenic",
        "role": "Natural Disturbance",
        "asd_rel": r"ChapterS_SoilsAndMixtures\s07_ASD_BurnArea_TopSurface_WRF00-02_ASDFRb_AREF.txt",
        "l8_rel":  r"ChapterS_SoilsAndMixtures\S07LSAT8_BurnArea_TopSurface_WRF00-02_ASDFRb_AREF.txt",
        "s2_rel":  r"ChapterS_SoilsAndMixtures\S07SNTL2_BurnArea_TopSurface_WRF00-02_ASDFRb_AREF.txt",
    },
    {
        "id": "oak_leaves",
        "name": "Oak Leaf (Oak-Leaf-1)",
        "contrast": "Healthy Vegetation",
        "role": "Vegetation",
        "asd_rel": r"ChapterV_Vegetation\s07_ASD_Oak_Oak-Leaf-1_fresh_ASDFRa_AREF.txt",
        "l8_rel":  r"ChapterV_Vegetation\S07LSAT8_Oak_Oak-Leaf-1_fresh_ASDFRa_AREF.txt",
        "s2_rel":  r"ChapterV_Vegetation\S07SNTL2_Oak_Oak-Leaf-1_fresh_ASDFRa_AREF.txt",
    },
    {
        "id": "oak_shrub",
        "name": "Oak Shrub (CA01-QUDU-2)",
        "contrast": "Canopy Foliage",
        "role": "Vegetation",
        "asd_rel": r"ChapterV_Vegetation\s07_ASD_Oak_QUDU_CA01-QUDU-2_bush_2_ASDFRa_AREF.txt",
        "l8_rel":  r"ChapterV_Vegetation\S07LSAT8_Oak_QUDU_CA01-QUDU-2_bush_2_ASDFRa_AREF.txt",
        "s2_rel":  r"ChapterV_Vegetation\S07SNTL2_Oak_QUDU_CA01-QUDU-2_bush_2_ASDFRa_AREF.txt",
    },
    {
        "id": "aspen_YellowGreenTop",
        "name": "Yellow-Green Aspen Leaf (Aspen-3)",
        "contrast": "Canopy Foliage",
        "role": "Vegetation",
        "asd_rel": r"ChapterV_Vegetation\s07_ASD_Aspen_Aspen-3_yellowGreenTop_ASDFRa_AREF.txt",
        "l8_rel":  r"ChapterV_Vegetation\S07LSAT8_Aspen_Aspen-3_yellowGreenTop_ASDFRa_AREF.txt",
        "s2_rel":  r"ChapterV_Vegetation\S07SNTL2_Aspen_Aspen-3_yellowGreenTop_ASDFRa_AREF.txt",
    },
    {
        "id": "lawn_grass",
        "name": "Lawn Grass  (GDS91)",
        "contrast": "Healthy Vegetation",
        "role": "Vegetation",
        "asd_rel": r"ChapterV_Vegetation\s07_ASD_Lawn_Grass_GDS91_green_BECKa_AREF.txt",
        "l8_rel":  r"ChapterV_Vegetation\S07LSAT8_Lawn_Grass_GDS91_green_BECKa_AREF.txt",
        "s2_rel":  r"ChapterV_Vegetation\S07SNTL2_Lawn_Grass_GDS91_green_BECKa_AREF.txt",
    },
    {
        "id": "turbid_water",
        "name": "Water + 0.5g/L Montmorillonite (SWy-2+0.5g/l)",
        "contrast": "Inundation",
        "role": "Flood / Inundation",
        "asd_rel": r"ChapterL_Liquids\s07_ASD_Water+Montmor_SWy-2+0.50g-l_ASDFRa_AREF.txt",
        "l8_rel":  r"ChapterL_Liquids\S07LSAT8_Water+Montmor_SWy-2+0.50g-l_ASDFRa_AREF.txt",
        "s2_rel":  r"ChapterL_Liquids\S07SNTL2_Water+Montmor_SWy-2+0.50g-l_ASDFRa_AREF.txt",
    },
    {
        "id": "turbid_water",
        "name": "Water + 1.67g/L Montmorillonite (SWy-2+1.67g/l)",
        "contrast": "Inundation",
        "role": "Flood / Inundation",
        "asd_rel": r"ChapterL_Liquids\s07_ASD_Water+Montmor_SWy-2+1.67g-l_ASDFRa_AREF.txt",
        "l8_rel":  r"ChapterL_Liquids\S07LSAT8_Water+Montmor_SWy-2+1.67g-l_ASDFRa_AREF.txt",
        "s2_rel":  r"ChapterL_Liquids\S07SNTL2_Water+Montmor_SWy-2+1.67g-l_ASDFRa_AREF.txt",
    },
    {
        "id": "turbid_water",
        "name": "Water + 5.01g/L Montmorillonite (SWy-2+5.01g/l)",
        "contrast": "Inundation",
        "role": "Flood / Inundation",
        "asd_rel": r"ChapterL_Liquids\s07_ASD_Water+Montmor_SWy-2+5.01g-l_ASDFRa_AREF.txt",
        "l8_rel":  r"ChapterL_Liquids\S07LSAT8_Water+Montmor_SWy-2+5.01g-l_ASDFRa_AREF.txt",
        "s2_rel":  r"ChapterL_Liquids\S07SNTL2_Water+Montmor_SWy-2+5.01g-l_ASDFRa_AREF.txt",
    },
    {
        "id": "turbid_water",
        "name": "Water + 16.5g/L Montmorillonite (SWy-2+16.5g/l)",
        "contrast": "Inundation",
        "role": "Flood / Inundation",
        "asd_rel": r"ChapterL_Liquids\s07_ASD_Water+Montmor_SWy-2+16.5g-l_ASDFRa_AREF.txt",
        "l8_rel":  r"ChapterL_Liquids\S07LSAT8_Water+Montmor_SWy-2+16.5g-l_ASDFRa_AREF.txt",
        "s2_rel":  r"ChapterL_Liquids\S07SNTL2_Water+Montmor_SWy-2+16.5g-l_ASDFRa_AREF.txt",
    },
]

# Sentinel-2 band indices matching HLSS30 (10 bands):
S2_SUB_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 11, 12]

# Injected pixel local offsets inside each 13x13 patch (stride = 4, 3 baseline pixels separation)
# Inner 9x9 evaluated region is slice 2:11; outer 2-pixel margin is slice :2 and 11:
INJ_OFFSETS = [
    (4, 4),  # P1
    (4, 8),  # P2
    (8, 4),  # P3
    (8, 8),  # P4
]

# Fractional abundance levels
MIXTURE_FRACTIONS = np.arange(0.00,1.05, step=0.05)

# Compound data type for tabular instance results (1 record per test instance)
BREAK_RECORD_DTYPE = np.dtype([
    ("condition_idx", np.int16),        # 0 to num_conditions - 1
    ("material_id", "S32"),             # Target material identifier string
    ("material_name", "S48"),           # Material descriptive name
    ("mixture_fraction", np.float32),   # 0.00 to 1.00
    ("patch_id", "S16"),                # 'CROPLAND', 'TREE COVER', 'WETLAND', 'DEVELOPED'
    ("patch_name", "S48"),              # Full stratum description
    ("pixel_num", np.int8),             # 1 to 4
    ("scene_row", np.int16),            # Absolute row coordinate in master scene
    ("scene_col", np.int16),            # Absolute col coordinate in master scene
    ("break_detected", np.bool_),       # True if break detected post-onset
    ("break_timestamp", np.float64),    # Acquisition timestamp of detected post-onset break
    ("latency_days", np.float32),       # Days between onset date and detected break
    ("break_magnitude", np.float32),    # Magnitude of break or residual shift
    ("p_value", np.float32),            # Chi-square or model significance
    ("total_breaks", np.int16),         # Total number of breaks across entire time series
    ("post_onset_breaks", np.int16),    # Number of breaks occurring post-onset
    ("has_multiple_post_breaks", np.bool_), # True if post_onset_breaks > 1
    ("confirmation_timestamp", np.float64), # Timestamp of the 6th anomaly (confirmation)
    ("confirmation_latency_days", np.float32), # Days from onset to confirmation
])

# Compound data type for all temporal curve segments and breaks across the time series
SEGMENT_RECORD_DTYPE = np.dtype([
    ("condition_idx", np.int16),
    ("material_id", "S32"),
    ("material_name", "S48"),
    ("mixture_fraction", np.float32),
    ("patch_id", "S16"),
    ("patch_name", "S48"),
    ("pixel_num", np.int8),
    ("scene_row", np.int16),
    ("scene_col", np.int16),
    ("segment_num", np.int8),                  # 1, 2, ...
    ("start_timestamp", np.float64),           # Timestamp of segment start observation
    ("end_timestamp", np.float64),             # Timestamp of segment end observation
    ("has_break", np.bool_),                   # True if segment terminated by break, False if active until end
    ("break_num", np.int8),                    # 1, 2, ... (or 0 if no break)
    ("break_timestamp", np.float64),           # Timestamp of first anomaly in run
    ("confirmation_timestamp", np.float64),    # Timestamp of confirmation observation
    ("is_post_onset", np.bool_),               # True if break occurred on or after injection onset
    ("latency_days", np.float32),              # Days from onset date to break_timestamp (NaN if pre-onset)
    ("confirmation_latency_days", np.float32), # Days from onset date to confirmation (NaN if pre-onset)
    ("break_magnitude", np.float32),           # Residual jump magnitude
    ("rmse", np.float32),                      # Segment model RMSE
    ("num_observations", np.int16),            # Number of observations in segment
])


# ==============================================================================
# STEP 1: EXTRACTION OF BASELINE CHIPS & METADATA
# ==============================================================================
def get_dataset_metadata(ds):
    """
    Extracts fill value and scale factor from HDF5 dataset attributes without assumed defaults.
    Fails explicitly if mandatory attributes are missing.
    """
    # 1. Fill value extraction
    fill_val = None
    if hasattr(ds, "attrs"):
        if "_FillValue" in ds.attrs:
            fill_val = ds.attrs["_FillValue"]
        elif "fillvalue" in ds.attrs:
            fill_val = ds.attrs["fillvalue"]
        elif "fill_value" in ds.attrs:
            fill_val = ds.attrs["fill_value"]

    if fill_val is None and hasattr(ds, "fillvalue") and ds.fillvalue is not None:
        if np.issubdtype(ds.dtype, np.floating) or ds.fillvalue != 0:
            fill_val = ds.fillvalue

    if fill_val is None:
        if np.issubdtype(ds.dtype, np.integer):
            raise KeyError(f"Dataset '{getattr(ds, 'name', 'dataset')}' has integer datatype ({ds.dtype}) but lacks mandatory '_FillValue' attribute.")
        elif np.issubdtype(ds.dtype, np.floating):
            fill_val = np.nan

    if fill_val is not None:
        if np.issubdtype(ds.dtype, np.integer):
            fill_val = int(fill_val)
        elif np.issubdtype(ds.dtype, np.floating):
            fill_val = float(fill_val)

    # 2. Scale factor extraction
    if np.issubdtype(ds.dtype, np.floating):
        # Native floating-point reflectance is already unscaled [0, 1]
        scale = float(ds.attrs["scale_factor"]) if (hasattr(ds, "attrs") and "scale_factor" in ds.attrs) else 1.0
    else:
        if hasattr(ds, "attrs") and "scale_factor" in ds.attrs:
            scale = float(ds.attrs["scale_factor"])
        elif hasattr(ds, "attrs") and "scale_to_float" in ds.attrs:
            sf = float(ds.attrs["scale_to_float"])
            scale = (1.0 / sf) if sf < 1.0 else sf
        else:
            raise KeyError(f"Dataset '{getattr(ds, 'name', 'dataset')}' has integer datatype ({ds.dtype}) lacks mandatory 'scale_factor' or 'scale_to_float' attribute.")

    return fill_val, scale


def extract_baseline_chips(h5_file, target_metric=None):
    """
    Extracts the unperturbed 9x9 time series for each of the 4 patches across sensors.
    Dynamically identifies scale_factor and _FillValue attributes from each dataset without assumed defaults.
    """
    harm_field_path = "/HDFEOS/GRIDS/HARMONIZED/Data Fields"
    if harm_field_path in h5_file:
        harm_grp = h5_file[harm_field_path]
        if target_metric is None:
            if "volume_map_3d_zscore" in harm_grp:
                target_metric = "volume_map_3d_zscore"
            elif "sliding_volume_z_score" in harm_grp:
                target_metric = "sliding_volume_z_score"
            else:
                target_metric = DEFAULT_TARGET_METRIC
    else:
        if target_metric is None:
            target_metric = DEFAULT_TARGET_METRIC

    print(f"\n[Step 1] Extracting unperturbed {PATCH_SIZE}x{PATCH_SIZE} baseline chips from master scene (metric: {target_metric})...")

    # 1. Verify presence of dataset produced by HLST_SC_calculations.py
    if harm_field_path not in h5_file or target_metric not in h5_file[harm_field_path]:
        raise RuntimeError(
            f"Harmonized Spectral Complexity dataset '{harm_field_path}/{target_metric}' "
            f"(or 'sliding_volume_z_score') "
            f"not found in master container '{getattr(h5_file, 'filename', MASTER_H5_PATH)}'. "
            f"Please run HLST_SC_calculations.py first to compute spectral complexity metrics."
        )

    # 2. Verify presence of baseline SC-CCD dataset produced by harmonized_CCD_main.py
    scccd_grp_name = "SCCCD_Results"
    if scccd_grp_name not in h5_file or "change_date_timestamp" not in h5_file[scccd_grp_name]:
        raise RuntimeError(
            f"Baseline SC-CCD dataset '/{scccd_grp_name}' not found in master container "
            f"'{getattr(h5_file, 'filename', MASTER_H5_PATH)}'. "
            f"Please run harmonized_CCD_main.py first to generate full-frame baseline SC-CCD change detection results."
        )

    # 3. Verify presence of baseline PyCCD dataset produced by pyccd_landsat_main.py
    pyccd_grp_name = "PYCCD_Results"
    if pyccd_grp_name not in h5_file or "change_date_timestamp" not in h5_file[pyccd_grp_name]:
        raise RuntimeError(
            f"Baseline PyCCD dataset '/{pyccd_grp_name}' not found in master container "
            f"'{getattr(h5_file, 'filename', MASTER_H5_PATH)}'. "
            f"Please run pyccd_landsat_main.py first to generate full-frame PyCCD change detection results."
        )

    harm_grp = h5_file[harm_field_path]
    l30_grp = h5_file["/HDFEOS/GRIDS/HLSL30/Data Fields"]
    s30_grp = h5_file["/HDFEOS/GRIDS/HLSS30/Data Fields"]
    tan_grp = h5_file.get("/HDFEOS/GRIDS/TANAGER/Data Fields")

    harm_z_ds = harm_grp[target_metric]
    harm_cm_ds = harm_grp["common_mask"]
    l30_sr_ds = l30_grp["surface_reflectance"]
    l30_fm_ds = l30_grp["Fmask"]
    l30_cm_ds = l30_grp["common_mask"]
    s30_sr_ds = s30_grp["surface_reflectance"]
    s30_fm_ds = s30_grp["Fmask"]
    s30_cm_ds = s30_grp["common_mask"]

    n_harm_frames = harm_z_ds.shape[0]
    n_l30_frames = l30_sr_ds.shape[0]
    n_s30_frames = s30_sr_ds.shape[0]
    n_patches = len(LAND_COVER_PATCHES)

    chips = {
        "HLSL30": np.zeros((n_patches, n_l30_frames, 7, PATCH_SIZE, PATCH_SIZE), dtype=l30_sr_ds.dtype),
        "HLSS30": np.zeros((n_patches, n_s30_frames, 10, PATCH_SIZE, PATCH_SIZE), dtype=s30_sr_ds.dtype),
        "HLSL30_Fmask": np.zeros((n_patches, n_l30_frames, PATCH_SIZE, PATCH_SIZE), dtype=l30_fm_ds.dtype),
        "HLSS30_Fmask": np.zeros((n_patches, n_s30_frames, PATCH_SIZE, PATCH_SIZE), dtype=s30_fm_ds.dtype),
        "HLSL30_common_mask": np.zeros((n_patches, n_l30_frames, PATCH_SIZE, PATCH_SIZE), dtype=l30_cm_ds.dtype),
        "HLSS30_common_mask": np.zeros((n_patches, n_s30_frames, PATCH_SIZE, PATCH_SIZE), dtype=s30_cm_ds.dtype),
        "HARMONIZED_z_score": np.zeros((n_patches, n_harm_frames, PATCH_SIZE, PATCH_SIZE), dtype=harm_z_ds.dtype),
        "HARMONIZED_common_mask": np.zeros((n_patches, n_harm_frames, PATCH_SIZE, PATCH_SIZE), dtype=harm_cm_ds.dtype),
    }

    l30_fill, l30_scale = get_dataset_metadata(l30_sr_ds)
    s30_fill, s30_scale = get_dataset_metadata(s30_sr_ds)

    scale_factors = {
        "HLSL30": l30_scale,
        "HLSS30": s30_scale,
    }
    fill_values = {
        "HLSL30": l30_fill,
        "HLSS30": s30_fill,
    }

    mask_fill_values = {
        "HLSL30_Fmask": l30_fm_ds.attrs.get("_FillValue", getattr(l30_fm_ds, "fillvalue", 255)),
        "HLSL30_common_mask": l30_cm_ds.attrs.get("_FillValue", getattr(l30_cm_ds, "fillvalue", 255)),
        "HLSS30_Fmask": s30_fm_ds.attrs.get("_FillValue", getattr(s30_fm_ds, "fillvalue", 255)),
        "HLSS30_common_mask": s30_cm_ds.attrs.get("_FillValue", getattr(s30_cm_ds, "fillvalue", 255)),
        "HARMONIZED_common_mask": harm_cm_ds.attrs.get("_FillValue", getattr(harm_cm_ds, "fillvalue", 255)),
    }

    tan_sr_ds = None
    if tan_grp is not None and "surface_reflectance" in tan_grp:
        tan_sr_ds = tan_grp["surface_reflectance"]
        n_tan_frames = tan_sr_ds.shape[0]
        n_tan_bands = tan_sr_ds.shape[1]
        chips["TANAGER"] = np.zeros((n_patches, n_tan_frames, n_tan_bands, PATCH_SIZE, PATCH_SIZE), dtype=tan_sr_ds.dtype)
        tan_fill, tan_scale = get_dataset_metadata(tan_sr_ds)
        scale_factors["TANAGER"] = tan_scale
        fill_values["TANAGER"] = tan_fill

    for p_idx, p in enumerate(LAND_COVER_PATCHES):
        r0, c0 = p["r0"], p["c0"]
        chips["HLSL30"][p_idx] = l30_sr_ds[:, :, r0:r0+PATCH_SIZE, c0:c0+PATCH_SIZE]
        chips["HLSS30"][p_idx] = s30_sr_ds[:, :, r0:r0+PATCH_SIZE, c0:c0+PATCH_SIZE]
        chips["HLSL30_Fmask"][p_idx] = l30_fm_ds[:, r0:r0+PATCH_SIZE, c0:c0+PATCH_SIZE]
        chips["HLSS30_Fmask"][p_idx] = s30_fm_ds[:, r0:r0+PATCH_SIZE, c0:c0+PATCH_SIZE]
        chips["HLSL30_common_mask"][p_idx] = l30_cm_ds[:, r0:r0+PATCH_SIZE, c0:c0+PATCH_SIZE]
        chips["HLSS30_common_mask"][p_idx] = s30_cm_ds[:, r0:r0+PATCH_SIZE, c0:c0+PATCH_SIZE]
        chips["HARMONIZED_z_score"][p_idx] = harm_z_ds[:, r0:r0+PATCH_SIZE, c0:c0+PATCH_SIZE]
        chips["HARMONIZED_common_mask"][p_idx] = harm_cm_ds[:, r0:r0+PATCH_SIZE, c0:c0+PATCH_SIZE]
        if tan_sr_ds is not None:
            chips["TANAGER"][p_idx] = tan_sr_ds[:, :, r0:r0+PATCH_SIZE, c0:c0+PATCH_SIZE]

    # Extract baseline SC-CCD results directly from master container /SCCCD_Results (produced by harmonized_CCD_main.py)
    # The outer 2-pixel surrounding region is NOT evaluated by SC-CCD (initialized to NaN date, 0 count)
    baseline_scccd = {
        "date": np.full((n_patches, PATCH_SIZE, PATCH_SIZE), np.nan, dtype=np.float64),
        "count": np.zeros((n_patches, PATCH_SIZE, PATCH_SIZE), dtype=np.int32),
    }
    sc_date_ds = h5_file[f"{scccd_grp_name}/change_date_timestamp"]
    sc_count_ds = h5_file[f"{scccd_grp_name}/change_count"]
    for p_idx, p in enumerate(LAND_COVER_PATCHES):
        r0, c0 = p["r0"], p["c0"]
        baseline_scccd["date"][p_idx, 2:11, 2:11] = sc_date_ds[r0+2:r0+11, c0+2:c0+11]
        baseline_scccd["count"][p_idx, 2:11, 2:11] = sc_count_ds[r0+2:r0+11, c0+2:c0+11]
    chips["BASELINE_SCCCD"] = baseline_scccd

    # Extract baseline PyCCD results directly from master container /PYCCD_Results (produced by pyccd_landsat_main.py)
    # The outer 2-pixel surrounding region is NOT evaluated (initialized to NaN date, 0 count)
    baseline_pyccd = {
        "date": np.full((n_patches, PATCH_SIZE, PATCH_SIZE), np.nan, dtype=np.float64),
        "count": np.zeros((n_patches, PATCH_SIZE, PATCH_SIZE), dtype=np.int32),
    }
    py_date_ds = h5_file[f"{pyccd_grp_name}/change_date_timestamp"]
    py_count_ds = h5_file[f"{pyccd_grp_name}/change_count"]
    for p_idx, p in enumerate(LAND_COVER_PATCHES):
        r0, c0 = p["r0"], p["c0"]
        baseline_pyccd["date"][p_idx, 2:11, 2:11] = py_date_ds[r0+2:r0+11, c0+2:c0+11]
        baseline_pyccd["count"][p_idx, 2:11, 2:11] = py_count_ds[r0+2:r0+11, c0+2:c0+11]
    chips["BASELINE_PYCCD"] = baseline_pyccd

    metadata = {
        "target_metric": target_metric,
        "harm_acq_times": harm_z_ds.attrs["acquisition_time"][:],
        "harm_source_grid": np.array([g.decode() if isinstance(g, bytes) else str(g) for g in harm_z_ds.attrs["source_grid"]]),
        "harm_source_idx": harm_z_ds.attrs["source_frame_index"][:],
        "frame_global_means": harm_z_ds.attrs["frame_global_means"][:],
        "frame_global_stds": harm_z_ds.attrs["frame_global_stds"][:],
        "l30_acq_times": l30_sr_ds.attrs["acquisition_time"][:],
        "s30_acq_times": s30_sr_ds.attrs["acquisition_time"][:],
        "scale_factors": scale_factors,
        "fill_values": fill_values,
        "mask_fill_values": mask_fill_values,
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
# STEP 2 & 3: SIMULATE SPECTRAL COMPLEXITY CUBES (BATCHED SLIDING VOLUME)
# ==============================================================================
def simulate_spectral_complexity_cubes(chips, metadata, target_spectra, onset_ts, target_metric=DEFAULT_TARGET_METRIC):
    """
    Computes 9x9 sliding volume Gramian complexity across all perturbation conditions.
    Utilizes high-throughput batched sliding-window volume unmixing via SpecComplexTorch (GPU)
    or SpecComplex (CPU), vectorizing linear mixture simulations across all 4 patches and post-onset frames.
    """
    num_materials = len(target_spectra)
    num_fractions = len(MIXTURE_FRACTIONS)
    num_conditions = num_materials * num_fractions

    print(f"\n[Step 2 & 3] Simulating Spectral Complexity Z-scores across {num_conditions} conditions (metric: {target_metric})...")

    harm_acq_times = metadata["harm_acq_times"]
    source_grid = metadata["harm_source_grid"]
    source_idx = metadata["harm_source_idx"]
    means = metadata["frame_global_means"]
    stds = metadata["frame_global_stds"]

    n_frames = len(harm_acq_times)
    n_patches = len(LAND_COVER_PATCHES)

    # 1. Identify pre-onset and post-onset frame indices
    pre_onset_mask = harm_acq_times < onset_ts
    post_onset_indices = np.where(~pre_onset_mask)[0]
    pre_onset_indices = np.where(pre_onset_mask)[0]

    print(f"Timeline partitioning: {len(pre_onset_indices)} pre-onset frames, {len(post_onset_indices)} post-onset frames.")

    # 2. Extract or precompute unperturbed baseline Z-scores for all patches
    if "HARMONIZED_z_score" in chips and np.count_nonzero(~np.isnan(chips["HARMONIZED_z_score"])) > 0:
        print("Using precomputed baseline Z-score time series from master container chips...")
        unperturbed_z = chips["HARMONIZED_z_score"].copy()
    else:
        print("Precomputing unperturbed baseline Z-score time series for all patches...")
        unperturbed_z = np.full((n_patches, n_frames, PATCH_SIZE, PATCH_SIZE), np.nan, dtype=np.float32)

        scale_factors = metadata["scale_factors"]
        fill_values = metadata["fill_values"]
        tan_gw = metadata.get("tanager_good_wavelengths")

        for t in range(n_frames):
            mu_t = means[t]
            std_t = stds[t]
            if np.isnan(mu_t) or np.isnan(std_t) or std_t <= 0:
                continue

            grid_name = source_grid[t]
            local_idx = source_idx[t]
            if grid_name not in chips:
                continue

            scale = scale_factors[grid_name]
            fill_val = fill_values[grid_name]

            raw_patches = chips[grid_name][:, local_idx]  # (n_patches, bands, H, W)
            flt_sr = raw_patches.astype(np.float32) / float(scale)
            if np.issubdtype(raw_patches.dtype, np.integer):
                flt_sr[raw_patches == fill_val] = np.nan
            else:
                if fill_val is not None and not np.isnan(fill_val):
                    flt_sr[np.isclose(raw_patches, fill_val, equal_nan=True)] = np.nan

            if grid_name == "TANAGER" and tan_gw is not None:
                gw_mask = tan_gw[local_idx].astype(bool)
                eval_sr = flt_sr[:, gw_mask, ...]
            else:
                eval_sr = flt_sr

            if target_metric == "volume_map_3d_zscore":
                vol_maps = sc.process_volume_3d_sliding_tile_batch(
                    eval_sr, tile_size=TILE_SIZE, stride=SLIDING_STRIDE, num_endmembers=NUM_ENDMEMBERS_3D, separability_threshold=0.01
                )
            else:
                vol_maps = sc.process_volume_sliding_tile_batch(
                    eval_sr, tile_size=TILE_SIZE, stride=SLIDING_STRIDE, num_endmembers=NUM_ENDMEMBERS
                )
            valid_masks = (chips["HARMONIZED_common_mask"][:, t] == 0)
            z_maps, _, _ = sc.calculate_global_z_score_batch(
                vol_maps,
                valid_pixel_mask=valid_masks,
                array_mean=mu_t,
                array_std=std_t,
            )
            unperturbed_z[:, t] = z_maps

    # 3. Simulate post-onset linear mixture across all conditions using batched tensors
    simulated_z_cubes = np.zeros((num_conditions, n_patches, n_frames, PATCH_SIZE, PATCH_SIZE), dtype=np.float32)

    # Initialize all conditions with unperturbed baseline Z-scores
    for c_idx in range(num_conditions):
        for p_idx in range(n_patches):
            simulated_z_cubes[c_idx, p_idx] = unperturbed_z[p_idx].copy()

    # Partition valid post-onset frames by sensor grid
    scale_factors = metadata["scale_factors"]
    fill_values = metadata["fill_values"]
    tan_gw = metadata.get("tanager_good_wavelengths")

    l8_post = [t for t in post_onset_indices if source_grid[t] == "HLSL30" and not (np.isnan(means[t]) or np.isnan(stds[t]) or stds[t] <= 0)]
    s2_post = [t for t in post_onset_indices if source_grid[t] == "HLSS30" and not (np.isnan(means[t]) or np.isnan(stds[t]) or stds[t] <= 0)]
    tan_post = [t for t in post_onset_indices if source_grid[t] == "TANAGER" and not (np.isnan(means[t]) or np.isnan(stds[t]) or stds[t] <= 0)]

    # Pre-extract normalized base surface reflectance for post-onset frames
    flt_l8 = None
    if len(l8_post) > 0 and "HLSL30" in chips:
        l8_local = [source_idx[t] for t in l8_post]
        raw_l8 = chips["HLSL30"][:, l8_local]  # (n_patches, n_l8, 7, H, W)
        scale_l8 = float(scale_factors["HLSL30"])
        fill_l8 = fill_values["HLSL30"]
        flt_l8 = raw_l8.astype(np.float32) / scale_l8
        if np.issubdtype(raw_l8.dtype, np.integer):
            flt_l8[raw_l8 == fill_l8] = np.nan
        elif fill_l8 is not None and not np.isnan(fill_l8):
            flt_l8[np.isclose(raw_l8, fill_l8, equal_nan=True)] = np.nan
        means_l8 = means[l8_post]
        stds_l8 = stds[l8_post]
        mask_l8 = (chips["HARMONIZED_common_mask"][:, l8_post] == 0)

    flt_s2 = None
    if len(s2_post) > 0 and "HLSS30" in chips:
        s2_local = [source_idx[t] for t in s2_post]
        raw_s2 = chips["HLSS30"][:, s2_local]  # (n_patches, n_s2, 10, H, W)
        scale_s2 = float(scale_factors["HLSS30"])
        fill_s2 = fill_values["HLSS30"]
        flt_s2 = raw_s2.astype(np.float32) / scale_s2
        if np.issubdtype(raw_s2.dtype, np.integer):
            flt_s2[raw_s2 == fill_s2] = np.nan
        elif fill_s2 is not None and not np.isnan(fill_s2):
            flt_s2[np.isclose(raw_s2, fill_s2, equal_nan=True)] = np.nan
        means_s2 = means[s2_post]
        stds_s2 = stds[s2_post]
        mask_s2 = (chips["HARMONIZED_common_mask"][:, s2_post] == 0)

    flt_tan = None
    if len(tan_post) > 0 and "TANAGER" in chips:
        tan_local = [source_idx[t] for t in tan_post]
        raw_tan = chips["TANAGER"][:, tan_local]
        scale_tan = float(scale_factors.get("TANAGER", 1.0))
        fill_tan = fill_values.get("TANAGER")
        flt_tan = raw_tan.astype(np.float32) / scale_tan
        if fill_tan is not None and not np.isnan(fill_tan):
            flt_tan[np.isclose(raw_tan, fill_tan, equal_nan=True)] = np.nan
        means_tan = means[tan_post]
        stds_tan = stds[tan_post]
        mask_tan = (chips["HARMONIZED_common_mask"][:, tan_post] == 0)

    t_start_sim = time.perf_counter()
    cond_counter = 0

    for m_idx, mat in enumerate(target_spectra):
        spec_l8 = mat["spec_l8"]
        spec_s2 = mat["spec_s2"]
        spec_tanager = mat.get("spec_tanager")

        for f_idx, f_weight in enumerate(MIXTURE_FRACTIONS):
            # 1. HLSL30 Batched simulation
            if flt_l8 is not None and spec_l8 is not None:
                inj_l8 = flt_l8.copy()
                for roff, coff in INJ_OFFSETS:
                    bg = flt_l8[:, :, :, roff, coff]
                    inj_val = (1.0 - f_weight) * bg + f_weight * spec_l8
                    valid_bg = ~np.isnan(bg).any(axis=-1, keepdims=True)
                    inj_l8[:, :, :, roff, coff] = np.where(valid_bg, inj_val, bg)

                n_p, n_l8, n_b, H, W = inj_l8.shape
                b_l8 = inj_l8.reshape(n_p * n_l8, n_b, H, W)
                if target_metric == "volume_map_3d_zscore":
                    v_l8 = sc.process_volume_3d_sliding_tile_batch(
                        b_l8, tile_size=TILE_SIZE, stride=SLIDING_STRIDE, num_endmembers=NUM_ENDMEMBERS_3D, separability_threshold=0.01
                    )
                else:
                    v_l8 = sc.process_volume_sliding_tile_batch(
                        b_l8, tile_size=TILE_SIZE, stride=SLIDING_STRIDE, num_endmembers=NUM_ENDMEMBERS
                    )
                v_l8 = v_l8.reshape(n_p, n_l8, H, W)

                m_exp = means_l8[np.newaxis, :, np.newaxis, np.newaxis]
                s_exp = stds_l8[np.newaxis, :, np.newaxis, np.newaxis]
                pos = (v_l8 > 0.0)
                val = pos & mask_l8 & (s_exp > 0) & (~np.isnan(m_exp))
                with np.errstate(divide="ignore", invalid="ignore"):
                    z_l8 = np.where(val, (np.log(v_l8) - m_exp) / s_exp, np.nan)
                simulated_z_cubes[cond_counter][:, l8_post] = z_l8

            # 2. HLSS30 Batched simulation
            if flt_s2 is not None and spec_s2 is not None:
                inj_s2 = flt_s2.copy()
                for roff, coff in INJ_OFFSETS:
                    bg = flt_s2[:, :, :, roff, coff]
                    inj_val = (1.0 - f_weight) * bg + f_weight * spec_s2
                    valid_bg = ~np.isnan(bg).any(axis=-1, keepdims=True)
                    inj_s2[:, :, :, roff, coff] = np.where(valid_bg, inj_val, bg)

                n_p, n_s2, n_b, H, W = inj_s2.shape
                b_s2 = inj_s2.reshape(n_p * n_s2, n_b, H, W)
                if target_metric == "volume_map_3d_zscore":
                    v_s2 = sc.process_volume_3d_sliding_tile_batch(
                        b_s2, tile_size=TILE_SIZE, stride=SLIDING_STRIDE, num_endmembers=NUM_ENDMEMBERS_3D, separability_threshold=0.01
                    )
                else:
                    v_s2 = sc.process_volume_sliding_tile_batch(
                        b_s2, tile_size=TILE_SIZE, stride=SLIDING_STRIDE, num_endmembers=NUM_ENDMEMBERS
                    )
                v_s2 = v_s2.reshape(n_p, n_s2, H, W)

                m_exp = means_s2[np.newaxis, :, np.newaxis, np.newaxis]
                s_exp = stds_s2[np.newaxis, :, np.newaxis, np.newaxis]
                pos = (v_s2 > 0.0)
                val = pos & mask_s2 & (s_exp > 0) & (~np.isnan(m_exp))
                with np.errstate(divide="ignore", invalid="ignore"):
                    z_s2 = np.where(val, (np.log(v_s2) - m_exp) / s_exp, np.nan)
                simulated_z_cubes[cond_counter][:, s2_post] = z_s2

            # 3. TANAGER Batched simulation
            if flt_tan is not None and spec_tanager is not None:
                inj_tan = flt_tan.copy()
                for roff, coff in INJ_OFFSETS:
                    bg = flt_tan[:, :, :, roff, coff]
                    inj_val = (1.0 - f_weight) * bg + f_weight * spec_tanager
                    valid_bg = ~np.isnan(bg).any(axis=-1, keepdims=True)
                    inj_tan[:, :, :, roff, coff] = np.where(valid_bg, inj_val, bg)

                n_p, n_tan, n_b, H, W = inj_tan.shape
                b_tan = inj_tan.reshape(n_p * n_tan, n_b, H, W)
                if target_metric == "volume_map_3d_zscore":
                    v_tan = sc.process_volume_3d_sliding_tile_batch(
                        b_tan, tile_size=TILE_SIZE, stride=SLIDING_STRIDE, num_endmembers=NUM_ENDMEMBERS_3D, separability_threshold=0.01
                    )
                else:
                    v_tan = sc.process_volume_sliding_tile_batch(
                        b_tan, tile_size=TILE_SIZE, stride=SLIDING_STRIDE, num_endmembers=NUM_ENDMEMBERS
                    )
                v_tan = v_tan.reshape(n_p, n_tan, H, W)

                m_exp = means_tan[np.newaxis, :, np.newaxis, np.newaxis]
                s_exp = stds_tan[np.newaxis, :, np.newaxis, np.newaxis]
                pos = (v_tan > 0.0)
                val = pos & mask_tan & (s_exp > 0) & (~np.isnan(m_exp))
                with np.errstate(divide="ignore", invalid="ignore"):
                    z_tan = np.where(val, (np.log(v_tan) - m_exp) / s_exp, np.nan)
                simulated_z_cubes[cond_counter][:, tan_post] = z_tan

            cond_counter += 1

    t_elapsed_sim = time.perf_counter() - t_start_sim
    print(f"Completed simulation of {num_conditions} conditions in {t_elapsed_sim:.2f} s ({t_elapsed_sim/num_conditions:.3f} s/condition).")

    return simulated_z_cubes, unperturbed_z


# ==============================================================================
# STEP 4: RUN HARMONIZED SC-CCD BREAK DETECTION
# ==============================================================================
def run_scccd_benchmark(sim_z_cubes, chips, metadata, target_spectra, onset_ts, baseline_scccd=None, return_spatial=True):
    """
    Executes Harmonized SC-CCD break regression across all 800 test instances.
    Evaluates the inner 9x9 region of each 13x13 patch across time to accurately characterize
    the spatial impact on neighboring pixels when modifying the reflectance of implanted pixels.
    The outer 2-pixel surrounding region is NOT evaluated by SC-CCD.
    Evaluates detected breaks against the unperturbed baseline SC-CCD results produced by harmonized_CCD_main.py.
    """
    print("\n[Step 4] Running Harmonized SC-CCD break detection on inner 9x9 regions (outer 2px not evaluated)...")

    if baseline_scccd is None and isinstance(chips, dict):
        baseline_scccd = chips.get("BASELINE_SCCCD")

    if baseline_scccd is None:
        raise RuntimeError(
            f"Baseline SC-CCD dataset '/SCCCD_Results' was not found in chips or master container '{MASTER_H5_PATH}'. "
            f"Please run harmonized_CCD_main.py first before executing the stratified patch benchmark."
        )
    harm_acq_times = metadata["harm_acq_times"]
    frac_years = scccm.extract_fractional_years(harm_acq_times)
    X_full = scccm.build_harmonic_matrix(frac_years)

    common_mask = chips["HARMONIZED_common_mask"]
    valid_mask_all = (common_mask == 0)

    num_materials = len(target_spectra)
    num_fractions = len(MIXTURE_FRACTIONS)
    n_patches = len(LAND_COVER_PATCHES)
    num_conditions = num_materials * num_fractions
    total_records = num_conditions * n_patches * len(INJ_OFFSETS)

    results_table = np.zeros(total_records, dtype=BREAK_RECORD_DTYPE)
    scccd_change_dates = np.full((num_conditions, n_patches, PATCH_SIZE, PATCH_SIZE), np.nan, dtype=np.float64)
    scccd_change_counts = np.zeros((num_conditions, n_patches, PATCH_SIZE, PATCH_SIZE), dtype=np.int32)
    scccd_segment_list = []

    # Prepare chunks for parallel execution
    chunk_list = []
    chunk_meta = []
    cond_idx = 0

    for m_idx, mat in enumerate(target_spectra):
        m_id = mat["id"].encode("utf-8")
        m_name = mat["name"].encode("utf-8")

        for f_idx, f_weight in enumerate(MIXTURE_FRACTIONS):
            for p_idx, p in enumerate(LAND_COVER_PATCHES):
                p_id = p["id"].encode("utf-8")
                p_name = p["name"].encode("utf-8")
                r0, c0 = p["r0"], p["c0"]

                # Run CCD row chunk ONLY across the inner 9x9 region (rows 2:11, cols 2:11)
                # The outer 2-pixel surrounding region is NOT evaluated by SC-CCD
                y_data_patch = sim_z_cubes[cond_idx, p_idx]
                valid_patch = valid_mask_all[p_idx]

                y_inner = y_data_patch[:, 2:11, 2:11]
                valid_inner = valid_patch[:, 2:11, 2:11]

                chunk_args = (
                    0, 9, 9,
                    y_inner, valid_inner,
                    frac_years, X_full, harm_acq_times,
                    scccm.MIN_SAMPLES, scccm.CHANGE_PROBABILITY, scccm.CONSECUTIVE_ANOMALIES
                )
                chunk_list.append(chunk_args)
                chunk_meta.append((cond_idx, p_idx, m_id, m_name, f_weight, p_id, p_name, r0, c0, y_inner, valid_inner))

            cond_idx += 1

    print(f"Parallel dispatching {len(chunk_list)} SC-CCD 9x9 chunks across all CPU cores...")
    results_gen = Parallel(n_jobs=-1, return_as='generator')(
        delayed(scccm._process_row_chunk)(args) for args in chunk_list
    )

    rec_idx = 0
    for meta, res in zip(chunk_meta, tqdm(results_gen, total=len(chunk_list), desc="SC-CCD chunks")):
        (cond_idx, p_idx, m_id, m_name, f_weight, p_id, p_name, r0, c0, y_inner, valid_inner) = meta
        chunk_pred = res[2]
        chunk_date = res[6]
        chunk_count = res[7]
        chunk_segments = res[8] if len(res) > 8 else {}

        # Record spatial results for inner 9x9 evaluated region; store post-onset break date if present
        for r_in in range(9):
            for c_in in range(9):
                p_segs = chunk_segments.get((r_in, c_in), [])
                p_cnt = len([s for s in p_segs if s.get("has_break", False)])
                scccd_change_counts[cond_idx, p_idx, 2 + r_in, 2 + c_in] = p_cnt

                eff_ts_p = np.nan
                for s in p_segs:
                    if s.get("has_break", False):
                        b_ts_cand = s.get("break_date", np.nan)
                        c_ts_cand = s.get("confirm_date", np.nan)
                        if b_ts_cand >= onset_ts:
                            eff_ts_p = b_ts_cand
                            break
                        elif c_ts_cand >= onset_ts:
                            run_idx = [k for k in range(s["break_idx"], s["confirm_idx"] + 1)
                                       if valid_inner[k, r_in, c_in] and harm_acq_times[k] >= onset_ts]
                            eff_ts_p = float(harm_acq_times[run_idx[0]]) if run_idx else c_ts_cand
                            break
                scccd_change_dates[cond_idx, p_idx, 2 + r_in, 2 + c_in] = eff_ts_p

        for p_num, (roff, coff) in enumerate(INJ_OFFSETS, 1):
            scene_r = r0 + roff
            scene_c = c0 + coff

            inner_r = roff - 2
            inner_c = coff - 2
            b_count = int(chunk_count[inner_r, inner_c])
            pixel_segs = chunk_segments.get((inner_r, inner_c), [])

            # Parse all segments and identify post-onset breaks
            post_onset_breaks = []
            for s in pixel_segs:
                has_b = s.get("has_break", False)
                b_ts = s.get("break_date", np.nan)
                c_ts = s.get("confirm_date", np.nan)
                is_post = False
                eff_b_ts = np.nan

                if has_b:
                    if b_ts >= onset_ts:
                        is_post = True
                        eff_b_ts = b_ts
                    elif c_ts >= onset_ts:
                        run_idx = [k for k in range(s["break_idx"], s["confirm_idx"] + 1)
                                   if valid_inner[k, inner_r, inner_c] and harm_acq_times[k] >= onset_ts]
                        eff_b_ts = float(harm_acq_times[run_idx[0]]) if run_idx else c_ts
                        is_post = True

                    if is_post:
                        post_onset_breaks.append({
                            "segment": s,
                            "break_timestamp": eff_b_ts,
                            "confirmation_timestamp": c_ts,
                            "raw_break_timestamp": b_ts,
                        })

                # Segment residual magnitude
                s_start = s["start_idx"]
                s_end = s["end_idx"]
                v_seg = np.where(valid_inner[s_start:s_end+1, inner_r, inner_c])[0] + s_start
                if len(v_seg) > 0:
                    act_s = y_inner[v_seg, inner_r, inner_c]
                    pred_s = chunk_pred[v_seg, inner_r, inner_c]
                    mag_seg = float(np.nanmean(np.abs(act_s - pred_s)))
                else:
                    mag_seg = 0.0

                seg_rec = np.zeros(1, dtype=SEGMENT_RECORD_DTYPE)[0]
                seg_rec["condition_idx"] = cond_idx
                seg_rec["material_id"] = m_id
                seg_rec["material_name"] = m_name
                seg_rec["mixture_fraction"] = f_weight
                seg_rec["patch_id"] = p_id
                seg_rec["patch_name"] = p_name
                seg_rec["pixel_num"] = p_num
                seg_rec["scene_row"] = scene_r
                seg_rec["scene_col"] = scene_c
                seg_rec["segment_num"] = s["segment_idx"]
                seg_rec["start_timestamp"] = s["start_date"]
                seg_rec["end_timestamp"] = s["end_date"]
                seg_rec["has_break"] = has_b
                seg_rec["break_num"] = s.get("break_num", 0)
                seg_rec["break_timestamp"] = eff_b_ts if is_post else b_ts
                seg_rec["confirmation_timestamp"] = c_ts
                seg_rec["is_post_onset"] = is_post
                seg_rec["latency_days"] = float((eff_b_ts - onset_ts) / 86400.0) if is_post else np.nan
                seg_rec["confirmation_latency_days"] = float((c_ts - onset_ts) / 86400.0) if is_post else np.nan
                seg_rec["break_magnitude"] = mag_seg
                seg_rec["rmse"] = float(s["rmse"])
                seg_rec["num_observations"] = int(s["n_points"])
                scccd_segment_list.append(seg_rec)

            # Compare against baseline SC-CCD evaluation (produced by harmonized_CCD_main.py)
            base_count = int(baseline_scccd["count"][p_idx, roff, coff])
            base_date = float(baseline_scccd["date"][p_idx, roff, coff])

            n_post = len(post_onset_breaks)
            has_mult = (n_post > 1)

            # Detection criteria: break occurred on or after onset date that was not already present in baseline
            if n_post > 0:
                primary_brk = post_onset_breaks[0]
                pri_b_ts = primary_brk["break_timestamp"]
                pri_c_ts = primary_brk["confirmation_timestamp"]
                is_new_break = (
                    (base_count == 0) or
                    np.isnan(base_date) or
                    (b_count > base_count) or
                    (pri_b_ts < base_date)
                )
                detected = is_new_break
                latency = float((pri_b_ts - onset_ts) / 86400.0) if detected else np.nan
                confirm_lat = float((pri_c_ts - onset_ts) / 86400.0) if detected else np.nan
            else:
                pri_b_ts = np.nan
                pri_c_ts = np.nan
                detected = False
                latency = np.nan
                confirm_lat = np.nan

            # Break magnitude: residual jump post-onset
            if detected:
                post_valid = np.where(valid_inner[:, inner_r, inner_c] & (harm_acq_times >= onset_ts))[0]
                if len(post_valid) > 0:
                    actuals = y_inner[post_valid, inner_r, inner_c]
                    preds = chunk_pred[post_valid, inner_r, inner_c]
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
            rec["break_timestamp"] = pri_b_ts if detected else np.nan
            rec["latency_days"] = latency
            rec["break_magnitude"] = mag
            rec["p_value"] = 0.99 if detected else 0.0
            rec["total_breaks"] = b_count
            rec["post_onset_breaks"] = n_post
            rec["has_multiple_post_breaks"] = has_mult
            rec["confirmation_timestamp"] = pri_c_ts if detected else np.nan
            rec["confirmation_latency_days"] = confirm_lat

            rec_idx += 1

    detected_count = np.sum(results_table["break_detected"])
    multiple_break_count = np.sum(results_table["has_multiple_post_breaks"])
    scccd_segments = np.array(scccd_segment_list, dtype=SEGMENT_RECORD_DTYPE)
    print(f"SC-CCD complete: {detected_count}/{total_records} breaks detected ({detected_count/total_records*100:.1f}%).")
    print(f"SC-CCD multiple post-implant breaks: {multiple_break_count}/{total_records} ({multiple_break_count/total_records*100:.1f}%).")
    print(f"SC-CCD total segment models recorded: {len(scccd_segments)} across {total_records} instances.")

    spatial_results = {
        "change_date": scccd_change_dates,
        "change_count": scccd_change_counts,
    }
    if return_spatial:
        return results_table, spatial_results, scccd_segments
    return results_table, scccd_segments


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
            params=proc_params
        )
        return res.get("change_models", [])
    except Exception:
        return []


def run_pyccd_benchmark(chips, metadata, target_spectra, onset_ts, baseline_pyccd=None, return_spatial=True):
    """
    Executes USGS LCMAP Landsat PyCCD break detection across all 800 test instances.
    Only evaluates the 4 implanted pixels per patch for each perturbation condition.
    Because PyCCD has no spatial context for neighboring pixels, the neighboring baseline
    pixels copy the baseline PyCCD evaluation (produced by pyccd_landsat_main.py)
    to maintain the full 13x13 patch output array size.
    """
    print("\n[Step 5] Running USGS LCMAP Landsat PyCCD on 4 implanted pixels per patch...")

    if baseline_pyccd is None and isinstance(chips, dict):
        baseline_pyccd = chips.get("BASELINE_PYCCD")

    if baseline_pyccd is None:
        raise RuntimeError(
            f"Baseline PyCCD dataset '/PYCCD_Results' was not found in chips or master container '{MASTER_H5_PATH}'. "
            f"Please run pyccd_landsat_main.py first before executing the stratified patch benchmark."
        )

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

    results_table = np.zeros(total_records, dtype=BREAK_RECORD_DTYPE)
    pyccd_change_dates = np.full((num_conditions, n_patches, PATCH_SIZE, PATCH_SIZE), np.nan, dtype=np.float64)
    pyccd_change_counts = np.zeros((num_conditions, n_patches, PATCH_SIZE, PATCH_SIZE), dtype=np.int32)
    pyccd_segment_list = []

    # Initialize all condition spatial grids with baseline PyCCD results
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

    l30_scale = metadata["scale_factors"]["HLSL30"]
    l30_fill = metadata["fill_values"]["HLSL30"]

    # Pre-build injected pixel time series for all evaluations
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

                l30_patch = chips["HLSL30"][p_idx]          # shape: (n_frames, 7, PATCH_SIZE, PATCH_SIZE)
                fmask_patch = chips["HLSL30_Fmask"][p_idx]  # shape: (n_frames, PATCH_SIZE, PATCH_SIZE)
                cmask_patch = chips["HLSL30_common_mask"][p_idx] # shape: (n_frames, PATCH_SIZE, PATCH_SIZE)

                # ONLY evaluate the 4 implanted pixels per patch
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
                        obs_pixel = np.array([blues[t], greens[t], reds[t], nirs[t], swir1s[t], swir2s[t]])
                        if not np.any(obs_pixel == l30_fill):
                            blues[t] = int(round((1.0 - f_weight) * blues[t] + f_weight * spec_l8_flt[1] * l30_scale))
                            greens[t] = int(round((1.0 - f_weight) * greens[t] + f_weight * spec_l8_flt[2] * l30_scale))
                            reds[t] = int(round((1.0 - f_weight) * reds[t] + f_weight * spec_l8_flt[3] * l30_scale))
                            nirs[t] = int(round((1.0 - f_weight) * nirs[t] + f_weight * spec_l8_flt[4] * l30_scale))
                            swir1s[t] = int(round((1.0 - f_weight) * swir1s[t] + f_weight * spec_l8_flt[5] * l30_scale))
                            swir2s[t] = int(round((1.0 - f_weight) * swir2s[t] + f_weight * spec_l8_flt[6] * l30_scale))

                    fm_pixel = fmask_patch[:, roff, coff]
                    cm_pixel = cmask_patch[:, roff, coff]
                    sr_pixel_bands = np.column_stack([blues, greens, reds, nirs, swir1s, swir2s])
                    qas_pixel = pccd.decode_categorical_qa(fm_pixel, cm_pixel, sr_pixel_bands, fill_val=l30_fill)

                    task_args = (
                        ordinal_dates, blues, greens, reds, nirs, swir1s, swir2s,
                        nominal_thermals, qas_pixel, proc_params
                    )
                    pyccd_tasks.append(task_args)
                    pyccd_meta.append((cond_idx, p_idx, m_id, m_name, f_weight, p_id, p_name, p_num, roff, coff, scene_r, scene_c))

            cond_idx += 1

    print(f"Parallel dispatching {len(pyccd_tasks)} PyCCD pixel evaluations across all CPU cores (batch_size=64)...")
    results_gen = Parallel(n_jobs=-1, batch_size=64, return_as='generator')(
        delayed(_eval_pyccd_pixel)(*task) for task in pyccd_tasks
    )

    rec_idx = 0
    for meta, models in zip(pyccd_meta, tqdm(results_gen, total=len(pyccd_tasks), desc="PyCCD pixels")):
        (cond_idx, p_idx, m_id, m_name, f_weight, p_id, p_name, p_num, roff, coff, scene_r, scene_c) = meta

        breaks = [m for m in models if m.get("change_probability", 0) > 0]
        b_count = len(breaks)
        post_onset_breaks = []

        for m_idx_seg, m in enumerate(models, 1):
            s_day = m.get("start_day")
            e_day = m.get("end_day")
            b_day = m.get("break_day")
            has_b = (b_day is not None and b_day > 0 and m.get("change_probability", 0) > 0)

            s_dt = datetime.date.fromordinal(s_day)
            s_ts = datetime.datetime(s_dt.year, s_dt.month, s_dt.day, tzinfo=datetime.timezone.utc).timestamp()
            e_dt = datetime.date.fromordinal(e_day)
            e_ts = datetime.datetime(e_dt.year, e_dt.month, e_dt.day, tzinfo=datetime.timezone.utc).timestamp()

            if has_b:
                b_dt = datetime.date.fromordinal(b_day)
                b_time = datetime.datetime(b_dt.year, b_dt.month, b_dt.day, tzinfo=datetime.timezone.utc).timestamp()
                is_post = (b_day >= onset_ordinal)
                lat_seg = float((b_time - onset_ts) / 86400.0) if is_post else np.nan
                mags = m.get("magnitudes", [0.0])
                mag_seg = float(np.linalg.norm(mags))
                if is_post:
                    post_onset_breaks.append((b_time, lat_seg, mag_seg))
            else:
                b_time = np.nan
                is_post = False
                lat_seg = np.nan
                mag_seg = 0.0

            seg_rec = np.zeros(1, dtype=SEGMENT_RECORD_DTYPE)[0]
            seg_rec["condition_idx"] = cond_idx
            seg_rec["material_id"] = m_id
            seg_rec["material_name"] = m_name
            seg_rec["mixture_fraction"] = f_weight
            seg_rec["patch_id"] = p_id
            seg_rec["patch_name"] = p_name
            seg_rec["pixel_num"] = p_num
            seg_rec["scene_row"] = scene_r
            seg_rec["scene_col"] = scene_c
            seg_rec["segment_num"] = m_idx_seg
            seg_rec["start_timestamp"] = s_ts
            seg_rec["end_timestamp"] = e_ts
            seg_rec["has_break"] = has_b
            seg_rec["break_num"] = m_idx_seg if has_b else 0
            seg_rec["break_timestamp"] = b_time
            seg_rec["confirmation_timestamp"] = b_time
            seg_rec["is_post_onset"] = is_post
            seg_rec["latency_days"] = lat_seg
            seg_rec["confirmation_latency_days"] = lat_seg
            seg_rec["break_magnitude"] = mag_seg
            rmse_val = m.get("rmse", [0.0])
            seg_rec["rmse"] = float(np.mean(rmse_val)) if hasattr(rmse_val, '__len__') else float(rmse_val)
            seg_rec["num_observations"] = int(m.get("observation_count", 0))
            pyccd_segment_list.append(seg_rec)

        # Compare against baseline PyCCD evaluation (produced by pyccd_landsat_main.py)
        base_count = int(baseline_pyccd["count"][p_idx, roff, coff])
        base_date = float(baseline_pyccd["date"][p_idx, roff, coff])

        n_post = len(post_onset_breaks)
        has_mult = (n_post > 1)

        if n_post > 0:
            pri_b_ts, pri_lat, pri_mag = post_onset_breaks[0]
            has_break_post_onset = (not np.isnan(pri_b_ts)) and (pri_b_ts >= onset_ts)
            is_new_break = has_break_post_onset and (
                (base_count == 0) or
                np.isnan(base_date) or
                (b_count > base_count) or
                (pri_b_ts < base_date)
            )
            detected = is_new_break
            break_ts = pri_b_ts if detected else np.nan
            latency = pri_lat if detected else np.nan
            mag = pri_mag if detected else 0.0
        else:
            detected = False
            break_ts = np.nan
            latency = np.nan
            mag = 0.0

        # Update the patch spatial arrays for the implanted pixel
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
        rec["break_timestamp"] = break_ts
        rec["latency_days"] = latency
        rec["break_magnitude"] = mag
        rec["p_value"] = 1.0 if detected else 0.0
        rec["total_breaks"] = b_count
        rec["post_onset_breaks"] = n_post
        rec["has_multiple_post_breaks"] = has_mult
        rec["confirmation_timestamp"] = break_ts
        rec["confirmation_latency_days"] = latency

        rec_idx += 1

    detected_count = np.sum(results_table["break_detected"])
    multiple_break_count = np.sum(results_table["has_multiple_post_breaks"])
    pyccd_segments = np.array(pyccd_segment_list, dtype=SEGMENT_RECORD_DTYPE)
    print(f"PyCCD complete: {detected_count}/{total_records} breaks detected ({detected_count/total_records*100:.1f}%).")
    print(f"PyCCD multiple post-implant breaks: {multiple_break_count}/{total_records} ({multiple_break_count/total_records*100:.1f}%).")
    print(f"PyCCD total segment models recorded: {len(pyccd_segments)} across {total_records} instances.")

    spatial_results = {
        "change_date": pyccd_change_dates,
        "change_count": pyccd_change_counts,
    }
    if return_spatial:
        return results_table, spatial_results, pyccd_segments
    return results_table, pyccd_segments


# ==============================================================================
# STEP 6: COMMIT BENCHMARK DATASET TO MASTER HDF5 FILE
# ==============================================================================
def save_benchmark_to_hdf5(h5_file, chips, metadata, target_spectra, sim_z_cubes, scccd_table, pyccd_table,
                           scccd_spatial=None, pyccd_spatial=None,
                           scccd_segments=None, pyccd_segments=None,
                           target_metric=DEFAULT_TARGET_METRIC, group_name=None):
    """
    Writes the complete benchmark hierarchy to the master HDF5 container.
    """
    if group_name is None:
        group_name = "/STRATIFIED_CHIP_BENCHMARK_3D" if target_metric == "volume_map_3d_zscore" else "/STRATIFIED_CHIP_BENCHMARK"
    print(f"\n[Step 6] Saving '{group_name}' group to master HDF5 file (metric: {target_metric})...")

    if group_name in h5_file:
        print(f"Overwriting existing '{group_name}' group...")
        del h5_file[group_name]

    base_grp = h5_file.create_group(group_name)
    base_grp.attrs["target_metric"] = target_metric

    # 1. METADATA Subgroup
    meta_grp = base_grp.create_group("METADATA")
    meta_grp.attrs["patch_size"] = PATCH_SIZE
    meta_grp.attrs["stride"] = 4
    meta_grp.attrs["injection_onset_date"] = INJECTION_ONSET_DATE
    meta_grp.attrs["num_conditions"] = len(target_spectra) * len(MIXTURE_FRACTIONS)
    meta_grp.attrs["patch_ids"] = np.array([p["id"] for p in LAND_COVER_PATCHES], dtype="S16")
    meta_grp.attrs["patch_names"] = np.array([p["name"] for p in LAND_COVER_PATCHES], dtype="S48")
    meta_grp.attrs["patch_bounds"] = np.array([[p["r0"], p["c0"], PATCH_SIZE, PATCH_SIZE] for p in LAND_COVER_PATCHES], dtype=np.int32)
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
    l30_ds = base_chips_grp.create_dataset(
        "HLSL30",
        data=chips["HLSL30"],
        dtype=chips["HLSL30"].dtype,
        fillvalue=metadata["fill_values"]["HLSL30"],
        compression="gzip",
        compression_opts=1
    )
    l30_ds.attrs["scale_factor"] = metadata["scale_factors"]["HLSL30"]
    l30_ds.attrs["_FillValue"] = metadata["fill_values"]["HLSL30"]

    fm_l30_ds = base_chips_grp.create_dataset(
        "HLSL30_Fmask",
        data=chips["HLSL30_Fmask"],
        dtype=chips["HLSL30_Fmask"].dtype,
        fillvalue=metadata["mask_fill_values"]["HLSL30_Fmask"],
        compression="gzip",
        compression_opts=1
    )
    fm_l30_ds.attrs["_FillValue"] = metadata["mask_fill_values"]["HLSL30_Fmask"]

    cm_l30_ds = base_chips_grp.create_dataset(
        "HLSL30_common_mask",
        data=chips["HLSL30_common_mask"],
        dtype=chips["HLSL30_common_mask"].dtype,
        fillvalue=metadata["mask_fill_values"]["HLSL30_common_mask"],
        compression="gzip",
        compression_opts=1
    )
    cm_l30_ds.attrs["_FillValue"] = metadata["mask_fill_values"]["HLSL30_common_mask"]

    s30_ds = base_chips_grp.create_dataset(
        "HLSS30",
        data=chips["HLSS30"],
        dtype=chips["HLSS30"].dtype,
        fillvalue=metadata["fill_values"]["HLSS30"],
        compression="gzip",
        compression_opts=1
    )
    s30_ds.attrs["scale_factor"] = metadata["scale_factors"]["HLSS30"]
    s30_ds.attrs["_FillValue"] = metadata["fill_values"]["HLSS30"]

    fm_s30_ds = base_chips_grp.create_dataset(
        "HLSS30_Fmask",
        data=chips["HLSS30_Fmask"],
        dtype=chips["HLSS30_Fmask"].dtype,
        fillvalue=metadata["mask_fill_values"]["HLSS30_Fmask"],
        compression="gzip",
        compression_opts=1
    )
    fm_s30_ds.attrs["_FillValue"] = metadata["mask_fill_values"]["HLSS30_Fmask"]

    cm_s30_ds = base_chips_grp.create_dataset(
        "HLSS30_common_mask",
        data=chips["HLSS30_common_mask"],
        dtype=chips["HLSS30_common_mask"].dtype,
        fillvalue=metadata["mask_fill_values"]["HLSS30_common_mask"],
        compression="gzip",
        compression_opts=1
    )
    cm_s30_ds.attrs["_FillValue"] = metadata["mask_fill_values"]["HLSS30_common_mask"]

    if "TANAGER" in chips:
        tan_ds = base_chips_grp.create_dataset(
            "TANAGER",
            data=chips["TANAGER"],
            dtype=chips["TANAGER"].dtype,
            fillvalue=metadata["fill_values"]["TANAGER"],
            compression="gzip",
            compression_opts=1
        )
        tan_ds.attrs["scale_factor"] = metadata["scale_factors"]["TANAGER"]
        tan_ds.attrs["_FillValue"] = metadata["fill_values"]["TANAGER"]

    harm_cm_ds = base_chips_grp.create_dataset(
        "HARMONIZED_common_mask",
        data=chips["HARMONIZED_common_mask"],
        dtype=chips["HARMONIZED_common_mask"].dtype,
        fillvalue=metadata["mask_fill_values"]["HARMONIZED_common_mask"],
        compression="gzip",
        compression_opts=1
    )
    harm_cm_ds.attrs["_FillValue"] = metadata["mask_fill_values"]["HARMONIZED_common_mask"]

    harm_z = base_chips_grp.create_dataset(
        "HARMONIZED_z_score",
        data=chips["HARMONIZED_z_score"],
        dtype=chips["HARMONIZED_z_score"].dtype,
        fillvalue=np.nan,
        compression="gzip",
        compression_opts=1
    )
    harm_z.attrs["_FillValue"] = np.nan

    if "BASELINE_SCCCD" in chips:
        base_chips_grp.create_dataset(
            "scccd_baseline_change_date",
            data=chips["BASELINE_SCCCD"]["date"],
            compression="gzip",
            compression_opts=1
        )
        base_chips_grp.create_dataset(
            "scccd_baseline_change_count",
            data=chips["BASELINE_SCCCD"]["count"],
            compression="gzip",
            compression_opts=1
        )
    if "BASELINE_PYCCD" in chips:
        base_chips_grp.create_dataset(
            "pyccd_baseline_change_date",
            data=chips["BASELINE_PYCCD"]["date"],
            compression="gzip",
            compression_opts=1
        )
        base_chips_grp.create_dataset(
            "pyccd_baseline_change_count",
            data=chips["BASELINE_PYCCD"]["count"],
            compression="gzip",
            compression_opts=1
        )

    # 3. SIMULATED_CUBES Subgroup (Stored as native float32 with gzip level 6)
    sim_grp = base_grp.create_group("SIMULATED_CUBES")
    sim_ds = sim_grp.create_dataset(
        target_metric,
        data=sim_z_cubes,
        dtype=sim_z_cubes.dtype,
        fillvalue=np.nan,
        compression="gzip",
        compression_opts=1,
        chunks=(1, 1, min(sim_z_cubes.shape[2], 128), PATCH_SIZE, PATCH_SIZE),
    )
    metric_desc = "Simulated 3D Equivalent Spectral Complexity Z-scores" if target_metric == "volume_map_3d_zscore" else "Simulated Gramian Spectral Complexity Z-scores"
    sim_ds.attrs["_FillValue"] = np.nan
    sim_ds.attrs["description"] = f"{metric_desc} under 50 perturbation conditions"

    # 4. RESULTS Subgroup
    res_grp = base_grp.create_group("RESULTS")
    res_grp.create_dataset("scccd_break_table", data=scccd_table, compression="gzip")
    res_grp.create_dataset("pyccd_break_table", data=pyccd_table, compression="gzip")
    if scccd_segments is not None:
        res_grp.create_dataset("scccd_break_segments", data=scccd_segments, compression="gzip")
    if pyccd_segments is not None:
        res_grp.create_dataset("pyccd_break_segments", data=pyccd_segments, compression="gzip")

    if scccd_spatial is not None:
        for k, arr in scccd_spatial.items():
            res_grp.create_dataset(
                f"scccd_{k}",
                data=arr,
                compression="gzip",
                compression_opts=1,
                chunks=(1, 1, PATCH_SIZE, PATCH_SIZE),
            )
    if pyccd_spatial is not None:
        for k, arr in pyccd_spatial.items():
            res_grp.create_dataset(
                f"pyccd_{k}",
                data=arr,
                compression="gzip",
                compression_opts=1,
                chunks=(1, 1, PATCH_SIZE, PATCH_SIZE),
            )

    print(f"Successfully committed {group_name} to {getattr(h5_file, 'filename', MASTER_H5_PATH)}.")


# ==============================================================================
# STEP 7: PRINT QUANTITATIVE PERFORMANCE SUMMARY
# ==============================================================================
def print_performance_summary(scccd_table, pyccd_table):
    """
    Prints comparative detection accuracy, latency, and multiple-break statistics stratified by land-cover class.
    """
    print("\n" + "=" * 115)
    print("LAND-COVER STRATIFIED BENCHMARK PERFORMANCE SUMMARY: SC-CCD vs. USGS PYCCD")
    print("=" * 115)

    patch_ids = [p["id"] for p in LAND_COVER_PATCHES]
    patch_names = [p["name"] for p in LAND_COVER_PATCHES]

    # 1. Performance Stratified by Land-Cover Class
    print(f"\n{'Stratum ID':<10} | {'Land-Cover Stratum':<32} | {'SC-CCD Recall':<15} | {'PyCCD Recall':<15} | {'SC Multi-Break':<15} | {'Py Multi-Break':<15} | {'Median Latency (SC / Py)'}")
    print("-" * 135)

    for p_id, p_name in zip(patch_ids, patch_names):
        sc_sub = scccd_table[scccd_table["patch_id"] == p_id.encode()]
        py_sub = pyccd_table[pyccd_table["patch_id"] == p_id.encode()]

        sc_rec = np.mean(sc_sub["break_detected"]) * 100.0
        py_rec = np.mean(py_sub["break_detected"]) * 100.0

        sc_multi = np.mean(sc_sub["has_multiple_post_breaks"]) * 100.0
        py_multi = np.mean(py_sub["has_multiple_post_breaks"]) * 100.0

        sc_lat = sc_sub["latency_days"][sc_sub["break_detected"]]
        py_lat = py_sub["latency_days"][py_sub["break_detected"]]

        sc_med_lat = f"{np.median(sc_lat):.1f} d" if len(sc_lat) > 0 else "N/A"
        py_med_lat = f"{np.median(py_lat):.1f} d" if len(py_lat) > 0 else "N/A"

        print(f"{p_id:<10} | {p_name:<32} | {sc_rec:5.1f}% ({np.sum(sc_sub['break_detected']):4d}/{len(sc_sub):4d}) | "
              f"{py_rec:5.1f}% ({np.sum(py_sub['break_detected']):4d}/{len(py_sub):4d}) | "
              f"{sc_multi:5.1f}% ({np.sum(sc_sub['has_multiple_post_breaks']):4d}/{len(sc_sub):4d}) | "
              f"{py_multi:5.1f}% ({np.sum(py_sub['has_multiple_post_breaks']):4d}/{len(py_sub):4d}) | "
              f"{sc_med_lat:<12} / {py_med_lat}")

    # 2. Performance Stratified by Target Material
    print(f"\n{'Material ID':<18} | {'Material Name':<34} | {'SC Recall':<12} | {'Py Recall':<12} | {'SC Multi %':<12} | {'Py Multi %':<12} | {'Median Latency (SC / Py)'}")
    print("-" * 135)

    mat_ids = np.unique(scccd_table["material_id"])
    for m_id in mat_ids:
        sc_sub = scccd_table[scccd_table["material_id"] == m_id]
        py_sub = pyccd_table[pyccd_table["material_id"] == m_id]

        m_name = sc_sub["material_name"][0].decode()
        m_id_str = m_id.decode()

        sc_rec = np.mean(sc_sub["break_detected"]) * 100.0
        py_rec = np.mean(py_sub["break_detected"]) * 100.0

        sc_multi = np.mean(sc_sub["has_multiple_post_breaks"]) * 100.0
        py_multi = np.mean(py_sub["has_multiple_post_breaks"]) * 100.0

        sc_lat = sc_sub["latency_days"][sc_sub["break_detected"]]
        py_lat = py_sub["latency_days"][py_sub["break_detected"]]

        sc_med_lat = f"{np.median(sc_lat):.1f} d" if len(sc_lat) > 0 else "N/A"
        py_med_lat = f"{np.median(py_lat):.1f} d" if len(py_lat) > 0 else "N/A"

        print(f"{m_id_str:<18} | {m_name:<34} | {sc_rec:5.1f}% ({np.sum(sc_sub['break_detected']):3d}/{len(sc_sub):3d}) | "
              f"{py_rec:5.1f}% ({np.sum(py_sub['break_detected']):3d}/{len(py_sub):3d}) | "
              f"{sc_multi:5.1f}% ({np.sum(sc_sub['has_multiple_post_breaks']):3d}/{len(sc_sub):3d}) | "
              f"{py_multi:5.1f}% ({np.sum(py_sub['has_multiple_post_breaks']):3d}/{len(py_sub):3d}) | "
              f"{sc_med_lat:<12} / {py_med_lat}")

    # 3. Overall Aggregate Performance
    total_n = len(scccd_table)
    total_sc_rec = np.mean(scccd_table["break_detected"]) * 100.0
    total_py_rec = np.mean(pyccd_table["break_detected"]) * 100.0
    total_sc_multi = np.mean(scccd_table["has_multiple_post_breaks"]) * 100.0
    total_py_multi = np.mean(pyccd_table["has_multiple_post_breaks"]) * 100.0

    all_sc_lat = scccd_table["latency_days"][scccd_table["break_detected"]]
    all_py_lat = pyccd_table["latency_days"][pyccd_table["break_detected"]]

    print("-" * 135)
    print(f"{'OVERALL AGGREGATE':<18} | {f'All {len(patch_ids)} Strata & {len(mat_ids)} Materials (N={total_n})':<34} | "
          f"{total_sc_rec:5.1f}% ({np.sum(scccd_table['break_detected']):4d}/{total_n}) | "
          f"{total_py_rec:5.1f}% ({np.sum(pyccd_table['break_detected']):4d}/{total_n}) | "
          f"{total_sc_multi:5.1f}% ({np.sum(scccd_table['has_multiple_post_breaks']):4d}/{total_n}) | "
          f"{total_py_multi:5.1f}% ({np.sum(pyccd_table['has_multiple_post_breaks']):4d}/{total_n}) | "
          f"{np.median(all_sc_lat):.1f} d        / {np.median(all_py_lat):.1f} d")
    print("=" * 135 + "\n")


# ==============================================================================
# MAIN PIPELINE ENTRY POINT
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Land-cover stratified patches benchmark pipeline.")
    parser.add_argument("--file", type=str, default=MASTER_H5_PATH, help="Path to master HDF5 container")
    parser.add_argument("--metric", type=str, default=DEFAULT_TARGET_METRIC, choices=["volume_map_3d_zscore", "sliding_volume_z_score"], help="Spectral complexity metric to benchmark")
    parser.add_argument("--group-name", type=str, default=None, help="HDF5 group name to save benchmark to")
    args = parser.parse_args()

    h5_path = args.file
    target_metric = args.metric
    group_name = args.group_name

    t_start_total = time.perf_counter()
    print("=" * 80)
    print("LAND-COVER STRATIFIED PATCHES BENCHMARK PIPELINE")
    print(f"Multi-Sensor Semi-Empirical Target Injection (SC-CCD vs. USGS PyCCD) [metric: {target_metric}]")
    print("=" * 80)

    # 1. Parse onset timestamp
    onset_dt = datetime.datetime.strptime(INJECTION_ONSET_DATE, "%Y-%m-%d").replace(tzinfo=datetime.timezone.utc)
    onset_ts = onset_dt.timestamp()
    print(f"Injection onset date: {INJECTION_ONSET_DATE} (timestamp: {onset_ts:.0f})")

    # 2. Open Master HDF5 Container
    print(f"Opening master container: {h5_path}")
    with h5py.File(h5_path, "r+", locking=False) as h5_file:
        # Load Tanager wavelengths if available
        tanager_wl = None
        if "/HDFEOS/GRIDS/TANAGER/Data Fields/surface_reflectance" in h5_file:
            tanager_wl = h5_file["/HDFEOS/GRIDS/TANAGER/Data Fields/surface_reflectance"].attrs.get("wavelengths")

        # Load 10 resampled target spectra
        print(f"Loading USGS Splib07 spectra from: {SPLIB_BASE}")
        target_spectra = load_all_target_spectra(SPLIB_BASE, tanager_wl)
        print(f"Successfully prepared {len(target_spectra)} target spectra.")

        # Step 1: Extract baseline chips and metadata
        chips, metadata = extract_baseline_chips(h5_file, target_metric=target_metric)

        # Step 2 & 3: Simulate Spectral Complexity Z-scores on GPU
        sim_z_cubes, unperturbed_z = simulate_spectral_complexity_cubes(chips, metadata, target_spectra, onset_ts, target_metric=target_metric)
        chips["HARMONIZED_z_score"] = unperturbed_z

        # Step 4: Run Harmonized SC-CCD break detection (full 9x9 patch across time to evaluate neighbor effects)
        scccd_table, scccd_spatial, scccd_segments = run_scccd_benchmark(sim_z_cubes, chips, metadata, target_spectra, onset_ts)

        # Step 5: Run USGS LCMAP Landsat PyCCD break detection (4 implanted pixels per patch, copying baseline for neighbors)
        pyccd_table, pyccd_spatial, pyccd_segments = run_pyccd_benchmark(chips, metadata, target_spectra, onset_ts)

        # Step 6: Commit complete benchmark group to master HDF5
        save_benchmark_to_hdf5(
            h5_file, chips, metadata, target_spectra, sim_z_cubes, scccd_table, pyccd_table,
            scccd_spatial=scccd_spatial, pyccd_spatial=pyccd_spatial,
            scccd_segments=scccd_segments, pyccd_segments=pyccd_segments,
            target_metric=target_metric, group_name=group_name
        )

    # Step 7: Print quantitative summary
    print_performance_summary(scccd_table, pyccd_table)

    t_elapsed_total = (time.perf_counter() - t_start_total) / 60.0
    print(f"Stratified benchmark pipeline completed successfully in {t_elapsed_total:.2f} minutes.")


if __name__ == "__main__":
    main()

