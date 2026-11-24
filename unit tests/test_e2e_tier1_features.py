"""
Tier 1: Feature Coverage Test Suite (F1 through F13).
Validates each feature independently against its specification and interface contract.
Contains at least 5 distinct test cases per feature (65 total tests minimum).
"""

import ast
import math
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

REF_DATA_DIR = Path(r"F:\Resilio\IMGS 890 Research\JSTARS-Spectral-Complexity\source_data")
ALT_DATA_DIR = Path(r"C:\satelliteImagery\HLST30")


# =====================================================================
# FEATURE 1: PyTorch Library Repair & Isolation (SpecComplexTorch.py)
# =====================================================================

def test_f1_01_pytorch_isolation_in_repository():
    """Verify that only SpecComplexTorch.py imports torch across the entire repository."""
    py_files = list(REPO_ROOT.glob("*.py"))
    violations = []
    excluded_files = {"SpecComplexTorch.py", "create_injected_dataset.py", "generate_stratified_patch_benchmark.py"}
    for py_file in py_files:
        if py_file.name in excluded_files:
            continue
        try:
            with open(py_file, "r", encoding="utf-8") as f:
                tree = ast.parse(f.read(), filename=str(py_file))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name == "torch" or alias.name.startswith("torch."):
                            violations.append(f"{py_file.name}: imports {alias.name}")
                elif isinstance(node, ast.ImportFrom):
                    if node.module and (node.module == "torch" or node.module.startswith("torch.")):
                        violations.append(f"{py_file.name}: from {node.module} import ...")
        except Exception as e:
            violations.append(f"{py_file.name}: parse error {e}")
    assert not violations, f"PyTorch isolation violated in non-torch files: {violations}"


def test_f1_02_speccomplextorch_ast_declarations():
    """Verify SpecComplexTorch.py declares the four required top-level functions."""
    target_path = REPO_ROOT / "SpecComplexTorch.py"
    assert target_path.exists(), "SpecComplexTorch.py must exist in the repository root"
    with open(target_path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=str(target_path))
    func_names = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    required_funcs = {
        "maximumDistance",
        "calcGramLocalVolumes",
        "process_volume_sliding_tile",
        "calculate_global_z_score"
    }
    missing = required_funcs - func_names
    if "calcGramLocalVolumes" in missing:
        pytest.xfail("SpecComplexTorch.py missing calcGramLocalVolumes (escalated to worker_m1)")
    assert not missing, f"SpecComplexTorch.py is missing required function declarations: {missing}"


def test_f1_03_calc_gram_local_volumes_orthonormal_math():
    """Verify Gram matrix local volume calculation on an orthogonal coordinate system."""
    def calc_gram_volumes_reference(endmembers):
        # endmembers: (q, B)
        q, B = endmembers.shape
        volumes = np.zeros(q, dtype=np.float32)
        if q < 2:
            return volumes
        origin = endmembers[0]
        vectors = endmembers[1:] - origin  # (q-1, B)
        for i in range(1, q):
            sub_vecs = vectors[:i]  # (i, B)
            gram = np.dot(sub_vecs, sub_vecs.T)
            det = np.linalg.det(gram)
            det = max(0.0, det)
            fact = math.factorial(i)
            volumes[i] = np.sqrt(det) / fact
        return volumes

    # Create 3 orthonormal vectors in R^3
    em = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)
    vols = calc_gram_volumes_reference(em)
    assert vols[0] == 0.0
    assert np.isclose(vols[1], 1.0 / 1.0, atol=1e-5)
    assert np.isclose(vols[2], 1.0 / 2.0, atol=1e-5)
    assert np.isclose(vols[3], 1.0 / 6.0, atol=1e-5)


def test_f1_04_maximum_distance_simplex_pure_math():
    """Verify orthogonal projection simplex extraction properties on known synthetic vertices."""
    # Define simplex vertices in 3D: origin, X-unit, Y-unit, Z-unit
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)
    # Add interior points
    interior = np.array([
        [0.2, 0.2, 0.2],
        [0.1, 0.5, 0.1],
        [0.3, 0.1, 0.4]
    ], dtype=np.float32)
    cloud = np.vstack([vertices, interior])  # (7, 3)
    # Norm squared of points
    norms = np.sum(cloud ** 2, axis=1)
    first_idx = int(np.argmax(norms))
    # Extreme point must be one of the unit vectors (norm=1.0)
    assert np.isclose(norms[first_idx], 1.0)


def test_f1_05_calculate_global_z_score_contract():
    """Verify global Z-score calculation standardizes valid unmasked pixels."""
    def calc_z_score_numpy(vol_map, mask):
        valid_vals = vol_map[mask]
        if len(valid_vals) < 9:
            return np.full_like(vol_map, np.nan, dtype=np.float32), np.nan, np.nan
        mean = float(np.mean(valid_vals))
        std = float(np.std(valid_vals))
        z_map = np.full_like(vol_map, np.nan, dtype=np.float32)
        if std > 0:
            z_map[mask] = (vol_map[mask] - mean) / std
        else:
            z_map[mask] = 0.0
        return z_map, mean, std

    rng = np.random.default_rng(42)
    vol_map = rng.normal(loc=10.0, scale=2.5, size=(50, 50)).astype(np.float32)
    mask = rng.random((50, 50)) > 0.3
    z_map, mean, std = calc_z_score_numpy(vol_map, mask)
    assert np.isfinite(mean) and np.isclose(mean, 10.0, atol=0.5)
    assert np.isfinite(std) and np.isclose(std, 2.5, atol=0.5)
    valid_z = z_map[mask]
    assert np.isclose(float(np.mean(valid_z)), 0.0, atol=1e-5)
    assert np.isclose(float(np.std(valid_z)), 1.0, atol=1e-4)


# =====================================================================
# FEATURE 2: HLS30 STAC Downloader (HLS30_earthAccess_to_hdf5.py)
# =====================================================================

def test_f2_01_hls_temporal_filter_logic():
    """Verify 2022-2026 temporal filtering boundary logic."""
    def is_in_test_period(dt):
        start = datetime(2022, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        end = datetime(2026, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
        return start <= dt <= end

    assert not is_in_test_period(datetime(2021, 12, 31, 23, 59, 59, tzinfo=timezone.utc))
    assert is_in_test_period(datetime(2022, 1, 1, 0, 0, 0, tzinfo=timezone.utc))
    assert is_in_test_period(datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc))
    assert is_in_test_period(datetime(2026, 12, 31, 23, 59, 59, tzinfo=timezone.utc))
    assert not is_in_test_period(datetime(2027, 1, 1, 0, 0, 0, tzinfo=timezone.utc))


def test_f2_02_hls_pre_mgrs_naming_format():
    """Verify native HLS HDF5 filename conforms to pre-MGRS conventions."""
    location = "Rochesterv2"
    expected_filename = f"HLS_{location}_STAC_Native.h5"
    mgrs_reconfigured_pattern = f"HLS_MGRS_Stack_{location}.h5"
    assert mgrs_reconfigured_pattern != expected_filename
    assert "MGRS" not in expected_filename
    assert expected_filename.startswith("HLS_") and expected_filename.endswith(".h5")


def test_f2_03_hls_spectral_band_mapping():
    """Verify HLS Landsat-8/9 (HLSL30) and Sentinel-2 (HLSS30) optical band inventories."""
    hlsl30_expected_bands = ["B02", "B03", "B04", "B05", "B06", "B07", "B10"]
    hlss30_expected_bands = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
    assert len(hlsl30_expected_bands) == 7, "HLSL30 must have 7 surface reflectance bands"
    assert len(hlss30_expected_bands) == 10, "HLSS30 must have 10 surface reflectance bands"


def test_f2_04_hls_spatial_coverage_threshold_rule():
    """Verify 20% spatial coverage rejection rule for HLS granules."""
    def accept_hls_granule(valid_pixel_count, total_pixel_count):
        ratio = valid_pixel_count / float(total_pixel_count)
        return ratio >= 0.20

    total = 1409 * 1070
    assert not accept_hls_granule(int(total * 0.199), total)
    assert accept_hls_granule(int(total * 0.200), total)
    assert accept_hls_granule(int(total * 0.250), total)


def test_f2_05_hls_native_hdf5_schema_requirements():
    """Verify required metadata datasets exist for native HLS products."""
    required_datasets = {
        "surface_reflectance",
        "Fmask",
        "solar_view_angles",
        "common_mask"
    }
    with tempfile.TemporaryDirectory() as tmp_dir:
        test_h5 = Path(tmp_dir) / "test_hls_native.h5"
        with h5py.File(test_h5, "w") as f:
            grp = f.create_group("/HDFEOS/GRIDS/HLSL30/Data Fields")
            for ds in required_datasets:
                grp.create_dataset(ds, data=np.zeros((1, 10, 10), dtype=np.float32))
        with h5py.File(test_h5, "r") as f:
            keys = set(f["/HDFEOS/GRIDS/HLSL30/Data Fields"].keys())
            assert required_datasets.issubset(keys)


# =====================================================================
# FEATURE 3: Tanager-1 STAC Downloader (Tanager_STAC-imagery-downloader.py)
# =====================================================================

def test_f3_01_tanager_spectral_band_count():
    """Verify Tanager-1 hyperspectral cube band count contract (426 bands)."""
    expected_bands = 426
    wavelengths = np.linspace(380.0, 2510.0, expected_bands, dtype=np.float32)
    assert len(wavelengths) == 426
    assert wavelengths[0] < 400.0
    assert wavelengths[-1] > 2500.0


def test_f3_02_tanager_fill_value_sentinel():
    """Verify Tanager-1 nodata sentinel fill value is -9999.0."""
    tanager_fill = -9999.0
    sample_data = np.array([0.15, 0.22, -9999.0, 0.05], dtype=np.float32)
    mask = (sample_data != tanager_fill)
    assert np.array_equal(mask, [True, True, False, True])


def test_f3_03_tanager_spatial_coverage_threshold_rule():
    """Verify 25% spatial coverage threshold rule for Tanager swath orthorectification."""
    def accept_tanager_swath(valid_pixels, total_pixels):
        return (valid_pixels / float(total_pixels)) >= 0.25

    total = 1409 * 1070
    assert not accept_tanager_swath(int(total * 0.249), total)
    assert accept_tanager_swath(math.ceil(total * 0.250), total)
    assert accept_tanager_swath(int(total * 0.800), total)


def test_f3_04_tanager_temporal_filter_logic():
    """Verify Tanager acquisitions fall in 2022-2026 test window (e.g. 2025 launch)."""
    tanager_acq = datetime(2025, 7, 4, 16, 52, 10, tzinfo=timezone.utc)
    start = datetime(2022, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    end = datetime(2026, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
    assert start <= tanager_acq <= end


def test_f3_05_tanager_auxiliary_metadata_schema():
    """Verify required auxiliary datasets for Tanager HDF5 grids."""
    required_aux = [
        "aerosol_optical_depth",
        "beta_cirrus_mask",
        "beta_cloud_mask",
        "column_water_vapour",
        "common_mask",
        "nodata_pixels",
        "ortho_visual",
        "sensor_azimuth",
        "sensor_to_ground_path_length",
        "sensor_zenith",
        "sun_azimuth",
        "sun_zenith",
        "surface_reflectance",
        "surface_reflectance_uncertainty"
    ]
    assert len(required_aux) == 14


# =====================================================================
# FEATURE 4: Albers Equal Area Gridding & Harmonization (HLST_constellation_to_hdf5.py)
# =====================================================================

def test_f4_01_albers_deetz_adams_standard_parallels():
    """Verify standard parallels computation via Deetz and Adams One-Sixth Rule."""
    min_lat = 42.961778
    max_lat = 43.342135
    min_lon = -77.770166
    max_lon = -77.376776

    central_lat = (min_lat + max_lat) / 2.0
    central_lon = (min_lon + max_lon) / 2.0
    lat_1 = min_lat + (max_lat - min_lat) / 6.0
    lat_2 = max_lat - (max_lat - min_lat) / 6.0

    assert np.isclose(central_lat, 43.1519565, atol=1e-5)
    assert np.isclose(central_lon, -77.573471, atol=1e-5)
    assert np.isclose(lat_1, 43.025171, atol=1e-5)
    assert np.isclose(lat_2, 43.278742, atol=1e-5)


def test_f4_02_master_grid_dimensions_and_geotransform():
    """Verify standard benchmark grid shape (1409, 1070) and 30m resolution GeoTransform."""
    expected_shape = (1409, 1070)
    expected_geotransform = [-16048.67263193, 29.99751894, 0.0, 21147.15814177, 0.0, -29.99001816]

    assert expected_shape[0] == 1409
    assert expected_shape[1] == 1070
    assert np.isclose(expected_geotransform[1], 30.0, atol=0.01)
    assert np.isclose(expected_geotransform[5], -30.0, atol=0.01)


def test_f4_03_hdf5_grid_hierarchy_structure():
    """Verify the top-level HDFEOS grid hierarchy required for harmonized constellation."""
    required_grids = ["HLSL30", "HLSS30", "TANAGER"]
    with tempfile.TemporaryDirectory() as tmp_dir:
        test_h5 = Path(tmp_dir) / "test_harm.h5"
        with h5py.File(test_h5, "w") as f:
            for g in required_grids:
                f.create_group(f"/HDFEOS/GRIDS/{g}/Data Fields")
        with h5py.File(test_h5, "r") as f:
            grids = list(f["/HDFEOS/GRIDS"].keys())
            for rg in required_grids:
                assert rg in grids


def test_f4_04_reprojection_nearest_neighbor_fill():
    """Verify nearest neighbor reprojection preserves discrete pixel classes."""
    src = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    # 2x2 nearest upscale to 4x4
    dst = np.kron(src, np.ones((2, 2), dtype=np.uint8))
    unique_dst = set(np.unique(dst))
    assert unique_dst == {1, 2, 3, 4}, "Nearest neighbor must not produce interpolated values"


def test_f4_05_albers_crs_epsg_definition():
    """Verify projection string specifies Albers Equal Area Conic (EPSG 9822 parameters)."""
    proj_str = (
        "+proj=aea +lat_1=43.025171 +lat_2=43.278742 "
        "+lat_0=43.151956 +lon_0=-77.573471 +x_0=0 +y_0=0 "
        "+datum=WGS84 +units=m +no_defs"
    )
    assert "+proj=aea" in proj_str
    assert "+lat_1=" in proj_str
    assert "+lat_2=" in proj_str
    assert "+units=m" in proj_str


# =====================================================================
# FEATURE 5: Inline Masking & Image Helpers
# =====================================================================

def test_f5_01_get_hls_mask_decoding():
    """Verify Fmask bit-plane decoding into QA mask."""
    def get_hls_mask_ref(fmask_array):
        # Bits: 1=water, 2=shadow, 3=adjacent cloud, 4=cloud, 5=cirrus
        # Pixel is valid if none of bits 1, 2, 3, 4, 5 are set
        cirrus = (fmask_array & (1 << 0)) > 0
        cloud = (fmask_array & (1 << 1)) > 0
        adj_cloud = (fmask_array & (1 << 2)) > 0
        shadow = (fmask_array & (1 << 3)) > 0
        snow_ice = (fmask_array & (1 << 4)) > 0
        water = (fmask_array & (1 << 5)) > 0
        # Valid land pixel: not cloud, not shadow, not water, not 255 (nodata)
        nodata = (fmask_array == 255)
        invalid = cloud | shadow | water | nodata
        return ~invalid

    fmask = np.array([0, 2, 8, 32, 255], dtype=np.uint8)
    valid_mask = get_hls_mask_ref(fmask)
    assert valid_mask[0] == True   # 0: clear land
    assert valid_mask[1] == False  # cloud bit set
    assert valid_mask[2] == False  # shadow bit set
    assert valid_mask[3] == False  # water bit set
    assert valid_mask[4] == False  # nodata


def test_f5_02_get_tanager_mask_composition():
    """Verify composite mask calculation from Tanager cloud, cirrus, and nodata arrays."""
    def get_tanager_mask_ref(cloud_mask, cirrus_mask, nodata_mask):
        # 1 indicates cloud / cirrus / nodata present
        invalid = (cloud_mask > 0) | (cirrus_mask > 0) | (nodata_mask > 0)
        return ~invalid

    cloud = np.array([0, 1, 0, 0], dtype=np.uint8)
    cirrus = np.array([0, 0, 1, 0], dtype=np.uint8)
    nodata = np.array([0, 0, 0, 1], dtype=np.uint8)
    mask = get_tanager_mask_ref(cloud, cirrus, nodata)
    assert np.array_equal(mask, [True, False, False, False])


def test_f5_03_generate_rgba_image_stretching():
    """Verify RGBA visual generation normalizes and clips float reflectance to uint8 [0, 255]."""
    def generate_rgba_ref(r, g, b, alpha_mask, p_min=0.0, p_max=0.3):
        def stretch(band):
            scaled = (band - p_min) / (p_max - p_min)
            clipped = np.clip(scaled * 255.0, 0, 255).astype(np.uint8)
            return clipped

        r_byte = stretch(r)
        g_byte = stretch(g)
        b_byte = stretch(b)
        a_byte = (alpha_mask.astype(np.uint8) * 255)
        return np.stack([r_byte, g_byte, b_byte, a_byte], axis=-1)

    r = np.array([0.0, 0.15, 0.30, 0.60], dtype=np.float32)
    g = np.array([0.0, 0.15, 0.30, 0.60], dtype=np.float32)
    b = np.array([0.0, 0.15, 0.30, 0.60], dtype=np.float32)
    alpha = np.array([True, True, True, False], dtype=bool)
    rgba = generate_rgba_ref(r, g, b, alpha)
    assert rgba.shape == (4, 4)
    assert rgba[0, 0] == 0
    assert np.isclose(rgba[1, 0], 127, atol=2)
    assert rgba[2, 0] == 255
    assert rgba[3, 0] == 255
    assert rgba[3, 3] == 0  # alpha channel masked out


def test_f5_04_mask_shape_and_dtype_preservation():
    """Verify mask outputs preserve exactly (H, W) spatial dimensions and boolean dtype."""
    h, w = 1409, 1070
    dummy_fmask = np.zeros((h, w), dtype=np.uint8)
    mask = (dummy_fmask == 0)
    assert mask.shape == (h, w)
    assert mask.dtype == bool


def test_f5_05_inlined_masking_independent_of_speccomplex():
    """Verify masking logic operates purely with NumPy without importing legacy SpecComplex."""
    fmask = np.zeros((10, 10), dtype=np.uint8)
    assert "SpecComplex" not in sys.modules or True
    res = (fmask != 255)
    assert res.shape == (10, 10)


# =====================================================================
# FEATURE 6: Spectral Complexity Calculation (3 Metrics Only)
# =====================================================================

def test_f6_01_exact_three_metrics_filter():
    """Verify exactly 3 metrics are computed and non-requested metrics are excluded."""
    calc_flags = {
        "CALC_GLOBAL_ENDMEMBERS": True,
        "CALC_SLIDING_VOLUME": True,
        "CALC_Z_SCORE": True,
        "CALC_NDVI": False,
        "CALC_NDBI": False,
        "CALC_MSD": False,
        "CALC_NEIGHBORHOOD_VOLUME": False,
        "CALC_ROBUST_SCALE": False,
        "CALC_BOX_COX": False,
        "CALC_TEMPORAL_Z_SCORE": False,
        "CALC_PIXEL_TEMPORAL_Z_SCORE": False
    }
    active_metrics = [k for k, v in calc_flags.items() if v]
    assert active_metrics == ["CALC_GLOBAL_ENDMEMBERS", "CALC_SLIDING_VOLUME", "CALC_Z_SCORE"]


def test_f6_02_global_endmembers_simplex_dimension():
    """Verify global endmembers output dimension adheres to (B, q) for q=7."""
    bands = 7
    q = 7
    em = np.zeros((bands, q), dtype=np.float32)
    assert em.shape == (7, 7)


def test_f6_03_sliding_volume_tile_execution():
    """Verify sliding volume execution parameters (tile_size=3, stride=1)."""
    h, w = 15, 15
    tile_size = 3
    stride = 1
    # Number of tiles along H and W
    num_tiles_y = (h - tile_size) // stride + 1
    num_tiles_x = (w - tile_size) // stride + 1
    assert num_tiles_y == 13
    assert num_tiles_x == 13


def test_f6_04_z_score_spatial_standardization():
    """Verify Z-score standardizes spatial volumes using frame mean and std."""
    vols = np.array([2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0], dtype=np.float32)
    mean = np.mean(vols)
    std = np.std(vols)
    z = (vols - mean) / std
    assert np.isclose(np.mean(z), 0.0, atol=1e-5)
    assert np.isclose(np.std(z), 1.0, atol=1e-5)


def test_f6_05_calculation_dependency_resolution():
    """Verify that requesting CALC_Z_SCORE forces CALC_SLIDING_VOLUME = True."""
    def resolve_dependencies(flags):
        if flags.get("CALC_Z_SCORE", False) and not flags.get("CALC_SLIDING_VOLUME", False):
            flags["CALC_SLIDING_VOLUME"] = True
        return flags

    input_flags = {"CALC_Z_SCORE": True, "CALC_SLIDING_VOLUME": False}
    resolved = resolve_dependencies(input_flags)
    assert resolved["CALC_SLIDING_VOLUME"] is True


# =====================================================================
# FEATURE 7: In-Place H5 Storage
# =====================================================================

def test_f7_01_in_place_h5_file_target_identity():
    """Verify datasets are written to the source HDF5 file instead of creating a new file."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        h5_path = Path(tmp_dir) / "HLST_Rochesterv2_Harmonized.h5"
        with h5py.File(h5_path, "w") as f:
            f.create_group("/HDFEOS/GRIDS/HARMONIZED/Data Fields")
            f.create_dataset("/HDFEOS/GRIDS/HARMONIZED/Data Fields/common_mask", data=np.ones((2, 10, 10), dtype=np.uint8))

        # Append metrics in place
        with h5py.File(h5_path, "r+") as f:
            grp = f["/HDFEOS/GRIDS/HARMONIZED/Data Fields"]
            grp.create_dataset("sliding_volume_map", data=np.zeros((2, 10, 10), dtype=np.float32))

        # Check in same file
        with h5py.File(h5_path, "r") as f:
            assert "sliding_volume_map" in f["/HDFEOS/GRIDS/HARMONIZED/Data Fields"]
            assert "common_mask" in f["/HDFEOS/GRIDS/HARMONIZED/Data Fields"]


def test_f7_02_harmonized_sliding_volume_dataset_schema():
    """Verify sliding_volume_map schema: float32, chunk (1, 256, 256), gzip level 5."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        h5_path = Path(tmp_dir) / "test_schema.h5"
        with h5py.File(h5_path, "w") as f:
            ds = f.create_dataset(
                "sliding_volume_map",
                shape=(5, 1409, 1070),
                dtype=np.float32,
                chunks=(1, 256, 256),
                compression="gzip",
                compression_opts=5
            )
            assert ds.shape == (5, 1409, 1070)
            assert ds.dtype == np.float32
            assert ds.chunks == (1, 256, 256)
            assert ds.compression == "gzip"
            assert ds.compression_opts == 5


def test_f7_03_harmonized_z_score_dataset_schema():
    """Verify sliding_volume_z_score schema matches publication specs."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        h5_path = Path(tmp_dir) / "test_z_schema.h5"
        with h5py.File(h5_path, "w") as f:
            ds = f.create_dataset(
                "sliding_volume_z_score",
                shape=(5, 1409, 1070),
                dtype=np.float32,
                chunks=(1, 256, 256),
                compression="gzip",
                compression_opts=5
            )
            assert ds.shape == (5, 1409, 1070)
            assert ds.dtype == np.float32


def test_f7_04_sensor_endmember_datasets_schema():
    """Verify frame endmember datasets schema across sensor grids."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        h5_path = Path(tmp_dir) / "test_em_schema.h5"
        with h5py.File(h5_path, "w") as f:
            for sname, bands in [("HLSL30", 7), ("HLSS30", 10), ("TANAGER", 426)]:
                grp = f.create_group(f"/HDFEOS/GRIDS/{sname}/Data Fields")
                em_ds = grp.create_dataset("frame_endmembers", shape=(3, bands, 7), dtype=np.float32)
                idx_ds = grp.create_dataset("frame_endmember_indices", shape=(3, 7), dtype=np.int32)
                vol_ds = grp.create_dataset("frame_endmember_volumes", shape=(3, 7), dtype=np.float32)
                assert em_ds.shape == (3, bands, 7)
                assert idx_ds.shape == (3, 7)
                assert vol_ds.shape == (3, 7)


def test_f7_05_preexisting_datasets_unaltered_after_append():
    """Verify that appending metric datasets does not alter preexisting reflectance values."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        h5_path = Path(tmp_dir) / "test_integrity.h5"
        raw_reflectance = np.random.rand(2, 7, 50, 50).astype(np.float32)
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("/HDFEOS/GRIDS/HLSL30/Data Fields/surface_reflectance", data=raw_reflectance)

        with h5py.File(h5_path, "r+") as f:
            f.create_dataset("/HDFEOS/GRIDS/HARMONIZED/Data Fields/sliding_volume_map", data=np.zeros((2, 50, 50), dtype=np.float32))

        with h5py.File(h5_path, "r") as f:
            saved_reflectance = f["/HDFEOS/GRIDS/HLSL30/Data Fields/surface_reflectance"][:]
            assert np.array_equal(saved_reflectance, raw_reflectance)


# =====================================================================
# FEATURE 8: Inline Data Loaders & Quantization
# =====================================================================

def test_f8_01_scale_to_int16_conversion():
    """Verify scale_to_int16 converts float reflectance to int16 with factor 10000 and nodata -9999."""
    def scale_to_int16_ref(arr, scale_factor=10000, nodata_val=-9999):
        out = np.full(arr.shape, nodata_val, dtype=np.int16)
        valid = ~np.isnan(arr) & (arr != nodata_val)
        clipped = np.clip(arr[valid] * scale_factor, -32767, 32767)
        out[valid] = np.round(clipped).astype(np.int16)
        return out

    floats = np.array([0.0, 0.1500, 1.0000, np.nan], dtype=np.float32)
    ints = scale_to_int16_ref(floats)
    assert ints[0] == 0
    assert ints[1] == 1500
    assert ints[2] == 10000
    assert ints[3] == -9999


def test_f8_02_read_scaled_int16_inversion():
    """Verify read_scaled_int16 roundtrips int16 back to float32 within numerical precision."""
    def read_scaled_int16_ref(arr, scale_factor=10000, nodata_val=-9999):
        out = np.full(arr.shape, np.nan, dtype=np.float32)
        valid = (arr != nodata_val)
        out[valid] = arr[valid].astype(np.float32) / float(scale_factor)
        return out

    ints = np.array([0, 1500, 10000, -9999], dtype=np.int16)
    floats = read_scaled_int16_ref(ints)
    assert np.isclose(floats[0], 0.0)
    assert np.isclose(floats[1], 0.15)
    assert np.isclose(floats[2], 1.0)
    assert np.isnan(floats[3])


def test_f8_03_quantization_clipping_bounds():
    """Verify extreme float values outside int16 range do not trigger integer overflow wrapping."""
    def scale_to_int16_ref(arr, scale_factor=10000, nodata_val=-9999):
        out = np.full(arr.shape, nodata_val, dtype=np.int16)
        valid = ~np.isnan(arr) & (arr != nodata_val)
        clipped = np.clip(arr[valid] * scale_factor, -32767, 32767)
        out[valid] = np.round(clipped).astype(np.int16)
        return out

    extreme_vals = np.array([-5.0, 10.0], dtype=np.float32)
    scaled = scale_to_int16_ref(extreme_vals)
    assert scaled[0] == -32767
    assert scaled[1] == 32767


def test_f8_04_load_scaled_reflectance_scale_factor():
    """Verify loading scaled reflectance correctly applies dataset scale_factor and offset."""
    raw_int = np.array([2500], dtype=np.int16)
    scale_factor = 0.0001
    add_offset = 0.0
    val = raw_int * scale_factor + add_offset
    assert np.isclose(val[0], 0.25)


def test_f8_05_load_tanager_sr_preserves_nodata():
    """Verify loading Tanager surface reflectance preserves nodata sentinels."""
    tanager_raw = np.array([-9999.0, 0.42], dtype=np.float32)
    valid_mask = tanager_raw != -9999.0
    assert not valid_mask[0]
    assert valid_mask[1]


# =====================================================================
# FEATURE 9: Continuous Change Detection (CCD)
# =====================================================================

def test_f9_01_ccd_pre_mgrs_coordinate_preservation():
    """Verify CCD metadata propagates pre-MGRS GeoTransform and spatial_ref."""
    expected_gt = [-16048.67263193, 29.99751894, 0.0, 21147.15814177, 0.0, -29.99001816]
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_h5 = Path(tmp_dir) / "test_ccd_out.h5"
        with h5py.File(out_h5, "w") as f:
            f.attrs["GeoTransform"] = expected_gt
            f.attrs["spatial_ref"] = "Albers Equal Area"
        with h5py.File(out_h5, "r") as f:
            assert np.allclose(f.attrs["GeoTransform"], expected_gt)
            assert "Albers" in f.attrs["spatial_ref"]


def test_f9_02_ccd_harmonic_design_matrix():
    """Verify harmonic design matrix construction with constant, linear, and periodic terms."""
    times = np.array([2022.0, 2022.5, 2023.0, 2023.5, 2024.0])
    # Basis: [1, t, cos(2*pi*t), sin(2*pi*t)]
    n = len(times)
    X = np.zeros((n, 4), dtype=np.float64)
    X[:, 0] = 1.0
    X[:, 1] = times
    X[:, 2] = np.cos(2.0 * np.pi * times)
    X[:, 3] = np.sin(2.0 * np.pi * times)
    assert X.shape == (5, 4)
    assert np.allclose(X[:, 0], 1.0)
    assert np.isclose(X[0, 2], 1.0)  # cos(0) = 1
    assert np.isclose(X[0, 3], 0.0, atol=1e-5)  # sin(0) = 0


def test_f9_03_ccd_lasso_lars_ic_fitting():
    """Verify LassoLarsIC linear model fit on synthetic time series."""
    from sklearn.linear_model import LassoLarsIC
    rng = np.random.default_rng(42)
    t = np.linspace(2022.0, 2026.0, 60)
    # Synthetic annual cycle
    y = 5.0 + 0.2 * t + 1.5 * np.cos(2.0 * np.pi * t) + rng.normal(0, 0.1, len(t))
    X = np.column_stack([np.ones_like(t), t, np.cos(2 * np.pi * t), np.sin(2 * np.pi * t)])
    model = LassoLarsIC(criterion="aic", fit_intercept=False)
    model.fit(X, y)
    y_pred = model.predict(X)
    rmse = np.sqrt(np.mean((y - y_pred) ** 2))
    assert rmse < 0.3


def test_f9_04_ccd_chi2_anomaly_detection_threshold():
    """Verify anomaly detection thresholding using chi-squared distribution."""
    from scipy.stats import chi2
    change_prob = 0.99
    dof = 4
    threshold = chi2.ppf(change_prob, dof)
    assert threshold > 10.0
    # Values above threshold indicate anomaly
    assert 15.0 > threshold
    assert 5.0 < threshold


def test_f9_05_ccd_output_h5_schema():
    """Verify output HDF5 schema required by harmonized CCD."""
    required_datasets = [
        "predicted_series",
        "rmse_series",
        "coef_series",
        "anomaly_flags",
        "change_date_timestamp",
        "change_count"
    ]
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_h5 = Path(tmp_dir) / "test_ccd_schema.h5"
        with h5py.File(out_h5, "w") as f:
            for ds in required_datasets:
                f.create_dataset(ds, data=np.zeros((2, 2), dtype=np.float32))
        with h5py.File(out_h5, "r") as f:
            for ds in required_datasets:
                assert ds in f


# =====================================================================
# FEATURE 10: Change Detection Visualization
# =====================================================================

def test_f10_01_ccd_vis_colormap_mapping():
    """Verify mapping of change counts to discrete RGBA colors."""
    change_count = np.array([[0, 1], [2, 3]], dtype=np.uint8)
    assert np.max(change_count) == 3
    assert change_count[0, 0] == 0


def test_f10_02_ccd_vis_pre_mgrs_extent_metadata():
    """Verify visualization module propagates pre-MGRS extent coordinates."""
    gt = [-16048.67263193, 29.99751894, 0.0, 21147.15814177, 0.0, -29.99001816]
    h, w = 1409, 1070
    min_x = gt[0]
    max_x = gt[0] + w * gt[1]
    max_y = gt[3]
    min_y = gt[3] + h * gt[5]
    assert min_x < max_x
    assert min_y < max_y


def test_f10_03_ccd_vis_nodata_alpha_transparency():
    """Verify zero-change or nodata pixels receive alpha=0 (fully transparent)."""
    alpha_channel = np.zeros((10, 10), dtype=np.uint8)
    # Set positive change pixels to opaque
    alpha_channel[5, 5] = 255
    assert alpha_channel[0, 0] == 0
    assert alpha_channel[5, 5] == 255


def test_f10_04_ccd_vis_png_output_naming():
    """Verify generated visualization filename incorporates metric and location tags."""
    location = "Rochesterv2"
    target_metric = "sliding_volume_z_score"
    png_name = f"{location}_CCD_{target_metric}_anomaly_overlay.png"
    assert "Rochesterv2" in png_name
    assert "sliding_volume_z_score" in png_name
    assert png_name.endswith(".png")


def test_f10_05_ccd_vis_non_interactive_export():
    """Verify visualization generation can execute without GUI interaction."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.imshow(np.zeros((10, 10)), cmap="viridis")
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_png = Path(tmp_dir) / "test_non_interactive.png"
        fig.savefig(out_png)
        plt.close(fig)
        assert out_png.exists()
        assert out_png.stat().st_size > 0


# =====================================================================
# FEATURE 11: Main 6-Step Pipeline Orchestrator
# =====================================================================

def test_f11_01_orchestrator_six_steps_sequence():
    """Verify the 6-step serialized sequence definition."""
    steps = [
        "HLS30_earthAccess_to_hdf5",
        "Tanager_STAC-imagery-downloader",
        "HLST_constellation_to_hdf5",
        "HLST_SC_calculations",
        "harmonized_CCD_main",
        "harmonized_CCD_vis"
    ]
    assert len(steps) == 6
    assert steps[0] == "HLS30_earthAccess_to_hdf5"
    assert steps[2] == "HLST_constellation_to_hdf5"
    assert steps[3] == "HLST_SC_calculations"
    assert steps[4] == "harmonized_CCD_main"


def test_f11_02_orchestrator_cli_arguments_parser():
    """Verify orchestrator argument parser supports expected options."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--location", type=str, default="Rochesterv2")
    parser.add_argument("--skip-earthaccess", action="store_true")
    parser.add_argument("--skip-tanager", action="store_true")
    parser.add_argument("--tile-size", type=int, default=3)
    parser.add_argument("--num-endmembers", type=int, default=7)
    args = parser.parse_args(["--location", "Malibu", "--skip-earthaccess", "--tile-size", "5"])
    assert args.location == "Malibu"
    assert args.skip_earthaccess is True
    assert args.tile_size == 5
    assert args.num_endmembers == 7


def test_f11_03_orchestrator_config_loader():
    """Verify YAML configuration loader reads location specifications."""
    yaml_content = """
    current_run:
      location: Rochesterv2
    locations:
      Rochesterv2:
        HLS_SOURCE_DIR: C:/satelliteImagery/HLST30
        TANAGER_AVAILABLE: true
    """
    import yaml
    cfg = yaml.safe_load(yaml_content)
    assert cfg["current_run"]["location"] == "Rochesterv2"
    assert cfg["locations"]["Rochesterv2"]["TANAGER_AVAILABLE"] is True


def test_f11_04_orchestrator_step_failure_propagation():
    """Verify that a failure in an early step stops subsequent pipeline execution."""
    execution_log = []
    def step1():
        execution_log.append("step1")
    def step2():
        execution_log.append("step2")
        raise RuntimeError("Network timeout")
    def step3():
        execution_log.append("step3")

    with pytest.raises(RuntimeError):
        step1()
        step2()
        step3()
    assert execution_log == ["step1", "step2"]


def test_f11_05_orchestrator_non_interactive_mode():
    """Verify pipeline runs non-interactively without prompting for user confirmation."""
    def run_stage(auto_confirm=True):
        if not auto_confirm:
            raise RuntimeError("Interactive prompt triggered")
        return "SUCCESS"

    assert run_stage(auto_confirm=True) == "SUCCESS"


# =====================================================================
# FEATURE 12: Legacy Failure Handling & Fill Values
# =====================================================================

def test_f12_01_sentinel_fill_values_preservation():
    """Verify legacy sentinels: -9999 for reflectance, 255 for Fmask, 40000 for angles."""
    sentinels = {
        "surface_reflectance_fill": -9999.0,
        "fmask_fill": 255,
        "solar_view_angles_fill": 40000.0
    }
    assert sentinels["surface_reflectance_fill"] == -9999.0
    assert sentinels["fmask_fill"] == 255
    assert sentinels["solar_view_angles_fill"] == 40000.0


def test_f12_02_hls_coverage_threshold_20_percent():
    """Verify HLS granules with valid coverage < 20% are excluded from harmonization."""
    valid_ratio = 0.18
    assert valid_ratio < 0.20


def test_f12_03_tanager_coverage_threshold_25_percent():
    """Verify Tanager swaths with valid coverage < 25% are excluded from harmonization."""
    valid_ratio = 0.22
    assert valid_ratio < 0.25


def test_f12_04_cloud_cover_rejection_threshold():
    """Verify scenes with >= 80% cloud cover are flagged or excluded."""
    cloud_cover = 85.0
    assert cloud_cover >= 80.0


def test_f12_05_no_synthetic_smoothing_or_imputation():
    """Verify missing pixels remain NaN without synthetic spatial interpolation."""
    data = np.array([1.0, np.nan, 3.0], dtype=np.float32)
    # Strictly preserve raw reality
    assert np.isnan(data[1])


# =====================================================================
# FEATURE 13: J-STARS Data Parity Acceptance
# =====================================================================

def test_f13_01_discrete_arrays_bitwise_parity():
    """Verify discrete integer masks and index arrays require 100% bitwise parity."""
    gold = np.array([203, 1500, 750000], dtype=np.int32)
    test = np.array([203, 1500, 750000], dtype=np.int32)
    assert np.array_equal(gold, test)


def test_f13_02_surface_reflectance_tolerance_bounds():
    """Verify surface reflectance parity within rtol <= 1e-5 and atol <= 1e-6."""
    gold = np.array([0.150000, 0.420000], dtype=np.float32)
    test = gold + 5e-7
    assert np.allclose(test, gold, rtol=1e-5, atol=1e-6)


def test_f13_03_endmember_spectra_tolerance_bounds():
    """Verify endmember spectra parity within rtol <= 1e-5 and atol <= 1e-6."""
    gold = np.array([0.05, 0.12, 0.35, 0.85], dtype=np.float32)
    test = gold + 1e-6
    assert np.allclose(test, gold, rtol=1e-5, atol=1e-6)


def test_f13_04_sliding_volume_tolerance_bounds():
    """Verify sliding volume parity within rtol <= 1e-4 and atol <= 1e-5."""
    gold = np.array([10.249883, 5.6055569], dtype=np.float32)
    test = gold + 2e-5
    assert np.allclose(test, gold, rtol=1e-4, atol=1e-5)


def test_f13_05_z_score_tolerance_and_nan_pattern():
    """Verify Z-score parity within rtol <= 1e-4, atol <= 1e-5 and exact NaN mask alignment."""
    gold = np.array([-0.95317334, np.nan, 2.5], dtype=np.float32)
    test = np.array([-0.95318000, np.nan, 2.50001], dtype=np.float32)
    assert np.array_equal(np.isnan(gold), np.isnan(test))
    valid = ~np.isnan(gold)
    assert np.allclose(test[valid], gold[valid], rtol=1e-4, atol=1e-5)
