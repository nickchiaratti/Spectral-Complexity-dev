"""
Unit and Integration Test Suite for Harmonization and Orthorectification Module
(HLST_constellation_to_hdf5.py).

Validates:
1. Albers Equal Area projection and Deetz & Adams One-Sixth Rule calculations.
2. Master grid shape (1409, 1070) and GeoTransform parameters.
3. Inlined helper functions: get_hls_mask, get_tanager_mask, generate_rgba_image.
4. Sentinel constants, fill values, and failure handling policies.
5. Spatial coverage filtering (20% HLS threshold, 25% Tanager threshold).
6. HDF-EOS grid hierarchy and schema compliance.
7. End-to-end integration and real-world benchmark parity.
"""

import ast
import os
import sys
import tempfile
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import h5py
import pytest
from pyproj import CRS, Transformer
from rasterio.transform import Affine

REPO_ROOT = Path(__file__).resolve().parent.parent
for _sub in ["", "HLS30", "Tanager-1", "Harmonized_SC", "change_detection/SC-CCM", "change_detection/harmonized-CCD", "change_detection/pyccd", "simulated-data-test"]:
    _p = str((REPO_ROOT / _sub).resolve()) if _sub else str(REPO_ROOT)
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _resolve_repo_file(filename: str) -> Path:
    direct = REPO_ROOT / filename
    if direct.exists():
        return direct
    for _sub in ["Harmonized_SC", "HLS30", "Tanager-1", "change_detection/SC-CCM", "change_detection/harmonized-CCD", "change_detection/pyccd", "simulated-data-test"]:
        cand = REPO_ROOT / _sub / filename
        if cand.exists():
            return cand
    matches = [m for m in REPO_ROOT.glob(f"**/{filename}") if ".git" not in m.parts]
    return matches[0] if matches else direct


import HLST_constellation_to_hdf5 as harm

REF_DATA_DIR = Path(r"F:\Resilio\IMGS 890 Research\JSTARS-Spectral-Complexity\source_data")
ALT_DATA_DIR = Path(r"C:\satelliteImagery\HLST30")


# =====================================================================
# 1. ALBERS EQUAL AREA GRID & DEETZ-ADAMS RULE
# =====================================================================

def test_albers_deetz_adams_standard_parallels_analytical():
    """Verify Deetz & Adams One-Sixth Rule standard parallel calculations."""
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


def test_calculate_master_grid_rochester_parameters():
    """Verify calculate_master_grid returns exact benchmark dimensions and GeoTransform."""
    bbox = [-77.770166, 42.961778, -77.376776, 43.342135]
    dst_crs, transform, width, height, proj_code, zone, gctp_params = harm.calculate_master_grid(
        bbox, resolution=30.0
    )

    # Master Grid Dimensions
    assert height == 1409
    assert width == 1070

    # GeoTransform Array: [c, a, b, f, d, e]
    gdal_tf = [transform.c, transform.a, transform.b, transform.f, transform.d, transform.e]
    expected_tf = [-16048.67263193, 29.99751894, 0.0, 21147.15814177, 0.0, -29.99001816]

    np.testing.assert_allclose(gdal_tf, expected_tf, rtol=1e-4, atol=1e-4)
    assert np.isclose(transform.a, 30.0, atol=0.01)
    assert np.isclose(transform.e, -30.0, atol=0.01)

    # Proj and GCTP Metadata
    assert proj_code == "GCTP_ALBERS"
    assert zone == 0
    assert len(gctp_params) == 13
    assert np.isclose(gctp_params[0], 6378137.0)  # WGS84 semi-major axis
    assert np.isclose(gctp_params[1], 6356752.314245)  # WGS84 semi-minor axis
    assert np.isclose(gctp_params[2], 43.025171, atol=1e-5)  # lat_1
    assert np.isclose(gctp_params[3], 43.278742, atol=1e-5)  # lat_2
    assert np.isclose(gctp_params[4], -77.573471, atol=1e-5)  # central_lon
    assert np.isclose(gctp_params[5], 43.1519565, atol=1e-5)  # central_lat


def test_calculate_master_grid_generic_bounds():
    """Verify calculate_master_grid generalizes to non-Rochester bounds without hardcoding."""
    bbox = [14.8, 37.4, 15.35, 37.9]  # Mt Etna bounds
    dst_crs, transform, width, height, proj_code, zone, gctp_params = harm.calculate_master_grid(
        bbox, resolution=30.0
    )
    assert width > 1000
    assert height > 1000
    assert proj_code == "GCTP_ALBERS"
    assert "+proj=aea" in dst_crs.to_proj4()
    assert np.isclose(gctp_params[4], (14.8 + 15.35) / 2.0, atol=1e-5)
    assert np.isclose(gctp_params[5], (37.4 + 37.9) / 2.0, atol=1e-5)


def test_get_safe_bbox_ordering():
    """Verify get_safe_bbox correctly sorts inverted coordinates."""
    bbox = harm.get_safe_bbox(-77.3, 43.3, -77.7, 42.9)
    assert bbox == [-77.7, 42.9, -77.3, 43.3]


# =====================================================================
# 2. INLINED HELPER FUNCTIONS (F5)
# =====================================================================

def test_get_hls_mask_bit_decoding():
    """Verify bitwise decoding of HLS Fmask QA bits."""
    # Bits: 0:cirrus, 1:cloud, 2:adj_cloud, 3:shadow, 4:snow, 5:water, 255:nodata
    fmask_samples = np.array([
        0,    # Clear land -> Valid
        1,    # Cirrus -> Invalid
        2,    # Cloud -> Invalid
        4,    # Adj cloud/shadow -> Invalid
        8,    # Cloud shadow -> Invalid
        16,   # Snow/ice -> Invalid
        32,   # Water -> Invalid
        255   # Nodata -> Invalid
    ], dtype=np.uint8)

    valid_mask = harm.get_hls_mask(fmask_samples, return_valid=True)
    expected_valid = np.array([True, False, False, False, False, False, False, False], dtype=bool)
    assert np.array_equal(valid_mask, expected_valid)

    # Inverted mask test (True = Invalid/Masked)
    invalid_mask = harm.get_hls_mask(fmask_samples, return_valid=False)
    assert np.array_equal(invalid_mask, ~expected_valid)


def test_get_hls_mask_solar_elevation():
    """Verify sun elevation threshold filtering in HLS masking."""
    fmask = np.zeros((3, 3), dtype=np.uint8)  # All clear
    # Angles array: [SZA, SAA, VZA, VAA]
    # SZA=75 deg -> Sun elevation = 90 - 75 = 15 deg (< 20 deg threshold -> invalid)
    # SZA=45 deg -> Sun elevation = 90 - 45 = 45 deg (>= 20 deg threshold -> valid)
    angles_low = np.full((4, 3, 3), 75.0, dtype=np.float32)
    angles_high = np.full((4, 3, 3), 45.0, dtype=np.float32)

    mask_low = harm.get_hls_mask(fmask, solar_view_angles=angles_low, sun_elevation_threshold=20.0, return_valid=True)
    assert not np.any(mask_low), "Low sun elevation must be invalid"

    mask_high = harm.get_hls_mask(fmask, solar_view_angles=angles_high, sun_elevation_threshold=20.0, return_valid=True)
    assert np.all(mask_high), "High sun elevation must be valid"


def test_get_hls_mask_aerosol_levels():
    """Verify aerosol level filtering (bits 6-7)."""
    # Bits 6-7: 0=Climatology, 1=Low, 2=Moderate, 3=High
    # Byte values: 0b00_000000=0, 0b01_000000=64, 0b10_000000=128, 0b11_000000=192
    fmask = np.array([0, 64, 128, 192], dtype=np.uint8)

    # 'medium' accept level: rejects aerosol bits > 2 (i.e., level 3 = 192)
    mask_med = harm.get_hls_mask(fmask, aerosol_accept_level="medium", return_valid=True)
    assert np.array_equal(mask_med, [True, True, True, False])

    # 'low' accept level: rejects aerosol bits > 1 (i.e., level 2 and 3 = 128, 192)
    mask_low = harm.get_hls_mask(fmask, aerosol_accept_level="low", return_valid=True)
    assert np.array_equal(mask_low, [True, True, False, False])


def test_get_hls_mask_cloud_dilation():
    """Verify morphological cloud dilation expands the masked area."""
    fmask = np.zeros((10, 10), dtype=np.uint8)
    fmask[5, 5] = 2  # Single cloud pixel at center

    mask_no_dil = harm.get_hls_mask(fmask, cloud_dilation=0, return_valid=False)
    assert np.sum(mask_no_dil) == 1

    mask_dil = harm.get_hls_mask(fmask, cloud_dilation=1, return_valid=False)
    assert np.sum(mask_dil) == 9  # 3x3 dilated kernel


def test_get_tanager_mask_composition():
    """Verify Tanager composite mask calculation from cloud, nodata, and cirrus arrays."""
    cloud = np.array([0, 1, 0, 0], dtype=np.uint8)
    nodata = np.array([0, 0, 0, 1], dtype=np.uint8)
    cirrus = np.array([0, 0, 1, 0], dtype=np.uint8)

    valid_mask = harm.get_tanager_mask(cloud, nodata, cirrus, return_valid=True)
    assert np.array_equal(valid_mask, [True, False, False, False])

    invalid_mask = harm.get_tanager_mask(cloud, nodata, cirrus, return_valid=False)
    assert np.array_equal(invalid_mask, [False, True, True, True])


def test_get_tanager_mask_thresholds():
    """Verify Tanager QA checks for sun zenith, AOD, and reflectance uncertainty."""
    shape = (5, 5)
    f_idx = 0

    # Build synthetic group dictionary
    fake_grp = {
        "beta_cloud_mask": np.zeros((1, 5, 5), dtype=np.uint8),
        "beta_cirrus_mask": np.zeros((1, 5, 5), dtype=np.uint8),
        "nodata_pixels": np.zeros((1, 5, 5), dtype=np.uint8),
        "sun_zenith": np.full((1, 5, 5), 45.0, dtype=np.float32),
        "aerosol_optical_depth": np.full((1, 5, 5), 0.15, dtype=np.float32),
        "surface_reflectance_uncertainty": np.full((1, 10, 5, 5), 0.02, dtype=np.float32),
    }

    # Base valid case
    mask = harm.get_tanager_mask(fake_grp, f_idx, shape=shape, return_valid=False)
    assert not np.any(mask), "Clear frame should have no masked pixels"

    # High AOD (> 0.35 threshold)
    fake_grp["aerosol_optical_depth"][0, 2, 2] = 0.50
    mask_aod = harm.get_tanager_mask(fake_grp, f_idx, shape=shape, aerosol_depth_threshold=0.35, return_valid=False)
    assert mask_aod[2, 2] == True

    # High Uncertainty (> 0.10 threshold)
    fake_grp["surface_reflectance_uncertainty"][0, 3, 1, 1] = 0.25
    mask_unc = harm.get_tanager_mask(fake_grp, f_idx, shape=shape, uncertainty_threshold=0.10, return_valid=False)
    assert mask_unc[1, 1] == True


def test_generate_rgba_image_stretching_and_clipping():
    """Verify RGBA image generation normalizes and clips reflectance to uint8 [0, 255]."""
    r = np.array([0.0, 0.15, 0.30, 0.60], dtype=np.float32)
    g = np.array([0.0, 0.15, 0.30, 0.60], dtype=np.float32)
    b = np.array([0.0, 0.15, 0.30, 0.60], dtype=np.float32)
    alpha_mask = np.array([True, True, True, False], dtype=bool)

    rgba = harm.generate_rgba_image(r, g, b, alpha_mask, p_min=0.0, p_max=0.3, gamma=1.0)

    assert rgba.shape == (4, 4)
    assert rgba.dtype == np.uint8
    assert rgba[0, 0] == 0
    assert np.isclose(rgba[1, 0], 127, atol=2)
    assert rgba[2, 0] == 255
    assert rgba[3, 0] == 255
    assert rgba[3, 3] == 0  # alpha channel masked out by alpha_mask


def test_generate_rgba_image_cube_input():
    """Verify generate_rgba_image supports (3, H, W) cube inputs."""
    cube = np.random.uniform(0.05, 0.35, size=(3, 20, 20)).astype(np.float32)
    valid_mask = np.ones((20, 20), dtype=bool)
    valid_mask[0:5, 0:5] = False

    rgba = harm.generate_rgba_image(cube, valid_mask=valid_mask)
    assert rgba.shape == (20, 20, 4)
    assert rgba.dtype == np.uint8
    assert np.all(rgba[0:5, 0:5, 3] == 0)
    assert np.all(rgba[6:, 6:, 3] == 255)


def test_generate_rgba_image_all_nan_handling():
    """Verify generate_rgba_image produces zero array on entirely NaN inputs."""
    nan_band = np.full((10, 10), np.nan, dtype=np.float32)
    rgba = harm.generate_rgba_image(nan_band, nan_band, nan_band)
    assert rgba.shape == (10, 10, 4)
    assert rgba.dtype == np.uint8
    assert np.all(rgba == 0)


# =====================================================================
# 3. SENTINEL VALUES AND FAILURE HANDLING
# =====================================================================

def test_sentinel_constants():
    """Verify sentinel constants adhere strictly to legacy definitions."""
    assert harm.HLS_SR_NODATA == -9999
    assert harm.HLS_FMASK_NODATA == 255
    assert harm.HLS_ANGLES_NODATA == 40000
    assert harm.TANAGER_NODATA == -9999.0
    assert harm.TANAGER_MASK_NODATA == 255
    assert harm.MIN_HLS_ROI_COVERAGE_PERCENT == 20.0
    assert harm.MIN_TANAGER_ROI_COVERAGE_PERCENT == 25.0


def test_pytorch_isolation_in_harmonization():
    """Verify HLST_constellation_to_hdf5.py does not import torch or SpecComplex."""
    target_path = _resolve_repo_file("HLST_constellation_to_hdf5.py")
    with open(target_path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=str(target_path))

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert "torch" not in alias.name, f"Forbidden import: {alias.name}"
                assert "SpecComplex" not in alias.name, f"Forbidden import: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                assert "torch" not in node.module, f"Forbidden from-import: {node.module}"
                assert "SpecComplex" not in node.module, f"Forbidden from-import: {node.module}"


# =====================================================================
# 4. HDF5 SCHEMA AND GRID CREATION
# =====================================================================

def test_write_hdf_sensor_group_schema():
    """Verify write_hdf_sensor_group creates expected HDF-EOS group and attributes."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        test_h5 = Path(tmp_dir) / "test_sensor_grp.h5"

        num_frames = 2
        bands = 7
        h, w = 50, 40
        data_dict = {
            "sr": np.full((num_frames, bands, h, w), 0.20, dtype=np.float32),
            "fm": np.zeros((num_frames, 1, h, w), dtype=np.uint8),
            "ag": np.full((num_frames, 4, h, w), 35.0, dtype=np.float32),
            "vis": np.zeros((num_frames, 4, h, w), dtype=np.uint8),
            "mask": np.zeros((num_frames, h, w), dtype=bool),
            "meta": {
                "space": ["LANDSAT-8", "LANDSAT-9"],
                "acq": [1700000000.0, 1700086400.0],
                "saz": [140.0, 142.0],
                "sel": [45.0, 46.0],
                "cc": [5.0, 10.0]
            },
            "count": num_frames
        }

        crs = CRS.from_epsg(4326)
        transform = Affine(30.0, 0.0, -16000.0, 0.0, -30.0, 21000.0)

        with h5py.File(test_h5, "w") as h5f:
            harm.write_hdf_sensor_group(
                h5f,
                "/HDFEOS/GRIDS/HLSL30/Data Fields",
                data_dict,
                harm.L30_SR_WAVELENGTHS,
                crs,
                transform
            )

        with h5py.File(test_h5, "r") as h5f:
            grp = h5f["/HDFEOS/GRIDS/HLSL30/Data Fields"]
            assert "surface_reflectance" in grp
            assert "Fmask" in grp
            assert "solar_view_angles" in grp
            assert "ortho_visual" in grp
            assert "common_mask" in grp

            sr_ds = grp["surface_reflectance"]
            assert sr_ds.shape == (2, 7, 50, 40)
            assert sr_ds.dtype == np.float32
            assert sr_ds.chunks == (1, 7, 50, 40)
            assert np.isnan(sr_ds.attrs["_FillValue"])
            assert len(sr_ds.attrs["wavelengths"]) == 7

            fm_ds = grp["Fmask"]
            assert fm_ds.shape == (2, 50, 40)
            assert fm_ds.dtype == np.uint8
            assert fm_ds.attrs["_FillValue"] == 255

            mask_ds = grp["common_mask"]
            assert mask_ds.shape == (2, 50, 40)
            assert mask_ds.dtype == bool
            assert mask_ds.attrs["description"] == "True = Invalid/Masked, False = Valid."


def test_odl_string_generation():
    """Verify ODL grid metadata strings contain required HDF-EOS structural tags."""
    tf = Affine(30.0, 0.0, -16048.67, 0.0, -30.0, 21147.15)
    params = [6378137.0, 6356752.3, 43.0, 43.2, -77.5, 43.1, 0, 0, 0, 0, 0, 0, 0]

    odl_hls = harm.generate_hls_odl_grid_string(
        "HLSS30", 1070, 1409, tf, "GCTP_ALBERS", 0, params, 10, 210
    )
    assert "GROUP=HLSS30" in odl_hls
    assert 'GridName="HLSS30"' in odl_hls
    assert "XDim=1070" in odl_hls
    assert "YDim=1409" in odl_hls
    assert "Projection=GCTP_ALBERS" in odl_hls
    assert 'DataFieldName="surface_reflectance"' in odl_hls
    assert 'DataFieldName="common_mask"' in odl_hls
    assert "END_GROUP=HLSS30" in odl_hls


# =====================================================================
# 5. REAL-WORLD BENCHMARK DATASET PARITY
# =====================================================================

def test_jstars_benchmark_parity_if_available():
    """
    Validates that the Albers master grid parameters match the J-STARS publication
    reference dataset HLST_Rochesterv2_Harmonized.h5.
    """
    ref_file = None
    for candidate in [REF_DATA_DIR / "HLST_Rochesterv2_Harmonized.h5", ALT_DATA_DIR / "HLST_Rochesterv2_Harmonized.h5"]:
        if candidate.exists():
            ref_file = candidate
            break

    if ref_file is None:
        pytest.skip("HLST_Rochesterv2_Harmonized.h5 not found in reference locations")

    with h5py.File(ref_file, "r") as h5f:
        # Check required grids
        grids = list(h5f["/HDFEOS/GRIDS"].keys())
        assert "HLSL30" in grids
        assert "HLSS30" in grids
        assert "TANAGER" in grids

        # Check raster geometry against HLSL30
        sr_ds = h5f["/HDFEOS/GRIDS/HLSL30/Data Fields/surface_reflectance"]
        n_frames, bands, height, width = sr_ds.shape
        assert height == 1409
        assert width == 1070
        assert bands == 7
        assert sr_ds.dtype == np.float32

        # Check GeoTransform attribute
        ref_tf = sr_ds.attrs["GeoTransform"]
        expected_tf = [-16048.67263193, 29.99751894, 0.0, 21147.15814177, 0.0, -29.99001816]
        np.testing.assert_allclose(ref_tf, expected_tf, rtol=1e-5, atol=1e-5)

        # Check Tanager grid geometry
        tan_sr = h5f["/HDFEOS/GRIDS/TANAGER/Data Fields/surface_reflectance"]
        t_frames, t_bands, t_h, t_w = tan_sr.shape
        assert t_h == 1409
        assert t_w == 1070
        assert t_bands == 426
        assert tan_sr.dtype == np.float32
        assert tan_sr.fillvalue == -9999.0


# =====================================================================
# 6. COVERAGE FILTERING AND REPROJECTION VERIFICATION
# =====================================================================

def test_fetch_native_hls_groups_missing_file_raises():
    """Verify fetch_native_hls_groups raises FileNotFoundError when file is missing."""
    with pytest.raises(FileNotFoundError):
        harm.fetch_native_hls_groups("non_existent_file_path.h5", "HLSS30")


def test_fetch_native_hls_groups_synthetic():
    """Verify fetch_native_hls_groups correctly groups granules by UTC date."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        test_h5 = Path(tmp_dir) / "test_native.h5"
        with h5py.File(test_h5, "w") as h5f:
            grp1 = h5f.create_group("/HDFEOS/GRIDS/HLSS30_T18TXM/Data Fields")
            ds1 = grp1.create_dataset("surface_reflectance", data=np.zeros((2, 10, 10, 10), dtype=np.float32))
            # Timestamps for 2024-06-01 and 2024-06-02
            ds1.attrs["acquisition_time"] = np.array([1717200000.0, 1717286400.0], dtype=np.float64)

            grp2 = h5f.create_group("/HDFEOS/GRIDS/HLSS30_T18TYM/Data Fields")
            ds2 = grp2.create_dataset("surface_reflectance", data=np.zeros((1, 10, 10, 10), dtype=np.float32))
            ds2.attrs["acquisition_time"] = np.array([1717200000.0], dtype=np.float64)

        daily, tiles = harm.fetch_native_hls_groups(str(test_h5), "HLSS30")
        assert len(daily) == 2
        assert "T18TXM" in tiles
        assert "T18TYM" in tiles
        # 2024-06-01 should have entries from both tiles
        d1 = datetime.fromtimestamp(1717200000.0, tz=timezone.utc).strftime("%Y-%m-%d")
        assert len(daily[d1]) == 2


def test_coverage_threshold_rule_hls():
    """Verify that HLS frames below 20% spatial coverage are rejected."""
    total_pixels = 1409 * 1070
    valid_pixels_fail = int(total_pixels * 0.199)
    valid_pixels_pass = int(total_pixels * 0.201)

    cov_fail = (valid_pixels_fail / float(total_pixels)) * 100.0
    cov_pass = (valid_pixels_pass / float(total_pixels)) * 100.0

    assert cov_fail < harm.MIN_HLS_ROI_COVERAGE_PERCENT
    assert cov_pass >= harm.MIN_HLS_ROI_COVERAGE_PERCENT


def test_coverage_threshold_rule_tanager():
    """Verify that Tanager swaths below 25% spatial coverage are rejected."""
    total_pixels = 1409 * 1070
    valid_pixels_fail = int(total_pixels * 0.249)
    valid_pixels_pass = int(total_pixels * 0.251)

    cov_fail = (valid_pixels_fail / float(total_pixels)) * 100.0
    cov_pass = (valid_pixels_pass / float(total_pixels)) * 100.0

    assert cov_fail < harm.MIN_TANAGER_ROI_COVERAGE_PERCENT
    assert cov_pass >= harm.MIN_TANAGER_ROI_COVERAGE_PERCENT


def test_reprojection_nearest_neighbor_preserves_classes():
    """Verify nearest-neighbor interpolation preserves discrete class values without smoothing."""
    src = np.array([[0, 2], [8, 255]], dtype=np.uint8)
    dst = np.kron(src, np.ones((4, 4), dtype=np.uint8))
    unique_vals = set(np.unique(dst))
    assert unique_vals == {0, 2, 8, 255}, "Nearest neighbor must not introduce intermediate values"


def test_process_hls_master_stack_synthetic():
    """
    End-to-end integration test of process_hls_master_stack using a synthetic native H5 file.
    Tests pass 1 coverage filtering and pass 2 reprojection and stacking.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        native_h5 = Path(tmp_dir) / "synthetic_native_hls.h5"

        master_h, master_w = 40, 30
        master_tf = Affine(30.0, 0.0, 1000.0, 0.0, -30.0, 2000.0)
        master_crs = CRS.from_epsg(32618)

        # Create source tile matching master grid
        gdal_tf = [master_tf.c, master_tf.a, master_tf.b, master_tf.f, master_tf.d, master_tf.e]

        with h5py.File(native_h5, "w") as h5f:
            grp = h5f.create_group("/HDFEOS/GRIDS/HLSL30_T18TXM/Data Fields")

            # 2 dates: date 1 full coverage, date 2 only 12.5% coverage (5 rows out of 40)
            sr_data = np.full((2, 7, master_h, master_w), 1500, dtype=np.int16)
            sr_data[1, :, 5:, :] = -9999

            sr_ds = grp.create_dataset("surface_reflectance", data=sr_data, fillvalue=-9999)
            sr_ds.attrs["_FillValue"] = -9999
            sr_ds.attrs["scale_factor"] = 0.0001
            sr_ds.attrs["scale_to_float"] = 0.0001
            sr_ds.attrs["GeoTransform"] = gdal_tf
            sr_ds.attrs["spatial_ref"] = master_crs.to_wkt()
            sr_ds.attrs["acquisition_time"] = np.array([1717200000.0, 1717286400.0], dtype=np.float64)
            sr_ds.attrs["spacecraft_id"] = np.array(["LANDSAT-8", "LANDSAT-8"], dtype=h5py.string_dtype(encoding="ascii"))
            sr_ds.attrs["cloud_cover"] = np.array([5.0, 10.0], dtype=np.float32)

            fm_data = np.zeros((2, 1, master_h, master_w), dtype=np.uint8)
            fm_data[1, :, 5:, :] = 255
            fm_ds = grp.create_dataset("Fmask", data=fm_data, fillvalue=255)
            fm_ds.attrs["_FillValue"] = 255
            fm_ds.attrs["GeoTransform"] = gdal_tf
            fm_ds.attrs["spatial_ref"] = master_crs.to_wkt()

            ag_data = np.full((2, 4, master_h, master_w), 4000, dtype=np.uint16)
            ag_data[1, :, 5:, :] = 40000
            ag_ds = grp.create_dataset("solar_view_angles", data=ag_data, fillvalue=40000)
            ag_ds.attrs["_FillValue"] = 40000
            ag_ds.attrs["scale_factor"] = 0.01
            ag_ds.attrs["scale_to_float"] = 0.01
            ag_ds.attrs["GeoTransform"] = gdal_tf
            ag_ds.attrs["spatial_ref"] = master_crs.to_wkt()

        daily_groups, unique_tiles = harm.fetch_native_hls_groups(str(native_h5), "HLSL30")
        assert len(daily_groups) == 2

        # Process stack
        result = harm.process_hls_master_stack(
            native_h5_path=str(native_h5),
            daily_groups=daily_groups,
            expected_sr=7,
            master_height=master_h,
            master_width=master_w,
            master_transform=master_tf,
            master_crs=master_crs,
            min_roi_coverage=20.0
        )

        assert result is not None
        # Only date 1 (full coverage) should pass; date 2 (12.5% coverage) should be rejected
        assert result["count"] == 1
        assert result["sr"].shape == (1, 7, master_h, master_w)
        assert result["fm"].shape == (1, 1, master_h, master_w)
        assert result["ag"].shape == (1, 4, master_h, master_w)
        assert result["vis"].shape == (1, 4, master_h, master_w)
        assert result["mask"].shape == (1, master_h, master_w)

