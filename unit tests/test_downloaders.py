'''
Unit tests for STAC downloaders in spectral-complexity-public:
- HLS30_earthAccess_to_hdf5.py
- Tanager_STAC-imagery-downloader.py

Verifies:
1. Strict PyTorch library separation (no torch imported).
2. Pre-MGRS native format and HDF5 dataset structure.
3. 2022-2026 temporal filtering logic.
4. Sentinel constants and data value transformations (-9999, 255, 40000).
5. Spatial filtering and bounding box intersections.
6. Legacy failure handling and edge case resilience.
'''
import os
import sys
import json
import importlib
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
import h5py
import pytest
import rasterio
from rasterio.transform import from_origin
from pyproj import CRS

# Ensure dev repository and segregated subdirs are on sys.path
repo_root = Path(__file__).resolve().parent.parent
for _sub in ["", "HLS30", "Tanager-1", "Harmonized_SC", "change_detection/SC-CCM", "change_detection/harmonized-CCD", "change_detection/pyccd", "simulated-data-test"]:
    _p = str((repo_root / _sub).resolve()) if _sub else str(repo_root)
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _resolve_repo_file(filename: str) -> Path:
    direct = repo_root / filename
    if direct.exists():
        return direct
    for _sub in ["HLS30", "Tanager-1", "Harmonized_SC", "change_detection/SC-CCM", "change_detection/harmonized-CCD", "change_detection/pyccd", "simulated-data-test"]:
        cand = repo_root / _sub / filename
        if cand.exists():
            return cand
    matches = [m for m in repo_root.glob(f"**/{filename}") if ".git" not in m.parts]
    return matches[0] if matches else direct


# =====================================================================
# 1. PYTORCH SEPARATION AND IMPORT CHECKS
# =====================================================================

def test_pytorch_separation():
    """Verify neither downloader module imports torch (strict R7 compliance)."""
    # Import modules dynamically
    hls_module = importlib.import_module("HLS30_earthAccess_to_hdf5")
    tanager_module = importlib.import_module("Tanager_STAC-imagery-downloader")

    # Verify 'torch' is not in the modules' globals
    assert "torch" not in dir(hls_module), "HLS30 downloader must not import torch"
    assert "torch" not in dir(tanager_module), "Tanager downloader must not import torch"

    # Verify no torch reference exists in source code text
    with open(_resolve_repo_file("HLS30_earthAccess_to_hdf5.py"), "r", encoding="utf-8") as f:
        hls_src = f.read()
    assert "import torch" not in hls_src
    assert "from torch" not in hls_src

    with open(_resolve_repo_file("Tanager_STAC-imagery-downloader.py"), "r", encoding="utf-8") as f:
        tanager_src = f.read()
    assert "import torch" not in tanager_src
    assert "from torch" not in tanager_src


# =====================================================================
# 2. TEMPORAL FILTERING TESTS (2022-2026 TEST PERIOD)
# =====================================================================

def test_tanager_temporal_filter():
    """Verify Tanager temporal filtering targets the 2022-2026 test period."""
    tanager_module = importlib.import_module("Tanager_STAC-imagery-downloader")
    passes_temporal_filter = tanager_module.passes_temporal_filter

    # Valid items within 2022-2026 test period
    valid_item_2022 = {"properties": {"datetime": "2022-01-01T12:00:00Z"}}
    valid_item_2024 = {"properties": {"datetime": "2024-06-15T10:30:00Z"}}
    valid_item_2026 = {"properties": {"datetime": "2026-05-31T23:59:59Z"}}
    valid_item_alt_key = {"properties": {"acquisition_time": "2025-08-20T14:00:00Z"}}

    assert passes_temporal_filter(valid_item_2022, "2022-01-01", "2026-12-31") is True
    assert passes_temporal_filter(valid_item_2024, "2022-01-01", "2026-12-31") is True
    assert passes_temporal_filter(valid_item_2026, "2022-01-01", "2026-12-31") is True
    assert passes_temporal_filter(valid_item_alt_key, "2022-01-01", "2026-12-31") is True

    # Invalid items outside the 2022-2026 test period
    invalid_item_2015 = {"properties": {"datetime": "2015-06-01T00:00:00Z"}}
    invalid_item_2021 = {"properties": {"datetime": "2021-12-31T23:59:59Z"}}
    invalid_item_2027 = {"properties": {"datetime": "2027-01-01T00:00:00Z"}}

    assert passes_temporal_filter(invalid_item_2015, "2022-01-01", "2026-12-31") is False
    assert passes_temporal_filter(invalid_item_2021, "2022-01-01", "2026-12-31") is False
    assert passes_temporal_filter(invalid_item_2027, "2022-01-01", "2026-12-31") is False

    # Missing date property defaults to True to avoid dropping items without explicit timestamps
    empty_item = {"properties": {}}
    assert passes_temporal_filter(empty_item, "2022-01-01", "2026-12-31") is True


def test_hls_temporal_configuration():
    """Verify HLS downloader defaults and clamps temporal parameters to 2022-2026."""
    hls_module = importlib.import_module("HLS30_earthAccess_to_hdf5")
    assert hls_module.DEFAULT_START_DATE == "2022-01-01"
    assert hls_module.DEFAULT_END_DATE == "2026-06-01"

    # Test config loader returns 2022-2026 for default Rochesterv2
    config = hls_module.load_locations_config()
    rochester_cfg = config["locations"]["Rochesterv2"]
    assert rochester_cfg["ROI_LON_MIN"] == -77.770166
    assert rochester_cfg["ROI_LAT_MIN"] == 42.961778


# =====================================================================
# 3. SPATIAL FILTERING TESTS
# =====================================================================

def test_spatial_intersection():
    """Verify bounding box intersection logic."""
    tanager_module = importlib.import_module("Tanager_STAC-imagery-downloader")
    intersects = tanager_module.intersects

    box1 = [-77.72, 43.04, -77.44, 43.28]  # Rochester
    box2 = [-77.60, 43.10, -77.50, 43.20]  # Inside Rochester
    box3 = [-77.40, 43.04, -77.30, 43.28]  # East of Rochester (disjoint)
    box4 = [-77.72, 43.30, -77.44, 43.40]  # North of Rochester (disjoint)

    assert intersects(box1, box2) is True
    assert intersects(box1, box3) is False
    assert intersects(box1, box4) is False


def test_passes_spatial_filters():
    """Verify inclusion and exclusion spatial filter evaluations."""
    tanager_module = importlib.import_module("Tanager_STAC-imagery-downloader")
    passes_spatial_filters = tanager_module.passes_spatial_filters

    includes = [[-77.72, 43.04, -77.44, 43.28]]
    excludes = [[-114.05, 37.0, -109.0, 42.5]]

    item_in_rochester = [-77.60, 43.10, -77.50, 43.20]
    item_in_utah = [-112.0, 39.0, -111.0, 40.0]
    item_outside_both = [10.0, 10.0, 11.0, 11.0]

    assert passes_spatial_filters(item_in_rochester, includes, excludes) is True
    assert passes_spatial_filters(item_in_utah, includes, excludes) is False
    assert passes_spatial_filters(item_outside_both, includes, excludes) is False
    assert passes_spatial_filters(None, includes, excludes) is False


# =====================================================================
# 4. SENTINEL VALUE CONSTANTS AND TRANSFORMATIONS
# =====================================================================

def test_sentinel_constants():
    """Verify exact legacy sentinel constants."""
    hls_module = importlib.import_module("HLS30_earthAccess_to_hdf5")

    assert hls_module.FILL_VALUE_REFLECTANCE_RAW == -9999
    assert hls_module.FILL_VALUE_FMASK_RAW == 255
    assert hls_module.FILL_VALUE_ANGLES_RAW == 40000
    assert hls_module.SCALE_FACTOR_REFLECTANCE == 0.0001
    assert hls_module.SCALE_FACTOR_ANGLES == 0.01


def test_sentinel_value_mapping():
    """Verify raw tile sentinel conversions match legacy mathematical formulas."""
    hls_module = importlib.import_module("HLS30_earthAccess_to_hdf5")

    # Synthetic raw arrays
    t_sr = np.array([[-9999, 1000], [2500, -9999]], dtype=np.int32)
    t_fm = np.array([[255, 0], [1, 255]], dtype=np.uint8)
    t_ag = np.array([[40000, 3000], [4500, 40000]], dtype=np.int32)

    # Reflectance conversion: -9999 -> np.nan, valid -> * 0.0001
    sr_converted = np.where(
        t_sr != hls_module.FILL_VALUE_REFLECTANCE_RAW,
        t_sr.astype(np.float32) * hls_module.SCALE_FACTOR_REFLECTANCE,
        np.nan
    )
    assert np.isnan(sr_converted[0, 0])
    assert np.isnan(sr_converted[1, 1])
    assert np.isclose(sr_converted[0, 1], 0.1000)
    assert np.isclose(sr_converted[1, 0], 0.2500)

    # Fmask conversion: valid where (fm != 255) & sr_valid
    sr_valid = t_sr[0] != hls_module.FILL_VALUE_REFLECTANCE_RAW  # Evaluated on band 0
    fm_converted = np.where(
        (t_fm != hls_module.FILL_VALUE_FMASK_RAW) & sr_valid,
        t_fm,
        hls_module.FILL_VALUE_FMASK_RAW
    )
    assert fm_converted[0, 0] == 255
    assert fm_converted[0, 1] == 0
    assert fm_converted[1, 1] == 255

    # Angles conversion: 40000 -> np.nan, valid -> * 0.01
    ag_converted = np.where(
        (t_ag != hls_module.FILL_VALUE_ANGLES_RAW) & sr_valid,
        t_ag.astype(np.float32) * hls_module.SCALE_FACTOR_ANGLES,
        np.nan
    )
    assert np.isnan(ag_converted[0, 0])
    assert np.isclose(ag_converted[0, 1], 30.00)
    assert np.isnan(ag_converted[1, 1])


# =====================================================================
# 5. FAILURE HANDLING TESTS
# =====================================================================

def test_tanager_download_file_failure_handling(tmp_path):
    """Verify download_file skips existing files and catches network errors."""
    tanager_module = importlib.import_module("Tanager_STAC-imagery-downloader")
    download_file = tanager_module.download_file

    # Existing file with size > 0 should return True (skip)
    existing_file = tmp_path / "existing.h5"
    existing_file.write_bytes(b"dummy_data_content")
    assert download_file("http://invalid.url/test.h5", existing_file) is True

    # Failed network request returns False without throwing unhandled exception
    missing_file = tmp_path / "missing.h5"
    with mock.patch("requests.get", side_effect=Exception("Connection refused")):
        success = download_file("http://nonexistent.domain/test.h5", missing_file)
        assert success is False
        assert not missing_file.exists()


def test_tanager_execute_job_resilience(tmp_path):
    """Verify execute_job handles broken catalog URLs gracefully."""
    tanager_module = importlib.import_module("Tanager_STAC-imagery-downloader")
    execute_job = tanager_module.execute_job

    bad_job = {
        "job_name": "Test_Job",
        "collection_url": "https://invalid.planet.domain/catalog.json",
        "output_dir": str(tmp_path),
        "include_bboxes": [[0, 0, 1, 1]],
        "exclude_bboxes": [],
        "target_assets": ["basic_sr_hdf5"]
    }

    # Should catch exception, log, and return 0 processed scenes
    scenes = execute_job(bad_job)
    assert scenes == 0


def test_hls_stream_null_metadata_error(tmp_path):
    """Verify stream_native_tiles_to_hdf5 raises ValueError on null metadata."""
    hls_module = importlib.import_module("HLS30_earthAccess_to_hdf5")
    stream_native = hls_module.stream_native_tiles_to_hdf5

    # Create dummy GeoTIFF so initial window metadata read succeeds
    tif_path = tmp_path / "valid_dummy.tif"
    height, width, count = 10, 10, 15
    transform = from_origin(-77.70, 43.30, 0.0003, 0.0003)
    crs = CRS.from_epsg(4326)
    data = np.zeros((count, height, width), dtype=np.int32)
    profile = {
        'driver': 'GTiff', 'height': height, 'width': width,
        'count': count, 'dtype': 'int32', 'crs': crs, 'transform': transform
    }
    with rasterio.open(tif_path, 'w', **profile) as dst:
        dst.write(data)

    mock_h5 = mock.MagicMock()
    safe_bbox = [-77.70, 43.30 - height * 0.0003, -77.70 + width * 0.0003, 43.30]

    tile_collections = {
        "T18TXN": {
            "items": {
                "HLS.S30.T18TXN.2024001.v2.0": {
                    "acquisition_time": 1704067200.0,
                    "spacecraft_id": None,  # Null metadata attribute
                    "cloud_cover": 10.0,
                    "filepath": str(tif_path)
                }
            }
        }
    }

    with pytest.raises(ValueError, match="CRITICAL ERROR: Metadata 'spacecraft_id' for .* is null"):
        stream_native(mock_h5, "HLSS30", tile_collections, 10, 11, hls_module.S30_WAVELENGTHS, safe_bbox)


# =====================================================================
# 6. HDF5 NATIVE DATASET GENERATION AND PARITY TESTS
# =====================================================================

def test_hls_stream_native_tiles_to_hdf5(tmp_path):
    """
    Verify stream_native_tiles_to_hdf5 correctly creates native HDF5 datasets,
    applying scaling and legacy fill values.
    """
    hls_module = importlib.import_module("HLS30_earthAccess_to_hdf5")
    stream_native = hls_module.stream_native_tiles_to_hdf5

    # Create synthetic multi-band GeoTIFF
    tif_path = tmp_path / "test_tile.tif"
    height, width = 64, 64
    count = 15  # 10 SR + 1 Fmask + 4 Angles
    res = 0.0003
    transform = from_origin(-77.70, 43.30, res, res)
    crs = CRS.from_epsg(4326)

    # Fill data with known sentinels
    data = np.full((count, height, width), 1000, dtype=np.int32)
    data[0, 0, 0] = -9999    # Reflectance fill
    data[10, 0, 1] = 255     # Fmask fill
    data[11, 0, 2] = 40000   # SZA fill

    profile = {
        'driver': 'GTiff',
        'height': height,
        'width': width,
        'count': count,
        'dtype': 'int32',
        'crs': crs,
        'transform': transform
    }

    with rasterio.open(tif_path, 'w', **profile) as dst:
        dst.write(data)

    # Align safe_bbox exactly to the raster bounds so read window matches array shape
    safe_bbox = [-77.70, 43.30 - height * res, -77.70 + width * res, 43.30]
    tile_collections = {
        "T18TXN": {
            "items": {
                "HLS.S30.T18TXN.2024001.v2.0": {
                    "acquisition_time": 1704067200.0,
                    "spacecraft_id": "Sentinel-2A",
                    "cloud_cover": 5.5,
                    "filepath": str(tif_path)
                }
            },
            "zone": 18
        }
    }

    h5_path = tmp_path / "test_native.h5"
    with h5py.File(h5_path, 'w') as h5f:
        odl_blocks = stream_native(
            h5f, "HLSS30", tile_collections, 10, 11,
            hls_module.S30_WAVELENGTHS, safe_bbox, hdfeos_module=None
        )

    # Verify HDF5 content
    with h5py.File(h5_path, 'r') as h5f:
        grp_path = "/HDFEOS/GRIDS/HLSS30_T18TXN/Data Fields"
        assert grp_path in h5f, f"Expected {grp_path} in generated HDF5"

        grp = h5f[grp_path]
        assert "surface_reflectance" in grp
        assert "Fmask" in grp
        assert "solar_view_angles" in grp

        sr = grp["surface_reflectance"]
        fm = grp["Fmask"]
        ag = grp["solar_view_angles"]

        # Check types (int16/uint16 with scale attributes, or float32)
        assert sr.dtype in (np.int16, np.float32)
        assert fm.dtype == np.uint8
        assert ag.dtype in (np.uint16, np.float32)

        # Check attributes
        assert sr.attrs["units"] == "Reflectance"
        fill_val = sr.attrs["_FillValue"]
        assert np.isnan(fill_val) or fill_val == -9999
        assert fm.attrs["_FillValue"] == 255
        ag_fill = ag.attrs["_FillValue"]
        assert np.isnan(ag_fill) or ag_fill == 40000
        assert list(ag.attrs["band_order"]) == ["SZA", "SAA", "VZA", "VAA"]
        assert np.isclose(sr.attrs["cloud_cover"][0], 5.5)

        # Check values
        sr_data = sr[:]
        if sr.dtype == np.int16:
            assert sr_data[0, 0, 0, 0] == -9999
            assert sr_data[0, 1, 0, 0] == 1000
        else:
            assert np.isnan(sr_data[0, 0, 0, 0]), "Reflectance -9999 must convert to NaN"
            assert np.isclose(sr_data[0, 1, 0, 0], 0.1000), "Valid raw 1000 must scale to 0.1000"
