"""
Tier 4: Real-World Application Scenarios Test Suite.
Validates real publication datasets against IEEE J-STARS benchmark cubes in:
- F:\\Resilio\\IMGS 890 Research\\JSTARS-Spectral-Complexity\\source_data\\
- C:\\satelliteImagery\\HLST30\\
Verifies 2022-2026 test period frames (461 total: 244 HLSL30, 210 HLSS30, 7 TANAGER),
spatial coordinate geometry, statistical benchmark distributions, and data parity contracts.
"""

import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Primary and fallback data directories
PRIMARY_REF_DIR = Path(r"F:\Resilio\IMGS 890 Research\JSTARS-Spectral-Complexity\source_data")
FALLBACK_REF_DIR = Path(r"C:\satelliteImagery\HLST30")


def get_reference_file(filename):
    """Locate reference file in primary or fallback directory."""
    primary = PRIMARY_REF_DIR / filename
    if primary.exists():
        return primary
    fallback = FALLBACK_REF_DIR / filename
    if fallback.exists():
        return fallback
    return None


@pytest.fixture(scope="module")
def harmonized_file():
    path = get_reference_file("HLST_Rochesterv2_Harmonized.h5")
    if path is None:
        pytest.skip("HLST_Rochesterv2_Harmonized.h5 not found in reference locations")
    return path


@pytest.fixture(scope="module")
def sc_metric_file():
    path = get_reference_file("HLST_Rochesterv2_Harmonized_SC_EM-7_Norm-None.h5")
    if path is None:
        pytest.skip("HLST_Rochesterv2_Harmonized_SC_EM-7_Norm-None.h5 not found in reference locations")
    return path


def test_real_01_reference_file_presence_and_byte_integrity(harmonized_file, sc_metric_file):
    """Verify physical presence and exact byte size of J-STARS publication datasets."""
    assert harmonized_file.exists(), f"Missing harmonized file at {harmonized_file}"
    assert sc_metric_file.exists(), f"Missing SC metric file at {sc_metric_file}"

    harm_size = harmonized_file.stat().st_size
    sc_size = sc_metric_file.stat().st_size

    # Benchmark sizes verified by explorer surveys
    assert harm_size == 53054508945, f"Harmonized file size mismatch: {harm_size} vs 53054508945"
    assert sc_size == 29145269378, f"SC metric file size mismatch: {sc_size} vs 29145269378"


def test_real_02_spatial_reference_and_geotransform(harmonized_file):
    """Verify Albers Equal Area projection and 30m resolution GeoTransform on benchmark data."""
    expected_gt = [-16048.67263193, 29.99751894, 0.0, 21147.15814177, 0.0, -29.99001816]
    with h5py.File(harmonized_file, "r") as f:
        ds = f["/HDFEOS/GRIDS/HLSL30/Data Fields/surface_reflectance"]
        gt = ds.attrs["GeoTransform"]
        spatial_ref = str(ds.attrs["spatial_ref"])
        assert np.allclose(gt, expected_gt, atol=1e-4)
        assert "Albers Equal Area" in spatial_ref or "Albers_Conic_Equal_Area" in spatial_ref
        assert ds.shape[2:] == (1409, 1070)


def test_real_03_test_period_scene_inventory(harmonized_file):
    """Verify exact 461 test period scenes (2022-2026): 244 HLSL30, 210 HLSS30, 7 TANAGER."""
    start_ts = datetime(2022, 1, 1, 0, 0, 0, tzinfo=timezone.utc).timestamp()
    end_ts = datetime(2026, 12, 31, 23, 59, 59, tzinfo=timezone.utc).timestamp()

    counts = {}
    with h5py.File(harmonized_file, "r") as f:
        for grid_name in ["HLSL30", "HLSS30", "TANAGER"]:
            ds = f[f"/HDFEOS/GRIDS/{grid_name}/Data Fields/surface_reflectance"]
            acq_times = ds.attrs["acquisition_time"]
            test_mask = (acq_times >= start_ts) & (acq_times <= end_ts)
            counts[grid_name] = int(np.count_nonzero(test_mask))

    assert counts["HLSL30"] == 244, f"Expected 244 HLSL30 test period frames, got {counts['HLSL30']}"
    assert counts["HLSS30"] == 210, f"Expected 210 HLSS30 test period frames, got {counts['HLSS30']}"
    assert counts["TANAGER"] == 7, f"Expected 7 TANAGER test period frames, got {counts['TANAGER']}"
    total_test_frames = sum(counts.values())
    assert total_test_frames == 461, f"Expected 461 total test period frames, got {total_test_frames}"


def test_real_04_tanager_hyperspectral_profile_and_atmospheric_nan_bands(harmonized_file, sc_metric_file):
    """Verify Tanager 426-band profile and 58 NaN atmospheric absorption bands in endmembers."""
    with h5py.File(harmonized_file, "r") as f:
        tanager_sr = f["/HDFEOS/GRIDS/TANAGER/Data Fields/surface_reflectance"]
        assert tanager_sr.shape == (7, 426, 1409, 1070)
        assert tanager_sr.dtype == np.float32
        assert tanager_sr.attrs["_FillValue"] == -9999.0
        assert len(tanager_sr.attrs["wavelengths"]) == 426

    with h5py.File(sc_metric_file, "r") as f:
        tanager_em = f["/HDFEOS/GRIDS/TANAGER/Data Fields/frame_endmembers"]
        assert tanager_em.shape == (7, 426, 7)
        # Verify 58 atmospheric bands * 7 endmembers * 7 frames = 2842 NaNs
        nan_count = int(np.count_nonzero(np.isnan(tanager_em[:])))
        expected_nans = 7 * 58 * 7
        assert nan_count == expected_nans, f"Expected {expected_nans} NaNs in Tanager endmembers, got {nan_count}"


def test_real_05_statistical_benchmark_validation(sc_metric_file):
    """Verify benchmark sliding volume and z-score statistics over 2022-2026 test period."""
    with h5py.File(sc_metric_file, "r") as f:
        harm_grp = f["/HDFEOS/GRIDS/HARMONIZED/Data Fields"]
        slide_ds = harm_grp["sliding_volume_map"]
        z_ds = harm_grp["sliding_volume_z_score"]

        assert slide_ds.shape == (844, 1409, 1070)
        assert z_ds.shape == (844, 1409, 1070)

        # Inspect subset of test period frames (e.g. frames 400 to 410)
        sample_slice = slice(400, 405)
        slide_sample = slide_ds[sample_slice, 500:600, 500:600]
        z_sample = z_ds[sample_slice, 500:600, 500:600]

        valid_slide = slide_sample[~np.isnan(slide_sample)]
        valid_z = z_sample[~np.isnan(z_sample)]

        assert len(valid_slide) > 0
        assert np.all(valid_slide >= 0.0), "Sliding volumes must be non-negative"
        if len(valid_z) > 0:
            assert np.all(valid_z > -40.0) and np.all(valid_z < 40.0), "Z-scores should fall in realistic bounds"


def test_real_06_discrete_mask_and_index_schema(harmonized_file, sc_metric_file):
    """Verify discrete mask schemas and endmember integer indices."""
    with h5py.File(harmonized_file, "r") as f:
        for grid_name, count in [("HLSL30", 441), ("HLSS30", 396)]:
            fmask = f[f"/HDFEOS/GRIDS/{grid_name}/Data Fields/Fmask"]
            assert fmask.dtype == np.uint8
            assert fmask.attrs["_FillValue"] == 255
            assert fmask.shape == (count, 1409, 1070)

    with h5py.File(sc_metric_file, "r") as f:
        for grid_name, count in [("HLSL30", 441), ("HLSS30", 396), ("TANAGER", 7)]:
            idx_ds = f[f"/HDFEOS/GRIDS/{grid_name}/Data Fields/frame_endmember_indices"]
            assert idx_ds.dtype == np.int32
            assert idx_ds.shape == (count, 7)
            # Flattened pixel index must be within 0 and 1409 * 1070 - 1
            max_idx = 1409 * 1070
            assert np.all(idx_ds[:] >= 0)
            assert np.all(idx_ds[:] < max_idx)
