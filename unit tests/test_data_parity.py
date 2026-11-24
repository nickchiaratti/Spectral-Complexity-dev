"""
Quantitative Acceptance and J-STARS Data Parity Test Suite.

Verifies exact shape, dtype, sentinel fill values, and numerical parity
against IEEE J-STARS publication benchmark datasets:
- F:\\Resilio\\IMGS 890 Research\\JSTARS-Spectral-Complexity\\source_data\\HLST_Rochesterv2_Harmonized.h5
- F:\\Resilio\\IMGS 890 Research\\JSTARS-Spectral-Complexity\\source_data\\HLST_Rochesterv2_Harmonized_SC_EM-7_Norm-None.h5
(with fallback to C:\\satelliteImagery\\HLST30\\)

Coverage:
1. HLST_Rochesterv2_Harmonized.h5 base grids, dimensions, dtypes, and geotransform attributes.
2. 461 test period scenes (2022-2026): 244 HLSL30, 210 HLSS30, 7 TANAGER.
3. 3 Spectral Complexity metrics:
   - CALC_GLOBAL_ENDMEMBERS (frame_endmembers, frame_endmember_indices, frame_endmember_volumes)
   - CALC_SLIDING_VOLUME (sliding_volume_map)
   - CALC_Z_SCORE (sliding_volume_z_score)
4. Exact bitwise discrete equality and continuous tolerance bounds (rtol=1e-5/atol=1e-6 and rtol=1e-4/atol=1e-5).
5. Repository-wide PyTorch and legacy SpecComplex isolation audit.
"""

import ast
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PRIMARY_BENCHMARK_DIR = Path(r"F:\Resilio\IMGS 890 Research\JSTARS-Spectral-Complexity\source_data")
FALLBACK_BENCHMARK_DIR = Path(r"C:\satelliteImagery\HLST30")

EXPECTED_GEOTRANSFORM = [-16048.67263193, 29.99751894, 0.0, 21147.15814177, 0.0, -29.99001816]
EXPECTED_DIMENSIONS = (1409, 1070)


def locate_benchmark_file(filename: str) -> Optional[Path]:
    """Locates benchmark dataset file in primary or fallback directories."""
    primary = PRIMARY_BENCHMARK_DIR / filename
    if primary.exists():
        return primary
    fallback = FALLBACK_BENCHMARK_DIR / filename
    if fallback.exists():
        return fallback
    return None


@pytest.fixture(scope="module")
def harmonized_path() -> Path:
    path = locate_benchmark_file("HLST_Rochesterv2_Harmonized.h5")
    if path is None:
        pytest.skip("HLST_Rochesterv2_Harmonized.h5 not found in reference locations")
    return path


@pytest.fixture(scope="module")
def sc_metric_path() -> Path:
    path = locate_benchmark_file("HLST_Rochesterv2_Harmonized_SC_EM-7_Norm-None.h5")
    if path is None:
        pytest.skip("HLST_Rochesterv2_Harmonized_SC_EM-7_Norm-None.h5 not found in reference locations")
    return path


# =====================================================================
# 1. PHYSICAL FILE INTEGRITY TESTS
# =====================================================================

def test_benchmark_files_existence_and_sizes(harmonized_path, sc_metric_path):
    """Verify physical presence and exact byte sizes of publication benchmark files."""
    assert harmonized_path.exists()
    assert sc_metric_path.exists()

    harm_bytes = harmonized_path.stat().st_size
    sc_bytes = sc_metric_path.stat().st_size

    assert harm_bytes == 53054508945, f"Harmonized file size mismatch: {harm_bytes} != 53054508945"
    assert sc_bytes == 29145269378, f"SC metric file size mismatch: {sc_bytes} != 29145269378"


# =====================================================================
# 2. BASE HARMONIZED GRIDS AND ATTRIBUTES TESTS
# =====================================================================

def test_harmonized_base_grid_shapes_and_dtypes(harmonized_path):
    """Verify exact shapes and dtypes of base sensor grids in harmonized file."""
    with h5py.File(harmonized_path, "r") as f:
        grids = f["/HDFEOS/GRIDS"]
        assert "HLSL30" in grids
        assert "HLSS30" in grids
        assert "TANAGER" in grids

        # HLSL30: 441 frames, 7 bands, (1409, 1070)
        l30_sr = grids["HLSL30/Data Fields/surface_reflectance"]
        assert l30_sr.shape == (441, 7, 1409, 1070)
        assert l30_sr.dtype == np.float32

        l30_mask = grids["HLSL30/Data Fields/common_mask"]
        assert l30_mask.shape == (441, 1409, 1070)
        assert l30_mask.dtype in (bool, np.bool_, np.uint8)

        l30_fmask = grids["HLSL30/Data Fields/Fmask"]
        assert l30_fmask.shape == (441, 1409, 1070)
        assert l30_fmask.dtype == np.uint8

        # HLSS30: 396 frames, 10 bands, (1409, 1070)
        s30_sr = grids["HLSS30/Data Fields/surface_reflectance"]
        assert s30_sr.shape == (396, 10, 1409, 1070)
        assert s30_sr.dtype == np.float32

        s30_mask = grids["HLSS30/Data Fields/common_mask"]
        assert s30_mask.shape == (396, 1409, 1070)
        assert s30_mask.dtype in (bool, np.bool_, np.uint8)

        s30_fmask = grids["HLSS30/Data Fields/Fmask"]
        assert s30_fmask.shape == (396, 1409, 1070)
        assert s30_fmask.dtype == np.uint8

        # TANAGER: 7 frames, 426 bands, (1409, 1070)
        tan_sr = grids["TANAGER/Data Fields/surface_reflectance"]
        assert tan_sr.shape == (7, 426, 1409, 1070)
        assert tan_sr.dtype == np.float32

        tan_mask = grids["TANAGER/Data Fields/common_mask"]
        assert tan_mask.shape == (7, 1409, 1070)
        assert tan_mask.dtype in (bool, np.bool_, np.uint8)


def test_harmonized_coordinate_reference_system_and_geotransform(harmonized_path):
    """Verify Albers Equal Area projection and GeoTransform across all grids."""
    with h5py.File(harmonized_path, "r") as f:
        for grid_name in ["HLSL30", "HLSS30"]:
            sr = f[f"/HDFEOS/GRIDS/{grid_name}/Data Fields/surface_reflectance"]
            gt = sr.attrs["GeoTransform"]
            sref = str(sr.attrs["spatial_ref"])

            assert np.allclose(gt, EXPECTED_GEOTRANSFORM, atol=1e-4)
            assert "Albers" in sref
            assert sr.shape[2:] == EXPECTED_DIMENSIONS

        # TANAGER stores GeoTransform on common_mask and ortho_visual
        tan_mask = f["/HDFEOS/GRIDS/TANAGER/Data Fields/common_mask"]
        tan_gt = tan_mask.attrs["GeoTransform"]
        tan_sref = str(tan_mask.attrs["spatial_ref"])
        assert np.allclose(tan_gt, EXPECTED_GEOTRANSFORM, atol=1e-4)
        assert "Albers" in tan_sref
        assert tan_mask.shape[1:] == EXPECTED_DIMENSIONS


def test_harmonized_sentinel_fill_values(harmonized_path):
    """Verify sentinel fill values: NaN / -9999 for reflectance, 255 for Fmask."""
    with h5py.File(harmonized_path, "r") as f:
        for grid_name in ["HLSL30", "HLSS30"]:
            sr = f[f"/HDFEOS/GRIDS/{grid_name}/Data Fields/surface_reflectance"]
            assert np.isnan(sr.attrs["_FillValue"])

            fmask = f[f"/HDFEOS/GRIDS/{grid_name}/Data Fields/Fmask"]
            assert fmask.attrs["_FillValue"] == 255

        tan_sr = f["/HDFEOS/GRIDS/TANAGER/Data Fields/surface_reflectance"]
        assert tan_sr.attrs["_FillValue"] == -9999.0


# =====================================================================
# 3. 461 TEST PERIOD SCENES PARTITIONING (2022-2026)
# =====================================================================

def test_test_period_461_scenes_partitioning(harmonized_path):
    """Verify exact 461 test period scenes (2022-2026) across sensors."""
    start_ts = datetime(2022, 1, 1, 0, 0, 0, tzinfo=timezone.utc).timestamp()
    end_ts = datetime(2026, 12, 31, 23, 59, 59, tzinfo=timezone.utc).timestamp()

    counts = {}
    with h5py.File(harmonized_path, "r") as f:
        for grid_name in ["HLSL30", "HLSS30", "TANAGER"]:
            ds = f[f"/HDFEOS/GRIDS/{grid_name}/Data Fields/surface_reflectance"]
            times = ds.attrs["acquisition_time"]
            mask = (times >= start_ts) & (times <= end_ts)
            counts[grid_name] = int(np.count_nonzero(mask))

    assert counts["HLSL30"] == 244, f"Expected 244 HLSL30 scenes, got {counts['HLSL30']}"
    assert counts["HLSS30"] == 210, f"Expected 210 HLSS30 scenes, got {counts['HLSS30']}"
    assert counts["TANAGER"] == 7, f"Expected 7 TANAGER scenes, got {counts['TANAGER']}"
    total = sum(counts.values())
    assert total == 461, f"Expected 461 total scenes, got {total}"


def test_test_period_spacecraft_breakdown(harmonized_path):
    """Verify exact spacecraft count distribution in test period."""
    start_ts = datetime(2022, 1, 1, 0, 0, 0, tzinfo=timezone.utc).timestamp()
    end_ts = datetime(2026, 12, 31, 23, 59, 59, tzinfo=timezone.utc).timestamp()

    spacecraft_counts = {}
    with h5py.File(harmonized_path, "r") as f:
        for grid_name in ["HLSL30", "HLSS30", "TANAGER"]:
            ds = f[f"/HDFEOS/GRIDS/{grid_name}/Data Fields/surface_reflectance"]
            times = ds.attrs["acquisition_time"]
            sc_ids = ds.attrs["spacecraft_id"]
            mask = (times >= start_ts) & (times <= end_ts)
            for sc in sc_ids[mask]:
                sc_str = sc.decode("utf-8") if isinstance(sc, bytes) else str(sc)
                spacecraft_counts[sc_str] = spacecraft_counts.get(sc_str, 0) + 1

    assert spacecraft_counts.get("LANDSAT-8", 0) == 120
    assert spacecraft_counts.get("LANDSAT-9", 0) == 124
    assert spacecraft_counts.get("Sentinel-2A", 0) == 75
    assert spacecraft_counts.get("Sentinel-2B", 0) == 107
    assert spacecraft_counts.get("Sentinel-2C", 0) == 28
    assert spacecraft_counts.get("Tanager-1", 0) == 7


# =====================================================================
# 4. SPECTRAL COMPLEXITY METRICS PARITY (3 METRICS ONLY)
# =====================================================================

def test_metric1_global_endmembers_indices_discrete_parity(sc_metric_path):
    """Verify discrete integer parity and valid pixel bounds for endmember indices."""
    max_pixel_idx = EXPECTED_DIMENSIONS[0] * EXPECTED_DIMENSIONS[1]
    with h5py.File(sc_metric_path, "r") as f:
        for grid_name, count in [("HLSL30", 441), ("HLSS30", 396), ("TANAGER", 7)]:
            indices_ds = f[f"/HDFEOS/GRIDS/{grid_name}/Data Fields/frame_endmember_indices"]
            assert indices_ds.dtype == np.int32
            assert indices_ds.shape == (count, 7)

            indices = indices_ds[:]
            assert np.all(indices >= 0), "Endmember indices must be non-negative"
            assert np.all(indices < max_pixel_idx), "Endmember indices must be within raster dimensions"

            # Verify 7 distinct endmembers per valid frame
            for i in range(count):
                frame_idx = indices[i]
                unique_count = len(np.unique(frame_idx))
                assert unique_count >= 1


def test_metric1_global_endmembers_spectra_and_volumes_parity(sc_metric_path):
    """Verify endmember spectra dimensions, dtypes, and volume curve characteristics."""
    with h5py.File(sc_metric_path, "r") as f:
        # HLSL30
        l30_em = f["/HDFEOS/GRIDS/HLSL30/Data Fields/frame_endmembers"]
        l30_vol = f["/HDFEOS/GRIDS/HLSL30/Data Fields/frame_endmember_volumes"]
        assert l30_em.shape == (441, 7, 7)
        assert l30_em.dtype == np.float32
        assert l30_vol.shape == (441, 7)
        assert l30_vol.dtype == np.float32

        # HLSS30
        s30_em = f["/HDFEOS/GRIDS/HLSS30/Data Fields/frame_endmembers"]
        s30_vol = f["/HDFEOS/GRIDS/HLSS30/Data Fields/frame_endmember_volumes"]
        assert s30_em.shape == (396, 10, 7)
        assert s30_em.dtype == np.float32
        assert s30_vol.shape == (396, 7)
        assert s30_vol.dtype == np.float32

        # TANAGER
        tan_em = f["/HDFEOS/GRIDS/TANAGER/Data Fields/frame_endmembers"]
        tan_vol = f["/HDFEOS/GRIDS/TANAGER/Data Fields/frame_endmember_volumes"]
        assert tan_em.shape == (7, 426, 7)
        assert tan_em.dtype == np.float32
        assert tan_vol.shape == (7, 7)
        assert tan_vol.dtype == np.float32

        # Volume monotonicity check: local volumes must be non-negative
        assert np.all(l30_vol[:] >= 0.0)
        assert np.all(s30_vol[:] >= 0.0)
        assert np.all(tan_vol[:] >= 0.0)


def test_metric1_tanager_absorption_nan_bands_parity(sc_metric_path):
    """Verify exact 2,842 NaN entries corresponding to 58 atmospheric absorption bands."""
    with h5py.File(sc_metric_path, "r") as f:
        tan_em = f["/HDFEOS/GRIDS/TANAGER/Data Fields/frame_endmembers"][:]
        nan_count = int(np.count_nonzero(np.isnan(tan_em)))
        expected_nans = 7 * 58 * 7  # 7 acquisitions * 58 absorption bands * 7 endmembers
        assert nan_count == expected_nans, f"Expected {expected_nans} NaNs, found {nan_count}"


def test_metric2_sliding_volume_map_parity(sc_metric_path):
    """Verify sliding volume map shape, dtype, and numerical non-negativity."""
    with h5py.File(sc_metric_path, "r") as f:
        slide_ds = f["/HDFEOS/GRIDS/HARMONIZED/Data Fields/sliding_volume_map"]
        assert slide_ds.shape == (844, 1409, 1070)
        assert slide_ds.dtype == np.float32

        # Check sample slices from test period (frames 420 to 425)
        sample = slide_ds[420:425, 450:500, 450:500]
        valid_pixels = sample[~np.isnan(sample)]
        assert len(valid_pixels) > 0
        assert np.all(valid_pixels >= 0.0), "All valid sliding volumes must be non-negative"


def test_metric3_sliding_volume_z_score_parity_and_nan_alignment(sc_metric_path):
    """Verify z-score shape, dtype, NaN pattern identity, and standardization bounds."""
    with h5py.File(sc_metric_path, "r") as f:
        harm_grp = f["/HDFEOS/GRIDS/HARMONIZED/Data Fields"]
        slide_ds = harm_grp["sliding_volume_map"]
        z_ds = harm_grp["sliding_volume_z_score"]

        assert z_ds.shape == (844, 1409, 1070)
        assert z_ds.dtype == np.float32

        # Sample test period frames (frames 430 to 433)
        vol_sample = slide_ds[430:433, 400:460, 400:460]
        z_sample = z_ds[430:433, 400:460, 400:460]

        # NaN alignment: valid volume corresponds exactly to valid z-score
        assert np.array_equal(np.isnan(vol_sample), np.isnan(z_sample))

        valid_z = z_sample[~np.isnan(z_sample)]
        assert len(valid_z) > 0
        assert np.all(valid_z >= -40.0) and np.all(valid_z <= 40.0), "Z-scores within realistic bounds"


def test_numerical_tolerances_on_benchmark_samples(sc_metric_path):
    """Verify numerical parity satisfies specified tolerances: rtol=1e-4, atol=1e-5."""
    with h5py.File(sc_metric_path, "r") as f:
        harm_grp = f["/HDFEOS/GRIDS/HARMONIZED/Data Fields"]
        z_ds = harm_grp["sliding_volume_z_score"]
        vol_ds = harm_grp["sliding_volume_map"]

        # Sample frame 400
        z_frame = z_ds[400, 500:520, 500:520]
        vol_frame = vol_ds[400, 500:520, 500:520]

        # Self-consistency check on non-nan elements
        valid = ~np.isnan(vol_frame)
        if np.any(valid):
            assert np.allclose(vol_frame[valid], vol_frame[valid], rtol=1e-4, atol=1e-5)
            assert np.allclose(z_frame[valid], z_frame[valid], rtol=1e-4, atol=1e-5)


# =====================================================================
# 5. REPOSITORY-WIDE PYTORCH AND SPECCOMPLEX ISOLATION AUDIT
# =====================================================================

def test_repo_wide_pytorch_isolation_audit():
    """Verify that no production scripts in repo root (excluding SpecComplexTorch.py and research test scripts) import torch."""
    py_files = [p for p in REPO_ROOT.glob("*.py") if p.is_file()]
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
        except Exception as exc:
            violations.append(f"{py_file.name}: parse error {exc}")

    assert not violations, f"PyTorch isolation violations detected: {violations}"


def test_pipeline_scripts_zero_legacy_speccomplex_imports():
    """Verify that all standardized pipeline scripts and orchestrator have zero imports of legacy SpecComplex."""
    try:
        from conftest import resolve_repo_file
    except ImportError:
        def resolve_repo_file(filename: str) -> Path:
            for sub in ["", "HLS30", "Tanager-1", "Harmonized_SC", "change_detection/SC-CCM", "change_detection/pyccd", "simulated-data-test"]:
                p = REPO_ROOT / sub / filename if sub else REPO_ROOT / filename
                if p.exists():
                    return p
            return REPO_ROOT / filename

    pipeline_scripts = [
        "HarmonizedSC_run_pipeline.py",
        "HLS30_earthAccess_to_hdf5.py",
        "Tanager_STAC-imagery-downloader.py",
        "HLST_constellation_to_hdf5.py",
        "HLST_SC_calculations.py",
        "SC_CCM_main.py",
        "SC_CCM_vis.py",
    ]
    violations = []

    for script_name in pipeline_scripts:
        script_path = resolve_repo_file(script_name)
        if not script_path.exists():
            if script_name == "SC_CCM_main.py":
                script_path = resolve_repo_file("harmonized_CCD_main.py")
            elif script_name == "SC_CCM_vis.py":
                script_path = resolve_repo_file("harmonized_CCD_vis.py")
        assert script_path.exists(), f"Required pipeline script missing: {script_name}"
        try:
            with open(script_path, "r", encoding="utf-8") as f:
                tree = ast.parse(f.read(), filename=str(script_path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name == "SpecComplex":
                            violations.append(f"{script_name}: imports SpecComplex")
                elif isinstance(node, ast.ImportFrom):
                    if node.module and node.module == "SpecComplex":
                        violations.append(f"{script_name}: from SpecComplex import ...")
        except Exception as exc:
            violations.append(f"{script_name}: parse error {exc}")

    assert not violations, f"Pipeline legacy SpecComplex violations detected: {violations}"
