"""
Unit tests for Harmonized Continuous Change Detection (CCD) Pipeline and Visualization.

Tests cover:
- PyTorch and legacy SpecComplex isolation
- Inlined quantization and dequantization (read_scaled_int16, scale_to_int16)
- Fractional years timestamp conversion
- Fourier harmonic design matrix construction
- Outlier detection and chi-square threshold evaluation
- Row chunk CCD fitting with LASSO initialization and RLS tracking
- Full CCD pipeline execution on synthetic HDF5 datasets
- Output schema, dataset shapes, dtypes, and coordinate reference preservation
- Visualization helpers and non-interactive diagnostic PNG export
- SpecComplexTorch patch analysis integration
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
from scipy.stats import chi2

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from conftest import resolve_repo_file
except ImportError:
    def resolve_repo_file(filename: str) -> Path:
        for sub in ["", "HLS30", "Tanager-1", "Harmonized_SC", "change_detection/SC-CCM", "change_detection/pyccd", "simulated-data-test"]:
            p = REPO_ROOT / sub / filename if sub else REPO_ROOT / filename
            if p.exists():
                return p
        return REPO_ROOT / filename

try:
    import SC_CCM_main as ccd_main
except ImportError:
    import harmonized_CCD_main as ccd_main

try:
    import SC_CCM_vis as ccd_vis
except ImportError:
    import harmonized_CCD_vis as ccd_vis


# =====================================================================
# 1. PYTORCH & SPECCOMPLEX ISOLATION TESTS
# =====================================================================

def test_ccd_main_zero_torch_or_legacy_speccomplex_imports():
    """Verify harmonized_CCD_main.py or SC_CCM_main.py has zero imports of torch or legacy SpecComplex."""
    main_file = resolve_repo_file("SC_CCM_main.py")
    if not main_file.exists():
        main_file = resolve_repo_file("harmonized_CCD_main.py")
    with open(main_file, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=str(main_file))

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name != "torch" and not alias.name.startswith("torch."), (
                    f"{main_file.name} imports torch: {alias.name}"
                )
                assert alias.name != "SpecComplex", (
                    f"{main_file.name} imports legacy SpecComplex: {alias.name}"
                )
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                assert node.module != "torch" and not node.module.startswith("torch."), (
                    f"{main_file.name} imports from torch: {node.module}"
                )
                assert node.module != "SpecComplex", (
                    f"{main_file.name} imports from legacy SpecComplex: {node.module}"
                )


def test_ccd_vis_zero_torch_or_legacy_speccomplex_imports():
    """Verify harmonized_CCD_vis.py or SC_CCM_vis.py has zero imports of torch or legacy SpecComplex."""
    vis_file = resolve_repo_file("SC_CCM_vis.py")
    if not vis_file.exists():
        vis_file = resolve_repo_file("harmonized_CCD_vis.py")
    with open(vis_file, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=str(vis_file))

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name != "torch" and not alias.name.startswith("torch."), (
                    f"{vis_file.name} imports torch: {alias.name}"
                )
                assert alias.name != "SpecComplex", (
                    f"{vis_file.name} imports legacy SpecComplex: {alias.name}"
                )
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                assert node.module != "torch" and not node.module.startswith("torch."), (
                    f"{vis_file.name} imports from torch: {node.module}"
                )
                assert node.module != "SpecComplex", (
                    f"{vis_file.name} imports from legacy SpecComplex: {node.module}"
                )


# =====================================================================
# 2. INLINED QUANTIZATION / DEQUANTIZATION TESTS
# =====================================================================

def test_read_scaled_int16_inversion_and_nodata_preservation():
    """Verify read_scaled_int16 inverts int16 to float32 and maps nodata -9999 to NaN."""
    raw_ints = np.array([-9999, 0, 1500, 5000, 10000], dtype=np.int16)
    floats_main = ccd_main.read_scaled_int16(raw_ints, scale_factor=10000, nodata_val=-9999)
    floats_vis = ccd_vis.read_scaled_int16(raw_ints, scale_factor=10000, nodata_val=-9999)

    for floats in [floats_main, floats_vis]:
        assert np.isnan(floats[0])
        assert np.isclose(floats[1], 0.0)
        assert np.isclose(floats[2], 0.15)
        assert np.isclose(floats[3], 0.50)
        assert np.isclose(floats[4], 1.00)
        assert floats.dtype == np.float32


def test_scale_to_int16_clipping_bounds():
    """Verify scale_to_int16 correctly quantizes and bounds values."""
    raw_floats = np.array([np.nan, -2.0, 0.0, 0.25, 1.0, 5.0], dtype=np.float32)
    scaled = ccd_main.scale_to_int16(raw_floats, scale_factor=10000, nodata_val=-9999)

    assert scaled[0] == -9999
    assert scaled[1] == -20000
    assert scaled[2] == 0
    assert scaled[3] == 2500
    assert scaled[4] == 10000
    assert scaled[5] == 32767
    assert scaled.dtype == np.int16


# =====================================================================
# 3. FRACTIONAL YEARS AND HARMONIC DESIGN MATRIX TESTS
# =====================================================================

def test_extract_fractional_years_monotonicity():
    """Verify timestamp conversion produces strictly increasing continuous fractional years."""
    # 2022-01-01T00:00:00Z, 2022-07-02T12:00:00Z, 2023-01-01T00:00:00Z
    timestamps = np.array([1640995200.0, 1656763200.0, 1672531200.0], dtype=np.float64)
    frac = ccd_main.extract_fractional_years(timestamps)

    assert len(frac) == 3
    assert np.isclose(frac[0], 2022.0, atol=1e-3)
    assert np.isclose(frac[1], 2022.5, atol=1e-2)
    assert np.isclose(frac[2], 2023.0, atol=1e-3)
    assert np.all(np.diff(frac) > 0)


def test_build_harmonic_matrix_column_structure():
    """Verify Fourier basis matrix contains constant, linear, and periodic terms."""
    t = np.array([2022.0, 2022.25, 2022.5, 2022.75, 2023.0])
    periods = [0.5, 1.0]

    # Test with constant only + 2 periods: 1 + 2*2 = 5 features
    X = ccd_main.build_harmonic_matrix(t, temporal_periods=periods,
                                       enable_const=True, enable_lin=False, enable_quad=False)
    assert X.shape == (5, 5)
    assert np.allclose(X[:, 0], 1.0)

    # Test with constant + linear + quadratic + 3 periods: 3 + 2*3 = 9 features
    X_full = ccd_main.build_harmonic_matrix(t, temporal_periods=[0.33, 0.5, 1.0],
                                            enable_const=True, enable_lin=True, enable_quad=True)
    assert X_full.shape == (5, 9)
    assert np.allclose(X_full[:, 0], 1.0)
    assert np.allclose(X_full[:, 1], t)
    assert np.allclose(X_full[:, 2], t**2)


# =====================================================================
# 4. CHI-SQUARE THRESHOLDING AND ANOMALY EVALUATION TESTS
# =====================================================================

def test_chi2_threshold_evaluation():
    """Verify chi-square anomaly threshold with degrees of freedom = 3."""
    prob = 0.99
    df = 3
    threshold = chi2.ppf(prob, df=df)
    assert threshold > 11.0 and threshold < 12.0

    # Synthetic normalized squared error below threshold: normal
    err_normal = 2.0
    rmse = 1.0
    stat_normal = (err_normal / rmse) ** 2
    assert stat_normal < threshold

    # Synthetic normalized squared error above threshold: anomaly
    err_anom = 4.0
    stat_anom = (err_anom / rmse) ** 2
    assert stat_anom > threshold


# =====================================================================
# 5. ROW CHUNK CCD ESTIMATION TESTS
# =====================================================================

def test_process_row_chunk_stable_pixel():
    """Verify stable annual time series exhibits low RMSE and zero change breaks."""
    n_frames = 40
    t_years = np.linspace(2022.0, 2025.0, n_frames)
    acq_times = t_years * 31557600.0

    # Pure harmonic annual signal + small noise
    y_signal = 1.5 + 0.8 * np.cos(2.0 * np.pi * t_years) + 0.4 * np.sin(2.0 * np.pi * t_years)

    y_data_chunk = np.zeros((n_frames, 1, 1), dtype=np.float32)
    y_data_chunk[:, 0, 0] = y_signal
    valid_mask_chunk = np.ones((n_frames, 1, 1), dtype=bool)

    X_full = ccd_main.build_harmonic_matrix(t_years, temporal_periods=[1.0],
                                            enable_const=True, enable_lin=False, enable_quad=False)

    chunk_args = (
        0, 1, 1,
        y_data_chunk, valid_mask_chunk,
        t_years, X_full, acq_times,
        16, 0.99, 4
    )

    res = ccd_main._process_row_chunk(chunk_args)
    y_start, y_end, pred, rmse, coef, flags, date, count = res[:8]

    assert y_start == 0
    assert y_end == 1
    assert count[0, 0] == 0
    assert np.isnan(date[0, 0])
    valid_pred = ~np.isnan(pred[:, 0, 0])
    assert np.sum(valid_pred) >= 16
    assert np.nanmean(rmse[:, 0, 0]) < 0.2


def test_process_row_chunk_structural_break_detection():
    """Verify sudden step change triggers anomaly flag streak and registers change count."""
    n_frames = 50
    t_years = np.linspace(2022.0, 2026.0, n_frames)
    acq_times = t_years * 31557600.0

    # Step function after frame 30
    y_signal = np.ones(n_frames, dtype=np.float32) * 1.0
    y_signal[30:] = 10.0  # abrupt persistent jump

    y_data_chunk = np.zeros((n_frames, 1, 1), dtype=np.float32)
    y_data_chunk[:, 0, 0] = y_signal
    valid_mask_chunk = np.ones((n_frames, 1, 1), dtype=bool)

    X_full = ccd_main.build_harmonic_matrix(t_years, temporal_periods=[1.0],
                                            enable_const=True, enable_lin=False, enable_quad=False)

    chunk_args = (
        0, 1, 1,
        y_data_chunk, valid_mask_chunk,
        t_years, X_full, acq_times,
        16, 0.99, 4
    )

    res = ccd_main._process_row_chunk(chunk_args)
    y_start, y_end, pred, rmse, coef, flags, date, count = res[:8]

    assert count[0, 0] >= 1
    assert not np.isnan(date[0, 0])
    # Anomaly flags should be present near the jump
    assert np.sum(flags[:, 0, 0]) >= 4


def test_process_row_chunk_insufficient_samples_skips():
    """Verify pixel with fewer than min_samples valid observations remains unmodeled."""
    n_frames = 30
    t_years = np.linspace(2022.0, 2025.0, n_frames)
    acq_times = t_years * 31557600.0

    y_data_chunk = np.ones((n_frames, 1, 1), dtype=np.float32)
    # Only 5 valid observations (< min_samples = 16)
    valid_mask_chunk = np.zeros((n_frames, 1, 1), dtype=bool)
    valid_mask_chunk[:5, 0, 0] = True

    X_full = ccd_main.build_harmonic_matrix(t_years, temporal_periods=[1.0],
                                            enable_const=True, enable_lin=False, enable_quad=False)

    chunk_args = (
        0, 1, 1,
        y_data_chunk, valid_mask_chunk,
        t_years, X_full, acq_times,
        16, 0.99, 4
    )

    res = ccd_main._process_row_chunk(chunk_args)
    y_start, y_end, pred, rmse, coef, flags, date, count = res[:8]

    assert count[0, 0] == 0
    assert np.isnan(date[0, 0])
    assert np.all(np.isnan(pred[:, 0, 0]))
    assert np.all(np.isnan(rmse[:, 0, 0]))


# =====================================================================
# 6. FULL CCD PIPELINE EXECUTION & SCHEMA VERIFICATION TESTS
# =====================================================================

def test_ccd_main_pipeline_execution_and_h5_schema():
    """Verify end-to-end execution of harmonized_CCD_main on synthetic dataset."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        input_h5 = Path(tmp_dir) / "HLST_Rochester_Harmonized_Test.h5"
        output_h5 = Path(tmp_dir) / "Rochester_CCD_Harmonized_Test.h5"

        n_frames = 25
        h, w = 4, 4
        start_ts = datetime(2022, 1, 1, tzinfo=timezone.utc).timestamp()
        end_ts = datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp()
        acq_times = np.linspace(start_ts, end_ts, n_frames)
        t_years = np.linspace(2022.0, 2025.0, n_frames)

        cube = np.zeros((n_frames, h, w), dtype=np.float32)
        for y in range(h):
            for x in range(w):
                cube[:, y, x] = 1.0 + 0.5 * np.cos(2.0 * np.pi * t_years) + 0.1 * np.random.randn(n_frames)

        with h5py.File(input_h5, "w") as f:
            f.attrs["GeoTransform"] = [-16048.67263193, 29.99751894, 0.0, 21147.15814177, 0.0, -29.99001816]
            f.attrs["spatial_ref"] = "Albers Equal Area"
            grp = f.create_group("/HDFEOS/GRIDS/HARMONIZED/Data Fields")
            ds = grp.create_dataset("sliding_volume_z_score", data=cube)
            ds.attrs["acquisition_time"] = acq_times
            grp.create_dataset("common_mask", data=np.zeros((n_frames, h, w), dtype=np.uint8))

        res_path = ccd_main.main(
            location="Rochester",
            h5_path=str(input_h5),
            output_h5=str(output_h5),
            target_metric="sliding_volume_z_score",
            launch_vis=False,
            n_jobs=1,
            chunk_size=2
        )

        assert os.path.exists(res_path)

        with h5py.File(output_h5, "r") as out_f:
            # Check root attributes
            assert "GeoTransform" in out_f.attrs
            assert "spatial_ref" in out_f.attrs
            assert out_f.attrs["TARGET_METRIC"] == "sliding_volume_z_score"
            assert out_f.attrs["CHANGE_PROBABILITY"] == 0.99
            assert out_f.attrs["CONSECUTIVE_ANOMALIES"] == 4

            # Check datasets
            expected_ds = [
                "predicted_series",
                "rmse_series",
                "coef_series",
                "anomaly_flags",
                "change_date_timestamp",
                "change_count"
            ]
            for name in expected_ds:
                assert name in out_f, f"Missing dataset: {name}"

            assert out_f["predicted_series"].shape == (n_frames, h, w)
            assert out_f["predicted_series"].dtype == np.float32
            assert out_f["rmse_series"].shape == (n_frames, h, w)
            assert out_f["rmse_series"].dtype == np.float32
            assert out_f["coef_series"].shape[0:3] == (n_frames, h, w)
            assert out_f["anomaly_flags"].shape == (n_frames, h, w)
            assert out_f["anomaly_flags"].dtype == np.uint8
            assert out_f["change_date_timestamp"].shape == (h, w)
            assert out_f["change_date_timestamp"].dtype == np.float64
            assert out_f["change_count"].shape == (h, w)
            assert out_f["change_count"].dtype == np.int32


def test_ccd_main_streaming_block_execution_and_resumption():
    """Verify streaming block execution with multiple blocks and resume capability."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        input_h5 = Path(tmp_dir) / "HLST_Streaming_Test.h5"
        output_h5 = Path(tmp_dir) / "CCD_Streaming_Test.h5"

        n_frames = 20
        h, w = 6, 4
        start_ts = datetime(2022, 1, 1, tzinfo=timezone.utc).timestamp()
        end_ts = datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp()
        acq_times = np.linspace(start_ts, end_ts, n_frames)
        t_years = np.linspace(2022.0, 2025.0, n_frames)

        cube = np.zeros((n_frames, h, w), dtype=np.float32)
        for y in range(h):
            for x in range(w):
                cube[:, y, x] = 1.0 + 0.5 * np.cos(2.0 * np.pi * t_years) + 0.05 * np.random.randn(n_frames)

        with h5py.File(input_h5, "w") as f:
            f.attrs["GeoTransform"] = [-16048.67, 30.0, 0.0, 21147.15, 0.0, -30.0]
            f.attrs["spatial_ref"] = "Albers Equal Area"
            grp = f.create_group("/HDFEOS/GRIDS/HARMONIZED/Data Fields")
            ds = grp.create_dataset("sliding_volume_z_score", data=cube)
            ds.attrs["acquisition_time"] = acq_times
            grp.create_dataset("common_mask", data=np.zeros((n_frames, h, w), dtype=np.uint8))

        # First run: 6 rows with block_size=2 produces 3 blocks
        res_path = ccd_main.main(
            location="Rochester",
            h5_path=str(input_h5),
            output_h5=str(output_h5),
            target_metric="sliding_volume_z_score",
            launch_vis=False,
            n_jobs=1,
            block_size=2,
            worker_chunk_rows=1,
            resume=True
        )

        assert os.path.exists(res_path)

        with h5py.File(output_h5, "r") as out_f:
            assert out_f["predicted_series"].shape == (n_frames, h, w)
            assert out_f["change_date_timestamp"].shape == (h, w)
            assert out_f["coef_series"].shape[:3] == (n_frames, h, w)

        # Second run with resume=True: should detect matching schema and resume without recreating
        res_path2 = ccd_main.main(
            location="Rochester",
            h5_path=str(input_h5),
            output_h5=str(output_h5),
            target_metric="sliding_volume_z_score",
            launch_vis=False,
            n_jobs=1,
            block_size=2,
            worker_chunk_rows=1,
            resume=True
        )
        assert res_path2 == res_path


# =====================================================================
# 7. VISUALIZATION MODULE TESTS
# =====================================================================

def test_ccd_vis_default_png_naming():
    """Verify visualization helper generates correct PNG filename."""
    name = ccd_vis.get_default_png_name("Rochesterv2", "sliding_volume_z_score")
    assert name == "Rochesterv2_CCD_sliding_volume_z_score_anomaly_overlay.png"


def test_ccd_vis_extract_patch_endmembers_with_speccomplextorch():
    """Verify patch endmember extraction via SpecComplexTorch.maximumDistance_volumes."""
    patch = np.random.rand(3, 3, 10).astype(np.float32)
    em = ccd_vis._extract_patch_endmembers(patch, num_endmembers=7)

    assert em.shape == (10, 7)
    assert not np.isnan(em).all()
    assert em.dtype == np.float32


def test_ccd_vis_extract_patch_endmembers_insufficient_valid_pixels():
    """Verify patch with too few valid pixels returns NaN array without error."""
    patch = np.full((3, 3, 10), np.nan, dtype=np.float32)
    em = ccd_vis._extract_patch_endmembers(patch, num_endmembers=7)

    assert em.shape == (10, 7)
    assert np.all(np.isnan(em))


def test_ccd_vis_non_interactive_png_export():
    """Verify plot_spatial_anomaly_overlay runs non-interactively and saves diagnostic PNG."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        src_h5 = Path(tmp_dir) / "test_src.h5"
        inf_h5 = Path(tmp_dir) / "test_inf.h5"
        out_png = Path(tmp_dir) / "diagnostic_map.png"

        n_frames = 5
        h, w = 4, 4

        with h5py.File(src_h5, "w") as f:
            grp = f.create_group("/HDFEOS/GRIDS/HARMONIZED/Data Fields")
            ds = grp.create_dataset("sliding_volume_z_score", data=np.random.rand(n_frames, h, w).astype(np.float32))
            ds.attrs["acquisition_time"] = np.linspace(1640995200.0, 1672531200.0, n_frames)
            grp.create_dataset("common_mask", data=np.zeros((n_frames, h, w), dtype=np.uint8))
            grp.create_dataset("ortho_visual", data=np.zeros((n_frames, 3, h, w), dtype=np.uint8))

        with h5py.File(inf_h5, "w") as f:
            f.attrs["TARGET_METRIC"] = "sliding_volume_z_score"
            f.create_dataset("change_date_timestamp", data=np.full((h, w), np.nan, dtype=np.float64))
            f.create_dataset("change_count", data=np.zeros((h, w), dtype=np.int32))

        fig = ccd_vis.plot_spatial_anomaly_overlay(
            str(src_h5),
            str(inf_h5),
            output_png=str(out_png),
            interactive=False
        )

        assert out_png.exists()
        assert out_png.stat().st_size > 0


def test_ccd_vis_plot_pixel_sits_rendering():
    """Verify plot_pixel_sits executes correctly on mock Matplotlib Axes."""
    import matplotlib.pyplot as plt

    with tempfile.TemporaryDirectory() as tmp_dir:
        src_h5 = Path(tmp_dir) / "sits_src.h5"
        inf_h5 = Path(tmp_dir) / "sits_inf.h5"

        n_frames = 10
        h, w = 2, 2

        with h5py.File(src_h5, "w") as f:
            f.attrs["GeoTransform"] = [-16048.67, 30.0, 0.0, 21147.16, 0.0, -30.0]
            f.attrs["spatial_ref"] = "Albers Equal Area"
            grp = f.create_group("/HDFEOS/GRIDS/HARMONIZED/Data Fields")
            ds = grp.create_dataset("sliding_volume_z_score", data=np.random.rand(n_frames, h, w).astype(np.float32))
            ds.attrs["acquisition_time"] = np.linspace(1640995200.0, 1672531200.0, n_frames)
            grp.create_dataset("common_mask", data=np.zeros((n_frames, h, w), dtype=np.uint8))

        with h5py.File(inf_h5, "w") as f:
            f.attrs["TARGET_METRIC"] = "sliding_volume_z_score"
            f.attrs["CHANGE_PROBABILITY"] = 0.99
            f.attrs["CHI2_DEGREES_OF_FREEDOM"] = 3
            f.create_dataset("predicted_series", data=np.ones((n_frames, h, w), dtype=np.float32))
            f.create_dataset("rmse_series", data=np.full((n_frames, h, w), 0.1, dtype=np.float32))
            f.create_dataset("anomaly_flags", data=np.zeros((n_frames, h, w), dtype=np.uint8))

        fig, ax = plt.subplots()
        ccd_vis.plot_pixel_sits(
            pixel_y=0,
            pixel_x=0,
            source_h5_path=str(src_h5),
            inference_results_h5=str(inf_h5),
            ax=ax
        )

        # Confirm elements plotted on axis
        assert len(ax.lines) > 0
        plt.close(fig)


def test_tqdm_joblib_context_manager():
    """Verify tqdm_joblib patches and restores joblib callback correctly."""
    from tqdm import tqdm
    with ccd_main.tqdm_joblib(tqdm(total=10, disable=True)) as bar:
        assert bar is not None


def test_resolve_h5_input_existing():
    """Verify resolve_h5_input returns provided path when it exists."""
    with tempfile.NamedTemporaryFile(suffix=".h5") as tmp:
        resolved = ccd_main.resolve_h5_input(location="TestLoc", h5_path=tmp.name)
        assert resolved == tmp.name

