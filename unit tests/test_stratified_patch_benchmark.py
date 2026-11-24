"""
Unit tests for generate_stratified_patch_benchmark.py
Verifies:
1. Pre-execution checks for:
   - /HDFEOS/GRIDS/HARMONIZED/Data Fields/sliding_volume_z_score (produced by HLST_SC_calculations.py)
   - /SCCCD_Results (produced by harmonized_CCD_main.py)
   - /PYCCD_Results (produced by pyccd_landsat_main.py)
2. PyCCD only evaluates 4 implanted pixels, copies baseline results for neighbors, and evaluates against BASELINE_PYCCD.
3. SC-CCD evaluates the full 9x9 patch across time to characterize spatial neighbor impacts and evaluates against BASELINE_SCCCD.
"""

import os
import tempfile
import datetime
import pytest
import h5py
import numpy as np
from unittest.mock import patch
import joblib

try:
    import implant_generate_patch_benchmark as gsb
except ImportError:
    try:
        import generate_patch_benchmark as gsb
    except ImportError:
        import generate_stratified_patch_benchmark as gsb


def test_missing_sliding_volume_z_score_raises_error():
    """Verify that extract_baseline_chips raises RuntimeError if sliding_volume_z_score is missing."""
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        with h5py.File(tmp_path, "w") as f:
            f.create_group("HDFEOS/GRIDS/HARMONIZED/Data Fields")
            # sliding_volume_z_score omitted
            f.create_group("SCCCD_Results")
            f.create_group("PYCCD_Results")

        with h5py.File(tmp_path, "r") as f:
            with pytest.raises(RuntimeError) as excinfo:
                gsb.extract_baseline_chips(f)
            assert "HLST_SC_calculations.py" in str(excinfo.value)
            assert "sliding_volume_z_score" in str(excinfo.value)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_missing_scccd_results_raises_error():
    """Verify that extract_baseline_chips raises RuntimeError if SCCCD_Results is missing."""
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        with h5py.File(tmp_path, "w") as f:
            harm_fields = f.create_group("HDFEOS/GRIDS/HARMONIZED/Data Fields")
            harm_fields.create_dataset("sliding_volume_z_score", data=np.zeros((5, 10, 10), dtype=np.float32))
            # SCCCD_Results omitted
            f.create_group("PYCCD_Results")

        with h5py.File(tmp_path, "r") as f:
            with pytest.raises(RuntimeError) as excinfo:
                gsb.extract_baseline_chips(f)
            assert "harmonized_CCD_main.py" in str(excinfo.value)
            assert "SCCCD_Results" in str(excinfo.value)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_missing_pyccd_results_raises_error():
    """Verify that extract_baseline_chips raises RuntimeError if PYCCD_Results is missing."""
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        with h5py.File(tmp_path, "w") as f:
            harm_fields = f.create_group("HDFEOS/GRIDS/HARMONIZED/Data Fields")
            harm_fields.create_dataset("sliding_volume_z_score", data=np.zeros((5, 10, 10), dtype=np.float32))
            scccd_grp = f.create_group("SCCCD_Results")
            scccd_grp.create_dataset("change_date_timestamp", data=np.zeros((10, 10), dtype=np.float64))
            scccd_grp.create_dataset("change_count", data=np.zeros((10, 10), dtype=np.int32))
            # PYCCD_Results omitted

        with h5py.File(tmp_path, "r") as f:
            with pytest.raises(RuntimeError) as excinfo:
                gsb.extract_baseline_chips(f)
            assert "pyccd_landsat_main.py" in str(excinfo.value)
            assert "PYCCD_Results" in str(excinfo.value)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_pyccd_neighbor_preservation_and_implanted_evaluation():
    """
    Verify that run_pyccd_benchmark:
    1. Evaluates only the 4 implanted pixels.
    2. Faithfully copies baseline PyCCD values for all neighbor pixels.
    3. Evaluates detected breaks against baseline PyCCD values.
    4. Produces a full (num_conditions, n_patches, 9, 9) spatial array.
    """
    n_patches = len(gsb.LAND_COVER_PATCHES)
    patch_size = gsb.PATCH_SIZE
    n_frames = 20
    now_ts = 1700000000.0
    l30_acq = np.linspace(now_ts - 86400 * 30, now_ts + 86400 * 30, n_frames)

    # Initialize baseline PyCCD with distinct marker values (unbroken at implanted pixels)
    baseline_date = np.full((n_patches, patch_size, patch_size), 1234567.0, dtype=np.float64)
    baseline_count = np.full((n_patches, patch_size, patch_size), 42, dtype=np.int32)
    # Set implanted pixels to clean in baseline
    for r, c in gsb.INJ_OFFSETS:
        baseline_date[:, r, c] = np.nan
        baseline_count[:, r, c] = 0

    chips = {
        "HLSL30": np.ones((n_patches, n_frames, 7, patch_size, patch_size), dtype=np.int16) * 1000,
        "HLSL30_Fmask": np.zeros((n_patches, n_frames, patch_size, patch_size), dtype=np.uint8),
        "HLSL30_common_mask": np.zeros((n_patches, n_frames, patch_size, patch_size), dtype=np.uint8),
        "BASELINE_PYCCD": {
            "date": baseline_date,
            "count": baseline_count,
        }
    }

    metadata = {
        "l30_acq_times": l30_acq,
        "scale_factors": {"HLSL30": 10000.0},
        "fill_values": {"HLSL30": -9999},
    }

    target_spectra = [{
        "id": "test_mat",
        "name": "Test Material",
        "spec_l8": np.ones(8, dtype=np.float32) * 0.5,
    }]

    # Patch ccd.detect to track calls and return a simulated break
    detect_calls = []

    def mock_detect(*args, **kwargs):
        detect_calls.append(args)
        # Mock break model
        return {
            "change_models": [{
                "start_day": datetime.datetime.fromtimestamp(now_ts - 86400 * 10, tz=datetime.timezone.utc).date().toordinal(),
                "end_day": datetime.datetime.fromtimestamp(now_ts + 86400 * 10, tz=datetime.timezone.utc).date().toordinal(),
                "change_probability": 1.0,
                "break_day": datetime.datetime.fromtimestamp(now_ts + 86400, tz=datetime.timezone.utc).date().toordinal(),
                "magnitudes": [100.0, 100.0],
            }]
        }

    target_mod = gsb.__name__
    with patch(f"{target_mod}.ccd.detect", side_effect=mock_detect):
        with patch(f"{target_mod}.MIXTURE_FRACTIONS", [0.25]):
            with patch(f"{target_mod}.Parallel", side_effect=lambda *a, **kw: joblib.Parallel(n_jobs=1, prefer="threads", **{k: v for k, v in kw.items() if k != "n_jobs"})):
                res = gsb.run_pyccd_benchmark(
                    chips, metadata, target_spectra, onset_ts=now_ts, return_spatial=True
                )
                table, spatial = res[:2]

    # 1 material x 1 fraction = 1 condition; 4 patches x 4 pixels = 16 detect calls
    expected_calls = 1 * n_patches * 4
    assert len(detect_calls) == expected_calls, f"Expected {expected_calls} ccd.detect calls, got {len(detect_calls)}"

    # Check spatial output shape
    # 1 condition, 4 patches, 9x9 or 13x13
    assert spatial["change_date"].shape == (1, n_patches, patch_size, patch_size)
    assert spatial["change_count"].shape == (1, n_patches, patch_size, patch_size)

    # Verify that all neighbor pixels retained baseline values, and only INJ_OFFSETS were updated
    for p_idx in range(n_patches):
        for r in range(patch_size):
            for c in range(patch_size):
                if (r, c) in gsb.INJ_OFFSETS:
                    # Implanted pixels should have detected break
                    assert spatial["change_count"][0, p_idx, r, c] == 1
                    assert not np.isnan(spatial["change_date"][0, p_idx, r, c])
                else:
                    # Neighbor pixels must match baseline PyCCD
                    assert spatial["change_count"][0, p_idx, r, c] == 42
                    assert spatial["change_date"][0, p_idx, r, c] == 1234567.0


def test_scccd_evaluates_full_patch_for_spatial_neighbors():
    """
    Verify that run_scccd_benchmark processes the full 9x9 patch across time,
    evaluating detected breaks against BASELINE_SCCCD produced by harmonized_CCD_main.py.
    """
    n_patches = len(gsb.LAND_COVER_PATCHES)
    patch_size = gsb.PATCH_SIZE
    eval_size = 9 if patch_size == 13 else patch_size
    n_frames = 20
    now_ts = 1700000000.0
    harm_acq = np.linspace(now_ts - 86400 * 50, now_ts + 86400 * 50, n_frames)

    # Simulated z-cubes: 1 condition x 4 patches x 20 frames x patch_size x patch_size
    sim_z = np.random.randn(1, n_patches, n_frames, patch_size, patch_size).astype(np.float32)

    chips = {
        "HARMONIZED_common_mask": np.zeros((n_patches, n_frames, patch_size, patch_size), dtype=np.uint8),
        "BASELINE_SCCCD": {
            "date": np.full((n_patches, patch_size, patch_size), np.nan, dtype=np.float64),
            "count": np.zeros((n_patches, patch_size, patch_size), dtype=np.int32),
        }
    }

    metadata = {
        "harm_acq_times": harm_acq,
    }

    target_spectra = [{
        "id": "test_mat",
        "name": "Test Material",
    }]

    chunk_calls = []

    def mock_process_row_chunk(args):
        chunk_calls.append(args)
        r0, r_end, c_end, y_data, valid_mask = args[0], args[1], args[2], args[3], args[4]
        # Return dummy chunk predictions and break maps matching eval region size
        pred = np.zeros_like(y_data)
        date_map = np.full((eval_size, eval_size), now_ts + 86400, dtype=np.float64)
        count_map = np.ones((eval_size, eval_size), dtype=np.int32)
        mock_seg = {
            "segment_idx": 0,
            "start_idx": 0,
            "start_date": now_ts - 86400 * 10,
            "end_idx": 18,
            "end_date": now_ts + 86400 * 10,
            "has_break": True,
            "break_num": 1,
            "break_idx": 10,
            "break_date": now_ts + 86400,
            "confirm_idx": 12,
            "confirm_date": now_ts + 86400 * 2,
            "rmse": 0.1,
            "coef": np.zeros(8, dtype=np.float64),
            "n_points": 18,
        }
        segs = {(r, c): [mock_seg] for r in range(eval_size) for c in range(eval_size)}
        return (None, None, pred, None, None, None, date_map, count_map, segs)

    target_mod = gsb.__name__
    target_ccd = f"{target_mod}.scccm._process_row_chunk" if hasattr(gsb, "scccm") else f"{target_mod}.hccd._process_row_chunk"

    with patch(target_ccd, side_effect=mock_process_row_chunk):
        with patch(f"{target_mod}.MIXTURE_FRACTIONS", [0.25]):
            with patch(f"{target_mod}.Parallel", side_effect=lambda *a, **kw: joblib.Parallel(n_jobs=1, prefer="threads", **{k: v for k, v in kw.items() if k != "n_jobs"})):
                res = gsb.run_scccd_benchmark(
                    sim_z, chips, metadata, target_spectra, onset_ts=now_ts, return_spatial=True
                )
                table, spatial = res[:2]

    # Exactly 1 call per patch across full patch
    assert len(chunk_calls) == n_patches
    for call_args in chunk_calls:
        # Check that row chunk covers patch dimensions
        assert call_args[0] == 0
        assert call_args[1] == eval_size
        assert call_args[2] == eval_size

    # Check spatial output shape and values
    assert spatial["change_date"].shape == (1, n_patches, patch_size, patch_size)
    assert spatial["change_count"].shape == (1, n_patches, patch_size, patch_size)
    assert np.all(table["break_detected"] == True)

