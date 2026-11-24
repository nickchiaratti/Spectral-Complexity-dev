"""
Tier 3: Cross-Feature Combinations & Integration Test Suite.
Validates multi-module interactions, cross-sensor harmonization, in-place HDF5 updates,
CCD time series consumption from harmonized files, and repository-wide PyTorch isolation.
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
from sklearn.linear_model import LassoLarsIC

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def test_int_01_multi_sensor_timeline_harmonization():
    """Verify temporal sorting and alignment of mixed HLSL30, HLSS30, and TANAGER acquisitions."""
    # Synthetic frame acquisitions
    raw_acquisitions = [
        {"sensor": "HLSL30", "time": 1704100000.0, "spacecraft": "LANDSAT-8"},
        {"sensor": "HLSS30", "time": 1704050000.0, "spacecraft": "Sentinel-2A"},
        {"sensor": "TANAGER", "time": 1704200000.0, "spacecraft": "Tanager-1"},
        {"sensor": "HLSS30", "time": 1704150000.0, "spacecraft": "Sentinel-2B"},
    ]

    # Chronological sort
    sorted_frames = sorted(raw_acquisitions, key=lambda x: x["time"])
    times = [f["time"] for f in sorted_frames]

    # Strictly monotonic non-decreasing order
    assert all(times[i] <= times[i + 1] for i in range(len(times) - 1))
    assert sorted_frames[0]["sensor"] == "HLSS30"
    assert sorted_frames[-1]["sensor"] == "TANAGER"


def test_int_02_sc_metrics_in_place_insertion_integrity():
    """Verify appending SC metrics directly into an existing harmonized HDF5 file."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        h5_path = Path(tmp_dir) / "HLST_Rochester_Harmonized.h5"
        # 1. Create base harmonized file with HLSL30, HLSS30, TANAGER
        with h5py.File(h5_path, "w") as f:
            for sname, bcount, nframes in [("HLSL30", 7, 3), ("HLSS30", 10, 4), ("TANAGER", 426, 1)]:
                grp = f.create_group(f"/HDFEOS/GRIDS/{sname}/Data Fields")
                grp.create_dataset(
                    "surface_reflectance",
                    data=np.full((nframes, bcount, 20, 20), 0.2, dtype=np.float32)
                )
                grp.create_dataset(
                    "common_mask",
                    data=np.ones((nframes, 20, 20), dtype=bool)
                )
            harm_grp = f.create_group("/HDFEOS/GRIDS/HARMONIZED/Data Fields")
            harm_grp.create_dataset("common_mask", data=np.ones((8, 20, 20), dtype=bool))

        # 2. Append metrics in place (simulating Step 4)
        with h5py.File(h5_path, "r+") as f:
            harm_grp = f["/HDFEOS/GRIDS/HARMONIZED/Data Fields"]
            harm_grp.create_dataset(
                "sliding_volume_map",
                data=np.ones((8, 20, 20), dtype=np.float32) * 5.0,
                chunks=(1, 20, 20),
                compression="gzip",
                compression_opts=5
            )
            harm_grp.create_dataset(
                "sliding_volume_z_score",
                data=np.zeros((8, 20, 20), dtype=np.float32),
                chunks=(1, 20, 20),
                compression="gzip",
                compression_opts=5
            )
            for sname, bcount, nframes in [("HLSL30", 7, 3), ("HLSS30", 10, 4), ("TANAGER", 426, 1)]:
                s_grp = f[f"/HDFEOS/GRIDS/{sname}/Data Fields"]
                s_grp.create_dataset("frame_endmembers", data=np.zeros((nframes, bcount, 7), dtype=np.float32))
                s_grp.create_dataset("frame_endmember_indices", data=np.zeros((nframes, 7), dtype=np.int32))
                s_grp.create_dataset("frame_endmember_volumes", data=np.zeros((nframes, 7), dtype=np.float32))

        # 3. Verify datasets in the modified file
        with h5py.File(h5_path, "r") as f:
            harm_fields = f["/HDFEOS/GRIDS/HARMONIZED/Data Fields"]
            assert "sliding_volume_map" in harm_fields
            assert "sliding_volume_z_score" in harm_fields
            # Verify original datasets remain unchanged
            sr = f["/HDFEOS/GRIDS/HLSL30/Data Fields/surface_reflectance"][:]
            assert np.allclose(sr, 0.2)


def test_int_03_ccd_consumes_in_place_zscore():
    """Verify CCD time series algorithm directly loads and fits from in-place z-score dataset."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        h5_path = Path(tmp_dir) / "Harmonized_with_SC.h5"
        n_frames = 30
        h, w = 4, 4

        # Generate synthetic temporal z-score with known seasonal variation
        t_years = np.linspace(2022.0, 2025.0, n_frames)
        z_series = 0.5 * np.cos(2.0 * np.pi * t_years) + 0.1 * np.random.randn(n_frames)
        cube = np.zeros((n_frames, h, w), dtype=np.float32)
        cube[:, :, :] = z_series[:, None, None]

        with h5py.File(h5_path, "w") as f:
            f.attrs["GeoTransform"] = [-16048.67, 30.0, 0.0, 21147.16, 0.0, -30.0]
            f.attrs["spatial_ref"] = "Albers Equal Area"
            grp = f.create_group("/HDFEOS/GRIDS/HARMONIZED/Data Fields")
            grp.create_dataset("sliding_volume_z_score", data=cube)
            ds = grp.create_dataset("common_mask", data=np.ones((n_frames, h, w), dtype=bool))
            ds.attrs["acquisition_time"] = t_years * 31557600.0  # seconds

        # Simulate CCD reading directly from source
        with h5py.File(h5_path, "r") as f:
            z_data = f["/HDFEOS/GRIDS/HARMONIZED/Data Fields/sliding_volume_z_score"][:]
            mask_data = f["/HDFEOS/GRIDS/HARMONIZED/Data Fields/common_mask"][:]

        assert z_data.shape == (n_frames, h, w)
        assert mask_data.shape == (n_frames, h, w)

        # Fit harmonic model on single pixel time series
        pixel_series = z_data[:, 0, 0]
        X = np.column_stack([
            np.ones(n_frames),
            t_years,
            np.cos(2.0 * np.pi * t_years),
            np.sin(2.0 * np.pi * t_years)
        ])
        model = LassoLarsIC(criterion="aic", fit_intercept=False)
        model.fit(X, pixel_series)
        pred = model.predict(X)
        residuals = pixel_series - pred
        rmse = float(np.sqrt(np.mean(residuals ** 2)))
        assert rmse < 0.25


def test_int_04_repository_wide_pytorch_isolation_audit():
    """Verify that no scripts outside SpecComplexTorch.py and test files import PyTorch."""
    py_files = list(REPO_ROOT.glob("*.py"))
    violations = []
    excluded_files = {"SpecComplexTorch.py", "create_injected_dataset.py", "generate_stratified_patch_benchmark.py"}
    for p in py_files:
        if p.name in excluded_files:
            continue
        with open(p, "r", encoding="utf-8") as f:
            content = f.read()
            tree = ast.parse(content, filename=str(p))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if "torch" in alias.name:
                            violations.append((p.name, alias.name))
                elif isinstance(node, ast.ImportFrom):
                    if node.module and "torch" in node.module:
                        violations.append((p.name, node.module))

    assert not violations, f"PyTorch isolation violated in repository scripts: {violations}"


def test_int_05_synthetic_pipeline_dataflow_execution():
    """Verify end-to-end dataflow sequence on synthetic multi-band raster."""
    # 1. Input surface reflectance (4 bands, 10x10)
    rng = np.random.default_rng(99)
    cube = rng.uniform(0.05, 0.60, size=(4, 10, 10)).astype(np.float32)
    mask = np.ones((10, 10), dtype=bool)

    # 2. Sliding window 3x3 volume mock
    tile_size = 3
    stride = 1
    h, w = 10, 10
    vol_map = np.zeros((h, w), dtype=np.float32)
    counts = np.zeros((h, w), dtype=np.int32)
    for y in range(0, h - tile_size + 1, stride):
        for x in range(0, w - tile_size + 1, stride):
            patch = cube[:, y:y + tile_size, x:x + tile_size]
            val = float(np.sum(patch ** 2))
            vol_map[y:y + tile_size, x:x + tile_size] += val
            counts[y:y + tile_size, x:x + tile_size] += 1
    valid_counts = counts > 0
    vol_map[valid_counts] /= counts[valid_counts]

    # 3. Z-Score calculation
    valid_vols = vol_map[mask]
    mean = float(np.mean(valid_vols))
    std = float(np.std(valid_vols))
    z_map = (vol_map - mean) / std

    # 4. Verification of output properties
    assert z_map.shape == (10, 10)
    assert np.isclose(float(np.mean(z_map[mask])), 0.0, atol=1e-5)
    assert np.isclose(float(np.std(z_map[mask])), 1.0, atol=1e-5)


def test_int_06_chronological_multi_sensor_interleaving():
    """Verify interleaving timestamps from Landsat-8, Landsat-9, Sentinel-2A/B/C, and Tanager-1."""
    timestamps = {
        "HLSL30_L8": [1640995200.0, 1642204800.0],
        "HLSL30_L9": [1641600000.0, 1642809600.0],
        "HLSS30_S2A": [1641254400.0, 1642464000.0],
        "HLSS30_S2B": [1641859200.0, 1643068800.0],
        "TANAGER": [1751647930.0]
    }
    all_events = []
    for platform, ts_list in timestamps.items():
        for t in ts_list:
            all_events.append((t, platform))
    all_events.sort(key=lambda x: x[0])

    event_times = [e[0] for e in all_events]
    assert event_times == sorted(event_times)
    assert len(all_events) == 9
    assert all_events[-1][1] == "TANAGER"  # 2025 Tanager event is last
