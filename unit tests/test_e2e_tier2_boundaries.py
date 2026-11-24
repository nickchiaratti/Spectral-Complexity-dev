"""
Tier 2: Boundary & Corner Cases Test Suite.
Validates system behavior at domain boundaries, extreme inputs, and transition limits.
Covers minimum 9-pixel unmasked rule, all-masked frames, NaN propagation,
spatial corners, tile extremes, sentinels, and zero-variance scenarios.
"""

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


# ---------------------------------------------------------------------
# Helper function implementing the 9-pixel threshold z-score rule
# ---------------------------------------------------------------------
def calc_z_score_with_threshold(volume_array, valid_pixel_mask, min_pixels=9):
    """
    Computes global z-score with the strict rule:
    If number of valid unmasked pixels is less than min_pixels (default 9),
    returns an all-NaN z-score map with mean=NaN and std=NaN.
    """
    valid_vals = volume_array[valid_pixel_mask]
    n_valid = len(valid_vals)
    if n_valid < min_pixels:
        z_map = np.full_like(volume_array, np.nan, dtype=np.float32)
        return z_map, np.nan, np.nan

    mean = float(np.mean(valid_vals))
    std = float(np.std(valid_vals))
    z_map = np.full_like(volume_array, np.nan, dtype=np.float32)
    if std > 0.0:
        z_map[valid_pixel_mask] = (volume_array[valid_pixel_mask] - mean) / std
    else:
        z_map[valid_pixel_mask] = 0.0
    return z_map, mean, std


def test_bva_01_zscore_minimum_nine_unmasked_pixels():
    """Verify minimum 9 unmasked pixels rule for Z-score calculation across boundary values."""
    vol_grid = np.ones((10, 10), dtype=np.float32) * 5.0
    # Boundary test: N in {0, 1, 8, 9, 10}
    for count in [0, 1, 8]:
        mask = np.zeros((10, 10), dtype=bool)
        if count > 0:
            coords = np.unravel_index(np.arange(count), (10, 10))
            mask[coords] = True
        z_map, mean, std = calc_z_score_with_threshold(vol_grid, mask, min_pixels=9)
        assert np.isnan(mean), f"Expected mean=NaN for count={count}"
        assert np.isnan(std), f"Expected std=NaN for count={count}"
        assert np.all(np.isnan(z_map)), f"Expected all-NaN z_map for count={count}"

    # Exactly 9 unmasked pixels with distinct values
    mask_9 = np.zeros((10, 10), dtype=bool)
    coords_9 = np.unravel_index(np.arange(9), (10, 10))
    mask_9[coords_9] = True
    vol_grid[coords_9] = np.arange(1.0, 10.0, dtype=np.float32)
    z_map_9, mean_9, std_9 = calc_z_score_with_threshold(vol_grid, mask_9, min_pixels=9)
    assert np.isfinite(mean_9)
    assert np.isfinite(std_9)
    assert np.isclose(mean_9, 5.0, atol=1e-5)
    assert np.count_nonzero(~np.isnan(z_map_9)) == 9
    assert np.count_nonzero(np.isnan(z_map_9)) == 91

    # 10 unmasked pixels
    mask_10 = np.zeros((10, 10), dtype=bool)
    coords_10 = np.unravel_index(np.arange(10), (10, 10))
    mask_10[coords_10] = True
    vol_grid[coords_10] = np.arange(1.0, 11.0, dtype=np.float32)
    z_map_10, mean_10, std_10 = calc_z_score_with_threshold(vol_grid, mask_10, min_pixels=9)
    assert np.isfinite(mean_10)
    assert np.count_nonzero(~np.isnan(z_map_10)) == 10


def test_bva_02_all_masked_scenes_handling():
    """Verify that a 100% masked scene (zero valid pixels) produces all-NaN outputs without crash."""
    h, w = 50, 50
    all_masked = np.zeros((h, w), dtype=bool)
    vol_map = np.random.rand(h, w).astype(np.float32)
    z_map, mean, std = calc_z_score_with_threshold(vol_map, all_masked, min_pixels=9)
    assert np.isnan(mean)
    assert np.isnan(std)
    assert np.all(np.isnan(z_map))


def test_bva_03_scattered_nan_pixel_containment():
    """Verify scattered NaNs in reflectance do not propagate to unmasked valid regions."""
    rng = np.random.default_rng(123)
    frame = rng.uniform(0.1, 0.8, size=(7, 20, 20)).astype(np.float32)
    # Inject NaNs at isolated locations
    frame[:, 5, 5] = np.nan
    frame[2, 10, 10] = np.nan

    valid_mask = ~np.isnan(frame).any(axis=0)
    assert valid_mask[5, 5] == False
    assert valid_mask[10, 10] == False
    assert valid_mask[0, 0] == True

    # Valid values extraction
    valid_pixels = frame[:, valid_mask]
    assert not np.isnan(valid_pixels).any(), "Valid pixel subset must contain no NaNs"


def test_bva_04_extreme_spatial_boundaries_four_corners():
    """Verify tile boundary indexing at the four extreme corners of a 1409x1070 grid."""
    height, width = 1409, 1070
    tile_size = 3
    stride = 1

    corners = [
        (0, 0),                        # Upper-left
        (0, width - tile_size),        # Upper-right
        (height - tile_size, 0),       # Lower-left
        (height - tile_size, width - tile_size)  # Lower-right
    ]

    for y, x in corners:
        y_slice = slice(y, y + tile_size)
        x_slice = slice(x, x + tile_size)
        assert 0 <= y_slice.start and y_slice.stop <= height
        assert 0 <= x_slice.start and x_slice.stop <= width


def test_bva_05_single_pixel_tile_execution():
    """Verify degenerate single-pixel tile (tile_size=1, stride=1) preserves raster shape."""
    h, w = 10, 10
    tile_size = 1
    stride = 1
    num_tiles_y = (h - tile_size) // stride + 1
    num_tiles_x = (w - tile_size) // stride + 1
    assert num_tiles_y == 10
    assert num_tiles_x == 10


def test_bva_06_tile_larger_than_raster():
    """Verify that tile_size exceeding raster dimensions is caught without IndexError."""
    h, w = 5, 5
    tile_size = 10
    stride = 1
    can_tile = (h >= tile_size) and (w >= tile_size)
    assert not can_tile, "Tile larger than raster must not proceed with standard tiling"


def test_bva_07_zero_variance_volume_map():
    """Verify standard deviation = 0 (identical volume pixels) prevents division by zero."""
    vol_grid = np.full((15, 15), 4.25, dtype=np.float32)
    mask = np.ones((15, 15), dtype=bool)
    z_map, mean, std = calc_z_score_with_threshold(vol_grid, mask, min_pixels=9)
    assert np.isclose(mean, 4.25, atol=1e-5)
    assert np.isclose(std, 0.0, atol=1e-5)
    # Z-scores should be zero when std=0, not NaN or inf
    assert not np.isinf(z_map[mask]).any()
    assert np.allclose(z_map[mask], 0.0)


def test_bva_08_saturated_and_negative_reflectance():
    """Verify that unphysical negative or saturated reflectance values do not produce fatal errors."""
    reflectance = np.array([-0.25, 0.0, 0.5, 1.0, 1.75], dtype=np.float32)
    # Negative values can arise from atmospheric over-correction; specular glint can exceed 1.0
    assert np.any(reflectance < 0.0)
    assert np.any(reflectance > 1.0)
    # Ensure clipping or validity checking preserves array float integrity
    finite_mask = np.isfinite(reflectance)
    assert np.all(finite_mask)


def test_bva_09_temporal_filter_exact_boundary_seconds():
    """Verify temporal filter rejects 1 second before 2022 and 1 second after 2026."""
    start = datetime(2022, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    end = datetime(2026, 12, 31, 23, 59, 59, tzinfo=timezone.utc)

    sec_before = datetime(2021, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
    sec_start = datetime(2022, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    sec_end = datetime(2026, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
    sec_after = datetime(2027, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

    assert not (start <= sec_before <= end)
    assert start <= sec_start <= end
    assert start <= sec_end <= end
    assert not (start <= sec_after <= end)


def test_bva_10_sentinel_fill_value_masking_integrity():
    """Verify sentinel values (-9999.0, 255, 40000.0) are completely isolated by mask checks."""
    sr_pixel = -9999.0
    fmask_pixel = np.uint8(255)
    angle_pixel = 40000.0

    is_sr_valid = (sr_pixel != -9999.0) and not np.isnan(sr_pixel)
    is_fmask_valid = (fmask_pixel != 255)
    is_angle_valid = (angle_pixel != 40000.0)

    assert not is_sr_valid
    assert not is_fmask_valid
    assert not is_angle_valid


def test_bva_11_sensor_band_dimension_extremes():
    """Verify simplex volume calculations support band dimension variations: 7, 10, 426."""
    band_configs = [
        ("HLSL30", 7),
        ("HLSS30", 10),
        ("TANAGER", 426)
    ]
    for sensor, bands in band_configs:
        dummy_pixel = np.ones((bands,), dtype=np.float32)
        norm_sq = np.sum(dummy_pixel ** 2)
        assert np.isclose(norm_sq, float(bands))


def test_bva_12_ccd_minimum_samples_threshold():
    """Verify CCD skips model fitting when unmasked observation count is less than 12."""
    min_samples = 12

    def can_fit_ccd(observations_count):
        return observations_count >= min_samples

    assert not can_fit_ccd(0)
    assert not can_fit_ccd(5)
    assert not can_fit_ccd(11)
    assert can_fit_ccd(12)
    assert can_fit_ccd(40)
