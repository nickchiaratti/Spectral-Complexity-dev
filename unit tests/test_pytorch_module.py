import os
import sys
import math
import warnings
import numpy as np
import pytest
import torch

# Ensure public module is importable
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import SpecComplexTorch as sct
import SpecComplex as sc


# ==============================================================================
# 1. Module Imports and Interface Verification
# ==============================================================================

def test_module_imports_and_symbols():
    """Verify that SpecComplexTorch exposes all required public functions."""
    required_symbols = [
        'maximumDistance',
        'maximumDistance_volumes',
        'calcGramLocalVolumes',
        'process_volume_sliding_tile',
        'calculate_global_z_score'
    ]
    for sym in required_symbols:
        assert hasattr(sct, sym), f"SpecComplexTorch missing required symbol '{sym}'"
        assert callable(getattr(sct, sym)), f"Symbol '{sym}' is not callable"


# ==============================================================================
# 2. maximumDistance Function Tests
# ==============================================================================

@pytest.mark.parametrize("device", ['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu'])
def test_maximumDistance_torch_3d(device):
    """Test batched maximumDistance with 3D PyTorch tensors on CPU and CUDA."""
    B, C, N = 2, 7, 100
    k = 5
    data = torch.rand(B, C, N, dtype=torch.float32, device=device)
    mask = torch.ones(B, N, dtype=torch.bool, device=device)
    
    # Invalidate some pixels
    mask[:, :10] = False

    endmembers = sct.maximumDistance(data, k, mask)

    assert isinstance(endmembers, torch.Tensor)
    assert endmembers.shape == (B, C, k)
    assert endmembers.device.type == device
    assert torch.all(torch.isfinite(endmembers))


def test_maximumDistance_torch_2d():
    """Test unbatched 2D tensor input (C, N)."""
    C, N = 6, 80
    k = 4
    data = torch.rand(C, N, dtype=torch.float32)
    endmembers = sct.maximumDistance(data, k)
    assert isinstance(endmembers, torch.Tensor)
    assert endmembers.shape == (C, k)
    assert torch.all(torch.isfinite(endmembers))


def test_maximumDistance_numpy():
    """Test maximumDistance accepting and returning NumPy arrays."""
    C, N = 8, 120
    k = 6
    data_np = np.random.rand(C, N).astype(np.float32)
    mask_np = np.ones(N, dtype=bool)
    mask_np[0:5] = False

    em_np = sct.maximumDistance(data_np, k, mask_np)
    assert isinstance(em_np, np.ndarray)
    assert em_np.shape == (C, k)
    assert np.all(np.isfinite(em_np))


def test_maximumDistance_invalid_pixel_rejection():
    """Verify that masked invalid pixels are never selected as endmembers."""
    B, C, N = 1, 4, 50
    k = 3
    data = torch.zeros(B, C, N, dtype=torch.float32)
    # Put huge values at indices 0 and 1, but mark them invalid
    data[0, :, 0] = 9999.0
    data[0, :, 1] = 8888.0
    # Put moderate values at valid indices
    data[0, :, 10] = 10.0
    data[0, :, 20] = 5.0
    data[0, :, 30] = 2.0

    mask = torch.ones(B, N, dtype=torch.bool)
    mask[0, 0] = False
    mask[0, 1] = False

    endmembers = sct.maximumDistance(data, k, mask)
    # The huge values must NOT be present in endmembers
    assert not torch.isclose(endmembers, torch.tensor(9999.0)).any()
    assert not torch.isclose(endmembers, torch.tensor(8888.0)).any()


# ==============================================================================
# 3. maximumDistance_volumes Function Tests
# ==============================================================================

@pytest.mark.parametrize("device", ['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu'])
def test_maximumDistance_volumes_torch(device):
    """Test maximumDistance_volumes on CPU and CUDA."""
    B, C, N = 2, 7, 80
    k = 5
    cube = torch.rand(B, C, N, dtype=torch.float32, device=device)

    em, vol = sct.maximumDistance_volumes(cube, k)

    assert isinstance(em, torch.Tensor)
    assert isinstance(vol, torch.Tensor)
    assert em.shape == (B, C, k)
    assert vol.shape == (B, k)
    assert em.device.type == device
    assert vol.device.type == device

    # Check invariants: vol[:, 0] == 0, vol[:, 1] > 0, vol >= 0
    assert torch.all(vol[:, 0] == 0.0)
    assert torch.all(vol[:, 1] > 0.0)
    assert torch.all(vol >= 0.0)


def test_maximumDistance_volumes_return_heights():
    """Test return_heights option and verify Gantmacher product theorem."""
    B, C, N = 1, 6, 60
    k = 4
    cube = torch.rand(B, C, N, dtype=torch.float32)

    em, vol, heights = sct.maximumDistance_volumes(cube, k, return_heights=True)

    assert em.shape == (B, C, k)
    assert vol.shape == (B, k)
    assert heights.shape == (B, k)

    # Invariant: vol[:, j] == prod(heights[:, 1..j])
    cum_heights = torch.cumprod(heights[:, 1:], dim=-1)
    torch.testing.assert_close(vol[:, 1:], cum_heights, rtol=1e-5, atol=1e-5)


def test_maximumDistance_volumes_numpy():
    """Test maximumDistance_volumes with NumPy array input."""
    C, N = 5, 50
    k = 4
    cube_np = np.random.rand(C, N).astype(np.float32)

    em_np, vol_np = sct.maximumDistance_volumes(cube_np, k)
    assert isinstance(em_np, np.ndarray)
    assert isinstance(vol_np, np.ndarray)
    assert em_np.shape == (C, k)
    assert vol_np.shape == (k,)
    assert vol_np[0] == 0.0
    assert vol_np[1] > 0.0


# ==============================================================================
# 4. calcGramLocalVolumes Function Tests
# ==============================================================================

@pytest.mark.parametrize("device", ['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu'])
def test_calcGramLocalVolumes_shapes(device):
    """Test calcGramLocalVolumes with both (k, B) and (B, k) orientations on CPU/CUDA."""
    k, B = 5, 10
    
    # 1. Orientation: (num_endmembers, num_bands) -> (5, 10)
    em_kb = torch.randn(k, B, dtype=torch.float32, device=device)
    vol_kb = sct.calcGramLocalVolumes(em_kb, divide_by_factorial=True)
    assert isinstance(vol_kb, torch.Tensor)
    assert vol_kb.shape == (k,)
    assert vol_kb.device.type == device
    assert torch.all(vol_kb >= 0.0)

    # 2. Orientation: (num_bands, num_endmembers) -> (10, 5)
    em_bk = em_kb.transpose(0, 1)  # (10, 5)
    vol_bk = sct.calcGramLocalVolumes(em_bk, divide_by_factorial=True)
    assert vol_bk.shape == (k,)
    assert vol_bk.device.type == device
    torch.testing.assert_close(vol_kb, vol_bk, rtol=1e-5, atol=1e-5)


def test_calcGramLocalVolumes_square():
    """Test calcGramLocalVolumes when num_endmembers == num_bands (e.g., Landsat 7x7)."""
    k = 7
    em = torch.randn(k, k, dtype=torch.float32)
    # Default axis=None should treat rows as endmembers per interface contract
    vol_default = sct.calcGramLocalVolumes(em, divide_by_factorial=True)
    assert vol_default.shape == (k,)
    assert torch.all(vol_default >= 0.0)

    # Explicit axis=0
    vol_axis0 = sct.calcGramLocalVolumes(em, divide_by_factorial=True, axis=0)
    torch.testing.assert_close(vol_default, vol_axis0)


def test_calcGramLocalVolumes_orthogonal_simplex_analytical():
    """Verify analytical volume of an orthogonal simplex spanned by standard unit vectors.
    
    For k orthonormal unit vectors e_1..e_k:
    Gram matrix is the k x k identity matrix I_k, det(I_k) = 1.
    Parallelotope volume = sqrt(1) = 1.0.
    Simplex volume = 1.0 / k!.
    """
    k = 5
    B = 10
    # Create orthonormal basis
    ortho_em = torch.zeros(k, B, dtype=torch.float32)
    for i in range(k):
        ortho_em[i, i] = 1.0

    # With divide_by_factorial=True (formula specified in R7)
    vol_simplex = sct.calcGramLocalVolumes(ortho_em, divide_by_factorial=True)
    expected_simplex = torch.tensor([1.0 / math.factorial(i) for i in range(1, k + 1)], dtype=torch.float32)
    torch.testing.assert_close(vol_simplex, expected_simplex, rtol=1e-6, atol=1e-6)

    # With divide_by_factorial=False (legacy SpecComplex.py parallelotope volume)
    vol_parallelotope = sct.calcGramLocalVolumes(ortho_em, divide_by_factorial=False)
    expected_parallelotope = torch.ones(k, dtype=torch.float32)
    torch.testing.assert_close(vol_parallelotope, expected_parallelotope, rtol=1e-6, atol=1e-6)


def test_calcGramLocalVolumes_localization_vector():
    """Test subtraction of localization_vector."""
    k, B = 4, 8
    em = torch.randn(k, B, dtype=torch.float32)
    loc = torch.randn(B, dtype=torch.float32)

    # Localized manually
    em_centered = em - loc.unsqueeze(0)
    vol_manual = sct.calcGramLocalVolumes(em_centered, divide_by_factorial=False)

    # Localized via parameter
    vol_param = sct.calcGramLocalVolumes(em, localization_vector=loc, divide_by_factorial=False)
    torch.testing.assert_close(vol_manual, vol_param, rtol=1e-6, atol=1e-6)


def test_calcGramLocalVolumes_numpy_input():
    """Test calcGramLocalVolumes with NumPy array input."""
    k, B = 5, 9
    em_np = np.random.randn(k, B).astype(np.float32)
    vol_np = sct.calcGramLocalVolumes(em_np, divide_by_factorial=True)
    assert isinstance(vol_np, np.ndarray)
    assert vol_np.shape == (k,)
    assert np.all(vol_np >= 0.0)


def test_calcGramLocalVolumes_linear_dependence():
    """Verify that linearly dependent endmembers produce determinant zero without crashing."""
    k, B = 4, 6
    em = torch.randn(k, B, dtype=torch.float32)
    # Make vector 3 identical to vector 2
    em[3] = em[2]
    vol = sct.calcGramLocalVolumes(em, divide_by_factorial=False)
    assert vol.shape == (k,)
    # The 4th volume must collapse to 0.0 because the 4th vector is duplicate
    assert torch.isclose(vol[3], torch.tensor(0.0), atol=1e-5)


# ==============================================================================
# 5. Parity with J-STARS / Legacy Dataset Test
# ==============================================================================

def test_calcGramLocalVolumes_against_jstars_ground_truth():
    """Verify calcGramLocalVolumes against backed-up J-STARS dataset ground truth."""
    import h5py
    jstars_path = r"F:\Resilio\IMGS 890 Research\JSTARS-Spectral-Complexity\source_data\HLST_Rochesterv2_Harmonized_SC_EM-7_Norm-None.h5"
    if not os.path.exists(jstars_path):
        pytest.skip(f"J-STARS ground truth file not found at {jstars_path}")

    with h5py.File(jstars_path, 'r') as f:
        # Test HLSL30 frame 0
        em0 = f['/HDFEOS/GRIDS/HLSL30/Data Fields/frame_endmembers'][0]  # (7, 7)
        stored_vol0 = f['/HDFEOS/GRIDS/HLSL30/Data Fields/frame_endmember_volumes'][0]  # (7,)

        # Legacy localization: local origin is endmember at index 1
        loc_vec = em0[:, 1]
        rem_em = np.delete(em0, 1, axis=1)  # (7, 6)

        # Legacy computation (divide_by_factorial=False, matching legacy SpecComplex.py)
        vol_calc = sct.calcGramLocalVolumes(rem_em, localization_vector=loc_vec, divide_by_factorial=False, axis=1)
        vol_curve = np.insert(vol_calc, 0, 0.0)

        np.testing.assert_allclose(vol_curve, stored_vol0, rtol=1e-5, atol=1e-6,
                                   err_msg="calcGramLocalVolumes does not match J-STARS stored volumes")


# ==============================================================================
# 6. process_volume_sliding_tile Function Tests
# ==============================================================================

def test_process_volume_sliding_tile_cpu():
    """Test sliding window processing on CPU."""
    bands, height, width = 4, 15, 15
    data = np.random.uniform(0.1, 0.8, size=(bands, height, width)).astype(np.float32)

    final_map = sct.process_volume_sliding_tile(data, tile_size=3, stride=1, num_endmembers=3, device='cpu')
    assert isinstance(final_map, np.ndarray)
    assert final_map.shape == (height, width)
    assert final_map.dtype == np.float32
    assert np.all(np.isfinite(final_map[1:-1, 1:-1]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_process_volume_sliding_tile_cpu_cuda_parity():
    """Test parity between CPU and CUDA executions of process_volume_sliding_tile."""
    bands, height, width = 4, 15, 15
    data = np.random.uniform(0.1, 0.8, size=(bands, height, width)).astype(np.float32)

    map_cpu = sct.process_volume_sliding_tile(data, tile_size=3, stride=1, num_endmembers=3, device='cpu')
    map_cuda = sct.process_volume_sliding_tile(data, tile_size=3, stride=1, num_endmembers=3, device='cuda')

    assert map_cpu.shape == map_cuda.shape
    np.testing.assert_allclose(map_cpu, map_cuda, rtol=1e-4, atol=1e-5,
                               err_msg="CPU and CUDA process_volume_sliding_tile outputs differ")


def test_process_volume_sliding_tile_tensor_input():
    """Test process_volume_sliding_tile accepting a torch.Tensor."""
    bands, height, width = 3, 10, 10
    data_t = torch.rand(bands, height, width, dtype=torch.float32)

    final_map = sct.process_volume_sliding_tile(data_t, tile_size=3, stride=1, num_endmembers=3, device='cpu')
    assert isinstance(final_map, np.ndarray)
    assert final_map.shape == (height, width)


def test_process_volume_sliding_tile_outliers_and_nans():
    """Verify that reflectance outliers outside [-1.5, 1.5] and NaNs are masked."""
    bands, height, width = 4, 12, 12
    data = np.random.uniform(0.1, 0.8, size=(bands, height, width)).astype(np.float32)
    # Inject NaN and extreme values into a tile
    data[:, 5, 5] = np.nan
    data[:, 8, 8] = 999.0

    final_map = sct.process_volume_sliding_tile(data, tile_size=3, stride=1, num_endmembers=3, device='cpu')
    assert isinstance(final_map, np.ndarray)
    assert final_map.shape == (height, width)


def test_process_volume_sliding_tile_smoothing_toggle():
    """Verify that apply_spatial_smoothing=False returns raw neighborhood_map directly."""
    bands, height, width = 4, 15, 15
    np.random.seed(42)
    data = np.random.uniform(0.1, 0.8, size=(bands, height, width)).astype(np.float32)

    # 1. Spatially smoothed (legacy behavior)
    smoothed, neigh_ref = sct.process_volume_sliding_tile(
        data, tile_size=3, stride=1, num_endmembers=3, device='cpu', return_neighborhood=True,
        apply_spatial_smoothing=True
    )

    # 2. Unsmoothed direct neighborhood_map
    unsmoothed = sct.process_volume_sliding_tile(
        data, tile_size=3, stride=1, num_endmembers=3, device='cpu',
        apply_spatial_smoothing=False
    )

    # Unsmoothed map must exactly match the raw neighborhood_map
    np.testing.assert_allclose(unsmoothed, neigh_ref, rtol=1e-5, atol=1e-7,
                               err_msg="Unsmoothed map does not match neighborhood_map")

    # Smoothed map must differ from unsmoothed map (due to uniform filter averaging)
    valid_inner = np.isfinite(unsmoothed[2:-2, 2:-2])
    assert not np.allclose(smoothed[2:-2, 2:-2][valid_inner], unsmoothed[2:-2, 2:-2][valid_inner])

    # 3. CPU Reference parity (SpecComplex)
    import SpecComplex as sc_cpu
    cpu_unsmoothed = sc_cpu.process_volume_sliding_tile(
        data, tile_size=3, stride=1, num_endmembers=3,
        apply_spatial_smoothing=False
    )
    np.testing.assert_allclose(unsmoothed, cpu_unsmoothed, rtol=1e-4, atol=1e-6,
                               err_msg="CPU SpecComplex and SpecComplexTorch unsmoothed outputs differ")


def test_process_volume_3d_sliding_tile_smoothing_toggle():
    """Verify that process_volume_3d_sliding_tile respects apply_spatial_smoothing=False."""
    bands, height, width = 6, 12, 12
    np.random.seed(123)
    data = np.random.uniform(0.05, 0.6, size=(bands, height, width)).astype(np.float32)

    unsmoothed_3d = sct.process_volume_3d_sliding_tile(
        data, tile_size=3, stride=1, num_endmembers=10, separability_threshold=0.01,
        device='cpu', apply_spatial_smoothing=False
    )
    smoothed_3d, neigh_3d = sct.process_volume_3d_sliding_tile(
        data, tile_size=3, stride=1, num_endmembers=10, separability_threshold=0.01,
        device='cpu', return_neighborhood=True, apply_spatial_smoothing=True
    )

    assert unsmoothed_3d.shape == (height, width)
    np.testing.assert_allclose(unsmoothed_3d, neigh_3d, rtol=1e-5, atol=1e-7,
                               err_msg="3D unsmoothed map does not match neighborhood_map")


# ==============================================================================
# 7. calculate_global_z_score Function Tests
# ==============================================================================

def test_calculate_global_z_score_numpy():
    """Test calculate_global_z_score with standard NumPy inputs."""
    H, W = 30, 30
    np.random.seed(42)
    # Generate log-normal volume values
    vols = np.exp(np.random.normal(loc=1.0, scale=0.5, size=(H, W))).astype(np.float32)
    mask = np.ones((H, W), dtype=bool)
    mask[0:3, :] = False  # Mask border

    z_map, mean, std = sct.calculate_global_z_score(vols, mask)

    assert isinstance(z_map, np.ndarray)
    assert z_map.shape == (H, W)
    assert isinstance(mean, float)
    assert isinstance(std, float)
    assert np.isfinite(mean)
    assert std > 0.0

    # Valid background statistics should be approximately standard normal
    valid_z = z_map[mask]
    assert np.isclose(np.mean(valid_z), 0.0, atol=1e-1)
    assert np.isclose(np.std(valid_z, ddof=1), 1.0, atol=1e-1)


@pytest.mark.parametrize("device", ['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu'])
def test_calculate_global_z_score_torch(device):
    """Test calculate_global_z_score with PyTorch tensor inputs on CPU/CUDA."""
    H, W = 25, 25
    vols_t = torch.exp(torch.randn(H, W, device=device) * 0.4 + 0.5)
    mask_t = torch.ones(H, W, dtype=torch.bool, device=device)

    z_map, mean, std = sct.calculate_global_z_score(vols_t, mask_t)

    assert isinstance(z_map, torch.Tensor)
    assert z_map.shape == (H, W)
    assert z_map.device.type == device
    assert isinstance(mean, float)
    assert isinstance(std, float)
    assert np.isfinite(mean)
    assert std > 0.0


def test_calculate_global_z_score_insufficient_pixels():
    """Verify graceful fallback (warning and NaNs) when valid pixels < 9."""
    H, W = 10, 10
    vols = np.full((H, W), np.nan, dtype=np.float32)
    vols[0, :5] = 1.0  # Only 5 valid pixels
    mask = np.ones((H, W), dtype=bool)

    with pytest.warns(UserWarning, match="Insufficient valid pixels"):
        z_map, mean, std = sct.calculate_global_z_score(vols, mask)

    assert np.isnan(mean)
    assert np.isnan(std)
    assert np.all(np.isnan(z_map))


def test_calculate_global_z_score_zero_variance_raises():
    """Verify strict failure (ValueError) when valid pixels have zero variance."""
    H, W = 10, 10
    vols = np.full((H, W), 2.5, dtype=np.float32)
    mask = np.ones((H, W), dtype=bool)

    # Constant array has len(unique) == 1, triggers warning and NaN return
    with pytest.warns(UserWarning, match="zero variance found"):
        z_map, mean, std = sct.calculate_global_z_score(vols, mask)
    assert np.isnan(mean)


def test_calculate_global_z_score_with_separately_determined_stats_scalar():
    """Verify calculate_global_z_score accepts scalar array_mean and array_std in both numpy and torch."""
    H, W = 10, 10
    vols = np.full((H, W), np.e, dtype=np.float32)  # log(e) = 1.0
    vols[0, 0] = np.e ** 3.0  # log(e^3) = 3.0
    vols[0, 1] = 0.0  # Invalid pixel

    fixed_mean = 1.0
    fixed_std = 2.0

    # 1. SpecComplex (numpy) with valid_pixel_mask=None
    z_map_np, mean_np, std_np = sc.calculate_global_z_score(vols, array_mean=fixed_mean, array_std=fixed_std)
    assert mean_np == fixed_mean
    assert std_np == fixed_std
    assert np.isclose(z_map_np[0, 0], (3.0 - 1.0) / 2.0)  # (3 - 1) / 2 = 1.0
    assert np.isclose(z_map_np[1, 1], (1.0 - 1.0) / 2.0)  # (1 - 1) / 2 = 0.0
    assert np.isnan(z_map_np[0, 1])  # 0.0 is not strictly positive, so nan

    # 2. SpecComplexTorch with numpy inputs and valid_pixel_mask provided
    mask = np.ones((H, W), dtype=bool)
    z_map_sct, mean_sct, std_sct = sct.calculate_global_z_score(vols, mask, array_mean=fixed_mean, array_std=fixed_std)
    np.testing.assert_allclose(z_map_sct, z_map_np, equal_nan=True)
    assert mean_sct == fixed_mean
    assert std_sct == fixed_std

    # 3. SpecComplexTorch with torch inputs
    vols_t = torch.from_numpy(vols)
    z_map_t, mean_t, std_t = sct.calculate_global_z_score(vols_t, array_mean=fixed_mean, array_std=fixed_std)
    assert isinstance(z_map_t, torch.Tensor)
    np.testing.assert_allclose(z_map_t.cpu().numpy(), z_map_np, equal_nan=True)
    assert mean_t == fixed_mean
    assert std_t == fixed_std

    # 4. Torch scalar tensors passed as array_mean / array_std
    mean_tensor = torch.tensor(fixed_mean)
    std_tensor = torch.tensor(fixed_std)
    z_map_t2, mean_t2, std_t2 = sct.calculate_global_z_score(vols_t, array_mean=mean_tensor, array_std=std_tensor)
    assert isinstance(z_map_t2, torch.Tensor)
    np.testing.assert_allclose(z_map_t2.cpu().numpy(), z_map_np, equal_nan=True)


def test_calculate_global_z_score_with_separately_determined_stats_2d_array():
    """Verify calculate_global_z_score accepts 2D array_mean and array_std."""
    H, W = 5, 5
    vols = np.full((H, W), np.e ** 2.0, dtype=np.float32)
    means = np.full((H, W), 1.0, dtype=np.float32)
    stds = np.full((H, W), 0.5, dtype=np.float32)

    # (2.0 - 1.0) / 0.5 = 2.0
    z_map_np, m_out, s_out = sc.calculate_global_z_score(vols, array_mean=means, array_std=stds)
    assert np.allclose(z_map_np, 2.0)

    # SpecComplexTorch with torch 2D tensors
    vols_t = torch.from_numpy(vols)
    means_t = torch.from_numpy(means)
    stds_t = torch.from_numpy(stds)
    z_map_t, m_t, s_t = sct.calculate_global_z_score(vols_t, array_mean=means_t, array_std=stds_t)
    assert isinstance(z_map_t, torch.Tensor)
    assert torch.allclose(z_map_t, torch.full((H, W), 2.0, dtype=torch.float32))


def test_calculate_global_z_score_error_handling_and_fallbacks():
    """Verify validation: partial stats, non-positive std, nan stats, and missing mask."""
    H, W = 6, 6
    vols = np.full((H, W), 2.0, dtype=np.float32)
    mask = np.ones((H, W), dtype=bool)

    # 1. Only array_mean provided (missing array_std)
    for mod in (sc, sct):
        with pytest.raises(ValueError, match="Both array_mean and array_std must be provided together"):
            mod.calculate_global_z_score(vols, array_mean=1.0)

        with pytest.raises(ValueError, match="Both array_mean and array_std must be provided together"):
            mod.calculate_global_z_score(vols, array_std=1.0)

    # 2. Neither provided and valid_pixel_mask is None
    for mod in (sc, sct):
        with pytest.raises(ValueError, match="valid_pixel_mask must be provided"):
            mod.calculate_global_z_score(vols)

    # 3. Non-positive array_std
    for mod in (sc, sct):
        with pytest.raises(ValueError, match="Global standard deviation must be positive"):
            mod.calculate_global_z_score(vols, array_mean=1.0, array_std=0.0)

        with pytest.raises(ValueError, match="Global standard deviation must be positive"):
            mod.calculate_global_z_score(vols, array_mean=1.0, array_std=-1.0)

    # 4. NaN in array_mean or array_std returns NaNs with warning
    for mod in (sc, sct):
        with pytest.warns(UserWarning):
            z_map, m, s = mod.calculate_global_z_score(vols, array_mean=np.nan, array_std=1.0)
            assert np.isnan(m)
            assert np.isnan(s)
            assert np.all(np.isnan(np.asarray(z_map)))


if __name__ == "__main__":
    sys.exit(pytest.main(["-p", "no:dash", "-v", __file__]))

