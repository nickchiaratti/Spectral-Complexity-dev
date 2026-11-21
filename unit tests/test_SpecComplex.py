import pytest
import numpy as np
import torch
import h5py
import sys
import os
import math

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import SpecComplex
import SpecComplexTorch
from maximumDistance_Volumes import maximumDistance_parallelotopeVolumes

FILE_PATH = r"C:\satelliteImagery\HLST30\HLST_Rochesterv2.h5"

def get_test_cases():
    if not os.path.exists(FILE_PATH):
        return []
    
    test_cases = []
    try:
        with h5py.File(FILE_PATH, 'r') as f:
            grids = f.get('HDFEOS/GRIDS')
            if not grids:
                return []
            
            for grid_name in grids.keys():
                try:
                    ds = f[f'HDFEOS/GRIDS/{grid_name}/Data Fields/surface_reflectance']
                    num_frames = ds.shape[0]
                    
                    if num_frames < 10:
                        frame_indices = np.linspace(0,num_frames-1, dtype=int).tolist()
                    else:
                        num_test_frames = max(1, num_frames // 10)
                        # evenly distributed over time indices
                        frame_indices = np.linspace(0, num_frames - 1, num_test_frames, dtype=int).tolist()
                        
                    for idx in sorted(set(frame_indices)):
                        test_cases.append((grid_name, idx))
                except Exception:
                    continue
    except Exception:
        pass
        
    return test_cases

TEST_CASES = get_test_cases()


@pytest.fixture(params=TEST_CASES, ids=[f"{g}-f{i}" for g, i in TEST_CASES])
def real_data(request):
    grid_name, frame_idx = request.param
    
    with h5py.File(FILE_PATH, 'r') as f:
        ds = f[f'HDFEOS/GRIDS/{grid_name}/Data Fields/surface_reflectance']
        frame_data = ds[frame_idx]
        
        # Get scale to float from attributes, default to 0.0001 if not found
        scale = ds.attrs.get('scale_to_float', 0.0001)
        
        # Data is [bands, rows, cols]
        img = np.transpose(frame_data, (1, 2, 0)).astype(np.float32)
        
        if 'good_wavelengths' in ds.attrs:
            gw_mask = ds.attrs['good_wavelengths'].astype(bool)
            img = img[:, :, gw_mask]
        
        # Scale surface reflectance to [0, 1] range using the attribute
        img = img * scale
        
        img_flat = img.reshape((-1, img.shape[2]))
        
        # A valid pixel should have no negative values and no values > 1 in any band
        valid_mask = np.all((img_flat > 0) & (img_flat <= 1), axis=1)
        
        # Fallback if too strict
        if np.sum(valid_mask) < 25:
            valid_mask = np.all((img_flat >= 0) & (img_flat <= 1), axis=1)
            
        # Incorporate common_mask from ARD HDF5 file (True = Invalid/Masked, False = Valid)
        if 'common_mask' in ds.parent:
            cm = ds.parent['common_mask'][frame_idx]
            valid_mask = valid_mask & (~cm.flatten())
            
        valid_pixels = img_flat[valid_mask]
        
        if len(valid_pixels) < 25:
            pytest.skip(f"Not enough valid pixels in {grid_name} frame {frame_idx} for testing. Found {len(valid_pixels)}")
            
        # Get maximum perfect square of valid pixels to reshape into 2D grid
        num_valid = len(valid_pixels)
        if num_valid > 2500:
            num_valid = 2500
        side = int(math.sqrt(num_valid))
        total_pixels = side * side
        
        test_img = valid_pixels[:total_pixels].reshape((side, side, img.shape[2]))
        return test_img

def test_maximumDistance(real_data):
    num_endmembers = 7
    img = real_data
    rows, cols, bands = img.shape
    
    # 1. Run original Numpy version
    em_np, em_idx_np = SpecComplex.maximumDistance(img, num_endmembers)
    
    # 2. Run Torch version
    # The Numpy version reshapes with order="F". We replicate the exact layout to ensure
    # that any ties in argmax resolve to the identical flat pixel index.
    img_flat = np.reshape(img, (rows * cols, bands), order="F")
    data_torch = torch.from_numpy(img_flat).transpose(0, 1).unsqueeze(0) # (1, bands, pixels)
    
    valid_pixel_mask = ~torch.isnan(data_torch).any(dim=1)
    
    em_torch = SpecComplexTorch.maximumDistance_torch(
        data_torch, num_endmembers, valid_pixel_mask
    )
    
    # 3. Compare Results
    em_torch_np = em_torch.squeeze(0).cpu().numpy()
    
    em_diff = (em_np - em_torch_np).flatten()
    print(f"Mean EM: {np.mean(em_np):.10e}, Mean Diff: {np.mean(em_diff):.10e}, Var Diff: {np.var(em_diff):.10e}")
    
    np.testing.assert_allclose(em_np, em_torch_np, rtol=1e-8, atol=1e-5, 
                               err_msg="Maximum distance endmembers mismatch")


def test_calcGramLocalVolumes(real_data):
    num_endmembers = 7
    img = real_data
    
    # Get endmembers using Numpy version to test volume function with same inputs
    em_np, _ = SpecComplex.maximumDistance(img, num_endmembers)
    bands = em_np.shape[0]
    
    # Mean vector for localization
    img_flat = img.reshape((-1, bands))
    valid_pixels = img_flat[~np.isnan(img_flat).any(axis=1)]
    mean_vec_np = np.mean(valid_pixels, axis=0)
    
    # 1. Run Numpy version
    vol_np = SpecComplex.calcGramLocalVolumes(em_np, mean_vec_np)
    
    # 2. Run Torch version
    em_torch = torch.from_numpy(em_np).unsqueeze(0) # (1, bands, num_endmembers)
    mean_vec_torch = torch.from_numpy(mean_vec_np).unsqueeze(0) # (1, bands)
    
    vol_torch = SpecComplexTorch.calcGramLocalVolumes_QR_torch(em_torch, mean_vec_torch)
    vol_torch_np = vol_torch.squeeze(0).cpu().numpy()
    
    # 3. Compare Results
    vol_diff = (vol_np - vol_torch_np).flatten()
    print(f"Mean Vol: {np.mean(vol_np):.10e}, Mean Diff: {np.mean(vol_diff):.10e}, Var Diff: {np.var(vol_diff):.10e}")
    
    np.testing.assert_allclose(vol_np, vol_torch_np, rtol=1e-8, atol=1e-5,
                               err_msg="Local volumes mismatch")

def test_maximumDistance_volumes(real_data):
    num_endmembers = 7
    img = real_data
    rows, cols, bands = img.shape
    
    # 1. Run separate methods
    em_sep, _ = SpecComplex.maximumDistance(img, num_endmembers)
    
    localizationVec = em_sep[:, 1]
    remainingEndmembers = np.delete(em_sep, 1, axis=1)
    vol_sep = SpecComplex.calcGramLocalVolumes(remainingEndmembers, localizationVec)
    vol_sep = np.insert(vol_sep, 0, 0.0)
    
    # 2. Run combined method
    img_flat = np.reshape(img, (rows * cols, bands), order="F")
    valid_mask = ~np.isnan(img_flat).any(axis=1)
    valid_pixels = img_flat[valid_mask]
    pixels_t = valid_pixels.T # (bands, num_pixels)
    
    em_comb, vol_comb = SpecComplex.maximumDistance_volumes(pixels_t, num_endmembers)
    
    # 3. Compare Results
    em_diff = (em_sep - em_comb).flatten()
    print(f"Mean EM: {np.mean(em_sep):.10e}, Mean Diff: {np.mean(em_diff):.10e}, Var Diff: {np.var(em_diff):.10e}")
    np.testing.assert_allclose(em_sep, em_comb, rtol=1e-8, atol=1e-5,
                               err_msg="Endmembers mismatch between separate and combined methods")
                               
    vol_diff = (vol_sep - vol_comb).flatten()
    print(f"Mean Vol: {np.mean(vol_sep):.10e}, Mean Diff: {np.mean(vol_diff):.10e}, Var Diff: {np.var(vol_diff):.10e}")
    np.testing.assert_allclose(vol_sep, vol_comb, rtol=1e-8, atol=1e-5,
                               err_msg="Volumes mismatch between separate and combined methods")

def test_maximumDistance_volumes_torch(real_data):
    num_endmembers = 7
    img = real_data
    rows, cols, bands = img.shape
    
    # 1. Run Numpy combined method
    img_flat = np.reshape(img, (rows * cols, bands), order="F")
    valid_mask = ~np.isnan(img_flat).any(axis=1)
    valid_pixels = img_flat[valid_mask]
    pixels_t = valid_pixels.T # (bands, num_pixels)
    
    em_np, vol_np = SpecComplex.maximumDistance_volumes(pixels_t, num_endmembers)
    
    # 2. Run Torch combined method
    data_torch = torch.from_numpy(valid_pixels).transpose(0, 1).unsqueeze(0) # (1, bands, valid_pixels)
    
    em_torch, vol_torch = SpecComplexTorch.maximumDistance_volumes_torch(
        data_torch, num_endmembers
    )
    
    em_torch_np = em_torch.squeeze(0).cpu().numpy()
    vol_torch_np = vol_torch.squeeze(0).cpu().numpy()
    
    # 3. Compare Results
    em_diff = (em_np - em_torch_np).flatten()
    print(f"Mean EM NP: {np.mean(em_np):.10e}, Mean Torch EM: {np.mean(em_torch_np):.10e}")
    print(f"Mean EM Diff: {np.mean(em_diff):.10e}, Var EM Diff: {np.var(em_diff):.10e}")
    
    np.testing.assert_allclose(em_np, em_torch_np, rtol=1e-8, atol=1e-5,
                               err_msg="Endmembers mismatch between Numpy and Torch maximumDistance_volumes")
                               
    vol_diff = (vol_np - vol_torch_np).flatten()
    print(f"Mean Vol NP: {np.mean(vol_np):.10e}, Mean Torch Vol: {np.mean(vol_torch_np):.10e}")
    print(f"Mean Vol Diff: {np.mean(vol_diff):.10e}, Var Vol Diff: {np.var(vol_diff):.10e}")
    
    np.testing.assert_allclose(vol_np, vol_torch_np, rtol=1e-8, atol=1e-5,
                               err_msg="Volumes mismatch between Numpy and Torch maximumDistance_volumes")

def test_maximumDistance_volumes_torch_heights(real_data):
    num_endmembers = 7
    img = real_data
    rows, cols, bands = img.shape
    
    img_flat = np.reshape(img, (rows * cols, bands), order="F")
    valid_mask = ~np.isnan(img_flat).any(axis=1)
    valid_pixels = img_flat[valid_mask]
    pixels_t = valid_pixels.T  # (bands, num_pixels)
    
    # 1. Run Numpy test method
    em_np, vol_np, h_np = SpecComplex.maximumDistance_volumes_test(pixels_t, num_endmembers)
    
    # 2. Run Torch test method
    data_torch = torch.from_numpy(valid_pixels).transpose(0, 1).unsqueeze(0)  # (1, bands, valid_pixels)
    em_torch, vol_torch, h_torch = SpecComplexTorch.maximumDistance_volumes_torch_test(
        data_torch, num_endmembers
    )
    
    em_torch_np = em_torch.squeeze(0).cpu().numpy()
    vol_torch_np = vol_torch.squeeze(0).cpu().numpy()
    h_torch_np = h_torch.squeeze(0).cpu().numpy()
    
    # 3. Verify internal consistency: cumulative product of heights equals volumes
    expected_vol_torch = torch.cumprod(h_torch[:, 1:], dim=-1)
    torch.testing.assert_close(
        vol_torch[:, 1:], expected_vol_torch, rtol=1e-6, atol=1e-6,
        msg="Cumulative product of heights does not match volumes in Torch"
    )
    
    # 4. Compare Torch vs Numpy
    np.testing.assert_allclose(h_np, h_torch_np, rtol=1e-8, atol=1e-5,
                               err_msg="Heights mismatch between Numpy and Torch")
    np.testing.assert_allclose(vol_np, vol_torch_np, rtol=1e-8, atol=1e-5,
                               err_msg="Volumes mismatch between Numpy and Torch")
    np.testing.assert_allclose(em_np, em_torch_np, rtol=1e-8, atol=1e-5,
                               err_msg="Endmembers mismatch between Numpy and Torch")

def test_maximumDistance_parallelotopeVolumes(real_data):
    """
    Validates equivalence of maximumDistance_parallelotopeVolumes (3D input)
    against original separate maximumDistance and calcGramLocalVolumes implementations.
    Both endmember values and linear scale volume results must be within 1e-5.
    """
    num_endmembers = 7
    img = real_data
    
    # 1. Run separate original methods
    em_sep, _ = SpecComplex.maximumDistance(img, num_endmembers)
    
    localizationVec = em_sep[:, 1]
    remainingEndmembers = np.delete(em_sep, 1, axis=1)
    vol_sep = SpecComplex.calcGramLocalVolumes(remainingEndmembers, localizationVec)
    vol_sep = np.insert(vol_sep, 0, 0.0)
    
    # 2. Run maximumDistance_parallelotopeVolumes with 3D input
    em_par, vol_log_par, heights_par, idx_par = maximumDistance_parallelotopeVolumes(
        img, num_endmembers
    )
    vol_linear_par = np.exp(vol_log_par)
    
    # 3. Compare Results: Endmembers within 1e-5
    em_diff = (em_sep - em_par).flatten()
    print(f"Mean EM: {np.mean(em_sep):.10e}, Mean Diff: {np.mean(em_diff):.10e}, Max Abs Diff: {np.max(np.abs(em_diff)):.10e}, Var Diff: {np.var(em_diff):.10e}")
    np.testing.assert_allclose(
        em_sep, em_par, rtol=1e-8, atol=1e-5,
        err_msg="Endmembers mismatch between maximumDistance and maximumDistance_parallelotopeVolumes"
    )
    
    # 4. Compare Results: Linear scale volumes within 1e-5
    vol_diff = (vol_sep - vol_linear_par).flatten()
    print(f"Mean Vol: {np.mean(vol_sep):.10e}, Mean Diff: {np.mean(vol_diff):.10e}, Max Abs Diff: {np.max(np.abs(vol_diff)):.10e}, Var Diff: {np.var(vol_diff):.10e}")
    np.testing.assert_allclose(
        vol_sep, vol_linear_par, rtol=1e-8, atol=1e-5,
        err_msg="Linear scale volumes mismatch between calcGramLocalVolumes and maximumDistance_parallelotopeVolumes"
    )
    
    # 5. Verify internal consistency: cumulative product of heights matches linear volume
    expected_vol = np.cumprod(heights_par[1:])
    np.testing.assert_allclose(
        vol_linear_par[1:], expected_vol, rtol=1e-8, atol=1e-5,
        err_msg="Cumulative product of heights does not match linear volume in maximumDistance_parallelotopeVolumes"
    )


def test_maximumDistance_parallelotopeVolumes_2d(real_data):
    """
    Validates equivalence of maximumDistance_parallelotopeVolumes (2D (bands, pixels) input)
    against original separate maximumDistance and calcGramLocalVolumes implementations.
    Both endmember values and linear scale volume results must be within 1e-5.
    """
    num_endmembers = 7
    img = real_data
    rows, cols, bands = img.shape
    
    # 1. Run separate original methods
    em_sep, _ = SpecComplex.maximumDistance(img, num_endmembers)
    
    localizationVec = em_sep[:, 1]
    remainingEndmembers = np.delete(em_sep, 1, axis=1)
    vol_sep = SpecComplex.calcGramLocalVolumes(remainingEndmembers, localizationVec)
    vol_sep = np.insert(vol_sep, 0, 0.0)
    
    # 2. Run maximumDistance_parallelotopeVolumes with 2D input
    img_flat = np.reshape(img, (rows * cols, bands), order="F")
    valid_mask = ~np.isnan(img_flat).any(axis=1)
    valid_pixels = img_flat[valid_mask]
    pixels_t = valid_pixels.T  # (bands, num_pixels)
    
    em_par, vol_log_par, heights_par, idx_par = maximumDistance_parallelotopeVolumes(
        pixels_t, num_endmembers
    )
    vol_linear_par = np.exp(vol_log_par)
    
    # 3. Compare Results
    em_diff = (em_sep - em_par).flatten()
    print(f"Mean EM 2D: {np.mean(em_sep):.10e}, Mean Diff: {np.mean(em_diff):.10e}, Max Abs Diff: {np.max(np.abs(em_diff)):.10e}, Var Diff: {np.var(em_diff):.10e}")
    np.testing.assert_allclose(
        em_sep, em_par, rtol=1e-8, atol=1e-5,
        err_msg="Endmembers mismatch between maximumDistance and 2D maximumDistance_parallelotopeVolumes"
    )
    
    vol_diff = (vol_sep - vol_linear_par).flatten()
    print(f"Mean Vol 2D: {np.mean(vol_sep):.10e}, Mean Diff: {np.mean(vol_diff):.10e}, Max Abs Diff: {np.max(np.abs(vol_diff)):.10e}, Var Diff: {np.var(vol_diff):.10e}")
    np.testing.assert_allclose(
        vol_sep, vol_linear_par, rtol=1e-8, atol=1e-5,
        err_msg="Linear scale volumes mismatch between calcGramLocalVolumes and 2D maximumDistance_parallelotopeVolumes"
    )


def test_maximumDistance_parallelotopeVolumes_synthetic():
    """
    Standalone validation on synthetic spectral cube (independent of external HDF5 files).
    Validates both 3D and 2D inputs against maximumDistance and calcGramLocalVolumes within 1e-5.
    """
    np.random.seed(42)
    num_endmembers = 5
    img = np.random.uniform(0.05, 0.95, (10, 10, 15)).astype(np.float32)
    rows, cols, bands = img.shape
    
    # 1. Run separate original methods
    em_sep, _ = SpecComplex.maximumDistance(img, num_endmembers)
    localizationVec = em_sep[:, 1]
    remainingEndmembers = np.delete(em_sep, 1, axis=1)
    vol_sep = SpecComplex.calcGramLocalVolumes(remainingEndmembers, localizationVec)
    vol_sep = np.insert(vol_sep, 0, 0.0)
    
    # 2. Run 3D input
    em_par_3d, vol_log_3d, heights_3d, _ = maximumDistance_parallelotopeVolumes(img, num_endmembers)
    vol_linear_3d = np.exp(vol_log_3d)
    
    np.testing.assert_allclose(
        em_sep, em_par_3d, rtol=1e-8, atol=1e-5,
        err_msg="Synthetic: Endmembers mismatch on 3D input"
    )
    np.testing.assert_allclose(
        vol_sep, vol_linear_3d, rtol=1e-8, atol=1e-5,
        err_msg="Synthetic: Linear scale volumes mismatch on 3D input"
    )
    
    # 3. Run 2D input
    img_flat = np.reshape(img, (rows * cols, bands), order="F")
    pixels_t = img_flat.T
    em_par_2d, vol_log_2d, heights_2d, _ = maximumDistance_parallelotopeVolumes(pixels_t, num_endmembers)
    vol_linear_2d = np.exp(vol_log_2d)
    
    np.testing.assert_allclose(
        em_sep, em_par_2d, rtol=1e-8, atol=1e-5,
        err_msg="Synthetic: Endmembers mismatch on 2D input"
    )
    np.testing.assert_allclose(
        vol_sep, vol_linear_2d, rtol=1e-8, atol=1e-5,
        err_msg="Synthetic: Linear scale volumes mismatch on 2D input"
    )


def test_maximumDistance_cholesky(real_data):
    """
    Validates equivalence of maximumDistance_cholesky against maximumDistance_volumes
    on real satellite data filtered via common_mask.
    """
    num_endmembers = 7
    img = real_data
    rows, cols, bands = img.shape
    
    img_flat = np.reshape(img, (rows * cols, bands), order="F")
    pixels_t = img_flat.T  # (bands, num_pixels)
    
    em_chol, vol_chol, h_chol = SpecComplex.maximumDistance_cholesky(pixels_t, num_endmembers)
    em_vol, vol_vol, h_vol = SpecComplex.maximumDistance_volumes_test(pixels_t, num_endmembers)
    
    np.testing.assert_allclose(
        em_chol, em_vol, rtol=1e-5, atol=1e-5,
        err_msg="Endmembers mismatch between Cholesky and standard MaxD"
    )
    np.testing.assert_allclose(
        vol_chol, vol_vol, rtol=1e-5, atol=1e-5,
        err_msg="Volumes mismatch between Cholesky and standard MaxD"
    )
    np.testing.assert_allclose(
        h_chol, h_vol, rtol=1e-5, atol=1e-5,
        err_msg="Heights mismatch between Cholesky and standard MaxD"
    )
    
    expected_vol = np.cumprod(h_chol[1:])
    np.testing.assert_allclose(
        vol_chol[1:], expected_vol, rtol=1e-6, atol=1e-6,
        err_msg="Cumulative product of heights does not match volumes in Cholesky"
    )


def test_maximumDistance_cholesky_torch(real_data):
    """
    Validates equivalence of maximumDistance_cholesky_torch against maximumDistance_volumes_torch
    on real satellite data filtered via common_mask.
    """
    num_endmembers = 7
    img = real_data
    rows, cols, bands = img.shape
    
    img_flat = np.reshape(img, (rows * cols, bands), order="F")
    data_torch = torch.from_numpy(img_flat).transpose(0, 1).unsqueeze(0)  # (1, bands, pixels)
    
    em_chol, vol_chol, h_chol = SpecComplexTorch.maximumDistance_cholesky_torch(
        data_torch, num_endmembers, return_heights=True
    )
    em_vol, vol_vol, h_vol = SpecComplexTorch.maximumDistance_volumes_torch(
        data_torch, num_endmembers, return_heights=True
    )
    
    em_chol_np = em_chol.squeeze(0).cpu().numpy()
    vol_chol_np = vol_chol.squeeze(0).cpu().numpy()
    h_chol_np = h_chol.squeeze(0).cpu().numpy()
    
    em_vol_np = em_vol.squeeze(0).cpu().numpy()
    vol_vol_np = vol_vol.squeeze(0).cpu().numpy()
    h_vol_np = h_vol.squeeze(0).cpu().numpy()
    
    np.testing.assert_allclose(
        em_chol_np, em_vol_np, rtol=1e-5, atol=1e-5,
        err_msg="Endmembers mismatch between Torch Cholesky and Torch standard"
    )
    np.testing.assert_allclose(
        vol_chol_np, vol_vol_np, rtol=1e-5, atol=1e-5,
        err_msg="Volumes mismatch between Torch Cholesky and Torch standard"
    )
    np.testing.assert_allclose(
        h_chol_np, h_vol_np, rtol=1e-5, atol=1e-5,
        err_msg="Heights mismatch between Torch Cholesky and Torch standard"
    )


def test_process_cholesky_volume_sliding_tile_real_data():
    """
    Validates process_cholesky_volume_sliding_tile against process_volume_sliding_tile
    on real satellite imagery where common_mask is applied to exclude invalid pixels.
    Confirms identical invalid/NaN tile handling and relative difference within float32 limits.
    """
    if not os.path.exists(FILE_PATH):
        pytest.skip(f"Test file not found: {FILE_PATH}")
        
    with h5py.File(FILE_PATH, 'r') as f:
        if 'HDFEOS/GRIDS/HLSL30/Data Fields' not in f:
            pytest.skip("HLSL30 grid not found")
        df = f['HDFEOS/GRIDS/HLSL30/Data Fields']
        sr_ds = df['surface_reflectance']
        cm_ds = df['common_mask']
        scale = sr_ds.attrs.get('scale_to_float', 0.0001)
        
        frame_idx = None
        for i in range(min(10, cm_ds.shape[0])):
            cm = cm_ds[i]
            if np.sum(~cm) > 1000:
                frame_idx = i
                break
        if frame_idx is None:
            pytest.skip("No frame with valid common_mask pixels found")
            
        cm = cm_ds[frame_idx]
        sr = sr_ds[frame_idx].astype(np.float32) * scale
        sr[:, cm] = np.nan
        
        valid_coords = np.argwhere(~cm)
        r_mid, c_mid = valid_coords[len(valid_coords) // 2]
        r_start = max(0, r_mid - 50)
        c_start = max(0, c_mid - 50)
        sub_sr = sr[:, r_start:r_start+100, c_start:c_start+100]
        
        tile_size = 3
        stride = 1
        num_endmembers = 4
        
        final_std, neigh_std, esd_std, sii_std = SpecComplexTorch.process_volume_sliding_tile(
            sub_sr, tile_size, stride, num_endmembers, compute_esd=True, compute_sii=True, global_sigma_n=0.01
        )
        final_chol, neigh_chol, esd_chol, sii_chol = SpecComplexTorch.process_cholesky_volume_sliding_tile(
            sub_sr, tile_size, stride, num_endmembers, compute_esd=True, compute_sii=True, global_sigma_n=0.01
        )
        
        # 1. Both methods must produce identical NaN masks (strict tile validity match)
        np.testing.assert_array_equal(
            np.isnan(neigh_std), np.isnan(neigh_chol),
            err_msg="Tile validity mask differs between standard and Cholesky sliding tile"
        )
        
        # 2. On valid tiles, relative difference must be bounded by float32 precision
        valid_tiles = np.isfinite(neigh_std)
        assert np.sum(valid_tiles) > 0, "No valid tiles in sub-region"
        
        vol_std = neigh_std[valid_tiles]
        vol_chol = neigh_chol[valid_tiles]
        
        rel_diff = np.abs(vol_std - vol_chol) / np.maximum(vol_std, 1e-12)
        assert np.max(rel_diff) < 1e-5, f"Relative difference exceeds tolerance: max rel diff = {np.max(rel_diff)}"
        
        # 3. ESD counts must match exactly on all valid tiles
        np.testing.assert_array_equal(
            esd_std[valid_tiles], esd_chol[valid_tiles],
            err_msg="ESD counts mismatch on valid tiles"
        )
        
        # 4. SII values must match within tolerance
        np.testing.assert_allclose(
            sii_std[valid_tiles], sii_chol[valid_tiles], rtol=1e-5, atol=1e-5,
            err_msg="SII values mismatch on valid tiles"
        )


if __name__ == "__main__":
    import datetime
    
    log_file = os.path.join(os.path.dirname(__file__), "test_results.log")
    
    with open(log_file, "a") as f:
        f.write(f"\n{'='*50}\n")
        f.write(f"Test Run: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*50}\n")
        
    class CustomLogReporter:
        def pytest_runtest_logreport(self, report):
            if report.when == "setup" and report.outcome in ["skipped", "failed"]:
                with open(log_file, "a") as f:
                    f.write(f"[{report.nodeid}] {report.outcome.upper()}")
                    if report.longrepr:
                        if isinstance(report.longrepr, tuple):
                            f.write(f"  Reason: {report.longrepr[2]}\n")
                        else:
                            f.write(f"  Reason: {str(report.longrepr).splitlines()[-1]}\n")
                            
            elif report.when == "call":
                with open(log_file, "a") as f:
                    f.write(f"[{report.nodeid}] {report.outcome.upper()}")
                    stdout = dict(report.sections).get("Captured stdout call", "")
                    if stdout:
                        f.write(f"  {stdout.strip()}\n")
                    if report.failed:
                        f.write(f"{report.longrepr}\n")
                        
    pytest.main(["-q", "--no-header", __file__], plugins=[CustomLogReporter()])
