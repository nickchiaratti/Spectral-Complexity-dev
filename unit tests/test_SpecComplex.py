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

FILE_PATH = r"C:\satelliteImagery\MGRS30mConstellation\Harmonized_MGRS_Stack_Tait.h5"

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
                        frame_indices = [num_frames // 2] 
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
        
        # Scale surface reflectance to [0, 1] range using the attribute
        img = img * scale
        
        img_flat = img.reshape((-1, img.shape[2]))
        
        # A valid pixel should have no negative values and no values > 1 in any band
        valid_mask = np.all((img_flat > 0) & (img_flat <= 1), axis=1)
        
        # Fallback if too strict
        if np.sum(valid_mask) < 25:
            valid_mask = np.all((img_flat >= 0) & (img_flat <= 1), axis=1)
            
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
    num_endmembers = 5
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
    
    np.testing.assert_allclose(em_np, em_torch_np, rtol=1e-4, atol=1e-5, 
                               err_msg="Maximum distance endmembers mismatch")


def test_calcGramLocalVolumes(real_data):
    num_endmembers = 5
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
    np.testing.assert_allclose(vol_np, vol_torch_np, rtol=1e-4, atol=1e-5,
                               err_msg="Local volumes mismatch")


def test_generate_rgba_from_hsi():
    # Create a synthetic 10x10 hyperspectral cube with 200 bands
    bands = 200
    height, width = 10, 10
    wavelengths = np.linspace(400, 2400, bands, dtype=np.float32)
    
    # 1. Valid data (reflectance between 0 and 0.5, stored as int16 scaled by 10000)
    cube = np.random.randint(500, 5000, size=(bands, height, width), dtype=np.int16)
    
    # Add negative noise in a few bands to simulate atmospheric correction artifacts
    cube[0, 2:5, 2:5] = -5
    cube[10, 3, 3] = -15
    
    # Add a nodata region
    cube[:, 0:2, :] = -32768
    
    rgba = SpecComplex.generate_rgba_from_hsi(cube, wavelengths, nodata=-32768)
    
    assert rgba.shape == (height, width, 4)
    assert rgba.dtype == np.uint8
    
    # Nodata pixels should have alpha = 0
    assert np.all(rgba[0:2, :, 3] == 0)
    
    # Valid pixels (including those with negative noise in some bands) should have alpha = 255
    assert np.all(rgba[2:, :, 3] == 255)
    
    # RGB channels should contain non-zero valid visual data
    assert np.any(rgba[2:, :, :3] > 0)

