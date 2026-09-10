import pytest
import numpy as np
import torch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import SpecComplex as sc
import SpecComplexTorch as scTorch

def test_hesd_pure_noise():
    B, N = 7, 100
    np.random.seed(42)
    sigma_true = 1.0
    pixels = np.random.normal(0, sigma_true, (B, N)).astype(np.float32)
    esd, sii, heights = sc.estimate_effective_dimensionality(pixels, k_max=7, sigma_n=sigma_true, kappa=3.0)
    assert esd == 1, f"Expected ESD=1 for pure noise, got {esd}"
    assert sii == 0.0, "Expected SII=0 for pure noise"

def test_hesd_synthetic_rank():
    B, N = 10, 200
    np.random.seed(42)
    M = np.random.rand(B, 3).astype(np.float32) * 10.0
    A = np.random.dirichlet(np.ones(3), N).T
    signal = M @ A
    sigma_true = 0.1
    noise = np.random.normal(0, sigma_true, (B, N)).astype(np.float32)
    pixels = signal + noise
    esd, sii, heights = sc.estimate_effective_dimensionality(pixels, k_max=10, sigma_n=sigma_true, kappa=3.0)
    assert esd == 4, f"Expected ESD=4 for rank-3 signal, got {esd}"
    assert sii > 0.0, "Expected SII > 0 for strong signal"

def test_hesd_numpy_torch_parity():
    B, N = 15, 100
    np.random.seed(42)
    pixels = np.random.rand(B, N).astype(np.float32) * 5.0
    sigma_true = 0.2
    esd_np, sii_np, heights_np = sc.estimate_effective_dimensionality(pixels, k_max=10, sigma_n=sigma_true, kappa=2.0)
    img_cube = torch.from_numpy(pixels).unsqueeze(0)
    esd_pt, sii_pt, heights_pt = scTorch.estimate_effective_dimensionality_torch(img_cube, k_max=10, sigma_n=sigma_true, kappa=2.0)
    assert esd_np == esd_pt[0].item(), "ESD mismatch between NumPy and PyTorch"
    np.testing.assert_allclose(sii_np, sii_pt[0].item(), rtol=1e-5)
    np.testing.assert_allclose(heights_np, heights_pt[0].cpu().numpy(), rtol=1e-5, atol=1e-6)

if __name__ == "__main__":
    pytest.main(["-v", __file__])
