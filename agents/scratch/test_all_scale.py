import h5py
import numpy as np
import spectral
from scipy.fft import dctn

def estimate_noise_ssd(image_cube):
    img_hwc = np.transpose(image_cube, (1, 2, 0))
    n_right = spectral.noise_from_diffs(img_hwc, direction='right')
    n_lower = spectral.noise_from_diffs(img_hwc, direction='lower')
    sigma_n = (n_right.cov + n_lower.cov) / 2.0
    return np.mean(np.diag(sigma_n))

def estimate_noise_mnf(image_cube):
    img_hwc = np.transpose(image_cube, (1, 2, 0))
    signal_stats = spectral.calc_stats(img_hwc)
    noise_stats = spectral.noise_from_diffs(img_hwc)
    mnf_result = spectral.mnf(signal_stats, noise_stats)
    return np.mean(np.diag(mnf_result.noise.cov))

def estimate_noise_dct(image_cube, block_size=8):
    bands, height, width = image_cube.shape
    n_blocks_y = height // block_size
    n_blocks_x = width // block_size
    img = image_cube[:, :n_blocks_y*block_size, :n_blocks_x*block_size]
    blocks = img.reshape(bands, n_blocks_y, block_size, n_blocks_x, block_size)
    blocks = blocks.transpose(1, 3, 0, 2, 4).reshape(-1, bands, block_size, block_size)
    dct_blocks = dctn(blocks, axes=(2, 3), norm='ortho')
    C = dct_blocks[:, :, block_size-1, block_size-1]
    median_C = np.median(C, axis=0, keepdims=True)
    mad = np.median(np.abs(C - median_C), axis=0)
    mad_stds = mad / 0.6745
    centered_C = C - median_C
    magnitudes = np.linalg.norm(centered_C / (mad_stds + 1e-8), axis=1)
    threshold = np.percentile(magnitudes, 90)
    valid_m = magnitudes <= threshold
    C_robust = C[valid_m, :]
    corr = np.corrcoef(C_robust, rowvar=False) if C_robust.shape[0] > 1 else np.eye(bands)
    sigma_n = np.outer(mad_stds, mad_stds) * corr
    return np.mean(np.diag(sigma_n))

h5_path = r'C:\satelliteImagery\share2012\SpecTIR\SpecTIR_Ortho_Stack_0920-1706.h5'
with h5py.File(h5_path, 'r') as f:
    cube = f['HDFEOS/GRIDS/SPECTIR/Data Fields/surface_reflectance'][0, :, 100:200, 100:200]

cube_scaled = cube.astype(np.float64) / 10000.0

print(f"SSD noise on scaled cube: {estimate_noise_ssd(cube_scaled)}")
print(f"MNF noise on scaled cube: {estimate_noise_mnf(cube_scaled)}")
print(f"DCT noise on scaled cube: {estimate_noise_dct(cube_scaled)}")
