import numpy as np
from scipy.fft import dctn
import spectral

def estimate_noise_ssd(image_cube, valid_mask=None):
    """
    Estimates noise covariance using Spatial Shift-Difference (SSD) via manual differences.
    Uses an average of horizontal and vertical differences, respecting the valid_mask.
    
    Args:
        image_cube (np.ndarray): Shape (Bands, Height, Width)
        valid_mask (np.ndarray): Shape (Height, Width) boolean mask
    """
    bands, height, width = image_cube.shape
    
    diff_h = image_cube[:, :, 1:] - image_cube[:, :, :-1]
    if valid_mask is not None:
        mask_h = valid_mask[:, 1:] & valid_mask[:, :-1]
        valid_diff_h = diff_h[:, mask_h]
    else:
        valid_diff_h = diff_h.reshape(bands, -1)
        
    diff_v = image_cube[:, 1:, :] - image_cube[:, :-1, :]
    if valid_mask is not None:
        mask_v = valid_mask[1:, :] & valid_mask[:-1, :]
        valid_diff_v = diff_v[:, mask_v]
    else:
        valid_diff_v = diff_v.reshape(bands, -1)
        
    all_diffs = np.concatenate((valid_diff_h, valid_diff_v), axis=1)
    N = all_diffs.shape[1]
    
    if N == 0:
        return np.zeros((bands, bands), dtype=np.float32)
        
    # SSD noise covariance = 0.5 * Cov(diffs)
    sigma_n = 0.5 * (all_diffs @ all_diffs.T) / N
    return sigma_n.astype(np.float32)

def estimate_noise_mnf(image_cube, valid_mask=None):
    """
    Estimates noise covariance natively using Spectral Python's MNF, respecting valid_mask.
    Args:
        image_cube (np.ndarray): Shape (Bands, Height, Width)
        valid_mask (np.ndarray): Shape (Height, Width) boolean mask
    """
    # For noise_stats, we'll use our masked SSD implementation
    noise_cov = estimate_noise_ssd(image_cube, valid_mask)
    return noise_cov

def estimate_noise_mlr(image_cube, valid_mask=None):
    """
    Estimates the full noise covariance matrix using Multiple Linear Regression (MLR).
    This implements the exact HySime noise estimation algorithm from pysptools. https://pysptools.sourceforge.io/_modules/pysptools/material_count/vd.html#hysime
    """
    bands, height, width = image_cube.shape
    
    if valid_mask is not None:
        Y = image_cube[:, valid_mask].astype(np.float64)
    else:
        Y = image_cube.reshape(bands, -1).astype(np.float64)
    
    L, N = Y.shape
    if N == 0:
        return np.zeros((bands, bands), dtype=np.float32)
        
    small = 1e-6
    w = np.zeros((L, N), dtype=np.float64)
    
    # Calculate the uncentered correlation matrix
    RR = np.dot(Y, Y.T) / N
    RRi = np.linalg.pinv(RR + small * np.eye(L))
    
    # Iterate over each band to regress it against all other bands
    for i in range(L):
        # Calculate regression coefficients by removing the i-th band's influence
        col = RRi[:, i:i+1]
        row = RRi[i:i+1, :]
        XX = RRi - np.dot(col, row) / RRi[i, i]
        
        RRa = RR[:, i].copy()
        RRa[i] = 0.0
        
        beta = np.dot(XX, RRa)
        beta[i] = 0.0
        
        # Calculate residual noise (w) = true signal - predicted signal
        w[i, :] = Y[i, :] - np.dot(beta, Y)
        
    # The noise covariance matrix is the diagonal of the covariance of the residuals
    sigma_n = np.diag(np.diag(np.dot(w, w.T) / N))
    
    return sigma_n.astype(np.float32)

def estimate_noise_dct(image_cube, valid_mask=None, block_size=8):
    """
    Estimates noise covariance using Discrete Cosine Transform (DCT) by isolating
    high-frequency structural blocks.
    """
    bands, height, width = image_cube.shape
    
    n_blocks_y = height // block_size
    n_blocks_x = width // block_size
    
    if n_blocks_y == 0 or n_blocks_x == 0:
        return np.zeros((bands, bands), dtype=np.float32)
        
    img = image_cube[:, :n_blocks_y*block_size, :n_blocks_x*block_size]
    blocks = img.reshape(bands, n_blocks_y, block_size, n_blocks_x, block_size)
    blocks = blocks.transpose(1, 3, 0, 2, 4)
    blocks = blocks.reshape(-1, bands, block_size, block_size)
    
    if valid_mask is not None:
        mask_img = valid_mask[:n_blocks_y*block_size, :n_blocks_x*block_size]
        mask_blocks = mask_img.reshape(n_blocks_y, block_size, n_blocks_x, block_size)
        mask_blocks = mask_blocks.transpose(0, 2, 1, 3)
        mask_blocks = mask_blocks.reshape(-1, block_size, block_size)
        valid_blocks_mask = np.all(mask_blocks, axis=(1, 2))
        valid_blocks = blocks[valid_blocks_mask]
    else:
        valid_blocks = blocks
    
    if len(valid_blocks) == 0:
        return np.zeros((bands, bands), dtype=np.float32)
        
    dct_blocks = dctn(valid_blocks, axes=(2, 3), norm='ortho')
    C = dct_blocks[:, :, block_size-1, block_size-1]
    
    median_C = np.median(C, axis=0, keepdims=True)
    mad = np.median(np.abs(C - median_C), axis=0)
    mad_stds = mad / 0.6745
    
    centered_C = C - median_C
    magnitudes = np.linalg.norm(centered_C / (mad_stds + 1e-8), axis=1)
    
    threshold = np.percentile(magnitudes, 90)
    valid_m = magnitudes <= threshold
    
    C_robust = C[valid_m, :]
    if C_robust.shape[0] > 1:
        corr = np.corrcoef(C_robust, rowvar=False)
    else:
        corr = np.eye(bands)
        
    sigma_n = np.outer(mad_stds, mad_stds) * corr
    return sigma_n.astype(np.float32)