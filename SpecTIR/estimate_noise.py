import numpy as np
from scipy.fft import dctn
from scipy.ndimage import binary_erosion
import spectral

def get_max_valid_rectangle(mask):
    """
    Finds the maximum area rectangle of True values in a 2D boolean array
    using the maximal rectangle in histogram algorithm.
    
    Args:
        mask (np.ndarray): 2D boolean mask of shape (Height, Width).
        
    Returns:
        tuple: (row_start, col_start, height, width) of the largest fully-valid rectangle.
    """
    rows, cols = mask.shape
    max_area = 0
    best_rect = (0, 0, 0, 0)
    
    heights = np.zeros(cols, dtype=int)
    for r in range(rows):
        heights = np.where(mask[r], heights + 1, 0)
        stack = []
        for c in range(cols + 1):
            h = heights[c] if c < cols else 0
            start = c
            while stack and stack[-1][1] > h:
                idx, val = stack.pop()
                area = val * (c - idx)
                if area > max_area:
                    max_area = area
                    best_rect = (r - val + 1, idx, val, c - idx)
                start = idx
            stack.append((start, h))
            
    return best_rect

def estimate_noise_nfd(image_cube, valid_mask=None, scrub_edge_artifacts=True, return_signal_variance=False, direction='lowerright'):
    """
    Estimates noise covariance using Noise From Differences (NFD) via Spectral Python.
    Requires a contiguous unmasked rectangular region.
    
    When scrub_edge_artifacts is True, applies edge artifact scrubbing:
      - Masks out non-physical reflectance/swath boundary outliers (e.g. > 1.05 or < -0.05).
      - Applies morphological erosion along mask boundaries to remove partial-pixel interpolation artifacts at swath borders.
      
    Args:
        image_cube (np.ndarray): Shape (Bands, Height, Width)
        valid_mask (np.ndarray, optional): Shape (Height, Width) boolean mask
        scrub_edge_artifacts (bool): If True, removes edge/boundary artifacts before identifying the inscribed rectangle.
        return_signal_variance (bool): If True, returns (noise_cov, patch_signal_variance).
        direction (str): Pixel difference direction ('lowerright', 'lowerleft', 'right', 'lower'). Default 'lowerright'.
        
    Returns:
        np.ndarray or tuple: (Bands, Bands) float32 noise covariance matrix,
                             or (noise_cov, patch_signal_variance) if return_signal_variance is True.
    """
    bands, height, width = image_cube.shape
    
    if valid_mask is None:
        vmask = np.all(np.isfinite(image_cube), axis=0)
    else:
        vmask = valid_mask.copy() & np.all(np.isfinite(image_cube), axis=0)
        
    if scrub_edge_artifacts and np.any(vmask):
        # Identify if data is in reflectance scale (99.5th percentile <= 1.2)
        valid_sample = image_cube[:, vmask]
        if valid_sample.size > 0:
            p99 = np.percentile(valid_sample, 99.5)
            if p99 <= 1.2:
                # Mask out unphysical edge artifact spikes
                vmask = vmask & np.all(image_cube <= 1.05, axis=0) & np.all(image_cube >= -0.05, axis=0)
        
        # Erode valid mask boundary to remove edge-interpolation artifacts at raster swath boundaries
        if np.any(vmask):
            vmask = binary_erosion(vmask, iterations=2)
            
    ry, rx, rh, rw = get_max_valid_rectangle(vmask)
    if rh < 10 or rw < 10:
        print("Warning: Could not find a large enough contiguous valid area for NFD. Returning zeros.")
        noise_cov = np.zeros((bands, bands), dtype=np.float32)
        signal_var = np.zeros(bands, dtype=np.float32)
    else:
        patch = image_cube[:, ry:ry+rh, rx:rx+rw]
        patch_spy = np.transpose(patch, (1, 2, 0)) # (Height, Width, Bands)
        stats = spectral.noise_from_diffs(patch_spy, direction=direction)
        noise_cov = stats.cov.astype(np.float32)
        signal_var = np.var(patch_spy.reshape(-1, bands), axis=0, ddof=1).astype(np.float32)
        
    if return_signal_variance:
        return noise_cov, signal_var
    return noise_cov

def estimate_noise_ssd(
    image_cube, 
    valid_mask=None, 
    block_size=16, 
    homogeneity_percentile=2.0, 
    min_blocks=5, 
    direction='lowerright',
    return_signal_variance=False
):
    """
    Estimates noise covariance using Spatial Shift-Difference (SSD) with Local Homogeneity Filtering.
    Identifies homogeneous spatial blocks across the image cube to prevent scene texture and edge
    gradients from contaminating the detector noise estimation, and computes noise covariance
    on the selected homogeneous patches using Spectral Python's noise_from_diffs.
    
    Args:
        image_cube (np.ndarray): Shape (Bands, Height, Width)
        valid_mask (np.ndarray, optional): Shape (Height, Width) boolean mask
        block_size (int): Local square patch size in pixels (default: 16).
        homogeneity_percentile (float): Percentage of valid blocks to select as the most homogeneous (default: 5.0).
        min_blocks (int): Minimum number of homogeneous blocks to sample (default: 5).
        direction (str): Difference direction ('lowerright', 'lowerleft', 'right', 'lower') for spectral.noise_from_diffs.
        return_signal_variance (bool): If True, returns (noise_cov, patch_signal_variance).
        
    Returns:
        np.ndarray or tuple: (Bands, Bands) float32 noise covariance matrix,
                             or (noise_cov, patch_signal_variance) if return_signal_variance=True.
    """
    bands, height, width = image_cube.shape
    
    if valid_mask is None:
        vmask = np.all(np.isfinite(image_cube), axis=0)
    else:
        vmask = valid_mask.copy() & np.all(np.isfinite(image_cube), axis=0)
        
    # If image dimension is smaller than or equal to block_size, execute noise_from_diffs directly on valid region
    if height <= block_size or width <= block_size:
        ry, rx, rh, rw = 0, 0, height, width
        if not np.all(vmask):
            ry, rx, rh, rw = get_max_valid_rectangle(vmask)
        if rh < 4 or rw < 4:
            noise_cov = np.zeros((bands, bands), dtype=np.float32)
            sig_var = np.zeros(bands, dtype=np.float32)
        else:
            patch = np.transpose(image_cube[:, ry:ry+rh, rx:rx+rw], (1, 2, 0))
            stats = spectral.noise_from_diffs(patch, direction=direction)
            noise_cov = stats.cov.astype(np.float32)
            sig_var = np.var(patch.reshape(-1, bands), axis=0, ddof=1).astype(np.float32)
        if return_signal_variance:
            return noise_cov, sig_var
        return noise_cov

    n_by = height // block_size
    n_bx = width // block_size
    
    img_crop = image_cube[:, :n_by * block_size, :n_bx * block_size]
    mask_crop = vmask[:n_by * block_size, :n_bx * block_size]
    
    blocks = (
        img_crop.reshape(bands, n_by, block_size, n_bx, block_size)
        .transpose(1, 3, 0, 2, 4)
        .reshape(-1, bands, block_size, block_size)
    )
    mask_blocks = (
        mask_crop.reshape(n_by, block_size, n_bx, block_size)
        .transpose(0, 2, 1, 3)
        .reshape(-1, block_size, block_size)
    )
    
    valid_blocks_idx = np.where(np.all(mask_blocks, axis=(1, 2)))[0]
    
    if len(valid_blocks_idx) == 0:
        # Fallback to largest contiguous valid rectangle if no fully-valid grid blocks are found
        ry, rx, rh, rw = get_max_valid_rectangle(vmask)
        if rh >= 4 and rw >= 4:
            patch = np.transpose(image_cube[:, ry:ry+rh, rx:rx+rw], (1, 2, 0))
            stats = spectral.noise_from_diffs(patch, direction=direction)
            noise_cov = stats.cov.astype(np.float32)
            sig_var = np.var(patch.reshape(-1, bands), axis=0, ddof=1).astype(np.float32)
        else:
            noise_cov = np.zeros((bands, bands), dtype=np.float32)
            sig_var = np.zeros(bands, dtype=np.float32)
        if return_signal_variance:
            return noise_cov, sig_var
        return noise_cov
        
    v_blocks = blocks[valid_blocks_idx]
    
    # Calculate homogeneity score: mean spatial standard deviation across all bands
    block_stds = np.std(v_blocks, axis=(2, 3)) # (num_valid, bands)
    block_scores = np.mean(block_stds, axis=1) # (num_valid,)
    
    # Select most homogeneous blocks
    n_sel = max(min_blocks, int(np.ceil(len(valid_blocks_idx) * (homogeneity_percentile / 100.0))))
    n_sel = min(n_sel, len(valid_blocks_idx))
    sorted_order = np.argsort(block_scores)
    sel_blocks = v_blocks[sorted_order[:n_sel]]
    
    # Pool noise covariance across homogeneous patches using spectral.noise_from_diffs
    cov_sum = np.zeros((bands, bands), dtype=np.float64)
    total_samples = 0
    delta_n = (block_size - 1)**2 if direction in ('lowerright', 'lowerleft') else block_size * (block_size - 1)
    
    for b in range(n_sel):
        patch = np.transpose(sel_blocks[b], (1, 2, 0)) # (B, B, bands)
        stats = spectral.noise_from_diffs(patch, direction=direction)
        cov_sum += stats.cov * delta_n
        total_samples += delta_n
        
    noise_cov = (cov_sum / total_samples).astype(np.float32)
    sig_var = np.var(sel_blocks.transpose(0, 2, 3, 1).reshape(-1, bands), axis=0, ddof=1).astype(np.float32)
    
    if return_signal_variance:
        return noise_cov, sig_var
    return noise_cov

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