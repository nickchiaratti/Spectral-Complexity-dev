import os
import sys
import numpy as np
import h5py
from scipy.fft import dctn

def estimate_noise_ssd(image_cube, valid_mask):
    """
    Estimates the full noise covariance matrix using Spatial Shift-Difference (SSD).
    Computes horizontal and vertical differences of adjacent pixels.
    
    Args:
        image_cube (np.ndarray): Shape (Bands, Height, Width)
        valid_mask (np.ndarray): Boolean mask of valid pixels (Height, Width)
        
    Returns:
        np.ndarray: B x B noise covariance matrix
    """
    bands, height, width = image_cube.shape
    
    # Horizontal difference
    valid_x = valid_mask[:, :-1] & valid_mask[:, 1:]
    img_left = image_cube[:, :, :-1][:, valid_x]
    img_right = image_cube[:, :, 1:][:, valid_x]
    diff_x = img_left.astype(np.float32) - img_right.astype(np.float32)
    
    cov_x = np.zeros((bands, bands), dtype=np.float32)
    if diff_x.shape[1] > 1:
        # Variance of difference is 2 * variance of noise
        cov_x = np.cov(diff_x, ddof=1) / 2.0
        
    # Vertical difference
    valid_y = valid_mask[:-1, :] & valid_mask[1:, :]
    img_top = image_cube[:, :-1, :][:, valid_y]
    img_bottom = image_cube[:, 1:, :][:, valid_y]
    diff_y = img_top.astype(np.float32) - img_bottom.astype(np.float32)
    
    cov_y = np.zeros((bands, bands), dtype=np.float32)
    if diff_y.shape[1] > 1:
        cov_y = np.cov(diff_y, ddof=1) / 2.0
        
    # Average the spatial covariances
    sigma_n = (cov_x + cov_y) / 2.0
    return sigma_n


def estimate_noise_mlr(image_cube, valid_mask):
    """
    Estimates the full noise covariance matrix using Multiple Linear Regression (MLR).
    
    Args:
        image_cube (np.ndarray): Shape (Bands, Height, Width)
        valid_mask (np.ndarray): Boolean mask of valid pixels (Height, Width)
        
    Returns:
        np.ndarray: B x B noise covariance matrix
    """
    bands, height, width = image_cube.shape
    
    Y = image_cube[:, valid_mask].astype(np.float32)
    
    N_pixels = Y.shape[1]
    if N_pixels == 0:
        return np.zeros((bands, bands), dtype=np.float32)
        
    # Subtract mean
    mu = np.mean(Y, axis=1, keepdims=True)
    Y_c = Y - mu
    
    # Compute data covariance matrix R
    R = (Y_c @ Y_c.T) / (N_pixels - 1)
    
    # Ridge regularization to prevent singular matrix inversion
    epsilon = 1e-6 * np.trace(R) / bands
    R_reg = R + np.eye(bands) * epsilon
    
    Q = np.linalg.inv(R_reg)
    
    # Mathematical derivation of MLR residual covariance:
    # E = diag(Q)^{-1} Q Y_c  (the residual vectors)
    # Cov(E) = diag(Q)^{-1} Q R Q diag(Q)^{-1}
    # Since Q = R^{-1}, this simplifies exactly to: diag(Q)^{-1} Q diag(Q)^{-1}
    diag_Q_inv = np.diag(1.0 / np.diag(Q))
    sigma_n = diag_Q_inv @ Q @ diag_Q_inv
    
    return sigma_n

def estimate_noise_dct(image_cube, valid_mask, block_size=8):
    """
    Estimates the full noise covariance matrix using Discrete Cosine Transform (DCT).
    Isolates the highest spatial frequency to calculate robust variances and correlations.
    
    Args:
        image_cube (np.ndarray): Shape (Bands, Height, Width)
        valid_mask (np.ndarray): Boolean mask of valid pixels (Height, Width)
        block_size (int): Size of the blocks to use for DCT (default 8)
        
    Returns:
        np.ndarray: B x B noise covariance matrix
    """
    bands, height, width = image_cube.shape
    
    n_blocks_y = height // block_size
    n_blocks_x = width // block_size
    
    if n_blocks_y == 0 or n_blocks_x == 0:
        return np.zeros((bands, bands), dtype=np.float32)
        
    # Extract blocks synchronously across all bands to preserve spectral correlation
    img = image_cube[:, :n_blocks_y*block_size, :n_blocks_x*block_size]
    blocks = img.reshape(bands, n_blocks_y, block_size, n_blocks_x, block_size)
    blocks = blocks.transpose(1, 3, 0, 2, 4)
    blocks = blocks.reshape(-1, bands, block_size, block_size)
    
    # Extract blocks for the valid mask
    mask_img = valid_mask[:n_blocks_y*block_size, :n_blocks_x*block_size]
    mask_blocks = mask_img.reshape(n_blocks_y, block_size, n_blocks_x, block_size)
    mask_blocks = mask_blocks.transpose(0, 2, 1, 3)
    mask_blocks = mask_blocks.reshape(-1, block_size, block_size)
    
    # Mask out blocks that contain any invalid pixels
    valid_blocks_mask = np.all(mask_blocks, axis=(1, 2))
    valid_blocks = blocks[valid_blocks_mask]
    
    if len(valid_blocks) == 0:
        return np.zeros((bands, bands), dtype=np.float32)
        
    # Apply 2D DCT to spatial dimensions (axes 2 and 3)
    dct_blocks = dctn(valid_blocks, axes=(2, 3), norm='ortho')
    
    # Extract the highest frequency coefficient (bottom-right)
    # C shape: (N_valid_blocks, bands)
    C = dct_blocks[:, :, block_size-1, block_size-1]
    
    # 1. Robust Variance Estimation (MAD)
    median_C = np.median(C, axis=0, keepdims=True)
    mad = np.median(np.abs(C - median_C), axis=0)
    mad_stds = mad / 0.6745
    
    # 2. Spectral Correlation Matrix
    # We drop blocks with extreme outliers (e.g., sharp boundaries leaking into high frequencies)
    # to ensure the correlation represents the true sensor noise floor, not scene structure.
    centered_C = C - median_C
    magnitudes = np.linalg.norm(centered_C / (mad_stds + 1e-8), axis=1)
    
    # Keep the 90% most "noise-like" blocks for correlation
    threshold = np.percentile(magnitudes, 90)
    valid_m = magnitudes <= threshold
    
    C_robust = C[valid_m, :]
    if C_robust.shape[0] > 1:
        corr = np.corrcoef(C_robust, rowvar=False)
    else:
        corr = np.eye(bands)
        
    # 3. Construct the Full Noise Covariance Matrix
    sigma_n = np.outer(mad_stds, mad_stds) * corr
    
    return sigma_n

def main():
    h5_file = r"C:\satelliteImagery\share2012\SpecTIR\SpecTIR_Ortho_Stack_0920-1706.h5"
    if not os.path.exists(h5_file):
        print(f"Error: Could not find {h5_file}")
        return
        
    print(f"Opening {h5_file} for noise estimation...")
    with h5py.File(h5_file, 'r+') as f:
        grp_path = "HDFEOS/GRIDS/SPECTIR/Data Fields"
        if grp_path not in f:
            print(f"Error: Could not find group {grp_path} in HDF5 file.")
            return
            
        grp = f[grp_path]
        ds_ref = grp['surface_reflectance']
        fill_val = ds_ref.fillvalue if ds_ref.fillvalue is not None else 0
        
        print(f"Reading surface_reflectance cube into memory (Shape: {ds_ref.shape})...")
        image_cube = ds_ref[0, ...] 
        
        if 'nodata_mask' in grp:
            print("Reading nodata_mask from HDF5...")
            valid_mask = ~grp['nodata_mask'][0, 0, ...]
        else:
            print(f"No nodata_mask found. Falling back to fill_value ({fill_val}) heuristic...")
            valid_mask = np.all(image_cube != fill_val, axis=0)
        
        print("Computing SSD Noise Covariance...")
        ssd_noise = estimate_noise_ssd(image_cube, valid_mask)
        
        print("Computing MLR Noise Covariance...")
        mlr_noise = estimate_noise_mlr(image_cube, valid_mask)
        
        print("Computing DCT Noise Covariance...")
        dct_noise = estimate_noise_dct(image_cube, valid_mask)
        
        # Save to HDF5
        bands = image_cube.shape[0]
        
        for name, data in [("ssd_noise_cov", ssd_noise), ("mlr_noise_cov", mlr_noise), ("dct_noise_cov", dct_noise)]:
            if name in grp:
                print(f"Dataset '{name}' already exists. Overwriting...")
                del grp[name]
                
            ds = grp.create_dataset(
                name,
                data=data,
                shape=(bands, bands),
                dtype=np.float32,
                compression="gzip",
                compression_opts=5
            )
            method = name.split('_')[0].upper()
            ds.attrs['description'] = f"Estimated full noise covariance matrix (Bands x Bands) using {method} method"
            print(f"Saved dataset: {name}")

    print("Noise estimation completed successfully.")

if __name__ == "__main__":
    main()
