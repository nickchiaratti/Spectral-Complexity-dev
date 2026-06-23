import torch
import numpy as np
import warnings

def maximumDistance_torch(data, num_endmembers, valid_pixel_mask):
    """
    Batched, GPU-accelerated Maximum Distance (MaxD) algorithm.
    Extracts geometric simplices using orthogonal projections.
    
    Args:
        data: Tensor of shape (B, C, N) [Batch, Bands, Pixels]
        num_endmembers: int
        valid_pixel_mask: Boolean Tensor of shape (B, N). True for valid pixels.
    Returns:
        endmembers: Tensor of shape (B, C, num_endmembers)
    """
    B, C, N = data.shape
    device = data.device
    dtype = data.dtype
    
    # Calculate squared magnitude for all pixels
    magnitude_sq = torch.sum(data ** 2, dim=1) # (B, N)
    
    # Argmax: invalid pixels mapped to -inf so they are ignored
    mag_sq_max = magnitude_sq.clone()
    mag_sq_max[~valid_pixel_mask] = -float('inf')
    idx1 = torch.argmax(mag_sq_max, dim=1)
    
    # Argmin: invalid pixels mapped to +inf so they are ignored
    mag_sq_min = magnitude_sq.clone()
    mag_sq_min[~valid_pixel_mask] = float('inf')
    idx2 = torch.argmin(mag_sq_min, dim=1)
    
    endmembers = torch.zeros(B, C, num_endmembers, dtype=dtype, device=device)
    b_idx = torch.arange(B, device=device)
    
    endmembers[:, :, 0] = data[b_idx, :, idx1]
    endmembers[:, :, 1] = data[b_idx, :, idx2]
    
    data_proj = data.clone()
    
    for i in range(2, num_endmembers):
        # Extract previous endmember vector for projection
        diff = data_proj[b_idx, :, idx2].unsqueeze(2) - data_proj[b_idx, :, idx1].unsqueeze(2) # (B, C, 1)
        norm_sq = torch.sum(diff ** 2, dim=1, keepdim=True) # (B, 1, 1)
        
        # Calculate algebraic pseudoinverse safely
        pseudo = torch.where(
            norm_sq > 1e-12, 
            diff.transpose(1, 2) / norm_sq, 
            torch.zeros_like(diff.transpose(1, 2))
        )
        
        # Batch Matrix Multiply projection: data_proj -= diff @ (pseudo @ data_proj)
        proj_coef = torch.bmm(pseudo, data_proj)
        data_proj -= torch.bmm(diff, proj_coef)
        
        # Calculate new distances
        idx1 = idx2.clone()
        vec = data_proj[b_idx, :, idx2].unsqueeze(2)
        diff_new = torch.sum((vec - data_proj) ** 2, dim=1) # (B, N)
        
        # Mask out invalid pixels from being chosen as max distance
        diff_new[~valid_pixel_mask] = -float('inf')
        
        idx2 = torch.argmax(diff_new, dim=1)
        
        endmembers[:, :, i] = data[b_idx, :, idx2]
        
    return endmembers


def calcGramLocalVolumes_QR_torch(endmembers, localization_vector):
    """
    Batched QR Decomposition for Simplex Volumes.
    Follows Gantmacher's theorem equating volume to the product of orthogonal heights.
    
    Args:
        endmembers: Tensor of shape (B, C, E)
        localization_vector: Tensor of shape (B, C) or (C,)
    Returns:
        volumes: Tensor of shape (B, E)
    """
    if localization_vector.dim() == 1:
        localization_vector = localization_vector.unsqueeze(0).unsqueeze(2)
    elif localization_vector.dim() == 2:
        localization_vector = localization_vector.unsqueeze(2)
        
    # Localize (translate) the endmembers to the origin defined by localization_vector
    localized_vectors = endmembers - localization_vector
    
    # Batched QR Decomposition
    # Q: (B, C, E) Orthogonal rotations
    # R: (B, E, E) Upper-triangular scales (heights)
    Q, R = torch.linalg.qr(localized_vectors)
    
    # Extract absolute heights from the main diagonal of R
    heights = torch.abs(torch.diagonal(R, dim1=-2, dim2=-1)) # (B, E)
    
    # Parallelotope volume is the cumulative product of orthogonal heights
    volumes = torch.cumprod(heights, dim=-1) # (B, E)
    
    return volumes


def process_volume_sliding_tile(frame_data, tile_size, stride, num_endmembers, gram_type, norm_type):
    """
    Fully batched, hardware-agnostic Spectral Complexity calculation.
    Uses `torch.nn.functional.unfold` and `torch.nn.functional.fold` to compute 
    sliding window analytics in parallel across the GPU/CPU.
    """
    bands, height, width = frame_data.shape
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load data into PyTorch, enforcing float64 for absolute precision/stability
    tensor_data = torch.from_numpy(frame_data).to(device, dtype=torch.float64) # (C, H, W)
    
    # Extract all spatial sliding windows simultaneously
    # Shape: (1, C * tile_size * tile_size, L) where L = num_windows
    unfolded = torch.nn.functional.unfold(
        tensor_data.unsqueeze(0), 
        kernel_size=tile_size, 
        stride=stride
    )
    
    # Reshape and Permute to (L, C, N_pixels)
    N_pixels = tile_size * tile_size
    unfolded = unfolded.view(bands, N_pixels, -1)
    windows = unfolded.permute(2, 0, 1) # (L, C, N_pixels)
    L = windows.shape[0]
    
    # Calculate Dynamic Chunk Size to prevent OOM
    if device == 'cuda':
        total_vram = torch.cuda.get_device_properties(device).total_memory
        target_vram = total_vram * 0.8
        
        # Estimate bytes per window: (C * N_pixels * 8 bytes)
        # Overhead multiplier heavily increased to 150 to account for QR decomposition and broadcasting
        bytes_per_window = 150 * bands * N_pixels * 8
        batch_size = max(1, int(target_vram // bytes_per_window))
        # Cap batch size strictly to prevent unified memory swapping
        batch_size = min(batch_size, 100000)
    else:
        # Standard generous chunk size for CPU RAM
        batch_size = 5000 

        
    vol_vals = torch.zeros(L, dtype=torch.float64, device=device)
    valid_mask = torch.zeros(L, dtype=torch.bool, device=device)
    
    # Process windows in memory-safe batches without gradients
    with torch.no_grad():
        for i in range(0, L, batch_size):
            batch_windows = windows[i:i+batch_size] # (B, C, N)
        
            # Find windows that have enough valid pixels to extract endmembers
            # A pixel is valid if all its band values are not NaN.
            pixel_validity = ~torch.isnan(batch_windows).any(dim=1) # (B, N)
            valid_pixels_per_window = pixel_validity.sum(dim=1) # (B)
        
            # Strict Validity: Window is only processed if ALL pixels are valid.
            # This matches the Docstring specification of SpecComplex.py 
            # "Window is only processed if ALL pixels are valid."
            batch_valid = valid_pixels_per_window == N_pixels
        
            valid_mask[i:i+batch_size] = batch_valid
        
            if not batch_valid.any():
                continue
            
            valid_data = batch_windows[batch_valid].clone()
            valid_pixel_mask = pixel_validity[batch_valid] # (B_valid, N)
        
            # Zero out NaNs to prevent NaN propagation during tensor math
            valid_data[torch.isnan(valid_data)] = 0.0
        
            # 1. Batched Maximum Distance Simplices
            endmembers = maximumDistance_torch(valid_data, num_endmembers, valid_pixel_mask) # (B_valid, C, E)
        
            # 2. Batched Gram Volumes (QR Decomposition)
            if gram_type == 'datasetMean':
                # NaNs were zeroed, but since we require ALL pixels to be valid, mean is safe
                meanVector = valid_data.mean(dim=2) # (B_valid, C)
                volume = calcGramLocalVolumes_QR_torch(endmembers, meanVector)
            elif gram_type == 'minEndmember':
                localizationVec = endmembers[:, :, 1]
                remainingEndmembers = torch.cat((endmembers[:, :, 0:1], endmembers[:, :, 2:]), dim=2)
                volume = calcGramLocalVolumes_QR_torch(remainingEndmembers, localizationVec)
            
                # Prepend 0.0 volume for mathematical consistency
                zeros = torch.zeros(volume.shape[0], 1, dtype=torch.float64, device=device)
                volume = torch.cat((zeros, volume), dim=1)
            else:
                origin = torch.zeros(bands, dtype=torch.float64, device=device)
                volume = calcGramLocalVolumes_QR_torch(endmembers, origin)
            
            # 3. Optional Normalization
            if norm_type == 'bandCount':
                m_array = torch.arange(1, volume.shape[1] + 1, dtype=torch.float64, device=device)
                volume = volume / torch.pow(bands, (m_array / 2.0))
            
            # 4. Extract target metric
            if volume.shape[1] > 2:
                vol_val = torch.max(volume[:, 2:], dim=1)[0]
            else:
                vol_val = torch.zeros(volume.shape[0], dtype=torch.float64, device=device)
            
            vol_vals[i:i+batch_size][batch_valid] = vol_val
        
    # --- Fold spatial output map ---
    # Reconstruct the overlapping sum_map and count_map natively via F.fold
    vol_vals_expanded = vol_vals.unsqueeze(0).unsqueeze(1).expand(1, N_pixels, L)
    valid_expanded = valid_mask.to(torch.float64).unsqueeze(0).unsqueeze(1).expand(1, N_pixels, L)
    
    # Overlapping F.fold accumulation
    sum_map = torch.nn.functional.fold(
        vol_vals_expanded, 
        output_size=(height, width), 
        kernel_size=tile_size, 
        stride=stride
    )
    count_map = torch.nn.functional.fold(
        valid_expanded, 
        output_size=(height, width), 
        kernel_size=tile_size, 
        stride=stride
    )
    
    # Cast to specified lightweight datatypes for downstream saving
    sum_map = sum_map.squeeze().cpu().numpy().astype(np.float32)
    count_map = count_map.squeeze().cpu().numpy().astype(np.int8)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        final_map = sum_map / count_map
        
    return final_map
