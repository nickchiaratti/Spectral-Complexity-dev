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
    GPU-accelerated Spectral Complexity calculation with CPU-host memory spooling.
    
    The spatial unfold tensor is kept in system RAM to avoid saturating GPU VRAM.
    Only small batch chunks are transferred to the GPU for MaxD and QR computation,
    then results are immediately returned to CPU. This keeps VRAM utilization bounded
    to a fixed fraction of total GPU memory regardless of image or band count.
    
    Precision: float32, consistent with SpecComplex.py reference implementation.
    """
    COMPUTE_DTYPE = torch.float32
    BYTES_PER_ELEMENT = 4  # float32

    bands, height, width = frame_data.shape
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. Load data onto CPU only — system RAM absorbs the full unfold allocation
    tensor_data = torch.from_numpy(frame_data).to('cpu', dtype=COMPUTE_DTYPE)
    
    # 2. Calculate dynamic batch size to maximize GPU utilization without exceeding
    #    60% VRAM. The overhead multiplier accounts for the peak intermediate tensors
    #    created during MaxD (data_proj clone, diff broadcasts) and QR (Q, R, workspace).
    N_pixels = tile_size * tile_size
    if device == 'cuda':
        total_vram = torch.cuda.get_device_properties(device).total_memory
        target_vram = total_vram * 0.60
        overhead_multiplier = 20
        bytes_per_window = overhead_multiplier * bands * N_pixels * BYTES_PER_ELEMENT
        batch_size = max(1, int(target_vram // bytes_per_window))
    else:
        batch_size = 5000 
    
    # 3. Extract spatial sliding windows in chunks to avoid OOM on large images (e.g. EnMAP/Tanager)
    out_h = (height - tile_size) // stride + 1
    out_w = (width - tile_size) // stride + 1
    L = out_h * out_w
    
    # Accumulators live on CPU to prevent VRAM growth across iterations
    vol_vals = torch.zeros(L, dtype=COMPUTE_DTYPE, device='cpu')
    valid_mask = torch.zeros(L, dtype=torch.bool, device='cpu')
    
    chunk_rows = 100 # Process up to 100 rows of output windows at a time
    
    # 4. Process windows in memory-safe batches — transfer chunk to GPU, compute, return to CPU
    with torch.no_grad():
        for y_start in range(0, out_h, chunk_rows):
            y_end = min(y_start + chunk_rows, out_h)
            
            row_start = y_start * stride
            row_end = (y_end - 1) * stride + tile_size
            
            chunk_data = tensor_data[:, row_start:row_end, :].unsqueeze(0)
            
            chunk_unfold = torch.nn.functional.unfold(chunk_data, kernel_size=tile_size, stride=stride)
            chunk_windows = chunk_unfold.view(bands, N_pixels, -1).permute(2, 0, 1) # (L_chunk, C, N_pixels)
            L_chunk = chunk_windows.shape[0]
            
            global_start_idx = y_start * out_w
            
            for i in range(0, L_chunk, batch_size):
                batch_windows = chunk_windows[i:i+batch_size].to(device)  # (B, C, N)
        
                # Find windows that have enough valid pixels to extract endmembers
                # A pixel is valid if all its band values are not NaN.
                pixel_validity = ~torch.isnan(batch_windows).any(dim=1)  # (B, N)
                valid_pixels_per_window = pixel_validity.sum(dim=1)  # (B)
            
                # Strict Validity: Window is only processed if ALL pixels are valid.
                # This matches the Docstring specification of SpecComplex.py 
                # "Window is only processed if ALL pixels are valid."
                batch_valid = valid_pixels_per_window == N_pixels
            
                # Store validity on CPU
                valid_mask[global_start_idx + i : global_start_idx + i + batch_windows.shape[0]] = batch_valid.cpu()
            
                if not batch_valid.any():
                    continue
                
                valid_data = batch_windows[batch_valid].clone()
                valid_pixel_mask = pixel_validity[batch_valid]  # (B_valid, N)
            
                # Zero out NaNs to prevent NaN propagation during tensor math
                valid_data[torch.isnan(valid_data)] = 0.0
            
                # 4a. Batched Maximum Distance Simplices
                endmembers = maximumDistance_torch(valid_data, num_endmembers, valid_pixel_mask)  # (B_valid, C, E)
            
                # 4b. Batched Gram Volumes (QR Decomposition)
                if gram_type == 'datasetMean':
                    meanVector = valid_data.mean(dim=2)  # (B_valid, C)
                    volume = calcGramLocalVolumes_QR_torch(endmembers, meanVector)
                elif gram_type == 'minEndmember':
                    localizationVec = endmembers[:, :, 1]
                    remainingEndmembers = torch.cat((endmembers[:, :, 0:1], endmembers[:, :, 2:]), dim=2)
                    volume = calcGramLocalVolumes_QR_torch(remainingEndmembers, localizationVec)
                
                    # Prepend 0.0 volume for mathematical consistency
                    zeros = torch.zeros(volume.shape[0], 1, dtype=COMPUTE_DTYPE, device=device)
                    volume = torch.cat((zeros, volume), dim=1)
                else:
                    origin = torch.zeros(bands, dtype=COMPUTE_DTYPE, device=device)
                    volume = calcGramLocalVolumes_QR_torch(endmembers, origin)
                
                # 4c. Optional Normalization
                if norm_type == 'bandCount':
                    m_array = torch.arange(1, volume.shape[1] + 1, dtype=COMPUTE_DTYPE, device=device)
                    volume = volume / torch.pow(bands, (m_array / 2.0))
                
                # 4d. Extract target metric and immediately move result to CPU
                if volume.shape[1] > 2:
                    vol_val = torch.max(volume[:, 2:], dim=1)[0]
                else:
                    vol_val = torch.zeros(volume.shape[0], dtype=COMPUTE_DTYPE, device=device)
                
                vol_vals[global_start_idx + i : global_start_idx + i + batch_windows.shape[0]][batch_valid.cpu()] = vol_val.cpu()
        
    # 5. Fold spatial output map on CPU — overlap-add reconstruction
    vol_vals_expanded = vol_vals.unsqueeze(0).unsqueeze(1).expand(1, N_pixels, L)
    valid_expanded = valid_mask.to(COMPUTE_DTYPE).unsqueeze(0).unsqueeze(1).expand(1, N_pixels, L)
    
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
    sum_map = sum_map.squeeze().numpy().astype(np.float32)
    count_map = count_map.squeeze().numpy().astype(np.int8)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        final_map = sum_map / count_map
    
    # 6. Neighborhood map — assign each tile's volume directly to its center pixel.
    #    No spatial averaging: each pixel receives the unblurred volume of the
    #    neighborhood tile centered on it. Border pixels within center_offset of
    #    the image edge have no complete tile and remain NaN.
    out_h = (height - tile_size) // stride + 1
    out_w = (width - tile_size) // stride + 1
    center_offset = tile_size // 2
    
    neighborhood_map = np.full((height, width), np.nan, dtype=np.float32)
    vol_grid = vol_vals.numpy().reshape(out_h, out_w)
    valid_grid = valid_mask.numpy().reshape(out_h, out_w)
    
    neighborhood_map[center_offset:center_offset + out_h,
                     center_offset:center_offset + out_w] = vol_grid
    neighborhood_map[center_offset:center_offset + out_h,
                     center_offset:center_offset + out_w][~valid_grid] = np.nan
        
    return final_map, neighborhood_map

