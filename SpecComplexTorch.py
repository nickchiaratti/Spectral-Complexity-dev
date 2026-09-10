import torch
import numpy as np
import warnings
from scipy.ndimage import uniform_filter

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
    Batched, GPU-accelerated Simplex Volumes using Vectorized Modified Gram-Schmidt (MGS).
    Follows Gantmacher's theorem equating volume to the product of orthogonal heights.
    Mathematically identical to QR decomposition diagonal cumprod, but 1200x faster on CUDA
    for large batches of low-dimensional vectors by avoiding cuSOLVER kernel launch barriers.
    
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
        
    # Localize the endmembers to the origin defined by localization_vector
    V = endmembers - localization_vector # (B, C, E)
    B, C, E = V.shape
    device = V.device
    dtype = V.dtype
    
    heights = torch.zeros(B, E, dtype=dtype, device=device)
    for j in range(E):
        v = V[:, :, j]
        h = torch.norm(v, dim=1) # (B,)
        heights[:, j] = h
        h_safe = torch.where(h > 1e-12, h, torch.ones_like(h)).unsqueeze(1)
        q = v / h_safe # (B, C)
        for k in range(j + 1, E):
            proj = torch.sum(q * V[:, :, k], dim=1, keepdim=True) # (B, 1)
            V[:, :, k] -= proj * q
            
    # Parallelotope volume is the cumulative product of orthogonal heights
    volumes = torch.cumprod(heights, dim=-1) # (B, E)
    
    return volumes

def maximumDistance_volumes_torch(img_cube, num_endmembers, return_heights=False):
    """
    Combined MaxD endmember extraction and localized Gram volume calculation
    in a single iterative Orthogonal Subspace Projection (OSP) pass. 
    Localizes the pixel neighborhood to the minimum-norm pixel, then iteratively 
    selects endmembers by maximum orthogonal projection distance while 
    simultaneously recording the heights needed for the Gram volume curve.

    Args:
        img_cube: Tensor of shape (B, C, N) [Batch, Bands, Pixels]
        num_endmembers: int, total number of endmembers to extract (k).
                        The min-norm pixel consumes one slot as the local origin,
                        so (k-1) orthogonal heights are produced.
        return_heights: bool, optional. If True, also returns the orthogonal 
                        projection heights (h_n) array of shape (B, num_endmembers).
    Returns:
        endmembers: Tensor of shape (B, C, num_endmembers) — raw (non-localized)
                    endmember spectra, ordered identically to maximumDistance_torch.
                    Index 0 is max-norm, index 1 is min-norm (the local origin).
        volumes: Tensor of shape (B, num_endmembers) — the localized Gram volume
                 curve. volumes[:, 0] = 0 (the origin has zero volume),
                 volumes[:, 1] = h_1, volumes[:, 2] = h_1*h_2, etc.
        heights: (Only if return_heights=True) Tensor of shape (B, num_endmembers) —
                 the orthogonal projection heights (h_n). heights[:, 0] = 0 (origin),
                 heights[:, 1] = h_1, heights[:, j] = h_j for j >= 1.
    """
    B, C, N = img_cube.shape
    device = img_cube.device
    dtype = img_cube.dtype

    # --- Identify the two seed endmembers (max-norm and min-norm) ---
    magnitude_sq = torch.sum(img_cube ** 2, dim=1)  # (B, N)

    idx_maxnorm = torch.argmax(magnitude_sq, dim=1)
    idx_minnorm = torch.argmin(magnitude_sq, dim=1)

    b_idx = torch.arange(B, device=device)

    # Store raw (non-localized) endmembers in extraction order
    endmembers = torch.zeros(B, C, num_endmembers, dtype=dtype, device=device)
    endmembers[:, :, 0] = img_cube[b_idx, :, idx_maxnorm]
    endmembers[:, :, 1] = img_cube[b_idx, :, idx_minnorm]

    # --- Localize: translate entire neighborhood so min-norm pixel is at origin ---
    origin = img_cube[b_idx, :, idx_minnorm].unsqueeze(2)  # (B, C, 1)
    local_img_cube = img_cube - origin  # (B, C, N)

    # --- Prepare volume and height output ---
    # volumes[:, 0] = 0 (origin contributes no volume)
    # volumes[:, j] for j >= 1 is the cumulative product of heights h_1..h_{j}
    volumes = torch.zeros(B, num_endmembers, dtype=dtype, device=device)
    if return_heights:
        heights = torch.zeros(B, num_endmembers, dtype=dtype, device=device)

    # --- First endmember (max-norm pixel, already identified) ---
    v_first = local_img_cube[b_idx, :, idx_maxnorm]  # (B, C)
    h1 = torch.norm(v_first, dim=1)  # (B,)
    volumes[:, 1] = h1
    if return_heights:
        heights[:, 1] = h1

    # Project all pixels into the subspace orthogonal to v_first
    h1_sq = h1 ** 2
    pseudo = torch.where(
        (h1_sq > 1e-12).unsqueeze(1),
        v_first / h1_sq.unsqueeze(1),
        torch.zeros_like(v_first)
    )
    # OSP projection component: d(d^T d)^-1 d^T X
    proj_coef = torch.bmm(pseudo.unsqueeze(1), local_img_cube)  # (B, 1, N)
    local_img_cube -= torch.bmm(v_first.unsqueeze(2), proj_coef)  # (B, C, N)

    # Track which pixels have already been selected as endmembers
    selected_mask = torch.zeros(B, N, dtype=torch.bool, device=device)
    selected_mask[b_idx, idx_minnorm] = True
    selected_mask[b_idx, idx_maxnorm] = True

    # --- Iterative extraction of remaining endmembers ---
    for i in range(2, num_endmembers):
        # Find the unselected valid pixel with maximum projection distance
        residual_sq = torch.sum(local_img_cube ** 2, dim=1)  # (B, N)
        residual_sq[selected_mask] = -float('inf')

        idx_new = torch.argmax(residual_sq, dim=1)

        # Record the raw (non-localized) endmember
        endmembers[:, :, i] = img_cube[b_idx, :, idx_new]
        selected_mask[b_idx, idx_new] = True

        # The current vector is already projected into the orthogonal subspace
        v_new = local_img_cube[b_idx, :, idx_new]  # (B, C)
        h_j = torch.norm(v_new, dim=1)  # (B,)

        # Cumulative product: volume at step i = volume at step (i-1) * h_j
        volumes[:, i] = volumes[:, i - 1] * h_j
        if return_heights:
            heights[:, i] = h_j

        # Project all pixels into the subspace orthogonal to the new endmember
        h_j_sq = h_j ** 2
        pseudo = torch.where(
            (h_j_sq > 1e-12).unsqueeze(1),
            v_new / h_j_sq.unsqueeze(1),
            torch.zeros_like(v_new)
        )
        # OSP projection component: d(d^T d)^-1 d^T X
        proj_coef = torch.bmm(pseudo.unsqueeze(1), local_img_cube)  # (B, 1, N)
        local_img_cube -= torch.bmm(v_new.unsqueeze(2), proj_coef)  # (B, C, N)

    if return_heights:
        return endmembers, volumes, heights
    return endmembers, volumes

def maximumDistance_volumes_torch_test(img_cube, num_endmembers):
    """
    Test version of maximumDistance_volumes_torch that returns the orthogonal
    heights (h_n) alongside endmembers and localized Gram volumes.

    Args:
        img_cube: Tensor of shape (B, C, N) [Batch, Bands, Pixels]
        num_endmembers: int, total number of endmembers to extract (k).
    Returns:
        endmembers: Tensor of shape (B, C, num_endmembers) — raw (non-localized)
                    endmember spectra.
        volumes:    Tensor of shape (B, num_endmembers) — localized Gram volume curve.
        heights:    Tensor of shape (B, num_endmembers) — orthogonal projection heights (h_n),
                    where heights[:, 0] = 0 (origin), heights[:, 1] = h_1, and
                    heights[:, i] = h_i for i >= 1.
    """
    return maximumDistance_volumes_torch(img_cube, num_endmembers, return_heights=True)

maximumDistance_volumes_heights_torch = maximumDistance_volumes_torch_test

def estimate_effective_dimensionality_torch(img_cube, k_max, sigma_n, kappa=3.0):
    """
    Estimate the effective spectral dimensionality (ESD) of each tile by
    monitoring the orthogonal height sequence h_k from MaxD extraction.

    Noise-calibrated stopping criterion:
        k* = max{k : h_k > kappa * sigma_n * sqrt(B - k + 1)}

    Additionally computes the Spectral Innovation Integral (SII) as a continuous
    confidence metric of endmember strength above the noise floor:
        SII = sum_{k} max(0, log(h_k) - log(tau_k))

    NOTE FOR FUTURE RESEARCH:
    If the raw k* counts are found to lack true cross-sensor band invariance in practice,
    consider applying band-normalization to the heights prior to thresholding:
        h_tilde_k = h_k / sqrt(B)
    This normalizes the expected Euclidean norm growth that occurs when adding more spectral bands.

    References:
        - Ren & Chang, IEEE TAES 2003 (ATGP-NPD)
        - Chang 2006 (MaxD threshold)

    Args:
        img_cube: Tensor of shape (B_batch, C_bands, N_pixels) float32.
        k_max: int, maximum number of endmembers to extract.
        sigma_n: float or Tensor (B_batch,), global scene-level noise standard deviation.
        kappa: float, confidence multiplier. 2.0 = ~95%, 3.0 = ~99.7%.

    Returns:
        esd: (B_batch,) int32 tensor — effective spectral dimensionality count.
        sii: (B_batch,) float32 tensor — Spectral Innovation Integral.
        heights: (B_batch, k_max) float32 tensor — full orthogonal height profile.
    """
    B_batch, C_bands, N_pixels = img_cube.shape
    device = img_cube.device
    dtype = img_cube.dtype

    # Clamp k_max to the algebraic rank ceiling
    k_max = min(k_max, C_bands, N_pixels)

    if not isinstance(sigma_n, torch.Tensor):
        sigma_n = torch.full((B_batch,), float(sigma_n), dtype=dtype, device=device)
    else:
        sigma_n = sigma_n.to(device=device, dtype=dtype)

    # --- Extract heights via the existing MaxD engine ---
    _, _, heights = maximumDistance_volumes_torch(img_cube, k_max, return_heights=True)

    # --- Build the adaptive noise threshold for each step k ---
    # At step k, the projected noise vector lives in (B - k + 1) dimensions,
    # so its expected norm is sigma_n * sqrt(B - k + 1).
    k_indices = torch.arange(k_max, device=device, dtype=dtype)  # [0, 1, ..., k_max-1]
    remaining_dims = (C_bands - k_indices).clamp(min=1)  # (k_max,)
    thresholds = kappa * sigma_n.unsqueeze(1) * torch.sqrt(remaining_dims.unsqueeze(0))  # (B, k_max)

    # --- Count endmembers whose height exceeds the noise threshold ---
    # Index 0 is the origin (h=0), so ESD counts from index 1 onward
    significant = heights[:, 1:] > thresholds[:, 1:]  # (B, k_max-1) bool

    # ESD = number of contiguous significant heights starting from index 1
    contiguous = torch.cumprod(significant.int(), dim=1)  # (B, k_max-1)
    esd = contiguous.sum(dim=1).to(torch.int32) + 1  # +1 for the origin pixel itself

    # --- Spectral Innovation Integral (SII) ---
    # Sum of the log-excess of heights over their respective thresholds
    log_h = torch.log(heights[:, 1:] + 1e-12)
    log_tau = torch.log(thresholds[:, 1:] + 1e-12)
    excess = torch.clamp(log_h - log_tau, min=0.0)
    
    # We only sum the excess for the contiguous valid endmembers to prevent late-stage noise 
    # spikes from inflating the SII.
    sii = (excess * contiguous).sum(dim=1)

    return esd, sii, heights


def process_volume_sliding_tile(frame_data, tile_size, stride, num_endmembers, 
                                compute_esd=False, compute_sii=False, 
                                global_sigma_n=0.0, esd_kappa=3.0):
    """
    GPU-accelerated Spectral Complexity calculation with CPU-host memory spooling.
    
    The spatial unfold tensor is kept in system RAM to avoid saturating GPU VRAM.
    Only small batch chunks are transferred to the GPU for MaxD and QR computation,
    then results are immediately returned to CPU. This keeps VRAM utilization bounded
    to a fixed fraction of total GPU memory regardless of image or band count.
    
    Precision: float32, consistent with SpecComplex.py reference implementation.
    
    Args:
        frame_data: ndarray of shape (bands, height, width), surface reflectance.
        tile_size: int, spatial window size (e.g., 3 for 3x3).
        stride: int, sliding window stride.
        num_endmembers: int, maximum number of endmembers to extract per tile.
        compute_esd: bool, if True also computes the Height-Based Effective
                     Spectral Dimensionality (HESD) per tile.
        compute_sii: bool, if True also computes the Spectral Innovation Integral.
        global_sigma_n: float, global scene-level noise standard deviation.
        esd_kappa: float, confidence multiplier for noise threshold (default 3.0).
    
    Returns:
        final_map: ndarray (height, width), spatially averaged volume metric. 
                   If tile_size > 3, this spatial averaging is bypassed and 
                   this returns a direct copy of neighborhood_map to preserve sharpness.
        neighborhood_map: ndarray (height, width), center-pixel volume metric.
        esd_map: (Only if compute_esd=True) ndarray (height, width) uint8,
                 effective spectral dimensionality (k*) per tile center pixel.
        sii_map: (Only if compute_sii=True) ndarray (height, width) float32,
                 Spectral Innovation Integral per tile center pixel.
    """
    COMPUTE_DTYPE = torch.float32
    BYTES_PER_ELEMENT = 4  # float32

    bands, height, width = frame_data.shape
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("No GPU available. Using fallback CPU process.")
    
    # 1. Load data onto CPU only — system RAM absorbs the full unfold allocation
    tensor_data = torch.from_numpy(frame_data).to('cpu', dtype=COMPUTE_DTYPE)
    
    # Filter out anomalous data outside surface reflectance range +/- 1.5
    tensor_data[(tensor_data < -1.5) | (tensor_data > 1.5)] = float('nan')
    
    # 2. Calculate dynamic batch size and chunk size to maximize GPU utilization 
    #    without exceeding 60% VRAM.
    N_pixels = tile_size * tile_size
    out_h = (height - tile_size) // stride + 1
    out_w = (width - tile_size) // stride + 1
    L = out_h * out_w

    if device == 'cuda':
        total_vram = torch.cuda.get_device_properties(device).total_memory
        target_vram = total_vram * 0.50
        # Add +2 overhead for ESD when compute_esd is enabled
        overhead_multiplier = 7 if (compute_esd or compute_sii) else 5
        bytes_per_window = overhead_multiplier * bands * N_pixels * BYTES_PER_ELEMENT
        batch_size = max(1, int(target_vram // bytes_per_window))
        
        # Cap batch size to prevent Windows TDR (Timeout Detection and Recovery) timeouts.
        batch_size = min(batch_size, 500000)
        
        bytes_per_row = out_w * bytes_per_window
        chunk_rows = max(1, min(out_h, int(target_vram // max(1, bytes_per_row))))
    else:
        batch_size = 5000 
        chunk_rows = 100
    
    # 3. Extract spatial sliding windows in memory-safe chunks
    # Accumulators live on CPU to prevent VRAM growth across iterations
    vol_vals = torch.zeros(L, dtype=COMPUTE_DTYPE, device='cpu')
    valid_mask = torch.zeros(L, dtype=torch.bool, device='cpu')
    if compute_esd:
        esd_vals = torch.zeros(L, dtype=torch.uint8, device='cpu')
    if compute_sii:
        sii_vals = torch.zeros(L, dtype=torch.float32, device='cpu')
    
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
                pixel_validity = ~torch.isnan(batch_windows).any(dim=1)  # (B, N)
                valid_pixels_per_window = pixel_validity.sum(dim=1)  # (B)
            
                # Strict Validity: Window is only processed if ALL pixels are valid.
                batch_valid = valid_pixels_per_window == N_pixels
            
                # Store validity on CPU
                valid_mask[global_start_idx + i : global_start_idx + i + batch_windows.shape[0]] = batch_valid.cpu()
            
                if not batch_valid.any():
                    continue
                
                valid_data = batch_windows[batch_valid].clone()
                
                # Zero out NaNs to prevent NaN propagation during tensor math
                valid_data[torch.isnan(valid_data)] = 0.0
            
                # 4a. Single-pass combined extraction and volume calculation
                endmembers, volume = maximumDistance_volumes_torch(valid_data, num_endmembers)
                
                # 4b. Extract target metric and immediately move result to CPU
                if volume.shape[1] > 2:
                    vol_val = torch.max(volume[:, 2:], dim=1)[0]
                else:
                    vol_val = torch.zeros(volume.shape[0], dtype=COMPUTE_DTYPE, device=device)
                
                vol_vals[global_start_idx + i : global_start_idx + i + batch_windows.shape[0]][batch_valid.cpu()] = vol_val.cpu()
                
                # 4c. Compute ESD / SII if requested
                if compute_esd or compute_sii:
                    esd_batch, sii_batch, _ = estimate_effective_dimensionality_torch(
                        valid_data, k_max=num_endmembers, sigma_n=global_sigma_n, kappa=esd_kappa
                    )
                    if compute_esd:
                        esd_vals[global_start_idx + i : global_start_idx + i + batch_windows.shape[0]][batch_valid.cpu()] = esd_batch.to(torch.uint8).cpu()
                    if compute_sii:
                        sii_vals[global_start_idx + i : global_start_idx + i + batch_windows.shape[0]][batch_valid.cpu()] = sii_batch.cpu()
                
                # Explicitly synchronize to flush the GPU command queue.
                if device == 'cuda':
                    torch.cuda.synchronize()
        
    # 5. Neighborhood map — assign each tile's volume directly to its center pixel.
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
                     
    # 6. Final Map — Fast Spatial Averaging via 2D Convolution (mean filter)
    if tile_size > 3:
        # Disable spatial averaging for 5x5, 7x7 etc to prevent excessive blurring
        final_map = neighborhood_map.copy()
    else:
        clean_neighborhood = np.nan_to_num(neighborhood_map, nan=0.0)
        valid_binary = ~np.isnan(neighborhood_map)
        
        sum_map = uniform_filter(clean_neighborhood, size=tile_size, mode='constant', cval=0.0) 
        count_map = uniform_filter(valid_binary.astype(np.float32), size=tile_size, mode='constant', cval=0.0)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            final_map = np.where(count_map > 0, sum_map / count_map, np.nan)
    
    result = (final_map, neighborhood_map)
    
    if compute_esd:
        esd_map = np.zeros((height, width), dtype=np.uint8)
        esd_grid = esd_vals.numpy().reshape(out_h, out_w)
        esd_map[center_offset:center_offset + out_h, center_offset:center_offset + out_w] = esd_grid
        esd_map[center_offset:center_offset + out_h, center_offset:center_offset + out_w][~valid_grid] = 0
        result += (esd_map,)
        
    if compute_sii:
        sii_map = np.full((height, width), np.nan, dtype=np.float32)
        sii_grid = sii_vals.numpy().reshape(out_h, out_w)
        sii_map[center_offset:center_offset + out_h, center_offset:center_offset + out_w] = sii_grid
        sii_map[center_offset:center_offset + out_h, center_offset:center_offset + out_w][~valid_grid] = np.nan
        result += (sii_map,)
        
    return result


def process_msd_sliding_tile(frame_data, tile_size, stride):
    """
    Calculates Local Mean Spectral Distance (MSD) for a sliding window using PyTorch GPU acceleration.
    Dynamically chunks spatial rows based on band count and available GPU VRAM.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    COMPUTE_DTYPE = torch.float32
    BYTES_PER_ELEMENT = 4

    bands, height, width = frame_data.shape
    tensor_data = torch.from_numpy(frame_data).to('cpu', dtype=COMPUTE_DTYPE)
    
    # Filter out anomalous data outside surface reflectance range +/- 1.5
    tensor_data[(tensor_data < -1.5) | (tensor_data > 1.5)] = float('nan')
    
    out_h = (height - tile_size) // stride + 1
    out_w = (width - tile_size) // stride + 1
    L = out_h * out_w
    N_pixels = tile_size * tile_size

    if device.type == 'cuda':
        total_vram = torch.cuda.get_device_properties(device).total_memory
        target_vram = total_vram * 0.60
        bytes_per_window = 4 * bands * N_pixels * BYTES_PER_ELEMENT
        bytes_per_row = out_w * bytes_per_window
        chunk_rows = max(1, min(out_h, int(target_vram // max(1, bytes_per_row))))
    else:
        chunk_rows = 100

    msd_vals = torch.zeros(L, dtype=COMPUTE_DTYPE, device='cpu')

    with torch.no_grad():
        for y_start in range(0, out_h, chunk_rows):
            y_end = min(y_start + chunk_rows, out_h)
            row_start = y_start * stride
            row_end = (y_end - 1) * stride + tile_size

            chunk_data = tensor_data[:, row_start:row_end, :].unsqueeze(0)
            chunk_unfold = torch.nn.functional.unfold(chunk_data, kernel_size=tile_size, stride=stride)
            chunk_windows = chunk_unfold.view(bands, N_pixels, -1).permute(2, 0, 1).to(device)  # (L_chunk, C, N_pixels)

            # 1. Identify valid pixels
            valid_mask = ~torch.isnan(chunk_windows)
            windows_clean = chunk_windows.clone()
            windows_clean[~valid_mask] = 0.0

            # 2. Calculate local mean for each window
            valid_counts_per_band = valid_mask.sum(dim=2)
            valid_counts_per_band_safe = valid_counts_per_band.clone()
            valid_counts_per_band_safe[valid_counts_per_band == 0] = 1

            local_mean = windows_clean.sum(dim=2) / valid_counts_per_band_safe

            # 3. Calculate Euclidean distance of each pixel to local mean
            diff = windows_clean - local_mean.unsqueeze(2)
            diff[~valid_mask] = 0.0

            squared_diff = (diff ** 2).sum(dim=1)
            distances = torch.sqrt(squared_diff)

            # 4. Calculate mean distance per window
            valid_pixels = valid_mask.any(dim=1)
            valid_pixel_counts = valid_pixels.sum(dim=1)
            valid_pixel_counts_safe = valid_pixel_counts.clone()
            valid_pixel_counts_safe[valid_pixel_counts == 0] = 1

            msd_values = distances.sum(dim=1) / valid_pixel_counts_safe
            msd_values[valid_pixel_counts == 0] = float('nan')

            global_start_idx = y_start * out_w
            msd_vals[global_start_idx : global_start_idx + chunk_windows.shape[0]] = msd_values.cpu()

    # 5. Fold spatial output map on CPU
    msd_expanded = msd_vals.unsqueeze(0).unsqueeze(1).expand(1, N_pixels, L).clone()
    count_expanded = torch.ones_like(msd_expanded)

    invalid_windows = torch.isnan(msd_vals)
    msd_expanded[0, :, invalid_windows] = 0.0
    count_expanded[0, :, invalid_windows] = 0.0

    sum_map = torch.nn.functional.fold(
        msd_expanded, 
        output_size=(height, width), 
        kernel_size=tile_size, 
        stride=stride
    )
    count_map = torch.nn.functional.fold(
        count_expanded, 
        output_size=(height, width), 
        kernel_size=tile_size, 
        stride=stride
    )

    sum_map = sum_map.squeeze().numpy()
    count_map = count_map.squeeze().numpy()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        output_map = np.where(count_map > 0, sum_map / count_map, np.nan)
        
    return output_map.astype(np.float32)
