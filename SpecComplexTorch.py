import os
import math
import warnings
from warnings import simplefilter
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage import uniform_filter



def maximumDistance(data, num_endmembers, valid_pixel_mask=None):
    """
    Batched, GPU-accelerated Maximum Distance (MaxD) algorithm.
    Extracts geometric simplices using orthogonal projections.
    
    Args:
        data: Tensor of shape (B, C, N) [Batch, Bands, Pixels] or (C, N),
              or numpy ndarray of corresponding shape.
        num_endmembers: int
        valid_pixel_mask: Boolean Tensor/ndarray of shape (B, N) or (N,).
                          True for valid pixels. If None, derived from finite non-NaN data.
    Returns:
        endmembers: Tensor or ndarray of shape (B, C, num_endmembers) or (C, num_endmembers)
    """
    is_numpy = isinstance(data, np.ndarray)
    if is_numpy:
        orig_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        data = torch.from_numpy(data).to(device=orig_device, dtype=torch.float32)
    else:
        orig_device = data.device

    is_2d = (data.dim() == 2)
    if is_2d:
        data = data.unsqueeze(0)  # (1, C, N)

    B, C, N = data.shape
    device = data.device
    dtype = data.dtype

    if valid_pixel_mask is None:
        valid_pixel_mask = ~torch.isnan(data).any(dim=1)
    elif isinstance(valid_pixel_mask, np.ndarray):
        valid_pixel_mask = torch.from_numpy(valid_pixel_mask).to(device=device, dtype=torch.bool)
        if is_2d and valid_pixel_mask.dim() == 1:
            valid_pixel_mask = valid_pixel_mask.unsqueeze(0)
    elif isinstance(valid_pixel_mask, torch.Tensor):
        valid_pixel_mask = valid_pixel_mask.to(device=device, dtype=torch.bool)
        if is_2d and valid_pixel_mask.dim() == 1:
            valid_pixel_mask = valid_pixel_mask.unsqueeze(0)

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
        
    if is_2d:
        endmembers = endmembers.squeeze(0)

    if is_numpy:
        return endmembers.cpu().numpy()
    return endmembers

def maximumDistance_volumes(img_cube, num_endmembers, return_heights=False):
    """
    Combined MaxD endmember extraction and localized Gram volume calculation
    in a single iterative Orthogonal Subspace Projection (OSP) pass with GPU batching,
    lazy projection, float64 norm precision, and robust per-tile epsilon thresholding.
    Optimized for high-throughput steady-state processing with zero per-iteration allocations.

    Localizes the pixel neighborhood to the minimum-norm pixel, then iteratively 
    selects endmembers by maximum orthogonal projection distance while 
    simultaneously recording the heights needed for the Gram volume curve.

    Args:
        img_cube: Tensor of shape (B, C, N) [Batch, Bands, Pixels] or (C, N),
                  or numpy ndarray of corresponding shape.
        num_endmembers: int, total number of endmembers to extract (k).
                        The min-norm pixel consumes one slot as the local origin,
                        so (k-1) orthogonal heights are produced.
        return_heights: bool, optional. If True, also returns the orthogonal 
                        projection heights (h_n) array of shape (B, num_endmembers).
    Returns:
        endmembers: Tensor or ndarray of shape (B, C, num_endmembers) or (C, num_endmembers) — raw (non-localized)
                    endmember spectra, ordered identically to maximumDistance.
                    Index 0 is max-norm, index 1 is min-norm (the local origin).
        volumes: Tensor or ndarray of shape (B, num_endmembers) or (num_endmembers,) — the localized Gram volume
                 curve. volumes[..., 0] = 0 (the origin has zero volume),
                 volumes[..., 1] = h_1, volumes[..., 2] = h_1*h_2, etc.
        heights: (Only if return_heights=True) Tensor or ndarray of orthogonal projection heights.
    """
    is_numpy = isinstance(img_cube, np.ndarray)
    if is_numpy:
        orig_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        img_cube = torch.from_numpy(img_cube).to(device=orig_device, dtype=torch.float32)
    else:
        orig_device = img_cube.device

    is_2d = (img_cube.dim() == 2)
    if is_2d:
        img_cube = img_cube.unsqueeze(0)  # (1, C, N)

    B, C, N = img_cube.shape
    orig_dtype = img_cube.dtype
    img_cube = img_cube.to(torch.float32)
    device = img_cube.device
    dtype = torch.float32

    # Vector norm noise floor for float32
    epsilon = np.sqrt(C) * torch.finfo(dtype).eps

    # --- Identify the two seed endmembers (max-norm and min-norm) ---
    magnitude_sq = torch.sum(img_cube.square(), dim=1)  # (B, N)
    idx_maxnorm = torch.argmax(magnitude_sq, dim=1)
    idx_minnorm = torch.argmin(magnitude_sq, dim=1)

    b_idx = torch.arange(B, device=device)

    # Store raw (non-localized) endmembers in extraction order
    endmembers = torch.zeros(B, C, num_endmembers, dtype=dtype, device=device)
    endmembers[:, :, 0] = img_cube[b_idx, :, idx_maxnorm]
    endmembers[:, :, 1] = img_cube[b_idx, :, idx_minnorm]

    # --- Localize to min-norm pixel ---
    origin = img_cube[b_idx, :, idx_minnorm].unsqueeze(2)  # (B, C, 1)
    local_data = img_cube - origin  # (B, C, N)

    volumes = torch.zeros((B, num_endmembers), dtype=dtype, device=device)
    heights = torch.zeros((B, num_endmembers), dtype=dtype, device=device)

    U_basis = torch.zeros((B, C, num_endmembers - 1), dtype=dtype, device=device)

    residual_sq = torch.sum(local_data.square(), dim=1)  # (B, N)
    residual_sq[b_idx, idx_minnorm] = -float('inf')
    residual_sq[b_idx, idx_maxnorm] = -float('inf')

    # --- First endmember (max-norm pixel, already identified) ---
    v_first = local_data[b_idx, :, idx_maxnorm]  # (B, C)
    h1 = torch.norm(v_first.to(torch.float64), dim=1).to(dtype)  # (B,)

    heights[:, 1] = h1
    is_valid_1 = h1 > epsilon
    all_active = is_valid_1.all().item()

    if all_active:
        volumes[:, 1] = h1
        u1 = v_first / h1.unsqueeze(1)
        U_basis[:, :, 0] = u1
        proj_coeffs = torch.bmm(u1.unsqueeze(1), local_data).squeeze(1)  # (B, N)
        residual_sq.sub_(proj_coeffs.square_())
        active_mask = None
    else:
        active_mask = is_valid_1
        volumes[active_mask, 1] = h1[active_mask]
        safe_h1 = torch.where(active_mask, h1, torch.ones_like(h1))
        u1 = torch.where(active_mask.unsqueeze(1), v_first / safe_h1.unsqueeze(1), torch.zeros_like(v_first))
        U_basis[:, :, 0] = u1
        proj_coeffs = torch.bmm(u1.unsqueeze(1), local_data).squeeze(1)
        residual_sq.sub_(proj_coeffs.square_())

    # --- Iterative extraction of remaining endmembers ---
    for i in range(2, num_endmembers):
        if active_mask is not None and not active_mask.any():
            break  # All tiles in the batch have reached the noise floor

        idx_new = torch.argmax(residual_sq, dim=1)
        residual_sq[b_idx, idx_new] = -float('inf')

        endmembers[:, :, i] = img_cube[b_idx, :, idx_new]

        x_new = local_data[b_idx, :, idx_new]  # (B, C)
        k = i - 1
        U_k = U_basis[:, :, :k]  # (B, C, k)
        coeffs = torch.bmm(U_k.transpose(1, 2), x_new.unsqueeze(2))  # (B, k, 1)
        v_new = x_new - torch.bmm(U_k, coeffs).squeeze(2)  # (B, C)

        h_j = torch.norm(v_new.to(torch.float64), dim=1).to(dtype)  # (B,)

        if all_active:
            is_valid = h_j > epsilon
            if is_valid.all().item():
                heights[:, i] = h_j
                volumes[:, i] = volumes[:, i - 1] * h_j
                uj = v_new / h_j.unsqueeze(1)
                U_basis[:, :, k] = uj
                proj_coeffs = torch.bmm(uj.unsqueeze(1), local_data).squeeze(1)
                residual_sq.sub_(proj_coeffs.square_())
            else:
                all_active = False
                active_mask = is_valid
                heights[active_mask, i] = h_j[active_mask]
                volumes[active_mask, i] = volumes[active_mask, i - 1] * h_j[active_mask]
                safe_hj = torch.where(active_mask, h_j, torch.ones_like(h_j))
                uj = torch.where(active_mask.unsqueeze(1), v_new / safe_hj.unsqueeze(1), torch.zeros_like(v_new))
                U_basis[:, :, k] = uj
                proj_coeffs = torch.bmm(uj.unsqueeze(1), local_data).squeeze(1)
                residual_sq.sub_(proj_coeffs.square_())
        else:
            is_valid = (h_j > epsilon) & active_mask
            active_mask = active_mask & is_valid
            heights[active_mask, i] = h_j[active_mask]
            volumes[active_mask, i] = volumes[active_mask, i - 1] * h_j[active_mask]
            safe_hj = torch.where(active_mask, h_j, torch.ones_like(h_j))
            uj = torch.where(active_mask.unsqueeze(1), v_new / safe_hj.unsqueeze(1), torch.zeros_like(v_new))
            U_basis[:, :, k] = uj
            proj_coeffs = torch.bmm(uj.unsqueeze(1), local_data).squeeze(1)
            residual_sq.sub_(proj_coeffs.square_())

    if is_2d:
        endmembers = endmembers.squeeze(0)
        volumes = volumes.squeeze(0)
        if return_heights:
            heights = heights.squeeze(0)

    if is_numpy:
        if return_heights:
            return endmembers.cpu().numpy(), volumes.cpu().numpy(), heights.cpu().numpy()
        return endmembers.cpu().numpy(), volumes.cpu().numpy()

    if return_heights:
        return endmembers.to(orig_dtype), volumes.to(orig_dtype), heights.to(orig_dtype)
    return endmembers.to(orig_dtype), volumes.to(orig_dtype)


maximumDistance_volumes_torch = maximumDistance_volumes

def maximumDistance_volumes_torch_test(img_cube, num_endmembers):
    return maximumDistance_volumes(img_cube, num_endmembers, return_heights=True)

maximumDistance_volumes_heights_torch = maximumDistance_volumes_torch_test


def calcGramLocalVolumes(endmembers, localization_vector=None, divide_by_factorial=True, axis=None):
    """
    Computes the Gram matrix determinant volume curve for k=1..num_endmembers using PyTorch tensors.

    Mathematical formula:
        G_k = E_k @ E_k.T (or E_k.T @ E_k)
        vol_k = sqrt(|det(G_k)|) / (k! if divide_by_factorial else 1.0)

    Args:
        endmembers: Tensor or ndarray of shape (num_endmembers, num_bands) or (num_bands, num_endmembers),
                    or batched (B, num_endmembers, num_bands) / (B, num_bands, num_endmembers).
        localization_vector: Optional Tensor or ndarray of shape (num_bands,).
                             If provided, subtracted from endmembers before Gram matrix calculation.
        divide_by_factorial: bool, default True.
                             If True, divides by k! (simplex volume: sqrt(|det|) / k! as specified in R7).
                             If False, returns parallelotope volume sqrt(|det|) matching legacy SpecComplex.py.
        axis: int or None. Explicit axis designating the endmembers dimension. If None, inferred automatically.

    Returns:
        volumes: Tensor or ndarray of shape (num_endmembers,) or (B, num_endmembers) on the same device/type as input.
    """
    is_numpy = isinstance(endmembers, np.ndarray)
    if is_numpy:
        orig_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        endmembers = torch.from_numpy(endmembers).to(device=orig_device, dtype=torch.float32)
    else:
        orig_device = endmembers.device
        endmembers = endmembers.to(dtype=torch.float32)

    device = endmembers.device

    if endmembers.dim() == 2:
        if axis is not None:
            k_dim = axis % 2
        else:
            if localization_vector is not None:
                loc_len = localization_vector.shape[0] if hasattr(localization_vector, 'shape') else len(localization_vector)
                if endmembers.shape[0] == loc_len and endmembers.shape[1] != loc_len:
                    k_dim = 1
                elif endmembers.shape[1] == loc_len and endmembers.shape[0] != loc_len:
                    k_dim = 0
                else:
                    k_dim = 0 if endmembers.shape[0] <= endmembers.shape[1] else 1
            else:
                k_dim = 0 if endmembers.shape[0] <= endmembers.shape[1] else 1

        if k_dim == 0:
            E = endmembers  # (k, B)
        else:
            E = endmembers.transpose(0, 1)  # (k, B)

        if localization_vector is not None:
            if isinstance(localization_vector, np.ndarray):
                loc_t = torch.from_numpy(localization_vector).to(device=device, dtype=torch.float32)
            else:
                loc_t = localization_vector.to(device=device, dtype=torch.float32)
            if loc_t.dim() == 1:
                loc_t = loc_t.unsqueeze(0)
            E = E - loc_t

        G = torch.matmul(E, E.transpose(0, 1))  # (k, k)
        k_total = G.shape[0]
        volumes = torch.zeros(k_total, dtype=torch.float32, device=device)
        for i in range(1, k_total + 1):
            sub_G = G[:i, :i]
            det = torch.linalg.det(sub_G)
            vol = torch.sqrt(torch.clamp(torch.abs(det), min=0.0))
            if divide_by_factorial:
                vol = vol / math.factorial(i)
            volumes[i - 1] = vol

        if is_numpy:
            return volumes.cpu().numpy()
        return volumes

    elif endmembers.dim() == 3:
        if axis is not None:
            k_dim = axis % 3
        else:
            if localization_vector is not None:
                loc_len = localization_vector.shape[-1] if hasattr(localization_vector, 'shape') else len(localization_vector)
                if endmembers.shape[1] == loc_len and endmembers.shape[2] != loc_len:
                    k_dim = 2
                elif endmembers.shape[2] == loc_len and endmembers.shape[1] != loc_len:
                    k_dim = 1
                else:
                    k_dim = 1 if endmembers.shape[1] <= endmembers.shape[2] else 2
            else:
                k_dim = 1 if endmembers.shape[1] <= endmembers.shape[2] else 2

        if k_dim == 1:
            E = endmembers  # (B, k, C)
        else:
            E = endmembers.transpose(1, 2)  # (B, k, C)

        if localization_vector is not None:
            if isinstance(localization_vector, np.ndarray):
                loc_t = torch.from_numpy(localization_vector).to(device=device, dtype=torch.float32)
            else:
                loc_t = localization_vector.to(device=device, dtype=torch.float32)
            if loc_t.dim() == 1:
                loc_t = loc_t.unsqueeze(0).unsqueeze(0)
            elif loc_t.dim() == 2:
                loc_t = loc_t.unsqueeze(1)
            E = E - loc_t

        G = torch.bmm(E, E.transpose(1, 2))  # (B, k, k)
        batch_size, k_total, _ = G.shape
        volumes = torch.zeros(batch_size, k_total, dtype=torch.float32, device=device)
        for i in range(1, k_total + 1):
            sub_G = G[:, :i, :i]
            det = torch.linalg.det(sub_G)
            vol = torch.sqrt(torch.clamp(torch.abs(det), min=0.0))
            if divide_by_factorial:
                vol = vol / math.factorial(i)
            volumes[:, i - 1] = vol

        if is_numpy:
            return volumes.cpu().numpy()
        return volumes

    else:
        raise ValueError(f"calcGramLocalVolumes expects 2D or 3D tensor, got shape {endmembers.shape}")

def process_volume_sliding_tile_batch(batch_data, tile_size, stride, num_endmembers, device=None, return_neighborhood=False):
    """
    High-throughput batched GPU sliding window processing across multiple patch tiles.
    Bypasses whole-scene CPU memory spooling and chunking to execute direct batched unfolding,
    batched MaxD Gramian volume unmixing, and 2D convolution spatial averaging directly in VRAM.

    Args:
        batch_data: ndarray or Tensor of shape (B, bands, height, width) or (bands, height, width).
        tile_size: int, spatial window size (e.g., 3 for 3x3).
        stride: int, sliding window stride.
        num_endmembers: int, maximum number of endmembers to extract per tile.
        device: torch.device, str, or None. Target device ('cuda' or 'cpu'). If None, uses CUDA if available.
        return_neighborhood: bool, default False. If True, returns (final_maps, neighborhood_maps).

    Returns:
        final_maps: ndarray or Tensor of shape (B, height, width) or (height, width).
    """
    if num_endmembers > tile_size * tile_size:
        warnings.warn(f"Clamping num_endmembers ({num_endmembers}) to tile pixel count ({tile_size * tile_size})")
        num_endmembers = tile_size * tile_size

    if device is None:
        target_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        target_device = torch.device(device)

    is_numpy = isinstance(batch_data, np.ndarray)
    if is_numpy:
        t_data = torch.from_numpy(batch_data).to(device=target_device, dtype=torch.float32)
    elif isinstance(batch_data, torch.Tensor):
        t_data = batch_data.to(device=target_device, dtype=torch.float32)
    else:
        t_data = torch.as_tensor(batch_data, device=target_device, dtype=torch.float32)

    is_3d = (t_data.dim() == 3)
    if is_3d:
        t_data = t_data.unsqueeze(0)

    B, bands, height, width = t_data.shape

    # Filter out anomalous data outside surface reflectance range +/- 1.5
    t_data[(t_data < -1.5) | (t_data > 1.5)] = float('nan')

    N_pixels = tile_size * tile_size
    out_h = (height - tile_size) // stride + 1
    out_w = (width - tile_size) // stride + 1
    L = out_h * out_w

    # Direct batched unfolding on GPU
    patches = torch.nn.functional.unfold(t_data, kernel_size=(tile_size, tile_size), stride=stride)
    windows = patches.view(B, bands, N_pixels, L).permute(0, 3, 1, 2)  # (B, L, bands, N_pixels)
    flat_windows = windows.reshape(B * L, bands, N_pixels)

    flat_is_valid = ~torch.isnan(flat_windows).reshape(B * L, -1).any(dim=1)
    flat_vols = torch.zeros(B * L, dtype=torch.float32, device=target_device)

    if flat_is_valid.any():
        valid_data = flat_windows[flat_is_valid]
        chunk_size = 30000
        n_val = valid_data.shape[0]
        v_out = torch.zeros(n_val, dtype=torch.float32, device=target_device)
        for s in range(0, n_val, chunk_size):
            e = min(s + chunk_size, n_val)
            _, vol = maximumDistance_volumes(valid_data[s:e], num_endmembers)
            if vol.shape[1] > 2:
                v_out[s:e] = torch.max(vol[:, 2:], dim=1)[0]
        flat_vols[flat_is_valid] = v_out

    grid_vols = flat_vols.view(B, out_h, out_w)
    grid_valid = flat_is_valid.view(B, out_h, out_w)

    center_offset = tile_size // 2
    neighborhood_map = torch.full((B, height, width), float('nan'), dtype=torch.float32, device=target_device)
    neighborhood_map[:, center_offset:center_offset + out_h, center_offset:center_offset + out_w] = torch.where(grid_valid, grid_vols, float('nan'))

    clean_neighborhood = torch.nan_to_num(neighborhood_map, nan=0.0)
    valid_binary = (~torch.isnan(neighborhood_map)).to(torch.float32)

    pad = tile_size // 2
    kernel = torch.ones((1, 1, tile_size, tile_size), dtype=torch.float32, device=target_device)
    sum_map = torch.nn.functional.conv2d(clean_neighborhood.unsqueeze(1), kernel, padding=pad)
    count_map = torch.nn.functional.conv2d(valid_binary.unsqueeze(1), kernel, padding=pad)

    nan_tensor = torch.tensor(float('nan'), dtype=torch.float32, device=target_device)
    final_map = torch.where(count_map > 0, sum_map / count_map, nan_tensor).squeeze(1)

    if is_3d:
        final_map = final_map.squeeze(0)
        neighborhood_map = neighborhood_map.squeeze(0)

    # Always return NumPy-compatible arrays to callers
    if return_neighborhood:
        return final_map.detach().cpu().numpy(), neighborhood_map.detach().cpu().numpy()
    return final_map.detach().cpu().numpy()


def process_volume_sliding_tile(frame_data, tile_size, stride, num_endmembers, device=None, return_neighborhood=False):
    """
    GPU-accelerated sliding window processing with CPU-host memory spooling.
    Strict Validity: Window is only processed if ALL pixels are valid.
    
    The spatial unfold tensor is kept in system RAM or dynamically chunked onto GPU VRAM
    to avoid saturating memory. Only small spatial chunks and batches are processed on GPU,
    then results are aggregated and returned. This keeps VRAM utilization bounded
    to a fixed fraction of total GPU memory regardless of image or band count.
    
    Precision: float32, consistent with SpecComplex.py reference implementation.
    
    Args:
        frame_data: ndarray or Tensor of shape (bands, height, width), surface reflectance.
        tile_size: int, spatial window size (e.g., 3 for 3x3).
        stride: int, sliding window stride.
        num_endmembers: int, maximum number of endmembers to extract per tile.
        device: torch.device, str, or None. Device to use ('cuda' or 'cpu').
                If None, uses CUDA if available, else CPU.
        return_neighborhood: bool, default False. If True, returns (final_map, neighborhood_map).
    
    Returns:
        final_map: ndarray (height, width), spatially averaged volume metric. 
        neighborhood_map: (optional) ndarray (height, width) if return_neighborhood=True.
    """
    # If 4D batch or Tensor of 4D, delegate directly to high-throughput batch function
    if hasattr(frame_data, "ndim") and frame_data.ndim == 4:
        return process_volume_sliding_tile_batch(frame_data, tile_size, stride, num_endmembers, device=device, return_neighborhood=return_neighborhood)
    if hasattr(frame_data, "dim") and frame_data.dim() == 4:
        return process_volume_sliding_tile_batch(frame_data, tile_size, stride, num_endmembers, device=device, return_neighborhood=return_neighborhood)

    if num_endmembers > tile_size * tile_size:
        warnings.warn(f"Clamping num_endmembers ({num_endmembers}) to tile pixel count ({tile_size * tile_size})")
        num_endmembers = tile_size * tile_size

    COMPUTE_DTYPE = torch.float32
    BYTES_PER_ELEMENT = 4  # float32

    bands, height, width = frame_data.shape
    if device is None:
        target_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        target_device = torch.device(device)
    if target_device.type == 'cpu':
        pass

    # 1. Load data onto CPU only — system RAM absorbs the full allocation
    if isinstance(frame_data, torch.Tensor):
        tensor_data = frame_data.detach().cpu().to(dtype=COMPUTE_DTYPE).clone()
    else:
        tensor_data = torch.from_numpy(frame_data).to('cpu', dtype=COMPUTE_DTYPE)

    # Filter out anomalous data outside surface reflectance range +/- 1.5
    tensor_data[(tensor_data < -1.5) | (tensor_data > 1.5)] = float('nan')

    # 2. Calculate dynamic batch size and chunk size to maximize GPU utilization 
    N_pixels = tile_size * tile_size
    out_h = (height - tile_size) // stride + 1
    out_w = (width - tile_size) // stride + 1

    if target_device.type == 'cuda':
        total_vram = torch.cuda.get_device_properties(target_device).total_memory
        target_vram = total_vram * 0.50
        overhead_multiplier = 5
        bytes_per_window_compute = overhead_multiplier * bands * N_pixels * BYTES_PER_ELEMENT
        batch_size = max(1, int(target_vram // max(1, bytes_per_window_compute)))
        batch_size = min(batch_size, 20000)

        unfold_vram_budget = total_vram * 0.30
        bytes_per_window_unfold = bands * N_pixels * BYTES_PER_ELEMENT
        windows_per_chunk = max(1, int(unfold_vram_budget // max(1, bytes_per_window_unfold)))

        if windows_per_chunk >= out_w:
            chunk_cols = out_w
            chunk_rows = max(1, windows_per_chunk // out_w)
            chunk_rows = min(chunk_rows, 50)  # Cap rows to keep progress bar moving smoothly
        else:
            chunk_rows = 1
            chunk_cols = windows_per_chunk
    else:
        batch_size = 5000 
        chunk_rows = 100
        chunk_cols = out_w

    # 3. Accumulators live on CPU to prevent VRAM growth across iterations
    vol_vals = torch.zeros((out_h, out_w), dtype=COMPUTE_DTYPE, device='cpu')
    valid_mask = torch.zeros((out_h, out_w), dtype=torch.bool, device='cpu')

    with torch.no_grad():
        for y_start in range(0, out_h, chunk_rows):
            y_end = min(y_start + chunk_rows, out_h)
            chunk_h = y_end - y_start

            row_start = y_start * stride
            row_end = (y_end - 1) * stride + tile_size

            for x_start in range(0, out_w, chunk_cols):
                x_end_out = min(x_start + chunk_cols, out_w)
                chunk_w = x_end_out - x_start

                col_start = x_start * stride
                col_end = (x_end_out - 1) * stride + tile_size

                # Push small raw chunk to GPU FIRST, avoiding CPU memory bandwidth bottleneck
                chunk_data = tensor_data[:, row_start:row_end, col_start:col_end].to(dtype=COMPUTE_DTYPE, device=target_device).unsqueeze(0)

                # Unfold directly on GPU VRAM
                chunk_unfold = torch.nn.functional.unfold(chunk_data, kernel_size=tile_size, stride=stride)
                chunk_windows = chunk_unfold.view(bands, N_pixels, -1).permute(2, 0, 1)  # (L_chunk, C, N_pixels)
                L_chunk = chunk_windows.shape[0]

                # Strict Validity: Window is only processed if ALL pixels and bands are valid
                chunk_is_valid = ~torch.isnan(chunk_windows).reshape(L_chunk, -1).any(dim=1)

                # Accumulate results on GPU, copy once per spatial chunk
                chunk_vol_vals = torch.zeros(L_chunk, dtype=COMPUTE_DTYPE, device=target_device)

                for i in range(0, L_chunk, batch_size):
                    batch_valid = chunk_is_valid[i:i + batch_size]
                    batch_len = batch_valid.shape[0]

                    if not batch_valid.any():
                        continue

                    valid_data = chunk_windows[i:i + batch_size][batch_valid]

                    _, volume = maximumDistance_volumes(valid_data, num_endmembers)

                    if volume.shape[1] > 2:
                        vol_val = torch.max(volume[:, 2:], dim=1)[0]
                    else:
                        vol_val = torch.zeros(volume.shape[0], dtype=COMPUTE_DTYPE, device=target_device)

                    chunk_vol_vals[i:i + batch_len][batch_valid] = vol_val

                # Single bulk D2H copy per spatial chunk
                vol_vals[y_start:y_end, x_start:x_end_out] = chunk_vol_vals.view(chunk_h, chunk_w).cpu()
                valid_mask[y_start:y_end, x_start:x_end_out] = chunk_is_valid.view(chunk_h, chunk_w).cpu()

                # Explicitly delete all GPU tensors to reduce CUDA allocator fragmentation
                del chunk_data, chunk_unfold, chunk_windows, chunk_vol_vals, chunk_is_valid

    # 4. Neighborhood map — assign each tile's volume directly to its center pixel
    center_offset = tile_size // 2

    neighborhood_map = np.full((height, width), np.nan, dtype=np.float32)
    vol_grid = vol_vals.numpy()
    valid_grid = valid_mask.numpy()

    neighborhood_map[center_offset:center_offset + out_h,
                     center_offset:center_offset + out_w] = vol_grid
    neighborhood_map[center_offset:center_offset + out_h,
                     center_offset:center_offset + out_w][~valid_grid] = np.nan

    # 5. Final Map — Fast Spatial Averaging via 2D Convolution (mean filter)
    clean_neighborhood = np.nan_to_num(neighborhood_map, nan=0.0)
    valid_binary = ~np.isnan(neighborhood_map)

    sum_map = uniform_filter(clean_neighborhood, size=tile_size, mode='constant', cval=0.0)
    count_map = uniform_filter(valid_binary.astype(np.float32), size=tile_size, mode='constant', cval=0.0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        final_map = np.where(count_map > 0, sum_map / count_map, np.nan)

    if return_neighborhood:
        return final_map, neighborhood_map
    return final_map


process_volume_sliding_tile_torch = process_volume_sliding_tile


def calculate_global_z_score_batch(volume_array, valid_pixel_mask=None, array_mean=None, array_std=None):
    """
    Calculates global Z-scores across a batch of spectral complexity volume frames (B, H, W).
    Accepts numpy ndarrays or torch Tensors on CPU/CUDA and returns numpy ndarrays.

    Args:
        volume_array (torch.Tensor or np.ndarray): 3D array/tensor [batch, height, width] of spectral complexity volumes.
        valid_pixel_mask (torch.Tensor or np.ndarray, optional): Boolean 2D or 3D mask of valid pixels.
        array_mean (float, torch.Tensor, or np.ndarray, optional): Pre-computed mean(s) (in log space).
        array_std (float, torch.Tensor, or np.ndarray, optional): Pre-computed standard deviation(s) (in log space).

    Returns:
        tuple: (z_scores, array_mean, array_std)
            - z_scores (np.ndarray): 3D array [batch, height, width] of Z-scores.
            - array_mean: Mean(s) used for normalization.
            - array_std: Standard deviation(s) used for normalization.
    """
    if isinstance(volume_array, torch.Tensor):
        vol_tensor = volume_array
        device = vol_tensor.device
        B, height, width = vol_tensor.shape
        global_valid_mask = vol_tensor > 0.0

        if array_mean is not None and array_std is not None:
            mean_t = array_mean if isinstance(array_mean, torch.Tensor) else torch.as_tensor(array_mean, dtype=torch.float32, device=device)
            std_t = array_std if isinstance(array_std, torch.Tensor) else torch.as_tensor(array_std, dtype=torch.float32, device=device)
            mean_t = mean_t.to(device=device, dtype=torch.float32)
            std_t = std_t.to(device=device, dtype=torch.float32)

            if mean_t.dim() == 0:
                mean_t = mean_t.view(1, 1, 1).expand(B, 1, 1)
                std_t = std_t.view(1, 1, 1).expand(B, 1, 1)
            elif mean_t.dim() == 1:
                mean_t = mean_t.view(-1, 1, 1)
                std_t = std_t.view(-1, 1, 1)

            if valid_pixel_mask is not None:
                mask_t = valid_pixel_mask if isinstance(valid_pixel_mask, torch.Tensor) else torch.as_tensor(valid_pixel_mask, dtype=torch.bool, device=device)
                mask_t = mask_t.to(device=device, dtype=torch.bool)
                if mask_t.dim() == 2:
                    mask = global_valid_mask & mask_t.unsqueeze(0)
                else:
                    mask = global_valid_mask & mask_t
            else:
                mask = global_valid_mask

            valid_eval = mask & (std_t > 0) & (~torch.isnan(mean_t))
            safe_vol = torch.where(valid_eval, vol_tensor, torch.ones_like(vol_tensor))
            log_vol = torch.log(safe_vol)
            z_scores_t = torch.where(valid_eval, (log_vol - mean_t) / std_t, torch.tensor(float("nan"), device=device, dtype=torch.float32))

            mean_ret = array_mean.detach().cpu().numpy() if isinstance(array_mean, torch.Tensor) else array_mean
            std_ret = array_std.detach().cpu().numpy() if isinstance(array_std, torch.Tensor) else array_std
            return z_scores_t.detach().cpu().numpy(), mean_ret, std_ret
        else:
            vol_np = vol_tensor.detach().cpu().numpy()
            mask_np = (
                valid_pixel_mask.detach().cpu().numpy() 
                if isinstance(valid_pixel_mask, torch.Tensor) 
                else np.asarray(valid_pixel_mask, dtype=bool)
            ) if valid_pixel_mask is not None else None
            z_scores = np.full((B, height, width), np.nan, dtype=np.float32)
            for b in range(B):
                vm = mask_np[b] if (mask_np is not None and mask_np.ndim == 3) else mask_np
                z_scores[b], _, _ = calculate_global_z_score(vol_np[b], valid_pixel_mask=vm)
            return z_scores, np.nan, np.nan

    vol_np = np.asarray(volume_array)
    mask_np = (
        valid_pixel_mask.detach().cpu().numpy() 
        if isinstance(valid_pixel_mask, torch.Tensor) 
        else np.asarray(valid_pixel_mask, dtype=bool)
    ) if valid_pixel_mask is not None else None

    B, height, width = vol_np.shape
    z_scores = np.full((B, height, width), np.nan, dtype=np.float32)
    global_valid_mask = vol_np > 0.0

    if array_mean is not None and array_std is not None:
        mean_val = array_mean.detach().cpu().numpy() if isinstance(array_mean, torch.Tensor) else array_mean
        std_val = array_std.detach().cpu().numpy() if isinstance(array_std, torch.Tensor) else array_std
        mean_arr = np.asarray(mean_val, dtype=np.float32)
        std_arr = np.asarray(std_val, dtype=np.float32)

        if mean_arr.ndim == 0:
            mean_arr = np.full((B, 1, 1), float(mean_arr), dtype=np.float32)
            std_arr = np.full((B, 1, 1), float(std_arr), dtype=np.float32)
        elif mean_arr.ndim == 1:
            mean_arr = mean_arr[:, np.newaxis, np.newaxis]
            std_arr = std_arr[:, np.newaxis, np.newaxis]

        if mask_np is not None:
            if mask_np.ndim == 2:
                mask = global_valid_mask & mask_np[np.newaxis, :, :]
            else:
                mask = global_valid_mask & mask_np
        else:
            mask = global_valid_mask

        valid_eval = mask & (std_arr > 0) & (~np.isnan(mean_arr))
        with np.errstate(divide="ignore", invalid="ignore"):
            z_scores = np.where(valid_eval, (np.log(vol_np) - mean_arr) / std_arr, np.nan)

        return z_scores, mean_val, std_val
    else:
        for b in range(B):
            vm = mask_np[b] if (mask_np is not None and mask_np.ndim == 3) else mask_np
            z_scores[b], _, _ = calculate_global_z_score(vol_np[b], valid_pixel_mask=vm)
        return z_scores, np.nan, np.nan


def calculate_global_z_score(volume_array, valid_pixel_mask=None, array_mean=None, array_std=None):
    """
    Calculates the global Z-score for an entire frame of spectral complexity volumes.
    Decouples the evaluation space from the background statistics space by strictly 
    calculating the mean and standard deviation from radiometrically valid pixels, 
    preventing artifacts from skewing the background model.
    Alternatively accepts a separately determined mean and standard deviation (array_mean, array_std).

    Supports volume_array and valid_pixel_mask as numpy ndarrays or torch Tensors on CPU/CUDA.

    Args:
        volume_array (torch.Tensor or np.ndarray): 2D array/tensor [height, width] of spectral complexity volumes.
        valid_pixel_mask (torch.Tensor or np.ndarray, optional): Boolean 2D mask [height, width] of radiometrically
            valid pixels. Required if array_mean and array_std are not provided.
        array_mean (float, torch.Tensor, or np.ndarray, optional): Pre-computed mean (in log space) to use for Z-score.
        array_std (float, torch.Tensor, or np.ndarray, optional): Pre-computed standard deviation (in log space) to use for Z-score.

    Returns:
        tuple: (z_scores, global_mean, global_std)
            - z_scores (torch.Tensor or np.ndarray): 2D array/tensor [height, width] of Z-scores.
            - global_mean (float, torch.Tensor, or np.ndarray): Mean used for normalization.
            - global_std (float, torch.Tensor, or np.ndarray): Standard deviation used for normalization.
    """
    # Support 3D batched inputs (B, height, width)
    if (isinstance(volume_array, torch.Tensor) and volume_array.dim() == 3) or (hasattr(volume_array, "ndim") and volume_array.ndim == 3):
        return calculate_global_z_score_batch(volume_array, valid_pixel_mask=valid_pixel_mask, array_mean=array_mean, array_std=array_std)

    is_torch = isinstance(volume_array, torch.Tensor)
    if is_torch:
        orig_device = volume_array.device
        orig_dtype = volume_array.dtype
        vol_np = volume_array.detach().cpu().numpy()
        mask_np = (
            valid_pixel_mask.detach().cpu().numpy() 
            if isinstance(valid_pixel_mask, torch.Tensor) 
            else np.asarray(valid_pixel_mask, dtype=bool)
        ) if valid_pixel_mask is not None else None
    else:
        vol_np = np.asarray(volume_array)
        mask_np = np.asarray(valid_pixel_mask, dtype=bool) if valid_pixel_mask is not None else None

    height, width = vol_np.shape
    z_scores = np.full((height, width), np.nan, dtype=np.float32)
    
    # Identify globally valid pixels (strictly positive for log transform)
    global_valid_mask = vol_np > 0.0

    if array_mean is not None and array_std is not None:
        mean_val = array_mean.detach().cpu().numpy() if isinstance(array_mean, torch.Tensor) else array_mean
        std_val = array_std.detach().cpu().numpy() if isinstance(array_std, torch.Tensor) else array_std

        if np.ndim(mean_val) == 0:
            global_mean = float(mean_val)
            global_std = float(std_val)
            if np.isnan(global_mean) or np.isnan(global_std):
                warnings.warn("NaN found in array_mean or array_std", UserWarning)
                return z_scores, np.nan, np.nan
            if global_std <= 0:
                raise ValueError(f"calculate_global_z_score failed: Global standard deviation must be positive, got {global_std}.")
            mean_eval = global_mean
            std_eval = global_std
        else:
            mean_arr = np.asarray(mean_val, dtype=np.float32)
            std_arr = np.asarray(std_val, dtype=np.float32)
            if mean_arr.size == 1:
                global_mean = float(mean_arr.item())
                global_std = float(std_arr.item())
                if np.isnan(global_mean) or np.isnan(global_std):
                    warnings.warn("NaN found in array_mean or array_std", UserWarning)
                    return z_scores, np.nan, np.nan
                if global_std <= 0:
                    raise ValueError(f"calculate_global_z_score failed: Global standard deviation must be positive, got {global_std}.")
                mean_eval = global_mean
                std_eval = global_std
            else:
                if mean_arr.shape != (height, width) or std_arr.shape != (height, width):
                    raise ValueError(f"calculate_global_z_score failed: array_mean shape {mean_arr.shape} and array_std shape {std_arr.shape} must match volume_array shape ({height}, {width}).")
                if np.any(std_arr[global_valid_mask] <= 0):
                    raise ValueError("calculate_global_z_score failed: Global standard deviation must be positive.")
                if np.any(np.isnan(mean_arr[global_valid_mask])) or np.any(np.isnan(std_arr[global_valid_mask])):
                    warnings.warn("NaN found in array_mean or array_std", UserWarning)
                    return z_scores, np.nan, np.nan
                global_mean = mean_arr
                global_std = std_arr
                mean_eval = mean_arr[global_valid_mask]
                std_eval = std_arr[global_valid_mask]
    elif (array_mean is not None) or (array_std is not None):
        raise ValueError("calculate_global_z_score failed: Both array_mean and array_std must be provided together.")
    else:
        if mask_np is None:
            raise ValueError("calculate_global_z_score failed: valid_pixel_mask must be provided when array_mean and array_std are not specified.")

        # Intersect with radiometrically valid pixels for the statistical background model
        stats_mask = global_valid_mask & mask_np
        
        # Extract subset volumes strictly for statistical estimation
        stats_vols = vol_np[stats_mask]

        # Graceful fallback per user directive: Return NaNs for entire frame if no valid background exists.
        if len(stats_vols) < 9 or len(np.unique(stats_vols)) <= 1:
            if len(stats_vols) < 9:
                warnings.warn("Insufficient valid pixels", UserWarning)
            elif len(np.unique(stats_vols)) <= 1:
                warnings.warn("zero variance found", UserWarning)
            if is_torch:
                z_scores = torch.from_numpy(z_scores).to(device=orig_device, dtype=orig_dtype if orig_dtype.is_floating_point else torch.float32)
            return z_scores, np.nan, np.nan
            
        log_stats_vols = np.log(stats_vols)
        
        # Calculate global scene statistics (using ddof=1 for unbiased sample estimator)
        global_mean = float(np.mean(log_stats_vols))
        global_std = float(np.std(log_stats_vols, ddof=1))
        
        # Strict failure handling: Prevent training on synthetically flat frames
        if global_std == 0:
            raise ValueError("calculate_global_z_score failed: Global standard deviation of the radiometrically valid subset is exactly zero.")
        mean_eval = global_mean
        std_eval = global_std
        
    # Evaluate ALL geometrically valid pixels using the pure background model
    apply_vols = vol_np[global_valid_mask]
    if len(apply_vols) > 0:
        log_apply_vols = np.log(apply_vols)
        # Apply standard Z-score equation
        z_scores[global_valid_mask] = (log_apply_vols - mean_eval) / std_eval
    
    if is_torch:
        z_scores = torch.from_numpy(z_scores).to(device=orig_device, dtype=orig_dtype if orig_dtype.is_floating_point else torch.float32)
    return z_scores, global_mean, global_std

def calc_ndvi_frame(frame_data, red_idx, nir_idx):
    """
    Calculates the Normalized Difference Vegetation Index (NDVI) per pixel.
    
    Formula: NDVI = (NIR - Red) / (NIR + Red)
    
    Args:
        frame_data (np.ndarray or torch.Tensor): 3D image cube [bands, height, width]
        red_idx (int): Index of the Red band (~665-670nm).
        nir_idx (int): Index of the Near-Infrared (NIR) band (~842-865nm).
                       
    Returns:
        ndvi (np.ndarray or torch.Tensor): 2D array [height, width] containing NDVI values bound between [-1.0, 1.0].
                            Invalid pixels evaluate strictly to np.nan.
    """
    is_torch = False
    if isinstance(frame_data, torch.Tensor):
        is_torch = True
        orig_device = frame_data.device
        orig_dtype = frame_data.dtype
        frame_np = frame_data.detach().cpu().numpy()
    else:
        frame_np = np.asarray(frame_data)

    bands, height, width = frame_np.shape
    if red_idx < 0 or red_idx >= bands or nir_idx < 0 or nir_idx >= bands:
        raise IndexError(f"Band index out of bounds: red_idx={red_idx}, nir_idx={nir_idx} for bands={bands}")

    red = frame_np[red_idx, :, :].astype(np.float32)
    nir = frame_np[nir_idx, :, :].astype(np.float32)

    denominator = nir + red

    with np.errstate(divide='ignore', invalid='ignore'):
        ndvi = (nir - red) / denominator

    ndvi[np.isinf(ndvi)] = np.nan
    ndvi = np.clip(ndvi, -1.0, 1.0)

    if is_torch:
        out_dtype = orig_dtype if orig_dtype.is_floating_point else torch.float32
        return torch.from_numpy(ndvi).to(device=orig_device, dtype=out_dtype)
    return ndvi

def scale_to_int16(arr, scale_factor=10000.0, nodata_val=-32768, nodata_value=None):
    """
    Quantizes floating point array or tensor to signed int16 with scaling and nodata handling.
    """
    if nodata_value is not None:
        nodata_val = nodata_value
    out = np.full(arr.shape, nodata_val, dtype=np.int16)
    valid = ~np.isnan(arr) & (arr != nodata_val)
    scaled = np.round(arr[valid] * scale_factor)
    out[valid] = scaled.astype(np.int16)
    return out

# =====================================================================
# BACKWARD COMPATIBILITY ALIASES & DEV RESEARCH EXTENSIONS
# =====================================================================

maximumDistance_torch = maximumDistance
def calcGramLocalVolumes_QR_torch(endmembers, localization_vector):
    """
    Batched, GPU-accelerated Simplex Volumes using Vectorized Modified Gram-Schmidt (MGS).
    Follows Gantmacher's theorem equating volume to the product of orthogonal heights.
    """
    if localization_vector.dim() == 1:
        localization_vector = localization_vector.unsqueeze(0).unsqueeze(2)
    elif localization_vector.dim() == 2:
        localization_vector = localization_vector.unsqueeze(2)
        
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
            
    volumes = torch.cumprod(heights, dim=-1) # (B, E)
    return volumes

maximumDistance_volumes_torch = maximumDistance_volumes

def maximumDistance_cholesky_torch(img_cube, num_endmembers, return_heights=False):
    """
    Batched, GPU-accelerated endmember extraction and localized Gram volume calculation
    via Gram-space Pivoted Cholesky decomposition (Harbrecht et al., 2012).

    Localizes the pixel neighborhood to the minimum-norm pixel, forms the local spatial
    Gram matrix K = Y_loc^T * Y_loc, and extracts endmembers, volumes, and orthogonal heights
    via batched Schur complement downdating.

    Args:
        img_cube: Tensor of shape (B, C, N) [Batch, Bands, Pixels]
        num_endmembers: int, total number of endmembers to extract (k).
        return_heights: bool, optional. If True, also returns orthogonal heights (B, k).

    Returns:
        endmembers: Tensor of shape (B, C, num_endmembers)
        volumes:    Tensor of shape (B, num_endmembers)
        heights:    (Only if return_heights=True) Tensor of shape (B, num_endmembers)
    """
    B, C, N = img_cube.shape
    device = img_cube.device
    dtype = img_cube.dtype
    b_idx = torch.arange(B, device=device)

    # 1. Identify seed pixels by magnitude
    magnitude_sq = torch.sum(img_cube ** 2, dim=1)
    idx_max = torch.argmax(magnitude_sq, dim=1)
    idx_min = torch.argmin(magnitude_sq, dim=1)

    endmembers = torch.zeros(B, C, num_endmembers, dtype=dtype, device=device)
    endmembers[:, :, 0] = img_cube[b_idx, :, idx_max]
    endmembers[:, :, 1] = img_cube[b_idx, :, idx_min]

    # 2. Localize: translate neighborhood so min-norm pixel is at origin (cast to float64)
    origin = img_cube[b_idx, :, idx_min].unsqueeze(2)
    Y_loc = (img_cube - origin).to(torch.float64)

    # 3. Local spatial Gram matrix K = Y_loc^T * Y_loc (B, N, N)
    K = torch.bmm(Y_loc.transpose(1, 2), Y_loc)
    diag_K = torch.diagonal(K, dim1=1, dim2=2).clone()

    # 4. First localized endmember height
    h1_sq = diag_K[b_idx, idx_max].clamp(min=0.0)
    h1 = torch.sqrt(h1_sq)

    volumes = torch.zeros(B, num_endmembers, dtype=torch.float64, device=device)
    volumes[:, 1] = h1
    if return_heights:
        heights = torch.zeros(B, num_endmembers, dtype=torch.float64, device=device)
        heights[:, 1] = h1

    L = torch.zeros(B, N, num_endmembers, dtype=torch.float64, device=device)
    h1_safe = torch.where(h1 > 1e-12, h1, torch.ones_like(h1)).unsqueeze(1)
    L[:, :, 0] = K[b_idx, :, idx_max] / h1_safe
    diag_K -= L[:, :, 0] ** 2

    selected_mask = torch.zeros(B, N, dtype=torch.bool, device=device)
    selected_mask[b_idx, idx_min] = True
    selected_mask[b_idx, idx_max] = True
    diag_K[selected_mask] = -float('inf')

    # 5. Iterative extraction via greedy diagonal Schur downdating
    for j in range(2, num_endmembers):
        idx_next = torch.argmax(diag_K, dim=1)
        h_sq = diag_K[b_idx, idx_next].clamp(min=0.0)
        h_k = torch.sqrt(h_sq)

        endmembers[:, :, j] = img_cube[b_idx, :, idx_next]
        volumes[:, j] = volumes[:, j - 1] * h_k
        if return_heights:
            heights[:, j] = h_k

        selected_mask[b_idx, idx_next] = True

        k_col = K[b_idx, :, idx_next]
        L_prev = L[:, :, :j-1]
        L_pivot = L[b_idx, idx_next, :j-1].unsqueeze(2)
        proj = torch.bmm(L_prev, L_pivot).squeeze(2)
        col = k_col - proj

        h_k_safe = torch.where(h_k > 1e-12, h_k, torch.ones_like(h_k)).unsqueeze(1)
        L[:, :, j - 1] = col / h_k_safe

        diag_K -= L[:, :, j - 1] ** 2
        diag_K[selected_mask] = -float('inf')

    if return_heights:
        return endmembers, volumes.to(dtype), heights.to(dtype)
    return endmembers, volumes.to(dtype)


def maximumDistance_cholesky_torch_test(img_cube, num_endmembers):
    """
    Test version of maximumDistance_cholesky_torch that returns orthogonal heights.
    """
    return maximumDistance_cholesky_torch(img_cube, num_endmembers, return_heights=True)


def estimate_effective_dimensionality_torch(img_cube, k_max, sigma_n, kappa=3.0, heights=None):
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
        heights: Tensor of shape (B_batch, k_max) float32, optional precomputed heights.

    Returns:
        esd: (B_batch,) int32 tensor: effective spectral dimensionality count.
        sii: (B_batch,) float32 tensor: Spectral Innovation Integral.
        heights: (B_batch, k_max) float32 tensor: full orthogonal height profile.
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

    # --- Extract heights via the existing MaxD engine if not supplied ---
    if heights is None:
        _, _, heights = maximumDistance_volumes_torch(img_cube, k_max, return_heights=True)
    elif heights.shape[1] > k_max:
        heights = heights[:, :k_max]

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


def process_cholesky_volume_sliding_tile(frame_data, tile_size, stride, num_endmembers, 
                                          compute_esd=False, compute_sii=False, 
                                          global_sigma_n=0.0, esd_kappa=3.0):
    """
    GPU-accelerated Spectral Complexity calculation using Gram-space Pivoted Cholesky
    decomposition with CPU-host memory spooling.
    
    The spatial unfold tensor is kept in system RAM to avoid saturating GPU VRAM.
    Small batch chunks are transferred to the GPU for batched Gram matrix formation
    and pivoted Cholesky Schur complement downdating, then volume results are
    immediately returned to CPU. This keeps VRAM utilization bounded to a fixed
    fraction of total GPU memory regardless of image or band count.
    
    Precision: float32 interface, internally promotes to float64 for Gram Cholesky
    decomposition to avoid catastrophic numerical cancellation.
    
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
    
    # 1. Load data onto CPU only: system RAM absorbs the full unfold allocation
    tensor_data = torch.from_numpy(frame_data).to('cpu', dtype=COMPUTE_DTYPE)
    
    # Filter out anomalous data outside surface reflectance range +/- 1.5
    tensor_data[(tensor_data < -1.5) | (tensor_data > 1.5)] = float('nan')
    
    # 2. Calculate dynamic batch size and chunk size to maximize GPU utilization 
    #    without exceeding 50% VRAM.
    N_pixels = tile_size * tile_size
    out_h = (height - tile_size) // stride + 1
    out_w = (width - tile_size) // stride + 1
    L = out_h * out_w

    if device == 'cuda':
        total_vram = torch.cuda.get_device_properties(device).total_memory
        target_vram = total_vram * 0.50
        # Cholesky uses float32 input (bands*N*4) and internal float64 for Gram K (N*N*8),
        # localized data Y_loc (bands*N*8), and Cholesky factor L (N*k*8)
        bytes_per_window = (bands * N_pixels * 4) + (bands * N_pixels * 8) + (N_pixels * N_pixels * 8) + (N_pixels * num_endmembers * 8)
        if compute_esd or compute_sii:
            bytes_per_window += num_endmembers * 8
        # Add safety margin for PyTorch memory allocator overhead
        bytes_per_window = int(bytes_per_window * 2.5)
        batch_size = max(1, int(target_vram // max(1, bytes_per_window)))
        
        # Cap batch size to prevent Windows TDR (Timeout Detection and Recovery) timeouts.
        batch_size = min(batch_size, 20000)
        
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
    
    # 4. Process windows in memory-safe batches: transfer chunk to GPU, compute, return to CPU
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
            
                # 4a. Combined extraction and volume calculation via Pivoted Cholesky
                if compute_esd or compute_sii:
                    endmembers, volume, heights = maximumDistance_cholesky_torch(
                        valid_data, num_endmembers, return_heights=True
                    )
                else:
                    endmembers, volume = maximumDistance_cholesky_torch(
                        valid_data, num_endmembers, return_heights=False
                    )
                
                # 4b. Extract target metric and immediately move result to CPU
                if volume.shape[1] > 2:
                    vol_val = torch.max(volume[:, 2:], dim=1)[0]
                else:
                    vol_val = torch.zeros(volume.shape[0], dtype=COMPUTE_DTYPE, device=device)
                
                vol_vals[global_start_idx + i : global_start_idx + i + batch_windows.shape[0]][batch_valid.cpu()] = vol_val.cpu()
                
                # 4c. Compute ESD / SII if requested using precomputed heights
                if compute_esd or compute_sii:
                    esd_batch, sii_batch, _ = estimate_effective_dimensionality_torch(
                        valid_data, k_max=num_endmembers, sigma_n=global_sigma_n, kappa=esd_kappa, heights=heights
                    )
                    if compute_esd:
                        esd_vals[global_start_idx + i : global_start_idx + i + batch_windows.shape[0]][batch_valid.cpu()] = esd_batch.to(torch.uint8).cpu()
                    if compute_sii:
                        sii_vals[global_start_idx + i : global_start_idx + i + batch_windows.shape[0]][batch_valid.cpu()] = sii_batch.cpu()
                
                # Explicitly synchronize to flush the GPU command queue.
                if device == 'cuda':
                    torch.cuda.synchronize()
        
    # 5. Neighborhood map: assign each tile's volume directly to its center pixel.
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
                     
    # 6. Final Map: Fast spatial averaging via 2D convolution (mean filter)
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


