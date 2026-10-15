import numpy as np
import h5py
import tkinter as tk
from tqdm import tqdm
from tkinter import filedialog
import torch

TILE_SIZE=3
TILE_SIZES = [101]#3,5,7,9,11,15,19,35,55,65,81,101]#[3,5,7,9,11,15,19,35,55,65,81,101]
NUM_ENDMEMBERS = 40
CALCULATE_NOISE_STATS = True  # Set to False to skip calculating noise equivalent heights

def maximumDistance_volumes_torch(img_cube, num_endmembers, Sigma_n=None, Sigma_n_tensor=None):
    """
    Combined MaxD endmember extraction and localized Gram volume calculation
    in a single iterative Orthogonal Subspace Projection (OSP) pass. 
    Localizes the pixel neighborhood to the minimum-norm pixel, then iteratively 
    selects endmembers by maximum orthogonal projection distance while 
    simultaneously recording the heights needed for the Gram volume curve.

    Uses a lazy projection strategy: the localized data tensor is kept immutable
    and residual squared norms are updated incrementally, reducing per-iteration
    cost from 3x to 1x full-tensor passes. The orthogonal residual for each
    selected pixel is computed on-demand via the accumulated orthonormal basis.

    Args:
        img_cube: Tensor of shape (B, C, N) [Batch, Bands, Pixels]
        num_endmembers: int, total number of endmembers to extract (k).
                        The min-norm pixel consumes one slot as the local origin,
                        so (k-1) orthogonal heights are produced.
        Sigma_n: numpy array of noise covariance matrices (optional).
        Sigma_n_tensor: Pre-created GPU tensor of Sigma_n (optional). If provided,
                        avoids repeated numpy->GPU copies. Created once per tile size
                        in the caller to reduce CUDA allocator fragmentation.
    Returns:
        endmembers: Tensor of shape (B, C, num_endmembers) — raw (non-localized)
                    endmember spectra. Index 0 is max-norm, index 1 is min-norm
                    (the local origin).
        volumes: Tensor of shape (B, num_endmembers) — the localized Gram volume
                 curve. volumes[:, 0] = -inf (the origin has zero volume),
                 volumes[:, 1] = log(h_1), volumes[:, 2] = log(h_1)+log(h_2), etc.
        heights: Tensor of shape (B, num_endmembers) —
                 the orthogonal projection heights (h_n). heights[:, 0] = 0 (origin),
                 heights[:, 1] = h_1, heights[:, j] = h_j for j >= 1.
        noise_heights: Tensor or None.
    """
    B, C, N = img_cube.shape
    orig_dtype = img_cube.dtype
    img_cube = img_cube.to(torch.float64)
    device = img_cube.device
    dtype = torch.float64

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
    # local_data is kept IMMUTABLE for the rest of the function (lazy projection)
    origin = img_cube[b_idx, :, idx_minnorm].unsqueeze(2)  # (B, C, 1)
    local_data = img_cube - origin  # (B, C, N)

    # --- Prepare volume and height output ---
    # volumes[:, 0] = -inf (origin contributes no volume)
    # volumes[:, j] for j >= 1 is the cumulative sum of log heights log(h_1)..-log(h_{j})
    volumes = torch.full((B, num_endmembers), -float('inf'), dtype=dtype, device=device)
    heights = torch.zeros(B, num_endmembers, dtype=dtype, device=device)

    # Orthonormal basis accumulated across iterations (always needed for lazy projection)
    U_basis = torch.zeros((B, C, num_endmembers - 1), dtype=dtype, device=device)

    # Incremental residual squared norms — avoids recomputing from the full (B,C,N) tensor
    residual_sq = torch.sum(local_data ** 2, dim=1)  # (B, N)
    # Mark seed pixels so they are never re-selected (-inf survives subsequent subtractions)
    residual_sq[b_idx, idx_minnorm] = -float('inf')
    residual_sq[b_idx, idx_maxnorm] = -float('inf')

    if Sigma_n is not None:
        N_cov = Sigma_n.shape[0]
        noise_heights = torch.zeros((N_cov, B, num_endmembers), dtype=dtype, device=device)
        # Use pre-created tensor if provided (avoids repeated numpy->GPU copy per batch)
        if Sigma_n_tensor is None:
            Sigma_n_tensor = torch.tensor(Sigma_n, dtype=dtype, device=device)
        else:
            Sigma_n_tensor = Sigma_n_tensor.to(dtype=dtype)
        
        # Precompute initial traces for P_perp = I
        trace_1 = torch.diagonal(Sigma_n_tensor, dim1=-2, dim2=-1).sum(-1).unsqueeze(1).repeat(1, B) # (N_cov, B)
        Sigma_sq = torch.matmul(Sigma_n_tensor, Sigma_n_tensor)
        trace_2 = torch.diagonal(Sigma_sq, dim1=-2, dim2=-1).sum(-1).unsqueeze(1).repeat(1, B) # (N_cov, B)
    else:
        noise_heights = None

    # --- First endmember (max-norm pixel, already identified) ---
    v_first = local_data[b_idx, :, idx_maxnorm]  # (B, C)
    h1 = torch.norm(v_first, dim=1)  # (B,)
    volumes[:, 1] = torch.log(h1)
    heights[:, 1] = h1

    u1 = v_first / h1.unsqueeze(1)  # (B, C)
    U_basis[:, :, 0] = u1

    # Incremental residual_sq update: since u1 is orthogonal to the (empty) prior basis,
    # u1^T @ residual_j = u1^T @ x_j for all pixels j
    proj_coeffs = torch.bmm(u1.unsqueeze(1), local_data).squeeze(1)  # (B, N)
    residual_sq -= proj_coeffs ** 2

    if Sigma_n is not None:
        noise_heights[:, :, 1] = calculate_noise_threshold_from_traces_torch(trace_1, trace_2, N)
        
        # Vectorized across N_cov (Sigma_n matrices)
        w = torch.einsum('bi,nij->nbj', u1, Sigma_n_tensor)
        u_Sigma_u = (u1.unsqueeze(0) * w).sum(dim=2)
        
        # P_perp is I at step 0, so P_w = w
        P_w = w
        w_P_w = (w * P_w).sum(dim=2)
        
        trace_1 = trace_1 - u_Sigma_u
        trace_2 = trace_2 - 2.0 * w_P_w + u_Sigma_u ** 2

    # --- Iterative extraction of remaining endmembers ---
    for i in range(2, num_endmembers):
        # Find pixel with maximum residual squared norm (selected pixels are already -inf)
        idx_new = torch.argmax(residual_sq, dim=1)
        residual_sq[b_idx, idx_new] = -float('inf')  # mark as selected

        # Record the raw (non-localized) endmember
        endmembers[:, :, i] = img_cube[b_idx, :, idx_new]

        # Extract original localized vector and project into orthogonal complement
        # of the accumulated basis: v_new = x_new - U_k @ (U_k^T @ x_new)
        x_new = local_data[b_idx, :, idx_new]  # (B, C)
        k = i - 1  # number of basis vectors so far
        U_k = U_basis[:, :, :k]  # (B, C, k)
        coeffs = torch.bmm(U_k.transpose(1, 2), x_new.unsqueeze(2))  # (B, k, 1)
        v_new = x_new - torch.bmm(U_k, coeffs).squeeze(2)  # (B, C)

        h_j = torch.norm(v_new, dim=1)  # (B,)

        # Cumulative addition in log space: log_volume at step i = log_volume at step (i-1) + log(h_j)
        volumes[:, i] = volumes[:, i - 1] + torch.log(h_j)
        heights[:, i] = h_j

        uj = v_new / h_j.unsqueeze(1)  # (B, C)
        U_basis[:, :, k] = uj

        # Incremental residual_sq update: since uj is orthogonal to U_k by construction,
        # uj^T @ residual_j = uj^T @ x_j (project against immutable local_data)
        proj_coeffs = torch.bmm(uj.unsqueeze(1), local_data).squeeze(1)  # (B, N)
        residual_sq -= proj_coeffs ** 2

        if Sigma_n is not None:
            noise_heights[:, :, i] = calculate_noise_threshold_from_traces_torch(trace_1, trace_2, N)
            
            # Vectorized across N_cov (Sigma_n matrices)
            w = torch.einsum('bi,nij->nbj', uj, Sigma_n_tensor)
            u_Sigma_u = (uj.unsqueeze(0) * w).sum(dim=2)
            
            # U_k has shape (B, C, k) where C is bands, k is number of basis vectors so far
            # w has shape (N_cov, B, C)
            # U_k^T @ w -> (N_cov, B, k)
            U_T_w = torch.einsum('bik,nbi->nbk', U_k, w)
            
            # U_k @ (U_k^T @ w) -> (N_cov, B, C)
            proj_w = torch.einsum('bik,nbk->nbi', U_k, U_T_w)
            
            P_w = w - proj_w
            w_P_w = (w * P_w).sum(dim=2)
            
            trace_1 = trace_1 - u_Sigma_u
            trace_2 = trace_2 - 2.0 * w_P_w + u_Sigma_u ** 2
    if noise_heights is not None:
        noise_heights = noise_heights.to(orig_dtype)
    return endmembers.to(orig_dtype), volumes.to(orig_dtype), heights.to(orig_dtype), noise_heights


def maximumDistance_volumes(pixels, num_endmembers, Sigma_n=None):
    """
    Combined endmember extraction and Gram volume estimation in a single
    iterative Orthogonal Subspace Projection (OSP) pass.

    Localizes the pixel neighborhood by subtracting the minimum-norm pixel
    (the local origin), then iteratively selects endmembers by largest
    orthogonal projection distance while recording the heights needed for 
    the volume curve.
    """
    if pixels.ndim == 3:
        h, w, num_bands = pixels.shape
        pixels = np.reshape(pixels, (h * w, num_bands), order="F").astype(np.float64).T
    elif pixels.ndim == 2:
        pixels = pixels.astype(np.float64)

    num_bands, num_pixels = pixels.shape

    if num_pixels < num_endmembers:
        return (
            np.full((num_bands, num_endmembers), np.nan),
            np.full(num_endmembers, np.nan),
            np.full(num_endmembers, np.nan),
            None
        )

    # --- Identify the two seed pixels by spectral magnitude ---
    magnitudes = np.linalg.norm(pixels, axis=0)
    idx_maxnorm = np.argmax(magnitudes)
    idx_minnorm = np.argmin(magnitudes)

    # Store raw endmembers in extraction order
    endmembers = np.zeros((num_bands, num_endmembers), dtype=pixels.dtype)
    endmembers[:, 0] = pixels[:, idx_maxnorm]
    endmembers[:, 1] = pixels[:, idx_minnorm]

    # --- Localize: shift so the min-norm pixel is at the origin ---
    origin = pixels[:, idx_minnorm].reshape(-1, 1)    # (bands, 1)
    residuals = pixels - origin                        # (bands, num_pixels)

    # --- Prepare volume and height output ---
    volumes = np.full(num_endmembers, -np.inf, dtype=np.float64)
    heights = np.zeros(num_endmembers, dtype=np.float64)

    if Sigma_n is not None:
        N_cov = Sigma_n.shape[0]
        noise_heights = np.zeros((N_cov, num_endmembers), dtype=np.float64)
        P_perp = np.eye(num_bands, dtype=np.float64)
    else:
        noise_heights = None

    selected_indices = [idx_minnorm, idx_maxnorm]

    # --- First endmember: the localized max-norm pixel ---
    v = residuals[:, idx_maxnorm].copy()
    h_dist = np.linalg.norm(v)
    volumes[1] = np.log(h_dist)
    heights[1] = h_dist

    u1 = v / h_dist

    if Sigma_n is not None:
        noise_heights[:, 1] = calculate_noise_threshold_closed_form(Sigma_n, P_perp, num_pixels)
        P_perp -= np.outer(u1, u1)

    # Project all pixels into the subspace orthogonal to v
    proj_component = np.outer(u1, np.dot(u1, residuals))
    residuals -= proj_component

    # --- Iteratively extract remaining endmembers ---
    for j in range(2, num_endmembers):
        # Magnitudes of unselected pixels in the current orthogonal subspace
        residual_mag = np.sum(residuals ** 2, axis=0)  # (num_pixels,)
        residual_mag[selected_indices] = -np.inf

        idx_new = np.argmax(residual_mag)
        endmembers[:, j] = pixels[:, idx_new]
        selected_indices.append(idx_new)

        # The orthogonal distance is the magnitude of the projected vector
        v = residuals[:, idx_new].copy()
        h_dist = np.linalg.norm(v)

        # Log volume is the cumulative sum of log orthogonal heights
        volumes[j] = volumes[j - 1] + np.log(h_dist)
        heights[j] = h_dist

        uj = v / h_dist

        if Sigma_n is not None:
            noise_heights[:, j] = calculate_noise_threshold_closed_form(Sigma_n, P_perp, num_pixels)
            P_perp -= np.outer(uj, uj)

        # Project all pixels into the subspace orthogonal to the new endmember
        proj_component = np.outer(uj, np.dot(uj, residuals))
        residuals -= proj_component

    return endmembers, volumes, heights, noise_heights


def process_volume_sliding_tile(frame_data, valid_mask=None, tile_size=TILE_SIZE, num_endmembers=NUM_ENDMEMBERS, Sigma_n=None, valid_sources=None):
    """
    Sliding window processing.
    Strict Validity: Window is only processed if ALL pixels are valid.
    Output is masked with NaN for any pixel identified as invalid.
    """
    if num_endmembers > tile_size * tile_size:
        print(f"Warning: num_endmembers ({num_endmembers}) > pixels in a tile ({tile_size*tile_size}). Clamping to {tile_size*tile_size+1}.")
        num_endmembers = tile_size * tile_size+1

    bands, height, width = frame_data.shape
    img = np.transpose(frame_data, (1, 2, 0))
    
    center_offset = tile_size // 2
    
    neighborhood_map = np.full((height, width), np.nan, dtype=np.float32)
    heights_map = np.full((height, width, num_endmembers), np.nan, dtype=np.float32)
    if Sigma_n is not None and valid_sources is not None:
        noise_heights_map = np.full((len(valid_sources), height, width, num_endmembers), np.nan, dtype=np.float32)

    
    for y_start in tqdm(range(0, height - tile_size + 1), desc="Sliding window", leave=False):
        for x_start in range(0, width - tile_size + 1):
            y_end, x_end = y_start + tile_size, x_start + tile_size
            
            tile = img[y_start:y_end, x_start:x_end, :]
            
            # If explicit valid_mask is provided, ensure all pixels in the tile are valid
            if valid_mask is not None:
                if not np.all(valid_mask[y_start:y_end, x_start:x_end]):
                    continue
            
            # Pre-emptive validity check for NaN (legacy fallback)
            if np.isnan(tile).any():
                continue
                
            tile_flat = np.reshape(tile, (tile_size * tile_size, bands), order="F").T
            
            endmembers, volumes, heights, noise_heights = maximumDistance_volumes(tile_flat, num_endmembers, Sigma_n)
            
            if len(volumes) > 2:
                vol_val = np.max(volumes[2:])
            else:
                vol_val = -np.inf
                
            cy = y_start + center_offset
            cx = x_start + center_offset
            neighborhood_map[cy, cx] = vol_val
            heights_map[cy, cx, :] = heights
            if Sigma_n is not None and valid_sources is not None:
                noise_heights_map[:, cy, cx, :] = noise_heights

    # Add the final datasets using naming conventions
    datasets = {
        f'neighborhood_map_{tile_size}x{tile_size}': neighborhood_map,
        f'heights_map_{tile_size}x{tile_size}': heights_map
    }
    if Sigma_n is not None and valid_sources is not None:
        for i, src in enumerate(valid_sources):
            datasets[f'noise_heights_map_{src}_{tile_size}x{tile_size}'] = noise_heights_map[i]

        
    return datasets

@torch.no_grad()
def process_volume_sliding_tile_torch(frame_data, valid_mask=None, tile_size=TILE_SIZE, num_endmembers=NUM_ENDMEMBERS, Sigma_n=None, valid_sources=None, scale_factor=1.0):
    """
    GPU-accelerated sliding window processing with CPU-host memory spooling.
    Strict Validity: Window is only processed if ALL pixels are valid.
    """
    if num_endmembers > tile_size * tile_size:
        print(f"Warning: num_endmembers ({num_endmembers}) > pixels in a tile ({tile_size*tile_size}). Clamping to {tile_size*tile_size}.")
        num_endmembers = tile_size * tile_size

    COMPUTE_DTYPE = torch.float32
    BYTES_PER_ELEMENT = 4  # float32
    stride = 1

    bands, height, width = frame_data.shape
    device = 'cuda'
    
    # 1. Load data onto CPU only — system RAM absorbs the full unfold allocation
    # Do not cast to COMPUTE_DTYPE yet to save massive RAM (int16 array is 13 GiB instead of 26 GiB)
    tensor_data = torch.from_numpy(frame_data)
    
    # 2. Calculate dynamic batch size and chunk size to maximize GPU utilization 
    N_pixels = tile_size * tile_size
    out_h = (height - tile_size) // stride + 1
    out_w = (width - tile_size) // stride + 1
    L = out_h * out_w

    total_vram = torch.cuda.get_device_properties(device).total_memory
    target_vram = total_vram * 0.50
    overhead_multiplier = 5
    bytes_per_window_compute = overhead_multiplier * bands * N_pixels * BYTES_PER_ELEMENT
    batch_size = max(1, int(target_vram // max(1, bytes_per_window_compute)))
    batch_size = min(batch_size, 20000)
    
    # 3. Dynamic Chunking: We want to unfold on the GPU to leverage massive GPU memory bandwidth.
    # We restrict chunk size to ~30% of VRAM so unfold output comfortably fits.
    unfold_vram_budget = total_vram * 0.30
    bytes_per_window_unfold = bands * N_pixels * BYTES_PER_ELEMENT
    windows_per_chunk = max(1, int(unfold_vram_budget // max(1, bytes_per_window_unfold)))
    
    if windows_per_chunk >= out_w:
        chunk_cols = out_w
        chunk_rows = max(1, windows_per_chunk // out_w)
        chunk_rows = min(chunk_rows, 50) # Cap rows to keep progress bar moving smoothly
    else:
        chunk_rows = 1
        chunk_cols = windows_per_chunk
    
    # 4. Accumulators live on CPU to prevent VRAM growth across iterations. 
    # Shape as 2D directly to avoid 1D indexing bugs across chunks.
    vol_vals = torch.zeros((out_h, out_w), dtype=COMPUTE_DTYPE, device='cpu')
    heights_vals = torch.zeros((out_h, out_w, num_endmembers), dtype=COMPUTE_DTYPE, device='cpu')
    window_valid_mask = torch.zeros((out_h, out_w), dtype=torch.bool, device='cpu')
    if Sigma_n is not None and valid_sources is not None:
        noise_heights_vals = torch.zeros((len(valid_sources), out_h, out_w, num_endmembers), dtype=COMPUTE_DTYPE, device='cpu')
    
    # Pre-create Sigma_n_tensor ONCE on GPU to avoid repeated numpy->GPU copies
    # in every batch call to maximumDistance_volumes_torch (~5000x fewer allocations)
    if Sigma_n is not None:
        Sigma_n_tensor_gpu = torch.tensor(Sigma_n, dtype=torch.float64, device=device)
    else:
        Sigma_n_tensor_gpu = None
    
    if valid_mask is not None:
        valid_mask_tensor = torch.from_numpy(valid_mask).to('cpu')
    else:
        valid_mask_tensor = None

    with torch.no_grad():
        for y_start in tqdm(range(0, out_h, chunk_rows), desc=f"Sliding chunk {tile_size}x{tile_size}", leave=False):
            y_end = min(y_start + chunk_rows, out_h)
            chunk_h = y_end - y_start
            
            row_start = y_start * stride
            row_end = (y_end - 1) * stride + tile_size
            
            for x_start in range(0, out_w, chunk_cols):
                x_end_out = min(x_start + chunk_cols, out_w)
                chunk_w = x_end_out - x_start
                
                col_start = x_start * stride
                col_end = (x_end_out - 1) * stride + tile_size
            
                # Push small, raw chunk to GPU FIRST, avoiding CPU memory bandwidth bottleneck
                chunk_data = tensor_data[:, row_start:row_end, col_start:col_end].to(dtype=COMPUTE_DTYPE, device=device).unsqueeze(0)
                if scale_factor != 1.0:
                    chunk_data.div_(scale_factor)
                
                # Unfold directly on GPU VRAM
                chunk_unfold = torch.nn.functional.unfold(chunk_data, kernel_size=tile_size, stride=stride)
                chunk_windows = chunk_unfold.view(bands, N_pixels, -1).permute(2, 0, 1) # (L_chunk, C, N_pixels)
                L_chunk = chunk_windows.shape[0]
                
                # Fix #4: Unfold valid mask on CPU as int8, avoiding float promotion and GPU transfer
                if valid_mask_tensor is not None:
                    chunk_valid_cpu = valid_mask_tensor[row_start:row_end, col_start:col_end].to(torch.int8).unsqueeze(0).unsqueeze(0)
                    chunk_valid_unfold = torch.nn.functional.unfold(chunk_valid_cpu.float(), kernel_size=tile_size, stride=stride)
                    chunk_valid_windows = chunk_valid_unfold.view(1, N_pixels, -1).permute(2, 0, 1) # (L_chunk, 1, N_pixels)
                    chunk_is_valid = (chunk_valid_windows.sum(dim=2).squeeze(1) == N_pixels)  # bool on CPU
                else:
                    chunk_is_valid = torch.ones(L_chunk, dtype=torch.bool)
                
                # Fix #3: Accumulate results on GPU, copy once per spatial chunk
                chunk_vol_vals = torch.full((L_chunk,), -float('inf'), dtype=COMPUTE_DTYPE, device=device)
                chunk_heights_vals = torch.zeros((L_chunk, num_endmembers), dtype=COMPUTE_DTYPE, device=device)
                chunk_window_valid_mask = chunk_is_valid.clone()  # already computed from valid_mask
                if Sigma_n is not None and valid_sources is not None:
                    chunk_noise_heights_vals = torch.zeros((len(valid_sources), L_chunk, num_endmembers), dtype=COMPUTE_DTYPE, device=device)
                
                for i in range(0, L_chunk, batch_size):
                    batch_valid = chunk_is_valid[i:i+batch_size].to(device)
                    batch_len = batch_valid.shape[0]
                    
                    if not batch_valid.any():
                        continue
                    
                    # Fix #1: Skip redundant NaN scan — valid_mask from process_h5_file already
                    # guarantees all pixels in valid windows contain real data.
                    valid_data = chunk_windows[i:i+batch_size][batch_valid]
                
                    endmembers, volume, heights, noise_heights = maximumDistance_volumes_torch(
                        valid_data, num_endmembers, Sigma_n, Sigma_n_tensor=Sigma_n_tensor_gpu)
                    
                    if volume.shape[1] > 2:
                        vol_val = torch.max(volume[:, 2:], dim=1)[0]
                    else:
                        vol_val = torch.full((volume.shape[0],), -float('inf'), dtype=COMPUTE_DTYPE, device=device)
                    
                    # Accumulate directly on GPU — no per-batch .cpu() transfers
                    chunk_vol_vals[i:i+batch_len][batch_valid] = vol_val
                    chunk_heights_vals[i:i+batch_len][batch_valid] = heights
                    if Sigma_n is not None and valid_sources is not None:
                        chunk_noise_heights_vals[:, i:i+batch_len, :][:, batch_valid, :] = noise_heights
                    
                    # Explicitly free per-batch GPU tensors to reduce CUDA allocator fragmentation
                    del endmembers, volume, heights, vol_val, valid_data, batch_valid
                    if noise_heights is not None:
                        del noise_heights
                
                # Single bulk D2H copy per spatial chunk (replaces 3-5 tiny PCIe transfers per batch)
                chunk_vol_2d = chunk_vol_vals.view(chunk_h, chunk_w).cpu()
                chunk_heights_2d = chunk_heights_vals.view(chunk_h, chunk_w, num_endmembers).cpu()
                chunk_valid_2d = chunk_window_valid_mask.view(chunk_h, chunk_w)
                
                vol_vals[y_start:y_end, x_start:x_end_out] = chunk_vol_2d
                heights_vals[y_start:y_end, x_start:x_end_out] = chunk_heights_2d
                window_valid_mask[y_start:y_end, x_start:x_end_out] = chunk_valid_2d
                if Sigma_n is not None and valid_sources is not None:
                    chunk_noise_2d = chunk_noise_heights_vals.view(len(valid_sources), chunk_h, chunk_w, num_endmembers).cpu()
                    noise_heights_vals[:, y_start:y_end, x_start:x_end_out, :] = chunk_noise_2d
                
                # Explicitly delete all GPU tensors to reduce CUDA allocator fragmentation
                del chunk_data, chunk_unfold, chunk_windows
                del chunk_vol_vals, chunk_heights_vals, chunk_window_valid_mask
                del chunk_vol_2d, chunk_heights_2d, chunk_valid_2d
                if valid_mask_tensor is not None:
                    del chunk_valid_cpu, chunk_valid_unfold, chunk_valid_windows, chunk_is_valid
                else:
                    del chunk_is_valid
                if Sigma_n is not None and valid_sources is not None:
                    del chunk_noise_heights_vals, chunk_noise_2d
        
    # Free the pre-created Sigma_n GPU tensor now that processing is done
    if Sigma_n_tensor_gpu is not None:
        del Sigma_n_tensor_gpu
    
    center_offset = tile_size // 2
    
    neighborhood_map = np.full((height, width), np.nan, dtype=np.float32)
    heights_map = np.full((height, width, num_endmembers), np.nan, dtype=np.float32)
    if Sigma_n is not None and valid_sources is not None:
        noise_heights_map = np.full((len(valid_sources), height, width, num_endmembers), np.nan, dtype=np.float32)

    
    vol_grid = vol_vals.numpy()
    heights_grid = heights_vals.numpy()
    valid_grid = window_valid_mask.numpy()
    
    neighborhood_map[center_offset:center_offset + out_h,
                     center_offset:center_offset + out_w] = vol_grid
    neighborhood_map[center_offset:center_offset + out_h,
                     center_offset:center_offset + out_w][~valid_grid] = np.nan
                     
    heights_map[center_offset:center_offset + out_h,
                center_offset:center_offset + out_w, :] = heights_grid
    heights_map[center_offset:center_offset + out_h,
                center_offset:center_offset + out_w, :][~valid_grid] = np.nan

    if Sigma_n is not None and valid_sources is not None:
        noise_heights_grid = noise_heights_vals.numpy()
        noise_heights_map = np.full((len(valid_sources), height, width, num_endmembers), np.nan, dtype=np.float32)
        noise_heights_map[:, center_offset:center_offset + out_h, center_offset:center_offset + out_w, :] = noise_heights_grid
        noise_heights_map[:, center_offset:center_offset + out_h, center_offset:center_offset + out_w, :][:, ~valid_grid] = np.nan


    datasets = {
        f'neighborhood_map_{tile_size}x{tile_size}': neighborhood_map,
        f'heights_map_{tile_size}x{tile_size}': heights_map
    }
    if Sigma_n is not None and valid_sources is not None:
        for i, src in enumerate(valid_sources):
            datasets[f'noise_heights_map_{src}_{tile_size}x{tile_size}'] = noise_heights_map[i]

        
    return datasets


def calculate_noise_threshold_from_traces_torch(trace_1, trace_2, num_pixels, alpha=0.01):
    trace_1_safe = torch.clamp(trace_1, min=1e-12)
    trace_2_safe = torch.clamp(trace_2, min=1e-12)
    v = (trace_1_safe ** 2) / trace_2_safe
    c = trace_2_safe / trace_1_safe
    N_candidates = num_pixels - 1
    p_adj = (1.0 - alpha) ** (1.0 / max(N_candidates, 1))
    
    try:
        Z_p = torch.special.ndtri(torch.tensor(p_adj, dtype=trace_1.dtype, device=trace_1.device))
    except (ImportError, AttributeError):
        from torch.distributions.normal import Normal
        Z_p = Normal(0, 1).icdf(torch.tensor(p_adj, dtype=trace_1.dtype, device=trace_1.device))
        
    chi2_approx = v * (1.0 - (2.0 / (9.0 * v)) + Z_p * torch.sqrt(2.0 / (9.0 * v))) ** 3
    threshold_squared = c * chi2_approx
    h_noise_threshold = torch.sqrt(torch.clamp(threshold_squared, min=1e-12))
    return h_noise_threshold


def calculate_noise_threshold_closed_form(Sigma_n, P_perp, num_pixels, alpha=0.01):
    from scipy.stats import chi2
    PS = np.matmul(P_perp, Sigma_n)
    trace_1 = np.trace(PS, axis1=1, axis2=2)
    PS_sq = np.matmul(PS, PS)
    trace_2 = np.trace(PS_sq, axis1=1, axis2=2)
    
    trace_1_safe = np.maximum(trace_1, 1e-12)
    trace_2_safe = np.maximum(trace_2, 1e-12)
    
    v = (trace_1_safe ** 2) / trace_2_safe
    c = trace_2_safe / trace_1_safe
    
    N_candidates = num_pixels - 1
    p_adj = (1.0 - alpha) ** (1.0 / max(N_candidates, 1))
    
    threshold_squared = c * chi2.ppf(p_adj, v)
    h_noise_threshold = np.sqrt(np.maximum(threshold_squared, 1e-12))
    return h_noise_threshold

def process_h5_file(file_path, tile_size=TILE_SIZE, num_endmembers=NUM_ENDMEMBERS):
    import time
    start_time = time.time()
    print(f"Opening {file_path}...")
    try:
        with h5py.File(file_path, 'r+') as f:
            grp_path = "HDFEOS/GRIDS/HYSPEX/Data Fields"
            if grp_path not in f:
                print(f"Error: Could not find {grp_path} in the HDF5 file.")
                return
                
            grp = f[grp_path]
            print("Reading surface_reflectance...")
            # Avoid loading all frames into RAM at once; use dataset reference
            sr_data = grp['surface_reflectance']
            num_frames = sr_data.shape[0]
            
            fill_val = sr_data.fillvalue
        
            # Dynamically read Scale_Factor and safely extract scalar
            scale_factor = sr_data.attrs.get('Scale_Factor')
            if scale_factor is not None:
                scale_factor = float(scale_factor[0])
            else:
                scale_factor = 1.0
            
            covariances = []
            valid_sources = []
            if CALCULATE_NOISE_STATS:
                cov_sources = ['ssd', 'mlr', 'dct', 'mnf','nfd']
                for src in cov_sources:
                    cov_name = f"{src}_noise_cov"
                    if cov_name in grp:
                        covariances.append(grp[cov_name][:])
                        valid_sources.append(src)
                    else:
                        print(f"Warning: {cov_name} not found.")
            
            if len(covariances) > 0:
                Sigma_n = np.stack(covariances, axis=0) # shape (n, B, B)
            else:
                Sigma_n = None

            # Extract spatial reference directly from surface_reflectance
            spatial_ref = None
            geo_transform = None
            if 'spatial_ref' in sr_data.attrs:
                spatial_ref = sr_data.attrs['spatial_ref']
            if 'GeoTransform' in sr_data.attrs:
                geo_transform = sr_data.attrs['GeoTransform']

            for i in range(num_frames):
                print(f"Processing frame {i+1}/{num_frames} with tile_size={tile_size} and num_endmembers={num_endmembers}...")
                
                # Load frame i (shape: bands, lines, samples)
                raw_frame = sr_data[i]
                
                # Valid mask based on dataset fill_value 
                # A pixel is valid if it has data in at least some bands (ignoring globally masked water bands)
                v_mask = np.any(raw_frame != fill_val, axis=0)
                
                # Zero out the fill values so they don't artificially skew the Euclidean distance in the complexity metric
                raw_frame[raw_frame == fill_val] = 0.0
                    
                import gc
                gc.collect()
                
                if torch.cuda.is_available():
                    results = process_volume_sliding_tile_torch(raw_frame, valid_mask=v_mask, tile_size=tile_size, num_endmembers=num_endmembers, Sigma_n=Sigma_n, valid_sources=valid_sources, scale_factor=scale_factor)
                    torch.cuda.empty_cache()
                else:
                    print("CUDA NOT AVAILABLE, CPU PROCESSING ONLY...")
                    frame_data = raw_frame.astype(np.float32) / scale_factor
                    results = process_volume_sliding_tile(frame_data, valid_mask=v_mask, tile_size=tile_size, num_endmembers=num_endmembers, Sigma_n=Sigma_n, valid_sources=valid_sources)
                    
                for ds_name, ds_data in results.items():
                    if ds_name in grp and i == 0:
                        print(f"Dataset {ds_name} already exists in {grp_path}. Overwriting...")
                        del grp[ds_name]
                        
                    if ds_name not in grp:
                        shape = (num_frames,) + ds_data.shape
                        print(f"Saving dataset: {ds_name} to {grp_path} with shape {shape}")
                        ds = grp.create_dataset(ds_name, shape=shape, dtype=ds_data.dtype, compression="gzip", compression_opts=5)
                        if spatial_ref is not None:
                            ds.attrs['spatial_ref'] = spatial_ref
                        if geo_transform is not None:
                            ds.attrs['GeoTransform'] = geo_transform
                            
                    grp[ds_name][i] = ds_data
                
                # Free memory heavily to prevent out of memory issues for the next tile iteration
                del raw_frame
                del v_mask
                del results
                gc.collect()
                        
            end_time = time.time()
            duration = end_time - start_time
            print(f"Processing complete for TILE_SIZE={tile_size} in {duration:.2f} seconds.")
            
            # Save the calculation duration as an attribute
            nm_name = f'neighborhood_map_{tile_size}x{tile_size}'
            if nm_name in grp:
                grp[nm_name].attrs['processing_duration_seconds'] = duration
                
            print("Processing complete and saved to file.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    import os
    import sys
    
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    
    h5_file = filedialog.askopenfilename(
        title="Select HySpex Ortho Stack (.h5)",
        filetypes=[("HDF5 files", "*.h5"), ("All files", "*.*")],
        initialdir=r"C:\satelliteImagery\hyspex\20250918_GE18\dual"
    )
    
    if not h5_file:
        print("No .h5 file selected.")
        sys.exit(1)
        
    h5_files = [h5_file]
    
    for h5_file in h5_files:
        print(f"\n{'='*50}\nProcessing File: {h5_file}\n{'='*50}")
        for i, TILE_SIZE in enumerate(TILE_SIZES):
            print(f"Processing for TILE_SIZE={TILE_SIZE}...")
            process_h5_file(h5_file, TILE_SIZE, NUM_ENDMEMBERS)
            
            # Full memory cleanup between tile sizes to prevent CUDA allocator
            # fragmentation and CPU memory accumulation that causes progressive slowdown
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
