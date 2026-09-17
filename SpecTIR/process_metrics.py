import numpy as np
import h5py
import tkinter as tk
from tqdm import tqdm
from tkinter import filedialog
import torch

TILE_SIZE = 3
NUM_ENDMEMBERS = 40

def maximumDistance_volumes_torch(img_cube, num_endmembers, Sigma_n=None):
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
    Returns:
        endmembers: Tensor of shape (B, C, num_endmembers) — raw (non-localized)
                    endmember spectra, ordered identically to maximumDistance_torch.
                    Index 0 is max-norm, index 1 is min-norm (the local origin).
        volumes: Tensor of shape (B, num_endmembers) — the localized Gram volume
                 curve. volumes[:, 0] = 0 (the origin has zero volume),
                 volumes[:, 1] = h_1, volumes[:, 2] = h_1*h_2, etc.
        heights: Tensor of shape (B, num_endmembers) —
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
    # volumes[:, 0] = -inf (origin contributes no volume)
    # volumes[:, j] for j >= 1 is the cumulative sum of log heights log(h_1)..-log(h_{j})
    volumes = torch.full((B, num_endmembers), -float('inf'), dtype=dtype, device=device)
    heights = torch.zeros(B, num_endmembers, dtype=dtype, device=device)

    if Sigma_n is not None:
        N_cov = Sigma_n.shape[0]
        noise_heights = torch.zeros((N_cov, B, num_endmembers), dtype=dtype, device=device)
        U_basis = torch.zeros((B, C, num_endmembers), dtype=dtype, device=device)
        Sigma_n_tensor = torch.tensor(Sigma_n, dtype=dtype, device=device)
        
        # Precompute initial traces for P_perp = I
        trace_1 = torch.diagonal(Sigma_n_tensor, dim1=-2, dim2=-1).sum(-1).unsqueeze(1).repeat(1, B) # (N_cov, B)
        Sigma_sq = torch.matmul(Sigma_n_tensor, Sigma_n_tensor)
        trace_2 = torch.diagonal(Sigma_sq, dim1=-2, dim2=-1).sum(-1).unsqueeze(1).repeat(1, B) # (N_cov, B)
    else:
        noise_heights = None

    # --- First endmember (max-norm pixel, already identified) ---
    v_first = local_img_cube[b_idx, :, idx_maxnorm]  # (B, C)
    h1 = torch.norm(v_first, dim=1)  # (B,)
    volumes[:, 1] = torch.log(torch.clamp(h1, min=1e-12))
    heights[:, 1] = h1

    if Sigma_n is not None:
        noise_heights[:, :, 1] = calculate_noise_threshold_from_traces_torch(trace_1, trace_2, N)
        u1 = v_first / torch.clamp(h1.unsqueeze(1), min=1e-12)
        
        for k in range(N_cov):
            w = torch.matmul(u1, Sigma_n_tensor[k])
            u_Sigma_u = (u1 * w).sum(dim=1)
            
            # P_perp is I at step 0, so P_w = w
            P_w = w
            w_P_w = (w * P_w).sum(dim=1)
            
            trace_1[k] = trace_1[k] - u_Sigma_u
            trace_2[k] = trace_2[k] - 2.0 * w_P_w + u_Sigma_u ** 2
            
        U_basis[:, :, 0] = u1


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

        # Cumulative addition in log space: log_volume at step i = log_volume at step (i-1) + log(h_j)
        volumes[:, i] = volumes[:, i - 1] + torch.log(torch.clamp(h_j, min=1e-12))
        heights[:, i] = h_j

        if Sigma_n is not None:
            noise_heights[:, :, i] = calculate_noise_threshold_from_traces_torch(trace_1, trace_2, N)
            uj = v_new / torch.clamp(h_j.unsqueeze(1), min=1e-12)
            
            k_idx = i - 1
            U_k = U_basis[:, :, :k_idx]
            
            for k in range(N_cov):
                w = torch.matmul(uj, Sigma_n_tensor[k])
                u_Sigma_u = (uj * w).sum(dim=1)
                
                # P_w = w - U_k @ (U_k^T @ w)
                w_unsq = w.unsqueeze(2)
                U_T_w = torch.bmm(U_k.transpose(1, 2), w_unsq)
                P_w = w_unsq - torch.bmm(U_k, U_T_w)
                P_w = P_w.squeeze(2)
                
                w_P_w = (w * P_w).sum(dim=1)
                
                trace_1[k] = trace_1[k] - u_Sigma_u
                trace_2[k] = trace_2[k] - 2.0 * w_P_w + u_Sigma_u ** 2
                
            U_basis[:, :, k_idx] = uj


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

    return endmembers, volumes, heights, noise_heights


def maximumDistance_volumes(pixels, num_endmembers, Sigma_n=None):
    """
    Combined endmember extraction and Gram volume estimation in a single
    iterative Orthogonal Subspace Projection (OSP) pass.

    Localizes the pixel neighborhood by subtracting the minimum-norm pixel
    (the local origin), then iteratively selects endmembers by largest
    orthogonal projection distance while recording the heights needed for 
    the volume curve.

    Rejects inputs containing any NaN values. Always returns heights.
    """
    nan_return = (
        np.full((pixels.shape[-1] if pixels.ndim == 3 else pixels.shape[0], num_endmembers), np.nan),
        np.full(num_endmembers, np.nan),
        np.full(num_endmembers, np.nan),
        None
    )

    if pixels.ndim == 3:
        h, w, num_bands = pixels.shape
        if np.isnan(pixels).any():
            return nan_return
        pixels = np.reshape(pixels, (h * w, num_bands), order="F").astype(np.float32).T
    elif pixels.ndim == 2:
        pixels = pixels.astype(np.float32)
        if np.isnan(pixels).any():
            return nan_return

    num_bands, num_pixels = pixels.shape

    if num_pixels < num_endmembers:
        return nan_return

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

    selected = np.zeros(num_pixels, dtype=bool)
    selected[idx_minnorm] = True
    selected[idx_maxnorm] = True

    # --- First endmember: the localized max-norm pixel ---
    v = residuals[:, idx_maxnorm].copy()
    h_dist = np.linalg.norm(v)
    volumes[1] = np.log(max(h_dist, 1e-12))
    heights[1] = h_dist

    if Sigma_n is not None:
        noise_heights[:, 1] = calculate_noise_threshold_closed_form(Sigma_n, P_perp, num_pixels)
        u1 = v / max(h_dist, 1e-12)
        P_perp -= np.outer(u1, u1)


    # Project all pixels into the subspace orthogonal to v
    if h_dist > 1e-12:
        # OSP projection component: d(d^T d)^-1 d^T X
        proj_component = np.outer(v, np.dot(v, residuals) / (h_dist ** 2))
        residuals -= proj_component

    # --- Iteratively extract remaining endmembers ---
    for j in range(2, num_endmembers):
        # Magnitudes of unselected pixels in the current orthogonal subspace
        residual_mag = np.sum(residuals ** 2, axis=0)  # (num_pixels,)
        residual_mag[selected] = -np.inf

        idx_new = np.argmax(residual_mag)
        endmembers[:, j] = pixels[:, idx_new]
        selected[idx_new] = True

        # The orthogonal distance is the magnitude of the projected vector
        v = residuals[:, idx_new].copy()
        h_dist = np.linalg.norm(v)

        # Log volume is the cumulative sum of log orthogonal heights
        volumes[j] = volumes[j - 1] + np.log(max(h_dist, 1e-12))
        heights[j] = h_dist

        if Sigma_n is not None:
            noise_heights[:, j] = calculate_noise_threshold_closed_form(Sigma_n, P_perp, num_pixels)
            uj = v / max(h_dist, 1e-12)
            P_perp -= np.outer(uj, uj)


        # Project all pixels into the subspace orthogonal to the new endmember
        if h_dist > 1e-12:
            proj_component = np.outer(v, np.dot(v, residuals) / (h_dist ** 2))
            residuals -= proj_component

    return endmembers, volumes, heights, noise_heights


def process_volume_sliding_tile(frame_data, valid_mask=None, tile_size=TILE_SIZE, num_endmembers=NUM_ENDMEMBERS, Sigma_n=None, valid_sources=None):
    """
    Sliding window processing.
    Strict Validity: Window is only processed if ALL pixels are valid.
    Output is masked with NaN for any pixel identified as invalid.
    """
    if num_endmembers > tile_size * tile_size:
        print(f"Warning: num_endmembers ({num_endmembers}) > pixels in a tile ({tile_size*tile_size}). Clamping to {tile_size*tile_size}.")
        num_endmembers = tile_size * tile_size

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
            
            tile_flat_torch = torch.from_numpy(tile_flat).unsqueeze(0)  # Add batch dimension: (1, C, N)
            if torch.cuda.is_available():
                tile_flat_torch = tile_flat_torch.cuda()
                
            endmembers, volumes, heights, noise_heights = maximumDistance_volumes_torch(tile_flat_torch, num_endmembers, Sigma_n)
            
            volumes = volumes[0].cpu().numpy()
            heights = heights[0].cpu().numpy()
            if Sigma_n is not None:
                noise_heights = noise_heights[:, 0, :].cpu().numpy()
            
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

    if Sigma_n is not None and valid_sources is not None:
        for i, src in enumerate(valid_sources):
            datasets[f'noise_heights_map_{src}_{tile_size}x{tile_size}'] = noise_heights_map[i]

        
    return datasets

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
    bytes_per_window = overhead_multiplier * bands * N_pixels * BYTES_PER_ELEMENT
    batch_size = max(1, int(target_vram // max(1, bytes_per_window)))
    batch_size = min(batch_size, 20000) # Increased back to 20000 since P_perp is removed
    
    bytes_per_row = out_w * bytes_per_window
    chunk_rows = max(1, min(out_h, int(target_vram // max(1, bytes_per_row))))
    
    # 3. Accumulators live on CPU to prevent VRAM growth across iterations
    vol_vals = torch.zeros(L, dtype=COMPUTE_DTYPE, device='cpu')
    heights_vals = torch.zeros((L, num_endmembers), dtype=COMPUTE_DTYPE, device='cpu')
    window_valid_mask = torch.zeros(L, dtype=torch.bool, device='cpu')
    if Sigma_n is not None and valid_sources is not None:
        noise_heights_vals = torch.zeros((len(valid_sources), L, num_endmembers), dtype=COMPUTE_DTYPE, device='cpu')
    
    if valid_mask is not None:
        valid_mask_tensor = torch.from_numpy(valid_mask).to('cpu')
    else:
        valid_mask_tensor = None

    with torch.no_grad():
        for y_start in tqdm(range(0, out_h, chunk_rows), desc=f"Sliding chunk {tile_size}x{tile_size}", leave=False):
            y_end = min(y_start + chunk_rows, out_h)
            
            row_start = y_start * stride
            row_end = (y_end - 1) * stride + tile_size
            
            # Cast chunk to float32 on the CPU just before unfolding to avoid entire array allocation
            chunk_data = tensor_data[:, row_start:row_end, :].to(dtype=COMPUTE_DTYPE).unsqueeze(0)
            if scale_factor != 1.0:
                chunk_data.div_(scale_factor)
            
            chunk_unfold = torch.nn.functional.unfold(chunk_data, kernel_size=tile_size, stride=stride)
            chunk_windows = chunk_unfold.view(bands, N_pixels, -1).permute(2, 0, 1) # (L_chunk, C, N_pixels)
            L_chunk = chunk_windows.shape[0]
            
            if valid_mask_tensor is not None:
                chunk_valid = valid_mask_tensor[row_start:row_end, :].unsqueeze(0).unsqueeze(0).float()
                chunk_valid_unfold = torch.nn.functional.unfold(chunk_valid, kernel_size=tile_size, stride=stride)
                chunk_valid_windows = chunk_valid_unfold.view(1, N_pixels, -1).permute(2, 0, 1) # (L_chunk, 1, N_pixels)
                chunk_is_valid = (chunk_valid_windows.sum(dim=2).squeeze(1) == N_pixels)
            else:
                chunk_is_valid = torch.ones(L_chunk, dtype=torch.bool)
            
            global_start_idx = y_start * out_w
            
            for i in range(0, L_chunk, batch_size):
                batch_valid_mask = chunk_is_valid[i:i+batch_size]
                batch_windows = chunk_windows[i:i+batch_size]
                
                pixel_validity = ~torch.isnan(batch_windows).any(dim=1)  # (B, N)
                valid_pixels_per_window = pixel_validity.sum(dim=1)  # (B)
                batch_valid = batch_valid_mask & (valid_pixels_per_window == N_pixels)
                
                window_valid_mask[global_start_idx + i : global_start_idx + i + batch_windows.shape[0]] = batch_valid
                
                if not batch_valid.any():
                    continue
                
                valid_data = batch_windows[batch_valid].to(device)
                
                valid_data[torch.isnan(valid_data)] = 0.0
            
                endmembers, volume, heights, noise_heights = maximumDistance_volumes_torch(valid_data, num_endmembers, Sigma_n)
                
                if volume.shape[1] > 2:
                    vol_val = torch.max(volume[:, 2:], dim=1)[0]
                else:
                    vol_val = torch.full((volume.shape[0],), -float('inf'), dtype=COMPUTE_DTYPE, device=device)
                
                vol_vals[global_start_idx + i : global_start_idx + i + batch_windows.shape[0]][batch_valid] = vol_val.cpu()
                heights_vals[global_start_idx + i : global_start_idx + i + batch_windows.shape[0]][batch_valid] = heights.cpu()
                if Sigma_n is not None and valid_sources is not None:
                    noise_heights_vals[:, global_start_idx + i : global_start_idx + i + batch_windows.shape[0], :][:, batch_valid, :] = noise_heights.cpu()
                
                torch.cuda.synchronize()
        
    center_offset = tile_size // 2
    
    neighborhood_map = np.full((height, width), np.nan, dtype=np.float32)
    heights_map = np.full((height, width, num_endmembers), np.nan, dtype=np.float32)
    if Sigma_n is not None and valid_sources is not None:
        noise_heights_map = np.full((len(valid_sources), height, width, num_endmembers), np.nan, dtype=np.float32)

    
    vol_grid = vol_vals.numpy().reshape(out_h, out_w)
    heights_grid = heights_vals.numpy().reshape(out_h, out_w, num_endmembers)
    valid_grid = window_valid_mask.numpy().reshape(out_h, out_w)
    
    neighborhood_map[center_offset:center_offset + out_h,
                     center_offset:center_offset + out_w] = vol_grid
    neighborhood_map[center_offset:center_offset + out_h,
                     center_offset:center_offset + out_w][~valid_grid] = np.nan
                     
    heights_map[center_offset:center_offset + out_h,
                center_offset:center_offset + out_w, :] = heights_grid
    heights_map[center_offset:center_offset + out_h,
                center_offset:center_offset + out_w, :][~valid_grid] = np.nan

    if Sigma_n is not None and valid_sources is not None:
        noise_heights_grid = noise_heights_vals.numpy().reshape(len(valid_sources), out_h, out_w, num_endmembers)
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

    if Sigma_n is not None and valid_sources is not None:
        for i, src in enumerate(valid_sources):
            datasets[f'noise_heights_map_{src}_{tile_size}x{tile_size}'] = noise_heights_map[i]

        
    return datasets

def calculate_noise_threshold_mc(Sigma_n, P_perp, num_pixels, M=5000, alpha=0.01):
    """
    Calculates the noise-equivalent orthogonal height threshold using a 
    Monte Carlo simulation (as described in maxD-Volumes slides 15-18).
    
    Args:
        Sigma_n (np.ndarray): Full B x B noise covariance matrix.
        P_perp (np.ndarray): B x B orthogonal projection matrix from current MaxD iteration.
        num_pixels (int): Total number of pixels in the search window (N).
        M (int): Number of Monte Carlo realizations.
        alpha (float): False alarm probability (upper-tail quantile).
        
    Returns:
        float: The noise-equivalent height threshold h_noise(alpha).
    """
    bands = Sigma_n.shape[0]
    
    # 1. Generate M realizations of N noise pixels
    # Using SVD factorization is mathematically robust even if Sigma_n is numerically singular
    U, S, _ = np.linalg.svd(Sigma_n)
    S_sqrt = np.sqrt(np.maximum(S, 0))
    
    # Z ~ N(0, I)
    Z = np.random.standard_normal((M, num_pixels, bands))
    # Transform to N(0, Sigma_n)
    noise_draws = Z @ (U * S_sqrt).T
    
    # 2. Localize by differencing candidates from the reference noise vector (n_0)
    n_0 = noise_draws[:, 0:1, :]
    n_i = noise_draws[:, 1:, :]
    eta = n_i - n_0  # Shape: (M, num_pixels - 1, bands)
    
    # 3. Project the localized noise through the observed MaxD projector
    # Since P_perp is symmetric, eta @ P_perp mathematically applies the projection
    projected_eta = eta @ P_perp
    
    # Calculate magnitudes of projected noise vectors
    magnitudes = np.linalg.norm(projected_eta, axis=-1)  # Shape: (M, num_pixels - 1)
    
    # 4. Retain the maximum projected-noise magnitude from the simulated population
    H_noise = np.max(magnitudes, axis=1)  # Shape: (M,)
    
    # 5. Define the noise-equivalent height as the selected upper-tail quantile
    h_noise_threshold = np.quantile(H_noise, 1.0 - alpha)
    
    return float(h_noise_threshold)


def calculate_noise_threshold_from_traces_torch(trace_1, trace_2, num_pixels, alpha=0.01):
    trace_1_safe = torch.clamp(trace_1, min=1e-12)
    trace_2_safe = torch.clamp(trace_2, min=1e-12)
    v = (trace_1 ** 2) / trace_2_safe
    c = 2.0 * (trace_2 / trace_1_safe)
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
    
    v = (trace_1 ** 2) / trace_2_safe
    c = 2.0 * (trace_2 / trace_1_safe)
    
    N_candidates = num_pixels - 1
    p_adj = (1.0 - alpha) ** (1.0 / max(N_candidates, 1))
    
    threshold_squared = c * chi2.ppf(p_adj, v)
    h_noise_threshold = np.sqrt(np.maximum(threshold_squared, 1e-12))
    return h_noise_threshold

def process_h5_file(file_path, tile_size=TILE_SIZE, num_endmembers=NUM_ENDMEMBERS):
    print(f"Opening {file_path}...")
    try:
        with h5py.File(file_path, 'r+') as f:
            grp_path = "HDFEOS/GRIDS/SPECTIR/Data Fields"
            if grp_path not in f or 'surface_reflectance' not in f[grp_path]:
                print(f"Error: '{grp_path}/surface_reflectance' dataset not found in the file.")
            else:
                grp = f[grp_path]
                print("Reading surface_reflectance...")
                # Avoid loading all frames into RAM at once; use dataset reference
                sr_data = grp['surface_reflectance']
                num_frames = sr_data.shape[0]
                
                neighborhood_maps = []
                heights_maps = []
                
                nodata_mask = None
                if 'nodata_mask' in grp:
                    nodata_mask = grp['nodata_mask'][:]
                
                fill_val = sr_data.fillvalue if sr_data.fillvalue is not None else 0
                
                # Dynamically read Scale_Factor
                scale_factor = sr_data.attrs.get('Scale_Factor')
                
                
                cov_sources = ['ssd', 'mlr', 'dct', 'mnf']
                covariances = []
                valid_sources = []
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
    
                
                frame_results_list = []
                for i in range(num_frames):
                    print(f"Processing frame {i+1}/{num_frames} with tile_size={tile_size} and num_endmembers={num_endmembers}...")
                    
                    # Load single frame as raw int16 (takes ~13 GiB instead of ~40 GiB for float conversion)
                    raw_frame = sr_data[i]
                    
                    v_mask = None
                    if nodata_mask is not None:
                        v_mask = ~nodata_mask[i, 0]
                    else:
                        # Fallback to checking for fill_value across all bands using raw_frame
                        v_mask = np.all(raw_frame != fill_val, axis=0)
                    
                    # Filter out anomalous data outside surface reflectance range (equivalent to +/- 1.5 in float)
                    invalid_values = (raw_frame < -15000) | (raw_frame > 15000)
                    v_mask = v_mask & ~np.any(invalid_values, axis=0)
                        
                    import gc
                    gc.collect()
                    
                    if torch.cuda.is_available():
                        results = process_volume_sliding_tile_torch(raw_frame, valid_mask=v_mask, tile_size=tile_size, num_endmembers=num_endmembers, Sigma_n=Sigma_n, valid_sources=valid_sources, scale_factor=scale_factor)
                    else:
                        frame_data = raw_frame.astype(np.float32) / scale_factor
                        results = process_volume_sliding_tile(frame_data, valid_mask=v_mask, tile_size=tile_size, num_endmembers=num_endmembers, Sigma_n=Sigma_n, valid_sources=valid_sources)
                        
                    frame_results_list.append(results)
                    
                final_results = {}
                if len(frame_results_list) > 0:
                    for key in frame_results_list[0].keys():
                        final_results[key] = np.stack([res[key] for res in frame_results_list], axis=0)
                
                for ds_name, ds_data in final_results.items():
                    if ds_name in grp:
                        print(f"Dataset {ds_name} already exists in {grp_path}. Overwriting...")
                        del grp[ds_name]
                    print(f"Saving dataset: {ds_name} to {grp_path} with shape {ds_data.shape}")
                    grp.create_dataset(ds_name, data=ds_data)
                print("Processing complete and saved to file.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    import os
    import sys
    target_dir = r"C:\satelliteImagery\share2012\SpecTIR"
    if not os.path.exists(target_dir):
        print(f"Error: Target directory {target_dir} not found.")
        sys.exit(1)
        
    h5_files = [
        os.path.join(target_dir, f) for f in os.listdir(target_dir) 
        if f.startswith("SpecTIR_Ortho_Stack_") and f.endswith(".h5")
    ]
    
    if not h5_files:
        print("No matching .h5 files found.")
        sys.exit(1)
        
    for h5_file in h5_files:
        print(f"\n{'='*50}\nProcessing File: {h5_file}\n{'='*50}")
        for i, TILE_SIZE in enumerate([3, 4, 5, 7, 9, 11, 13, 15, 17, 19, 21]):
            print(f"Processing for TILE_SIZE={TILE_SIZE}...")
            process_h5_file(h5_file, TILE_SIZE, NUM_ENDMEMBERS)
