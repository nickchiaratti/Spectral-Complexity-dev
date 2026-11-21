import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import h5py
import tkinter as tk
from tqdm import tqdm
from tkinter import filedialog
import torch
import spectral

TILE_SIZE=15
TILE_SIZES = [3]#[3,5,7,9,11,15,25,35,45,55,65,101]#3,5,7,9,11,15,19,35,55,65,81,101]#[3,5,7,9,11,15,19,35,55,65,81,101]
NUM_ENDMEMBERS = 20
CALCULATE_NOISE_STATS = False  # Set to False to skip calculating noise equivalent heights


def maximumDistance_volumes_torch(img_cube, num_endmembers, Sigma_n=None, Sigma_n_tensor=None):
    """
    Combined MaxD endmember extraction and localized Gram volume calculation
    in a single iterative Orthogonal Subspace Projection (OSP) pass with GPU batching,
    lazy projection, float64 norm precision, and robust per-tile epsilon thresholding.
    Optimized for high-throughput steady-state processing with zero per-iteration allocations.
    """
    B, C, N = img_cube.shape
    orig_dtype = img_cube.dtype
    img_cube = img_cube.to(torch.float32)
    device = img_cube.device
    dtype = torch.float32

    # Vector norm noise floor for float32
    epsilon = np.sqrt(C) * torch.finfo(dtype).eps

    # --- Identify seed endmembers ---
    magnitude_sq = torch.sum(img_cube.square(), dim=1)  # (B, N)
    idx_maxnorm = torch.argmax(magnitude_sq, dim=1)
    idx_minnorm = torch.argmin(magnitude_sq, dim=1)

    b_idx = torch.arange(B, device=device)

    # Store raw endmembers in extraction order
    endmembers = torch.zeros(B, C, num_endmembers, dtype=dtype, device=device)
    endmembers[:, :, 0] = img_cube[b_idx, :, idx_maxnorm]
    endmembers[:, :, 1] = img_cube[b_idx, :, idx_minnorm]

    # --- Localize to min-norm pixel ---
    origin = img_cube[b_idx, :, idx_minnorm].unsqueeze(2)  # (B, C, 1)
    local_data = img_cube - origin  # (B, C, N)

    volumes = torch.full((B, num_endmembers), -float('inf'), dtype=dtype, device=device)
    heights = torch.full((B, num_endmembers), -float('inf'), dtype=dtype, device=device)
    heights[:, 0] = 0.0  # Origin has zero height

    U_basis = torch.zeros((B, C, num_endmembers - 1), dtype=dtype, device=device)

    residual_sq = torch.sum(local_data.square(), dim=1)  # (B, N)
    residual_sq[b_idx, idx_minnorm] = -float('inf')
    residual_sq[b_idx, idx_maxnorm] = -float('inf')

    if Sigma_n is not None:
        N_cov = Sigma_n.shape[0]
        noise_heights = torch.full((N_cov, B, num_endmembers), -float('inf'), dtype=dtype, device=device)
        noise_heights[:, :, 0] = 0.0
        if Sigma_n_tensor is None:
            Sigma_n_tensor = torch.tensor(Sigma_n, dtype=dtype, device=device)
        else:
            Sigma_n_tensor = Sigma_n_tensor.to(dtype=dtype)
        
        trace_1 = torch.diagonal(Sigma_n_tensor, dim1=-2, dim2=-1).sum(-1).unsqueeze(1).repeat(1, B)
        Sigma_sq = torch.matmul(Sigma_n_tensor, Sigma_n_tensor)
        trace_2 = torch.diagonal(Sigma_sq, dim1=-2, dim2=-1).sum(-1).unsqueeze(1).repeat(1, B)
    else:
        noise_heights = None

    # --- First endmember ---
    v_first = local_data[b_idx, :, idx_maxnorm]  # (B, C)
    h1 = torch.norm(v_first.to(torch.float64), dim=1).to(dtype)  # (B,)

    heights[:, 1] = h1
    is_valid_1 = h1 > epsilon
    all_active = is_valid_1.all().item()

    if all_active:
        volumes[:, 1] = torch.log(h1)
        u1 = v_first / h1.unsqueeze(1)
        U_basis[:, :, 0] = u1
        proj_coeffs = torch.bmm(u1.unsqueeze(1), local_data).squeeze(1)  # (B, N)
        residual_sq.sub_(proj_coeffs.square_())
        active_mask = None
    else:
        active_mask = is_valid_1
        volumes[active_mask, 1] = torch.log(h1[active_mask])
        safe_h1 = torch.where(active_mask, h1, torch.ones_like(h1))
        u1 = torch.where(active_mask.unsqueeze(1), v_first / safe_h1.unsqueeze(1), torch.zeros_like(v_first))
        U_basis[:, :, 0] = u1
        proj_coeffs = torch.bmm(u1.unsqueeze(1), local_data).squeeze(1)
        residual_sq.sub_(proj_coeffs.square_())

    if Sigma_n is not None:
        if active_mask is None or active_mask.any():
            mask_sel = slice(None) if active_mask is None else active_mask
            noise_heights[:, mask_sel, 1] = calculate_noise_threshold_from_traces_torch(
                trace_1[:, mask_sel], trace_2[:, mask_sel], N
            )
        w = torch.einsum('bi,nij->nbj', u1, Sigma_n_tensor)
        u_Sigma_u = (u1.unsqueeze(0) * w).sum(dim=2)
        P_w = w
        w_P_w = (w * P_w).sum(dim=2)
        trace_1 = trace_1 - u_Sigma_u
        trace_2 = trace_2 - 2.0 * w_P_w + u_Sigma_u ** 2

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
                volumes[:, i] = volumes[:, i - 1] + torch.log(h_j)
                uj = v_new / h_j.unsqueeze(1)
                U_basis[:, :, k] = uj
                proj_coeffs = torch.bmm(uj.unsqueeze(1), local_data).squeeze(1)
                residual_sq.sub_(proj_coeffs.square_())
            else:
                all_active = False
                active_mask = is_valid
                heights[active_mask, i] = h_j[active_mask]
                volumes[active_mask, i] = volumes[active_mask, i - 1] + torch.log(h_j[active_mask])
                safe_hj = torch.where(active_mask, h_j, torch.ones_like(h_j))
                uj = torch.where(active_mask.unsqueeze(1), v_new / safe_hj.unsqueeze(1), torch.zeros_like(v_new))
                U_basis[:, :, k] = uj
                proj_coeffs = torch.bmm(uj.unsqueeze(1), local_data).squeeze(1)
                residual_sq.sub_(proj_coeffs.square_())
        else:
            is_valid = (h_j > epsilon) & active_mask
            active_mask = active_mask & is_valid
            heights[active_mask, i] = h_j[active_mask]
            volumes[active_mask, i] = volumes[active_mask, i - 1] + torch.log(h_j[active_mask])
            safe_hj = torch.where(active_mask, h_j, torch.ones_like(h_j))
            uj = torch.where(active_mask.unsqueeze(1), v_new / safe_hj.unsqueeze(1), torch.zeros_like(v_new))
            U_basis[:, :, k] = uj
            proj_coeffs = torch.bmm(uj.unsqueeze(1), local_data).squeeze(1)
            residual_sq.sub_(proj_coeffs.square_())

        if Sigma_n is not None:
            if active_mask is None or active_mask.any():
                mask_sel = slice(None) if active_mask is None else active_mask
                noise_heights[:, mask_sel, i] = calculate_noise_threshold_from_traces_torch(
                    trace_1[:, mask_sel], trace_2[:, mask_sel], N
                )
            w = torch.einsum('bi,nij->nbj', uj, Sigma_n_tensor)
            u_Sigma_u = (uj.unsqueeze(0) * w).sum(dim=2)
            U_T_w = torch.einsum('bik,nbi->nbk', U_k, w)
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
    Combined endmember extraction and Gram volume estimation using the maximum distance (MaxD) method with iterative orthogonal projection.

    Localizes the pixel neighborhood by subtracting the minimum-norm pixel (the local origin), then iteratively selects endmembers by largest
    orthogonal projection distance while recording the heights needed for the volume curve.

    Args:
        pixels (np.ndarray): (bands, pixels) or (height, width, bands) array of x pixel spectra.
        num_endmembers (int): The number of r endmembers to extract.
        Sigma_n (np.ndarray, optional): (N_cov, bands, bands) array of noise covariance matrices. Defaults to None.

    Returns:
        endmembers (np.ndarray): (bands, num_endmembers) array of endmember spectra v_r.
        volumes (np.ndarray): (num_endmembers,) array of ln(Vol_r).
        heights (np.ndarray): (num_endmembers,) array of heights h_r.
        noise_heights (np.ndarray): (N_cov, num_endmembers) array of noise-limited orthogonal distances.
    """
    if pixels.ndim == 3:
        h, w, num_bands = pixels.shape
        pixels = np.reshape(pixels, (h * w, num_bands), order="F").astype(pixels.dtype).T

    num_bands, num_pixels = pixels.shape
    epsilon=np.sqrt(num_bands)*np.finfo(pixels.dtype).eps

    # Initialize Variables
    endmembers = np.zeros((num_bands, num_endmembers), dtype=pixels.dtype)
    volumes = np.full(num_endmembers, -np.inf, dtype=pixels.dtype)
    heights = np.zeros(num_endmembers, dtype=pixels.dtype)

    if Sigma_n is not None:
        N_cov = Sigma_n.shape[0]
        noise_heights = np.zeros((N_cov, num_endmembers), dtype=pixels.dtype)
        P_perp = np.eye(num_bands, dtype=pixels.dtype)
    else:
        noise_heights = None

    # Initialization
    # 𝒗_0=arg⁡𝑚𝑖𝑛‖𝒙‖
    # 𝒗_1=arg⁡𝑚𝑎𝑥‖𝒙‖
    magnitudes = np.linalg.norm(pixels, axis=0)
    idx_maxnorm = np.argmax(magnitudes)
    idx_minnorm = np.argmin(magnitudes)
    endmembers[:, 0] = pixels[:, idx_maxnorm]
    endmembers[:, 1] = pixels[:, idx_minnorm]
    selected_indices = [idx_minnorm, idx_maxnorm]

    # Shift all pixel spectra to a localized origin at 𝒗_0 to form the initial residual vectors 𝒓_1
    # 𝒓_1 (𝒙)=𝒙−𝒗_0
    origin = pixels[:, idx_minnorm].reshape(-1, 1)    # (bands, 1)
    residuals = pixels - origin                        # (bands, num_pixels)

    # Calculate the first direction vector 𝒅_1, height ℎ_1, and orthonormal basis 𝒒_1
    # 𝒅_1=𝒗_0−𝒗_1
    # ℎ_1=‖𝒅_1 ‖
    # 𝒒_1=𝒅_1/ℎ_1  
    d1 = residuals[:, idx_maxnorm].copy().astype(np.float64)              # (bands, )
    h1 = float(np.linalg.norm(d1))
    volumes[1] = np.log(h1) if h1 > epsilon else -np.inf
    heights[1] = h1
    if h1 > epsilon:
        q1 = (d1 / h1).astype(pixels.dtype)  # (bands,1)
        proj_component = np.outer(q1, np.dot(q1, residuals))  # equivalent to q1*q1.T*residuals in vector operations
        residuals -= proj_component
    else:
        # all pixels are the same, set all subsequent endmembers to -inf
        volumes[2:] = -np.inf
        heights[2:] = -np.inf
        return endmembers, volumes, heights, noise_heights

    if Sigma_n is not None:
        noise_heights[:, 1] = calculate_noise_threshold_closed_form(Sigma_n, P_perp, num_pixels)
        P_perp -= np.outer(q1, q1)

    # The residual vectors are deflated across subsequent iterations 𝑗=2,…,𝑟 to select 𝑟 endmembers
    for j in range(2, num_endmembers):
        # Select the next endmember 𝒗_𝑗 as the pixel with maximum magnitude in the current residual subspace 
        # excluding the indices for 𝒗_0,𝒗_1,𝒗_2,…𝒗_(𝑗−1)
        # 𝒗_𝑗=arg𝑚𝑎𝑥‖𝒓_𝑗 (𝒙)‖
        residual_norm = np.linalg.norm(residuals, axis=0)  # (num_pixels,)
        if len(selected_indices) < num_pixels:
            residual_norm[selected_indices] = -np.inf
            idx_new = np.argmax(residual_norm)
            selected_indices.append(idx_new)
        else:
            idx_new = 0

        v_j = residuals[:, idx_new].copy()
        endmembers[:, j] = pixels[:, idx_new]

        # Calculate the subsequent height ℎ_𝑗 and orthonormal basis 𝒒_𝑗
        # ℎ_𝑗=‖𝒓_𝑗 (𝒗_𝑗 )‖
        # 𝒒_𝒋=(𝒓_𝑗 (𝒗_𝑗 ))/ℎ_𝑗 
        v_64 = v_j.astype(np.float64)
        h_j = float(np.linalg.norm(v_64))

        if h_j > epsilon:
            heights[j] = h_j
            q_j = (v_64 / h_j).astype(pixels.dtype)
            volumes[j] = volumes[j - 1] + np.log(h_j)
            proj_component = np.outer(q_j, np.dot(q_j, residuals)) # equivalent to q_j*q_j.T*residuals in vector operations
            residuals -= proj_component
            if Sigma_n is not None:
                noise_heights[:, j] = calculate_noise_threshold_closed_form(Sigma_n, P_perp, num_pixels)
                P_perp -= np.outer(q_j, q_j)
        else:
            heights[j:] = -np.inf
            volumes[j:] = -np.inf
            if Sigma_n is not None:
                noise_heights[:, j:] = -np.inf
            break


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
                
            cy = y_start + center_offset
            cx = x_start + center_offset
            heights_map[cy, cx, :] = heights
            if Sigma_n is not None and valid_sources is not None:
                noise_heights_map[:, cy, cx, :] = noise_heights

    # Add the final datasets using naming conventions
    datasets = {
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
    heights_vals = torch.zeros((out_h, out_w, num_endmembers), dtype=COMPUTE_DTYPE, device='cpu')
    window_valid_mask = torch.zeros((out_h, out_w), dtype=torch.bool, device='cpu')
    if Sigma_n is not None and valid_sources is not None:
        noise_heights_vals = torch.zeros((len(valid_sources), out_h, out_w, num_endmembers), dtype=COMPUTE_DTYPE, device='cpu')
    
    # Pre-create Sigma_n_tensor ONCE on GPU to avoid repeated numpy->GPU copies
    # in every batch call to maximumDistance_volumes_torch (~5000x fewer allocations)
    if Sigma_n is not None:
        Sigma_n_tensor_gpu = torch.tensor(Sigma_n, dtype=torch.float32, device=device)
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
                    
                    # Accumulate directly on GPU — no per-batch .cpu() transfers
                    chunk_heights_vals[i:i+batch_len][batch_valid] = heights
                    if Sigma_n is not None and valid_sources is not None:
                        chunk_noise_heights_vals[:, i:i+batch_len, :][:, batch_valid, :] = noise_heights
                    
                    # Explicitly free per-batch GPU tensors to reduce CUDA allocator fragmentation
                    del endmembers, volume, heights, valid_data, batch_valid
                    if noise_heights is not None:
                        del noise_heights
                
                # Single bulk D2H copy per spatial chunk (replaces 3-5 tiny PCIe transfers per batch)
                chunk_heights_2d = chunk_heights_vals.view(chunk_h, chunk_w, num_endmembers).cpu()
                chunk_valid_2d = chunk_window_valid_mask.view(chunk_h, chunk_w)
                
                heights_vals[y_start:y_end, x_start:x_end_out] = chunk_heights_2d
                window_valid_mask[y_start:y_end, x_start:x_end_out] = chunk_valid_2d
                if Sigma_n is not None and valid_sources is not None:
                    chunk_noise_2d = chunk_noise_heights_vals.view(len(valid_sources), chunk_h, chunk_w, num_endmembers).cpu()
                    noise_heights_vals[:, y_start:y_end, x_start:x_end_out, :] = chunk_noise_2d
                
                # Explicitly delete all GPU tensors to reduce CUDA allocator fragmentation
                del chunk_data, chunk_unfold, chunk_windows
                del chunk_heights_vals, chunk_window_valid_mask
                del chunk_heights_2d, chunk_valid_2d
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
    
    heights_map = np.full((height, width, num_endmembers), np.nan, dtype=np.float32)
    if Sigma_n is not None and valid_sources is not None:
        noise_heights_map = np.full((len(valid_sources), height, width, num_endmembers), np.nan, dtype=np.float32)

    heights_grid = heights_vals.numpy()
    valid_grid = window_valid_mask.numpy()
    
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


def compute_and_save_rx_anomaly(grp, sr_data, spatial_ref=None, geo_transform=None):
    """
    Computes Reed-Xiaoli (RX) anomaly detector score using spectral.rx for each frame
    and stores the results as an additional dataset named 'rxanomaly' in the HDF5 group.
    Only executed if 'rxanomaly' does not already exist in the file.
    """
    import time

    if 'rxanomaly' in grp:
        print("Dataset 'rxanomaly' already exists. Skipping calculation.")
        return

    print("Computing RX anomaly detector scores (spectral.rx)...")
    t0_rx = time.time()

    if sr_data.ndim == 4:
        num_frames, orig_B, orig_H, orig_W = sr_data.shape
    else:
        num_frames = 1
        orig_B, orig_H, orig_W = sr_data.shape

    out_shape = (num_frames, orig_H, orig_W)
    chunks_rx = (1, min(orig_H, 256), min(orig_W, 256))
    print(f"Creating dataset 'rxanomaly' with shape {out_shape} and chunks {chunks_rx}...")
    rx_ds = grp.create_dataset(
        'rxanomaly',
        shape=out_shape,
        dtype=np.float32,
        compression="gzip",
        compression_opts=4,
        chunks=chunks_rx
    )
    if spatial_ref is not None:
        rx_ds.attrs['spatial_ref'] = spatial_ref
    if geo_transform is not None:
        rx_ds.attrs['GeoTransform'] = geo_transform

    for frame_idx in range(num_frames):
        if num_frames > 1:
            print(f"Processing RX anomaly for frame {frame_idx + 1}/{num_frames}...")

        # Sample contiguous lines to estimate clean background covariance
        sample_y = np.linspace(orig_H // 6, orig_H - orig_H // 6, 6, dtype=int)
        if sr_data.ndim == 4:
            train_list = [sr_data[frame_idx, :, y, :] for y in sample_y]
        else:
            train_list = [sr_data[:, y, :] for y in sample_y]
        train_arr = np.concatenate(train_list, axis=1)  # (B, 6 * W)

        valid_bands = np.mean(train_arr > 0.0, axis=1) > 0.1
        if not np.any(valid_bands):
            valid_bands = np.ones(orig_B, dtype=bool)

        train_valid = train_arr[valid_bands].T  # (N, B_valid)
        mask = np.all(train_valid > 0.0, axis=-1)
        bg_pixels = train_valid[mask]

        if len(bg_pixels) > 10:
            mean = np.mean(bg_pixels, axis=0)
            cov = np.cov(bg_pixels, rowvar=False)
            cov_reg = cov + (1e-4 * np.trace(cov) / cov.shape[0]) * np.eye(cov.shape[0])
            bg_stats = spectral.GaussianStats(mean, cov_reg)
            inv_cov = np.linalg.pinv(cov_reg).astype(np.float32)
        else:
            bg_stats = None
            inv_cov = None

        rx_map = np.full((orig_H, orig_W), np.nan, dtype=np.float32)
        chunk_size = 256

        # GPU-accelerated Mahalanobis distance when CUDA is available
        if bg_stats is not None and torch.cuda.is_available():
            device = 'cuda'
            mean_t = torch.tensor(mean, dtype=torch.float32, device=device)
            inv_cov_t = torch.tensor(inv_cov, dtype=torch.float32, device=device)
            for y_start in range(0, orig_H, chunk_size):
                y_end = min(orig_H, y_start + chunk_size)
                if sr_data.ndim == 4:
                    c_raw = sr_data[frame_idx, :, y_start:y_end, :]
                else:
                    c_raw = sr_data[:, y_start:y_end, :]
                c_tensor = torch.from_numpy(c_raw[valid_bands]).permute(1, 2, 0).to(device, dtype=torch.float32)
                valid_t = torch.all(c_tensor > 0.0, dim=-1)
                diff = c_tensor - mean_t
                temp = torch.matmul(diff, inv_cov_t)
                rx_chunk = torch.sum(temp * diff, dim=-1)
                rx_chunk[~valid_t] = float('nan')
                rx_map[y_start:y_end] = rx_chunk.cpu().numpy()
        else:
            for y_start in range(0, orig_H, chunk_size):
                y_end = min(orig_H, y_start + chunk_size)
                if sr_data.ndim == 4:
                    c_raw = sr_data[frame_idx, :, y_start:y_end, :]
                else:
                    c_raw = sr_data[:, y_start:y_end, :]
                c = np.transpose(c_raw[valid_bands], (1, 2, 0))
                valid_mask = np.all(c > 0.0, axis=-1)
                if np.any(valid_mask):
                    rx_res = spectral.rx(c, background=bg_stats)
                    rx_map[y_start:y_end] = np.where(valid_mask, rx_res, np.nan)

        rx_ds[frame_idx] = rx_map

    duration = time.time() - t0_rx
    rx_ds.attrs['processing_duration_seconds'] = duration
    print(f"Dataset 'rxanomaly' created and saved in {duration:.2f} seconds.")


def get_hsi_data_group(f):
    """
    Dynamically finds the HSI data fields group under HDFEOS/GRIDS,
    supporting HYSPEX, HEADWALL, or any other sensor grid.
    """
    if "HDFEOS/GRIDS" in f:
        grids = f["HDFEOS/GRIDS"]
        for sensor in list(grids.keys()):
            path = f"HDFEOS/GRIDS/{sensor}/Data Fields"
            if path in f:
                return f[path], path
    if "HDFEOS/GRIDS/HYSPEX/Data Fields" in f:
        return f["HDFEOS/GRIDS/HYSPEX/Data Fields"], "HDFEOS/GRIDS/HYSPEX/Data Fields"
    return None, None


def process_rx_anomaly_h5_file(file_path):
    """
    Opens file_path, checks if 'rxanomaly' dataset exists, and computes/stores it if not.
    """
    try:
        with h5py.File(file_path, 'r+') as f:
            grp, grp_path = get_hsi_data_group(f)
            if grp is None:
                print(f"Error: Could not find HDFEOS/GRIDS Data Fields in {file_path}")
                return
            if 'rxanomaly' in grp:
                print("Dataset 'rxanomaly' already exists. Skipping.")
                return
            if 'surface_reflectance' not in grp:
                print(f"Error: 'surface_reflectance' not found in {grp_path}")
                return
            sr_data = grp['surface_reflectance']
            spatial_ref = sr_data.attrs.get('spatial_ref', None)
            geo_transform = sr_data.attrs.get('GeoTransform', None)
            compute_and_save_rx_anomaly(grp, sr_data, spatial_ref=spatial_ref, geo_transform=geo_transform)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error processing RX anomaly for {file_path}: {e}")


def process_h5_file(file_path, tile_size=TILE_SIZE, num_endmembers=NUM_ENDMEMBERS):
    import time
    start_time = time.time()
    print(f"Opening {file_path}...")
    try:
        with h5py.File(file_path, 'r+') as f:
            grp, grp_path = get_hsi_data_group(f)
            if grp is None:
                print(f"Error: Could not find HDFEOS/GRIDS Data Fields in {file_path}")
                return
                
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

            # Check and compute RX anomaly detector score if not already present
            if 'rxanomaly' not in grp:
                compute_and_save_rx_anomaly(grp, sr_data, spatial_ref=spatial_ref, geo_transform=geo_transform)
            else:
                print("Dataset 'rxanomaly' already exists. Skipping RX anomaly calculation.")

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
            hm_name = f'heights_map_{tile_size}x{tile_size}'
            if hm_name in grp:
                grp[hm_name].attrs['processing_duration_seconds'] = duration
                
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
    
    # Default to Headwall or HySpex directory if available
    init_dir = r"C:\satelliteImagery\headwall-vnir\pfm1-landmine-uav-vnir-hsi-IGARSS-2026"
    if not os.path.exists(init_dir):
        init_dir = r"C:\satelliteImagery\hyspex\20250918_GE18\dual"

    h5_file = filedialog.askopenfilename(
        title="Select Hyperspectral Ortho Stack (.h5)",
        filetypes=[("HDF5 files", "*.h5"), ("All files", "*.*")],
        initialdir=init_dir
    )
    
    if not h5_file:
        print("No .h5 file selected.")
        sys.exit(1)
        
    h5_files = [h5_file]
    
    for h5_file in h5_files:
        print(f"\n{'='*50}\nProcessing File: {h5_file}\n{'='*50}")
        # Compute RX anomaly if not already present
        process_rx_anomaly_h5_file(h5_file)
        for i, TILE_SIZE in enumerate(TILE_SIZES):
            print(f"Processing for TILE_SIZE={TILE_SIZE}...")
            process_h5_file(h5_file, TILE_SIZE, NUM_ENDMEMBERS)
            
            # Full memory cleanup between tile sizes to prevent CUDA allocator
            # fragmentation and CPU memory accumulation that causes progressive slowdown
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
