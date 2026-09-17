import numpy as np
import h5py
import tkinter as tk
from tqdm import tqdm
from tkinter import filedialog
import torch

TILE_SIZE = 3
NUM_ENDMEMBERS = 40

def maximumDistance_volumes_torch(img_cube, num_endmembers):
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
    # volumes[:, 0] = 0 (origin contributes no volume)
    # volumes[:, j] for j >= 1 is the cumulative product of heights h_1..h_{j}
    volumes = torch.zeros(B, num_endmembers, dtype=dtype, device=device)
    heights = torch.zeros(B, num_endmembers, dtype=dtype, device=device)

    # --- First endmember (max-norm pixel, already identified) ---
    v_first = local_img_cube[b_idx, :, idx_maxnorm]  # (B, C)
    h1 = torch.norm(v_first, dim=1)  # (B,)
    volumes[:, 1] = h1
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

    return endmembers, volumes, heights


def maximumDistance_volumes(pixels, num_endmembers):
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
        np.full(num_endmembers, np.nan)
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
    volumes = np.zeros(num_endmembers, dtype=np.float64)
    heights = np.zeros(num_endmembers, dtype=np.float64)

    selected = np.zeros(num_pixels, dtype=bool)
    selected[idx_minnorm] = True
    selected[idx_maxnorm] = True

    # --- First endmember: the localized max-norm pixel ---
    v = residuals[:, idx_maxnorm].copy()
    h_dist = np.linalg.norm(v)
    volumes[1] = h_dist
    heights[1] = h_dist

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

        # Volume is the cumulative product of orthogonal heights
        volumes[j] = volumes[j - 1] * h_dist
        heights[j] = h_dist

        # Project all pixels into the subspace orthogonal to the new endmember
        if h_dist > 1e-12:
            proj_component = np.outer(v, np.dot(v, residuals) / (h_dist ** 2))
            residuals -= proj_component

    return endmembers, volumes, heights


def process_volume_sliding_tile(frame_data, valid_mask=None, tile_size=TILE_SIZE, num_endmembers=NUM_ENDMEMBERS):
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
                
            endmembers, volumes, heights = maximumDistance_volumes_torch(tile_flat_torch, num_endmembers)
            
            volumes = volumes[0].cpu().numpy()
            heights = heights[0].cpu().numpy()
            
            if len(volumes) > 2:
                vol_val = np.max(volumes[2:])
            else:
                vol_val = 0.0
                
            cy = y_start + center_offset
            cx = x_start + center_offset
            neighborhood_map[cy, cx] = vol_val
            heights_map[cy, cx, :] = heights

    # Add the final datasets using naming conventions
    datasets = {
        f'neighborhood_map_{tile_size}x{tile_size}': neighborhood_map,
        f'heights_map_{tile_size}x{tile_size}': heights_map
    }
        
    return datasets

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
                sr_data = grp['surface_reflectance'][:]
                num_frames = sr_data.shape[0]
                
                neighborhood_maps = []
                heights_maps = []
                
                nodata_mask = None
                if 'nodata_mask' in grp:
                    nodata_mask = grp['nodata_mask'][:]
                
                fill_val = grp['surface_reflectance'].fillvalue if grp['surface_reflectance'].fillvalue is not None else 0
                
                for i in range(num_frames):
                    print(f"Processing frame {i+1}/{num_frames} with tile_size={tile_size} and num_endmembers={num_endmembers}...")
                    
                    # Load and scale to true reflectance [0.0, 1.0]
                    # Retain original fill_val behavior by referencing the unscaled data for mask fallback
                    raw_frame = sr_data[i]
                    frame_data = raw_frame.astype(np.float32) / 10000.0
                    
                    v_mask = None
                    if nodata_mask is not None:
                        v_mask = ~nodata_mask[i, 0]
                    else:
                        # Fallback to checking for fill_value across all bands using raw_frame
                        v_mask = np.all(raw_frame != fill_val, axis=0)
                        
                    results = process_volume_sliding_tile(frame_data, valid_mask=v_mask, tile_size=tile_size, num_endmembers=num_endmembers)
                    neighborhood_maps.append(results[f'neighborhood_map_{tile_size}x{tile_size}'])
                    heights_maps.append(results[f'heights_map_{tile_size}x{tile_size}'])
                    
                final_results = {
                    f'neighborhood_map_{tile_size}x{tile_size}': np.stack(neighborhood_maps, axis=0),
                    f'heights_map_{tile_size}x{tile_size}': np.stack(heights_maps, axis=0)
                }
                
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
    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename(
        title="Select SpecTIR HDF5 File",
        filetypes=[("HDF5 files", "*.h5"), ("All files", "*.*")]
    )

    if not file_path:
        print("No file selected. Exiting.")
        import sys
        sys.exit(0)

    for i, TILE_SIZE in enumerate([3, 4, 5, 7,9,11,13,15]):
        print(f"Processing for TILE_SIZE={TILE_SIZE}...")
        process_h5_file(file_path, TILE_SIZE, NUM_ENDMEMBERS)
