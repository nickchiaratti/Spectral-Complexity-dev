import numpy as np
import warnings
from scipy import ndimage


def maximumDistance(data, num_endmembers):
    '''
    Args:
        data (np.ndarray): 3D image cube [nPixels0, nPixels1, nbands]
        num_endmembers (int): number of endmembers to be calculated (choose more than expected to find)
    Returns:
        endmembers [bands, num_endmembers]
        endmembers_index [1, num_endmembers]
    '''      
    # Flatten 3D cube [rows, cols, bands] -> 2D [pixels, bands]
    image2D = np.reshape(data, (data.shape[0] * data.shape[1], data.shape[2]), order="F")

    if np.min(image2D) < 0:
        warnings.warn('Data contains negative values')
        image2D = np.clip(image2D, 0, 2)
    if np.max(image2D) > 1:
        warnings.warn('Data contains values greater than 1')
        image2D = np.clip(image2D, 0, 1)

    # --- NaN Handling ---
    valid_mask = ~np.isnan(image2D).any(axis=1)
    
    if np.sum(valid_mask) < num_endmembers:
        print(f"Not enough valid pixels (no NaNs) to find {num_endmembers} endmembers. Found {np.sum(valid_mask)} valid pixels.")
        return np.full((image2D.shape[1], num_endmembers), np.nan), np.full((1, num_endmembers), np.nan)

    valid_data = image2D[valid_mask]
    valid_indices = np.where(valid_mask)[0]

    # Transpose to [bands, pixels]
    data_t = np.transpose(valid_data)
    num_bands, num_pix = data_t.shape

    # calculate magnitude of all vectors to find min and max
    magnitude = np.linalg.norm(data_t, axis=0)
    idx1 = np.argmax(magnitude)
    idx2 = np.argmin(magnitude)

    # create empty output arrays for endmembers
    endmembers = np.zeros([num_bands, num_endmembers])
    endmembers_index = np.zeros([1, num_endmembers], dtype=int)   

    # assign largest and smallest vector as first and second endmembers
    endmembers[:, 0] = data_t[:, idx1]
    endmembers[:, 1] = data_t[:, idx2]
    
    endmembers_index[0, 0] = valid_indices[idx1]
    endmembers_index[0, 1] = valid_indices[idx2]

    # Use standard ndarray instead of deprecated np.matrix. 
    # Create a copy so we don't modify the original data_t needed for extraction
    data_proj = data_t.copy()
    identity_matrix = np.identity(num_bands)

    for i in range(2, num_endmembers):
        # Extract difference vector as 2D column array (bands, 1) to maintain shape for broadcasting
        diff = data_proj[:, idx2:idx2+1] - data_proj[:, idx1:idx1+1]
        # calculate pseudo inverse of difference vector
        pseudo = np.linalg.pinv(diff)
        data_proj = np.matmul((identity_matrix - np.matmul(diff, pseudo)), data_proj)

        idx1 = idx2
        vec = data_proj[:, idx2:idx2+1] # Shape (bands, 1)
            
        diff_new = np.sum(np.square(vec - data_proj), axis=0)

        # find the maximum distance for next endmember
        # np.argmax returns the index of the first occurrence of the maximum value
        idx2 = np.argmax(diff_new)

        # assign to endmember file
        endmembers[:, i] = data_t[:, idx2]
        
        # Map back to original index
        endmembers_index[0, i] = valid_indices[idx2]

    return endmembers, endmembers_index

def calcGramLocalVolumes(endmembers, localization_vector):
    """
    Calculates the Local Gram matrix.
    1. Subtracts the localization vector from all other endmembers (centering the simplex on x).
    2. Calculates the Gram matrix of these centered vectors.
    3. Calculates the parallelotope volume estimate for 1 through N endmembers
    4. Returns volume values in an array of length N
    """
    # Reduce to current number of endmembers
    # Shape: (Bands, N)
    localized_vectors = endmembers - localization_vector[:, np.newaxis]

    # Calculate Gram Matrix
    # G = V^T * V (Shape: N x N)
    gram = np.matmul(localized_vectors.T, localized_vectors)
    
    # Initialize array to store the volume sequence
    N = gram.shape[0]
    volumes = np.zeros(N)
    
    # Calculate the parallelotope volume estimate for 1 through N endmembers
    for i in range(1, N + 1):
        # Extract the i x i top-left submatrix
        sub_gram = gram[:i, :i]
        
        # Calculate the Gramian determinant
        det = np.linalg.det(sub_gram)
        
        # Guard against floating-point inaccuracies that can cause tiny 
        # negative determinants near the linear dependence threshold
        if det < 0:
            det = 0.0
            
        # Volume is the square root of the Gramian determinant
        volumes[i-1] = np.sqrt(det)
        
    return volumes


def process_volume_frame(frame_data, num_endmembers, gram_type, norm_type):
    """
    Process the image to identify endmembers for the entire frame.
    Pixel Filtering: Only valid pixels are extracted into the 2D matrix.
    Returns the full volume curve, endmembers, and indices.
    """
    bands, height, width = frame_data.shape
    img = np.transpose(frame_data, (1, 2, 0))
    image2D = np.reshape(img, (height * width, bands))
    # Check gram type
    if gram_type == 'datasetMean': print("Localizing Gram to dataset mean")
    elif gram_type == 'minEndmember': print("Localizing Gram to second endmember")
    else: print("Localizing Gram to 0")
    # Check norm type
    if norm_type == 'bandCount': print(f"Normalizing Endmembers by √{bands}")
    else: print("No Endmember Normalization Applied")
    # Find endmembers
    endmembers, endmember_indices = maximumDistance(img, num_endmembers)
    meanVector = img.mean(axis=(0, 1))
    localizationVec = endmembers[:,1]

    if gram_type == 'datasetMean':
        volume = calcGramLocalVolumes(endmembers,meanVector)
    elif gram_type == 'minEndmember':
        remainingEndmembers = np.delete(endmembers,1,axis=1)
        volume = calcGramLocalVolumes(remainingEndmembers,localizationVec)
        volume = np.insert(volume,0,0.0)
    else:
        volume = calcGramLocalVolumes(endmembers,np.zeros(bands))

    if norm_type == 'bandCount':
        m_array = np.arange(1, len(volume) + 1)
        volume = volume / np.power(bands, (m_array / 2.0))

    # Return full volume array (curve) instead of just the maximum
    return endmembers, endmember_indices, volume

def process_volume_sliding_tile(frame_data, tile_size, stride, num_endmembers, gram_type, norm_type):
    """
    Sliding window processing.
    Strict Validity: Window is only processed if ALL pixels are valid.
    Output is masked with NaN for any pixel identified as invalid.
    """
    bands, height, width = frame_data.shape
    img = np.transpose(frame_data, (1, 2, 0))
    
    sum_map = np.zeros((height, width), dtype=np.float32)
    count_map = np.zeros((height, width), dtype=np.int8)
    #if gram_type == 'minEndmember':
    #    print("Localizing Gram to second endmember")
    #else:
    #    print("Localizing Gram to 0")
#
    #if norm_type == 'bandCount':
    #    print(f"Normalizing Endmembers by √{bands}")
    #else:
    #    print("No Endmember Normalization Applied")
    
    for y_start in range(0, height - tile_size + 1, stride):
        for x_start in range(0, width - tile_size + 1, stride):
            y_end, x_end = y_start + tile_size, x_start + tile_size
            
            tile = img[y_start:y_end, x_start:x_end, :]
            meanVector = tile.mean(axis=(0, 1))
            endmembers, _ = maximumDistance(tile, num_endmembers)
            localizationVec = endmembers[:,1]

            if gram_type == 'minEndmember':
                remainingEndmembers = np.delete(endmembers,1,axis=1)
                volume = calcGramLocalVolumes(remainingEndmembers,localizationVec)
                volume = np.insert(volume,0,0.0)
            else:
                volume = calcGramLocalVolumes(endmembers,np.zeros(bands))

            if norm_type == 'bandCount':
                m_array = np.arange(1, len(volume) + 1)
                volume = volume / np.power(bands, (m_array / 2.0))

            vol_val = np.max(volume[2:])

            sum_map[y_start:y_end, x_start:x_end] += vol_val
            count_map[y_start:y_end, x_start:x_end] += 1
            
    return sum_map / count_map

def calcGramLocalVolumes_QR(endmembers, localization_vector):
    """
    See page 251 in "The Theory of Matrices" Volume 1 by F.R. Gantmacher for equating volumes to product of heights
    Calculates the parallelotope volume estimate using QR Decomposition.
    """
    # 1. Reduce to current number of endmembers (Shape: Bands x N)
    localized_vectors = endmembers - localization_vector[:, np.newaxis]
    
    # 2. Perform QR decomposition directly on the localized vectors
    # Q is orthogonal (rotations), R is upper-triangular (scale/height)
    Q, R = np.linalg.qr(localized_vectors)
    
    # 3. The absolute values of the main diagonal of R are exactly 
    # the perpendicular heights (h) of the vectors!
    heights = np.abs(np.diag(R))
    
    # 4. Volume sequence: Vol_m = Vol_{m-1} * h_m
    # The cumulative product of the heights gives the expanding volumes
    volumes = np.cumprod(heights)
    
    return volumes

def _process_chunk(chunk_args):
    y_starts_global, y_start_global, y_end_global, img_slice, tile_size, stride, num_endmembers, gram_type, norm_type = chunk_args
    bands = img_slice.shape[2]
    width = img_slice.shape[1]
    chunk_height = y_end_global - y_start_global
    
    chunk_sum_map = np.zeros((chunk_height, width), dtype=np.float32)
    chunk_count_map = np.zeros((chunk_height, width), dtype=np.int8)

    for y_start in y_starts_global:
        y_start_local = y_start - y_start_global
        for x_start in range(0, width - tile_size + 1, stride):
            y_end_local = y_start_local + tile_size
            x_end = x_start + tile_size
            
            tile = img_slice[y_start_local:y_end_local, x_start:x_end, :]
            meanVector = tile.mean(axis=(0, 1))
            endmembers, _ = maximumDistance(tile, num_endmembers)
            localizationVec = endmembers[:, 1]
            
            if gram_type == 'minEndmember':
                remainingEndmembers = np.delete(endmembers, 1, axis=1)
                volume = calcGramLocalVolumes(remainingEndmembers, localizationVec)
                volume = np.insert(volume, 0, 0.0)
            else:
                volume = calcGramLocalVolumes(endmembers, np.zeros(bands))

            if norm_type == 'bandCount':
                m_array = np.arange(1, len(volume) + 1)
                volume = volume / np.power(bands, (m_array / 2.0))

            vol_val = np.max(volume[2:])

            chunk_sum_map[y_start_local:y_end_local, x_start:x_end] += vol_val
            chunk_count_map[y_start_local:y_end_local, x_start:x_end] += 1
            
    return y_start_global, y_end_global, chunk_sum_map, chunk_count_map

def process_volume_sliding_tile_parallel(frame_data, tile_size, stride, num_endmembers, gram_type, norm_type, n_jobs=None):
    """
    Parallelized sliding window processing.
    Produces equivalent output to process_volume_sliding_tile.
    """
    bands, height, width = frame_data.shape
    img = np.transpose(frame_data, (1, 2, 0))
    
    sum_map = np.zeros((height, width), dtype=np.float32)
    count_map = np.zeros((height, width), dtype=np.int8)
    
    #if gram_type == 'minEndmember':
    #    print("Localizing Gram to second endmember")
    #else:
    #    print("Localizing Gram to 0")
#
    #if norm_type == 'bandCount':
    #    print(f"Normalizing Endmembers by √{bands}")
    #else:
    #    print("No Endmember Normalization Applied")

    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor
    if n_jobs is None:
        n_jobs = multiprocessing.cpu_count()
        
    y_starts = list(range(0, height - tile_size + 1, stride))
    
    if len(y_starts) == 0:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return sum_map / count_map

    # Determine a good chunk size based on available workers
    # Divide the workload into roughly n_jobs * 2 chunks for load balancing
    num_chunks = max(1, n_jobs * 2)
    chunk_size_y = max(1, (len(y_starts) + num_chunks - 1) // num_chunks)
    
    chunks = []
    for i in range(0, len(y_starts), chunk_size_y):
        chunk_y_starts = y_starts[i : i + chunk_size_y]
        
        y_first = chunk_y_starts[0]
        y_last = chunk_y_starts[-1]

        y_start_global = y_first
        y_end_global = min(y_last + tile_size, height)
        
        img_slice = img[y_start_global:y_end_global, :, :]
        chunks.append((chunk_y_starts, y_start_global, y_end_global, img_slice, tile_size, stride, num_endmembers, gram_type, norm_type))
        
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        results = executor.map(_process_chunk, chunks)
        
    for y_start_global, y_end_global, chunk_sum_map, chunk_count_map in results:
        sum_map[y_start_global:y_end_global, :] += chunk_sum_map
        count_map[y_start_global:y_end_global, :] += chunk_count_map
        
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        final_map = sum_map / count_map
        
    return final_map


# ---------------------------------------------------------------------------
# GPU Implementation (CuPy) — Tier 2: Batched rank-1 projection + batched QR
# ---------------------------------------------------------------------------

def _gpu_batch_maximum_distance(data_t, num_endmembers, cp):
    """
    Batched equivalent of maximumDistance() using a rank-1 orthogonal projection
    update instead of pinv/SVD.  This eliminates the SVD bottleneck and enables
    all T tiles to be processed simultaneously on the GPU.

    Mathematical equivalence
    ------------------------
    The original loop computes:
        data_proj = (I - diff @ pinv(diff)) @ data_proj

    For a column vector diff of shape (B, 1), the Moore-Penrose pseudo-inverse is:
        pinv(diff) = diff.T / (diff.T @ diff) = diff.T / ||diff||²

    Substituting:
        projection_matrix = I - (diff @ diff.T) / ||diff||²

    This is a rank-1 Householder-like update identical to the Gram-Schmidt step.
    Applied to the full [B, P] pixel matrix (in batch across T tiles):
        proj   = diff.T @ data_proj          →  [T, P]  (dot product with each pixel)
        update = diff[:, :, None] * proj[:, None, :] / ||diff||²
        data_proj -= update

    Parameters
    ----------
    data_t : cp.ndarray  shape [T, B, P]
        T tile pixel matrices (B bands × P pixels each).
    num_endmembers : int
    cp : cupy module

    Returns
    -------
    endmembers : cp.ndarray  shape [T, B, num_endmembers]
        Selected endmember spectral vectors for each tile.
    """
    T, B, P = data_t.shape
    T_idx = cp.arange(T, dtype=cp.int32)

    # -- initialise with highest and lowest L2-norm pixels (mirrors maximumDistance) --
    magnitude = cp.linalg.norm(data_t, axis=1)          # [T, P]
    idx1 = cp.argmax(magnitude, axis=1).astype(cp.int32) # [T]
    idx2 = cp.argmin(magnitude, axis=1).astype(cp.int32) # [T]

    endmembers = cp.zeros((T, B, num_endmembers), dtype=cp.float32)
    endmembers[:, :, 0] = data_t[T_idx, :, idx1]        # brightest
    endmembers[:, :, 1] = data_t[T_idx, :, idx2]        # darkest

    # data_proj is modified in-place across iterations (copy preserves data_t)
    data_proj = data_t.copy()                            # [T, B, P]

    for i in range(2, num_endmembers):
        # diff: direction between the two active endmembers — [T, B]
        diff = data_proj[T_idx, :, idx2] - data_proj[T_idx, :, idx1]

        # ||diff||² per tile — clamp to avoid division by zero: [T]
        norm_sq = cp.maximum(cp.sum(diff * diff, axis=1), 1e-30)

        # Scalar projection of every pixel onto diff direction: [T, P]
        proj = cp.einsum('tb,tbp->tp', diff, data_proj)

        # Rank-1 orthogonal projection update (in-place): [T, B, P]
        # data_proj -= outer(diff, proj) / norm_sq
        data_proj -= (diff[:, :, None] * proj[:, None, :]) / norm_sq[:, None, None]

        idx1 = idx2.copy()

        # Pixel farthest from current idx2 in the projected space: [T, P]
        vec       = data_proj[T_idx, :, idx2]            # [T, B]
        diff_new  = cp.sum((vec[:, :, None] - data_proj) ** 2, axis=1)  # [T, P]
        idx2      = cp.argmax(diff_new, axis=1).astype(cp.int32)        # [T]

        # Record from the *original* (un-projected) data so endmembers are actual spectra
        endmembers[:, :, i] = data_t[T_idx, :, idx2]

    return endmembers


def _gpu_batch_gram_volumes_qr(endmembers, bands, gram_type, norm_type, num_endmembers, cp):
    """
    Batched parallelotope volume computation using QR decomposition.

    Mathematical equivalence with calcGramLocalVolumes
    --------------------------------------------------
    For a matrix V of shape [B, N] (B bands, N endmembers):
        Gram method : vol_i = sqrt(det(V[:,:i].T @ V[:,:i]))
        QR method   : V = QR  →  vol_i = prod(|diag(R[:i,:i])|)  = prod(|R[j,j]| for j<i)

    These are identical because det(V.T V) = det(R.T R) = det(R)² for an economy QR
    of a tall-thin matrix, and the lower principal submatrices of R give the partial volumes.
    |R[j,j]| is the perpendicular height of endmember j above the hyperplane spanned
    by endmembers 0..j-1  (Gram-Schmidt height), matching Gantmacher §5.2.

    CuPy's cp.linalg.qr supports batched input [T, B, N] → Q:[T,B,N], R:[T,N,N].

    Parameters
    ----------
    endmembers : cp.ndarray  shape [T, B, num_endmembers]
    bands, gram_type, norm_type, num_endmembers : same semantics as serial function
    cp : cupy module

    Returns
    -------
    volumes : cp.ndarray  shape [T, num_endmembers]
    """
    T = endmembers.shape[0]

    if gram_type == 'minEndmember':
        # Localise all vectors relative to the second endmember (index 1)
        loc_vec     = endmembers[:, :, 1:2]                          # [T, B, 1]
        em_remain   = cp.concatenate(
            [endmembers[:, :, :1], endmembers[:, :, 2:]], axis=2)    # [T, B, N-1]
        localized   = em_remain - loc_vec                            # [T, B, N-1]
        N           = num_endmembers - 1
    else:
        localized   = endmembers                                     # [T, B, N]
        N           = num_endmembers

    # Batched economy QR: localized [T, B, N]  →  R [T, N, N]
    _, R = cp.linalg.qr(localized)                                   # R: [T, N, N]

    # |diag(R)| = perpendicular heights [T, N]
    heights = cp.abs(cp.diagonal(R, axis1=-2, axis2=-1))             # [T, N]

    # Cumulative product of heights = expanding parallelotope volumes [T, N]
    volumes = cp.cumprod(heights, axis=1)                            # [T, N]

    if gram_type == 'minEndmember':
        # Prepend 0 to mirror:  volume = np.insert(volume, 0, 0.0)
        volumes = cp.concatenate(
            [cp.zeros((T, 1), dtype=cp.float32), volumes], axis=1)  # [T, num_endmembers]

    if norm_type == 'bandCount':
        m_array = cp.arange(1, num_endmembers + 1, dtype=cp.float32)
        volumes = volumes / cp.power(float(bands), m_array[None, :] / 2.0)

    return volumes                                                    # [T, num_endmembers]


def process_volume_sliding_tile_gpu(frame_data, tile_size, stride, num_endmembers,
                                    gram_type, norm_type,
                                    chunk_tiles=None, device=0):
    """
    GPU-accelerated sliding-window spectral-complexity map (CuPy backend).

    Produces output equivalent to process_volume_sliding_tile() but processes
    all tile positions in large batches on the GPU rather than sequentially on
    the CPU.  Two key algorithm substitutions enable full GPU batch execution:

    1. **Rank-1 projection instead of pinv/SVD** inside maximumDistance:
       For a column vector ``diff`` of shape (B,1), pinv(diff) = diff.T/||diff||²,
       reducing the projection update to a single outer-product rank-1 update
       (no SVD).  All T tiles are updated simultaneously in one batched einsum.

    2. **Batched QR instead of sequential Gramian determinants** for volumes:
       CuPy's batched ``linalg.qr`` decomposes the [T,B,N] endmember tensor in one
       call; |diag(R)| gives the perpendicular heights whose cumprod equals the
       parallelotope volumes (see Gantmacher, *Theory of Matrices*, Vol. 1, §5.2).

    Memory
    ------
    Tiles are streamed to the GPU in chunks of ``chunk_tiles`` to avoid OOM
    errors for large images.  The full image is transferred to the GPU once
    (as read-only) if GPU memory is sufficient; otherwise tiles are extracted
    on the CPU and transferred per chunk.

    Scatter-add (stride=1 optimisation)
    ------------------------------------
    For stride=1 the vol_val grid is a rectangular [n_y, n_x] array.  The
    sum_map / count_map accumulation across overlapping tiles is mathematically
    a 2-D separable box-filter convolution and is replaced by a double-nested
    loop of O(tile_size²) whole-array additions — far cheaper than per-tile
    scatter.

    Parameters
    ----------
    frame_data      : np.ndarray  [bands, height, width], values in [0, 1]
    tile_size       : int   sliding window side length
    stride          : int   step size (optimised path for stride=1)
    num_endmembers  : int   number of endmembers per tile (>= bands recommended < 8)
    gram_type       : str   'minEndmember' | anything else → zero localisation
    norm_type       : str   'bandCount'    | anything else → no normalisation
    chunk_tiles     : int | None
        Number of tile positions per GPU batch.  If None, auto-sized to fit
        roughly 256 MB of GPU working memory.
    device          : int   CUDA device index (default 0)

    Returns
    -------
    np.ndarray  [height, width], dtype float32  —  per-pixel average volume map

    Raises
    ------
    ImportError  if CuPy is not installed.

    Notes
    -----
    * NaN handling and value-range clipping (present in the CPU code) are
      omitted here for GPU efficiency.  Ensure frame_data is valid before
      calling.
    * Results will differ from the CPU version at the ~1e-4 level due to
      float32 GPU arithmetic vs float64 CPU arithmetic and different ordering
      (QR vs sequential Gramian determinants), but are mathematically equivalent.
    """
    try:
        import cupy as cp
    except ImportError:
        raise ImportError(
            "CuPy is required for GPU processing.  "
            "Install the wheel matching your CUDA version, e.g.:\n"
            "  pip install cupy-cuda12x"
        )

    bands, height, width = frame_data.shape
    # Transpose once to [H, W, B] for tile extraction; ensure float32 on GPU
    img_np = np.transpose(frame_data, (1, 2, 0)).astype(np.float32)  # [H, W, B]

    # Number of valid tile origins
    n_y = (height - tile_size) // stride + 1
    n_x = (width  - tile_size) // stride + 1
    total_tiles = n_y * n_x

    # Auto-determine chunk size: target ≈256 MB for the working tile buffer
    # (data_proj copy + endmembers + intermediates ≈ 5× raw tile size)
    if chunk_tiles is None:
        P               = tile_size * tile_size
        bytes_per_tile  = bands * P * 4 * 5          # float32 ×5 buffers
        budget          = 256 * 1024 * 1024           # 256 MB
        chunk_tiles     = max(64, min(total_tiles, budget // max(bytes_per_tile, 1)))

    # Flat arrays of every tile's top-left corner [total_tiles]
    ys_grid, xs_grid = np.meshgrid(
        np.arange(n_y, dtype=np.int32) * stride,
        np.arange(n_x, dtype=np.int32) * stride,
        indexing='ij')
    all_ys = ys_grid.ravel()   # [total_tiles]
    all_xs = xs_grid.ravel()   # [total_tiles]

    # Output buffer holds one scalar per tile; reconstructed into grid at the end
    vol_all = np.empty(total_tiles, dtype=np.float32)

    with cp.cuda.Device(device):
        # --- attempt to keep the full image on GPU to avoid repeated transfers ---
        try:
            img_gpu      = cp.asarray(img_np)
            use_gpu_img  = True
        except cp.cuda.memory.OutOfMemoryError:
            img_gpu      = None
            use_gpu_img  = False

        dy_offsets = cp.arange(tile_size, dtype=cp.int32)   # reused every chunk
        dx_offsets = cp.arange(tile_size, dtype=cp.int32)

        for c_start in range(0, total_tiles, chunk_tiles):
            c_end = min(c_start + chunk_tiles, total_tiles)
            T     = c_end - c_start

            ys_c = all_ys[c_start:c_end]  # CPU [T]
            xs_c = all_xs[c_start:c_end]  # CPU [T]

            if use_gpu_img:
                # Build 2-D offset grids on the GPU and let CuPy broadcast:
                #   y_idx : [T, tile_size, 1]   x_idx : [T, 1, tile_size]
                # → fancy-indexed result : [T, tile_size, tile_size, B]
                ys_cp  = cp.asarray(ys_c)                             # [T]
                xs_cp  = cp.asarray(xs_c)                             # [T]
                y_idx  = ys_cp[:, None, None] + dy_offsets[None, :, None]  # [T, ts, 1]
                x_idx  = xs_cp[:, None, None] + dx_offsets[None, None, :]  # [T, 1, ts]
                # img_gpu[y_idx, x_idx, :] broadcasts to [T, ts, ts, B]
                data_t = (img_gpu[y_idx, x_idx, :]
                          .reshape(T, tile_size * tile_size, bands)
                          .transpose(0, 2, 1))                        # [T, B, P]
            else:
                # Extract tiles on CPU with fancy indexing, transfer once per chunk
                y_idx_np = (ys_c[:, None, None]
                            + np.arange(tile_size)[None, :, None])   # [T, ts, 1]
                x_idx_np = (xs_c[:, None, None]
                            + np.arange(tile_size)[None, None, :])   # [T, 1, ts]
                tiles_np = img_np[y_idx_np, x_idx_np, :]             # [T, ts, ts, B]
                data_t   = cp.asarray(
                    tiles_np.reshape(T, tile_size * tile_size, bands)
                            .transpose(0, 2, 1)
                )                                                     # [T, B, P]

            # ----------------------------------------------------------------
            # 1. Batched maximum-distance endmember selection (rank-1 update)
            # ----------------------------------------------------------------
            endmembers = _gpu_batch_maximum_distance(data_t, num_endmembers, cp)
            # endmembers: [T, B, num_endmembers]

            # ----------------------------------------------------------------
            # 2. Batched parallelotope volumes via QR decomposition
            # ----------------------------------------------------------------
            volumes = _gpu_batch_gram_volumes_qr(
                endmembers, bands, gram_type, norm_type, num_endmembers, cp)
            # volumes: [T, num_endmembers]

            # Maximum volume excluding first two elements (mirrors vol_val = max(volume[2:]))
            vol_val = cp.max(volumes[:, 2:], axis=1)                 # [T]

            vol_all[c_start:c_end] = cp.asnumpy(vol_val)

    # --------------------------------------------------------------------
    # Scatter-add: accumulate tile contributions into sum_map / count_map
    #
    # For stride=1, vol_all reshaped to [n_y, n_x] is a regular grid.
    # A pixel at (r, c) is covered by tiles whose origin (ys, xs) satisfies
    #   max(0, r-ts+1) ≤ ys ≤ min(r, n_y-1)  (and symmetric for x).
    # This is identical to a 2-D box convolution of the vol_grid with a
    # tile_size×tile_size all-ones kernel.  The O(tile_size²) loop below
    # exploits this structure to avoid an O(total_tiles × tile_size²) loop.
    # --------------------------------------------------------------------
    vol_grid = vol_all.reshape(n_y, n_x).astype(np.float64)

    if stride == 1:
        sum_map   = np.zeros((height, width), dtype=np.float64)
        count_map = np.zeros((height, width), dtype=np.int32)
        for dy in range(tile_size):
            for dx in range(tile_size):
                sum_map  [dy:dy + n_y, dx:dx + n_x] += vol_grid
                count_map[dy:dy + n_y, dx:dx + n_x] += 1
    else:
        # Generic path for arbitrary strides
        sum_map   = np.zeros((height, width), dtype=np.float64)
        count_map = np.zeros((height, width), dtype=np.int32)
        for t_idx in range(total_tiles):
            ys_t = all_ys[t_idx]
            xs_t = all_xs[t_idx]
            v    = vol_grid.ravel()[t_idx]
            sum_map  [ys_t:ys_t + tile_size, xs_t:xs_t + tile_size] += v
            count_map[ys_t:ys_t + tile_size, xs_t:xs_t + tile_size] += 1

    with np.errstate(invalid='ignore', divide='ignore'):
        return (sum_map / count_map).astype(np.float32)