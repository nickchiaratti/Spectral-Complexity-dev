import numpy as np
import warnings
from scipy import ndimage
from joblib import Parallel, delayed
import multiprocessing


def maximumDistance(data, num_endmembers, chunk_size=50000):
    '''
    Memory-optimized MaxD geometric simplex extraction.
    Utilizes strict float32 typing and chunked vector broadcasting to 
    prevent ArrayMemoryErrors on hyperspectral (300+ band) datasets.
    
    Returns:
        endmembers [bands, num_endmembers]
        endmembers_index [1, num_endmembers]
    '''      
    image2D = np.reshape(data, (data.shape[0] * data.shape[1], data.shape[2]), order="F")

    if np.min(image2D) < 0:
        image2D = np.clip(image2D, 0, 1)
    if np.max(image2D) > 1:
        image2D = np.clip(image2D, 0, 1)

    valid_mask = ~np.isnan(image2D).any(axis=1)
    
    if np.sum(valid_mask) < num_endmembers:
        return np.full((image2D.shape[1], num_endmembers), np.nan), np.full((1, num_endmembers), np.nan)

    valid_data = image2D[valid_mask].astype(np.float32)
    valid_indices = np.where(valid_mask)[0]

    data_t = np.transpose(valid_data)
    num_bands, num_pix = data_t.shape

    # Use squared norm to avoid sqrt
    magnitude_sq = np.sum(np.square(data_t), axis=0)
    idx1 = np.argmax(magnitude_sq)
    idx2 = np.argmin(magnitude_sq)

    endmembers = np.zeros([num_bands, num_endmembers], dtype=np.float32)
    endmembers_index = np.zeros([1, num_endmembers], dtype=int)   

    endmembers[:, 0] = data_t[:, idx1]
    endmembers[:, 1] = data_t[:, idx2]
    
    endmembers_index[0, 0] = valid_indices[idx1]
    endmembers_index[0, 1] = valid_indices[idx2]

    # Pre-allocate strictly as float32 to prevent memory doubling
    data_proj = data_t.copy()

    for i in range(2, num_endmembers):
        diff = data_proj[:, idx2:idx2+1] - data_proj[:, idx1:idx1+1]
        
        # Omit SVD, replace with algebraic pseudoinverse for a column vector
        norm_sq = np.sum(np.square(diff))
        if norm_sq > 1e-12:
            pseudo = diff.T / norm_sq
        else:
            pseudo = np.zeros_like(diff.T)

        # EVIDENCE-BASED FIX: Chunked In-Place Projection
        # Applies the projection algebraically without constructing the Identity matrix
        for c in range(0, num_pix, chunk_size):
            c_end = min(c + chunk_size, num_pix)
            chunk = data_proj[:, c:c_end]
            proj_coef = np.matmul(pseudo, chunk)
            data_proj[:, c:c_end] -= np.matmul(diff, proj_coef)

        idx1 = idx2
        vec = data_proj[:, idx2:idx2+1] 
            
        # EVIDENCE-BASED FIX: Chunked Distance Calculation
        # Prevents NumPy from allocating a massive intermediate array for np.square()
        diff_new = np.zeros(num_pix, dtype=np.float32)
        for c in range(0, num_pix, chunk_size):
            c_end = min(c + chunk_size, num_pix)
            chunk = data_proj[:, c:c_end]
            diff_new[c:c_end] = np.sum(np.square(vec - chunk), axis=0)
            
        idx2 = np.argmax(diff_new)

        endmembers[:, i] = data_t[:, idx2]
        endmembers_index[0, i] = valid_indices[idx2]

    return endmembers, endmembers_index
#****************************
# QR Methods
#****************************

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


def _process_chunk_QR(chunk_args):
    y_starts_global, y_start_global, y_end_global, img_slice, tile_size, stride, num_endmembers, gram_type, norm_type = chunk_args
    bands = img_slice.shape[2]
    width = img_slice.shape[1]
    chunk_height = y_end_global - y_start_global
    
    chunk_sum_map = np.zeros((chunk_height, width), dtype=np.float32)
    chunk_count_map = np.zeros((chunk_height, width), dtype=np.int8)

    # Pre-allocate buffer for F-contiguous tile to prevent reshape copying in maximumDistance
    tile_buffer = np.empty((tile_size * tile_size, bands), dtype=np.float32, order='F')
    tile_buffer_3d = tile_buffer.reshape((tile_size, tile_size, bands), order='F')

    for y_start in y_starts_global:
        y_start_local = y_start - y_start_global
        for x_start in range(0, width - tile_size + 1, stride):
            y_end_local = y_start_local + tile_size
            x_end = x_start + tile_size
            
            tile_buffer_3d[...] = img_slice[y_start_local:y_end_local, x_start:x_end, :]
            endmembers, _ = maximumDistance(tile_buffer_3d, num_endmembers)
            localizationVec = endmembers[:, 1]
            
            if gram_type == 'minEndmember':
                remainingEndmembers = np.delete(endmembers, 1, axis=1)
                volume = calcGramLocalVolumes_QR(remainingEndmembers, localizationVec)
                volume = np.insert(volume, 0, 0.0)
            else:
                volume = calcGramLocalVolumes_QR(endmembers, np.zeros(bands))

            if norm_type == 'bandCount':
                m_array = np.arange(1, len(volume) + 1)
                volume = volume / np.power(bands, (m_array / 2.0))

            vol_val = np.max(volume[2:])

            chunk_sum_map[y_start_local:y_end_local, x_start:x_end] += vol_val
            chunk_count_map[y_start_local:y_end_local, x_start:x_end] += 1
            
    return y_start_global, y_end_global, chunk_sum_map, chunk_count_map

def process_volume_sliding_tile(frame_data, tile_size, stride, num_endmembers, gram_type, norm_type, n_jobs=None):
    """
    Parallelized sliding window processing.
    Produces equivalent output to process_volume_sliding_tile.
    """
    bands, height, width = frame_data.shape
    img = np.transpose(frame_data, (1, 2, 0))
    
    sum_map = np.zeros((height, width), dtype=np.float32)
    count_map = np.zeros((height, width), dtype=np.int8)

    
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
        
    results = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(_process_chunk_QR)(chunk) for chunk in chunks
    )
        
    for y_start_global, y_end_global, chunk_sum_map, chunk_count_map in results:
        sum_map[y_start_global:y_end_global, :] += chunk_sum_map
        count_map[y_start_global:y_end_global, :] += chunk_count_map
        
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        final_map = sum_map / count_map
        
    return final_map
