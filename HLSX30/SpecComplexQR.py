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
        #warnings.warn('Data contains negative values')
        image2D = np.clip(image2D, 0, 2)
    if np.max(image2D) > 1:
        #warnings.warn('Data contains values greater than 1')
        image2D = np.clip(image2D, 0, 1)

    # --- NaN Handling ---
    valid_mask = ~np.isnan(image2D).any(axis=1)
    
    if np.sum(valid_mask) < num_endmembers:
        #print(f"Not enough valid pixels (no NaNs) to find {num_endmembers} endmembers. Found {np.sum(valid_mask)} valid pixels.")
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
        results = executor.map(_process_chunk_QR, chunks)
        
    for y_start_global, y_end_global, chunk_sum_map, chunk_count_map in results:
        sum_map[y_start_global:y_end_global, :] += chunk_sum_map
        count_map[y_start_global:y_end_global, :] += chunk_count_map
        
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        final_map = sum_map / count_map
        
    return final_map
