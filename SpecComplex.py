import numpy as np
import math
import warnings
from skimage import exposure
from scipy import ndimage


# Mapped levels for Aerosol_Optical_Depth
AEROSOL_DICT = {
    'low': [2, 4, 32, 66, 68, 96, 100],
    'medium': [2, 4, 32, 66, 68, 96, 100, 130, 132, 160, 164],
    'high': [2, 4, 32, 66, 68, 96, 100, 130, 132, 160, 164, 192, 194, 196, 224, 228] # Aerosol_Optical_Depth > 0.3
}

def get_landsat_mask(data_grp, f_idx, shape, 
                     sun_elevation_threshold=25, 
                     cloud_dilation=0, 
                     qa_reject_mask=0b111111, 
                     radsat_accept_value=0, 
                     aerosol_accept_level='medium'):
    """
    Generates a boolean spatial mask for LANDSAT data using Quality Assessment (QA) bands.
    Valid pixels return True, masked pixels return False.
    """
    valid_mask = np.ones(shape, dtype=bool)
    kernel = np.ones((3, 3), dtype=bool)
    
    # Sun Elevation Check (Fails loudly if attribute is missing)
    sun_elev_arr = data_grp['surface_reflectance'].attrs['sun_elevation']
    if sun_elev_arr[f_idx] < sun_elevation_threshold:
        return np.zeros(shape, dtype=bool)

    # QA Reject Mask
    qa_pixel = data_grp['QUALITY_L1_PIXEL'][f_idx, ...]
    bad_qa_mask = (qa_pixel & qa_reject_mask) != 0
    if cloud_dilation > 0:
        bad_qa_mask = ndimage.binary_dilation(bad_qa_mask, structure=kernel, iterations=cloud_dilation)
    valid_mask &= ~bad_qa_mask

    # RADSAT Accept Value
    bad_radsat = data_grp['RADIOMETRIC_SATURATION'][f_idx, ...] != radsat_accept_value
    valid_mask &= ~bad_radsat

    # Aerosol Accept Values
    if aerosol_accept_level != 'all':
        aerosol = data_grp['QUALITY_L2_AEROSOL'][f_idx, ...]
        
        accepted_values = AEROSOL_DICT.get(aerosol_accept_level)
        if accepted_values is None:
            raise ValueError(f"Invalid aerosol_accept_level: '{aerosol_accept_level}'. Must be 'low', 'medium', or 'high'.")
            
        invalid_aerosol = ~np.isin(aerosol, accepted_values)
        if cloud_dilation > 0:
            invalid_aerosol = ndimage.binary_dilation(invalid_aerosol, structure=kernel, iterations=cloud_dilation)
        valid_mask &= ~invalid_aerosol

    return valid_mask

def get_tanager_mask(data_grp, f_idx, shape, 
                     sun_elevation_threshold=25, 
                     cloud_dilation=2, 
                     apply_cloud_mask=True, 
                     uncertainty_threshold=0.1, 
                     aerosol_depth_threshold=0.3):
    """
    Generates a boolean spatial mask for TANAGER data using beta masks and uncertainty.
    Valid pixels return True, masked pixels return False.
    """
    valid_mask = np.ones(shape, dtype=bool)
    kernel = np.ones((3, 3), dtype=bool)
    
    # Cloud Mask Check
    if apply_cloud_mask:
        c_mask = (data_grp['beta_cloud_mask'][f_idx, ...] == 1)
        cirrus_mask = (data_grp['beta_cirrus_mask'][f_idx, ...] == 1)
        combined_cloud = c_mask | cirrus_mask
        if cloud_dilation > 0:
            combined_cloud = ndimage.binary_dilation(combined_cloud, structure=kernel, iterations=cloud_dilation)
        valid_mask &= ~combined_cloud
    
    # Sun Elevation Check (Derived from Sun Zenith)
    zenith = data_grp['sun_zenith'][f_idx, ...]
    valid_mask &= (zenith != -9999.0) & ((90.0 - zenith) >= sun_elevation_threshold)
        
    # Aerosol Optical Depth Check
    aod = data_grp['aerosol_optical_depth'][f_idx, ...]
    bad_aod_mask = (aod == -9999.0) | (aod >= aerosol_depth_threshold) | np.isnan(aod)
    if cloud_dilation > 0:
        bad_aod_mask = ndimage.binary_dilation(bad_aod_mask, structure=kernel, iterations=cloud_dilation)
    valid_mask &= ~bad_aod_mask
        
    # Surface Reflectance Uncertainty Check
    gw_mask = data_grp['surface_reflectance'].attrs['all_good_wavelengths']
    valid_bands = gw_mask[f_idx].astype(bool)
    
    # Suppress all-NaN slice warnings since we explicitly catch the resulting NaNs on the next line
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        unc = np.nanmax(data_grp['surface_reflectance_uncertainty'][f_idx, valid_bands, ...], axis=0)
        
    unc_mask = (unc == -9999.0) | (unc >= uncertainty_threshold) | np.isnan(unc)
    if cloud_dilation > 0:
        unc_mask = ndimage.binary_dilation(unc_mask, structure=kernel, iterations=cloud_dilation)
    valid_mask &= ~unc_mask
        
    return valid_mask

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

def generate_rgba_image(frame_sr, red_idx=3, green_idx=2, blue_idx=1, low=2, high=98, gamma=1.2):
    """
    Extracts, stretches, and gamma-corrects the RGB bands from a surface 
    reflectance frame to create a true color image with an alpha channel.
    
    The alpha channel follows standard RGBA opacity conventions:
    255 (Opaque) where pixels are valid, and 0 (Transparent) where 
    pixels are invalid (all 0s or containing NaNs).
    
    Returns:
        rgba_8bit (np.ndarray): Shape (height, width, 4), dtype uint8.
    """
    bands, height, width = frame_sr.shape

    # Handle case where the entire frame is NaN
    if np.all(np.isnan(frame_sr)): 
        return np.zeros((height, width, 4), dtype=np.uint8)

    # 1. Determine Invalid Pixel Mask
    all_zeros_mask = np.all(frame_sr == 0, axis=0)
    has_nan_mask = np.any(np.isnan(frame_sr), axis=0)
    invalid_pixel_mask = all_zeros_mask | has_nan_mask
    valid_mask = ~invalid_pixel_mask

    # 2. Extract, Stretch, and Gamma Correct RGB bands
    rgb_indices = [red_idx, green_idx, blue_idx]
    rgb = np.zeros((height, width, 3), dtype=np.float32)

    for i, idx in enumerate(rgb_indices):
        band_data = frame_sr[idx, :, :]
        
        # Calculate percentiles ONLY on valid pixels to bypass slow NaN handling
        valid_pixels = band_data[valid_mask]
        
        if valid_pixels.size == 0:
            continue # Leave as 0 if band is entirely empty
            
        p_low, p_high = np.percentile(valid_pixels, (low, high))
        
        if p_low < p_high: 
            # Manual linear stretch (highly optimized)
            stretched = np.clip((band_data - p_low) / (p_high - p_low), 0.0, 1.0)
            
            # Non-linear Gamma Correction to improve mid-tone visibility
            if gamma != 1.0:
                # Prevent power warnings on 0 values
                with np.errstate(invalid='ignore', divide='ignore'):
                    stretched = np.power(stretched, 1.0 / gamma)
                    # Clean up any potential inf/nan from power operation
                    stretched = np.nan_to_num(stretched, nan=0.0, posinf=1.0, neginf=0.0)
            
            rgb[:, :, i] = stretched

    # Scale to 8-bit for highly efficient HDF5 storage
    rgb_8bit = (rgb * 255).astype(np.uint8)
    
    # 3. Construct the Alpha Channel
    alpha = np.full((height, width), 255, dtype=np.uint8)
    alpha[invalid_pixel_mask] = 0
    
    # 4. Stack alpha onto RGB to create RGBA
    rgba_8bit = np.dstack((rgb_8bit, alpha))
    
    return rgba_8bit


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

def process_volume_tiles(frame_data, tile_size, num_endmembers, gram_type, norm_type):
    """
    Grid-based processing (Non-overlapping tiles).
    Strict Validity: Window is only processed if ALL pixels are valid.
    Any pixel that is part of an invalid tile is set to NaN.
    """
    bands, height, width = frame_data.shape
    img = np.transpose(frame_data, (1, 2, 0))
    output_map = np.full((height, width), np.nan, dtype=np.float32)
    # Check gram type
    if gram_type == 'datasetMean': print("Localizing Gram to dataset mean")
    elif gram_type == 'minEndmember': print("Localizing Gram to second endmember")
    else: print("Localizing Gram to 0")
    # Check norm type
    if norm_type == 'bandCount': print(f"Normalizing Endmembers by √{bands}")
    else: print("No Endmember Normalization Applied")
    
    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            y_end, x_end = min(y + tile_size, height), min(x + tile_size, width)
            tile = img[y:y_end, x:x_end, :]
            if tile[:,:,0].size >= num_endmembers:
                meanVector = tile.mean(axis=(0, 1))
                volume = np.zeros(num_endmembers)
                endmembers, _ = maximumDistance(tile, num_endmembers)
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
                
                output_map[y:y_end, x:x_end] = np.max(volume[2:])
    
    return output_map

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
    if gram_type == 'datasetMean':
        print("Localizing Gram to dataset mean")
    elif gram_type == 'minEndmember':
        print("Localizing Gram to second endmember")
    else:
        print("Localizing Gram to 0")

    if norm_type == 'bandCount':
        print(f"Normalizing Endmembers by √{bands}")
    else:
        print("No Endmember Normalization Applied")
    
    for y_start in range(0, height - tile_size + 1, stride):
        for x_start in range(0, width - tile_size + 1, stride):
            y_end, x_end = y_start + tile_size, x_start + tile_size
            
            tile = img[y_start:y_end, x_start:x_end, :]
            meanVector = tile.mean(axis=(0, 1))
            endmembers, _ = maximumDistance(tile, num_endmembers)
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

            vol_val = np.max(volume[2:])

            sum_map[y_start:y_end, x_start:x_end] += vol_val
            count_map[y_start:y_end, x_start:x_end] += 1
            
    return sum_map / count_map


def plot_endmember_locations(image_cube, rgb_image, endmember_indices, endmembers):
    """
    Plots the spatial (row, col) location of each found endmember
    on top of a true-color representation of the original image.

    Args:
        image_cube (np.ndarray): The *original* [rows, cols, bands] image.
        rgb_image (np.ndarray): The true color image to overlay.
        endmember_indices (np.ndarray): The 1D [numEndmembers] array of indices.
        endmembers (np.ndarray): The [bands, numEndmembers] array, used to check for validity.
    """
    print("\nGenerating endmember spatial location plot...")
    
    # --- MODIFICATION: Get cols from image_cube ---

    rows, cols, bands = image_cube.shape

    plt.figure(figsize=(12, 9))
    
    # 2. Display the RGB image
    plt.imshow(rgb_image)
    
    # 3. Plot endmember locations
    for i, idx in enumerate(endmember_indices):
        # Check if this endmember was actually found (is not all zeros)
        if np.any(endmembers[:, i] != 0):
            # Convert 1D index back to 2D (row, col)
            row = idx // cols
            col = idx % cols
            
            # Plot a crosshair
            plt.plot(col, row, 'r+', markersize=15, markeredgewidth=2,  label=f'V[{i}]' if i < 2 else None) # Only label V0, V1 in legend
            
            # Annotate
            plt.annotate(f'V[{i}]', (col, row),  textcoords="offset points", xytext=(0, -15),  ha='center', fontsize=12, color='r', fontweight='bold')

    plt.title('Spatial Locations of Found Endmembers')
    plt.xlabel('Pixel Column')
    plt.ylabel('Pixel Row')
    # Create a custom legend for V[0] and V[1]
    if endmembers.shape[1] >= 2:
        plt.legend(handles=[
            plt.Line2D([0], [0], color='r', marker='+', linestyle='None', markersize=15, label='Endmember Location (e.g., V[0], V[1])')
        ])
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()

def plot_spectral_profiles(endmembers, band_count):
    """
    Plots the spectral signature of each endmember as a line graph.
    This is the most direct way to visualize what the endmembers are.

    Args:
        endmembers (np.ndarray): The [bands, numEndmembers] array.
        band_count (int): The number of bands (e.g., 8).
    """
    print("\nGenerating spectral profile plot...")
    plt.figure(figsize=(12, 9))
    
    # Create an x-axis representing the band number (1-indexed)
    x_axis = np.arange(1, band_count + 1)
    num_found_endmembers = endmembers.shape[1]
    
    for i in range(num_found_endmembers):
        em_signature = endmembers[:, i]
        # Check if this endmember was actually found (not just zeros)
        if np.any(em_signature != 0):
            plt.plot(x_axis, em_signature, label=f'Endmember {i} (V[{i}])', lw=2)
    
    plt.title('Spectral Signatures of Found Endmembers')
    plt.xlabel('Band Number')
    plt.ylabel('Pixel Value / Reflectance')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(x_axis) # Ensure every band has a tick

'''
Sandbox 

'''

def calculate_global_z_score(volume_array, valid_pixel_mask):
    """
    Calculates the global Z-score for an entire frame of spectral complexity volumes.
    Decouples the evaluation space from the background statistics space by strictly 
    calculating the mean and standard deviation from radiometrically valid pixels, 
    preventing artifacts from skewing the background model.
    """
    print("Calculating global Z-score for frame")
    height, width = volume_array.shape
    z_scores = np.full((height, width), np.nan, dtype=np.float32)
    
    # Identify globally valid pixels (strictly positive for log transform)
    global_valid_mask = volume_array > 0.0
    
    # Intersect with radiometrically valid pixels for the statistical background model
    stats_mask = global_valid_mask & valid_pixel_mask
    
    # Graceful fallback per user directive: Return NaNs for entire frame if no valid background exists.
    if not np.any(stats_mask):
        warnings.warn("calculate_global_z_score warning: No radiometrically valid pixels with volume > 0 found. Returning NaNs.")
        return z_scores
        
    # Extract subset volumes strictly for statistical estimation
    stats_vols = volume_array[stats_mask]
    log_stats_vols = np.log(stats_vols)
    
    # Calculate global scene statistics (using ddof=1 for unbiased sample estimator)
    global_mean = np.mean(log_stats_vols)
    global_std = np.std(log_stats_vols, ddof=1)
    
    # Strict failure handling: Prevent training on synthetically flat frames
    if global_std == 0:
        raise ValueError("calculate_global_z_score failed: Global standard deviation of the radiometrically valid subset is exactly zero.")
        
    # Evaluate ALL geometrically valid pixels using the pure background model
    apply_vols = volume_array[global_valid_mask]
    log_apply_vols = np.log(apply_vols)
    
    # Apply standard Z-score equation
    z_scores[global_valid_mask] = (log_apply_vols - global_mean) / global_std
    
    return z_scores

def calculate_local_z_score(volume_array, window_size, stride):
    """
    Calculates the local sliding-window Z-score for a frame of spectral complexity volumes.
    Uses a sum_map and count_map to average the Z-scores across all overlapping sliding windows,
    creating an ensemble anomaly detection map.
    """
    print(f"Calculating local {window_size}x{window_size} neighborhood Z-score for frame")
    height, width = volume_array.shape
    
    sum_map = np.zeros((height, width), dtype=np.float32)
    count_map = np.zeros((height, width), dtype=np.int32)
    
    # 1. Identify globally valid pixels (strictly positive for log transform)
    global_valid_mask = volume_array > 0.0
    
    if not np.any(global_valid_mask):
        warnings.warn("No valid volumes found in frame > 0. Returning NaNs.")
        return np.full((height, width), np.nan, dtype=np.float32)

    for y_start in range(0, height - window_size + 1, stride):
        for x_start in range(0, width - window_size + 1, stride):
            y_end = y_start + window_size
            x_end = x_start + window_size
            
            window = volume_array[y_start:y_end, x_start:x_end]
            valid_mask = window > 0.0
            
            # We need at least two valid pixels in the window to calculate a standard deviation
            if np.sum(valid_mask) < 2:
                continue
            
            valid_vols = window[valid_mask]
            log_vols = np.log(valid_vols)
            
            local_mean = np.mean(log_vols)
            local_std = np.std(log_vols)
            
            # Initialize a Z-score window of 0s
            z_window = np.zeros((window_size, window_size), dtype=np.float32)
            
            # Calculate Z-scores for valid pixels only
            if local_std > 1e-12:
                z_window[valid_mask] = (log_vols - local_mean) / local_std
            
            # Accumulate the calculated Z-scores into the sum map
            sum_map[y_start:y_end, x_start:x_end] += z_window
            
            # Only increment the count map for pixels that actually received a calculation
            count_map[y_start:y_end, x_start:x_end] += valid_mask.astype(np.int32)

    # NumPy will safely evaluate 0/0 to NaN for the untouched margins
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        output_map = sum_map / count_map
        
    # Explicitly enforce the spatial mask on the final output to clear out invalid pixels
    output_map[~global_valid_mask] = np.nan
    
    return output_map

def calculate_annular_z_score(volume_array, bg_window_size, guard_window_size, stride):
    """
    Calculates the local sliding-window Z-score for a frame of spectral complexity volumes.
    Uses an ensemble annular (dual-window) guard band approach to prevent signal swamping.
    """
    height, width = volume_array.shape
    
    sum_map = np.zeros((height, width), dtype=np.float32)
    count_map = np.zeros((height, width), dtype=np.int32)
    
    # Identify globally valid pixels (strictly positive for log transform)
    global_valid_mask = volume_array > 0.0
    
    if not np.any(global_valid_mask):
        warnings.warn("No valid volumes found in frame > 0. Returning NaNs.")
        return np.full((height, width), np.nan, dtype=np.float32)

    # Calculate guard window boundaries relative to the outer window
    center_idx = bg_window_size // 2
    g_half = guard_window_size // 2
    g_start = center_idx - g_half
    g_end = center_idx + g_half + 1

    for y_start in range(0, height - bg_window_size + 1, stride):
        for x_start in range(0, width - bg_window_size + 1, stride):
            y_end = y_start + bg_window_size
            x_end = x_start + bg_window_size
            
            window = volume_array[y_start:y_end, x_start:x_end]
            valid_mask = window > 0.0
            
            # Create the Annular (Donut) Background Mask
            bg_mask = valid_mask.copy()
            bg_mask[g_start:g_end, g_start:g_end] = False # Hollow out the guard area
            
            # We need enough background pixels to calculate a meaningful standard deviation
            if np.sum(bg_mask) < 5: 
                continue
            
            # Calculate background statistics ONLY on the outer ring
            bg_vols = window[bg_mask]
            log_bg_vols = np.log(bg_vols)
            
            local_mean = np.mean(log_bg_vols)
            local_std = np.std(log_bg_vols)
            
            if local_std <= 1e-6:
                continue # Perfectly flat background, cannot calculate Z-score
            
            # Create the Target Mask (Only score pixels inside the Guard Window)
            target_mask = np.zeros_like(valid_mask, dtype=bool)
            target_mask[g_start:g_end, g_start:g_end] = valid_mask[g_start:g_end, g_start:g_end]
            
            if not np.any(target_mask):
                continue # No valid pixels in the target area to score
            
            # Apply Z-score equation to target pixels based on outer background stats
            target_vols = window[target_mask]
            log_target_vols = np.log(target_vols)
            
            z_scores = (log_target_vols - local_mean) / local_std
            
            # Accumulate results exactly into the guard spatial locations
            z_window = np.zeros((bg_window_size, bg_window_size), dtype=np.float32)
            z_window[target_mask] = z_scores
            
            sum_map[y_start:y_end, x_start:x_end] += z_window
            count_map[y_start:y_end, x_start:x_end] += target_mask.astype(np.int32)

    # Average overlapping ensemble scores mimicking SpecComplex.py pattern
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        output_map = sum_map / count_map
        
    # Enforce global mask
    output_map[~global_valid_mask] = np.nan
    
    return output_map

def process_msd_sliding_tile(frame_data, tile_size, stride):
    """
    Calculates the Local Mean Spectral Distance (MSD) for a sliding window.
    Acts as a benchmark for local spectral heterogeneity.
    """
    bands, height, width = frame_data.shape
    img = np.transpose(frame_data, (1, 2, 0))
    
    sum_map = np.zeros((height, width), dtype=np.float32)
    count_map = np.zeros((height, width), dtype=np.int32)

    for y_start in range(0, height - tile_size + 1, stride):
        for x_start in range(0, width - tile_size + 1, stride):
            y_end, x_end = y_start + tile_size, x_start + tile_size
            
            # Extract window and reshape to (N_pixels, Bands)
            tile_cube = img[y_start:y_end, x_start:x_end, :]
            tile_2d = np.reshape(tile_cube, (-1, bands))
            
            # 1. Calculate the local mean spectral vector
            local_mean = np.nanmean(tile_2d, axis=0)
            
            # 2. Calculate the Euclidean distance of each pixel to the mean
            # np.linalg.norm with axis=1 computes distance for each row (pixel)
            distances = np.linalg.norm(tile_2d - local_mean, axis=1)
            
            # 3. Calculate the Mean Spectral Distance for the window
            msd_value = np.nanmean(distances)
            
            # 4. Assign to maps (applies spatial smoothing equivalent to sliding volume)
            sum_map[y_start:y_end, x_start:x_end] += msd_value
            count_map[y_start:y_end, x_start:x_end] += 1
            
    output_map = np.where(count_map > 0, sum_map / count_map, np.nan)
    return output_map


def calc_evi_frame(frame_data):
    """
    Calculates the Enhanced Vegetation Index (EVI) per pixel for a given frame.
    
    Formula: EVI = G * ((NIR - Red) / (NIR + C1 * Red - C2 * Blue + L))
    where G=2.5, C1=6, C2=7.5, L=1 (Reference: Huete et al., 2002).
    
    Assumes standard Landsat 8/9 Level-2 stacker band order:
    Index 1: Blue  (L8/9 Band 2)
    Index 3: Red   (L8/9 Band 4)
    Index 4: NIR   (L8/9 Band 5)
    """
    bands, height, width = frame_data.shape
    
    if bands < 5:
        warnings.warn("Insufficient bands to calculate EVI. Returning NaNs.")
        return np.full((height, width), np.nan, dtype=np.float32)
        
    blue = frame_data[1, :, :]
    red = frame_data[3, :, :]
    nir = frame_data[4, :, :]
    
    # Standard EVI coefficients (Huete et al., 2002)
    G = 2.5
    C1 = 6.0
    C2 = 7.5
    L = 1.0
    
    # Calculate denominator
    denominator = nir + (C1 * red) - (C2 * blue) + L
    
    # Safely calculate EVI avoiding division by zero
    evi = G * ((nir - red) / denominator)
    
    # Mask out infinity caused by extreme outliers or zeros
    evi[np.isinf(evi)] = np.nan
    return evi

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