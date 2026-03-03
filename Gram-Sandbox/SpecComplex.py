import numpy as np
import math
import warnings

'''
methods pulled from MaxD_Gram.py with minor updates
'''
def calcGramLocal(endmembers, mean_vector):
    """
    Calculates the Local Gram matrix.
    1. Subtracts the mean vector from all other endmembers (centering the simplex on x).
    2. Calculates the Gram matrix of these centered vectors.
    """
    # Reduce to current number of endmembers
    # Shape: (Bands, N)
    localized_vectors = endmembers - mean_vector[:, np.newaxis]

    # Calculate Gram Matrix
    # G = V^T * V
    gram = np.matmul(localized_vectors.T, localized_vectors)
    return gram

def maximumDistance(data, num_endmembers):
    '''
    Args:
        data (np.ndarray): 2D data [npixels, nbands]
        num_endmembers (int): number of endmembers to be calculated (choose more than expected to find)
    Returns:
        endmembers [bands, num_endmembers]
        endmembers_index [1, num_endmembers]
    '''
    # data = 2D data [npixels, nbands]
    # num_endmembers = number of endmembers to be calculated (choose more than expected to find)
    #print('---> In MaxD extracting endmembers and Grammian ...')
       
    # Ensure data is 2D [npixels, nbands]
    if data.ndim == 3:
        # Flatten 3D cube [rows, cols, bands] -> 2D [pixels, bands]
        image2D = np.reshape(data, (data.shape[0] * data.shape[1], data.shape[2]), order="F")
    else:
        image2D = data
    if np.min(data) < -1:
        warnings.warn('Data contains negative values')
        data = np.clip(data, 0, 2)
    if np.max(data) > 1:
        warnings.warn('Data contains values greater than 1')
        data = np.clip(data, 0, 1)

    # --- NaN Handling ---
    # Identify valid pixels (rows) that do not contain any NaN values
    valid_mask = ~np.isnan(image2D).any(axis=1)
    
    # Check if we have enough valid pixels
    if np.sum(valid_mask) < num_endmembers:
        print(f"Not enough valid pixels (no NaNs) to find {num_endmembers} endmembers. Found {np.sum(valid_mask)} valid pixels.")
        # Return empty/zero arrays with correct shape [bands, num]
        return np.zeros([image2D.shape[1], num_endmembers]), np.zeros([1, num_endmembers]), np.zeros([num_endmembers])

    # Filter data to keep only valid pixels
    valid_data = image2D[valid_mask]
    
    # Store original indices to map back later
    # valid_indices[i] contains the index in the original flattened image2D corresponding to the i-th row in valid_data
    valid_indices = np.where(valid_mask)[0]

    data = np.transpose(valid_data)
    if np.min(data) < -1:
        raise ValueError('Data contains negative values')

    # find data size
    num_bands = data.shape[0]
    num_pix = data.shape[1]

    # calculate magnitude of all vectors to find min and max
    magnitude = np.linalg.norm(data, axis=0)
    idx1 = np.argmax(magnitude)
    idx2 = np.argmin(magnitude)

    # create empty output arrays for endmembers
    endmembers = np.zeros([num_bands, num_endmembers])
    endmembers_index = np.zeros([1, num_endmembers])   

    # assign largest and smallest vector as first and second endmembers
    endmembers[:, 0] = np.transpose(data[:, idx1])
    endmembers[:, 1] = np.transpose(data[:, idx2])
    
    # Map back to original indices
    endmembers_index[0, 0] = valid_indices[idx1]
    endmembers_index[0, 1] = valid_indices[idx2]

    data_proj = np.matrix(data)
    identity_matrix = np.identity(num_bands)

    loop = np.arange(3, num_endmembers + 1)
    for i in loop:
        diff = []
        pseudo = []
        # calc difference between endmembers
        diff = np.matrix(data_proj[:, idx2] - data_proj[:, idx1])
        # caclualte pseudo inverse of difference vector
        pseudo = np.linalg.pinv(diff)
        data_proj = np.matmul((identity_matrix - np.matmul(diff, pseudo)), data_proj)

        idx1 = idx2
        # Optimize: avoid creating (bands x num_pix) matrix of ones
        # np.matmul(data_proj[:, idx2], np.ones([1, num_pix])) creates a huge matrix repeating the vector
        # We can just use broadcasting: data_proj[:, idx2] is (bands, 1), data_proj is (bands, num_pix)
        vec = data_proj[:, idx2] # Shape (bands, 1)
        # Ensure it's a column vector
        if vec.ndim == 1:
            vec = vec[:, np.newaxis]
            
        diff_new = np.sum(np.square(vec - data_proj), axis=0)

        # find the maximum distance for next endmember
        # np.argmax returns the index of the first occurrence of the maximum value
        idx2 = np.argmax(diff_new)

        # assign to endmember file
        endmembers[:, i - 1] = np.transpose(data[:, idx2])
        
        # Map back to original index
        endmembers_index[0, i - 1] = valid_indices[idx2]

    return endmembers, endmembers_index


def plot_endmember_locations(image_cube, rgb_indices, endmember_indices, endmembers):
    """
    Plots the spatial (row, col) location of each found endmember
    on top of a true-color representation of the original image.

    Args:
        image_cube (np.ndarray): The *original* [rows, cols, bands] image.
        rgb_indices (tuple): (r, g, b) indices for plotting the true color image.
        endmember_indices (np.ndarray): The 1D [numEndmembers] array of indices.
        endmembers (np.ndarray): The [bands, numEndmembers] array, used to check for validity.
    """
    print("\nGenerating endmember spatial location plot...")
    
    # --- MODIFICATION: Get cols from image_cube ---

    rows, cols, bands = image_cube.shape

    plt.figure(figsize=(12, 9))
    
    # 1. Create a normalized RGB image for display
    r_idx, g_idx, b_idx = rgb_indices    
    image_rgb = make_color_image(image_cube, r_idx, g_idx, b_idx)
    
    # 2. Display the RGB image
    plt.imshow(image_rgb)
    
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

def process_volume_frame(frame_data, num_endmembers, gram_type='general', valid_mask=None, norm_type=None):
    """
    Process the image to identify endmembers for the entire frame.
    Pixel Filtering: Only valid pixels are extracted into the 2D matrix.
    Returns the full volume curve, endmembers, and indices.
    """
    bands, height, width = frame_data.shape
    img = np.transpose(frame_data, (1, 2, 0))
    image2D = np.reshape(img, (height * width, bands))

    if valid_mask is not None:
        # Flatten mask and extract only valid spectral signatures
        flat_mask = valid_mask.flatten()
        image2D = image2D[flat_mask]
    else:
        print("No valid mask provided, assuming all pixels in frame are valid.")
        
    if image2D.shape[0] < num_endmembers:
        print("Not enough pixels to find endmembers")
        return np.zeros([bands, num_endmembers]), np.zeros([1, num_endmembers]), np.zeros([num_endmembers])

    endmembers, endmember_indices = maximumDistance(image2D, num_endmembers)
    volume = np.zeros(num_endmembers)
    if gram_type == 'datasetMean':
        print("Localizing Gram to dataset mean")
        meanVector = image2D.mean(axis=0)
        for i in range(1, num_endmembers):
            volume[i] = np.sqrt(np.abs(np.linalg.det(calcGramLocal(endmembers[:, 0:i],meanVector))))
    else:
        print("Localizing Gram to 0")
        for i in range(1, num_endmembers):
            volume[i] = np.sqrt(np.abs(np.linalg.det(calcGramLocal(endmembers[:, 0:i],np.zeros(bands)))))

    if norm_type == 'bandCount':
        print(f"Normalizing volume by sqrt({bands})")
        volume = volume / np.sqrt(bands)

    # Return full volume array (curve) instead of just the maximum
    return endmembers, endmember_indices, volume

def process_volume_tiles(frame_data, tile_size, num_endmembers, gram_type='general', valid_mask=None, norm_type=None):
    """
    Grid-based processing (Non-overlapping tiles).
    Strict Validity: Window is only processed if ALL pixels are valid.
    Any pixel that is part of an invalid tile is set to NaN.
    """
    bands, height, width = frame_data.shape
    img = np.transpose(frame_data, (1, 2, 0))
    output_map = np.full((height, width), np.nan, dtype=np.float32)
    if valid_mask is None:
        print("No valid mask provided, assuming all pixels in tiles are valid.")
        valid_mask = np.ones((height, width), dtype=bool)
    
    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            y_end, x_end = min(y + tile_size, height), min(x + tile_size, width)
            
            # Check mask for this tile
            tile_mask = valid_mask[y:y_end, x:x_end]
            
            # REQUIREMENT: Window must be 100% valid (no clouds/shadows/nodata)
            if not np.all(tile_mask):
                continue 
            
            chunk = img[y:y_end, x:x_end, :]
            chunk_2d = np.reshape(chunk, (-1, bands))
            
            if chunk_2d.shape[0] >= num_endmembers:
                meanVector = chunk_2d.mean(axis=0)
                volume = np.zeros(num_endmembers)
                endmembers, _ = maximumDistance(chunk_2d, num_endmembers)
                for i in range(2, num_endmembers):
                    if gram_type == 'datasetMean':
                        volume[i] = np.sqrt(np.abs(np.linalg.det(calcGramLocal(endmembers[:, 0:i],meanVector))))
                    else:
                        volume[i] = np.sqrt(np.abs(np.linalg.det(calcGramLocal(endmembers[:, 0:i],np.zeros(bands)))))

                if norm_type == 'bandCount':
                    output_map[y:y_end, x:x_end] = np.max(volume[2:])/np.sqrt(bands)
                else:
                    output_map[y:y_end, x:x_end] = np.max(volume[2:])
    
    # Explicitly enforce spatial mask on final output
    output_map[valid_mask == 0] = np.nan
    return output_map

def process_volume_sliding_tile(frame_data, tile_size, stride, num_endmembers, gram_type='general', valid_mask=None, norm_type=None):
    """
    Sliding window processing.
    Strict Validity: Window is only processed if ALL pixels are valid.
    Output is masked with NaN for any pixel identified as invalid.
    """
    bands, height, width = frame_data.shape
    img = np.transpose(frame_data, (1, 2, 0))
    
    sum_map = np.zeros((height, width), dtype=np.float32)
    count_map = np.zeros((height, width), dtype=np.int8)

    if valid_mask is None:
        print("No valid mask provided, assuming all pixels in tiles are valid.")
        valid_mask = np.ones((height, width), dtype=bool)
    
    for y_start in range(0, height - tile_size + 1, stride):
        for x_start in range(0, width - tile_size + 1, stride):
            y_end, x_end = y_start + tile_size, x_start + tile_size
            
            # Use valid mask to verify window integrity
            window_mask = valid_mask[y_start:y_end, x_start:x_end]
            
            if not np.all(window_mask):
                continue 

            tile_cube = img[y_start:y_end, x_start:x_end, :]
            tile_2d = np.reshape(tile_cube, (-1, bands))
            
            if tile_2d.shape[0] >= num_endmembers:
                meanVector = tile_2d.mean(axis=0)
                endmembers, _ = maximumDistance(tile_2d, num_endmembers)
                volume = np.zeros(num_endmembers)
                for i in range(2, num_endmembers):
                    if gram_type == 'datasetMean':
                        volume[i] = np.sqrt(np.abs(np.linalg.det(calcGramLocal(endmembers[:, 0:i],meanVector))))
                    else:
                        volume[i] = np.sqrt(np.abs(np.linalg.det(calcGramLocal(endmembers[:, 0:i],np.zeros(bands)))))
                vol_val = np.max(volume[2:])
                if norm_type == 'bandCount':
                    vol_val = vol_val/np.sqrt(bands)
                sum_map[y_start:y_end, x_start:x_end] += vol_val
                count_map[y_start:y_end, x_start:x_end] += 1
            
    # Finalize normalization
    output_map = np.full((height, width), np.nan, dtype=np.float32)
    valid_pixels = (count_map > 0)
    output_map[valid_pixels] = sum_map[valid_pixels] / count_map[valid_pixels]
    
    # Explicitly enforce spatial mask on final output
    output_map[valid_mask == 0] = np.nan
    return output_map