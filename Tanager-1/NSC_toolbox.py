import tkinter as tk
from tkinter import filedialog
import numpy as np
from skimage import exposure
import matplotlib.pyplot as plt


def linear_normalize_array(arr):
    '''Normalizes a numpy array to the range [0, 1] using linear scaling, ignoring NaNs.'''
    arr_min = np.nanmin(arr)
    arr_max = np.nanmax(arr)
    norm_arr = (arr - arr_min) / (arr_max - arr_min)
    return norm_arr.clip(0, 1)

def log_normalize_array(arr):
    '''Normalizes a numpy array to the range [0, 1] using logarithmic scaling, ignoring NaNs.'''
    log_arr = np.log1p(arr)
    log_min = np.nanmin(log_arr)
    log_max = np.nanmax(log_arr)
            
    normalized_log_arr = (log_arr - log_min) / (log_max - log_min)
    return normalized_log_arr.clip(0, 1)

def percentile_normalize_array(arr, lower_percentile=1, upper_percentile=99):
    '''Normalizes a numpy array to the range [0, 1] using percentiles, ignoring NaNs.'''
    p_low, p_high = np.nanpercentile(arr, (lower_percentile, upper_percentile))
    norm_arr = exposure.rescale_intensity(arr, in_range=(p_low, p_high), out_range=(0, 1))
    return norm_arr.clip(0, 1)

def make_color_image(image, b1, b2, b3):

    nrows = image.shape[0]
    ncols = image.shape[1]
    # extract the red, green, and blue bands and rescale them to 0-255 for display
    if image[:, :, b1].min() < 0:
        redimg = image[:, :, b1] - image[:, :, b1].min()
    else:
        redimg = image[:, :, b1]
    redmin, redmax = np.percentile(redimg, (1, 99))
    redimg = exposure.rescale_intensity(redimg, in_range=(redmin, redmax), out_range=(0, 255))
    redimg = redimg.reshape(nrows, ncols)
    # 
    if image[:, :, b2].min() < 0:
        greenimg = image[:, :, b2] - image[:, :, b2].min()
    else:
        greenimg = image[:, :, b2]
    greenmin, greenmax = np.percentile(greenimg, (1, 99))
    greenimg = exposure.rescale_intensity(greenimg, in_range=(greenmin, greenmax), out_range=(0, 255))
    greenimg = greenimg.reshape(nrows, ncols)
    #
    if image[:, :, b3].min() < 0:
        blueimg = image[:, :, b3] - image[:, :, b3].min()
    else:
        blueimg = image[:, :, b3]
    bluemin, bluemax = np.percentile(blueimg, (1, 99))
    blueimg = exposure.rescale_intensity(blueimg, in_range=(bluemin, bluemax), out_range=(0, 255))
    blueimg = blueimg.reshape(nrows, ncols)
    #
    # NOTE: displaying an RGB image imshow assumes indices [0,1,2] are [red, green, blue]
    #
    rgbimg = np.ndarray((nrows, ncols, 3), dtype=np.uint8)
    rgbimg[:, :, 2] = blueimg
    rgbimg[:, :, 1] = greenimg
    rgbimg[:, :, 0] = redimg
    return (rgbimg)

def prompt_for_file(title, filetypes, initial_dir):
    """
    Opens a graphical file selection dialog for the user.
    inputs:
        title (str): The title of the file dialog window.
        filetypes (list): A list of tuples specifying the file types to display.
        initial_dir (str): The initial directory to open in the file dialog.
    returns:
        str: The full path to the selected file, or None if no file is selected.    
    """
    try:
        root = tk.Tk()
        root.withdraw()  # Hide the main tkinter window
        filepath = filedialog.askopenfilename(
            title=title,
            filetypes=filetypes,
            initialdir=initial_dir
        )
        root.destroy()
        return filepath
    except Exception as e:
        print(f"Error opening file dialog (is tkinter available?): {e}")
        return None

def get_rgb_indices(header):
        """
        Finds the band indices closest to true color wavelengths.
        Args:
            header (dict): The ENVI header dictionary.
        Returns:
            tuple: (red_band_index, green_band_index, blue_band_index) as indices for RGB channels in the loaded image cube. 
        """
        if 'wavelength' not in header or not header['wavelength']:
            print("Warning: 'wavelength' not in header or is empty. Defaulting to bands 0, 1, 2.")
            return (0, 1, 2)
            
        wavelengths = np.array(header['wavelength'], dtype=float)
        units = header.get('wavelength units', 'Micrometers').lower()
        
        # Normalize target wavelengths to the header's units
        # Targets are in Micrometers
        targets_um = {'red': 0.65, 'green': 0.56, 'blue': 0.48}
        
        if 'nano' in units:
            # Convert targets to nanometers
            targets_wl = {k: v * 1000 for k, v in targets_um.items()}
        else:
            # Assume micrometers
            targets_wl = targets_um
            
        r_idx = int(np.argmin(np.abs(wavelengths - targets_wl['red'])))
        g_idx = int(np.argmin(np.abs(wavelengths - targets_wl['green'])))
        b_idx = int(np.argmin(np.abs(wavelengths - targets_wl['blue'])))

        print(f"Wavelength unit: {units}")
        print(f"  - Red   (target {targets_wl['red']}): Found band {r_idx} ({wavelengths[r_idx]})")
        print(f"  - Green (target {targets_wl['green']}): Found band {g_idx} ({wavelengths[g_idx]})")
        print(f"  - Blue  (target {targets_wl['blue']}): Found band {b_idx} ({wavelengths[b_idx]})")
        
        # Basic check to see if all indices are the same (e.g., in a 1-band file)
        if r_idx == g_idx and g_idx == b_idx:
            print("Warning: All RGB channels mapped to the same band. Display will be grayscale.")
            # Use this single band for all 3 channels
            return (r_idx, r_idx, r_idx)
            
        return (r_idx, g_idx, b_idx)

def plot_flattened_data_heatmap(image_cube, endmember_indices):
    """
    Plots the entire [num_pixels, bands] array as a heatmap and
    draws horizontal lines to indicate which rows were selected
    as endmembers.

    Args:
        pixel_data (np.ndarray): The [num_pixels, bands] array.
        endmember_indices (np.ndarray): The 1D [numEndmembers] array of indices.
        bands (int): The number of bands (for x-axis labeling).
    """
    print("\nGenerating flattened data heatmap plot...")
    plt.figure(figsize=(12, 6))
    rows, cols, bands = image_cube.shape
    pixel_data = image_cube.reshape((rows * cols, bands))
        
    # 1. Normalize the pixel data for better heatmap visibility
    # We normalize each *band* (column) independently
    p2, p98 = np.percentile(pixel_data, (2, 98), axis=0)
    # Handle non-varying bands (p2 == p98) to avoid division by zero
    scale = p98 - p2
    scale[scale == 0] = 1.0
    
    data_normalized = (pixel_data - p2) / scale
    data_normalized = np.clip(data_normalized, 0, 1)

    # 2. Display the heatmap
    plt.imshow(data_normalized, aspect='auto', cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Normalized Pixel Value')
    
    # 3. Draw horizontal lines for each endmember
    for i, idx in enumerate(endmember_indices):
        # Only draw if the index is valid (was found)
        if idx > 0:
            plt.axhline(y=idx, color='red', linestyle='--', 
                        linewidth=1.5, label=f'V[{i}]' if i < 3 else None)
            
            # Annotate the index
            plt.text(bands, idx, f' V[{i}] (pixel {idx})', 
                     color='red', ha='left', va='center', fontweight='bold')

    plt.title('Endmember Locations in Flattened `pixel_data` Array')
    plt.xlabel('Band Number')
    plt.ylabel('Pixel Index (0 to num_pixels-1)')
    plt.xticks(np.arange(0, bands, max(1, bands // 10))) # Ticks for bands
    
    # Only show legend if we have valid endmembers
    if np.any(endmember_indices > 0):
        plt.legend(handles=[
            plt.Line2D([0], [0], color='r', linestyle='--', 
                       label='Endmember Row')
        ], loc='upper left')



def maximumDistanceVolumeProjection(image_cube, numEndmembers):
    """
    Implements the Maximum Distance Method (MDM) from Lee (2003) to find endmembers 
    and calculates the iterative parallelotope volume for each set of endmembers 
    using the SVE method's *standard* (origin-rooted) Gram matrix.

    This function's logic is aligned with the implementation found in 
    the "MaxD_Gram.py" library, which is a faithful implementation of 
    the algorithm described by Lee (2003).

    Args:
        image_cube (np.ndarray): A 3D array shaped as [rows, cols, bands] 
                                 representing the hyperspectral image data.
        numEndmembers (int): The desired number of endmembers to extract (must be >= 2).

    Returns:
        tuple: A tuple containing:
            - endmembers (np.ndarray): A 2D array shaped as [bands, numEndmembers] 
                                       containing the extracted endmembers.
            - volumes (np.ndarray): A 1D array of length numEndmembers. volumes[k] 
                                    holds the parallelotope hypervolume calculated 
                                    using the first (k+1) endmembers (V[0]...V[k]).
                                    volumes[0] is 0. volumes[1] is the first calculated
                                    area.
            - endmember_indices (np.ndarray): A 1D array of length numEndmembers
                                              containing the 1D index of each endmember.
    """
    
    # --- 0. Initialization ---
    try:
        rows, cols, bands = image_cube.shape
    except ValueError as e:
        raise ValueError(f"image_cube must be a 3D numpy array. Got shape {image_cube.shape}") from e

    if numEndmembers < 2:
        raise ValueError("numEndmembers must be at least 2 to perform MDM.")

    num_pixels = rows * cols
    
    # Reshape data to [num_pixels, bands]
    pixel_data = image_cube.reshape((num_pixels, bands)).astype(np.float64)
    
    # --- REFACTOR: Align with MaxD_Gram.py ---
    # Transpose data to [bands, num_pixels]
    data = pixel_data.T
    # data_proj is the matrix that will be iteratively projected
    data_proj = data.copy()
    
    # We store endmembers as [bands, numEndmembers]
    endmembers_list = np.zeros((bands, numEndmembers), dtype=np.float64) 
    volumes = np.zeros(numEndmembers, dtype=np.float64)
    endmember_indices = np.zeros(numEndmembers, dtype=int)
    # --- END REFACTOR ---

    # --- 1. MDM Step 1: Find First Endmember (V[0]) ---
    # Calculate the L2 norm for every pixel vector
    magnitude = np.sum(np.square(data), axis=0)
    
    # Find the pixel with the maximum norm [cite: Lee, 2003]
    idx1 = np.argmax(magnitude) # This will be our "from" index
    endmembers_list[:, 0] = data[:, idx1]
    endmember_indices[0] = idx1
    
    # --- 2. MDM Step 2: Find Second Endmember (V[1]) ---
    # Find the pixel with the minimum norm [cite: Lee, 2003]
    idx2 = np.argmin(magnitude) # This will be our "to" index
    endmembers_list[:, 1] = data[:, idx2]
    endmember_indices[1] = idx2
    
    # --- 3. MDM Iteration: Find V[2] through V[numEndmembers-1] ---
    
    # --- SVE: Calculate Volume for k=1 (First calculation) ---
    # Get the set of endmembers found so far {V[0], V[1]}
    # We must transpose to [k+1, bands] for the Gram matrix G = V @ V.T
    current_endmember_set_k1 = endmembers_list[:, 0:2].T # Shape [2, bands]
    G_standard_k1 = current_endmember_set_k1 @ current_endmember_set_k1.T # Shape [2, 2]
    sign_k1, logdet_k1 = np.linalg.slogdet(G_standard_k1)
    if sign_k1 > 0:
        volumes[1] = np.exp(logdet_k1 / 2.0)
    
    # Handle case where max and min norm pixels are the same
    if idx1 == idx2:
         print("Warning: First two endmembers (max/min norm) are identical. "
               "MDM (Lee) cannot proceed further. Returning full-size arrays "
               "containing only 1 endmember.")
         return endmembers_list, volumes, endmember_indices

    # Loop to find the rest of the endmembers
    for k in range(2, numEndmembers):
        
        # --- 3a. MDM: Project data ---
        # Get the *projected* vectors for the last two endmember indices
        v_idx1 = data_proj[:, idx1]
        v_idx2 = data_proj[:, idx2]

        # Create the difference vector *in the projected space*
        diff = v_idx2 - v_idx1
        d_dot = np.dot(diff, diff)

        # Handle potential division by zero
        if d_dot < 1e-9:
            print(f"Warning: Projection collapsed at endmember {k}. "
                  f"Endmember vectors are linearly dependent. "
                  f"Stopping search and returning {numEndmembers}-length arrays. "
                  f"Found {k} endmembers.")
            break # Exit the loop

        # Calculate the projection matrix P_k
        diff_col = diff.reshape(-1, 1) # Shape [bands, 1]
        pseudo = np.linalg.pinv(diff_col) # Shape [1, bands]
        P = np.identity(bands, dtype=np.float64) - (diff_col @ pseudo)
        
        # Project the *entire* data matrix into the new subspace
        data_proj = P @ data_proj
        
        # --- 3b. MDM: Find next endmember ---
        # The "from" point for the next distance search is the
        # projection of the endmember we just found
        idx1 = idx2 
        current_projected_vector = data_proj[:, idx1] # Shape [bands,]

        # Calculate squared distances from this point to all other points
        # Use broadcasting (more efficient than np.ones/matmul)
        # diffs = data_proj - current_projected_vector.reshape(-1, 1)
        # distances_sq = np.sum(np.square(diffs), axis=0)
        
        # Use einsum for high performance
        diffs = data_proj - current_projected_vector.reshape(-1, 1)
        distances_sq = np.einsum('ij,ij->j', diffs, diffs) # sum over axis 0

        # Find the index of the pixel with the maximum distance
        idx2 = np.argmax(distances_sq)

        # --- 3c. Store Endmember ---
        # This ensures we get real, non-negative pixel values.
        endmembers_list[:, k] = data[:, idx2]
        endmember_indices[k] = idx2
        
        # --- 3d. SVE: Calculate Volume (Standard Gram Matrix) ---
        # Get the set of all endmembers found so far {V[0], ..., V[k]}
        current_endmember_set = endmembers_list[:, 0:k+1].T # Shape [k+1, bands]
        
        # Calculate the standard (origin-rooted) Gram matrix G = V @ V.T
        G_standard = current_endmember_set @ current_endmember_set.T # Shape [k+1, k+1]
        
        sign, logdet = np.linalg.slogdet(G_standard)
        
        if sign > 0:
            volumes[k] = np.exp(logdet / 2.0)
        else:
            volumes[k] = 0.0

    # --- 4. Final Return ---
    # Return [bands, numEndmembers]
    return endmembers_list, volumes, endmember_indices

def maximumDistanceVolumeGeometric(image_cube, numEndmembers, stop_threshold=1e-9):
    """
    Implements the Maximum Distance Analysis (MDA) from Tao et al. (2021)
    to find endmembers and calculates the iterative parallelotope volume 
    using the *standard* (origin-rooted) Gram matrix.

    This function implements the geometric distance-based methodology:
    1. Tao, X., et al. (2021). "Endmember Estimation with Maximum Distance Analysis."
       (Provides the geometric distance method for endmember selection).
    2. Ziemann, A. (2010). "Using n-dimensional volumes for mathematical applications 
       in spectral image analysis." (Provides the standard Gram matrix definition
       for parallelotope volume calculation [Chapter 3]).

    Args:
        image_cube (np.ndarray): A 3D array shaped as [rows, cols, bands].
        numEndmembers (int): The *maximum* desired number of endmembers.
        stop_threshold (float): The stopping criterion from Tao et al. (2021).
                                 Search stops if max distance is below this.

    Returns:
        tuple: A tuple containing:
            - endmembers (np.ndarray): [bands, numEndmembers] array of found endmembers.
            - volumes (np.ndarray): [numEndmembers] array of iterative volumes.
            - endmember_indices (np.ndarray): [numEndmembers] 1D array of indices.
    """
    
    # --- 0. Initialization ---
    try:
        rows, cols, bands = image_cube.shape
    except ValueError as e:
        raise ValueError(f"image_cube must be a 3D numpy array. Got shape {image_cube.shape}") from e

    if numEndmembers < 2:
        raise ValueError("numEndmembers must be at least 2.")

    num_pixels = rows * cols
    pixel_data = image_cube.reshape((num_pixels, bands)).astype(np.float64)
    
    endmembers_list = np.zeros((numEndmembers, bands), dtype=np.float64) 
    volumes = np.zeros(numEndmembers, dtype=np.float64)
    endmember_indices = np.zeros(numEndmembers, dtype=int)
    
    # --- 1. Tao Def. 1: Find First Endmember (V[0]) ---
    # "farthest pixel point from the coordinate origin"
    norms = np.linalg.norm(pixel_data, axis=1)
    max_norm_index = np.argmax(norms)
    V_0 = pixel_data[max_norm_index, :]
    endmembers_list[0, :] = V_0
    endmember_indices[0] = max_norm_index
    
    # --- 2. Tao Def. 2: Find Second Endmember (V[1]) ---
    # "farthest pixel point from the first extracted endmember"
    diffs_v0 = pixel_data - V_0
    distances_sq_v0 = np.einsum('ij,ij->i', diffs_v0, diffs_v0)
    max_dist_index = np.argmax(distances_sq_v0)
    V_1 = pixel_data[max_dist_index, :]
    endmembers_list[1, :] = V_1
    endmember_indices[1] = max_dist_index
    
    # --- SVE: Calculate Volume for k=1 (First calculation) ---
    current_endmember_set_k1 = endmembers_list[0:2, :] 
    G_standard_k1 = current_endmember_set_k1 @ current_endmember_set_k1.T
    sign_k1, logdet_k1 = np.linalg.slogdet(G_standard_k1)
    if sign_k1 > 0:
        volumes[1] = np.exp(logdet_k1 / 2.0)
    
    # --- 3. Tao Iteration: Find V[2] through V[numEndmembers-1] ---
    for k in range(2, numEndmembers):
        
        # --- 3a. Tao: Find next endmember ---
        max_dist_sq = -1.0
        next_endmember_index = -1
        
        if k == 2:
            # --- Tao Def. 3: Farthest from line L(V0, V1) ---
            # Use vector rejection: d = || (p-B) - proj_u(p-B) ||
            # A = V0, B = V1
            A = endmembers_list[0, :]
            B = endmembers_list[1, :]
            u = A - B # Vector defining the line
            u_dot_u = np.dot(u, u)
            
            if u_dot_u < 1e-9:
                print(f"Warning: V[0] and V[1] are identical. "
                      f"Tao (MDA) cannot proceed. Returning {numEndmembers}-length arrays.")
                break
                
            # v_vectors = all pixels relative to B
            v_vectors = pixel_data - B # Shape [num_pixels, bands]
            # v_dot_u = projection scalar for each pixel
            v_dot_u = v_vectors @ u # Shape [num_pixels]
            
            # proj = ( (v.u) / (u.u) ) * u
            projections = np.outer(v_dot_u / u_dot_u, u) # Shape [num_pixels, bands]
            rejections = v_vectors - projections
            
            distances_sq = np.einsum('ij,ij->i', rejections, rejections)
            max_dist_sq = np.max(distances_sq)
            next_endmember_index = np.argmax(distances_sq)

        else:
            # --- Tao Def. 5: Farthest from affine hull Aff(V0...Vk-1) ---
            # Use stable QR decomposition to find rejection distance
            # A = V0 (our affine hull origin)
            A = endmembers_list[0, :]
            # U = [V1-A, V2-A, ... Vk-1-A]
            # Transpose to get [bands, k-1]
            U = (endmembers_list[1:k, :] - A).T 
            
            # V_T = [p1-A, p2-A, ... pN-A]
            # Transpose to get [bands, num_pixels]
            V_T = (pixel_data - A).T
            
            # Get orthonormal basis Q for the subspace U
            try:
                Q, _ = np.linalg.qr(U)
            except np.linalg.LinAlgError:
                 print(f"Warning: QR decomposition failed at k={k}. "
                       f"Endmembers are likely linearly dependent. "
                       f"Stopping search.")
                 break

            # Projection matrix onto the affine hull's subspace
            P_proj = Q @ Q.T
            
            # Project all pixel vectors (V_T) onto the hull
            V_T_proj = P_proj @ V_T
            
            # Get the rejection (orthogonal) vectors
            V_T_rej = V_T - V_T_proj
            
            # Find the squared distances
            distances_sq = np.einsum('ij,ij->j', V_T_rej, V_T_rej)
            max_dist_sq = np.max(distances_sq)
            next_endmember_index = np.argmax(distances_sq)

        # --- 3b. Tao Def. 6: Stopping Criterion ---
        # "maximum distance... is zero" (or below our noise threshold)
        if max_dist_sq < stop_threshold:
            print(f"\nStopping at k={k} (Tao Def. 6): "
                  f"Max distance to affine hull ({np.sqrt(max_dist_sq):.2e}) "
                  f"is below threshold ({np.sqrt(stop_threshold):.2e}).")
            print(f"Found {k} endmembers.")
            break # Exit the loop
            
        # --- 3c. Store the new endmember ---
        endmembers_list[k, :] = pixel_data[next_endmember_index, :]
        endmember_indices[k] = next_endmember_index
        
        # --- 3d. SVE: Calculate Volume (Standard Gram Matrix) ---
        current_endmember_set = endmembers_list[0:k+1, :]
        G_standard = current_endmember_set @ current_endmember_set.T
        sign, logdet = np.linalg.slogdet(G_standard)
        
        if sign > 0:
            volumes[k] = np.exp(logdet / 2.0)
        else:
            volumes[k] = 0.0

    # --- 4. Final Return ---
    # Transpose endmembers to match output spec [bands, numEndmembers]
    return endmembers_list.T, volumes, endmember_indices


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
    
def plot_band_scatter(image_cube, endmembers, band_x_idx, band_y_idx):
    """
    Plots all pixels as a 2D scatter plot using two selected bands.
    This shows the geometric relationship and how endmembers form the
    "corners" (simplex) of the pixel cloud.

    Args:
        pixel_data (np.ndarray): The [num_pixels, bands] array of all pixels.
        endmembers (np.ndarray): The [bands, numEndmembers] array.
        band_x_idx (int): The 0-based index for the x-axis band.
        band_y_idx (int): The 0-based index for the y-axis band.
    """
    rows, cols, bands = image_cube.shape
    pixel_data = image_cube.reshape((rows * cols, bands))
    print(f"\nGenerating band scatter plot (Band {band_x_idx+1} vs Band {band_y_idx+1})...")
    plt.figure(figsize=(9, 9))
    
    # Plot all pixels
    max_x = np.max(pixel_data[:, band_x_idx])
    max_y = np.max(pixel_data[:, band_y_idx])
    plt.scatter(pixel_data[:, band_x_idx]/max_x, pixel_data[:, band_y_idx]/max_y, 
                alpha=0.1, c='gray', label='All Image Pixels')
    
    # Get the endmember values for the two selected bands
    em_x = endmembers[band_x_idx, :]
    em_y = endmembers[band_y_idx, :]
    
    # Plot endmembers
    plt.scatter(em_x, em_y, s=200, c='red', edgecolor='black', 
                label='Found Endmembers', zorder=10)
    
    # Annotate endmembers
    for i in range(endmembers.shape[1]):
        if np.any(endmembers[:, i] != 0):
            plt.annotate(f'V[{i}]', (em_x[i], em_y[i]), 
                         textcoords="offset points", xytext=(0,10), 
                         ha='center', fontsize=12, fontweight='bold')
    
    plt.title(f'Endmember Positions in Spectral Space (Bands {band_x_idx+1} vs {band_y_idx+1})')
    plt.xlabel(f'Band {band_x_idx+1} Value')
    plt.ylabel(f'Band {band_y_idx+1} Value')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.gca().set_aspect('equal', adjustable='box')

def plot_3d_band_scatter(image_cube, endmembers, band_indices=(0, 1, 2)):
    """
    Plots all pixels as a 3D scatter plot using three selected bands.
    This shows the 3D geometric relationship of the pixel cloud.

    Args:
        pixel_data (np.ndarray): The [num_pixels, bands] array of all pixels.
        endmembers (np.ndarray): The [bands, numEndmembers] array.
        band_indices (tuple): A tuple of 3 integers for the (x, y, z) axes.
    """
    b_x_idx, b_y_idx, b_z_idx = band_indices
    print(f"\nGenerating 3D band scatter plot (Bands {b_x_idx+1}, {b_y_idx+1}, {b_z_idx+1})...")

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    rows, cols, bands = image_cube.shape
    pixel_data = image_cube.reshape((rows * cols, bands))
    

    # 1. Plot all pixels
    ax.scatter(pixel_data[:, b_x_idx], 
               pixel_data[:, b_y_idx], 
               pixel_data[:, b_z_idx], 
               alpha=0.05, c='gray', label='All Image Pixels', s=1)

    # 2. Get the endmember values for the three selected bands
    em_x = endmembers[b_x_idx, :]
    em_y = endmembers[b_y_idx, :]
    em_z = endmembers[b_z_idx, :]

    # 3. Plot endmembers and their vectors from origin
    for i in range(endmembers.shape[1]):
        # Check if this endmember was actually found (is not all zeros)
        if np.any(endmembers[:, i] != 0):
            # Get coordinates for this endmember
            x, y, z = em_x[i], em_y[i], em_z[i]

            # --- Plot the endmember POINT ---
            # Only add label for the first point to avoid duplicate legend entries
            ax.scatter(x, y, z, s=200, c='red', edgecolor='black', 
                       zorder=10, depthshade=False,
                       label='Endmember Point' if i == 0 else "")

            # --- NEW: Plot the vector LINE from origin ---
            ax.plot([0, x], [0, y], [0, z], color='red', linestyle='--', 
                    linewidth=1.5, zorder=5,
                    label='Endmember Vector' if i == 0 else "")

            # --- Plot the ANNOTATION ---
            ax.text(x, y, z, f' V[{i}]', 
                    color='black', fontsize=12, fontweight='bold')

    ax.set_title('Endmember Positions and Vectors in 3D Spectral Space')
    ax.set_xlabel(f'Band {b_x_idx+1} Value')
    ax.set_ylabel(f'Band {b_y_idx+1} Value')
    ax.set_zlabel(f'Band {b_z_idx+1} Value')
    ax.legend()
    # Set origin point for clarity
    ax.plot([0], [0], [0], 'ko', markersize=5) 
    ax.legend()

'''
methods pulled from MaxD_Gram.py with minor updates
'''
def maximumDistance(data, num, mnf_data=0, gram='general'):
    # data = 2D data [npixels, nbands]
    # num = number of endmembers to be calculated (choose more than expected to find)
    # if MNF data is not available, code will assign img as mnf_data
    #print('---> In MaxD extracting endmembers and Grammian ...')
       
    # Ensure data is 2D [npixels, nbands]
    if data.ndim == 3:
        # Flatten 3D cube [rows, cols, bands] -> 2D [pixels, bands]
        image2D = np.reshape(data, (data.shape[0] * data.shape[1], data.shape[2]), order="F")
    else:
        image2D = data
    if np.min(data) < -1:
        raise ValueError('Data contains negative values')

    # --- NaN Handling ---
    # Identify valid pixels (rows) that do not contain any NaN values
    valid_mask = ~np.isnan(image2D).any(axis=1)
    
    # Check if we have enough valid pixels
    if np.sum(valid_mask) < num:
        print(f"Not enough valid pixels (no NaNs) to find {num} endmembers. Found {np.sum(valid_mask)} valid pixels.")
        # Return empty/zero arrays with correct shape [bands, num]
        return np.zeros([image2D.shape[1], num]), np.zeros([1, num]), np.zeros([num])

    # Filter data to keep only valid pixels
    valid_data = image2D[valid_mask]
    
    # Store original indices to map back later
    # valid_indices[i] contains the index in the original flattened image2D corresponding to the i-th row in valid_data
    valid_indices = np.where(valid_mask)[0]
    
    if mnf_data == 0:
        mnf_data = valid_data
    else:
        # If mnf_data was provided, we must reshape and filter it exactly the same way
        mnf_2D = np.reshape(mnf_data, (mnf_data.shape[0] * mnf_data.shape[1], mnf_data.shape[2]), order="F")
        mnf_data = mnf_2D[valid_mask]

    data = np.transpose(valid_data)
    data2 = np.transpose(mnf_data)

    # find data size
    num_bands = data.shape[0]
    num_pix = data.shape[1]

    # calculate magnitude of all vectors to find min and max
    magnitude = np.sum(np.square(data), axis=0)
    idx1 = np.argmax(magnitude)
    idx2 = np.argmin(magnitude)

    # create empty output arrays for endmembers
    endmembers = np.zeros([num_bands, num])
    endmembers_index = np.zeros([1, num])

    # assign largest and smallest vector as first and second endmembers
    endmembers[:, 0] = np.transpose(data[:, idx1])
    endmembers[:, 1] = np.transpose(data[:, idx2])
    
    # Map back to original indices
    endmembers_index[0, 0] = valid_indices[idx1]
    endmembers_index[0, 1] = valid_indices[idx2]

    data_proj = np.matrix(data2)
    identity_matrix = np.identity(num_bands)

    # create array for volume of determinant of Gram matrix
    volume = np.zeros([num])

    loop = np.arange(3, num + 1)
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

        # find ne maximum distance for next endmember
        idx2 = np.int_(np.where(diff_new == np.max(diff_new))[1])

        #print('DEBUG: idx2: ', idx2,np.size(idx2))
        ###
        # DWM: looks like there may be cases where idx2 has more than one element, i.e., there are
        # two elements of diff_new that are equal to the max.  In that case, just grab the first one
        ###
        if np.size(idx2) > 1:
            idx2 = idx2[0]

        # assign to endmember file
        endmembers[:, i - 1] = np.transpose(data[:, idx2])
        
        # Map back to original index
        endmembers_index[0, i - 1] = valid_indices[idx2]

        if gram == 'local':
            # calculate local gram matrix
            loc_gram = calcGramLocal(endmembers, i)
            volume[i - 1] = np.sqrt(np.abs(np.linalg.det(loc_gram)))

        elif gram == 'general':
            # calculate general gram matrix
            gen_gram = calcGramGeneral(endmembers[:, 0:i])
            volume[i - 1] = np.sqrt(np.abs(np.linalg.det(gen_gram)))

    return endmembers, endmembers_index, volume


def calcGramGeneral(data_endmembers):
    # calculate gram matrix = V^T * V
    gram = np.matmul(np.transpose(data_endmembers), data_endmembers)

    return gram


def calcGramLocal(data_endmembers, iteration):
    # use only endmembers already calculated
    data_endmembers = data_endmembers[:, 0:iteration]

    # calculate the Gram matrix based on local information (points nearest to mean)
    # num_bands = data_endmembers.shape[0]
    num_pix = data_endmembers.shape[1]

    # create mean vector
    mean_spec = np.mean(data_endmembers, axis=1)

    # calculate normalized difference between mean vector and endmembers and find closest vector to mean vector
    diffdist = np.linalg.norm(np.transpose(np.matlib.repmat(mean_spec, num_pix, 1)) - data_endmembers, axis=0)
    min_idx = np.argmin(diffdist)

    # create index of rows to keep
    index = np.ones([num_pix])
    # keep all but min distance one
    index[min_idx] = 0
    # find index of all nonzero entires
    keep_idx = np.squeeze(np.where(index == 1))
    nearpix = data_endmembers[:, keep_idx]

    # calculate local Gram
    num_neighbors = nearpix.shape[1]
    # gram = np.zeros([num_neighbors, num_neighbors])
    diff_matrix = nearpix - np.transpose(np.matlib.repmat(mean_spec, num_neighbors, 1))

    gram = np.matmul(np.transpose(diff_matrix), diff_matrix)
    #print('<--- done')

    return gram

def maximumDistanceNormalized(data, num, mnf_data=0, gram='general'):
    # data = 2D data [npixels, nbands]
    # num = number of endmembers to be calculated (choose more than expected to find)
    # if MNF data is not available, code will assign img as mnf_data
    #print('---> In MaxD extracting endmembers and Grammian ...')
       
    # Ensure data is 2D [npixels, nbands]
    if data.ndim == 3:
        # Flatten 3D cube [rows, cols, bands] -> 2D [pixels, bands]
        image2D = np.reshape(data, (data.shape[0] * data.shape[1], data.shape[2]), order="F")
    else:
        image2D = data
    if np.min(data) < -1:
        raise ValueError('Data contains negative values')

    # --- NaN Handling ---
    # Identify valid pixels (rows) that do not contain any NaN values
    valid_mask = ~np.isnan(image2D).any(axis=1)
    
    # Check if we have enough valid pixels
    if np.sum(valid_mask) < num:
        print(f"Not enough valid pixels (no NaNs) to find {num} endmembers. Found {np.sum(valid_mask)} valid pixels.")
        # Return empty/zero arrays with correct shape [bands, num]
        return np.zeros([image2D.shape[1], num]), np.zeros([1, num]), np.zeros([num])

    # Filter data to keep only valid pixels
    valid_data = image2D[valid_mask]
    
    # Store original indices to map back later
    # valid_indices[i] contains the index in the original flattened image2D corresponding to the i-th row in valid_data
    valid_indices = np.where(valid_mask)[0]
    
    if mnf_data == 0:
        mnf_data = valid_data
    else:
        # If mnf_data was provided, we must reshape and filter it exactly the same way
        mnf_2D = np.reshape(mnf_data, (mnf_data.shape[0] * mnf_data.shape[1], mnf_data.shape[2]), order="F")
        mnf_data = mnf_2D[valid_mask]

    data = np.transpose(valid_data)
    data2 = np.transpose(mnf_data)
    vec_norms = np.linalg.norm(data, axis=0)
    vec_norms[vec_norms == 0] = 1
    

    # find data size
    num_bands = data.shape[0]
    num_pix = data.shape[1]

    # calculate magnitude of all vectors to find min and max
    magnitude = np.sum(np.square(data), axis=0)
    idx1 = np.argmax(magnitude)
    idx2 = np.argmin(magnitude)

    # create empty output arrays for endmembers
    endmembers = np.zeros([num_bands, num])
    endmembers_index = np.zeros([1, num])

    # normalize pixel vectors
    data = data / vec_norms
    data2 = data2 / vec_norms
    if np.min(data) < -1:
        raise ValueError('Data contains negative values')

    # assign largest and smallest vector as first and second endmembers
    endmembers[:, 0] = np.transpose(data[:, idx1])
    endmembers[:, 1] = np.transpose(data[:, idx2])
    
    # Map back to original indices
    endmembers_index[0, 0] = valid_indices[idx1]
    endmembers_index[0, 1] = valid_indices[idx2]

    data_proj = np.matrix(data2)
    identity_matrix = np.identity(num_bands)

    # create array for volume of determinant of Gram matrix
    volume = np.zeros([num])

    loop = np.arange(3, num + 1)
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

        # find ne maximum distance for next endmember
        idx2 = np.int_(np.where(diff_new == np.max(diff_new))[1])

        #print('DEBUG: idx2: ', idx2,np.size(idx2))
        ###
        # DWM: looks like there may be cases where idx2 has more than one element, i.e., there are
        # two elements of diff_new that are equal to the max.  In that case, just grab the first one
        ###
        if np.size(idx2) > 1:
            idx2 = idx2[0]

        # assign to endmember file
        endmembers[:, i - 1] = np.transpose(data[:, idx2])
        
        # Map back to original index
        endmembers_index[0, i - 1] = valid_indices[idx2]

        if gram == 'local':
            # calculate local gram matrix
            loc_gram = calcGramLocal(endmembers, i)
            volume[i - 1] = np.sqrt(np.abs(np.linalg.det(loc_gram)))

        elif gram == 'general':
            # calculate general gram matrix
            gen_gram = calcGramGeneral(endmembers[:, 0:i])
            volume[i - 1] = np.sqrt(np.abs(np.linalg.det(gen_gram)))

    return endmembers, endmembers_index, volume