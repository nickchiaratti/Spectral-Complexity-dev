import numpy as np


def maximumDistance_parallelotopeVolumes(pixels, num_endmembers):
    """
    Combined endmember extraction and Gram volume estimation using the maximum distance (MaxD) method with iterative orthogonal projection.

    Localizes the pixel neighborhood by subtracting the minimum-norm pixel (the local origin), 
    then iteratively selects endmembers by largest orthogonal projection distance while recording the heights needed for the volume curve.

    Args:
        pixels (np.ndarray): (bands, pixels) or (height, width, bands) array of x pixel spectra.
        num_endmembers (int): The number of r endmembers to extract.
    Returns:
        endmembers (np.ndarray): (bands, num_endmembers) array of endmember spectra v_r.
        volumes (np.ndarray): (num_endmembers,) array of ln(Vol_r).
        heights (np.ndarray): (num_endmembers,) array of heights h_r.
        indices (np.ndarray): (num_endmembers,) array of indices i_r of the extracted endmembers. Used to identify endmember selection order in source pixel set. 
    """
    if pixels.ndim == 3:
        h, w, num_bands = pixels.shape
        pixels = np.reshape(pixels, (h * w, num_bands), order="F").astype(np.float64).T
    elif pixels.ndim == 2:
        pixels = pixels.astype(np.float64)

    num_bands, num_pixels = pixels.shape

    # Initialize Variables
    endmembers = np.zeros((num_bands, num_endmembers), dtype=pixels.dtype)
    volumes = np.full(num_endmembers, -np.inf, dtype=np.float64)
    heights = np.zeros(num_endmembers, dtype=np.float64)
    indices = np.zeros(num_endmembers, dtype=int)

    # Initialization
    # 𝒗_0=arg⁡𝑚𝑖𝑛‖𝒙‖
    # 𝒗_1=arg⁡𝑚𝑎𝑥‖𝒙‖
    magnitudes = np.linalg.norm(pixels, axis=0)
    idx_maxnorm = np.argmax(magnitudes)
    idx_minnorm = np.argmin(magnitudes)
    endmembers[:, 0] = pixels[:, idx_maxnorm]
    endmembers[:, 1] = pixels[:, idx_minnorm]
    indices[0] = idx_maxnorm
    indices[1] = idx_minnorm
    selected_indices = [idx_minnorm, idx_maxnorm]

    # Shift all pixel spectra to a localized origin at 𝒗_0 to form the initial residual vectors 𝒓_1
    # 𝒓_1 (𝒙)=𝒙−𝒗_0
    origin = pixels[:, idx_minnorm].reshape(-1, 1)    # (bands, 1)
    residuals = pixels - origin                        # (bands, num_pixels)

    # Calculate the first direction vector 𝒅_1, height ℎ_1, and orthonormal basis 𝒒_1
    # 𝒅_1=𝒗_1−𝒗_0
    # ℎ_1=‖𝒅_1 ‖
    # 𝒒_1=𝒅_1/ℎ_1  
    d1 = residuals[:, idx_maxnorm].copy()              # (bands, 1)
    h1 = np.linalg.norm(d1)
    volumes[1] = np.log(h1)
    heights[1] = h1
    q1 = d1 / h1  # (bands,1)

    # Remove the component along the latest orthonormal direction 𝒒_(𝑗−1) from all pixel residuals
    # 𝒓_𝑗 (𝒙)=𝒓_(𝑗−1)−(𝒒_(𝑗−1)^𝑇 𝒓_(𝑗−1)) 𝒒_(𝑗−1)
    proj_component = np.outer(q1, np.dot(q1, residuals)) 
    residuals -= proj_component

    # The residual vectors are deflated across subsequent iterations 𝑗=2,…,𝑟 to select 𝑟 endmembers
    for j in range(2, num_endmembers):
        # Select the next endmember 𝒗_𝑗 as the pixel with maximum magnitude in the current residual subspace 
        # excluding the indices for 𝒗_0,𝒗_1,𝒗_2,…𝒗_(𝑗−1)
        # 𝒗_𝑗=arg𝑚𝑎𝑥‖𝒓_𝑗 (𝒙)‖
        residual_norm = np.linalg.norm(residuals, axis=0)  # (num_pixels,)
        residual_norm[selected_indices] = -np.inf
        idx_new = np.argmax(residual_norm)
        v_j = residuals[:, idx_new].copy()
        endmembers[:, j] = pixels[:, idx_new]
        indices[j] = idx_new
        selected_indices.append(idx_new)

        # Calculate the subsequent height ℎ_𝑗 and orthonormal basis 𝒒_𝑗
        # ℎ_𝑗=‖𝒓_𝑗 (𝒗_𝑗 )‖
        # 𝒒_𝒋=(𝒓_𝑗 (𝒗_𝑗 ))/ℎ_𝑗 
        h_j = np.linalg.norm(v_j)
        q_j = v_j / h_j
        heights[j] = h_j

        # Log volume is the cumulative sum of log orthogonal heights
        # 〖𝑽𝒐𝒍〗_𝑟=∏ ℎ_𝑗 
        # Implemented as a sum of logs because of floating point errors
        volumes[j] = volumes[j - 1] + np.log(h_j)

        # Remove the component along the latest orthonormal direction 𝒒_(𝑗−1) from all pixel residuals
        # 𝒓_𝑗 (𝒙)=𝒓_(𝑗−1)−(𝒒_(𝑗−1)^𝑇 𝒓_(𝑗−1)) 𝒒_(𝑗−1)
        proj_component = np.outer(q_j, np.dot(q_j, residuals)) 
        residuals -= proj_component

    return endmembers, volumes, heights, indices