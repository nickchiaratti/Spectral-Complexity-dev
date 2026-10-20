import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import pandas as pd
from bs4 import BeautifulSoup
# Set default color sequence for line plots and scatter plots
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.colormaps['okabe_ito'].colors)

# --- Configuration ---
SIMULATED_SENSORS=["rsLandsat8", "rsSentinel2", "rsWorldView3", "cvHYMAP2014", "cvAVIRISc2014", "cvASD"]
'Uses splib07 https://doi.org/10.5066/F7RR1WDJ'
cvASD_HTML_INDEX_PATH = r"C:\satelliteImagery\ground_truth\splib07\indexes\datatable_splib07b_cvASD.html"
cvASD_WV_FILE = r"C:\satelliteImagery\ground_truth\splib07\ASCIIdata\ASCIIdata_splib07b_cvASD\s07_ASD_Wavelengths_ASD_0.35-2.5_microns_2151_ch.txt"
rsLandsat8_HTML_INDEX_PATH = r"C:\satelliteImagery\ground_truth\splib07\indexes\datatable_splib07b_rsLandsat8.html"
rsLandsat8_WV_FILE = r"C:\satelliteImagery\ground_truth\splib07\ASCIIdata\ASCIIdata_splib07b_rsLandsat8\S07LSAT8_Wavelengths_Landsat8_(7_bands)_microns.txt"
rsSentinel2_HTML_INDEX_PATH = r"C:\satelliteImagery\ground_truth\splib07\indexes\datatable_splib07b_rsSentinel2.html"
rsSentinel2_WV_FILE = r"C:\satelliteImagery\ground_truth\splib07\ASCIIdata\ASCIIdata_splib07b_rsSentinel2\S07SNTL2_Wavelengths_Sentinel2_(13_bands)_microns.txt"
rsWorldView3_HTML_INDEX_PATH = r"C:\satelliteImagery\ground_truth\splib07\indexes\datatable_splib07b_rsWorldView3.html"
rsWorldView3_WV_FILE = r"C:\satelliteImagery\ground_truth\splib07\ASCIIdata\ASCIIdata_splib07b_rsWorldView3\S07WV3_Wavelengths_WorldView3_(16_bands)_micron.txt"
cvHYMAP2014_HTML_INDEX_PATH = r"C:\satelliteImagery\ground_truth\splib07\indexes\datatable_splib07b_cvHYMAP2014.html"
cvHYMAP2014_WV_FILE = r"C:\satelliteImagery\ground_truth\splib07\ASCIIdata\ASCIIdata_splib07b_cvHYMAP2014\s07_HY14_HyMap2_Alaska2014_wavelengths_126ch.txt"
cvAVIRISc2014_HTML_INDEX_PATH = r"C:\satelliteImagery\ground_truth\splib07\indexes\datatable_splib07b_cvAVIRISc2014.html"
cvAVIRISc2014_WV_FILE = r"C:\satelliteImagery\ground_truth\splib07\ASCIIdata\ASCIIdata_splib07b_cvAVIRISc2014\s07_AV14_Wavelengths_in_microns_224_ch_AVIRIS14a.txt"

# Self collected Leaf spectral data using an ASD Field Spectrometer. Leaf1 type is a norway maple, Leaf2 is a silver maple. The disease is tar spot.
NICKS_LEAVES_DIR  ="C:/satelliteImagery/ground_truth/NickLeaves_20261001"

# Provide materials (outermost to innermost square)
MATERIALS_TO_USE = {
    #"121": {"materials": ["Oak Oak-Leaf-1"],                                                                             "SNR": 500, "Variance": 0.45},
    #"100": {"materials": ["Oak Oak-Leaf-1", "Lodgepole-Pine", "Aspen Aspen-1","Douglas-Fir","Aspen Aspen-4","Goethite0.02+Quartz","Conifer-Meadow"],               "SNR": 500,  "Variance": 0.45},
    #"81":  {"materials": ["Oak Oak-Leaf-1", "Maple_Leaves","Goethite0.02+Quartz","Aspen Aspen-1","Douglas-Fir","Aspen Aspen-4","Conifer-Meadow"],               "SNR": 500,  "Variance": 0.45},
    "49":  {"materials": ["Lodgepole-Pine", "Maple_Leaves","Aspen Aspen-1","Douglas-Fir","Aspen Aspen-4","Conifer-Meadow"],               "SNR": np.inf,  "Variance": 1},
    #"25":  {"materials": ["Sheet_Metal GDS352", "Ultramarine GDS791", "Cuprite HS127.3B","Maple_Leaves",],               "SNR": None,  "Variance": 0.1}
    #"25":  {"materials": ["Oak Oak-Leaf-1", "Lodgepole-Pine", "Maple_Leaves","Aspen Aspen-1","Douglas-Fir","Aspen Aspen-4","Conifer-Meadow","Goethite0.02+Quartz"],               "SNR": np.inf,  "Variance": 1}
    #"49":  {"materials": ["NICKLEAVES:DiseaseLeaf1", "NICKLEAVES:DiseaseLeaf2","NICKLEAVES:HealthyLeaf1", "NICKLEAVES:HealthyLeaf2"],   "SNR": np.inf,  "Variance": 1}
    #"49":  {"materials": ["NICKLEAVES:HealthyLeaf1", "NICKLEAVES:HealthyLeaf2"],   "SNR": np.inf,  "Variance": 1}
}
NUM_ADDITIONAL_EM = 8
ENABLE_MODEL_PREDICTIONS = True

def maximumDistance_parallelotopeVolumes(pixels, num_endmembers):
    """
    Combined endmember extraction and Gram volume estimation using the maximum distance (MaxD) method with iterative orthogonal projection.

    Localizes the pixel neighborhood by subtracting the minimum-norm pixel (the local origin), then iteratively selects endmembers by largest
    orthogonal projection distance while recording the heights needed for the volume curve.

    Args:
        pixels (np.ndarray): (bands, pixels) or (height, width, bands) array of x pixel spectra.
        num_endmembers (int): The number of r endmembers to extract.
    Returns:
        endmembers (np.ndarray): (bands, num_endmembers) array of endmember spectra v_r.
        volumes (np.ndarray): (num_endmembers,) array of ln(Vol_r).
        heights (np.ndarray): (num_endmembers,) array of heights h_r.
        indices (np.ndarray): (num_endmembers,) array of indices i_r of the extracted endmembers.
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
    volumes[1] = np.log(h1) if h1 > 1e-15 else -np.inf
    heights[1] = h1
    if h1 > 1e-15:
        q1 = d1 / h1  # (bands,1)
        proj_component = np.outer(q1, np.dot(q1, residuals))  # equivalent to q1*q1.T*residuals in vector operations
        residuals -= proj_component

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
        indices[j] = idx_new

        # Calculate the subsequent height ℎ_𝑗 and orthonormal basis 𝒒_𝑗
        # ℎ_𝑗=‖𝒓_𝑗 (𝒗_𝑗 )‖
        # 𝒒_𝒋=(𝒓_𝑗 (𝒗_𝑗 ))/ℎ_𝑗 
        h_j = np.linalg.norm(v_j)
        heights[j] = h_j

        if h_j > 1e-15:
            q_j = v_j / h_j
            volumes[j] = volumes[j - 1] + np.log(h_j)
            proj_component = np.outer(q_j, np.dot(q_j, residuals)) # equivalent to q_j*q_j.T*residuals in vector operations
            residuals -= proj_component
        else:
            volumes[j] = -np.inf

    return endmembers, volumes, heights, indices

def maximumDistance_parallelotopeVolumes_noise(pixels, num_endmembers, noise_pixels):
    """
    Combined endmember extraction and Gram volume estimation using the maximum distance (MaxD) method with iterative orthogonal projection.

    Localizes the pixel neighborhood by subtracting the minimum-norm pixel (the local origin), then iteratively selects endmembers by largest
    orthogonal projection distance while recording the heights needed for the volume curve.

    Args:
        pixels (np.ndarray): (bands, pixels) or (height, width, bands) array of x pixel spectra.
        num_endmembers (int): The number of r endmembers to extract.
        noise_pixels (np.ndarray): (bands, pixels) or (height, width, bands) array of noise pixel spectra.
    Returns:
        endmembers (np.ndarray): (bands, num_endmembers) array of endmember spectra v_r.
        volumes (np.ndarray): (num_endmembers,) array of ln(Vol_r).
        heights (np.ndarray): (num_endmembers,) array of heights h_r.
        indices (np.ndarray): (num_endmembers,) array of indices i_r of the extracted endmembers.
        noise_norms_mean (np.ndarray): (num_endmembers,) array of mean noise heights Nh_r.
    """
    if pixels.ndim == 3:
        h, w, num_bands = pixels.shape
        pixels = np.reshape(pixels, (h * w, num_bands), order="F").astype(np.float64).T
    elif pixels.ndim == 2:
        pixels = pixels.astype(np.float64)

    if noise_pixels.ndim == 3:
        hn, wn, num_bands_n = noise_pixels.shape
        noise_pixels = np.reshape(noise_pixels, (hn * wn, num_bands_n), order="F").astype(np.float64).T
    elif noise_pixels.ndim == 2:
        noise_pixels = noise_pixels.astype(np.float64)
    
    noise_residuals = noise_pixels.copy()
    noise_norms_mean = np.zeros(num_endmembers, dtype=np.float64)
    noise_heights = np.zeros(num_endmembers, dtype=np.float64)

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
    volumes[1] = np.log(h1) if h1 > 1e-15 else -np.inf
    heights[1] = h1
    noise_heights[1] = np.linalg.norm(noise_residuals[:, idx_maxnorm].copy())
    noise_norms_mean[1] = np.mean(np.linalg.norm(noise_residuals, axis=0))
    if h1 > 1e-15:
        q1 = d1 / h1  # (bands,1)
        proj_component = np.outer(q1, np.dot(q1, residuals))  # equivalent to q1*q1.T*residuals in vector operations
        residuals -= proj_component
        noise_proj = np.outer(q1, np.dot(q1, noise_residuals))
        noise_residuals -= noise_proj

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
        indices[j] = idx_new

        # Calculate the subsequent height ℎ_𝑗 and orthonormal basis 𝒒_𝑗
        # ℎ_𝑗=‖𝒓_𝑗 (𝒗_𝑗 )‖
        # 𝒒_𝒋=(𝒓_𝑗 (𝒗_𝑗 ))/ℎ_𝑗 
        h_j = np.linalg.norm(v_j)
        heights[j] = h_j
        noise_heights[j] = np.linalg.norm(noise_residuals[:, idx_new].copy())
        noise_norms_mean[j] = np.mean(np.linalg.norm(noise_residuals, axis=0))

        if h_j > 1e-15:
            q_j = v_j / h_j
            volumes[j] = volumes[j - 1] + np.log(h_j)
            proj_component = np.outer(q_j, np.dot(q_j, residuals)) # equivalent to q_j*q_j.T*residuals in vector operations
            residuals -= proj_component
            noise_proj = np.outer(q_j, np.dot(q_j, noise_residuals))
            noise_residuals -= noise_proj
        else:
            volumes[j] = -np.inf

    return endmembers, volumes, heights, indices, noise_heights, noise_norms_mean

def generate_rgba_from_hsi(frame_data, wavelengths, low=0.5, high=99.5, gamma=1.1, nodata=None):
    """
    Extracts, stretches, and gamma-corrects a full hyperspectral 
    frame (using CIE 1931 CMFs) to create a true color image with an alpha channel.
    
    The alpha channel follows standard RGBA opacity conventions:
    255 (Opaque) where pixels are valid, and 0 (Transparent) where 
    pixels are invalid (all 0s or all NaNs).
    
    Parameters:
        frame_data (np.ndarray): Hyperspectral image data in float format of shape (bands, height, width).
        wavelengths (array-like): Band center wavelengths in nanometers (nm).
        low (float): Lower percentile for contrast stretching.
        high (float): Higher percentile for contrast stretching.
        gamma (float): Gamma correction value.
        nodata (float, optional): Value to treat as no-data/NaN.
        
    Returns:
        rgba_8bit (np.ndarray): Shape (height, width, 4), dtype uint8.
    """
    import colour
    from scipy.interpolate import interp1d
    
    bands, height, width = frame_data.shape
    
    wavelengths = np.asarray(wavelengths, dtype=np.float32)
    if not np.any(wavelengths > 0):
        raise ValueError("Invalid wavelengths provided to generate_rgba_from_hsi: all wavelengths are <= 0.")
    if np.nanmax(wavelengths) < 50.0:
        raise ValueError(f"Wavelengths must be provided in nanometers (nm), but got max wavelength {np.nanmax(wavelengths):.2f} < 50 nm.")
        
    # Ensure input frame_data is in float format
    if not np.issubdtype(frame_data.dtype, np.floating):
        raise TypeError(f"frame_data must be floating point dtype, got {frame_data.dtype}")

    if nodata is not None:
        frame_data = frame_data.copy()
        frame_data[frame_data == nodata] = np.nan
            
    # 1. Determine Invalid Pixel Mask
    # A pixel is invalid only if all bands are NaN or all bands are 0
    all_nan_mask = np.all(np.isnan(frame_data), axis=0)
    all_zero_mask = np.all(frame_data == 0, axis=0)
    invalid_pixel_mask = all_nan_mask | all_zero_mask
    valid_mask = ~invalid_pixel_mask
    valid_mask_flat = valid_mask.reshape(-1)
    
    if not np.any(valid_mask):
        return np.zeros((height, width, 4), dtype=np.uint8)
        
    # 2. Extract processes from truecolorhsi.visualization
    from truecolorhsi.visualization import get_illuminant_spd_and_xyz, get_band_index
    
    # Replace individual missing band values with 0.0 for CMF integration and clip negative noise
    frame_data_clean = np.nan_to_num(frame_data, nan=0.0)
    frame_data_clean = np.clip(frame_data_clean, 0.0, None)
    
    i_cutoff = get_band_index(wavelengths, 830.0)
    hyperspec_wavelengths = wavelengths[:i_cutoff]
    
    std_wavelengths, illuminant_values, xyz = get_illuminant_spd_and_xyz(illuminant='D65', plot_flag=False, run_example=False)
    
    interp_function = interp1d(std_wavelengths, illuminant_values, kind='linear', fill_value="extrapolate")
    illuminant_interp = interp_function(hyperspec_wavelengths)
    
    interp_func_0 = interp1d(std_wavelengths, xyz[:, 0], kind='linear', fill_value='extrapolate')
    interp_func_1 = interp1d(std_wavelengths, xyz[:, 1], kind='linear', fill_value='extrapolate')
    interp_func_2 = interp1d(std_wavelengths, xyz[:, 2], kind='linear', fill_value='extrapolate')
    
    cie_x_interp = interp_func_0(hyperspec_wavelengths)
    cie_y_interp = interp_func_1(hyperspec_wavelengths)
    cie_z_interp = interp_func_2(hyperspec_wavelengths)
    xyz_interp = np.column_stack((cie_x_interp, cie_y_interp, cie_z_interp))
    
    # Get the reflectance data in the visible range
    visible_range_data = frame_data_clean[:i_cutoff, :, :].reshape((i_cutoff, -1))
    
    # Calculate the XYZ of a perfect white diffuser under the same illuminant
    visible_range_data_white = np.ones((i_cutoff, 1), dtype=np.float32)
    XYZ_white = xyz_interp.T @ np.diag(illuminant_interp) @ visible_range_data_white
    
    # Convert Reflectance to CIEXYZ tristimulus values
    XYZ = xyz_interp.T @ np.diag(illuminant_interp) @ visible_range_data
    
    # Normalize the XYZ values using the Y component of the perfect white diffuser.
    # This preserves the physical chromaticity and scales the luminance to [0, 1] range
    # without dynamic scene-dependent stretching that ruins uniform simulations.
    XYZ_normalized = XYZ / XYZ_white[1]
    
    # XYZ to sRGB
    XYZ_image = XYZ_normalized.T.reshape(height, width, 3)
    SRGB_image = colour.XYZ_to_sRGB(XYZ_image)
    
    SRGB_image = np.clip(SRGB_image, 0, 1)
    
    if gamma != 1.0 and gamma > 0:
        SRGB_image = SRGB_image ** (1 / gamma)
        
    # Scale to 8-bit
    rgb_8bit = (SRGB_image * 255).astype(np.uint8)
    
    # 3. Construct the Alpha Channel
    alpha = np.full((height, width), 255, dtype=np.uint8)
    alpha[invalid_pixel_mask] = 0
    
    # 4. Stack alpha onto RGB to create RGBA
    rgba_8bit = np.dstack((rgb_8bit, alpha))
    
    return rgba_8bit

def lookup_material_file(html_path, search_str, soup=None):
    """Parses the splib07 HTML datatable to find the Spectrum Data path for a material."""
    if soup is None:
        with open(html_path, 'r', encoding='utf-8', errors='ignore') as f:
            soup = BeautifulSoup(f, 'html.parser')
            
    for tr in soup.find_all('tr'):
        th_or_td = tr.find(lambda tag: tag.name in ['th', 'td'] and tag.get('scope') == 'row')
        if th_or_td and search_str.lower() in th_or_td.get_text().lower():
            for a in tr.find_all('a'):
                if 'Spectrum Data' in a.get_text():
                    href = a.get('href')
                    base_dir = os.path.dirname(html_path)
                    return os.path.abspath(os.path.join(base_dir, href))
    raise ValueError(f"Could not find Spectrum Data for material: {search_str}")

def load_asd_txt(filepath):
    """Loads ASD ASCII file, ignoring header and setting -1.23e34 to np.nan"""
    with open(filepath, 'r') as f:
        lines = f.readlines()
    data = []
    for line in lines[1:]:
        if not line.strip():
            continue
        val = float(line.strip())
        if val < -1.0: # typically -1.23e34
            data.append(np.nan)
        else:
            data.append(val)
    return np.array(data, dtype=np.float32)

def add_gaussian_noise(img, snr=None, seed=42, return_noise=False):
    """
    Injects Additive White Gaussian Noise (AWGN) based on a target linear SNR.
    SNR = mu_signal / sigma_noise, where mu_signal is the mean reflectance.
    """
    if snr in [None, np.inf]:
        if return_noise:
            return img, np.zeros_like(img)
        return img
        
    np.random.seed(seed)
    mu_signal = np.mean(img)
    # Prevent division by zero if layer is completely dark
    sigma = mu_signal / snr
    
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    if return_noise:
        return img + noise, noise
    return img + noise

def create_mixture_layer(materials, size, variance=1, seed=256):
    """
    Creates a square layer of the given size containing a linear mixture 
    of the provided materials. If one material is provided, it returns a pure layer.
    """
    num_mats = len(materials)
    num_bands = materials[0].shape[0]
    layer = np.zeros((num_bands, size, size), dtype=np.float32)
    
    if num_mats == 1:
        layer[:, :, :] = materials[0][:, None, None]
    else:
        np.random.seed(seed)
        # The Dirichlet distribution naturally enforces that all fractions sum to 1.0
        # Map 'variance' to the Dirichlet alpha concentration parameter.
        # Low variance = high alpha = highly mixed. High variance = low alpha = purer pixels.
        alpha = np.ones(num_mats) * max(0.1, 1.0 / variance)
        
        # Output of np.random.dirichlet is (size*size, num_mats). 
        # Transpose and reshape to (num_mats, size, size) to match the layer shape.
        fractions = np.random.dirichlet(alpha, size=(size * size)).astype(np.float32)
        fractions = fractions.T.reshape((num_mats, size, size))
        
        for i, mat in enumerate(materials):
            layer += mat[:, None, None] * fractions[i]
            
    return layer

def create_4corner_mixture_layer(materials, size):
    """
    Places 4 materials in the corners of the square using bilinear interpolation.
    Guarantees pure corners and sums to 1.0 at all pixels.
    """
    num_bands = materials[0].shape[0]
    layer = np.zeros((num_bands, size, size), dtype=np.float32)
    
    if size <= 1:
        layer[:, :, :] = materials[0][:, None, None]
        return layer

    # Create 1D gradients from 0 to 1
    x = np.linspace(0, 1, size, dtype=np.float32)
    y = np.linspace(0, 1, size, dtype=np.float32)
    xv, yv = np.meshgrid(x, y)

    # Bilinear weights sum to 1.0 everywhere
    w0 = (1 - xv) * (1 - yv)  # Top-Left
    w1 = xv * (1 - yv)        # Top-Right
    w2 = (1 - xv) * yv        # Bottom-Left
    w3 = xv * yv              # Bottom-Right
    
    fractions = np.array([w0, w1, w2, w3])
    
    for i, mat in enumerate(materials):
        layer += mat[:, None, None] * fractions[i]
        
    return layer

def main(sensor_name):
    print("Parsing HTML index to find material files...")
    if sensor_name == "rsLandsat8":
        HTML_INDEX_PATH = rsLandsat8_HTML_INDEX_PATH
        WV_FILE = rsLandsat8_WV_FILE
    elif sensor_name == "rsSentinel2":
        HTML_INDEX_PATH = rsSentinel2_HTML_INDEX_PATH
        WV_FILE = rsSentinel2_WV_FILE
    elif sensor_name == "rsWorldView3":
        HTML_INDEX_PATH = rsWorldView3_HTML_INDEX_PATH
        WV_FILE = rsWorldView3_WV_FILE
    elif sensor_name == "cvHYMAP2014":
        HTML_INDEX_PATH = cvHYMAP2014_HTML_INDEX_PATH
        WV_FILE = cvHYMAP2014_WV_FILE
    elif sensor_name == "cvAVIRISc2014":
        HTML_INDEX_PATH = cvAVIRISc2014_HTML_INDEX_PATH
        WV_FILE = cvAVIRISc2014_WV_FILE
    elif sensor_name == "cvASD":
        HTML_INDEX_PATH = cvASD_HTML_INDEX_PATH
        WV_FILE = cvASD_WV_FILE
    else:
        raise ValueError(f"Unknown sensor name: {sensor_name}")
        
    with open(HTML_INDEX_PATH, 'r', encoding='utf-8', errors='ignore') as f:
        soup = BeautifulSoup(f, 'html.parser')
        
    loaded_materials = {}
    all_mat_names = []
    for layer_config in MATERIALS_TO_USE.values():
        if isinstance(layer_config, dict):
            all_mat_names.extend(layer_config.get("materials", []))
        else:
            all_mat_names.extend(layer_config)
        
    for mat_name in all_mat_names:
        if mat_name.startswith("NICKLEAVES:"):
            filename_str = mat_name.split(":", 1)[1]
            csv_path = os.path.join(NICKS_LEAVES_DIR, f"{sensor_name}_{filename_str}.csv")
            print(f"Found {mat_name}: {os.path.basename(csv_path)}")
            df = pd.read_csv(csv_path)
            loaded_materials[mat_name] = df['DHR_mean'].values.astype(np.float32)
        else:
            mat_file = lookup_material_file(HTML_INDEX_PATH, mat_name, soup=soup)
            print(f"Found {mat_name}: {os.path.basename(mat_file)}")
            loaded_materials[mat_name] = load_asd_txt(mat_file)
        
    print("Loading data...")
    wavelengths = load_asd_txt(WV_FILE)
    original_band_count = len(wavelengths)
    
    # Create valid mask across all materials
    valid_mask = ~np.isnan(wavelengths)
    for mat_data in loaded_materials.values():
        valid_mask &= ~np.isnan(mat_data)
        
    # Apply valid mask
    wavelengths = wavelengths[valid_mask]
    for mat_name in loaded_materials:
        loaded_materials[mat_name] = loaded_materials[mat_name][valid_mask]
        
    num_bands = len(wavelengths)
    print(f"Number of valid bands: {num_bands}")
    
    # Sort sizes descending to build image from outside in
    sizes = sorted([int(s) for s in MATERIALS_TO_USE.keys()], reverse=True)
    img_size = sizes[0]
    
    # 1. Create the simulated image
    img = np.zeros((num_bands, img_size, img_size), dtype=np.float32)
    noise_img = np.zeros((num_bands, img_size, img_size), dtype=np.float32)
    
    for tz in sizes:
        config = MATERIALS_TO_USE[str(tz)]
        
        if isinstance(config, dict):
            mat_names = config.get("materials", [])
            snr = config.get("SNR", None)
            variance = config.get("Variance", 0.45)
        else:
            mat_names = config
            snr = None
            variance = 0.45
            
        mats = [loaded_materials[name] for name in mat_names]
        
        if len(mats) == 4:
            layer = create_4corner_mixture_layer(mats, tz)
        else:
            layer = create_mixture_layer(mats, tz, variance=variance)
            
        if snr is not None:
            layer, layer_noise = add_gaussian_noise(layer, snr=snr, return_noise=True)
        else:
            layer_noise = np.zeros_like(layer)
        
        # Center the layer in the image
        start = (img_size - tz) // 2
        end = start + tz
        img[:, start:end, start:end] = layer
        noise_img[:, start:end, start:end] = layer_noise
    
    # Optional global noise: img = add_gaussian_noise(img, snr=500)

    # 2. Calculate metrics
    tile_sizes = sizes[::-1] # Ascending for evaluation
    
    results = {}
    
    for tz in tile_sizes:
        start = (img_size - tz) // 2
        end = start + tz
        tile = img[:, start:end, start:end]
        tile_noise = noise_img[:, start:end, start:end]
        
        # Flatten for maximumDistance_volumes
        pixels = tile.reshape(num_bands, -1)
        noise_pixels = tile_noise.reshape(num_bands, -1)
        
        # Calculate number of unique materials present in this tile and smaller tiles
        unique_mats = set()
        for t in tile_sizes:
            if t <= tz:
                config = MATERIALS_TO_USE[str(t)]
                mats = config.get("materials", []) if isinstance(config, dict) else config
                unique_mats.update(mats)
                
        num_em_eval = len(unique_mats) + NUM_ADDITIONAL_EM
        
        # Extract endmembers, volumes, heights, indices
        if noise_pixels is not None:
            em, vol, h, indices, noise_heights, noise_norms_mean = maximumDistance_parallelotopeVolumes_noise(pixels, num_em_eval, noise_pixels=noise_pixels)
        else:
            em, vol, h, indices = maximumDistance_parallelotopeVolumes(pixels, num_em_eval)
        
        # Convert 1D tile indices back to 2D global coordinates
        r_rel, c_rel = np.unravel_index(indices, (tz, tz))
        r_global = r_rel + start
        c_global = c_rel + start
        
        # Compute PCA eigenvalues for Scree plot
        from sklearn.decomposition import PCA
        pca = PCA(n_components=min(num_em_eval, num_bands))
        pca.fit(pixels.T)
        
        results[tz] = {
            'endmembers': em,
            'volumes': vol,
            'heights': h,
            'r_global': r_global,
            'c_global': c_global,
            'noise_heights': noise_heights,
            'noise_norms_mean': noise_norms_mean,
            'pca_variance_percentage': pca.explained_variance_ratio_ * 100
        }
        
    # 3. Displays
    
    # Create dynamic output directory based on the HTML index suffix
    html_basename = os.path.basename(HTML_INDEX_PATH)
    suffix = html_basename.split('_')[-1].replace('.html', '')
    output_dir = os.path.join(r"C:\satelliteImagery\simulated", suffix)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving results to: {output_dir}")

    # a. Displays an RGB image of the 17x17 image and a spectral plot of the source materials
    rgba = generate_rgba_from_hsi(img, wavelengths*1e3)
    
    fig, (ax_img, ax_spec) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"{sensor_name} ({num_bands} Bands) - Simulated Image & Source Spectra", fontsize=14)
    
    # Left: RGB Image
    ax_img.imshow(rgba)
    ax_img.set_title(f"Simulated {img_size}x{img_size} Image ({num_bands} Bands)")
    ax_img.axis('off')
    
    # Okabe-Ito accessible color sequence (natively supported in Matplotlib >= 3.11)
    tz_colors = plt.colormaps['okabe_ito'].colors
    for tz_idx, tz in enumerate(tile_sizes):
        color = tz_colors[tz_idx % len(tz_colors)]
        r_coords = results[tz]['r_global']
        c_coords = results[tz]['c_global']
        
        # Smaller tiles are evaluated first (lower tz_idx). Give them higher zorder so they render on top.
        z = 100 - tz_idx
        
        # Plot numbered markers
        ax_img.scatter(c_coords, r_coords, facecolors='none', edgecolors=[color], marker='o', s=30, linewidth=1, label=f'{tz}x{tz} tile', zorder=z)
        for i, (r, c) in enumerate(zip(r_coords, c_coords)):
            # Draw text without bbox to avoid color blending
            ax_img.text(c + 1.5, r - 1.5, str(i + 1), color=color, fontsize=5, fontweight='bold', zorder=z)
            
    ax_img.legend(loc='upper right', fontsize='small', framealpha=0.8)
    
    # Right: Source Spectra
    for tz in sizes:
        config = MATERIALS_TO_USE[str(tz)]
        mat_names = config.get("materials", []) if isinstance(config, dict) else config
        for mat_name in mat_names:
            is_mixed = "Mixed" if len(mat_names) > 1 else f"{tz}x{tz}"
            ax_spec.plot(wavelengths, loaded_materials[mat_name], label=f"{mat_name} ({is_mixed})")
            
    ax_spec.set_title(f"Source Material Spectra ({num_bands} Bands)")
    ax_spec.set_xlabel('Wavelength (microns)')
    ax_spec.set_ylabel('Reflectance')
    ax_spec.legend(title="Materials", loc='upper right', fontsize='small')
    ax_spec.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    rgb_path = os.path.join(output_dir, f'{sensor_name}_simulated_rgb.png')
    plt.savefig(rgb_path, dpi=300, bbox_inches='tight')
    print(f"Saved {rgb_path}")
    #plt.show()
    plt.close(fig)
    
    # b. Plots of volumes and heights vs endmember count
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    ax1_twin = ax1.twinx()
    
    max_num_em = max(len(results[t]['volumes']) for t in tile_sizes)
    max_em_idx = np.arange(1, max_num_em + 1)
    
    if ENABLE_MODEL_PREDICTIONS:
        # Attempt to load the trained models
        import glob
        import joblib
        import importlib
        import __main__
        
        mc_module = importlib.import_module("monte_carlo_stopping_criteria")
        __main__.FeatureSelectorWrapper = mc_module.FeatureSelectorWrapper
        
        model_dir = f"C:/satelliteImagery/simulated/monte_carlo_results/{original_band_count} bands"
        hgb_files = glob.glob(f"{model_dir}/endmember_regressor_hgb_*.joblib")
        linear_files = glob.glob(f"{model_dir}/endmember_regressor_linear_*.joblib")
        slope_multiplier_files = glob.glob(f"{model_dir}/slope_multiplier_bands_*.json")
        
        import json
        best_multiplier = None
        if slope_multiplier_files:
            with open(max(slope_multiplier_files, key=os.path.getctime), 'r') as f:
                best_multiplier = json.load(f).get('best_multiplier', 0.1)
                
        hgb_model = joblib.load(max(hgb_files, key=os.path.getctime)) if hgb_files else None
        linear_model = joblib.load(max(linear_files, key=os.path.getctime)) if linear_files else None
    else:
        hgb_model = None
        linear_model = None
        best_multiplier = None
    
    for tz in tile_sizes:
        vol = results[tz]['volumes']
        h = results[tz]['heights']
        diffs = -np.diff(h) # Calculate absolute height drops
        
        em_idx_tz = np.arange(1, len(vol) + 1)
        
        # Replace -inf and nan with np.nan for plotting to break lines correctly
        vol_plot = np.where(np.isinf(vol), np.nan, vol)
        linear_vol_plot = np.exp(vol_plot)
        
        # Predict Endmembers
        label_suffix = ""
        pred_hgb = None
        pred_linear = None
        pred_slope_thresh = None
        pred_log_vol_jump = None
        pred_log_ht_drop = None
        
        if hgb_model and linear_model:
            config = MATERIALS_TO_USE[str(tz)]
            tile_snr_raw = config.get("SNR") if isinstance(config, dict) else None
            tile_snr = tile_snr_raw if tile_snr_raw is not None else np.inf
            
            # Simulate true online iterative extraction
            pred_hgb = None
            pred_linear = None
            pred_slope_thresh = None
            
            for k in range(3, len(vol) + 1):
                partial_vol = vol[:k]
                partial_h = h[:k]
                
                feat = mc_module.extract_curve_features(partial_vol, partial_h, tile_snr, int(tz))
                feat['__raw_curve_volume'] = np.array(partial_vol, dtype=np.float64)
                feat['__raw_curve_height'] = np.array(partial_h, dtype=np.float64)
                
                X = pd.DataFrame([feat]).fillna(0.0)
                
                if pred_hgb is None:
                    p = int(np.round(hgb_model.predict(X)[0]))
                    if p <= k:
                        pred_hgb = k
                        
                if pred_linear is None:
                    p = int(np.round(linear_model.predict(X)[0]))
                    if p <= k:
                        pred_linear = k
                        
                # 3. Online Heuristic: Median Slope Threshold
                if pred_slope_thresh is None and best_multiplier is not None:
                    current_drop = diffs[k-2]
                    if k-2 > 0:
                        median_drop = np.nanmedian(diffs[:k-2])
                        if current_drop < best_multiplier * median_drop:
                            pred_slope_thresh = k
                            
                # If all valid models have triggered a stop, break the loop early
                if pred_hgb is not None and pred_linear is not None and (best_multiplier is None or pred_slope_thresh is not None):
                    break
                    
            # Cap any runaway models at the max extracted endmembers
            if pred_hgb is None: pred_hgb = len(vol)
            if pred_linear is None: pred_linear = len(vol)
            if pred_slope_thresh is None: pred_slope_thresh = len(vol)
            
            # --- Offline Heuristics (calculated on the full curve) ---
            # 1. Log-Height Max Drop Index
            log_ht = np.log10(np.maximum(h, 1e-30))
            pred_log_ht_drop = np.nanargmin(np.diff(log_ht)) + 1
            
            # 2. Log-Volume Max Jump Index
            log_vol = np.log10(np.maximum(vol, 1e-30))
            pred_log_vol_jump = np.nanargmax(np.diff(log_vol)) + 1
                
            label_suffix = f" (H:{pred_hgb}, L:{pred_linear}, S:{pred_slope_thresh}, VJ:{pred_log_vol_jump}, HD:{pred_log_ht_drop})"
            
        # Plot curves first to get the color
        p1 = ax1.plot(em_idx_tz, linear_vol_plot, marker='o', label=f'{tz}x{tz} tile{label_suffix}')
        color = p1[0].get_color()
        
        # Manhattan distance summation (cumulative sum of orthogonal heights)
        manhattan_sum = np.cumsum(h)
        ax1_twin.plot(em_idx_tz[1:], manhattan_sum[1:], linestyle='--', marker='^', color=color, alpha=0.7, label=f'{tz}x{tz} Manhattan dist')
        
        ax2.plot(em_idx_tz, h, marker='o', color=color, label=f'{tz}x{tz} tile{label_suffix}')
        ax2.plot(em_idx_tz[1:], np.abs(diffs), linestyle='-.', marker='s', color=color, alpha=0.7, label=f'{tz}x{tz} height diff')
        
        noise_heights = results[tz].get('noise_heights')
        if noise_heights is not None:
            ax2.plot(em_idx_tz, noise_heights, linestyle=':', marker='x', color=color, alpha=0.7, label=f'{tz}x{tz} noise height')
            noise_volumes = np.full_like(noise_heights, -np.inf)
            if noise_heights[1] > 0:
                noise_volumes[1] = np.log(noise_heights[1])
                for j in range(2, len(noise_heights)):
                    if noise_heights[j] > 0:
                        noise_volumes[j] = noise_volumes[j - 1] + np.log(noise_heights[j])
                        
            linear_noise_vol = np.exp(np.where(np.isinf(noise_volumes), np.nan, noise_volumes))
            ax1.plot(em_idx_tz, linear_noise_vol, linestyle='--', marker='x', color=color, alpha=0.7, label=f'{tz}x{tz} noise volume')
            
            # Cumulative Manhattan distance for noise
            noise_manhattan_sum = np.cumsum(noise_heights)
            ax1_twin.plot(em_idx_tz[1:], noise_manhattan_sum[1:], linestyle=':', marker='v', color=color, alpha=0.5, label=f'{tz}x{tz} noise Manhattan')


        noise_norms_mean = results[tz].get('noise_norms_mean')
        if noise_norms_mean is not None:
            ax2.plot(em_idx_tz[1:], noise_norms_mean[1:], linestyle='--', marker='x', color=color, alpha=0.7, label=f'{tz}x{tz} mean noise')
            
            # Cumulative product of noise norms (noise equivalent volume)
            
        
        # Add vertical lines using the same color as the curve
        if pred_hgb is not None and pred_linear is not None:
            offset = 0.05 if pred_hgb == pred_linear else 0.0
            ax1.axvline(pred_hgb - offset, color=color, linestyle='--', alpha=0.7)
            ax1.axvline(pred_linear + offset, color=color, linestyle=':', alpha=0.7)
            ax2.axvline(pred_hgb - offset, color=color, linestyle='--', alpha=0.7)
            ax2.axvline(pred_linear + offset, color=color, linestyle=':', alpha=0.7)
            if pred_slope_thresh is not None:
                ax2.axvline(pred_slope_thresh + offset * 2, color=color, linestyle='-.', alpha=0.5)
            if pred_log_vol_jump is not None:
                ax1.axvline(pred_log_vol_jump + offset * 3, color=color, linestyle='-.', alpha=0.5)
            if pred_log_ht_drop is not None:
                ax2.axvline(pred_log_ht_drop + offset * 3, color=color, linestyle='-.', alpha=0.5)
        
        
    total_materials = sum(len(config.get("materials", [])) for config in MATERIALS_TO_USE.values())
    unique_materials = len(set([mat_name for mat_name in sum([config.get("materials", []) for config in MATERIALS_TO_USE.values()], [])]))
    snrs = [config.get("SNR") for config in MATERIALS_TO_USE.values() if isinstance(config, dict)]
    numeric_snrs = [s for s in snrs if s is not None]
    min_snr = min(numeric_snrs) if numeric_snrs else "None"
    fig.suptitle(f'{sensor_name} Parallelotope Metrics ({num_bands} Bands)', fontsize=14)
    ax1.set_title(f'Volumes vs Endmember Count\n{sensor_name} ({num_bands} Bands) | Minimum SNR={min_snr} | # Materials={total_materials} | # Unique Materials={unique_materials}')
    ax1.set_xlabel('Number of Endmembers (k)')
    ax1.set_ylabel('Volume')
    rho_max = 0.7
    float32_minimum_threshold = rho_max * np.sqrt(num_bands) * np.finfo(np.float32).eps
    float64_spacing_precision= np.spacing(1)
    ax1.set_yscale('symlog', linthresh=float32_minimum_threshold)
    ax1.set_ylim(bottom=0)
    ax1.set_xticks(max_em_idx)
    
    # Configure twin axis for Manhattan distance
    ax1_twin.set_ylabel('Cumulative Manhattan Distance (Sum of Heights)')
    ax1_twin.set_ylim(bottom=0)
    ax1_twin.grid(False)
    
    # Combined legend for ax1 and ax1_twin
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='best', fontsize='x-small')
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    secax1 = ax1.secondary_xaxis('top', functions=(lambda x: x - 1, lambda x: x + 1))
    secax1.set_xlabel('Subspace Dimensionality')
    secax1.set_xticks(max_em_idx - 1)
    
    ax2.set_title(f'Heights vs Endmember Count\n{sensor_name} ({num_bands} Bands) | Minimum SNR={min_snr} | # Materials={total_materials} | # Unique Materials={unique_materials}')
    ax2.set_xlabel('Number of Endmembers (k)')
    ax2.set_ylabel('Orthogonal Distance (Height)')
    
    ax2.set_yscale('symlog', linthresh=float32_minimum_threshold)
    ax2.set_xticks(max_em_idx)
    ax2.legend(loc='best', fontsize='x-small')
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    secax2 = ax2.secondary_xaxis('top', functions=(lambda x: x - 1, lambda x: x + 1))
    secax2.set_xlabel('Subspace Dimensionality')
    secax2.set_xticks(max_em_idx - 1)
    
    plt.tight_layout()
    fig.subplots_adjust(wspace=0.28)
    metrics_path = os.path.join(output_dir, f'{sensor_name}_metrics_plot.png')
    plt.savefig(metrics_path, dpi=300,pad_inches=0,bbox_inches='tight')
    print(f"Saved {metrics_path}")
    plt.close(fig)
    
    # c. PCA Scree Plot
    fig_scree, ax_scree = plt.subplots(figsize=(10, 6))
    for tz in tile_sizes:
        evals = results[tz]['pca_variance_percentage']
        ax_scree.plot(np.arange(1, len(evals) + 1), evals, marker='.', linestyle='-', alpha=0.7, label=f'{tz}x{tz} tile')
        
    ax_scree.set_title(f'PCA Scree Plot\n{sensor_name} ({num_bands} Bands) | Minimum SNR={min_snr} | # Materials={total_materials} | # Unique Materials={unique_materials}')
    ax_scree.set_xlabel('Principal Component Number')
    ax_scree.set_ylabel('Percentage of Total Variance (%)')
    #ax_scree.set_yscale('symlog', linthresh=1e-5)
    ax_scree.set_ylim(top=10,bottom=0)
    ax_scree.legend(loc='upper right', fontsize='small')
    ax_scree.grid(True, linestyle='--', alpha=0.6)
    
    # Zoom x-axis to the interesting region (the elbow)
    zoom_max = min(max_num_em * 2, max(len(evals) for tz in tile_sizes))
    ax_scree.set_xlim(0.5, zoom_max + 0.5)
    ax_scree.set_xticks(np.arange(1, int(zoom_max) + 1))
    
    plt.tight_layout()
    scree_path = os.path.join(output_dir, f'{sensor_name}_scree_plot.png')
    plt.savefig(scree_path, dpi=300, bbox_inches='tight')
    print(f"Saved {scree_path}")
    plt.close(fig_scree)
    
    # c. Spectral plots of the endmembers with a legend indicating the index order of the endmember spectra.
    num_tiles = len(tile_sizes)
    cols = 3 if num_tiles > 4 else 2
    rows = (num_tiles + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 6*rows))
    fig.suptitle(f'{sensor_name} Extracted Endmember Spectra ({num_bands} Bands)', fontsize=14)
    axes = axes.flatten()
    
    for i in range(len(axes)):
        if i >= num_tiles:
            axes[i].axis('off')
            continue
            
        tz = tile_sizes[i]
        ax = axes[i]
        em = results[tz]['endmembers']
        n_em = em.shape[1]
        
        # Plot in reverse order so the first endmembers are drawn last and appear on top
        for j in reversed(range(n_em)):
            # If the endmember is all NaN, it means extraction failed (e.g. identical pixels)
            if np.isnan(em[:, j]).all():
                continue
            
            # Label with the extraction index order (1-indexed)
            ax.plot(wavelengths, em[:, j], label=f'Index {j+1}')
            
        ax.set_title(f'Extracted Endmembers for {tz}x{tz} Tile ({num_bands} Bands)')
        ax.set_xlabel('Wavelength (microns)')
        ax.set_ylabel('Reflectance')
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1], title='Extraction Order', loc='upper right')
        ax.grid(True, linestyle='--', alpha=0.6)
        
    plt.tight_layout()
    spectra_path = os.path.join(output_dir, f'{sensor_name}_endmember_spectra.png')
    plt.savefig(spectra_path, dpi=300,pad_inches=0,bbox_inches='tight')
    print(f"Saved {spectra_path}")
    
if __name__ == "__main__":
    for sensor_name in SIMULATED_SENSORS:
        main(sensor_name)
