import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from bs4 import BeautifulSoup
# Set default color sequence for line plots and scatter plots
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.colormaps['okabe_ito'].colors)

# --- Configuration ---
SIMULATED_SENSOR="cvASD"
'Uses splib07 https://doi.org/10.5066/F7RR1WDJ'
if SIMULATED_SENSOR=="cvASD":
    HTML_INDEX_PATH = r"C:\satelliteImagery\ground_truth\splib07\indexes\datatable_splib07b_cvASD.html"
    WV_FILE = r"C:\satelliteImagery\ground_truth\splib07\ASCIIdata\ASCIIdata_splib07b_cvASD\s07_ASD_Wavelengths_ASD_0.35-2.5_microns_2151_ch.txt"
elif SIMULATED_SENSOR=="rsLandsat8":
    HTML_INDEX_PATH = r"C:\satelliteImagery\ground_truth\splib07\indexes\datatable_splib07b_rsLandsat8.html"
    WV_FILE = r"C:\satelliteImagery\ground_truth\splib07\ASCIIdata\ASCIIdata_splib07b_rsLandsat8\S07LSAT8_Wavelengths_Landsat8_(7_bands)_microns.txt"
elif SIMULATED_SENSOR=="rsSentinel2":
    HTML_INDEX_PATH = r"C:\satelliteImagery\ground_truth\splib07\indexes\datatable_splib07b_rsSentinel2.html"
    WV_FILE = r"C:\satelliteImagery\ground_truth\splib07\ASCIIdata\ASCIIdata_splib07b_rsSentinel2\S07SNTL2_Wavelengths_Sentinel2_(13_bands)_microns.txt"
elif SIMULATED_SENSOR=="rsWorldView3":
    HTML_INDEX_PATH = r"C:\satelliteImagery\ground_truth\splib07\indexes\datatable_splib07b_rsWorldView3.html"
    WV_FILE = r"C:\satelliteImagery\ground_truth\splib07\ASCIIdata\ASCIIdata_splib07b_rsWorldView3\S07WV3_Wavelengths_WorldView3_(16_bands)_micron.txt"

# Provide materials (outermost to innermost square)
MATERIALS_TO_USE = {
    "121": {"materials": ["Aspen Aspen-1"],                                                                             "SNR": None, "Variance": 0.1},
    "100": {"materials": ["Maple_Leaves"],                                                                              "SNR": None, "Variance": 0.1},
    "81":  {"materials": ["Lodgepole-Pine"],                                                                            "SNR": None, "Variance": 0.1},
    "49":  {"materials": ["Hyacinth DWO-3-DEL-2"],                                                                      "SNR": None, "Variance": 0.1},
    "25":  {"materials": ["Sheet_Metal GDS352", "Ultramarine GDS791", "Cuprite HS127.3B","Maple_Leaves"],               "SNR": None,  "Variance": 0.1}
}
NUM_ADDITIONAL_EM = 3

def maximumDistance_volumes(pixels, num_endmembers):
    """
    Combined endmember extraction and Gram volume estimation in a single
    iterative Orthogonal Subspace Projection (OSP) pass.

    Localizes the pixel neighborhood by subtracting the minimum-norm pixel
    (the local origin), then iteratively selects endmembers by largest
    orthogonal projection distance while recording the heights needed for 
    the volume curve.
    """
    if pixels.ndim == 3:
        h, w, num_bands = pixels.shape
        pixels = np.reshape(pixels, (h * w, num_bands), order="F").astype(np.float64).T
    elif pixels.ndim == 2:
        pixels = pixels.astype(np.float64)

    num_bands, num_pixels = pixels.shape

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
    volumes = np.full(num_endmembers, -np.inf, dtype=np.float64)
    heights = np.zeros(num_endmembers, dtype=np.float64)
    indices = np.zeros(num_endmembers, dtype=int)
    
    indices[0] = idx_maxnorm
    indices[1] = idx_minnorm
    selected_indices = [idx_minnorm, idx_maxnorm]
    
    # --- First endmember: the localized max-norm pixel ---
    v = residuals[:, idx_maxnorm].copy()
    h_dist = np.linalg.norm(v)
    volumes[1] = np.log(h_dist)
    heights[1] = h_dist

    u1 = v / h_dist

    # Project all pixels into the subspace orthogonal to v
    proj_component = np.outer(u1, np.dot(u1, residuals))
    residuals -= proj_component

    # --- Iteratively extract remaining endmembers ---
    for j in range(2, num_endmembers):
        # Magnitudes of unselected pixels in the current orthogonal subspace
        residual_mag = np.sum(residuals ** 2, axis=0)  # (num_pixels,)
        residual_mag[selected_indices] = -np.inf

        idx_new = np.argmax(residual_mag)
        endmembers[:, j] = pixels[:, idx_new]
        indices[j] = idx_new
        selected_indices.append(idx_new)

        # The orthogonal distance is the magnitude of the projected vector
        v = residuals[:, idx_new].copy()
        h_dist = np.linalg.norm(v)

        # Log volume is the cumulative sum of log orthogonal heights
        volumes[j] = volumes[j - 1] + np.log(h_dist)
        heights[j] = h_dist

        uj = v / h_dist

        # Project all pixels into the subspace orthogonal to the new endmember
        proj_component = np.outer(uj, np.dot(uj, residuals))
        residuals -= proj_component

    return endmembers, volumes, heights, indices

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

def add_gaussian_noise(img, snr=None, seed=42):
    """
    Injects Additive White Gaussian Noise (AWGN) based on a target linear SNR.
    SNR = mu_signal / sigma_noise, where mu_signal is the mean reflectance.
    """
    if snr is None:
        return img
        
    np.random.seed(seed)
    mu_signal = np.mean(img)
    # Prevent division by zero if layer is completely dark
    sigma = mu_signal / snr if mu_signal > 0 else 0.005
    
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    return img + noise

def create_mixture_layer(materials, size, variance=0.45, seed=42):
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
        mean_fraction = 1.0 / num_mats
        fractions = np.random.normal(mean_fraction, variance, (num_mats, size, size)).astype(np.float32)
        fractions = np.clip(fractions, 0, 1)
        sums = np.sum(fractions, axis=0, keepdims=True)
        sums[sums == 0] = 1.0
        fractions /= sums
        
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

def main():
    print("Parsing HTML index to find material files...")
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
        mat_file = lookup_material_file(HTML_INDEX_PATH, mat_name, soup=soup)
        print(f"Found {mat_name}: {os.path.basename(mat_file)}")
        loaded_materials[mat_name] = load_asd_txt(mat_file)
        
    print("Loading data...")
    wavelengths = load_asd_txt(WV_FILE)
    
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
            layer = add_gaussian_noise(layer, snr=snr)
        
        # Center the layer in the image
        start = (img_size - tz) // 2
        end = start + tz
        img[:, start:end, start:end] = layer
    
    # Optional global noise: img = add_gaussian_noise(img, snr=500)

    # 2. Calculate metrics
    tile_sizes = sizes[::-1] # Ascending for evaluation
    total_materials = sum(len(v.get("materials", [])) if isinstance(v, dict) else len(v) for v in MATERIALS_TO_USE.values())
    num_em_eval = total_materials + NUM_ADDITIONAL_EM
    
    results = {}
    
    for tz in tile_sizes:
        start = (img_size - tz) // 2
        end = start + tz
        tile = img[:, start:end, start:end]
        
        # Flatten for maximumDistance_volumes
        pixels = tile.reshape(num_bands, -1)
        
        # Extract endmembers, volumes, heights, indices
        em, vol, h, indices = maximumDistance_volumes(pixels, num_em_eval)
        
        # Convert 1D tile indices back to 2D global coordinates
        r_rel, c_rel = np.unravel_index(indices, (tz, tz))
        r_global = r_rel + start
        c_global = c_rel + start
        
        results[tz] = {
            'endmembers': em,
            'volumes': vol,
            'heights': h,
            'r_global': r_global,
            'c_global': c_global
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
    
    # Left: RGB Image
    ax_img.imshow(rgba)
    ax_img.set_title(f"Simulated {img_size}x{img_size} Image")
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
            
    ax_spec.set_title("Source Material Spectra")
    ax_spec.set_xlabel('Wavelength (microns)')
    ax_spec.set_ylabel('Reflectance')
    ax_spec.legend(title="Materials", loc='upper right', fontsize='small')
    ax_spec.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    rgb_path = os.path.join(output_dir, f'{SIMULATED_SENSOR}_simulated_rgb.png')
    plt.savefig(rgb_path, dpi=300, bbox_inches='tight')
    print(f"Saved {rgb_path}")
    #plt.show()
    plt.close(fig)
    
    # b. Plots of volumes and heights vs endmember count
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    em_idx = np.arange(1, em.shape[1]+1)
    
    for tz in tile_sizes:
        vol = results[tz]['volumes']
        h = results[tz]['heights']
        
        # Replace -inf and nan with np.nan for plotting to break lines correctly
        vol_plot = np.where(np.isinf(vol), np.nan, vol)
        linear_vol_plot = np.exp(vol_plot)
        
        ax1.plot(em_idx, linear_vol_plot, marker='o', label=f'{tz}x{tz} tile')
        
        ax2.plot(em_idx, h, marker='o', label=f'{tz}x{tz} tile')
        
        
    ax1.set_title('Volumes vs Endmember Count')
    ax1.set_xlabel('Number of Endmembers (k)')
    ax1.set_ylabel('Volume')
    ax1.set_yscale('symlog', linthresh=1e-7)
    ax1.set_xticks(em_idx)
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    secax1 = ax1.secondary_xaxis('top', functions=(lambda x: x - 1, lambda x: x + 1))
    secax1.set_xlabel('Subspace Dimensionality')
    secax1.set_xticks(em_idx - 1)
    
    ax2.set_title('Heights vs Endmember Count')
    ax2.set_xlabel('Number of Endmembers (k)')
    ax2.set_ylabel('Orthogonal Distance (Height)')
    ax2.set_yscale('asinh')
    ax2.set_xticks(em_idx)
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    secax2 = ax2.secondary_xaxis('top', functions=(lambda x: x - 1, lambda x: x + 1))
    secax2.set_xlabel('Subspace Dimensionality')
    secax2.set_xticks(em_idx - 1)
    
    plt.tight_layout()
    metrics_path = os.path.join(output_dir, f'{SIMULATED_SENSOR}_metrics_plot.png')
    plt.savefig(metrics_path, dpi=300,pad_inches=0,bbox_inches='tight')
    print(f"Saved {metrics_path}")
    
    # c. Spectral plots of the endmembers with a legend indicating the index order of the endmember spectra.
    num_tiles = len(tile_sizes)
    cols = 3 if num_tiles > 4 else 2
    rows = (num_tiles + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 6*rows))
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
            
        ax.set_title(f'Extracted Endmembers for {tz}x{tz} Tile')
        ax.set_xlabel('Wavelength (microns)')
        ax.set_ylabel('Reflectance')
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1], title='Extraction Order', loc='upper right')
        ax.grid(True, linestyle='--', alpha=0.6)
        
    plt.tight_layout()
    spectra_path = os.path.join(output_dir, f'{SIMULATED_SENSOR}_endmember_spectra.png')
    plt.savefig(spectra_path, dpi=300,pad_inches=0,bbox_inches='tight')
    print(f"Saved {spectra_path}")
    
if __name__ == "__main__":
    main()
