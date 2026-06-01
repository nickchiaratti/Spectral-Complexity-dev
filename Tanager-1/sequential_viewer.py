import h5py
import numpy as np
import matplotlib.pyplot as plt
import re

# Target wavelengths for True Color composite (in nanometers)
TARGET_RED_NM = 670.0
TARGET_GREEN_NM = 550.0
TARGET_BLUE_NM = 480.0

def get_geospatial_extent(h5_file):
    """
    Parses the internal StructMetadata.0 ODL string to extract the physical
    bounding coordinates (in meters) of the dataset.
    """
    meta_data = h5_file["HDFEOS INFORMATION/StructMetadata.0"][()]
    if isinstance(meta_data, (np.ndarray, list)): 
        meta_data = meta_data[0]
    odl = meta_data.decode('ascii') if isinstance(meta_data, bytes) else str(meta_data)
    
    # Extract coordinates; intentionally allowing this to fail with AttributeError if regex finds nothing
    ul_match = re.search(r'UpperLeftPointMtrs=\(\s*(-?[\d\.]+)\s*,\s*(-?[\d\.]+)\s*\)', odl)
    lr_match = re.search(r'LowerRightMtrs=\(\s*(-?[\d\.]+)\s*,\s*(-?[\d\.]+)\s*\)', odl)
    
    ul_x, ul_y = float(ul_match.group(1)), float(ul_match.group(2))
    lr_x, lr_y = float(lr_match.group(1)), float(lr_match.group(2))
    
    # Matplotlib extent format: [xmin, xmax, ymin, ymax]
    return [ul_x, lr_x, lr_y, ul_y]

def robust_percentile_stretch(band_data, lower_pct=2, upper_pct=98):
    """
    Applies a robust linear contrast stretch based on data percentiles,
    ignoring NaN values to prevent nodata areas from skewing the histogram.
    """
    valid_data = band_data[~np.isnan(band_data)]
    
    # Allow failure if valid_data is empty
    p_low, p_high = np.percentile(valid_data, (lower_pct, upper_pct))
    
    # Avoid division by zero if image has zero variance
    if p_high == p_low:
        return np.zeros_like(band_data)
        
    stretched = (band_data - p_low) / (p_high - p_low)
    return np.clip(stretched, 0, 1)

def load_and_process_scene(filepath):
    """
    Loads the Tanager hyperspectral scene, extracts RGB bands, applies
    the nodata mask, and returns the normalized RGB array and spatial extent.
    """
    print(f"Loading scene: {filepath}")
    with h5py.File(filepath, 'r') as f:
        # Source files use the 'HYP' grid name
        grid_base = 'HDFEOS/GRIDS/HYP/Data Fields'
        
        # Load datasets
        sr_dset = f[f'{grid_base}/surface_reflectance']
        nodata_mask = f[f'{grid_base}/nodata_pixels'][:]
        wavelengths = sr_dset.attrs['wavelengths']
        
        # Determine closest band indices for RGB
        r_idx = np.argmin(np.abs(wavelengths - TARGET_RED_NM))
        g_idx = np.argmin(np.abs(wavelengths - TARGET_GREEN_NM))
        b_idx = np.argmin(np.abs(wavelengths - TARGET_BLUE_NM))
        
        print(f"  Extracted Bands -> R: {wavelengths[r_idx]:.1f}nm, "
              f"G: {wavelengths[g_idx]:.1f}nm, B: {wavelengths[b_idx]:.1f}nm")
        
        # Extract the specific 2D arrays (Rank 3 tensor: Bands x Height x Width)
        r_band = sr_dset[r_idx, :, :].astype(float)
        g_band = sr_dset[g_idx, :, :].astype(float)
        b_band = sr_dset[b_idx, :, :].astype(float)
        
        # Apply NoData Mask (1 = NoData, 0 = Valid)
        # NaN is used so matplotlib natively renders these pixels as transparent
        invalid_pixels = (nodata_mask == 1)
        r_band[invalid_pixels] = np.nan
        g_band[invalid_pixels] = np.nan
        b_band[invalid_pixels] = np.nan
        
        # Normalize for visualization
        r_norm = robust_percentile_stretch(r_band)
        g_norm = robust_percentile_stretch(g_band)
        b_norm = robust_percentile_stretch(b_band)
        
        rgb_composite = np.stack([r_norm, g_norm, b_norm], axis=-1)
        extent = get_geospatial_extent(f)
        
        return rgb_composite, extent

def main():
    file1 = r"C:\satelliteImagery\Tanager\SourceData\20250903_164714_08_4001\20250903_164714_08_4001_ortho_sr_hdf5.h5"
    file2 = r"C:\satelliteImagery\Tanager\SourceData\20250903_164719_92_4001\20250903_164719_92_4001_ortho_sr_hdf5.h5"

    rgb1, extent1 = load_and_process_scene(file1)
    rgb2, extent2 = load_and_process_scene(file2)

    # Visualization
    fig, ax = plt.subplots(figsize=(12, 10))
    fig.suptitle("Sequential Tanager Acquisitions in Shared Geographic Space", fontsize=14, fontweight='bold')

    # Calculate global limits to encompass both images in the shared view
    global_xmin = min(extent1[0], extent2[0])
    global_xmax = max(extent1[1], extent2[1])
    global_ymin = min(extent1[2], extent2[2])
    global_ymax = max(extent1[3], extent2[3])

    # Display both images on the single axis
    # The first image is plotted normally
    ax.imshow(rgb1, extent=extent1, origin='upper')

    # The second image is plotted over the first. Since invalid pixels were set to NaN,
    # they will be transparent, allowing the first image to show through where the
    # second image has no data. Where the second image has valid data, it will
    # overwrite the first image in the overlapping region.
    ax.imshow(rgb2, extent=extent2, origin='upper')

    ax.set_xlabel("Easting (Meters)")
    ax.set_ylabel("Northing (Meters)")
    ax.grid(True, linestyle='--', alpha=0.5)

    ax.set_xlim(global_xmin, global_xmax)
    ax.set_ylim(global_ymin, global_ymax)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()