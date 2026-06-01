import h5py
import numpy as np
import matplotlib.pyplot as plt
import re

# Target wavelengths for True Color composite (in nanometers)
TARGET_RED_NM = 670.0
TARGET_GREEN_NM = 550.0
TARGET_BLUE_NM = 480.0

# Grouped sequential acquisitions
file_groups = [
    [
        r"C:\satelliteImagery\Tanager\SourceData\20250704_165204_61_4001\20250704_165204_61_4001_ortho_sr_hdf5.h5",
        r"C:\satelliteImagery\Tanager\SourceData\20250704_165208_78_4001\20250704_165208_78_4001_ortho_sr_hdf5.h5",
        r"C:\satelliteImagery\Tanager\SourceData\20250704_165212_93_4001\20250704_165212_93_4001_ortho_sr_hdf5.h5"
    ],
    [
        r"C:\satelliteImagery\Tanager\SourceData\20250801_165544_61_4001\20250801_165544_61_4001_ortho_sr_hdf5.h5",
        r"C:\satelliteImagery\Tanager\SourceData\20250801_165548_86_4001\20250801_165548_86_4001_ortho_sr_hdf5.h5",
        r"C:\satelliteImagery\Tanager\SourceData\20250801_165553_11_4001\20250801_165553_11_4001_ortho_sr_hdf5.h5"
    ],
    [
        r"C:\satelliteImagery\Tanager\SourceData\20250903_164714_08_4001\20250903_164714_08_4001_ortho_sr_hdf5.h5",
        r"C:\satelliteImagery\Tanager\SourceData\20250903_164719_92_4001\20250903_164719_92_4001_ortho_sr_hdf5.h5"
    ],
    [
        r"C:\satelliteImagery\Tanager\SourceData\20250919_170233_04_4001\20250919_170233_04_4001_ortho_sr_hdf5.h5",
        r"C:\satelliteImagery\Tanager\SourceData\20250919_170238_88_4001\20250919_170238_88_4001_ortho_sr_hdf5.h5"
    ],
    [
        r"C:\satelliteImagery\Tanager\SourceData\20251001_165049_07_4001\20251001_165049_07_4001_ortho_sr_hdf5.h5",
        r"C:\satelliteImagery\Tanager\SourceData\20251001_165054_91_4001\20251001_165054_91_4001_ortho_sr_hdf5.h5"
    ],
    [
        r"C:\satelliteImagery\Tanager\SourceData\20251015_165255_92_4001\20251015_165255_92_4001_ortho_sr_hdf5.h5",
        r"C:\satelliteImagery\Tanager\SourceData\20251015_165301_89_4001\20251015_165301_89_4001_ortho_sr_hdf5.h5"
    ],
    [
        r"C:\satelliteImagery\Tanager\SourceData\20251029_165455_83_4001\20251029_165455_83_4001_ortho_sr_hdf5.h5",
        r"C:\satelliteImagery\Tanager\SourceData\20251029_165501_94_4001\20251029_165501_94_4001_ortho_sr_hdf5.h5"
    ]
]

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

def percentile_stretch(band_data, lower_pct=1, upper_pct=99):
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
        r_norm = percentile_stretch(r_band)
        g_norm = percentile_stretch(g_band)
        b_norm = percentile_stretch(b_band)
        
        rgb_composite = np.stack([r_norm, g_norm, b_norm], axis=-1)
        extent = get_geospatial_extent(f)
        
        # Explicitly cast indices to standard Python integers for clean string formatting
        return rgb_composite, extent, (int(r_idx), int(g_idx), int(b_idx))

def main():
    
    for files in file_groups:
        date = files[0].split("\\")[-2][0:8]
        figureName = "Tanager_Sequential_Acquisitions_" + date + ".png"
        print(f"\n--- Processing grouping for date: {date} ---")

        rgb_list = []
        extent_list = []
        band_indices = None
        
        for file in files:
            rgb, extent, indices = load_and_process_scene(file)
            rgb_list.append(rgb)
            extent_list.append(extent)
            if band_indices is None:
                band_indices = indices

        # Visualization
        fig, ax = plt.subplots(figsize=(12, 10))
        
        formatted_date = f"{date[0:4]}-{date[4:6]}-{date[6:8]}"
        fig.suptitle(f"Sequential Tanager Acquisitions in Shared Geographic Space\nAcquisition Date: {formatted_date} | RGB Band Indices: {band_indices}", fontsize=14, fontweight='bold')

        # Calculate global limits to encompass all images in the shared view
        global_xmin = min(extent[0] for extent in extent_list)
        global_xmax = max(extent[1] for extent in extent_list)
        global_ymin = min(extent[2] for extent in extent_list)
        global_ymax = max(extent[3] for extent in extent_list)

        # Display images on the single axis sequentially.
        # Since invalid pixels were set to NaN, they will be transparent, allowing
        # earlier images to show through where the later image has no data.
        for rgb, extent in zip(rgb_list, extent_list):
            ax.imshow(rgb, extent=extent, origin='upper')

        ax.set_xlabel("Easting (Meters)")
        ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%1.0f'))
        ax.set_ylabel("Northing (Meters)")
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%1.0f'))

        ax.set_xlim(global_xmin, global_xmax)
        ax.set_ylim(global_ymin, global_ymax)
        plt.tight_layout()
        
        print(f"Saving figure: {figureName}")
        plt.savefig(figureName, dpi=300)
        
        # Explicitly close the figure to free memory between loop iterations
        plt.close(fig)
        
    print("\nAll sequential acquisition figures generated successfully.")

if __name__ == "__main__":
    main()