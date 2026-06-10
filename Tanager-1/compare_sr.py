import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

SOURCE_FILE = "C:/satelliteImagery/Tanager/Rochesterv2_SourceData/Tanager_Native_Stack_Rochesterv2.h5"

def plot_comparison():
    if not os.path.exists(SOURCE_FILE):
        print(f"File not found: {SOURCE_FILE}")
        return

    with h5py.File(SOURCE_FILE, 'r', swmr=True) as f:
        grp = f['HDFEOS/GRIDS/TANAGER/Data Fields']
        
        if 'surface_reflectance' not in grp or 'surface_reflectance_py6s' not in grp:
            print("Both 'surface_reflectance' and 'surface_reflectance_py6s' must be present in the HDF5 file.")
            return
            
        sr_mfg = grp['surface_reflectance']
        sr_py6s = grp['surface_reflectance_py6s']
        
        n_times, n_bands, height, width = sr_mfg.shape
        fill_val = sr_mfg.attrs.get('_FillValue', -9999.0)
        if isinstance(fill_val, (np.ndarray, list, tuple)):
            fill_val = fill_val[0]
            
        # Get good wavelengths mask
        if 'good_wavelengths' in sr_mfg.attrs:
            good_bands = sr_mfg.attrs['good_wavelengths'] == 1
        else:
            print("Warning: 'good_wavelengths' attribute not found. Using all bands.")
            good_bands = np.ones(n_bands, dtype=bool)

        print(f"Using {np.sum(good_bands)} good bands out of {n_bands} for comparison.")

        out_dir = os.path.dirname(SOURCE_FILE)

        for t_idx in range(n_times):
            print(f"Generating 2D relative difference map for Frame {t_idx+1}/{n_times}...")
            
            # Load cubes for the current frame
            mfg_cube = sr_mfg[t_idx, :, :, :]
            py6s_cube = sr_py6s[t_idx, :, :, :]
            
            # Mask out fill values and bad bands
            # We want to process spatial pixels. Let's create an array to hold the mean relative difference.
            rel_diff_map = np.full((height, width), np.nan, dtype=np.float32)
            
            # Filter to only good bands
            mfg_good = mfg_cube[good_bands, :, :]
            py6s_good = py6s_cube[good_bands, :, :]
            
            # Valid mask: where BOTH are not fill value
            valid_mask = (mfg_good != fill_val) & (py6s_good != fill_val) & (mfg_good > 0.001) # Avoid div by zero on noisy 0.0 values
            
            # Calculate absolute relative difference
            # Formula: |Py6S - Mfg| / Mfg * 100
            diff = np.zeros_like(mfg_good, dtype=np.float32)
            np.divide(np.abs(py6s_good - mfg_good), mfg_good, out=diff, where=valid_mask)
            diff *= 100.0
            
            # Calculate the mean relative difference across the spectral axis (ignoring NaNs/zeros where invalid)
            diff[~valid_mask] = np.nan
            
            # Suppress RuntimeWarning for All-NaN slices
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                mean_rel_diff = np.nanmean(diff, axis=0)
            
            # Plot the 2D map
            plt.figure(figsize=(10, 8))
            
            # Use vmax to cap outliers (e.g. 50% difference) so the color scale is readable
            im = plt.imshow(mean_rel_diff, cmap='viridis', vmin=0, vmax=50)
            cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
            cbar.set_label('Mean Relative Difference (%)')
            
            plt.title(f"Mean Relative % Difference (Py6S vs Planet SR)\nFrame {t_idx} - Bad Bands Excluded")
            plt.xlabel("X (Pixels)")
            plt.ylabel("Y (Pixels)")
            
            out_path = os.path.join(out_dir, f"SR_RelDiff_Map_Frame_{t_idx}.png")
            plt.savefig(out_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  Saved {out_path}")

if __name__ == "__main__":
    plot_comparison()
