import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import tifffile
from pathlib import Path
import warnings

SOURCE_DIR = "C:/satelliteImagery/Tanager/Rochesterv2_SourceData"

class ComparisonViewer:
    def __init__(self, cache):
        self.cache = cache
        self.current_idx = 0
        self.fig, (self.ax_img, self.ax_spec) = plt.subplots(1, 2, figsize=(16, 7))
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.update_view()

    def update_view(self):
        self.ax_img.clear()
        self.ax_spec.clear()
        
        data = self.cache[self.current_idx]
        self.im = self.ax_img.imshow(data['diff_map'], cmap='viridis', vmin=0, vmax=50)
        self.ax_img.set_title(f"Pair {self.current_idx+1}/{len(self.cache)}: {data['name']}\nClick a pixel to see spectra. Use Left/Right arrows to navigate.")
        
        if not hasattr(self, 'cbar'):
            self.cbar = self.fig.colorbar(self.im, ax=self.ax_img, fraction=0.046, pad=0.04)
            self.cbar.set_label('Mean Relative Difference (%)')
        else:
            self.im.set_clim(vmin=0, vmax=50)
            
        self.ax_spec.set_title("Spectra (Click on image)")
        self.ax_spec.set_xlabel("Wavelength (nm)")
        self.ax_spec.set_ylabel("Reflectance")
        self.fig.canvas.draw()

    def on_key(self, event):
        if event.key == 'right':
            self.current_idx = (self.current_idx + 1) % len(self.cache)
            self.update_view()
        elif event.key == 'left':
            self.current_idx = (self.current_idx - 1) % len(self.cache)
            self.update_view()

    def on_click(self, event):
        if event.inaxes != self.ax_img: return
        if event.xdata is None or event.ydata is None: return
        
        x, y = int(event.xdata + 0.5), int(event.ydata + 0.5)
        data = self.cache[self.current_idx]
        
        height, width = data['diff_map'].shape
        if not (0 <= x < width and 0 <= y < height): return
        
        with h5py.File(data['planet_path'], 'r') as fp, h5py.File(data['py6s_path'], 'r') as fs:
            spec_planet = fp['HDFEOS/SWATHS/HYP/Data Fields/surface_reflectance'][:, y, x]
            spec_py6s = fs['HDFEOS/SWATHS/HYP/Data Fields/surface_reflectance'][:, y, x]
            
            fill_val = data['fill_val']
            # Plot all data, ignoring fill_val
            valid = (spec_planet != fill_val) & (spec_py6s != fill_val)
            wls = data['wavelengths']
            
            if data['has_sg']:
                spec_py6s_sg = fs['HDFEOS/SWATHS/HYP/Data Fields/surface_reflectance_SG'][:, y, x]
                
            self.ax_spec.clear()
            self.ax_spec.plot(wls[valid], spec_planet[valid], label='Planet SR', color='blue')
            self.ax_spec.plot(wls[valid], spec_py6s[valid], label='Py6S SR', color='red', linestyle='dashed')
            if data['has_sg']:
                self.ax_spec.plot(wls[valid], spec_py6s_sg[valid], label='Py6S SG Smoothed', color='orange', linestyle='dotted')
            
            # Fill bad bands with a vertical gray span
            good_bands = data['good_bands']
            bad_indices = np.where(~good_bands)[0]
            if len(bad_indices) > 0:
                contiguous_regions = np.split(bad_indices, np.where(np.diff(bad_indices) != 1)[0] + 1)
                for i, region in enumerate(contiguous_regions):
                    if len(region) == 0: continue
                    wl_start = wls[region[0]]
                    wl_end = wls[region[-1]]
                    self.ax_spec.axvspan(wl_start, wl_end, color='gray', alpha=0.3, label='Bad Bands' if i == 0 else "")
            self.ax_spec.set_title(f"Spectra at X={x}, Y={y}")
            self.ax_spec.set_xlabel("Wavelength (nm)")
            self.ax_spec.set_ylabel("Reflectance")
            self.ax_spec.legend()
            self.fig.canvas.draw()

def plot_comparison():
    source_path = Path(SOURCE_DIR)
    
    # Find all basic_6Ssr_hdf5.h5 files
    py6s_files = list(source_path.rglob("*_basic_6Ssr_hdf5.h5"))
    
    if not py6s_files:
        print("No processed 6S basic swath files found.")
        return

    cache = []
    
    for py6s_path in py6s_files:
        planet_name = py6s_path.name.replace("_basic_6Ssr_hdf5.h5", "_basic_sr_hdf5.h5")
        planet_path = py6s_path.parent / planet_name
        
        if not planet_path.exists():
            print(f"Matching Planet SR file not found for {py6s_path.name}")
            continue
            
        print(f"\nCaching: {planet_name}")
        
        with h5py.File(planet_path, 'r', swmr=True) as f_planet, h5py.File(py6s_path, 'r', swmr=True) as f_py6s:
            grp_planet = f_planet['HDFEOS/SWATHS/HYP/Data Fields']
            grp_py6s = f_py6s['HDFEOS/SWATHS/HYP/Data Fields']
            
            if 'surface_reflectance' not in grp_planet or 'surface_reflectance' not in grp_py6s:
                print("  Missing 'surface_reflectance' dataset. Skipping.")
                continue
                
            sr_planet = grp_planet['surface_reflectance']
            sr_py6s = grp_py6s['surface_reflectance']
            
            has_sg = 'surface_reflectance_SG' in grp_py6s
            
            fill_val = sr_planet.attrs.get('_FillValue', -9999.0)
            if isinstance(fill_val, (np.ndarray, list, tuple)):
                fill_val = fill_val[0]
                
            wavelengths = sr_planet.attrs.get('wavelengths', np.arange(sr_planet.shape[0]))
            
            if 'good_wavelengths' in sr_planet.attrs:
                good_bands = sr_planet.attrs['good_wavelengths'] == 1
            else:
                good_bands = np.ones(sr_planet.shape[0], dtype=bool)

            print(f"  Using {np.sum(good_bands)} good bands out of {sr_planet.shape[0]}.")
            
            planet_cube = sr_planet[()]
            py6s_cube = sr_py6s[()]
            
            planet_good = planet_cube[good_bands, :, :]
            py6s_good = py6s_cube[good_bands, :, :]
            
            valid_mask = (planet_good != fill_val) & (py6s_good != fill_val) & (planet_good > 0.001)
            
            diff = np.zeros_like(planet_good, dtype=np.float32)
            np.divide(np.abs(py6s_good - planet_good), planet_good, out=diff, where=valid_mask)
            diff *= 100.0
            diff[~valid_mask] = np.nan
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                mean_rel_diff = np.nanmean(diff, axis=0)
            
            out_tif_path = py6s_path.parent / py6s_path.name.replace("_basic_6Ssr_hdf5.h5", "_comparison_reldiff.tif")
            tif_data = np.nan_to_num(mean_rel_diff, nan=-9999.0).astype(np.float32)
            tifffile.imwrite(out_tif_path, tif_data)
            print(f"  Saved per-pixel difference TIF to {out_tif_path}")
            
            cache.append({
                'name': planet_name,
                'diff_map': mean_rel_diff,
                'planet_path': planet_path,
                'py6s_path': py6s_path,
                'wavelengths': wavelengths,
                'good_bands': good_bands,
                'fill_val': fill_val,
                'has_sg': has_sg
            })
            
    if not cache:
        print("No valid pairs successfully cached.")
        return
        
    print("\nStarting Interactive Viewer. Use Left/Right arrow keys to navigate pairs.")
    viewer = ComparisonViewer(cache)
    plt.show()

if __name__ == "__main__":
    plot_comparison()
