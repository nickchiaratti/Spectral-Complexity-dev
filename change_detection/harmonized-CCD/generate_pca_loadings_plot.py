import os
import sys
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scienceplots
from datetime import datetime, timezone
from sklearn.decomposition import PCA
import glob

plt.style.use(['science', 'no-latex'])
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.dpi': 300
})
loc = "Tait"
px_y = 16
px_x = 76

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if root_dir not in sys.path:
    sys.path.append(root_dir)
import SpecComplex as sc

def enforce_sign_convention(components):
    """
    PCA eigenvectors have a sign ambiguity (v and -v are both valid).
    We enforce a convention: the element with the largest absolute magnitude 
    must be positive. This ensures consistent visual comparison between segments.
    """
    for i in range(components.shape[0]):
        max_idx = np.argmax(np.abs(components[i]))
        if components[i, max_idx] < 0:
            components[i] *= -1
    return components

def save_pca_eigenvector_spectra(pixel_y, pixel_x, source_h5_path, inference_results_h5, out_path):
    print("Generating PCA Eigenvector Spectra Plot...")
    
    with h5py.File(inference_results_h5, 'r') as f_inf:
        rmse_series = f_inf['rmse_series'][:, pixel_y, pixel_x]
        
    with h5py.File(source_h5_path, 'r') as f_sc:
        harm_grp = f_sc['/HDFEOS/GRIDS/HARMONIZED/Data Fields']
        target_metric = f_sc.attrs.get('TARGET_METRIC', 'sliding_volume_z_score')
        if target_metric not in harm_grp:
            target_metric = list(harm_grp.keys())[0]
            
        attrs = harm_grp[target_metric].attrs
        acq_time = attrs['acquisition_time'][:]
        source_grids = [s.decode('utf-8') if isinstance(s, bytes) else str(s) for s in attrs['source_grid'][:]]
        source_frames = attrs['source_frame_index'][:]
        spacecrafts = [s.decode('utf-8') if isinstance(s, bytes) else str(s) for s in attrs['source_spacecraft'][:]]
        unified_masks = harm_grp['common_mask'][:, pixel_y, pixel_x]

    # Identify structural segments based on RMSE plateaus
    segments = []
    current_seg = []
    current_rmse = None
    for i in range(len(acq_time)):
        if unified_masks[i]: continue
        r = rmse_series[i]
        if np.isnan(r): continue
        if current_rmse is None:
            current_rmse = r
            current_seg.append(i)
        elif abs(r - current_rmse) < 1e-6:
            current_seg.append(i)
        else:
            segments.append(current_seg)
            current_seg = [i]
            current_rmse = r
    if current_seg:
        segments.append(current_seg)

    if len(segments) < 2:
        print("Less than 2 segments found. Cannot perform pre/post break comparison.")
        return

    # Extract all spectral endmembers grouped by segment and sensor
    # We will restrict the PCA to Landsat (7 bands) to ensure mathematical alignment
    # because you cannot run a joint PCA on 7-band and 10-band data simultaneously.
    
    segment_data = {i: [] for i in range(len(segments))}
    wavelength_grid = None
    
    import re
    raw_h5_path = re.sub(r'_SC_.*?\.h5$', '.h5', source_h5_path)
    if not os.path.exists(raw_h5_path):
        print(f"Error: Raw H5 file not found at {raw_h5_path}")
        return
        
    with h5py.File(raw_h5_path, 'r') as f_sc:
        for seg_idx, seg in enumerate(segments):
            for i in seg:
                sensor = str(spacecrafts[i]).upper()
                if 'LANDSAT' not in sensor:
                    # Filter to Landsat for mathematically rigorous eigenvector shape matching
                    continue
                    
                grid = source_grids[i]
                frame_idx = source_frames[i]
                
                sr_ds = f_sc[f"/HDFEOS/GRIDS/{grid}/Data Fields/surface_reflectance"]
                patch = sr_ds[frame_idx, :, max(0, pixel_y-1):pixel_y+2, max(0, pixel_x-1):pixel_x+2]
                
                if wavelength_grid is None:
                    w_attr = sr_ds.attrs.get('wavelengths', sr_ds.attrs.get('wavelength'))
                    wavelength_grid = w_attr[:] if w_attr is not None else np.arange(1, patch.shape[0]+1, dtype=float)
                    if np.max(wavelength_grid) < 10: wavelength_grid *= 1000
                
                patch = np.transpose(patch, (1, 2, 0))
                # Flatten the 3x3 patch into 9 pixels instead of extracting endmembers, 
                # or extract the 4 endmembers to represent the bounding simplex.
                # Since we want to capture physical variance of the landscape, using all 9 valid pixels 
                # provides a robust covariance matrix.
                valid_mask = np.all(~np.isnan(patch), axis=-1)
                valid_pixels = patch[valid_mask]
                
                if valid_pixels.size > 0:
                    segment_data[seg_idx].append(valid_pixels)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"PCA Eigenvector Spectra Pre and Post Structural Break\n(Landsat Features | Pixel: x={pixel_x}, y={pixel_y})", fontsize=16)
    
    segment_colors = ['blue', 'red', 'green', 'purple']
    
    for seg_idx in range(min(3, len(segments))):  # Only plot first 3 segments if many exist
        if not segment_data[seg_idx]:
            continue
            
        # Concatenate all valid pixels for the entire time-span of the segment
        X = np.vstack(segment_data[seg_idx])
        
        # Fit PCA
        pca = PCA(n_components=3)
        pca.fit(X)
        
        # Extract Eigenvectors and enforce sign convention
        components = enforce_sign_convention(pca.components_)
        explained_var = pca.explained_variance_ratio_ * 100
        
        seg_color = segment_colors[seg_idx % len(segment_colors)]
        
        for pc_idx in range(3):
            ax = axes[pc_idx]
            eigenvector = components[pc_idx]
            var = explained_var[pc_idx]
            
            ax.plot(wavelength_grid, eigenvector, color=seg_color, marker='o', linewidth=2, 
                    label=f"Seg {seg_idx+1} ({var:.1f}% var)")
            
            ax.set_title(f"Principal Component {pc_idx+1}")
            ax.set_xlabel("Wavelength (nm)")
            ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
            
            if pc_idx == 0:
                ax.set_ylabel("Loading Weight")
            ax.legend()
            
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    from harmonized_CCD_main import LOCATION, H5_PATH, ENABLE_CONSTANT, ENABLE_LINEAR, ENABLE_QUADRATIC, TEMPORAL_PERIODS, TARGET_METRIC
    
    
    
    term_str = f"C{int(ENABLE_CONSTANT)}L{int(ENABLE_LINEAR)}Q{int(ENABLE_QUADRATIC)}"
    period_str = f"P{len(TEMPORAL_PERIODS)}"
    config = f"{term_str}_{period_str}"
    
    def get_inference_h5(location, config, target_metric):
        search_pattern = f"C:/satelliteImagery/HLST30/CCD/{location}_CCD_Harmonized_Change_Detection_{target_metric}_{config}.h5"
        files = glob.glob(search_pattern)
        if not files: return None
        files.sort(key=os.path.getmtime, reverse=True)
        return files[0]

    inf_h5 = get_inference_h5(loc, config, TARGET_METRIC)
    source_h5 = H5_PATH
        
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'presentation_plots')
    os.makedirs(out_dir, exist_ok=True)
    
    out_path = os.path.join(out_dir, f'pca_eigenvectors_y{px_y}_x{px_x}.png')
    save_pca_eigenvector_spectra(px_y, px_x, source_h5, inf_h5, out_path)
    
    print(f"Saved PCA Eigenvector Spectra to: {out_path}")
