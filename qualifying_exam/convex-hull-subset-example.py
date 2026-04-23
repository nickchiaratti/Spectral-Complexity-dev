import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from skimage import exposure
from mpl_toolkits.mplot3d import Axes3D
import SpecComplex as sc

# --- Configuration ---
FILE_PATH = "C:/satelliteImagery/LANDSAT/Tait/LANDSAT_Stack_Tait_HDFEOS_SC_EM-7_Gram-corrected_Norm-None_QA-Strict.h5"
FRAME_IDX = 3  # 0-based index for the 4th frame
SUBSET_SIZE = 10
NUM_ENDMEMBERS = 7

# Coordinates (x, y) = (Column, Row)
SUBSET_CONFIGS = [
    {'x': 122, 'y': 47, 'label': 'Subset 1: Nature Area [122, 47]'},
    {'x': 52, 'y': 96, 'label': 'Subset 2: Shopping Center [52, 96]'}
]

# Visualization Bands
LANDSAT_RGB_BANDS = (3, 2, 1)  # Red, Green, Blue
HULL_BANDS = (4, 3, 2)         # SWIR2, NIR, Green (for 3D Projection)

def percentile_normalize_array(arr, low=2, high=98):
    """Normalizes array for RGB display."""
    if np.all(np.isnan(arr)): return np.zeros_like(arr)
    p_low, p_high = np.nanpercentile(arr, (low, high))
    if p_low == p_high: return np.zeros_like(arr)
    return exposure.rescale_intensity(arr, in_range=(p_low, p_high), out_range=(0, 1)).clip(0, 1)

def extract_subset(data, center_x, center_y, size):
    """
    Extracts a spatial subset from the data cube.
    Data format expected: [Bands, Height, Width]
    """
    half = size // 2
    x1 = int(center_x - half)
    x2 = int(x1 + size)
    y1 = int(center_y - half)
    y2 = int(y1 + size)
    
    # Check bounds
    _, h, w = data.shape
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    
    return data[:, y1:y2, x1:x2]

def main():
    print(f"Opening file: {FILE_PATH}")
    
    try:
        with h5py.File(FILE_PATH, 'r') as h5:
            # Locate Data
            grid_keys = list(h5['/HDFEOS/GRIDS'].keys())
            if 'LANDSAT' not in grid_keys:
                print("Error: LANDSAT grid not found.")
                return
            
            data_grp = h5['/HDFEOS/GRIDS/LANDSAT/Data Fields']
            sr_dset = data_grp['surface_reflectance']
            wavelengths = sr_dset.attrs.get('wavelengths', np.arange(sr_dset.shape[1]))
            
            print(f"Loading Frame {FRAME_IDX + 1}...")
            # Load entire frame into memory [Bands, Y, X]
            frame_data = sr_dset[FRAME_IDX, ...]
            
            # Setup Plotting
            fig = plt.figure(figsize=(18, 10))
            fig.suptitle(f"Spectral Complexity Analysis: 10x10 Subsets (Frame {FRAME_IDX + 1})", fontsize=16)
            
            rows = len(SUBSET_CONFIGS)
            cols = 4 # RGB, Spectral, Curve, 3D Hull
            
            for i, config in enumerate(SUBSET_CONFIGS):
                print(f"\n--- Processing {config['label']} ---")
                
                # 1. Extract Subset
                subset = extract_subset(frame_data, config['x'], config['y'], SUBSET_SIZE)
                n_bands, h, w = subset.shape
                print(f"Subset shape: {subset.shape}")
                
                # 2. Reshape for Calculation [N_Pixels, N_Bands]
                # Transpose to [Y, X, Bands] then flatten spatial dims
                flat_data = subset.transpose(1, 2, 0).reshape(-1, n_bands)
                
                # 3. Calculate Complexity Metrics
                # Using SpecComplex directly, bypassing tile/sliding logic
                endmembers, em_indices, volumes = sc.maximumDistance(
                    flat_data, NUM_ENDMEMBERS, gram='corrected', normalization=None
                )
                
                # --- VISUALIZATION ---
                row_offset = i * cols
                
                # A. True Color RGB
                ax_rgb = fig.add_subplot(rows, cols, row_offset + 1)
                r = percentile_normalize_array(subset[LANDSAT_RGB_BANDS[0]])
                g = percentile_normalize_array(subset[LANDSAT_RGB_BANDS[1]])
                b = percentile_normalize_array(subset[LANDSAT_RGB_BANDS[2]])
                rgb = np.dstack((r, g, b))
                ax_rgb.imshow(rgb)#, interpolation='nearest')
                ax_rgb.set_title(f"{config['label']}")
                ax_rgb.axis('off')
                
                # B. Spectral Signatures
                ax_spec = fig.add_subplot(rows, cols, row_offset + 2)
                for e_idx in range(endmembers.shape[1]):
                    sig = endmembers[:, e_idx]
                    # Only plot valid endmembers (not all zeros)
                    if np.any(sig != 0):
                        ax_spec.plot(wavelengths, sig, label=f'EM {e_idx}')
                ax_spec.set_title("Endmember Signatures")
                ax_spec.set_xlabel("Wavelength (um)")
                ax_spec.set_ylabel("Reflectance")
                ax_spec.grid(True, alpha=0.3)
                # ax_spec.legend(fontsize='x-small') 
                
                # C. Complexity Curve
                ax_vol = fig.add_subplot(rows, cols, row_offset + 3)
                ax_vol.plot(np.arange(1, len(volumes) + 1), np.pad(volumes[2:], (2,0), 'constant', constant_values=0), 'o-', color='green')
                ax_vol.set_title("Complexity (Volume) Curve")
                ax_vol.set_xlabel("Endmember Count")
                ax_vol.set_ylabel("Volume")
                ax_vol.grid(True, alpha=0.3)
                
                # D. 3D Convex Hull
                ax_hull = fig.add_subplot(rows, cols, row_offset + 4, projection='3d')
                
                # Indices for projection
                b1, b2, b3 = HULL_BANDS
                
                # Plot all pixels in subset (grey cloud)
                # Filter NaNs for plotting
                valid_pixels = flat_data[~np.isnan(flat_data).any(axis=1)]
                ax_hull.scatter(valid_pixels[:, b1], valid_pixels[:, b2], valid_pixels[:, b3], 
                                c='gray', alpha=0.3, s=10, label='Pixels')
                
                # Plot Endmembers (red dots)
                em_subset = endmembers[[b1, b2, b3], :]
                ax_hull.scatter(em_subset[0, 0:4], em_subset[1, 0:4], em_subset[2, 0:4], 
                                c='red', s=50, label='Endmembers')
                
                # Plot Convex Hull Wireframe
                try:
                    # Need at least 4 points for 3D hull
                    points_3d = em_subset[:,0:4].T
                    valid_em_mask = ~np.all(points_3d == 0, axis=1)
                    points_3d = points_3d[valid_em_mask]
                    
                    if points_3d.shape[0] >= 4:
                        hull = ConvexHull(points_3d)
                        for s in hull.simplices:
                            # Close the loop for drawing
                            s = np.append(s, s[0])
                            ax_hull.plot(points_3d[s, 0], points_3d[s, 1], points_3d[s, 2], 'r-', alpha=0.5)
                except Exception as e:
                    print(f"Could not compute hull for {config['label']}: {e}")

                ax_hull.set_title(f"3D Hull (Bands {b1},{b2},{b3})")
                ax_hull.set_xlim(0, 0.5)
                ax_hull.set_ylim(0, 0.5)
                ax_hull.set_zlim(0, 0.5)
                ax_hull.set_xlabel(f"Band {b1}")
                ax_hull.set_ylabel(f"Band {b2}")
                ax_hull.set_zlabel(f"Band {b3}")
            
            plt.tight_layout()
            plt.show()
            
    except FileNotFoundError:
        print(f"Error: File not found at {FILE_PATH}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()