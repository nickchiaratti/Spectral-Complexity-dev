import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d import Axes3D
import tkinter as tk
from tkinter import filedialog
from datetime import datetime, timezone
import NSC_toolbox as nsc

# --- Configuration ---
RGB_BANDS = (3, 2, 1)  # Landsat 8/9: Red, Green, Blue
THREED_BANDS = (2,4,6)
PIXEL_SAMPLE_SIZE = 5000 # Downsample for scatter plots to maintain performance

# QA Bitmasks for filtering
QA_BAD_MASK = (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3) | (1 << 4) | (1 << 5)

def get_metadata_str(h5_file, idx):
    """Extracts timestamp and spacecraft ID from HDF5 attributes."""
    try:
        dset = h5_file['HDFEOS/GRIDS/LANDSAT/Data Fields/surface_reflectance']
        ts = dset.attrs['acquisition_time'][idx]
        sat_id = dset.attrs['spacecraft_id'][idx]
        
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        time_str = dt.strftime('%Y-%m-%dT%H:%M:%S')
        
        if isinstance(sat_id, bytes):
            sat_id = sat_id.decode('ascii')
            
        return f"{sat_id} - {time_str}"
    except Exception as e:
        return f"Frame {idx}"

def visualize_hull(h5_path, frame_idx=0):
    print(f"Loading data from frame {frame_idx}...")
    
    with h5py.File(h5_path, 'r') as h5:
        # Load Datasets
        grid_path = 'HDFEOS/GRIDS/LANDSAT/Data Fields'
        sr_dset = h5[f"{grid_path}/surface_reflectance"]
        em_dset = h5[f"{grid_path}/endmembers"]
        idx_dset = h5[f"{grid_path}/endmember_indices"]
        qa_dset = h5[f"{grid_path}/QUALITY_L1_PIXEL"]
        
        num_frames, num_bands, height, width = sr_dset.shape
        if frame_idx >= num_frames:
            print(f"Frame {frame_idx} out of bounds (max {num_frames-1})")
            return

        # Load Frame Data
        # Shape: (Bands, Height, Width) -> Transpose to (Height, Width, Bands)
        img_cube = np.transpose(sr_dset[frame_idx, ...], (1, 2, 0))
        qa_mask = qa_dset[frame_idx, ...]
        endmembers = em_dset[frame_idx, ...] # Shape: (Bands, Endmembers)
        em_indices = idx_dset[frame_idx, ...] # Shape: (Endmembers,)
        
        # Valid Mask (Clear pixels only)
        valid_mask = (qa_mask & QA_BAD_MASK) == 0
        
        # Flatten and Filter
        pixels = img_cube.reshape(-1, num_bands)
        valid_flat = valid_mask.flatten()
        
        # Extract only valid pixels for PCA/Hull
        clean_pixels = pixels[valid_flat]
        
        if clean_pixels.shape[0] < 100:
            print("Not enough valid pixels in this frame for analysis.")
            return

        print(f"Processing {clean_pixels.shape[0]} valid pixels...")

        # --- PCA Projection ---
        pca = PCA(n_components=3)
        # Fit PCA on valid pixels
        pixel_proj = pca.fit_transform(clean_pixels)
        # Project Endmembers using the SAME transform (Endmembers [Bands, N] -> Transpose to [N, Bands])
        em_proj = pca.transform(endmembers.T) 

        # Downsample for plotting
        if pixel_proj.shape[0] > PIXEL_SAMPLE_SIZE:
            indices = np.random.choice(pixel_proj.shape[0], PIXEL_SAMPLE_SIZE, replace=False)
            plot_pixels_pca = pixel_proj[indices]
        else:
            plot_pixels_pca = pixel_proj

        # --- Visualization Setup ---
        fig = plt.figure(figsize=(18, 6))
        meta_str = get_metadata_str(h5, frame_idx)
        fig.suptitle(f"Convex Hull & PCA Analysis: {meta_str}", fontsize=14)

        # --- Panel 1: Spatial Endmember Locations (Left) ---
        ax1 = fig.add_subplot(131)
        
        # Create True Color RGB
        r = nsc.percentile_normalize_array(img_cube[:, :, RGB_BANDS[0]])
        g = nsc.percentile_normalize_array(img_cube[:, :, RGB_BANDS[1]])
        b = nsc.percentile_normalize_array(img_cube[:, :, RGB_BANDS[2]])
        rgb = np.stack([r, g, b], axis=-1)
        rgb = np.nan_to_num(rgb, nan=0.0) 
        
        ax1.imshow(rgb)
        
        # Plot Endmember Locations
        for i, flat_idx in enumerate(em_indices):
            # Calculate 2D coordinates from flat index
            row = flat_idx // width
            col = flat_idx % width
            
            ax1.plot(col, row, 'r+', markersize=12, markeredgewidth=2)
            ax1.annotate(f'V{i}', (col, row), xytext=(3, 3), 
                                    textcoords='offset points', color='yellow', 
                                    fontweight='bold', fontsize=10)
            
        ax1.set_title("Endmember Locations (True Color)")
        ax1.axis('off')

        # --- Panel 2: PCA 2D (Center) ---
        ax2 = fig.add_subplot(132)
        ax2.scatter(plot_pixels_pca[:, 0], plot_pixels_pca[:, 1], c='gray', alpha=0.1, s=1, label='Pixels')
        
        # Calculate 2D Hull on Projected Endmembers
        try:
            # We compute hull on endmembers in PC space
            if em_proj.shape[0] >= 3:
                hull_2d = ConvexHull(em_proj[:, :2])
                for simplex in hull_2d.simplices:
                    ax2.plot(em_proj[simplex, 0], em_proj[simplex, 1], 'r--', lw=1)
        except Exception as e:
            print(f"2D Hull Error: {e}")

        # Plot Endmembers
        ax2.scatter(em_proj[:, 0], em_proj[:, 1], c='red', s=80, marker='*', label='Endmembers')
        for i in range(em_proj.shape[0]):
             ax2.annotate(f'V{i}', (em_proj[i, 0], em_proj[i, 1]), xytext=(5, 5), 
                          textcoords='offset points', color='darkred', fontweight='bold')

        ax2.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
        ax2.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
        ax2.set_title("2D PCA Mixing Space (PC1 vs PC2)")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # --- Panel 3: PCA 3D (Right) ---
        ax3 = fig.add_subplot(133, projection='3d')
        ax3.scatter(clean_pixels[:, THREED_BANDS[0]], clean_pixels[:, THREED_BANDS[1]], clean_pixels[:, THREED_BANDS[2]], c='gray', alpha=0.1, s=1, label='Image Pixels')
        em_3d = endmembers[list(THREED_BANDS), :].T
        ax3.scatter(em_3d[:, 0], em_3d[:, 1], em_3d[:, 2], c='red', s=80, marker='*', label='Endmembers')
        
        # Calculate 3D Hull on Endmembers
        try:
            hull_3d = ConvexHull(em_3d)
            for simplex in hull_3d.simplices:
                # Plot edges of simplices
                # simplex is indices of vertices forming a triangle face
                # Draw lines between them: 0-1, 1-2, 2-0
                pts = em_3d[simplex]
                # Close the loop
                pts = np.vstack([pts, pts[0]]) 
                ax3.plot(pts[:, 0], pts[:, 1], pts[:, 2], 'r-', lw=1, alpha=0.5)
        except Exception as e:
            print(f"3D Hull Error: {e}")

        ax3.set_xlabel(f"Band {THREED_BANDS[0]+1}")
        ax3.set_ylabel(f"Band {THREED_BANDS[1]+1}")
        ax3.set_zlabel(f"Band {THREED_BANDS[2]+1}")
        ax3.set_title(f"3D Spectral Projection (Bands {THREED_BANDS[0]+1}, {THREED_BANDS[1]+1}, {THREED_BANDS[2]+1})")
        ax3.legend()        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    
    file_path = filedialog.askopenfilename(
        title="Select Complexity Analysis HDF5",
        filetypes=[("HDF5", "*.h5")]
    )
    
    if file_path:
        # Currently defaults to frame 0, could be extended to have navigation like the other viewer
        visualize_hull(file_path, frame_idx=77)
    
    root.destroy()