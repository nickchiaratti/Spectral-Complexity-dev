import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull
import tkinter as tk
from tkinter import filedialog
from datetime import datetime
import json

def get_timestamp(h5, idx):
    try:
        if "METADATA" in h5:
            meta_attr = f"frame_{idx}_json"
            if meta_attr in h5["METADATA"].attrs:
                meta_json = json.loads(h5["METADATA"].attrs[meta_attr])
                raw_time = meta_json['properties'].get('datetime', 'Unknown')
                dt = datetime.fromisoformat(raw_time.replace('Z', '+00:00'))
                return dt.strftime('%Y-%m-%dT%H:%M:%S.%f')
    except: pass
    return f"Frame {idx}"

def visualize_hull(h5_path, frame_idx=0):
    print(f"Loading data from frame {frame_idx}...")
    with h5py.File(h5_path, 'r') as h5:
        grid_path = '/HDFEOS/GRIDS/TANAGER/Data Fields'
        sr_dset = h5[f"{grid_path}/surface_reflectance"]
        em_dset = h5[f"{grid_path}/endmembers"]
        
        # Load Masks to ensure we only plot "Good" data
        # Using cloud/nodata/cirrus logic from your stacker
        cloud = h5[f"{grid_path}/beta_cloud_mask"][frame_idx, ...]
        nodata = h5[f"{grid_path}/nodata_pixels"][frame_idx, ...]
        valid_spatial = (cloud == 0) & (nodata == 0)
        
        # Load spectral data and endmembers
        # sr is (Bands, H, W) -> (H*W, Bands)
        bands, h, w = sr_dset.shape[1:]
        img_data = sr_dset[frame_idx, ...].reshape(bands, -1).T
        
        # Filter spatial no-data
        valid_img_data = img_data[valid_spatial.flatten()]
        
        # endmembers is (Bands, num_em) -> (num_em, Bands)
        ems = em_dset[frame_idx, ...].T 
        
        # Filter out NaN bands (spectral masking)
        # Identify bands that are not NaN in the first endmember
        good_bands_mask = ~np.isnan(ems[0])
        valid_img_data = valid_img_data[:, good_bands_mask]
        valid_ems = ems[:, good_bands_mask]

        timestamp = get_timestamp(h5, frame_idx)

    print("Performing PCA dimensionality reduction...")
    # Reduce to 3 components for 2D and 3D visualization
    pca = PCA(n_components=3)
    # Fit on all pixels to establish the spectral space
    pixel_proj = pca.fit_transform(valid_img_data)
    # Project endmembers into that same space
    em_proj = pca.transform(valid_ems)

    # Calculate Hull in 2D space (PC1 vs PC2)
    hull = ConvexHull(em_proj[:, :2])

    # Plotting
    fig = plt.figure(figsize=(14, 6))
    fig.suptitle(f"Spectral Convex Hull - {timestamp}\n{os.path.basename(h5_path)}", fontsize=14)

    # Subplot 1: 2D Projection
    ax1 = fig.add_subplot(121)
    # Plot ROI Pixels as a density cloud
    ax1.scatter(pixel_proj[:, 0], pixel_proj[:, 1], s=1, c='gray', alpha=0.1, label='ROI Pixels')
    # Plot Endmembers
    ax1.scatter(em_proj[:, 0], em_proj[:, 1], c='red', s=50, marker='*', edgecolors='black', label='Endmembers', zorder=5)
    
    # Draw Hull edges
    for simplex in hull.simplices:
        ax1.plot(em_proj[simplex, 0], em_proj[simplex, 1], 'r-', lw=2, alpha=0.8)
    
    # Label Endmembers
    for i in range(len(em_proj)):
        ax1.annotate(f'V{i}', (em_proj[i, 0], em_proj[i, 1]), xytext=(5, 5), textcoords='offset points', color='darkred', fontweight='bold')

    ax1.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax1.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax1.set_title("2D Spectral Mixing Space (PC1 vs PC2)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Subplot 2: 3D Projection
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(pixel_proj[::10, 0], pixel_proj[::10, 1], pixel_proj[::10, 2], s=1, c='gray', alpha=0.05)
    ax2.scatter(em_proj[:, 0], em_proj[:, 1], em_proj[:, 2], c='red', s=60, marker='*')
    
    # Simple lines for 3D hull structure visibility
    for simplex in ConvexHull(em_proj).simplices:
        for i in range(len(simplex)):
            start, end = simplex[i], simplex[(i+1)%len(simplex)]
            ax2.plot(em_proj[[start, end], 0], em_proj[[start, end], 1], em_proj[[start, end], 2], 'r-', alpha=0.5)

    ax2.set_xlabel("PC1")
    ax2.set_ylabel("PC2")
    ax2.set_zlabel("PC3")
    ax2.set_title("3D Spectral Mixing Space")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    root = tk.Tk(); root.withdraw()
    path = filedialog.askopenfilename(title="Select Calculated Tanager HDF5", filetypes=[("HDF5", "*.h5")])
    if path:
        # You can prompt for a frame index or just use 0
        visualize_hull(path, frame_idx=0)
    root.destroy()