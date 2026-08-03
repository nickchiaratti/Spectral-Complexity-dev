import os
import sys
import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.append(r'F:\Resilio\IMGS 890 Research\Spectral-Complexity-dev')
from Harmonized_SC.HLST_skeleton_loader import HLST_ARD_Interface
import SpecComplexTorch as sct

def generate_maps():
    print("Loading datasets...")
    filepath = r'C:\satelliteImagery\HLST30\HLST_Tait_Harmonized_SC_EM-7_Norm-None.h5'
    ard = HLST_ARD_Interface(filepath)
    raw_h5 = h5py.File(r'C:\satelliteImagery\HLST30\HLST_Tait_Harmonized.h5', 'r')
    
    # t=570 corresponds to TANAGER 2025-09-19
    t = 570
    sensor_grid = ard.grids[t]
    local_idx = ard.indices[t]
    date_str = ard.times[t]
    if isinstance(date_str, bytes):
        date_str = date_str.decode('utf-8')
    print(f"Target frame: {sensor_grid} on {date_str} (t={t})")
    
    # Load surface reflectance
    native_ds = raw_h5[f'/HDFEOS/GRIDS/{sensor_grid}/Data Fields/surface_reflectance']
    sr = native_ds[local_idx, ...]
    
    # Apply good wavelengths mask
    good_bands = native_ds.attrs.get('good_wavelengths')
    if good_bands is not None:
        valid_indices = np.where(good_bands == 1)[0]
        sr = sr[valid_indices, :, :]
    
    # Convert to float32 (SpecComplexTorch precision)
    sr = sr.astype(np.float32)
    
    # No spatial or bounds masking applied as requested; 
    # all pixels will be processed.
    
    # Convert to torch tensor (C, H, W)
    frame_tensor = torch.from_numpy(sr)
    
    # Process sliding windows
    window_sizes = [2, 3, 5, 7]
    num_endmembers = 7
    gram_type = 'minEndmember'
    norm_type = 'None'
    
    output_dir = r'F:\Resilio\IMGS 890 Research\TGRS-Spectral-Complexity\media'
    os.makedirs(output_dir, exist_ok=True)
    
    non_avg_plotted = False
    
    for s in window_sizes:
        print(f"Processing {s}x{s} window...")
        avg_map, non_avg_map = sct.process_volume_sliding_tile(
            sr, 
            tile_size=s, 
            stride=1, 
            num_endmembers=num_endmembers, 
            gram_type=gram_type, 
            norm_type=norm_type
        )
        
        # Plot spatially averaged map
        plt.figure(figsize=(5, 5))
        plt.imshow(avg_map, cmap='viridis')
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(os.path.join(output_dir, f'image_avg_{s}x{s}.png'), bbox_inches='tight', pad_inches=0, dpi=600)
        plt.close()
        print(f"Saved image_avg_{s}x{s}.png")
        
        # For non-spatially averaged map, they are generated at each window size (with different edge padding)
        # We only need the one for 3x3 as the representative non-spatially averaged metric.
        if s == 3 and not non_avg_plotted:
            plt.figure(figsize=(5, 5))
            plt.imshow(non_avg_map, cmap='viridis')
            plt.axis('off')
            plt.tight_layout(pad=0)
            plt.savefig(os.path.join(output_dir, f'image_non_avg_{s}x{s}.png'), bbox_inches='tight', pad_inches=0, dpi=600)
            plt.close()
            print(f"Saved image_non_avg_{s}x{s}.png")
            non_avg_plotted = True

    raw_h5.close()
    print("Done!")

if __name__ == '__main__':
    generate_maps()
