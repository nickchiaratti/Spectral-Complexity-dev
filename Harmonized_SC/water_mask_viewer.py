import os
import argparse
import yaml
import h5py
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timezone
import glob

# ==========================================
# CONFIGURATION
# ==========================================
script_dir = os.path.dirname(os.path.abspath(__file__))
LOCATION = "Rochesterv2"
CONFIG_FILE_PATH = os.path.join(script_dir, "locations_config.yaml")
if not os.path.exists(CONFIG_FILE_PATH):
    CONFIG_FILE_PATH = os.path.join(os.path.dirname(script_dir), "locations_config.yaml")
SATELLITE_DATA_DIR = "C:/satelliteImagery/HLST30"

def get_file_path(location):
    """
    Attempt to find the correct base HDF5 file path for the location.
    The exact filename can vary (some include year, some don't).
    """
    # Search for any Harmonized files for this location
    matches = glob.glob(os.path.join(SATELLITE_DATA_DIR, f"HLST_{location}_Harmonized*.h5"))
    
    if matches:
        # Sort so that SC_EM files are preferred and selected first
        matches.sort(key=lambda x: "SC_EM" in os.path.basename(x), reverse=True)
        return matches[0]
        
    # Default fallback
    base_name = f"HLST_{location}_Harmonized.h5"
    return os.path.join(SATELLITE_DATA_DIR, base_name)

def main():
    location = LOCATION
    print(f"Using location: {location}")

    file_path = get_file_path(location)
    print(f"Opening data cube: {file_path}")
    
    base_file_path = os.path.join(SATELLITE_DATA_DIR, f"HLST_{location}_Harmonized.h5")
    
    if not os.path.exists(file_path) or not os.path.exists(base_file_path):
        print("ERROR: Required HDF5 files not found! Ensure the pipeline has run and base cubes exist.")
        return

    with h5py.File(file_path, 'r') as h5, h5py.File(base_file_path, 'r') as h5_base:
        harm_grp = h5['/HDFEOS/GRIDS/HARMONIZED/Data Fields']
        ortho_visual_ds = harm_grp['ortho_visual'] 
        
        # Extract global timeline provenance from ortho_visual (since common_mask is in SC cube)
        times = ortho_visual_ds.attrs['acquisition_time']
        indices = ortho_visual_ds.attrs['source_frame_index']
        raw_grids = ortho_visual_ds.attrs['source_grid']
        grids = [g.decode('utf-8') if isinstance(g, bytes) else str(g) for g in raw_grids]
        
        num_frames, _, height, width = ortho_visual_ds.shape
        
        # 1. Find frame closest to 2025-09-10
        target_dt = datetime(2025, 9, 10, tzinfo=timezone.utc)
        target_timestamp = target_dt.timestamp()
        
        time_diffs = np.abs(times - target_timestamp)
        closest_idx = np.argmin(time_diffs)
        
        closest_dt = datetime.fromtimestamp(times[closest_idx], tz=timezone.utc)
        print(f"Closest frame to 2025-09-10 is index {closest_idx} (Date: {closest_dt.strftime('%Y-%m-%d %H:%M:%S')})")
        
        # ortho_visual is typically shape (Time, Band, Y, X)
        closest_ortho = ortho_visual_ds[closest_idx]
        # Ensure we only take the first 3 bands (RGB/BGR) and reorder to (Y, X, Bands)
        img_visual = np.transpose(closest_ortho[:3, :, :], (1, 2, 0))
        
        # 2. Accumulate water mask over time
        print("Accumulating water mask over time, ignoring fill data...")
        water_sum = np.zeros((height, width), dtype=np.int32)
        valid_sum = np.zeros((height, width), dtype=np.int32)
        
        for t in range(num_frames):
            grid_name = grids[t]
            local_idx = indices[t]
            
            # Pivot to native grid to access FMASK
            # Tanager datasets won't have FMASK, so we safely check if it exists
            fmask_path = f"/HDFEOS/GRIDS/{grid_name}/Data Fields/Fmask"
            if fmask_path in h5_base:
                fmask_frame = h5_base[fmask_path][local_idx, :, :]
                
                # Identify valid pixels (FillValue is 255)
                valid_pixels = (fmask_frame != 255)
                
                # Extract bit 5 (value 32)
                water_pixels = ((fmask_frame & 32) > 0)
                
                water_sum[valid_pixels] += water_pixels[valid_pixels]
                valid_sum[valid_pixels] += 1
                
        print("Accumulation complete.")
        
        # 3. Visualization
        fig, axes = plt.subplots(1, 3, figsize=(22, 7))
        
        axes[0].imshow(img_visual)
        axes[0].set_title(f"Ortho Visual\n{closest_dt.strftime('%Y-%m-%d')}")
        axes[0].axis('off')
        
        # Create a masked array to hide regions with zero valid observations
        water_sum_masked = np.ma.masked_where(valid_sum == 0, water_sum)
        
        im1 = axes[1].imshow(water_sum_masked, cmap='Blues')
        axes[1].set_title(f"Temporally Summed Open Water (Fmask Bit 5)\nMax Sum: {np.max(water_sum)}")
        axes[1].axis('off')
        
        cbar1 = plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        cbar1.set_label('Water Detection Count')
        
        binary_mask_masked = water_sum_masked >= (1/3*np.max(water_sum))
        
        im2 = axes[2].imshow(binary_mask_masked, cmap='gray', vmin=0, vmax=1)
        axes[2].set_title("Persistent Water Mask\n(>= 33% of Valid Frames)")
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}")
