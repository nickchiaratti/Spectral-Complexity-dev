import os
import h5py
import numpy as np
try:
    from skimage.registration import phase_cross_correlation
except ImportError:
    print("Error: scikit-image is not installed. Please run: pip install scikit-image")
    exit(1)

def analyze_offsets(h5_path):
    print(f"Loading {h5_path}...")
    
    with h5py.File(h5_path, 'r') as h5:
        grp = h5["HDFEOS/GRIDS/GHOST/Data Fields"]
        rad_dset = grp["radiance"]
        mask_dset = grp["common_mask"]
        
        num_frames = rad_dset.shape[0]
        if num_frames < 2:
            print("Not enough frames to calculate offset.")
            return

        # Extract resolution from spatial_transform to convert pixels to meters
        st = rad_dset.attrs['spatial_transform']
        res_deg = st[1]
        
        # Approximate meters per degree for rough physical offset conversion
        center_lat = st[3] - (rad_dset.shape[2] / 2) * res_deg
        meters_per_deg_lat = 111320.0
        meters_per_deg_lon = 111320.0 * np.cos(np.radians(center_lat))

        # We will use a central green band for visual feature correlation
        band_idx = 34 
        
        print(f"\nAnalyzing Frame-to-Frame Offsets using Phase Cross-Correlation")
        print(f"Using Band {band_idx} for correlation.")
        print("-" * 65)
        
        total_shift_x = 0
        total_shift_y = 0

        for t in range(1, num_frames):
            ref_img = rad_dset[t-1, band_idx, :, :]
            mov_img = rad_dset[t, band_idx, :, :]
            
            ref_mask = mask_dset[t-1, :, :] == 1
            mov_mask = mask_dset[t, :, :] == 1
            
            # Replace NaNs with 0 to prevent math errors during correlation
            ref_clean = np.nan_to_num(ref_img, nan=0.0)
            mov_clean = np.nan_to_num(mov_img, nan=0.0)
            
            try:
                # Masked phase cross correlation limits calculations to valid data regions
                shift, error, diffphase = phase_cross_correlation(
                    ref_clean, mov_clean, 
                    reference_mask=ref_mask, moving_mask=mov_mask,
                    upsample_factor=10
                )
            except TypeError:
                # Fallback for older scikit-image versions that don't support masking
                shift, error, diffphase = phase_cross_correlation(
                    ref_clean, mov_clean, 
                    upsample_factor=10
                )

            # shift is returned as (shift_y, shift_x)
            shift_y, shift_x = shift
            
            dist_x_meters = shift_x * res_deg * meters_per_deg_lon
            dist_y_meters = shift_y * res_deg * meters_per_deg_lat
            
            total_shift_x += dist_x_meters
            total_shift_y += dist_y_meters
            
            print(f"Frame {t-1} vs Frame {t}:")
            print(f"  Pixel Shift:   X = {shift_x:>6.2f} px, Y = {shift_y:>6.2f} px")
            print(f"  Ground Shift:  Easting ~ {dist_x_meters:>6.2f} m, Northing ~ {-dist_y_meters:>6.2f} m")

        print("-" * 65)
        print("NOTE: Positive Easting means the new frame drifted East.")
        print("      Positive Northing means the new frame drifted North.")

if __name__ == "__main__":
    h5_file = r"C:\satelliteImagery\OSK-Ghost\SourceData\GHOST_Native_Stack_HDFEOS.h5"
    if os.path.exists(h5_file):
        analyze_offsets(h5_file)
    else:
        print(f"File not found: {h5_file}")
