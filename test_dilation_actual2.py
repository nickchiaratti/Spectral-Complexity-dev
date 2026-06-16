import h5py
import numpy as np

file_path = r"C:\satelliteImagery\HLST30\HLST_Malibu_Harmonized.h5"
try:
    with h5py.File(file_path, 'r') as f:
        fm = f['HDFEOS/GRIDS/HLSS30/Data Fields/Fmask']
        sva = f['HDFEOS/GRIDS/HLSS30/Data Fields/solar_view_angles']
        
        found_idx = -1
        # Check dimensions
        is_4d = len(fm.shape) == 4
        
        for i in range(50, min(150, fm.shape[0])):
            frame = fm[i, 0, :, :] if is_4d else fm[i, :, :]
            clouds = np.sum((frame & 2) > 0)
            if 1000 < clouds < 100000:
                found_idx = i
                break
                
        if found_idx == -1:
            print("Could not find a suitable frame")
        else:
            print(f"Selected HLSS30 frame {found_idx}")
            fmask = fm[found_idx, 0, :, :] if is_4d else fm[found_idx, :, :]
            angles = sva[found_idx, :, :, :]
            
            qa_reject_mask = 0b111111
            aerosol_accept_level = 'medium'
            sun_elevation_threshold = 30
            
            qa_invalid = (fmask & qa_reject_mask) != 0
            aerosol_bits = (fmask >> 6) & 0b11
            aerosol_invalid = aerosol_bits > 2
            
            sza = angles[0, :, :]
            sun_elev = 90.0 - sza
            sun_invalid = (sun_elev < sun_elevation_threshold) | np.isnan(sun_elev)
            
            base_mask = qa_invalid | aerosol_invalid | sun_invalid
            
            from scipy import ndimage
            kernel = np.ones((3, 3), dtype=bool)
            dilated_mask_2 = ndimage.binary_dilation(base_mask, structure=kernel, iterations=2)
            
            acq_time = f['HDFEOS/GRIDS/HLSS30/Data Fields/surface_reflectance'].attrs['acquisition_time'][found_idx]
            harm_times = f['HDFEOS/GRIDS/HARMONIZED/Data Fields/common_mask'].attrs['acquisition_time']
            harm_idx = np.where(harm_times == acq_time)[0]
            
            if len(harm_idx) > 0:
                harm_idx = harm_idx[0]
                actual_mask = f['HDFEOS/GRIDS/HARMONIZED/Data Fields/common_mask'][harm_idx, :, :]
                
                print(f"Pixels in base_mask (dilation=0): {np.sum(base_mask)}")
                print(f"Pixels in dilated_mask_2 (dilation=2): {np.sum(dilated_mask_2)}")
                print(f"Pixels in actual common_mask: {np.sum(actual_mask)}")
                
                match_0 = np.sum(base_mask == actual_mask)
                match_2 = np.sum(dilated_mask_2 == actual_mask)
                total = actual_mask.size
                
                print(f"Match with dilation=0: {match_0}/{total} ({match_0/total*100:.2f}%)")
                print(f"Match with dilation=2: {match_2}/{total} ({match_2/total*100:.2f}%)")
            else:
                print("Could not find corresponding harmonized frame")

except Exception as e:
    print(f"Error: {e}")
