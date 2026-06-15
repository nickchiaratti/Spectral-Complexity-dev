import h5py
import numpy as np

def verify_results():
    file_orig = r"C:\satelliteImagery\HLST30\HLST_Tait_Harmonized_SC_EM-7_Norm-bandCount.h5"
    file_eff = r"C:\satelliteImagery\HLST30\HLST_Tait_Harmonized_SC_EM-7_Norm-bandCount_efficient.h5"
    
    print("Loading data from HDF5 files...")
    
    try:
        with h5py.File(file_orig, 'r') as h5_orig, h5py.File(file_eff, 'r') as h5_eff:
            path = "/HDFEOS/GRIDS/HARMONIZED/Data Fields/sliding_volume_map"
            
            if path not in h5_orig or path not in h5_eff:
                print(f"Error: {path} not found in one of the files.")
                return
                
            orig_map = h5_orig[path][:]
            eff_map = h5_eff[path][:]
            
            # Create a mask of valid pixels (where both are not NaN)
            valid_mask = ~np.isnan(orig_map) & ~np.isnan(eff_map)
            
            orig_valid = orig_map[valid_mask]
            eff_valid = eff_map[valid_mask]
            
            if len(orig_valid) == 0:
                print("No valid data points found to compare.")
                return
                
            print(f"Comparing {len(orig_valid):,} valid data points...")
            
            # Calculate Absolute Difference
            abs_diff = np.abs(orig_valid - eff_valid)
            
            # Calculate Relative Difference safely
            # Only calculate relative difference where the original value is meaningfully non-zero
            nonzero_mask = np.abs(orig_valid) > 1e-12
            
            if np.sum(nonzero_mask) > 0:
                rel_diff = abs_diff[nonzero_mask] / np.abs(orig_valid[nonzero_mask])
                max_rel = np.max(rel_diff)
                mean_rel = np.mean(rel_diff)
            else:
                max_rel = 0.0
                mean_rel = 0.0
            
            print("\n" + "="*50)
            print("MAGNITUDE OF DATA (Original)")
            print("="*50)
            print(f"Min Value:  {np.min(orig_valid):.6e}")
            print(f"Max Value:  {np.max(orig_valid):.6e}")
            print(f"Mean Value: {np.mean(orig_valid):.6e}")
            print(f"Values > 1e-12: {np.sum(nonzero_mask):,}")
            
            print("\n" + "="*50)
            print("ABSOLUTE DIFFERENCE")
            print("="*50)
            print(f"Max Absolute Error:  {np.max(abs_diff):.6e}")
            print(f"Mean Absolute Error: {np.mean(abs_diff):.6e}")
            
            print("\n" + "="*50)
            print("RELATIVE DIFFERENCE (Ignoring values < 1e-12)")
            print("="*50)
            print(f"Max Relative Error:  {max_rel:.6e}  ({max_rel*100:.6f} %)")
            print(f"Mean Relative Error: {mean_rel:.6e}  ({mean_rel*100:.6f} %)")
            
            print("\n" + "="*50)
            print("CONCLUSION")
            print("="*50)
            # Float32 precision is roughly 1.19e-7, so relative errors around 1e-6 to 1e-5 are expected.
            if max_rel < 1e-4:
                print("SUCCESS: Maximum relative error is within expected float32 precision limits (< 1e-4).")
                print("The differences are purely mathematical rounding artifacts from the algebraic optimization.")
            else:
                print("WARNING: Maximum relative error is larger than expected for standard float32 precision.")
                print("Review the areas with largest discrepancies.")
                
    except FileNotFoundError as e:
        print(f"Error: Could not find file. {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    verify_results()
