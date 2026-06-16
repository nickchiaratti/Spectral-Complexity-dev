import h5py
import numpy as np

file_path = r"C:\satelliteImagery\HLST30\HLST_Malibu_Harmonized_SC_EM-7_Norm-bandCount.h5"
try:
    with h5py.File(file_path, 'r') as f:
        harm = f['HDFEOS/GRIDS/HARMONIZED']
        print("HARMONIZED keys:", list(harm.keys()))
        
        # Check attributes of HARMONIZED
        print("HARMONIZED attrs:")
        for k, v in harm.attrs.items():
            print(f"  {k}: {v}")
            
        # Also check common_mask attributes
        mask_ds = harm['Data Fields/common_mask']
        print("common_mask attrs:")
        for k, v in mask_ds.attrs.items():
            print(f"  {k}: {v}")
            
        # Also try to find timestamps anywhere
        def print_if_time(name, obj):
            if 'time' in name.lower() or 'date' in name.lower() or 'year' in name.lower():
                print(f"Found time related: {name}")
                if isinstance(obj, h5py.Dataset):
                    print(f"  shape: {obj.shape}, dtype: {obj.dtype}")
                    if obj.size > 0 and obj.size < 10000:
                        try:
                            # if string
                            print(f"  first few: {obj[:5]}")
                        except:
                            pass
        f.visititems(print_if_time)
        
        # Now let's extract the mask values for pixel x=644, y=464 (col=644, row=464)
        mask_values = mask_ds[:, 464, 644]
        print(f"Mask values for (464, 644) over time: unique values = {np.unique(mask_values, return_counts=True)}")
        
except Exception as e:
    print(f"Error: {e}")
