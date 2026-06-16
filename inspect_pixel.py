import h5py
import numpy as np

file_path = r"C:\satelliteImagery\HLST30\HLST_Malibu_Harmonized_SC_EM-7_Norm-bandCount.h5"
try:
    with h5py.File(file_path, 'r') as f:
        print("Keys at root:", list(f.keys()))
        if 'HDFEOS' in f:
            grids = f['HDFEOS/GRIDS']
            print("Grids available:", list(grids.keys()))
            
            # Let's see what arrays we have in the grids
            for grid_name in grids.keys():
                df = grids[grid_name]['Data Fields']
                print(f"--- {grid_name} Data Fields ---")
                for k, v in df.items():
                    if isinstance(v, h5py.Dataset):
                        print(f"  {k}: shape {v.shape}, dtype {v.dtype}")
                        
        print("\nChecking metadata or time information...")
        if 'METADATA' in f:
            print("Metadata items:", list(f['METADATA'].keys()))
            
except Exception as e:
    print(f"Error: {e}")
