import h5py

file_path = r"C:\satelliteImagery\HLST30\HLST_Malibu_Harmonized.h5"
try:
    with h5py.File(file_path, 'r') as f:
        print("HLSS30 common_mask attrs:")
        if 'HDFEOS/GRIDS/HLSS30/Data Fields/common_mask' in f:
            mask_ds = f['HDFEOS/GRIDS/HLSS30/Data Fields/common_mask']
            for k, v in mask_ds.attrs.items():
                print(f"  {k}: {v}")
                
        print("\nTANAGER common_mask attrs:")
        if 'HDFEOS/GRIDS/TANAGER/Data Fields/common_mask' in f:
            mask_ds = f['HDFEOS/GRIDS/TANAGER/Data Fields/common_mask']
            for k, v in mask_ds.attrs.items():
                print(f"  {k}: {v}")
except Exception as e:
    print(f"Error: {e}")
