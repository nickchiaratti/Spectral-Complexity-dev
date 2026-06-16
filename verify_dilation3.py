import h5py

file_path = r"C:\satelliteImagery\HLST30\HLST_Malibu_Harmonized.h5"
try:
    with h5py.File(file_path, 'r') as f:
        print("Keys in original file:", list(f.keys()))
        if 'HDFEOS/GRIDS/HARMONIZED/Data Fields/common_mask' in f:
            mask_ds = f['HDFEOS/GRIDS/HARMONIZED/Data Fields/common_mask']
            print("HLST_Malibu_Harmonized.h5 common_mask attrs:")
            for k, v in mask_ds.attrs.items():
                print(f"  {k}: {v}")
except Exception as e:
    print(f"Error on original: {e}")
