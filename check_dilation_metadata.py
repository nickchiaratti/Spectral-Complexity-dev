import h5py

file_path = r"C:\satelliteImagery\HLST30\HLST_Malibu_Harmonized_SC_EM-7_Norm-bandCount.h5"
try:
    with h5py.File(file_path, 'r') as f:
        mask_ds = f['HDFEOS/GRIDS/HARMONIZED/Data Fields/common_mask']
        print("Attributes of common_mask:")
        for k, v in mask_ds.attrs.items():
            print(f"  {k}: {v}")
except Exception as e:
    print(f"Error: {e}")
