import h5py

file_path = r"C:\satelliteImagery\HLST30\HLST_Malibu_Harmonized.h5"
try:
    with h5py.File(file_path, 'r') as f:
        if 'HDFEOS/GRIDS/HARMONIZED/Data Fields/common_mask' in f:
            mask_ds = f['HDFEOS/GRIDS/HARMONIZED/Data Fields/common_mask']
            print("HLST_Malibu_Harmonized.h5 common_mask attrs:")
            if 'cloud_dilation' in mask_ds.attrs:
                print("  cloud_dilation:", mask_ds.attrs['cloud_dilation'])
            if 'aerosol_accept_level' in mask_ds.attrs:
                print("  aerosol_accept_level:", mask_ds.attrs['aerosol_accept_level'])
except Exception as e:
    pass

file_path2 = r"C:\satelliteImagery\HLST30\HLST_Malibu_Harmonized_SC_EM-7_Norm-bandCount.h5"
try:
    with h5py.File(file_path2, 'r') as f:
        if 'HDFEOS/GRIDS/HARMONIZED/Data Fields/common_mask' in f:
            mask_ds = f['HDFEOS/GRIDS/HARMONIZED/Data Fields/common_mask']
            print("HLST_Malibu_Harmonized_SC_EM-7_Norm-bandCount.h5 common_mask attrs:")
            if 'cloud_dilation' in mask_ds.attrs:
                print("  cloud_dilation:", mask_ds.attrs['cloud_dilation'])
            if 'aerosol_accept_level' in mask_ds.attrs:
                print("  aerosol_accept_level:", mask_ds.attrs['aerosol_accept_level'])
except Exception as e:
    pass
