import h5py

def print_keys(name, obj):
    print(name)

with h5py.File('C:/satelliteImagery/HLST30/HLST_Rochesterv2_Harmonized.h5', 'r') as f:
    f.visititems(print_keys)
