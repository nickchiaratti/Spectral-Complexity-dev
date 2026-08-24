with open(r'F:\Resilio\IMGS 890 Research\Spectral-Complexity-dev\change_detection\harmonized-CCD\generate_pca_loadings_plot.py', 'r') as f:
    text = f.read()

target = '''    with h5py.File(source_h5_path, 'r') as f_sc:
        for seg_idx, seg in enumerate(segments):'''

replacement = '''    import re
    raw_h5_path = re.sub(r'_SC_.*?\.h5$', '.h5', source_h5_path)
    if not os.path.exists(raw_h5_path):
        print(f"Error: Raw H5 file not found at {raw_h5_path}")
        return
        
    with h5py.File(raw_h5_path, 'r') as f_sc:
        for seg_idx, seg in enumerate(segments):'''

text = text.replace(target, replacement)
with open(r'F:\Resilio\IMGS 890 Research\Spectral-Complexity-dev\change_detection\harmonized-CCD\generate_pca_loadings_plot.py', 'w') as f:
    f.write(text)
print("Patched raw h5 path")
