import h5py
import numpy as np

file_path = r"C:\satelliteImagery\share2012\SpecTIR\SpecTIR_Ortho_Stack_0920-1706.h5"
base_path = "HDFEOS/GRIDS/SPECTIR/Data Fields/"
keys = ['ssd_noise_cov', 'mnf_noise_cov', 'mlr_noise_cov', 'dct_noise_cov']

def ascii_plot(data, width=80, height=20):
    if len(data) == 0: return ""
    min_val, max_val = np.min(data), np.max(data)
    if max_val == min_val: return "Flat line"
    
    # Downsample to width
    indices = np.linspace(0, len(data) - 1, width).astype(int)
    sampled = data[indices]
    
    plot = [[' ' for _ in range(width)] for _ in range(height)]
    
    for x, val in enumerate(sampled):
        y = int((val - min_val) / (max_val - min_val) * (height - 1))
        plot[height - 1 - y][x] = '*'
        
    return '\n'.join(''.join(row) for row in plot)

try:
    with h5py.File(file_path, 'r') as f:
        for key in keys:
            full_path = base_path + key
            cov = f[full_path][:]
            diag = np.diag(cov)
            print(f"\n--- {key} ---")
            print(ascii_plot(diag, width=100, height=15))
except Exception as e:
    print("Error:", e)
