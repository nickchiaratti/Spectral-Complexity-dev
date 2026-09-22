import h5py
import numpy as np

def estimate_noise_mlr(image_cube):
    bands, height, width = image_cube.shape
    Y = image_cube.reshape(bands, -1).astype(np.float64)
    
    L, N = Y.shape
    small = 1e-6
    w = np.zeros((L, N), dtype=np.float64)
    
    RR = np.dot(Y, Y.T)
    RRi = np.linalg.pinv(RR + small * np.eye(L))
    
    for i in range(L):
        col = RRi[:, i:i+1]
        row = RRi[i:i+1, :]
        XX = RRi - np.dot(col, row) / RRi[i, i]
        
        RRa = RR[:, i].copy()
        RRa[i] = 0.0
        
        beta = np.dot(XX, RRa)
        beta[i] = 0.0
        
        w[i, :] = Y[i, :] - np.dot(beta, Y)
        
    sigma_n = np.diag(np.diag(np.dot(w, w.T) / N))
    return sigma_n

h5_path = r'C:\satelliteImagery\share2012\SpecTIR\SpecTIR_Ortho_Stack_0920-1706.h5'
with h5py.File(h5_path, 'r') as f:
    cube = f['HDFEOS/GRIDS/SPECTIR/Data Fields/surface_reflectance'][0, :, 100:200, 100:200]

print(f"Cube max: {np.max(cube)}, min: {np.min(cube)}")
sigma = estimate_noise_mlr(cube)
print("MLR noise mean diagonal on raw cube:", np.mean(np.diag(sigma)))

# Try scaling the cube to [0, 1] range to simulate proper reflectance
cube_scaled = cube.astype(np.float64) / 10000.0
sigma_scaled = estimate_noise_mlr(cube_scaled)
print("MLR noise mean diagonal on scaled cube:", np.mean(np.diag(sigma_scaled)))
print("Scaled cube MLR noise mean * 10000^2:", np.mean(np.diag(sigma_scaled)) * 10000**2)

# Check matrix condition number
Y = cube.reshape(cube.shape[0], -1).astype(np.float64)
RR = np.dot(Y, Y.T)
eigvals = np.linalg.eigvalsh(RR)
print("RR Condition number:", eigvals[-1] / (eigvals[0] + 1e-12))
