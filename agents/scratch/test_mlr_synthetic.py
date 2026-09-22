import h5py
import numpy as np

def estimate_noise_mlr(Y):
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
    return np.mean(np.diag(sigma_n))

# Generate a synthetic hyperspectral cube (very smooth)
# 360 bands, 10000 pixels
N = 10000
L = 360
# 5 endmembers
endmembers = np.linspace(0.1, 0.9, L).reshape(L, 1) + np.sin(np.linspace(0, 10, L)).reshape(L, 1) * 0.1
endmembers = np.repeat(endmembers, 5, axis=1)
# slightly change endmembers
for i in range(5):
    endmembers[:, i] += np.cos(np.linspace(0, 5*(i+1), L)) * 0.05

abundances = np.random.dirichlet(np.ones(5), N).T # 5 x N
Y_clean = np.dot(endmembers, abundances) # L x N

# Add independent Gaussian noise with variance 1e-4 (std 1e-2)
true_var = 1e-4
Y_noisy = Y_clean + np.random.normal(0, np.sqrt(true_var), Y_clean.shape)

mlr_noise_clean = estimate_noise_mlr(Y_clean)
mlr_noise_noisy = estimate_noise_mlr(Y_noisy)

print(f"True added variance: {true_var}")
print(f"MLR estimated variance on CLEAN data: {mlr_noise_clean}")
print(f"MLR estimated variance on NOISY data: {mlr_noise_noisy}")
