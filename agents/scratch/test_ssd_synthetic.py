import numpy as np
import spectral

def estimate_noise_ssd(image_cube):
    img_hwc = np.transpose(image_cube, (1, 2, 0))
    n_right = spectral.noise_from_diffs(img_hwc, direction='right')
    return np.mean(np.diag(n_right.cov))

# 360 bands, 100x100 pixels
L = 360
H = 100
W = 100
endmembers = np.linspace(0.1, 0.9, L).reshape(L, 1) + np.sin(np.linspace(0, 10, L)).reshape(L, 1) * 0.1
endmembers = np.repeat(endmembers, 5, axis=1)
for i in range(5):
    endmembers[:, i] += np.cos(np.linspace(0, 5*(i+1), L)) * 0.05

# Generate abundances with spatial structure
x = np.linspace(0, 4*np.pi, W)
y = np.linspace(0, 4*np.pi, H)
xv, yv = np.meshgrid(x, y)
abundances = np.zeros((5, H, W))
abundances[0] = (np.sin(xv) + 1) / 2
abundances[1] = (np.cos(yv) + 1) / 2
abundances[2] = (np.sin(xv + yv) + 1) / 2
abundances[3] = (np.cos(xv - yv) + 1) / 2
abundances[4] = 1 - (abundances[0]+abundances[1]+abundances[2]+abundances[3])/4

# Normalize
abundances = abundances / np.sum(abundances, axis=0, keepdims=True)

# Image cube
Y_clean = np.tensordot(endmembers, abundances, axes=([1], [0])) # L x H x W

print("SSD on purely deterministic signal with spatial structure:", estimate_noise_ssd(Y_clean))
