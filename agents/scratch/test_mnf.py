import numpy as np
import spectral

# Create dummy image cube
np.random.seed(0)
img = np.random.rand(10, 10, 5)

signal_stats = spectral.calc_stats(img)
noise_stats = spectral.noise_from_diffs(img)

mnf_result = spectral.mnf(signal_stats, noise_stats)
print("noise_stats.cov:\n", noise_stats.cov)
print("mnf_result.noise.cov:\n", mnf_result.noise.cov)
print("Are they the same?", np.allclose(noise_stats.cov, mnf_result.noise.cov))
