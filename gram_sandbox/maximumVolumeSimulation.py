import numpy as np
import matplotlib.pyplot as plt

def simulate_max_orthogonal_endmembers(N_bands):
    """
    Generates the exact 4 binary endmembers in [0,1] that produce 
    the theoretical maximum volume in N dimensions for m=3 radiating vectors.
    """
    best_vol_sq = 0
    
    # 1. Combinatorial Search for Maximum Volume Allocation
    for A in range((N_bands // 3) + 1):
        for B in range((N_bands - 3*A) // 3 + 1):
            
            # Pack remaining bands into C
            C = N_bands - 3*A - 3*B
            
            # Determinant of the Gram matrix
            vol_sq = ((A + B)**2) * (A + 4*B + 3*C)
            
            if vol_sq > best_vol_sq:
                best_vol_sq = vol_sq

    # The scaled volume includes the 1/(N^1.5) normalization factor
    scaled_volume = np.sqrt(best_vol_sq) / (N_bands**1.5)
    return scaled_volume

# --- Run the Simulation from 4 to 1000 Bands ---
N_vals = np.arange(4, 101)
max_volumes = np.zeros(len(N_vals))

for i, N in enumerate(N_vals):
    max_volumes[i] = simulate_max_orthogonal_endmembers(N)

# --- The True Geometric Asymptotic Limit (Hadamard Bound) ---
# Derived from maximizing the combinatorial packing: B = N/3
true_asymptotic_limit = np.sqrt(4 / 27)

# --- Plot the Results ---
plt.figure(figsize=(10, 6))
plt.plot(N_vals, max_volumes, label='Discrete Max Volume (Combinatorial Packing)', color='blue')
plt.axhline(true_asymptotic_limit, color='red', linestyle='--', label=f'True Limit ($\sqrt{{4/27}} \\approx {true_asymptotic_limit:.4f}$)')
plt.xlabel('Number of Spectral Bands ($N$)')
plt.ylabel('Normalized Volume Limit')
plt.title('Theoretical Maximum Simplex Volume ($m=3$) vs. Band Count')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()