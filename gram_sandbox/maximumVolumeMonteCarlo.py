import numpy as np
import matplotlib.pyplot as plt
import time

def calculate_true_discrete_limit(N_bands):
    """Calculates the exact theoretical maximum discrete volume for N bands."""
    best_vol_sq = 0
    for A in range((N_bands // 3) + 1):
        for B in range((N_bands - 3*A) // 3 + 1):
            C = N_bands - 3*A - 3*B
            vol_sq = ((A + B)**2) * (A + 4*B + 3*C)
            if vol_sq > best_vol_sq:
                best_vol_sq = vol_sq
    return np.sqrt(best_vol_sq) * np.power(N_bands,-3/2)

def binary_monte_carlo_sweep(N_min=4, N_max=100, iterations_per_N=200_000):
    """
    Sweeps through spectral band counts, performing a vectorized binary 
    Monte Carlo simulation to find the maximum scaled volume.
    """
    N_vals = np.arange(N_min, N_max + 1)
    mc_max_volumes = np.zeros(len(N_vals))
    mc_min_volumes = np.zeros(len(N_vals))
    true_max_volumes = np.zeros(len(N_vals))

    
    
    print(f"Running Binary Monte Carlo Sweep ({iterations_per_N:,} iterations per N)...")
    start_time = time.time()
    
    for i, N in enumerate(N_vals):
        
        #iterations_per_N = int(np.maximum(i/10,1)*iterations_per_N)
        # 1. Calculate the true theoretical limit for reference
        true_max_volumes[i] = calculate_true_discrete_limit(N)
        
        # 2. Generate strictly binary matrices {0, 1}
        # Shape: (iterations, 3 radiating vectors, N bands)
        V = np.random.randint(0, 2, size=(iterations_per_N, 3, N))
        
        # 3. Vectorized Gram Matrix Calculation
        G = np.matmul(V, V.transpose(0, 2, 1))
        
        # 4. Vectorized Determinant Calculation
        dets = np.clip(np.linalg.det(G), 0.0,None)
        
        # 5. Scale and extract the maximum
        vol = np.sqrt(dets)* np.power(N,-3/2)
        max_vol = np.max(vol) 
        mc_max_volumes[i] = max_vol
        min_vol = np.min(vol) 
        mc_min_volumes[i] = min_vol
        
        # Print progress every 20 bands
        if N % 20 == 0:
            print(f"Processed up to N={N}...")

    elapsed = time.time() - start_time
    print(f"Sweep completed in {elapsed:.2f} seconds.")
    
    return N_vals, mc_max_volumes, mc_min_volumes, true_max_volumes

# --- Execute Simulation ---
N_min = 5
N_max = 10
iterations = 100000_000

N_vals, mc_max_volumes, mc_min_volumes, true_max_volumes = binary_monte_carlo_sweep(N_min, N_max, iterations)

# --- Plot the Results ---
true_asymptotic_limit = np.sqrt(4 / 27)

plt.figure(figsize=(11, 6))

# Plot the True Discrete Limit
#plt.plot(N_vals, true_volumes, color='blue', linewidth=2, 
         #label='True Combinatorial Limit')

# Plot the Binary Monte Carlo Results
plt.plot(N_vals, mc_max_volumes, color='orange', alpha=0.8, label=f'Binary Monte Carlo Max ({iterations:,} iter/N)')
plt.plot(N_vals, mc_min_volumes, color='green', alpha=0.8, label=f'Binary Monte Carlo Min ({iterations:,} iter/N)')
plt.yscale('log')

# Plot the Asymptotic Bound
#plt.axhline(true_asymptotic_limit, color='red', linestyle='--', label=f'Asymptotic Limit ($\\approx {true_asymptotic_limit:.4f}$)')

plt.xlabel('Number of Spectral Bands ($N$)')
plt.ylabel('Normalized Volume')
plt.title('Binary Monte Carlo vs. Theoretical Maximum Simplex Volume ($m=3$)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()