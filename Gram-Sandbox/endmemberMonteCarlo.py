import numpy as np
import time

def monte_carlo_max_volume(N_bands, num_iterations=1000000, batch_size=100000):
    """
    Monte Carlo simulation to find the maximum scaled volume for 4 endmembers
    (m=3 radiating vectors) in N spectral bands.
    
    Args:
        N_bands: Number of spectral bands.
        num_iterations: Total number of random matrices to simulate.
        batch_size: How many matrices to process in memory simultaneously.
    """
    print(f"Starting Monte Carlo for N={N_bands} bands ({num_iterations:,} iterations)...")
    start_time = time.time()
    
    max_scaled_vol = 0.0
    best_radiating_vectors = None
    
    # We lock the anchor endmember to [0, 0, ..., 0].
    # Therefore, we only need to simulate the 3 radiating vectors.
    for i in range(0, num_iterations, batch_size):
        
        # 1. Generate random uniform surface reflectance data in [0, 1]
        # Shape: (batch_size, 3 radiating vectors, N bands)
        #V = np.random.rand(batch_size, 3, N_bands)
        V = np.random.randint(0, 2, size=(batch_size, 3, N_bands))
        
        # 2. Vectorized Gram Matrix Calculation
        # We multiply V by its transpose.
        # V shape: (batch_size, 3, N)
        # V^T shape: (batch_size, N, 3) --> achieved via transpose(0, 2, 1)
        # G shape: (batch_size, 3, 3)
        G = np.matmul(V, V.transpose(0, 2, 1))
        
        # 3. Vectorized Determinant Calculation
        dets = np.linalg.det(G)
        
        # 4. Guard against negative floating point errors
        dets = np.clip(dets, 0.0, None)
        
        # 5. Calculate sensor-agnostic scaled volumes
        scaled_vols = np.sqrt(dets) / (N_bands ** 1.5)
        
        # 6. Extract the maximum from this batch
        batch_max_idx = np.argmax(scaled_vols)
        if scaled_vols[batch_max_idx] > max_scaled_vol:
            max_scaled_vol = scaled_vols[batch_max_idx]
            best_radiating_vectors = V[batch_max_idx]
            
    elapsed = time.time() - start_time
    print(f"Simulation complete in {elapsed:.2f} seconds.")
    print(f"Max Scaled Volume Found: {max_scaled_vol:.5f}")
    
    return max_scaled_vol, best_radiating_vectors

# --- Run the Simulation ---
# Example: 7-band multispectral simulation
N = 7
max_vol, best_vectors = monte_carlo_max_volume(N_bands=N, num_iterations=1_000_000)

print("\nBest 3 Radiating Vectors Found (The Anchor is implicitly all 0s):")
print(np.round(best_vectors, 3))