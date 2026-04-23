import numpy as np
import matplotlib.pyplot as plt
import time
import warnings

# ==========================================
# 1. The User's Exact Pipeline Functions
# ==========================================

def maximumDistance_fast(data_2d, num_endmembers):
    """
    Highly optimized version of the user's Max-D algorithm.
    Expects data_2d shape: [bands, pixels]
    """
    num_bands, num_pix = data_2d.shape
    endmembers = np.zeros([num_bands, num_endmembers])

    # 1. Max & Min Magnitude (EM1 and EM2)
    magnitude = np.linalg.norm(data_2d, axis=0)
    idx1 = np.argmax(magnitude)
    idx2 = np.argmin(magnitude)

    endmembers[:, 0] = data_2d[:, idx1]
    endmembers[:, 1] = data_2d[:, idx2]

    data_proj = data_2d.copy().astype(float)

    # 2. Sequential Max Distance Projection
    for i in range(2, num_endmembers):
        diff = data_proj[:, idx2:idx2+1] - data_proj[:, idx1:idx1+1]
        diff_norm_sq = np.sum(diff**2)
        
        if diff_norm_sq > 1e-15:
            # Fast in-place orthogonal projection
            proj_component = diff @ (diff.T @ data_proj) / diff_norm_sq
            data_proj -= proj_component

        idx1 = idx2
        vec = data_proj[:, idx2:idx2+1] 
        diff_new = np.sum(np.square(vec - data_proj), axis=0)
        idx2 = np.argmax(diff_new)
        
        endmembers[:, i] = data_2d[:, idx2]

    return endmembers

def calcGramLocalVolumes(endmembers, localization_vector):
    localized_vectors = endmembers - localization_vector[:, np.newaxis]
    gram = np.matmul(localized_vectors.T, localized_vectors)
    det = np.clip(np.linalg.det(gram),0.0,None)
    volume = np.sqrt(det)
    return volume

# ==========================================
# 2. The Pipeline-Integrated Monte Carlo
# ==========================================

def simulate_max_d_ceiling(N_min, N_max, iterations, pixels_per_scene):
    """
    Simulates random extreme data clouds, runs them through the Max-D 
    algorithm, and calculates the maximum achievable normalized volume.
    """
    N_vals = np.arange(N_min, N_max + 1)
    max_localized_pipeline_volumes = np.zeros(len(N_vals))
    
    print(f"Starting Max-D Integrated Simulation ({iterations} scenes per band count)...")
    start_time = time.time()
    
    for idx, N in enumerate(N_vals):
        best_localized_scaled_vol = 0.0
        best_localized_endmembers = None
        
        for _ in range(iterations):
            # A. Generate a synthetic scene of extreme binary pixels [0, 1]
            # Binary pixels naturally push the volume to its mathematical limits
            scene_data = np.random.randint(0, 2, size=(N, pixels_per_scene)).astype(float)
            scene_data = np.clip(scene_data, 0.001, 1.0)
            
            em = maximumDistance_fast(scene_data, num_endmembers=4)
            
            localizationVec = em[:, 1]
            remainingEndmembers = np.delete(em, 1, axis=1) # Remove EM2
            localized_vol = calcGramLocalVolumes(remainingEndmembers, localizationVec) / (N ** 1.5)

            if localized_vol > best_localized_scaled_vol:
                best_localized_scaled_vol = localized_vol
                if N==7:
                    best_localized_endmembers = remainingEndmembers-localizationVec[:, np.newaxis]
                    print(best_localized_endmembers)

            
                
        max_localized_pipeline_volumes[idx] = best_localized_scaled_vol
        
        if N % 10 == 0:
            print(f"Processed up to N={N}... Current Max-D Limit: {best_localized_scaled_vol:.4f}")

    print(f"Simulation completed in {time.time() - start_time:.2f} seconds.")
    return N_vals, max_localized_pipeline_volumes

# --- Execute and Plot ---
N_vals, localized_pipeline_limits = simulate_max_d_ceiling(N_min=5, N_max=370, iterations=500, pixels_per_scene=50000)

#combine N_vals and localized_pipeline_limits into a csv file
np.savetxt('localized_volume_maxD_limits.csv', np.column_stack((N_vals, localized_pipeline_limits)), delimiter=',')

plt.subplots(figsize=(11, 6))
plt.plot(N_vals, localized_pipeline_limits, linewidth=2, label='Localized Volume')
plt.ylabel('Localized Volume')
plt.xlabel('Number of Spectral Bands ($N$)')
plt.title('Maximum Volume Localized by Minimum Endmember')
plt.grid(True, alpha=0.3)
plt.legend(loc='best')
plt.show()