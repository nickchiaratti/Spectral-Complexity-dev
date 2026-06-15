import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

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

def generate_physical_endmembers(N_bands):
    """
    Generates continuous, highly correlated spectral curves typical of physical materials.
    (e.g., bright soil, dark water, vegetation red-edge)
    """
    x = np.linspace(0.1, 10, N_bands)
    E = np.zeros((N_bands, 4))
    
    # EM1: Bright Soil (High Albedo, slow continuous climb)
    E[:, 0] = 0.2 + 0.6 * (x / 10)**0.5
    # EM2: Deep Water (Dark, exponentially decreasing)
    E[:, 1] = 0.1 * np.exp(-x/2)
    # EM3: Vegetation (Sharp red edge jump, plateau in NIR)
    E[:, 2] = 0.05 + 0.5 / (1 + np.exp(-(x - 4)*3))
    # EM4: Concrete/Urban (Flat, mid-high albedo)
    E[:, 3] = 0.4 + 0.1 * np.sin(x)
    
    return np.clip(E, 0, 1)

def simulate_3x3_lmm_monte_carlo(N_bands, vertices=3, iterations=10000):
    """
    Simulates 10,000 unique 3x3 spatial windows using the Linear Mixture Model, 
    extracts the local endmembers via Max-D, and calculates the complexity volume.
    """
    parallelotope_vertices = vertices
    scaled_volumes = np.zeros(iterations)
    
    # The true unmixed materials existing in the "world"
    true_E = generate_physical_endmembers(N_bands)
    
    for i in range(iterations):
        # 1. The Spatial Abundance Constraints (ASC and ANC)
        # Using a Dirichlet distribution ensures all fractions sum to 1.0 and are >= 0.
        # Generating exactly 9 pixels to simulate a 3x3 spatial window.
        abundances = np.random.dirichlet(alpha=[1.0, 1.0, 1.0, 1.0], size=9).T 
        
        # 2. The Linear Mixture Model (X = E * A)
        clean_pixels = np.matmul(true_E, abundances)
        
        # 3. Additive Sensor Noise (Gaussian SNR)
        noise = np.random.normal(0, 0.01, size=(N_bands, 9))
        scene_pixels = np.clip(clean_pixels + noise, 0, 1)
        
        # 4. Extract endmembers using the user's Max-D pipeline
        extracted_em = maximumDistance_fast(scene_pixels, parallelotope_vertices+1)
        
        # 5. Localize to the minimum endmember (EM2)
        localizationVec = extracted_em[:, 1] 
        remainingEndmembers = np.delete(extracted_em, 1, axis=1)
        
        # 6. Calculate Volume and Scale by (1 / N^(1.5))
        raw_vols = calcGramLocalVolumes(remainingEndmembers, localizationVec)
        scaled_volumes[i] = raw_vols / (N_bands ** (parallelotope_vertices/2)) # Index 2 is m=3
        
    return scaled_volumes

N_min = 5
N_max = 50
N = np.arange(N_min, N_max + 1)
N= np.append(N, np.arange(N_max, 370, 10))
meanVols3 = np.zeros(len(N))
varVols3 = np.zeros(len(N))
#meanVols4 = np.zeros(len(N))
#varVols4 = np.zeros(len(N))

for idx, bands in enumerate(N):
    vols3 = simulate_3x3_lmm_monte_carlo(N_bands=bands, vertices=3, iterations=10000)
    meanVols3[idx] = np.mean(vols3)
    varVols3[idx] = np.var(vols3)
    #vols4 = simulate_3x3_lmm_monte_carlo(N_bands=bands, vertices=4, iterations=10000)
    #meanVols4[idx] = np.mean(vols4)
    #varVols4[idx] = np.var(vols4)


plt.subplots(figsize=(11, 6))

plt.plot(N,meanVols3,label='LMM Simulated Volume (4 Endmembers)')
plt.yscale('log')
twin1 = plt.twinx()
twin1.plot(N,varVols3,color='orange',label='LMM Simulated Variance (4 Endmembers)')
twin1.set_yscale('log')
plt.xlabel('Number of Spectral Bands ($N$)')
plt.ylabel('Localized Volume')
plt.title('Maximum Volume Localized by Minimum Endmember')
plt.grid(True, alpha=0.3)
plt.legend(loc='best')
plt.show()

#plt.figure()
#plt.errorbar(N,meanVols4,xerr=varVols4,label='LMM Simulated Volume (5 Endmembers)')
#plt.yscale('log')
#plt.xlabel('Number of Spectral Bands ($N$)')
#plt.ylabel('Localized Volume')
#plt.title('Maximum Volume Localized by Minimum Endmember')
#plt.grid(True, alpha=0.3)
#plt.legend(loc='best')
#plt.show()