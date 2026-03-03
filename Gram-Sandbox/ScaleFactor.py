import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Configuration ---
# Update these paths if the files are in a different directory
LANDSAT_CSV = "C:/satelliteImagery/MultiSensor_Analysis/Landsat_2025-09-12_Loc-datasetMean_Norm-bandCount.csv"
TANAGER_CSV = "C:/satelliteImagery/MultiSensor_Analysis/Tanager_2025-09-19_Loc-datasetMean_Norm-bandCount.csv"

titleStr = LANDSAT_CSV[60:-4]

def main():
    print("Loading CSV files...")
    try:
        # Load the CSVs as numpy arrays and flatten them to 1D lists
        landsat_data = pd.read_csv(LANDSAT_CSV, header=None).values.flatten()
        tanager_data = pd.read_csv(TANAGER_CSV, header=None).values.flatten()
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        return

    # Ensure both arrays are exactly the same size
    if landsat_data.shape != tanager_data.shape:
        print("Warning: CSV dimensions do not match! Truncating to minimum size.")
        min_len = min(len(landsat_data), len(tanager_data))
        landsat_data = landsat_data[:min_len]
        tanager_data = tanager_data[:min_len]

    # Create a mask to remove NaNs (NoData pixels) or exact zeros from BOTH arrays
    valid_mask = (~np.isnan(landsat_data)) & (~np.isnan(tanager_data)) & (landsat_data > 0) & (tanager_data > 0)
    
    l_valid = landsat_data[valid_mask]
    t_valid = tanager_data[valid_mask]

    print(f"Analyzing {len(l_valid)} valid coincident tiles...")

    # --- Calculate Scale Factors ---
    
    # 1. Least Squares Scale Factor (Best for minimizing overall error)
    # Equation: Tanager = m * Landsat. We solve for m.
    # m = Sum(L * T) / Sum(L^2)
    optimal_scale_factor = np.sum(l_valid * t_valid) / np.sum(l_valid**2)
    
    # 2. Median Ratio (Highly resistant to outlier pixels/clouds)
    ratios = t_valid / l_valid
    median_ratio = np.median(ratios)
    mean_ratio = np.mean(ratios)

    print("-" * 40)
    print(f"Optimal Least-Squares Scale Factor: {optimal_scale_factor:.4f}")
    print(f"Median Ratio Scale Factor:          {median_ratio:.4f}")
    print(f"Mean Ratio Scale Factor:            {mean_ratio:.4f}")
    print("-" * 40)
    print(f"To convert Landsat to Tanager scale: Landsat_Volume * {optimal_scale_factor:.4f}")

    # --- Visualization ---
    plt.figure(figsize=(9, 8))
    
    # Plot raw data points
    plt.scatter(l_valid, t_valid, alpha=0.3, s=10, label='Tile Volumes')
    
    # Plot the scaling lines
    max_l = np.max(l_valid)
    line_x = np.array([0, max_l])
    
    plt.plot(line_x, line_x * optimal_scale_factor, color='red', linewidth=2, 
             label=f'Least Squares Fit (x{optimal_scale_factor:.2f})')
    plt.plot(line_x, line_x * median_ratio, color='green', linewidth=2, linestyle='--', 
             label=f'Median Ratio Fit (x{median_ratio:.2f})')

    plt.title(f"Landsat vs Tanager Tiled Volume Map Correlation\n{titleStr}", fontsize=14)
    plt.xlabel("Landsat Volume")
    plt.ylabel("Tanager Volume")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()