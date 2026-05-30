"""
Transformer CCDC Baseline Visualizer
Generates a CCDC-comparable time-series plot mapping the non-parametric 
learned phenology baseline against true irregular observations.
Includes a spatial context subplot of the first post-training observation.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from datetime import datetime
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" 

# Import the architecture and data extraction logic
from transformer_ccdc import PhenologyTransformer, extract_valid_pixel_history
import data_loader as DL

# ==========================================
# CONFIGURATION
# ==========================================
LOCATION = "Tait"
LANDSAT_PATH = f"C:/satelliteImagery/LANDSAT/{LOCATION}/LANDSAT_Stack_{LOCATION}_GEE_2015_2025_WRS16_SC_EM-7_Gram-minEndmember_Norm-bandCount.h5"
TANAGER_PATH = f"C:/satelliteImagery/Tanager/{LOCATION}/Tanager_Stack_{LOCATION}_HDFEOS_SC_EM-7_Gram-minEndmember_Norm-bandCount.h5"

TARGET_ROW = 72
TARGET_COL = 25

WINDOW_YEARS = 4.0
LAMBDA_THRESH = 3.0
K_CONSECUTIVE = 5

def plot_pixel_baseline(cube: list, times: np.ndarray, values: np.ndarray, start_idx: int = 0):
    """
    Trains the Transformer on a specific temporal window and plots the learned 
    continuous baseline against the true observations. Includes a spatial 
    context image of the first future observation.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PhenologyTransformer().to(device)
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = optim.AdamW(model.parameters(), lr=0.01)

    # 1. Isolate the training window
    window_start_time = times[start_idx]
    window_end_time = window_start_time + WINDOW_YEARS
    
    train_indices = np.where((times >= window_start_time) & (times < window_end_time))[0]
    if len(train_indices) < 10:
        raise ValueError("Insufficient valid data in the selected window to train a baseline.")
        
    t_train = torch.tensor(times[train_indices], dtype=torch.float32).unsqueeze(-1).unsqueeze(0).to(device)
    x_train = torch.tensor(values[train_indices], dtype=torch.float32).unsqueeze(-1).unsqueeze(0).to(device)

    # 2. Train the model (Fast local fit)
    print("Training continuous baseline...")
    model.train()
    for _ in range(150):
        optimizer.zero_grad()
        pred = model(x_train, t_train)
        loss = criterion(pred, x_train)
        loss.backward()
        optimizer.step()

    # 3. Calculate baseline variance (Sigma)
    model.eval()
    with torch.no_grad():
        train_preds = model(x_train, t_train)
        residuals = torch.abs(train_preds - x_train).cpu().numpy().flatten()
        sigma_train = np.std(residuals)
        threshold = LAMBDA_THRESH * sigma_train

    # 4. Generate the Dense Continuous Baseline Curve
    forecast_end = window_end_time + 1.5
    t_dense = np.linspace(window_start_time, forecast_end, 500)
    
    t_dense_tensor = torch.tensor(t_dense, dtype=torch.float32).unsqueeze(-1).unsqueeze(0).to(device)
    x_query = torch.zeros_like(t_dense_tensor)
    
    with torch.no_grad():
        dense_preds = model(x_query, t_dense_tensor).cpu().numpy().flatten()

    # 5. Extract Future Evaluation Points
    future_indices = np.where((times >= window_end_time) & (times <= forecast_end))[0]
    
    # ==========================================
    # PLOTTING
    # ==========================================
    fig, (ax_ts, ax_img) = plt.subplots(1, 2, figsize=(18, 7), gridspec_kw={'width_ratios': [3, 1.2]})
    
    # --- Subplot 1: Time Series Baseline ---
    # A. Plot the Shaded Threshold Band
    ax_ts.fill_between(
        t_dense, 
        dense_preds - threshold, 
        dense_preds + threshold, 
        color='lightgreen', alpha=0.3, 
        label=f'Normal Variance Bound (\u00B1 {LAMBDA_THRESH}\u03C3)'
    )
    
    # B. Plot the Dense Transformer Baseline
    ax_ts.plot(t_dense, dense_preds, color='green', linewidth=2, label='Transformer Learned Baseline')
    
    # C. Plot the True Historical Training Observations
    ax_ts.scatter(
        times[train_indices], values[train_indices], 
        color='blue', edgecolor='k', s=50, zorder=5, 
        label='Training Observations'
    )
    
    # D. Plot Future Observations and Highlight Anomalies
    if len(future_indices) > 0:
        t_future = times[future_indices]
        x_future = values[future_indices]
        
        t_future_tensor = torch.tensor(t_future, dtype=torch.float32).unsqueeze(-1).unsqueeze(0).to(device)
        with torch.no_grad():
             future_preds = model(torch.zeros_like(t_future_tensor), t_future_tensor).cpu().numpy().flatten()
             
        future_residuals = np.abs(future_preds - x_future)
        is_anomaly = future_residuals > threshold
        
        # Plot Normal Future Points
        ax_ts.scatter(
            t_future[~is_anomaly], x_future[~is_anomaly], 
            color='gray', edgecolor='k', s=50, zorder=5, 
            label='Expected Future Observations'
        )
        
        # Plot Anomalous Future Points
        ax_ts.scatter(
            t_future[is_anomaly], x_future[is_anomaly], 
            color='red', edgecolor='k', s=70, marker='X', zorder=6, 
            label='Structural Anomalies (Break Points)'
        )
        
        # --- Subplot 2: Spatial Context Image ---
        # Map the first future observation back to the original datacube chronologically
        first_future_time = t_future[0]
        start_time = cube[0]['datetime_utc']
        cube_time_fractions = np.array([
            (frame['datetime_utc'] - start_time).total_seconds() / (365.25 * 24 * 3600) 
            for frame in cube
        ])
        
        # Find exact index in the datacube avoiding fill/interpolation mapping
        frame_idx = int(np.argmin(np.abs(cube_time_fractions - first_future_time)))
        context_img = cube[frame_idx]['ortho_visual']
        img_date = cube[frame_idx]['datetime_utc'].strftime('%Y-%m-%d')
        sensor = cube[frame_idx]['spacecraft']
        
        ax_img.imshow(context_img)
        ax_img.scatter(
            [TARGET_COL], [TARGET_ROW], 
            color='orange', marker='o', s=80, 
            edgecolor='black', linewidth=1.5, zorder=10, 
            label='Target Pixel'
        )
        ax_img.set_title(f"Spatial Context: Frame {frame_idx}\n{sensor} | {img_date}", fontsize=13)
        ax_img.set_xlabel("Pixel Column (X)", fontsize=11)
        ax_img.set_ylabel("Pixel Row (Y)", fontsize=11)
        ax_img.legend(loc='upper right', fontsize=10)

    else:
        ax_img.text(0.5, 0.5, "No future observations\navailable for context.", ha='center', va='center', fontsize=12)
        ax_img.set_title("Context Image Unavailable")
        ax_img.axis('off')

    # Formatting Time Series Subplot
    ax_ts.axvline(x=window_end_time, color='k', linestyle='--', label='End of Training Window')
    ax_ts.set_title(f"Transformer CCDC Temporal Evaluation - Pixel (Row {TARGET_ROW}, Col {TARGET_COL})", fontsize=16)
    ax_ts.set_xlabel("Continuous Time (Fractional Years)", fontsize=12)
    ax_ts.set_ylabel("Z-Score Complexity (Monochromatic Value)", fontsize=12)
    ax_ts.legend(loc='upper right', fontsize=10)
    ax_ts.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.show()

def main():
    print("Loading multi-sensor datacube...")
    cube, h5_l, h5_t = DL.load_merged_datacube(LANDSAT_PATH, TANAGER_PATH)
    
    print(f"Extracting strictly valid history for pixel ({TARGET_ROW}, {TARGET_COL})...")
    try:
        times_ary, values_ary = extract_valid_pixel_history(cube, TARGET_ROW, TARGET_COL)
        plot_pixel_baseline(cube, times_ary, values_ary)
    except ValueError as e:
        print(f"Cannot generate plot: {e}")
        
    h5_l.close()
    h5_t.close()

if __name__ == "__main__":
    main()