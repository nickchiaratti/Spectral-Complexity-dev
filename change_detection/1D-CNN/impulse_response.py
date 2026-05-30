"""
1D-CNN Impulse Response Rollout (Sensitivity Analysis)

This script acts as an isolated diagnostic probe for the Multi-Scale SITS Net. 
It generates a pure synthetic step-function (representing a sudden target placement) 
to evaluate the network's detection latency, memory decay, and response to 
varying temporal cadences and seasonal starting points.

Author: [Your Name/Lab]
Date: 2026-04-16
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
import datetime
from model import MultiScaleSITSNet

# ==========================================
# 1. DIAGNOSTIC CONFIGURATION
# ==========================================
WEIGHTS_PATH = r"C:\satelliteImagery\HLST30\1D-CNN-Tait\sits_baseline_weights_pre2024.pth"

SIMULATION_STEPS = 45
HISTORY_WINDOW = 20

# Synthetic Data Parameters
START_DATE = datetime.datetime(2024, 4, 1, tzinfo=datetime.timezone.utc)
SPATIAL_LOCATION = (78,27)
# START_DOY = 365//2          # Day of Year the simulation begins (Impacts seasonal trigonometric encoding)
CADENCE_DAYS = 8.0      # Simulated time sampling interval between frames

# Anomaly Parameters
ANOMALY_START_STEP = 25  # The frame where the "tarp" is placed
ANOMALY_MAGNITUDE = 2  # Z-score of the structural anomaly

# Evaluation Frame for Concatenated Visualization
# Captures the exact sliding window state at this step (e.g., right as the anomaly hits)
EVALUATION_STEP = 25

# Dummy Baseline RMSE (Used purely for plotting the 3x visual threshold line)
# In production, this is calculated dynamically in train_evaluate.py
ASSUMED_BASELINE_RMSE = 0.65 

# ==========================================
# 2. SYNTHETIC TENSOR GENERATION
# ==========================================

def fourier_features(val, freqs=10):
    """Matches the static spatial encoding from dataset.py"""
    features = []
    for i in range(freqs):
        features.append(math.sin((2**i) * math.pi * val))
        features.append(math.cos((2**i) * math.pi * val))
    return features

def generate_synthetic_frame(step_idx):
    """
    Generates a single time-step based on configurable cadence and DOY.
    Injects a sudden structural step-function at ANOMALY_START_STEP.
    """
    # 1. Temporal Progression
    start_doy = START_DATE.timetuple().tm_yday
    doy = (start_doy + step_idx * CADENCE_DAYS) % 365.25
    doy_sin = math.sin(2 * math.pi * doy / 365.25)
    doy_cos = math.cos(2 * math.pi * doy / 365.25)
    
    # 2. Time Delta placeholder (Will be calculated dynamically in rollout loop)
    dt_log = 0.0
    
    # 3. Time of Day (TOD) — cyclical encoding of collection hour (UTC)
    # Simulate a fixed acquisition time anchored to the start offset.
    # Using a representative mid-morning UTC hour (e.g., 15.5h ≈ 10:30am EST)
    tod_hour = (15.5 + step_idx * (CADENCE_DAYS % 1.0)) % 24.0
    tod_sin = math.sin(2 * math.pi * tod_hour / 24.0)
    tod_cos = math.cos(2 * math.pi * tod_hour / 24.0)
    
    # 4. The Impulse Injection (Step-Function)
    z_true = ANOMALY_MAGNITUDE if step_idx >= ANOMALY_START_STEP else 0.0
    
    # Feature order must match dataset.py: [doy_sin, doy_cos, tod_sin, tod_cos, dt_log, z_score]
    return [doy_sin, doy_cos, tod_sin, tod_cos, dt_log, z_true]

# ==========================================
# 3. DIAGNOSTIC ROLLOUT EXECUTION
# ==========================================

def main():
    print("--- 1D-CNN Impulse Response Diagnostic Probe ---")
    
    if not os.path.exists(WEIGHTS_PATH):
        print(f"Error: Weights not found at {WEIGHTS_PATH}. Please run training first.")
        return

    # Load Frozen Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MultiScaleSITSNet().to(device)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model.eval()
    
    # Generate Spatial Anchor (Center of frame)
    sf_x = fourier_features(SPATIAL_LOCATION[0], 10)
    sf_y = fourier_features(SPATIAL_LOCATION[1], 10)
    X_spatial = torch.tensor([sf_x + sf_y], dtype=torch.float32).to(device)
    
    # Pre-allocate tracking arrays
    true_z_scores = []
    predicted_z_scores = []  # Specifically tracking the 1-step ahead prediction (t+1)
    mean_residuals = []      # The 3-step forecast error that triggers the anomaly
    
    # State capture for concatenated visualization
    captured_history_z = []
    captured_preds = []
    captured_actuals = []
    
    # Generate the full synthetic timeline
    full_sequence = [generate_synthetic_frame(i) for i in range(SIMULATION_STEPS)]
    
    print(f"Executing Sequential Rollout ({SIMULATION_STEPS} steps @ {CADENCE_DAYS} day cadence)...")
    
    with torch.no_grad():
        for t in range(HISTORY_WINDOW, SIMULATION_STEPS - 2):
            # Extract the 20-frame historical window leading up to 't'
            # Must copy the inner lists to avoid mutating full_sequence
            history_window = [list(frame) for frame in full_sequence[t - HISTORY_WINDOW : t]]
            
            # Recalculate time delta: elapsed time between current target forecast obs (step t)
            # and each historical frame
            for j in range(HISTORY_WINDOW):
                k = t - HISTORY_WINDOW + j
                delta_t = (t - k) * CADENCE_DAYS
                history_window[j][4] = math.log(1 + delta_t)
                
            X_seq = torch.tensor([history_window], dtype=torch.float32).to(device)
            
            # Extract the actual ground truth for the next 3 frames (z_score is index 5 in 6-feature vector)
            actuals = [full_sequence[t][5], full_sequence[t+1][5], full_sequence[t+2][5]]
            
            # Predict the next 3 frames
            preds = model(X_seq, X_spatial).cpu().numpy()[0]
            
            # Calculate metrics
            residual_1 = abs(preds[0] - actuals[0])
            residual_2 = abs(preds[1] - actuals[1])
            residual_3 = abs(preds[2] - actuals[2])
            mean_res = (residual_1 + residual_2 + residual_3) / 3.0
            
            true_z_scores.append(actuals[0])
            predicted_z_scores.append(preds[0]) # Store the immediate 1-step prediction
            mean_residuals.append(mean_res)
            
            # Capture specific sequence state for the sliding window visualization
            if t == EVALUATION_STEP:
                captured_history_z = [frame[5] for frame in history_window]
                captured_preds = preds.tolist()
                captured_actuals = actuals
            
            if t == ANOMALY_START_STEP:
                print(f"  [Step {t}] Anomaly Injected! True Z: {actuals[0]:.1f} | Predicted Z: {preds[0]:.2f} | Mean Res: {mean_res:.2f}")

    # ==========================================
    # 4. VISUALIZATION DASHBOARD
    # ==========================================
    
    # Shift the X-axis for plotting to align with the prediction target (starting at t=20)
    plot_x = np.arange(HISTORY_WINDOW, SIMULATION_STEPS - 2)
    
    def step_to_date(x, pos=None):
        try:
            current_date = START_DATE + datetime.timedelta(days=float(x) * CADENCE_DAYS)
            return current_date.strftime('%Y-%m-%d')
        except:
            return ""
            
    date_formatter = FuncFormatter(step_to_date)
    
    fig = plt.figure(figsize=(16, 12))
    fig.canvas.manager.set_window_title(f"1D-CNN Diagnostics | Start Date: {START_DATE.strftime('%Y-%m-%d')} | Cadence: {CADENCE_DAYS}d")
    gs = gridspec.GridSpec(3, 1, height_ratios=[2, 1, 1.5], hspace=0.3)
    
    ax_sig = fig.add_subplot(gs[0])
    ax_res = fig.add_subplot(gs[1], sharex=ax_sig)
    ax_win = fig.add_subplot(gs[2])
    
    # --- Top Plot: Signal Tracking ---
    ax_sig.plot(plot_x, true_z_scores, 'k-', linewidth=3, label='Synthetic Ground Truth (Step Function)')
    ax_sig.plot(plot_x, predicted_z_scores, 'r--', linewidth=2, marker='o', markersize=6, label=r'CNN 1-Step Prediction ($\hat{y}_{t}$)')
    
    secax_sig = ax_sig.secondary_xaxis('top')
    secax_sig.xaxis.set_major_formatter(date_formatter)
    
    ax_sig.axvline(ANOMALY_START_STEP, color='orange', linestyle=':', linewidth=2, label='Anomaly Injected')
    
    ax_sig.set_title(f"Network Prediction Inertia vs. Step-Function Anomaly (Start Date: {START_DATE.strftime('%Y-%m-%d')}, Cadence: {CADENCE_DAYS}d)", fontweight='bold', fontsize=14)
    ax_sig.set_ylabel("Complexity Z-Score", fontweight='bold')
    ax_sig.grid(True, alpha=0.3, linestyle='--')
    ax_sig.legend(loc='upper left')
    
    # --- Middle Plot: Detection Metric ---
    ax_res.bar(plot_x, mean_residuals, color='tab:blue', alpha=0.7, label='3-Step Mean Absolute Residual')
    
    secax_res = ax_res.secondary_xaxis('top')
    secax_res.xaxis.set_major_formatter(date_formatter)
    
    threshold = 3.0 * ASSUMED_BASELINE_RMSE
    ax_res.axhline(threshold, color='red', linestyle='-', linewidth=2, label=f'Anomaly Trigger Threshold (3x RMSE $\\approx$ {threshold:.2f})')
    
    for i, res in enumerate(mean_residuals):
        if res > threshold:
            ax_res.plot(plot_x[i], res + 0.2, 'rv', markersize=8) 
            
    ax_res.set_title("Detection Trigger Metric", fontweight='bold')
    ax_res.set_xlabel("Absolute Time Step (Index)", fontweight='bold')
    ax_res.set_ylabel("Mean Residual", fontweight='bold')
    ax_res.grid(True, alpha=0.3, linestyle='--')
    ax_res.legend(loc='upper right')
    
    # --- Bottom Plot: Concatenated Sliding Window State ---
    win_x_history = np.arange(1, HISTORY_WINDOW + 1)
    win_x_future = np.arange(HISTORY_WINDOW + 1, HISTORY_WINDOW + 4)
    
    if captured_history_z:
        ax_win.plot(win_x_history, captured_history_z, 'ko-', linewidth=2, markersize=6, label='Historical Sequence Tensor ($X_{seq}$)')
        ax_win.plot(win_x_future, captured_actuals, 'k^-', linewidth=2, markersize=8, alpha=0.5, label='Actual Future')
        ax_win.plot(win_x_future, captured_preds, 'rX--', linewidth=2, markersize=10, label='CNN 3-Step Forecast ($Y_{pred}$)')
        
        ax_win.axvline(HISTORY_WINDOW + 0.5, color='gray', linestyle='-', linewidth=2)
        ax_win.text(HISTORY_WINDOW - 0.5, max(max(captured_history_z), max(captured_actuals), max(captured_preds)) * 0.9, 
                    '← Model Receptive Field', ha='right', va='center', fontweight='bold', color='gray')
        ax_win.text(HISTORY_WINDOW + 1.5, max(max(captured_history_z), max(captured_actuals), max(captured_preds)) * 0.9, 
                    'Forecast Horizon →', ha='left', va='center', fontweight='bold', color='gray')

    ax_win.set_title(f"Isolated Tensor State at Evaluation Step = {EVALUATION_STEP}", fontweight='bold')
    ax_win.set_xlabel("Relative Sequence Index", fontweight='bold')
    ax_win.set_ylabel("Complexity Z-Score", fontweight='bold')
    ax_win.set_xticks(np.arange(1, HISTORY_WINDOW + 4))
    ax_win.grid(True, alpha=0.3, linestyle='--')
    ax_win.legend(loc='upper left')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()