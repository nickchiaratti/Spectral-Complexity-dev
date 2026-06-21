"""
FT-OOD: Fourier Transform Out-of-Distribution Anomaly Detection Pipeline

Architecture:
    Layer 1: Deterministic Multi-Scale ALFT Encoder (frozen, no learnable parameters)
             - Batched Orthogonal Matching Pursuit on Non-Uniform DFT
             - Parallel continuous temporal windows [0.5yr, 1.0yr, 3.0yr]
    Layer 2: Time2Vec continuous temporal encoding (learnable)
    Layer 3: Masked Transformer Encoder (learnable, bidirectional for retrospective)
    Layer 4: Deep SVDD projection head (learnable) → OOD anomaly score
    Layer 5: Batched Streaming Drift Detector (statistical, EMA-based)

This pipeline replaces the 1D-CNN Inception architecture (.CNN_generalized_main.py)
which failed due to fixed Conv1d kernel lengths on irregularly sampled time series.
The ALFT encoder natively handles irregular sampling via the NDFT design matrix,
eliminating the need for synthetic interpolation.

ALFT logic adapted from: dhr_main_pytorch.py (Dynamic Harmonic Regression)
Pipeline structure adapted from: .CNN_generalized_main.py

References:
    Kazemi et al. (2019). Time2Vec: Learning a Vector Representation of Time.
    Ruff et al. (2018). Deep One-Class Classification. ICML.
    VanderPlas (2018). Understanding the Lomb-Scargle Periodogram.
    Zhu & Woodcock (2014). Continuous Change Detection and Classification.
"""

import os
import h5py
import numpy as np
import datetime
import math
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# ==========================================
# 0. CONFIGURATION & CONSTANTS
# ==========================================
LOCATION = "Tait"
H5_PATH = f"C:/satelliteImagery/HLST30/HLST_{LOCATION}_Harmonized_SC_EM-7_Norm-bandCount.h5"
TARGET_METRIC = 'sliding_volume_z_score'
MASK_PATH = '/HDFEOS/GRIDS/HARMONIZED/Data Fields/common_mask'
DATA_PATH = f'/HDFEOS/GRIDS/HARMONIZED/Data Fields/{TARGET_METRIC}'
OUTPUT_DIR = "C:/satelliteImagery/HLST30/FT-OOD"

# ALFT Multi-Scale Encoder Configuration
# Lookback windows in continuous years. Configurable: try adding 0.25 for sub-annual
# sensitivity once late-mission sampling density (5 days/sample) supports it.
WINDOWS = [0.5, 1.0, 3.0]
K_FREQUENCIES = 2               # Dominant frequencies per window via OMP
MIN_SAMPLES = 2 * K_FREQUENCIES + 1 + 3  # 8: 2K+1 parameters + 3 DOF for robust RMSE

# Frequency search grid for NDFT (identical to dhr_main_pytorch.py)
F_GRID_MIN = 0.2    # cycles/year
F_GRID_MAX = 4.0
F_GRID_N = 150

# Sequence & Model Architecture
L_MAX = 256          # Maximum sequence length for Transformer window
D_MODEL = 64         # Transformer hidden dimension (divisible by NUM_HEADS)
NUM_HEADS = 8
NUM_LAYERS = 4
TIME_DIM = 16        # Time2Vec output features
SVDD_DIM = 64        # Deep SVDD hypersphere dimension

# Training
TRAIN_END_DATE = "2024-01-01"
SKIP_TRAIN = False
EPOCHS = 10
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-6
BATCH_SIZE_TRAIN = 2048
TRAIN_STRIDE = 4     # Subsample training target timesteps (every Nth)
CENTER_INIT_SAMPLES = 10000  # Max samples for hypersphere center initialization

# Spatial Processing
CHUNK_SIZE = 128

# Drift Detection Hyperparameters
WARNING_SIGMA = 2.0
DRIFT_SIGMA = 3.0
CONSECUTIVE_ANOMALIES = 3
EMA_ALPHA = 0.05
WARMUP_PERIOD = 20

# Inference
INFERENCE_BATCH = 256  # Sub-batch size for model forward pass

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Derived constants
FEATURES_PER_WINDOW = 2 * K_FREQUENCIES + 2  # [beta_0, cos_1..K, sin_1..K, RMSE]
ALFT_DIM = len(WINDOWS) * FEATURES_PER_WINDOW


from dataset import load_dataset
from train_evaluate import precompute_alft_features, train_svdd, run_inference
from model import OOD_Anomaly_Detector, BatchedStreamingDriftDetector


# ==========================================
# 8. OUTPUT & MAIN
# ==========================================
def save_results(output_h5, score_map, status_map, first_drift_ts,
                 drift_count_map, acq_times, geo_transform, spatial_ref):
    """Saves inference results to HDF5 with spatial metadata."""
    os.makedirs(os.path.dirname(output_h5), exist_ok=True)

    print(f"\nSaving results to {output_h5}...")
    with h5py.File(output_h5, 'w') as out:
        # Spatial metadata
        if spatial_ref is not None:
            out.attrs['spatial_ref'] = spatial_ref
        if geo_transform is not None:
            out.attrs['GeoTransform'] = geo_transform
        out.attrs['acquisition_time'] = acq_times

        # Configuration
        out.attrs['LOCATION'] = LOCATION
        out.attrs['TARGET_METRIC'] = TARGET_METRIC
        out.attrs['WINDOWS'] = WINDOWS
        out.attrs['K_FREQUENCIES'] = K_FREQUENCIES
        out.attrs['L_MAX'] = L_MAX
        out.attrs['D_MODEL'] = D_MODEL
        out.attrs['NUM_HEADS'] = NUM_HEADS
        out.attrs['NUM_LAYERS'] = NUM_LAYERS
        out.attrs['TIME_DIM'] = TIME_DIM
        out.attrs['SVDD_DIM'] = SVDD_DIM
        out.attrs['TRAIN_END_DATE'] = TRAIN_END_DATE
        out.attrs['WARNING_SIGMA'] = WARNING_SIGMA
        out.attrs['DRIFT_SIGMA'] = DRIFT_SIGMA
        out.attrs['CONSECUTIVE_ANOMALIES'] = CONSECUTIVE_ANOMALIES
        out.attrs['EMA_ALPHA'] = EMA_ALPHA
        out.attrs['WARMUP_PERIOD'] = WARMUP_PERIOD
        out.attrs['SOURCE_DATA'] = H5_PATH

        # Datasets
        out.create_dataset(
            'anomaly_scores', data=score_map, compression='gzip'
        )
        out.create_dataset(
            'drift_status', data=status_map, compression='gzip'
        )
        out.create_dataset(
            'first_drift_timestamp', data=first_drift_ts, compression='gzip'
        )
        out.create_dataset(
            'drift_count', data=drift_count_map, compression='gzip'
        )


def main():
    print(f"FT-OOD Pipeline: {LOCATION}")
    print(f"Device: {DEVICE}")
    print(f"ALFT Windows: {WINDOWS} years, K={K_FREQUENCIES} frequencies")
    print(f"ALFT feature dim: {ALFT_DIM}")
    print(f"Model: d_model={D_MODEL}, heads={NUM_HEADS}, layers={NUM_LAYERS}")
    print(f"Train cutoff: {TRAIN_END_DATE}")

    # Parse train end date to fractional year
    train_end_dt = datetime.datetime.strptime(TRAIN_END_DATE, "%Y-%m-%d")
    train_end_dt = train_end_dt.replace(tzinfo=datetime.timezone.utc)
    train_end_year = train_end_dt.year
    _sy = datetime.datetime(train_end_year, 1, 1, tzinfo=datetime.timezone.utc)
    _sn = datetime.datetime(train_end_year + 1, 1, 1, tzinfo=datetime.timezone.utc)
    _dur = (_sn - _sy).total_seconds()
    _el = (train_end_dt - _sy).total_seconds()
    train_end_frac = train_end_year + (_el / _dur)

    # Output paths
    _win_str = '-'.join(f"{w:.2g}" for w in WINDOWS)
    weights_path = os.path.join(
        OUTPUT_DIR,
        f'FT-OOD_{LOCATION}_weights_W{_win_str}_K{K_FREQUENCIES}_pre{train_end_year}.pth'
    )
    output_h5 = os.path.join(
        OUTPUT_DIR,
        f'FT-OOD_{LOCATION}_results_W{_win_str}_K{K_FREQUENCIES}_pre{train_end_year}.h5'
    )

    # ── 1. Load Dataset ──
    y_data, valid_mask, acq_times, frac_years, geo_transform, spatial_ref = (
        load_dataset(H5_PATH, TARGET_METRIC)
    )

    # ── 2. Pre-compute ALFT Features ──
    alft_features, alft_valid = precompute_alft_features(
        y_data, valid_mask, frac_years, ALFT_DIM, WINDOWS, K_FREQUENCIES,
        F_GRID_MIN, F_GRID_MAX, F_GRID_N, MIN_SAMPLES, CHUNK_SIZE, DEVICE
    )

    # ── 3. Build and Train Model ──
    model = OOD_Anomaly_Detector(
        alft_dim=ALFT_DIM, time_dim=TIME_DIM, d_model=D_MODEL,
        num_heads=NUM_HEADS, num_layers=NUM_LAYERS, svdd_dim=SVDD_DIM
    ).to(DEVICE)

    if SKIP_TRAIN and os.path.exists(weights_path):
        print(f"\nSkipping training. Loading weights from {weights_path}...")
        model.load_state_dict(
            torch.load(weights_path, map_location=DEVICE, weights_only=True)
        )
    else:
        train_svdd(
            model, alft_features, alft_valid, frac_years,
            train_end_frac, L_MAX, ALFT_DIM, TRAIN_STRIDE, BATCH_SIZE_TRAIN,
            CENTER_INIT_SAMPLES, EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
            weights_path, DEVICE
        )

    # ── 4. Retrospective Inference ──
    score_map, status_map, first_drift_ts, drift_count_map = run_inference(
        model, alft_features, alft_valid, frac_years, acq_times, 
        L_MAX, ALFT_DIM, CHUNK_SIZE, INFERENCE_BATCH, WARNING_SIGMA, 
        DRIFT_SIGMA, CONSECUTIVE_ANOMALIES, EMA_ALPHA, WARMUP_PERIOD, DEVICE
    )

    # ── 5. Save Results ──
    save_results(
        output_h5, score_map, status_map, first_drift_ts,
        drift_count_map, acq_times, geo_transform, spatial_ref
    )

    # ── Summary ──
    num_pixels = alft_valid.shape[1] * alft_valid.shape[2]
    drift_pixels = np.sum(drift_count_map > 0)
    total_drifts = np.sum(drift_count_map)
    warning_count = np.sum(status_map == BatchedStreamingDriftDetector.STATUS_WARNING)

    print(f"\n{'=' * 50}")
    print(f"FT-OOD Pipeline Complete")
    print(f"{'=' * 50}")
    print(f"Pixels with confirmed drift: {drift_pixels:,} / {num_pixels:,} "
          f"({100.0 * drift_pixels / num_pixels:.2f}%)")
    print(f"Total drift events: {total_drifts:,}")
    print(f"Total warning events: {warning_count:,}")
    print(f"Results: {output_h5}")
    print(f"Weights: {weights_path}")


if __name__ == '__main__':
    main()