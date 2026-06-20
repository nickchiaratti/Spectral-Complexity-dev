# CNN-Transformer Change Detection Pipeline

This directory contains a CNN-Transformer-based pipeline for spatial-temporal change detection and forecasting using satellite imagery time series (SITS). The primary orchestrator script is `.CNN_transformer_main.py`.

## Pipeline Overview (`.CNN_transformer_main.py`)

The pipeline script configures the hyper-parameters and manages the end-to-end execution:
1. **Configuration:** Defines the target location, training end dates, and data paths.
2. **Train and Evaluate:** Calls `train_and_evaluate` to train the model up to a specified end date and evaluate it on monitoring data, generating predictions and anomalies with Monte Carlo (MC) dropout sampling for confidence intervals.
3. **Visualization:** If inference results are available, it runs `plot_spatial_anomaly_overlay` to visualize the detected anomalies spatially.

## Model Architecture (`model.py`)

The core model is `MultiScaleSITSNet`, a PyTorch-based neural network designed for sequence-to-point forecasting on irregular time series. It focuses strictly on temporal features, omitting spatial dimensions for per-pixel analysis.

*   **Feature Projection:** Projects the 15-channel input temporal features (which include spectral bands, harmonic terms, and time-elapsed features) into a higher-dimensional space using a linear layer.
    *   *Operation:* `Linear(in_features=15, out_features=96)`
    *   *Shape Transformation:* `(Batch, SeqLen, 15) -> (Batch, SeqLen, 96)`
*   **Transformer Block:** Utilizes a `TransformerEncoderLayer` for temporal self-attention across the historical sequence. The attention mechanism explicitly ignores padded sequence elements using a boolean `src_key_padding_mask` derived from the input validity masks, ensuring irregular and padded sequences do not negatively impact the learned representations.
    *   *Operation:* `TransformerEncoderLayer(d_model=96, nhead=4, dim_feedforward=128, dropout=0.1, batch_first=True)`
    *   *Shape Transformation:* `(Batch, SeqLen, 96) -> (Batch, SeqLen, 96)`
*   **Global Average Pooling:** Summarizes the sequence into a single context vector of dimension 96 by performing masked global average pooling over the temporal dimension. This process averages only the valid observations, clamping the denominator to prevent division by zero for entirely padded sequences.
    *   *Operation:* Masked sum divided by valid count
    *   *Shape Transformation:* `(Batch, SeqLen, 96) -> (Batch, 96)`
*   **Target Feature Conditioning:** The pooled temporal context is concatenated with 56-dimensional target features representing the exact time and context of the prediction (such as dynamic harmonic and elapsed time features for the target date).
    *   *Operation:* `torch.cat([context, targets], dim=1)`
    *   *Shape Transformation:* `(Batch, 96)` and `(Batch, 56)` -> `(Batch, 152)`
*   **Regression Head:** The concatenated vector (152 dimensions) is processed by a Multi-Layer Perceptron (MLP) containing two linear layers. The first layer reduces the dimensionality to 128, followed by a ReLU activation and 0.2 dropout for regularization. The final linear layer outputs the predicted values (e.g., 3-channel predicted spectral values or z-scores).
    *   *Operation:* `Linear(152, 128) -> ReLU -> Dropout(0.2) -> Linear(128, 3)`
    *   *Shape Transformation:* `(Batch, 152) -> (Batch, 128) -> (Batch, 3)`

## Dataset Preparation (`dataset.py`)

The `SITSDataset` class handles the loading and preprocessing of HDF5 SITS data. It is heavily optimized for speed and memory efficiency.

*   **Shared Memory Optimization:** To prevent memory duplication across PyTorch dataloader workers, large HDF5 arrays (z-scores, validity masks, acquisition times) are loaded once into CPU contiguous PyTorch shared memory tensors.
*   **Elastic Temporal Windowing:** Extracts a historical sequence of valid observations over a specific time window (e.g., 3 years). If a pixel has fewer than the required minimum samples (e.g., 38), an "elastic window" mechanism expands the search backwards up to a maximum limit (e.g., 5 years) to ensure sufficient context.
*   **Harmonic and Temporal Features:**
    *   Computes time elapsed between the target date and each historical observation.
    *   Generates multi-scale dynamic harmonic features using predefined temporal periods (1.0, 0.5, and 0.33 years).
    *   Incorporates sinusoidal transformations for Day-of-Year (DOY) and Time-of-Day (TOD).
    *   Normalizes time differences and includes constant and squared time terms.
*   **Sequence Padding & Masking:** Pads all historical sequences to a fixed maximum length (350) and generates binary padding masks for the Transformer's self-attention layers to ignore padded values.
*   **Target Features:** Calculates equivalent temporal and harmonic features for the specific target prediction dates to condition the model's regression head.
