# 1D-CNN Generalized Change Detection Pipeline

This directory contains a generalized 1D Convolutional Neural Network (CNN) pipeline for satellite imagery time series (SITS) change detection and forecasting. The primary orchestration script is `.CNN_generalized_main.py`.

## Pipeline Overview (`.CNN_generalized_main.py`)

The pipeline script defines the execution flow and configures hyper-parameters:
1. **Configuration:** Sets the target location, training end date, and paths for input/output data. It defines specific SITS windowing parameters like `TIME_WINDOW_YEARS`, `MIN_SAMPLES`, and a `TEMPORAL_DECAY_RATE`.
2. **Train and Evaluate:** Invokes `train_and_evaluate` to handle model training and testing. It applies Monte Carlo (MC) sampling to estimate uncertainty and generates an inference HDF5 file containing the anomaly detections.
3. **Visualization:** If the inference results are generated successfully, it calls `plot_spatial_anomaly_overlay` to map the detected anomalies.

## Model Architecture (`model.py`)

The core architecture, `MultiScaleSITSNet`, relies on 1D convolutions across the time dimension to extract temporal patterns at multiple receptive fields. 

Here is the detailed layer-by-layer breakdown:

1. **Input Permutation**: The input sequence `(Batch, SeqLen, in_channels)` is transposed to `(Batch, in_channels, SeqLen)` for 1D convolutions.
2. **Inception Block (Multi-Scale Feature Extraction)**: 
   - Three parallel `Conv1d` branches with different kernel sizes (`3`, `5`, and `7`), each taking `in_channels` and outputting `32` channels with `padding='same'`.
   - The outputs of the three branches are concatenated along the channel dimension to form a `96`-channel tensor.
   - A `ReLU` activation is applied.
   - The output is multiplied by the temporal sequence mask to properly ignore padded values.
3. **First Pooling Block**:
   - `MaxPool1d(kernel_size=2)` is applied to halve the sequence length. The mask is similarly pooled.
   - `Dropout1d(0.2)` drops entire channels randomly to prevent overfitting.
4. **Secondary Extractor**:
   - A single `Conv1d` layer (in: 96, out: 64, kernel: 3, padding='same') refines the extracted multi-scale features.
   - A `ReLU` activation is applied, followed again by mask multiplication.
5. **Second Pooling Block**:
   - `MaxPool1d(kernel_size=2)` is applied again.
   - `Dropout1d(0.2)` is applied.
6. **Global Average Pooling (GAP)**:
   - A masked Global Average Pooling operation sums the remaining temporal features and divides by the sum of the mask, compressing the sequence into a flat `(Batch, 64)` vector.
7. **Regression Head**:
   - The pooled 64-dimensional feature vector is concatenated with the encoded target features (the exact time step to predict).
   - `Linear(64 + target_features_dim, 128)`
   - `ReLU` activation
   - `Dropout(0.2)`
   - `Linear(128, out_features)` to output the final predictions.

## Encoded Dataset Values (`dataset.py`)

The `SITSDataset` prepares a rich set of temporal and harmonic features for each valid observation in the sequence, making up the `in_channels` for the model. The encoded values include:

*   **Pixel Value**: The actual standardized observation (`pixel_z`).
*   **Cyclical Time-of-Day (TOD)**: `sin` and `cos` encoded continuous hour of acquisition (`pixel_tod_sin`, `pixel_tod_cos`).
*   **Cyclical Day-of-Year (DOY)**: `sin` and `cos` encoded day of the year (`pixel_doy_sin`, `pixel_doy_cos`).
*   **Elapsed Time**: Normalized time difference in years between the historical observation and the target forecast date (`dt_years_norm`).
*   **Multi-Scale Harmonic Features**: Dynamic sine and cosine pairs representing elapsed time wrapped over different predefined temporal periods (1.0, 0.5, 0.33, and 0.25 years). This yields 8 distinct harmonic features.
*   **Temporal Decay Masking**: Instead of a strict binary padding mask, the sequence mask incorporates a temporal decay weight (`1.0 - exp(-temporal_decay_rate * delta_t)`). This allows the model to naturally down-weight older observations in the temporal aggregation steps.
