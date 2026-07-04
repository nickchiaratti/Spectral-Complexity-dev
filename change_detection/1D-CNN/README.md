# 1D-CNN Change Detection Pipeline

This directory contains a 1D Convolutional Neural Network (CNN) pipeline for satellite imagery time series (SITS) change detection and forecasting. The primary orchestration script is `.CNN_main.py`.

## Pipeline Overview (`.CNN_main.py`)

The pipeline script defines the execution flow and configures hyper-parameters:
1. **Configuration:** Sets the target location, training end date, and paths for input/output data. It defines specific SITS windowing parameters like `TIME_WINDOW_YEARS`, `MIN_SAMPLES`, `ENABLE_ELASTIC_WINDOW`, and `MAX_ELASTIC_WINDOW_YEARS`.
2. **Train and Evaluate:** Invokes `train_and_evaluate` to handle model training and testing. It applies Monte Carlo (MC) sampling to estimate uncertainty and generates an inference HDF5 file containing the anomaly detections and PnPXAI attribution metrics (Sensitivity & Complexity).
3. **Visualization:** If the inference results are generated successfully, it calls `plot_spatial_anomaly_overlay` to map the detected anomalies across space and time.

## Model Architecture (`model.py`)

The core architecture, `MultiScaleSITSNet`, relies on 1D convolutions across the time dimension to extract temporal patterns at multiple receptive fields.

Here is the detailed layer-by-layer breakdown:

1. **Input Permutation**: The input sequence `(Batch, SeqLen, in_channels)` is transposed to `(Batch, in_channels, SeqLen)` for 1D convolutions.
2. **Inception Block (Multi-Scale Feature Extraction)**: 
   - Three parallel `Conv1d` branches with kernel sizes `5`, `7`, and `9`, each taking `in_channels` and outputting `32` channels with `padding='same'`.
   - The outputs of the three branches are concatenated along the channel dimension to form a `96`-channel tensor.
   - A `ReLU` activation is applied.
   - The output is multiplied by the temporal sequence mask to properly ignore padded values.
3. **First Pooling Block**:
   - `MaxPool1d(kernel_size=2)` is applied to halve the sequence length. The mask is similarly pooled.
   - `Dropout1d(0.2)` drops entire channels randomly to prevent overfitting.
4. **Secondary Extractor**:
   - A single `Conv1d` layer (in: 96, out: 64, kernel: 7, padding='same') refines the extracted multi-scale features.
   - A `ReLU` activation is applied, followed again by mask multiplication.
5. **Second Pooling Block**:
   - `MaxPool1d(kernel_size=2)` is applied again. The mask is similarly pooled.
   - `Dropout1d(0.2)` is applied.
6. **Global Average Pooling (GAP)**:
   - A masked Global Average Pooling operation sums the remaining temporal features and divides by the sum of the mask, compressing the sequence into a flat `(Batch, 64)` vector.
7. **Regression Head**:
   - The pooled 64-dimensional feature vector is concatenated with the encoded spatial features (`spatial_dim=40`) and the target features (`target_features_dim=6`).
   - `Linear(64 + 40 + 6 = 110, 128)`
   - `ReLU` activation
   - `Dropout(0.2)`
   - `Linear(128, out_features)` (where `out_features=1`) to output the final prediction.

## Encoded Dataset Values (`dataset.py`)

The `SITSDataset` prepares a rich set of temporal, spatial, and cyclical features for each valid observation in the sequence, making up the `in_channels` for the model. The encoded values include:

*   **Pixel Value**: The actual standardized observation (`pixel_z`).
*   **Cyclical Time-of-Day (TOD)**: `sin` and `cos` encoded continuous hour of acquisition (`pixel_tod_sin`, `pixel_tod_cos`).
*   **Cyclical Day-of-Year (DOY)**: `sin` and `cos` encoded day of the year (`pixel_doy_sin`, `pixel_doy_cos`).
*   **Fourier-Encoded Elapsed Time Delta**: The elapsed time in years between the historical observation and the target forecast date (`dt_years`) is Fourier-encoded into orthogonal sine and cosine components (`dt_years_sin`, `dt_years_cos`). This avoids injecting rigid multi-year periodicity assumptions while providing continuous phase metrics.
*   **Spatial Features**: Multi-frequency Fourier features capturing the normalized spatial coordinates (X and Y), producing 40 dimensions (20 features for X and 20 features for Y, derived from 10 sine/cosine frequencies). This allows pixel-specific baseline calibration.
*   **Dynamic Sequence Length**: The dataset class scans valid data sequences at initialization to determine the maximum observed sequence length (`max_seq_len`) and checks it against an upper limit of `150`.
*   **Elastic Windowing**: An advanced sequence slicing logic that looks back up to `max_elastic_window_years` if the static `time_window_years` does not meet the minimum required sample count (`MIN_SAMPLES`).

## Model Training & Optimization (`train_evaluate.py`)

*   **Optimizer**: Trained using `AdamW` with selective weight decay:
    *   Weight decay (`1e-2`) is applied only to 2D/3D weights (Conv filters).
    *   Weight decay is disabled (`0.0`) on biases, 1D parameters, and linear projection weights.
*   **Loss Function**: `HuberLoss(delta=3.0)` to handle noisy training residuals robustly.
*   **Uncertainty Estimation**: Monte Carlo (MC) dropout is applied during inference to estimate epistemic uncertainty.

## Visualization Features (`visualization.py`)

*   **Aesthetics**: Utilizes `scienceplots` styling (`['science', 'no-latex']`).
*   **Twin-Axis Layout**: Shifted twin-axis labels (`Attribution %` and `Complexity`) explicitly to the right side of the plot using `.yaxis.set_label_position("right")` to prevent left-side overlapping caused by `scienceplots` overrides.
*   **Anomaly Scatter Alignment**: Red anomaly markers (`Anomaly (Unconfirmed)`) are correctly scattered on actual observed values (`Actual_1`) rather than the prediction line.
*   **Spatial Overlay Map**: Uses the cyclic `hsv` colormap to map the detected anomaly dates across space.
*   **Anomaly Flagging Constraint**: Anomaly detection and flagging is restricted strictly to the monitoring period (post-`TRAIN_END_YEAR`) because pixel-specific spatial calibration ensures very low training residuals and high baseline accuracy.
