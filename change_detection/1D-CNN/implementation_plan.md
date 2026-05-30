# 1D-CNN Multi-Scale SITS Anomaly Detection

This plan implements a PyTorch anomaly detection pipeline based on a 1D Convolutional Neural Network processing Satellite Image Time Series (SITS). This will be achieved by writing a central Python script, `main.py`, in the `c:\satelliteImagery\AnomalyDetector\1D-CNN` directory, and auxiliary modules if necessary.

## User Review Required

> [!IMPORTANT]
> - Note that the `plot_spatial_anomaly_overlay` function uses `matplotlib` interactivity event loops (`mpl_connect`). This allows clicking pixels to spawn the 1D time-series visualization. This will run locally and open UI windows on your machine.
> - The dataset could take significant time to process if we loop over all pixels sequentially dynamically in Python. Optimizations like processing valid pixels by flattening them and extracting rolling windows vectorized over pixels will be used to maintain performance while preserving data validity. Please confirm this is acceptable.

## Proposed Changes

### 1D-CNN Model and Pipeline Implementation

#### [NEW] [dataset.py](file:///c:/satelliteImagery/AnomalyDetector/1D-CNN/dataset.py)
This module will define the data ingestion and preprocessing pipeline.
- Implement `SITSDataset` extending `torch.utils.data.Dataset`.
- Load the HDF5 data fields: `sliding_volume_z_score_masked`, `QUALITY_L1_PIXEL`, `surface_reflectance`.
- Extract timestamps and calculate `DOY`, time deltas, and normalized `sun_elevation` and `sun_azimuth` per frame.
- Utilize `get_landsat_mask` from `SpecComplex.py` crossing multiple QA tensors to evaluate masking. Masked pixels will have 1D linear interpolation using computationally efficient `numpy.interp` applied sequentially for `z_score`, `sun_elevation`, etc.
- Exclude pixels that have less than 23 valid initial unmasked observations.
- Globally clip the input `sliding_volume_z_score_masked` to `[-5.0, 5.0]` to prevent extreme radiometrically-induced outliers from destabilizing gradient descent.
- Split generating 23-frame rolling windows (20 history, 3 target) grouped into Calibration (target pre-2024-01-01) and Monitoring (target post-2024-01-01) datasets.
- Generate Fourier Feature mappings (40 dimensions) for spatial coordinates.

#### [NEW] [model.py](file:///c:/satelliteImagery/AnomalyDetector/1D-CNN/model.py)
This module will define the core neural network PyTorch architecture `MultiScaleSITSNet`.
- A multi-branch `InceptionTime` block utilizing parallel `Conv1d` layers (kernels 3, 5, 7), concatenated and pooled.
- A secondary Conv1D temporal feature extractor.
- Flatten and concatenate with static spatial features (40 dims).
- Regression Head predicting 3 contiguous targets applying linear activations.

#### [NEW] [train_evaluate.py](file:///c:/satelliteImagery/AnomalyDetector/1D-CNN/train_evaluate.py)
Handles training and inference logic.
- Optimizer: `AdamW`. Loss: `HuberLoss`.
- Train strictly on Calibration dataset and save weights to `sits_baseline_weights_pre2024.pth`.
- Compute Pre-2024 `baseline_rmse`.
- Run Inference on Monitoring dataset.
- Calculate residual and flag anomalies if `mean(abs(residual)) > 3.0 * baseline_rmse`.
- Output results array mapping `[Pixel_X, Pixel_Y, Timestamp_T21, Timestamp_T23, Pred_1, Pred_2, Pred_3, Actual_1, Actual_2, Actual_3, Mean_Residual, Anomaly_Flag]` into `inference_results.h5`.

#### [NEW] [visualization.py](file:///c:/satelliteImagery/AnomalyDetector/1D-CNN/visualization.py)
- `plot_pixel_sits(pixel_results)`: Displays historical validity vs predictions (with RMSE confidence bounds) on a 1D graph.
- `plot_spatial_anomaly_overlay(source_h5_path, inference_results)`: Side-by-side subplot visualization employing `matplotlib` interaction. Plots the spatial anomaly indicator overlaid on the ortho reference frame. Anomalous sequences (> 2 flags) are colored by first occurrence. Clicking a pixel updates the adjacent 1D timeline visualization pane dynamically and updates the base map frame along with a temporal reference line.

#### [NEW] [main.py](file:///c:/satelliteImagery/AnomalyDetector/1D-CNN/main.py)
- The main entry point tying all modules together, orchestrating execution flow.

## Open Questions
- To apply interpolation efficiently over the time dimension for multiple pixels, the `dataset.py` might pre-process this into a memory-mapped array or iterate efficiently in numpy format to avoid excessive Pandas Series initializations.

## Verification Plan

### Automated Tests
- Syntax parsing and test imports will be verified.

### Manual Verification
- We will execute `main.py` against the input data source. We will verify successful completion, that `inference_results.h5` is correctly generated, the CLI metrics are produced, and interactive Matplotlib map opens correctly with your data flow.
