# Adaptive Harmonic Analysis (ALLSSA) Change Detection

This directory contains the implementation of a Adaptive Harmonic Analysis (ALLSSA) pipeline used for detecting structural anomalies in satellite image time series data. 

## Overview of Processing (`ALLSSA_main_pytorch.py`)

The `ALLSSA_main_pytorch.py` script implements a highly-optimized, GPU-accelerated harmonic regression pipeline to detect changes in spectral indices (e.g., sliding volume z-scores) over time. Instead of relying on predefined frequencies (like annual or semi-annual cycles), this method dynamically determines the most prominent frequencies for every pixel within a rolling temporal window.

### Key Processing Steps:

1. **Temporal Conversion & Data Loading:**
   - UNIX timestamps are converted into continuous fractional years to handle irregular sampling (e.g., due to cloud cover or differing satellite overpasses).
   - Data is loaded from an HDF5 file containing Harmonized Landsat/Sentinel-2 (HLS) time series.
   - The script accepts a `--location` command-line argument to specify the target site, which dynamically configures the HDF5 input/output paths.

2. **GPU-Accelerated Dynamic Spatial Chunking:**
   - To manage memory and leverage GPU parallelization efficiently, the spatial domain is divided into discrete chunks.
   - The script dynamically computes the optimal `CHUNK_SIZE` by querying available GPU VRAM (`torch.cuda.mem_get_info`) or CPU RAM, targeting 80% memory allocation to maximize hardware utilization.
   - Processing heavily utilizes PyTorch tensor operations, aligning with best practices for computationally intensive remote sensing tasks.

3. **Dynamic Frequency Estimation:**
   - Instead of static predefined frequencies, the pipeline utilizes a configurable `FREQUENCY_ESTIMATOR` framework to identify the top $K$ dominant frequencies from the irregularly spaced data.
   - The default algorithm is **ALFT (Adaptive / Iterative Least-Squares Frequency Tracking)**. It sequentially extracts dominant frequencies by fitting a harmonic component, computing the residual signal, and evaluating the continuous Fourier integral on the updated orthogonal residuals to find the next frequency.
   - Other supported estimators include:
     - `NDFT`: Static Non-Uniform Discrete Fourier Transform grid search.
     - `NOMP`: Newtonized Orthogonal Matching Pursuit.
     - `CBPDN`: Continuous Basis Pursuit DeNoising (using L1-regularized Adam optimization).
     - `CIRL`: Continuous Iterative Reweighted Least-Squares (using log-penalty Adam optimization).

4. **Adaptive Harmonic Analysis:**
   - A design matrix is constructed dynamically using the top $K$ frequencies identified by the selected estimator. The matrix includes a constant term, along with cosine and sine terms for each frequency.
   - Ordinary Least Squares (OLS) is performed using batched matrix multiplications (`torch.linalg.solve(XtX, Xty)`) to calculate the regression coefficients (amplitudes and phases).

5. **Robust Uncertainty Estimation and Prediction:**
   - The model predicts the expected value for the current target time step.
   - A rigorous statistical prediction bound ($S$) is established. Rather than using classic OLS residual variance (which is sensitive to unmasked clouds or transient noise), the system computes the standard error robustly using **Median Absolute Deviation (MAD)**. 
   - The robust scale factor is calculated as $\sigma_{\text{robust}} = 1.4826 \times \text{MAD}$ (assuming asymptotic normality of inliers).

6. **Anomaly Detection & Consecutive Tracking:**
   - A pixel at a specific time step is flagged as anomalous if the absolute residual (difference between actual and predicted value) exceeds a predefined threshold: `RMSE_MULTIPLIER * S`.
   - To reduce false positives from transient noise, the system tracks consecutive anomalies. A structural change is only confirmed when `CONSECUTIVE_ANOMALIES` (e.g., 4) are detected in a row.

7. **Output Generation:**
   - The pipeline exports the predicted series, dynamic frequency tracks (saved as angular frequencies $\Omega = 2\pi f$ in radians/year), amplitudes, robust RMSE bounds, anomaly flags, and the dates/counts of detected structural changes into an output HDF5 format.

## Configuration Parameters

The script exposes several configurable variables directly at the top of the file:
- `FREQUENCY_ESTIMATOR`: Selection of the gridless/grid-based frequency extraction algorithm (default: `ALFT`).
- `IGNORE_COMMON_MASK`: Whether to rely on robust statistics to filter out cloudy/noisy pixels naturally (when `True`) or explicitly remove them using a pre-calculated mask. 
- `RMSE_MULTIPLIER`: The statistical threshold multiplier for anomaly detection.
- `CONSECUTIVE_ANOMALIES`: The number of sequential anomalous observations required to register a structural change.
- `MAX_WINDOW_YEARS` / `MIN_WINDOW_YEARS`: Lookback window size constraints for fitting the harmonic model.
- `K_FREQUENCIES`: The number of dynamic harmonic components to extract.
- `MIN_SAMPLES`: The minimum valid observations needed to prevent overfitting and ensure robust statistical bounds.

## Visualization (`ALLSSA_vis.py`)

A supporting visualization script is included to interactively inspect the results. It plots the raw data against the dynamic harmonic predictions, highlights the confidence bounds, and visualizes the changing frequencies and amplitudes over time for selected pixels.
