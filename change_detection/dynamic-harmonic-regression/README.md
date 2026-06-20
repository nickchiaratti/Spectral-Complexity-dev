# Dynamic Harmonic Regression (DHR) Change Detection

This directory contains the implementation of a Dynamic Harmonic Regression (DHR) pipeline used for detecting structural anomalies in satellite image time series data. 

## Overview of Processing (`dhr_main.py`)

The `dhr_main.py` script implements a highly-optimized, GPU-accelerated harmonic regression pipeline to detect changes in spectral indices (e.g., sliding volume z-scores) over time. Instead of relying on predefined frequencies (like annual or semi-annual cycles), this method dynamically determines the most prominent frequencies for every pixel within a rolling temporal window.

### Key Processing Steps:

1. **Temporal Conversion & Data Loading:**
   - UNIX timestamps are converted into continuous fractional years to handle irregular sampling (e.g., due to cloud cover or differing satellite overpasses).
   - Data is loaded from an HDF5 file containing Harmonized Landsat/Sentinel-2 (HLS) time series.

2. **GPU-Accelerated Spatial Chunking:**
   - To manage memory and leverage GPU parallelization efficiently, the spatial domain is divided into discrete chunks (e.g., 256x256 pixels). 
   - Processing heavily utilizes PyTorch tensor operations, aligning with best practices for computationally intensive remote sensing tasks.

3. **Non-Uniform Discrete Fourier Transform (NUDFT):**
   - For each pixel, within a defined lookback window (e.g., 5 years), an NUDFT is computed. This explicitly handles the irregularly spaced data without requiring synthetic interpolation or temporal smoothing (preserving the raw reality of the input datasets).
   - The calculation evaluates the continuous Fourier integral as a discrete summation directly over the irregular acquisition times ($T_{win}$).
   - The frequency grid ($\Omega$) is defined as a set of continuous frequencies to probe (e.g., $f_{grid} \in [0.2, 4.0]$ Hz).
   - A complex exponential basis ($E$) is constructed directly mapping every frequency to every non-uniform time step ($E = e^{-i \Omega T_{win}}$).
   - The valid active signal ($Y_{active}$) is mean-centered to remove the zero-frequency (DC) component.
   - The amplitude spectrum is derived from the matrix multiplication of the exponential basis $E$ and the centered signal $Y_{active\_centered}$, mathematically representing the summation: $X(\omega) = \left| \sum x(t_n) e^{-i \omega t_n} \right|$. 
   - By using batched dense matrix multiplication operations on the GPU (`torch.matmul`), the periodogram is evaluated extremely fast across the specific frequency grid to identify the top $K$ dominant frequencies ($K$ is typically 2). This direct evaluation is computationally superior to gridding-based NUFFT approaches (like Kaiser-Bessel interpolation) due to the highly-targeted narrow frequency band and short sequence lengths, maximizing GPU Tensor Core utilization and avoiding sparse memory overheads.

4. **Dynamic Harmonic Regression:**
   - A design matrix is constructed dynamically using the top $K$ frequencies identified by the NDFT. The matrix includes a constant term, along with cosine and sine terms for each frequency.
   - Ordinary Least Squares (OLS) is performed using batched matrix multiplications (`torch.linalg.solve(XtX, Xty)`) to calculate the regression coefficients (amplitudes and phases).

5. **Uncertainty Estimation and Prediction:**
   - The training root-mean-square error (RMSE) is calculated based on the degrees of freedom.
   - The model predicts the expected value for the current target time step.
   - A rigorous statistical prediction bound is established by computing the standard error of prediction ($S$), which accounts for both the residual variance and the uncertainty of the regression estimates.

6. **Anomaly Detection & Consecutive Tracking:**
   - A pixel at a specific time step is flagged as anomalous if the absolute residual (difference between actual and predicted value) exceeds a predefined threshold: `RMSE_MULTIPLIER * S`.
   - To reduce false positives from transient noise (e.g., undetected clouds or shadows), the system tracks consecutive anomalies. A structural change is only confirmed when `CONSECUTIVE_ANOMALIES` (e.g., 4) are detected in a row.

7. **Output Generation:**
   - The pipeline exports the predicted series, dynamic frequency tracks, amplitudes, RMSE bounds, anomaly flags, and the dates/counts of detected structural changes into an output HDF5 format.

## Configuration Parameters

The script exposes several configurable variables directly at the top of the file:
- `IGNORE_COMMON_MASK`: Whether to rely on the NDFT to filter out cloudy/noisy pixels naturally (when `True`) or explicitly remove them using a pre-calculated mask. 
- `RMSE_MULTIPLIER`: The statistical threshold multiplier for anomaly detection.
- `CONSECUTIVE_ANOMALIES`: The number of sequential anomalous observations required to register a structural change.
- `MAX_WINDOW_YEARS` / `MIN_WINDOW_YEARS`: Lookback window size constraints for fitting the harmonic model.
- `K_FREQUENCIES`: The number of dynamic harmonic components to extract.
- `MIN_SAMPLES`: The minimum valid observations needed to prevent overfitting and ensure robust statistical bounds.

## Visualization (`dhr_vis.py`)

A supporting visualization script is included to interactively inspect the results. It plots the raw data against the dynamic harmonic predictions, highlights the confidence bounds, and visualizes the changing frequencies and amplitudes over time for selected pixels.
