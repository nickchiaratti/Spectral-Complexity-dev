# OOD Frequency Autoencoder Change Detection Pipeline

This directory contains the Out-of-Distribution (OOD) Autoencoder pipeline for identifying spectral anomalies in satellite imagery time series (SITS). The primary orchestration script is `OOD_autoencoder_main.py`.

## Pipeline Overview (`OOD_autoencoder_main.py`)

The OOD Autoencoder pipeline operates entirely in the frequency domain, taking advantage of the Non-Uniform Fast Fourier Transform (NUFFT) to handle irregularly sampled time series natively.

1. **Configuration**: Configures data paths, hyper-parameters (e.g., latent dimension, batch size, contamination rate), and spatial constraints.
2. **Data Loading**: Loads the time series and acquisition times via `TimeSeriesH5Dataset`.
3. **Training**: Trains the `FrequencyAutoencoder` on the data. A key architectural decision is the use of **L1 Loss (Mean Absolute Error)** instead of MSE. This is critical to prevent the model from heavily penalizing and subsequently learning the out-of-distribution shifts.
4. **Scoring**: Computes reconstruction errors (L1 distance between true and reconstructed amplitude spectrums) to generate anomaly scores. High reconstruction error indicates an out-of-distribution (anomalous) pixel.

## Model Architecture (`models.py`)

The core architecture is the `FrequencyAutoencoder`, an autoencoder designed to compress and reconstruct the amplitude spectrum of irregular time series data.

*   **Non-Uniform FFT (NUFFT)**: Utilizes `torchkbnufft.KbNufftAdjoint` (Type 1 NUFFT) to process irregular, gappy data natively without interpolation. Time coordinates are mapped to the required `[-pi, pi]` range.
    *   *Operation:* `tkbn.KbNufftAdjoint(im_size=(num_bins,))`
    *   *Shape Transformation:* `points: (1, 1, SeqLen), values: (Batch, 1, SeqLen)` -> `(Batch, 1, num_bins)`
*   **Amplitude Spectrum**: Slices the positive frequencies from the complex NUFFT output and computes the absolute amplitude spectrum.
    *   *Operation:* `freq_complex[:, num_bins // 2:]` followed by `torch.abs()`
    *   *Shape Transformation:* `(Batch, num_bins)` -> `(Batch, freq_length)` where `freq_length = num_bins - (num_bins // 2)`
*   **Encoder**: A Multi-Layer Perceptron (MLP) that reduces the frequency dimensionality down to a bottleneck vector (`latent_dim`).
    *   *Operation:* `Linear(freq_length, 128) -> ReLU -> Linear(128, 64) -> ReLU -> Linear(64, latent_dim)`
    *   *Shape Transformation:* `(Batch, freq_length) -> (Batch, 128) -> (Batch, 64) -> (Batch, latent_dim)`
*   **Decoder**: Reconstructs the amplitude spectrum from the latent representation. The final layer applies a `ReLU` activation because frequency amplitudes are strictly non-negative.
    *   *Operation:* `Linear(latent_dim, 64) -> ReLU -> Linear(64, 128) -> ReLU -> Linear(128, freq_length) -> ReLU`
    *   *Shape Transformation:* `(Batch, latent_dim) -> (Batch, 64) -> (Batch, 128) -> (Batch, freq_length)`

## Dataset Preparation (`dataset.py`)

The `TimeSeriesH5Dataset` class loads the spatiotemporal data and prepares it for the frequency-domain autoencoder.

*   **Memory Management**: Loads the entire 3D HDF5 dataset into memory for fast access, flattening it spatially into `(num_pixels, time_steps)`.
*   **Irregularity Handling (Zero-Padding)**: For Type 1 NUFFT, any invalid or masked point (due to clouds, shadows, or NaNs) is simply zeroed out. A coefficient of 0.0 perfectly ignores the point in the NUFFT algorithm, preserving the physical reality of the data without injecting synthetic or interpolated values.
*   **Time Scaling**: Retrieves raw acquisition times and strictly scales them to the `[-pi, pi]` range as required by the `torchkbnufft` implementation.
*   **Batch Structure**: Returns the mapped points (time coordinates) and the valid values for each pixel simultaneously.
