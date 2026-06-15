# **AI Coding Agent Instructions: Multi-Scale 1D-CNN SITS Anomaly Detection**

## **Context and Prime Directives**

You are tasked with implementing a proof-of-concept PyTorch pipeline for Satellite Image Time Series (SITS) anomaly detection. This is for doctoral-level research.

**PROOF OF CONCEPT MANDATE:** To ensure pipeline execution and validate the tensor architecture, data interpolation is explicitly permitted. Do not drop pixels purely for having data gaps; instead, interpolate across masked observations to create dense temporal sequences.

## **1\. Data Ingestion & Preprocessing Module**

Implement a robust PyTorch Dataset class to ingest data from C:\\satelliteImagery\\LANDSAT\\Tait\\LANDSAT\_Stack\_Tait\_GEE\_2015\_2025\_WRS16\_SC\_EM-7\_Gram-minEndmember\_Norm-bandCount.h5.

* **Target Group:** /HDFEOS/GRIDS/LANDSAT/Data Fields/  
* **Masking & Interpolation Logic:** Import and utilize `get_landsat_mask` from `SpecComplex.py` crossing data groups like `QUALITY_L1_PIXEL`, `RADIOMETRIC_SATURATION`, and `QUALITY_L2_AEROSOL` to identify valid observations. Use this mask to set the corresponding `sliding_volume_z_score_masked` and sun geometry values to `np.nan`.  
  * Apply computationally efficient 1D temporal linear interpolation (e.g., via customized `numpy.interp` wrappers) to fill the NaN gaps without relying on Pandas overhead.  
  * If a pixel does not possess a strict minimum of 23 valid historically unmasked observations, flag it as "insufficient data" and exclude it from the active training tensor.  
* **Sequence Extraction (Stride=1):** Extract overlapping windows of 23 frames (20 history, 3 target) from the now-dense interpolated timelines.  
* **Feature Engineering:**  
  * Extract timestamps from surface\_reflectance attributes.  
  * $DOY_{sin}$, $DOY_{cos}$ (from timestamp).  
  * $SunAz_{sin}$, $SunAz_{cos}$ (from sun\_azimuth).  
  * $SunElev_{norm}$: Min-Max normalize sun\_elevation using the global min/max across the *entire* dataset timeline.  
  * $\Delta t'_{log}$: Calculate days since the previous frame. Since data is interpolated, this will generally be the standard acquisition interval, but encode it as np.log(1 \+ delta\_t) to maintain architectural consistency.  
  * Target: sliding\_volume\_z\_score\_masked value. Note: clip values globally to `[-5.0, 5.0]` to prevent unmanageable gradients expanding beyond valid radiometric statistics.
* **Spatial Encoding:** Implement a Fourier Feature Mapping function for the normalized $(x, y)$ coordinates using $L=10$ frequencies to yield a 40-dimensional static vector per pixel.  
* **Chronological Splitting:** \* CalibrationSet: Filter sequences where the timestamp of the 23rd frame is strictly \< '2024-01-01'.  
  * MonitoringSet: Filter sequences where the timestamp of the 23rd frame is \>= '2024-01-01'.

## **2\. Network Architecture Implementation**

Implement the MultiScaleSITSNet as a PyTorch nn.Module.

* **Inputs:** X\_seq shape (Batch, 20, 7), X\_spatial shape (Batch, 40).  
* **Note on PyTorch Conv1D:** PyTorch expects input shape (Batch, Channels, Length). You must permute X\_seq to (Batch, 7, 20\) in the forward pass.  
* **Inception Block (Parallel Branches):**  
  * Branch 1: nn.Conv1d(7, 16, kernel\_size=3, padding='same')  
  * Branch 2: nn.Conv1d(7, 16, kernel\_size=5, padding='same')  
  * Branch 3: nn.Conv1d(7, 16, kernel\_size=7, padding='same')  
* **Concatenation & Pooling:** torch.cat the branches along the channel dimension (out=48). Apply nn.ReLU() and nn.MaxPool1d(kernel\_size=2).  
* **Secondary Extractor:** nn.Conv1d(48, 64, kernel\_size=3, padding='same') \-\> nn.ReLU() \-\> nn.MaxPool1d(kernel\_size=2).  
* **Bridging:** nn.Flatten()  
* **Regression Head:**  
  * torch.cat flattened output with X\_spatial.  
  * nn.Linear(flattened\_dim \+ 40, 128\) \-\> nn.ReLU() \-\> nn.Dropout(0.2).  
  * nn.Linear(128, 3\) \-\> Linear Activation (no activation function).

## **3\. Training & Weight Preservation**

* **Optimizer:** Use torch.optim.AdamW. AdamW decouples weight decay from the gradient update, providing vastly superior generalization for CNNs over standard Adam.  
* **Loss Function:** torch.nn.HuberLoss(delta=1.0).  
* **Training Loop:** Train *exclusively* on the CalibrationSet (pre-2024) using a standard DataLoader.  
* **State Preservation:** Upon completion, save the weights strictly to disk: torch.save(model.state\_dict(), 'sits\_baseline\_weights\_pre2024.pth').

## **4\. Inference & Monitoring Period**

* **Phase 1: Baseline RMSE Calculation:** Set model to eval(). Run a forward pass over the entire CalibrationSet. Calculate the global Root Mean Square Error (RMSE) between predictions and actuals. Store this scalar baseline\_rmse.  
* **Phase 2: Monitoring Inference:** Run a forward pass over the MonitoringSet (post-2024 data).  
* **Anomaly Detection Logic:** \* Calculate residuals for the 3 predicted frames.  
  * Trigger Anomaly: if mean(abs(residual)) \> (3.0 \* baseline\_rmse): is\_anomaly \= True

## **5\. Result Serialization & Visualization**

* **Serialization:** Save the inference results to a structured HDF5 file (inference\_results.h5) to maintain strict metrological consistency with the source data. The HDF5 file must contain a dataset or a structured compound array with the following explicit columns and data types:  
  * Pixel\_X, Pixel\_Y: Integer spatial coordinates.  
  * Timestamp\_T21, Timestamp\_T23: float64 UNIX timestamps (UTC) extracted from the acquisition\_time attribute of the 21st and 23rd frames in the sequence, allowing anomalies to be mapped to exact collection dates.  
  * Pred\_1, Pred\_2, Pred\_3: float32 continuous scalars. These are the direct predictions from the 1D-CNN's regression head for the 3 target frames.  
  * Actual\_1, Actual\_2, Actual\_3: float32 continuous scalars. The ground-truth sliding\_volume\_z\_score\_masked values from the source HDF5 for the matching timestamps.  
  * Mean\_Residual: float32 positive scalar. The Mean Absolute Error (MAE) across the 3-frame forecast window: $\frac{1}{3}\sum_{i=1}^3 |y_i - \hat{y}_i|$.  
  * Anomaly\_Flag: Boolean (True/False or uint8 1/0). Evaluates to True strictly if the Mean\_Residual $> 3 \times \text{RMSE}_{baseline}$.  
* **Quantifiable Metrics Output:** Print a standard console report containing:  
  * Baseline RMSE (Pre-2024).  
  * Total sequences evaluated in Monitoring Set.  
  * Total Anomalies flagged.  
  * Anomaly rate (%).  
* **1D Time-Series Visualization:** Implement plot\_pixel\_sits(pixel\_results).  
  * X-axis: Time. Y-axis: sliding\_volume\_z\_score\_masked value.  
  * **Visual Distinction for Data Validity:** Utilize the original QUALITY\_L1\_PIXEL mask to differentiate data points. Plot valid/used historical observations as **filled black dots**, and masked/invalid (interpolated) observations as **open circles**. Plot post-2024 actuals using the same logic (e.g., filled blue dots for valid, open blue circles for interpolated).  
  * Plot predictions (red dashed line) with a shaded confidence interval representing $\pm 3 \times \text{RMSE}_{baseline}$. Highlight anomalies with red markers.  
* **Spatial Anomaly Overlay Visualization:** Implement plot\_spatial\_anomaly\_overlay(source\_h5\_path, inference\_results).  
  * **Base Layer Initialization:** Extract the ortho\_visual frame corresponding to the date **2025-09-12** from the source HDF5 file (converted from BSQ to BIP format as \[H, W, C\]). Set this as the default background image.  
  * Create a 2D map of the pixel grid.  
  * **Overlay Logic:**  
    1. **Anomalies:** For pixels accumulating *more than 2 overall anomaly sequence flags* across the evaluation span, color them using a chronological colormap (e.g., matplotlib viridis) mapped to the *first date/timestamp* a sequence anomaly was detected on that spatial coordinate. Add a colorbar indicating the date range (2024-2025).  
    2. **No/Minor Anomalies:** Leave pixels completely transparent (alpha=0) so the underlying ortho\_visual imagery is clearly visible. This acts as a spatial filter for sequence anomaly counts `<= 2`.
    3. **Insufficient Data:** Color pixels solid gray (\#808080 with 50% opacity) where there were insufficient valid observations to perform interpolation/inference.  
  * **Interactivity & Dynamic Base Layer Update:** Implement an interactive click-event listener on the spatial map (e.g., using matplotlib's mpl\_connect('button\_press\_event', onclick) or interactive frameworks like plotly or bokeh). When a user clicks on a specific pixel in the 2D grid:  
    1. **Trigger 1D Time-Series:** Dynamically trigger and display the plot\_pixel\_sits() 1D time-series visualization for that exact pixel's coordinate.  
    2. **Dynamic Base Layer Update:** \* If the clicked pixel is flagged as an **Anomaly**, dynamically update the underlying ortho\_visual base layer to the frame that corresponds with the **first date/timestamp** the anomaly was detected.  
       * If the clicked pixel has **No Anomaly**, default the underlying ortho\_visual base layer back to the **2025-09-12** frame.

## **References to Implement**

* Loshchilov, I., & Hutter, F. (2019). Decoupled Weight Decay Regularization. *International Conference on Learning Representations (ICLR)*.  
* Ismail Fawaz, H., et al. (2020). InceptionTime: Finding AlexNet for time series classification. *Data Mining and Knowledge Discovery*.