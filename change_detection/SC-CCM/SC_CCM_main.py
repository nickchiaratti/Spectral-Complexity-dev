"""
Harmonized Continuous Change Monitoring (SC-CCM) Pipeline.

This module implements continuous change detection and monitoring for satellite 
time-series observations (e.g., Harmonized Landsat Sentinel-2 and Tanager-1 
spectral complexity and biophysical index cubes).

Theoretical Framework:
----------------------
1. Harmonic Fourier Regression:
   Models cyclical seasonal surface dynamics and long-term trends:
       y(t) = a_0 + c_1*t + c_2*t^2 + sum_{k=1}^K [a_k*cos(2*pi*t / T_k) + b_k*sin(2*pi*t / T_k)]
   where t is continuous time in fractional years and T_k represents temporal harmonic 
   periods (e.g., annual, semi-annual, trimonthly cycles).

2. Sparse Model Selection & Outlier Rejection:
   A stable initialization window is selected spanning at least a minimum temporal 
   baseline (e.g., >= 1.5 years). Least Absolute Shrinkage and Selection Operator 
   (LASSO) via the LARS algorithm with Bayesian Information Criterion (BIC) performs 
   sparse harmonic selection. Normalized squared residuals are tested against a 
   Chi-squared distribution (df=3) to iteratively remove statistical outliers.

3. Parameter & Covariance Initialization:
   Ordinary Least Squares (OLS) estimates initial parameters on active features, 
   initializing the parameter vector theta_0 and the parameter covariance matrix P_0.

4. Online Recursive Least Squares (RLS) Tracking:
   As new observations arrive sequentially:
     - Step-ahead prediction: y_hat_t = x_t^T * theta_{t-1}
     - Innovation residual: e_t = y_t - y_hat_t
     - Chi-squared anomaly test: chi2_t = (abs(e_t) / RMSE)^2
     - Structural break confirmation: If consecutive anomalies >= threshold, a break 
       is confirmed at the first anomaly of the run, terminating the current segment.
     - In-control parameter update (when chi2_t <= threshold):
         K_t = (P_{t-1} * x_t) / (lambda + x_t^T * P_{t-1} * x_t)
         theta_t = theta_{t-1} + K_t * e_t
         P_t = (1 / lambda) * (P_{t-1} - K_t * x_t^T * P_{t-1})
       where lambda is the exponential forgetting factor.
"""


import sys
from pathlib import Path
_script_dir = Path(__file__).resolve().parent
_repo_root = _script_dir.parent.parent
for _p in [_repo_root, _script_dir, _script_dir.parent / "pyccd"]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import math
import time
import gc
import multiprocessing
import contextlib
import h5py
import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.linear_model import LassoLarsIC
import joblib
from joblib import Parallel, delayed
from tqdm import tqdm


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar."""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def read_scaled_int16(arr, scale_factor=10000, nodata_val=-9999):
    """
    Invert int16 quantized array back to float32 reflectance.
    Preserves nodata sentinels as NaN without injecting synthetic fill values.
    """
    arr = np.asarray(arr)
    out = np.full(arr.shape, np.nan, dtype=np.float32)
    valid = (arr != nodata_val)
    out[valid] = arr[valid].astype(np.float32) / float(scale_factor)
    return out


def scale_to_int16(raw_floats, scale_factor=10000, nodata_val=-9999, nodata_value=None):
    """
    Quantizes float32 array to int16, mapping NaNs to nodata value.
    """
    if nodata_value is not None:
        nodata_val = nodata_value
    raw_floats = np.asarray(raw_floats)
    out = np.full(raw_floats.shape, nodata_val, dtype=np.int16)
    valid = ~np.isnan(raw_floats)
    scaled = np.round(raw_floats[valid] * float(scale_factor))
    clipped = np.clip(scaled, -32768, 32767)
    out[valid] = clipped.astype(np.int16)
    return out


def resolve_metric_naming(metric_name, injected=False):
    """Resolve display name, short prefix, and dataset key from target metric identifier."""
    m_raw = str(metric_name).strip()
    m_lower = m_raw.lower()
    is_inj = injected or m_lower.startswith('injected_') or 'injected' in m_lower
    base = m_lower.replace('injected_', '').replace('injected', '').strip('_')

    if 'z_score' in base or base in ('sc', 'sliding_volume_z_score'):
        disp = 'Spectral Complexity (Z-Score)'
        short = 'SC'
        ds_key = 'injected_sliding_volume_z_score' if is_inj else 'sliding_volume_z_score'
    elif 'ndvi' in base:
        disp = 'NDVI'
        short = 'NDVI'
        ds_key = 'injected_ndvi' if is_inj else 'ndvi'
    else:
        clean = base.replace(' ', '_').upper()
        disp = metric_name
        short = clean
        ds_key = f"injected_{m_raw}" if (is_inj and not m_lower.startswith('injected_')) else m_raw

    group_prefix = f"INJECTED_{short}" if is_inj else short
    return disp, short, ds_key, group_prefix, is_inj


def resolve_h5_input(location, h5_path=None):
    """Resolve location or explicit container path to an existing HDF5 file."""
    if h5_path and os.path.exists(h5_path):
        return h5_path
    candidate = f"C:/satelliteImagery/HLST30/HLST_{location}.h5"
    if os.path.exists(candidate):
        return candidate
    return h5_path or candidate


# ==========================================
# CONFIGURATION & HYPERPARAMETERS
# ==========================================
LOCATION = "Rochesterv2"
H5_PATH = f"C:/satelliteImagery/HLST30/HLST_{LOCATION}.h5"

TARGET_METRIC = 'sliding_volume_z_score'
TARGET_NAME, TARGET_SHORT_NAME, *_ = resolve_metric_naming(TARGET_METRIC)

# Anomaly Detection Thresholds
CHANGE_PROBABILITY = 0.99
CONSECUTIVE_ANOMALIES = 4
CHI2_DEGREES_OF_FREEDOM = 3

# Segment Initialization & Stability Constraints
MAX_RMSE = 1.0
MIN_RMSE_CLAMP = 1e-5
MIN_YEARS_FOR_INIT = 1.5
MIN_SAMPLES = 16

# Harmonic Regression Configurations
ENABLE_CONSTANT = True
ENABLE_LINEAR = False
ENABLE_QUADRATIC = False
TEMPORAL_PERIODS = [0.33, 0.5, 1]

# Recursive Least Squares Tuning
RLS_RIDGE_PENALTY = 1e-6
RLS_FORGETTING_FACTOR = 0.999

# Streaming Block Execution Parameters
BLOCK_SIZE_ROWS = 64    # Streaming block size (rows) written directly to HDF5
WORKER_CHUNK_ROWS = 8   # Rows per parallel worker task

# ==========================================
#  DATA INGESTION & SCALING
# ==========================================
def read_scaled_dataset(ds, r_slice=slice(None), c_slice=slice(None)):
    """
    Read HDF5 dataset slice, scaling raw values to float32 and masking fill values as NaN.
    Extracts '_FillValue' and 'scale_factor' from dataset attributes without hardcoding.
    """
    fill_val = ds.attrs.get("_FillValue", ds.attrs.get("fillvalue", getattr(ds, "fillvalue", None)))
    scale = float(ds.attrs.get("scale_factor", 1.0))
    if "scale_to_float" in ds.attrs:
        sf = float(ds.attrs["scale_to_float"])
        scale = (1.0 / sf) if sf < 1.0 else sf

    raw = ds[:, r_slice, c_slice]
    out = raw.astype(np.float32) / scale
    if fill_val is not None and not np.isnan(fill_val):
        is_fill = np.isclose(raw, fill_val, equal_nan=True) if np.issubdtype(raw.dtype, np.floating) else (raw == fill_val)
        out[is_fill] = np.nan
    return out


def _resolve_metric_dataset(data_grp, target_metric, metric_key, is_inj):
    """Select the HDF5 dataset matching baseline or injected metric naming."""
    candidates = [metric_key]
    if is_inj:
        if metric_key == 'injected_ndvi':
            candidates.append('injected_ndvi_map')
    else:
        candidates.append(target_metric)
        if target_metric == 'ndvi':
            candidates.append('ndvi_map')

    for key in dict.fromkeys(candidates):
        if key in data_grp:
            return data_grp[key]

    kind = "Injected target metric" if is_inj else "Target metric"
    raise KeyError(
        f"{kind} '{metric_key}' not found in {data_grp.name}. "
        f"Available datasets: {list(data_grp.keys())}"
    )


# ==========================================
# 3. MATHEMATICAL CORE FORMULATIONS
# ==========================================
def extract_fractional_years(acq_times):
    """
    Convert UNIX timestamps or year floats into continuous fractional years (t).
    Vectorized using pandas datetime and calendar day handling.
    """
    acq_arr = np.asarray(acq_times, dtype=np.float64)
    if len(acq_arr) == 0 or np.all((acq_arr >= 1970.0) & (acq_arr <= 2100.0)):
        return acq_arr

    dt = pd.to_datetime(acq_arr, unit='s', utc=True)
    start_year = pd.to_datetime(dt.year, format='%Y', utc=True)
    next_year = pd.to_datetime(dt.year + 1, format='%Y', utc=True)
    year_duration = (next_year - start_year).total_seconds()
    elapsed = (dt - start_year).total_seconds()
    return (dt.year + elapsed / year_duration).to_numpy(dtype=np.float64)


def build_harmonic_matrix(t, temporal_periods=None,
                          enable_const=ENABLE_CONSTANT,
                          enable_lin=ENABLE_LINEAR,
                          enable_quad=ENABLE_QUADRATIC):
    """
    Construct the Fourier harmonic design matrix X(t).

    For each observation time t, the row vector consists of:
      - Intercept: 1 (if enable_const)
      - Linear trend: t (if enable_lin)
      - Quadratic trend: t^2 (if enable_quad)
      - Cosine and sine harmonic pairs: cos(2*pi*t / T_k), sin(2*pi*t / T_k)
    """
    if temporal_periods is None:
        temporal_periods = TEMPORAL_PERIODS

    cols = []
    if enable_const:
        cols.append(np.ones_like(t, dtype=np.float64))
    if enable_lin:
        cols.append(np.asarray(t, dtype=np.float64))
    if enable_quad:
        cols.append(np.asarray(t, dtype=np.float64)**2)

    w = 2.0 * math.pi
    for p in temporal_periods:
        omega = w / float(p)
        cols.append(np.cos(omega * t))
        cols.append(np.sin(omega * t))

    return np.column_stack(cols) if cols else np.zeros((len(t), 0), dtype=np.float64)


def fit_lasso_inliers(X, y, min_samples, chi2_threshold, min_rmse=MIN_RMSE_CLAMP):
    """
    Fit LassoLarsIC with BIC model selection and iterative Chi-squared outlier rejection.

    Iteratively estimates parameters and rejects observations exceeding the Chi-squared
    residual threshold until an inlier subset is reached.
    """
    inliers = np.ones(len(y), dtype=bool)
    while np.sum(inliers) >= min_samples:
        try:
            model = LassoLarsIC(criterion='bic', fit_intercept=True).fit(X[inliers], y[inliers])
            residuals = np.abs(y[inliers] - model.predict(X[inliers]))
            rmse = max(float(np.sqrt(np.mean(residuals**2))), min_rmse)
        except Exception:
            return None, None, None, None

        outliers = (residuals / rmse)**2 > chi2_threshold
        if not np.any(outliers):
            return model, inliers, rmse, np.where(model.coef_ != 0)[0]

        inlier_indices = np.where(inliers)[0]
        inliers[inlier_indices[outliers]] = False

    return None, None, None, None


def rls_filter_step(theta, P, x_t, actual_y, forgetting_factor=RLS_FORGETTING_FACTOR):
    """
    Perform a single textbook Recursive Least Squares (RLS) parameter update.

    Innovation residual:
        e_t = y_t - x_t^T * theta_{t-1}
    Kalman-like gain:
        Px = P_{t-1} * x_t
        gamma = lambda + x_t^T * Px
        K_t = Px / gamma
    State & covariance updates:
        theta_t = theta_{t-1} + K_t * e_t
        P_t = (P_{t-1} - K_t * Px^T) / lambda
    """
    y_pred = np.dot(x_t, theta)
    innovation_error = actual_y - y_pred

    Px = P @ x_t
    denom = forgetting_factor + np.dot(x_t, Px)
    K = Px / denom

    theta_updated = theta + K * innovation_error
    P_updated = (P - np.outer(K, Px)) / forgetting_factor

    return theta_updated, P_updated, denom, innovation_error


# ==========================================
# 4. SINGLE PIXEL CONTINUOUS CHANGE DETECTION
# ==========================================
def detect_pixel_change(
    y_series,
    valid_indices,
    frac_years,
    X_full,
    acq_times,
    min_samples=MIN_SAMPLES,
    change_prob=CHANGE_PROBABILITY,
    consec_anom=CONSECUTIVE_ANOMALIES,
    min_years_for_init=MIN_YEARS_FOR_INIT,
    max_rmse=MAX_RMSE,
    min_rmse_clamp=MIN_RMSE_CLAMP,
    rls_forgetting_factor=RLS_FORGETTING_FACTOR,
    rls_ridge_penalty=RLS_RIDGE_PENALTY,
    enable_constant=ENABLE_CONSTANT,
):
    """
    Continuous Change Detection for a single pixel time series.

    Executes piecewise continuous monitoring through sequential stages:
      Step 1: Baseline initialization window search satisfying minimum sample count 
              and minimum temporal duration (>= 1.5 years).
      Step 2: Sparse harmonic selection via LassoLarsIC (BIC criterion) and iterative 
              Chi-squared outlier rejection.
      Step 3: Ordinary Least Squares (OLS) estimation on active features and covariance initialization.
      Step 4: Backfill model predictions across the initialization baseline.
      Step 5: Sequential online Recursive Least Squares (RLS) forward tracking with 
              Chi-squared anomaly hypothesis testing.
      Step 6: Structural break confirmation upon consecutive anomaly threshold breach.
    """
    num_frames = len(y_series)
    num_features = X_full.shape[1]

    pred_out = np.full(num_frames, np.nan, dtype=np.float32)
    rmse_out = np.full(num_frames, np.nan, dtype=np.float32)
    coef_out = np.full((num_frames, num_features), np.nan, dtype=np.float32)
    flags_out = np.zeros(num_frames, dtype=np.uint8)

    first_break_date = np.nan
    break_count = 0
    pixel_segments = []

    chi2_threshold = chi2.ppf(change_prob, df=CHI2_DEGREES_OF_FREEDOM)
    n_valid = len(valid_indices)

    seg_counter = 0
    i = 0

    while i < n_valid - min_samples:
        # Step 1: Initialization Window Identification
        segment_start_idx = i
        train_end = segment_start_idx + min_samples

        while train_end <= n_valid:
            time_span = frac_years[valid_indices[train_end - 1]] - frac_years[valid_indices[segment_start_idx]]
            if time_span >= min_years_for_init:
                break
            train_end += 1

        if train_end > n_valid:
            break

        # Step 2: Sparse Harmonic Selection via LASSO-BIC & Outlier Pruning
        window_indices = valid_indices[segment_start_idx:train_end]
        X_window = X_full[window_indices]
        y_window = y_series[window_indices]

        lasso_model, inlier_mask, init_rmse, active_features = fit_lasso_inliers(
            X=X_window,
            y=y_window,
            min_samples=max(num_features + 1, min_samples),
            chi2_threshold=chi2_threshold,
            min_rmse=min_rmse_clamp
        )

        if lasso_model is None:
            i += 1
            continue

        # Step 3: OLS Parameter Estimation & Covariance Initialization
        if enable_constant and (0 not in active_features):
            active_features = np.insert(active_features, 0, 0)
        if len(active_features) == 0:
            active_features = np.array([0])

        X_train_active = X_window[inlier_mask][:, active_features]
        Y_train_active = y_window[inlier_mask]
        X_train_all = X_window[inlier_mask]

        try:
            P_active = np.linalg.inv(X_train_active.T @ X_train_active + np.eye(len(active_features)) * rls_ridge_penalty)
            theta_active = P_active @ X_train_active.T @ Y_train_active

            theta = np.zeros(num_features, dtype=np.float64)
            theta[active_features] = theta_active
            P = np.linalg.inv(X_train_all.T @ X_train_all + np.eye(num_features) * rls_ridge_penalty)
        except np.linalg.LinAlgError:
            i += 1
            continue

        Y_pred_init = X_train_all @ theta
        SSE = np.sum((Y_train_active - Y_pred_init)**2)
        dof = len(active_features)
        n_points = int(np.sum(inlier_mask))
        rmse = max(float(np.sqrt(SSE / max(1, n_points - dof))), min_rmse_clamp)

        # Transient Event Rejection: Slide forward if initialization window has high variance
        if rmse > max_rmse:
            i += 1
            continue

        # Step 4: Backfill Predictions Across Initialization Window
        for k in range(segment_start_idx, train_end):
            global_idx = valid_indices[k]
            pred_out[global_idx] = np.dot(X_full[global_idx], theta)
            rmse_out[global_idx] = rmse
            coef_out[global_idx, :] = theta

        seg_counter += 1
        consecutive_count = 0
        break_detected = False
        first_anomaly_idx = -1
        target_j = train_end - 1

        # Step 5: Online Forward Tracking & Chi-Squared Anomaly Detection
        for j in range(train_end, n_valid):
            target_j = j
            target_idx = valid_indices[j]
            x_target = X_full[target_idx]
            actual = y_series[target_idx]

            y_pred = np.dot(x_target, theta)
            error = abs(actual - y_pred)

            pred_out[target_idx] = y_pred
            rmse_out[target_idx] = rmse
            coef_out[target_idx, :] = theta

            chi2_val = (error / rmse)**2
            is_anomaly = chi2_val > chi2_threshold

            if is_anomaly:
                flags_out[target_idx] = 1
                consecutive_count += 1
                if consecutive_count >= consec_anom:
                    first_anomaly_idx = valid_indices[j - consec_anom + 1]
                    break_detected = True
                    break
            else:
                consecutive_count = 0
                theta, P, denom, e = rls_filter_step(theta, P, x_target, actual, rls_forgetting_factor)
                SSE += (e**2) / denom
                n_points += 1
                rmse = max(float(np.sqrt(SSE / max(1, n_points - dof))), min_rmse_clamp)

        # Step 6: Segment Recording & Break Confirmation
        if break_detected:
            break_count += 1
            if np.isnan(first_break_date):
                first_break_date = acq_times[first_anomaly_idx]

            pixel_segments.append({
                "segment_idx": seg_counter,
                "start_idx": int(valid_indices[segment_start_idx]),
                "start_date": float(acq_times[valid_indices[segment_start_idx]]),
                "end_idx": int(valid_indices[target_j]),
                "end_date": float(acq_times[valid_indices[target_j]]),
                "has_break": True,
                "break_num": int(break_count),
                "break_idx": int(first_anomaly_idx),
                "break_date": float(acq_times[first_anomaly_idx]),
                "confirm_idx": int(valid_indices[target_j]),
                "confirm_date": float(acq_times[valid_indices[target_j]]),
                "rmse": float(rmse),
                "coef": theta.copy(),
                "n_points": int(n_points),
            })
            i = target_j - consec_anom + 1
        else:
            pixel_segments.append({
                "segment_idx": seg_counter,
                "start_idx": int(valid_indices[segment_start_idx]),
                "start_date": float(acq_times[valid_indices[segment_start_idx]]),
                "end_idx": int(valid_indices[target_j]),
                "end_date": float(acq_times[valid_indices[target_j]]),
                "has_break": False,
                "break_num": 0,
                "break_idx": -1,
                "break_date": np.nan,
                "confirm_idx": -1,
                "confirm_date": np.nan,
                "rmse": float(rmse),
                "coef": theta.copy(),
                "n_points": int(n_points),
            })
            break

    return (
        pred_out,
        rmse_out,
        coef_out,
        flags_out,
        first_break_date,
        break_count,
        pixel_segments
    )


# ==========================================
# 5. ROW CHUNK DISPATCH WORKER
# ==========================================
def _process_row_chunk(chunk_args):
    """
    Worker function for parallel processing across row chunks.
    """
    (y_start, y_end, width, y_data_chunk, valid_mask_chunk,
     frac_years, X_full, acq_times, min_samples, change_prob, consec_anom) = chunk_args

    num_frames = y_data_chunk.shape[0]
    chunk_height = y_end - y_start
    num_features = X_full.shape[1]

    chunk_pred = np.full((num_frames, chunk_height, width), np.nan, dtype=np.float32)
    chunk_rmse = np.full((num_frames, chunk_height, width), np.nan, dtype=np.float32)
    chunk_coef = np.full((num_frames, chunk_height, width, num_features), np.nan, dtype=np.float32)
    chunk_flags = np.zeros((num_frames, chunk_height, width), dtype=np.uint8)
    chunk_date = np.full((chunk_height, width), np.nan, dtype=np.float64)
    chunk_count = np.zeros((chunk_height, width), dtype=np.int32)
    chunk_segments = {}

    for y_local in range(chunk_height):
        for x in range(width):
            pixel_valid = valid_mask_chunk[:, y_local, x]
            valid_indices = np.where(pixel_valid)[0]

            if len(valid_indices) <= min_samples:
                continue

            (
                p_pred,
                p_rmse,
                p_coef,
                p_flags,
                p_date,
                p_count,
                p_segs
            ) = detect_pixel_change(
                y_series=y_data_chunk[:, y_local, x],
                valid_indices=valid_indices,
                frac_years=frac_years,
                X_full=X_full,
                acq_times=acq_times,
                min_samples=min_samples,
                change_prob=change_prob,
                consec_anom=consec_anom,
                min_years_for_init=MIN_YEARS_FOR_INIT,
                max_rmse=MAX_RMSE,
                min_rmse_clamp=MIN_RMSE_CLAMP,
                rls_forgetting_factor=RLS_FORGETTING_FACTOR,
                rls_ridge_penalty=RLS_RIDGE_PENALTY,
                enable_constant=ENABLE_CONSTANT
            )

            chunk_pred[:, y_local, x] = p_pred
            chunk_rmse[:, y_local, x] = p_rmse
            chunk_coef[:, y_local, x, :] = p_coef
            chunk_flags[:, y_local, x] = p_flags
            chunk_date[y_local, x] = p_date
            chunk_count[y_local, x] = p_count

            if p_segs:
                chunk_segments[(y_local, x)] = p_segs

    return (
        y_start,
        y_end,
        chunk_pred,
        chunk_rmse,
        chunk_coef,
        chunk_flags,
        chunk_date,
        chunk_count,
        chunk_segments
    )


# ==========================================
# 6. MAIN DRIVER & HDF5 STORAGE
# ==========================================
def main(enable_const=ENABLE_CONSTANT,
         enable_lin=ENABLE_LINEAR,
         enable_quad=ENABLE_QUADRATIC,
         temporal_periods=None,
         target_metric=TARGET_METRIC,
         injected=False,
         output_group=None,
         location=LOCATION,
         h5_path=None,
         output_h5=None,
         n_jobs=None,
         chunk_size=None,
         block_size=BLOCK_SIZE_ROWS,
         worker_chunk_rows=WORKER_CHUNK_ROWS,
         resume=True,
         overwrite=False,
         launch_vis=False,
         row_start=None,
         row_end=None,
         col_start=None,
         col_end=None,
         **kwargs):
    """
    Main driver for continuous change detection regression analysis.
    Implements streaming block execution to keep system RAM footprint bounded.
    """
    if temporal_periods is None:
        temporal_periods = TEMPORAL_PERIODS

    min_samples = MIN_SAMPLES

    if chunk_size is not None and chunk_size > 0:
        worker_chunk_rows = chunk_size

    target_name, target_short_name, metric_key, group_prefix, is_inj = resolve_metric_naming(target_metric, injected=injected)
    group_name = output_group if output_group is not None else f"{group_prefix}CCD_Results"

    if h5_path is None:
        h5_path = H5_PATH
        if not os.path.exists(h5_path):
            alt_path = f"C:/satelliteImagery/HLST30/HLST_{location}v2.h5"
            if os.path.exists(alt_path):
                h5_path = alt_path

    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"Input HDF5 file not found: {h5_path}")

    if output_h5 is None:
        output_h5 = h5_path

    print(f"Inspecting metadata from {h5_path}...")
    with h5py.File(h5_path, 'r') as f:
        data_grp = f['/HDFEOS/GRIDS/HARMONIZED/Data Fields']
        metric_ds = _resolve_metric_dataset(data_grp, target_metric, metric_key, is_inj)
        print(f"Selected dataset '{metric_ds.name}' -> output group '{group_name}'")

        acq_times = metric_ds.attrs.get('acquisition_time')
        if acq_times is None and 'acquisition_time' in data_grp.attrs:
            acq_times = data_grp.attrs['acquisition_time']
        elif acq_times is None and 'common_mask' in data_grp and 'acquisition_time' in data_grp['common_mask'].attrs:
            acq_times = data_grp['common_mask'].attrs['acquisition_time']
        elif acq_times is None and 'acquisition_time' in f.attrs:
            acq_times = f.attrs['acquisition_time']

        if acq_times is None:
            raise KeyError(
                f"Mandatory 'acquisition_time' attribute not found for dataset '{metric_ds.name}' "
                "or parent container. Cannot construct harmonic temporal matrix without timestamps."
            )

        acq_times = np.array(acq_times[:], dtype=np.float64)
        geo_transform = metric_ds.attrs.get('GeoTransform', f.attrs.get('GeoTransform'))
        spatial_ref = metric_ds.attrs.get('spatial_ref', f.attrs.get('spatial_ref'))

        full_num_frames, full_height, full_width = metric_ds.shape

        y0 = 0 if row_start is None else max(0, row_start)
        y1 = full_height if row_end is None else min(full_height, row_end)
        x0 = 0 if col_start is None else max(0, col_start)
        x1 = full_width if col_end is None else min(full_width, col_end)

        height = y1 - y0
        width = x1 - x0

    if block_size is None or block_size <= 0:
        block_size = BLOCK_SIZE_ROWS
    if block_size > height:
        block_size = max(1, height)

    if worker_chunk_rows is None or worker_chunk_rows <= 0:
        worker_chunk_rows = WORKER_CHUNK_ROWS
    if worker_chunk_rows > block_size:
        worker_chunk_rows = block_size

    frac_years = extract_fractional_years(acq_times)

    # Sort observations chronologically
    sort_idx = np.argsort(acq_times)
    acq_times = acq_times[sort_idx]
    frac_years = frac_years[sort_idx]
    num_frames = len(acq_times)

    print(f"Target grid: {num_frames} frames, processing {height}x{width} pixels (rows {y0}:{y1}, cols {x0}:{x1})")

    X_full = build_harmonic_matrix(
        frac_years,
        temporal_periods=temporal_periods,
        enable_const=enable_const,
        enable_lin=enable_lin,
        enable_quad=enable_quad
    )
    num_features = X_full.shape[1]

    if n_jobs is None:
        n_jobs = min(8, multiprocessing.cpu_count())
    print(f"Execution parameters: {n_jobs} cores | {block_size}-row streaming blocks | {worker_chunk_rows}-row worker chunks")

    needs_creation = True
    with h5py.File(output_h5, 'a') as out_file:
        if group_name in out_file:
            if overwrite or not resume:
                del out_file[group_name]
            else:
                existing_grp = out_file[group_name]
                exp_3d = (num_frames, height, width)
                exp_2d = (height, width)
                if ('predicted_series' in existing_grp and 'change_date_timestamp' in existing_grp and
                        existing_grp['predicted_series'].shape == exp_3d and
                        existing_grp['change_date_timestamp'].shape == exp_2d):
                    needs_creation = False
                    print(f"Target group '{group_name}' already exists with matching grid schema. Resuming existing progress...")
                else:
                    del out_file[group_name]

        if needs_creation:
            print(f"Initializing chunked datasets in {output_h5} -> group '{group_name}'...")
            ccd_grp = out_file.create_group(group_name)

            if spatial_ref is not None:
                ccd_grp.attrs['spatial_ref'] = spatial_ref
                out_file.attrs['spatial_ref'] = spatial_ref
            if geo_transform is not None:
                gt = list(geo_transform)
                if x0 > 0 or y0 > 0:
                    gt[0] += x0 * gt[1]
                    gt[3] += y0 * gt[5]
                ccd_grp.attrs['GeoTransform'] = gt
                out_file.attrs['GeoTransform'] = gt

            ccd_grp.attrs['CHANGE_PROBABILITY'] = CHANGE_PROBABILITY
            ccd_grp.attrs['CONSECUTIVE_ANOMALIES'] = CONSECUTIVE_ANOMALIES
            ccd_grp.attrs['CHI2_DEGREES_OF_FREEDOM'] = CHI2_DEGREES_OF_FREEDOM
            ccd_grp.attrs['MAX_RMSE'] = MAX_RMSE
            ccd_grp.attrs['MIN_RMSE_CLAMP'] = MIN_RMSE_CLAMP
            ccd_grp.attrs['MIN_YEARS_FOR_INIT'] = MIN_YEARS_FOR_INIT
            ccd_grp.attrs['MIN_SAMPLES'] = min_samples
            ccd_grp.attrs['TEMPORAL_PERIODS'] = temporal_periods
            ccd_grp.attrs['ENABLE_CONSTANT'] = enable_const
            ccd_grp.attrs['ENABLE_LINEAR'] = enable_lin
            ccd_grp.attrs['ENABLE_QUADRATIC'] = enable_quad
            ccd_grp.attrs['TARGET_METRIC'] = metric_key
            ccd_grp.attrs['TARGET_SHORT_NAME'] = target_short_name
            ccd_grp.attrs['TARGET_NAME'] = target_name
            ccd_grp.attrs['SOURCE_DATA'] = h5_path
            ccd_grp.attrs['acquisition_time'] = acq_times

            out_file.attrs['CHANGE_PROBABILITY'] = CHANGE_PROBABILITY
            out_file.attrs['CONSECUTIVE_ANOMALIES'] = CONSECUTIVE_ANOMALIES
            out_file.attrs['CHI2_DEGREES_OF_FREEDOM'] = CHI2_DEGREES_OF_FREEDOM
            out_file.attrs['TARGET_METRIC'] = metric_key

            if (height, width) != (full_height, full_width):
                ccd_grp.attrs['ROI_ROWS'] = [y0, y1]
                ccd_grp.attrs['ROI_COLS'] = [x0, x1]

            chunk_3d = (min(64, num_frames), min(16, height), min(64, width))
            chunk_4d = (min(32, num_frames), min(8, height), min(32, width), num_features)

            for name, shape, fill, dtype, opt, chunks in [
                ('predicted_series', (num_frames, height, width), np.nan, np.float32, 1, chunk_3d),
                ('rmse_series', (num_frames, height, width), np.nan, np.float32, 1, chunk_3d),
                ('coef_series', (num_frames, height, width, num_features), np.nan, np.float32, 1, chunk_4d),
                ('anomaly_flags', (num_frames, height, width), 0, np.uint8, 1, chunk_3d),
                ('change_date_timestamp', (height, width), np.nan, np.float64, 1, True),
                ('change_count', (height, width), 0, np.int32, 1, True),
            ]:
                ds = ccd_grp.create_dataset(name, shape=shape, dtype=dtype, fillvalue=fill,
                                            shuffle=True, compression='gzip', compression_opts=opt,
                                            chunks=chunks)
                ds.attrs['_FillValue'] = fill
                if name not in out_file:
                    out_file[name] = ds

    total_blocks = math.ceil(height / block_size)
    print(f"Beginning streaming execution across {total_blocks} blocks ({block_size} rows per block)...")
    t0 = time.time()
    total_changed_pixels = 0

    with h5py.File(output_h5, 'a') as out_file:
        ccd_grp = out_file[group_name]
        ds_pred = ccd_grp['predicted_series']
        ds_rmse = ccd_grp['rmse_series']
        ds_coef = ccd_grp['coef_series']
        ds_flags = ccd_grp['anomaly_flags']
        ds_date = ccd_grp['change_date_timestamp']
        ds_count = ccd_grp['change_count']

        same_file = os.path.abspath(output_h5) == os.path.abspath(h5_path)
        f_in = out_file if same_file else h5py.File(h5_path, 'r')
        try:
            data_grp = f_in['/HDFEOS/GRIDS/HARMONIZED/Data Fields']
            metric_ds = _resolve_metric_dataset(data_grp, target_metric, metric_key, is_inj)
            common_mask_ds = data_grp['common_mask']

            for b_idx in range(total_blocks):
                b_y0 = b_idx * block_size
                b_y1 = min(b_y0 + block_size, height)
                b_height = b_y1 - b_y0

                # Check if block was already computed in a previous run (resume)
                if resume and not needs_creation:
                    block_check = ds_date[b_y0:b_y1, :]
                    is_completed = (np.count_nonzero(block_check == 0.0) == 0) and np.any(np.isnan(block_check) | (block_check > 0))
                    if is_completed:
                        b_changed = int(np.count_nonzero(~np.isnan(block_check)))
                        total_changed_pixels += b_changed
                        pct_complete = (b_y1 / height) * 100.0
                        print(f"Block {b_idx + 1:2d}/{total_blocks} (rows {b_y0:4d}:{b_y1:4d}) already completed ({pct_complete:5.1f}%) | "
                              f"Changed pixels in block: {b_changed:4d} (Skipping to resume)")
                        continue

                # Stream-load ONLY the current spatial block from disk
                r_slice = slice(y0 + b_y0, y0 + b_y1)
                c_slice = slice(x0, x1)
                y_block = read_scaled_dataset(metric_ds, r_slice=r_slice, c_slice=c_slice)
                cm_block = common_mask_ds[:, r_slice, c_slice]

                # Sort observations chronologically along time axis
                y_block = y_block[sort_idx, ...]
                cm_block = cm_block[sort_idx, ...]
                valid_mask_block = (cm_block == 0) & ~np.isnan(y_block)

                # Allocate block buffers
                block_pred = np.full((num_frames, b_height, width), np.nan, dtype=np.float32)
                block_rmse = np.full((num_frames, b_height, width), np.nan, dtype=np.float32)
                block_coef = np.full((num_frames, b_height, width, num_features), np.nan, dtype=np.float32)
                block_flags = np.zeros((num_frames, b_height, width), dtype=np.uint8)
                block_date = np.full((b_height, width), np.nan, dtype=np.float64)
                block_count = np.zeros((b_height, width), dtype=np.int32)

                # Partition block into worker subchunks
                subchunks = []
                for sub_y0 in range(0, b_height, worker_chunk_rows):
                    sub_y1 = min(sub_y0 + worker_chunk_rows, b_height)
                    chunk_args = (
                        sub_y0, sub_y1, width,
                        np.ascontiguousarray(y_block[:, sub_y0:sub_y1, :]),
                        np.ascontiguousarray(valid_mask_block[:, sub_y0:sub_y1, :]),
                        frac_years, X_full, acq_times,
                        min_samples, CHANGE_PROBABILITY, CONSECUTIVE_ANOMALIES
                    )
                    subchunks.append(chunk_args)

                # Process block subchunks in parallel
                if n_jobs > 1 and len(subchunks) > 1:
                    results = Parallel(n_jobs=n_jobs, backend='loky')(
                        delayed(_process_row_chunk)(sc) for sc in subchunks
                    )
                else:
                    results = [_process_row_chunk(sc) for sc in subchunks]

                # Assemble subchunks into block buffers
                for res in results:
                    sub_y0, sub_y1, c_pred, c_rmse, c_coef, c_flags, c_date, c_count = res[:8]
                    block_pred[:, sub_y0:sub_y1, :] = c_pred
                    block_rmse[:, sub_y0:sub_y1, :] = c_rmse
                    block_coef[:, sub_y0:sub_y1, :, :] = c_coef
                    block_flags[:, sub_y0:sub_y1, :] = c_flags
                    block_date[sub_y0:sub_y1, :] = c_date
                    block_count[sub_y0:sub_y1, :] = c_count

                # Stream write directly into HDF5 datasets
                ds_pred[:, b_y0:b_y1, :] = block_pred
                ds_rmse[:, b_y0:b_y1, :] = block_rmse
                ds_coef[:, b_y0:b_y1, :, :] = block_coef
                ds_flags[:, b_y0:b_y1, :] = block_flags
                ds_date[b_y0:b_y1, :] = block_date
                ds_count[b_y0:b_y1, :] = block_count

                out_file.flush()

                b_changed = int(np.count_nonzero(~np.isnan(block_date)))
                total_changed_pixels += b_changed
                elapsed_block = time.time() - t0
                pct_complete = (b_y1 / height) * 100.0
                print(f"Block {b_idx + 1:2d}/{total_blocks} (rows {b_y0:4d}:{b_y1:4d}) complete ({pct_complete:5.1f}%) | "
                      f"Changed pixels in block: {b_changed:4d} | Elapsed: {elapsed_block/60:4.1f} min")

                # Explicit memory release
                del y_block, cm_block, valid_mask_block
                del block_pred, block_rmse, block_coef, block_flags, block_date, block_count
                del results, subchunks
                gc.collect()

                # Recycle worker processes between blocks to eliminate C-level memory corruption and fragmentation
                try:
                    from joblib.externals.loky import get_reusable_executor
                    get_reusable_executor().shutdown(wait=True)
                except Exception:
                    pass

            total_elapsed = time.time() - t0
            ccd_grp.attrs['ELAPSED_SECONDS'] = total_elapsed
        finally:
            if not same_file:
                f_in.close()

    if launch_vis:
        try:
            import SC_CCM_vis
            SC_CCM_vis.plot_spatial_anomaly_overlay(
                source_h5_path=output_h5,
                target_metric=target_short_name,
                injected=is_inj,
                interactive=True
            )
        except Exception as e:
            print(f"Warning: Failed to launch visualization: {e}")

    print("Harmonized CCD Pipeline Complete!")
    return output_h5


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Harmonized CCD Main")
    parser.add_argument('--location', type=str, default=LOCATION, help="Location name")
    parser.add_argument('--h5-path', type=str, default=None, help="Input source HDF5 path")
    parser.add_argument('--output-h5', type=str, default=None, help="Output CCD HDF5 path")
    parser.add_argument('--metric', type=str, default=TARGET_METRIC, help="Target metric")
    parser.add_argument('--injected', action='store_true', help="Run CCD on injected dataset")
    parser.add_argument('--no-vis', action='store_true', help="Skip launching visualization")
    parser.add_argument('--block-size', type=int, default=BLOCK_SIZE_ROWS, help="Rows per streaming block written to HDF5")
    parser.add_argument('--worker-chunk-rows', type=int, default=WORKER_CHUNK_ROWS, help="Rows per worker task")
    parser.add_argument('--chunk-size', type=int, default=None, help="Alias for worker-chunk-rows")
    parser.add_argument('--n-jobs', type=int, default=None, help="Number of parallel worker processes")
    parser.add_argument('--row-start', type=int, default=None, help="Start row for ROI")
    parser.add_argument('--row-end', type=int, default=None, help="End row for ROI")
    parser.add_argument('--col-start', type=int, default=None, help="Start col for ROI")
    parser.add_argument('--col-end', type=int, default=None, help="End col for ROI")
    parser.add_argument('--no-resume', action='store_false', dest='resume', default=True, help="Disable automatic block resumption")
    parser.add_argument('--overwrite', action='store_true', default=False, help="Overwrite existing output group")
    args = parser.parse_args()

    main(
        location=args.location,
        h5_path=args.h5_path,
        output_h5=args.output_h5,
        target_metric=args.metric,
        injected=args.injected,
        launch_vis=not args.no_vis,
        block_size=args.block_size,
        worker_chunk_rows=args.worker_chunk_rows,
        chunk_size=args.chunk_size,
        n_jobs=args.n_jobs,
        row_start=args.row_start,
        row_end=args.row_end,
        col_start=args.col_start,
        col_end=args.col_end,
        resume=args.resume,
        overwrite=args.overwrite
    )
