"""
Harmonized Continuous Change Detection (CCD) Pipeline.
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

import warnings
import h5py
import numpy as np
import datetime
import math
from tqdm import tqdm
from sklearn.linear_model import LassoLarsIC
from sklearn.linear_model._least_angle import _lars_path_solver
import multiprocessing
import joblib
import contextlib
from scipy.stats import chi2
from joblib import Parallel, delayed
import numba as nb
_HAVE_NUMBA = True
_njit_opt = nb.njit(fastmath=True)




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


def get_dataset_metadata(ds):
    """
    Extracts fill value and scale factor from HDF5 dataset attributes without assumed defaults.
    Fails explicitly if mandatory attributes are missing.
    """
    # 1. Fill value extraction
    fill_val = None
    if hasattr(ds, "attrs"):
        if "_FillValue" in ds.attrs:
            fill_val = ds.attrs["_FillValue"]
        elif "fillvalue" in ds.attrs:
            fill_val = ds.attrs["fillvalue"]
        elif "fill_value" in ds.attrs:
            fill_val = ds.attrs["fill_value"]

    if fill_val is None and hasattr(ds, "fillvalue") and ds.fillvalue is not None:
        if np.issubdtype(ds.dtype, np.floating) or ds.fillvalue != 0:
            fill_val = ds.fillvalue

    if fill_val is None and np.issubdtype(ds.dtype, np.integer):
        raise KeyError(
            f"Dataset '{getattr(ds, 'name', 'dataset')}' has integer datatype ({ds.dtype}) "
            "but lacks mandatory '_FillValue' or 'fillvalue' attribute."
        )

    if fill_val is not None:
        if np.issubdtype(ds.dtype, np.integer):
            fill_val = int(fill_val)
        elif np.issubdtype(ds.dtype, np.floating):
            fill_val = float(fill_val)

    # 2. Scale factor extraction
    if np.issubdtype(ds.dtype, np.floating):
        # Native floating-point data is unscaled unless explicitly specified
        scale = float(ds.attrs["scale_factor"]) if (hasattr(ds, "attrs") and "scale_factor" in ds.attrs) else 1.0
    else:
        if hasattr(ds, "attrs") and "scale_factor" in ds.attrs:
            scale = float(ds.attrs["scale_factor"])
        elif hasattr(ds, "attrs") and "scale_to_float" in ds.attrs:
            sf = float(ds.attrs["scale_to_float"])
            scale = (1.0 / sf) if sf < 1.0 else sf
        else:
            raise KeyError(
                f"Dataset '{getattr(ds, 'name', 'dataset')}' has integer datatype ({ds.dtype}) "
                "but lacks mandatory 'scale_factor' or 'scale_to_float' attribute."
            )

    return fill_val, scale


def read_scaled_int16(arr, scale_factor, nodata_val):
    """
    Invert int16 quantized array back to float32 reflectance.
    Preserves nodata sentinels as NaN without injecting synthetic fill values.
    Requires explicit scale_factor and nodata_val without assumed defaults.
    """
    if scale_factor is None:
        raise ValueError("scale_factor must be explicitly provided without default fallback.")
    if nodata_val is None:
        raise ValueError("nodata_val must be explicitly provided without default fallback.")

    out = np.full(arr.shape, np.nan, dtype=np.float32)
    valid = (arr != nodata_val)
    out[valid] = arr[valid].astype(np.float32) / float(scale_factor)
    return out


def scale_to_int16(arr, scale_factor, nodata_val):
    """
    Quantize float array to int16 format.
    Preserves nodata sentinels without synthetic smoothing.
    Requires explicit scale_factor and nodata_val without assumed defaults.
    """
    if scale_factor is None or nodata_val is None:
        raise ValueError("scale_factor and nodata_val must be explicitly specified without assumed defaults.")

    out = np.full(arr.shape, nodata_val, dtype=np.int16)
    valid = ~np.isnan(arr) & (arr != nodata_val)
    clipped = np.clip(arr[valid] * scale_factor, -32767, 32767)
    out[valid] = np.round(clipped).astype(np.int16)
    return out


# ==========================================
# 1. CONFIGURATION
# ==========================================
# Input/Output
LOCATION = "Rochesterv2"
H5_PATH = f"C:/satelliteImagery/HLST30/HLST_{LOCATION}.h5"
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
    elif 'robust' in base:
        disp = 'Spectral Complexity (Robust)'
        short = 'SC_Robust'
        ds_key = 'injected_sliding_volume_robust_scale' if is_inj else 'sliding_volume_robust_scale'
    elif 'ndbi' in base:
        disp = 'NDBI'
        short = 'NDBI'
        ds_key = 'injected_ndbi' if is_inj else 'ndbi'
    else:
        clean = base.replace(' ', '_').upper()
        disp = metric_name
        short = clean
        ds_key = f"injected_{m_raw}" if (is_inj and not m_lower.startswith('injected_')) else m_raw

    group_prefix = f"INJECTED_{short}" if is_inj else short
    return disp, short, ds_key, group_prefix, is_inj


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

# RLS Tuning
RLS_RIDGE_PENALTY = 1e-6
RLS_FORGETTING_FACTOR = 0.999


def extract_fractional_years(acq_times):
    """Converts UNIX timestamps into continuous fractional years (t)."""
    frac_years = []
    for dt in acq_times:
        val = float(dt)
        if 1970.0 <= val <= 2100.0:
            frac_years.append(val)
            continue
        try:
            dt_obj = datetime.datetime.fromtimestamp(val, tz=datetime.timezone.utc)
            year = dt_obj.year
            start_of_year = datetime.datetime(year, 1, 1, tzinfo=datetime.timezone.utc)
            start_of_next = datetime.datetime(year + 1, 1, 1, tzinfo=datetime.timezone.utc)
            year_duration = (start_of_next - start_of_year).total_seconds()
            elapsed = (dt_obj - start_of_year).total_seconds()
            frac_years.append(year + (elapsed / year_duration))
        except (OSError, OverflowError, ValueError):
            frac_years.append(1970.0 + (val / 31557600.0))
    return np.array(frac_years)


def fast_lasso_lars_bic(X, y):
    """
    Lightweight, exact drop-in replacement for LassoLarsIC(criterion='bic', fit_intercept=True).
    Bypasses scikit-learn estimator overhead, input validation, and redundant copying.

    Returns:
        coef: ndarray of shape (n_features,)
        intercept: float
        y_pred: ndarray of shape (n_samples,)
        active_features: ndarray of active feature indices
    """
    n_samples, n_features = X.shape

    # 1. Center X and y (equivalent to fit_intercept=True)
    X_mean = np.mean(X, axis=0)
    y_mean = np.mean(y)
    X_c = X - X_mean
    y_c = y - y_mean

    # 2. Estimate unpenalized noise variance via OLS on centered variables
    coef_ols, _, _, _ = np.linalg.lstsq(X_c, y_c, rcond=None)
    residuals_ols = y_c - X_c @ coef_ols
    dof_noise = max(1, n_samples - n_features - 1)
    noise_variance = np.sum(residuals_ols**2) / dof_noise
    noise_variance = max(noise_variance, 1e-12)

    # 3. Compute LARS path directly via the compiled solver
    alphas, _, coef_path = _lars_path_solver(
        X=X_c, y=y_c, method='lasso', copy_X=True, return_path=True
    )

    # 4. Evaluate BIC criterion across the regularization path
    residuals = y_c[:, np.newaxis] - X_c @ coef_path
    rss = np.sum(residuals**2, axis=0)

    eps = np.finfo(coef_path.dtype).eps
    dof = np.sum(np.abs(coef_path) > eps, axis=0)

    bic_criterion = (
        n_samples * np.log(2.0 * np.pi * noise_variance)
        + rss / noise_variance
        + np.log(n_samples) * dof
    )

    # 5. Extract optimal model
    best_idx = np.argmin(bic_criterion)
    coef = coef_path[:, best_idx]
    intercept = y_mean - np.dot(X_mean, coef)
    y_pred = X @ coef + intercept
    active_features = np.where(np.abs(coef) > eps)[0]

    return coef, intercept, y_pred, active_features


def build_harmonic_matrix(t, temporal_periods=None,
                          enable_const=ENABLE_CONSTANT,
                          enable_lin=ENABLE_LINEAR,
                          enable_quad=ENABLE_QUADRATIC):
    """
    Constructs a Fourier basis matrix with configurable polynomial trends and temporal periods.
    """
    if temporal_periods is None:
        temporal_periods = TEMPORAL_PERIODS

    cols = []

    if enable_const:
        cols.append(np.ones_like(t))

    if enable_lin:
        cols.append(t)
    if enable_quad:
        cols.append(t**2)

    w = 2.0 * math.pi
    for p in temporal_periods:
        cols.append(np.cos((w / p) * t))
        cols.append(np.sin((w / p) * t))

    return np.column_stack(cols) if cols else np.zeros((len(t), 0))


@_njit_opt
def backfill_init_window_numba(
    valid_indices, segment_start_idx, train_end,
    X_full, theta, rmse,
    chunk_pred, chunk_rmse, chunk_coef,
    y_local, x
):
    """JIT-compiled backfill of model predictions across the initialization window."""
    num_features = len(theta)
    for k in range(segment_start_idx, train_end):
        global_idx = valid_indices[k]
        y_pred = 0.0
        for f in range(num_features):
            y_pred += X_full[global_idx, f] * theta[f]
            chunk_coef[global_idx, y_local, x, f] = np.float32(theta[f])
        chunk_pred[global_idx, y_local, x] = np.float32(y_pred)
        chunk_rmse[global_idx, y_local, x] = np.float32(rmse)


@_njit_opt
def rls_forward_pass_numba(
    valid_indices, train_end, y_series, X_full,
    theta_in, P_in, rmse, SSE_in, dof, n_points_in,
    chi2_threshold, consec_anom, forgetting_factor,
    chunk_pred, chunk_rmse, chunk_coef, chunk_flags,
    y_local, x
):
    """
    JIT-compiled forward expanding RLS pass for a single pixel time series.
    Eliminates all Python interpreter and temporary array allocation overhead.
    """
    num_features = len(theta_in)
    theta = theta_in.copy()
    P = P_in.copy()
    SSE = SSE_in
    n_points = n_points_in
    consecutive_count = 0
    break_detected = False
    first_anomaly_idx = -1
    target_j = train_end - 1

    n_valid = len(valid_indices)
    Px = np.zeros(num_features, dtype=np.float64)
    K = np.zeros(num_features, dtype=np.float64)
    inv_ff = 1.0 / forgetting_factor

    for j in range(train_end, n_valid):
        target_j = j
        target_idx = valid_indices[j]
        actual = y_series[target_idx]

        y_pred = 0.0
        for f in range(num_features):
            y_pred += X_full[target_idx, f] * theta[f]

        error = abs(actual - y_pred)

        chunk_pred[target_idx, y_local, x] = np.float32(y_pred)
        chunk_rmse[target_idx, y_local, x] = np.float32(rmse)
        for f in range(num_features):
            chunk_coef[target_idx, y_local, x, f] = np.float32(theta[f])

        chi2_val = (error / rmse) ** 2
        is_anomaly = chi2_val > chi2_threshold

        if is_anomaly:
            chunk_flags[target_idx, y_local, x] = np.uint8(1)
            consecutive_count += 1
            if consecutive_count >= consec_anom:
                first_anomaly_idx = valid_indices[j - consec_anom + 1]
                break_detected = True
                break
        else:
            consecutive_count = 0
            xTx = 0.0
            for r in range(num_features):
                val = 0.0
                for c in range(num_features):
                    val += P[r, c] * X_full[target_idx, c]
                Px[r] = val
                xTx += X_full[target_idx, r] * val

            denom = forgetting_factor + xTx
            inv_denom = 1.0 / denom
            for f in range(num_features):
                K[f] = Px[f] * inv_denom

            e = actual - y_pred
            for f in range(num_features):
                theta[f] += K[f] * e

            for r in range(num_features):
                for c in range(num_features):
                    P[r, c] = (P[r, c] - K[r] * Px[c]) * inv_ff

            SSE += (e * e) * inv_denom
            n_points += 1

    return break_detected, target_j, first_anomaly_idx, theta, P, SSE, n_points


def rls_forward_pass_batched(
    y_batch, X_full, valid_indices, train_end,
    thetas_init, P_matrices_init, rmses,
    chi2_threshold, consec_anom, forgetting_factor=0.999
):
    """
    Batched RLS forward pass across multiple pixels sharing identical observation indices.
    Vectorizes matrix-vector operations across all pixels simultaneously using @ and np.einsum.

    Parameters:
        y_batch: ndarray of shape (num_frames, batch_size)
        X_full: ndarray of shape (num_frames, num_features)
        valid_indices: 1D array of valid time frame indices
        train_end: index in valid_indices where forward expansion begins
        thetas_init: ndarray of shape (batch_size, num_features)
        P_matrices_init: ndarray of shape (batch_size, num_features, num_features)
        rmses: ndarray of shape (batch_size,)
        chi2_threshold: float threshold for chi2 anomaly check
        consec_anom: consecutive anomaly count to confirm break
        forgetting_factor: RLS lambda parameter

    Returns:
        preds: ndarray of shape (num_frames, batch_size)
        anomalies: ndarray of shape (num_frames, batch_size)
        thetas: updated parameter vectors (batch_size, num_features)
        P_matrices: updated covariance matrices (batch_size, num_features, num_features)
        break_indices: 1D array of length batch_size (or -1 if no break)
    """
    num_frames, batch_size = y_batch.shape

    thetas = thetas_init.copy()
    P = P_matrices_init.copy()
    consec = np.zeros(batch_size, dtype=np.int32)
    active = np.ones(batch_size, dtype=bool)
    break_indices = np.full(batch_size, -1, dtype=np.int32)

    preds = np.full((num_frames, batch_size), np.nan, dtype=np.float32)
    anomalies = np.zeros((num_frames, batch_size), dtype=np.uint8)

    inv_ff = 1.0 / forgetting_factor

    for j in range(train_end, len(valid_indices)):
        if not np.any(active):
            break

        idx = valid_indices[j]
        x_t = X_full[idx, :]

        # Batched prediction: (M, K) @ (K,) -> (M,)
        y_pred = thetas @ x_t
        preds[idx, :] = y_pred

        actual = y_batch[idx, :]
        error = np.abs(actual - y_pred)
        chi2_vals = (error / rmses) ** 2

        is_anom = (chi2_vals > chi2_threshold) & active
        anomalies[idx, is_anom] = 1

        consec[is_anom] += 1
        new_breaks = (consec >= consec_anom) & active
        if np.any(new_breaks):
            break_indices[new_breaks] = valid_indices[j - consec_anom + 1]
            active[new_breaks] = False

        non_anom = (~is_anom) & active
        consec[non_anom] = 0

        if not np.any(non_anom):
            continue

        P_act = P[non_anom]
        Px = P_act @ x_t
        denom = forgetting_factor + (Px @ x_t)
        K_gain = Px / denom[:, None]

        e = actual[non_anom] - y_pred[non_anom]
        thetas[non_anom] += K_gain * e[:, None]

        P_act = (P_act - np.einsum('ni,nj->nij', K_gain, Px)) * inv_ff
        P[non_anom] = P_act

    return preds, anomalies, thetas, P, break_indices


def _process_row_chunk(chunk_args):
    """Worker function for processing a row chunk across temporal frames."""
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

    chi2_threshold = chi2.ppf(change_prob, df=CHI2_DEGREES_OF_FREEDOM)

    for y_local in range(chunk_height):
        for x in range(width):
            pixel_valid = valid_mask_chunk[:, y_local, x]
            valid_indices = np.where(pixel_valid)[0]

            if len(valid_indices) <= min_samples:
                continue

            pixel_segments = []
            seg_counter = 0
            i = 0
            while i < len(valid_indices) - min_samples:
                # 1. Initialize segment
                segment_start_idx = i
                train_end = segment_start_idx + min_samples

                # Minimum time span constraint for stable initialization
                while train_end <= len(valid_indices):
                    time_span = frac_years[valid_indices[train_end - 1]] - frac_years[valid_indices[segment_start_idx]]
                    if time_span >= MIN_YEARS_FOR_INIT:
                        break
                    train_end += 1

                if train_end > len(valid_indices):
                    # Could not satisfy minimum initialization time span
                    break

                active_pool = list(range(segment_start_idx, train_end))
                min_dof = X_full.shape[1] + 1
                absolute_min_samples = max(min_dof, min_samples)

                initialization_successful = False
                last_active_features = None

                while len(active_pool) >= absolute_min_samples:
                    X_train = X_full[valid_indices[active_pool], :]
                    Y_train = y_data_chunk[valid_indices[active_pool], y_local, x]

                    try:
                        coef, intercept, y_train_pred, active_features = fast_lasso_lars_bic(X_train, Y_train)
                        rmse = np.sqrt(np.mean((Y_train - y_train_pred)**2))
                        rmse = max(rmse, MIN_RMSE_CLAMP)
                    except Exception:
                        break

                    errors = np.abs(Y_train - y_train_pred)
                    chi2_vals_init = (errors / rmse)**2
                    outlier_mask = chi2_vals_init > chi2_threshold

                    if np.any(outlier_mask):
                        # Multi-outlier pruning: remove all observations exceeding threshold simultaneously
                        active_pool = [active_pool[k] for k in range(len(active_pool)) if not outlier_mask[k]]
                    else:
                        initialization_successful = True
                        last_active_features = active_features
                        break

                if not initialization_successful:
                    i += 1
                    continue

                # Hand-off to OLS for RLS initialization
                active_features = last_active_features

                # If constant term is enabled, force it to be active
                if ENABLE_CONSTANT and 0 not in active_features:
                    active_features = np.insert(active_features, 0, 0)

                if len(active_features) == 0:
                    active_features = np.array([0])

                X_train_active = X_full[valid_indices[active_pool]][:, active_features]
                Y_train_active = y_data_chunk[valid_indices[active_pool], y_local, x]

                # Initialize Full RLS State (track all terms)
                X_train_all = X_full[valid_indices[active_pool]]
                num_total_features = X_train_all.shape[1]

                try:
                    # OLS only on active features to prevent overfitting on initialization
                    P_active = np.linalg.inv(X_train_active.T @ X_train_active + np.eye(len(active_features)) * RLS_RIDGE_PENALTY)
                    theta_active = P_active @ X_train_active.T @ Y_train_active

                    # Map to full theta vector (inactive terms start at 0)
                    theta = np.zeros(num_total_features)
                    theta[active_features] = theta_active

                    # Initialize P matrix on all features so RLS can adapt them later
                    P = np.linalg.inv(X_train_all.T @ X_train_all + np.eye(num_total_features) * RLS_RIDGE_PENALTY)
                except np.linalg.LinAlgError:
                    i += 1
                    continue

                Y_pred_init = X_train_all @ theta
                SSE = np.sum((Y_train_active - Y_pred_init)**2)
                dof = len(active_features)
                n_points = len(active_pool)
                rmse = np.sqrt(SSE / max(1, n_points - dof))
                rmse = max(rmse, MIN_RMSE_CLAMP)

                # Transient Event Rejection: If the initialization window is volatile, slide forward
                if rmse > MAX_RMSE:
                    i += 1
                    continue

                # Backfill predictions for initialization window (Numba JIT accelerated)
                backfill_init_window_numba(
                    valid_indices, segment_start_idx, train_end,
                    X_full, theta, rmse,
                    chunk_pred, chunk_rmse, chunk_coef,
                    y_local, x
                )

                seg_counter += 1
                consecutive_count = 0
                break_detected = False
                target_j = train_end - 1

                # 2. Forward Expanding Phase (Numba JIT accelerated)
                break_detected, target_j, first_anomaly_idx, theta, P, SSE, n_points = rls_forward_pass_numba(
                    valid_indices, train_end, y_data_chunk[:, y_local, x], X_full,
                    theta, P, rmse, SSE, dof, n_points,
                    chi2_threshold, consec_anom, RLS_FORGETTING_FACTOR,
                    chunk_pred, chunk_rmse, chunk_coef, chunk_flags,
                    y_local, x
                )

                if break_detected:
                    chunk_count[y_local, x] += 1
                    if np.isnan(chunk_date[y_local, x]):
                        chunk_date[y_local, x] = acq_times[first_anomaly_idx]

                    pixel_segments.append({
                        "segment_idx": seg_counter,
                        "start_idx": int(valid_indices[segment_start_idx]),
                        "start_date": float(acq_times[valid_indices[segment_start_idx]]),
                        "end_idx": int(valid_indices[target_j]),
                        "end_date": float(acq_times[valid_indices[target_j]]),
                        "has_break": True,
                        "break_num": int(chunk_count[y_local, x]),
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

            if pixel_segments:
                chunk_segments[(y_local, x)] = pixel_segments

    return y_start, y_end, chunk_pred, chunk_rmse, chunk_coef, chunk_flags, chunk_date, chunk_count, chunk_segments


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
         chunk_size=1,
         launch_vis=False,
         row_start=0,
         row_end=None,
         col_start=0,
         col_end=None,
         **kwargs):
    """
    Main driver for continuous change detection regression analysis.
    """
    if temporal_periods is None:
        temporal_periods = TEMPORAL_PERIODS

    min_samples = MIN_SAMPLES

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

    print(f"Loading data from {h5_path}...")
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

        if acq_times is not None:
            acq_times = np.array(acq_times[:], dtype=np.float64)

        if row_end is None:
            r_slice = slice(row_start, None)
        else:
            r_slice = slice(row_start, row_end)
        if col_end is None:
            c_slice = slice(col_start, None)
        else:
            c_slice = slice(col_start, col_end)

        if row_start == 0 and row_end is None and col_start == 0 and col_end is None:
            raw_y = metric_ds[...]
            common_mask = data_grp['common_mask'][...]
        else:
            raw_y = metric_ds[:, r_slice, c_slice]
            common_mask = data_grp['common_mask'][:, r_slice, c_slice]

        fill_val, scale = get_dataset_metadata(metric_ds)

        if np.issubdtype(raw_y.dtype, np.integer):
            y_data = read_scaled_int16(raw_y, scale_factor=scale, nodata_val=fill_val)
        elif np.issubdtype(raw_y.dtype, np.floating):
            y_data = raw_y.astype(np.float32) / float(scale)
            if fill_val is not None and not np.isnan(fill_val):
                y_data[np.isclose(raw_y, fill_val, equal_nan=True)] = np.nan
        else:
            raise TypeError(f"Unsupported dataset datatype '{raw_y.dtype}' for {metric_ds.name}")

        # Determine valid mask
        valid_mask = (common_mask == 0) & ~np.isnan(y_data)

        geo_transform = metric_ds.attrs.get('GeoTransform')
        if geo_transform is None and 'GeoTransform' in f.attrs:
            geo_transform = f.attrs['GeoTransform']

        spatial_ref = metric_ds.attrs.get('spatial_ref')
        if spatial_ref is None and 'spatial_ref' in f.attrs:
            spatial_ref = f.attrs['spatial_ref']

    num_frames, height, width = y_data.shape

    if acq_times is None:
        raise KeyError(
            f"Mandatory 'acquisition_time' attribute not found for dataset '{metric_ds.name}' "
            "or parent container. Cannot construct harmonic temporal matrix without timestamps."
        )

    frac_years = extract_fractional_years(acq_times)

    # Sort chronologically
    sort_idx = np.argsort(acq_times)
    acq_times = acq_times[sort_idx]
    frac_years = frac_years[sort_idx]
    y_data = y_data[sort_idx, ...]
    valid_mask = valid_mask[sort_idx, ...]

    print(f"Dataset shape: {num_frames} frames, {height}x{width} pixels")

    # Output arrays
    change_date_map = np.zeros((height, width), dtype=np.float64)
    change_date_map[:] = np.nan
    change_count_map = np.zeros((height, width), dtype=np.int32)

    predicted_series = np.full((num_frames, height, width), np.nan, dtype=np.float32)
    rmse_series = np.full((num_frames, height, width), np.nan, dtype=np.float32)
    anomaly_flags = np.zeros((num_frames, height, width), dtype=np.uint8)

    X_full = build_harmonic_matrix(frac_years, temporal_periods=temporal_periods,
                                   enable_const=enable_const, enable_lin=enable_lin, enable_quad=enable_quad)

    num_features = X_full.shape[1]
    coef_series = np.full((num_frames, height, width, num_features), np.nan, dtype=np.float32)

    print("\nExecuting Sliding Window LASSO Harmonic Regression...")

    if n_jobs is None:
        n_jobs = min(8, multiprocessing.cpu_count())
    print(f"Using {n_jobs} cores for parallel processing.")

    num_chunks = max(1, math.ceil(height / chunk_size))
    chunks = []
    for y_start in range(0, height, chunk_size):
        y_end = min(y_start + chunk_size, height)
        chunk_args = (
            y_start, y_end, width,
            y_data[:, y_start:y_end, :], valid_mask[:, y_start:y_end, :],
            frac_years, X_full, acq_times,
            min_samples, CHANGE_PROBABILITY, CONSECUTIVE_ANOMALIES
        )
        chunks.append(chunk_args)

    if n_jobs > 1 and len(chunks) > 1:
        results = Parallel(n_jobs=n_jobs, backend='loky', return_as='generator')(
            delayed(_process_row_chunk)(chunk) for chunk in chunks
        )
        for res in tqdm(results, total=len(chunks), desc="Processing row chunks"):
            y_start, y_end, c_pred, c_rmse, c_coef, c_flags, c_date, c_count = res[:8]
            predicted_series[:, y_start:y_end, :] = c_pred
            rmse_series[:, y_start:y_end, :] = c_rmse
            coef_series[:, y_start:y_end, :, :] = c_coef
            anomaly_flags[:, y_start:y_end, :] = c_flags
            change_date_map[y_start:y_end, :] = c_date
            change_count_map[y_start:y_end, :] = c_count
    else:
        for chunk in tqdm(chunks, desc="Processing row chunks"):
            res = _process_row_chunk(chunk)
            y_start, y_end, c_pred, c_rmse, c_coef, c_flags, c_date, c_count = res[:8]
            predicted_series[:, y_start:y_end, :] = c_pred
            rmse_series[:, y_start:y_end, :] = c_rmse
            coef_series[:, y_start:y_end, :, :] = c_coef
            anomaly_flags[:, y_start:y_end, :] = c_flags
            change_date_map[y_start:y_end, :] = c_date
            change_count_map[y_start:y_end, :] = c_count

    print(f"\nSaving Results directly to {output_h5} -> group '{group_name}'...")
    with h5py.File(output_h5, 'a') as out_file:
        if group_name in out_file:
            del out_file[group_name]
        ccd_grp = out_file.create_group(group_name)

        if spatial_ref is not None:
            ccd_grp.attrs['spatial_ref'] = spatial_ref
        if geo_transform is not None:
            ccd_grp.attrs['GeoTransform'] = geo_transform

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

        ds_pred = ccd_grp.create_dataset('predicted_series', data=predicted_series, fillvalue=np.nan, shuffle=True, compression='gzip', compression_opts=1)
        ds_pred.attrs['_FillValue'] = np.nan

        ds_rmse = ccd_grp.create_dataset('rmse_series', data=rmse_series, fillvalue=np.nan, shuffle=True, compression='gzip', compression_opts=1)
        ds_rmse.attrs['_FillValue'] = np.nan

        ds_coef = ccd_grp.create_dataset('coef_series', data=coef_series, fillvalue=np.nan, shuffle=True, compression='gzip', compression_opts=1)
        ds_coef.attrs['_FillValue'] = np.nan

        ds_flags = ccd_grp.create_dataset('anomaly_flags', data=anomaly_flags, fillvalue=0, shuffle=True, compression='gzip', compression_opts=1)
        ds_flags.attrs['_FillValue'] = np.uint8(0)

        ds_date = ccd_grp.create_dataset('change_date_timestamp', data=change_date_map, fillvalue=np.nan, shuffle=True, compression='gzip', compression_opts=1)
        ds_date.attrs['_FillValue'] = np.nan

        ds_count = ccd_grp.create_dataset('change_count', data=change_count_map, fillvalue=0, shuffle=True, compression='gzip', compression_opts=1)
        ds_count.attrs['_FillValue'] = np.int32(0)

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
    args = parser.parse_args()

    main(
        location=args.location,
        h5_path=args.h5_path,
        output_h5=args.output_h5,
        target_metric=args.metric,
        injected=args.injected,
        launch_vis=not args.no_vis
    )
