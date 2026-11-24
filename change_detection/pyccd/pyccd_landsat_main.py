"""
USGS LCMAP Continuous Change Detection (CCDC) Baseline Pipeline for Landsat.

Implements the official pyccd algorithm (https://code.usgs.gov/lcmap/pyccd)
using only Landsat observations from the harmonized multi-sensor HDF5 file.
Configured for high-throughput streaming block execution with an in-memory budget up to 32 GB.
"""


import sys
from pathlib import Path
_script_dir = Path(__file__).resolve().parent
_repo_root = _script_dir.parent.parent
for _p in [_repo_root, _script_dir]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import os
import sys

# Prevent OpenMP / BLAS thread oversubscription and memory corruption in parallel workers
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import gc
import math
import time
import datetime
import warnings
import h5py
import numpy as np
import multiprocessing
import joblib
import contextlib
from tqdm import tqdm
from joblib import Parallel, delayed

import ccd
import ccd.math_utils
import ccd.models.lasso

# ==============================================================================
# Compatibility and Stability Patches for pyccd
# ==============================================================================
def patch_pyccd():
    """
    Ensure robust pyccd execution:
      1. SciPy >= 1.9 compatibility: ensure ccd.math_utils.mode returns 1D arrays.
      2. Memory layout and numerical safety for Scikit-Learn coordinate descent:
         guarantee contiguous Fortran float64 for X and contiguous float64 for y,
         preventing C-level access violations in cd_fast.enet_coordinate_descent.
      3. Fallback to Ridge regression if coordinate descent fails on degenerate pixels.
      4. Suppress coordinate descent convergence and runtime warnings.
    """
    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    import ccd
    import ccd.math_utils
    import ccd.models.lasso

    import warnings
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    # 1. Mode array fix
    _orig_mode = ccd.math_utils.mode

    def _patched_mode(*args, **kwargs):
        res = _orig_mode(*args, **kwargs)
        mode_arr = np.atleast_1d(res[0])
        count_arr = np.atleast_1d(res[1])
        return (mode_arr, count_arr)

    ccd.math_utils.mode = _patched_mode

    # 2. Safe coordinate descent wrapper
    from sklearn import linear_model
    from ccd.models import FittedModel
    from ccd.math_utils import calc_rmse

    def _safe_fitted_model(dates, spectra_obs, max_iter, avg_days_yr, num_coefficients):
        coef_matrix = ccd.models.lasso.coefficient_matrix(dates, avg_days_yr, num_coefficients)

        # Enforce exact memory layouts required by Cython _cd_fast (F-order X, C-order y, float64)
        X = np.asfortranarray(coef_matrix, dtype=np.float64)
        y = np.ascontiguousarray(spectra_obs, dtype=np.float64)

        try:
            lasso = linear_model.Lasso(max_iter=max_iter)
            model = lasso.fit(X, y)
        except Exception:
            ridge = linear_model.Ridge(alpha=1e-4)
            model = ridge.fit(X, y)

        predictions = model.predict(X)
        rmse, residuals = calc_rmse(y, predictions, num_pm=num_coefficients)

        return FittedModel(fitted_model=model, rmse=rmse, residual=residuals)

    ccd.models.lasso.fitted_model = _safe_fitted_model

patch_pyccd()


# ==============================================================================
# Custom 7-Band Landsat 8 and 9 (OLI) Continuous Change Detection (pyccd)
# ==============================================================================
def results_to_changemodel_7band(
    fitted_models, start_day, end_day, break_day,
    magnitudes, observation_count, change_probability, curve_qa
):
    """Consolidate 7-band regression models for Landsat 8/9 into standardized dictionary."""
    band_keys = ['coastal_aerosol', 'blue', 'green', 'red', 'nir', 'swir1', 'swir2']
    spectral_models = []
    for ix, model in enumerate(fitted_models):
        spectral = {
            'rmse': float(model.rmse),
            'coefficients': tuple(float(c) for c in model.fitted_model.coef_),
            'intercept': float(model.fitted_model.intercept_),
            'magnitude': float(magnitudes[ix]) if ix < len(magnitudes) else 0.0
        }
        spectral_models.append(spectral)

    model_dict = {
        'start_day': int(start_day),
        'end_day': int(end_day),
        'break_day': int(break_day),
        'observation_count': int(observation_count),
        'change_probability': float(change_probability),
        'curve_qa': int(curve_qa),
    }
    for k, sm in zip(band_keys, spectral_models):
        model_dict[k] = sm
    model_dict['coastal'] = spectral_models[0]
    return model_dict


def catch_7band(dates, observations, fitter_fn, processing_mask, model_window, curve_qa, proc_params):
    """Fit general models on unmodeled observations for 7 bands."""
    avg_days_yr = proc_params.AVG_DAYS_YR
    fit_max_iter = proc_params.LASSO_MAX_ITER
    num_coef = proc_params.COEFFICIENT_MIN

    period = dates[processing_mask]
    spectral_obs = observations[:, processing_mask]
    model_period = period[model_window]
    model_spectral = spectral_obs[:, model_window]

    models = [fitter_fn(model_period, spectrum, fit_max_iter, avg_days_yr, num_coef)
              for spectrum in model_spectral]

    break_day = period[-1] if model_window.stop >= period.shape[0] else period[model_window.stop]
    return results_to_changemodel_7band(
        fitted_models=models,
        start_day=period[model_window.start],
        end_day=period[model_window.stop - 1],
        break_day=break_day,
        magnitudes=np.zeros(observations.shape[0]),
        observation_count=(model_window.stop - model_window.start),
        change_probability=0,
        curve_qa=curve_qa
    )


def lookforward_7band(dates, observations, model_window, fitter_fn, processing_mask, variogram, proc_params):
    """Expand model window across 7 bands until change is detected or observations exhausted."""
    peek_size = proc_params.PEEK_SIZE
    coef_min = proc_params.COEFFICIENT_MIN
    coef_mid = proc_params.COEFFICIENT_MID
    coef_max = proc_params.COEFFICIENT_MAX
    num_obs_fact = proc_params.NUM_OBS_FACTOR
    detection_bands = proc_params.DETECTION_BANDS
    change_thresh = proc_params.CHANGE_THRESHOLD
    outlier_thresh = proc_params.OUTLIER_THRESHOLD
    avg_days_yr = proc_params.AVG_DAYS_YR
    fit_max_iter = proc_params.LASSO_MAX_ITER

    fit_window = model_window
    models = None
    change = 0

    period = dates[processing_mask]
    spectral_obs = observations[:, processing_mask]
    fit_span = ccd.procedures.span(period, fit_window)

    while model_window.stop + peek_size <= period.shape[0]:
        num_coefs = ccd.procedures.determine_num_coefs(period[model_window], coef_min, coef_mid, coef_max, num_obs_fact)
        peek_window = slice(model_window.stop, model_window.stop + peek_size)
        model_span = ccd.procedures.span(period, model_window)

        if not models or model_window.stop - model_window.start < 24 or model_span >= 1.33 * fit_span:
            fit_window = model_window
            fit_span = ccd.procedures.span(period, fit_window)
            models = [fitter_fn(period[fit_window], spectrum, fit_max_iter, avg_days_yr, num_coefs)
                      for spectrum in spectral_obs[:, fit_window]]

        residuals = np.array([ccd.procedures.calc_residuals(period[peek_window],
                                                            spectral_obs[idx, peek_window],
                                                            models[idx], avg_days_yr)
                              for idx in range(observations.shape[0])])

        if model_window.stop - model_window.start <= 24:
            comp_rmse = [models[idx].rmse for idx in detection_bands]
        else:
            closest_indexes = ccd.procedures.find_closest_doy(period, peek_window.stop - 1, fit_window, 24)
            comp_rmse = [ccd.procedures.euclidean_norm(models[idx].residual[closest_indexes]) / 4
                         for idx in detection_bands]

        magnitude = ccd.procedures.change_magnitude(residuals[detection_bands, :],
                                                    variogram[detection_bands],
                                                    comp_rmse)

        if ccd.procedures.detect_change(magnitude, change_thresh):
            change = 1
            break
        elif ccd.procedures.detect_outlier(magnitude[0], outlier_thresh):
            processing_mask = ccd.procedures.update_processing_mask(processing_mask, peek_window.start)
            period = dates[processing_mask]
            spectral_obs = observations[:, processing_mask]
            continue

        if model_window.stop + peek_size > period.shape[0]:
            break
        model_window = slice(model_window.start, model_window.stop + 1)

    result = results_to_changemodel_7band(
        fitted_models=models,
        start_day=period[model_window.start],
        end_day=period[model_window.stop - 1],
        break_day=period[peek_window.start],
        magnitudes=np.median(residuals, axis=1),
        observation_count=(model_window.stop - model_window.start),
        change_probability=change,
        curve_qa=num_coefs
    )
    return result, processing_mask, model_window


def standard_procedure_7band(dates, observations, fitter_fn, quality, prev_results, proc_params):
    """Standard CCDC iterative procedure adapted for 7-band Landsat 8/9 surface reflectance."""
    meow_size = proc_params.MEOW_SIZE
    defpeek = proc_params.PEEK_SIZE
    curve_qa = proc_params.CURVE_QA

    clear = proc_params.QA_CLEAR
    water = proc_params.QA_WATER
    unsaturated = np.all((observations > 0) & (observations < 10000), axis=0)
    processing_mask = ((quality == clear) | (quality == water)) & unsaturated
    date_mask = ccd.qa.mask_duplicate_values(dates[processing_mask])
    processing_mask[processing_mask] = date_mask

    stat_mask = ccd.procedures.statmask(dates, processing_mask, proc_params.STAT_ORD)
    if np.sum(stat_mask) < meow_size:
        stat_mask = np.copy(processing_mask)

    if prev_results:
        results = ccd.procedures.results_fromprev(prev_results)
        processing_mask = ccd.procedures.prevmask(processing_mask, dates, prev_results['processing_mask'], results)
        js = ccd.procedures.jumpstart(results, dates[processing_mask], proc_params)
        model_window, previous_end = js
        start = (model_window.start == 0)
    else:
        results = []
        model_window = slice(0, meow_size)
        previous_end = 0
        start = True

    obs_count = np.sum(processing_mask)
    if obs_count <= meow_size:
        return results, processing_mask

    peek_size = ccd.procedures.adjustpeek(dates[stat_mask], defpeek)
    proc_params.PEEK_SIZE = peek_size
    proc_params.CHANGE_THRESHOLD = ccd.procedures.adjustchgthresh(peek_size, defpeek, proc_params.CHANGE_THRESHOLD)

    variogram = ccd.procedures.adjusted_variogram(dates[stat_mask], observations[:, stat_mask])
    if not ccd.procedures.check_variogram(variogram):
        return results, processing_mask

    while model_window.stop <= dates[processing_mask].shape[0] - meow_size:
        if len(results) > 0:
            start = False
        init = ccd.procedures.initialize(dates, observations, fitter_fn, model_window, processing_mask, variogram, proc_params)
        model_window, init_models, processing_mask = init
        if init_models is None:
            break
        if model_window.start > previous_end:
            lb = ccd.procedures.lookback(dates, observations, model_window, init_models, previous_end, processing_mask, variogram, proc_params)
            model_window, processing_mask = lb
        if model_window.start - previous_end > peek_size and start is True:
            results.append(catch_7band(dates, observations, fitter_fn, processing_mask, slice(previous_end, model_window.start), curve_qa['START'], proc_params))
            start = False
        if model_window.stop + peek_size > dates[processing_mask].shape[0]:
            break
        lf = lookforward_7band(dates, observations, model_window, fitter_fn, processing_mask, variogram, proc_params)
        res, processing_mask, model_window = lf
        results.append(res)
        previous_end = model_window.stop
        model_window = slice(model_window.stop, model_window.stop + meow_size)

    if previous_end + peek_size < dates[processing_mask].shape[0]:
        results.append(catch_7band(dates, observations, fitter_fn, processing_mask, slice(previous_end, dates[processing_mask].shape[0]), curve_qa['END'], proc_params))

    return results, processing_mask


def insufficient_clear_procedure_7band(dates, observations, fitter_fn, quality, prev_results, proc_params):
    """Fallback procedure when insufficient clear observations are available for 7 bands."""
    meow_size = proc_params.MEOW_SIZE
    curve_qa = proc_params.CURVE_QA['INSUF_CLEAR']
    avg_days_yr = proc_params.AVG_DAYS_YR
    fit_max_iter = proc_params.LASSO_MAX_ITER
    num_coef = proc_params.COEFFICIENT_MIN

    green_idx = proc_params.GREEN_IDX
    filter_range = proc_params.MEDIAN_GREEN_FILTER
    max_ord = proc_params.STAT_ORD

    clear = proc_params.QA_CLEAR
    water = proc_params.QA_WATER
    unsaturated = np.all((observations > 0) & (observations < 10000), axis=0)
    standard_mask = ((quality == water) | (quality == clear)) & unsaturated

    green_mask = ccd.qa.limited_grmedian_filter(
        observations[:, standard_mask][green_idx],
        dates[standard_mask],
        filter_range,
        max_ord
    )
    standard_mask[standard_mask][:green_mask.shape[0]] &= green_mask
    date_mask = ccd.qa.mask_duplicate_values(dates[standard_mask])
    standard_mask[standard_mask] = date_mask
    processing_mask = standard_mask

    period = dates[processing_mask]
    spectral_obs = observations[:, processing_mask]

    if np.sum(processing_mask) < meow_size:
        return [], processing_mask

    models = [fitter_fn(period, spectrum, fit_max_iter, avg_days_yr, num_coef)
              for spectrum in spectral_obs]

    magnitudes = np.zeros(shape=(observations.shape[0],))
    result = results_to_changemodel_7band(
        fitted_models=models,
        start_day=dates[0],
        end_day=dates[-1],
        break_day=dates[-1],
        magnitudes=magnitudes,
        observation_count=np.sum(processing_mask),
        change_probability=0,
        curve_qa=curve_qa
    )
    return (result,), processing_mask


def permanent_snow_procedure_7band(dates, observations, fitter_fn, quality, prev_results, proc_params):
    """Procedure for areas with persistent snow cover across 7 bands."""
    meow_size = proc_params.MEOW_SIZE
    curve_qa = proc_params.CURVE_QA['PERSIST_SNOW']
    avg_days_yr = proc_params.AVG_DAYS_YR
    fit_max_iter = proc_params.LASSO_MAX_ITER
    num_coef = proc_params.COEFFICIENT_MIN

    clear = proc_params.QA_CLEAR
    water = proc_params.QA_WATER
    snow = proc_params.QA_SNOW
    unsaturated = np.all((observations > 0) & (observations < 10000), axis=0)
    mask = (((quality == water) | (quality == clear)) & unsaturated) | (quality == snow)
    date_mask = ccd.qa.mask_duplicate_values(dates[mask])
    mask[mask] = date_mask
    processing_mask = mask

    period = dates[processing_mask]
    spectral_obs = observations[:, processing_mask]

    if np.sum(processing_mask) < meow_size:
        return [], processing_mask

    models = [fitter_fn(period, spectrum, fit_max_iter, avg_days_yr, num_coef)
              for spectrum in spectral_obs]

    magnitudes = np.zeros(shape=(observations.shape[0],))
    result = results_to_changemodel_7band(
        fitted_models=models,
        start_day=dates[0],
        end_day=dates[-1],
        break_day=dates[-1],
        magnitudes=magnitudes,
        observation_count=np.sum(processing_mask),
        change_probability=0,
        curve_qa=curve_qa
    )
    return (result,), processing_mask


def detect_7band(
    dates,
    coastal_aerosols,
    blues=None,
    greens=None,
    reds=None,
    nirs=None,
    swir1s=None,
    swir2s=None,
    qas=None,
    prev_results=None,
    params=None,
):
    """
    Custom method of pyccd implementing all 7 spectral bands available in Landsat 8 and 9 (OLI):
      0: Coastal Aerosol (0.443 um / 443 nm) - B01
      1: Blue (0.482 um / 482 nm)            - B02
      2: Green (0.561 um / 561 nm)           - B03
      3: Red (0.655 um / 655 nm)             - B04
      4: NIR (0.865 um / 865 nm)             - B05
      5: SWIR 1 (1.609 um / 1609 nm)         - B06
      6: SWIR 2 (2.201 um / 2201 nm)         - B07

    Bypasses dummy thermal conversions and checks saturation across all 7 reflective bands.
    """
    proc_params = ccd.app.get_default_params()
    proc_params.QA_BITPACKED = False
    proc_params.QA_FILL = 0
    proc_params.QA_CLEAR = 1
    proc_params.QA_WATER = 2
    proc_params.QA_SHADOW = 3
    proc_params.QA_SNOW = 4
    proc_params.QA_CLOUD = 5
    proc_params.COASTAL_IDX = 0
    proc_params.BLUE_IDX = 1
    proc_params.GREEN_IDX = 2
    proc_params.RED_IDX = 3
    proc_params.NIR_IDX = 4
    proc_params.SWIR1_IDX = 5
    proc_params.SWIR2_IDX = 6
    proc_params.THERMAL_IDX = -1
    proc_params.TMASK_BANDS = [2, 5]  # Green (B03) and SWIR1 (B06) in 7-band indexing
    proc_params.DETECTION_BANDS = [0, 1, 2, 3, 4, 5, 6]  # All 7 Landsat 8/9 bands
    proc_params.CHANGE_THRESHOLD = float(scipy.stats.chi2.ppf(0.99, df=7))

    if params:
        proc_params.update(params)
        if 'DETECTION_BANDS' in params and 'CHANGE_THRESHOLD' not in params:
            proc_params.CHANGE_THRESHOLD = float(scipy.stats.chi2.ppf(0.99, df=len(proc_params.DETECTION_BANDS)))

    if blues is None and hasattr(coastal_aerosols, 'shape') and coastal_aerosols.shape[0] == 7:
        spectra = np.asarray(coastal_aerosols, dtype=np.float64)
        if greens is not None and qas is None:
            qas = greens
    elif blues is not None and greens is not None and reds is not None and nirs is not None and swir1s is not None and swir2s is not None:
        spectra = np.stack([coastal_aerosols, blues, greens, reds, nirs, swir1s, swir2s]).astype(np.float64)
    else:
        raise ValueError("detect_7band requires 7 band inputs or a (7, N) spectra array, plus QA array.")

    dates = np.asarray(dates)
    qas = np.asarray(qas)
    indices = np.argsort(dates)
    dates = dates[indices]
    spectra = spectra[:, indices]
    qas = qas[indices]

    fitter_fn = ccd.models.lasso.fitted_model
    if proc_params.QA_BITPACKED is True:
        qas = ccd.qa.unpackqa(qas, proc_params)

    probs = ccd.qa.quality_probabilities(qas, proc_params)

    clear_count = np.sum((qas == proc_params.QA_CLEAR) | (qas == proc_params.QA_WATER))
    clear_pct = clear_count / len(qas) if len(qas) > 0 else 0
    snow_count = np.sum(qas == proc_params.QA_SNOW)
    snow_pct = snow_count / len(qas) if len(qas) > 0 else 0

    if clear_pct >= proc_params.CLEAR_PCT_THRESHOLD:
        proc_fn = standard_procedure_7band
    elif snow_pct >= proc_params.SNOW_PCT_THRESHOLD:
        proc_fn = permanent_snow_procedure_7band
    else:
        proc_fn = insufficient_clear_procedure_7band

    procedure_results = proc_fn(dates, spectra, fitter_fn, qas, prev_results, proc_params)
    change_models, processing_mask = procedure_results

    return {
        'algorithm': 'pyccd-7band-landsat89',
        'processing_mask': [int(_) for _ in processing_mask],
        'change_models': change_models,
        'cloud_prob': probs[0],
        'snow_prob': probs[1],
        'water_prob': probs[2],
    }

# Alias detect to custom 7-band method
detect = detect_7band


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


# ==============================================================================
# Configuration Defaults
# ==============================================================================
LOCATION = "Rochesterv2"
H5_PATH = f"C:/satelliteImagery/HLST30/HLST_{LOCATION}.h5"
OUTPUT_GROUP = "PYCCD_Results"
INJECTED = False

BAND_NAMES = ['coastal_aerosol', 'blue', 'green', 'red', 'nir', 'swir1', 'swir2']
BAND_INDICES = {
    'coastal_aerosol': 0,
    'coastal': 0,
    'blue': 1,
    'green': 2,
    'red': 3,
    'nir': 4,
    'swir1': 5,
    'swir2': 6
}
TARGET_BAND = 'nir'

MIN_CLEAR_SAMPLES = 12
NOMINAL_THERMAL_KELVIN_SCALED = 2931

def check_ram_capacity():
    """Detect system RAM to determine whether to enable high-RAM preloading."""
    try:
        import psutil
        vm = psutil.virtual_memory()
        return vm.total / (1024 ** 3), vm.available / (1024 ** 3)
    except Exception:
        return 64.0, 32.0


# Scaled parameters targeting a ~20-25 GB memory footprint (within 32 GB RAM budget)
# 192 rows per block produces 8 blocks across 1409 scene rows.
# 16 rows per worker chunk produces 12 worker subchunks per block, matching 12 workers.
BLOCK_SIZE_ROWS = 192        # Streaming block size written directly to HDF5
WORKER_CHUNK_ROWS = 16       # Row chunk per parallel worker task
PRELOAD_INPUT = True         # Preload 10.6 GB Landsat inputs into RAM to eliminate disk I/O
DEFAULT_N_JOBS = max(1, min(12, multiprocessing.cpu_count() - 4))  # 12 workers = 75% CPU load on 16 threads


def build_harmonic_basis(ordinal_dates, avg_days_yr=365.2425):
    """
    Constructs the 8-feature Fourier basis matrix matching pyccd's lasso representation:
    [1.0, t, cos(w*t), sin(w*t), cos(2w*t), sin(2w*t), cos(3w*t), sin(3w*t)]
    where w = 2*pi / avg_days_yr.
    """
    w = 2.0 * math.pi / avg_days_yr
    w12 = w * ordinal_dates
    cols = [
        np.ones_like(ordinal_dates, dtype=np.float64),
        ordinal_dates.astype(np.float64),
        np.cos(w12),
        np.sin(w12),
        np.cos(2.0 * w12),
        np.sin(2.0 * w12),
        np.cos(3.0 * w12),
        np.sin(3.0 * w12)
    ]
    return np.column_stack(cols)


def decode_categorical_qa(fmask_arr, common_mask_arr, sr_data, fill_val=None):
    """
    Decodes HLS Fmask bits and common_mask into pyccd categorical QA codes
    consistent with USGS LCMAP CCDC ADD (LSDS-1982 Section 2.6.1 & Appendix B):
      0 = Fill / Nodata
      1 = Clear land
      2 = Water
      3 = Cloud shadow
      4 = Snow / Ice
      5 = Cloud / Cirrus / Adjacent cloud / Masked invalid

    Precedence hierarchy (following ccd.qa.qabitval / LSDS-1982 Section 2.6.1):
      Fill (0) > Cloud (5) > Shadow (3) > Snow (4) > Water (2) > Clear (1)
    """
    if sr_data is None:
        sr_is_fill = False
    elif sr_data.ndim == 1:
        if fill_val is not None:
            sr_is_fill = (sr_data == fill_val)
        else:
            sr_is_fill = (sr_data == -9999) | (sr_data == -32768) | (sr_data < -9000)
    elif sr_data.ndim == 2:
        axis = 1 if sr_data.shape[0] == fmask_arr.shape[0] else 0
        if fill_val is not None:
            sr_is_fill = np.any(sr_data == fill_val, axis=axis)
        else:
            sr_is_fill = np.any((sr_data == -9999) | (sr_data == -32768) | (sr_data < -9000), axis=axis)
    else:
        axis = tuple(range(1, sr_data.ndim))
        if fill_val is not None:
            sr_is_fill = np.any(sr_data == fill_val, axis=axis)
        else:
            sr_is_fill = np.any((sr_data == -9999) | (sr_data == -32768) | (sr_data < -9000), axis=axis)

    is_fill = (fmask_arr == 255) | sr_is_fill
    is_cloud = (fmask_arr & ((1 << 1) | (1 << 0) | (1 << 2))) > 0
    is_shadow = (fmask_arr & (1 << 3)) > 0
    is_snow = (fmask_arr & (1 << 4)) > 0
    is_water = (fmask_arr & (1 << 5)) > 0

    # Start with default Clear land (1)
    qas = np.ones(fmask_arr.shape, dtype=np.uint8)

    # 2. Water = 2
    qas[is_water] = 2
    # 3. Snow = 4 (Snow > Water)
    qas[is_snow] = 4
    # 4. Shadow = 3 (Shadow > Snow > Water)
    qas[is_shadow] = 3
    # 5. Cloud = 5 (Cloud > Shadow > Snow > Water)
    # Includes Fmask cloud/cirrus/adj_cloud, plus any residual common_mask invalidity
    qas[is_cloud | (common_mask_arr & ~is_shadow & ~is_snow & ~is_fill)] = 5
    # 6. Fill = 0 (Fill beats all)
    qas[is_fill] = 0

    return qas


def _process_row_subchunk(subchunk_args):
    """Worker function for processing a small row subchunk with pyccd."""
    patch_pyccd()

    if len(subchunk_args) >= 16:
        (sub_y0, sub_y1, width,
         sr_sub, fm_sub, cm_sub,
         ordinal_dates, acq_times, X_full,
         target_band, save_coef, save_all_bands, save_ndvi,
         min_clear_samples,
         fill_val, scale_factor) = subchunk_args[:16]
    else:
        (sub_y0, sub_y1, width,
         sr_sub, fm_sub, cm_sub,
         ordinal_dates, acq_times, X_full,
         target_band, save_coef, save_all_bands, save_ndvi,
         min_clear_samples) = subchunk_args[:14]
        fill_val = None
        scale_factor = 10000.0

    num_frames = sr_sub.shape[0]
    chunk_h = sub_y1 - sub_y0

    sub_pred = np.full((num_frames, chunk_h, width), np.nan, dtype=np.float32)
    sub_rmse = np.full((num_frames, chunk_h, width), np.nan, dtype=np.float32)
    sub_coef = np.full((num_frames, chunk_h, width, 8), np.nan, dtype=np.float32) if save_coef else None
    sub_flags = np.zeros((num_frames, chunk_h, width), dtype=np.uint8)
    sub_date = np.full((chunk_h, width), np.nan, dtype=np.float64)
    sub_count = np.zeros((chunk_h, width), dtype=np.int32)
    sub_water_prob = np.full((chunk_h, width), np.nan, dtype=np.float32)

    sub_all_bands = {}
    if save_all_bands:
        for b_name in BAND_NAMES:
            sub_all_bands[b_name] = np.full((num_frames, chunk_h, width), np.nan, dtype=np.float32)

    sub_ndvi = None
    if save_ndvi:
        sub_ndvi = np.full((num_frames, chunk_h, width), np.nan, dtype=np.float32)

    proc_params = {
        'QA_BITPACKED': False,
        'QA_FILL': 0,
        'QA_CLEAR': 1,
        'QA_WATER': 2,
        'QA_SHADOW': 3,
        'QA_SNOW': 4,
        'QA_CLOUD': 5,
        'LASSO_MAX_ITER': 2000,
        'DETECTION_BANDS': [0, 1, 2, 3, 4, 5, 6]
    }

    for y_local in range(chunk_h):
        for x in range(width):
            fm_pixel = fm_sub[:, y_local, x]
            cm_pixel = cm_sub[:, y_local, x]
            sr_pixel_bands = sr_sub[:, 0:7, y_local, x]

            qas_pixel = decode_categorical_qa(fm_pixel, cm_pixel, sr_pixel_bands, fill_val=fill_val)

            # LSDS-1982 Section 2.6.2.1 / 2.6.1: usable observations for model fitting are Clear (1) and Water (2),
            # with Snow (4) also valid under the permanent snow procedure.
            usable_count = int(np.count_nonzero((qas_pixel == 1) | (qas_pixel == 2) | (qas_pixel == 4)))
            if usable_count < min_clear_samples:
                continue

            coastal = sr_sub[:, 0, y_local, x]
            blues = sr_sub[:, 1, y_local, x]
            greens = sr_sub[:, 2, y_local, x]
            reds = sr_sub[:, 3, y_local, x]
            nirs = sr_sub[:, 4, y_local, x]
            swir1s = sr_sub[:, 5, y_local, x]
            swir2s = sr_sub[:, 6, y_local, x]

            try:
                res = detect_7band(
                    ordinal_dates,
                    coastal, blues, greens, reds, nirs, swir1s, swir2s,
                    qas_pixel,
                    params=proc_params
                )
            except Exception:
                continue

            sub_water_prob[y_local, x] = float(res.get('water_prob', np.nan))

            models = res.get('change_models', [])
            if not models:
                continue

            breaks = [m for m in models if m.get('change_probability', 0) > 0]
            sub_count[y_local, x] = len(breaks)

            if breaks:
                first_break_ord = int(breaks[0]['break_day'])
                try:
                    dt = datetime.datetime.fromordinal(first_break_ord).replace(tzinfo=datetime.timezone.utc)
                    sub_date[y_local, x] = dt.timestamp()
                except (ValueError, OverflowError):
                    sub_date[y_local, x] = np.nan

                for b in breaks:
                    b_day = b['break_day']
                    t_b = int(np.argmin(np.abs(ordinal_dates - b_day)))
                    sub_flags[t_b, y_local, x] = 1

            num_models = len(models)
            if num_models == 1:
                active_model_indices = np.zeros(num_frames, dtype=int)
            else:
                split_days = [
                    m['break_day'] if m.get('change_probability', 0) > 0 else m['end_day']
                    for m in models[:-1]
                ]
                active_model_indices = np.zeros(num_frames, dtype=int)
                for t_idx, ord_d in enumerate(ordinal_dates):
                    m_idx = 0
                    while m_idx < len(split_days) and ord_d > split_days[m_idx]:
                        m_idx += 1
                    active_model_indices[t_idx] = m_idx

            target_preds = []
            target_rmses = []
            target_coefs = []

            for m in models:
                b_info = m[target_band]
                intercept = float(b_info['intercept'])
                coefs_7 = list(b_info['coefficients'])
                theta_8 = np.array([intercept] + coefs_7, dtype=np.float64)

                pred_scaled = (X_full @ theta_8) / scale_factor
                rmse_scaled = float(b_info['rmse']) / scale_factor

                target_preds.append(pred_scaled)
                target_rmses.append(rmse_scaled)
                if save_coef:
                    target_coefs.append(theta_8 / scale_factor)

            for t_idx in range(num_frames):
                m_active = active_model_indices[t_idx]
                sub_pred[t_idx, y_local, x] = target_preds[m_active][t_idx]
                sub_rmse[t_idx, y_local, x] = target_rmses[m_active]
                if save_coef:
                    sub_coef[t_idx, y_local, x, :] = target_coefs[m_active]

            if save_all_bands or save_ndvi:
                all_band_preds = {}
                for b_name in BAND_NAMES:
                    b_preds = []
                    for m in models:
                        b_info = m[b_name]
                        b_theta = np.array([float(b_info['intercept'])] + list(b_info['coefficients']), dtype=np.float64)
                        b_preds.append((X_full @ b_theta) / scale_factor)
                    all_band_preds[b_name] = b_preds

                if save_all_bands:
                    for b_name in BAND_NAMES:
                        for t_idx in range(num_frames):
                            m_active = active_model_indices[t_idx]
                            sub_all_bands[b_name][t_idx, y_local, x] = all_band_preds[b_name][m_active][t_idx]

                if save_ndvi:
                    for t_idx in range(num_frames):
                        m_active = active_model_indices[t_idx]
                        p_nir = all_band_preds['nir'][m_active][t_idx]
                        p_red = all_band_preds['red'][m_active][t_idx]
                        denom = p_nir + p_red
                        if np.abs(denom) > 1e-6:
                            sub_ndvi[t_idx, y_local, x] = (p_nir - p_red) / denom

    return (
        sub_y0, sub_y1,
        sub_pred, sub_rmse, sub_coef, sub_flags,
        sub_date, sub_count,
        sub_all_bands if save_all_bands else None,
        sub_ndvi if save_ndvi else None,
        sub_water_prob
    )


def main(
    location=LOCATION,
    h5_path=None,
    output_h5=None,
    output_group=None,
    injected=INJECTED,
    target_band=TARGET_BAND,
    save_coef=True,
    save_all_bands=False,
    save_ndvi=False,
    min_clear_samples=MIN_CLEAR_SAMPLES,
    block_size=BLOCK_SIZE_ROWS,
    worker_chunk_rows=WORKER_CHUNK_ROWS,
    preload_input=PRELOAD_INPUT,
    n_jobs=None,
    row_start=None,
    row_end=None,
    col_start=None,
    col_end=None,
    resume=False,
    overwrite=True,
    **kwargs,
):
    """
    Executes USGS pyccd change detection with streaming block writing to HDF5.
    Configured for high throughput utilizing up to 32 GB of system RAM.
    Supports automatic block-level checkpoint resumption.
    """

    if output_group is None:
        output_group = "INJECTED_PYCCD_Results" if injected else OUTPUT_GROUP
    elif injected and output_group == OUTPUT_GROUP:
        output_group = "INJECTED_PYCCD_Results"

    sr_dataset_name = "injected_surface_reflectance" if injected else "surface_reflectance"

    if h5_path is None:
        h5_path = f"C:/satelliteImagery/HLST30/HLST_{location}.h5"
        if not os.path.exists(h5_path):
            alt_path = f"C:/satelliteImagery/HLST30/HLST_{location}v2.h5"
            if os.path.exists(alt_path):
                h5_path = alt_path

    if output_h5 is None:
        output_h5 = h5_path

    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"Source HDF5 file not found: {h5_path}")

    total_ram_gb, avail_ram_gb = check_ram_capacity()
    cpu_cores = multiprocessing.cpu_count()

    if n_jobs is None:
        n_jobs = DEFAULT_N_JOBS

    print(f"Opening Landsat dataset: {h5_path} (SR dataset: '{sr_dataset_name}')")
    print(f"System Hardware: {cpu_cores} logical cores | {total_ram_gb:.1f} GB RAM ({avail_ram_gb:.1f} GB free)")
    with h5py.File(h5_path, 'r') as f_in:
        hlsl30_grp = f_in['/HDFEOS/GRIDS/HLSL30/Data Fields']
        if sr_dataset_name not in hlsl30_grp:
            raise KeyError(f"Dataset '{sr_dataset_name}' not found in {hlsl30_grp.name}. Available datasets: {list(hlsl30_grp.keys())}")
        sr_ds = hlsl30_grp[sr_dataset_name]

        # Datatype validation
        if sr_ds.dtype != np.int16:
            raise TypeError(
                f"Dataset '{sr_dataset_name}' has datatype {sr_ds.dtype}; pyccd Landsat pipeline requires int16."
            )

        # Retrieve _FillValue attribute
        if "_FillValue" in sr_ds.attrs:
            fill_val = sr_ds.attrs["_FillValue"]
        elif "fillvalue" in sr_ds.attrs:
            fill_val = sr_ds.attrs["fillvalue"]
        elif hasattr(sr_ds, "fillvalue") and sr_ds.fillvalue is not None and sr_ds.fillvalue != 0:
            fill_val = sr_ds.fillvalue
        else:
            raise KeyError(
                f"Dataset '{sr_dataset_name}' lacks mandatory '_FillValue' attribute."
            )
        fill_val = int(fill_val)

        # Retrieve scale_factor attribute
        if "scale_factor" in sr_ds.attrs:
            scale_factor = float(sr_ds.attrs["scale_factor"])
        elif "scale_to_float" in sr_ds.attrs:
            sf = float(sr_ds.attrs["scale_to_float"])
            scale_factor = (1.0 / sf) if sf < 1.0 else sf
        else:
            raise KeyError(
                f"Dataset '{sr_dataset_name}' lacks mandatory 'scale_factor' or 'scale_to_float' attribute."
            )

        acq_times = sr_ds.attrs.get('acquisition_time')
        if acq_times is None:
            acq_times = f_in['/HDFEOS/GRIDS/HARMONIZED/Data Fields'].attrs.get('acquisition_time')
        acq_times = np.array(acq_times[:], dtype=np.float64)

        geo_transform = sr_ds.attrs.get('GeoTransform', f_in.attrs.get('GeoTransform'))
        spatial_ref = sr_ds.attrs.get('spatial_ref', f_in.attrs.get('spatial_ref'))

        full_num_frames, num_bands, full_height, full_width = sr_ds.shape

        y0 = 0 if row_start is None else max(0, row_start)
        y1 = full_height if row_end is None else min(full_height, row_end)
        x0 = 0 if col_start is None else max(0, col_start)
        x1 = full_width if col_end is None else min(full_width, col_end)

        height = y1 - y0
        width = x1 - x0

    ordinal_dates = np.array([
        datetime.date.fromtimestamp(ts).toordinal() for ts in acq_times
    ], dtype=np.int64)

    X_full = build_harmonic_basis(ordinal_dates)
    num_frames = len(ordinal_dates)

    print(f"Landsat grid: {full_num_frames} frames, {num_bands} bands, processing {height}x{width} pixels")
    print(f"Target band: '{target_band}' | Storing coefficients: {save_coef}")
    print(f"Execution parameters: {n_jobs} parallel workers, {block_size}-row streaming blocks, {worker_chunk_rows}-row worker chunks")
    print(f"Checkpoint resume enabled: {resume} (Overwrite: {overwrite})")

    # Preload inputs if enabled
    sr_preloaded = None
    fm_preloaded = None
    cm_preloaded = None

    if preload_input:
        print(f"Preloading Landsat data into RAM (~10.6 GB) to maximize processing throughput...")
        t_load = time.time()
        with h5py.File(h5_path, 'r') as f_in:
            hlsl30_grp = f_in['/HDFEOS/GRIDS/HLSL30/Data Fields']
            sr_preloaded = np.array(hlsl30_grp[sr_dataset_name][:, :, y0:y1, x0:x1], dtype=np.int16)
            fm_preloaded = np.array(hlsl30_grp['Fmask'][:, y0:y1, x0:x1], dtype=np.uint8)
            cm_preloaded = np.array(hlsl30_grp['common_mask'][:, y0:y1, x0:x1], dtype=bool)
        load_sec = time.time() - t_load
        print(f"Preload complete in {load_sec:.1f}s. Landsat inputs memory-resident (active RAM: ~10.6 GB).")
    else:
        print("Preloading disabled: streaming blocks on-demand from disk.")

    # Check and initialize chunked HDF5 datasets in target group
    needs_creation = True
    with h5py.File(output_h5, 'a') as out_file:
        if output_group in out_file:
            if overwrite or not resume:
                del out_file[output_group]
            else:
                existing_grp = out_file[output_group]
                exp_3d = (num_frames, height, width)
                exp_2d = (height, width)
                if ('predicted_series' in existing_grp and 'change_date_timestamp' in existing_grp and
                        existing_grp['predicted_series'].shape == exp_3d and
                        existing_grp['change_date_timestamp'].shape == exp_2d):
                    needs_creation = False
                    print(f"Target group '{output_group}' already exists with matching grid schema. Resuming existing progress...")
                else:
                    del out_file[output_group]

        if needs_creation:
            print(f"Initializing chunked datasets in {output_h5} -> group '{output_group}'...")
            ccd_grp = out_file.create_group(output_group)

            if spatial_ref is not None:
                ccd_grp.attrs['spatial_ref'] = spatial_ref
            if geo_transform is not None:
                ccd_grp.attrs['GeoTransform'] = geo_transform

            ccd_grp.attrs['ALGORITHM'] = 'lcmap-pyccd-7band-landsat89'
            ccd_grp.attrs['TARGET_BAND'] = target_band
            ccd_grp.attrs['TARGET_BANDS'] = BAND_NAMES
            ccd_grp.attrs['DETECTION_BANDS'] = [0, 1, 2, 3, 4, 5, 6]
            ccd_grp.attrs['SOURCE_DATA'] = h5_path
            ccd_grp.attrs['SOURCE_GRID'] = '/HDFEOS/GRIDS/HLSL30'
            ccd_grp.attrs['SR_DATASET'] = sr_dataset_name
            ccd_grp.attrs['INJECTED'] = bool(injected)
            ccd_grp.attrs['NUM_LANDSAT_FRAMES'] = num_frames
            ccd_grp.attrs['MIN_CLEAR_SAMPLES'] = min_clear_samples
            ccd_grp.attrs['PEEK_SIZE'] = 6
            ccd_grp.attrs['MEOW_SIZE'] = 12
            ccd_grp.attrs['CHANGE_THRESHOLD'] = 18.475306906582357
            ccd_grp.attrs['SOURCE_FILL_VALUE'] = fill_val
            ccd_grp.attrs['SOURCE_SCALE_FACTOR'] = scale_factor

            if (height, width) != (full_height, full_width):
                ccd_grp.attrs['ROI_ROWS'] = [y0, y1]
                ccd_grp.attrs['ROI_COLS'] = [x0, x1]

            chunk_3d = (min(64, num_frames), min(16, height), min(64, width))
            ccd_grp.create_dataset('predicted_series', shape=(num_frames, height, width), dtype='float32', chunks=chunk_3d, compression='gzip', compression_opts=6, shuffle=True)
            ccd_grp.create_dataset('rmse_series', shape=(num_frames, height, width), dtype='float32', chunks=chunk_3d, compression='gzip', compression_opts=6, shuffle=True)
            ccd_grp.create_dataset('anomaly_flags', shape=(num_frames, height, width), dtype='uint8', chunks=chunk_3d, compression='gzip', compression_opts=6, shuffle=True)
            ccd_grp.create_dataset('change_date_timestamp', shape=(height, width), dtype='float64', chunks=True, compression='gzip', compression_opts=6, shuffle=True)
            ccd_grp.create_dataset('change_count', shape=(height, width), dtype='int32', chunks=True, compression='gzip', compression_opts=6, shuffle=True)
            ccd_grp.create_dataset('water_probability', shape=(height, width), dtype='float32', chunks=True, compression='gzip', compression_opts=6, shuffle=True)
            ccd_grp['water_probability'].attrs['description'] = 'LCMAP CCDC water probability: ratio of water observations to total clear and water observations (LSDS-1982 Section 2.6.1)'

            if save_coef:
                chunk_4d = (min(32, num_frames), min(8, height), min(32, width), 8)
                ccd_grp.create_dataset('coef_series', shape=(num_frames, height, width, 8), dtype='float32', chunks=chunk_4d, compression='gzip', compression_opts=6, shuffle=True)

            if save_all_bands:
                for b_name in BAND_NAMES:
                    ccd_grp.create_dataset(f'predicted_{b_name}', shape=(num_frames, height, width), dtype='float32', chunks=chunk_3d, compression='gzip', compression_opts=6, shuffle=True)

            if save_ndvi:
                ccd_grp.create_dataset('predicted_ndvi', shape=(num_frames, height, width), dtype='float32', chunks=chunk_3d, compression='gzip', compression_opts=6, shuffle=True)

    # Streaming block execution loop
    total_blocks = math.ceil(height / block_size)
    print(f"Beginning streaming execution across {total_blocks} blocks ({block_size} rows per block)...")
    t0 = time.time()
    total_changed_pixels = 0

    with h5py.File(output_h5, 'a') as out_file:
        ccd_grp = out_file[output_group]
        ds_pred = ccd_grp['predicted_series']
        ds_rmse = ccd_grp['rmse_series']
        ds_flags = ccd_grp['anomaly_flags']
        ds_date = ccd_grp['change_date_timestamp']
        ds_count = ccd_grp['change_count']
        ds_water = ccd_grp['water_probability']
        ds_coef = ccd_grp['coef_series'] if save_coef else None
        ds_all_bands = {b: ccd_grp[f'predicted_{b}'] for b in BAND_NAMES} if save_all_bands else {}
        ds_ndvi = ccd_grp['predicted_ndvi'] if save_ndvi else None

        for b_idx in range(total_blocks):
            b_y0 = b_idx * block_size
            b_y1 = min(b_y0 + block_size, height)
            b_height = b_y1 - b_y0

            # Check if block was already computed in a previous run
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

            if preload_input:
                sr_block = sr_preloaded[:, :, b_y0:b_y1, :]
                fm_block = fm_preloaded[:, b_y0:b_y1, :]
                cm_block = cm_preloaded[:, b_y0:b_y1, :]
            else:
                with h5py.File(h5_path, 'r') as f_in:
                    hlsl30_grp = f_in['/HDFEOS/GRIDS/HLSL30/Data Fields']
                    sr_block = hlsl30_grp[sr_dataset_name][:, :, (y0 + b_y0):(y0 + b_y1), x0:x1]
                    fm_block = hlsl30_grp['Fmask'][:, (y0 + b_y0):(y0 + b_y1), x0:x1]
                    cm_block = hlsl30_grp['common_mask'][:, (y0 + b_y0):(y0 + b_y1), x0:x1]

            # Allocate block buffers
            block_pred = np.full((num_frames, b_height, width), np.nan, dtype=np.float32)
            block_rmse = np.full((num_frames, b_height, width), np.nan, dtype=np.float32)
            block_coef = np.full((num_frames, b_height, width, 8), np.nan, dtype=np.float32) if save_coef else None
            block_flags = np.zeros((num_frames, b_height, width), dtype=np.uint8)
            block_date = np.full((b_height, width), np.nan, dtype=np.float64)
            block_count = np.zeros((b_height, width), dtype=np.int32)
            block_water = np.full((b_height, width), np.nan, dtype=np.float32)
            block_bands = {b: np.full((num_frames, b_height, width), np.nan, dtype=np.float32) for b in BAND_NAMES} if save_all_bands else None
            block_ndvi = np.full((num_frames, b_height, width), np.nan, dtype=np.float32) if save_ndvi else None

            # Partition block into worker subchunks
            subchunks = []
            for sub_y0 in range(0, b_height, worker_chunk_rows):
                sub_y1 = min(sub_y0 + worker_chunk_rows, b_height)
                sub_args = (
                    sub_y0, sub_y1, width,
                    np.ascontiguousarray(sr_block[:, :, sub_y0:sub_y1, :]),
                    np.ascontiguousarray(fm_block[:, sub_y0:sub_y1, :]),
                    np.ascontiguousarray(cm_block[:, sub_y0:sub_y1, :]),
                    ordinal_dates, acq_times, X_full,
                    target_band, save_coef, save_all_bands, save_ndvi,
                    min_clear_samples,
                    fill_val, scale_factor
                )
                subchunks.append(sub_args)

            # Process block subchunks in parallel
            if n_jobs > 1 and len(subchunks) > 1:
                results = Parallel(n_jobs=n_jobs, backend='loky')(
                    delayed(_process_row_subchunk)(sc) for sc in subchunks
                )
            else:
                results = [_process_row_subchunk(sc) for sc in subchunks]

            # Assemble subchunks into block
            for (sub_y0, sub_y1, s_pred, s_rmse, s_coef, s_flags, s_date, s_count, s_bands, s_ndvi, s_water) in results:
                block_pred[:, sub_y0:sub_y1, :] = s_pred
                block_rmse[:, sub_y0:sub_y1, :] = s_rmse
                block_flags[:, sub_y0:sub_y1, :] = s_flags
                block_date[sub_y0:sub_y1, :] = s_date
                block_count[sub_y0:sub_y1, :] = s_count
                block_water[sub_y0:sub_y1, :] = s_water
                if save_coef and s_coef is not None:
                    block_coef[:, sub_y0:sub_y1, :, :] = s_coef
                if save_all_bands and s_bands is not None:
                    for b_name in BAND_NAMES:
                        block_bands[b_name][:, sub_y0:sub_y1, :] = s_bands[b_name]
                if save_ndvi and s_ndvi is not None:
                    block_ndvi[:, sub_y0:sub_y1, :] = s_ndvi

            # Stream write directly into HDF5 datasets
            ds_pred[:, b_y0:b_y1, :] = block_pred
            ds_rmse[:, b_y0:b_y1, :] = block_rmse
            ds_flags[:, b_y0:b_y1, :] = block_flags
            ds_date[b_y0:b_y1, :] = block_date
            ds_count[b_y0:b_y1, :] = block_count
            ds_water[b_y0:b_y1, :] = block_water
            if save_coef:
                ds_coef[:, b_y0:b_y1, :, :] = block_coef
            if save_all_bands:
                for b_name in BAND_NAMES:
                    ds_all_bands[b_name][:, b_y0:b_y1, :] = block_bands[b_name]
            if save_ndvi:
                ds_ndvi[:, b_y0:b_y1, :] = block_ndvi

            out_file.flush()

            b_changed = int(np.count_nonzero(~np.isnan(block_date)))
            total_changed_pixels += b_changed
            elapsed_block = time.time() - t0
            pct_complete = (b_y1 / height) * 100.0
            print(f"Block {b_idx + 1:2d}/{total_blocks} (rows {b_y0:4d}:{b_y1:4d}) complete ({pct_complete:5.1f}%) | "
                  f"Changed pixels in block: {b_changed:4d} | Elapsed: {elapsed_block/60:4.1f} min")

            # Explicit memory release
            del results, subchunks
            del block_pred, block_rmse, block_flags, block_date, block_count, block_water
            if not preload_input:
                del sr_block, fm_block, cm_block
            if save_coef:
                del block_coef
            if save_all_bands:
                del block_bands
            if save_ndvi:
                del block_ndvi
            gc.collect()

            # Recycle worker processes between blocks to eliminate C-level memory corruption and fragmentation
            try:
                from joblib.externals.loky import get_reusable_executor
                get_reusable_executor().shutdown(wait=True)
            except Exception:
                pass

        total_elapsed = time.time() - t0
        ccd_grp.attrs['ELAPSED_SECONDS'] = total_elapsed

    if preload_input:
        del sr_preloaded, fm_preloaded, cm_preloaded
        gc.collect()

    total_pixels = height * width
    mode_str = "Injected Landsat SR" if injected else "Baseline Landsat SR"
    print(f"\nPyCCD execution complete ({mode_str}) in {total_elapsed/60:.2f} minutes.")
    print(f"Total changed pixels detected: {total_changed_pixels:,} / {total_pixels:,} ({total_changed_pixels / total_pixels * 100:.2f}%)")
    print(f"Results safely flushed to: {output_h5} -> group '{output_group}'")
    return output_h5


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Memory-Efficient Streaming USGS PyCCD Baseline on Landsat Data")
    parser.add_argument('--location', type=str, default=LOCATION, help="Location name")
    parser.add_argument('--h5-path', type=str, default=None, help="Input source HDF5 path")
    parser.add_argument('--output-h5', type=str, default=None, help="Output HDF5 path")
    parser.add_argument('--group', type=str, default=None, help="Output HDF5 group name (defaults to 'PYCCD_Results' or 'INJECTED_PYCCD_Results' when --injected)")
    parser.add_argument('--injected', action='store_true', default=INJECTED, help="Run PyCCD change detection on injected surface reflectance ('injected_surface_reflectance')")
    parser.add_argument('--target-band', type=str, default=TARGET_BAND, choices=BAND_NAMES, help="Primary band for predicted series")
    parser.add_argument('--no-coef', action='store_true', help="Skip storing 8 harmonic coefficients series to save disk space")
    parser.add_argument('--save-all-bands', action='store_true', help="Also save predicted series for all reflective bands")
    parser.add_argument('--save-ndvi', action='store_true', help="Also compute and save predicted NDVI series")
    parser.add_argument('--block-size', type=int, default=BLOCK_SIZE_ROWS, help="Rows per streaming block written to HDF5")
    parser.add_argument('--worker-chunk-rows', type=int, default=WORKER_CHUNK_ROWS, help="Rows per worker task")
    parser.add_argument('--preload-input', action='store_true', default=PRELOAD_INPUT, help="Preload all Landsat inputs into RAM (~10.6 GB) to eliminate disk I/O")
    parser.add_argument('--no-preload-input', action='store_false', dest='preload_input', help="Stream blocks on-demand from disk (RAM-conserving mode)")
    parser.add_argument('--n-jobs', type=int, default=DEFAULT_N_JOBS, help="Number of parallel worker processes")
    parser.add_argument('--row-start', type=int, default=None, help="Optional starting row")
    parser.add_argument('--row-end', type=int, default=None, help="Optional ending row")
    parser.add_argument('--col-start', type=int, default=None, help="Optional starting col")
    parser.add_argument('--col-end', type=int, default=None, help="Optional ending col")
    parser.add_argument('--overwrite', action='store_true', default=True, help="Overwrite existing output group instead of resuming")
    parser.add_argument('--no-resume', action='store_false', dest='resume', default=True, help="Disable automatic block resumption")
    args = parser.parse_args()

    main(
        location=args.location,
        h5_path=args.h5_path,
        output_h5=args.output_h5,
        output_group=args.group,
        injected=args.injected,
        target_band=args.target_band,
        save_coef=not args.no_coef,
        save_all_bands=args.save_all_bands,
        save_ndvi=args.save_ndvi,
        block_size=args.block_size,
        worker_chunk_rows=args.worker_chunk_rows,
        preload_input=args.preload_input,
        n_jobs=args.n_jobs,
        row_start=args.row_start,
        row_end=args.row_end,
        col_start=args.col_start,
        col_end=args.col_end,
        resume=args.resume,
        overwrite=args.overwrite,
    )
