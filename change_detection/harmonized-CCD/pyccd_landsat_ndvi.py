"""
USGS LCMAP Continuous Change Detection (CCDC) Single-Index NDVI Baseline Pipeline.

Implements the official pyccd algorithm (https://code.usgs.gov/lcmap/pyccd)
configured strictly for single-index NDVI change detection on Landsat observations.
Outputs are stored in the harmonized HDF5 file under group 'PYCCD_NDVI_Results',
enabling direct 3-way benchmarking against:
  1. Full-Spectrum Multi-Band LCMAP CCD ('PYCCD_Results')
  2. Single-Index NDVI LCMAP CCD ('PYCCD_NDVI_Results')
  3. Spatiotemporal Spectral Complexity CCD ('CCD_Results')

Configured for high-throughput streaming block execution with an in-memory budget up to 32 GB.
"""

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
from scipy.stats import chi2

import ccd
import ccd.math_utils
import ccd.qa
import ccd.procedures
import ccd.models.lasso

# ==============================================================================
# Compatibility and Single-Index Patches for pyccd
# ==============================================================================
def patch_pyccd():
    """
    Apply essential runtime patches to pyccd:
      1. SciPy >= 1.9 compatibility: ensure ccd.math_utils.mode returns 1D arrays.
      2. Dynamic degrees of freedom in ccd.procedures.adjustchgthresh:
         The stock pyccd hardcodes df=5 in adjustchgthresh. For single-index NDVI,
         df=1 must be enforced so threshold dynamic scaling does not revert to df=5.
      3. Range expansion in ccd.qa.filter_saturated:
         Allow slot 3 (NDVI scaled by 10,000) to span [-10000, 10000] without being
         flagged as saturated/invalid when negative values occur.
      4. Memory layout and numerical safety for Scikit-Learn coordinate descent:
         guarantee contiguous Fortran float64 for X and contiguous float64 for y,
         preventing C-level access violations in cd_fast.enet_coordinate_descent.
      5. Fallback to Ridge regression if coordinate descent fails on degenerate pixels.
      6. Suppress coordinate descent convergence and runtime warnings.
    """
    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    import ccd
    import ccd.math_utils
    import ccd.qa
    import ccd.procedures
    import ccd.models.lasso

    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    # 1. Mode array fix
    _orig_mode = ccd.math_utils.mode
    def _patched_mode(*args, **kwargs):
        res = _orig_mode(*args, **kwargs)
        return (np.atleast_1d(res[0]), np.atleast_1d(res[1]))
    ccd.math_utils.mode = _patched_mode

    # 2. Dynamic chi-square degrees of freedom
    def _dynamic_adjustchgthresh(peek, defpeek, defthresh):
        thresh = defthresh
        if peek > defpeek:
            pt_cg = 1.0 - (1.0 - 0.99) ** (defpeek / peek)
            # In single-index NDVI mode, df = 1
            thresh = chi2.ppf(pt_cg, 1)
        return thresh
    ccd.procedures.adjustchgthresh = _dynamic_adjustchgthresh

    # 3. Filter saturated adjustment for NDVI slot 3
    def _patched_filter_saturated(obs):
        unsaturated = (
            (0 < obs[1, :]) & (obs[1, :] < 10000) &
            (0 < obs[2, :]) & (obs[2, :] < 10000) &
            (-10000 <= obs[3, :]) & (obs[3, :] <= 10000) &
            (0 < obs[4, :]) & (obs[4, :] < 10000) &
            (0 < obs[5, :]) & (obs[5, :] < 10000) &
            (0 < obs[0, :]) & (obs[0, :] < 10000)
        )
        return unsaturated
    ccd.qa.filter_saturated = _patched_filter_saturated

    # 4. Safe coordinate descent wrapper
    from sklearn import linear_model
    from ccd.models import FittedModel
    from ccd.math_utils import calc_rmse

    def _safe_fitted_model(dates, spectra_obs, max_iter, avg_days_yr, num_coefficients):
        coef_matrix = ccd.models.lasso.coefficient_matrix(dates, avg_days_yr, num_coefficients)

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
OUTPUT_GROUP = "PYCCD_NDVI_Results"

MIN_CLEAR_SAMPLES = 12
NOMINAL_THERMAL_KELVIN_SCALED = 2931

# Scaled parameters targeting a ~20-25 GB memory footprint (within 32 GB RAM budget)
BLOCK_SIZE_ROWS = 192        # Streaming block size written directly to HDF5
WORKER_CHUNK_ROWS = 16       # Row chunk per parallel worker task
PRELOAD_INPUT = True         # Preload 10.6 GB Landsat inputs into RAM to eliminate disk I/O
DEFAULT_N_JOBS = max(1, min(12, multiprocessing.cpu_count() - 4))


def check_ram_capacity():
    """Detect system RAM to determine capacity."""
    try:
        import psutil
        vm = psutil.virtual_memory()
        return vm.total / (1024 ** 3), vm.available / (1024 ** 3)
    except Exception:
        return 64.0, 32.0


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


def decode_categorical_qa(fmask_arr, common_mask_arr, sr_blue):
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
    is_fill = (fmask_arr == 255) | (sr_blue == -32768)
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
    qas[is_cloud | (common_mask_arr & ~is_shadow & ~is_snow & ~is_fill)] = 5
    # 6. Fill = 0 (Fill beats all)
    qas[is_fill] = 0

    return qas


def _process_row_subchunk_ndvi(subchunk_args):
    """Worker function for processing a row subchunk with pyccd strictly on NDVI."""
    patch_pyccd()

    (sub_y0, sub_y1, width,
     sr_sub, fm_sub, cm_sub,
     ordinal_dates, acq_times, X_full,
     save_coef, min_clear_samples) = subchunk_args

    num_frames = sr_sub.shape[0]
    chunk_h = sub_y1 - sub_y0

    sub_pred_ndvi = np.full((num_frames, chunk_h, width), np.nan, dtype=np.float32)
    sub_rmse_ndvi = np.full((num_frames, chunk_h, width), np.nan, dtype=np.float32)
    sub_coef_ndvi = np.full((num_frames, chunk_h, width, 8), np.nan, dtype=np.float32) if save_coef else None
    sub_flags = np.zeros((num_frames, chunk_h, width), dtype=np.uint8)
    sub_date = np.full((chunk_h, width), np.nan, dtype=np.float64)
    sub_count = np.zeros((chunk_h, width), dtype=np.int32)
    sub_water_prob = np.full((chunk_h, width), np.nan, dtype=np.float32)

    nominal_thermals = np.full(num_frames, NOMINAL_THERMAL_KELVIN_SCALED, dtype=np.int16)

    # Single-index NDVI parameters:
    # df = 1: chi2.ppf(0.99, 1) = 6.6348966
    # Outlier: chi2.ppf(0.9999, 1) = 15.136705
    # Detection band: 3 (NDVI routed into NIR slot)
    proc_params = {
        'DETECTION_BANDS': [3],
        'CHANGE_THRESHOLD': 6.6348966,
        'OUTLIER_THRESHOLD': 15.136705,
        'QA_BITPACKED': False,
        'QA_FILL': 0,
        'QA_CLEAR': 1,
        'QA_WATER': 2,
        'QA_SHADOW': 3,
        'QA_SNOW': 4,
        'QA_CLOUD': 5,
        'STAT_ORD': int(np.max(ordinal_dates)) + 1,
        'LASSO_MAX_ITER': 2000
    }

    for y_local in range(chunk_h):
        for x in range(width):
            fm_pixel = fm_sub[:, y_local, x]
            cm_pixel = cm_sub[:, y_local, x]
            sr_blue = sr_sub[:, 1, y_local, x]

            qas_pixel = decode_categorical_qa(fm_pixel, cm_pixel, sr_blue)

            usable_count = int(np.count_nonzero((qas_pixel == 1) | (qas_pixel == 2) | (qas_pixel == 4)))
            if usable_count < min_clear_samples:
                continue

            # Landsat reflectance bands (HLSL30 indices: 1=Blue, 2=Green, 3=Red, 4=NIR, 5=SWIR1, 6=SWIR2)
            blues = sr_sub[:, 1, y_local, x]
            greens = sr_sub[:, 2, y_local, x]
            reds = sr_sub[:, 3, y_local, x]
            nirs = sr_sub[:, 4, y_local, x]
            swir1s = sr_sub[:, 5, y_local, x]
            swir2s = sr_sub[:, 6, y_local, x]

            # Compute NDVI directly on raw surface reflectance integers
            # Unscaled float NDVI: (NIR - Red) / (NIR + Red)
            denom = (nirs.astype(np.float64) + reds.astype(np.float64))
            valid_sr = (blues != -32768) & (reds != -32768) & (nirs != -32768) & (np.abs(denom) > 1e-4)

            raw_ndvi = np.zeros(num_frames, dtype=np.float64)
            raw_ndvi[valid_sr] = (nirs[valid_sr].astype(np.float64) - reds[valid_sr].astype(np.float64)) / denom[valid_sr]
            raw_ndvi = np.clip(raw_ndvi, -1.0, 1.0)

            # Scale NDVI to standard Landsat surface reflectance range [-10000, 10000]
            ndvi_scaled = np.round(raw_ndvi * 10000.0).astype(np.int16)

            try:
                # Slot 3 carries ndvi_scaled; Green and SWIR1 carry real reflectance for Tmask
                res = ccd.detect(
                    ordinal_dates,
                    blues, greens, reds, ndvi_scaled, swir1s, swir2s,
                    nominal_thermals,
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

            ndvi_preds = []
            ndvi_rmses = []
            ndvi_coefs = []

            for m in models:
                # Slot 3 is named 'nir' in pyccd model structure
                b_info = m['nir']
                intercept = float(b_info['intercept'])
                coefs_7 = list(b_info['coefficients'])
                theta_8 = np.array([intercept] + coefs_7, dtype=np.float64)

                pred_scaled = (X_full @ theta_8) / 10000.0
                rmse_scaled = float(b_info['rmse']) / 10000.0

                ndvi_preds.append(pred_scaled)
                ndvi_rmses.append(rmse_scaled)
                if save_coef:
                    ndvi_coefs.append(theta_8 / 10000.0)

            for t_idx in range(num_frames):
                m_active = active_model_indices[t_idx]
                sub_pred_ndvi[t_idx, y_local, x] = ndvi_preds[m_active][t_idx]
                sub_rmse_ndvi[t_idx, y_local, x] = ndvi_rmses[m_active]
                if save_coef:
                    sub_coef_ndvi[t_idx, y_local, x, :] = ndvi_coefs[m_active]

    return (
        sub_y0, sub_y1,
        sub_pred_ndvi, sub_rmse_ndvi, sub_coef_ndvi, sub_flags,
        sub_date, sub_count,
        sub_water_prob
    )


def main(
    location=LOCATION,
    h5_path=None,
    output_h5=None,
    output_group=OUTPUT_GROUP,
    save_coef=True,
    min_clear_samples=MIN_CLEAR_SAMPLES,
    block_size=BLOCK_SIZE_ROWS,
    worker_chunk_rows=WORKER_CHUNK_ROWS,
    preload_input=PRELOAD_INPUT,
    n_jobs=None,
    row_start=None,
    row_end=None,
    col_start=None,
    col_end=None,
    resume=True,
    overwrite=False,
    **kwargs,
):
    """
    Executes USGS pyccd single-index NDVI change detection with streaming block writing to HDF5.
    Supports automatic block-level checkpoint resumption.
    """
    patch_pyccd()

    if h5_path is None:
        h5_path = H5_PATH
    if output_h5 is None:
        output_h5 = h5_path
    if n_jobs is None:
        n_jobs = DEFAULT_N_JOBS

    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"Input file not found: {h5_path}")

    total_ram_gb, avail_ram_gb = check_ram_capacity()
    cpu_cores = multiprocessing.cpu_count()

    print("=" * 80)
    print("USGS LCMAP Single-Index NDVI Continuous Change Detection (CCDC) Baseline")
    print(f"HDF5 Input Path    : {h5_path}")
    print(f"HDF5 Output Target : {output_h5} -> Group: {output_group}")
    print(f"Parallel Workers   : {n_jobs} (Host: {cpu_cores} cores | {total_ram_gb:.1f} GB RAM, {avail_ram_gb:.1f} GB free)")
    print(f"Block Streaming    : {block_size} rows/block | Worker Subchunks: {worker_chunk_rows} rows")
    print(f"Preload Inputs     : {preload_input}")
    print(f"Detection Threshold: chi2(df=1, p=0.99) = 6.6348966")
    print(f"Checkpoint Resume  : {resume} (Overwrite: {overwrite})")
    print("=" * 80)

    # Read temporal metadata and grid dimensions from HLSL30
    with h5py.File(h5_path, 'r') as h5_in:
        hlsl30_grp = h5_in['/HDFEOS/GRIDS/HLSL30/Data Fields']
        sr_ds = hlsl30_grp['surface_reflectance']

        acq_times = sr_ds.attrs.get('acquisition_time')
        if acq_times is None:
            acq_times = h5_in['/HDFEOS/GRIDS/HARMONIZED/Data Fields'].attrs.get('acquisition_time')
        acq_times = np.array(acq_times[:], dtype=np.float64)

        geo_transform = sr_ds.attrs.get('GeoTransform', h5_in.attrs.get('GeoTransform'))
        spatial_ref = sr_ds.attrs.get('spatial_ref', h5_in.attrs.get('spatial_ref'))

        full_num_frames, num_bands, full_height, full_width = sr_ds.shape

        y0 = 0 if row_start is None else max(0, row_start)
        y1 = full_height if row_end is None else min(full_height, row_end)
        x0 = 0 if col_start is None else max(0, col_start)
        x1 = full_width if col_end is None else min(full_width, col_end)

        height = y1 - y0
        width = x1 - x0

    num_landsat = len(acq_times)
    print(f"Landsat Observations          : {num_landsat} frames (Grid: {full_height}x{full_width})")
    print(f"Active Processing Window      : Rows [{y0}:{y1}], Cols [{x0}:{x1}] ({height}x{width})")

    # Ordinal dates and harmonic Fourier design matrix
    ordinal_dates = np.array([
        datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc).date().toordinal()
        for ts in acq_times
    ], dtype=np.int64)

    X_full = build_harmonic_basis(ordinal_dates)

    # Preload input data into RAM if enabled
    sr_preloaded = None
    fm_preloaded = None
    cm_preloaded = None

    if preload_input:
        t_load0 = time.time()
        print(f"\n[Preload] Loading Landsat inputs into RAM for rows {y0} to {y1}...")
        with h5py.File(h5_path, 'r') as h5_in:
            hlsl30_grp = h5_in['/HDFEOS/GRIDS/HLSL30/Data Fields']
            sr_preloaded = np.array(hlsl30_grp['surface_reflectance'][:, :, y0:y1, x0:x1], dtype=np.int16)
            fm_preloaded = np.array(hlsl30_grp['Fmask'][:, y0:y1, x0:x1], dtype=np.uint8)
            cm_preloaded = np.array(hlsl30_grp['common_mask'][:, y0:y1, x0:x1], dtype=bool)
        mem_mb = (sr_preloaded.nbytes + fm_preloaded.nbytes + cm_preloaded.nbytes) / (1024 ** 2)
        print(f"[Preload] Successfully loaded {mem_mb:.1f} MB into RAM in {time.time() - t_load0:.2f} s")
    else:
        print("Preloading disabled: streaming blocks on-demand from disk.")

    # Check and initialize destination HDF5 datasets
    needs_creation = True
    with h5py.File(output_h5, 'a') as out_file:
        if output_group in out_file:
            if overwrite or not resume:
                del out_file[output_group]
            else:
                existing_grp = out_file[output_group]
                exp_3d = (num_landsat, height, width)
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

            ccd_grp.attrs['ALGORITHM'] = 'lcmap-pyccd-ndvi'
            ccd_grp.attrs['TARGET_BAND'] = 'ndvi'
            ccd_grp.attrs['DETECTION_BANDS'] = np.array([3], dtype=np.int32)
            ccd_grp.attrs['DEGREES_OF_FREEDOM'] = 1
            ccd_grp.attrs['CHANGE_THRESHOLD'] = 6.6348966
            ccd_grp.attrs['OUTLIER_THRESHOLD'] = 15.136705
            ccd_grp.attrs['SOURCE_DATA'] = h5_path
            ccd_grp.attrs['SOURCE_GRID'] = '/HDFEOS/GRIDS/HLSL30'
            ccd_grp.attrs['NUM_LANDSAT_FRAMES'] = num_landsat
            ccd_grp.attrs['MIN_CLEAR_SAMPLES'] = min_clear_samples
            ccd_grp.attrs['description'] = "USGS LCMAP pyccd CCDC Baseline executed strictly on Landsat NDVI"
            ccd_grp.attrs['created_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()

            if (height, width) != (full_height, full_width):
                ccd_grp.attrs['ROI_ROWS'] = [y0, y1]
                ccd_grp.attrs['ROI_COLS'] = [x0, x1]

            chunk_3d = (min(64, num_landsat), min(16, height), min(64, width))
            ccd_grp.create_dataset('predicted_series', shape=(num_landsat, height, width), dtype='float32', chunks=chunk_3d, compression='gzip', compression_opts=6, shuffle=True)
            ccd_grp.create_dataset('rmse_series', shape=(num_landsat, height, width), dtype='float32', chunks=chunk_3d, compression='gzip', compression_opts=6, shuffle=True)
            ccd_grp.create_dataset('anomaly_flags', shape=(num_landsat, height, width), dtype='uint8', chunks=chunk_3d, compression='gzip', compression_opts=6, shuffle=True)
            ccd_grp.create_dataset('change_date_timestamp', shape=(height, width), dtype='float64', chunks=True, compression='gzip', compression_opts=6, shuffle=True)
            ccd_grp.create_dataset('change_count', shape=(height, width), dtype='int32', chunks=True, compression='gzip', compression_opts=6, shuffle=True)
            ccd_grp.create_dataset('water_probability', shape=(height, width), dtype='float32', chunks=True, compression='gzip', compression_opts=6, shuffle=True)
            ccd_grp['water_probability'].attrs['description'] = 'LCMAP CCDC water probability: ratio of water observations to total clear and water observations'

            if save_coef:
                chunk_4d = (min(32, num_landsat), min(8, height), min(32, width), 8)
                ccd_grp.create_dataset('coef_series', shape=(num_landsat, height, width, 8), dtype='float32', chunks=chunk_4d, compression='gzip', compression_opts=6, shuffle=True)

            # Soft links for aliases
            ccd_grp['predicted_ndvi'] = h5py.SoftLink(f'/{output_group}/predicted_series')
            ccd_grp['rmse_ndvi'] = h5py.SoftLink(f'/{output_group}/rmse_series')
            if save_coef:
                ccd_grp['coef_ndvi'] = h5py.SoftLink(f'/{output_group}/coef_series')

    # Block-by-block processing and streaming write
    total_blocks = math.ceil(height / block_size)
    total_changed_pixels = 0
    t_global_start = time.time()
    print(f"\nProcessing {total_blocks} streaming spatial blocks with {n_jobs} parallel workers...\n")

    with h5py.File(output_h5, 'a') as out_file:
        ccd_grp = out_file[output_group]
        ds_pred = ccd_grp['predicted_series']
        ds_rmse = ccd_grp['rmse_series']
        ds_flags = ccd_grp['anomaly_flags']
        ds_date = ccd_grp['change_date_timestamp']
        ds_count = ccd_grp['change_count']
        ds_water = ccd_grp['water_probability']
        ds_coef = ccd_grp['coef_series'] if save_coef else None

        for b_idx in range(total_blocks):
            b_y0 = b_idx * block_size
            b_y1 = min(b_y0 + block_size, height)
            b_height = b_y1 - b_y0
            t_block0 = time.time()

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
                    sr_block = hlsl30_grp['surface_reflectance'][:, :, (y0 + b_y0):(y0 + b_y1), x0:x1]
                    fm_block = hlsl30_grp['Fmask'][:, (y0 + b_y0):(y0 + b_y1), x0:x1]
                    cm_block = hlsl30_grp['common_mask'][:, (y0 + b_y0):(y0 + b_y1), x0:x1]

            # Allocate block buffers
            block_pred = np.full((num_landsat, b_height, width), np.nan, dtype=np.float32)
            block_rmse = np.full((num_landsat, b_height, width), np.nan, dtype=np.float32)
            block_coef = np.full((num_landsat, b_height, width, 8), np.nan, dtype=np.float32) if save_coef else None
            block_flags = np.zeros((num_landsat, b_height, width), dtype=np.uint8)
            block_date = np.full((b_height, width), np.nan, dtype=np.float64)
            block_count = np.zeros((b_height, width), dtype=np.int32)
            block_water = np.full((b_height, width), np.nan, dtype=np.float32)

            subchunks = []
            for sub_y0 in range(0, b_height, worker_chunk_rows):
                sub_y1 = min(sub_y0 + worker_chunk_rows, b_height)
                sub_args = (
                    sub_y0, sub_y1, width,
                    np.ascontiguousarray(sr_block[:, :, sub_y0:sub_y1, :]),
                    np.ascontiguousarray(fm_block[:, sub_y0:sub_y1, :]),
                    np.ascontiguousarray(cm_block[:, sub_y0:sub_y1, :]),
                    ordinal_dates, acq_times, X_full,
                    save_coef, min_clear_samples
                )
                subchunks.append(sub_args)

            if n_jobs > 1 and len(subchunks) > 1:
                with tqdm_joblib(tqdm(desc=f"Block {b_idx + 1:2d}/{total_blocks}", total=len(subchunks), unit="chunk", leave=False)):
                    worker_results = Parallel(n_jobs=n_jobs, backend='loky')(
                        delayed(_process_row_subchunk_ndvi)(sc) for sc in subchunks
                    )
            else:
                worker_results = [_process_row_subchunk_ndvi(sc) for sc in subchunks]

            # Assemble subchunks into block
            for (sub_y0, sub_y1, s_pred, s_rmse, s_coef, s_flags, s_date, s_count, s_water) in worker_results:
                block_pred[:, sub_y0:sub_y1, :] = s_pred
                block_rmse[:, sub_y0:sub_y1, :] = s_rmse
                block_flags[:, sub_y0:sub_y1, :] = s_flags
                block_date[sub_y0:sub_y1, :] = s_date
                block_count[sub_y0:sub_y1, :] = s_count
                block_water[sub_y0:sub_y1, :] = s_water
                if save_coef and s_coef is not None:
                    block_coef[:, sub_y0:sub_y1, :, :] = s_coef

            # Stream write directly into HDF5 datasets
            ds_pred[:, b_y0:b_y1, :] = block_pred
            ds_rmse[:, b_y0:b_y1, :] = block_rmse
            ds_flags[:, b_y0:b_y1, :] = block_flags
            ds_date[b_y0:b_y1, :] = block_date
            ds_count[b_y0:b_y1, :] = block_count
            ds_water[b_y0:b_y1, :] = block_water
            if save_coef:
                ds_coef[:, b_y0:b_y1, :, :] = block_coef

            out_file.flush()

            b_changed = int(np.count_nonzero(~np.isnan(block_date)))
            total_changed_pixels += b_changed
            elapsed_block = time.time() - t_block0
            elapsed_total = time.time() - t_global_start
            pct_complete = (b_y1 / height) * 100.0
            print(f"Block {b_idx + 1:2d}/{total_blocks} (rows {b_y0:4d}:{b_y1:4d}) complete ({pct_complete:5.1f}%) | "
                  f"Changed pixels: {b_changed:4d} | Block time: {elapsed_block:.1f}s ({b_height / max(1e-3, elapsed_block):.1f} rows/s) | Total: {elapsed_total/60:.1f}m")

            del worker_results, subchunks
            del block_pred, block_rmse, block_flags, block_date, block_count, block_water
            if not preload_input:
                del sr_block, fm_block, cm_block
            if save_coef:
                del block_coef
            gc.collect()

            # Recycle worker processes between blocks to eliminate C-level memory corruption and fragmentation
            try:
                from joblib.externals.loky import get_reusable_executor
                get_reusable_executor().shutdown(wait=True)
            except Exception:
                pass

        ccd_grp.attrs['ELAPSED_SECONDS'] = float(time.time() - t_global_start)

    if preload_input:
        del sr_preloaded, fm_preloaded, cm_preloaded
        gc.collect()

    total_elapsed = time.time() - t_global_start
    print("\n" + "=" * 80)
    print("USGS LCMAP Single-Index NDVI Change Detection Complete!")
    print(f"Total Elapsed Time : {total_elapsed:.2f} s ({total_elapsed / 60:.2f} min)")
    print(f"Total Changed Pixels: {total_changed_pixels}")
    print(f"Results Saved To   : {output_h5} -> Group: {output_group}")
    print("=" * 80)
    return output_h5


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Run Memory-Efficient Streaming USGS PyCCD Single-Index NDVI Baseline")
    parser.add_argument('--location', type=str, default=LOCATION, help="Location name")
    parser.add_argument('--h5-path', type=str, default=None, help="Input source HDF5 path")
    parser.add_argument('--output-h5', type=str, default=None, help="Output HDF5 path")
    parser.add_argument('--group', type=str, default=OUTPUT_GROUP, help="Output HDF5 group name")
    parser.add_argument('--no-coef', action='store_true', help="Skip storing 8 harmonic coefficients series to save disk space")
    parser.add_argument('--block-size', type=int, default=BLOCK_SIZE_ROWS, help="Rows per streaming block written to HDF5")
    parser.add_argument('--worker-chunk-rows', type=int, default=WORKER_CHUNK_ROWS, help="Rows per worker task")
    parser.add_argument('--preload-input', action='store_true', default=PRELOAD_INPUT, help="Preload all Landsat inputs into RAM to eliminate disk I/O")
    parser.add_argument('--no-preload-input', action='store_false', dest='preload_input', help="Stream blocks on-demand from disk (RAM-conserving mode)")
    parser.add_argument('--n-jobs', type=int, default=DEFAULT_N_JOBS, help="Number of parallel worker processes")
    parser.add_argument('--row-start', type=int, default=None, help="Optional starting row")
    parser.add_argument('--row-end', type=int, default=None, help="Optional ending row")
    parser.add_argument('--col-start', type=int, default=None, help="Optional starting col")
    parser.add_argument('--col-end', type=int, default=None, help="Optional ending col")
    parser.add_argument('--overwrite', action='store_true', help="Overwrite existing output group instead of resuming")
    parser.add_argument('--no-resume', action='store_false', dest='resume', help="Disable automatic block resumption")
    args = parser.parse_args()

    main(
        location=args.location,
        h5_path=args.h5_path,
        output_h5=args.output_h5,
        output_group=args.group,
        save_coef=not args.no_coef,
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
