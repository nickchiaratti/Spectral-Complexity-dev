import os
import h5py
import numpy as np
import pandas as pd
import datetime
import math
from tqdm import tqdm
import multiprocessing
import joblib
import contextlib
from joblib import Parallel, delayed

from adtk.data import validate_series
from adtk.detector import InterQuartileRangeAD, LevelShiftAD

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

# ==========================================
# 1. CONFIGURATION
# ==========================================
LOCATION = "Tait"
H5_PATH = f"C:/satelliteImagery/HLST30/HLST_{LOCATION}_Harmonized_SC_EM-7_Norm-None.h5"
TARGET_METRIC = 'sliding_volume_z_score'

# ADTK specific parameters
MIN_SAMPLES = 20
IQR_C = 3.0
LS_C = 2.0
LS_WINDOW = 15
# If anomalies occur within this many days of each other, group them into a single event
EVENT_GROUPING_DAYS = 30


def _process_row_chunk(chunk_args):
    (y_start, y_end, width, y_data, valid_mask, acq_times,
     iqr_c, ls_c, ls_window, min_samples, event_grouping_days) = chunk_args

    num_frames = y_data.shape[0]
    chunk_height = y_end - y_start

    # Output arrays
    chunk_pred = np.full((num_frames, chunk_height, width), np.nan, dtype=np.float32)
    chunk_rmse = np.full((num_frames, chunk_height, width), np.nan, dtype=np.float32)
    chunk_flags = np.zeros((num_frames, chunk_height, width), dtype=np.uint8)
    chunk_date = np.full((chunk_height, width), np.nan, dtype=np.float64)
    chunk_count = np.zeros((chunk_height, width), dtype=np.int32)

    for y_local in range(chunk_height):
        y_global = y_start + y_local
        for x in range(width):
            pixel_valid = valid_mask[:, y_global, x]
            valid_indices = np.where(pixel_valid)[0]

            if len(valid_indices) < min_samples:
                continue

            # Extract valid signal and corresponding timestamps
            signal = y_data[valid_indices, y_global, x].astype(np.float64)
            signal_acq_times = acq_times[valid_indices]
            
            dates = pd.to_datetime(signal_acq_times, unit='s')
            s = pd.Series(signal, index=dates)
            
            try:
                s = validate_series(s)
                # Apply IQR anomaly detector
                iqr_ad = InterQuartileRangeAD(c=iqr_c)
                anomalies_iqr = iqr_ad.fit_detect(s)
                
                # Apply LevelShift detector
                ls_ad = LevelShiftAD(c=ls_c, side='both', window=ls_window)
                anomalies_ls = ls_ad.fit_detect(s)
                
                # Combine flags
                anomalies = anomalies_iqr | anomalies_ls
            except Exception:
                continue
                
            # If nothing flagged or all NaNs, skip
            if anomalies.sum() == 0 or anomalies.isna().all():
                continue
                
            # Find the indices of flagged anomalies
            anom_series_indices = np.where(anomalies == True)[0]
            
            # Map back to original frame indices and flag them
            for idx in anom_series_indices:
                frame_idx = valid_indices[idx]
                chunk_flags[frame_idx, y_local, x] = 1

            # Group anomalies into events
            events = []
            current_event = []
            
            for idx in anom_series_indices:
                ts = signal_acq_times[idx]
                if not current_event:
                    current_event = [ts]
                else:
                    # Check if within grouping window of the previous anomaly in this event
                    last_ts = current_event[-1]
                    if (ts - last_ts) <= (event_grouping_days * 86400):
                        current_event.append(ts)
                    else:
                        events.append(current_event)
                        current_event = [ts]
            
            if current_event:
                events.append(current_event)
                
            change_count_val = len(events)
            chunk_count[y_local, x] = change_count_val
            
            if change_count_val > 0:
                # We will record the start date of the first event as the change date
                chunk_date[y_local, x] = events[0][0]

    return y_start, y_end, chunk_pred, chunk_rmse, chunk_flags, chunk_date, chunk_count


def main():
    import warnings
    warnings.filterwarnings('ignore', category=FutureWarning)

    output_h5 = f"C:/satelliteImagery/HLST30/CCD/{LOCATION}_CCD_adtk_Change_Detection_{TARGET_METRIC}.h5"

    print(f"Loading data from {H5_PATH}...")
    with h5py.File(H5_PATH, 'r') as f:
        data_grp = f['/HDFEOS/GRIDS/HARMONIZED/Data Fields']
        metric_ds = data_grp[TARGET_METRIC]

        acq_times = metric_ds.attrs['acquisition_time'][:]
        y_data = metric_ds[...]

        common_mask = data_grp['common_mask'][...]
        valid_mask = (common_mask == 0) & ~np.isnan(y_data)

        geo_transform = metric_ds.attrs.get('GeoTransform')
        spatial_ref = metric_ds.attrs.get('spatial_ref')

    num_frames, height, width = y_data.shape

    # Sort chronologically
    sort_idx = np.argsort(acq_times)
    acq_times = acq_times[sort_idx]
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

    print(f"\\nExecuting ADTK detection (IQR_C={IQR_C}, LS_C={LS_C}, LS_WINDOW={LS_WINDOW})...")

    n_jobs = multiprocessing.cpu_count()
    print(f"Using {n_jobs} cores for parallel processing.")

    num_chunks = max(1, n_jobs * 4)
    chunk_size = max(1, math.ceil(height / num_chunks))

    chunks = []
    for y_start in range(0, height, chunk_size):
        y_end = min(y_start + chunk_size, height)
        chunk_args = (
            y_start, y_end, width,
            y_data, valid_mask,
            acq_times,
            IQR_C, LS_C, LS_WINDOW, MIN_SAMPLES, EVENT_GROUPING_DAYS
        )
        chunks.append(chunk_args)

    with tqdm_joblib(tqdm(desc="Processing row chunks", total=len(chunks))):
        results = Parallel(n_jobs=n_jobs, backend='loky')(
            delayed(_process_row_chunk)(chunk) for chunk in chunks
        )

    for y_start, y_end, c_pred, c_rmse, c_flags, c_date, c_count in results:
        predicted_series[:, y_start:y_end, :] = c_pred
        rmse_series[:, y_start:y_end, :] = c_rmse
        anomaly_flags[:, y_start:y_end, :] = c_flags
        change_date_map[y_start:y_end, :] = c_date
        change_count_map[y_start:y_end, :] = c_count

    os.makedirs(os.path.dirname(output_h5), exist_ok=True)
    print(f"\\nSaving Results to {output_h5}...")
    with h5py.File(output_h5, 'w') as out_file:
        if spatial_ref is not None:
            out_file.attrs['spatial_ref'] = spatial_ref
        if geo_transform is not None:
            out_file.attrs['GeoTransform'] = geo_transform

        out_file.attrs['RMSE_MULTIPLIER'] = 1.0
        out_file.attrs['MIN_SAMPLES'] = MIN_SAMPLES
        out_file.attrs['TARGET_METRIC'] = TARGET_METRIC
        out_file.attrs['SOURCE_DATA'] = H5_PATH
        out_file.attrs['ADTK_IQR_C'] = IQR_C
        out_file.attrs['ADTK_LS_C'] = LS_C
        out_file.attrs['ADTK_LS_WINDOW'] = LS_WINDOW
        out_file.attrs['ADTK_EVENT_GROUPING_DAYS'] = EVENT_GROUPING_DAYS

        out_file.create_dataset('predicted_series', data=predicted_series, compression='gzip')
        out_file.create_dataset('rmse_series', data=rmse_series, compression='gzip')
        out_file.create_dataset('anomaly_flags', data=anomaly_flags, compression='gzip')
        out_file.create_dataset('change_date_timestamp', data=change_date_map, compression='gzip')
        out_file.create_dataset('change_count', data=change_count_map, compression='gzip')

    print("ADTK Pipeline Complete!")
    return output_h5

if __name__ == '__main__':
    main()
