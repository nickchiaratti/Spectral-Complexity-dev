import os
import h5py
import numpy as np
import datetime
import math
from tqdm import tqdm
import multiprocessing
import joblib
import contextlib
from joblib import Parallel, delayed
import ruptures

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
# PELT penalty: higher = fewer breakpoints (more conservative), lower = more breakpoints.
# For z-score data in the range [-5, 5], pen=10 is a moderate starting point.
PENALTY = 10
# Minimum number of valid observations required before running PELT on a pixel.
MIN_SAMPLES = 20
# Cost model: 'rbf' detects shifts in full distribution (mean + variance),
# 'l2' detects only mean shifts.
COST_MODEL = 'rbf'
# Minimum segment length between breakpoints (in number of observations).
MIN_SEGMENT_LENGTH = 4


def _process_row_chunk(chunk_args):
    (y_start, y_end, width, y_data, valid_mask, acq_times,
     penalty, min_samples, cost_model, min_seg_len) = chunk_args

    num_frames = y_data.shape[0]
    chunk_height = y_end - y_start

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

            # Extract only valid observations for PELT
            signal = y_data[valid_indices, y_global, x].astype(np.float64)

            try:
                algo = ruptures.Pelt(model=cost_model, min_size=min_seg_len).fit(signal)
                breakpoints = algo.predict(pen=penalty)
            except Exception:
                continue

            # breakpoints from ruptures are 1-indexed positions in the valid signal,
            # with the last element always being len(signal) (end of series).
            # Remove the terminal index — it is not a structural break.
            break_indices_in_valid = [bp for bp in breakpoints if bp < len(valid_indices)]

            change_count_val = len(break_indices_in_valid)
            chunk_count[y_local, x] = change_count_val

            if change_count_val > 0:
                # Map the first breakpoint back to the original frame index
                first_break_frame_idx = valid_indices[break_indices_in_valid[0]]
                chunk_date[y_local, x] = acq_times[first_break_frame_idx]

                # Flag all breakpoints in the anomaly_flags array
                for bp in break_indices_in_valid:
                    frame_idx = valid_indices[bp]
                    chunk_flags[frame_idx, y_local, x] = 1

            # Compute piecewise segment means and standard deviations for visualization.
            # Segments are defined by [0, bp1), [bp1, bp2), ..., [bpN, end).
            segment_bounds = [0] + break_indices_in_valid + [len(valid_indices)]
            for seg_i in range(len(segment_bounds) - 1):
                seg_start = segment_bounds[seg_i]
                seg_end = segment_bounds[seg_i + 1]
                seg_vals = signal[seg_start:seg_end]

                if len(seg_vals) == 0:
                    continue

                seg_mean = np.mean(seg_vals)
                seg_std = np.std(seg_vals) if len(seg_vals) > 1 else 0.0

                # Map back to original frame indices
                for k in range(seg_start, seg_end):
                    frame_idx = valid_indices[k]
                    chunk_pred[frame_idx, y_local, x] = seg_mean
                    chunk_rmse[frame_idx, y_local, x] = seg_std

    return y_start, y_end, chunk_pred, chunk_rmse, chunk_flags, chunk_date, chunk_count


def main():
    output_h5 = f"C:/satelliteImagery/HLST30/CCD/{LOCATION}_CCD_ruptures_Change_Detection_{TARGET_METRIC}.h5"

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

    print("\nExecuting ruptures PELT change point detection...")

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
            PENALTY, MIN_SAMPLES, COST_MODEL, MIN_SEGMENT_LENGTH
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
    print(f"\nSaving Results to {output_h5}...")
    with h5py.File(output_h5, 'w') as out_file:
        if spatial_ref is not None:
            out_file.attrs['spatial_ref'] = spatial_ref
        if geo_transform is not None:
            out_file.attrs['GeoTransform'] = geo_transform

        out_file.attrs['RMSE_MULTIPLIER'] = 1.0
        out_file.attrs['MIN_SAMPLES'] = MIN_SAMPLES
        out_file.attrs['TARGET_METRIC'] = TARGET_METRIC
        out_file.attrs['SOURCE_DATA'] = H5_PATH
        out_file.attrs['PENALTY'] = PENALTY
        out_file.attrs['COST_MODEL'] = COST_MODEL
        out_file.attrs['MIN_SEGMENT_LENGTH'] = MIN_SEGMENT_LENGTH

        out_file.create_dataset('predicted_series', data=predicted_series, compression='gzip')
        out_file.create_dataset('rmse_series', data=rmse_series, compression='gzip')
        out_file.create_dataset('anomaly_flags', data=anomaly_flags, compression='gzip')
        out_file.create_dataset('change_date_timestamp', data=change_date_map, compression='gzip')
        out_file.create_dataset('change_count', data=change_count_map, compression='gzip')

    print("Ruptures PELT Pipeline Complete!")
    return output_h5

if __name__ == "__main__":
    main()
