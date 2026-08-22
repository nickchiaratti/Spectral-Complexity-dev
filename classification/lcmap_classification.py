import os
import re
import sys
import h5py
import numpy as np
from datetime import datetime, timezone

# Ensure usgs_sam_classifier can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from usgs_sam_classifier import USGS_SAM_Classifier

def get_sensor_group(sc_str):
    s = str(sc_str).upper()
    if 'LANDSAT' in s: return 'Landsat'
    if 'SENTINEL' in s: return 'Sentinel'
    if 'TANAGER' in s: return 'Tanager'
    return s

def extract_lcmap_time_series(pixel_y, pixel_x, source_h5_path, inference_results_h5, lcmap_classes=None):
    """
    Extracts surface reflectance patches and classifies them using USGS_SAM_Classifier.
    Returns:
        tuple: (segments, sensors, extracted_data)
        extracted_data is a nested dict:
        extracted_data[row_idx][col_idx] = {
            'dates': [datetime, datetime, ...],
            'class_counts': {
                'Water': [0.1, 0.2, ...],
                ...
            },
            'start_acq_time': int,
            'end_acq_time': int
        }
    """
    if lcmap_classes is None:
        lcmap_classes = ['Developed', 'Cropland', 'Grass/Shrub', 'Tree Cover', 'Water', 'Wetland', 'Ice/Snow', 'Barren']

    with h5py.File(inference_results_h5, 'r') as f_inf:
        rmse = f_inf['rmse_series'][:, pixel_y, pixel_x]
        
    with h5py.File(source_h5_path, 'r') as f_sc:
        harm_grp = f_sc['/HDFEOS/GRIDS/HARMONIZED/Data Fields']
        target_metric = f_sc.attrs.get('TARGET_METRIC', 'sliding_volume_z_score')
        if target_metric not in harm_grp:
            target_metric = list(harm_grp.keys())[0]
            
        attrs = harm_grp[target_metric].attrs
        acq_time = attrs['acquisition_time'][:]
        source_grids = [s.decode('utf-8') if isinstance(s, bytes) else str(s) for s in attrs['source_grid'][:]]
        source_frames = attrs['source_frame_index'][:]
        spacecrafts = [s.decode('utf-8') if isinstance(s, bytes) else str(s) for s in attrs['source_spacecraft'][:]]
        unified_masks = harm_grp['common_mask'][:, pixel_y, pixel_x]
        
        _, H, W = harm_grp['common_mask'].shape

    y_start = max(0, pixel_y - 1)
    y_end = min(H, pixel_y + 2)
    x_start = max(0, pixel_x - 1)
    x_end = min(W, pixel_x + 2)
    
    segments = []
    current_segment = []
    current_rmse = None
    for i in range(len(acq_time)):
        if unified_masks[i]: continue
        r = rmse[i]
        
        if len(current_segment) == 0:
            current_rmse = r
            current_segment.append(i)
        elif (np.isnan(r) and np.isnan(current_rmse)):
            current_segment.append(i)
        elif (not np.isnan(r) and not np.isnan(current_rmse) and abs(r - current_rmse) < 1e-6):
            current_segment.append(i)
        else:
            segments.append(current_segment)
            current_segment = [i]
            current_rmse = r
    if current_segment:
        segments.append(current_segment)
        
    if not segments:
        return [], [], {}
        
    raw_h5_path = re.sub(r'_SC_.*?\.h5$', '.h5', source_h5_path)
    
    present_sensors = set()
    for seg in segments:
        for i in seg:
            present_sensors.add(get_sensor_group(spacecrafts[i]))
    sensors = sorted([s for s in present_sensors if 'Landsat' in s or 'Sentinel' in s])
    
    if not sensors:
        return segments, [], {}

    # Load compiled classifiers
    classifiers = {}
    compiled_base = r'C:\satelliteImagery\ground_truth\splib07'
    for s in sensors:
        s_lower = s.lower()
        npz_name = f'usgs_{s_lower}_compiled.npz'
        npz_path = os.path.join(compiled_base, npz_name)
        if os.path.exists(npz_path):
            classifiers[s] = USGS_SAM_Classifier(npz_path)
        else:
            print(f'Warning: Compiled library not found for {s} at {npz_path}.')

    extracted_data = {row_idx: {} for row_idx in range(len(sensors))}
    
    with h5py.File(raw_h5_path, 'r') as f_raw:
        for row_idx, sensor in enumerate(sensors):
            if sensor not in classifiers:
                continue
            clf = classifiers[sensor]
            
            for col_idx, seg in enumerate(segments):
                dates = []
                class_counts = {c: [] for c in lcmap_classes}
                
                for i in seg:
                    grid = source_grids[i]
                    frame_idx = source_frames[i]
                    s = get_sensor_group(spacecrafts[i])
                    
                    if s != sensor: continue
                    
                    sr_ds = f_raw[f'/HDFEOS/GRIDS/{grid}/Data Fields/surface_reflectance']
                    patch = sr_ds[frame_idx, :, y_start:y_end, x_start:x_end]
                    
                    C, H_p, W_p = patch.shape
                    pixels = patch.reshape(C, -1).T
                    
                    if "Landsat" in sensor:
                        data_indices = [1, 2, 3, 4, 5, 6]
                    elif "Sentinel" in sensor:
                        data_indices = [1, 2, 3, 7, 8, 9]
                    else:
                        data_indices = slice(None)
                        
                    preds = clf.classify_pixels(pixels[:, data_indices])
                    
                    counts = {c: 0 for c in lcmap_classes}
                    total_valid = 0
                    for p in preds:
                        if p in counts:
                            counts[p] += 1
                            total_valid += 1
                            
                    if total_valid > 0:
                        for c in lcmap_classes:
                            class_counts[c].append(counts[c] / total_valid)
                        dates.append(datetime.fromtimestamp(acq_time[i], timezone.utc))
                        
                extracted_data[row_idx][col_idx] = {
                    'dates': dates,
                    'class_counts': class_counts,
                    'start_acq_time': acq_time[seg[0]],
                    'end_acq_time': acq_time[seg[-1]]
                }
                
    return segments, sensors, extracted_data
