"""
Harmonized Continuous Change Detection (CCD) Visualization & Diagnostic Analysis.
"""


import sys
from pathlib import Path
_script_dir = Path(__file__).resolve().parent
_repo_root = _script_dir.parent.parent
for _p in [_repo_root, _script_dir, _script_dir.parent / "pyccd"]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import os
import re
import glob
import h5py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button
from datetime import datetime, timezone
import pyproj
from scipy.stats import chi2
from scipy.optimize import linear_sum_assignment

from SC_CCM_main import (
    LOCATION, H5_PATH, ENABLE_CONSTANT, ENABLE_LINEAR, ENABLE_QUADRATIC,
    TEMPORAL_PERIODS, TARGET_METRIC, TARGET_NAME
)

_term_str = f"C{int(ENABLE_CONSTANT)}L{int(ENABLE_LINEAR)}Q{int(ENABLE_QUADRATIC)}"
_period_str = f"P{len(TEMPORAL_PERIODS)}"
CONFIG = f"{_term_str}_{_period_str}"


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


def get_inference_h5(location, config, target_metric):
    """Finds latest inference results HDF5 matching pattern."""
    search_patterns = [
        f"C:/satelliteImagery/HLST30/CCD/{location}_CCD_Harmonized_Change_Detection_{target_metric}_{config}.h5",
        f"C:/satelliteImagery/HLST30/CCD/*{location}*{target_metric}*.h5",
    ]
    for pattern in search_patterns:
        files = glob.glob(pattern)
        if files:
            files.sort(key=os.path.getmtime, reverse=True)
            return files[0]
    return None


def get_default_png_name(location=LOCATION, target_metric=TARGET_METRIC, injected=False):
    """Generates standard diagnostic PNG filename matching interface contract."""
    prefix = "injected_" if injected else ""
    return f"{location}_CCD_{prefix}{target_metric}_anomaly_overlay.png"


def resolve_metric_and_group(target_metric, injected=False):
    """Resolves (group_name, dataset_key, display_name, is_injected) from user metric argument."""
    m_str = str(target_metric).lower().strip()
    is_inj = injected or m_str.startswith('injected_') or 'injected' in m_str
    base = m_str.replace('injected_', '').replace('injected', '').strip('_')

    if base in ('sc', 'sliding_volume_z_score'):
        short = 'SC'
        ds_key = 'injected_sliding_volume_z_score' if is_inj else 'sliding_volume_z_score'
        disp = 'Spectral Complexity (Z-Score)'
    elif base in ('ndvi', 'ndvi_map'):
        short = 'NDVI'
        ds_key = 'injected_ndvi' if is_inj else 'ndvi'
        disp = 'NDVI'
    else:
        short = base.upper()
        ds_key = target_metric
        disp = target_metric

    prefix = f"INJECTED_{short}" if is_inj else short
    group_name = f"{prefix}CCD_Results"
    return group_name, ds_key, disp, is_inj


def _extract_patch_endmembers(patch, num_endmembers=7):
    """Extract geometric simplex endmembers from a patch [H, W, B] using SpecComplexTorch."""
    H, W, B = patch.shape
    pix = patch.reshape(-1, B)
    valid = ~np.isnan(pix).any(axis=1)
    if np.sum(valid) < num_endmembers:
        return np.full((B, num_endmembers), np.nan, dtype=np.float32)
    valid_data = pix[valid].T.astype(np.float32)
    import SpecComplexTorch as sct
    em = sct.maximumDistance(valid_data, num_endmembers)
    if hasattr(em, 'cpu'):
        em = em.detach().cpu().numpy()
    return np.asarray(em, dtype=np.float32)


def plot_pixel_sits(pixel_y, pixel_x, source_h5_path, target_metric='SC', injected=False, ax=None, current_date=None, inference_results_h5=None, **kwargs):
    """Plots pixel-level satellite image time series (SITS) with CCD model fit and confidence bounds."""
    from pathlib import Path
    if inference_results_h5 is None and isinstance(target_metric, (str, Path)) and (str(target_metric).endswith('.h5') or '/' in str(target_metric) or '\\' in str(target_metric)):
        inference_results_h5 = str(target_metric)
        target_metric = 'SC'

    inf_source = inference_results_h5 if (inference_results_h5 and os.path.exists(inference_results_h5)) else source_h5_path
    lat, lon = None, None
    group_name, default_ds, target_name, is_inj = resolve_metric_and_group(target_metric, injected=injected)

    with h5py.File(inf_source, 'r') as f_inf, h5py.File(source_h5_path, 'r') as f:
        if group_name in f_inf:
            ccd_grp = f_inf[group_name]
        elif f"{target_metric}_CCD_Results" in f_inf:
            ccd_grp = f_inf[f"{target_metric}_CCD_Results"]
        elif 'CCD_Results' in f_inf:
            ccd_grp = f_inf['CCD_Results']
        elif 'predicted_series' in f_inf:
            ccd_grp = f_inf
        else:
            raise KeyError(f"Group '{group_name}' not found in {inf_source}. Available: {list(f_inf.keys())}")
        actual_metric = ccd_grp.attrs.get('TARGET_METRIC', default_ds)

        harm_grp = f['/HDFEOS/GRIDS/HARMONIZED/Data Fields']
        if actual_metric in harm_grp:
            metric_ds = harm_grp[actual_metric]
        elif default_ds in harm_grp:
            metric_ds = harm_grp[default_ds]
        elif default_ds == 'ndvi' and 'ndvi_map' in harm_grp:
            metric_ds = harm_grp['ndvi_map']
        else:
            raise KeyError(f"Metric dataset '{actual_metric}' not found in {harm_grp.name}")
        acq_time = metric_ds.attrs.get('acquisition_time')
        if acq_time is None and 'acquisition_time' in harm_grp.attrs:
            acq_time = harm_grp.attrs['acquisition_time']
        elif acq_time is None and 'common_mask' in harm_grp and 'acquisition_time' in harm_grp['common_mask'].attrs:
            acq_time = harm_grp['common_mask'].attrs['acquisition_time']
        elif acq_time is None and 'acquisition_time' in f.attrs:
            acq_time = f.attrs['acquisition_time']

        acq_time = acq_time[:]
        z_score = metric_ds[:, pixel_y, pixel_x]
        if np.issubdtype(z_score.dtype, np.integer):
            z_score = read_scaled_int16(z_score)

        unified_masks = harm_grp['common_mask'][:, pixel_y, pixel_x]
        is_invalid = unified_masks.astype(bool)

        if 'source_spacecraft' in metric_ds.attrs:
            spacecraft_bytes = metric_ds.attrs['source_spacecraft'][:]
            spacecrafts = [s.decode('utf-8') if isinstance(s, bytes) else str(s) for s in spacecraft_bytes]
        elif 'source_spacecraft' in harm_grp['common_mask'].attrs:
            spacecraft_bytes = harm_grp['common_mask'].attrs['source_spacecraft'][:]
            spacecrafts = [s.decode('utf-8') if isinstance(s, bytes) else str(s) for s in spacecraft_bytes]
        else:
            spacecrafts = ["Harmonized"] * len(acq_time)

        geo_transform = metric_ds.attrs.get('GeoTransform', f.attrs.get('GeoTransform'))
        spatial_ref = metric_ds.attrs.get('spatial_ref', f.attrs.get('spatial_ref'))
        if geo_transform is not None and spatial_ref is not None:
            try:
                gt = geo_transform
                x_geo = gt[0] + (pixel_x + 0.5) * gt[1] + (pixel_y + 0.5) * gt[2]
                y_geo = gt[3] + (pixel_x + 0.5) * gt[4] + (pixel_y + 0.5) * gt[5]

                if isinstance(spatial_ref, bytes):
                    spatial_ref_str = spatial_ref.decode('utf-8')
                else:
                    spatial_ref_str = str(spatial_ref)

                crs = pyproj.CRS.from_user_input(spatial_ref_str)
                transformer = pyproj.Transformer.from_crs(crs, "epsg:4326", always_xy=True)
                lon, lat = transformer.transform(x_geo, y_geo)
            except Exception as e:
                print(f"Warning: Could not compute lat/lon: {e}")

        predicted = ccd_grp['predicted_series'][:, pixel_y, pixel_x]
        rmse = ccd_grp['rmse_series'][:, pixel_y, pixel_x]
        anomalies = ccd_grp['anomaly_flags'][:, pixel_y, pixel_x]
        if 'CHANGE_PROBABILITY' in ccd_grp.attrs:
            p = ccd_grp.attrs['CHANGE_PROBABILITY']
            df = ccd_grp.attrs.get('CHI2_DEGREES_OF_FREEDOM', 3)
            rmse_multiplier = np.sqrt(chi2.ppf(p, df=df))
            bound_label = f"Prediction \u00b1{rmse_multiplier:.2f}\u03c3 (p={p}, df={df})"
        else:
            rmse_multiplier = ccd_grp.attrs.get('RMSE_MULTIPLIER', 3.0)
            bound_label = f"Prediction \u00b1{rmse_multiplier}\u03c3"

    dates = [datetime.fromtimestamp(ts, timezone.utc) for ts in acq_time]
        

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
        show_plot = True
    else:
        show_plot = False

    valid_mask = ~is_invalid
    dates_arr = np.array(dates)
    spacecrafts_arr = np.array(spacecrafts)

    # Plot Actuals
    is_anomaly = (anomalies == 1)
    markers = [('s', 'Sentinel'), ('o', 'Landsat'), ('D', 'Tanager')]
    for marker_type, sc_keyword in markers:
        sc_mask = np.array([sc_keyword.lower() in str(sc).lower() for sc in spacecrafts_arr])

        idx_valid_normal = valid_mask & sc_mask & ~is_anomaly
        if np.any(idx_valid_normal):
            ax.plot(dates_arr[idx_valid_normal], z_score[idx_valid_normal], color='k',
                    marker=marker_type, linestyle='None', label=f'Valid ({sc_keyword})')

        idx_valid_anomaly = valid_mask & sc_mask & is_anomaly
        if np.any(idx_valid_anomaly):
            ax.plot(dates_arr[idx_valid_anomaly], z_score[idx_valid_anomaly], color='r',
                    marker=marker_type, linestyle='None', label=f'Anomaly ({sc_keyword})')

        idx_invalid = is_invalid & sc_mask
        if np.any(idx_invalid):
            ax.plot(dates_arr[idx_invalid], z_score[idx_invalid], color='gray',
                    marker=marker_type, linestyle='None', markerfacecolor='none', label=f'Invalid ({sc_keyword})')

    # Plot Predictions
    pred_mask = ~np.isnan(predicted)
    if np.any(pred_mask):
        pred_dates = dates_arr[pred_mask]
        preds = predicted[pred_mask]
        rmses = rmse[pred_mask]

        # Detect segments based on changes in the frozen RMSE
        rmse_diff = np.diff(rmses)
        # Add 1 because diff shifts indices by 1
        break_indices = np.where(rmse_diff != 0)[0] + 1

        date_segments = np.split(pred_dates, break_indices)
        pred_segments = np.split(preds, break_indices)
        rmse_segments = np.split(rmses, break_indices)

        segment_colors = ['blue', 'green', 'purple', 'orange', 'cyan', 'magenta', 'brown']

        for i, (seg_dates, seg_preds, seg_rmses) in enumerate(zip(date_segments, pred_segments, rmse_segments)):
            if len(seg_dates) == 0:
                continue

            c = segment_colors[i % len(segment_colors)]
            upper_bound = seg_preds + rmse_multiplier * seg_rmses
            lower_bound = seg_preds - rmse_multiplier * seg_rmses

            label = 'Harmonic Prediction' if i == 0 else f'Prediction (Seg {i+1})'
            ax.plot(seg_dates, seg_preds, color=c, linestyle='--', label=label)

            fill_label = bound_label if i == 0 else None
            ax.fill_between(seg_dates, lower_bound, upper_bound, color=c, alpha=0.15, label=fill_label)

    # Current date marker
    if current_date is not None:
        ax.axvline(x=current_date, color='orange', linestyle='--', linewidth=2, label='Current Frame')

    title_str = f"SITS for Pixel (y={pixel_y}, x={pixel_x})"
    if lat is not None and lon is not None:
        title_str += f" | Lat: {lat:.5f}\u00b0, Lon: {lon:.5f}\u00b0"
    ax.set_title(title_str)
    ax.set_ylabel(target_name)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))

    # Deduplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)

    if show_plot:
        plt.tight_layout()
        plt.show()


def plot_spatial_anomaly_overlay(source_h5_path, target_metric='SC', injected=False, output_png=None, interactive=True, inference_results_h5=None, **kwargs):
    """
    Renders spatial anomaly frequency and acquisition map over satellite background.
    Supports non-interactive export to PNG without interactive GUI blocking.
    """
    from pathlib import Path
    if inference_results_h5 is None and isinstance(target_metric, (str, Path)) and (str(target_metric).endswith('.h5') or '/' in str(target_metric) or '\\' in str(target_metric)):
        inference_results_h5 = str(target_metric)
        target_metric = 'SC'

    inf_source = inference_results_h5 if (inference_results_h5 and os.path.exists(inference_results_h5)) else source_h5_path
    group_name, default_ds, target_name, is_inj = resolve_metric_and_group(target_metric, injected=injected)

    with h5py.File(inf_source, 'r') as f_inf:
        if group_name in f_inf:
            ccd_grp = f_inf[group_name]
        elif f"{target_metric}_CCD_Results" in f_inf:
            ccd_grp = f_inf[f"{target_metric}_CCD_Results"]
        elif 'CCD_Results' in f_inf:
            ccd_grp = f_inf['CCD_Results']
        elif 'predicted_series' in f_inf or 'change_date_timestamp' in f_inf:
            ccd_grp = f_inf
        else:
            raise KeyError(f"Group '{group_name}' not found in {inf_source}. Available: {list(f_inf.keys())}")
        actual_metric = ccd_grp.attrs.get('TARGET_METRIC', default_ds)
        anomaly_map = ccd_grp['change_date_timestamp'][:]
        change_count_map = ccd_grp['change_count'][:]
        min_samples = ccd_grp.attrs.get('MIN_SAMPLES', 16)

    with h5py.File(source_h5_path, 'r') as f:
        harm_grp = f['/HDFEOS/GRIDS/HARMONIZED/Data Fields']
        if actual_metric in harm_grp:
            metric_ds = harm_grp[actual_metric]
        elif default_ds in harm_grp:
            metric_ds = harm_grp[default_ds]
        elif default_ds == 'ndvi' and 'ndvi_map' in harm_grp:
            metric_ds = harm_grp['ndvi_map']
        else:
            raise KeyError(f"Metric dataset '{actual_metric}' not found in {harm_grp.name}")
        acq_time = metric_ds.attrs.get('acquisition_time')
        if acq_time is None and 'acquisition_time' in harm_grp.attrs:
            acq_time = harm_grp.attrs['acquisition_time']
        elif acq_time is None and 'common_mask' in harm_grp and 'acquisition_time' in harm_grp['common_mask'].attrs:
            acq_time = harm_grp['common_mask'].attrs['acquisition_time']
        elif acq_time is None and 'acquisition_time' in f.attrs:
            acq_time = f.attrs['acquisition_time']
        acq_time = acq_time[:]
        unified_masks = harm_grp['common_mask'][:]
        full_valid_mask = ~unified_masks.astype(bool)

    def get_ortho(idx):
        with h5py.File(source_h5_path, 'r') as f:
            harm_grp = f['/HDFEOS/GRIDS/HARMONIZED/Data Fields']
            if actual_metric in harm_grp and 'source_spacecraft' in harm_grp[actual_metric].attrs:
                spc = harm_grp[actual_metric].attrs['source_spacecraft'][idx]
            elif 'common_mask' in harm_grp and 'source_spacecraft' in harm_grp['common_mask'].attrs:
                spc = harm_grp['common_mask'].attrs['source_spacecraft'][idx]
                spc = spc.decode('utf-8') if isinstance(spc, bytes) else str(spc)
            else:
                spc = "Harmonized"

            if 'ortho_visual' in harm_grp:
                o = harm_grp['ortho_visual'][idx]
                if o.ndim == 3 and o.shape[0] in (3, 4):
                    o = np.transpose(o, (1, 2, 0))
                o = o.astype(np.float32) / 255.0
                valid_mask = np.all(o > 0, axis=-1)
                o[~valid_mask] = 0.0
            else:
                # Synthetic fallback for tests lacking ortho_visual
                if actual_metric in harm_grp:
                    metric_frame = harm_grp[actual_metric][idx]
                elif default_ds in harm_grp:
                    metric_frame = harm_grp[default_ds][idx]
                else:
                    metric_frame = harm_grp['common_mask'][idx]
                h_f, w_f = metric_frame.shape
                o = np.zeros((h_f, w_f, 3), dtype=np.float32)
                valid_m = ~np.isnan(metric_frame)
                if np.any(valid_m):
                    mn, mx = np.nanmin(metric_frame), np.nanmax(metric_frame)
                    denom = mx - mn if mx > mn else 1.0
                    normed = np.clip((metric_frame - mn) / denom, 0, 1)
                    o[valid_m, 0] = normed[valid_m]
                    o[valid_m, 1] = normed[valid_m]
                    o[valid_m, 2] = normed[valid_m]
            return o, spc

    dates = [datetime.fromtimestamp(ts, timezone.utc) for ts in acq_time]
    target_date = datetime(2025, 9, 12, tzinfo=timezone.utc).date()
    diffs = [abs((d.date() - target_date).days) for d in dates]
    base_idx = int(np.argmin(diffs)) if diffs else 0
    base_frame, base_sg = get_ortho(base_idx)
    base_date = dates[base_idx] if dates else datetime.now(timezone.utc)

    H, W = full_valid_mask.shape[1], full_valid_mask.shape[2]
    anomaly_map = anomaly_map.copy()
    anomaly_map[change_count_map == 0] = np.nan

    fig, (ax_img, ax_ts) = plt.subplots(1, 2, figsize=(18, 8))
    fig.subplots_adjust(bottom=0.15)

    ax_img.imshow(base_frame)
    ax_img.set_title(f"{base_sg} Acquisition: {base_date.strftime('%Y-%m-%d %H:%M:%S')} UTC")

    valid_initial_counts = np.sum(full_valid_mask, axis=0)
    insufficient_data = valid_initial_counts < min_samples

    gray = np.zeros((H, W, 4), dtype=np.float32)
    gray[insufficient_data, 0] = 0.5
    gray[insufficient_data, 1] = 0.5
    gray[insufficient_data, 2] = 0.5
    gray[insufficient_data, 3] = 0.5
    ax_img.imshow(gray)

    if not np.all(np.isnan(anomaly_map)):
        from matplotlib.cm import gist_rainbow
        masked_anom = np.ma.masked_invalid(anomaly_map)
        cmap = gist_rainbow
        cmap.set_bad(color='white', alpha=0)
        im = ax_img.imshow(masked_anom, cmap=cmap, alpha=0.7)
        
        y_anom, x_anom = np.where(~np.isnan(anomaly_map))
        cbar = plt.colorbar(im, ax=ax_img)
        ticks = cbar.get_ticks()
        min_anom, max_anom = np.nanmin(anomaly_map), np.nanmax(anomaly_map)
        if not np.isnan(min_anom):
            ticks = ticks[(ticks >= min_anom) & (ticks <= max_anom)]
            if len(ticks) > 0:
                cbar.set_ticks(ticks)
                cbar.set_ticklabels([datetime.fromtimestamp(t, timezone.utc).strftime('%Y-%m-%d') for t in ticks])

    rect = patches.Rectangle((-1, -1), 1, 1, linewidth=2, edgecolor='orange', facecolor='none', visible=False)
    ax_img.add_patch(rect)

    ax_ts.text(0.5, 0.5, 'Click a pixel on the map to view data',
               horizontalalignment='center', verticalalignment='center', transform=ax_ts.transAxes)

    current_selected = {'x': None, 'y': None}

    if interactive:

        def onclick(event):
            if event.inaxes != ax_img: return
            x, y = int(event.xdata), int(event.ydata)
            if x < 0 or x >= W or y < 0 or y >= H: return
            print(f"Clicked on {x}, {y}")

            rect.set_xy((x - 0.5, y - 0.5))
            rect.set_visible(True)

            current_selected['x'] = x
            current_selected['y'] = y

            if not np.isnan(anomaly_map[y, x]):
                anom_ts = anomaly_map[y, x]
                idx = int(np.argmin(np.abs(acq_time - anom_ts)))
                new_base, current_sg = get_ortho(idx)
                ax_img.images[0].set_array(new_base)
                current_date_ts = acq_time[idx]
            else:
                ax_img.images[0].set_array(base_frame)
                current_date_ts = acq_time[base_idx]
                current_sg = base_sg

            current_date = datetime.fromtimestamp(current_date_ts, timezone.utc)
            ax_img.set_title(f"{current_sg} Acquisition: {current_date.strftime('%Y-%m-%d %H:%M:%S')} UTC")
            ax_ts.clear()
            plot_pixel_sits(y, x, source_h5_path, target_metric=target_metric, injected=injected, ax=ax_ts, current_date=current_date, inference_results_h5=inf_source)
            fig.canvas.draw()

        fig.canvas.mpl_connect('button_press_event', onclick)

    if output_png is None and not interactive:
        loc = os.path.basename(source_h5_path).split('_')[1] if '_' in os.path.basename(source_h5_path) else "Rochesterv2"
        output_png = get_default_png_name(location=loc, target_metric=target_metric)

    if output_png:
        if not os.path.isabs(output_png) and not os.path.dirname(output_png):
            output_png = os.path.join(os.path.dirname(source_h5_path), output_png)
        out_parent = os.path.dirname(output_png)
        if out_parent:
            os.makedirs(out_parent, exist_ok=True)
        fig.savefig(output_png, bbox_inches='tight', dpi=150)
        print(f"Saved diagnostic PNG map to {output_png}")
        if not interactive:
            plt.close(fig)
            return fig

    if interactive:
        plt.show()

    return fig


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Harmonized CCD Visualization")
    parser.add_argument('--source-h5', type=str, default="C:/satelliteImagery/HLST30/HLST_Rochesterv2.h5", help="Path to source harmonized HDF5")
    parser.add_argument('--output-png', type=str, default=False, help="Path to save diagnostic PNG map")
    parser.add_argument('--metric', type=str, default=TARGET_METRIC, help="Target metric ('SC', 'NDVI')")
    parser.add_argument('--injected', action='store_true', help="Visualize injected results ('INJECTED_SCCCD_Results', etc.)")
    parser.add_argument('--no-gui', action='store_true', help="Run in non-interactive batch mode")
    args = parser.parse_args()

    source_h5 = args.source_h5 

    if args.metric == 'sliding_volume_z_score':
        TARGET_NAME = 'Spectral Complexity (Z-Score)'
        TARGET_SHORT_NAME = 'SC'
    elif args.metric == 'ndvi':
        TARGET_NAME = 'NDVI'
        TARGET_SHORT_NAME = 'NDVI'
    else:
        TARGET_NAME = args.metric

    if source_h5 and os.path.exists(source_h5):
        print(f"Loading latest inference results: {source_h5}")
        plot_spatial_anomaly_overlay(
            source_h5,
            target_metric=TARGET_SHORT_NAME,
            injected=args.injected,
            output_png=args.output_png,
            interactive=not args.no_gui
        )
    else:
        print("Run harmonized_CCD_main.py first to create output h5")
