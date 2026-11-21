"""
USGS PyCCD Landsat Visualization and SITS Diagnostic Tool.

Provides interactive and programmatic visualization of USGS CCDC baseline
change detection results stored in the multi-sensor HDF5 file.
Designed for IDE execution (Run button in VS Code / IDE) and script integration.
"""

import os
import sys
import datetime
from datetime import timezone
import h5py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.dates as mdates
import pyproj

# ==============================================================================
# Configuration Defaults (IDE friendly)
# ==============================================================================
LOCATION = "Rochesterv2"
H5_PATH = f"C:/satelliteImagery/HLST30/HLST_{LOCATION}.h5"
GROUP_NAME = "PYCCD_Results"
INJECTED = False

BAND_INDICES = {
    'blue': 1,
    'green': 2,
    'red': 3,
    'nir': 4,
    'swir1': 5,
    'swir2': 6
}
BAND_LABELS = {
    'blue': 'Blue Surface Reflectance (0.482 \u03bcm)',
    'green': 'Green Surface Reflectance (0.561 \u03bcm)',
    'red': 'Red Surface Reflectance (0.655 \u03bcm)',
    'nir': 'NIR Surface Reflectance (0.865 \u03bcm)',
    'swir1': 'SWIR1 Surface Reflectance (1.609 \u03bcm)',
    'swir2': 'SWIR2 Surface Reflectance (2.201 \u03bcm)',
    'ndvi': 'Normalized Difference Vegetation Index (NDVI)'
}


def get_default_h5_path(location=LOCATION):
    """Finds available HDF5 path."""
    candidates = [
        f"C:/satelliteImagery/HLST30/HLST_{location}.h5",
        f"C:/satelliteImagery/HLST30/HLST_{location}v2.h5",
        f"C:/satelliteImagery/HLST30/HLST_{location}_Harmonized_SC_EM-7_Norm-None.h5"
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[0]


def get_default_png_name(location=LOCATION, injected=INJECTED):
    """Generates standard diagnostic PNG filename matching interface contract."""
    prefix = "injected_" if injected else ""
    return f"{location}_PYCCD_{prefix}anomaly_overlay.png"


def pixel_to_latlon(pixel_y, pixel_x, geo_transform, spatial_ref):
    """Converts pixel indices (y, x) to WGS84 (lon, lat) using GDAL GeoTransform and CRS."""
    if geo_transform is None or spatial_ref is None:
        return None, None
    try:
        gt = geo_transform
        x_geo = gt[0] + (pixel_x + 0.5) * gt[1] + (pixel_y + 0.5) * gt[2]
        y_geo = gt[3] + (pixel_x + 0.5) * gt[4] + (pixel_y + 0.5) * gt[5]

        if isinstance(spatial_ref, bytes):
            s_str = spatial_ref.decode('utf-8')
        else:
            s_str = str(spatial_ref)

        crs = pyproj.CRS.from_user_input(s_str)
        transformer = pyproj.Transformer.from_crs(crs, "epsg:4326", always_xy=True)
        lon, lat = transformer.transform(x_geo, y_geo)
        return lat, lon
    except Exception:
        return None, None


def plot_pixel_sits(
    pixel_y,
    pixel_x,
    source_h5_path=None,
    group_name=None,
    injected=INJECTED,
    band=None,
    ax=None,
    current_date=None,
    show_plot=True
):
    """
    Plots pixel-level Landsat Satellite Image Time Series (SITS) with CCDC harmonic fit.
    """
    if source_h5_path is None:
        source_h5_path = get_default_h5_path()

    if not os.path.exists(source_h5_path):
        raise FileNotFoundError(f"Source file not found: {source_h5_path}")

    if group_name is None:
        group_name = 'INJECTED_PYCCD_Results' if injected else GROUP_NAME
    elif injected and group_name == GROUP_NAME:
        group_name = 'INJECTED_PYCCD_Results'

    with h5py.File(source_h5_path, 'r') as f_root:
        if group_name not in f_root:
            if injected and 'INJECTED_PYCCD_Results' in f_root:
                group_name = 'INJECTED_PYCCD_Results'
            else:
                raise KeyError(f"Group '{group_name}' not found in {source_h5_path}. Run pyccd_landsat_main.py first.")

        grp = f_root[group_name]
        target_band = grp.attrs.get('TARGET_BAND', 'nir') if band is None else band
        if isinstance(target_band, bytes):
            target_band = target_band.decode('utf-8')

        ds_name = f'predicted_{target_band}' if f'predicted_{target_band}' in grp else 'predicted_series'
        predicted = grp[ds_name][:, pixel_y, pixel_x]
        rmse_key = 'rmse_series' if 'rmse_series' in grp else 'rmse_ndvi'
        rmse = grp[rmse_key][:, pixel_y, pixel_x]
        anomalies = grp['anomaly_flags'][:, pixel_y, pixel_x]
        first_break_ts = grp['change_date_timestamp'][pixel_y, pixel_x]
        change_count = int(grp['change_count'][pixel_y, pixel_x])
        water_prob = float(grp['water_probability'][pixel_y, pixel_x]) if 'water_probability' in grp else np.nan

        geo_transform = grp.attrs.get('GeoTransform')
        spatial_ref = grp.attrs.get('spatial_ref')
        roi_rows = grp.attrs.get('ROI_ROWS')
        roi_cols = grp.attrs.get('ROI_COLS')
        source_data_file = grp.attrs.get('SOURCE_DATA')

    # Resolve file containing original Landsat observations
    obs_file = source_h5_path
    with h5py.File(obs_file, 'r') as f_check:
        if '/HDFEOS/GRIDS/HLSL30/Data Fields' not in f_check:
            if source_data_file and os.path.exists(source_data_file):
                obs_file = source_data_file

    with h5py.File(obs_file, 'r') as f_obs:
        l_grp = f_obs['/HDFEOS/GRIDS/HLSL30/Data Fields']
        sr_target = grp.attrs.get('SR_DATASET', 'injected_surface_reflectance' if injected or 'INJECTED' in group_name else 'surface_reflectance')
        if isinstance(sr_target, bytes):
            sr_target = sr_target.decode('utf-8')
        if sr_target not in l_grp:
            sr_target = 'surface_reflectance'
        sr_ds = l_grp[sr_target]
        fm_ds = l_grp['Fmask']
        cm_ds = l_grp['common_mask']

        acq_time = sr_ds.attrs.get('acquisition_time')
        if acq_time is None:
            acq_time = f_obs['/HDFEOS/GRIDS/HARMONIZED/Data Fields'].attrs.get('acquisition_time')
        acq_time = np.array(acq_time[:], dtype=np.float64)

        if geo_transform is None:
            geo_transform = sr_ds.attrs.get('GeoTransform', f_obs.attrs.get('GeoTransform'))
        if spatial_ref is None:
            spatial_ref = sr_ds.attrs.get('spatial_ref', f_obs.attrs.get('spatial_ref'))

        # Map to global coordinates if results were computed on a sub-region ROI
        y_global = roi_rows[0] + pixel_y if roi_rows is not None else pixel_y
        x_global = roi_cols[0] + pixel_x if roi_cols is not None else pixel_x

        fm = fm_ds[:, y_global, x_global]
        cm = cm_ds[:, y_global, x_global]

        if target_band.lower() == 'ndvi':
            red_raw = sr_ds[:, 3, y_global, x_global]
            nir_raw = sr_ds[:, 4, y_global, x_global]
            denom = nir_raw.astype(np.float64) + red_raw.astype(np.float64)
            valid_refl = (red_raw != -32768) & (nir_raw != -32768) & (np.abs(denom) > 1e-4)
            sr_vals = np.full(red_raw.shape, np.nan, dtype=np.float32)
            sr_vals[valid_refl] = ((nir_raw[valid_refl].astype(np.float64) - red_raw[valid_refl].astype(np.float64)) / denom[valid_refl]).astype(np.float32)
        else:
            band_idx = BAND_INDICES.get(target_band, 4)
            raw_sr = sr_ds[:, band_idx, y_global, x_global]
            sr_vals = np.full(raw_sr.shape, np.nan, dtype=np.float32)
            valid_refl = (raw_sr != -32768)
            sr_vals[valid_refl] = raw_sr[valid_refl].astype(np.float32) / 10000.0

    valid_mask = ~cm.astype(bool) & (fm != 255) & valid_refl
    is_anomaly = (anomalies == 1)

    lat, lon = pixel_to_latlon(y_global, x_global, geo_transform, spatial_ref)
    dates = [datetime.datetime.fromtimestamp(ts, timezone.utc) for ts in acq_time]
    dates_arr = np.array(dates)

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
        standalone = True
    else:
        standalone = False

    # 1. Observed Points
    idx_normal = valid_mask & ~is_anomaly
    if np.any(idx_normal):
        ax.plot(
            dates_arr[idx_normal], sr_vals[idx_normal],
            color='black', marker='o', markersize=4, linestyle='None',
            label='Valid Observation (Landsat)'
        )

    idx_anom = valid_mask & is_anomaly
    if np.any(idx_anom):
        ax.plot(
            dates_arr[idx_anom], sr_vals[idx_anom],
            color='crimson', marker='D', markersize=7, linestyle='None',
            label=f'Break Event ({np.sum(idx_anom)})'
        )

    idx_invalid = ~valid_mask
    if np.any(idx_invalid):
        ax.plot(
            dates_arr[idx_invalid], sr_vals[idx_invalid],
            color='silver', marker='o', markersize=3, linestyle='None',
            markerfacecolor='none', alpha=0.5, label='Masked (Cloud/Shadow)'
        )

    # 2. CCDC Harmonic Segments
    pred_mask = ~np.isnan(predicted)
    if np.any(pred_mask):
        pred_dates = dates_arr[pred_mask]
        preds = predicted[pred_mask]
        rmses = rmse[pred_mask]

        # Detect segment transitions where frozen RMSE changes
        rmse_diff = np.diff(rmses)
        break_indices = np.where(rmse_diff != 0)[0] + 1

        date_segs = np.split(pred_dates, break_indices)
        pred_segs = np.split(preds, break_indices)
        rmse_segs = np.split(rmses, break_indices)

        colors = ['#1f77b4', '#2ca02c', '#9467bd', '#ff7f0e', '#17becf', '#e377c2']

        for i, (s_dates, s_preds, s_rmses) in enumerate(zip(date_segs, pred_segs, rmse_segs)):
            if len(s_dates) == 0:
                continue
            c = colors[i % len(colors)]
            upper = s_preds + 3.0 * s_rmses
            lower = s_preds - 3.0 * s_rmses

            seg_label = f"CCDC Fit (Seg {i+1})" if len(date_segs) > 1 else "CCDC Harmonic Fit"
            ax.plot(s_dates, s_preds, color=c, linestyle='-', linewidth=2, label=seg_label)
            ax.fill_between(s_dates, lower, upper, color=c, alpha=0.15, label='Fit \u00b13\u03c3' if i == 0 else None)

    # 3. Mark Break Dates
    if not np.isnan(first_break_ts) and first_break_ts > 0:
        dt_break = datetime.datetime.fromtimestamp(first_break_ts, timezone.utc)
        ax.axvline(x=dt_break, color='crimson', linestyle=':', linewidth=1.8, label=f'First Break: {dt_break.strftime("%Y-%m-%d")}')

    if current_date is not None:
        ax.axvline(x=current_date, color='darkorange', linestyle='--', linewidth=2, label='Current Frame')

    title_prefix = "USGS CCDC (Injected Landsat SR)" if (injected or 'INJECTED' in group_name) else "USGS CCDC Baseline"
    title_str = f"{title_prefix} for Pixel (y={y_global}, x={x_global})"
    if lat is not None and lon is not None:
        title_str += f" | Lat: {lat:.5f}\u00b0, Lon: {lon:.5f}\u00b0"
    title_str += f" | Breaks: {change_count}"
    if not np.isnan(water_prob):
        title_str += f" | Water: {water_prob * 100.0:.1f}%"

    ax.set_title(title_str, fontsize=12, fontweight='bold')
    ax.set_ylabel(BAND_LABELS.get(target_band, f'{target_band.upper()} Reflectance'), fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))

    # Deduplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0., fontsize=9)

    if standalone and show_plot:
        plt.tight_layout()
        plt.show()

    return ax


def plot_spatial_change_map(
    source_h5_path=None,
    group_name=None,
    injected=INJECTED,
    output_png=None,
    interactive=True
):
    """
    Renders 2-pane interactive GUI: spatial change map on left, click-to-plot SITS on right.
    """
    if source_h5_path is None:
        source_h5_path = get_default_h5_path()

    if not os.path.exists(source_h5_path):
        raise FileNotFoundError(f"Source file not found: {source_h5_path}")

    if group_name is None:
        group_name = 'INJECTED_PYCCD_Results' if injected else GROUP_NAME
    elif injected and group_name == GROUP_NAME:
        group_name = 'INJECTED_PYCCD_Results'

    print(f"Loading CCDC results from {source_h5_path} -> group '{group_name}'...")
    with h5py.File(source_h5_path, 'r') as f_root:
        if group_name not in f_root:
            if injected and 'INJECTED_PYCCD_Results' in f_root:
                group_name = 'INJECTED_PYCCD_Results'
            else:
                raise KeyError(f"Group '{group_name}' not found. Run pyccd_landsat_main.py first.")

        grp = f_root[group_name]
        change_dates = grp['change_date_timestamp'][:]
        change_counts = grp['change_count'][:]
        target_band = grp.attrs.get('TARGET_BAND', 'nir')
        if isinstance(target_band, bytes):
            target_band = target_band.decode('utf-8')

        # Load RGB visual
        if '/HDFEOS/GRIDS/HLSL30/Data Fields/ortho_visual' in f_root:
            ortho_ds = f_root['/HDFEOS/GRIDS/HLSL30/Data Fields/ortho_visual']
            mid_idx = ortho_ds.shape[0] // 2
            ortho = ortho_ds[mid_idx, :3, ...]
            ortho = np.transpose(ortho, (1, 2, 0)).astype(np.float32) / 255.0
        elif '/HDFEOS/GRIDS/HARMONIZED/Data Fields/ortho_visual' in f_root:
            ortho_ds = f_root['/HDFEOS/GRIDS/HARMONIZED/Data Fields/ortho_visual']
            mid_idx = ortho_ds.shape[0] // 2
            ortho = ortho_ds[mid_idx]
            if ortho.shape[0] == 3:
                ortho = np.transpose(ortho, (1, 2, 0))
            ortho = ortho.astype(np.float32) / 255.0
        else:
            H, W = change_dates.shape
            ortho = np.full((H, W, 3), 0.3, dtype=np.float32)

    H, W = change_dates.shape
    anomaly_map = change_dates.copy()
    anomaly_map[change_counts == 0] = np.nan

    fig, (ax_img, ax_ts) = plt.subplots(1, 2, figsize=(18, 8))
    fig.subplots_adjust(bottom=0.10)

    # Left pane: image backdrop
    ax_img.imshow(ortho)
    title_prefix = "USGS CCDC (Injected Landsat Surface Reflectance)" if (injected or 'INJECTED' in group_name) else "USGS CCDC Baseline"
    ax_img.set_title(f"{title_prefix}: Change Break Dates\n(Click any pixel to inspect SITS fit)", fontsize=11, fontweight='bold')

    # Overlay change dates heatmap
    if not np.all(np.isnan(anomaly_map)):
        from matplotlib.cm import gist_rainbow
        masked_anom = np.ma.masked_invalid(anomaly_map)
        cmap = gist_rainbow
        cmap.set_bad(color='white', alpha=0)
        im = ax_img.imshow(masked_anom, cmap=cmap, alpha=0.7)

        cbar = plt.colorbar(im, ax=ax_img, fraction=0.046, pad=0.04)
        cbar.set_label('Detected Break Date', fontsize=10)
        ticks = cbar.get_ticks()
        min_anom, max_anom = np.nanmin(anomaly_map), np.nanmax(anomaly_map)
        if not np.isnan(min_anom):
            ticks = ticks[(ticks >= min_anom) & (ticks <= max_anom)]
            if len(ticks) > 0:
                cbar.set_ticks(ticks)
                cbar.set_ticklabels([datetime.datetime.fromtimestamp(t, timezone.utc).strftime('%Y-%m') for t in ticks])

    rect = patches.Rectangle((-1, -1), 1, 1, linewidth=2, edgecolor='crimson', facecolor='none', visible=False)
    ax_img.add_patch(rect)

    ax_ts.text(0.5, 0.5, 'Click any pixel on the left map to inspect\nLandsat SITS and CCDC harmonic fit',
               horizontalalignment='center', verticalalignment='center', transform=ax_ts.transAxes, fontsize=12, color='gray')

    if interactive:
        def onclick(event):
            if event.inaxes != ax_img:
                return
            x, y = int(round(event.xdata)), int(round(event.ydata))
            if x < 0 or x >= W or y < 0 or y >= H:
                return

            rect.set_xy((x - 0.5, y - 0.5))
            rect.set_visible(True)

            ax_ts.clear()
            plot_pixel_sits(y, x, source_h5_path=source_h5_path, group_name=group_name, injected=injected, band=target_band, ax=ax_ts, show_plot=False)
            fig.canvas.draw_idle()

        fig.canvas.mpl_connect('button_press_event', onclick)

    if output_png:
        fig.savefig(output_png, bbox_inches='tight', dpi=150)
        print(f"Saved diagnostic PNG map to {output_png}")

    if interactive:
        plt.show()

    return fig


def compare_sc_vs_pyccd(
    pixel_y,
    pixel_x,
    source_h5_path=None,
    group_name=None,
    injected=INJECTED,
    target_metric='SC',
    output_png=None,
    show_plot=True
):
    """
    Plots side-by-side comparison between Spectral Complexity CCD and USGS PyCCD at the same pixel.
    """
    if source_h5_path is None:
        source_h5_path = get_default_h5_path()

    if group_name is None:
        group_name = 'INJECTED_PYCCD_Results' if injected else GROUP_NAME
    elif injected and group_name == GROUP_NAME:
        group_name = 'INJECTED_PYCCD_Results'

    from harmonized_CCD_vis import plot_pixel_sits as plot_sc_sits

    fig, (ax_sc, ax_ccd) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # 1. Spectral Complexity (Harmonized Constellation)
    plot_sc_sits(pixel_y, pixel_x, source_h5_path=source_h5_path, target_metric=target_metric, injected=injected, ax=ax_sc)
    metric_label = f"{target_metric} Z-Score (Injected)" if injected else f"{target_metric} Z-Score (Baseline)"
    ax_sc.set_title(f"A. {metric_label} (Harmonized Virtual Constellation) | Pixel ({pixel_y}, {pixel_x})", fontsize=11, fontweight='bold')

    # 2. USGS PyCCD (Landsat-only)
    plot_pixel_sits(pixel_y, pixel_x, source_h5_path=source_h5_path, group_name=group_name, injected=injected, ax=ax_ccd, show_plot=False)
    ccd_label = "USGS CCDC (Injected Landsat SR)" if injected else "USGS CCDC Baseline (Landsat SR)"
    ax_ccd.set_title(f"B. {ccd_label} ({group_name}) | Pixel ({pixel_y}, {pixel_x})", fontsize=11, fontweight='bold')

    plt.tight_layout()

    if output_png:
        fig.savefig(output_png, bbox_inches='tight', dpi=150)
        print(f"Saved comparative figure to {output_png}")

    if show_plot:
        plt.show()

    return fig


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Visualize USGS PyCCD Baseline Results")
    parser.add_argument('--source-h5', type=str, default=None, help="Path to HDF5 file")
    parser.add_argument('--group', type=str, default=None, help="HDF5 group name (defaults to 'PYCCD_Results' or 'INJECTED_PYCCD_Results' when --injected)")
    parser.add_argument('--injected', action='store_true', default=INJECTED, help="Visualize injected results ('INJECTED_PYCCD_Results')")
    parser.add_argument('--target-metric', type=str, default='SC', help="Metric for comparison (default: 'SC')")
    parser.add_argument('--pixel-y', type=int, default=None, help="Pixel Y index to plot")
    parser.add_argument('--pixel-x', type=int, default=None, help="Pixel X index to plot")
    parser.add_argument('--output-png', type=str, default=None, help="Save figure to PNG")
    parser.add_argument('--compare', action='store_true', help="Plot SC vs PyCCD comparison")
    parser.add_argument('--no-gui', action='store_true', help="Run non-interactively")
    args = parser.parse_args()

    h5_file = args.source_h5 if args.source_h5 else get_default_h5_path()

    if args.pixel_y is not None and args.pixel_x is not None:
        if args.compare:
            compare_sc_vs_pyccd(
                args.pixel_y, args.pixel_x,
                source_h5_path=h5_file,
                group_name=args.group,
                injected=args.injected,
                target_metric=args.target_metric,
                output_png=args.output_png,
                show_plot=not args.no_gui,
            )
        else:
            plot_pixel_sits(
                args.pixel_y, args.pixel_x,
                source_h5_path=h5_file,
                group_name=args.group,
                injected=args.injected,
                show_plot=not args.no_gui,
            )
    else:
        plot_spatial_change_map(
            source_h5_path=h5_file,
            group_name=args.group,
            injected=args.injected,
            output_png=args.output_png,
            interactive=not args.no_gui,
        )
