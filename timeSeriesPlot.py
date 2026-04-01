import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from datetime import datetime, timezone
from zoneinfo import ZoneInfo


import rasterio.transform
from pyproj import Transformer, CRS
import SpecComplex as sc

# ==========================================
# 1. CONFIGURATION
# ==========================================
Location = "Tait"
Frame_Reg = "WRS16" # "CoReg"

# Time Series Range
START_YEAR = 2015
END_YEAR = 2025
TS_START_DATE = datetime(START_YEAR, 1, 1, tzinfo=timezone.utc)
TS_END_DATE = datetime(END_YEAR, 12, 31, tzinfo=timezone.utc)

# Combined Pixel Mask Configuration
MASKING = True
SUN_ELEVATION_THRESHOLD = 30
CLOUD_DILATION = 2

# Tanager Pixel Mask Configuration
TANAGER_AEROSOL_DEPTH_THRESHOLD = 0.3
TANAGER_SR_UNCERTAINTY_THRESHOLD = 0.10

# LANDSAT Pixel Mask Configuration
QA_REJECT_MASK = 0b111111
RADSAT_ACCEPT_VALUE = 0
AEROSOL_ACCEPT_LEVEL = 'medium' #'low' 'medium' 'high'

# Primary & Secondary Datasets for the axes
PRIM_METRIC = 'sliding_volume_map' # 'sliding_volume_map' 'sliding_volume_z_score'
SEC_METRIC = 'sliding_volume_z_score_masked'

COMPLEXITY_DICT = {
    'sliding_volume_map': 'Spectral Complexity',
    'sliding_volume_z_score': 'Spectral Complexity Z-Score',
    'sliding_volume_z_score_masked': 'Spectral Complexity Z-Score',
    'sliding_volume_local_z_score': 'Spectral Complexity Local Z-Score'
}

landsat_path = f"C:/satelliteImagery/LANDSAT/{Location}/LANDSAT_Stack_{Location}_GEE_2015_2025_{Frame_Reg}_SC_EM-7_Gram-minEndmember_Norm-bandCount.h5"
tanager_path = f"C:/satelliteImagery/Tanager/{Location}/Tanager_Stack_{Location}_HDFEOS_SC_EM-7_Gram-minEndmember_Norm-bandCount.h5"

# Time Series Locations (Latitude, Longitude)
TS_LOCATIONS = [
    {'latlon': (43.142856, -77.508451), 'label': "West Tait Forest",     'color': 'tab:green'},
    {'latlon': (43.144861, -77.501176), 'label': "East Tait Forest",             'color': 'tab:olive'},
    {'latlon': (43.136910, -77.469462), 'label': "Artificial turf football field",  'color': 'tab:blue'},
    {'latlon': (43.138241, -77.470873), 'label': "Recently added artificial turf",  'color': 'tab:cyan'},
    {'latlon': (43.141297, -77.506256), 'label': "Tait Parking Lot",                'color': 'tab:red'},
    {'latlon': (43.139411, -77.504005), 'label': "ROCX NITE Tarp",                  'color': 'tab:purple'},
]

# ==========================================
# 2. DATA EXTRACTION
# ==========================================

def extract_data():
    files = [landsat_path, tanager_path]
    
    # Initialize Data Structure
    ts_data = {
        'LANDSAT': {loc['label']: {'t': [], 'prim_v': [], 'sec_v': []} for loc in TS_LOCATIONS},
        'TANAGER': {loc['label']: {'t': [], 'prim_v': [], 'sec_v': []} for loc in TS_LOCATIONS}
    }
    
    for path in files:
        if not os.path.exists(path):
            print(f"Warning: File not found: {path}")
            continue
            
        with h5py.File(path, 'r') as h5:
            source_name = list(h5['/HDFEOS/GRIDS'].keys())[0]
            key = 'LANDSAT' if 'LANDSAT' in source_name.upper() else 'TANAGER'
            data_grp = h5[f'/HDFEOS/GRIDS/{source_name}/Data Fields']
            
            sr_dset = data_grp['surface_reflectance']
            vis_dset = data_grp['ortho_visual']
            prim_dset = data_grp[PRIM_METRIC]
            sec_dset = data_grp[SEC_METRIC] if SEC_METRIC else None
            
            # 2.1 Map Geographic Coordinates to Pixel Coordinates
            geo_transform = sr_dset.attrs['GeoTransform']
            spatial_ref = sr_dset.attrs['spatial_ref']
            if isinstance(spatial_ref, bytes): spatial_ref = spatial_ref.decode('utf-8')
            
            crs = CRS.from_wkt(spatial_ref)
            transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
            affine = rasterio.transform.Affine.from_gdal(*geo_transform)
            inv_affine = ~affine
            
            mapped_locations = []
            for loc in TS_LOCATIONS:
                lat, lon = loc['latlon']
                proj_x, proj_y = transformer.transform(lon, lat)
                px, py = inv_affine * (proj_x, proj_y)
                mapped_locations.append({
                    'label': loc['label'], 
                    'y': int(round(py)), 
                    'x': int(round(px))
                })
            
            # 2.2 Extract Time Series
            acq_times = sr_dset.attrs['acquisition_time']
            num_frames = prim_dset.shape[0]
            
            for f_idx in range(num_frames):
                dt = datetime.fromtimestamp(acq_times[f_idx], tz=timezone.utc)
                if not (TS_START_DATE <= dt <= TS_END_DATE):
                    continue
                
                shape = prim_dset[f_idx].shape
                
                # 1. ARD Alpha Masking: Channel 3 (index 3) is the explicit Alpha channel.
                # Valid pixels are strictly greater than 0.
                alpha_mask = vis_dset[f_idx, 3, ...] > 0
                
                # 2. Apply Strict Spatial Pixel Masking via SpecComplex
                if not MASKING:
                    spatial_mask = np.ones(shape, dtype=bool)
                elif key == 'LANDSAT':
                    spatial_mask = sc.get_landsat_mask(
                        data_grp=data_grp,
                        f_idx=f_idx,
                        shape=shape,
                        sun_elevation_threshold=SUN_ELEVATION_THRESHOLD,
                        cloud_dilation=CLOUD_DILATION,
                        qa_reject_mask=QA_REJECT_MASK,
                        radsat_accept_value=RADSAT_ACCEPT_VALUE,
                        aerosol_accept_level=AEROSOL_ACCEPT_LEVEL
                    )
                else:
                    spatial_mask = sc.get_tanager_mask(
                        data_grp=data_grp,
                        f_idx=f_idx,
                        shape=shape,
                        sun_elevation_threshold=SUN_ELEVATION_THRESHOLD,
                        cloud_dilation=CLOUD_DILATION,
                        apply_cloud_mask=True,
                        uncertainty_threshold=TANAGER_SR_UNCERTAINTY_THRESHOLD,
                        aerosol_depth_threshold=TANAGER_AEROSOL_DEPTH_THRESHOLD
                    )
                    
                # Combine both masks to ensure data integrity
                combined_mask = alpha_mask & spatial_mask
                
                for loc in mapped_locations:
                    y, x = loc['y'], loc['x']
                    # Verify array bounds
                    if 0 <= y < shape[0] and 0 <= x < shape[1]:
                        if combined_mask[y, x]:
                            prim_val = prim_dset[f_idx, y, x]
                            sec_val = sec_dset[f_idx, y, x] if sec_dset is not None else np.nan
                            
                            if not np.isnan(prim_val):
                                ts_data[key][loc['label']]['t'].append(dt)
                                ts_data[key][loc['label']]['prim_v'].append(prim_val)
                                ts_data[key][loc['label']]['sec_v'].append(sec_val)

    return ts_data

# ==========================================
# 3. PLOTTING
# ==========================================

def plot_time_series(ts_data):
    # ==========================================
    # Figure 1: Twin Y-Axis Overlay
    # ==========================================
    fig, ax1 = plt.subplots(figsize=(16, 8))
    fig.canvas.manager.set_window_title("Twin-Axis Metric Overlay")
    ax2 = ax1.twinx() if SEC_METRIC else None

    # Plot LANDSAT Time Series
    for loc in TS_LOCATIONS:
        label = loc['label']
        data = ts_data['LANDSAT'][label]
        if data['t']:
            ax1.plot(data['t'], data['prim_v'], marker='^', color=loc['color'], 
                     markersize=5, linestyle='--', linewidth=1, alpha=0.7)
            if ax2 and SEC_METRIC:
                ax2.plot(data['t'], data['sec_v'], marker='x', color=loc['color'], 
                         markersize=4, linestyle=':', linewidth=1, alpha=0.4)

    # Plot TANAGER Time Series
    for loc in TS_LOCATIONS:
        label = loc['label']
        data = ts_data['TANAGER'][label]
        if data['t']:
            ax1.plot(data['t'], data['prim_v'], marker='s', color=loc['color'], 
                     markersize=6, linestyle='-', linewidth=2, alpha=0.9)
            if ax2 and SEC_METRIC:
                ax2.plot(data['t'], data['sec_v'], marker='d', color=loc['color'], 
                         markersize=5, linestyle='-.', linewidth=1.5, alpha=0.6)

    if len(ax1.lines) > 0:
        xlims = ax1.get_xlim()
        # Seasonal background shading
        for yr in range(START_YEAR, END_YEAR + 2):
            ax1.axvspan(datetime(yr - 1, 12, 1, tzinfo=timezone.utc), datetime(yr, 3, 1, tzinfo=timezone.utc), color='lightgray', alpha=0.3, zorder=0, lw=0)
            ax1.axvspan(datetime(yr, 3, 1, tzinfo=timezone.utc), datetime(yr, 6, 1, tzinfo=timezone.utc), color='lightgreen', alpha=0.2, zorder=0, lw=0)
            ax1.axvspan(datetime(yr, 6, 1, tzinfo=timezone.utc), datetime(yr, 9, 1, tzinfo=timezone.utc), color='lightyellow', alpha=0.3, zorder=0, lw=0)
            ax1.axvspan(datetime(yr, 9, 1, tzinfo=timezone.utc), datetime(yr, 12, 1, tzinfo=timezone.utc), color='orange', alpha=0.15, zorder=0, lw=0)
        ax1.set_xlim(xlims)

    # Styling and Formatting
    title_suffix = f" & {COMPLEXITY_DICT[SEC_METRIC]}" if SEC_METRIC else ""
    ax1.set_title(f"Multi-Sensor Time Series Analysis: {COMPLEXITY_DICT[PRIM_METRIC]}{title_suffix}\n[{START_YEAR}-{END_YEAR}]", fontsize=14, fontweight='bold')
    
    ax1.grid(True, alpha=0.4, which="both", ls="--")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.tick_params(axis='x', rotation=45)
    
    ax1.set_ylabel(COMPLEXITY_DICT[PRIM_METRIC], color='black', fontweight='bold', fontsize=12)
    if PRIM_METRIC == 'sliding_volume_map':
        ax1.set_yscale('log')
        
    if ax2:
        ax2.set_ylabel(COMPLEXITY_DICT[SEC_METRIC], color='gray', fontweight='bold', fontsize=12)
        if SEC_METRIC == 'sliding_volume_map':
            ax2.set_yscale('log')

    # Twin Axis Legend Construction
    loc_handles = [Line2D([0], [0], color=loc['color'], lw=4, label=loc['label']) for loc in TS_LOCATIONS]
    leg1 = ax1.legend(handles=loc_handles, title="Spatial Locations", loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    ax1.add_artist(leg1)
    
    style_handles = [
        Line2D([0], [0], color='black', marker='^', linestyle='--', linewidth=1, label=f"L: {PRIM_METRIC}"),
        Line2D([0], [0], color='black', marker='s', linestyle='-', linewidth=2, label=f"T: {PRIM_METRIC}")
    ]
    if ax2 and SEC_METRIC:
        style_handles.extend([
            Line2D([0], [0], color='gray', marker='x', linestyle=':', linewidth=1, label=f"L: {SEC_METRIC}"),
            Line2D([0], [0], color='gray', marker='d', linestyle='-.', linewidth=1.5, label=f"T: {SEC_METRIC}")
        ])
        
    ax1.legend(handles=style_handles, title="Sensors & Metrics", loc='center left', bbox_to_anchor=(1.05, 0.4), borderaxespad=0.)
    fig.tight_layout(rect=[0, 0, 0.82, 1])

    # ==========================================
    # Figure 2: Small Multiples (Separated Metrics)
    # ==========================================
    if SEC_METRIC:
        fig2, (ax_prim, ax_sec) = plt.subplots(2, 1, figsize=(16, 12), sharex=True)
        fig2.canvas.manager.set_window_title("Separated Metric Subplots")

        # Plot LANDSAT
        for loc in TS_LOCATIONS:
            label = loc['label']
            data = ts_data['LANDSAT'][label]
            if data['t']:
                ax_prim.plot(data['t'], data['prim_v'], marker='^', color=loc['color'], 
                             markersize=5, linestyle='--', linewidth=1, alpha=0.7)
                ax_sec.plot(data['t'], data['sec_v'], marker='^', color=loc['color'], 
                             markersize=5, linestyle='--', linewidth=1, alpha=0.7)

        # Plot TANAGER
        for loc in TS_LOCATIONS:
            label = loc['label']
            data = ts_data['TANAGER'][label]
            if data['t']:
                ax_prim.plot(data['t'], data['prim_v'], marker='s', color=loc['color'], 
                             markersize=6, linestyle='-', linewidth=2, alpha=0.9)
                ax_sec.plot(data['t'], data['sec_v'], marker='s', color=loc['color'], 
                             markersize=6, linestyle='-', linewidth=2, alpha=0.9)

        # Formatting axes
        for ax, metric in zip([ax_prim, ax_sec], [PRIM_METRIC, SEC_METRIC]):
            # Background shading
            if len(ax.lines) > 0:
                xlims = ax.get_xlim()
                for yr in range(START_YEAR, END_YEAR + 2):
                    ax.axvspan(datetime(yr - 1, 12, 1, tzinfo=timezone.utc), datetime(yr, 3, 1, tzinfo=timezone.utc), color='lightgray', alpha=0.3, zorder=0, lw=0)
                    ax.axvspan(datetime(yr, 3, 1, tzinfo=timezone.utc), datetime(yr, 6, 1, tzinfo=timezone.utc), color='lightgreen', alpha=0.2, zorder=0, lw=0)
                    ax.axvspan(datetime(yr, 6, 1, tzinfo=timezone.utc), datetime(yr, 9, 1, tzinfo=timezone.utc), color='lightyellow', alpha=0.3, zorder=0, lw=0)
                    ax.axvspan(datetime(yr, 9, 1, tzinfo=timezone.utc), datetime(yr, 12, 1, tzinfo=timezone.utc), color='orange', alpha=0.15, zorder=0, lw=0)
                ax.set_xlim(xlims)
            
            ax.grid(True, alpha=0.4, which="both", ls="--")
            ax.set_ylabel(COMPLEXITY_DICT[metric], color='black', fontweight='bold', fontsize=12)
            
            if metric == 'sliding_volume_map':
                ax.set_yscale('log')
            
            # Independent Legends for Subplots
            loc_handles_sep = [Line2D([0], [0], color=loc['color'], lw=4, label=loc['label']) for loc in TS_LOCATIONS]
            leg_sep1 = ax.legend(handles=loc_handles_sep, title="Spatial Locations", loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0.)
            ax.add_artist(leg_sep1)
            
            sensor_handles = [
                Line2D([0], [0], color='black', marker='^', linestyle='--', linewidth=1, label="Landsat 8/9"),
                Line2D([0], [0], color='black', marker='s', linestyle='-', linewidth=2, label="Tanager")
            ]
            ax.legend(handles=sensor_handles, title="Sensors", loc='center left', bbox_to_anchor=(1.01, 0.3), borderaxespad=0.)

        ax_prim.set_title(f"Separated Multi-Sensor Time Series Analysis\n[{START_YEAR}-{END_YEAR}]", fontsize=14, fontweight='bold')
        ax_sec.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax_sec.tick_params(axis='x', rotation=45)
        
        fig2.tight_layout(rect=[0, 0, 0.85, 1])

    plt.show()

if __name__ == "__main__":
    print(f"Extracting data from Multi-Sensor Virtual Constellation...")
    data = extract_data()
    print("Plotting time series...")
    plot_time_series(data)