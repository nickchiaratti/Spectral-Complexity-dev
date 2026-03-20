import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timezone
from scipy import ndimage

import rasterio.transform
from pyproj import Transformer, CRS

# ==========================================
# 1. CONFIGURATION
# ==========================================

complexity_type = 'sliding_volume_z_score'
secondary_metric = 'evi_map'

TS_START_DATE = datetime(2015, 1, 1, tzinfo=timezone.utc)
TS_END_DATE = datetime(2025, 12, 31, tzinfo=timezone.utc)

# Combined Pixel Mask Configuration
SUN_ELEVATION_THRESHOLD = 0
CLOUD_DILATION = 0

# Tanager Pixel Mask Configuration
TANAGER_AEROSOL_DEPTH_THRESHOLD = 10
TANAGER_SR_UNCERTAINTY_THRESHOLD = 0.10

# LANDSAT Pixel Mask Configuration
QA_REJECT_MASK = 0b111111
RADSAT_ACCEPT_VALUE = 0
AEROSOL_ACCEPT_LEVEL = 'all' # 'low' 'medium' 'high' 'all'

AEROSOL_DICT = {
    'low': [2, 4, 32, 66, 68, 96, 100],
    'medium': [2, 4, 32, 66, 68, 96, 100, 130, 132, 160, 164],
    'high': [2, 4, 32, 66, 68, 96, 100, 130, 132, 160, 164, 192, 194, 196, 224, 228]
}

landsat_path = "C:/satelliteImagery/LANDSAT/Tait/LANDSAT_Stack_Tait_GEE_2015_2025_SC_EM-7_Gram-minEndmember_Norm-bandCount_Aerosol-low_QA-AllFrames_sunElMin-40.h5"
tanager_path = "C:/satelliteImagery/Tanager/Tait/Tanager_Stack_Tait_HDFEOS_SC_EM-7_Gram-minEndmember_Norm-bandCount_Aerosol-low_QA-AllFrames_sunElMin-40.h5"

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
# 2. MASKING FUNCTIONS
# ==========================================

def get_landsat_mask(data_grp, f_idx, shape):
    """Generates a boolean mask for LANDSAT data based on configurations."""
    valid_mask = np.ones(shape, dtype=bool)
    
    sun_elev_arr = data_grp['surface_reflectance'].attrs.get('sun_elevation')
    if sun_elev_arr is not None and f_idx < len(sun_elev_arr):
        if sun_elev_arr[f_idx] < SUN_ELEVATION_THRESHOLD:
            return np.zeros(shape, dtype=bool)

    if 'QUALITY_L1_PIXEL' in data_grp:
        qa_pixel = data_grp['QUALITY_L1_PIXEL'][f_idx, ...]
        bad_qa_mask = (qa_pixel & QA_REJECT_MASK) != 0
        if CLOUD_DILATION > 0:
            kernel = np.ones((3, 3), dtype=bool)
            bad_qa_mask = ndimage.binary_dilation(bad_qa_mask, structure=kernel, iterations=CLOUD_DILATION)
        valid_mask &= ~bad_qa_mask

    if 'RADIOMETRIC_SATURATION' in data_grp:
        bad_radsat = data_grp['RADIOMETRIC_SATURATION'][f_idx, ...] != RADSAT_ACCEPT_VALUE
        kernel = np.ones((3, 3), dtype=bool)
        bad_radsat = ndimage.binary_dilation(bad_radsat, structure=kernel, iterations=1)
        valid_mask &= ~bad_radsat

    if 'QUALITY_L2_AEROSOL' in data_grp and AEROSOL_ACCEPT_LEVEL != 'all':
        aerosol = data_grp['QUALITY_L2_AEROSOL'][f_idx, ...]
        invalid_aerosol = ~np.isin(aerosol, AEROSOL_DICT[AEROSOL_ACCEPT_LEVEL])
        kernel = np.ones((3, 3), dtype=bool)
        invalid_aerosol = ndimage.binary_dilation(invalid_aerosol, structure=kernel, iterations=1)
        valid_mask &= ~invalid_aerosol

    return valid_mask

def get_tanager_mask(data_grp, f_idx, shape):
    """Generates a boolean mask for TANAGER data based on configurations."""
    valid_mask = np.ones(shape, dtype=bool)

    if 'beta_cloud_mask' in data_grp:
        cloud_mask = (data_grp['beta_cloud_mask'][f_idx, ...]==1)
        cirrus_mask = (data_grp['beta_cirrus_mask'][f_idx, ...]==1)
        combined_cloud = cloud_mask | cirrus_mask
        if CLOUD_DILATION > 0:
            kernel = np.ones((3, 3), dtype=bool)
            combined_cloud = ndimage.binary_dilation(combined_cloud, structure=kernel, iterations=CLOUD_DILATION)
        valid_mask &= ~combined_cloud
    
    if 'sun_zenith' in data_grp:
        zenith = data_grp['sun_zenith'][f_idx, ...]
        valid_mask &= (zenith != -9999.0) & ((90.0 - zenith) >= SUN_ELEVATION_THRESHOLD)
        
    if 'aerosol_optical_depth' in data_grp:
        aod = data_grp['aerosol_optical_depth'][f_idx, ...]
        bad_aod_mask = (aod == -9999.0) | (aod <= TANAGER_AEROSOL_DEPTH_THRESHOLD)
        if TANAGER_AEROSOL_DEPTH_THRESHOLD > 0:
            kernel = np.ones((3, 3), dtype=bool)
            bad_aod_mask = ndimage.binary_dilation(bad_aod_mask, structure=kernel, iterations=1)
        valid_mask &= ~bad_aod_mask
        
    if 'surface_reflectance_uncertainty' in data_grp:
        gw_mask = data_grp['surface_reflectance'].attrs.get('all_good_wavelengths')
        if gw_mask is not None:
            valid_bands = gw_mask[f_idx].astype(bool)
            unc = np.nanmax(data_grp['surface_reflectance_uncertainty'][f_idx, valid_bands, ...], axis=0)
        else:
            unc = np.nanmax(data_grp['surface_reflectance_uncertainty'][f_idx, ...], axis=0)
            
        unc_mask = (unc == -9999.0) | (unc >= TANAGER_SR_UNCERTAINTY_THRESHOLD)
        if TANAGER_SR_UNCERTAINTY_THRESHOLD > 0:
            kernel = np.ones((3, 3), dtype=bool)
            unc_mask = ndimage.binary_dilation(unc_mask, structure=kernel, iterations=1)
        valid_mask &= ~unc_mask
        
    return valid_mask

# ==========================================
# 3. DATA EXTRACTION
# ==========================================

def extract_data():
    files = [landsat_path, tanager_path]
    
    # Initialize Data Structure
    ts_data = {
        'LANDSAT': {loc['label']: {'t': [], 'primary': [], 'secondary': []} for loc in TS_LOCATIONS},
        'TANAGER': {loc['label']: {'t': [], 'primary': [], 'secondary': []} for loc in TS_LOCATIONS}
    }
    
    for filepath in files:
        if not os.path.exists(filepath):
            print(f"Warning: File not found: {filepath}")
            continue
            
        print(f"Processing {os.path.basename(filepath)}...")
        with h5py.File(filepath, 'r') as h5:
            source_name = list(h5['/HDFEOS/GRIDS'].keys())[0]
            key = 'LANDSAT' if 'LANDSAT' in source_name.upper() else 'TANAGER'
            data_grp = h5[f'HDFEOS/GRIDS/{source_name}/Data Fields']
            
            # 3.1 Verify Datasets Exist
            if complexity_type not in data_grp:
                print(f"  -> Skipping: {complexity_type} not found.")
                continue
            
            prim_dset = data_grp[complexity_type]
            sec_dset = data_grp[secondary_metric] if secondary_metric in data_grp else None
            if sec_dset is None:
                print(f"  -> Warning: {secondary_metric} not found in this file.")
            
            # 3.2 Map Geocoordinates to Pixels (using the first frame's metadata)
            sr_dset = data_grp['surface_reflectance']
            geo_transform = sr_dset.attrs.get('GeoTransform')
            spatial_ref = sr_dset.attrs.get('spatial_ref')
            
            mapped_locations = []
            if geo_transform is not None and spatial_ref is not None:
                if isinstance(spatial_ref, bytes):
                    spatial_ref = spatial_ref.decode('utf-8')
                crs = CRS.from_wkt(spatial_ref)
                transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
                affine = rasterio.transform.Affine(*geo_transform)
                inv_affine = ~affine
                
                for loc in TS_LOCATIONS:
                    lat, lon = loc['latlon']
                    px, py = inv_affine * transformer.transform(lon, lat)
                    mapped_locations.append({'label': loc['label'], 'y': int(round(py)), 'x': int(round(px))})
            else:
                print("  -> Error: Georeferencing metadata missing.")
                continue
            
            # 3.3 Extract Time Series
            acq_times = sr_dset.attrs.get('acquisition_time')
            num_frames = prim_dset.shape[0]
            
            for f_idx in range(num_frames):
                dt = datetime.fromtimestamp(acq_times[f_idx], tz=timezone.utc)
                if not (TS_START_DATE <= dt <= TS_END_DATE):
                    continue
                
                shape = prim_dset[f_idx].shape
                mask = get_landsat_mask(data_grp, f_idx, shape) if key == 'LANDSAT' else get_tanager_mask(data_grp, f_idx, shape)
                
                for loc in mapped_locations:
                    y, x = loc['y'], loc['x']
                    # Verify array bounds
                    if 0 <= y < shape[0] and 0 <= x < shape[1]:
                        if mask[y, x]:
                            prim_val = prim_dset[f_idx, y, x]
                            sec_val = sec_dset[f_idx, y, x] if sec_dset is not None else np.nan
                            
                            if not np.isnan(prim_val):
                                ts_data[key][loc['label']]['t'].append(dt)
                                ts_data[key][loc['label']]['primary'].append(prim_val)
                                ts_data[key][loc['label']]['secondary'].append(sec_val)

    return ts_data

# ==========================================
# 4. PLOTTING
# ==========================================

def plot_static_time_series(ts_data):
    fig, ax_prim = plt.subplots(figsize=(16, 8))
    ax_sec = ax_prim.twinx()
    
    # Store legend elements
    legend_elements_prim = []
    legend_elements_sec = []
    
    print("Plotting data...")
    for loc in TS_LOCATIONS:
        label = loc['label']
        color = loc['color']
        
        # Plot Landsat (Triangles)
        l_data = ts_data['LANDSAT'][label]
        if l_data['t']:
            # Primary Metric (Solid Line)
            line_p, = ax_prim.plot(l_data['t'], l_data['primary'], marker='^', color=color, 
                                   linestyle='-', linewidth=1.5, markersize=5, alpha=0.8,
                                   label=f"L: {label} ({complexity_type})")
            legend_elements_prim.append(line_p)
            
            # Secondary Metric (Dashed Line)
            # Filter out NaNs if the secondary metric was missing
            valid_sec = [(t, v) for t, v in zip(l_data['t'], l_data['secondary']) if not np.isnan(v)]
            if valid_sec:
                t_sec, v_sec = zip(*valid_sec)
                line_s, = ax_sec.plot(t_sec, v_sec, marker='^', color=color, 
                                      linestyle='--', linewidth=1.5, markersize=5, alpha=0.6,
                                      label=f"L: {label} ({secondary_metric})")
                legend_elements_sec.append(line_s)
                
        # Plot Tanager (Squares)
        t_data = ts_data['TANAGER'][label]
        if t_data['t']:
            # Primary Metric (Solid Line)
            line_p, = ax_prim.plot(t_data['t'], t_data['primary'], marker='s', color=color, 
                                   linestyle='-', linewidth=1.5, markersize=6, alpha=0.9,
                                   label=f"T: {label} ({complexity_type})")
            legend_elements_prim.append(line_p)
            
            # Secondary Metric (Dashed Line)
            valid_sec = [(t, v) for t, v in zip(t_data['t'], t_data['secondary']) if not np.isnan(v)]
            if valid_sec:
                t_sec, v_sec = zip(*valid_sec)
                line_s, = ax_sec.plot(t_sec, v_sec, marker='s', color=color, 
                                      linestyle='--', linewidth=1.5, markersize=6, alpha=0.7,
                                      label=f"T: {label} ({secondary_metric})")
                legend_elements_sec.append(line_s)

    # --- Formatting & Background Shading ---
    if legend_elements_prim:
        xlims = ax_prim.get_xlim()
        for yr in range(TS_START_DATE.year, TS_END_DATE.year + 2):
            # Winter -> light gray
            ax_prim.axvspan(datetime(yr - 1, 12, 1, tzinfo=timezone.utc), datetime(yr, 3, 1, tzinfo=timezone.utc), color='lightgray', alpha=0.3, zorder=0, lw=0)
            # Spring -> light green
            ax_prim.axvspan(datetime(yr, 3, 1, tzinfo=timezone.utc), datetime(yr, 6, 1, tzinfo=timezone.utc), color='lightgreen', alpha=0.2, zorder=0, lw=0)
            # Summer -> light yellow
            ax_prim.axvspan(datetime(yr, 6, 1, tzinfo=timezone.utc), datetime(yr, 9, 1, tzinfo=timezone.utc), color='lightyellow', alpha=0.3, zorder=0, lw=0)
            # Fall -> light orange
            ax_prim.axvspan(datetime(yr, 9, 1, tzinfo=timezone.utc), datetime(yr, 12, 1, tzinfo=timezone.utc), color='orange', alpha=0.15, zorder=0, lw=0)
        ax_prim.set_xlim(xlims)

    # Axis Labels & Titles
    ax_prim.set_title(f"Twin Axis Time Series\nPrimary: {complexity_type} | Secondary: {secondary_metric}\n({TS_START_DATE.strftime('%Y-%m-%d')} to {TS_END_DATE.strftime('%Y-%m-%d')})", fontsize=14)
    
    ax_prim.set_ylabel(f"Primary Metric ({complexity_type})", fontweight='bold', fontsize=11)
    ax_sec.set_ylabel(f"Secondary Metric ({secondary_metric})", fontweight='bold', fontsize=11, color='gray')
    
    ax_prim.grid(True, alpha=0.4, which="major", ls="-")
    ax_prim.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax_prim.tick_params(axis='x', rotation=45, labelsize=10)
    
    # Custom Legend Configuration
    # To keep it clean, we group the legends.
    if legend_elements_prim and legend_elements_sec:
        # Create a unified legend to avoid overlapping boxes
        ax_prim.legend(legend_elements_prim + legend_elements_sec, 
                       [l.get_label() for l in legend_elements_prim] + [l.get_label() for l in legend_elements_sec], 
                       loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=9, title="Location & Metric Mapping")
    elif legend_elements_prim:
        ax_prim.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=9)

    plt.tight_layout()
    
    # Save the static plot
    out_dir = "C:/satelliteImagery/MultiSensor_Analysis_Outputs"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"Static_TimeSeries_{complexity_type}_vs_{secondary_metric}.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved figure to: {out_path}")
    
    plt.show()

if __name__ == "__main__":
    extracted_data = extract_data()
    plot_static_time_series(extracted_data)