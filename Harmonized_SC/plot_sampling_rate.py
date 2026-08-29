import argparse
import os
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import sys
from pathlib import Path
script_dir = Path(__file__).resolve().parent
if str(script_dir.parent) not in sys.path:
    sys.path.insert(0, str(script_dir.parent))
import SpecComplex

LOCATION = 'SanRafael'

SPACECRAFT_STYLE = {
    'Landsat-8': {'color': '#E69F00', 'marker': '^'},
    'Landsat-9': {'color': '#D55E00', 'marker': 'v'},
    'HLSL30': {'color': '#E69F00', 'marker': '^'},
    'Sentinel-2A': {'color': '#009E73', 'marker': 'o'},
    'Sentinel-2B': {'color': '#56B4E9', 'marker': 's'},
    'Sentinel-2C': {'color': '#0072B2', 'marker': 'p'},
    'HLSS30': {'color': '#009E73', 'marker': 'o'},
    'Tanager-1': {'color': '#CC79A7', 'marker': 'D'},
    'Tanager': {'color': '#CC79A7', 'marker': 'D'},
    'TANAGER': {'color': '#CC79A7', 'marker': 'D'},
    'EnMAP': {'color': '#F0E442', 'marker': 'X'},
    'ENMAP': {'color': '#F0E442', 'marker': 'X'},
    'Dragonette-001': {'color': '#882255', 'marker': 'P'},
    'Dragonette': {'color': '#882255', 'marker': 'P'},
    'DRAGONETTE': {'color': '#882255', 'marker': 'P'},
    'Wyvern': {'color': '#882255', 'marker': 'P'},
}

# Fallback palettes
FALLBACK_COLORS = ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7', '#882255', '#44AA99', '#117733']
FALLBACK_MARKERS = ['o', 's', '^', 'D', 'p', 'v', 'X', 'P', '<', '>']


def analyze_sampling_rate(h5_path, output_plot_path=None, max_masked_ratio=0.60):
    print(f"Analyzing {h5_path}...")
    
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"File not found: {h5_path}")
        
    with h5py.File(h5_path, 'r') as f:
        # Load the common mask and acquisition times
        mask_dset = f['HDFEOS/GRIDS/HARMONIZED/Data Fields/common_mask']
        mask = SpecComplex.read_scaled_int16(mask_dset)
        times = mask_dset.attrs['acquisition_time']
        
        # Load source spacecraft or source grid if available
        if 'source_spacecraft' in mask_dset.attrs:
            spacecraft_arr = mask_dset.attrs['source_spacecraft']
        elif 'source_spacecraft' in f.get('HDFEOS/GRIDS/HARMONIZED/Data Fields/surface_reflectance', {}).attrs:
            spacecraft_arr = f['HDFEOS/GRIDS/HARMONIZED/Data Fields/surface_reflectance'].attrs['source_spacecraft']
        elif 'source_grid' in mask_dset.attrs:
            spacecraft_arr = mask_dset.attrs['source_grid']
        else:
            spacecraft_arr = None
        
    num_frames = mask.shape[0]
    total_pixels_per_frame = mask.shape[1] * mask.shape[2]
    
    valid_times = []
    valid_spacecrafts = []
    
    for i in range(num_frames):
        # Calculate percentage of masked pixels. Assuming >0 means masked.
        masked_pixels = np.count_nonzero(mask[i])
        masked_ratio = masked_pixels / total_pixels_per_frame
        
        if masked_ratio <= max_masked_ratio:
            valid_times.append(times[i])
            if spacecraft_arr is not None and i < len(spacecraft_arr):
                sc = spacecraft_arr[i]
                sc_str = sc.decode('utf-8') if isinstance(sc, bytes) else str(sc)
            else:
                sc_str = "Unknown"
            valid_spacecrafts.append(sc_str)
            
    # Convert epoch times to pandas datetime
    datetimes = pd.to_datetime(valid_times, unit='s')
    
    # Create DataFrame and compute time differences
    df = pd.DataFrame({
        'datetime': datetimes,
        'spacecraft': valid_spacecrafts
    })
    df = df.sort_values('datetime').reset_index(drop=True)
    df['year'] = df['datetime'].dt.year
    df['delta_days'] = df['datetime'].diff().dt.total_seconds() / (24 * 3600)
    
    stats_list = []
    
    print("\nSummary Statistics per Year:")
    print("-" * 65)
    print(f"{'Year':<10} | {'Mean Rate (days)':<20} | {'Std Dev (days)':<20} | {'Valid Frames':<15}")
    print("-" * 65)
    
    years = np.sort(df['year'].unique())
    for year in years:
        year_df = df[df['year'] == year]
        deltas = year_df['delta_days'].dropna()
        
        if len(deltas) > 0:
            mean_rate = deltas.mean()
            std_rate = deltas.std()
        else:
            mean_rate = np.nan
            std_rate = np.nan
            
        print(f"{year:<10} | {mean_rate:<20.4f} | {std_rate:<20.4f} | {len(year_df):<15}")
        
        stats_list.append({
            'Year': year,
            'Mean': mean_rate,
            'StdDev': std_rate,
            'Count': len(year_df)
        })
        
    stats_df = pd.DataFrame(stats_list, columns=['Year', 'Mean', 'StdDev', 'Count'])
    valid_stats = stats_df.dropna(subset=['Mean'])
    
    # Print summary statistics per sensor
    unique_scs = np.sort(df['spacecraft'].unique())
    print("\nSummary Statistics per Observation Source / Spacecraft:")
    print("-" * 75)
    print(f"{'Spacecraft / Sensor':<25} | {'Mean Rate (days)':<18} | {'Median (days)':<15} | {'Valid Frames':<15}")
    print("-" * 75)
    for sc in unique_scs:
        sc_df = df[df['spacecraft'] == sc]
        deltas = sc_df['delta_days'].dropna()
        m_rate = deltas.mean() if len(deltas) > 0 else np.nan
        med_rate = deltas.median() if len(deltas) > 0 else np.nan
        print(f"{sc:<25} | {m_rate:<18.4f} | {med_rate:<15.4f} | {len(sc_df):<15}")
    print("-" * 75)
    
    # Write to METADATA group in HDF5
    print("\nWriting statistics to METADATA group in HDF5...")
    try:
        with h5py.File(h5_path, 'r+') as f:
            if 'METADATA' not in f:
                meta_grp = f.create_group('METADATA')
            else:
                meta_grp = f['METADATA']
                
            if 'Sampling_Statistics' in meta_grp:
                del meta_grp['Sampling_Statistics']
                
            if len(stats_df) > 0:
                stats_grp = meta_grp.create_group('Sampling_Statistics')
                stats_grp.attrs['description'] = 'Summary statistics per year for sampling rate (days)'
                stats_grp.create_dataset('Year', data=stats_df['Year'].values.astype('int32'), dtype='int32')
                stats_grp.create_dataset('Mean_Rate_Days', data=stats_df['Mean'].values.astype('float32'), dtype='float32')
                stats_grp.create_dataset('Std_Dev_Days', data=stats_df['StdDev'].values.astype('float32'), dtype='float32')
                stats_grp.create_dataset('Valid_Frames_Count', data=stats_df['Count'].values.astype('int32'), dtype='int32')
    except Exception as e:
        print(f"Warning: Could not write Sampling_Statistics to HDF5 METADATA: {e}")
    
    # Create plot with GridSpec to make room for text box
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, width_ratios=[3, 1])
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax_text = fig.add_subplot(gs[:, 1])
    ax_text.axis('off')
    
    # Top plot: scatter of all deltas over time colored and styled by spacecraft / sensor
    for i, sc in enumerate(unique_scs):
        sc_df = df[df['spacecraft'] == sc]
        style = SPACECRAFT_STYLE.get(sc, {
            'color': FALLBACK_COLORS[i % len(FALLBACK_COLORS)],
            'marker': FALLBACK_MARKERS[i % len(FALLBACK_MARKERS)]
        })
        ax1.scatter(
            sc_df['datetime'], sc_df['delta_days'],
            label=sc,
            color=style['color'],
            marker=style['marker'],
            alpha=0.85,
            edgecolors='k',
            linewidths=0.5,
            s=35
        )
        
    ax1.set_title('Sampling Interval over Time', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Days Between Valid Observations', fontsize=10)
    if len(unique_scs) > 0:
        ax1.legend(
            title="Observation Source",
            loc='upper right',
            prop={'size': 8},
            title_fontsize=9,
            ncol=min(3, max(1, len(unique_scs))),
            framealpha=0.9
        )
    ax1.grid(True, linestyle='--', alpha=0.5)
    if len(df) > 0:
        ax1.xaxis.set_major_locator(mdates.YearLocator())
    
    # Bottom plot: Bar chart of yearly means
    if len(valid_stats) > 0:
        yerr = valid_stats['StdDev'].fillna(0)
        ax2.bar(valid_stats['Year'], valid_stats['Mean'], yerr=yerr, 
                capsize=5, color='skyblue', edgecolor='black', alpha=0.7)
        ax2.set_xticks(valid_stats['Year'])
    
    ax2.set_title('Average Sampling Rate per Year (with Std Dev Error Bars)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Year', fontsize=10)
    ax2.set_ylabel('Mean Sampling Rate (Days)', fontsize=10)
    ax2.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Text box for summary statistics
    header = f"{'Year':<6} | {'Mean':<8} | {'Std Dev':<8} | {'Count':<6}"
    separator = "-" * len(header)
    table_str = header + "\n" + separator + "\n"
    
    for _, row in stats_df.iterrows():
        y_val = int(row['Year'])
        m_val = f"{row['Mean']:.1f}" if pd.notna(row['Mean']) else "NaN"
        s_val = f"{row['StdDev']:.1f}" if pd.notna(row['StdDev']) else "NaN"
        c_val = int(row['Count'])
        table_str += f"{y_val:<6} | {m_val:<8} | {s_val:<8} | {c_val:<6}\n"
        
    props = dict(boxstyle='round', facecolor='whitesmoke', alpha=0.8)
    ax_text.text(0.05, 0.5, "Summary Statistics\n\n" + table_str, 
                 fontsize=11, fontfamily='monospace', 
                 verticalalignment='center', bbox=props)
    
    if output_plot_path is None:
        output_dir = os.path.dirname(h5_path)
        base_name = os.path.splitext(os.path.basename(h5_path))[0]
        output_plot = os.path.join(output_dir, f"{base_name}_sampling_rate.png")
    else:
        output_plot = output_plot_path
        os.makedirs(os.path.dirname(output_plot), exist_ok=True)
    
    plt.tight_layout()
    plt.savefig(output_plot, dpi=500)
    print(f"\nPlot saved to: {output_plot}")
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Analyze sampling rate of HDF5 Harmonized dataset.")
    parser.add_argument('--file', '-f', default=f"C:/satelliteImagery/MGRS30mConstellation/Harmonized_MGRS_Stack_{LOCATION}_SC_EM-7_Norm-None.h5", 
                        help="Path to the HDF5 file.")
    parser.add_argument('--output', '-o', default=None, help="Optional path to output plot image file.")
    parser.add_argument('--cloud-thresh', '-c', type=float, default=0.60, 
                        help="Maximum allowed masked/cloud ratio per frame (default: 0.60).")
    args = parser.parse_args()
    
    analyze_sampling_rate(args.file, output_plot_path=args.output, max_masked_ratio=args.cloud_thresh)

