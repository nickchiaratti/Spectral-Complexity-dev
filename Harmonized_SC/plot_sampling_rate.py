import argparse
import os
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

LOCATION='Hurlingham'

def analyze_sampling_rate(h5_path):
    print(f"Analyzing {h5_path}...")
    
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"File not found: {h5_path}")
        
    with h5py.File(h5_path, 'r') as f:
        # Load the common mask and acquisition times
        mask = f['HDFEOS/GRIDS/HARMONIZED/Data Fields/common_mask'][:]
        times = f['HDFEOS/GRIDS/HARMONIZED/Data Fields/common_mask'].attrs['acquisition_time']
        
    num_frames = mask.shape[0]
    total_pixels_per_frame = mask.shape[1] * mask.shape[2]
    
    valid_times = []
    
    for i in range(num_frames):
        # Calculate percentage of masked pixels. Assuming >0 means masked.
        masked_pixels = np.count_nonzero(mask[i])
        masked_ratio = masked_pixels / total_pixels_per_frame
        
        if masked_ratio <= 0.60:
            valid_times.append(times[i])
            
    # Convert epoch times to pandas datetime
    datetimes = pd.to_datetime(valid_times, unit='s')
    
    # Create DataFrame and compute time differences
    df = pd.DataFrame({'datetime': datetimes})
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
        
    stats_df = pd.DataFrame(stats_list)
    valid_stats = stats_df.dropna(subset=['Mean'])
    
    # Write to METADATA group in HDF5
    print("\nWriting statistics to METADATA group in HDF5...")
    with h5py.File(h5_path, 'r+') as f:
        if 'METADATA' not in f:
            meta_grp = f.create_group('METADATA')
        else:
            meta_grp = f['METADATA']
            
        if 'Sampling_Statistics' in meta_grp:
            del meta_grp['Sampling_Statistics']
            
        stats_grp = meta_grp.create_group('Sampling_Statistics')
        stats_grp.attrs['description'] = 'Summary statistics per year for sampling rate (days)'
        stats_grp.create_dataset('Year', data=stats_df['Year'].values, dtype='int32')
        stats_grp.create_dataset('Mean_Rate_Days', data=stats_df['Mean'].values.astype('float32'))
        stats_grp.create_dataset('Std_Dev_Days', data=stats_df['StdDev'].values.astype('float32'))
        stats_grp.create_dataset('Valid_Frames_Count', data=stats_df['Count'].values, dtype='int32')
    
    # Create plot with GridSpec to make room for text box
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, width_ratios=[3, 1])
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax_text = fig.add_subplot(gs[:, 1])
    ax_text.axis('off')
    
    # Top plot: scatter of all deltas over time
    ax1.scatter(df['datetime'], df['delta_days'], color='coral', alpha=0.7, edgecolors='k')
    ax1.set_title('Sampling Interval over Time')
    ax1.set_ylabel('Days between collections')
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # Bottom plot: Bar chart of yearly means
    yerr = valid_stats['StdDev'].fillna(0)
    ax2.bar(valid_stats['Year'], valid_stats['Mean'], yerr=yerr, 
            capsize=5, color='skyblue', edgecolor='black', alpha=0.7)
    
    ax2.set_title('Average Sampling Rate per Year (with Std Dev Error Bars)')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Mean Sampling Rate (Days)')
    ax2.set_xticks(valid_stats['Year'])
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
    
    output_dir = os.path.dirname(h5_path)
    base_name = os.path.splitext(os.path.basename(h5_path))[0]
    output_plot = os.path.join(output_dir, f"{base_name}_sampling_rate.png")
    
    plt.tight_layout()
    plt.savefig(output_plot, dpi=500)
    print(f"\nPlot saved to: {output_plot}")
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Analyze sampling rate of HDF5 Harmonized dataset.")
    parser.add_argument('--file', '-f', default=f"C:/satelliteImagery/HLST30/HLST_{LOCATION}_Harmonized_SC_EM-7_Norm-bandCount.h5", 
                        help="Path to the HDF5 file.")
    args = parser.parse_args()
    
    analyze_sampling_rate(args.file)
