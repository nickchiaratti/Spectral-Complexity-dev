import os
import h5py
import numpy as np
import itertools
import csv
from harmonized_CCD_main import main as run_pipeline

LOCATION = "Hurlingham"

def main():
    base_periods = [1.0, 0.5, 0.33, 0.25, 0.1]#0.75, 0.66, 0.1]
    
    # Generate all subsets of base_periods
    period_subsets = []
    for r in range(1, len(base_periods) + 1):
        period_subsets.extend(list(itertools.combinations(base_periods, r)))
        
    booleans = [True, False]
    
    results = []
    
    print("Starting Comparative Analysis Orchestrator...")
    
    configs = []
    for c in booleans:
        for l in booleans:
            for q in booleans:
                for e in booleans:
                    for periods in period_subsets:
                        periods = list(periods)
                        num_trend_terms = int(c) + int(l) + int(q)
                        total_terms = num_trend_terms + len(periods) * 2
                        if 6 <= total_terms <= 10:
                            config_name = f"C{int(c)}L{int(l)}Q{int(q)}_P{len(periods)}_E{int(e)}"
                            configs.append((c, l, q, e, periods, total_terms, config_name))
                            
    print(f"Total configurations to evaluate: {len(configs)}")
    
    # Phase 1: Execution
    print("\n--- Phase 1: Pipeline Execution ---")
    for c, l, q, e, periods, total_terms, config_name in configs:
        expected_h5 = f"C:/satelliteImagery/HLST30/CCD/{LOCATION}_CCD_Harmonized_Change_Detection_{config_name}.h5"
        if not os.path.exists(expected_h5):
            print(f"\nExecuting Configuration: {config_name} (Terms: {total_terms})")
            print(f"Periods: {periods}")
            run_pipeline(enable_const=c, enable_lin=l, enable_quad=q, 
                         temporal_periods=periods, enable_elastic_window=e, launch_vis=False)
        else:
            print(f"File {expected_h5} already exists. Skipping pipeline execution.")

    # Phase 2: Metric Extraction
    print("\n--- Phase 2: Metric Extraction ---")
    for c, l, q, e, periods, total_terms, config_name in configs:
        output_h5 = f"C:/satelliteImagery/HLST30/CCD/{LOCATION}_CCD_Harmonized_Change_Detection_{config_name}.h5"
        if os.path.exists(output_h5):
            print(f"Extracting metrics for {config_name}...")
            try:
                with h5py.File(output_h5, 'r') as f:
                    rmse_series = f['rmse_series'][:]
                    rmse_mult = f.attrs.get('RMSE_MULTIPLIER', 3.0)
                    
                    # bounds size = 2 * multiplier * rmse
                    bounds_series = 2.0 * rmse_mult * rmse_series
                    
                    median_bound = np.nanmedian(bounds_series)
                    mean_bound = np.nanmean(bounds_series)
                    max_bound = np.nanmax(bounds_series)
                    
                    anomaly_flags = f['anomaly_flags'][:]
                    change_count = f['change_count'][:]
                    
                    valid_st_mask = ~np.isnan(rmse_series)
                    num_valid_st = np.sum(valid_st_mask)
                    st_anomaly_pct = (np.sum(anomaly_flags[valid_st_mask]) / num_valid_st * 100.0) if num_valid_st > 0 else np.nan
                    
                    valid_spatial_mask = ~np.all(np.isnan(rmse_series), axis=0)
                    num_valid_spatial = np.sum(valid_spatial_mask)
                    spatial_anomaly_pct = (np.sum((change_count > 0) & valid_spatial_mask) / num_valid_spatial * 100.0) if num_valid_spatial > 0 else np.nan

                    print(f"  Extracted metrics -> Global Median: {median_bound:.5f} | Spatial Anomaly: {spatial_anomaly_pct:.2f}%")
                    
                    row_data = {
                        'Config': config_name,
                        'Constant': c,
                        'Linear': l,
                        'Quadratic': q,
                        'Elastic_Window': e,
                        'Periods': str(periods),
                        'Total_Terms': total_terms,
                        'Global_Median_Bound': median_bound,
                        'Global_Mean_Bound': mean_bound,
                        'Global_Max_Bound': max_bound,
                        'ST_Anomaly_Pct': st_anomaly_pct,
                        'Spatial_Anomaly_Pct': spatial_anomaly_pct
                    }
                    
                    pixels_of_interest = [(78, 28), (131, 69), (33, 73), (29, 65)]
                    for py, px in pixels_of_interest:
                        if py < bounds_series.shape[1] and px < bounds_series.shape[2]:
                            pixel_ts = bounds_series[:, py, px]
                            if np.all(np.isnan(pixel_ts)):
                                p_med, p_mean, p_max = np.nan, np.nan, np.nan
                            else:
                                p_med = np.nanmedian(pixel_ts)
                                p_mean = np.nanmean(pixel_ts)
                                p_max = np.nanmax(pixel_ts)
                            row_data[f'Px_{py}_{px}_Median'] = p_med
                            row_data[f'Px_{py}_{px}_Mean'] = p_mean
                            row_data[f'Px_{py}_{px}_Max'] = p_max
                        else:
                            row_data[f'Px_{py}_{px}_Median'] = np.nan
                            row_data[f'Px_{py}_{px}_Mean'] = np.nan
                            row_data[f'Px_{py}_{px}_Max'] = np.nan
                            
                    results.append(row_data)
            except Exception as ex:
                print(f"Failed to read metrics from {output_h5}: {ex}")

    # Save results to CSV
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, f"{LOCATION}_comparative_analysis_results.csv")
    if results:
        keys = results[0].keys()
        with open(csv_path, 'w', newline='') as output_file:
            dict_writer = csv.DictWriter(output_file, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(results)
        print(f"\nResults saved to absolute path:\n  {csv_path}")
        
        # Print summary table
        print(f"\n=== Top 15 Configurations by Lowest Global Median Bound ===")
        print(f"{'Config':<20} | {'Terms':<6} | {'Glob Med':<10} | {'Glob Mean':<10} | {'Anom Pct':<10} | {'P(78,28) Med':<12}")
        print("-" * 81)
        # Sort by global median bound ascending
        results.sort(key=lambda x: x['Global_Median_Bound'] if not np.isnan(x['Global_Median_Bound']) else float('inf'))
        for r in results[:15]:
            gm = r.get('Global_Median_Bound', np.nan)
            gmean = r.get('Global_Mean_Bound', np.nan)
            apct = r.get('Spatial_Anomaly_Pct', np.nan)
            p78 = r.get('Px_78_28_Median', np.nan)
            print(f"{r['Config']:<20} | {r['Total_Terms']:<6} | {gm:<10.5f} | {gmean:<10.5f} | {apct:<10.2f} | {p78:<12.5f}")
            
        if len(results) > 15:
            print(f"... and {len(results) - 15} more.")
            print(f"Please open the CSV file to review all {len(results)} configurations.")
            
if __name__ == "__main__":
    main()
