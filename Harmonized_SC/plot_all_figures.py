import os
import sys

# Windows WMI Hang Bypass (MUST occur before any other imports)
if 'PROCESSOR_IDENTIFIER' not in os.environ:
    os.environ['PROCESSOR_IDENTIFIER'] = 'Bypass WMI'
import platform
def _dummy_wmi_query(*args, **kwargs):
    raise OSError("WMI disabled to prevent hangs")
platform._wmi_query = _dummy_wmi_query

import h5py
import tkinter as tk
from tkinter import filedialog
from pathlib import Path

# Fix path to allow importing sibling modules when run directly
script_dir = Path(__file__).resolve().parent
if str(script_dir.parent) not in sys.path:
    sys.path.insert(0, str(script_dir.parent))

import Harmonized_SC.plot_sampling_rate as plot_sampling_rate
import Harmonized_SC.plot_water_mask as plot_water_mask
import Harmonized_SC.plot_sliding_volume_global_stats as plot_sliding_volume_global_stats
import Harmonized_SC.plot_SpecComplex_cross_sensor_correlation as plot_SpecComplex_cross_sensor_correlation
import Harmonized_SC.plot_registration_quality_multisensor as plot_registration_quality_multisensor

def generate_all_plots(h5_path: str):
    """
    Generate all summary figures for the provided HDF5 pipeline output.
    """
    if not os.path.exists(h5_path):
        print(f"Error: File not found at {h5_path}")
        return

    # Try to extract a location name from the filename for plot_water_mask
    filename = os.path.basename(h5_path)
    target_location = "UnknownLocation"
    if "Harmonized_MGRS_Stack_" in filename:
        target_location = filename.split("Harmonized_MGRS_Stack_")[1].split("_SC_")[0]
    elif "HLST_" in filename:
        target_location = filename.split("HLST_")[1].split("_Harmonized")[0]

    print(f"Generating all summary plots for: {target_location}")
    print(f"Source Data: {h5_path}")
    print("=" * 60)

    print("\n1. Generating Sampling Rate Plot...")
    try:
        plot_sampling_rate.analyze_sampling_rate(h5_path=h5_path)
    except Exception as e:
        print(f" -> Error: {e}")

    print("\n2. Generating Water Mask Plot...")
    try:
        plot_water_mask.main(h5_path=h5_path)
    except Exception as e:
        print(f" -> Error: {e}")

    print("\n3. Generating Sliding Volume Global Stats Plots...")
    try:
        with h5py.File(h5_path, 'r') as h5f:
            if '/HDFEOS/GRIDS/HARMONIZED/Data Fields' in h5f:
                data_fields = list(h5f['/HDFEOS/GRIDS/HARMONIZED/Data Fields'].keys())
            else:
                data_fields = []
                
        metrics_to_plot = []
        if 'sliding_volume_z_score' in data_fields: metrics_to_plot.append('zscore')
        if 'sliding_volume_robust_scale' in data_fields: metrics_to_plot.append('robust')
        if 'sliding_volume_box_cox' in data_fields: metrics_to_plot.append('box_cox')

        for m in metrics_to_plot:
            print(f"  -> Plotting metric: {m}")
            plot_sliding_volume_global_stats.plot_global_stats(h5_path=h5_path, metric=m)
    except Exception as e:
        print(f" -> Error: {e}")

    print("\n4. Generating Cross-Sensor Correlation Summary...")
    try:
        corr_datasets = []
        if 'sliding_volume_z_score' in data_fields: corr_datasets.append('sliding_volume_z_score')
        if 'sliding_volume_robust_scale' in data_fields: corr_datasets.append('sliding_volume_robust_scale')
        if 'sliding_volume_box_cox' in data_fields: corr_datasets.append('sliding_volume_box_cox')
        
        for dset in corr_datasets:
            print(f"  -> Plotting correlation for: {dset}")
            plot_SpecComplex_cross_sensor_correlation.plot_cross_sensor_correlations(h5_path=h5_path, target_dataset=dset)
    except Exception as e:
        print(f" -> Error: {e}")

    print("\n5. Generating Registration Quality Summary...")
    try:
        plot_registration_quality_multisensor.analyze_and_plot_registration_quality(h5_path=h5_path)
    except Exception as e:
        print(f" -> Error: {e}")

    print("\n" + "=" * 60)
    print("All plotting operations completed.")


if __name__ == "__main__":
    # Hide the main tkinter window
    root = tk.Tk()
    root.withdraw()
    
    print("Please select the processed HLST/MGRS ARD HDF5 Cube...")
    file_path = tk.filedialog.askopenfilename(
        title="Select Pipeline HDF5 Cube",
        filetypes=[("HDF5 files", "*.h5"), ("All files", "*.*")],
        initialdir=r"C:\satelliteImagery\MGRS30mConstellation"
    )
    
    if not file_path:
        print("Execution cancelled by user.")
        sys.exit(0)
        
    generate_all_plots(file_path)
