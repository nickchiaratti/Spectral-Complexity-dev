import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys

# Windows WMI Hang Bypass (MUST occur before any other imports)
# This prevents h5py/platform from hanging when multiprocessing spawns child processes
if 'PROCESSOR_IDENTIFIER' not in os.environ:
    os.environ['PROCESSOR_IDENTIFIER'] = 'Bypass WMI'
import platform
def _dummy_wmi_query(*args, **kwargs):
    raise OSError("WMI disabled to prevent hangs")
platform._wmi_query = _dummy_wmi_query

import yaml
import subprocess
from pathlib import Path

import HLS30.HLS30_earthAccess_to_hdf5 as HLS30_earthAccess_to_hdf5
import Harmonized_SC.HLST_constellation_to_hdf5 as HLST_constellation_to_hdf5
import Harmonized_SC.HLST_SC_calculations as HLST_SC_calculations
import Harmonized_SC.HLST_specComplex_viewer as HLST_specComplex_viewer
import Harmonized_SC.plot_sampling_rate as plot_sampling_rate
import Harmonized_SC.plot_water_mask as plot_water_mask

# ==========================================
# PIPELINE CONFIGURATION
# ==========================================
script_dir = Path(__file__).resolve().parent
CONFIG_FILE_PATH = os.path.join(script_dir, "locations_config.yaml")
SATELLITE_DATA_DIR = "C:/satelliteImagery/HLST30"

def load_config(config_path="locations_config.yaml", location=None):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if location is None:
        location = config.get("current_run", {}).get("location")
    
    if not location or location not in config.get("locations", {}):
        print(f"Error: Location '{location}' not found in configuration.")
        sys.exit(1)
        
    loc_config = config["locations"][location]
    loc_config["LocationName"] = location
    return loc_config

import argparse

def main():
    parser = argparse.ArgumentParser(description="Run Harmonized SC Pipeline")
    parser.add_argument('--location', type=str, default=None,
                        help='Target location to process (e.g., Rochesterv2, Malibu). Overrides current_run in yaml.')
    parser.add_argument('--show-view', action='store_true',
                        help='Do not skip the interactive mgrs_view step')
    parser.add_argument('--show-spec-viewer', action='store_true',
                        help='Do not skip the interactive SpecComplex viewer step')
    parser.add_argument('--tile-size', type=int, default=3,
                        help='Tile size for spatial sliding window calculation')
    parser.add_argument('--num-endmembers', type=int, default=7,
                        help='Number of endmembers to extract')
    parser.add_argument('--norm-param', type=str, default='bandCount',
                        help='Normalization parameter (e.g. bandCount)')
    args = parser.parse_args()

    # Load configuration
    loc_config = load_config(CONFIG_FILE_PATH, args.location)
    location_name = loc_config["LocationName"]
    target_location = args.location if args.location else location_name
    print(f"Starting pipeline for location: {location_name}")    
    # Define the execution order
    steps = [
        ("mgrs_view.py", None),
        ("HLS30_earthAccess_to_hdf5", HLS30_earthAccess_to_hdf5.main),
        ("HLST_constellation_to_hdf5", HLST_constellation_to_hdf5.main),
        ("HLST_SC_calculations", HLST_SC_calculations.main),
        ("plot_sampling_rate", plot_sampling_rate.analyze_sampling_rate),
        ("plot_water_mask", plot_water_mask.main),
        ("HLST_specComplex_viewer", HLST_specComplex_viewer.main)
    ]

    for name, func in steps:
        print(f"\n{'='*50}")
        print(f"Executing: {name}")
        print(f"{'='*50}")
        
        if name == "mgrs_view.py":
            if not args.show_view:
                print("Skipping mgrs_view.py as requested.")
                continue
            else:
                script_path = os.path.join(script_dir, "HLSX30", "mgrs_view.py")
                print(f"Launching Streamlit interface for {script_path}...")
                print("Please interact with the browser window, then click 'Close App & Continue Pipeline' to proceed.")
                subprocess.run([sys.executable, "-m", "streamlit", "run", script_path], check=True)
                continue
        elif name == "HLST_SC_calculations":
            func(target_location=target_location, tile_size=args.tile_size, num_endmembers=args.num_endmembers, norm_param=args.norm_param)
            continue
        elif name == "plot_sampling_rate":
            print(f"Plotting sampling rate for {location_name}...")
            file_path = f"{SATELLITE_DATA_DIR}/HLST_{location_name}_Harmonized_SC_EM-{args.num_endmembers}_Norm-{args.norm_param}.h5"
            func(h5_path=file_path)
            continue
        elif name == "HLST_specComplex_viewer":
            if not args.show_spec_viewer:
                print("Skipping HLST_specComplex_viewer.")
                continue
                
            print(f"Launching SpecComplex Viewer for {location_name}...")
            # Compute file path based on default naming convention in calculation script
            file_path = f"{SATELLITE_DATA_DIR}/HLST_{location_name}_Harmonized_SC_EM-{args.num_endmembers}_Norm-{args.norm_param}.h5"
            
            # Extract start and end year from config
            start_date_str = loc_config.get("START_DATE", "2024-01-01")
            end_date_str = loc_config.get("END_DATE", "2026-01-01")
            start_year = int(start_date_str.split("-")[0])
            end_year = int(end_date_str.split("-")[0])
            
            func(target_location=target_location, file_path=file_path, start_year=start_year, end_year=end_year)
            continue
            
        func(target_location=target_location)

    print("\nPipeline execution completed successfully.")

if __name__ == "__main__":
    main()
