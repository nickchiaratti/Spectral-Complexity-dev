import os
import h5py
import sys
import yaml
from pathlib import Path

script_dir = Path(__file__).resolve().parent

def main(target_location=None):
    if target_location is None:
        import argparse
        parser = argparse.ArgumentParser(description="Create a virtual HDF5 constellation file mapping MGRS stacks.")
        parser.add_argument('--location', type=str, default=None, help='Target location prefix')
        args = parser.parse_args()
        location_arg = args.location
    else:
        location_arg = target_location

    # Load configuration
    config_path = os.path.join(script_dir.parent, "locations_config.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if location_arg:
        location = location_arg
    else:
        location = config.get("current_run", {}).get("location")
        
    if not location or location not in config.get("locations", {}):
        print(f"CRITICAL: Location '{location}' not found in configuration.")
        sys.exit(1)

    print(f"Creating MGRS Virtual Constellation for: {location}")

    # Define paths
    out_dir = "C:/satelliteImagery/MGRS30mConstellation"
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"Harmonized_MGRS_Stack_{location}.h5")

    # Define potential sources
    sources = {
        'ENMAP': {
            'file': f"C:/satelliteImagery/enmap/EnMAP_MGRS_Stack_{location}.h5",
            'link_path': '/HDFEOS/GRIDS/ENMAP'
        },
        'TANAGER': {
            'file': f"C:/satelliteImagery/tanager/Tanager_MGRS_Stack_{location}.h5",
            'link_path': '/HDFEOS/GRIDS/TANAGER'
        },
        'HLSS30': {
            'file': f"C:/satelliteImagery/HLS30/HLS_{location}_MGRS_Stack.h5",
            'link_path': '/HDFEOS/GRIDS/HLSS30'
        },
        'HLSL30': {
            'file': f"C:/satelliteImagery/HLS30/HLS_{location}_MGRS_Stack.h5",
            'link_path': '/HDFEOS/GRIDS/HLSL30'
        },
        'DRAGONETTE': {
            'file': f"C:/satelliteImagery/dragonette/Dragonette_MGRS_Stack_{location}.h5",
            'link_path': '/HDFEOS/GRIDS/DRAGONETTE'
        }
    }

    # Create virtual linking HDF5
    with h5py.File(out_file, 'w') as h5_out:
        # Recreate expected hierarchy
        h5_out.create_group('HDFEOS')
        grids_grp = h5_out.create_group('HDFEOS/GRIDS')
        
        linked_count = 0
        for sensor, info in sources.items():
            src_file = info['file']
            if os.path.exists(src_file):
                print(f"  -> Mounting {sensor} from {src_file}")
                # HDFEOS/GRIDS/SENSOR links to the exact same group in the source file
                # Need to use an absolute path so h5py correctly locates the target
                grids_grp[sensor] = h5py.ExternalLink(src_file, info['link_path'])
                linked_count += 1
            else:
                print(f"  -> Skipping {sensor}: File not found ({src_file})")
                
        if linked_count == 0:
            print("WARNING: No MGRS Stack files were found. The constellation file is empty.")
            
    print(f"\nSuccessfully created virtual MGRS Constellation file:")
    print(f" -> {out_file}")

if __name__ == "__main__":
    main()
