import argparse
import os
import sys
import importlib
from pathlib import Path

# Add project root to sys.path so we can import from other directories
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import downloaders and stackers
from HLS30 import HLS30_earthAccess_to_MGRS_grid
from enmap import enmap_STAC_imagery_downloader, enmap_MGRS_stacker
from Dragonette import dragonette_STAC_imagery_downloader, dragonette_MGRS_stacker

# Since Tanager directory has a hyphen (Tanager-1), use importlib
Tanager_STAC_imagery_downloader = importlib.import_module("Tanager-1.Tanager_STAC-imagery-downloader")
tanager_MGRS_stacker = importlib.import_module("Tanager-1.tanager_MGRS_stacker")

def check_h5_exists(sensor, location):
    paths = {
        "HLS": f"C:/satelliteImagery/HLS30/HLS_{location}_MGRS_Stack.h5",
        "Tanager": f"C:/satelliteImagery/Tanager/Tanager_MGRS_Stack_{location}.h5",
        "EnMAP": f"C:/satelliteImagery/enmap/EnMAP_MGRS_Stack_{location}.h5",
        "Dragonette": rf"C:\satelliteImagery\dragonette\Dragonette_MGRS_Stack_{location}.h5"
    }
    path = paths.get(sensor)
    if path and os.path.exists(path):
        return True
    return False

import yaml

def main(target_location=None, force=False):
    if target_location is None:
        parser = argparse.ArgumentParser(description="MGRS Grid Constellation Orchestrator")
        parser.add_argument("--location", type=str, required=True, help="Target location for the MGRS grid constellation")
        parser.add_argument("--force", action="store_true", help="Force regeneration of MGRS gridded .h5 files even if they exist")
        args = parser.parse_args()
        
        location = args.location
        force = args.force
    else:
        location = target_location
    
    print(f"Starting MGRS Grid Constellation processing for location: {location}")
    print(f"Force regeneration: {force}")

    # Load location configuration
    config_path = os.path.join(project_root, "locations_config.yaml")
    config_data = {}
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f) or {}
    loc_data = config_data.get("locations", {}).get(location, {})
    
    # HLS
    hls_avail = loc_data.get("HLS_AVAILABLE", True)
    if not hls_avail:
        print("\n--- Skipping HLS (HLS_AVAILABLE is False in config) ---")
    elif force or not check_h5_exists("HLS", location):
        print("\n" + "="*40)
        print("--- Processing HLS ---")
        print("="*40)
        try:
            HLS30_earthAccess_to_MGRS_grid.main(target_location=location)
        except (FileNotFoundError, ValueError) as e:
            print(f"Warning: Skipping HLS stack generation: {e}")
    else:
        print("\n--- Skipping HLS (already exists) ---")
        
    # Tanager
    tanager_avail = loc_data.get("TANAGER_AVAILABLE", True)
    if not tanager_avail:
        print("\n--- Skipping Tanager (TANAGER_AVAILABLE is False in config) ---")
    elif force or not check_h5_exists("Tanager", location):
        print("\n" + "="*40)
        print("--- Processing Tanager ---")
        print("="*40)
        try:
            Tanager_STAC_imagery_downloader.main(target_location=location)
            tanager_MGRS_stacker.process_tanager_mgrs_stack(location)
        except (FileNotFoundError, ValueError) as e:
            print(f"Warning: Skipping Tanager stack generation: {e}")
    else:
        print("\n--- Skipping Tanager (already exists) ---")
        
    # EnMAP
    enmap_avail = loc_data.get("ENMAP_AVAILABLE", True)
    if not enmap_avail:
        print("\n--- Skipping EnMAP (ENMAP_AVAILABLE is False in config) ---")
    elif force or not check_h5_exists("EnMAP", location):
        print("\n" + "="*40)
        print("--- Processing EnMAP ---")
        print("="*40)
        try:
            enmap_STAC_imagery_downloader.main(target_location=location)
            enmap_MGRS_stacker.process_mgrs_stack(location)
        except (FileNotFoundError, ValueError) as e:
            print(f"Warning: Skipping EnMAP stack generation: {e}")
    else:
        print("\n--- Skipping EnMAP (already exists) ---")
        
    # Dragonette
    dragonette_avail = loc_data.get("DRAGONETTE_AVAILABLE", True)
    if not dragonette_avail:
        print("\n--- Skipping Dragonette (DRAGONETTE_AVAILABLE is False in config) ---")
    elif force or not check_h5_exists("Dragonette", location):
        print("\n" + "="*40)
        print("--- Processing Dragonette ---")
        print("="*40)
        try:
            dragonette_STAC_imagery_downloader.main(target_location=location)
            dragonette_MGRS_stacker.process_dragonette_mgrs_stack(location)
        except (FileNotFoundError, ValueError) as e:
            print(f"Warning: Skipping Dragonette stack generation: {e}")
    else:
        print("\n--- Skipping Dragonette (already exists) ---")
        
    print(f"\nMGRS Grid Constellation processing complete for location: {location}")

if __name__ == "__main__":
    main()
