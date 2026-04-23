import os
import getpass
import shutil
import glob
from landsatxplore.api import API
from landsatxplore.earthexplorer import EarthExplorer
import rasterio

def get_credentials():
    """Securely prompt user for USGS EarthExplorer credentials."""
    print("Please enter your USGS EarthExplorer (EROS) credentials.")
    username = input("Username: ")
    password = getpass.getpass("Password: ")
    return username, password

def get_date_range():
    """Prompt user for start and end dates."""
    start_date = input("Enter start date (YYYY-MM-DD): ")
    end_date = input("Enter end date (YYYY-MM-DD): ")
    return start_date, end_date

def search_scenes(api, bbox, start_date, end_date, max_cloud):
    """Search for Landsat scenes."""
    print("Searching for scenes...")
    try:
        scenes = api.search(
            dataset='LANDSAT_OT_C2_L1',  # Landsat 8-9 OLI/TIRS Collection 2 Level 1
            bbox=bbox,
            start_date=start_date,
            end_date=end_date,
            max_cloud_cover=max_cloud
        )
        print(f"Found {len(scenes)} scenes.")
        return scenes
    except Exception as e:
        print(f"Error during search: {e}")
        return []

def download_scene(username, password, scene_id, output_dir):
    """Download a single scene bundle using EarthExplorer."""
    print(f"Initializing download for scene: {scene_id}...")
    try:
        ee = EarthExplorer(username, password)
        # Download returns the path to the downloaded .tar file
        tar_path = ee.download(scene_id, output_dir=output_dir)
        ee.logout()
        print(f"Download complete: {tar_path}")
        return tar_path
    except Exception as e:
        print(f"Error during download: {e}")
        return None

def extract_and_find_bands(tar_path, extract_dir):
    """Extract the .tar file and find all spectral band .TIF files."""
    print(f"Extracting {tar_path} to {extract_dir}...")
    try:
        shutil.unpack_archive(tar_path, extract_dir)
        print("Extraction complete.")
        
        # Find all spectral band files (B1-B11)
        # Using a pattern that captures B1..B9, B10, B11
        band_pattern = os.path.join(extract_dir, f"{os.path.basename(tar_path).split('.')[0]}*_B*.TIF")
        band_files = sorted(glob.glob(band_pattern))
        
        # Filter out non-spectral bands if any (e.g., BQA)
        spectral_band_files = [
            f for f in band_files 
            if f.endswith(tuple([f"_B{i}.TIF" for i in range(1, 12)]))
        ]
        
        print(f"Found {len(spectral_band_files)} spectral band files.")
        return spectral_band_files
        
    except Exception as e:
        print(f"Error during extraction: {e}")
        return []

def stack_bands_to_envi(band_files, output_filename, scene_id):
    """Stack all band files into a single ENVI .dat/.hdr file."""
    if not band_files:
        print("No band files found to stack.")
        return

    print(f"Stacking {len(band_files)} bands to {output_filename}...")
    
    try:
        # Open all band files
        src_files_to_open = [rasterio.open(fp) for fp in band_files]
        
        # Read metadata from the first band
        profile = src_files_to_open[0].profile
        
        # Update metadata for the stacked ENVI output
        profile.update(
            driver='ENVI',
            count=len(src_files_to_open), # Number of bands
            interleave='BAND'
        )
        
        # Write the stacked ENVI file
        with rasterio.open(output_filename, 'w', **profile) as dst:
            for i, src in enumerate(src_files_to_open, start=1):
                # Read the band
                band_data = src.read(1)
                # Write to the destination file
                dst.write(band_data, i)
                # Set band description (e.g., "Band 1", "Band 2")
                dst.set_band_description(i, f"Band {i}")
        
        print("Stacking complete.")
        print(f"Successfully created ENVI file: {output_filename}")
        print(f"Associated header file: {output_filename.replace('.dat', '.hdr')}")

    except Exception as e:
        print(f"Error during stacking: {e}")
    finally:
        # Ensure all source files are closed
        for src in src_files_to_open:
            src.close()

def main():
    # --- Configuration ---
    # Bounding box for Rochester, NY [min_lon, min_lat, max_lon, max_lat]
    rochester_bbox = [-77.8, 43.1, -77.4, 43.3]
    max_cloud_cover = 10
    
    # Directories for download and extraction
    download_dir = "landsat_download"
    extract_dir = "landsat_extracted"
    output_dir = "landsat_output"
    
    # Create directories if they don't exist
    for d in [download_dir, extract_dir, output_dir]:
        os.makedirs(d, exist_ok=True)
        
    # --- 1. Get Credentials and Dates ---
    username = 'text'
    password = 'text'
    start_date, end_date = get_date_range()
    
    try:
        # --- 2. Login and Search ---
        api = ("NFwnV4Ci5VozWHlDC5D7DxYzna2q0Hr3M3x7fJXevwE9TvGhtCRPXKtZVbDpzJnz").strip()
        scenes = search_scenes(api, rochester_bbox, start_date, end_date, max_cloud_cover)
        
        if not scenes:
            print("No matching scenes found for the given criteria.")
            api.logout()
            return
            
        # Select the first (often best) scene
        scene_to_download = scenes[0]
        scene_id = scene_to_download['display_id']
        print(f"Selected scene: {scene_id} (Cloud Cover: {scene_to_download['cloud_cover']}%)")
        
        # --- 3. Download Scene ---
        tar_path = download_scene(username, password, scene_id, download_dir)
        
        if not tar_path:
            api.logout()
            return
            
        # --- 4. Extract and Find Bands ---
        band_files = extract_and_find_bands(tar_path, extract_dir)
        
        if not band_files:
            print("Could not find band files in the downloaded archive.")
            api.logout()
            return
            
        # --- 5. Stack Bands to ENVI ---
        # We use .dat for the ENVI file, .hdr will be created automatically
        output_filename = os.path.join(output_dir, f"{scene_id}_stack.dat")
        stack_bands_to_envi(band_files, output_filename, scene_id)
        
        # --- 6. Logout ---
        api.logout()
        print("Logged out from USGS API.")
        
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        # --- 7. Cleanup ---
        # Clean up the downloaded and extracted files
        try:
            print("Cleaning up temporary download and extraction directories...")
            shutil.rmtree(download_dir)
            shutil.rmtree(extract_dir)
            print("Cleanup complete.")
        except OSError as e:
            print(f"Error cleaning up directories (files may be in use): {e}")

if __name__ == "__main__":
    print("--- Landsat Download and Stacking Tool ---")
    print("This script will download a Landsat 8/9 scene and stack all spectral bands into a single ENVI file.")
    print("You will need your USGS EarthExplorer credentials.")
    print("Required Python libraries: landsatxplore, rasterio\n")
    main()

