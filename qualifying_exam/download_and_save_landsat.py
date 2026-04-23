import pystac_client
import requests
import shutil
import os
import sys
from datetime import datetime
import rasterio
import numpy as np

# --- Configuration ---

STAC_SERVER_URL = "https://landsatlook.usgs.gov/stac-server"

def get_user_input():
    """Gathers necessary input from the user."""
    print("--- USGS STAC Search & ENVI Save ---")
    
    api_key = ("NFwnV4Ci5VozWHlDC5D7DxYzna2q0Hr3M3x7fJXevwE9TvGhtCRPXKtZVbDpzJnz").strip()
    if not api_key:
        print("API Key is required. Exiting.")
        sys.exit(1)

    print("\nEnter Bounding Box (in decimal degrees, e.g., -77.5, 43.0, -77.4, 43.1):")
    try:
        bbox_str = input("min_lon, min_lat, max_lon, max_lat: ")
        bbox = [float(coord.strip()) for coord in bbox_str.split(',')]
        if len(bbox) != 4:
            raise ValueError
    except ValueError:
        print("Invalid Bounding Box format. Exiting.")
        sys.exit(1)

    print("\nEnter Date Range (YYYY-MM-DD):")
    try:
        start_date_str = input("Start Date (e.g., 2024-01-01): ").strip()
        end_date_str = input("End Date (e.g., 2024-05-30): ").strip()
        
        # Validate dates
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
        end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
        
        # Format for STAC query (inclusive of end date)
        datetime_range = f"{start_date_str}T00:00:00Z/{end_date_str}T23:59:59Z"
    except ValueError:
        print("Invalid Date format. Exiting.")
        sys.exit(1)
        
    return api_key, bbox, datetime_range, {"X-API-Key": api_key}

def search_stac(bbox, datetime_range, headers):
    """Searches the STAC server for matching Landsat items."""
    print(f"\nConnecting to {STAC_SERVER_URL}...")
    try:
        client = pystac_client.Client.open(STAC_SERVER_URL, headers=headers)
        
        search_params = {
            "collections": ["landsat-c2l2-sr","landsat-c2ard-sr"],
            "bbox": bbox,
            "datetime": datetime_range,
            "query": {
                "eo:cloud_cover": {"lt": 10},
                "platform": {"in": ["LANDSAT_8", "LANDSAT_9"]}
            }
        }
        
        print("Searching for items...")
        search = client.search(**search_params)
        items = search.item_collection()
        
        if not items:
            print("No items found matching your criteria.")
            return None
            
        return items

    except Exception as e:
        print(f"Error connecting or searching STAC: {e}")
        return None

def select_item(items):
    """Allows the user to select one item from the search results."""
    print(f"\nFound {len(items)} matching scenes:")
    for i, item in enumerate(items):
        print(f"  {i+1}: {item.id} "
              f"({item.datetime.date()}) - "
              f"{item.properties['eo:cloud_cover']:.2f}% clouds")
    
    try:
        choice = int(input(f"Which scene would you like to download? (1-{len(items)}): "))
        if not 1 <= choice <= len(items):
            raise ValueError
        
        selected_item = items[choice - 1]
        print(f"You selected: {selected_item.id}")
        return selected_item
        
    except ValueError:
        print("Invalid selection. Exiting.")
        sys.exit(1)

def download_bands(item, headers):
    """
    Downloads all available scientific and QA bands (SR B1-7, ST_B10, QA_PIXEL,
    QA_RADSAT, SR_QA_AEROSOL) for the selected item.
    Returns a list of downloaded filenames.
    """
    # All 11 bands in the Landsat 8 C2L2 SR collection
    # Note: Landsat 9 only has 10 (it lacks QA_RADSAT)
    target_bands = [f'SR_B{i}' for i in range(1, 8)] # 7 bands
    target_bands.extend(['ST_B10', 'QA_PIXEL', 'QA_RADSAT', 'SR_QA_AEROSOL']) # 4 more
    target_bands = ['coastal', 'blue', 'green', 'red', 'nir08', 'swir16', 'swir22', 'qa_aerosol', 'qa_pixel', 'qa_radsat', 'temp']
    
    downloaded_files = []
    
    # Create a temporary directory to hold band files
    temp_dir = "temp_bands"
    os.makedirs(temp_dir, exist_ok=True)
    platform = item.properties.get('platform', 'unknown')
    print(f"\nDownloading all available bands ({platform}) to '{temp_dir}/'...")
    print(item.assets.keys())
    try:
        for band_name in target_bands:
            if band_name not in item.assets:
                print(f"  Info: Asset '{band_name}' not found in this item (normal for L9). Skipping.")
                continue
                
            asset = item.assets[band_name]
            download_url = asset.href
            # Use the asset's original filename
            original_filename = os.path.basename(asset.href.split('?')[0])
            local_filepath = os.path.join(temp_dir, original_filename)
            
            print(f"  Downloading {band_name}...")
            
            with requests.get(download_url, stream=True, headers=headers) as r:
                r.raise_for_status()
                with open(local_filepath, 'wb') as f:
                    shutil.copyfileobj(r.raw, f)
            
            downloaded_files.append(local_filepath)
            
        if not downloaded_files:
            print("Error: No bands were successfully downloaded.")
            return None
            
        print(f"Successfully downloaded {len(downloaded_files)} bands.")
        return downloaded_files
        
    except KeyError as e:
        print(f"Error: A required asset key was not found: {e}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error during download: {e}")
        return None

def save_to_envi_format_rasterio(band_filenames):
    """
    Uses rasterio to stack the downloaded GeoTIFFs and save the result
    in the ENVI .dat/.hdr format.
    """
    if not band_filenames:
        print("No files to process.")
        return

    output_filename = "landsat_scene_stack.dat"
    # Get absolute paths
    abs_input_paths = [os.path.abspath(f) for f in band_filenames]
    abs_output_path = os.path.abspath(output_filename)

    print(f"\nInitializing rasterio to stack and save to ENVI format...")

    try:
        # Open the first file to get metadata
        with rasterio.open(abs_input_paths[0]) as src:
            meta = src.profile
            meta['driver'] = 'ENVI'
            meta['interleave'] = 'BSQ' # Band Sequential, simplest to write
            meta['count'] = len(abs_input_paths)
            meta['dtype'] = src.dtypes[0] # Ensure we use the correct data type

        # Create the output ENVI file
        with rasterio.open(abs_output_path, 'w', **meta) as dst:
            print(f"Creating stacked file: {abs_output_path} (and .hdr)")
            
            for i, band_file in enumerate(abs_input_paths):
                print(f"  Adding band {i+1}/{len(abs_input_paths)}...")
                with rasterio.open(band_file) as src_band:
                    # Read the first (and only) band from the source file
                    band_data = src_band.read(1)
                    # Write this data to the (i+1)th band in the output file
                    dst.write(band_data, i + 1)
        
        print(f"\nSuccessfully saved stacked file as {output_filename} (and .hdr)")

        # Clean up the intermediate GeoTIFFs and directory
        if abs_input_paths:
            temp_dir = os.path.dirname(abs_input_paths[0])
            shutil.rmtree(temp_dir)
            print(f"Cleaned up temporary directory: {temp_dir}")

    except Exception as e:
        print(f"An error occurred during rasterio processing: {e}")
        print("Please ensure 'rasterio' and 'numpy' are installed.")

def main():
    api_key, bbox, datetime_range, headers = get_user_input()
    
    items = search_stac(bbox, datetime_range, headers)
    
    if items:
        selected_item = select_item(items)
        downloaded_files = download_bands(selected_item, headers)
        
        if downloaded_files:
            save_to_envi_format_rasterio(downloaded_files)

if __name__ == "__main__":
    main()


