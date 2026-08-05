"""
Downloads EnMAP HSI L2A data for specified ROIs and time ranges from the DLR EOC Geoservice STAC.
Uses pystac-client for querying and requests with HTTP Basic Auth for downloading.
"""
import os
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from pathlib import Path
from datetime import datetime
import getpass
import json
import browser_cookie3
import yaml

# ==========================================
# 1. CONFIGURATION
# ==========================================
# Target collection
STAC_API_URL = "https://geoservice.dlr.de/eoc/ogc/stac/v1"
COLLECTION_ID = "ENMAP_HSI_L2A"
TARGET_LOCATION = 'Rochesterv2'

# Output Directory
OUTPUT_DIR = Path(f"C:/satelliteImagery/enmap/SourceData")

def create_retry_session():
    """Creates a robust requests Session equipped with exponential backoff and browser cookies."""
    session = requests.Session()
    
    # Attempt to load .env manually if dotenv is not installed
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, val = line.split('=', 1)
                    os.environ[key.strip()] = val.strip()

    enmap_user = os.environ.get('ENMAP_USERNAME')
    enmap_pass = os.environ.get('ENMAP_PASSWORD')
    
    session.basic_auth = None
    if enmap_user and enmap_pass:
        print("Using basic authentication from .env file for EnMAP downloads.")
        session.basic_auth = (enmap_user, enmap_pass)
    else:
        print("Extracting session cookies for dlr.de from your local browser...")
        try:
            cj = browser_cookie3.load(domain_name='dlr.de')
            session.cookies.update(cj)
        except Exception as e:
            print(f"Warning: Failed to extract browser cookies: {e}")
            print("Continuing without browser cookies, but downloads may fail if authentication is required.")
        
    retry_strategy = Retry(
        total=10,
        status_forcelist=[429, 500, 502, 503, 504],
        backoff_factor=2,
        allowed_methods=["HEAD", "GET", "OPTIONS"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

def download_file(url, destination_path, session):
    """Streams file download to disk to handle large multi-gigabyte assets efficiently."""
    if destination_path.exists():
        print(f"    -> Skipping {destination_path.name} (Already exists)")
        return True

    print(f"    -> Downloading to {destination_path.name}...")
    try:
        kwargs = {"stream": True}
        if getattr(session, 'basic_auth', None):
            kwargs['auth'] = session.basic_auth
            
        with session.get(url, **kwargs) as r:
            # Note: For some web-based portals, a 401/403 might indicate auth issues.
            if r.status_code in [401, 403]:
                print(f"    -> Authentication failed or access denied (HTTP {r.status_code}) for {url}")
                return False
            r.raise_for_status()
            with open(destination_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        return True
    except Exception as e:
        print(f"    -> Error downloading {url}: {e}")
        # Clean up partial file
        if destination_path.exists():
            destination_path.unlink()
        return False

def execute_job(job_config, session):
    job_name = job_config["job_name"]
    bbox = job_config["bbox"]
    dt_range = job_config["datetime"]
    cc_max = job_config.get("cloud_cover_max", 100)
    target_assets = job_config.get("target_assets", ['image','metadata','quality_classes'])
    
    out_dir = OUTPUT_DIR / job_name
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*50}")
    print(f"Executing Job: {job_name}")
    print(f"BBox: {bbox}")
    print(f"Time: {dt_range}")
    print(f"Target Directory: {out_dir}")
    print(f"{'='*50}")
    
    # 1. Query STAC
    print("Querying DLR EOC STAC...")
    
    # Target the collection directly instead of using the global /search endpoint
    collection_items_url = f"{STAC_API_URL}/collections/{COLLECTION_ID}/items"
    
    params = {
        "bbox": ",".join(map(str, bbox)),
        "datetime": dt_range,
        "limit": 500
    }
    
    try:
        response = session.get(collection_items_url, params=params)
        response.raise_for_status()
        feature_collection = response.json()
        items = feature_collection.get("features", [])
        
        # Handle simple pagination if there are more results
        next_link = next((link['href'] for link in feature_collection.get('links', []) if link['rel'] == 'next'), None)
        while next_link:
            resp = session.get(next_link)
            resp.raise_for_status()
            fc = resp.json()
            items.extend(fc.get("features", []))
            next_link = next((link['href'] for link in fc.get('links', []) if link['rel'] == 'next'), None)
            
    except Exception as e:
        print(f"Failed to query STAC API: {e}")
        return

    # Client-side cloud cover filtering
    filtered_items = []
    for item in items:
        # Check cloud cover property if it exists
        properties = item.get('properties', {})
        cloud_cov = properties.get('eo:cloud_cover', 0)
        try:
            if float(cloud_cov) <= float(cc_max):
                filtered_items.append(item)
        except (ValueError, TypeError):
            # If cloud cover is entirely missing or unparseable, keep the item just in case
            filtered_items.append(item)
            
    print(f"Found {len(filtered_items)} items matching criteria (Cloud Cover <= {cc_max}%).")
    
    matched_items = 0
    for idx, item in enumerate(filtered_items):
        matched_items += 1
        item_id = item.get('id', f'item_{idx}')
        print(f"\n  [{idx+1}/{len(filtered_items)}] Processing {item_id}...")
        
        item_folder = out_dir / item_id
        item_folder.mkdir(exist_ok=True)
        
        # Save STAC metadata locally
        json_file_name = f"{item_id}.json"
        json_dest = item_folder / json_file_name
        if not json_dest.exists():
            with open(json_dest, 'w') as f:
                json.dump(item, f, indent=2)
                
        # Download assets
        assets = item.get('assets', {})
        if not assets:
            print("    -> No assets found for this item.")
            continue
            
        for asset_key, asset in assets.items():
            if target_assets and asset_key not in target_assets:
                continue
                
            href = asset.get('href')
            if not href:
                continue
                
            file_name = os.path.basename(href.split("?")[0])
            dest_path = item_folder / file_name
            
            download_file(href, dest_path, session)
            
    print(f"\nJob '{job_name}' completed. Successfully processed {matched_items} scenes.")

def main(target_location=None):
    print("Initializing EnMAP STAC Job Queue...\n")
    
    script_dir = Path(__file__).resolve().parent
    config_path = os.path.join(script_dir.parent, "locations_config.yaml")
    
    if not os.path.exists(config_path):
        print(f"Error: Could not find configuration file at {config_path}")
        return
        
    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)
        
    download_jobs = []
    locations_to_process = [target_location] if target_location else config_data.get("locations", {}).keys()
    
    for loc_name in locations_to_process:
        if loc_name not in config_data.get("locations", {}):
            print(f"Warning: Location '{loc_name}' not found in locations_config.yaml")
            continue
            
        loc_config = config_data["locations"][loc_name]
        
        min_lon = loc_config["ROI_LON_MIN"]
        max_lon = loc_config["ROI_LON_MAX"]
        min_lat = loc_config["ROI_LAT_MIN"]
        max_lat = loc_config["ROI_LAT_MAX"]
        
        start_date = loc_config.get("START_DATE", "2023-01-01")
        end_date = loc_config.get("END_DATE", "2025-12-31")
        
        # Enforce valid bbox format [minx, miny, maxx, maxy]
        bbox = [min(min_lon, max_lon), min(min_lat, max_lat), max(min_lon, max_lon), max(min_lat, max_lat)]
        
        download_jobs.append({
            "job_name": f"{loc_name}_EnMAP",
            "bbox": bbox,
            "datetime": f"{start_date}/{end_date}",
            "cloud_cover_max": 85,
            "target_assets": [] 
        })
    
    session = create_retry_session()
    
    for job in download_jobs:
        execute_job(job, session)
        
    print("\nAll download jobs finished.")

if __name__ == "__main__":
    main(target_location=TARGET_LOCATION)
