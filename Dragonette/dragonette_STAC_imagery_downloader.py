import os
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import json
from pathlib import Path
from urllib.parse import urljoin

# --- Pre-defined Regions of Interest (Bounding Boxes: [min_lon, min_lat, max_lon, max_lat]) ---
REGIONS = {
    "Southern_California": [-121.5, 32, -113, 35.5],
    "Utah": [-114.05, 37.0, -109.0, 42.5],
    "Rochester_NY": [-77.72, 43.04, -77.44, 43.28],
    "Buenos_Aires": [ -65.0, -70.0, -41.0, -30.0],
    "Global": [-180.0, -90.0, 180.0, 90.0]
}

# --- Download Jobs Configuration ---
DOWNLOAD_JOBS = [
    #{
    #    "job_name": "Palisades_fire",
    #    "collection_url": "https://wyvern-odp.com/product-type/standard/collection.json",
    #    "output_dir": "C:/satelliteImagery/Wyvern/Palisades_SourceData",
    #    "include_bboxes": [REGIONS["Southern_California"]],
    #    "target_assets": ['zip_file'] 
    #},
    {
        "job_name": "ROCX",
        # Extracted the true machine-readable STAC endpoint from the item's 'parent' links
        "collection_url": "https://wyvern-odp.com/industry/spatio_temporal/collection.json",
        "output_dir": "C:/satelliteImagery/Wyvern/ROCX_SourceData",
        "include_bboxes": [REGIONS["Rochester_NY"]],
        # Specifically targeting the zip archive as requested
        "target_assets": ['zip_file'] 
    }


]

def create_retry_session():
    """
    Creates a robust requests Session equipped with exponential backoff.
    Automatically catches 429 (Too Many Requests) and 50x server errors,
    pausing execution and retrying before escalating to a hard failure.
    """
    session = requests.Session()
    retry_strategy = Retry(
        total=10,  # Maximum number of consecutive retries before failing
        status_forcelist=[429, 500, 502, 503, 504],
        backoff_factor=2,  # Wait times: 2s, 4s, 8s, 16s, 32s...
        allowed_methods=["HEAD", "GET", "OPTIONS"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

def intersects(bbox1, bbox2):
    """Evaluates whether two [min_lon, min_lat, max_lon, max_lat] bounding boxes intersect."""
    return not (bbox1[2] < bbox2[0] or bbox1[0] > bbox2[2] or
                bbox1[3] < bbox2[1] or bbox1[1] > bbox2[3])

def passes_spatial_filters(item_bbox, include_bboxes, exclude_bboxes):
    """
    Evaluates an item's bounding box against explicit inclusion and exclusion regions.
    """
    if not item_bbox:
        return False

    for ex_box in exclude_bboxes:
        if intersects(item_bbox, ex_box):
            return False

    if not include_bboxes:
        return True 
        
    for inc_box in include_bboxes:
        if intersects(item_bbox, inc_box):
            return True

    return False

def download_file(url, destination_path, session):
    """
    Streams file download to disk to handle large multi-gigabyte assets efficiently.
    Uses the robust session to handle rate limits (429) automatically.
    """
    if destination_path.exists():
        print(f"    -> Skipping {destination_path.name} (Already exists)")
        return

    with session.get(url, stream=True) as r:
        r.raise_for_status()  # Crashes on persistent errors not caught by the retry logic (e.g., 404, 403)
        with open(destination_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

def execute_job(job_config, session):
    """Processes a single download job dictionary."""
    job_name = job_config["job_name"]
    collection_url = job_config["collection_url"]
    out_dir = Path(job_config["output_dir"])
    includes = job_config.get("include_bboxes", [])
    excludes = job_config.get("exclude_bboxes", [])
    target_assets = job_config.get("target_assets", [])

    print(f"\n{'='*50}")
    print(f"Executing Job: {job_name}")
    print(f"Target Directory: {out_dir}")
    print(f"{'='*50}")

    out_dir.mkdir(parents=True, exist_ok=True)

    response = session.get(collection_url)
    response.raise_for_status()
    collection_data = response.json()

    item_links = [link['href'] for link in collection_data.get('links', []) if link.get('rel') == 'item']
    print(f"Discovered {len(item_links)} items in catalog.")
    
    matched_items = 0

    for idx, item_href in enumerate(item_links):
        item_url = urljoin(collection_url, item_href)
        
        item_resp = session.get(item_url)
        item_resp.raise_for_status()
        item_data = item_resp.json()

        item_id = item_data.get('id', f'item_{idx}')
        item_bbox = item_data.get('bbox')
        
        # Apply Spatial Filtering
        if not passes_spatial_filters(item_bbox, includes, excludes):
            continue

        matched_items += 1
        print(f"\n  [{item_id}] Matched spatial criteria. Processing assets...")
        
        item_folder = out_dir / item_id
        item_folder.mkdir(exist_ok=True)

        # Download STAC JSON Metadata
        json_file_name = f"{item_id}.json"
        json_dest = item_folder / json_file_name
        download_file(item_url, json_dest, session)

        # Download Specific Assets
        assets = item_data.get('assets', {})
        available_keys = list(assets.keys())
        
        for asset_key in target_assets:
            if asset_key in assets:
                asset_url = assets[asset_key].get('href')
                if not asset_url.startswith('http'):
                    asset_url = urljoin(item_url, asset_url)
                
                file_name = os.path.basename(asset_url)
                dest_path = item_folder / file_name
                print(f"    -> Downloading {asset_key} ({file_name})...")
                download_file(asset_url, dest_path, session)
            else:
                # Explicitly raise an error if the expected asset structure changes
                raise KeyError(f"CRITICAL: Asset '{asset_key}' not present in item {item_id}. Available assets: {available_keys}")

    print(f"\nJob '{job_name}' completed. Successfully downloaded {matched_items} scenes.")

def main():
    print("Initializing Wyvern STAC Job Queue...\n")
    
    # Initialize the robust network session once and pass it to all jobs
    session = create_retry_session()
    
    for job in DOWNLOAD_JOBS:
        execute_job(job, session)
        
    print("\nAll download jobs finished.")

if __name__ == "__main__":
    main()