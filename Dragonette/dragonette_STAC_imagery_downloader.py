import os
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import json
from pathlib import Path
from urllib.parse import urljoin
import yaml
LOCATION = "CentralGreece"  # Fallback location if not specified in YAML

# --- Dynamic Configuration Loading ---
script_dir = Path(__file__).resolve().parent
config_path = os.path.join(script_dir, "locations_config.yaml")
if not os.path.exists(config_path):
    config_path = os.path.join(script_dir.parent, "locations_config.yaml")

with open(config_path, "r") as f:
    config_data = yaml.safe_load(f)

DOWNLOAD_JOBS = []
target_location = config_data.get("current_run", {}).get("location", LOCATION)

locations_to_run = config_data.get("locations", {})
if target_location and target_location in locations_to_run:
    locations_to_run = {target_location: locations_to_run[target_location]}

for loc_name, loc_data in locations_to_run.items():
    source_cache = loc_data.get("SOURCE_CACHE")
    if source_cache is None:
        source_cache = loc_name
        
    bbox = [
        loc_data["ROI_LON_MIN"],
        loc_data["ROI_LAT_MIN"],
        loc_data["ROI_LON_MAX"],
        loc_data["ROI_LAT_MAX"]
    ]
    # Use the top-level catalog to recursively find all items across all sub-collections
    url = "https://wyvern-odp.com/year/catalog.json"
    
    job = {
        "job_name": loc_name,
        "collection_url": url,
        "output_dir": f"C:/satelliteImagery/dragonette/{source_cache}_SourceData",
        "include_bboxes": [bbox],
        "target_assets": ['zip_file'] 
    }
    DOWNLOAD_JOBS.append(job)

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

    def fetch_all_item_links(start_url, max_depth=3):
        visited = set()
        item_urls = []
        
        def crawl(url, depth):
            if depth > max_depth or url in visited:
                return
            visited.add(url)
            
            try:
                resp = session.get(url)
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                print(f"  [Warning] Failed to fetch {url}: {e}")
                return
                
            links = data.get('links', [])
            for link in links:
                rel = link.get('rel')
                href = link.get('href')
                if not href:
                    continue
                
                full_url = urljoin(url, href)
                if rel == 'item':
                    item_urls.append(full_url)
                elif rel == 'child':
                    crawl(full_url, depth + 1)
                    
        crawl(start_url, 0)
        return list(set(item_urls)) # deduplicate
        
    print("Crawling STAC Catalog for items...")
    item_urls = fetch_all_item_links(collection_url)
    print(f"Discovered {len(item_urls)} items across the catalog.")
    
    matched_items = 0

    for idx, item_url in enumerate(item_urls):
        
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
                
                if file_name.endswith('.zip'):
                    import zipfile
                    print(f"    -> Extracting {file_name}...")
                    try:
                        with zipfile.ZipFile(dest_path, 'r') as zip_ref:
                            zip_ref.extractall(item_folder)
                    except zipfile.BadZipFile:
                        print(f"    -> WARNING: Bad zip file {file_name}")
            else:
                # Explicitly raise an error if the expected asset structure changes
                raise KeyError(f"CRITICAL: Asset '{asset_key}' not present in item {item_id}. Available assets: {available_keys}")

    print(f"\nJob '{job_name}' completed. Successfully downloaded {matched_items} scenes.")

def main(target_location=None):
    print("Initializing Wyvern STAC Job Queue...\n")
    
    # Initialize the robust network session once and pass it to all jobs
    session = create_retry_session()
    
    # Filter jobs if target_location is provided
    jobs_to_run = DOWNLOAD_JOBS
    if target_location:
        jobs_to_run = [job for job in DOWNLOAD_JOBS if job["job_name"] == target_location]
        
    for job in jobs_to_run:
        execute_job(job, session)
        
    print("\nAll download jobs finished.")

if __name__ == "__main__":
    main()