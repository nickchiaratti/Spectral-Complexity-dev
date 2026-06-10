import pystac_client
from datetime import datetime
import json

catalog = pystac_client.Client.open("https://cmr.earthdata.nasa.gov/stac/LPCLOUD")
collections = ["HLSS30.v2.0", "HLSL30.v2.0"]
bbox = [-77.50341, 43.13926, -77.50339, 43.13928]
dates = [
    "2024-04-01",
    "2024-03-08",
    "2024-02-17",
    "2024-05-03",
    "2024-05-22",
    "2024-06-01",
    "2024-07-07",
    "2025-11-22"
]

results = []

for date_str in dates:
    print(f"Checking {date_str}...")
    search = catalog.search(
        collections=collections,
        bbox=bbox,
        datetime=f"{date_str}T00:00:00Z/{date_str}T23:59:59Z"
    )
    items = list(search.items())
    for item in items:
        # Check if it is T17TQH
        tile = item.id.split('.')[2]
        if tile == "T17TQH":
            results.append({
                "date": date_str,
                "id": item.id,
                "collection": item.collection_id,
                "cloud_cover": item.properties.get('eo:cloud_cover'),
                "tile": tile
            })

print(json.dumps(results, indent=2))
