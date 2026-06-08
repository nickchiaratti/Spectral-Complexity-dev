import pystac_client
from datetime import datetime

catalog = pystac_client.Client.open("https://cmr.earthdata.nasa.gov/stac/LPCLOUD")
bbox = [-77.770166, 42.961778, -77.376776, 43.342135]
search = catalog.search(collections=["HLSL30.v2.0"], bbox=bbox, datetime="2014-01-01/2026-06-01", limit=500)

items = list(search.items())
print(f"Total items found: {len(items)}")
if items:
    dates = sorted([i.datetime for i in items])
    print(f"Earliest date: {dates[0]}")
    print(f"Latest date: {dates[-1]}")
