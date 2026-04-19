import pystac_client
import folium
import webbrowser
import os

# Your ROI Configuration
#42.961778,-77.770166,43.342135,-77.376776
ROI_LON_MIN = 14.9100; ROI_LON_MAX = 15.15
ROI_LAT_MIN = 37.6900; ROI_LAT_MAX = 37.8300
#ROI_LON_MIN = -77.770166; ROI_LON_MAX = -77.376776
#ROI_LAT_MIN = 42.961778; ROI_LAT_MAX = 43.342135
safe_bbox = [
    min(ROI_LON_MIN, ROI_LON_MAX), min(ROI_LAT_MIN, ROI_LAT_MAX),
    max(ROI_LON_MIN, ROI_LON_MAX), max(ROI_LAT_MIN, ROI_LAT_MAX)
]

print("Querying NASA STAC for spatial footprints...")
catalog = pystac_client.Client.open("https://cmr.earthdata.nasa.gov/stac/LPCLOUD")
search = catalog.search(
    collections=["HLSS30.v2.0"],
    bbox=safe_bbox,
    datetime="2024-06-01/2024-06-15", # Short timeframe just to grab spatial footprints
    limit=50
)

# Extract unique MGRS tiles and their geometries
unique_tiles = {}
for item in search.items():
    tile_id = item.id.split('.')[2]
    if tile_id not in unique_tiles:
        unique_tiles[tile_id] = item.geometry

# Create an interactive map centered on your ROI
center_lat = (ROI_LAT_MIN + ROI_LAT_MAX) / 2
center_lon = (ROI_LON_MIN + ROI_LON_MAX) / 2
m = folium.Map(location=[center_lat, center_lon], zoom_start=9, tiles="CartoDB positron")

# Plot the MGRS Tiles
for tile_id, geometry in unique_tiles.items():
    folium.GeoJson(
        geometry,
        name=f"MGRS Tile: {tile_id}",
        style_function=lambda x: {'color': 'blue', 'fillColor': 'blue', 'weight': 2, 'fillOpacity': 0.1},
        tooltip=f"HLS Native MGRS Tile: {tile_id}"
    ).add_to(m)

# Plot your specific ROI
folium.Rectangle(
    bounds=[[safe_bbox[1], safe_bbox[0]], [safe_bbox[3], safe_bbox[2]]],
    color='red', fill=True, weight=3, fillOpacity=0.3,
    tooltip="Your Region of Interest (ROI)"
).add_to(m)

# Save and open
output_file = "mgrs_preflight_map.html"
m.save(output_file)
webbrowser.open('file://' + os.path.realpath(output_file))
print("Map opened in your web browser!")