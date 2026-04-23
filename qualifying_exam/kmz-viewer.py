import zipfile
import xml.etree.ElementTree as ET
import re
import folium

KMZ_PATH = r"C:\satelliteImagery\ground_truth\GPS_ROCX_ALL_METADATA_NOTFINAL.kmz"
TARGET_LAT, TARGET_LON = 43.13944, -77.50354


def extract_kmz_points(kmz_file_path: str) -> dict:
    """
    Extracts point features from a KMZ file strictly using the Placemark 'id' attribute.
    Returns a dictionary mapping the numeric IDs to (latitude, longitude) tuples.
    """
    points = {}
    
    with zipfile.ZipFile(kmz_file_path, 'r') as kmz:
        kml_files = [name for name in kmz.namelist() if name.endswith('.kml')]
        if not kml_files:
            raise FileNotFoundError("No KML file found inside the provided KMZ archive.")
            
        with kmz.open(kml_files[0], 'r') as kml_file:
            tree = ET.parse(kml_file)
            root = tree.getroot()

            # Strip XML namespaces for robust searching across different KML schema versions
            for elem in root.iter():
                if '}' in elem.tag:
                    elem.tag = elem.tag.split('}', 1)[1]

            # Parse Placemarks strictly for their 'id' attribute
            for placemark in root.findall('.//Placemark'):
                point_elem = placemark.find('.//Point/coordinates')
                
                # Validate we actually have point data
                if point_elem is None or not point_elem.text:
                    continue

                # KML coordinates are typically formatted as "longitude,latitude,altitude"
                coords_raw = point_elem.text.strip().split(',')
                if len(coords_raw) < 2:
                    continue
                    
                lon = float(coords_raw[0])
                lat = float(coords_raw[1])
                coords = (lat, lon)

                # Extract by Placemark 'id' attribute (e.g., id="ID_00270" -> "270")
                pm_id = placemark.get('id')
                if pm_id:
                    # Extract the pure numeric portion, handling leading zeros
                    m = re.search(r'\d+', pm_id)
                    if m:
                        numeric_id = str(int(m.group()))
                        points[numeric_id] = coords
                            
    return points

def main(kmz_path, center_lat, center_lon):
    
    print("Extracting data from KMZ...")
    all_points = extract_kmz_points(kmz_path)

    # 1. Validate that all required points (270 through 302) actually exist in the data.
    # Per research requirements, we do NOT use fill values. We halt execution if data is missing.
    required_ids = [str(i) for i in range(269, 302)]
    missing_ids = [pid for pid in required_ids if pid not in all_points]
    
    if missing_ids:
        raise ValueError(
            f"CRITICAL DATA MISSING: The following required IDs were not found in the KMZ: {missing_ids}. "
            "Execution halted to prevent inaccurate geospatial rendering."
        )

    print("All required data points verified. Generating high-resolution map...")
    
    # Initialize the map with no default tiles so we can cleanly add the high-res Esri imagery
    m = folium.Map(location=[center_lat, center_lon], zoom_start=18, tiles=None)

    # Add Esri World Imagery (High Resolution Satellite Basemap)
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri',
        name='Esri World Imagery',
        max_zoom=20
    ).add_to(m)

    # 2. Add the specific blue marker at the required coordinates
    folium.Marker(
        location=[center_lat, center_lon],
        popup='Target Location<br>Lat: 43.13927<br>Lon: -77.50340',
        icon=folium.Icon(color='blue', icon='info-sign')
    ).add_to(m)

    # 3. Add all individual points (270-302) to the map as small circle markers
    # Using white/cyan styling to ensure they stand out against the dark satellite imagery
    for pid in required_ids:
        lat, lon = all_points[pid]
        folium.CircleMarker(
            location=[lat, lon],
            radius=2,
            popup=f"ID: {pid}",
            color='red',
            fill=True,
            fill_color='red',
            fill_opacity=0.9
        ).add_to(m)

    # 4. Create the first red filled polygon (270, 271, 272, 273)
    poly1_coords = [all_points[str(i)] for i in [269, 270, 271, 272]]
    folium.Polygon(
        locations=poly1_coords,
        color='red',
        weight=3,
        fill=True,
        fill_color='red',
        fill_opacity=0.9,
        tooltip='Polygon 1 (270-273)'
    ).add_to(m)

    # 5. Create the second red filled polygon (274, 275, 276, 277)
    poly2_coords = [all_points[str(i)] for i in [273, 274, 275, 276]]
    folium.Polygon(
        locations=poly2_coords,
        color='red',
        weight=3,
        fill=True,
        fill_color='red',
        fill_opacity=0.9,
        tooltip='Polygon 2 (274-277)'
    ).add_to(m)

    # Add layer control to allow toggling between Satellite and Street views
    folium.LayerControl().add_to(m)

    # Save the generated map to an HTML file
    output_filename = 'gps_visualization.html'
    m.save(output_filename)
    print(f"Map successfully generated and saved to {output_filename}. Open this file in a web browser to view.")

if __name__ == "__main__":
    main(KMZ_PATH, TARGET_LAT, TARGET_LON)