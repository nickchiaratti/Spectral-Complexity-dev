import pystac
import geopandas as gpd
import networkx as nx
from shapely.geometry import shape

def analyze_stac_overlaps(catalog_url: str, overlap_threshold: float = 0.5):
    """
    Parses a STAC collection and identifies locations with repeated imagery.
    
    Args:
        catalog_url (str): URL to the STAC collection.json
        overlap_threshold (float): Minimum fractional overlap (Intersection over Area) 
                                   required to consider two scenes as covering the same location.
    """
    print(f"Loading STAC Collection from: {catalog_url}")
    # Load the collection. pystac handles HTTP resolution automatically.
    collection = pystac.Collection.from_file(catalog_url)
    
    # Retrieve all items (this fetches individual item JSONs)
    items = list(collection.get_all_items())
    print(f"Discovered {len(items)} items in the collection.")

    # Extract exact data without failure-handling/fill values. 
    # Will raise Key/Attribute errors if the STAC catalog is malformed.
    features = []
    for item in items:
        features.append({
            "id": item.id,
            "datetime": item.datetime,
            "geometry": shape(item.geometry)
        })

    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(features, crs="EPSG:4326")

    # Project to an Equal Area CRS (EASE-Grid 2.0 Global) for rigorous area calculations.
    # Calculating area on EPSG:4326 (degrees) introduces significant latitudinal distortion.
    gdf_proj = gdf.to_crs("EPSG:6933")

    # Calculate exact areas for the threshold denominator
    gdf_proj['area'] = gdf_proj.geometry.area

    print("Computing spatial intersections...")
    # Perform a spatial join to find intersecting bounding boxes first (computational optimization)
    intersections = gpd.sjoin(gdf_proj, gdf_proj, how="inner", predicate="intersects")
    
    # Filter out self-intersections
    intersections = intersections[intersections.index != intersections.index_right]

    # Build an undirected graph where nodes are scenes and edges represent significant spatial overlap
    G = nx.Graph()
    G.add_nodes_from(gdf_proj.index)

    for idx1, idx2 in zip(intersections.index, intersections.index_right):
        # Avoid duplicate calculations for undirected pairs
        if idx1 > idx2:
            geom1 = gdf_proj.loc[idx1, 'geometry']
            geom2 = gdf_proj.loc[idx2, 'geometry']
            
            # Calculate the actual intersection polygon area
            intersection_area = geom1.intersection(geom2).area
            
            # Calculate Intersection over Area (IoA) relative to the smaller scene
            # This ensures a small scene fully contained within a large scene is clustered
            min_area = min(gdf_proj.loc[idx1, 'area'], gdf_proj.loc[idx2, 'area'])
            ioa = intersection_area / min_area

            if ioa >= overlap_threshold:
                G.add_edge(idx1, idx2)

    # Find connected components (clusters of overlapping scenes)
    clusters = list(nx.connected_components(G))
    
    print("\n--- Overlap Analysis Results ---")
    repeated_locations = 0
    for i, cluster in enumerate(clusters):
        if len(cluster) > 1: # Only report locations with repeated collections
            repeated_locations += 1
            # Retrieve the records for this cluster
            cluster_data = gdf.loc[list(cluster)].sort_values(by="datetime")
            
            # Calculate the centroid of the union of these geometries for location reference
            # Re-project to EPSG:4326 to print standard Lat/Lon coordinates
            cluster_union = gdf_proj.loc[list(cluster)].geometry.union_all()
            centroid = gpd.GeoSeries([cluster_union], crs="EPSG:6933").centroid.to_crs("EPSG:4326").iloc[0]
            
            print(f"\nLocation {repeated_locations}: Approximate Centroid (Lon: {centroid.x:.4f}, Lat: {centroid.y:.4f})")
            print(f"Total Collections: {len(cluster)}")
            
            for _, row in cluster_data.iterrows():
                print(f"  - Date: {row['datetime'].strftime('%Y-%m-%d %H:%M:%S UTC')} | ID: {row['id']}")

    if repeated_locations == 0:
        print("\nNo repeated collections found meeting the overlap threshold.")

if __name__ == "__main__":
    TARGET_URL = "https://www.planet.com/data/stac/tanager-core-imagery/urban/collection.json"
    # Using a 50% overlap threshold. This can be adjusted based on analytical requirements.
    analyze_stac_overlaps(TARGET_URL, overlap_threshold=0.5)