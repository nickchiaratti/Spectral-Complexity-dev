import streamlit as st
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium
import pystac_client
import pystac
import yaml
import os
from pathlib import Path

st.set_page_config(page_title="MGRS ROI Selector", layout="wide")

script_dir = Path(__file__).resolve().parent
CONFIG_FILE = os.path.join(script_dir, "locations_config.yaml")

def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            return yaml.safe_load(f)
    return {"locations": {}, "current_run": {"location": ""}}

def save_config(config_data):
    with open(CONFIG_FILE, "w") as f:
        yaml.dump(config_data, f, sort_keys=False)

@st.cache_data(show_spinner=False)
def query_custom_stac_collection(catalog_url):
    """Query a custom STAC catalog for footprint geometries."""
    if not catalog_url:
        return {}
    try:
        collection = pystac.Collection.from_file(catalog_url)
        items = list(collection.get_all_items())
        unique_tiles = {}
        for item in items:
            unique_tiles[item.id] = item.geometry
        return unique_tiles
    except Exception as e:
        return {"error": str(e)}

@st.cache_data(show_spinner=False)
def query_stac_for_tiles(bbox, start_date, end_date):
    """Query NASA STAC for HLS spatial footprints to get intersecting MGRS tiles and their geometries."""
    catalog = pystac_client.Client.open("https://cmr.earthdata.nasa.gov/stac/LPCLOUD")
    # For footprint queries, a short temporal window is fine, but we'll use the user's dates
    search = catalog.search(
        collections=["HLSL30.v2.0"],
        bbox=bbox,
        datetime=f"{start_date}/{end_date}",
        max_items=10
    )
    items = list(search.items())
    unique_tiles = {}
    for item in items:
        # HLS items usually have the tile id in their name, e.g. HLS.L30.T17TQH...
        tile_id = item.id.split('.')[2]
        if tile_id.startswith('T') and tile_id not in unique_tiles:
            unique_tiles[tile_id] = item.geometry
    return unique_tiles

def main():
    st.title("Interactive MGRS ROI Selector")
    
    config_data = load_config()
    all_locations = list(config_data.get("locations", {}).keys())
    current_run_loc = config_data.get("current_run", {}).get("location", "")
    
    if not all_locations:
        st.error("No locations found in configuration.")
        return

    # Sidebar configuration
    st.sidebar.header("Configuration")
    selected_loc = st.sidebar.selectbox(
        "Select Location to Edit", 
        all_locations, 
        index=all_locations.index(current_run_loc) if current_run_loc in all_locations else 0
    )
    
    loc_config = config_data["locations"][selected_loc]
    
    source_cache = st.sidebar.text_input("Source Cache (Blank for self)", value=loc_config.get("SOURCE_CACHE") or "")
    start_date = st.sidebar.text_input("Start Date", value=loc_config.get("START_DATE", "2024-01-01"))
    end_date = st.sidebar.text_input("End Date", value=loc_config.get("END_DATE", "2026-01-01"))
    tanager_avail = st.sidebar.checkbox("Tanager Available", value=loc_config.get("TANAGER_AVAILABLE", False))
    tanager_stac_url = st.sidebar.text_input("Tanager STAC URL", value=loc_config.get("TANAGER_STAC_URL", ""))
    dragonette_avail = st.sidebar.checkbox("Dragonette Available", value=loc_config.get("DRAGONETTE_AVAILABLE", False))
    dragonette_stac_url = st.sidebar.text_input("Dragonette STAC URL", value=loc_config.get("DRAGONETTE_STAC_URL", "https://wyvern-odp.com/industry/spatio_temporal/collection.json"))
    
    # Establish bounding box variables
    roi_lon_min = loc_config.get("ROI_LON_MIN", -118.847)
    roi_lon_max = loc_config.get("ROI_LON_MAX", -118.487)
    roi_lat_min = loc_config.get("ROI_LAT_MIN", 33.905)
    roi_lat_max = loc_config.get("ROI_LAT_MAX", 34.21)

    # Reorder properly
    safe_bbox = [
        min(roi_lon_min, roi_lon_max), min(roi_lat_min, roi_lat_max),
        max(roi_lon_min, roi_lon_max), max(roi_lat_min, roi_lat_max)
    ]
    
    center_lat = (safe_bbox[1] + safe_bbox[3]) / 2
    center_lon = (safe_bbox[0] + safe_bbox[2]) / 2

    # Check session state for drawn bounds before rendering map
    new_bbox = safe_bbox
    if "mgrs_map" in st.session_state:
        map_state = st.session_state["mgrs_map"]
        if map_state and map_state.get("last_active_drawing"):
            geom = map_state["last_active_drawing"]["geometry"]
            if geom["type"] == "Polygon":
                coords = geom["coordinates"][0]
                lons = [c[0] for c in coords]
                lats = [c[1] for c in coords]
                new_bbox = [min(lons), min(lats), max(lons), max(lats)]
                st.info(f"Using newly drawn ROI bounds: {new_bbox}")

    # Always query STAC for the active bbox
    with st.spinner("Querying STAC for MGRS Tiles..."):
        try:
            discovered_tiles_dict = query_stac_for_tiles(new_bbox, start_date, end_date)
        except Exception as e:
            st.error(f"Error querying STAC: {e}")
            discovered_tiles_dict = {}

    tanager_tiles_dict = {}
    if tanager_avail and tanager_stac_url:
        with st.spinner("Querying STAC for Tanager Tiles..."):
            res = query_custom_stac_collection(tanager_stac_url)
            if "error" in res:
                st.error(f"Error querying Tanager STAC: {res['error']}")
            else:
                tanager_tiles_dict = res

    dragonette_tiles_dict = {}
    if dragonette_avail and dragonette_stac_url:
        with st.spinner("Querying STAC for Dragonette Tiles..."):
            res = query_custom_stac_collection(dragonette_stac_url)
            if "error" in res:
                st.error(f"Error querying Dragonette STAC: {res['error']}")
            else:
                dragonette_tiles_dict = res

    # Map Rendering
    m = folium.Map(location=[center_lat, center_lon], zoom_start=9, tiles="CartoDB positron")
    
    # Existing ROI Rectangle
    folium.Rectangle(
        bounds=[[safe_bbox[1], safe_bbox[0]], [safe_bbox[3], safe_bbox[2]]],
        color='red', fill=True, weight=2, fillOpacity=0.2,
        tooltip=f"Current Saved ROI ({selected_loc})"
    ).add_to(m)
    
    # Plot discovered MGRS Tiles
    for tile_id, geom in discovered_tiles_dict.items():
        folium.GeoJson(
            geom,
            name=f"Tile: {tile_id}",
            tooltip=f"MGRS Tile: {tile_id}",
            style_function=lambda x: {'color': 'blue', 'fillColor': 'blue', 'weight': 2, 'fillOpacity': 0.1}
        ).add_to(m)
        
    # Plot discovered Tanager Tiles
    for tile_id, geom in tanager_tiles_dict.items():
        folium.GeoJson(
            geom,
            name=f"Tanager: {tile_id}",
            tooltip=f"Tanager Tile: {tile_id}",
            style_function=lambda x: {'color': 'orange', 'fillColor': 'orange', 'weight': 2, 'fillOpacity': 0.2}
        ).add_to(m)
        
    # Plot discovered Dragonette Tiles
    for tile_id, geom in dragonette_tiles_dict.items():
        folium.GeoJson(
            geom,
            name=f"Dragonette: {tile_id}",
            tooltip=f"Dragonette Tile: {tile_id}",
            style_function=lambda x: {'color': 'purple', 'fillColor': 'purple', 'weight': 2, 'fillOpacity': 0.2}
        ).add_to(m)
    
    # Add drawing tool
    draw = Draw(
        draw_options={
            'polyline': False, 'polygon': False, 'circle': False,
            'marker': False, 'circlemarker': False,
            'rectangle': True
        },
        edit_options={'edit': False}
    )
    draw.add_to(m)
    
    st.write("Draw a rectangle on the map to define a new ROI. If you don't draw anything, the current ROI is preserved.")
    
    # Render st_folium
    output = st_folium(m, width=1000, height=600, key="mgrs_map")
    

    
    if st.button("Save Config"):
        # Update config dictionary
        loc_config["SOURCE_CACHE"] = source_cache if source_cache.strip() != "" else None
        loc_config["ROI_LON_MIN"] = new_bbox[0]
        loc_config["ROI_LON_MAX"] = new_bbox[2]
        loc_config["ROI_LAT_MIN"] = new_bbox[1]
        loc_config["ROI_LAT_MAX"] = new_bbox[3]
        loc_config["START_DATE"] = start_date
        loc_config["END_DATE"] = end_date
        loc_config["TANAGER_AVAILABLE"] = tanager_avail
        loc_config["TANAGER_STAC_URL"] = tanager_stac_url
        loc_config["DRAGONETTE_AVAILABLE"] = dragonette_avail
        loc_config["DRAGONETTE_STAC_URL"] = dragonette_stac_url

        
        config_data["locations"][selected_loc] = loc_config
        config_data["current_run"]["location"] = selected_loc
        
        save_config(config_data)
        
        st.success(f"Configuration saved for {selected_loc}!")
                
    st.markdown("---")
    if st.button("Close App & Continue Pipeline"):
        st.warning("Shutting down Streamlit server... Please return to your terminal.")
        os._exit(0)

if __name__ == "__main__":
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    if get_script_run_ctx() is not None:
        main()
    else:
        import subprocess, sys
        print("Launching Streamlit automatically...")
        subprocess.run([sys.executable, "-m", "streamlit", "run", sys.argv[0]] + sys.argv[1:])