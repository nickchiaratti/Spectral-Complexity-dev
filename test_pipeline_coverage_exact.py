import sys
import os
sys.path.append(os.path.dirname(os.path.abspath('f:/Resilio/IMGS 890 Research/Spectral-Complexity-dev/HLSX30/HLST-constellation-to-hdf5.py')))

import h5py, numpy as np
from rasterio.warp import reproject, Resampling
from rasterio.transform import Affine
from rasterio.crs import CRS
from pyproj import Transformer
import yaml

with open("f:/Resilio/IMGS 890 Research/Spectral-Complexity-dev/HLSX30/locations_config.yaml", "r") as f:
    config_data = yaml.safe_load(f)
Location = config_data.get("current_run", {}).get("location", "Palisades")
config = config_data["locations"][Location]

ROI_LON_MIN = config["ROI_LON_MIN"]
ROI_LON_MAX = config["ROI_LON_MAX"]
ROI_LAT_MIN = config["ROI_LAT_MIN"]
ROI_LAT_MAX = config["ROI_LAT_MAX"]
safe_bbox = [
    min(ROI_LON_MIN, ROI_LON_MAX), max(ROI_LAT_MIN, ROI_LAT_MAX), 
    max(ROI_LON_MIN, ROI_LON_MAX), min(ROI_LAT_MIN, ROI_LAT_MAX)
]
safe_bbox = [min(safe_bbox[0], safe_bbox[2]), min(safe_bbox[1], safe_bbox[3]), max(safe_bbox[0], safe_bbox[2]), max(safe_bbox[1], safe_bbox[3])]
TARGET_RESOLUTION = 30.0

def calculate_master_grid(bbox, resolution):
    min_lon, min_lat, max_lon, max_lat = bbox
    central_lon = (min_lon + max_lon) / 2.0
    central_lat = (min_lat + max_lat) / 2.0
    lat_1 = min_lat + (max_lat - min_lat) / 6.0
    lat_2 = max_lat - (max_lat - min_lat) / 6.0
    
    proj_str = f"+proj=aea +lat_1={lat_1:.6f} +lat_2={lat_2:.6f} +lat_0={central_lat:.6f} +lon_0={central_lon:.6f} +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
    dst_crs = CRS.from_string(proj_str)
    from rasterio.transform import from_bounds as transform_from_bounds
    transformer = Transformer.from_crs("EPSG:4326", dst_crs, always_xy=True)
    xs, ys = transformer.transform([bbox[0], bbox[2], bbox[2], bbox[0]], [bbox[3], bbox[3], bbox[1], bbox[1]])
    minx, maxx, miny, maxy = min(xs), max(xs), min(ys), max(ys)
    width = int(np.ceil((maxx - minx) / resolution))
    height = int(np.ceil((maxy - miny) / resolution))
    transform = transform_from_bounds(minx, miny, maxx, maxy, width, height)
    return dst_crs, transform, width, height

master_crs, master_transform, master_width, master_height = calculate_master_grid(safe_bbox, TARGET_RESOLUTION)

f = h5py.File('C:/satelliteImagery/HLS30/HLS_Rochesterv2_STAC_Native_2025.h5', 'r')
sr_node = f['HDFEOS/GRIDS/HLSL30_Merged/Data Fields/surface_reflectance']
src_tf = Affine.from_gdal(*sr_node.attrs['GeoTransform'])
src_crs = CRS.from_wkt(sr_node.attrs['spatial_ref'])

for i in range(10):
    tmp_sr = np.full((1, master_height, master_width), np.nan, dtype=np.float32)
    reproject(source=sr_node[i, 0:1], destination=tmp_sr, src_transform=src_tf, src_crs=src_crs, dst_transform=master_transform, dst_crs=master_crs, resampling=Resampling.cubic, src_nodata=np.nan, dst_nodata=np.nan)
    valid_pixels = np.sum(~np.isnan(tmp_sr[0]))
    coverage = (valid_pixels / (master_height * master_width)) * 100
    print(f"Index {i}: Coverage = {coverage}%")
