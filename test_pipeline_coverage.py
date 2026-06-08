import h5py, numpy as np
from rasterio.warp import reproject, Resampling
from rasterio.transform import Affine
from rasterio.crs import CRS
from pyproj import Transformer

bbox=[-77.770166, 42.961778, -77.376776, 43.342135]
res=30.0
master_crs = CRS.from_string('+proj=aea +lat_1=29.5 +lat_2=45.5 +lat_0=23.0 +lon_0=-96.0 +x_0=0 +y_0=0 +datum=NAD83 +units=m +no_defs')
transformer = Transformer.from_crs('EPSG:4326', master_crs, always_xy=True)
xs, ys = transformer.transform([bbox[0], bbox[2], bbox[2], bbox[0]], [bbox[3], bbox[3], bbox[1], bbox[1]])

min_x, max_x = min(xs), max(xs)
min_y, max_y = min(ys), max(ys)
min_x = np.floor(min_x / res) * res
max_x = np.ceil(max_x / res) * res
min_y = np.floor(min_y / res) * res
max_y = np.ceil(max_y / res) * res
width = int((max_x - min_x) / res)
height = int((max_y - min_y) / res)
master_transform = Affine(res, 0, min_x, 0, -res, max_y)

f = h5py.File('C:/satelliteImagery/HLS30/HLS_Rochesterv2_STAC_Native_2025.h5', 'r')
sr_node = f['HDFEOS/GRIDS/HLSL30_Merged/Data Fields/surface_reflectance']
src_tf = Affine.from_gdal(*sr_node.attrs['GeoTransform'])
src_crs = CRS.from_wkt(sr_node.attrs['spatial_ref'])

valid_indices = []
for i in range(10):
    tmp_sr = np.full((1, height, width), np.nan, dtype=np.float32)
    reproject(source=sr_node[i, 0:1], destination=tmp_sr, src_transform=src_tf, src_crs=src_crs, dst_transform=master_transform, dst_crs=master_crs, resampling=Resampling.cubic, src_nodata=np.nan, dst_nodata=np.nan)
    coverage = np.sum(~np.isnan(tmp_sr[0])) / (height*width) * 100
    valid_indices.append(coverage)

print('First 10 coverages:', valid_indices)
