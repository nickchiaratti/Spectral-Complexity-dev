import h5py
import glob
import numpy as np
from rasterio.warp import reproject, Resampling
from rasterio.control import GroundControlPoint
from pyproj import CRS
from rasterio.transform import transform_from_bounds

f = glob.glob('C:/satelliteImagery/Tanager/**/*_basic_sr_hdf5.h5', recursive=True)[0]
h5f = h5py.File(f, 'r')
geo = h5f['HDFEOS/SWATHS/HYP/Geolocation Fields']
df = h5f['HDFEOS/SWATHS/HYP/Data Fields']

lat = geo['Latitude'][:]
lon = geo['Longitude'][:]
src_data = df['sun_azimuth'][:]

gcps = []
step = 10
for r in range(0, lat.shape[0], step):
    for c in range(0, lat.shape[1], step):
        gcps.append(GroundControlPoint(row=r, col=c, x=lon[r, c], y=lat[r, c]))
gcps.append(GroundControlPoint(row=lat.shape[0]-1, col=lat.shape[1]-1, x=lon[-1, -1], y=lat[-1, -1]))

master_transform = transform_from_bounds(-119, 33, -118, 34, 50, 50)
master_crs = CRS.from_epsg(4326)

src_data_3d = src_data[np.newaxis, ...]
incoming_3d = np.zeros((1, 50, 50), dtype=np.float32)

print("Starting reprojection 3D...")
try:
    reproject(
        source=src_data_3d, destination=incoming_3d,
        gcps=gcps, src_crs=CRS.from_epsg(4326),
        dst_transform=master_transform, dst_crs=master_crs,
        resampling=Resampling.nearest, src_nodata=-9999.0, dst_nodata=-9999.0
    )
    print('Successful reprojection 3D. Check for stripes:')
    print('Identical rows?', np.allclose(incoming_3d[0, 0, :], incoming_3d[0, 1, :]))
    print(incoming_3d[0, :3, :3])
except Exception as e:
    print('Error 3D:', e)

print("Starting reprojection 2D...")
try:
    incoming_2d = np.zeros((50, 50), dtype=np.float32)
    reproject(
        source=src_data, destination=incoming_2d,
        gcps=gcps, src_crs=CRS.from_epsg(4326),
        dst_transform=master_transform, dst_crs=master_crs,
        resampling=Resampling.nearest, src_nodata=-9999.0, dst_nodata=-9999.0
    )
    print('Successful reprojection 2D. Check for stripes:')
    print('Identical rows?', np.allclose(incoming_2d[0, :], incoming_2d[1, :]))
    print(incoming_2d[:3, :3])
except Exception as e:
    print('Error 2D:', e)
