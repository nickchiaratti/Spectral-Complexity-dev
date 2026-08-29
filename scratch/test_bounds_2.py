import rasterio
from rasterio.windows import from_bounds
from pyproj import Transformer, CRS
import os

cache_bbox = [-120.017395, 34.389913, -119.549103, 35.163705]
cache_dir = r"C:\satelliteImagery\HLS30\HLSS30-SourceData\SantaBarbara\STAC_CACHE"

files_to_test = [
    "HLS.S30.T10SGE.2023056T184341.v2.0.tif",
    "HLS.S30.T11SKV.2026035T185451.v2.0.tif"
]

for fname in files_to_test:
    fpath = os.path.join(cache_dir, fname)
    if not os.path.exists(fpath):
        print(f"{fname} not found.")
        continue
    with rasterio.open(fpath) as cached_src:
        transformer = Transformer.from_crs("EPSG:4326", cached_src.crs, always_xy=True)
        xs, ys = transformer.transform(
            [cache_bbox[0], cache_bbox[2], cache_bbox[2], cache_bbox[0]],
            [cache_bbox[3], cache_bbox[3], cache_bbox[1], cache_bbox[1]]
        )
        c_minx, c_maxx, c_miny, c_maxy = min(xs), max(xs), min(ys), max(ys)
        
        cache_valid = (
            cached_src.bounds.left <= c_minx + 40 and
            cached_src.bounds.right >= c_maxx - 40 and
            cached_src.bounds.bottom <= c_miny + 40 and
            cached_src.bounds.top >= c_maxy - 40
        )
        print(f"{fname}: {cache_valid}")
        if not cache_valid:
            print(f"  left diff: {cached_src.bounds.left - c_minx}")
            print(f"  right diff: {cached_src.bounds.right - c_maxx}")
            print(f"  bottom diff: {cached_src.bounds.bottom - c_miny}")
            print(f"  top diff: {cached_src.bounds.top - c_maxy}")
