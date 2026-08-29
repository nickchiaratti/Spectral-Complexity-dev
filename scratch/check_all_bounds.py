import rasterio
from pyproj import Transformer
import os

cache_bbox = [-120.017395, 34.389913, -119.549103, 35.163705]
cache_dir = r"C:\satelliteImagery\HLS30\HLSS30-SourceData\SantaBarbara\STAC_CACHE"

files = [f for f in os.listdir(cache_dir) if f.endswith('.tif')]
invalid_count = 0
for fname in files:
    fpath = os.path.join(cache_dir, fname)
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
        if not cache_valid:
            invalid_count += 1

print(f"Found {invalid_count} files with invalid bounds out of {len(files)}.")
