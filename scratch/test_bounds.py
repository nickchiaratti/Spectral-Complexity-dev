import rasterio
from rasterio.windows import from_bounds
from pyproj import Transformer, CRS
import os

print(f'Rasterio version: {rasterio.__version__}')

cache_bbox = [-120.017395, 34.389913, -119.549103, 35.163705]

# Find a cached tif in SantaBarbara
cache_dir = r"C:\satelliteImagery\HLS30\HLSS30-SourceData\SantaBarbara\STAC_CACHE"
files = [f for f in os.listdir(cache_dir) if f.endswith('.tif')]
if not files:
    print("No files found.")
else:
    fpath = os.path.join(cache_dir, files[0])
    with rasterio.open(fpath) as cached_src:
        transformer = Transformer.from_crs("EPSG:4326", cached_src.crs, always_xy=True)
        xs, ys = transformer.transform(
            [cache_bbox[0], cache_bbox[2], cache_bbox[2], cache_bbox[0]],
            [cache_bbox[3], cache_bbox[3], cache_bbox[1], cache_bbox[1]]
        )
        c_minx, c_maxx, c_miny, c_maxy = min(xs), max(xs), min(ys), max(ys)
        
        print("cached_src.bounds:")
        print(f"  left:   {cached_src.bounds.left}")
        print(f"  right:  {cached_src.bounds.right}")
        print(f"  bottom: {cached_src.bounds.bottom}")
        print(f"  top:    {cached_src.bounds.top}")
        
        print("\nc_bounds:")
        print(f"  c_minx: {c_minx}")
        print(f"  c_maxx: {c_maxx}")
        print(f"  c_miny: {c_miny}")
        print(f"  c_maxy: {c_maxy}")
        
        print("\nDifferences (cached - expected):")
        print(f"  left diff:   {cached_src.bounds.left - c_minx}")
        print(f"  right diff:  {cached_src.bounds.right - c_maxx}")
        print(f"  bottom diff: {cached_src.bounds.bottom - c_miny}")
        print(f"  top diff:    {cached_src.bounds.top - c_maxy}")
        
        cache_valid = (
            cached_src.bounds.left <= c_minx + 40 and
            cached_src.bounds.right >= c_maxx - 40 and
            cached_src.bounds.bottom <= c_miny + 40 and
            cached_src.bounds.top >= c_maxy - 40
        )
        
        print("\nConditions:")
        print(f"  left <= c_minx + 40:   {cached_src.bounds.left <= c_minx + 40}")
        print(f"  right >= c_maxx - 40:  {cached_src.bounds.right >= c_maxx - 40}")
        print(f"  bottom <= c_miny + 40: {cached_src.bounds.bottom <= c_miny + 40}")
        print(f"  top >= c_maxy - 40:    {cached_src.bounds.top >= c_maxy - 40}")
        
        print(f"\ncache_valid: {cache_valid}")
