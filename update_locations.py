import yaml
from rasterio.warp import transform_bounds
import math

with open('locations_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

for loc, params in config['locations'].items():
    if 'MGRS_EPSG' not in params: continue
    
    epsg = params['MGRS_EPSG']
    roi_bounds = [params['ROI_LON_MIN'], params['ROI_LAT_MIN'], params['ROI_LON_MAX'], params['ROI_LAT_MAX']]
    
    # Project ROI bounds to the MGRS EPSG
    left, bottom, right, top = transform_bounds('EPSG:4326', f'EPSG:{epsg}', *roi_bounds)
    
    # Snap UL to nearest 30m
    ul_x = math.floor(left / 30.0) * 30.0
    ul_y = math.ceil(top / 30.0) * 30.0
    
    # Snap BR to nearest 30m
    br_x = math.ceil(right / 30.0) * 30.0
    br_y = math.floor(bottom / 30.0) * 30.0
    
    # Calculate width and height
    width = int((br_x - ul_x) / 30.0)
    height = int((ul_y - br_y) / 30.0)
    
    params['MGRS_UL_X'] = ul_x
    params['MGRS_UL_Y'] = ul_y
    params['MGRS_WIDTH'] = width
    params['MGRS_HEIGHT'] = height

with open('locations_config.yaml', 'w') as f:
    yaml.dump(config, f, sort_keys=False)

print("Updated locations_config.yaml")
