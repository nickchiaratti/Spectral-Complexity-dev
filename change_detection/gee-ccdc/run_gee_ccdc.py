import ee
import geemap
import yaml
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.dates as mdates

# Configuration
CONFIG_PATH = r'F:\Resilio\IMGS 890 Research\Spectral-Complexity-dev\locations_config.yaml'
OUT_DIR = r'F:\Resilio\IMGS 890 Research\Spectral-Complexity-dev\change_detection\gee-ccdc\output'

def init_gee():
    try:
        ee.Initialize(project="project-ee18dbee-cd7e-4d08-812")
    except Exception as e:
        ee.Authenticate()
        ee.Initialize(project="project-ee18dbee-cd7e-4d08-812")

def load_config():
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    loc = config['locations']['Rochesterv2']
    return loc

def prep_landsat_c2(image):
    # Bit 1 is dilated cloud, Bit 3 is cloud, Bit 4 is cloud shadow
    qa = image.select('QA_PIXEL')
    cloud_shadow_bit_mask = (1 << 4)
    clouds_bit_mask = (1 << 3)
    dilated_cloud_mask = (1 << 1)
    
    mask = qa.bitwiseAnd(cloud_shadow_bit_mask).eq(0) \
        .And(qa.bitwiseAnd(clouds_bit_mask).eq(0)) \
        .And(qa.bitwiseAnd(dilated_cloud_mask).eq(0))
        
    # Apply scaling factors for C02 Level 2 and scale to 0-10000 for standard CCDC
    # (0.0000275 * 10000 = 0.275, -0.2 * 10000 = -2000)
    optical_bands = image.select(['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7']).multiply(0.275).add(-2000)
    
    return image.addBands(optical_bands, None, True).updateMask(mask).copyProperties(image, ["system:time_start"])

def run_ccdc():
    init_gee()
    os.makedirs(OUT_DIR, exist_ok=True)
    
    loc = load_config()
    start_date = loc['START_DATE']
    end_date = loc['END_DATE']
    
    roi = ee.Geometry.Rectangle([
        loc['ROI_LON_MIN'], loc['ROI_LAT_MIN'], 
        loc['ROI_LON_MAX'], loc['ROI_LAT_MAX']
    ])
    
    # Coordinates derived from array indices mapping
    # NITE Tarp (724, 752)
    nite_coords = [-77.5032, 43.1398]
    # Shadow Pines (771, 706)
    shadow_pines_coords = [-77.4859, 43.1522]
    
    points = {
        'NITE_Tarp': ee.Geometry.Point(nite_coords),
        'Shadow_Pines': ee.Geometry.Point(shadow_pines_coords)
    }
    
    print("Building Landsat Collection...")
    l8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").filterBounds(roi).filterDate(start_date, end_date)
    l9 = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2").filterBounds(roi).filterDate(start_date, end_date)
    
    col = l8.merge(l9).map(prep_landsat_c2)
    
    # Rename bands for CCDC
    col = col.select(
        ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7'], 
        ['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2']
    )
    
    print("Running CCDC Algorithm...")
    # CCDC Parameters
    ccdc_params = {
        'collection': col,
        'breakpointBands': ['Green', 'Red', 'NIR', 'SWIR1', 'SWIR2'],
        'tmaskBands': ['Green', 'SWIR2'],
        'minObservations': 6,
        'chiSquareProbability': 0.99,
        'minNumOfYearsScaler': 1.33,
        'dateFormat': 1, # fractional years
        'lambda': 20,
        'maxIterations': 25000
    }
    
    ccdc_result = ee.Algorithms.TemporalSegmentation.Ccdc(**ccdc_params)
    
    print("Extracting Time Series Data for Specific Pixels...")
    
    for name, pt in points.items():
        extract_and_plot_pixel(name, pt, col, ccdc_result)
        
    print("Generating Change Map Overlay...")
    # Extract the most recent break date (tBreak) for each pixel
    # tBreak is an array. arrayLength() gets number of breaks. arraySlice gets last break.
    tBreak_array = ccdc_result.select('tBreak')
    num_breaks = tBreak_array.arrayLength(0)
    # Mask where at least 1 break occurred
    has_break = num_breaks.gt(0)
    last_break_array = tBreak_array.arraySlice(0, -1)
    last_break_img = last_break_array.arrayFlatten([['last_break']]).updateMask(has_break)
    
    out_png = os.path.join(OUT_DIR, 'ccdc_last_break_map.png')
    if not os.path.exists(out_png):
        print("Downloading CCDC break map overlay as PNG...")
        try:
            # Create a visualization mapping fractional years 2015 to 2026 to colors
            vis_params = {
                'min': 2015,
                'max': 2026,
                'palette': ['#000080', '#0000D9', '#4000FF', '#8000FF', '#0080FF', '#00FFFF',
                            '#00FF80', '#80FF00', '#DAFF00', '#FFFF00', '#FFDA00', '#FF8000', '#FF0000']
            }
            
            # Create a smaller ROI around the points for the thumbnail to avoid GEE memory limits
            # CCDC over the full Rochesterv2 bounding box requires a batch export task
            vis_roi = points['NITE_Tarp'].buffer(150).bounds()
            
            # Reproject to Web Mercator for correct thumbnail aspect ratio, or use region directly
            thumb_url = last_break_img.visualize(**vis_params).getThumbURL({
                'region': vis_roi,
                'dimensions': 1024,
                'format': 'png'
            })
            
            import requests
            r = requests.get(thumb_url)
            with open(out_png, 'wb') as f:
                f.write(r.content)
            print(f"Successfully downloaded map overlay to {out_png}")
            
        except Exception as e:
            print("Export via getThumbURL failed.", e)
    else:
        print(f"Map overlay already exists at {out_png}")

def get_fractional_year(date):
    year = date.year
    start_of_year = datetime(year, 1, 1).timestamp()
    start_of_next_year = datetime(year+1, 1, 1).timestamp()
    year_duration = start_of_next_year - start_of_year
    fraction = (date.timestamp() - start_of_year) / year_duration
    return year + fraction

def frac_year_to_date(frac_year):
    year = int(frac_year)
    remainder = frac_year - year
    start_of_year = datetime(year, 1, 1).timestamp()
    start_of_next_year = datetime(year+1, 1, 1).timestamp()
    year_duration = start_of_next_year - start_of_year
    return datetime.fromtimestamp(start_of_year + remainder * year_duration)

def extract_and_plot_pixel(name, pt, col, ccdc_result):
    print(f"Processing {name}...")
    
    # 1. Extract raw observations
    def extract_point(img):
        val = img.reduceRegion(ee.Reducer.first(), pt, 30)
        return ee.Feature(None, {
            'time': img.date().millis(),
            'NIR': val.get('NIR')
        })
    
    timeseries = col.map(extract_point).getInfo()['features']
    
    obs_times = []
    obs_vals = []
    for f in timeseries:
        props = f['properties']
        if props.get('NIR') is not None:
            dt = datetime.fromtimestamp(props['time']/1000.0)
            obs_times.append(get_fractional_year(dt))
            obs_vals.append(props['NIR'])
            
    # 2. Extract CCDC coefficients
    ccdc_info = ccdc_result.reduceRegion(ee.Reducer.first(), pt, 30).getInfo()
    
    if ccdc_info.get('tStart') is None:
        print(f"No CCDC segments found for {name}.")
        return

    tStarts = ccdc_info['tStart']
    tEnds = ccdc_info['tEnd']
    tBreaks = ccdc_info['tBreak']
    coefs_nir = ccdc_info['NIR_coefs']
    
    plt.figure(figsize=(12, 6))
    
    # Plot raw data (scaled back to 0-1)
    obs_vals_scaled = np.array(obs_vals) / 10000.0
    plt.scatter(obs_times, obs_vals_scaled, color='gray', s=10, alpha=0.6, label='Landsat 8/9 NIR')
    
    # Evaluate segments
    omega = 2 * np.pi
    
    for i in range(len(tStarts)):
        start = tStarts[i]
        end = tEnds[i]
        coef = coefs_nir[i]
        
        # Coefs are [intercept, slope, cos(w), sin(w), cos(2w), sin(2w), cos(3w), sin(3w)]
        print(f"Segment {i} coefs: {coef}")
        t_grid = np.linspace(start, end, 100)
        y_fit = (coef[0] + 
                 coef[1] * t_grid + 
                 coef[2] * np.cos(omega * t_grid) + 
                 coef[3] * np.sin(omega * t_grid) + 
                 coef[4] * np.cos(2 * omega * t_grid) + 
                 coef[5] * np.sin(2 * omega * t_grid) + 
                 coef[6] * np.cos(3 * omega * t_grid) + 
                 coef[7] * np.sin(3 * omega * t_grid))
                 
        y_fit_scaled = y_fit / 10000.0
                 
        plt.plot(t_grid, y_fit_scaled, color='blue', linewidth=2, label='CCDC Fit' if i==0 else "")
        
    for brk in tBreaks:
        if brk > 0:
            plt.axvline(x=brk, color='red', linestyle='--', linewidth=1.5, label='Structural Break' if brk==tBreaks[0] else "")
            
    plt.title(f'CCDC Native Time Series: {name}')
    plt.xlabel('Year')
    plt.ylabel('NIR Reflectance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f'{name}_ccdc_timeseries.png'), dpi=300)
    plt.close()
    
    print(f"Finished plotting {name}.")

if __name__ == '__main__':
    run_ccdc()
