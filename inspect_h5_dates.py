import h5py
from datetime import datetime
import numpy as np

def min_date(file_path):
    with h5py.File(file_path, 'r') as f:
        min_ts = float('inf')
        
        if 'HDFEOS/GRIDS/HLSL30/Data Fields/surface_reflectance' in f:
            ds = f['HDFEOS/GRIDS/HLSL30/Data Fields/surface_reflectance']
            if 'acquisition_time' in ds.attrs:
                ts = ds.attrs['acquisition_time']
                min_ts = min(min_ts, np.min(ts))
                
        if 'HDFEOS/GRIDS/HLSS30/Data Fields/surface_reflectance' in f:
            ds = f['HDFEOS/GRIDS/HLSS30/Data Fields/surface_reflectance']
            if 'acquisition_time' in ds.attrs:
                ts = ds.attrs['acquisition_time']
                min_ts = min(min_ts, np.min(ts))
                
        if min_ts != float('inf'):
            print(f"Minimum timestamp: {min_ts}")
            print(f"Earliest date: {datetime.utcfromtimestamp(min_ts).strftime('%Y-%m-%d %H:%M:%S UTC')}")
        else:
            print("No acquisition_time found.")

min_date('C:/satelliteImagery/HLST30/HLST_Rochesterv2_Harmonized.h5')
