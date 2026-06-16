import h5py
import numpy as np
from datetime import datetime

file_path = r"C:\satelliteImagery\HLST30\HLST_Malibu_Harmonized_SC_EM-7_Norm-bandCount.h5"

try:
    with h5py.File(file_path, 'r') as f:
        # Check if Fmask exists in original files
        grids = ['HLSS30', 'HLSL30', 'TANAGER']
        
        # We know HARMONIZED common_mask has 866 frames
        harm_mask = f['HDFEOS/GRIDS/HARMONIZED/Data Fields/common_mask'][:, 464, 644]
        acq_times = f['HDFEOS/GRIDS/HARMONIZED/Data Fields/common_mask'].attrs['acquisition_time']
        
        years = np.array([datetime.fromtimestamp(t).year for t in acq_times])
        
        for yr in range(2015, 2026):
            mask_yr = harm_mask[years == yr]
            if len(mask_yr) > 0:
                print(f"Year {yr}: {len(mask_yr)} frames, {np.sum(mask_yr)} masked ({(np.sum(mask_yr)/len(mask_yr))*100:.1f}%)")
                
except Exception as e:
    print(f"Error: {e}")
