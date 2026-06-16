import h5py
import numpy as np
from datetime import datetime

file_path = r"C:\satelliteImagery\HLST30\HLST_Malibu_Harmonized_SC_EM-7_Norm-bandCount.h5"

try:
    with h5py.File(file_path, 'r') as f:
        harm_mask = f['HDFEOS/GRIDS/HARMONIZED/Data Fields/common_mask'][:, 464, 644]
        acq_times = f['HDFEOS/GRIDS/HARMONIZED/Data Fields/common_mask'].attrs['acquisition_time']
        spacecrafts = f['HDFEOS/GRIDS/HARMONIZED/Data Fields/common_mask'].attrs['source_spacecraft']
        
        years = np.array([datetime.fromtimestamp(t).year for t in acq_times])
        
        for yr in range(2021, 2026):
            mask_yr = harm_mask[years == yr]
            sc_yr = spacecrafts[years == yr]
            
            sc_strs = [s if isinstance(s, str) else s.decode('utf-8') for s in sc_yr]
            print(f"\n--- Year {yr} ---")
            unique_sc = np.unique(sc_strs)
            for sc in unique_sc:
                mask_for_sc = mask_yr[np.array(sc_strs) == sc]
                masked_count = np.sum(mask_for_sc)
                total = len(mask_for_sc)
                print(f"  {sc}: {masked_count}/{total} masked ({(masked_count/total)*100:.1f}%)")
                
except Exception as e:
    print(f"Error: {e}")
