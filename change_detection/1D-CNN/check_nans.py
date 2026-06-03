import h5py, numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'HLSX30')))
from SpecComplex import get_landsat_mask
f = h5py.File('c:/satelliteImagery/LANDSAT/Tait/LANDSAT_Stack_Tait_GEE_2015_2025_WRS16_SC_EM-7_Gram-minEndmember_Norm-bandCount.h5', 'r')
data_grp = f['/HDFEOS/GRIDS/LANDSAT/Data Fields']
z_score = data_grp['sliding_volume_z_score_masked'][:]
num_frames, h, w = z_score.shape
valid_masks=[]
for i in range(num_frames):
    # get_landsat_mask returns True for Invalid, so invert it
    valid_masks.append(~get_landsat_mask(data_grp, i, (h, w)))
valid_mask = np.stack(valid_masks)
# valid_mask is True for valid data
print('NaNs in valid frames:', np.isnan(z_score[valid_mask]).sum())
