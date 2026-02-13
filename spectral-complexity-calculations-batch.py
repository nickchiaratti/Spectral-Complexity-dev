from spectral_complexity_calculations import process_file

SC_Param_list = [
    'band_count',
    'dimensionality',
    'simplex',
    None, 
    'magnitude'
]

Stacked_file_list = [
    "C:/satelliteImagery/Tanager/Tait/Tanager_Stack_Tait_HDFEOS.h5"
]

for file in Stacked_file_list:
    for param in SC_Param_list:
        process_file(file, norm_param=param)

