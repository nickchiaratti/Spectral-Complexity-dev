from spectral_complexity_calculations import process_file

SC_Param_list = [ 
    'bandCount',
    None, 
]

Gram_param_list = [
    'datasetMean',
    'general',
]

Stacked_file_list = [
    "C:/satelliteImagery/Tanager/Tait/Tanager_Stack_Tait_HDFEOS.h5",
    "C:/satelliteImagery/LANDSAT/Tait/LANDSAT_Stack_Tait_HDFEOS.h5"
]

for file in Stacked_file_list:
    for param in SC_Param_list:
        for gram in Gram_param_list:
            process_file(file, norm_param=param, gram_type=gram)

