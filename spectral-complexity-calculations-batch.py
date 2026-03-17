from spectral_complexity_calculations import process_file

SC_Param_list = [ 
    #'bandCount',
    None, 
]

Gram_param_list = [
    #'minEndmember',
    #'datasetMean',
    'general',
]

Stacked_file_list = [
    "C:/satelliteImagery/Tanager/Tait/Tanager_Stack_Tait_HDFEOS.h5",
    "C:/satelliteImagery/LANDSAT/Tait/LANDSAT_Stack_Tait_GEE_2015_2025.h5",
    "C:/satelliteImagery/Tanager/Rochester/Tanager_Stack_Rochester_HDFEOS.h5",
    "C:/satelliteImagery/LANDSAT/Rochester/LANDSAT_Stack_Rochester_GEE_2015_2025.h5",
]

for file in Stacked_file_list:
    for gram in Gram_param_list:
        for param in SC_Param_list:
            process_file(file, norm_param=param, gram_type=gram)

