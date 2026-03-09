from spectral_complexity_calculations import process_file

SC_Param_list = [ 
    'bandCount',
    None, 
]

Gram_param_list = [
    'datasetMean',
    'minEndmember',
    'general',
]

Stacked_file_list = [
    "C:/satelliteImagery/LANDSAT/Tait/LANDSAT_Stack_Tait_HDFEOS.h5",
    "C:/satelliteImagery/Tanager/Tait/Tanager_Stack_Tait_HDFEOS.h5",
    "C:/satelliteImagery/Tanager/Tait-I-490/Tanager_Stack_Tait-I-490_HDFEOS.h5",
    "C:/satelliteImagery/LANDSAT/Tait-I-490/LANDSAT_Stack_Tait-I-490_HDFEOS.h5"
]

for file in Stacked_file_list:
    for param in SC_Param_list:
        for gram in Gram_param_list:
            process_file(file, norm_param=param, gram_type=gram)

