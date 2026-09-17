import numpy as np
import h5py

def load_scaled_reflectance(h5_dataset, frame_idx=None, as_float32=True, fill_value_in=-9999, fill_value_out=np.nan):
    if frame_idx is not None:
        data = h5_dataset[frame_idx]
    else:
        data = h5_dataset[:]
        
    if not as_float32:
        return data
        
    scale = h5_dataset.attrs.get("scale_to_float", 0.0001)
    _fill = h5_dataset.attrs.get("_FillValue", fill_value_in)
    
    float_data = data.astype(np.float32)
    valid_mask = (data != _fill)
    float_data[valid_mask] = float_data[valid_mask] * scale
    float_data[~valid_mask] = fill_value_out
    
    return float_data

def scale_reflectance_for_storage(float_data, scale_factor=10000.0, fill_value_in=np.nan, fill_value_out=-9999):
    if np.isnan(fill_value_in):
        invalid_mask = np.isnan(float_data)
    else:
        invalid_mask = (float_data == fill_value_in)
        
    int_data = np.zeros_like(float_data, dtype=np.int16)
    valid_mask = ~invalid_mask
    scaled = np.round(float_data[valid_mask] * scale_factor)
    scaled = np.clip(scaled, -32768, 32767)
    int_data[valid_mask] = scaled.astype(np.int16)
    int_data[invalid_mask] = fill_value_out
    
    return int_data

def add_reflectance_attributes(h5_dataset, scale_to_float=0.0001, fill_value=-9999):
    h5_dataset.attrs["scale_to_float"] = float(scale_to_float)
    h5_dataset.attrs["_FillValue"] = np.int16(fill_value)
    h5_dataset.attrs["units"] = "Reflectance"
