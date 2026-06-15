import os
import h5py
import numpy as np
import tkinter as tk
from tkinter import filedialog
import SpecComplex as sc
import pandas as pd
from skimage import exposure
import matplotlib.pyplot as plt
from datetime import datetime, timezone


# --- Configuration ---
TILE_SIZE = 3          # Size of the window (NxN pixels) for volume calc
SLIDING_STRIDE = 1      # Stride for sliding window (1 = every pixel, higher = faster)
#FRAME_IDX = 80  #Landsat frame 80 for 2025-09-19
#file_path = "C:/satelliteImagery/LANDSAT/Tait/LANDSAT_Stack_Tait_HDFEOS.h5"
FRAME_IDX = 3 #Tanager frame 3 for 2025-09-19
file_path = "C:/satelliteImagery/Tanager/Tait/Tanager_Stack_Tait_HDFEOS.h5"
SUBSET_SIZE = 3
SUBSET_CONFIGS = [
    #{'x': 122, 'y': 47, 'label': 'Nature Area [122, 47]'},
    {'x': 35, 'y': 74, 'label': 'ROCX Target [74, 35]'}
]

# --- Parameters for Maximum-Distance ---
num_endmembers = 7
MAX_DIST_P2 = 0
gram_type = None
SC_Param_Norm = None #'magnitude' 'band_count' None 'dimensionality' 'simplex'

def maximumDistance(data, num_endmembers, mnf_data=0):
    '''
    Args:
        data (np.ndarray): 2D data [npixels, nbands]
        num_endmembers (int): number of endmembers to be calculated (choose more than expected to find)
        mnf_data (np.ndarray): MNF data [npixels, nbands]
    Returns:
        endmembers [bands, num_endmembers]
        endmembers_index [1, num_endmembers]
    '''
    # data = 2D data [npixels, nbands]
    # num_endmembers = number of endmembers to be calculated (choose more than expected to find)
    # if MNF data is not available, code will assign img as mnf_data
    #print('---> In MaxD extracting endmembers and Grammian ...')
       
    # Ensure data is 2D [npixels, nbands]
    if data.ndim == 3:
        # Flatten 3D cube [rows, cols, bands] -> 2D [pixels, bands]
        image2D = np.reshape(data, (data.shape[0] * data.shape[1], data.shape[2]), order="F")
    else:
        image2D = data
    if np.min(data) < -1:
        warnings.warn('Data contains negative values')
    if np.max(data) > 1:
        warnings.warn('Data contains values greater than 1')
        data = np.clip(data, 0, 1)

    # --- NaN Handling ---
    # Identify valid pixels (rows) that do not contain any NaN values
    valid_mask = ~np.isnan(image2D).any(axis=1)
    
    # Check if we have enough valid pixels
    if np.sum(valid_mask) < num_endmembers:
        print(f"Not enough valid pixels (no NaNs) to find {num_endmembers} endmembers. Found {np.sum(valid_mask)} valid pixels.")
        # Return empty/zero arrays with correct shape [bands, num]
        return np.zeros([image2D.shape[1], num_endmembers]), np.zeros([1, num_endmembers]), np.zeros([num_endmembers])

    # Filter data to keep only valid pixels
    valid_data = image2D[valid_mask]
    
    # Store original indices to map back later
    # valid_indices[i] contains the index in the original flattened image2D corresponding to the i-th row in valid_data
    valid_indices = np.where(valid_mask)[0]
    
    if mnf_data == 0:
        mnf_data = valid_data
    else:
        # If mnf_data was provided, we must reshape and filter it exactly the same way
        mnf_2D = np.reshape(mnf_data, (mnf_data.shape[0] * mnf_data.shape[1], mnf_data.shape[2]), order="F")
        mnf_data = mnf_2D[valid_mask]

    data = np.transpose(valid_data)
    data2 = np.transpose(mnf_data)
    if np.min(data) < -1:
        raise ValueError('Data contains negative values')
    if np.max(data) > 1:
        raise ValueError('Data contains values greater than 1')

    # find data size
    num_bands = data.shape[0]
    num_pix = data.shape[1]

    # calculate magnitude of all vectors to find min and max
    magnitude = np.sum(np.square(data), axis=0)
    idx1 = np.argmax(magnitude)
    idx2 = np.argmin(magnitude)

    # create empty output arrays for endmembers
    endmembers = np.zeros([num_bands, num_endmembers])
    endmembers_index = np.zeros([1, num_endmembers])    

    # assign largest and smallest vector as first and second endmembers
    endmembers[:, 0] = np.transpose(data[:, idx1])
    endmembers[:, 1] = np.transpose(data[:, idx2])
    
    # Map back to original indices
    endmembers_index[0, 0] = valid_indices[idx1]
    endmembers_index[0, 1] = valid_indices[idx2]

    data_proj = np.matrix(data2)
    identity_matrix = np.identity(num_bands)


    loop = np.arange(3, num_endmembers + 1)
    for i in loop:
        diff = []
        pseudo = []
        # calc difference between endmembers
        diff = np.matrix(data_proj[:, idx2] - data_proj[:, idx1])
        # caclualte pseudo inverse of difference vector
        pseudo = np.linalg.pinv(diff)
        data_proj = np.matmul((identity_matrix - np.matmul(diff, pseudo)), data_proj)

        idx1 = idx2
        vec = data_proj[:, idx2] # Shape (bands, 1)
        # Ensure it's a column vector
        if vec.ndim == 1:
            vec = vec[:, np.newaxis]
            
        diff_new = np.sum(np.square(vec - data_proj), axis=0)

        # find ne maximum distance for next endmember
        idx2 = np.argmax(diff_new)

        # assign to endmember file
        endmembers[:, i - 1] = np.transpose(data[:, idx2])
        
        # Map back to original index
        endmembers_index[0, i - 1] = valid_indices[idx2]


    return endmembers, endmembers_index

def calcGramLocal(endmembers, mean_vector):
    """
    Calculates the Local Gram matrix.
    1. Subtracts the mean vector from all other endmembers (centering the simplex on x).
    2. Calculates the Gram matrix of these centered vectors.
    """
    # Reduce to current number of endmembers
    # Shape: (Bands, N)
    localized_vectors = endmembers - mean_vector[:, np.newaxis]

    # Calculate Gram Matrix
    # G = V^T * V
    gram = np.matmul(localized_vectors.T, localized_vectors)
    return gram

def calcGramGeneral(endmembers):
    # calculate gram matrix = V^T * V
    gram = np.matmul(np.transpose(endmembers), endmembers)
    return gram

def extract_subset(data, center_x, center_y, size):
    """
    Extracts a spatial subset from the data cube.
    Data format expected: [Bands, Height, Width]
    """
    half = size // 2
    x1 = int(center_x - half)
    x2 = int(x1 + size)
    y1 = int(center_y - half)
    y2 = int(y1 + size)
    
    # Check bounds
    _, h, w = data.shape
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    
    return data[:, y1:y2, x1:x2]

def percentile_normalize_array(arr, low=2, high=98):
    if np.all(np.isnan(arr)): return np.zeros_like(arr)
    p_low, p_high = np.nanpercentile(arr, (low, high))
    if p_low == p_high: return np.zeros_like(arr)
    return exposure.rescale_intensity(arr, in_range=(p_low, p_high), out_range=(0, 1)).clip(0, 1)

def process_file(filepath, norm_param=SC_Param_Norm, gram_type=gram_type):
    print(f"Processing: {filepath}")

    with h5py.File(filepath) as h5:
        grid_name = list(h5['/HDFEOS/GRIDS'].keys())[0]

        if grid_name not in ['TANAGER','LANDSAT']:
            print(f"Error: Unknown grid name: {grid_name}")
            return
            
        process_image_stack(h5, grid_name, norm_param, gram_type)

def process_image_stack(h5, sourceName, norm_param=None, gram_type='general'):
    folder_name = f"./Gram_calc_examples/{sourceName}_{num_endmembers}EM/"
    os.makedirs(folder_name, exist_ok=True)

    grid_name = sourceName
    base_fields_path = f"/HDFEOS/GRIDS/{grid_name}/Data Fields"
    ds_surfRef = h5[f"{base_fields_path}/surface_reflectance"]
    acq_times = ds_surfRef.attrs.get('acquisition_time')
    wavelengths = ds_surfRef.attrs.get("wavelengths")
    if grid_name == 'TANAGER':
        gw_mask = ds_surfRef.attrs.get("all_good_wavelengths").astype(bool)
        wavelengths = np.delete(wavelengths, np.where(~gw_mask[FRAME_IDX]))
    
    
    if grid_name == 'TANAGER':
        ds_ortho = h5[f"/HDFEOS/GRIDS/{grid_name}/Data Fields/ortho_visual"]
    

    for i, config in enumerate(SUBSET_CONFIGS): 
        
        output_filename = folder_name + f"{grid_name}_results_{i}"
        
        acq_time = acq_times[FRAME_IDX]
        dt = datetime.fromtimestamp(acq_time, tz=timezone.utc)
        subset_sr = extract_subset(ds_surfRef[FRAME_IDX, ...], config['x'], config['y'], SUBSET_SIZE)
        if grid_name == 'TANAGER':
            valid_sr = np.delete(subset_sr, np.where(~gw_mask[FRAME_IDX]), axis=0)
        else:
            valid_sr = subset_sr
        num_bands, height, width = valid_sr.shape
        #subset_sr = ds_surfRef[FRAME_IDX, ...]

        img = np.transpose(valid_sr, (1, 2, 0))
        image2D = np.reshape(img, (height * width, num_bands))
        valid_mask = ~np.isnan(image2D).any(axis=1)
        valid_data = image2D[valid_mask]

        #calculate magnitude of each pixel
        magnitudes = np.linalg.norm(valid_data, axis=1)
        magnitudes_array = np.reshape(magnitudes, (height, width))
        magnitudesdf = pd.DataFrame(magnitudes_array)
        magnitudesdf.columns = range(width)
        magnitudesdf.index = range(height)
        magnitudesdf.to_csv(output_filename+"_Magnitudes.csv", index=True, mode='w')
        plt.figure()
        plt.imshow(magnitudes_array)
        plt.colorbar()
        plt.title(f"Magnitudes of {grid_name} - {dt.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        plt.savefig(folder_name + f"{grid_name}_magnitudes_{i}.png")
        

        # Calculate Endmembers
        endmembers, endmember_idx, _ = sc.maximumDistance(valid_data, num_endmembers,0, None, norm_param)
        endmembersdf = pd.DataFrame(endmembers)
        endmembersdf.columns = range(num_endmembers)
        endmembersdf.index = wavelengths
        endmembersdf.to_csv(output_filename+"_General.csv", index=True, mode='w')

        # Calculate General Gram
        genGram=np.zeros([num_endmembers,num_endmembers,num_endmembers])
        genGramVol=np.zeros([num_endmembers,1])
        for i in range(1,num_endmembers+1):
            genGram[i-1,0:i,0:i] = calcGramGeneral(endmembers[:,0:i])
            genGramVol[i-1] = np.sqrt(np.abs(np.linalg.det(genGram[i-1,0:i,0:i])))

        genGramdf2= pd.DataFrame(genGram[2])
        genGramdf2.columns = np.arange(1,num_endmembers+1)
        genGramdf2.index = np.arange(1,num_endmembers+1)
        genGramdf7= pd.DataFrame(genGram[num_endmembers-1])
        genGramdf7.columns = np.arange(1,num_endmembers+1)
        genGramdf7.index = np.arange(1,num_endmembers+1)
        genGramdf2.to_csv(output_filename+"_General.csv", index=True, mode='a')
        genGramdf7.to_csv(output_filename+"_General.csv", index=True, mode='a')
        genGramVoldf = pd.DataFrame(genGramVol)
        genGramVoldf.index = np.arange(1,num_endmembers+1)
        genGramVoldf.to_csv(output_filename+"_General.csv", index=True, mode='a')    

        # Calculate Gram Localized by mean of dataset
        mean_dataset = valid_data.mean(axis=0)
        gramMeanDataset=np.zeros([num_endmembers,num_endmembers,num_endmembers])
        gramMeanDatasetVol=np.zeros(num_endmembers)
        
        for i in range(1,num_endmembers+1):
            gramMeanDataset[i-1,0:i,0:i] = calcGramLocal(endmembers[:,0:i], mean_dataset)
            gramMeanDatasetVol[i-1] = np.sqrt(np.abs(np.linalg.det(gramMeanDataset[i-1,0:i,0:i])))

        MeanDatasetEndmembers = pd.DataFrame(endmembers - mean_dataset[:, np.newaxis])
        MeanDatasetEndmembers.columns = range(num_endmembers)
        MeanDatasetEndmembers.index = wavelengths
        MeanDatasetEndmembers.to_csv(output_filename+"_MeanDataset.csv", index=True, mode='w')
        gramMeanDatasetdf2 = pd.DataFrame(gramMeanDataset[2])
        gramMeanDatasetdf2.columns = np.arange(1,num_endmembers+1)
        gramMeanDatasetdf2.index = np.arange(1,num_endmembers+1)
        gramMeanDatasetdf7 = pd.DataFrame(gramMeanDataset[num_endmembers-1])
        gramMeanDatasetdf7.columns = np.arange(1,num_endmembers+1)
        gramMeanDatasetdf7.index = np.arange(1,num_endmembers+1)
        gramMeanDatasetdf2.to_csv(output_filename+"_MeanDataset.csv", index=True, mode='a')
        gramMeanDatasetdf7.to_csv(output_filename+"_MeanDataset.csv", index=True, mode='a')
        gramMeanDatasetVoldf = pd.DataFrame(gramMeanDatasetVol)
        gramMeanDatasetVoldf.index = np.arange(1,num_endmembers+1)
        gramMeanDatasetVoldf.to_csv(output_filename+"_MeanDataset.csv", index=True, mode='a')

        # Calculate Gram Localized by mean Pixel
        dist_from_mean = np.linalg.norm(image2D - mean_dataset[np.newaxis,:], axis=1)
        closest_idx = np.argmin(dist_from_mean)
        meanClosest_pixel = image2D[closest_idx]
        gramMeanClosestPixel=np.zeros([num_endmembers,num_endmembers,num_endmembers])
        gramMeanClosestPixelVol=np.zeros(num_endmembers)
        for i in range(1,num_endmembers+1):
            gramMeanClosestPixel[i-1,0:i,0:i] = calcGramLocal(endmembers[:,0:i], meanClosest_pixel)
            gramMeanClosestPixelVol[i-1] = np.sqrt(np.abs(np.linalg.det(gramMeanClosestPixel[i-1,0:i,0:i])))
        
        MeanClosestPixelEndmembers = pd.DataFrame(endmembers - meanClosest_pixel[:, np.newaxis])
        MeanClosestPixelEndmembers.columns = range(num_endmembers)
        MeanClosestPixelEndmembers.index = wavelengths
        MeanClosestPixelEndmembers.to_csv(output_filename+"_MeanClosestPixel.csv", index=True, mode='w')
        gramMeanClosestPixeldf2 = pd.DataFrame(gramMeanClosestPixel[2])
        gramMeanClosestPixeldf2.columns = np.arange(1,num_endmembers+1)
        gramMeanClosestPixeldf2.index = np.arange(1,num_endmembers+1)
        gramMeanClosestPixeldf7 = pd.DataFrame(gramMeanClosestPixel[num_endmembers-1])
        gramMeanClosestPixeldf7.columns = np.arange(1,num_endmembers+1)
        gramMeanClosestPixeldf7.index = np.arange(1,num_endmembers+1)
        gramMeanClosestPixeldf2.to_csv(output_filename+"_MeanClosestPixel.csv", index=True, mode='a')
        gramMeanClosestPixeldf7.to_csv(output_filename+"_MeanClosestPixel.csv", index=True, mode='a')
        gramMeanClosestPixelVoldf = pd.DataFrame(gramMeanClosestPixelVol)
        gramMeanClosestPixelVoldf.index = np.arange(1,num_endmembers+1)
        gramMeanClosestPixelVoldf.to_csv(output_filename+"_MeanClosestPixel.csv", index=True, mode='a')

        # Calculate Gram Localized by mean of Endmembers
        gramMeanEndmembers=np.zeros([num_endmembers,num_endmembers,num_endmembers])
        gramMeanEndmembersVol=np.zeros(num_endmembers)
        for i in range(1,num_endmembers+1):
            mean_endmembers = np.mean(endmembers[:,0:i], axis=1)
            gramMeanEndmembers[i-1,0:i,0:i] = calcGramLocal(endmembers[:,0:i], mean_endmembers)
            gramMeanEndmembersVol[i-1] = np.sqrt(np.abs(np.linalg.det(gramMeanEndmembers[i-1,0:i,0:i])))
        
        MeanEndmembersEndmembers = pd.DataFrame(endmembers - mean_endmembers[:, np.newaxis])
        MeanEndmembersEndmembers.columns = range(num_endmembers)
        MeanEndmembersEndmembers.index = wavelengths
        MeanEndmembersEndmembers.to_csv(output_filename+"_MeanEndmembers.csv", index=True, mode='w')    
        
        gramMeanEndmembersdf2 = pd.DataFrame(gramMeanEndmembers[2])
        gramMeanEndmembersdf2.columns = np.arange(1,num_endmembers+1)
        gramMeanEndmembersdf2.index = np.arange(1,num_endmembers+1)
        gramMeanEndmembersdf7 = pd.DataFrame(gramMeanEndmembers[num_endmembers-1])
        gramMeanEndmembersdf7.columns = np.arange(1,num_endmembers+1)
        gramMeanEndmembersdf7.index = np.arange(1,num_endmembers+1)
        gramMeanEndmembersdf2.to_csv(output_filename+"_MeanEndmembers.csv", index=True, mode='a')
        gramMeanEndmembersdf7.to_csv(output_filename+"_MeanEndmembers.csv", index=True, mode='a')
        gramMeanEndmembersVoldf = pd.DataFrame(gramMeanEndmembersVol)
        gramMeanEndmembersVoldf.index = np.arange(1,num_endmembers+1)
        gramMeanEndmembersVoldf.to_csv(output_filename+"_MeanEndmembers.csv", index=True, mode='a')

        # Calculate Gram Localized by Endmember closest to mean
        gramMeanClosestEndmember=np.zeros([num_endmembers,num_endmembers,num_endmembers])
        gramMeanClosestEndmemberVol=np.zeros(num_endmembers)
        for i in range(2,num_endmembers+1):
            current_endmembers = endmembers[:,0:i]
            dist_from_mean = np.linalg.norm(current_endmembers - mean_dataset[:,np.newaxis], axis=0)    
            closest_idx = np.argmin(dist_from_mean)
            meanClosest_endmember = current_endmembers[:,closest_idx]
            endmembers_withoutX = np.delete(current_endmembers, closest_idx, axis=1)
            gramMeanClosestEndmember[i-1,0:i-1,0:i-1] = calcGramLocal(endmembers_withoutX, meanClosest_endmember)
            gramMeanClosestEndmemberVol[i-1] = np.sqrt(np.abs(np.linalg.det(gramMeanClosestEndmember[i-1,0:i-1,0:i-1])))
        
        ClosestEndmemberEndmembers = pd.DataFrame(endmembers - meanClosest_endmember[:, np.newaxis])
        ClosestEndmemberEndmembers.columns = range(num_endmembers)
        ClosestEndmemberEndmembers.index = wavelengths
        ClosestEndmemberEndmembers.to_csv(output_filename+"_ClosestEndmember.csv", index=True, mode='w')
        
        gramMeanClosestEndmemberdf3 = pd.DataFrame(gramMeanClosestEndmember[3])
        gramMeanClosestEndmemberdf3.columns = np.arange(1,num_endmembers+1)
        gramMeanClosestEndmemberdf3.index = np.arange(1,num_endmembers+1)
        gramMeanClosestEndmemberdf7 = pd.DataFrame(gramMeanClosestEndmember[num_endmembers-1])
        gramMeanClosestEndmemberdf7.columns = np.arange(1,num_endmembers+1)
        gramMeanClosestEndmemberdf7.index = np.arange(1,num_endmembers+1)
        gramMeanClosestEndmemberdf3.to_csv(output_filename+"_ClosestEndmember.csv", index=False, mode='a')
        gramMeanClosestEndmemberdf7.to_csv(output_filename+"_ClosestEndmember.csv", index=False, mode='a')
        gramMeanClosestEndmemberVoldf = pd.DataFrame(gramMeanClosestEndmemberVol)
        gramMeanClosestEndmemberVoldf.index = np.arange(1,num_endmembers+1)
        gramMeanClosestEndmemberVoldf.to_csv(output_filename+"_ClosestEndmember.csv", index=False, mode='a')
        
        if grid_name == 'LANDSAT':
            LANDSAT_RGB_BANDS = [3, 2, 1]
            r = percentile_normalize_array(subset_sr[LANDSAT_RGB_BANDS[0]])
            g = percentile_normalize_array(subset_sr[LANDSAT_RGB_BANDS[1]])
            b = percentile_normalize_array(subset_sr[LANDSAT_RGB_BANDS[2]])
            rgb = np.nan_to_num(np.stack([r, g, b], axis=-1), nan=0.0)
        else: # TANAGER
            subset_vis = extract_subset(ds_ortho[FRAME_IDX, ...], config['x'], config['y'], SUBSET_SIZE)
            rgb = np.transpose(subset_vis[:3, ...], (1, 2, 0)) # Drop alpha
        plt.figure()
        plt.imshow(rgb)
        plt.title(f"{grid_name} - {dt.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        plt.savefig(folder_name + f"{grid_name}_rgb_{i}.png")


    h5.close()



if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    
    print("Please select the HDF5 Image Stack...")
    #file_path = tk.filedialog.askopenfilename(title="Select HDF5 Image Stack",filetypes=[("HDF5 files", "*.h5")])
    
    if file_path:
        process_file(file_path)
    else:
        print("No file selected.")
    
    root.destroy()