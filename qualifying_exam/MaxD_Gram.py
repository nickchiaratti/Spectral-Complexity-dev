'''
+
=======================================================================

 NAME:
      MaxD_Gram

 DESCRIPTION:
Extracts end members from a spectral image using the MaxD approach.  Also estimates the
Material diversity in a scene using the Gram matrix approach, both global and local.

TO CALL:
   endmembers, endmembers_index, volume = maximumDistance(data, num, mnf_data, gram)

 USES:
numpy

 PARAMETERS:
    data = 2D data [npixels, nbands]
    num = number of endmembers to be calculated (choose more than expected to find)
    mnf_data = [MNF data cube or 0]; if MNF data is not available (i.e., mnf_data = 0),
        code will assign img as mnf_data

 KEYWORDS:


 RETURNS:
maximumDistance -  endmembers, endmembers_index, volume
calcGram; calcGramLocal - gram (Grammian, the volume of the parallelotope enclosed by the end members)

 NOTES:
- original python code developed by Tania Kleynhans based on papers by Amanda Ziemann and Dave Messinger

Publications to site:
- 

 HISTORY:
08/30/2023 - created
11/19/2025 - updated with argmax and argmin due to minimum distance endmembers returning multiple indices
02/15/2026 - updated mean vector calculation to use the mean of the data instead of the mean of the endmembers

=======================================================================
-
'''

import numpy as np
from numpy import matlib as mb


def maximumDistance(data, num, mnf_data, gram):
    # data = 2D data [npixels, nbands]
    # num = number of endmembers to be calculated (choose more than expected to find)
    # if MNF data is not available, code will assign img as mnf_data
    #print('---> In MaxD extracting endmembers and Grammian ...')
    if mnf_data == 0:
        mnf_data = data

    data = np.transpose(data)
    data2 = np.transpose(mnf_data)

    # Calculate the mean vector of the entire dataset axis=1 averages across pixels, resulting in a vector of size [bands]
    mean_vector = np.mean(data, axis=1) 

    # find data size
    num_bands = data.shape[0]
    num_pix = data.shape[1]

    # calculate magnitude of all vectors to find min and max
    magnitude = np.sum(np.square(data), axis=0)
    idx1 = np.argmax(magnitude)
    idx2 = np.argmin(magnitude)

    # create empty output arrays for endmembers
    endmembers = np.zeros([num_bands, num])
    endmembers_index = np.zeros([1, num])

    # assign largest and smallest vector as first and second endmembers
    endmembers[:, 0] = np.transpose(data[:, idx1])
    endmembers[:, 1] = np.transpose(data[:, idx2])
    endmembers_index[0, 0] = idx1
    endmembers_index[0, 1] = idx2

    data_proj = np.matrix(data2)
    identity_matrix = np.identity(num_bands)

    # create array for volume of determinant of Gram matrix
    volume = np.zeros([num])

    loop = np.arange(3, num + 1)
    for i in loop:
        diff = []
        pseudo = []
        # calc difference between endmembers
        diff = np.matrix(data_proj[:, idx2] - data_proj[:, idx1])
        # caclualte pseudo inverse of difference vector
        pseudo = np.linalg.pinv(diff)
        data_proj = np.matmul((identity_matrix - np.matmul(diff, pseudo)), data_proj)

        idx1 = idx2
        diff_new = np.sum(np.square((np.matmul(data_proj[:, idx2], np.ones([1, num_pix])) - data_proj)), axis=0)

        # find ne maximum distance for next endmember
        idx2 = np.int_(np.where(diff_new == np.max(diff_new))[1])

        #print('DEBUG: idx2: ', idx2,np.size(idx2))
        ###
        # DWM: looks like there may be cases where idx2 has more than one element, i.e., there are
        # two elements of diff_new that are equal to the max.  In that case, just grab the first one
        ###
        if np.size(idx2) > 1:
            idx2 = idx2[0]

        # assign to endmember file
        endmembers[:, i - 1] = np.transpose(data[:, idx2])
        endmembers_index[0, i - 1] = idx2

        if gram == 'local':
            # calculate local gram matrix
            loc_gram = calcGramLocal(endmembers, i, mean_vector)
            volume[i - 1] = np.sqrt(np.abs(np.linalg.det(loc_gram)))

        elif gram == 'general':
            # calculate general gram matrix
            gen_gram = calcGramGeneral(endmembers[:, 0:i])
            volume[i - 1] = np.sqrt(np.abs(np.linalg.det(gen_gram)))

    return endmembers, endmembers_index, volume


def calcGramGeneral(data_endmembers):
    # calculate gram matrix = V^T * V
    gram = np.matmul(np.transpose(data_endmembers), data_endmembers)

    return gram


def calcGramLocal(data_endmembers, num_endmembers, mean_vector):
    """
    Calculates the Local Gram matrix.
    
    1. Finds the endmember 'x' closest to the dataset mean.
    2. Subtracts 'x' from all other endmembers (centering the simplex on x).
    3. Calculates the Gram matrix of these centered vectors.
    """
    # Edge Case: If only 1 endmember, volume is 0.
    if num_endmembers < 2:
        return np.zeros((1, 1))

    # Reduce to current number of endmembers
    # Shape: (Bands, N)
    current_endmembers = data_endmembers[:, 0:num_endmembers]

    # 1. Find endmember closest to the dataset mean
    # mean_vector shape: (Bands,)
    
    # Determine difference between endmembers and dataset mean: (Bands, N) - (Bands, 1)
    diff_from_mean = current_endmembers - mean_vector[:, np.newaxis]
    dist_from_mean = np.linalg.norm(diff_from_mean, axis=0)
    
    # Index of the endmember closest to the dataset mean
    center_idx = np.argmin(dist_from_mean)
    
    # The endmember closest to the dataset mean
    vector_x = current_endmembers[:, center_idx] # Shape: (Bands,)

    # 2. Extract remaining vectors (exclude x)
    # Create a boolean mask of indices to keep
    mask = np.arange(num_endmembers) != center_idx
    remaining_vectors = current_endmembers[:, mask] # Shape: (Bands, N-1)

    # 3. Localize: Subtract x from remaining vectors
    # (Bands, N-1) - (Bands, 1)
    centered_vectors = remaining_vectors - vector_x[:, np.newaxis]

    # 4. Calculate Gram Matrix
    # G = V^T * V
    gram = np.matmul(centered_vectors.T, centered_vectors)

    return gram

