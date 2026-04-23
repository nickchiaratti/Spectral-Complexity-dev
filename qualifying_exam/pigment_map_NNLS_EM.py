'''
+
=======================================================================

 NAME:
      pigment_map_NNLS_EM

 DESCRIPTION:
Map pigments in a HSI using endmembers and spectral unmixing with NNLS; endmembers
are extracted using the MaxD and Gram matrix approach

 USES:
spectral
spectral.io.envi
numpy
matplotlib
PySpy-tools
openENVI
spectral_tools - my spectral tools in a separate file
MaxD_Gram - separate file

 PARAMETERS:

 KEYWORDS:
needs to know where the image is located.

 RETURNS:
saves as a pdf the class map with colors associated to the extracted EM's; EM's
are also saved as a pdf with matching colors to the class map

 NOTES:

 HISTORY:
09/08/2023: D. Messinger - created, based on pigment_map_NNLS.py



=======================================================================
-
'''

from spectral import *
import spectral.io.envi as envi
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from MaxD_Gram import *

#PySp Tools - abundance mapping by linear unmixing
import pysptools.eea as eea
import pysptools.abundance_maps as amaps

import time

import tkinter as tk
from tkinter import filedialog
import os

def select_file_to_process():
    """
    Opens a pop-up menu (file dialog) for the user to select a file.
    
    Returns:
        str: The full path to the selected file, or an empty string if no file is selected.
    """
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    # Open the file dialog and prompt the user to select a file.
    file_path = tk.filedialog.askopenfilename(
        title="Select a file to process",
        filetypes=(
            ("TIFF files", "*.tif;*.tiff"),
            ("ENVI files", "*.hdr;*.img"),
            ("All files", "*.*")
        )
    )
    
    return file_path

#define function to open up an ENVI file and return the image
#from openENVI import *
# my spectral tools
#from spectral_tools import *

###
# starting stuff
###
ifigure = 1
start_time = time.time()
start_hour = time.gmtime().tm_hour
start_min = time.gmtime().tm_min
start_sec = time.gmtime().tm_sec
print('Starting time [GMT]: ', start_hour,':', start_min,':',start_sec)

###
# input image file location and name
###
#infolder = '/Users/dwmpaci/Desktop/3_PROJECTS/0_Durham/WS_First_Folio/'
#infolder = '/Users/dwmpaci/Desktop/3_PROJECTS/0_Durham/HSI_Data/2023_03_15_09_18_14/'
#infolder = '/Users/dwmpaci/Desktop/3_PROJECTS/0_Durham/HSI_Data/Italian_leaf_complete/'
#infolder = '/Users/dwmpaci/Desktop/3_PROJECTS/python_dev/0_test_images/'
#infolder = "C:/Users/nnn_s/Downloads/1e1ce2d7-f450-427d-b2bf-a3f665f15c2c/1e1ce2d7-f450-427d-b2bf-a3f665f15c2c/wyvern_dragonette-002_20250526T200831_1e1ce2d7/"
#infolder = "C:/Users/nnn_s/Downloads/Tait Preserve/LC08_L1TP_016030_20250718_20250726_02_T1/"
#image_file = infolder + 'GM_HSI_3R_test_chip_4_1600x2000'
#image_file = infolder + 'GM_HSI_3R_test_chip_111bands'
#image_file = infolder + 'wyvern_dragonette-002_20250526T200831_1e1ce2d7.tiff
#image_file = infolder + 'LC08_L1TP_016030_20250718_20250726_02_T1_B1.TIF'
image_file = select_file_to_process()
infolder = os.path.dirname(image_file)
#
#infolder = '/Users/dwmpaci/Desktop/3_PROJECTS/2_GoughMap/2022_06_GM_data/070622/Position-2/6_ENVI/'
#infolder = '/Volumes/LaCie/GoughMapData/Gough_1R/'
#infolder = '/Users/dwmpaci/Desktop/3_PROJECTS/0_Durham/HSI_data/Symeon/Symeon-VNIR-2022_09_07_05_43_38/'
#image_file = infolder + 'Gough_1R'
#image_file = infolder + 'data'

###
# if desired (saveimages = 1), output image file location and name
###
saveimages = 0
# outfolder = '/Users/dwmpaci/Desktop/3_PROJECTS/2_GoughMap/2022_06_GM_data/070622/Position-2/6_ENVI/'
#outfolder = '/Users/dwmpaci/Desktop/3_PROJECTS/python_dev/Pigment_Mapping_SAM_EM/'
outfolder = infolder
outfile = outfolder+'GM_HSI_3R_test_chip_NNLS_EM_6_abundancemaps.pdf'
specfolder = outfolder
specfile = outfolder+'junk.pdf'

###
# enter number of endmembers to start off with (this should be more than the expected endmembers)
# and the number to keep in the end
###
num_EM = 10
num_to_keep = 8

###
# open the image file
###
#image_file = infolder +'refl_img'
#image_file = infolder +'GM_HSI_3R_test_chip_4_1600x2000'
print ('opening image file: ', image_file)
#image = open_image(image_file+'.hdr').load()
image = open_image(image_file).load()
nrows = image.nrows
ncols = image.ncols
nbands = image.nbands
print('IMAGE rows, cols, bands: ', image.nrows, image.ncols, image.nbands)
print('')

#
# extract the red green and blue bands to make a picture
#
bands = np.array(image.nbands)
bands = image.bands.centers
iblue = get_band_index(bands,475.0)
igreen = get_band_index(bands,535.0)
ired = get_band_index(bands,650.0)
# for SWIR
# iblue = get_band_index(bands,1034.0)
# igreen = get_band_index(bands,1195.0)
# ired = get_band_index(bands,1600.0)

if str(image.bands.centers) == 'None' :
    print('creating band centers array')
    bands = np.arange(nbands) + 1

###
# for MSI, max normalize the image
###
#image[:,:,:] = image[:,:,:]/np.amax(image[:,:,:])

# ####
# #For MSI, blue green and red are 2, 4, 6
# ####
# iblue = 2
# igreen = 4
# ired = 6

###
# make the RGB image and show it
###
rgbimg = make_color_image(image, ired, igreen, iblue)

plt.figure(ifigure)
picture = plt.imshow((rgbimg/np.amax(rgbimg)))
plt.title('RGB')
#plt.show()
ifigure += 1

###
# extract the endmembers with MaxD / Gram matrix
###
# reshape the image to pass to MaxD
image2D = np.reshape(image, [nrows * ncols, nbands], order="F")

# choose between using general gram matrix or local gram matrix
gram = 'general'
# gram = 'local'

# if you have mnf data of the image, set mnf_data to that, else code will use image data as mnf_data
mnf_data = 0

# number of EM's to extract
print(f'Extracting {num_EM} EMs: ')

endmembers, endmembers_index, volume = maximumDistance(image2D, num_EM, mnf_data, gram)

# normalize volume
volume_norm = volume / sum(volume)

# plot volume function
# x = np.arange(3, num_EM + 1)
# plt.figure(ifigure)
# plt.plot(x, volume_norm[2:])
# plt.xlabel('Number of endmembers')
# plt.ylabel('Normalized estimated volume')
# plt.title('Grammian Volume Function')
# #plt.show()
# ifigure +=1

#choose number of endmembers to keep based on graph
# value = input("How many EM's to keep? [integer]:\n")
# value = int(value)
# print(f'Keeping {value} endmembers....')
#num_to_keep = value

print(f'---> Keeping {num_to_keep} endmembers ....')
endmembers[:,0:num_to_keep-1]
endmembers_index[:,0:num_to_keep-1]

###
# build the class_spec array from the EMs
###
# class_spec = np.zeros((library.spectra.shape[0],library.spectra.shape[1]), float)
# for i,c in enumerate(library.spectra) :
#     class_spec[i] = library.spectra[i]
lib_nspec = num_to_keep
class_spec = np.zeros((num_to_keep,nbands), float)
for i in range(num_to_keep):
    class_spec[i,:] = endmembers[:,i]

###
# plot the spectra used for classification
#
# first, set up the color map for plotting and making the class map so they match up
#
#-> import the color map
mycmap = plt.get_cmap('jet', lib_nspec)
#-> generate color array
newcolors = mycmap(np.arange(0,mycmap.N))
#-> cast it as a ListedColormap object with attributes
newcmp = ListedColormap(newcolors)

###
# plot the class spectra
###
plt.figure(ifigure)
#plt.plot(library.bands.centers,class_spec[5,:],color='black',label = library.names[5])
for j in range(lib_nspec) :
    EM_names = 'EM: '+str(j+1)
    plt.plot(bands,class_spec[j,:],color=newcmp.colors[j], label = EM_names)

plt.xlabel('Wavelength (nm)')
plt.ylabel('Reflectance')
plt.title('Endmember Spectra')
plt.legend(loc='best')
# if saveimages == 1:
#     plt.savefig(specfile,dpi = 1200)
#plt.show()
ifigure += 1

###
# run spectral unmixing from PySp tools
###

img2unmix = np.ndarray([nrows, ncols, nbands])
img2unmix = np.copy(image[:,:,:])

mask = None
print('---> performing NNLS unmixing...', end = '')
nnls = amaps.NNLS()
result = nnls.map(np.array(img2unmix), class_spec, normalize=False,mask=mask)
print('... done <---')
print('Shape of result: ', result.shape)

###
# compute total abundance per pixel as a check on the unmixing model
###

total_abund = np.sum(result,axis=2)
print('shape of total_abund: ',total_abund.shape)
print('min / max / mean of total_abund: ', np.amin(total_abund), np.amax(total_abund), np.mean(total_abund))
plt.figure(ifigure)
picture = plt.imshow(total_abund, cmap='gray')
plt.title('Total Abundance')
#plt.show()
ifigure += 1

# plot a histogram
n_bins = 50
plt.figure(ifigure)
plt.hist(total_abund.flatten(),bins =n_bins)
plt.title('total_abund Histogram')
ifigure +=1

###
# plot the abundance maps with subfigures
###
figcols = 4
figrows = 2 #int(np.round(num_to_keep/figcols))
# print('figcols: ', figcols)
# print('figrows: ', figrows)
fig,axes = plt.subplots(nrows = figrows, ncols = figcols, figsize = (figcols*3, figrows*3))

for num in range (1,num_to_keep+1):
    plt.subplot(figrows,figcols,num)
    idx = num - 1
    # print('num: ',num)
    # print('idx: ',idx)
    plt.imshow(result[:,:,idx], cmap='gray',aspect = 'auto')
    plt.title('EM '+str(num))

fig.tight_layout() # used to adjust padding between subplots
if saveimages == 1:
    plt.savefig(outfile,dpi = 1200)

#ifigure +=1

print("--- %s seconds ---" % (time.time() - start_time))
plt.show()
