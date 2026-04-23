import os
from torchgeo.datasets import CDL, NLCD

# Create a directory to store the ground truth rasters
os.makedirs("C:/satelliteImagery/ground_truth", exist_ok=True)

# 1. Download CDL (Crop-specific land cover for the continental US)
# This will download the most recent available 30m resolution raster
#cdl = CDL(
#    paths="C:/satelliteImagery/ground_truth/cdl", 
#    download=True, 
#    checksum=True
#)

# 2. Download NLCD (17 broad land cover classes like Water, Developed, Forest)
# The annual NLCD products cover the period from 1985 to 2023
# Specify the year that best aligns with your Landsat 8/9 imagery
nlcd = NLCD(
    paths="C:/satelliteImagery/ground_truth/nlcd", 
    download=True, 
    years=[2024] 
)