'''
This script goes through the process of measuring, collecting and stitching 
the backscattering diffraction data from a measured aluminium sample. 
Firstly, 5MHz pulses are used on the sample to measure the speed of sound in the 
block experimentally. 
Next, the 3D printed guide is placed on the aluminium sample, to accomodate the 10MHz array. 
Images are taken at measured intervals, ensuring that each image taken is averaged 64 times. 
The images are processed and filtered, followed by a dimensionality reduction stitching step. 
The calculated pixel shifts are compared to the experimental shifts. 
'''
#%%
# Function Import
from pathlib import Path
import sys
import os
root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

import numpy as np
import matplotlib.pyplot as plt
import h5py

#%%
# Extracting Data
processed_data_type = '1D Processed Data'
processed_data_name = 'Al Pure 10MHz Ex 09032026'
imaged_data_name    = '1D TFM Data'

cwd      = Path.cwd().parent
filtered = True

# Input and Output paths.
PRO_DATA_DIR  = os.path.join(cwd, 'DATA', processed_data_type, (processed_data_name + ' Filtered'))
IMG_DATA_DIR = os.path.join(cwd, 'DATA', imaged_data_name, (processed_data_name + ' Filtered'))
os.makedirs(IMG_DATA_DIR, exist_ok=True)

# Image Folders Available
image_folders = [
    f for f in os.listdir(PRO_DATA_DIR)
    if os.path.isdir(os.path.join(PRO_DATA_DIR, f))
]
image_folders = np.sort(image_folders)
print('Folders available in directory:')
print(image_folders)
print()

speed_sound_files = ['Speed of Sound 1_filtered', 'Speed of Sound 2_filtered', 'Speed of Sound 3_filtered']
image_files1 = ['A1_filtered' 'A2_filtered' 'A3_filtered' 'A4_filtered' 'A5_filtered']
image_files2 = ['B1_filtered' 'B2_filtered' 'B3_filtered' 'B4_filtered' 'B5_filtered']




#%%
# 
