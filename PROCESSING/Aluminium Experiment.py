#%%
# Overview
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
root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

import numpy as np
import matplotlib.pyplot as plt
import h5py

#%%
# Calculating Speed of Sound


#%%
# 
