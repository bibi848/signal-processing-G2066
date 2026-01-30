'''
This script converts the 2D array .xlsx files processed previously into 3D TFM images. 
'''

#%%
# Importing Functions and Defining Correct Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import sys
import time

build_dir = os.path.join(
    os.path.dirname(__file__),
    "build", "CPP", "TFM", "Debug"
)
sys.path.insert(0, build_dir)

import tfm_cpp

# Point the script to the correct subfolder.
input_data_folder    = '2D Processed Data'
input_data_subfolder = 'Al Hole 3MHz 28012026'
output_data_folder   = '2D TFM Data'
cwd                  = os.getcwd()
display_picture      = 'n' # y/n
save_picture         = 'y' # y/n
all_pictures         = 'y' # y/n

# Image Parameters
c = 6320 # m/s

# Input and Output paths.
IN_DIR  = os.path.join(cwd, 'DATA', input_data_folder, input_data_subfolder)
OUT_DIR = os.path.join(cwd, 'DATA', output_data_folder, input_data_subfolder)

# Find all files in directory which are .xlsx files. 
xlsx_files = [
    f for f in os.listdir(IN_DIR)
    if f.lower().endswith(".xlsx")
    and os.path.isfile(os.path.join(IN_DIR, f))
]

print('Files available in directory:')
print(xlsx_files)
print()


#%%
# Looping over available files
for file in xlsx_files:
    print('Processing', file)

    file_path = os.path.join(IN_DIR, file)

    # Extract Data
    metadata  = pd.read_excel(file_path, "Metadata")
    time_data = pd.read_excel(file_path, "Time_Data").values
    time_sec  = pd.read_excel(file_path, "Time")["time_seconds"].values
    tx_rx     = pd.read_excel(file_path, "tx_rx")
    geometry  = pd.read_excel(file_path, "Array_Geometry")

    tx = tx_rx["tx"].values.astype(int)
    rx = tx_rx["rx"].values.astype(int)

    xc = geometry["el_xc"].values
    yc = geometry["el_yc"].values
    zc = geometry["el_zc"].values

    # Image grid
    x_img = np.linspace(xc.min(), xc.max(), 200)
    y_img = np.linspace(yc.min(), yc.max(), 200)
    z_img = np.linspace(0e-3, 40e-3, 300)
    tx0 = tx - 1
    rx0 = rx - 1

    Z, Y, X = np.meshgrid(z_img, y_img, x_img, indexing="ij")

    print('Starting TFM')
    start_time = time.time()
    img = tfm_cpp.tfm2D(time_data, time_sec, tx0, rx0, xc, yc, zc, X, Y, Z, c)
    end_time = time.time()
    print("TFM time:", end_time - start_time)
    print()

    if save_picture == 'y':
        out_name = os.path.splitext(file)[0] + "_TFM_3D.npy"
        np.save(
            os.path.join(OUT_DIR, out_name),
            img
        )

    if all_pictures == 'n':
        break