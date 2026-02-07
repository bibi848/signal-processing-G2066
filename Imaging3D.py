'''
This script converts the 2D array files processed previously into 3D TFM images. 
'''

#%%
# Importing Functions and Defining Correct Path
import pandas as pd
import numpy as np
import os
import sys
import time
import h5py

# Point the script to the correct subfolder.
input_data_folder    = '2D Processed Data'
input_data_subfolder = 'FeC_S Wavy 3MHz 04022026'
output_data_folder   = '2D TFM Data'
cwd                  = os.getcwd()
save_picture         = True
all_pictures         = True
filtered_data        = True
engine               = 'gpu' # cpp / gpu
threads              = 512

# Parameters
c        = 6320 # m/s
depth    = 80e-3 # mm
x_pixels = 200
y_pixels = 200
z_pixels = 400

# Input and Output paths.
if filtered_data:
    IN_DIR  = os.path.join(cwd, 'DATA', input_data_folder, (input_data_subfolder+' Filtered'))
    OUT_DIR = os.path.join(cwd, 'DATA', output_data_folder, (input_data_subfolder+' Filtered'))
    os.makedirs(OUT_DIR, exist_ok=True)
else:
    IN_DIR  = os.path.join(cwd, 'DATA', input_data_folder, input_data_subfolder)
    OUT_DIR = os.path.join(cwd, 'DATA', output_data_folder, input_data_subfolder)
    os.makedirs(OUT_DIR, exist_ok=True)

# List all available image folders
image_folders = [
    f for f in os.listdir(IN_DIR)
    if os.path.isdir(os.path.join(IN_DIR, f))
]

print('Files available in directory:')
print(image_folders)
print()

# Import the appropriate module
if engine == 'cpp':
    import platform
    if platform.system() == 'Windows':
        build_dir = os.path.join(os.path.dirname(__file__), "build", "CPP", "TFM", "Debug")
    else:
        build_dir = os.path.join(os.path.dirname(__file__), "build", "CPP", "TFM")
    sys.path.insert(0, build_dir)
    import tfm_cpp
    print('CPP Setup Successful')
    print()

elif engine == 'gpu':
    build_dir = os.path.join(os.path.dirname(__file__), "build", "CPP", "TFM_GPU")
    sys.path.insert(0, build_dir)
    import tfm_gpu
    print('GPU Setup Successful')
    print()


#%%
# Looping over available files
full_start = time.time()
for fol in image_folders:
    print('Processing', fol)

    file_path = os.path.join(IN_DIR, fol)

    # Extract Data
    metadata = pd.read_csv(os.path.join(file_path, "metadata.csv"))
    time_sec = pd.read_csv(os.path.join(file_path, "time.csv"))["time_seconds"].values
    tx_rx    = pd.read_csv(os.path.join(file_path, "tx_rx.csv"))
    geometry = pd.read_csv(os.path.join(file_path, "array_geometry.csv"))

    with h5py.File(os.path.join(file_path, "time_data.h5"), "r") as h5f:
        time_data = h5f["time_data"][:]

    tx = tx_rx["tx"].values.astype(int)
    rx = tx_rx["rx"].values.astype(int)

    xc = geometry["el_xc"].values
    yc = geometry["el_yc"].values
    zc = geometry["el_zc"].values

    # Image grid
    x_img = np.linspace(xc.min(), xc.max(), x_pixels)
    y_img = np.linspace(yc.min(), yc.max(), y_pixels)
    z_img = np.linspace(0e-3, depth, z_pixels)
    tx0 = tx - 1
    rx0 = rx - 1

    Z, Y, X = np.meshgrid(z_img, y_img, x_img, indexing="ij")

    # TFM computation
    print('Starting TFM')
    if engine == 'cpp':
        start_time = time.time()
        img = tfm_cpp.tfm2D(time_data, time_sec, tx0, rx0, xc, yc, zc, X, Y, Z, c)
        end_time = time.time()
        print(f"CPP TFM time: {end_time - start_time:.3f}s")

    elif engine == 'gpu':
        start_time = time.time()
        img = tfm_gpu.tfm2D_GPU(time_data, time_sec, tx0, rx0, xc, yc, zc, X, Y, Z, c, threads)
        end_time = time.time()
        print(f"GPU TFM time: {end_time - start_time:.3f}s")

    if save_picture:
        out_name = fol + "_3D_TFM.npy"
        np.save(
            os.path.join(OUT_DIR, out_name),
            img
        )

    if not all_pictures:
        break

    print()

full_end = time.time()
print(f'Time to process {len(image_folders)} images: {full_end - full_start:.6f}s')