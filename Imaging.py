'''
This script converts the .xlsx files processed previously into TFM images. 
'''

#%%
# Importing Functions and Defining Correct Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import sys
import time

from Classes.TFM1D import TFM1D

# Point the script to the correct subfolder.
input_data_folder    = '1D Processed Data'
input_data_subfolder = 'Al Pure 15MHz 26012026'
output_data_folder   = '1D TFM Data'
cwd                  = os.getcwd()
display_picture      = True
save_picture         = True
all_pictures         = True
filtered_data        = True
engine               = 'cpp'    # python/cpp/gpu
osys                 = 'ubuntu' # windows/ubuntu, choose windows if on mac
threads              = 512

# Image Parameters
c = 6320 # m/s
depth = 40e-3 # mm
x_pixels = 300
z_pixels = 500
cmap = 'viridis'

# Input and Output paths.
if filtered_data:
    IN_DIR  = os.path.join(cwd, 'DATA', input_data_folder, (input_data_subfolder+' Filtered'))
    OUT_DIR = os.path.join(cwd, 'DATA', output_data_folder, input_data_subfolder+' Filtered')
    os.makedirs(OUT_DIR, exist_ok=True)
else:
    IN_DIR  = os.path.join(cwd, 'DATA', input_data_folder, input_data_subfolder)
    OUT_DIR = os.path.join(cwd, 'DATA', output_data_folder, input_data_subfolder)
    os.makedirs(OUT_DIR, exist_ok=True)


# Find all files in directory which are .xlsx files.
xlsx_files = [
    f for f in os.listdir(IN_DIR)
    if f.lower().endswith(".xlsx")
    and os.path.isfile(os.path.join(IN_DIR, f))
]

print('Files available in directory:')
print(xlsx_files)
print()

# Import module
if engine == 'cpp':
    if osys == 'windows':
        build_dir = os.path.join(
            os.path.dirname(__file__),
            "build", "CPP", "TFM", "Debug"
        )
    elif osys == 'ubuntu':
        build_dir = os.path.join(
            os.path.dirname(__file__),
            "build", "CPP", "TFM"
        )
    sys.path.insert(0, build_dir)
    import tfm_cpp
    print('CPP Available')

elif engine == 'gpu':
    build_dir = os.path.join(
        os.path.dirname(__file__),
        "build", "CPP", "TFM_GPU"
    )
    sys.path.insert(0, build_dir)
    import tfm_gpu
    print('GPU Available')
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
    zc = geometry["el_zc"].values

    x_img = np.linspace(xc.min(), xc.max(), x_pixels)
    z_img = np.linspace(0e-3, depth, z_pixels)

    # TFM computation
    if engine == 'python':
        start_time = time.time()
        img = TFM1D(time_data, time_sec, tx, rx, xc, zc, c, x_img, z_img)
        end_time = time.time()
        print(f"Python execution time: {end_time - start_time:.6f}")

    elif engine == 'cpp':
        start_time = time.time()
        tx0 = tx - 1
        rx0 = rx - 1
        X, Z = np.meshgrid(x_img, z_img)
        img = tfm_cpp.tfm1D(time_data, time_sec, tx0, rx0, xc, zc, X, Z, c)
        end_time = time.time()
        print(f"CPP execution time: {end_time - start_time:.6f}")
        
    elif engine == 'gpu':
        start_time = time.time()
        tx0 = tx - 1
        rx0 = rx - 1
        X, Z = np.meshgrid(x_img, z_img)
        img = tfm_gpu.tfm1D_GPU(time_data, time_sec, tx0, rx0, xc, zc, X, Z, c, threads)
        end_time = time.time()
        print(f"GPU ROCm execution time: {end_time - start_time:.6f}")

    # Display picture
    if display_picture:
        plt.figure(figsize=(6, 8))
        plt.imshow(
            img,
            extent=[x_img[0]*1e3, x_img[-1]*1e3, z_img[-1]*1e3, z_img[0]*1e3],
            aspect="auto",
            cmap=cmap
        )
        plt.xlabel("x [mm]")
        plt.ylabel("z [mm]")
        plt.colorbar(label="Amplitude")
        plt.title(file)
        plt.tight_layout()
        plt.show()
    
    # Save clean file
    if save_picture:
        out_name = os.path.splitext(file)[0] + "_TFM.png"
        plt.imsave(
            os.path.join(OUT_DIR, out_name),
            img,
            cmap=cmap
        )

    if not all_pictures:
        break

    print()

#%%