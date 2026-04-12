'''
This script converts the files processed previously into TFM images. 
'''

#%%
# Importing Functions and Defining Correct Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import platform
import os
import sys
import time
import h5py
from scipy.signal import hilbert

from Classes.TFM1D import TFM1D
from Classes.TFM1D import CTFM1D
from Classes.TFM1D import TFM_angular1D

# Point the script to the correct subfolder.
input_data_folder    = '1D Processed Data'
input_data_subfolder = 'Al Pure 10MHz 17022026'
output_data_folder   = '1D TFM Data'
cwd                  = os.getcwd()

display_picture = True
save_picture    = False
all_pictures    = False
filtered_data   = False
img_output      = 'db' # real, complex, envelope, db

engine  = 'gpu' # python/cpp/gpu
threads = 512

# Threshold Parameters
vmax = 0.0
vmin = -20.0

# Image Parameters
c        = 6300 # m/s
z_max    = 10e-3   # m
z_min    = 40e-3   # m
x_min    = 'xc_min' # m, can specify length
x_max    = 'xc_max' # or just use xc_min/xc_max
x_pixels = 400
z_pixels = 400
cmap     = 'viridis'

# Angular Filter
half_angle_deg = 30
min_els        = 40

# Aspect Ratio
real_aspect_ratio = False
z_aspect = 8
x_aspect = 8

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
image_folders = np.sort(image_folders)
image_folders = [x for x in image_folders if 'Speed of Sound' not in x]
if "2D" in input_data_folder:
    image_folders = [x for x in image_folders if '1D' in x]    

print('Files available in directory:')
print(image_folders)
print()

# Import module
if engine == 'cpp':
    if platform.system() == 'Windows':
        build_dir = os.path.join(
            os.path.dirname(__file__),
            "build", "CPP", "TFM", "Debug"
        )
    else:
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

if filtered_data:
    CTFM, db_bool = True, True
else:
    CTFM, db_bool = False, False

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
    zc = geometry["el_zc"].values

    if x_max == 'xc_max':
        x_max = xc.max()
    if x_min == 'xc_min':
        x_min = xc.min()

    x_img = np.linspace(x_min, x_max, x_pixels)    
    z_img = np.linspace(z_max, z_min, z_pixels)
    if real_aspect_ratio:
        x_aspect = int(np.ceil(((x_max - x_min) / (z_min - z_max)) * z_aspect))

    # TFM computation
    if engine == 'python':
        start_time = time.time()

        if CTFM:
            img = CTFM1D(time_data, time_sec, tx, rx, xc, zc, c, x_img, z_img, output=img_output)
        else:
            img = TFM1D(time_data, time_sec, tx, rx, xc, zc, c, x_img, z_img)

        end_time = time.time()
        print(f"Python execution time: {end_time - start_time:.6f}")

    elif engine == 'cpp':
        start_time = time.time()
        tx0 = tx - 1
        rx0 = rx - 1
        X, Z = np.meshgrid(x_img, z_img)
        img = tfm_cpp.tfm1D(time_data, time_sec, tx0, rx0, xc, zc, X, Z, c)

        if img_output == 'real':
            img = img
        
        elif img_output == 'complex':
            img_analytic = hilbert(img, axis=0)
            img = img_analytic
        
        elif img_output == 'envelope':
            img_envelope = np.abs(hilbert(img, axis=0))
            img = img_envelope
        
        elif img_output == 'db':
            img_envelope = np.abs(hilbert(img, axis=0))
            img = 20 * np.log10(img_envelope / (img_envelope.max() + 1e-10) + 1e-10)

        end_time = time.time()
        print(f"CPP execution time: {end_time - start_time:.6f}")
        
    elif engine == 'gpu':
        start_time = time.time()
        tx0 = tx - 1
        rx0 = rx - 1
        X, Z = np.meshgrid(x_img, z_img)
        img = tfm_gpu.tfm1D_GPU(time_data, time_sec, tx0, rx0, xc, zc, X, Z, c, threads)

        if img_output == 'real':
            img = img
        
        elif img_output == 'complex':
            img_analytic = hilbert(img, axis=0)
            img = img_analytic
        
        elif img_output == 'envelope':
            img_envelope = np.abs(hilbert(img, axis=0))
            img = img_envelope
        
        elif img_output == 'db':
            img_envelope = np.abs(hilbert(img, axis=0))
            img = 20 * np.log10(img_envelope / (img_envelope.max() + 1e-10) + 1e-10)

        end_time = time.time()
        print(f"GPU ROCm execution time: {end_time - start_time:.6f}")

    # Display picture
    if display_picture:
        if img_output == 'complex':
            img_display = np.abs(img)  # show envelope for visualisation only
        else:
            img_display = img

        plt.figure(figsize=(x_aspect, z_aspect))
        if img_output == 'db':
            plt.imshow(
                img_display,
                vmax=vmax,
                vmin=vmin,
                extent=[x_img[0]*1e3, x_img[-1]*1e3, z_img[-1]*1e3, z_img[0]*1e3],
                aspect="auto",
                cmap=cmap
            )
        else:
            plt.imshow(
                img_display,
                extent=[x_img[0]*1e3, x_img[-1]*1e3, z_img[-1]*1e3, z_img[0]*1e3],
                aspect="auto",
                cmap=cmap
            )
        plt.xlabel("x [mm]")
        plt.ylabel("z [mm]")
        plt.colorbar(label="Amplitude")
        plt.title(fol)
        plt.tight_layout()
        plt.show()

    if save_picture:
        if img_output == 'complex':
            np.save(os.path.join(OUT_DIR, fol + "_TFM.npy"), img)
        else:
            img_save = img
            plt.imsave(
                os.path.join(OUT_DIR, fol + "_TFM.png"),
                img_save,
                cmap=cmap,
                vmin=vmin if img_output == 'db' else None,
                vmax=vmax if img_output == 'db' else None
            )

    if not all_pictures:
        break

    print()

full_end = time.time()
print(f'Time to process {len(image_folders)} images: {full_end - full_start:.6f}s')
#%%
# Pixel size
dx_mm = (x_img[-1] - x_img[0]) * 1e3 / (x_pixels - 1)
dz_mm = (z_img[-1] - z_img[0]) * 1e3 / (z_pixels - 1)
centre_frequency = float(metadata.loc[metadata['Field'] == 'centre_frequency_Hz', 'Value'].iloc[0])
wavelength = c / centre_frequency

print(f"Lateral pixel size: {dx_mm:.3f} mm")
print(f"Depth pixel size:   {dz_mm:.3f} mm")
print(f"Centre Frequency:   {centre_frequency/1e6} MHz")
print(f"Wavelength:         {wavelength * 1e3:.3f} mm")