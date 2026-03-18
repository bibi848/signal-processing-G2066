'''
This script goes through the process of measuring, collecting and stitching 
the backscattering diffraction data from a measured copper sample. 
Firstly, 5MHz pulses are used on the sample to measure the speed of sound in the 
block experimentally. 
Next, the 3D printed guide is placed on the copper sample, to accomodate the 7.5MHz array. 
Each position is scanned 4 separate times, rotating the 2D array 90 degrees anticlockwise
each time. 
'''
#%%
# Function Import
from pathlib import Path
import sys
import os
root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import napari
import h5py

from Classes.CalcSpeedOfSound import calcSpeedOfSound
from Classes.Stitch3D import normalised_correlation_3D
from Classes.Stitch3D import stitch_volumes
from Classes.Stitch2D import normalised_correlation_2D
from Classes.Stitch2D import stitch_images

#%%
# Self described functions
def read_npy(npy):
    return np.load(IMG_DATA_DIR + '/' + npy + '_3D_TFM.npy')

def read_png(png):
    return mpimg.imread(IMG_DATA_DIR + '/' + png)

#%%
# Extracting Data
processed_data_type = '2D Processed Data'
processed_data_name = 'Cu Pure 7.5MHz Ex 16032026'
imaged_data_name    = '2D TFM Data'

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

speed_of_sound_folders = [x for x in image_folders if 'Speed of Sound' in x]
array_1D_image_folders = [x for x in image_folders if '1D' in x]
array_2D_image_folders = [x for x in image_folders if "Speed of Sound" not in x]
array_2D_image_folders = [x for x in array_2D_image_folders if "1D" not in x]

#%%
# Speed of Sound Calculations
block_depth = 50e-3
t_threshold = 1e-5
threshold_shift = 2e-5
avg_speed = []

for folder in speed_of_sound_folders:
    # Locate Data
    loc = os.path.join(PRO_DATA_DIR, folder)

    time_path = loc + '/time.csv'
    h5_path   = loc + '/time_data.h5'

    time_df = pd.read_csv(time_path)
    time_np = time_df['time_seconds'].to_numpy()

    with h5py.File(h5_path, 'r') as f:
        time_data = np.array(f["time_data"])

    speed_sound = calcSpeedOfSound(time_np, time_data, t_threshold, threshold_shift,
                                   block_depth, displayBool=True, 
                                   elements=[1, 5, 10, 20, 25, 30, 50])

    print(f'Speed of Sound: {speed_sound:.2f} m/s')
    print()
    avg_speed.append(speed_sound)

print(f'Average Speed of Sound: {np.mean(avg_speed):.2f} m/s')
print()

#%%
# 2D Array Element Positions
geometry_path = (os.path.join(PRO_DATA_DIR, array_2D_image_folders[0]) + '/array_geometry.csv')
array_geometry = pd.read_csv(geometry_path)

x = array_geometry['el_xc'].to_numpy()
y = array_geometry['el_yc'].to_numpy()

x = x - np.mean(x)
y = y - np.mean(y)

rotations = [0, np.pi/2, np.pi, 3*np.pi/2]
labels = ['0', '90', '180', '270']

plt.figure(figsize=(6,6))

for theta, label in zip(rotations, labels):

    x_rot = x*np.cos(theta) - y*np.sin(theta)
    y_rot = x*np.sin(theta) + y*np.cos(theta)

    plt.scatter(x_rot*1e3, y_rot*1e3, s=10, label=label)

plt.xlabel('x [mm]')
plt.ylabel('y [mm]')
plt.title('Array Element Positions')
plt.legend()
plt.axis('equal')
plt.grid(True)

plt.show()

#%%
# 1D Imaging
'''
The imaging was performed using the calculated speed of sound above. 
The filter parameters were as follows:
Alpha = 1.0
Percentage band = 4%
Hanning window = False

The Imaging.py then used the following parameters for the imaging:
c = 4705.38 m/s
z_max = 15 mm
z_min = 40 mm
vmax = 0
vmin = -20
x_pixels = 800
z_pixels = 800
This resulted in the images used for stitching, as well as the pixel size.
'''

c = 4705.38 # m/s
x_pixels = 800
z_pixels = 800
x_pixel_size = 0.033e-3 # m
z_pixel_size = 0.031e-3 # m

#%%
# Example Binary and Cropped Images
# Processing Parameters
binary_threshold = 0.6
left_crop = 200
right_crop = int(left_crop + (800 - 2*left_crop))
top_crop = int(800 / 4)
bottom_crop = 0

img1 = read_png(array_1D_image_folders[2] + '_TFM.png')
img2 = read_png(array_1D_image_folders[3] + '_TFM.png')

# Grey scale
if img1.ndim == 3: img1 = img1.mean(axis=2)
else: img1 = img1
if img2.ndim == 3: img2 = img2.mean(axis=2)
else: img2 = img2

crop = (
    slice(top_crop, 800 - bottom_crop),
    slice(left_crop, right_crop)
)

binary1 = (img1 > binary_threshold).astype(float)
binary2 = (img2 > binary_threshold).astype(float)

plt.imshow(binary1, cmap="gray")
plt.axvline(left_crop, linewidth=1.5, c='r')
plt.axvline(right_crop, linewidth=1.5, c='r')
plt.axhline(top_crop, linewidth=1.5, c='r')
plt.axis("off")
plt.show()

plt.imshow(binary2, cmap="gray")
plt.axvline(left_crop, linewidth=1.5, c='r')
plt.axvline(right_crop, linewidth=1.5, c='r')
plt.axhline(top_crop, linewidth=1.5, c='r')
plt.axis("off")
plt.show()

#%%
# Binary-ing and Cropping All Data
reduced_images1 = []

for image_name in array_1D_image_folders:
    img = read_png(image_name + '_TFM.png')

    # Grey Scale
    if img.ndim == 3: img = img.mean(axis=2)
    else: img = img

    # Binary Image
    binary_img  = (img > binary_threshold).astype(float)
    h, w = binary_img.shape

    # Cropped Image
    cropped_img = binary_img[crop]
    reduced_images1.append(cropped_img)

#%%
# Example Stitch
img1 = reduced_images1[8]
img2 = reduced_images1[9]

dx, shifts, corr_values = normalised_correlation_2D(img1, img2)
combined_image, left_offset, w1, x2, w2 = stitch_images(img1, img2, dx)
error = abs(((5e-3 - abs(dx * x_pixel_size))/(5e-3)) * 100)

plt.figure(figsize=(10,6))
plt.imshow(combined_image)
plt.axis("off")

plt.axvline(left_offset, linestyle=":", linewidth=2)
plt.axvline(left_offset + w1, linestyle=":", linewidth=2)
plt.axvline(x2, linestyle=":", linewidth=2)
plt.axvline(x2 + w2, linestyle=":", linewidth=2)

plt.show()

print(f'Pixel Shift: {-1*dx}')
print(f'Distance Calculated: {-1 * dx * x_pixel_size * 1000:.3f} mm')
print(f'Actual Distance: 5 mm')
print(f'Approximate Error: {error:.3f}%')

#%%