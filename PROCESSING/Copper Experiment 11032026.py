'''
This script goes through the process of measuring, collecting and stitching 
the backscattering diffraction data from a measured copper sample. 
Firstly, 5MHz pulses are used on the sample to measure the speed of sound in the 
block experimentally. 
Next, the 3D printed guide is placed on the copper sample, to accomodate the 7.5MHz array. 
Images are taken at 5mm intervals, ensuring that each image taken is averaged 64 times. 
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

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import napari
import h5py

from Classes.CalcSpeedOfSound import calcSpeedOfSound
from Classes.Stitch3D import normalised_correlation_3D
from Classes.Stitch3D import stitch_volumes

#%%
# Self described functions
def read_npy(npy):
    return np.load(IMG_DATA_DIR + '/' + npy + '_3D_TFM.npy')

#%%
# Extracting Data
processed_data_type = '2D Processed Data'
processed_data_name = 'Cu Pure 7.5MHz Ex 11032026'
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
data_folders = [x for x in image_folders if "Calibration" not in x]
data_folders = [x for x in data_folders if "Speed of Sound" not in x]
groups = [data_folders[i:i+5] for i in range(0, len(data_folders), 5)]

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
geometry_path = (os.path.join(PRO_DATA_DIR, data_folders[0]) + '/array_geometry.csv')
array_geometry = pd.read_csv(geometry_path)

plt.figure(figsize=(6, 6))
plt.scatter(array_geometry['el_xc'], array_geometry['el_yc'])
plt.title('2D Array Element Positions')
plt.show()

#%%
# Imaging
'''
The imaging was performed using the calculated speed of sound above. 
The filter parameters were as follows:
Alpha = 0.9
Percentage band = 45%
Hanning window = False

The Imaging.py then used the following parameters for the imaging:
c = 4703.28 m/s
z_max = 10 mm
z_min = 40 mm
vmax = 0
vmin = -20
x_pixels = 200
y_pixels = 200
z_pixels = 400
This resulted in the images used for stitching, as well as the pixel size.
'''

c = 4703.28 # m/s
x_pixels = 200
y_pixels = 200
z_pixels = 400
x_pixel_size = 0.059e-3 # m
y_pixel_size = 0.059e-3 # m
z_pixel_size = 0.075e-3 # m

#%%
vol1 = read_npy(data_folders[10])
vol2 = read_npy(data_folders[11])

x_index = 100

plt.figure(figsize=(6,4))

plt.subplot(1,2,1)
plt.imshow(vol1[:,x_index,:], cmap='gray')

plt.subplot(1,2,2)
plt.imshow(vol2[:,x_index,:], cmap='gray')

plt.show()

#%%
print(data_folders[0])
print(data_folders[5])
print(data_folders[10])
print(data_folders[15])

vol1 = read_npy(data_folders[0])
vol2 = read_npy(data_folders[5])
vol3 = read_npy(data_folders[10])
vol4 = read_npy(data_folders[15])

y_index = 100

plt.figure(figsize=(4,4))

plt.subplot(2,2,1)
plt.imshow(vol1[:,:,y_index], cmap='gray')
plt.axis('off')

plt.subplot(2,2,2)
plt.imshow(vol2[:,:,y_index], cmap='gray')
plt.axis('off')

plt.subplot(2,2,3)
plt.imshow(vol3[:,:,y_index], cmap='gray')
plt.axis('off')

plt.subplot(2,2,4)
plt.imshow(vol4[:,:,y_index], cmap='gray')

plt.axis('off')
plt.show()

#%%
vol1 = read_npy(data_folders[0])

# Cropping
z_crop_top    = 200
z_crop_bottom = 150
x_crop_left   = 0
x_crop_right  = 0
y_crop_left   = 0
y_crop_right  = 0

z, x, y = vol1.shape

cropped_volume = vol1[
    z_crop_top : z - z_crop_bottom,
    x_crop_left : x - x_crop_right,
    y_crop_left : y - y_crop_right
]


viewer = napari.Viewer()

viewer.add_image(
    cropped_volume,
    name="vol1",
    colormap="gray"
)
napari.run()

#%%
vol2 = read_npy(data_folders[1])

z, x, y = vol2.shape

cropped_volume = vol2[
    z_crop_top : z - z_crop_bottom,
    x_crop_left : x - x_crop_right,
    y_crop_left : y - y_crop_right
]

viewer = napari.Viewer()

viewer.add_image(
    cropped_volume,
    name="vol2",
    colormap="gray"
)
napari.run()

#%%
vol1 = read_npy(data_folders[0])
vol2 = read_npy(data_folders[1])
max_shift = 100

# Cropping
z_crop_top    = 80
z_crop_bottom = 100
x_crop_left   = 10
x_crop_right  = 10
y_crop_left   = 10
y_crop_right  = 10

z, x, y = vol1.shape

vol1 = vol1[
    z_crop_top : z - z_crop_bottom,
    x_crop_left : x - x_crop_right,
    y_crop_left : y - y_crop_right
]

z, x, y = vol2.shape

vol2 = vol2[
    z_crop_top : z - z_crop_bottom,
    x_crop_left : x - x_crop_right,
    y_crop_left : y - y_crop_right
]

best_shiftx, shiftsx, corr_valuesx = normalised_correlation_3D(vol1, vol2, axis='x', max_shift=max_shift)
best_shifty, shiftsy, corr_valuesy = normalised_correlation_3D(vol1, vol2, axis='y', max_shift=max_shift)

actual_shift = round(5e-3 / x_pixel_size)

print(f'Actual shift: {actual_shift}')
print(f'Best y shift: {best_shifty}, max correlation: {max(corr_valuesy):.3f}')
print(f'Best x shift: {best_shiftx}, max correlation: {max(corr_valuesx):.3f}')

#%%
for i in range(5, 9):
    vol1 = read_npy(data_folders[i])
    vol2 = read_npy(data_folders[i+1])
    max_shift = 100

    print(f'{data_folders[i]} and {data_folders[i+1]}')

    # Cropping
    z_crop_top    = 80
    z_crop_bottom = 100
    x_crop_left   = 10
    x_crop_right  = 10
    y_crop_left   = 10
    y_crop_right  = 10

    z, x, y = vol1.shape

    vol1 = vol1[
        z_crop_top : z - z_crop_bottom,
        x_crop_left : x - x_crop_right,
        y_crop_left : y - y_crop_right
    ]

    z, x, y = vol2.shape

    vol2 = vol2[
        z_crop_top : z - z_crop_bottom,
        x_crop_left : x - x_crop_right,
        y_crop_left : y - y_crop_right
    ]

    best_shiftx, shiftsx, corr_valuesx = normalised_correlation_3D(vol1, vol2, axis='x', max_shift=max_shift)
    best_shifty, shiftsy, corr_valuesy = normalised_correlation_3D(vol1, vol2, axis='y', max_shift=max_shift)

    actual_shift = round(5e-3 / x_pixel_size)

    print(f'Actual shift: {actual_shift}')
    print(f'Best y shift: {best_shifty}, max correlation: {max(corr_valuesy):.3f}')
    print(f'Best x shift: {best_shiftx}, max correlation: {max(corr_valuesx):.3f}')
    print()

#%%
canvas1, canvas2 = stitch_volumes(vol1, vol2, best_shiftx, axis='x')

viewer = napari.Viewer()

viewer.add_image(canvas1, name="vol1", colormap="red")
viewer.add_image(canvas2, name="vol2", colormap="blue", opacity=0.5)

napari.run()

#%%
canvas1, canvas2 = stitch_volumes(vol1, vol2, actual_shift, axis='y')

viewer = napari.Viewer()

viewer.add_image(canvas1, name="vol1", colormap="red")
viewer.add_image(canvas2, name="vol2", colormap="blue", opacity=0.5)

napari.run()

