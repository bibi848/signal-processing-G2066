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

import numpy as np
import napari
import pandas as pd
import h5py

from Classes.CalcSpeedOfSound import calcSpeedOfSound

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
# Example Binary and Cropping
img = read_npy(data_folders[0])

# Greyscale
if img.ndim == 4:
    img = img.mean(axis=-1)

# Binary Mask
threshold = img.mean() + img.std()*1.2
binary_volume = (img > threshold).astype(float)

# Cropping
z_crop_top    = 50
z_crop_bottom = 20
x_crop_left   = 0
x_crop_right  = 0
y_crop_left   = 0
y_crop_right  = 0

z, x, y = binary_volume.shape

cropped_volume = binary_volume[
    z_crop_top : z - z_crop_bottom,
    x_crop_left : x - x_crop_right,
    y_crop_left : y - y_crop_right
]

viewer = napari.Viewer()

viewer.add_image(
    cropped_volume,
    name="Binary Cropped Volume",
    colormap="gray"
)
napari.run()

#%%
# Binary-ing and Cropping All Data
reduced_volumes = [[], [], [], []]

for i in range(len(groups)):
    print('i:', i)
    group = groups[i]

    for dat in group:
        img = read_npy(dat)
        
        if img.ndim == 4:
            img = img.mean(axis=-1)
        
        threshold = img.mean() + img.std()*2
        binary_volume = (img > threshold).astype(float)

        cropped_volume = binary_volume[
            z_crop_top : z - z_crop_bottom,
            x_crop_left : x - x_crop_right,
            y_crop_left : y - y_crop_right
        ]

        reduced_volumes[i].append(cropped_volume)

#%%
img = read_npy(groups[0][0])

viewer = napari.Viewer()

viewer.add_image(
    img,
    name="Test1",
    colormap="gray"
)
napari.run()

#%%
img = read_npy(groups[0][1])

viewer = napari.Viewer()

viewer.add_image(
    img,
    name="Test1",
    colormap="gray"
)
napari.run()


#%%
# Stitching Functions
def volume_correlation(vol1, vol2, max_shift=100):

    z1, x1, y1 = vol1.shape
    z2, x2, y2 = vol2.shape

    shifts = range(-max_shift, max_shift + 1)
    corr_values = []

    for dx in shifts:

        x1_start = max(0, dx)
        x1_end   = min(x1, x2 + dx)

        x2_start = max(0, -dx)
        x2_end   = min(x2, x1 - dx)

        if (x1_end - x1_start) <= 0:
            corr_values.append(0)
            continue

        region1 = vol1[:, x1_start:x1_end, :]
        region2 = vol2[:, x2_start:x2_end, :]

        numerator = np.sum(region1 * region2)
        denom = np.sqrt(np.sum(region1**2) * np.sum(region2**2))

        if denom > 0:
            corr_values.append(numerator / denom)
        else:
            corr_values.append(0)

    corr_values = np.array(corr_values)
    best_index = np.argmax(corr_values)
    best_dx = shifts[best_index]

    return best_dx, shifts, corr_values

def volume_correlation_y(vol1, vol2, max_shift=100):

    z1, x1, y1 = vol1.shape
    z2, x2, y2 = vol2.shape

    shifts = range(-max_shift, max_shift + 1)
    corr_values = []

    for dy in shifts:

        y1_start = max(0, dy)
        y1_end   = min(y1, y2 + dy)

        y2_start = max(0, -dy)
        y2_end   = min(y2, y1 - dy)

        if (y1_end - y1_start) <= 0:
            corr_values.append(0)
            continue

        region1 = vol1[:, :, y1_start:y1_end]
        region2 = vol2[:, :, y2_start:y2_end]

        numerator = np.sum(region1 * region2)
        denom = np.sqrt(np.sum(region1**2) * np.sum(region2**2))

        corr_values.append(numerator / denom if denom > 0 else 0)

    corr_values = np.array(corr_values)

    best_index = np.argmax(corr_values)
    best_dy = shifts[best_index]

    return best_dy, shifts, corr_values

def stitch_volumes(vol1, vol2, shift, axis=1):
    """
    vol1, vol2 : volumes with shape (z, x, y)
    shift      : pixel shift
    axis       : 1 = x direction, 2 = y direction
    """

    z1, x1, y1 = vol1.shape
    z2, x2, y2 = vol2.shape

    if axis == 1:   # shift in x

        left_offset = max(0, -shift)
        right_extent = max(x1, x2 + shift)
        total_x = left_offset + right_extent

        canvas1 = np.zeros((z1, total_x, y1))
        canvas2 = np.zeros((z1, total_x, y1))

        canvas1[:, left_offset:left_offset + x1, :] = vol1

        x2_start = left_offset + shift
        canvas2[:, x2_start:x2_start + x2, :] = vol2

    elif axis == 2:   # shift in y

        left_offset = max(0, -shift)
        right_extent = max(y1, y2 + shift)
        total_y = left_offset + right_extent

        canvas1 = np.zeros((z1, x1, total_y))
        canvas2 = np.zeros((z1, x1, total_y))

        canvas1[:, :, left_offset:left_offset + y1] = vol1

        y2_start = left_offset + shift
        canvas2[:, :, y2_start:y2_start + y2] = vol2

    else:
        raise ValueError("axis must be 1 (x) or 2 (y)")

    return canvas1, canvas2

#%%
# Example Stitch
vol1 = reduced_volumes[0][0]
vol2 = reduced_volumes[0][1]

dx, shifts, corr = volume_correlation(vol1, vol2, max_shift=100)

error = abs(((5e-3 - abs(dx * x_pixel_size))/(5e-3)) * 100)

print('x')
print(f'Pixel Shift: {-1*dx}')
print(f'Distance Calculated: {-1 * dx * x_pixel_size * 1000:.3f} mm')
print(f'Actual Distance: 5 mm')
print(f'Approximate Error: {error:.3f}%')

dy, shifts, corr = volume_correlation_y(vol1, vol2, max_shift=100)

error = abs(((5e-3 - abs(dy * y_pixel_size))/(5e-3)) * 100)

print('y')
print(f'Pixel Shift: {-1*dy}')
print(f'Distance Calculated: {-1 * dy * y_pixel_size * 1000:.3f} mm')
print(f'Actual Distance: 5 mm')
print(f'Approximate Error: {error:.3f}%')


#%%
canvas1, canvas2 = stitch_volumes(vol1, vol2, -85, axis=2)

viewer = napari.Viewer()

viewer.add_image(
    canvas1,
    name="Volume 1",
    colormap="red",
    blending="additive"
)

viewer.add_image(
    canvas2,
    name="Volume 2",
    colormap="cyan",
    blending="additive"
)

napari.run()

#%%
groupA = reduced_volumes[0]

shift = 85
axis = 1   # 1 = x direction, 2 = y direction

stitched_volume = groupA[0]

for i in range(1, len(groupA)):

    next_vol = groupA[i]

    canvas1, canvas2 = stitch_volumes(
        stitched_volume,
        next_vol,
        shift,
        axis=axis
    )

    # combine the two volumes
    stitched_volume = np.maximum(canvas1, canvas2)

#%%
viewer = napari.Viewer()

viewer.add_image(
    stitched_volume,
    name="Stitched Volume A",
    colormap="gray"
)

napari.run()

#%%
