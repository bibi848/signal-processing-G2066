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
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import h5py

from Classes.CalcSpeedOfSound import calcSpeedOfSound

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

#%%
# Speed of Sound Calculations

speed_of_sound_folders = [x for x in image_folders if 'Speed of Sound' in x]

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
c = 6126.44 m/s
z_max = 10 mm
z_min = 40 mm
vmax = 0
vmin = -20
x_pixels = 800
z_pixels = 800
This resulted in the images used for stitching, as well as the pixel size.
'''

c = 6126.44 # m/s
x_pixels = 800
z_pixels = 800
lateral_pixel_size = 0.048e-3 # m
depth_pixel_size   = 0.038e-3 # m

#%%
# Example Binary and Cropped Images
img1 = read_png(image_files1[1])
img2 = read_png(image_files1[2])

# Grey scale
if img1.ndim == 3: img1 = img1.mean(axis=2)
else: img1 = img1
if img2.ndim == 3: img2 = img2.mean(axis=2)
else: img2 = img2

threshold = 0.75

binary1 = (img1 > threshold).astype(float)
binary2 = (img2 > threshold).astype(float)

left_crop = 230
right_crop = int(left_crop + (800 - 2*left_crop))
top_crop = int(800 / 5)
bottom_crop = 0

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
binary_threshold = 0.75
left_crop = 240
right_crop = int(left_crop + (800 - 2*left_crop))
top_crop = int(800 / 4)
bottom_crop = 0

reduced_images1 = []
reduced_images2 = []

for image_name in image_files1:
    img = read_png(image_name)

    # Grey Scale
    if img.ndim == 3: img = img.mean(axis=2)
    else: img = img

    # Binary Image
    binary_img  = (img > binary_threshold).astype(float)
    h, w = binary_img.shape

    # Cropped Image
    cropped_img = binary_img[
        top_crop  : h - bottom_crop,
        left_crop : w - left_crop
    ]
    reduced_images1.append(cropped_img)

for image_name in image_files2:
    img = read_png(image_name)

    # Gray Scale
    if img.ndim == 3: img = img.mean(axis=2)
    else: img = img

    # Binary Image
    binary_img  = (img > binary_threshold).astype(float)
    h, w = binary_img.shape

    # Cropped Image
    cropped_img = binary_img[
        top_crop  : h - bottom_crop,
        left_crop : w - left_crop
    ]
    reduced_images2.append(cropped_img)

#%%
# Example Stitch
img1 = reduced_images1[0]
img2 = reduced_images2[1]

dx, shifts, corr_values = horizontal_correlation(img1, img2)
combined_image, left_offset, w1, x2, w2 = stitch_horizontal(img1, img2, dx)
error = abs(((5e-3 - abs(dx * lateral_pixel_size))/(5e-3)) * 100)

plt.figure(figsize=(10,6))
plt.imshow(combined_image)
plt.axis("off")

plt.axvline(left_offset, linestyle=":", linewidth=2)
plt.axvline(left_offset + w1, linestyle=":", linewidth=2)
plt.axvline(x2, linestyle=":", linewidth=2)
plt.axvline(x2 + w2, linestyle=":", linewidth=2)

plt.show()

print(f'Pixel Shift: {-1*dx}')
print(f'Distance Calculated: {-1 * dx * lateral_pixel_size * 1000:.3f} mm')
print(f'Actual Distance: 5 mm')
print(f'Approximate Error: {error:.3f}%')

#%%
# Finding all Pixel Shifts

dxes1 = []
dxes2 = []

for i, r_img in enumerate(reduced_images1[:-1]):
    img1 = r_img
    img2 = reduced_images1[i+1]

    dx, shifts, corr_values = horizontal_correlation(img1, img2)
    dxes1.append(dx)

for i, r_img in enumerate(reduced_images2[:-1]):
    img1 = r_img
    img2 = reduced_images2[i+1]

    dx, shifts, corr_values = horizontal_correlation(img1, img2)
    dxes2.append(dx)

#%%
# Converting Full Images to Greyscale
full_images1 = []
full_images2 = []

for image_name in image_files1:
    img = read_png(image_name)
    if img.ndim == 3: img = img.mean(axis=2)
    full_images1.append(img)

for image_name in image_files2:
    img = read_png(image_name)
    if img.ndim == 3: img = img.mean(axis=2)
    full_images2.append(img)

#%%
# Cumulative Shifting
stitched_image1 = full_images1[0]

for i, dx in enumerate(dxes1):

    next_img = full_images1[i+1]

    stitched_image1, left_offset, w1, x2, w2 = stitch_horizontal(
        stitched_image1,
        next_img,
        dx,
        colour_bool=False
    )

stitched_image2 = full_images2[0]
for i, dx in enumerate(dxes2):

    next_img = full_images2[i+1]

    stitched_image2, left_offset, w1, x2, w2 = stitch_horizontal(
        stitched_image2,
        next_img,
        dx,
        colour_bool=False
    )

#%%
# Display Shifted Data
plt.figure(figsize=(12,6))
plt.imshow(stitched_image1, cmap="gray")
plt.axis("off")
plt.show()

plt.figure(figsize=(12,6))
plt.imshow(stitched_image2, cmap="gray")
plt.axis("off")
plt.show()

avg_shift1 = np.mean(dxes1)
avg_shift2 = np.mean(dxes2)

avg_dist1 = -1 * avg_shift1 * lateral_pixel_size
avg_dist2 = -1 * avg_shift2 * lateral_pixel_size

error1 = abs((5e-3 - abs(avg_dist1)) / 5e-3) * 100
error2 = abs((5e-3 - abs(avg_dist2)) / 5e-3) * 100

print('Image 1')
print(f'Average Pixel Shift: {avg_shift1 * -1} pixels')
print(f'Average Calculated Distance: {avg_dist1 * 1000:.3f} mm')
print(f'Average Calculated Error: {error1:.3f}%')

print()
print('Image 2')
print(f'Average Pixel Shift: {avg_shift2 * -1} pixels')
print(f'Average Calculated Distance: {avg_dist2 * 1000:.3f} mm')
print(f'Average Calculated Error: {error2:.3f}%')

#%%