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
import matplotlib.image as mpimg
import pandas as pd
import h5py

#%%
# Self-defined Functions
def read_png(png):
    return mpimg.imread(IMG_DATA_DIR + '/' + png)

def horizontal_correlation(img1, img2, max_shift=400):

    h1, w1 = img1.shape
    h2, w2 = img2.shape

    corr_values = []

    shifts = range(-max_shift, max_shift + 1)

    for dx in shifts:
        x1_start = max(0, dx)
        x1_end   = min(w1, w2 + dx)

        x2_start = max(0, -dx)
        x2_end   = min(w2, w1 - dx)

        if (x1_end - x1_start) <= 0:
            corr_values.append(0)
            continue

        region1 = img1[:, x1_start:x1_end]
        region2 = img2[:, x2_start:x2_end]

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

def stitch_horizontal(img1, img2, dx, colour_bool=True):

    h1, w1 = img1.shape
    h2, w2 = img2.shape

    left_offset = max(0, -dx)
    right_extent = max(w1, w2 + dx)

    total_width = left_offset + right_extent
    height = h1

    canvas1 = np.zeros((height, total_width))
    canvas2 = np.zeros((height, total_width))

    canvas1[:, left_offset:left_offset + w1] = img1
    x2 = left_offset + dx
    canvas2[:, x2:x2 + w2] = img2

    if colour_bool:
        stitched = np.zeros((height, total_width, 3))

        stitched[:, :, 0] += canvas1 * 1.0
        stitched[:, :, 2] += canvas1 * 0.8

        stitched[:, :, 0] += canvas2 * 0.3
        stitched[:, :, 1] += canvas2 * 0.85
        stitched[:, :, 2] += canvas2 * 1.0

        overlap = (canvas1 > 0) & (canvas2 > 0)
        stitched[overlap] = [1, 1, 1]

        stitched = np.clip(stitched, 0, 1)

    else:
        # normal grayscale stitching
        stitched = np.maximum(canvas1, canvas2)

    return stitched, left_offset, w1, x2, w2

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
image_files1 = ['A1_filtered_TFM.png', 'A2_filtered_TFM.png', 'A3_filtered_TFM.png', 'A4_filtered_TFM.png', 'A5_filtered_TFM.png']
image_files2 = ['B1_filtered_TFM.png', 'B2_filtered_TFM.png', 'B3_filtered_TFM.png', 'B4_filtered_TFM.png', 'B5_filtered_TFM.png']

#%%
# Speed of Sound Calculations

block_depth = 53.3e-3
t_threshold = 1e-5

for folder in speed_sound_files:
    # Locate Data
    loc = os.path.join(PRO_DATA_DIR, folder)

    time_path = loc + '/time.csv'
    h5_path   = loc + '/time_data.h5'

    time_df = pd.read_csv(time_path)
    time_np = time_df['time_seconds'].to_numpy()

    with h5py.File(h5_path, 'r') as f:
        time_data = np.array(f["time_data"])

    # Find backwall reflection
    mask = time_np > t_threshold
    time_after = time_np[mask]
    signal_after = time_data[0][mask]

    max_idx  = np.argmax(signal_after)
    max_time = time_after[max_idx]
    max_val  = signal_after[max_idx]

    plt.plot(time_np, time_data[0])
    plt.scatter(max_time, max(time_data[0]), c='r')

    sound_speed = 2*(block_depth / max_time)

    print(f'Speed of Sound: {sound_speed:.2f} m/s')
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