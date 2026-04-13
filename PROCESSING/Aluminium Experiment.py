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

from Classes.CalcSpeedOfSound import calcSpeedOfSound
from Classes.Stitch2D import normalised_correlation_2D
from Classes.Stitch2D import stitch_images

#%%
# Self-defined Functions
def read_png(png):
    return mpimg.imread(IMG_DATA_DIR + '/' + png)

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

block_depth     = 53.3e-3
t_threshold     = 1e-5
threshold_shift = 1e-5
avg_speed = []

for folder in speed_sound_files:
    # Locate Data
    loc = os.path.join(PRO_DATA_DIR, folder)
    print(folder)

    time_path = loc + '/time.csv'
    h5_path   = loc + '/time_data.h5'

    time_df = pd.read_csv(time_path)
    time_np = time_df['time_seconds'].to_numpy()

    with h5py.File(h5_path, 'r') as f:
        time_data = np.array(f["time_data"])
    
    speed_sound = calcSpeedOfSound(time_np, time_data, t_threshold, threshold_shift, 
                                   block_depth, amplitude_threshold=0.2, calculation_type='interp',
                                   displayBool=True, elements=[1], savePicBool=False)
    
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
c = 6370.67 m/s
z_max = 10 mm
z_min = 40 mm
vmax = 0
vmin = -20
x_pixels = 800
z_pixels = 800
This resulted in the images used for stitching, as well as the pixel size.
'''

c = 6370.67 # m/s
x_pixels = 800
z_pixels = 800
lateral_pixel_size = 0.048e-3 # m
depth_pixel_size   = 0.038e-3 # m

#%%
# Example Binary and Cropped Images
from matplotlib.patches import Rectangle

# Processing Parameters
binary_threshold = 0.78
left_crop = 150
right_crop = int(left_crop + (800 - 2*left_crop))
top_crop = int(800 / 4)
bottom_crop = 20

img1 = read_png(image_files1[2])
img2 = read_png(image_files1[3])

# Grey scale
if img1.ndim == 3:
    img1 = img1.mean(axis=2)
if img2.ndim == 3:
    img2 = img2.mean(axis=2)

crop = (
    slice(top_crop, 800 - bottom_crop),
    slice(left_crop, right_crop)
)

binary1 = (img1 > binary_threshold).astype(float)
binary2 = (img2 > binary_threshold).astype(float)

# Make sure save directory exists
os.makedirs("Images", exist_ok=True)

def plot_cropped_overlay(binary_img, save_path):
    h, w = binary_img.shape

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(binary_img, cmap="gray")

    # RGBA overlay: red outside crop, transparent inside crop
    overlay = np.zeros((h, w, 4), dtype=float)
    overlay[..., 0] = 1.0   # red channel
    overlay[..., 1] = 0.0
    overlay[..., 2] = 0.0
    overlay[..., 3] = 0.0   # start fully transparent

    # Set alpha outside kept region
    overlay[:top_crop, :, 3] = 0.28
    if bottom_crop > 0:
        overlay[h-bottom_crop:, :, 3] = 0.28
    overlay[top_crop:h-bottom_crop if bottom_crop > 0 else h, :left_crop, 3] = 0.28
    overlay[top_crop:h-bottom_crop if bottom_crop > 0 else h, right_crop:, 3] = 0.28

    ax.imshow(overlay, interpolation="none")

    rect = Rectangle(
        (left_crop, top_crop),
        right_crop - left_crop,
        (h - bottom_crop) - top_crop,
        linewidth=2,
        edgecolor='red',
        facecolor='none'
    )
    ax.add_patch(rect)

    ax.axis("off")
    fig.savefig(
        save_path,
        dpi=300,
        bbox_inches='tight',
        pad_inches=0
    )
    plt.show()

plot_cropped_overlay(binary1, 'Images/cropped_and_binarised_example_1.png')
plot_cropped_overlay(binary2, 'Images/cropped_and_binarised_example_2.png')

#%%
# Binary-ing and Cropping All Data
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
    cropped_img = binary_img[crop]
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
    cropped_img = binary_img[crop]
    reduced_images2.append(cropped_img)

#%%
# Example Stitch
img1 = reduced_images1[1]
img2 = reduced_images1[2]

dx, shifts, corr_values = normalised_correlation_2D(img1, img2)
combined_image, left_offset, w1, x2, w2 = stitch_images(img1, img2, dx)
error = abs(((5e-3 - abs(dx * lateral_pixel_size))/(5e-3)) * 100)

plt.figure(figsize=(10,6))
plt.imshow(combined_image)
plt.axis("off")

plt.axvline(left_offset, linestyle=":", linewidth=2)
plt.axvline(left_offset + w1, linestyle=":", linewidth=2)
plt.axvline(x2, linestyle=":", linewidth=2)
plt.axvline(x2 + w2, linestyle=":", linewidth=2)

plt.savefig(
    'Images/example_stitch_overlay.png',
    dpi=300,
    bbox_inches='tight',
    pad_inches=0
)
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

    dx, shifts, corr_values = normalised_correlation_2D(img1, img2)
    dxes1.append(dx)

for i, r_img in enumerate(reduced_images2[:-1]):
    img1 = r_img
    img2 = reduced_images2[i+1]

    dx, shifts, corr_values = normalised_correlation_2D(img1, img2)
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

    stitched_image1, left_offset, w1, x2, w2 = stitch_images(
        stitched_image1,
        next_img,
        dx,
        colour_bool=False
    )

stitched_image2 = full_images2[0]
for i, dx in enumerate(dxes2):

    next_img = full_images2[i+1]

    stitched_image2, left_offset, w1, x2, w2 = stitch_images(
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

actual_shift = round(5e-3 / lateral_pixel_size)
abs_shift1   = []
abs_shift2   = []
for i in range(len(dxes1)):
    abs_shift1.append(abs(actual_shift - abs(dxes1[i])))
    abs_shift2.append(abs(actual_shift - abs(dxes2[i])))

avg_shift1 = np.mean(abs_shift1) + actual_shift
avg_shift2 = np.mean(abs_shift2) + actual_shift

avg_dist1 = avg_shift1 * lateral_pixel_size
avg_dist2 = avg_shift2 * lateral_pixel_size 

error1 = abs((5e-3 - abs(avg_dist1)) / 5e-3) * 100
error2 = abs((5e-3 - abs(avg_dist2)) / 5e-3) * 100

print('Image 1')
print(f'Average Pixel Shift: {avg_shift1} pixels')
print(f'Average Calculated Distance: {avg_dist1 * 1000:.3f} mm')
print(f'Average Calculated Error: {error1:.3f}%')

print()
print('Image 2')
print(f'Average Pixel Shift: {avg_shift2} pixels')
print(f'Average Calculated Distance: {avg_dist2 * 1000:.3f} mm')
print(f'Average Calculated Error: {error2:.3f}%')

#%%