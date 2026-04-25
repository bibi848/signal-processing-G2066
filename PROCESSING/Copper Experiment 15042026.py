'''
Author: OD
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

from scipy.ndimage import rotate
import matplotlib.image as mpimg
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

def read_png(png):
    return mpimg.imread(IMG_DATA_DIR + '/' + png)

#%%
# Extracting Data
processed_data_type = '2D Processed Data'
processed_data_name = 'Cu Pure 7.5MHz Ex 15042026'
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
# block_depth = 50e-3
# t_threshold = 1e-5
# threshold_shift = 2e-5
# avg_speed = []

# for folder in speed_of_sound_folders:
#     # Locate Data
#     loc = os.path.join(PRO_DATA_DIR, folder)

#     time_path = loc + '/time.csv'
#     h5_path   = loc + '/time_data.h5'

#     time_df = pd.read_csv(time_path)
#     time_np = time_df['time_seconds'].to_numpy()

#     with h5py.File(h5_path, 'r') as f:
#         time_data = np.array(f["time_data"])

#     speed_sound = calcSpeedOfSound(time_np, time_data, t_threshold, threshold_shift,
#                                    block_depth, amplitude_threshold=0.09, calculation_type='interp',
#                                    displayBool=True, elements=[1], savePicBool=False,)

#     print(f'Speed of Sound: {speed_sound:.2f} m/s')
#     print()
#     avg_speed.append(speed_sound)

# print(f'Average Speed of Sound: {np.mean(avg_speed):.2f} m/s')
# print()

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
# Imaging Parameters
'''
c        = 4636.1 # m/s
z_max    = 15e-3  # m
z_min    = 35e-3 # m
x_min    = 'xc_min' # m, can specify length
x_max    = 'xc_max' # or just use xc_min/xc_max
y_min    = 'yc_min'
y_max    = 'yc_max'
x_pixels = 300
y_pixels = 300
z_pixels = 500

X-dir pixel size: 0.039 mm
Y-dir pixel size: 0.040 mm
Z-dir pixel size: 0.040 mm
Centre Frequency:   7.5 MHz
Wavelength:         0.618 mm
'''

c = 4636.1 # m/s
x_pixels = 300
y_pixels = 300
z_pixels = 500
x_pixel_size = 0.039e-3 # m
y_pixel_size = 0.040e-3
z_pixel_size = 0.040e-3 # m

ROTATION_NPY_DIR = os.path.join(Path(__file__).resolve().parent, 'Rotation NPYs')
os.makedirs(ROTATION_NPY_DIR, exist_ok=True)

#%%
# Check overlap between two rotations before fusing

position_number = 1
rotation_a = 1     
rotation_b = 2     

correction_rotations_deg = [0, -90, -180, -270]

file_a = f'{position_number}{rotation_a}_filtered_3D_TFM.npy'
file_b = f'{position_number}{rotation_b}_filtered_3D_TFM.npy'

path_a = os.path.join(IMG_DATA_DIR, file_a)
path_b = os.path.join(IMG_DATA_DIR, file_b)

vol_a = np.load(path_a).astype(np.float32)
vol_b = np.load(path_b).astype(np.float32)

vol_a_rot = rotate(
    vol_a,
    angle=correction_rotations_deg[rotation_a - 1],
    axes=(1, 2),
    reshape=False,
    order=1,
    mode='constant',
    cval=0.0
).astype(np.float32)

vol_b_rot = rotate(
    vol_b,
    angle=correction_rotations_deg[rotation_b - 1],
    axes=(1, 2),
    reshape=False,
    order=1,
    mode='constant',
    cval=0.0
).astype(np.float32)

# Crop side edges
vol_a_rot = vol_a_rot[:, 40:-40, 40:-40]
vol_b_rot = vol_b_rot[:, 40:-40, 40:-40]

# Normalise for visual comparison
a_min, a_max = vol_a_rot.min(), vol_a_rot.max()
b_min, b_max = vol_b_rot.min(), vol_b_rot.max()

if a_max > a_min:
    vol_a_norm = (vol_a_rot - a_min) / (a_max - a_min)
else:
    vol_a_norm = np.zeros_like(vol_a_rot)

if b_max > b_min:
    vol_b_norm = (vol_b_rot - b_min) / (b_max - b_min)
else:
    vol_b_norm = np.zeros_like(vol_b_rot)

overlay = np.zeros(vol_a_norm.shape + (3,), dtype=np.float32)
overlay[..., 0] = vol_a_norm          # red
overlay[..., 2] = vol_a_norm + vol_b_norm   # blue
overlay = np.clip(overlay, 0, 1)

mip_a = np.max(vol_a_norm, axis=0)
mip_b = np.max(vol_b_norm, axis=0)
mip_overlay = np.max(overlay, axis=0)

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(mip_a, cmap='gray', origin='lower')
plt.title(f'Rotation {rotation_a}')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(mip_b, cmap='gray', origin='lower')
plt.title(f'Rotation {rotation_b}')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(mip_overlay, origin='lower')
plt.title('Overlay')
plt.axis('off')

plt.tight_layout()
plt.show()

viewer = napari.Viewer()
viewer.add_image(vol_a_norm, name=f'Rotation {rotation_a}', colormap='magenta')
viewer.add_image(vol_b_norm, name=f'Rotation {rotation_b}', colormap='blue')
viewer.add_image(overlay, name='Overlay RGB', rgb=True)
napari.run()

#%%
# Fuse each rotation at each position
correction_rotations_deg = [0, -90, -180, -270]

for i in range(4):
    print(i)
    aligned_volumes = []

    for j in range(4):

        file_name = f'{i+1}{j+1}_filtered_3D_TFM.npy'
        file_path = os.path.join(IMG_DATA_DIR, file_name)

        vol = np.load(file_path).astype(np.float32)

        vol_rot = rotate(
            vol,
            angle=correction_rotations_deg[j],
            axes=(1, 2),
            reshape=False,
            order=1,
            mode='constant',
            cval=0.0
        ).astype(np.float32)

        vol_rot = vol_rot[:, 40:-40, 40:-40]
        aligned_volumes.append(vol_rot)

    fused_volume = np.max(np.stack(aligned_volumes, axis=0), axis=0)

    save_path = os.path.join(ROTATION_NPY_DIR, f'position_{i+1}_overlay_max_raw.npy')
    np.save(save_path, fused_volume)

#%%
# Visualise Overlays

position_number = 3
aligned_volumes = []

for j in range(4):

    file_name = f'{position_number}{j+1}_filtered_3D_TFM.npy'
    file_path = os.path.join(IMG_DATA_DIR, file_name)

    vol = np.load(file_path).astype(np.float32)

    vol_rot = rotate(
        vol,
        angle=correction_rotations_deg[j],
        axes=(1, 2),
        reshape=False,
        order=1,
        mode='constant',
        cval=0.0
    ).astype(np.float32)

    vol_rot = vol_rot[:, 40:-40, 40:-40]
    aligned_volumes.append(vol_rot)

fused_volume = np.max(np.stack(aligned_volumes, axis=0), axis=0)

viewer = napari.Viewer()
viewer.add_image(fused_volume, name=f'Position {position_number} fused')

for j, vol in enumerate(aligned_volumes):
    viewer.add_image(vol, name=f'Position {position_number} rotation {j+1}')

napari.run()

#%%
# Stitch Test

position_a = 3
position_b = 4
stitch_axis = 'x' # x/y
max_shift = 100

if stitch_axis not in ['x', 'y']:
    raise ValueError("stitch_axis must be 'x' or 'y'")

vol_a_path = os.path.join(ROTATION_NPY_DIR, f'position_{position_a}_overlay_max_raw.npy')
vol_b_path = os.path.join(ROTATION_NPY_DIR, f'position_{position_b}_overlay_max_raw.npy')

vol_a = np.load(vol_a_path).astype(np.float32)
vol_b = np.load(vol_b_path).astype(np.float32)

pixel_size = x_pixel_size if stitch_axis == 'x' else y_pixel_size

shift, shifts, corr_values = normalised_correlation_3D(
    vol_a,
    vol_b,
    axis=stitch_axis,
    max_shift=max_shift
)

canvas_a, canvas_b = stitch_volumes(vol_a, vol_b, shift, axis=stitch_axis)

print(f'Stitch Axis: {stitch_axis}')
print(f'Pixel Shift: {shift} pixels')
print(f'Distance Calculated: {shift * pixel_size * 1000:.3f} mm')
print(f'Absolute Distance: {abs(shift * pixel_size * 1000):.3f} mm')

a_min, a_max = canvas_a.min(), canvas_a.max()
b_min, b_max = canvas_b.min(), canvas_b.max()

if a_max > a_min:
    canvas_a_norm = (canvas_a - a_min) / (a_max - a_min)
else:
    canvas_a_norm = np.zeros_like(canvas_a)

if b_max > b_min:
    canvas_b_norm = (canvas_b - b_min) / (b_max - b_min)
else:
    canvas_b_norm = np.zeros_like(canvas_b)

viewer = napari.Viewer()
viewer.add_image(
    canvas_a_norm,
    name=f'Position {position_a}',
    colormap='magenta',
    blending='additive'
)
viewer.add_image(
    canvas_b_norm,
    name=f'Position {position_b}',
    colormap='cyan',
    blending='additive'
)
napari.run()

#%%
# Find Shifts Between All Positions

stitch_axis = 'x' # x/y
max_shift = 100
number_of_positions = 4

pixel_size = x_pixel_size if stitch_axis == 'x' else y_pixel_size
volumes = []

for i in range(number_of_positions):

    file_path = os.path.join(ROTATION_NPY_DIR, f'position_{i+1}_overlay_max_raw.npy')
    vol = np.load(file_path).astype(np.float32)
    volumes.append(vol)

for i in range(number_of_positions - 1):

    vol_a = volumes[i]
    vol_b = volumes[i+1]

    shift, shifts, corr_values = normalised_correlation_3D(
        vol_a,
        vol_b,
        axis=stitch_axis,
        max_shift=max_shift
    )

    print(f'Positions {i+1} to {i+2}')
    print(f'Stitch Axis: {stitch_axis}')
    print(f'Pixel Shift: {shift} pixels')
    print(f'Distance Calculated: {shift * pixel_size * 1000:.3f} mm')
    print(f'Absolute Distance: {abs(shift * pixel_size * 1000):.3f} mm')

    print()
