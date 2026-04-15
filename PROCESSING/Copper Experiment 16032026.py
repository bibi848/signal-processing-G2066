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
                                   block_depth, amplitude_threshold=0.09, calculation_type='interp',
                                   displayBool=True, elements=[1], savePicBool=False,)

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
# 2D Imaging Parameters
'''
# 3D Images
c        = 4636.1 # m/s
z_max    = 5e-3  # m
z_min    = 25e-3 # m
x_min    = 'xc_min' # m, can specify length
x_max    = 'xc_max' # or just use xc_min/xc_max
y_min    = 'yc_min'
y_max    = 'yc_max'
x_pixels = 200
y_pixels = 200
z_pixels = 400

X-dir pixel size: 0.059 mm
Y-dir pixel size: 0.059 mm
Z-dir pixel size: 0.050 mm
Centre Frequency:   7.5 MHz
Wavelength:         0.627 mm
'''

c = 4636.1 # m/s
x_pixels = 800
z_pixels = 800
x_pixel_size = 0.033e-3 # m
z_pixel_size = 0.031e-3 # m

#%%
# 2D Array: fuse 4 rotations at each position, save max_raw to Images folder,
# then stitch position 1 and 2 with top/bottom crop.

from scipy.ndimage import rotate as nd_rotate
from pathlib import Path

# -------------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------------
def load_3d_tfm_file(file_stem):
    """
    Example:
        '11_filtered' -> IMG_DATA_DIR / '11_filtered_3D_TFM.npy'
    """
    npy_path = Path(IMG_DATA_DIR) / f"{file_stem}_3D_TFM.npy"

    if not npy_path.exists():
        raise FileNotFoundError(f"Could not find file: {npy_path}")

    volume = np.load(npy_path).astype(np.float32)
    print(f"Loaded {npy_path.name} with shape {volume.shape}")
    return volume


def rotate_volume_xy(volume, angle_deg):
    """
    Rotate in the x-y plane.
    Assumes volume shape is (z, y, x).
    """
    return nd_rotate(
        volume,
        angle=angle_deg,
        axes=(1, 2),
        reshape=False,
        order=1,
        mode='constant',
        cval=0.0,
        prefilter=True
    ).astype(np.float32)


def crop_rotation_edges(volume, crop_pixels=40):
    """
    Crop side edges after in-plane rotation to remove black/interpolation borders.
    Crops x and y only, not z.
    """
    if crop_pixels <= 0:
        return volume

    z, y, x = volume.shape
    return volume[:, crop_pixels:y-crop_pixels, crop_pixels:x-crop_pixels]


def crop_top_bottom(volume, top_crop=20, bottom_crop=10):
    """
    Crop z-direction before stitching.
    Assumes axis 0 is z.
    """
    z, y, x = volume.shape
    z_start = top_crop
    z_end = z - bottom_crop if bottom_crop > 0 else z
    return volume[z_start:z_end, :, :]


def normalise01(volume):
    v = volume.astype(np.float32)
    vmin = np.min(v)
    vmax = np.max(v)

    if vmax <= vmin:
        return np.zeros_like(v, dtype=np.float32)

    return (v - vmin) / (vmax - vmin)


def make_colour_overlay(vol1, vol2):
    """
    Build an RGB overlay volume:
      - vol1 shown in pink (R+B)
      - vol2 shown in blue (B)
    Returns array shape (z, y, x, 3)
    """
    a = normalise01(vol1)
    b = normalise01(vol2)

    overlay = np.zeros(a.shape + (3,), dtype=np.float32)

    # pink
    overlay[..., 0] += a   # red
    overlay[..., 2] += a   # blue

    # blue
    overlay[..., 2] += b

    overlay = np.clip(overlay, 0.0, 1.0)
    return overlay


def save_volume_in_images(volume, filename):
    """
    Save output .npy into local Images folder.
    """
    os.makedirs("Images", exist_ok=True)
    save_path = Path("Images") / filename
    np.save(save_path, volume)
    print(f"Saved: {save_path.resolve()}")


# -------------------------------------------------------------------------
# Fuse all positions
# -------------------------------------------------------------------------
# For each physical position p = 1..5, assume files:
#   p1_filtered, p2_filtered, p3_filtered, p4_filtered
# representing the 4 probe rotations at that position.

correction_rotations_deg = [0, -90, -180, -270]

fused_position_volumes = {}

for pos in range(1, 6):
    selected_scans = [f"{pos}{rot}_filtered" for rot in range(1, 5)]
    print(f"\nProcessing position {pos}: {selected_scans}")

    aligned_volumes = []

    for file_stem, corr_rot in zip(selected_scans, correction_rotations_deg):
        vol = load_3d_tfm_file(file_stem)

        # Rotate back to common orientation
        vol_aligned = rotate_volume_xy(vol, corr_rot)

        # Remove rotated-edge artifacts
        vol_aligned = crop_rotation_edges(vol_aligned, crop_pixels=40)

        aligned_volumes.append(vol_aligned)

    stack_raw = np.stack(aligned_volumes, axis=0)
    combined_max_raw = np.max(stack_raw, axis=0)

    fused_position_volumes[pos] = combined_max_raw

    # Save only max_raw in Images folder
    save_volume_in_images(combined_max_raw, f"position_{pos}_overlay_max_raw.npy")


# -------------------------------------------------------------------------
# Stitch position 1 and 2
# -------------------------------------------------------------------------
vol1 = fused_position_volumes[1]
vol2 = fused_position_volumes[2]

# Crop top and bottom before stitching
vol1_crop = crop_top_bottom(vol1, top_crop=20, bottom_crop=10)
vol2_crop = crop_top_bottom(vol2, top_crop=20, bottom_crop=10)

print("\nVolumes for stitching:")
print("Position 1 cropped shape:", vol1_crop.shape)
print("Position 2 cropped shape:", vol2_crop.shape)

# 3D correlation-based shift
dx, shifts, corr_values = normalised_correlation_3D(vol1_crop, vol2_crop)
print(f"Calculated pixel shift between position 1 and 2: {dx}")

# 3D stitched volume
stitched_12, left_offset, w1, x2, w2 = stitch_volumes(vol1_crop, vol2_crop, dx)

# Save stitched result
save_volume_in_images(stitched_12, "stitched_position_1_2.npy")

# -------------------------------------------------------------------------
# Build coloured overlay for visualisation
# -------------------------------------------------------------------------
# We want the same placement as the stitch, but without blending to grayscale.
# So manually place the two cropped volumes onto a common canvas.

z1, y1, x1_size = vol1_crop.shape
z2, y2, x2_size = vol2_crop.shape

if z1 != z2 or y1 != y2:
    raise ValueError("Volumes must match in z and y dimensions for this stitching assumption.")

canvas_width = max(left_offset + x1_size, x2 + x2_size)

canvas1 = np.zeros((z1, y1, canvas_width), dtype=np.float32)
canvas2 = np.zeros((z1, y1, canvas_width), dtype=np.float32)

canvas1[:, :, left_offset:left_offset + x1_size] = vol1_crop
canvas2[:, :, x2:x2 + x2_size] = vol2_crop

overlay_12 = make_colour_overlay(canvas1, canvas2)

save_volume_in_images(overlay_12, "stitched_position_1_2_colour_overlay.npy")

# -------------------------------------------------------------------------
# Visualise in napari
# -------------------------------------------------------------------------
viewer = napari.Viewer()
viewer.add_image(vol1_crop, name="Position 1 cropped", colormap="magenta")
viewer.add_image(vol2_crop, name="Position 2 cropped", colormap="blue")
viewer.add_image(stitched_12, name="Stitched 1+2")
viewer.add_image(overlay_12, name="Colour overlay 1+2", rgb=True)
napari.run()