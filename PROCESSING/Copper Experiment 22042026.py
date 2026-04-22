'''
This script goes through the process of analysing the backscattering
diffraction data from the copper sample scanned on 22/04/2026.
The data consists of two scan groups with no array rotation:

Group 1 (r1x) uses mostly x-axis shifts with one y-axis shift.
Group 2 (r2x) uses mostly y-axis shifts with one x-axis shift.

The scan order and imaging parameters are taken directly from the
Parameters.txt file stored alongside the imaged data.
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

from Classes.Stitch3D import normalised_correlation_3D
from Classes.Stitch3D import stitch_volumes

#%%
# Self described functions
def read_npy(scan_label):
    return np.load(os.path.join(IMG_DATA_DIR, f'{scan_label}_filtered_3D_TFM.npy'))

def crop_volume(vol, crop_pixels=40):
    if crop_pixels <= 0:
        return vol
    return vol[:, crop_pixels:-crop_pixels, crop_pixels:-crop_pixels]

def normalise_volume(vol):
    vmin = np.min(vol)
    vmax = np.max(vol)

    if vmax > vmin:
        return (vol - vmin) / (vmax - vmin)

    return np.zeros_like(vol)

def analyse_pair(scan_a, scan_b, stitch_axis='x', max_shift=100, crop_pixels=40):

    if stitch_axis not in ['x', 'y']:
        raise ValueError("stitch_axis must be 'x' or 'y'")

    vol_a = crop_volume(read_npy(scan_a).astype(np.float32), crop_pixels=crop_pixels)
    vol_b = crop_volume(read_npy(scan_b).astype(np.float32), crop_pixels=crop_pixels)

    pixel_size = x_pixel_size if stitch_axis == 'x' else y_pixel_size

    shift, shifts, corr_values = normalised_correlation_3D(
        vol_a,
        vol_b,
        axis=stitch_axis,
        max_shift=max_shift
    )

    canvas_a, canvas_b = stitch_volumes(vol_a, vol_b, shift, axis=stitch_axis)

    print(f'{scan_a} to {scan_b}')
    print(f'Stitch Axis: {stitch_axis}')
    print(f'Pixel Shift: {shift} pixels')
    print(f'Distance Calculated: {shift * pixel_size * 1000:.3f} mm')
    print(f'Absolute Distance: {abs(shift * pixel_size * 1000):.3f} mm')
    print(f'Peak Correlation: {np.max(corr_values):.4f}')
    print()

    return {
        'scan_a': scan_a,
        'scan_b': scan_b,
        'axis': stitch_axis,
        'shift_pixels': shift,
        'distance_mm': shift * pixel_size * 1000,
        'absolute_distance_mm': abs(shift * pixel_size * 1000),
        'correlation_peak': np.max(corr_values),
        'canvas_a': canvas_a,
        'canvas_b': canvas_b,
        'vol_a': vol_a,
        'vol_b': vol_b,
        'shifts': shifts,
        'corr_values': corr_values,
    }

def build_group_position_table(group_name, pair_results):

    scan_labels = SCAN_GROUPS[group_name]['scan_labels']
    x_positions_pixels = [0]
    y_positions_pixels = [0]

    x_current = 0
    y_current = 0

    for result in pair_results:

        if result['axis'] == 'x':
            x_current += result['shift_pixels']
        elif result['axis'] == 'y':
            y_current += result['shift_pixels']

        x_positions_pixels.append(x_current)
        y_positions_pixels.append(y_current)

    position_df = pd.DataFrame({
        'scan_label': scan_labels,
        'x_shift_pixels': x_positions_pixels,
        'y_shift_pixels': y_positions_pixels,
        'x_shift_mm': np.array(x_positions_pixels) * x_pixel_size * 1000,
        'y_shift_mm': np.array(y_positions_pixels) * y_pixel_size * 1000,
    })

    return position_df

def build_group_composite(scan_labels, position_df, crop_pixels=40):

    volumes = []
    for scan_label in scan_labels:
        vol = crop_volume(read_npy(scan_label).astype(np.float32), crop_pixels=crop_pixels)
        volumes.append(vol)

    z_dim, x_dim, y_dim = volumes[0].shape

    x_positions = position_df['x_shift_pixels'].to_numpy().astype(int)
    y_positions = position_df['y_shift_pixels'].to_numpy().astype(int)

    x_min = np.min(x_positions)
    x_max = np.max(x_positions)
    y_min = np.min(y_positions)
    y_max = np.max(y_positions)

    total_x = x_dim + (x_max - x_min)
    total_y = y_dim + (y_max - y_min)

    composite = np.full((z_dim, total_x, total_y), -np.inf, dtype=np.float32)

    for vol, x_pos, y_pos in zip(volumes, x_positions, y_positions):
        x_start = x_pos - x_min
        y_start = y_pos - y_min

        composite[:, x_start:x_start + x_dim, y_start:y_start + y_dim] = np.maximum(
            composite[:, x_start:x_start + x_dim, y_start:y_start + y_dim],
            vol
        )

    finite_values = composite[np.isfinite(composite)]
    if finite_values.size == 0:
        return np.zeros_like(composite)

    composite[~np.isfinite(composite)] = np.min(finite_values)

    return composite

def plot_scan_layouts():

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    for ax, (group_name, group_data) in zip(axes, SCAN_GROUPS.items()):
        layout = np.array(group_data['layout'])
        ax.imshow(np.zeros(layout.shape), cmap='gray', vmin=0, vmax=1)

        for row in range(layout.shape[0]):
            for col in range(layout.shape[1]):
                ax.text(col, row, str(layout[row, col]),
                        ha='center', va='center', fontsize=14, color='white')

        ax.set_title(f"{group_name}: {group_data['title']}")
        ax.set_xticks(range(layout.shape[1]))
        ax.set_yticks(range(layout.shape[0]))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(color='white', linewidth=1)

    plt.tight_layout()
    plt.show()

def plot_group_positions(group_name, position_df):

    plt.figure(figsize=(6, 5))
    plt.plot(position_df['x_shift_mm'], position_df['y_shift_mm'], '-o')

    for _, row in position_df.iterrows():
        plt.text(row['x_shift_mm'], row['y_shift_mm'], row['scan_label'])

    plt.xlabel('x shift [mm]')
    plt.ylabel('y shift [mm]')
    plt.title(f'Cumulative Scan Positions: {group_name}')
    plt.grid(True)
    plt.axis('equal')
    plt.show()

#%%
# Extracting Data
processed_data_type = '2D Processed Data'
processed_data_name = 'Cu Pure 7.5MHz Ex 22042026'
imaged_data_name    = '2D TFM Data'

cwd      = root_path
filtered = True

# Input and Output paths.
PRO_DATA_DIR = os.path.join(cwd, 'DATA', processed_data_type, (processed_data_name + ' Filtered'))
IMG_DATA_DIR = os.path.join(cwd, 'DATA', imaged_data_name, (processed_data_name + ' Filtered'))

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
# 2D Array Element Positions
geometry_path = os.path.join(PRO_DATA_DIR, array_2D_image_folders[0], 'array_geometry.csv')
array_geometry = pd.read_csv(geometry_path)

x = array_geometry['el_xc'].to_numpy()
y = array_geometry['el_yc'].to_numpy()

x = x - np.mean(x)
y = y - np.mean(y)

plt.figure(figsize=(6, 6))
plt.scatter(x * 1e3, y * 1e3, s=10)
plt.xlabel('x [mm]')
plt.ylabel('y [mm]')
plt.title('Array Element Positions')
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
y_pixel_size = 0.040e-3 # m
z_pixel_size = 0.040e-3 # m

# These pair axes follow the scan order noted in Parameters.txt.
SCAN_GROUPS = {
    'r1': {
        'title': '2 mm moves',
        'scan_labels': ['r11', 'r12', 'r13', 'r14', 'r15', 'r16'],
        'pair_axes': ['x', 'x', 'y', 'x', 'x'],
        'layout': [
            [6, 5, 4],
            [1, 2, 3],
        ],
    },
    'r2': {
        'title': '3 mm moves',
        'scan_labels': ['r21', 'r22', 'r23', 'r24', 'r25', 'r26'],
        'pair_axes': ['y', 'y', 'x', 'y', 'y'],
        'layout': [
            [6, 1],
            [5, 2],
            [4, 3],
        ],
    },
}

#%%
# Scan Layouts from Parameters.txt
plot_scan_layouts()

#%%
# Stitch Test

group_name = 'r1'
pair_number = 3
max_shift = 100
crop_pixels = 40

scan_labels = SCAN_GROUPS[group_name]['scan_labels']
pair_axes = SCAN_GROUPS[group_name]['pair_axes']

scan_a = scan_labels[pair_number - 1]
scan_b = scan_labels[pair_number]
stitch_axis = pair_axes[pair_number - 1]

pair_result = analyse_pair(
    scan_a,
    scan_b,
    stitch_axis=stitch_axis,
    max_shift=max_shift,
    crop_pixels=crop_pixels
)

canvas_a_norm = normalise_volume(pair_result['canvas_a'])
canvas_b_norm = normalise_volume(pair_result['canvas_b'])

viewer = napari.Viewer()
viewer.add_image(
    canvas_a_norm,
    name=f'{scan_a}',
    colormap='magenta',
    blending='additive'
)
viewer.add_image(
    canvas_b_norm,
    name=f'{scan_b}',
    colormap='cyan',
    blending='additive'
)
napari.run()

#%%
# Find Shifts Between All Positions

max_shift = 100
crop_pixels = 40
group_results = {}
group_position_tables = {}

for group_name, group_data in SCAN_GROUPS.items():

    print(f'Group: {group_name} ({group_data["title"]})')
    print()

    scan_labels = group_data['scan_labels']
    pair_axes = group_data['pair_axes']
    pair_results = []

    for i in range(len(scan_labels) - 1):

        pair_result = analyse_pair(
            scan_labels[i],
            scan_labels[i + 1],
            stitch_axis=pair_axes[i],
            max_shift=max_shift,
            crop_pixels=crop_pixels
        )
        pair_results.append(pair_result)

    group_results[group_name] = pair_results

    position_df = build_group_position_table(group_name, pair_results)
    group_position_tables[group_name] = position_df

    print('Cumulative Positions:')
    print(position_df.to_string(index=False))
    print()

#%%
# Plot cumulative x/y positions for each scan group

for group_name, position_df in group_position_tables.items():
    plot_group_positions(group_name, position_df)

#%%
# Build composite overlays for each group

group_name = 'r2'
crop_pixels = 40

position_df = group_position_tables[group_name]
scan_labels = SCAN_GROUPS[group_name]['scan_labels']
group_composite = build_group_composite(scan_labels, position_df, crop_pixels=crop_pixels)

viewer = napari.Viewer()
viewer.add_image(
    group_composite,
    name=f'{group_name} composite'
)
napari.run()

#%%
# Save a horizontal slice from the composite volume as a .png

group_name = 'r2'
z_depth_mm = 8.0
save_name = f'{group_name}_horizontal_slice_8p0mm.png'
vmin = -20
vmax = 0

position_df = group_position_tables[group_name]
scan_labels = SCAN_GROUPS[group_name]['scan_labels']
group_composite = build_group_composite(scan_labels, position_df, crop_pixels=crop_pixels)

images_dir = os.path.join(Path(__file__).resolve().parent, 'Images')
os.makedirs(images_dir, exist_ok=True)

z_index = int(np.round(z_depth_mm / (z_pixel_size * 1000)))
z_index = np.clip(z_index, 0, group_composite.shape[0] - 1)

horizontal_slice = group_composite[z_index, :, :]

plt.figure(figsize=(6, 6), frameon=False)
plt.imshow(horizontal_slice.T, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
plt.axis('off')

save_path = os.path.join(images_dir, save_name)
plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)
plt.show()