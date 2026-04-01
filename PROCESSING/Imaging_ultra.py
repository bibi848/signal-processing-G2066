"""
Batch GPU TFM imaging using the TFM_U pipeline (tfm_ultra module).
"""
#%%
# Imports
import os
import sys
import time
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import hilbert

root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

# Configuration─
input_data_folder    = '1D Processed Data'
input_data_subfolder = 'Al Pure 10MHz 18032026'
output_data_folder   = '1D TFM Data'
cwd                  = os.getcwd()

display_picture = False
save_picture    = False
filtered_data   = True
angular_filter  = False

img_output = 'real' # real, complex, envelope, dB

# GPU settings
threads    = 256 # threads per HIP block
batch_size = 10  # images per GPU batch

vmax = 0.0
vmin = -20.0

# Image grid
c        = 6300   # speed of sound m/s
z_max    = 10e-3  # m
z_min    = 40e-3  # m
x_min    = 'xc_min'
x_max    = 'xc_max'
x_pixels = 400
z_pixels = 400
cmap     = 'viridis'

# Aspect ratio
real_aspect_ratio = False
z_aspect = 8
x_aspect = 8

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

if filtered_data and not angular_filter:
    IN_DIR  = os.path.join(ROOT_DIR, 'DATA', input_data_folder,
                           input_data_subfolder + ' Filtered')
    OUT_DIR = os.path.join(ROOT_DIR, 'DATA', output_data_folder,
                           input_data_subfolder + ' Filtered')
elif filtered_data and angular_filter:
    IN_DIR  = os.path.join(ROOT_DIR, 'DATA', input_data_folder,
                           input_data_subfolder + ' Filtered')
    OUT_DIR = os.path.join(ROOT_DIR, 'DATA', output_data_folder,
                           input_data_subfolder + ' Angular')
else:
    IN_DIR  = os.path.join(ROOT_DIR, 'DATA', input_data_folder, input_data_subfolder)
    OUT_DIR = os.path.join(ROOT_DIR, 'DATA', output_data_folder, input_data_subfolder)

os.makedirs(OUT_DIR, exist_ok=True)

# tfm_ultra module
build_dir = os.path.join(os.path.dirname(__file__), '..', 'build', 'CPP', 'TFM_U')
sys.path.insert(0, os.path.normpath(build_dir))

import tfm_ultra
print('tfm_ultra successful')
print(f'batch_size = {batch_size}  and  threads = {threads}')
print()

# Image Folders
image_folders = sorted([
    f for f in os.listdir(IN_DIR)
    if os.path.isdir(os.path.join(IN_DIR, f))
])
if '2D' in input_data_folder:
    image_folders = [x for x in image_folders if '1D' in x]

print(f'Found {len(image_folders)} image folders')
#%%

# Post processing
def postprocess(raw, mode):
    if mode == 'real':
        return raw
    analytic = hilbert(raw, axis=0)
    if mode == 'complex':
        return analytic
    envelope = np.abs(analytic)
    if mode == 'envelope':
        return envelope
    if mode == 'db':
        return 20.0 * np.log10(envelope / (envelope.max() + 1e-10) + 1e-10)


# Display one image
def display_image(img, x_img, z_img, title, xa, za):
    img_disp = np.abs(img) if np.iscomplexobj(img) else img
    plt.figure(figsize=(xa, za))
    kwargs = dict(
        extent=[x_img[0]*1e3, x_img[-1]*1e3, z_img[-1]*1e3, z_img[0]*1e3],
        aspect='auto',
        cmap=cmap,
    )
    if img_output == 'db':
        kwargs.update(vmin=vmin, vmax=vmax)
    plt.imshow(img_disp, **kwargs)
    plt.xlabel('x [mm]')
    plt.ylabel('z [mm]')
    plt.colorbar(label='Amplitude')
    plt.title(title)
    plt.tight_layout()
    plt.show()

#%%
# Main loop
full_start = time.perf_counter()
total_folders = len(image_folders)

# 1. Read all time_data arrays from disk (CPU)
# 2. Submit the whole batch to tfm_ultra in one call (GPU)
# 3. Post-process and optionally display / save results (CPU)

for batch_start in range(0, total_folders, batch_size):
    batch_folders = image_folders[batch_start : batch_start + batch_size]
    n = len(batch_folders)
    print(f'Batch {batch_start // batch_size + 1} '
          f'images {batch_start + 1}–{batch_start + n} of {total_folders}')

    # Read data
    io_start = time.perf_counter()

    batch_time_data = []
    batch_meta      = []

    # Geometry is shared
    shared_geom = None

    for fol in batch_folders:
        file_path = os.path.join(IN_DIR, fol)

        metadata = pd.read_csv(os.path.join(file_path, 'metadata.csv'))
        time_sec = pd.read_csv(
            os.path.join(file_path, 'time.csv'))['time_seconds'].values
        tx_rx    = pd.read_csv(os.path.join(file_path, 'tx_rx.csv'))
        geometry = pd.read_csv(os.path.join(file_path, 'array_geometry.csv'))

        with h5py.File(os.path.join(file_path, 'time_data.h5'), 'r') as h5f:
            td = h5f['time_data'][:]

        batch_time_data.append(td.astype(np.float64, copy=False))

        tx_vals = tx_rx['tx'].values.astype(np.int32) - 1
        rx_vals = tx_rx['rx'].values.astype(np.int32) - 1

        xc_vals = geometry['el_xc'].values.astype(np.float64)
        zc_vals = geometry['el_zc'].values.astype(np.float64)

        _x_min = xc_vals.min() if x_min == 'xc_min' else x_min
        _x_max = xc_vals.max() if x_max == 'xc_max' else x_max

        x_img = np.linspace(_x_min, _x_max, x_pixels)
        z_img = np.linspace(z_max,  z_min,  z_pixels)

        _xa = x_aspect
        if real_aspect_ratio:
            _xa = int(np.ceil(((_x_max - _x_min) / (z_min - z_max)) * z_aspect))

        if shared_geom is None:
            X, Z = np.meshgrid(x_img, z_img)
            shared_geom = dict(
                time=time_sec,
                tx=tx_vals, rx=rx_vals,
                xc=xc_vals, zc=zc_vals,
                X=X, Z=Z,
                x_img=x_img, z_img=z_img,
            )
        else:
            assert tx_vals.shape == shared_geom['tx'].shape, \
                f"tx shape mismatch in {fol}: expected {shared_geom['tx'].shape}"

        batch_meta.append(dict(
            fol=fol,
            metadata=metadata,
            x_img=x_img, z_img=z_img,
            x_aspect=_xa,
        ))

    io_end = time.perf_counter()
    print(f'I/O load: {io_end - io_start:.3f} s for {n} files')

    # GPU batch call
    gpu_start = time.perf_counter()

    raw_imgs = tfm_ultra.tfm1D_batch_GPU(
        batch_time_data,
        shared_geom['time'],
        shared_geom['tx'],
        shared_geom['rx'],
        shared_geom['xc'],
        shared_geom['zc'],
        shared_geom['X'],
        shared_geom['Z'],
        c,
        threads,
    )

    gpu_end = time.perf_counter()
    print(f'GPU batch:  {gpu_end - gpu_start:.3f} s  '
          f'({(gpu_end - gpu_start)/n*1e3:.1f} ms per image)')

    # Post-process, display, save
    post_start = time.perf_counter()

    for idx_in_batch, (raw, meta) in enumerate(zip(raw_imgs, batch_meta)):
        img = postprocess(raw, img_output)

        if display_picture:
            display_image(
                img,
                meta['x_img'], meta['z_img'],
                meta['fol'],
                meta['x_aspect'], z_aspect,
            )

        if save_picture:
            if img_output == 'complex':
                np.save(os.path.join(OUT_DIR, meta['fol'] + '_TFM.npy'), img)
            else:
                img_save = np.abs(img) if np.iscomplexobj(img) else img
                plt.imsave(
                    os.path.join(OUT_DIR, meta['fol'] + '_TFM.png'),
                    img_save,
                    cmap=cmap,
                    vmin=vmin if img_output == 'db' else None,
                    vmax=vmax if img_output == 'db' else None,
                )

    post_end = time.perf_counter()
    print(f'Post processing: {post_end - post_start:.3f} s')
    print()

full_end = time.perf_counter()
elapsed  = full_end - full_start
print(f'Processed {total_folders} images in {elapsed:.3f} s '
      f'{elapsed / max(total_folders, 1) * 1e3:.1f} ms per image')
