"""
Batch GPU TFM imaging using the TFM_U pipeline (tfm_ultra module).
"""
# %%
import os
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import hilbert

root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

input_data_folder = '1D Processed Data'
input_data_subfolder = 'Al Pure 10MHz 18032026'
output_data_folder = '1D TFM Data'
cwd = os.getcwd()

display_picture = False
save_picture = False
filtered_data = True
angular_filter = False

img_output = 'real'  # real, complex, envelope, db

threads = 512
batch_size = 5

vmax = 0.0
vmin = -20.0

c = 6300
z_max = 10e-3
z_min = 40e-3
x_min = 'xc_min'
x_max = 'xc_max'
x_pixels = 400
z_pixels = 400
cmap = 'viridis'

real_aspect_ratio = False
z_aspect = 8
x_aspect = 8

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

if filtered_data and not angular_filter:
    IN_DIR = os.path.join(ROOT_DIR, 'DATA', input_data_folder, input_data_subfolder + ' Filtered')
    OUT_DIR = os.path.join(ROOT_DIR, 'DATA', output_data_folder, input_data_subfolder + ' Filtered')
elif filtered_data and angular_filter:
    IN_DIR = os.path.join(ROOT_DIR, 'DATA', input_data_folder, input_data_subfolder + ' Filtered')
    OUT_DIR = os.path.join(ROOT_DIR, 'DATA', output_data_folder, input_data_subfolder + ' Angular')
else:
    IN_DIR = os.path.join(ROOT_DIR, 'DATA', input_data_folder, input_data_subfolder)
    OUT_DIR = os.path.join(ROOT_DIR, 'DATA', output_data_folder, input_data_subfolder)

os.makedirs(OUT_DIR, exist_ok=True)

build_dir = os.path.join(os.path.dirname(__file__), '..', 'build', 'CPP', 'TFM_U')
sys.path.insert(0, os.path.normpath(build_dir))

import tfm_ultra

print('tfm_ultra successful')
print(f'batch_size = {batch_size}  and  threads = {threads}')
print()

image_folders = sorted([
    f for f in os.listdir(IN_DIR)
    if os.path.isdir(os.path.join(IN_DIR, f))
])
image_folders = np.sort(image_folders)
image_folders = [x for x in image_folders if 'Speed of Sound' not in x]
if '2D' in input_data_folder:
    image_folders = [x for x in image_folders if '1D' in x]

print(f'Found {len(image_folders)} image folders')


def postprocess(raw: np.ndarray, mode: str):
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
    return raw

def display_image(img: np.ndarray, x_img: np.ndarray, z_img: np.ndarray, title: str, xa: int, za: int):
    img_disp = np.abs(img) if np.iscomplexobj(img) else img
    plt.figure(figsize=(xa, za))
    kwargs = dict(
        extent=[x_img[0] * 1e3, x_img[-1] * 1e3, z_img[-1] * 1e3, z_img[0] * 1e3],
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

def load_reference_geometry(ref_folder: str):
    file_path = os.path.join(IN_DIR, ref_folder)

    metadata = pd.read_csv(os.path.join(file_path, 'metadata.csv'))
    time_sec = pd.read_csv(os.path.join(file_path, 'time.csv'))['time_seconds'].values.astype(np.float64)
    tx_rx = pd.read_csv(os.path.join(file_path, 'tx_rx.csv'))
    geometry = pd.read_csv(os.path.join(file_path, 'array_geometry.csv'))

    tx_vals = tx_rx['tx'].values.astype(np.int32) - 1
    rx_vals = tx_rx['rx'].values.astype(np.int32) - 1
    xc_vals = geometry['el_xc'].values.astype(np.float64)
    zc_vals = geometry['el_zc'].values.astype(np.float64)

    with h5py.File(os.path.join(file_path, 'time_data.h5'), 'r') as h5f:
        dset = h5f['time_data']
        if dset.ndim != 2:
            raise ValueError(f'{ref_folder}: expected 2D time_data, got shape {dset.shape}')
        Nf, Nt = map(int, dset.shape)

    x_lo = xc_vals.min() if x_min == 'xc_min' else x_min
    x_hi = xc_vals.max() if x_max == 'xc_max' else x_max

    x_img = np.linspace(x_lo, x_hi, x_pixels, dtype=np.float64)
    z_img = np.linspace(z_max, z_min, z_pixels, dtype=np.float64)
    X, Z = np.meshgrid(x_img, z_img)

    local_x_aspect = x_aspect
    if real_aspect_ratio:
        local_x_aspect = int(np.ceil(((x_hi - x_lo) / (z_min - z_max)) * z_aspect))

    return {
        'metadata': metadata,
        'time': np.ascontiguousarray(time_sec),
        'tx': np.ascontiguousarray(tx_vals),
        'rx': np.ascontiguousarray(rx_vals),
        'xc': np.ascontiguousarray(xc_vals),
        'zc': np.ascontiguousarray(zc_vals),
        'X': np.ascontiguousarray(X, dtype=np.float64),
        'Z': np.ascontiguousarray(Z, dtype=np.float64),
        'x_img': x_img,
        'z_img': z_img,
        'x_aspect': local_x_aspect,
        'Nf': Nf,
        'Nt': Nt,
        'Nelem': xc_vals.shape[0],
        'time_shape': time_sec.shape,
        'tx_shape': tx_vals.shape,
        'rx_shape': rx_vals.shape,
        'xc_shape': xc_vals.shape,
        'zc_shape': zc_vals.shape,
    }

def validate_folder_against_reference(folder: str, ref: dict) -> None:
    file_path = os.path.join(IN_DIR, folder)

    tx_rx = pd.read_csv(os.path.join(file_path, 'tx_rx.csv'))
    geometry = pd.read_csv(os.path.join(file_path, 'array_geometry.csv'))
    time_sec = pd.read_csv(os.path.join(file_path, 'time.csv'))['time_seconds'].values

    tx_vals = tx_rx['tx'].values.astype(np.int32) - 1
    rx_vals = tx_rx['rx'].values.astype(np.int32) - 1
    xc_vals = geometry['el_xc'].values.astype(np.float64)
    zc_vals = geometry['el_zc'].values.astype(np.float64)

    if time_sec.shape != ref['time_shape']:
        raise ValueError(f'{folder}: time axis shape mismatch {time_sec.shape} != {ref["time_shape"]}')
    if tx_vals.shape != ref['tx_shape']:
        raise ValueError(f'{folder}: tx shape mismatch {tx_vals.shape} != {ref["tx_shape"]}')
    if rx_vals.shape != ref['rx_shape']:
        raise ValueError(f'{folder}: rx shape mismatch {rx_vals.shape} != {ref["rx_shape"]}')
    if xc_vals.shape != ref['xc_shape'] or zc_vals.shape != ref['zc_shape']:
        raise ValueError(f'{folder}: geometry size mismatch with reference folder')

    if not np.array_equal(tx_vals, ref['tx']):
        raise ValueError(f'{folder}: tx values differ from reference; cannot reuse prepared GPU geometry')
    if not np.array_equal(rx_vals, ref['rx']):
        raise ValueError(f'{folder}: rx values differ from reference; cannot reuse prepared GPU geometry')
    if not np.allclose(time_sec, ref['time']):
        raise ValueError(f'{folder}: time axis differs from reference; cannot reuse prepared GPU geometry')
    if not np.allclose(xc_vals, ref['xc']) or not np.allclose(zc_vals, ref['zc']):
        raise ValueError(f'{folder}: element geometry differs from reference; cannot reuse prepared GPU geometry')

    with h5py.File(os.path.join(file_path, 'time_data.h5'), 'r') as h5f:
        dset = h5f['time_data']
        if tuple(dset.shape) != (ref['Nf'], ref['Nt']):
            raise ValueError(
                f'{folder}: time_data shape mismatch {tuple(dset.shape)} != {(ref["Nf"], ref["Nt"])}'
            )

def load_batch_into_buffer(buffer: np.ndarray, folders: list[str], ref: dict, validate_geometry: bool = False) -> tuple[list[dict], float]:
    """Fill the provided host buffer directly from HDF5 datasets."""
    io_start = time.perf_counter()
    batch_meta: list[dict] = []

    for i, fol in enumerate(folders):
        file_path = os.path.join(IN_DIR, fol)

        if validate_geometry:
            validate_folder_against_reference(fol, ref)

        with h5py.File(os.path.join(file_path, 'time_data.h5'), 'r') as h5f:
            dset = h5f['time_data']
            if tuple(dset.shape) != (ref['Nf'], ref['Nt']):
                raise ValueError(
                    f'{fol}: time_data shape mismatch {tuple(dset.shape)} != {(ref["Nf"], ref["Nt"])}'
                )
            dset.read_direct(buffer[i])

        batch_meta.append({
            'fol': fol,
            'x_img': ref['x_img'],
            'z_img': ref['z_img'],
            'x_aspect': ref['x_aspect'],
        })

    io_end = time.perf_counter()
    return batch_meta, (io_end - io_start)


# %%
# Main loop
full_start = time.perf_counter()

shared_geom = load_reference_geometry(image_folders[0])
Np = shared_geom['X'].size
print(f"Reference dataset: Nf={shared_geom['Nf']}, Nt={shared_geom['Nt']}, Np={Np}")

prepare_start = time.perf_counter()
tfm_ultra.prepare_tfm1D_GPU(
    shared_geom['time'],
    shared_geom['tx'],
    shared_geom['rx'],
    shared_geom['xc'],
    shared_geom['zc'],
    shared_geom['X'],
    shared_geom['Z'],
    c,
    batch_size,
    threads,
)
prepare_end = time.perf_counter()
print(f'GPU prepare/cache: {prepare_end - prepare_start:.6f} s')
print()

total_folders = len(image_folders)
all_batches = [image_folders[i:i + batch_size] for i in range(0, total_folders, batch_size)]

host_batch_a = np.empty((batch_size, shared_geom['Nf'], shared_geom['Nt']), dtype=np.float64)
host_batch_b = np.empty((batch_size, shared_geom['Nf'], shared_geom['Nt']), dtype=np.float64)

validate_next_batch = True

try:
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            load_batch_into_buffer,
            host_batch_a,
            all_batches[0],
            shared_geom,
            False,
        )

        for batch_idx, batch_folders in enumerate(all_batches):
            n = len(batch_folders)
            batch_number = batch_idx + 1
            first_image_num = batch_idx * batch_size + 1
            last_image_num = first_image_num + n - 1
            print(f'Batch {batch_number}, images {first_image_num} to {last_image_num} of {total_folders}')

            batch_meta, io_time = future.result()
            current_buffer = host_batch_a if (batch_idx % 2 == 0) else host_batch_b
            print(f'I/O load: {io_time:.6f} s for {n} files')

            if batch_idx + 1 < len(all_batches):
                next_buffer = host_batch_b if (batch_idx % 2 == 0) else host_batch_a
                future = executor.submit(
                    load_batch_into_buffer,
                    next_buffer,
                    all_batches[batch_idx + 1],
                    shared_geom,
                    validate_next_batch,
                )
                validate_next_batch = False

            gpu_start = time.perf_counter()
            raw_imgs = tfm_ultra.tfm1D_batch_GPU(current_buffer[:n])
            gpu_end = time.perf_counter()
            print(f'GPU batch: {gpu_end - gpu_start:.3f} s, {(gpu_end - gpu_start)/n:.6f} s per image')

            post_start = time.perf_counter()
            for raw, meta in zip(raw_imgs, batch_meta):
                img = postprocess(raw, img_output)

                if display_picture:
                    display_image(img, meta['x_img'], meta['z_img'], meta['fol'], meta['x_aspect'], z_aspect)

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
            print(f'Post processing: {post_end - post_start:.6f} s, {(post_end - post_start)/n:.6f}')
            print()
finally:
    tfm_ultra.clear_gpu_cache()

full_end = time.perf_counter()
elapsed = full_end - full_start
print(f'Processed {total_folders} images in {elapsed:.3f} s, {elapsed / max(total_folders, 1):.6f} s per image')