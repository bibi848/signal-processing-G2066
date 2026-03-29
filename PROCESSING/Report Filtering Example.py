#%%
from pathlib import Path
import sys
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import h5py

root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

from Classes.Filter import filter_signal

def remove_spikes(signal, threshold=500, verbose=False):
    signal = signal.copy()
    for i in range(1, len(signal)):
        if signal[i] > threshold:
            signal[i] = signal[i-1]
            if verbose:
                print('Signal exceeds threshold')
    return signal


raw_data_type       = '1D Raw Data'
raw_data_name       = 'Al Hole 15MHz 26012026'
processed_data_type = '1D Processed Data'

display_picture     = True
save_picture        = True  
all_pictures        = False  
filter_data         = False

crop_data           = True
crop_initial_amount = 100  
crop_latter_amount  = 800

# Filtering Parameters
filter_alpha   = 1.0
MHz_percentage = 1.5 
hanning_bool   = False

IN_DIR  = os.path.join(root_path, 'DATA', raw_data_type, raw_data_name)
if filter_data:
    OUT_DIR = os.path.join(root_path, 'DATA', processed_data_type, (raw_data_name + ' Filtered'))
else:
    OUT_DIR = os.path.join(root_path, 'DATA', processed_data_type, raw_data_name)

os.makedirs(OUT_DIR, exist_ok=True)

# Find all .mat files
mat_files = [
    f for f in os.listdir(IN_DIR)
    if f.lower().endswith(".mat")
    and os.path.isfile(os.path.join(IN_DIR, f))
]
mat_files = np.sort(mat_files)

print(f'Found {len(mat_files)} files in directory.')

#%%
# main

for file in mat_files:
    print(f'Processing {file}')
    file_path = os.path.join(IN_DIR, file)

    base_name = os.path.splitext(file)[0]
    if filter_data:
        base_name += "_filtered"

    file_out_dir = os.path.join(OUT_DIR, base_name)
    os.makedirs(file_out_dir, exist_ok=True)

    with h5py.File(file_path, "r") as f:
        centre_freq = f["exp_data/array/centre_freq"][()][0][0]
        
        el_x1 = np.array(f["exp_data/array/el_x1"]).flatten()
        tx = np.array(f["exp_data/tx"]).flatten().astype(int)
        rx = np.array(f["exp_data/rx"]).flatten().astype(int)
        time = np.array(f["exp_data/time"])[0]
        time_data = np.array(f["exp_data/time_data"])

    if crop_data:
        print(f'Cropping: Removing first {crop_initial_amount} samples and keeping up to sample {crop_latter_amount}.')
        time_data = time_data[:, crop_initial_amount:crop_latter_amount]
        time      = time[crop_initial_amount:crop_latter_amount]

    # Clean signal spikes
    time_data = np.apply_along_axis(remove_spikes, axis=1, arr=time_data, threshold=500)

    # Filtering Logic
    if filter_data:
        MHz_spacing = (centre_freq/1e6) * MHz_percentage
        f_start = (centre_freq/1e6) - MHz_spacing
        f_end   = (centre_freq/1e6) + MHz_spacing
        dt      = time[1] - time[0]

        print(f"Filtering signals ({f_start:.2f}MHz to {f_end:.2f}MHz)")
        time_data = np.apply_along_axis(
            filter_signal, 
            axis=1, 
            arr=time_data, 
            dt=dt, 
            f_start=f_start, 
            f_end=f_end, 
            filter_alpha=filter_alpha,
            hanning_bool=hanning_bool
            )
        
        time_data = time_data.astype(np.float32)

    if display_picture:
        print('Intensity')
        n_el = len(el_x1)
        n_t  = time_data.shape[1]
        fmc = np.zeros((n_el, n_el, n_t))

        for t in range(1, n_el + 1):
            mask = tx == t
            data_t = time_data[mask, :]
            rx_t   = rx[mask]
            order = np.argsort(rx_t)
            fmc[t-1, :, :] = data_t[order, :]

        img = np.sqrt(np.mean(fmc**2, axis=0))

        plt.figure(figsize=(10, 6))
        plt.imshow(
            img,
            aspect="auto",
            extent=[time[0], time[-1], n_el, 1],
            cmap="viridis"
        )
        plt.axis('off')
        
        if save_picture:
            out_name = base_name + ".png"
            save_path = os.path.join(file_out_dir, out_name)
            plt.savefig((f'Images/{out_name}'), dpi=300, bbox_inches='tight', pad_inches=0)
            print(f"Image saved to: {out_name}")
        
        plt.show()

    if not all_pictures:
        break

print("\nProcessing complete.")