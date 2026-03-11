'''
This script shows how the filtering is being done on the data. 
'''

#%%
# Importing Functions and Defining Correct Path
from scipy.signal import hilbert
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import numpy as np
import h5py
import sys
import os
plt.rcParams['font.size'] = 13

root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

from scipy.signal.windows import tukey
from Classes.Filter import filter_signal

# Point the script to the correct subfolder.
raw_data_type       = '1D Raw Data'
raw_data_name       = 'Al Pure 10MHz 17022026'
processed_data_type = '1D Processed Data'
cwd                 = Path.cwd().parent
file                = 'Al_70_1_1.mat'

# Filtering Parameters
filter_alpha = 0.9
MHz_percentage  = 0.3 # percentage
hanning_bool = False

# Input and Output paths.
IN_DIR  = os.path.join(cwd, 'DATA', raw_data_type, raw_data_name)
OUT_DIR = os.path.join(cwd, 'DATA', processed_data_type, raw_data_name)

# Find all files in directory which are .mat files. 
mat_files = [
    f for f in os.listdir(IN_DIR)
    if f.lower().endswith(".mat")
    and os.path.isfile(os.path.join(IN_DIR, f))
]
file_path = os.path.join(IN_DIR, file)

#%%
# Extract raw data

with h5py.File(file_path, "r") as f:
    centre_freq = f["exp_data/array/centre_freq"][()][0][0]
    manufacturer_raw = np.array(f["exp_data/array/manufacturer"])
    el_x1 = np.array(f["exp_data/array/el_x1"]).flatten()
    el_x2 = np.array(f["exp_data/array/el_x2"]).flatten()
    el_xc = np.array(f["exp_data/array/el_xc"]).flatten()
    el_y1 = np.array(f["exp_data/array/el_y1"]).flatten()
    el_y2 = np.array(f["exp_data/array/el_y2"]).flatten()
    el_yc = np.array(f["exp_data/array/el_yc"]).flatten()
    el_z1 = np.array(f["exp_data/array/el_z1"]).flatten()
    el_z2 = np.array(f["exp_data/array/el_z2"]).flatten()
    el_zc = np.array(f["exp_data/array/el_zc"]).flatten()
    tx = np.array(f["exp_data/tx"]).flatten().astype(int)
    rx = np.array(f["exp_data/rx"]).flatten().astype(int)
    time = np.array(f["exp_data/time"])[0]
    time_data = np.array(f["exp_data/time_data"])
manufacturer = ''.join(chr(c) for c in manufacturer_raw.flatten())

# Metadata
metadata_df = pd.DataFrame({
    "Field": [
        "origin",
        "centre_frequency_Hz",
        "manufacturer",
        "number_of_elements",
        "number_of_fmc_traces",
        "number_of_time_samples"
    ],
    "Value": [
        file,
        centre_freq,
        manufacturer,
        len(el_x1),
        time_data.shape[0],
        time_data.shape[1]
    ]
})

# Time Data
time_data_df = pd.DataFrame(time_data)

# Time
time_df = pd.DataFrame({
    "time_seconds": time
})

# tx / rx 
tx_rx_df = pd.DataFrame({
    "tx": tx,
    "rx": rx
})

# Geometry
geometry_df = pd.DataFrame({
    "el_x1": el_x1,
    "el_x2": el_x2,
    "el_xc": el_xc,
    "el_y1": el_y1,
    "el_y2": el_y2,
    "el_yc": el_yc,
    "el_z1": el_z1,
    "el_z2": el_z2,
    "el_zc": el_zc
})

#%%
# Example Raw Signal Collected
signal = time_data[8]
time   = time

signal = signal - np.mean(signal)
dt = time[1] - time[0]

plt.figure()
plt.plot(time, signal)
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.title("Time-Domain Signal")
plt.grid(True)
plt.tight_layout()
plt.show()

#%%
# Viewing data in frequency-domain
N        = len(signal)
# window   = np.hanning(N)
# signal_w = signal * window
# fft_vals = np.fft.rfft(signal_w)
fft_vals = np.fft.rfft(signal)
freqs    = np.fft.rfftfreq(N, dt) / 1e6
magnitude = np.abs(fft_vals)

plt.figure()
plt.plot(freqs, magnitude)
plt.xlabel("Frequency [MHz]")
plt.ylabel("Magnitude")
plt.title("Frequency-Domain Spectrum")
plt.grid(True)
plt.tight_layout()
plt.show()

#%%
# Time and Frequency Domains next to each other
signal = signal - np.mean(signal)
dt = time[1] - time[0]

# FFT
N = len(signal)
fft_vals = np.fft.rfft(signal)
freqs = np.fft.rfftfreq(N, dt) / 1e6
magnitude = np.abs(fft_vals)

# Plot time and frequency domain
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Time-domain
ax[0].plot(time, signal, c='b')
ax[0].set_xlabel("Time [s]")
ax[0].set_ylabel("Amplitude")
ax[0].set_title("Time Domain")
ax[0].grid(True)

# Frequency-domain
ax[1].plot(freqs, magnitude, c='r')
ax[1].set_xlabel("Frequency [MHz]")
ax[1].set_ylabel("Magnitude")
ax[1].set_title("Frequency Domain")
ax[1].grid(True)

plt.tight_layout()
plt.savefig("Images/time_frequency_signal.png", dpi=300, bbox_inches="tight")
plt.show()

#%%
# Filtering
MHz_spacing = (centre_freq/1e6) * MHz_percentage
f_start = (centre_freq/1e6) - MHz_spacing
f_end   = (centre_freq/1e6) + MHz_spacing

# Masking
mask = (freqs >= f_start) & (freqs <= f_end)
indices = np.where(mask)[0]

if len(indices) > 0:
    # Tukey is a tapered cosine filter
    filter_window = tukey(len(indices), alpha=filter_alpha)

    bp_filter = np.zeros_like(freqs)
    bp_filter[indices] = filter_window

    fft_filtered = fft_vals * bp_filter
    filtered_signal = np.fft.irfft(fft_filtered, n=N)

plt.figure()
plt.plot(freqs, magnitude / np.max(magnitude), label="Original")
plt.plot(freqs, bp_filter, label="Tukey Filter Mask", linestyle='--')
plt.fill_between(freqs, bp_filter, alpha=0.2, color='orange')
plt.xlabel("Frequency [MHz]")
plt.title(f"Tapered Cosine Filter ({f_start}-{f_end} MHz)")
plt.legend()
plt.show()

#%%
# Raw vs Filtered Signal
plt.figure(figsize=(8, 5))
plt.plot(time, signal, label="Original", alpha=0.5, color='gray')
plt.plot(time, filtered_signal, label="Filtered", color='blue')
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.title(f"Raw vs Filtered Signal ({f_start}-{f_end} MHz)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#%%
filtered_signal = filter_signal(signal, dt, f_start, f_end, filter_alpha, hanning_bool)

plt.figure(figsize=(8, 5))
plt.plot(time, signal, label="Original", alpha=0.5, color='gray')
plt.plot(time, filtered_signal, label="Filtered", color='blue')
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.title(f"Raw vs Filtered Signal ({f_start}-{f_end} MHz)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#%%
# FFT of filtered signal
fft_filtered = np.fft.rfft(filtered_signal)
magnitude_filtered = np.abs(fft_filtered)

fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Time-domain overlay
ax[0].plot(time, signal, label="Original", alpha=0.7, color='gray')
ax[0].plot(time, filtered_signal, label="Filtered", color='blue')
ax[0].set_xlabel("Time [s]")
ax[0].set_ylabel("Amplitude")
ax[0].set_title("Time Domain")
ax[0].grid(True)
ax[0].legend(loc='upper right')

# Frequency-domain overlay
ax[1].plot(freqs, magnitude / np.max(magnitude), c='r', label="Original")
ax[1].plot(freqs, bp_filter, label="Filter", linestyle='--', c='g')
ax[1].fill_between(freqs, bp_filter, alpha=0.2, color='g')
ax[1].set_xlabel("Frequency [MHz]")
ax[1].set_ylabel("Magnitude")
ax[1].set_title("Frequency Domain")
ax[1].grid(True)
ax[1].legend(loc='upper right')

plt.tight_layout()
plt.savefig("Images/filtered_time_frequency_signal.png", dpi=300, bbox_inches="tight")
plt.show()
# %%

#%%
# Hilbert Transform
plt.rcParams['font.size'] = 16
analytic_signal = hilbert(filtered_signal)
envelope = np.abs(analytic_signal)
crop = 300
plt.figure(figsize=(10,6))
plt.plot(time[:crop], filtered_signal[:crop], label="Filtered Signal", color="blue", alpha=0.9)
plt.plot(time[:crop], envelope[:crop], label="Hilbert Envelope", color="red", linewidth=2)

plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.legend(loc='upper right')
plt.grid(True)
plt.tight_layout()

plt.savefig("Images/hilbert_envelope_signal.png", dpi=300, bbox_inches="tight")
plt.show()

#%%
