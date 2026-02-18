'''
This script converts the .mat files collected from using the ultrasonic scanning array to .csv files. 
'''

#%%
# Importing Functions and Defining Correct Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import h5py
import os

from Classes.Filter import filter_signal

# Point the script to the correct subfolder.
raw_data_type       = '1D Raw Data'
raw_data_name       = 'Cu Pure 15MHz 17022026'
processed_data_type = '1D Processed Data'
cwd                 = os.getcwd()
display_picture     = True
save_picture        = False
all_pictures        = False
filter_data         = True
crop_data           = False
crop_amount         = 1200

# Filtering Parameters
filter_alpha = 0.9
MHz_percentage  = 0.1 # percentage
hanning_bool = False

# Input and Output paths.
IN_DIR  = os.path.join(cwd, 'DATA', raw_data_type, raw_data_name)
if filter_data:
    OUT_DIR = os.path.join(cwd, 'DATA', processed_data_type, (raw_data_name + ' Filtered'))
else:
    OUT_DIR = os.path.join(cwd, 'DATA', processed_data_type, raw_data_name)
os.makedirs(OUT_DIR, exist_ok=True)

# Find all files in directory which are .mat files. 
mat_files = [
    f for f in os.listdir(IN_DIR)
    if f.lower().endswith(".mat")
    and os.path.isfile(os.path.join(IN_DIR, f))
]

print('Files available in directory:')
print(mat_files)
print()

#%%
# Extracting all data from .mat files. 

for file in mat_files:
    print('Processing', file)
    file_path = os.path.join(IN_DIR, file)

    base_name = os.path.splitext(file)[0]
    if filter_data:
        base_name += "_filtered"

    file_out_dir = os.path.join(OUT_DIR, base_name)
    os.makedirs(file_out_dir, exist_ok=True)

    with h5py.File(file_path, "r") as f:
        centre_freq = f["exp_data/array/centre_freq"][()][0][0]

        if "manufacturer" in f["exp_data/array"]:
            manufacturer_raw = np.array(f["exp_data/array/manufacturer"])
            manufacturer = ''.join(chr(c) for c in manufacturer_raw.flatten())
        else:
            manufacturer = "UNKNOWN"

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

    if crop_data:
        time_data = time_data[:, :crop_amount]
        time      = time[:crop_amount]

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

    if filter_data:
        MHz_spacing = (centre_freq/1e6) * MHz_percentage
        f_start = (centre_freq/1e6) - MHz_spacing
        f_end   = (centre_freq/1e6) + MHz_spacing
        dt      = time[1] - time[0]

        print("Filtering Signals")
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

        # Compressing
        print('Compressing Signals')
        time_data = time_data = time_data.astype(np.float32)

    # Write CSV file
    print("Writing CSV files")

    metadata_df.to_csv(
        os.path.join(file_out_dir, "metadata.csv"),
        index=False
    )

    time_df.to_csv(
        os.path.join(file_out_dir, "time.csv"),
        index=False,
        float_format="%.10g"
    )

    tx_rx_df.to_csv(
        os.path.join(file_out_dir, "tx_rx.csv"),
        index=False
    )

    geometry_df.to_csv(
        os.path.join(file_out_dir, "array_geometry.csv"),
        index=False,
        float_format="%.10g"
    )

    h5_path = os.path.join(file_out_dir, "time_data.h5")

    with h5py.File(h5_path, "w") as h5f:
        dset = h5f.create_dataset(
            "time_data",
            data=time_data,
            dtype="float32",
            compression="gzip",
            compression_opts=4,
            chunks=(1, time_data.shape[1])
        )
    
        # Metadata as attributes
        dset.attrs["centre_frequency_Hz"] = centre_freq
        dset.attrs["dt"] = time[1] - time[0]
        dset.attrs["filtered"] = filter_data

    print(f"{base_name} done")

    if display_picture:
        print('Displaying transmit/receive data')

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

        plt.figure(figsize=(8, 5))
        plt.imshow(
            img,
            aspect="auto",
            extent=[time[0], time[-1], n_el, 1],
            cmap="viridis"
        )
        plt.title(f"Data from {file}")
        plt.xlabel("Time [s]")
        plt.ylabel("Receiver Number")
        plt.colorbar(label="Intensity")
        plt.tight_layout()
        
        if save_picture:
            out_name = os.path.splitext(file)[0] + ".png"
            plt.savefig(os.path.join(file_out_dir, out_name), dpi=300, bbox_inches='tight')
        
        plt.show()

    if not all_pictures:
        break
    print()
#%%