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

# Point the script to the correct subfolder.
raw_data_type       = '1D Raw Data'
raw_data_name       = 'Al Pure 15MHz 26012026'
processed_data_type = '1D Processed Data'
cwd                 = os.getcwd()
display_picture     = 'y' # y/n

# Input and Output paths.
IN_DIR  = os.path.join(cwd, raw_data_type, raw_data_name)
OUT_DIR = os.path.join(cwd, processed_data_type, raw_data_name)

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

    # Write Excel
    excel_name = os.path.splitext(file)[0] + ".xlsx"
    excel_path = os.path.join(OUT_DIR, excel_name)

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        metadata_df.to_excel(writer, sheet_name="Metadata", index=False)
        time_data_df.to_excel(writer, sheet_name="Time_Data", index=False)
        time_df.to_excel(writer, sheet_name="Time", index=False)
        tx_rx_df.to_excel(writer, sheet_name="tx_rx", index=False)
        geometry_df.to_excel(writer, sheet_name="Array_Geometry", index=False)

    print(f"{excel_name} done")

    if display_picture == 'y':
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
        out_name = os.path.splitext(file)[0] + ".png"
        plt.savefig(os.path.join(OUT_DIR, out_name), dpi=300, bbox_inches='tight')
        plt.show()

    print()
#%%