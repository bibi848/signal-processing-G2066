'''
This script converts the .xlsx files processed previously into TFM images. 
'''

#%%
# Importing Functions and Defining Correct Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

from Classes.TFM1D import TFM1D

# Point the script to the correct subfolder.
input_data_folder    = '1D Processed Data'
input_data_subfolder = 'Al Hole 5MHz 26012026'
output_data_folder   = '1D TFM Data'
cwd                  = os.getcwd()
display_picture      = 'y' # y/n

# Image Parameters
c = 6320 # m/s

# Input and Output paths.
IN_DIR  = os.path.join(cwd, input_data_folder, input_data_subfolder)
OUT_DIR = os.path.join(cwd, output_data_folder, input_data_subfolder)

# Find all files in directory which are .xlsx files. 
xlsx_files = [
    f for f in os.listdir(IN_DIR)
    if f.lower().endswith(".xlsx")
    and os.path.isfile(os.path.join(IN_DIR, f))
]

print('Files available in directory:')
print(xlsx_files)
print()

#%%
# Looping over available files
for file in xlsx_files:
    print('Processing', file)

    file_path = os.path.join(IN_DIR, file)

    # Extract Data
    metadata  = pd.read_excel(file_path, "Metadata")
    time_data = pd.read_excel(file_path, "Time_Data").values
    time      = pd.read_excel(file_path, "Time")["time_seconds"].values
    tx_rx     = pd.read_excel(file_path, "tx_rx")
    geometry  = pd.read_excel(file_path, "Array_Geometry")

    tx = tx_rx["tx"].values.astype(int)
    rx = tx_rx["rx"].values.astype(int)

    xc = geometry["el_xc"].values
    zc = geometry["el_zc"].values

    x_img = np.linspace(xc.min(), xc.max(), 200)
    z_img = np.linspace(0e-3, 40e-3, 300)

    # TFM
    img = TFM1D(time_data, time, tx, rx, xc, zc, c, x_img, z_img)

    if display_picture == 'y':
        plt.figure(figsize=(6, 8))
        plt.imshow(
            img,
            extent=[x_img[0]*1e3, x_img[-1]*1e3, z_img[-1]*1e3, z_img[0]*1e3],
            aspect="auto",
            cmap="hot"
        )
        plt.xlabel("x [mm]")
        plt.ylabel("z [mm]")
        plt.colorbar(label="Amplitude")
        plt.title(file)
        plt.tight_layout()
        out_name = os.path.splitext(file)[0] + "_TFM.png"
        plt.savefig(os.path.join(OUT_DIR, out_name), dpi=300, bbox_inches='tight')
        plt.show()
    
    # Save Clean File
    out_name = os.path.splitext(file)[0] + "_TFM_clean.png"
    plt.imsave(
        os.path.join(OUT_DIR, out_name),
        img,
        cmap="hot"
    )

#%%