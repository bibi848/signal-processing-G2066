'''
This script allows you to view the .npy files collected from 2D array scanning. 
It uses the library napari for viewing. 

pip install napari
pip install -U napari[pyqt5]
python Display3DData.py

In your terminal (one by one) worked for me. 
Linux: export QT_QPA_PLATFORM=xcb
'''

import numpy as np
import napari
import os

# Point the script to the correct subfolder.
input_data_folder    = '2D TFM Data'
input_data_subfolder = 'Al Hole 3MHz 28012026'
cwd                  = os.getcwd()

IN_DIR  = os.path.join(cwd, 'DATA', input_data_folder, input_data_subfolder)

# Find all files in directory which are .npy files. 
npy_files = [
    f for f in os.listdir(IN_DIR)
    if f.lower().endswith(".npy")
    and os.path.isfile(os.path.join(IN_DIR, f))
]
print('Files available in directory:')
print(npy_files)
print()

img = np.load(os.path.join(IN_DIR, "Al_40_3_TFM_3D.npy"))

viewer = napari.Viewer()
viewer.add_image(
    img,
    name="TFM",
    colormap="viridis",
    contrast_limits=(img.min(), img.max())
)

napari.run()
