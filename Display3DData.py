'''
This script allows you to view the .npy files collected from 2D array scanning. 
It uses the library napari for viewing. 

pip install napari
pip install -U napari[pyqt5]
python Display3DData.py

In your terminal (one by one) worked for me. 
Linux: export QT_QPA_PLATFORM=xcb
'''

# Configure Qt for macOS before importing napari
import os
import sys

if sys.platform == 'darwin':  # macOS
    import sysconfig
    try:
        import site
        site_packages = site.getsitepackages()[0] if site.getsitepackages() else None
        if site_packages:
            qt_plugin_path = os.path.join(site_packages, 'PyQt5', 'Qt5', 'plugins')
            if os.path.exists(qt_plugin_path):
                os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = qt_plugin_path
    except:
        pass
    if 'QT_QPA_PLATFORM_PLUGIN_PATH' not in os.environ:
        lib_path = sysconfig.get_path('purelib')
        if lib_path:
            qt_plugin_path = os.path.join(lib_path, 'PyQt5', 'Qt5', 'plugins')
            if os.path.exists(qt_plugin_path):
                os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = qt_plugin_path
    if 'QT_QPA_PLATFORM_PLUGIN_PATH' not in os.environ:
        base_path = os.path.join(sys.prefix, 'lib', f'python{sys.version_info.major}.{sys.version_info.minor}', 'site-packages')
        qt_plugin_path = os.path.join(base_path, 'PyQt5', 'Qt5', 'plugins')
        if os.path.exists(qt_plugin_path):
            os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = qt_plugin_path
    os.environ['QT_QPA_PLATFORM'] = 'cocoa'
    os.environ['QT_LOGGING_RULES'] = '*.debug=false;qt.qpa.*=false'

import numpy as np
import napari

# Point the script to the correct subfolder.
input_data_folder    = '2D TFM Data'
input_data_subfolder = 'Al Pure 15MHz 12022026 Filtered'
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

img = np.load(os.path.join(IN_DIR, "Al_70_1_1_filtered_3D_TFM.npy"))

viewer = napari.Viewer()
viewer.add_image(
    img,
    name="TFM",
    colormap="viridis",
    contrast_limits=(img.min(), img.max())
)

napari.run()
