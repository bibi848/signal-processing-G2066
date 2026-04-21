'''
This script allows you to view the .npy files collected from 2D array scanning. 
It uses the library napari for viewing, with X/Y/Z axis labels shown inside the viewer.

pip install napari
pip install -U napari[pyqt5]
python Display3DData.py

In your terminal (one by one) worked for me. 
Linux: export QT_QPA_PLATFORM=xcb
'''
#%%
import os
import sys
import numpy as np
import napari

# Configuring Napari for MacOS
if sys.platform == 'darwin':
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

# Point the script to the correct subfolder.
input_data_folder    = 'PROCESSING'
input_data_subfolder = 'Rotation NPYs'
cwd                  = os.getcwd()

IN_DIR = os.path.join(cwd, input_data_folder, input_data_subfolder)

# Find all files in directory which are .npy files.
npy_files = [
    f for f in os.listdir(IN_DIR)
    if f.lower().endswith(".npy")
    and os.path.isfile(os.path.join(IN_DIR, f))
]
print('Files available in directory:')
print(npy_files)
print()

# ---- USER SETTINGS: choose files here ----
files_to_load = [
    "position_2_fused_max.npy",
    "position_2_fused_median.npy",
    "position_2_fused_mean.npy",
]
# -----------------------------------------

viewer = napari.Viewer(ndisplay=3)  # Open directly in 3D mode

for fname in files_to_load:
    path = os.path.join(IN_DIR, fname)

    if not os.path.exists(path):
        print(f"Warning: {fname} not found, skipping.")
        continue

    img = np.load(path)

    if img.ndim != 3:
        print(f"Warning: {fname} has shape {img.shape} (expected 3D), skipping.")
        continue

    viewer.add_image(
        img,
        name=os.path.splitext(fname)[0],
        colormap="viridis",
        contrast_limits=(img.min(), img.max())
    )

# --- Axis labels (X, Y, Z) shown inside the viewer ---
# napari's axes overlay uses the dimension order of the array:
#   axis 0 → Z (depth / slices)
#   axis 1 → Y (rows)
#   axis 2 → X (columns)
viewer.dims.axis_labels = ("Z", "Y", "X")

# Enable the axes overlay so labels are rendered in the 3D canvas
viewer.axes.visible    = True   # Show the XYZ axes widget in the corner
viewer.axes.colored    = True   # Colour each axis (R=X, G=Y, B=Z)
viewer.axes.labels     = True   # Draw the X / Y / Z text labels
viewer.axes.dashed     = False  # Solid lines (set True for dashed)

napari.run()

