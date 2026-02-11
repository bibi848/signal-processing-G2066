import os
import sys
import numpy as np
import napari

# -----------------------------------------------------------------------------
# How to use:
# 1) Set DEFAULT_PATH to a file (.npy/.npz) or a folder containing such files.
# 2) Run: python "SYNTHETIC DATA/open_synthetic_data.py"
#
# Behavior:
# - Prints info (type, shape, dtype) for every .npy/.npz it loads.
# - Displays only arrays that look like images (ndim >= 2) in napari.
# - Skips metadata/0-D arrays automatically.
# -----------------------------------------------------------------------------

DEFAULT_PATH = "SYNTHETIC NPY/stitching_test/subvol_0_0_0.npy"  # file or folder

def describe(name, arr):
    """Print a quick summary of each loaded item."""
    print(f"{name}: type={type(arr).__name__}, shape={getattr(arr, 'shape', None)}, dtype={getattr(arr, 'dtype', None)}")

def is_image(arr):
    """Return True for arrays that can be shown as images (2D or higher)."""
    return isinstance(arr, np.ndarray) and arr.ndim >= 2

def load_path(path):
    """
    Load a file or a folder.
    - If a folder: recursively load all .npy/.npz files inside it.
    - If a file: load that single .npy/.npz.
    Returns a list of (name, array) for image-like arrays only.
    """
    items = []

    # Folder: walk its contents and load each file
    if os.path.isdir(path):
        for name in sorted(os.listdir(path)):
            full = os.path.join(path, name)
            if not os.path.isfile(full):
                continue
            items.extend(load_path(full))
        return items

    # If not a file, nothing to load
    if not os.path.isfile(path):
        return items

    # File: .npy or .npz
    name = os.path.basename(path)
    if name.lower().endswith(".npy"):
        arr = np.load(path, allow_pickle=True)
        describe(name, arr)
        if is_image(arr):
            items.append((name, arr))
    elif name.lower().endswith(".npz"):
        data = np.load(path, allow_pickle=True)
        for key in data.files:
            arr = data[key]
            describe(f"{name}:{key}", arr)
            if is_image(arr):
                items.append((f"{name}:{key}", arr))

    return items

def main():
    # Choose file/folder from DEFAULT_PATH
    path = DEFAULT_PATH
    if not (os.path.isdir(path) or os.path.isfile(path)):
        raise SystemExit(f"Path not found: {path}")

    # Load and filter image arrays
    images = load_path(path)
    if not images:
        raise SystemExit("No image arrays found.")

    # Show all image arrays in napari
    viewer = napari.Viewer()
    for name, arr in images:
        viewer.add_image(arr, name=name, colormap="viridis")
    napari.run()


main()