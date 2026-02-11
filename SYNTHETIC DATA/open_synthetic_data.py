import os
import sys
import numpy as np
import napari

DEFAULT_PATH = "SYNTHETIC NPY/stitching_test/subvol_0_0_0.npy"  # file or folder

def describe(name, arr):
    print(f"{name}: type={type(arr).__name__}, shape={getattr(arr, 'shape', None)}, dtype={getattr(arr, 'dtype', None)}")

def is_image(arr):
    return isinstance(arr, np.ndarray) and arr.ndim >= 2

def load_path(path):
    items = []

    if os.path.isdir(path):
        for name in sorted(os.listdir(path)):
            full = os.path.join(path, name)
            if not os.path.isfile(full):
                continue
            items.extend(load_path(full))
        return items

    if not os.path.isfile(path):
        return items

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
    path = DEFAULT_PATH
    if not (os.path.isdir(path) or os.path.isfile(path)):
        raise SystemExit(f"Path not found: {path}")

    images = load_path(path)
    if not images:
        raise SystemExit("No image arrays found.")

    viewer = napari.Viewer()
    for name, arr in images:
        viewer.add_image(arr, name=name, colormap="viridis")
    napari.run()

if __name__ == "__main__":
    main()

    