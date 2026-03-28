"""
Quick napari viewer for reconstructed 3D volumes.

Usage:
    python view_volumes.py VOL1.npy [VOL2.npy ...]
    python view_volumes.py "DATA/1D NPY Data/Al Pure 10MHz 18032026 Vol1/recon_volume.npy" \
                           "DATA/1D NPY Data/Al Pure 10MHz 18032026 Vol2/recon_volume.npy"
"""

import sys
import os
import numpy as np
import napari

COLORMAPS = ["viridis", "magma", "inferno", "plasma", "turbo", "hot"]


def main():
    paths = sys.argv[1:]
    if not paths:
        print("Usage: python view_volumes.py VOL1.npy [VOL2.npy ...]")
        sys.exit(1)

    viewer = napari.Viewer()

    for i, path in enumerate(paths):
        vol = np.load(path, allow_pickle=True)
        name = os.path.splitext(os.path.basename(os.path.dirname(path)))[0] or f"vol_{i}"
        cmap = COLORMAPS[i % len(COLORMAPS)]
        viewer.add_image(vol, name=name, colormap=cmap, rendering="mip")
        print(f"Loaded {name}: shape={vol.shape}, dtype={vol.dtype}")

    napari.run()


if __name__ == "__main__":
    main()
