"""
Same ground-truth cube as reconstruction_experiment.py, but reconstructed
via inverse Radon (iradon per z-plane) instead of nearest-angle stitching.

Note: `extract_rotated_slice` takes a **central slice** (not a line
integral), so the iradon premise is technically violated. This script
tests how well iradon recovers the cube from slice data anyway.
"""

import os
import sys
import numpy as np
import napari

sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..')))

from Classes.Reconstruct3D import reconstruct_volume


# Geometry (same as reconstruction_experiment.py)
BLOCK_HEIGHT = 50  # mm
BLOCK_LENGTH = 50  # mm
ARRAY_LENGTH = 50  # mm (the "aperture" / lateral extent of each slice)
ROT_ANGLE_DEG = 3  # angular step

SPHERES = [
    [10, 10, 10, 4], [20, 20, 20, 4], [30, 30, 30, 4], [40, 40, 40, 4],
    [15, 35, 25, 3], [35, 15, 25, 3],
]
CYLINDER_XZ = [15, 25, 4]    # [x, z, r]
CYLINDER_YZ = [25, 15, 3]    # [y, z, r]

NX = NY = NZ = 200


def build_ground_truth():
    x = np.linspace(0, BLOCK_LENGTH, NX)
    y = np.linspace(0, BLOCK_LENGTH, NY)
    z = np.linspace(0, BLOCK_HEIGHT, NZ)
    X, Y, Z = np.meshgrid(x, y, z, indexing='xy')
    X, Y, Z = (np.transpose(A, (2, 0, 1)) for A in (X, Y, Z))

    vol = np.zeros((NZ, NY, NX), dtype=np.uint8)
    for sx, sy, sz, r in SPHERES:
        vol[(X - sx)**2 + (Y - sy)**2 + (Z - sz)**2 <= r**2] = 255
    cx, cz, rc = CYLINDER_XZ
    vol[(X - cx)**2 + (Z - cz)**2 <= rc**2] = 255
    cy, cz, rc = CYLINDER_YZ
    vol[(Y - cy)**2 + (Z - cz)**2 <= rc**2] = 255
    return vol


def extract_rotated_slice(volume, theta_deg, array_length_mm,
                          block_length_mm, block_height_mm):
    nz, ny, nx = volume.shape
    n_line_samples = nx
    s = np.linspace(-array_length_mm / 2, array_length_mm / 2, n_line_samples)

    x_c = block_length_mm / 2
    y_c = block_length_mm / 2
    th = np.deg2rad(theta_deg)

    x_line = x_c + s * np.cos(th)
    y_line = y_c + s * np.sin(th)
    x_idx = np.round(x_line / block_length_mm * (nx - 1)).astype(int)
    y_idx = np.round(y_line / block_length_mm * (ny - 1)).astype(int)
    valid = (x_idx >= 0) & (x_idx < nx) & (y_idx >= 0) & (y_idx < ny)

    slice_2d = np.zeros((nz, n_line_samples), dtype=volume.dtype)
    for i in range(n_line_samples):
        if valid[i]:
            slice_2d[:, i] = volume[:, y_idx[i], x_idx[i]]
    return slice_2d


def main():
    vol = build_ground_truth()
    print(f"Ground truth {vol.shape}")

    angles_deg = np.arange(0, 180, ROT_ANGLE_DEG)
    slices = np.stack(
        [extract_rotated_slice(vol, a, ARRAY_LENGTH, BLOCK_LENGTH, BLOCK_HEIGHT)
         for a in angles_deg],
        axis=0,
    ).astype(np.float32)
    print(f"{len(angles_deg)} slices, shape {slices.shape}")

    # iradon per z-plane  (slices is (n_angles, n_z, n_lat), same as our B-scan stack)
    recon = reconstruct_volume(
        slices, angles_deg,
        filter_name='shepp-logan', circle=False,
    )
    print(f"Recon shape {recon.shape}")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'reconstructed_radon_volume.npy')
    np.save(out, recon)
    print(f"Saved → {out}")

    viewer = napari.Viewer(title='Radon reconstruction of ground-truth cube')
    viewer.add_image(vol, name='Ground truth', colormap='gray', opacity=0.6)
    viewer.add_image(recon, name='Radon reconstruction',
                     colormap='hot', opacity=0.9)
    napari.run()


if __name__ == '__main__':
    main()
