"""
Nearest-angle slice-stitching reconstruction on simulated B-scans.

Port of PROCESSING/Reconstruction Tests/reconstruction_experiment.py
`reconstruct_volume_from_rotated_slices`, applied to the rotational
B-scan stack produced by run_engine.py.

For each voxel (x,y,z):
    - pick the nearest measured angle θ_k
    - compute s = x·cosθ_k + y·sinθ_k
    - copy |bscan[k, z, s]| into the voxel

No filtering, no back-projection summation. Useful for comparison with
the inverse-Radon reconstruction of the same scan.
"""

from __future__ import annotations

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from Classes.Reconstruct3D import (
    load_bscans, load_bscans_complex, has_complex_bscans,
    view_reconstruction_napari,
)


# Point this at an existing scan directory (written by run_engine.scan_volume_3d)
SCAN_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'output', 'radon_tests', 'aperture_single_scatterer',
)


def reconstruct_stitch(
    slices: np.ndarray,       # (n_angles, n_z, n_lateral) real
    angles_deg: np.ndarray,   # (n_angles,)
    aperture_m: float,        # physical lateral extent of each B-scan
    output_size: int | None = None,
) -> np.ndarray:
    """
    Nearest-angle stitching. Returns (n_z, ny, nx) float32.

    Output grid is square with side `output_size` (default = n_lateral),
    spanning [-aperture/2, +aperture/2] in both x and y.
    """
    n_angles, n_z, n_lat = slices.shape
    if output_size is None:
        output_size = n_lat

    radius_max = aperture_m / 2.0
    s_coords = np.linspace(-radius_max, radius_max, n_lat)

    x = np.linspace(-radius_max, radius_max, output_size)
    y = np.linspace(-radius_max, radius_max, output_size)
    X, Y = np.meshgrid(x, y, indexing='xy')  # (ny, nx)

    rho = np.sqrt(X**2 + Y**2)
    inside = rho <= radius_max

    # angle of each voxel in x-y plane wrapped to [0, 180)
    phi_deg = (np.rad2deg(np.arctan2(Y, X)) + 360.0) % 360.0
    theta_target = np.where(phi_deg < 180.0, phi_deg, phi_deg - 180.0)

    # nearest measured angle (map measured angles to [0, 180))
    meas = (angles_deg + 360.0) % 360.0
    meas = np.where(meas < 180.0, meas, meas - 180.0)
    # for each pixel, pick nearest index in meas
    diff = np.abs(theta_target[..., None] - meas[None, None, :])
    diff = np.minimum(diff, 180.0 - diff)  # circular distance on [0,180)
    angle_idx = np.argmin(diff, axis=-1)

    chosen_theta_rad = np.deg2rad(angles_deg[angle_idx])
    s = X * np.cos(chosen_theta_rad) + Y * np.sin(chosen_theta_rad)

    s_idx = np.round((s - s_coords[0]) / (s_coords[-1] - s_coords[0])
                     * (n_lat - 1)).astype(int)
    s_idx = np.clip(s_idx, 0, n_lat - 1)

    volume = np.zeros((n_z, output_size, output_size), dtype=np.float32)
    for iz in range(n_z):
        plane = np.zeros((output_size, output_size), dtype=np.float32)
        plane[inside] = slices[angle_idx[inside], iz, s_idx[inside]]
        volume[iz] = plane

    return volume


def main() -> None:
    if has_complex_bscans(SCAN_DIR):
        bscans, meta = load_bscans_complex(SCAN_DIR)
        slices = np.abs(bscans).astype(np.float32)
    else:
        bscans_db, meta = load_bscans(SCAN_DIR)
        slices = np.float32(10.0 ** (bscans_db / 20.0))

    angles_deg = np.degrees(meta['angles_rad'])
    aperture = meta['array_aperture_m']
    print(f"Angles {angles_deg[0]:+.1f}..{angles_deg[-1]:+.1f} deg "
          f"({len(angles_deg)} frames), aperture {aperture*1e3:.1f} mm")

    volume = reconstruct_stitch(slices, angles_deg, aperture)
    out_path = os.path.join(SCAN_DIR, 'recon_stitch.npy')
    np.save(out_path, volume)
    print(f"Saved → {out_path}   shape {volume.shape}")

    # physical axes
    z_coords = np.linspace(meta['tfm_z_start_m'], meta['tfm_z_end_m'],
                           volume.shape[0])
    half = aperture / 2.0
    y_coords = np.linspace(-half, half, volume.shape[1])
    x_coords = np.linspace(-half, half, volume.shape[2])

    view_reconstruction_napari(volume, None, z_coords, y_coords, x_coords)


if __name__ == '__main__':
    main()
