"""
Same nearest-angle stitching method as reconstruction_experiment.py,
but applied to simulated B-scans produced by the synthetic engine
(run_engine.scan_volume_3d) instead of a hand-built cube of spheres.

Loads the rotational B-scan stack from SCAN_DIR, takes envelope,
normalises to uint8, and feeds it into the same
`reconstruct_volume_from_rotated_slices` function.
"""

import os
import sys
import numpy as np
import napari

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..')))

from Classes.Reconstruct3D import (
    load_bscans, load_bscans_complex, has_complex_bscans,
)


# Verbatim copy of reconstruction_experiment.reconstruct_volume_from_rotated_slices
def reconstruct_volume_from_rotated_slices(all_slices, angles_deg,
                                           nx, ny, nz,
                                           block_length_mm, block_height_mm,
                                           array_length_mm):
    angles_deg = np.asarray(angles_deg)
    slice_stack = np.stack(all_slices, axis=0)

    n_angles, nz_s, ns = slice_stack.shape

    recon_volume = np.zeros((nz, ny, nx), dtype=slice_stack.dtype)

    x = np.linspace(0, block_length_mm, nx)
    y = np.linspace(0, block_length_mm, ny)
    z = np.linspace(0, block_height_mm, nz)

    x_c = block_length_mm / 2
    y_c = block_length_mm / 2
    radius_max = array_length_mm / 2

    s_coords = np.linspace(-radius_max, radius_max, ns)

    X, Y = np.meshgrid(x, y, indexing='xy')
    dx = X - x_c
    dy = Y - y_c
    rho = np.sqrt(dx**2 + dy**2)

    phi_deg = (np.rad2deg(np.arctan2(dy, dx)) + 360) % 360
    theta_target = np.where(phi_deg < 180, phi_deg, phi_deg - 180)

    inside = rho <= radius_max

    angle_step = np.abs(angles_deg[1] - angles_deg[0]) if len(angles_deg) > 1 else 180
    angle_idx = np.round(theta_target / angle_step).astype(int)
    angle_idx = np.clip(angle_idx, 0, len(angles_deg) - 1)

    chosen_theta_deg = angles_deg[angle_idx]
    chosen_theta_rad = np.deg2rad(chosen_theta_deg)

    s = dx * np.cos(chosen_theta_rad) + dy * np.sin(chosen_theta_rad)

    s_idx = np.round((s - s_coords[0]) / (s_coords[-1] - s_coords[0]) * (ns - 1)).astype(int)
    s_idx = np.clip(s_idx, 0, ns - 1)

    for iz in range(nz):
        plane = np.zeros((ny, nx), dtype=recon_volume.dtype)

        valid_angle_idx = angle_idx[inside]
        valid_s_idx = s_idx[inside]

        plane_vals = slice_stack[valid_angle_idx, iz, valid_s_idx]
        plane[inside] = plane_vals

        recon_volume[iz] = plane

    return recon_volume


# Simulated rotational scan directory
SCAN_DIR = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', '..', 'SYNTHETIC DATA', 'output', 'radon_tests',
    'aperture_single_scatterer',
))


def main() -> None:
    # 1. Load simulated B-scans → (n_angles, n_z, n_lateral)
    if has_complex_bscans(SCAN_DIR):
        bscans, meta = load_bscans_complex(SCAN_DIR)
        env = np.abs(bscans).astype(np.float32)
    else:
        bscans_db, meta = load_bscans(SCAN_DIR)
        env = np.float32(10.0 ** (bscans_db / 20.0))

    angles_deg = np.degrees(meta['angles_rad'])
    aperture_m = float(meta['array_aperture_m'])
    z_start = float(meta['tfm_z_start_m'])
    z_end = float(meta['tfm_z_end_m'])

    # 2. Normalise each slice to uint8 [0,255] (same format as orig script)
    vmax = env.max() + 1e-12
    all_slices = [(env[k] / vmax * 255).astype(np.uint8)
                  for k in range(env.shape[0])]

    print(f"{len(all_slices)} simulated slices, "
          f"each {all_slices[0].shape}, "
          f"angles {angles_deg[0]:+.1f}..{angles_deg[-1]:+.1f} deg")

    # 3. Physical sizes in mm (same units as the original script)
    block_length_mm = aperture_m * 1e3        # x,y extent = aperture
    block_height_mm = (z_end - z_start) * 1e3  # z extent from TFM grid
    array_length_mm = aperture_m * 1e3

    n_z, n_lat = all_slices[0].shape
    nx = ny = n_lat
    nz = n_z

    # 4. Same stitching function, unchanged
    recon = reconstruct_volume_from_rotated_slices(
        all_slices=all_slices,
        angles_deg=angles_deg,
        nx=nx, ny=ny, nz=nz,
        block_length_mm=block_length_mm,
        block_height_mm=block_height_mm,
        array_length_mm=array_length_mm,
    )

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'reconstructed_simulated_volume.npy')
    np.save(out_path, recon)
    print(f"Saved → {out_path}   shape {recon.shape}")

    # 5. View
    dz = block_height_mm / max(nz - 1, 1)
    dxy = block_length_mm / max(nx - 1, 1)
    viewer = napari.Viewer(title='Reconstructed Simulated Volume')
    viewer.add_image(recon, name='Reconstructed (stitching)',
                     scale=(dz, dxy, dxy), colormap='hot')
    viewer.dims.axis_labels = ('z (mm)', 'y (mm)', 'x (mm)')
    napari.run()


if __name__ == '__main__':
    main()
