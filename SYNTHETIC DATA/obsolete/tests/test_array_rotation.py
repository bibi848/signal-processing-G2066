"""
Rotating array over a fixed off-centre scatterer.

The array rotates about its centre while a single scatterer sits at a fixed
world position (x0, y0, z0). For each rotation angle θ the scan plane
contains (cos θ, sin θ, 0), so the scatterer projects onto:

    L_parallel      = x0·cos θ + y0·sin θ   (lateral in the scan plane)
    d_perpendicular = −x0·sin θ + y0·cos θ  (distance from the plane)

With a finite elevation aperture h, the scatterer contributes to the TFM
image whenever |d_perpendicular| < h/2, at lateral position L_parallel.
"""

from __future__ import annotations

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import hilbert

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from engine.config import SimulationConfig, SpecimenConfig, ArrayConfig, AcquisitionConfig
from engine.fmc_engine import FMCEngine
from engine.materials import ALUMINUM
from engine.voxel_volume import VoxelVolume3D
from Classes.TFM1D import CTFM1D


# Fixed scatterer position in WORLD coordinates
SCAT_X = 1e-3         # 1 mm east
SCAT_Y = 2e-3         # 2 mm north
SCAT_Z = 20e-3        # 20 mm deep

SPECIMEN_THICKNESS = 40e-3
APERTURE = 5e-3       # 5 mm elevation aperture (real probe element height)
ANGLES_DEG = [0, 30, 60, 90, 120, 150]


def build_volume(voxel_size: float = 0.5e-3,
                 extent: float = 15e-3,
                 contrast: float = 0.5) -> VoxelVolume3D:
    """Uniform Al volume with ONE high-impedance voxel at (SCAT_X, SCAT_Y, SCAT_Z)."""
    n_xy = int(2 * extent / voxel_size) + 1
    n_z  = int(SPECIMEN_THICKNESS / voxel_size) + 1
    imp = np.full((n_z, n_xy, n_xy), ALUMINUM.Z_L, dtype=np.float32)

    origin_z, origin_y, origin_x = 0.0, -extent, -extent
    iz = int(round((SCAT_Z - origin_z) / voxel_size))
    iy = int(round((SCAT_Y - origin_y) / voxel_size))
    ix = int(round((SCAT_X - origin_x) / voxel_size))
    imp[iz, iy, ix] = ALUMINUM.Z_L * (1.0 + contrast)

    return VoxelVolume3D(
        impedance=imp,
        wavespeed=np.full_like(imp, ALUMINUM.c_L),
        voxel_size=voxel_size,
        origin_z=origin_z, origin_y=origin_y, origin_x=origin_x,
    )


def run_angle(vol: VoxelVolume3D, theta: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate FMC + TFM at one rotation angle θ. Returns (img_db, x_img, z_img)."""
    cfg = SimulationConfig(
        array=ArrayConfig(
            num_elements=128, element_pitch=0.3e-3, element_width=0.27e-3,
            element_height=5e-3, frequency=10e6, bandwidth=0.6,
            elevation_aperture=APERTURE, n_elevation_slices=17,
        ),
        specimen=SpecimenConfig(thickness=SPECIMEN_THICKNESS, width=50e-3),
        acquisition=AcquisitionConfig(time_samples=2048, snr_db=60.0, add_noise=False),
        wall_echoes=False,
    )

    half_w = cfg.array.aperture / 2
    born_step = vol.voxel_size / 2
    z_grid = np.arange(2e-3, SPECIMEN_THICKNESS + born_step, born_step)
    l_grid = np.arange(-half_w, half_w + born_step, born_step)

    z_s, x_s, amp_s = vol.extract_born_scatterers(
        theta=theta, z_grid=z_grid, lateral_grid=l_grid,
        background_Z=ALUMINUM.Z_L, threshold=1e-6,
        elevation_aperture=APERTURE, n_slices=cfg.array.n_elevation_slices,
    )

    engine = FMCEngine(cfg)
    if len(z_s) > 0:
        engine.set_born_scatterers(z_s, x_s, amp_s)
    result = engine.simulate()
    fmc = result['fmc_data']
    time_axis = result['time_axis']
    elem_x = result['element_positions']

    fmc[:, :, :int(2e-6 / cfg.dt)] = 0.0  # gate out front wall

    num_el = fmc.shape[0]
    tx_arr = np.repeat(np.arange(1, num_el + 1), num_el)
    rx_arr = np.tile(np.arange(1, num_el + 1), num_el)
    fmc_flat = fmc.reshape(-1, fmc.shape[-1])
    xc, zc = elem_x, np.zeros_like(elem_x)

    x_img = np.linspace(-half_w, half_w, 200)
    z_img = np.linspace(5e-3, 35e-3, 300)
    img = CTFM1D(fmc_flat, time_axis, tx_arr, rx_arr, xc, zc,
                 cfg.material.c_L, x_img, z_img, output='real')
    env = np.abs(hilbert(img, axis=0))
    img_db = 20 * np.log10(env / (env.max() + 1e-12) + 1e-12)
    return img_db, x_img, z_img


def main() -> None:
    vol = build_volume()

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    # Summary scatter plot of expected (L_parallel, d_perp) per angle
    print(f"\nScatterer at world (x={SCAT_X*1e3:.1f}, y={SCAT_Y*1e3:.1f}, "
          f"z={SCAT_Z*1e3:.1f}) mm  |  aperture = {APERTURE*1e3:.1f} mm\n")
    print(f"{'θ°':>6}  {'L∥ (mm)':>10}  {'d⊥ (mm)':>10}  visible?")

    for ax, th_deg in zip(axes, ANGLES_DEG):
        th = np.deg2rad(th_deg)
        L  =  SCAT_X * np.cos(th) + SCAT_Y * np.sin(th)
        d  = -SCAT_X * np.sin(th) + SCAT_Y * np.cos(th)
        visible = abs(d) < APERTURE / 2
        print(f"{th_deg:>6}  {L*1e3:>10.2f}  {d*1e3:>10.2f}  {'yes' if visible else 'no'}")

        img_db, x_img, z_img = run_angle(vol, th)
        im = ax.imshow(
            img_db,
            extent=[x_img[0]*1e3, x_img[-1]*1e3, z_img[-1]*1e3, z_img[0]*1e3],
            aspect='auto', cmap='inferno', vmin=-10, vmax=0,
        )
        # expected lateral position of the scatterer in the scan plane
        if visible:
            ax.plot(L * 1e3, SCAT_Z * 1e3, 'c+',
                    markersize=14, markeredgewidth=2, label='expected')
            ax.legend(loc='lower right', fontsize=8)
        ax.set_title(f'θ = {th_deg}°   L∥ = {L*1e3:+.2f} mm   d⊥ = {d*1e3:+.2f} mm'
                     + ('' if visible else '  (out of slab)'),
                     fontsize=10)
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('z (mm)')
        plt.colorbar(im, ax=ax, label='dB')

    fig.suptitle(
        f'Rotating array over fixed scatterer at (x={SCAT_X*1e3:.0f}, '
        f'y={SCAT_Y*1e3:.0f}, z={SCAT_Z*1e3:.0f}) mm,  '
        f'elevation aperture = {APERTURE*1e3:.0f} mm',
        fontsize=13,
    )
    fig.tight_layout()

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'array_rotation_tfm.png')
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved → {out_path}")


if __name__ == '__main__':
    main()
