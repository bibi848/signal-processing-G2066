"""
TFM comparison — scatterer on-axis vs off-axis, with elevation aperture on/off.

Runs 4 cases:
  (A) Scatterer at dy = 0 mm,   elevation_aperture = None   (baseline)
  (B) Scatterer at dy = 2 mm,   elevation_aperture = None   (should vanish)
  (C) Scatterer at dy = 0 mm,   elevation_aperture = 5 mm   (slab)
  (D) Scatterer at dy = 2 mm,   elevation_aperture = 5 mm   (now visible)

For each case, build a voxel volume with a single high-impedance voxel,
run the FMC engine, reconstruct TFM, and plot a 2x2 grid of B-scans.
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

from engine.config import (
    SimulationConfig, SpecimenConfig, ArrayConfig, AcquisitionConfig,
)
from engine.fmc_engine import FMCEngine
from engine.materials import ALUMINUM
from engine.voxel_volume import VoxelVolume3D
from Classes.TFM1D import CTFM1D


Z_SCATTERER = 20e-3      # scatterer depth (m)
SPECIMEN_THICKNESS = 40e-3
APERTURE_MM = 5.0        # elevation aperture to test (real probe: 5 mm element height)
DY_ON  = 0.0             # on-axis
DY_OFF = 2e-3            # off-axis (within ±2.5 mm slab)


def build_volume(dy: float,
                 voxel_size: float = 0.5e-3,
                 extent: float = 12e-3,
                 contrast: float = 0.5) -> VoxelVolume3D:
    """Uniform Al volume with one high-impedance voxel at (z, 0, dy)."""
    n_xy = int(2 * extent / voxel_size) + 1
    n_z  = int(SPECIMEN_THICKNESS / voxel_size) + 1
    imp = np.full((n_z, n_xy, n_xy), ALUMINUM.Z_L, dtype=np.float32)

    origin_z, origin_y, origin_x = 0.0, -extent, -extent
    iz = int(round((Z_SCATTERER - origin_z) / voxel_size))
    iy = int(round((dy         - origin_y) / voxel_size))
    ix = int(round((0.0        - origin_x) / voxel_size))
    imp[iz, iy, ix] = ALUMINUM.Z_L * (1.0 + contrast)

    return VoxelVolume3D(
        impedance=imp,
        wavespeed=np.full_like(imp, ALUMINUM.c_L),
        voxel_size=voxel_size,
        origin_z=origin_z, origin_y=origin_y, origin_x=origin_x,
    )


def run_case(dy: float, aperture: float | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate FMC + TFM for one configuration. Returns (img_db, x_img, z_img)."""
    cfg = SimulationConfig(
        array=ArrayConfig(
            num_elements=128,
            element_pitch=0.3e-3,
            element_width=0.27e-3,
            element_height=5e-3,
            frequency=10e6,
            bandwidth=0.6,
            elevation_aperture=aperture,
            n_elevation_slices=17 if aperture else 1,
        ),
        specimen=SpecimenConfig(thickness=SPECIMEN_THICKNESS, width=50e-3),
        acquisition=AcquisitionConfig(time_samples=2048, snr_db=60.0, add_noise=False),
        wall_echoes=False,
    )

    vol = build_volume(dy)
    half_w = cfg.array.aperture / 2
    # Sample finer than voxel size so the single-voxel gradient is captured
    born_step = vol.voxel_size / 2
    z_grid = np.arange(2e-3, SPECIMEN_THICKNESS + born_step, born_step)
    l_grid = np.arange(-half_w, half_w + born_step, born_step)

    z_s, x_s, amp_s = vol.extract_born_scatterers(
        theta=0.0,
        z_grid=z_grid,
        lateral_grid=l_grid,
        background_Z=ALUMINUM.Z_L,
        threshold=1e-6,
        elevation_aperture=aperture,
        n_slices=cfg.array.n_elevation_slices,
    )
    print(f"  Born scatterers extracted: {len(z_s)}  "
          f"max|amp|={np.max(np.abs(amp_s)) if len(amp_s) else 0:.4e}")

    engine = FMCEngine(cfg)
    if len(z_s) > 0:
        engine.set_born_scatterers(z_s, x_s, amp_s)
    result = engine.simulate()
    fmc = result['fmc_data']
    time_axis = result['time_axis']
    elem_x = result['element_positions']

    # Gate out front wall
    gate = int(2e-6 / cfg.dt)
    fmc[:, :, :gate] = 0.0

    # TFM via Python CTFM1D
    num_el = fmc.shape[0]
    tx_arr = np.repeat(np.arange(1, num_el + 1), num_el)
    rx_arr = np.tile(np.arange(1, num_el + 1), num_el)
    fmc_flat = fmc.reshape(-1, fmc.shape[-1])
    xc = elem_x
    zc = np.zeros_like(xc)

    x_img = np.linspace(-half_w, half_w, 200)
    z_img = np.linspace(5e-3, 35e-3, 300)

    img = CTFM1D(fmc_flat, time_axis, tx_arr, rx_arr, xc, zc,
                 cfg.material.c_L, x_img, z_img, output='real')
    env = np.abs(hilbert(img, axis=0))
    img_db = 20 * np.log10(env / (env.max() + 1e-12) + 1e-12)
    return img_db, x_img, z_img


def main() -> None:
    cases = [
        ('A: on-axis (dy=0),  aperture OFF',  DY_ON,  None),
        ('B: off-axis (dy=2mm), aperture OFF', DY_OFF, None),
        ('C: on-axis (dy=0),  aperture 5 mm',  DY_ON,  APERTURE_MM * 1e-3),
        ('D: off-axis (dy=2mm), aperture 5 mm', DY_OFF, APERTURE_MM * 1e-3),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for ax, (title, dy, aperture) in zip(axes, cases):
        print(f"\n--- {title} ---")
        img_db, x_img, z_img = run_case(dy, aperture)
        im = ax.imshow(
            img_db,
            extent=[x_img[0]*1e3, x_img[-1]*1e3, z_img[-1]*1e3, z_img[0]*1e3],
            aspect='auto', cmap='inferno', vmin=-40, vmax=0,
        )
        # Mark true scatterer projected position on scan plane (x=0, z=20mm)
        ax.plot(0.0, Z_SCATTERER * 1e3, 'c+', markersize=14, markeredgewidth=2,
                label='true (x,z)')
        ax.set_title(title)
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('z (mm)')
        ax.legend(loc='lower right', fontsize=8)
        plt.colorbar(im, ax=ax, label='dB')

    fig.suptitle(
        f'TFM reconstruction vs elevation aperture  '
        f'(scatterer at z={Z_SCATTERER*1e3:.0f} mm)',
        fontsize=13,
    )
    fig.tight_layout()

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'elevation_aperture_tfm.png')
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved TFM comparison → {out_path}")


if __name__ == '__main__':
    main()
