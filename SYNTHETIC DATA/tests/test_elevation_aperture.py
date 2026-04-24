"""
Off-plane point-scatterer validation for the finite-width elevation aperture.

Place a single high-impedance voxel at varying y-offsets from the scan plane
and measure the Born amplitude picked up by `extract_born_scatterers` with:
  - elevation_aperture = None  (thin slice, legacy)
  - elevation_aperture = 8 mm  (8 mm slab)

Expected:
  - Thin slice: Kronecker-like spike — only dy = 0 produces an echo.
  - 8 mm slab:  Top-hat — roughly uniform for |dy| < 4 mm, zero outside.
"""

from __future__ import annotations

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.voxel_volume import VoxelVolume3D
from engine.materials import ALUMINUM


def build_single_voxel_volume(dy: float,
                              voxel_size: float = 0.5e-3,
                              extent: float = 20e-3,
                              z_depth: float = 20e-3,
                              background_Z: float = ALUMINUM.Z_L,
                              contrast: float = 0.5) -> VoxelVolume3D:
    """Volume of uniform impedance with ONE voxel offset by dy in elevation."""
    n = int(2 * extent / voxel_size) + 1
    n_z = int(2 * z_depth / voxel_size) + 1
    shape = (n_z, n, n)

    imp = np.full(shape, background_Z, dtype=np.float32)

    # Centre the volume at x=0, y=0; z-origin at 0 (array surface)
    origin_x = -extent
    origin_y = -extent
    origin_z = 0.0

    # Indices of the target voxel (z=z_depth, x=0, y=dy)
    iz = int(round((z_depth - origin_z) / voxel_size))
    iy = int(round((dy - origin_y) / voxel_size))
    ix = int(round((0.0 - origin_x) / voxel_size))

    # Inject a high-contrast voxel AND the one below it so that the depth
    # gradient (ΔZ_z) is non-zero at the scatterer (extract_born_scatterers
    # uses np.diff along z — an isolated voxel produces two gradient pulses;
    # that is fine, we just need the peak amplitude).
    imp[iz, iy, ix] = background_Z * (1.0 + contrast)

    wavespeed = np.full(shape, ALUMINUM.c_L, dtype=np.float32)
    return VoxelVolume3D(
        impedance=imp,
        wavespeed=wavespeed,
        voxel_size=voxel_size,
        origin_z=origin_z,
        origin_y=origin_y,
        origin_x=origin_x,
    )


def measure_amplitude(vol: VoxelVolume3D,
                      elevation_aperture: float | None,
                      n_slices: int = 9) -> float:
    """Peak |Born amplitude| returned by extract_born_scatterers at theta=0."""
    z_grid = np.linspace(5e-3, 40e-3, 141)
    l_grid = np.linspace(-10e-3, 10e-3, 81)
    z_s, x_s, amp_s = vol.extract_born_scatterers(
        theta=0.0,
        z_grid=z_grid,
        lateral_grid=l_grid,
        background_Z=ALUMINUM.Z_L,
        threshold=1e-6,
        elevation_aperture=elevation_aperture,
        n_slices=n_slices,
    )
    if len(amp_s) == 0:
        return 0.0
    return float(np.max(np.abs(amp_s)))


def main() -> None:
    dy_values = np.linspace(-6e-3, 6e-3, 25)

    amps_thin: list[float] = []
    amps_slab: list[float] = []

    for dy in dy_values:
        vol = build_single_voxel_volume(dy=dy)
        amps_thin.append(measure_amplitude(vol, elevation_aperture=None))
        amps_slab.append(measure_amplitude(vol, elevation_aperture=5e-3, n_slices=17))
        print(f"  dy = {dy*1e3:+.2f} mm  |  thin = {amps_thin[-1]:.4e}  "
              f"slab = {amps_slab[-1]:.4e}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(dy_values * 1e3, amps_thin, 'o-', label='elevation_aperture = None (thin slice)')
    ax.plot(dy_values * 1e3, amps_slab, 's-', label='elevation_aperture = 5 mm')
    ax.axvspan(-2.5, 2.5, alpha=0.15, color='tab:orange',
               label='expected aperture coverage (±2.5 mm)')
    ax.set_xlabel('Off-plane offset  dy  (mm)')
    ax.set_ylabel('Peak |Born amplitude|')
    ax.set_title('Off-plane point-scatterer response vs elevation aperture')
    ax.grid(True, alpha=0.3)
    ax.legend()

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'elevation_aperture_validation.png')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved plot → {out_path}")


if __name__ == '__main__':
    main()
