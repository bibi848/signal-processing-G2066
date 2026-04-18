"""
Voxel-based 3D impedance field with full-3D Born scatterer extraction.

Reuses engine.voxel_volume.VoxelVolume3D unchanged (it is already 3D) and
extends it with a 3D-gradient extractor that walks all three axes of the
impedance field directly, producing (z, x, y, amp) tuples for the 3D Born
engine.

This mirrors engine.voxel_volume.extract_born_scatterers (2D) — same
gradient / threshold / Z₀ semantics, no slicing, no elevation aperture,
no per-scatterer wavespeed.
"""

from typing import Tuple
import numpy as np

from engine.voxel_volume import VoxelVolume3D


def extract_born_scatterers_3d(
    volume: VoxelVolume3D,
    background_Z: float,
    threshold: float = 0.005,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract 3D Born scatterers from an impedance volume.

    Born amplitude is the signed per-axis gradient of impedance normalised by
    the background impedance:

        amp_axis = ΔZ_axis / (2 · Z₀)

    A scatterer is emitted at every voxel whose |ΔZ / 2Z₀| exceeds the
    threshold, for each of the z / y / x gradient axes independently. Positions
    are the voxel centre — no sub-voxel jitter (matches the 2D sibling).

    Args:
        volume:        VoxelVolume3D with (n_z, n_y, n_x) impedance
        background_Z:  Background impedance Z₀ (Pa·s/m)
        threshold:     Minimum |ΔZ / 2Z₀| to emit a scatterer

    Returns:
        (z_s, x_s, y_s, amp_s) — scatterer world coordinates and signed
        Born amplitudes (dimensionless).
    """
    imp = volume.impedance
    vs = volume.voxel_size
    inv_2Z = 1.0 / (2.0 * background_Z)

    delta_z = np.diff(imp, axis=0, prepend=imp[:1]) * inv_2Z
    delta_y = np.diff(imp, axis=1, prepend=imp[:, :1]) * inv_2Z
    delta_x = np.diff(imp, axis=2, prepend=imp[:, :, :1]) * inv_2Z

    z_parts, y_parts, x_parts, a_parts = [], [], [], []
    for grid in (delta_z, delta_y, delta_x):
        iz, iy, ix = np.where(np.abs(grid) > threshold)
        if iz.size == 0:
            continue
        z_parts.append(volume.origin_z + iz * vs)
        y_parts.append(volume.origin_y + iy * vs)
        x_parts.append(volume.origin_x + ix * vs)
        a_parts.append(grid[iz, iy, ix])

    if not z_parts:
        empty = np.empty(0, dtype=np.float64)
        return empty, empty, empty, empty

    z_s = np.concatenate(z_parts).astype(np.float64)
    y_s = np.concatenate(y_parts).astype(np.float64)
    x_s = np.concatenate(x_parts).astype(np.float64)
    amp_s = np.concatenate(a_parts).astype(np.float64)
    return z_s, x_s, y_s, amp_s


__all__ = ['VoxelVolume3D', 'extract_born_scatterers_3d']
