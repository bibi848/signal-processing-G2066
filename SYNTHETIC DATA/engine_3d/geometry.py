"""
3D defect-to-scatterer helpers.

The 3D geometry classes (Specimen3D, SphericalDefect, CylindricalDefect,
PlanarCrack3D) already live in engine.geometry and are dimension-correct.
We reuse them as-is and provide a single dispatcher that turns each into a
3D surface-scatterer cloud (z, x, y, amp) suitable for the 3D Born engine.

The 2D slice_at_* helpers on those classes are ignored by this engine —
they exist for the legacy rotational 2D pipeline only.
"""

from typing import Tuple
import numpy as np

from engine.geometry import (
    Specimen3D,
    SphericalDefect,
    CylindricalDefect,
    PlanarCrack3D,
    DEFAULT_VOID_BORN_AMP,
)


def defect_to_born_scatterers_3d(
    defect,
    n_points: int = 600,
    amplitude: float = DEFAULT_VOID_BORN_AMP,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert a 3D defect into a point cloud of surface scatterers.

    Returns:
        (z_s, x_s, y_s, amp_s) — world coordinates and Born amplitudes.
    """
    if isinstance(defect, SphericalDefect):
        return _sphere_surface(defect, n_points, amplitude)
    if isinstance(defect, CylindricalDefect):
        return _cylinder_surface(defect, n_points, amplitude)
    if isinstance(defect, PlanarCrack3D):
        return _planar_crack_surface(defect, n_points, amplitude)
    raise TypeError(f"Unsupported 3D defect type: {type(defect).__name__}")


def _sphere_surface(
    d: SphericalDefect, n_points: int, amplitude: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    k = np.arange(n_points)
    phi = np.pi * (3.0 - np.sqrt(5.0))  # golden angle
    ang_y = phi * k
    cos_t = 1.0 - 2.0 * (k + 0.5) / n_points
    sin_t = np.sqrt(np.maximum(1.0 - cos_t ** 2, 0.0))
    nx = sin_t * np.cos(ang_y)
    ny = sin_t * np.sin(ang_y)
    nz = cos_t

    z = d.center_z + d.radius * nz
    x = d.center_x + d.radius * nx
    y = d.center_y + d.radius * ny
    amp = np.full(n_points, amplitude, dtype=np.float64)
    return z, x, y, amp


def _cylinder_surface(
    d: CylindricalDefect, n_points: int, amplitude: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    length = max(d.y_end - d.y_start, 1e-12)
    r = d.radius
    area_wall = 2.0 * np.pi * r * length
    area_caps = 2.0 * np.pi * r ** 2
    frac_wall = area_wall / (area_wall + area_caps)

    n_wall = max(int(round(n_points * frac_wall)), 1)
    n_caps = max(n_points - n_wall, 0)

    n_ang = max(int(round(np.sqrt(n_wall * 2 * np.pi * r / length))), 4)
    n_axial = max(int(round(n_wall / n_ang)), 1)
    angles = np.linspace(0.0, 2.0 * np.pi, n_ang, endpoint=False)
    ys_axial = np.linspace(d.y_start, d.y_end, n_axial)
    ang_grid, y_grid = np.meshgrid(angles, ys_axial, indexing='xy')
    cos_a = np.cos(ang_grid).ravel()
    sin_a = np.sin(ang_grid).ravel()
    z_wall = d.center_z + r * cos_a
    x_wall = d.center_x + r * sin_a
    y_wall = y_grid.ravel()

    n_per_cap = max(n_caps // 2, 0)
    if n_per_cap > 0:
        radii = r * np.sqrt(np.linspace(0.0, 1.0, n_per_cap, endpoint=False)
                             + 0.5 / n_per_cap)
        golden = np.pi * (3.0 - np.sqrt(5.0))
        theta = golden * np.arange(n_per_cap)
        cz = d.center_z + radii * np.cos(theta)
        cx = d.center_x + radii * np.sin(theta)
        z_cap = np.concatenate([cz, cz])
        x_cap = np.concatenate([cx, cx])
        y_cap = np.concatenate([
            np.full(n_per_cap, d.y_start),
            np.full(n_per_cap, d.y_end),
        ])
    else:
        z_cap = x_cap = y_cap = np.empty(0)

    z = np.concatenate([z_wall, z_cap]).astype(np.float64)
    x = np.concatenate([x_wall, x_cap]).astype(np.float64)
    y = np.concatenate([y_wall, y_cap]).astype(np.float64)
    amp = np.full(z.shape, amplitude, dtype=np.float64)
    return z, x, y, amp


def _planar_crack_surface(
    d: PlanarCrack3D, n_points: int, amplitude: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    seg_len = float(np.hypot(d.end_z - d.start_z, d.end_x - d.start_x))
    y_len = float(d.y_end - d.y_start)
    aspect = seg_len / max(y_len, 1e-12)
    n_s = max(int(round(np.sqrt(n_points * aspect))), 2)
    n_y = max(int(round(n_points / n_s)), 2)
    s = np.linspace(0.0, 1.0, n_s)
    y_line = np.linspace(d.y_start, d.y_end, n_y)
    S, Y = np.meshgrid(s, y_line, indexing='xy')
    z = (d.start_z + S * (d.end_z - d.start_z)).ravel().astype(np.float64)
    x = (d.start_x + S * (d.end_x - d.start_x)).ravel().astype(np.float64)
    y = Y.ravel().astype(np.float64)
    amp = np.full(z.shape, amplitude, dtype=np.float64)
    return z, x, y, amp


__all__ = [
    'Specimen3D',
    'SphericalDefect',
    'CylindricalDefect',
    'PlanarCrack3D',
    'DEFAULT_VOID_BORN_AMP',
    'defect_to_born_scatterers_3d',
]
