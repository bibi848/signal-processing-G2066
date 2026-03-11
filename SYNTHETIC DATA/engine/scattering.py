"""
Kirchhoff surface scattering model for defect echoes.

Implements the Kirchhoff (physical optics) approximation where the
scattered field is computed by integrating contributions from each
surface element of the defect boundary.
"""

import numpy as np
from typing import Tuple
from .geometry import Defect2D
from .propagation import (
    geometric_spreading_2d_array,
    material_attenuation_array,
    element_directivity_array,
)


def kirchhoff_scattering_2d(defect: Defect2D,
                             elem_positions: np.ndarray,
                             frequency: float,
                             c_L: float,
                             attenuation_L: float,
                             element_width: float,
                             n_surface_points: int = 120
                             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Kirchhoff surface scattering for all TX-RX pairs.

    Discretizes the defect surface and computes the scattered field
    contribution from each surface element for every TX-RX combination.

    Args:
        defect: Defect geometry with discretize_surface() method
        elem_positions: (num_el, 2) array of element [z, x] positions
        frequency: Center frequency (Hz)
        c_L: Longitudinal wave speed (m/s)
        attenuation_L: Attenuation coefficient (Np/m/MHz)
        element_width: Element width (m) for directivity
        n_surface_points: Number of surface discretization points

    Returns:
        tof: (num_el, num_el, n_pts) time-of-flight for each contribution
        amplitude: (num_el, num_el, n_pts) amplitude factor
        phase: (num_el, num_el, n_pts) phase (radians)
    """
    wavelength = c_L / frequency
    k = 2.0 * np.pi * frequency / c_L
    num_el = len(elem_positions)

    # Discretize defect surface
    surface_pts, surface_normals = defect.discretize_surface(n_surface_points)
    n_pts = len(surface_pts)

    # --- Distance vectors ---
    # d_tx[el, pt, :] = elem_pos[el] - surface_pts[pt]
    d_tx = elem_positions[:, np.newaxis, :] - surface_pts[np.newaxis, :, :]
    dist_tx = np.linalg.norm(d_tx, axis=2)  # (num_el, n_pts)
    dist_rx = dist_tx  # Same geometry (elements are both TX and RX)

    # Total distance: (num_el, num_el, n_pts)
    dist_total = dist_tx[:, np.newaxis, :] + dist_rx[np.newaxis, :, :]
    tof = dist_total / c_L

    # --- Geometric spreading: 1/√(d_tx × d_rx) ---
    spread = 1.0 / np.sqrt(
        np.maximum(dist_tx[:, np.newaxis, :] * dist_rx[np.newaxis, :, :], 1e-20)
    )

    # --- Material attenuation ---
    atten = np.exp(-attenuation_L * (frequency / 1e6) * dist_total)

    # --- Obliquity: cos(θ) at surface point ---
    dir_to_surface = -d_tx / np.maximum(dist_tx[:, :, np.newaxis], 1e-15)
    cos_theta = np.abs(np.sum(
        dir_to_surface * (-surface_normals[np.newaxis, :, :]), axis=2
    ))  # (num_el, n_pts)
    obliquity = cos_theta[:, np.newaxis, :] * cos_theta[np.newaxis, :, :]

    # --- Element directivity ---
    theta_el = np.arctan2(np.abs(d_tx[:, :, 1]), np.abs(d_tx[:, :, 0]))
    dir_factor = element_directivity_array(theta_el, element_width, wavelength)
    directivity = dir_factor[:, np.newaxis, :] * dir_factor[np.newaxis, :, :]

    # --- Surface element arc length ---
    if n_pts > 1:
        ds = np.linalg.norm(np.diff(surface_pts, axis=0), axis=1)
        ds = np.append(ds, ds[-1])
    else:
        ds = np.ones(1)

    # --- Kirchhoff integral factor ---
    kirchhoff_factor = np.sqrt(k / (2.0 * np.pi))

    # --- Combined amplitude ---
    amplitude = (kirchhoff_factor * spread * atten * obliquity
                * directivity * ds[np.newaxis, np.newaxis, :])

    # Phase: π for void (free surface) reflection
    phase = np.full_like(amplitude, np.pi)

    return tof, amplitude, phase
