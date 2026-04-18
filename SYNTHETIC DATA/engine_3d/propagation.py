"""
Wave propagation physics for the 3D engine.

Differences from the 2D engine:
    - Geometric spreading is 1/r (spherical), not 1/√r (cylindrical)
    - Element directivity is the product of two sinc patterns over the
      x- and y-apertures of the rectangular matrix element.

material_attenuation is reused from the 2D engine unchanged (isotropic).
"""

import numpy as np

from engine.propagation import (
    material_attenuation,
    material_attenuation_array,
)


def geometric_spreading_3d(distance: float) -> float:
    """
    Geometric spreading loss for 3D (spherical) wave propagation: 1/r.

    Args:
        distance: Propagation distance (m)

    Returns:
        Amplitude factor (dimensionless)
    """
    return 1.0 / max(distance, 1e-10)


def geometric_spreading_3d_array(distances: np.ndarray) -> np.ndarray:
    """Vectorised version of geometric_spreading_3d."""
    return 1.0 / np.maximum(distances, 1e-10)


def element_directivity_3d_array(
    theta_x: np.ndarray,
    theta_y: np.ndarray,
    element_width_x: float,
    element_width_y: float,
    wavelength: float,
) -> np.ndarray:
    """
    Rectangular-element directivity pattern in 3D.

    A rectangular element of width (w_x, w_y) radiates with
        D(θ_x, θ_y) = sinc(π·w_x·sin(θ_x)/λ) · sinc(π·w_y·sin(θ_y)/λ)

    where sinc here is the unnormalised sin(x)/x.

    Args:
        theta_x: Angle from element normal in the x–z plane (rad)
        theta_y: Angle from element normal in the y–z plane (rad)
        element_width_x: Element width along x (m)
        element_width_y: Element width along y (m)
        wavelength: Wavelength in the medium (m)

    Returns:
        Directivity factor in [0, 1] with the same shape as theta_x.
    """
    return (
        _sinc_unnormalised(theta_x, element_width_x, wavelength)
        * _sinc_unnormalised(theta_y, element_width_y, wavelength)
    )


def _sinc_unnormalised(theta: np.ndarray, element_width: float,
                        wavelength: float) -> np.ndarray:
    arg = np.pi * element_width * np.sin(theta) / wavelength
    result = np.ones_like(arg)
    mask = np.abs(arg) > 1e-10
    result[mask] = np.sin(arg[mask]) / arg[mask]
    return result


__all__ = [
    'geometric_spreading_3d',
    'geometric_spreading_3d_array',
    'element_directivity_3d_array',
    'material_attenuation',
    'material_attenuation_array',
]
