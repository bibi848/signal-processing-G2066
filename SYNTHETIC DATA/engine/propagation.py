"""
Wave propagation physics: geometric spreading, material attenuation,
and element directivity for 2D simulation.
"""

import numpy as np
from .config import MaterialProperties


def geometric_spreading_2d(distance: float) -> float:
    """
    Geometric spreading loss for 2D (cylindrical) wave propagation.

    In 2D, energy spreads over a cylinder → amplitude decays as 1/√r.
    (In 3D it would be 1/r for spherical spreading.)

    Args:
        distance: Propagation distance (m)

    Returns:
        Amplitude factor (dimensionless)
    """
    return 1.0 / np.sqrt(max(distance, 1e-10))


def geometric_spreading_2d_array(distances: np.ndarray) -> np.ndarray:
    """Vectorized version of geometric_spreading_2d."""
    return 1.0 / np.sqrt(np.maximum(distances, 1e-10))


def material_attenuation(distance: float, frequency_hz: float,
                         alpha: float) -> float:
    """
    Frequency-dependent material attenuation.

    Amplitude decay: exp(-α·f·d)

    Args:
        distance: Total propagation distance (m)
        frequency_hz: Wave frequency (Hz)
        alpha: Attenuation coefficient (Np/m/MHz)

    Returns:
        Amplitude factor (0 to 1)
    """
    freq_mhz = frequency_hz / 1e6
    return np.exp(-alpha * freq_mhz * distance)


def material_attenuation_array(distances: np.ndarray, frequency_hz: float,
                                alpha: float) -> np.ndarray:
    """Vectorized version of material_attenuation."""
    freq_mhz = frequency_hz / 1e6
    return np.exp(-alpha * freq_mhz * distances)


def element_directivity(theta: float, element_width: float,
                        wavelength: float) -> float:
    """
    Element directivity pattern (far-field approximation).

    A rectangular element of width w radiates with a sinc directivity:
        D(θ) = sinc(π·w·sin(θ)/λ)

    Args:
        theta: Angle from element normal (radians)
        element_width: Element width (m)
        wavelength: Wavelength in the medium (m)

    Returns:
        Directivity factor (0 to 1)
    """
    arg = np.pi * element_width * np.sin(theta) / wavelength
    if abs(arg) < 1e-10:
        return 1.0
    return abs(np.sin(arg) / arg)


def element_directivity_array(thetas: np.ndarray, element_width: float,
                               wavelength: float) -> np.ndarray:
    """Vectorized element directivity."""
    arg = np.pi * element_width * np.sin(thetas) / wavelength
    result = np.ones_like(arg)
    mask = np.abs(arg) > 1e-10
    result[mask] = np.abs(np.sin(arg[mask]) / arg[mask])
    return result


def incidence_angle(source: np.ndarray, target: np.ndarray,
                    normal: np.ndarray = None) -> float:
    """
    Calculate the angle of incidence of a ray hitting a surface.

    Args:
        source: Ray origin [z, x]
        target: Point on the surface [z, x]
        normal: Surface normal at target [nz, nx]. Default: vertical [1, 0]

    Returns:
        Angle in radians (0 = normal incidence, π/2 = grazing)
    """
    if normal is None:
        normal = np.array([1.0, 0.0])  # Default: horizontal surface

    direction = target - source
    dist = np.linalg.norm(direction)
    if dist < 1e-15:
        return 0.0
    direction /= dist

    # Angle between ray direction and inward normal (-normal)
    cos_theta = abs(np.dot(direction, -normal))
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.arccos(cos_theta)
