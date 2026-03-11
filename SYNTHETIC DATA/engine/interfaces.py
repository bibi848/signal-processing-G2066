"""
Acoustic interface physics: Snell's law and Fresnel reflection/transmission coefficients.

Handles wave behaviour at boundaries between different media
(e.g. couplant→specimen, specimen→air, specimen→void).
"""

import numpy as np
from .config import MaterialProperties


def reflection_coefficient_normal(Z1: float, Z2: float) -> float:
    """
    Pressure reflection coefficient at normal incidence.

    R = (Z2 - Z1) / (Z2 + Z1)

    Args:
        Z1: Impedance of incident medium (Pa·s/m)
        Z2: Impedance of transmitted medium (Pa·s/m)

    Returns:
        Reflection coefficient (-1 to +1).
        R < 0 means phase inversion on reflection.
    """
    return (Z2 - Z1) / (Z2 + Z1)


def transmission_coefficient_normal(Z1: float, Z2: float) -> float:
    """
    Pressure transmission coefficient at normal incidence.

    T = 2·Z2 / (Z2 + Z1)
    """
    return 2.0 * Z2 / (Z2 + Z1)


def snell_angle(theta_i: float, c_i: float, c_t: float) -> float:
    """
    Compute refracted/reflected angle using Snell's law.

    sin(θ_t)/c_t = sin(θ_i)/c_i

    Args:
        theta_i: Incident angle (radians)
        c_i: Wave speed in incident medium (m/s)
        c_t: Wave speed in transmitted medium (m/s)

    Returns:
        Refracted angle (radians), or np.nan if total internal reflection.
    """
    sin_t = (c_t / c_i) * np.sin(theta_i)
    if abs(sin_t) > 1.0:
        return np.nan  # Total internal reflection
    return np.arcsin(sin_t)


def fresnel_coefficients_fluid_solid(theta_i: float,
                                      rho1: float, c1: float,
                                      rho2: float, c2_L: float, c2_S: float):
    """
    Reflection/transmission coefficients at a fluid-solid interface.

    Used for: couplant (water) → specimen (aluminum) interface.
    The incident medium supports only longitudinal waves;
    the transmitted medium supports both L and S waves.

    Args:
        theta_i: Incident angle in fluid (radians)
        rho1: Fluid density (kg/m³)
        c1: Fluid longitudinal speed (m/s)
        rho2: Solid density (kg/m³)
        c2_L: Solid longitudinal speed (m/s)
        c2_S: Solid shear speed (m/s)

    Returns:
        (R, T_L, T_S) — reflection coefficient (fluid side),
        transmission coefficient for L-wave, transmission coefficient for S-wave
    """
    sin_i = np.sin(theta_i)
    cos_i = np.cos(theta_i)

    # Snell's law for transmitted angles
    sin_L = (c2_L / c1) * sin_i
    sin_S = (c2_S / c1) * sin_i

    # Check for total reflection
    if abs(sin_L) > 1.0 and abs(sin_S) > 1.0:
        return -1.0, 0.0, 0.0

    cos_L = np.sqrt(max(0, 1.0 - sin_L**2)) if abs(sin_L) <= 1.0 else 0.0
    cos_S = np.sqrt(max(0, 1.0 - sin_S**2)) if abs(sin_S) <= 1.0 else 0.0

    Z1 = rho1 * c1 / cos_i if cos_i > 1e-10 else 1e30
    Z2_L = rho2 * c2_L / cos_L if cos_L > 1e-10 else 1e30
    Z2_S = rho2 * c2_S / cos_S if cos_S > 1e-10 else 1e30

    # Simplified: treat as effective impedance problem
    # For more accuracy, use full Zoeppritz (Milestone 4)
    Z2_eff = Z2_L  # Dominant transmitted mode at small angles

    R = (Z2_eff - Z1) / (Z2_eff + Z1)
    T_L = 1.0 + R if abs(sin_L) <= 1.0 else 0.0
    T_S = 0.0  # Mode conversion — refined in Milestone 4

    return R, T_L, T_S


def fresnel_solid_free_surface(theta_i: float, c_L: float, c_S: float,
                                incident_mode: str = 'L'):
    """
    Reflection coefficients at a solid-free surface (solid→air/void).

    At a free surface, the stress must vanish. This gives rise to
    mode conversion: an incident L-wave generates reflected L and S waves.

    Args:
        theta_i: Incident angle (radians)
        c_L: Longitudinal speed in solid (m/s)
        c_S: Shear speed in solid (m/s)
        incident_mode: 'L' or 'S'

    Returns:
        (R_LL, R_LS) for incident L, or (R_SL, R_SS) for incident S.
        These are displacement amplitude reflection coefficients.
    """
    sin_i = np.sin(theta_i)
    cos_i = np.cos(theta_i)

    if incident_mode == 'L':
        # Incident L-wave
        sin_L = sin_i
        cos_L = cos_i
        sin_S = (c_S / c_L) * sin_i
        if abs(sin_S) > 1.0:
            # Beyond critical angle for mode conversion
            return -1.0, 0.0
        cos_S = np.sqrt(1.0 - sin_S**2)

        # Free surface reflection coefficients (Auld's formulation)
        p = sin_S
        q_L = cos_L
        q_S = cos_S

        D = (p**2 - q_S**2)**2 + 4 * p**2 * q_L * q_S
        if abs(D) < 1e-30:
            return -1.0, 0.0

        R_LL = ((p**2 - q_S**2)**2 - 4 * p**2 * q_L * q_S) / D
        R_LS = 4 * p * q_L * (p**2 - q_S**2) / D

        return R_LL, R_LS

    elif incident_mode == 'S':
        # Incident S-wave
        sin_S = sin_i
        cos_S = cos_i
        sin_L = (c_L / c_S) * sin_i
        if abs(sin_L) > 1.0:
            # Beyond critical angle — total reflection as S
            return 0.0, -1.0
        cos_L = np.sqrt(1.0 - sin_L**2)

        p = sin_S
        q_L = cos_L
        q_S = cos_S

        D = (p**2 - q_S**2)**2 + 4 * p**2 * q_L * q_S
        if abs(D) < 1e-30:
            return 0.0, -1.0

        R_SL = 4 * p * q_S * (p**2 - q_S**2) / D
        R_SS = ((p**2 - q_S**2)**2 - 4 * p**2 * q_L * q_S) / D

        return R_SL, R_SS

    else:
        raise ValueError(f"Unknown incident mode '{incident_mode}'")
