"""
Material property presets and acoustic impedance calculations.
"""

from .config import MaterialProperties


# --- Common NDT materials ---

ALUMINUM = MaterialProperties(
    name="Aluminum 6061-T6",
    c_L=6320.0,      # m/s longitudinal
    c_S=3130.0,      # m/s shear
    density=2700.0,   # kg/m³
    attenuation_L=0.03,  # Np/m/MHz
    attenuation_S=0.05,  # Np/m/MHz (shear attenuates more)
)

STEEL_MILD = MaterialProperties(
    name="Mild Steel",
    c_L=5960.0,
    c_S=3240.0,
    density=7850.0,
    attenuation_L=0.04,
    attenuation_S=0.07,
)

STEEL_STAINLESS = MaterialProperties(
    name="Stainless Steel 304",
    c_L=5790.0,
    c_S=3100.0,
    density=8000.0,
    attenuation_L=0.06,
    attenuation_S=0.10,
)

WATER = MaterialProperties(
    name="Water (20°C)",
    c_L=1480.0,
    c_S=0.0,          # No shear waves in fluids
    density=1000.0,
    attenuation_L=0.002,
    attenuation_S=0.0,
)

NDT_GEL = MaterialProperties(
    name="NDT Coupling Gel",
    # Gel is viscous water-based couplant — c_L close to water but slightly higher,
    # higher density due to additives, and much higher attenuation.
    # The gel layer is typically < 0.1 mm, so it acts as a coupling medium
    # rather than a wave propagation medium (layer << wavelength at 5-10 MHz).
    c_L=1500.0,      # m/s (similar to water)
    c_S=0.0,          # No shear waves in gel
    density=1050.0,   # kg/m³ (slightly denser than water)
    attenuation_L=0.05,   # Np/m/MHz (higher than water due to viscosity)
    attenuation_S=0.0,
)

AIR = MaterialProperties(
    name="Air",
    c_L=343.0,
    c_S=0.0,
    density=1.225,
    attenuation_L=1.0,
    attenuation_S=0.0,
)


__all__ = ['ALUMINUM', 'STEEL_MILD', 'STEEL_STAINLESS', 'WATER', 'NDT_GEL', 'AIR',
           'acoustic_impedance', 'wave_speed', 'attenuation_coefficient']


def acoustic_impedance(material: MaterialProperties, mode: str = 'L') -> float:
    """
    Calculate acoustic impedance Z = ρ × c.

    Args:
        material: Material properties
        mode: 'L' for longitudinal, 'S' for shear

    Returns:
        Acoustic impedance in Pa·s/m (Rayl)
    """
    if mode == 'L':
        return material.density * material.c_L
    elif mode == 'S':
        return material.density * material.c_S
    else:
        raise ValueError(f"Unknown mode '{mode}', use 'L' or 'S'")


def wave_speed(material: MaterialProperties, mode: str = 'L') -> float:
    """
    Get wave speed for the given mode.

    Args:
        material: Material properties
        mode: 'L' for longitudinal, 'S' for shear

    Returns:
        Wave speed in m/s
    """
    if mode == 'L':
        return material.c_L
    elif mode == 'S':
        return material.c_S
    else:
        raise ValueError(f"Unknown mode '{mode}', use 'L' or 'S'")


def attenuation_coefficient(material: MaterialProperties, mode: str = 'L') -> float:
    """
    Get attenuation coefficient for the given mode.

    Args:
        material: Material properties
        mode: 'L' for longitudinal, 'S' for shear

    Returns:
        Attenuation in Np/m/MHz
    """
    if mode == 'L':
        return material.attenuation_L
    elif mode == 'S':
        return material.attenuation_S
    else:
        raise ValueError(f"Unknown mode '{mode}', use 'L' or 'S'")
