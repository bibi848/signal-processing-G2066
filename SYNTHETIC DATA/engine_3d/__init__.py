"""
Born-only 3D ultrasonic NDT synthetic data engine with a 2D matrix array.

Parallel sibling of the 2D engine in ../engine/. Shares materials, waveforms,
and isotropic attenuation from the 2D engine; everything geometric is 3D.

Modules:
    config        — Dataclass configurations for the 3D simulation
    propagation   — 1/r spreading, rectangular-element 2D-sinc directivity
    geometry      — 3D defect point-cloud extraction (reuses engine.geometry)
    voxel_volume  — 3D Born scatterer extraction from a voxel impedance field
    fmc_engine    — Born-only 3D FMC simulator for a 2D matrix array
"""

from engine.materials import (
    ALUMINUM, STEEL_MILD, STEEL_STAINLESS, COPPER, WATER, NDT_GEL, AIR,
    acoustic_impedance, wave_speed, attenuation_coefficient,
)
from engine.waveforms import (
    Arrival, gabor_pulse, synthesize_ascan, synthesize_ascan_vectorized,
)

from .config import (
    ArrayConfig3D, SpecimenConfig3D, AcquisitionConfig3D,
    ReconstructionConfig3D, SimulationConfig3D,
)
from .propagation import (
    geometric_spreading_3d, geometric_spreading_3d_array,
    element_directivity_3d_array, material_attenuation,
    material_attenuation_array,
)
from .geometry import (
    Specimen3D, SphericalDefect, CylindricalDefect, PlanarCrack3D,
    DEFAULT_VOID_BORN_AMP, defect_to_born_scatterers_3d,
)
from .voxel_volume import VoxelVolume3D, extract_born_scatterers_3d
from .fmc_engine import FMCEngine3D


__all__ = [
    # Materials / waveforms (re-exports)
    'ALUMINUM', 'STEEL_MILD', 'STEEL_STAINLESS', 'COPPER',
    'WATER', 'NDT_GEL', 'AIR',
    'acoustic_impedance', 'wave_speed', 'attenuation_coefficient',
    'Arrival', 'gabor_pulse', 'synthesize_ascan', 'synthesize_ascan_vectorized',
    # Config
    'ArrayConfig3D', 'SpecimenConfig3D', 'AcquisitionConfig3D',
    'ReconstructionConfig3D', 'SimulationConfig3D',
    # Physics
    'geometric_spreading_3d', 'geometric_spreading_3d_array',
    'element_directivity_3d_array', 'material_attenuation',
    'material_attenuation_array',
    # Geometry
    'Specimen3D', 'SphericalDefect', 'CylindricalDefect', 'PlanarCrack3D',
    'DEFAULT_VOID_BORN_AMP', 'defect_to_born_scatterers_3d',
    # Voxel volume
    'VoxelVolume3D', 'extract_born_scatterers_3d',
    # FMC
    'FMCEngine3D',
]
