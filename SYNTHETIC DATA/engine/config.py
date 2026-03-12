"""
Configuration dataclasses for the NDT simulation engine.
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class MaterialProperties:
    """
    Acoustic properties of a material.

    Attributes:
        name: Material identifier
        c_L: Longitudinal wave speed (m/s)
        c_S: Shear wave speed (m/s)
        density: Mass density (kg/m³)
        attenuation_L: Longitudinal attenuation coefficient (Np/m/MHz)
        attenuation_S: Shear attenuation coefficient (Np/m/MHz)
    """
    name: str
    c_L: float
    c_S: float
    density: float
    attenuation_L: float = 0.0
    attenuation_S: float = 0.0

    @property
    def Z_L(self) -> float:
        """Longitudinal acoustic impedance (Pa·s/m)."""
        return self.density * self.c_L

    @property
    def Z_S(self) -> float:
        """Shear acoustic impedance (Pa·s/m)."""
        return self.density * self.c_S


@dataclass
class ArrayConfig:
    """
    1D linear phased array configuration.

    Attributes:
        num_elements: Number of array elements
        element_pitch: Center-to-center element spacing (m)
        element_width: Active width of each element (m)
        frequency: Center frequency (Hz)
        bandwidth: Fractional bandwidth (e.g. 0.6 = 60%)
        z_position: Depth position of the array surface (m)
    """
    num_elements: int = 64
    element_pitch: float = 0.6e-3
    element_width: float = 0.54e-3
    frequency: float = 10e6
    bandwidth: float = 0.6
    z_position: float = 0.0

    @property
    def wavelength_L(self) -> float:
        """Longitudinal wavelength — requires material context, use with caution."""
        # Default to aluminum for quick estimates
        return 6320.0 / self.frequency

    @property
    def aperture(self) -> float:
        """Total array aperture (m)."""
        return (self.num_elements - 1) * self.element_pitch

    @property
    def element_positions(self) -> np.ndarray:
        """
        X-positions of element centers, centered at x=0.

        Returns:
            (num_elements,) array of lateral positions in meters
        """
        indices = np.arange(self.num_elements)
        return (indices - (self.num_elements - 1) / 2) * self.element_pitch


@dataclass
class SpecimenConfig:
    """
    2D specimen geometry.

    The specimen is a rectangular block with the array on the top surface.
    Coordinate system: z = depth (downward), x = lateral.

    Attributes:
        thickness: Distance from front wall to back wall (m)
        width: Lateral extent of the specimen (m)
        front_wall_z: Z-coordinate of the front wall (m)
    """
    thickness: float = 50e-3     # 50 mm
    width: float = 50e-3         # 50 mm
    front_wall_z: float = 0.0

    @property
    def back_wall_z(self) -> float:
        """Z-coordinate of the back wall."""
        return self.front_wall_z + self.thickness


@dataclass
class AcquisitionConfig:
    """
    FMC acquisition parameters.

    Attributes:
        time_samples: Number of time samples per A-scan
        sampling_frequency: Sampling rate (Hz). None = 4× center frequency
        snr_db: Target signal-to-noise ratio for added noise (dB)
        grain_noise_level: Grain scattering amplitude relative to signal
        add_noise: Whether to add noise to FMC data
    """
    time_samples: int = 2048
    sampling_frequency: Optional[float] = None  # Set from array frequency if None
    snr_db: float = 35.0
    grain_noise_level: float = 0.05
    add_noise: bool = True


@dataclass
class ReconstructionConfig:
    """
    TFM reconstruction parameters.

    Attributes:
        pixel_size: Reconstruction grid pixel size (m)
        z_start: Start depth for reconstruction (m)
        z_end: End depth for reconstruction (m). None = specimen thickness
        x_start: Start lateral position (m). None = -aperture
        x_end: End lateral position (m). None = +aperture
        db_range: Dynamic range for dB display (negative value)
    """
    pixel_size: float = 0.1e-3
    z_start: float = 0.0
    z_end: Optional[float] = None
    x_start: Optional[float] = None
    x_end: Optional[float] = None
    db_range: float = -40.0


@dataclass
class ScanPlanConfig:
    """
    Mechanical scan plan for 3D volume acquisition.

    The 1D array rotates around the centre of the array (origin of the front
    surface) to acquire a stack of 2D B-scans at different azimuthal angles.
    Each scan plane contains the z-axis (depth) and the array direction
    (cos θ, sin θ, 0) in the x-y plane.

    Attributes:
        n_scans:      Number of angular frames to acquire
        theta_start:  First angle (rad).  Default −π/2  (array along −y)
        theta_end:    Last  angle (rad).  Default +π/2  (array along +y)
    """
    n_scans:     int   = 32
    theta_start: float = -np.pi / 2   # rad
    theta_end:   float =  np.pi / 2   # rad

    @property
    def angles(self) -> 'np.ndarray':
        """Rotation angles (rad) for each frame."""
        return np.linspace(self.theta_start, self.theta_end, self.n_scans)

    @property
    def angle_step(self) -> float:
        """Angular step between frames (rad)."""
        if self.n_scans < 2:
            return 0.0
        return float((self.theta_end - self.theta_start) / (self.n_scans - 1))


@dataclass
class SimulationConfig:
    """
    Top-level configuration bundling all simulation parameters.
    """
    material: Optional[MaterialProperties] = None
    couplant: Optional[MaterialProperties] = None
    array: ArrayConfig = field(default_factory=ArrayConfig)
    specimen: SpecimenConfig = field(default_factory=SpecimenConfig)
    acquisition: AcquisitionConfig = field(default_factory=AcquisitionConfig)
    reconstruction: ReconstructionConfig = field(default_factory=ReconstructionConfig)
    scan_plan: Optional[ScanPlanConfig] = None   # None = single 2D scan
    max_bounces: int = 3
    mode_conversion: bool = True
    gel_thickness: float = 0.075e-3  # Gel layer thickness (m). ~0.05-0.1 mm typical.

    def __post_init__(self):
        # Default material: aluminum
        if self.material is None:
            from .materials import ALUMINUM
            self.material = ALUMINUM
        # Default couplant: NDT gel (contact testing, not immersion)
        if self.couplant is None:
            from .materials import NDT_GEL
            self.couplant = NDT_GEL
        # Default sampling frequency
        if self.acquisition.sampling_frequency is None:
            self.acquisition.sampling_frequency = 4 * self.array.frequency
        # Default reconstruction bounds
        if self.reconstruction.z_end is None:
            self.reconstruction.z_end = self.specimen.thickness
        if self.reconstruction.x_start is None:
            self.reconstruction.x_start = -self.specimen.width / 2
        if self.reconstruction.x_end is None:
            self.reconstruction.x_end = self.specimen.width / 2

    @property
    def dt(self) -> float:
        """Time step (s)."""
        assert self.acquisition.sampling_frequency is not None
        return 1.0 / self.acquisition.sampling_frequency

    @property
    def time_axis(self) -> np.ndarray:
        """Time axis array (s)."""
        return np.arange(self.acquisition.time_samples) * self.dt

    def summary(self) -> str:
        """Print a human-readable summary of the configuration."""
        assert self.material is not None
        assert self.couplant is not None
        assert self.acquisition.sampling_frequency is not None
        lines = [
            f"{'='*70}",
            f"NDT SIMULATION CONFIGURATION",
            f"{'='*70}",
            f"  Material: {self.material.name} "
            f"(c_L={self.material.c_L:.0f} m/s, c_S={self.material.c_S:.0f} m/s, "
            f"ρ={self.material.density:.0f} kg/m³)",
            f"  Couplant: {self.couplant.name} "
            f"(c_L={self.couplant.c_L:.0f} m/s)",
            f"  Array: {self.array.num_elements} elements, "
            f"f={self.array.frequency/1e6:.1f} MHz, "
            f"pitch={self.array.element_pitch*1e3:.2f} mm, "
            f"BW={self.array.bandwidth*100:.0f}%",
            f"  Specimen: {self.specimen.thickness*1e3:.1f} mm thick × "
            f"{self.specimen.width*1e3:.1f} mm wide",
            f"  Acquisition: {self.acquisition.time_samples} samples @ "
            f"{self.acquisition.sampling_frequency/1e6:.1f} MHz",
            f"  Max depth: {self.material.c_L * self.time_axis[-1] / 2 * 1e3:.1f} mm",
            f"  Max bounces: {self.max_bounces}, "
            f"Mode conversion: {self.mode_conversion}",
        ]
        if self.scan_plan is not None:
            sp = self.scan_plan
            lines.append(
                f"  3D scan: {sp.n_scans} frames, "
                f"θ=[{np.degrees(sp.theta_start):.1f}°, {np.degrees(sp.theta_end):.1f}°], "
                f"step={np.degrees(sp.angle_step):.2f}°"
            )
        lines.append(f"{'='*70}")
        return "\n".join(lines)
