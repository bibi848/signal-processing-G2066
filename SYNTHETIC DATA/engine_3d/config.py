"""
Configuration dataclasses for the 3D NDT simulation engine.

Coordinate system (matches engine/geometry.py):
    z = depth (downward from array surface)
    x = lateral along the array's long axis
    y = elevation (perpendicular to the long axis, out-of-plane for a 1D array)

The 2D matrix array lies in the z = z_position plane and is described by a
tensor product of x-pitch and y-pitch. Element index ordering is flat, row-major
across (iy, ix).
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from engine.config import MaterialProperties


@dataclass
class ArrayConfig3D:
    """
    2D matrix phased array configuration.

    Attributes:
        n_elements_x: Number of elements along x (ignored if custom_positions set)
        n_elements_y: Number of elements along y (ignored if custom_positions set)
        pitch_x: Centre-to-centre spacing along x (m)
        pitch_y: Centre-to-centre spacing along y (m)
        element_width_x: Active width of each element along x (m)
        element_width_y: Active width of each element along y (m)
        frequency: Centre frequency (Hz)
        bandwidth: Fractional bandwidth (e.g. 0.6 = 60%)
        z_position: Depth position of the array surface (m)
        custom_positions: Optional (N, 2) array of (x, y) element centres.
            If set, overrides the rectangular tensor-product grid defined by
            n_elements_x/y and pitch_x/y. z is taken from z_position.
    """
    n_elements_x: int = 16
    n_elements_y: int = 16
    pitch_x: float = 0.6e-3
    pitch_y: float = 0.6e-3
    element_width_x: float = 0.54e-3
    element_width_y: float = 0.54e-3
    frequency: float = 10e6
    bandwidth: float = 0.6
    z_position: float = 0.0
    custom_positions: Optional[np.ndarray] = None

    @property
    def n_elements_total(self) -> int:
        if self.custom_positions is not None:
            return int(self.custom_positions.shape[0])
        return self.n_elements_x * self.n_elements_y

    @property
    def aperture_x(self) -> float:
        if self.custom_positions is not None:
            xs = self.custom_positions[:, 0]
            return float(xs.max() - xs.min())
        return (self.n_elements_x - 1) * self.pitch_x

    @property
    def aperture_y(self) -> float:
        if self.custom_positions is not None:
            ys = self.custom_positions[:, 1]
            return float(ys.max() - ys.min())
        return (self.n_elements_y - 1) * self.pitch_y

    def element_positions(self) -> np.ndarray:
        """
        World positions of every element centre, row-major in (iy, ix) for the
        default rectangular grid. With custom_positions set, positions are
        returned in their CSV order.

        Returns:
            (n_elements_total, 3) array with columns (z, x, y).
        """
        if self.custom_positions is not None:
            xy = np.asarray(self.custom_positions, dtype=float)
            flat_z = np.full(xy.shape[0], self.z_position)
            return np.stack([flat_z, xy[:, 0], xy[:, 1]], axis=1)
        ix = np.arange(self.n_elements_x)
        iy = np.arange(self.n_elements_y)
        x = (ix - (self.n_elements_x - 1) / 2.0) * self.pitch_x
        y = (iy - (self.n_elements_y - 1) / 2.0) * self.pitch_y
        xx, yy = np.meshgrid(x, y, indexing='xy')  # shape (n_y, n_x)
        flat_x = xx.ravel()
        flat_y = yy.ravel()
        flat_z = np.full_like(flat_x, self.z_position)
        return np.stack([flat_z, flat_x, flat_y], axis=1)


@dataclass
class SpecimenConfig3D:
    """
    3D rectangular specimen. Array sits on the front wall (z = front_wall_z).

    Attributes:
        thickness: Extent along z (m)
        width: Extent along x (m)
        depth: Extent along y (m)
        front_wall_z: Z-coordinate of the front wall (m)
    """
    thickness: float = 50e-3
    width: float = 50e-3
    depth: float = 50e-3
    front_wall_z: float = 0.0

    @property
    def back_wall_z(self) -> float:
        return self.front_wall_z + self.thickness


@dataclass
class AcquisitionConfig3D:
    """FMC acquisition parameters (identical semantics to the 2D engine)."""
    time_samples: int = 2048
    sampling_frequency: Optional[float] = None
    snr_db: float = 35.0
    grain_noise_level: float = 0.05
    add_noise: bool = True
    filter_alpha: float = 1.0
    hanning_bool: bool = False


@dataclass
class ReconstructionConfig3D:
    """
    Volumetric TFM reconstruction parameters. Bounds default to the full
    specimen extent in the respective axis.
    """
    pixel_size: float = 0.3e-3
    z_start: float = 0.0
    z_end: Optional[float] = None
    x_start: Optional[float] = None
    x_end: Optional[float] = None
    y_start: Optional[float] = None
    y_end: Optional[float] = None
    db_range: float = -40.0


@dataclass
class SimulationConfig3D:
    """
    Top-level configuration for the 3D engine.
    """
    material: Optional[MaterialProperties] = None
    couplant: Optional[MaterialProperties] = None
    array: ArrayConfig3D = field(default_factory=ArrayConfig3D)
    specimen: SpecimenConfig3D = field(default_factory=SpecimenConfig3D)
    acquisition: AcquisitionConfig3D = field(default_factory=AcquisitionConfig3D)
    reconstruction: ReconstructionConfig3D = field(default_factory=ReconstructionConfig3D)
    gel_thickness: float = 0.075e-3

    def __post_init__(self):
        if self.material is None:
            from engine.materials import ALUMINUM
            self.material = ALUMINUM
        if self.couplant is None:
            from engine.materials import NDT_GEL
            self.couplant = NDT_GEL
        if self.acquisition.sampling_frequency is None:
            self.acquisition.sampling_frequency = 4 * self.array.frequency
        r = self.reconstruction
        if r.z_end is None:
            r.z_end = self.specimen.thickness
        if r.x_start is None:
            r.x_start = -self.specimen.width / 2
        if r.x_end is None:
            r.x_end = self.specimen.width / 2
        if r.y_start is None:
            r.y_start = -self.specimen.depth / 2
        if r.y_end is None:
            r.y_end = self.specimen.depth / 2

    @property
    def dt(self) -> float:
        assert self.acquisition.sampling_frequency is not None
        return 1.0 / self.acquisition.sampling_frequency

    @property
    def time_axis(self) -> np.ndarray:
        return np.arange(self.acquisition.time_samples) * self.dt

    def summary(self) -> str:
        assert self.material is not None
        assert self.couplant is not None
        assert self.acquisition.sampling_frequency is not None
        a = self.array
        s = self.specimen
        if a.custom_positions is not None:
            layout = (f"  Matrix array (custom layout): {a.n_elements_total} "
                      f"elements, f={a.frequency/1e6:.1f} MHz, "
                      f"pitch=({a.pitch_x*1e3:.2f}, {a.pitch_y*1e3:.2f}) mm")
        else:
            layout = (f"  Matrix array: {a.n_elements_x}×{a.n_elements_y} = "
                      f"{a.n_elements_total} elements, "
                      f"f={a.frequency/1e6:.1f} MHz, "
                      f"pitch=({a.pitch_x*1e3:.2f}, {a.pitch_y*1e3:.2f}) mm")
        return "\n".join([
            "=" * 70,
            "3D NDT SIMULATION CONFIGURATION",
            "=" * 70,
            f"  Material: {self.material.name} "
            f"(c_L={self.material.c_L:.0f} m/s, ρ={self.material.density:.0f} kg/m³)",
            f"  Couplant: {self.couplant.name}",
            layout,
            f"  Aperture: {a.aperture_x*1e3:.2f} × {a.aperture_y*1e3:.2f} mm",
            f"  Specimen: {s.thickness*1e3:.1f} (z) × {s.width*1e3:.1f} (x) × "
            f"{s.depth*1e3:.1f} (y) mm",
            f"  Acquisition: {self.acquisition.time_samples} samples @ "
            f"{self.acquisition.sampling_frequency/1e6:.1f} MHz",
            f"  Max depth: {self.material.c_L * self.time_axis[-1] / 2 * 1e3:.1f} mm",
            f"  FMC shape: ({a.n_elements_total}, {a.n_elements_total}, "
            f"{self.acquisition.time_samples})",
            "=" * 70,
        ])
