"""
Specimen and defect geometry for 2D and 3D Born-only NDT simulation.

2D coordinate system:
    z = depth (downward from array surface, 0 = front wall)
    x = lateral (centered at 0, along array axis)

3D coordinate system (adds elevation axis):
    z = depth, x = lateral (along array length), y = elevation (mechanical
    scan direction). 3D defects implement slice_at_y / slice_at_angle to
    return a 2D cross-section. Each 2D defect knows how to emit a Born
    point-scatterer cloud (.to_born_scatterers()).
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional
from abc import ABC, abstractmethod


# Born amplitude for a void in a solid (R ≈ -1, amp = ΔZ/(2Z₀) ≈ -0.5).
# Same convention used by the voxel-volume defect-burn-in pipeline.
DEFAULT_VOID_BORN_AMP = -0.5


@dataclass
class Specimen2D:
    """2D rectangular specimen with front and back walls."""
    thickness: float
    width: float
    front_wall_z: float = 0.0

    @property
    def back_wall_z(self) -> float:
        return self.front_wall_z + self.thickness


# --- 2D defect base class ---

@dataclass
class Defect2D(ABC):
    """Abstract base class for 2D defect geometries."""

    @abstractmethod
    def to_born_scatterers(self,
                            n_points: int = 120,
                            amplitude: float = DEFAULT_VOID_BORN_AMP
                            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return surface point cloud as (z, x, amp) arrays of length n_points."""
        ...


@dataclass
class CircularDefect(Defect2D):
    """Circular void (2D cross-section of a side-drilled hole or pore)."""
    center_z: float
    center_x: float
    radius: float

    @property
    def center(self) -> np.ndarray:
        return np.array([self.center_z, self.center_x])

    def to_born_scatterers(self,
                            n_points: int = 120,
                            amplitude: float = DEFAULT_VOID_BORN_AMP
                            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
        z = self.center_z + self.radius * np.cos(angles)
        x = self.center_x + self.radius * np.sin(angles)
        amp = np.full(n_points, amplitude, dtype=np.float64)
        return z, x, amp


@dataclass
class CrackDefect(Defect2D):
    """Crack modeled as a line segment (zero-width planar reflector)."""
    start_z: float
    start_x: float
    end_z: float
    end_x: float

    @property
    def start(self) -> np.ndarray:
        return np.array([self.start_z, self.start_x])

    @property
    def end(self) -> np.ndarray:
        return np.array([self.end_z, self.end_x])

    @property
    def center(self) -> np.ndarray:
        return (self.start + self.end) / 2.0

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.end - self.start))

    def to_born_scatterers(self,
                            n_points: int = 120,
                            amplitude: float = DEFAULT_VOID_BORN_AMP
                            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        s = np.linspace(0, 1, n_points)
        z = self.start_z + s * (self.end_z - self.start_z)
        x = self.start_x + s * (self.end_x - self.start_x)
        amp = np.full(n_points, amplitude, dtype=np.float64)
        return z, x, amp


@dataclass
class FlatBottomHole(Defect2D):
    """Flat-bottom hole — flat disc reflector, modelled in 2D as a horizontal segment."""
    center_z: float
    center_x: float
    width: float

    @property
    def center(self) -> np.ndarray:
        return np.array([self.center_z, self.center_x])

    def to_born_scatterers(self,
                            n_points: int = 120,
                            amplitude: float = DEFAULT_VOID_BORN_AMP
                            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = np.linspace(self.center_x - self.width / 2,
                        self.center_x + self.width / 2,
                        n_points)
        z = np.full(n_points, self.center_z, dtype=np.float64)
        amp = np.full(n_points, amplitude, dtype=np.float64)
        return z, x, amp


# ---------------------------------------------------------------------------
# 3D geometry
# ---------------------------------------------------------------------------

@dataclass
class Specimen3D:
    """3D rectangular specimen. The 1D array sits on the front wall (z=0)."""
    thickness: float
    width: float
    depth: float
    front_wall_z: float = 0.0

    @property
    def back_wall_z(self) -> float:
        return self.front_wall_z + self.thickness

    def to_2d(self) -> 'Specimen2D':
        return Specimen2D(
            thickness=self.thickness,
            width=self.width,
            front_wall_z=self.front_wall_z,
        )

    def y_positions(self, n_scans: int) -> np.ndarray:
        return np.linspace(-self.depth / 2, self.depth / 2, n_scans)


class Defect3D(ABC):
    """Abstract base for 3D defects. See slice_at_y / slice_at_angle."""

    @abstractmethod
    def slice_at_y(self, y: float) -> Optional[Defect2D]:
        ...

    @abstractmethod
    def slice_at_angle(self, theta: float,
                       dy_offset: float = 0.0) -> Optional[Defect2D]:
        ...


@dataclass
class SphericalDefect(Defect3D):
    """Spherical void (e.g. pore, gas pocket)."""
    center_z: float
    center_x: float
    center_y: float
    radius: float

    @property
    def center(self) -> np.ndarray:
        return np.array([self.center_z, self.center_x, self.center_y])

    def slice_at_y(self, y: float) -> Optional[CircularDefect]:
        dy = y - self.center_y
        r_sq = self.radius ** 2 - dy ** 2
        if r_sq <= 0.0:
            return None
        return CircularDefect(
            center_z=self.center_z,
            center_x=self.center_x,
            radius=float(np.sqrt(r_sq)),
        )

    def slice_at_angle(self, theta: float,
                       dy_offset: float = 0.0) -> Optional[CircularDefect]:
        d = (-self.center_x * np.sin(theta)
             + self.center_y * np.cos(theta)
             - dy_offset)
        r_sq = self.radius ** 2 - d ** 2
        if r_sq <= 0.0:
            return None
        L = self.center_x * np.cos(theta) + self.center_y * np.sin(theta)
        return CircularDefect(
            center_z=self.center_z,
            center_x=float(L),
            radius=float(np.sqrt(r_sq)),
        )


@dataclass
class CylindricalDefect(Defect3D):
    """Cylindrical void with axis along the elevation (y) direction."""
    center_z: float
    center_x: float
    radius: float
    y_start: float
    y_end: float

    @property
    def center_y(self) -> float:
        return (self.y_start + self.y_end) / 2.0

    @property
    def center(self) -> np.ndarray:
        return np.array([self.center_z, self.center_x, self.center_y])

    def slice_at_y(self, y: float) -> Optional[CircularDefect]:
        if not (self.y_start <= y <= self.y_end):
            return None
        return CircularDefect(
            center_z=self.center_z,
            center_x=self.center_x,
            radius=self.radius,
        )

    def slice_at_angle(self, theta: float,
                       dy_offset: float = 0.0) -> Optional[CircularDefect]:
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        if abs(cos_t) < 1e-9:
            y_eff = (self.y_start + self.y_end) / 2.0
        else:
            y_star = (self.center_x * sin_t + dy_offset) / cos_t
            y_eff = float(np.clip(y_star, self.y_start, self.y_end))
        d = -self.center_x * sin_t + y_eff * cos_t - dy_offset
        r_sq = self.radius ** 2 - d ** 2
        if r_sq <= 0.0:
            return None
        L = self.center_x * cos_t + y_eff * sin_t
        return CircularDefect(
            center_z=self.center_z,
            center_x=float(L),
            radius=float(np.sqrt(r_sq)),
        )


@dataclass
class PlanarCrack3D(Defect3D):
    """Planar crack in the z-x plane, finite extent in elevation (y)."""
    start_z: float
    start_x: float
    end_z: float
    end_x: float
    y_start: float
    y_end: float

    def slice_at_y(self, y: float) -> Optional[CrackDefect]:
        if not (self.y_start <= y <= self.y_end):
            return None
        return CrackDefect(
            start_z=self.start_z,
            start_x=self.start_x,
            end_z=self.end_z,
            end_x=self.end_x,
        )

    def slice_at_angle(self, theta: float,
                       dy_offset: float = 0.0) -> Optional[CrackDefect]:
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        if abs(cos_t) < 1e-9:
            x_min = min(self.start_x, self.end_x)
            x_max = max(self.start_x, self.end_x)
            if not (x_min <= 0.0 <= x_max):
                return None
            return CrackDefect(
                start_z=self.start_z, start_x=self.start_x,
                end_z=self.end_z,   end_x=self.end_x,
            )

        tan_t = sin_t / cos_t
        y_shift = dy_offset / cos_t
        if abs(tan_t) > 1e-9:
            x_vis_min = (self.y_start - y_shift) / tan_t
            x_vis_max = (self.y_end   - y_shift) / tan_t
        else:
            x_vis_min = -1e9
            x_vis_max = 1e9
        if x_vis_min > x_vis_max:
            x_vis_min, x_vis_max = x_vis_max, x_vis_min

        x_crack_min = min(self.start_x, self.end_x)
        x_crack_max = max(self.start_x, self.end_x)

        x_lo = max(x_vis_min, x_crack_min)
        x_hi = min(x_vis_max, x_crack_max)
        if x_lo > x_hi:
            return None

        dx = self.end_x - self.start_x
        dz = self.end_z - self.start_z
        if abs(dx) < 1e-15:
            z_lo = z_hi = (self.start_z + self.end_z) / 2.0
        else:
            z_lo = self.start_z + dz * (x_lo - self.start_x) / dx
            z_hi = self.start_z + dz * (x_hi - self.start_x) / dx

        return CrackDefect(
            start_z=float(z_lo), start_x=float(x_lo / cos_t),
            end_z=float(z_hi),   end_x=float(x_hi / cos_t),
        )
