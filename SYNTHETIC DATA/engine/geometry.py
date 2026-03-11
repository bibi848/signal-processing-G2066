"""
Specimen and defect geometry for 2D NDT simulation.

Coordinate system:
    z = depth (downward from array surface, 0 = front wall)
    x = lateral (centered at 0, along array axis)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from abc import ABC, abstractmethod


@dataclass
class Specimen2D:
    """
    2D rectangular specimen with front and back walls.

    The array sits on the front wall (z = front_wall_z).
    """
    thickness: float          # m (front wall to back wall)
    width: float              # m (lateral extent)
    front_wall_z: float = 0.0

    @property
    def back_wall_z(self) -> float:
        return self.front_wall_z + self.thickness

    def mirror_across_backwall(self, point: np.ndarray) -> np.ndarray:
        """
        Mirror a point across the back wall.
        Used for computing skip paths via the image method.

        Args:
            point: (2,) array [z, x]

        Returns:
            Mirrored point [z_mirror, x]
        """
        mirrored = point.copy()
        mirrored[0] = 2.0 * self.back_wall_z - point[0]
        return mirrored

    def mirror_across_frontwall(self, point: np.ndarray) -> np.ndarray:
        """Mirror a point across the front wall."""
        mirrored = point.copy()
        mirrored[0] = 2.0 * self.front_wall_z - point[0]
        return mirrored

    def backwall_reflection_point(self, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        """
        Find the specular reflection point on the back wall
        for a ray going from p1 to the back wall to p2.

        Uses the mirror-image method: mirror p2 across the back wall,
        then the reflection point is where the line p1→p2_mirror
        intersects the back wall.

        Args:
            p1: Source point [z, x]
            p2: Receiver point [z, x]

        Returns:
            Reflection point [back_wall_z, x_reflect]
        """
        p2_mirror = self.mirror_across_backwall(p2)
        # Parametric line: P = p1 + t*(p2_mirror - p1)
        # At z = back_wall_z: t = (back_wall_z - p1[0]) / (p2_mirror[0] - p1[0])
        dz = p2_mirror[0] - p1[0]
        if abs(dz) < 1e-15:
            # Degenerate case: both at same depth
            x_reflect = 0.5 * (p1[1] + p2[1])
        else:
            t = (self.back_wall_z - p1[0]) / dz
            x_reflect = p1[1] + t * (p2_mirror[1] - p1[1])
        return np.array([self.back_wall_z, x_reflect])


# --- Defect base class ---

@dataclass
class Defect2D(ABC):
    """Abstract base class for 2D defect geometries."""

    @abstractmethod
    def discretize_surface(self, n_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Discretize the defect surface into points with outward normals.

        Args:
            n_points: Number of surface points

        Returns:
            points: (n_points, 2) array of [z, x] positions
            normals: (n_points, 2) array of outward unit normals [nz, nx]
        """
        ...

    @abstractmethod
    def contains_point(self, point: np.ndarray) -> bool:
        """Check if a point is inside the defect."""
        ...


@dataclass
class CircularDefect(Defect2D):
    """
    Circular void (2D cross-section of a side-drilled hole).

    Standard NDT calibration target. The boundary is a circle
    with total reflection (void/air interior).
    """
    center_z: float    # Depth of center (m)
    center_x: float    # Lateral position of center (m)
    radius: float      # Radius (m)

    @property
    def center(self) -> np.ndarray:
        return np.array([self.center_z, self.center_x])

    def discretize_surface(self, n_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Discretize circle perimeter into points with outward normals.
        """
        angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)

        # Points on the circle: [z, x]
        points = np.zeros((n_points, 2))
        points[:, 0] = self.center_z + self.radius * np.cos(angles)
        points[:, 1] = self.center_x + self.radius * np.sin(angles)

        # Outward normals (radial direction)
        normals = np.zeros((n_points, 2))
        normals[:, 0] = np.cos(angles)
        normals[:, 1] = np.sin(angles)

        return points, normals

    def contains_point(self, point: np.ndarray) -> bool:
        d = np.linalg.norm(point - self.center)
        return d < self.radius


@dataclass
class CrackDefect(Defect2D):
    """
    Crack modeled as a line segment (zero-width planar reflector).

    Critical for corner-trap detection. Typically vertical cracks
    extending from the back wall or internal cracks at arbitrary angles.
    """
    start_z: float     # Start point depth (m)
    start_x: float     # Start point lateral (m)
    end_z: float       # End point depth (m)
    end_x: float       # End point lateral (m)

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
        return np.linalg.norm(self.end - self.start)

    @property
    def tangent(self) -> np.ndarray:
        """Unit tangent along the crack."""
        d = self.end - self.start
        return d / np.linalg.norm(d)

    @property
    def normal(self) -> np.ndarray:
        """
        Unit normal to the crack (perpendicular to tangent).
        Convention: rotated 90° clockwise from tangent.
        """
        t = self.tangent
        return np.array([t[1], -t[0]])

    def discretize_surface(self, n_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Discretize crack into points along both faces (front and back).
        """
        # Parametric points along the crack
        s = np.linspace(0, 1, n_points)
        points = np.zeros((n_points, 2))
        points[:, 0] = self.start_z + s * (self.end_z - self.start_z)
        points[:, 1] = self.start_x + s * (self.end_x - self.start_x)

        # Both faces have the same normal (crack reflects from both sides)
        n = self.normal
        normals = np.tile(n, (n_points, 1))

        return points, normals

    def contains_point(self, point: np.ndarray) -> bool:
        # A crack is zero-width; nothing is "inside"
        return False


@dataclass
class FlatBottomHole(Defect2D):
    """
    Flat-bottom hole (FBH) — a flat circular reflector at a known depth.

    Standard NDT calibration target. Modeled as a horizontal line segment
    in 2D (the cross-section of a disc).
    """
    center_z: float    # Depth of the flat face (m)
    center_x: float    # Lateral position (m)
    width: float       # Diameter of the flat face (m)

    @property
    def center(self) -> np.ndarray:
        return np.array([self.center_z, self.center_x])

    def discretize_surface(self, n_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """Discretize the flat reflecting face."""
        x_positions = np.linspace(
            self.center_x - self.width / 2,
            self.center_x + self.width / 2,
            n_points
        )
        points = np.zeros((n_points, 2))
        points[:, 0] = self.center_z
        points[:, 1] = x_positions

        # Normal points upward (toward the array)
        normals = np.zeros((n_points, 2))
        normals[:, 0] = -1.0  # -z direction = toward array

        return points, normals

    def contains_point(self, point: np.ndarray) -> bool:
        return False
