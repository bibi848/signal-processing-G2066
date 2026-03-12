"""
Specimen and defect geometry for 2D and 3D NDT simulation.

2D coordinate system:
    z = depth (downward from array surface, 0 = front wall)
    x = lateral (centered at 0, along array axis)

3D coordinate system (adds elevation axis):
    z = depth (downward from array surface)
    x = lateral (along array length)
    y = elevation (perpendicular to array, mechanical scan direction)

The 1D array sits at a fixed y-position and produces a 2D B-scan (z-x plane).
Stepping the array in y and stacking B-scans gives the 3D volume.
3D defects implement slice_at_y(y) to return the 2D cross-section for the engine.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional
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
        return bool(d < self.radius)


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
        return float(np.linalg.norm(self.end - self.start))

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

    def contains_point(self, _point: np.ndarray) -> bool:
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


# ---------------------------------------------------------------------------
# 3D geometry
# ---------------------------------------------------------------------------

@dataclass
class Specimen3D:
    """
    3D rectangular specimen.

    The 1D array sits at a fixed y-position on the front wall (z = 0) and
    produces a B-scan in the z-x plane. Mechanically stepping the array
    along y and stacking B-scans yields the full 3D volume.

    Attributes:
        thickness: z-extent, front to back wall (m)
        width:     x-extent, lateral (m)
        depth:     y-extent, elevation / mechanical scan direction (m)
        front_wall_z: z-coordinate of the front wall (m)
    """
    thickness: float        # m  (z-direction)
    width: float            # m  (x-direction)
    depth: float            # m  (y-direction, mechanical scan axis)
    front_wall_z: float = 0.0

    @property
    def back_wall_z(self) -> float:
        return self.front_wall_z + self.thickness

    def to_2d(self) -> 'Specimen2D':
        """Return the 2D cross-section used by the ray-tracing engine."""
        return Specimen2D(
            thickness=self.thickness,
            width=self.width,
            front_wall_z=self.front_wall_z,
        )

    def y_positions(self, n_scans: int) -> np.ndarray:
        """
        Uniformly-spaced y-positions for a mechanical linear scan.

        Args:
            n_scans: Number of scan positions

        Returns:
            (n_scans,) array of y-coordinates centred on the specimen
        """
        return np.linspace(-self.depth / 2, self.depth / 2, n_scans)


class Defect3D(ABC):
    """
    Abstract base class for 3D defect geometries.

    Two slicing modes are supported:

    slice_at_y(y)        — linear scan: array translated along y, scan plane
                           is always the z-x plane at fixed elevation y.

    slice_at_angle(theta) — rotational scan: array rotated by angle theta
                            around the centre of the array (origin of the
                            front surface).  The scan plane contains the
                            z-axis and the direction (cos θ, sin θ, 0) in
                            the x-y plane.  Defect coordinates are projected
                            into this rotated frame before being handed to
                            the 2D engine.

    Rotational-frame geometry
    -------------------------
    For a point P = (x, y) in the original x-y plane:
        L  =  x cos θ + y sin θ   (lateral coord along array, the new "x")
        d  = -x sin θ + y cos θ   (perpendicular distance from scan plane)

    A defect is visible when |d| < its effective radius; otherwise None.
    """

    @abstractmethod
    def slice_at_y(self, y: float) -> Optional[Defect2D]:
        """
        Return the 2D cross-section of this defect at elevation y.

        Args:
            y: Elevation position of the array (m)

        Returns:
            A Defect2D instance if the defect intersects this y-plane,
            or None if it does not.
        """
        ...

    @abstractmethod
    def slice_at_angle(self, theta: float) -> Optional[Defect2D]:
        """
        Return the 2D cross-section of this defect in the scan plane at
        rotation angle theta.

        The scan plane passes through the z-axis (depth) and the direction
        (cos theta, sin theta, 0) in the x-y plane.

        Args:
            theta: Azimuthal rotation angle (rad), measured from x-axis
                   toward y-axis.

        Returns:
            A Defect2D instance in the rotated coordinate frame (lateral
            axis = projected distance along the array), or None if the
            defect is not intersected by the scan plane.
        """
        ...


@dataclass
class SphericalDefect(Defect3D):
    """
    Spherical void (e.g. pore, gas pocket).

    Slicing a sphere at elevation y gives a circle whose radius shrinks
    to zero at the poles.  When the slice plane misses the sphere, returns None.

    Attributes:
        center_z: Depth of centre (m)
        center_x: Lateral position of centre (m)
        center_y: Elevation position of centre (m)
        radius:   Sphere radius (m)
    """
    center_z: float
    center_x: float
    center_y: float
    radius: float

    @property
    def center(self) -> np.ndarray:
        return np.array([self.center_z, self.center_x, self.center_y])

    def slice_at_y(self, y: float) -> Optional[CircularDefect]:
        """
        Intersect the sphere with the plane at elevation y.

        Returns a CircularDefect whose radius is the chord radius at that y,
        or None if the plane does not intersect the sphere.
        """
        dy = y - self.center_y
        r_sq = self.radius ** 2 - dy ** 2
        if r_sq <= 0.0:
            return None
        return CircularDefect(
            center_z=self.center_z,
            center_x=self.center_x,
            radius=float(np.sqrt(r_sq)),
        )

    def slice_at_angle(self, theta: float) -> Optional[CircularDefect]:
        """
        Intersect the sphere with the rotated scan plane at angle theta.

        d  = -center_x sin θ + center_y cos θ  (distance to scan plane)
        L  =  center_x cos θ + center_y sin θ  (lateral in rotated frame)
        Visible circle radius = sqrt(r² − d²).
        """
        d = -self.center_x * np.sin(theta) + self.center_y * np.cos(theta)
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
    """
    Cylindrical void with axis along the elevation (y) direction.

    This is the 3D analogue of a side-drilled hole: the cross-section at
    every y-position within the cylinder extent is a circle of constant radius.

    Attributes:
        center_z: Depth of axis (m)
        center_x: Lateral position of axis (m)
        radius:   Cylinder radius (m)
        y_start:  Start of cylinder along elevation axis (m)
        y_end:    End of cylinder along elevation axis (m)
    """
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
        """
        Return a CircularDefect if y falls within the cylinder extent, else None.
        """
        if not (self.y_start <= y <= self.y_end):
            return None
        return CircularDefect(
            center_z=self.center_z,
            center_x=self.center_x,
            radius=self.radius,
        )

    def slice_at_angle(self, theta: float) -> Optional[CircularDefect]:
        """
        Intersect the cylinder with the rotated scan plane at angle theta.

        The cylinder axis runs along y at (center_x, center_z).
        Find the y-position on the axis closest to the scan plane, clamped
        to [y_start, y_end], then compute the perpendicular distance d from
        that axis point to the scan plane.

        d(y) = -center_x sin θ + y cos θ  →  zero at y* = center_x tan θ
        """
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        if abs(cos_t) < 1e-9:
            # Scan plane is nearly parallel to the y-axis; use midpoint
            y_eff = (self.y_start + self.y_end) / 2.0
        else:
            y_star = self.center_x * sin_t / cos_t   # = center_x * tan θ
            y_eff = float(np.clip(y_star, self.y_start, self.y_end))
        d = -self.center_x * sin_t + y_eff * cos_t
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
    """
    Planar crack lying in the z-x plane, with finite extent in elevation (y).

    The crack has the same z-x geometry at every y-position within its
    elevation span.  Slicing gives a CrackDefect at positions within the
    span, or None outside.

    Attributes:
        start_z, start_x: One end of the crack (m)
        end_z,   end_x:   Other end of the crack (m)
        y_start, y_end:   Elevation extent of the crack (m)
    """
    start_z: float
    start_x: float
    end_z: float
    end_x: float
    y_start: float
    y_end: float

    def slice_at_y(self, y: float) -> Optional[CrackDefect]:
        """
        Return a CrackDefect if y falls within the crack's elevation span, else None.
        """
        if not (self.y_start <= y <= self.y_end):
            return None
        return CrackDefect(
            start_z=self.start_z,
            start_x=self.start_x,
            end_z=self.end_z,
            end_x=self.end_x,
        )

    def slice_at_angle(self, theta: float) -> Optional[CrackDefect]:
        """
        Intersect the crack with the rotated scan plane at angle theta.

        The crack lies in the z-x plane at various y ∈ [y_start, y_end].
        The scan plane satisfies y = x tan θ.  A crack point at x_c is
        visible when x_c * tan θ ∈ [y_start, y_end].

        The projected lateral coordinate in the rotated frame is:
            L = x_c / cos θ  (since x = L cos θ, y = L sin θ = x tan θ)

        If no part of the crack satisfies the visibility condition, returns None.
        For the visible sub-segment the x-coordinates are clipped then projected.
        """
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        if abs(cos_t) < 1e-9:
            # Scan plane is nearly parallel to the y-axis (θ ≈ ±π/2).
            # Visible only if the crack spans x ≈ 0.
            x_min = min(self.start_x, self.end_x)
            x_max = max(self.start_x, self.end_x)
            if not (x_min <= 0.0 <= x_max):
                return None
            # Project: x → L = x/cos_t → ±∞, so approximate with x as-is
            return CrackDefect(
                start_z=self.start_z, start_x=self.start_x,
                end_z=self.end_z,   end_x=self.end_x,
            )

        tan_t = sin_t / cos_t
        # x-range visible in this scan plane: y_req = x * tan_t ∈ [y_start, y_end]
        x_vis_min = self.y_start / tan_t if abs(tan_t) > 1e-9 else -1e9
        x_vis_max = self.y_end   / tan_t if abs(tan_t) > 1e-9 else  1e9
        if x_vis_min > x_vis_max:
            x_vis_min, x_vis_max = x_vis_max, x_vis_min

        x_crack_min = min(self.start_x, self.end_x)
        x_crack_max = max(self.start_x, self.end_x)

        # Intersection of visible x-range with crack x-range
        x_lo = max(x_vis_min, x_crack_min)
        x_hi = min(x_vis_max, x_crack_max)
        if x_lo > x_hi:
            return None

        # Project clipped endpoints into rotated frame (L = x / cos θ)
        # Interpolate along the crack to get z at the clipped x values
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
