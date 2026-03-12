"""
Voxel-based 3D volume representation for NDT simulation.

The world is described by a 3D array of local acoustic impedance
Z(z, y, x) = ρ(z,y,x) · c_L(z,y,x).  Any voxel where Z differs from the
background material produces a weak reflection (Born / first-order scattering
approximation) with amplitude δZ / (2·Z₀).

This enables:
  - Grain-noise simulation: Voronoi grains with ±1-3 % impedance variation
  - Arbitrary defect geometry: embed high-contrast (R ≈ 1) voxel regions
  - Mixed scenes: geometric defects + grain background

Coordinate system (same as geometry.py)
    z  — depth, downward from array surface (axis 0)
    y  — elevation, mechanical scan / rotation direction (axis 1)
    x  — lateral, along array (axis 2)

The volume is stored as a (n_z, n_y, n_x) float32 array.  Origin coordinates
give the world position of voxel index [0, 0, 0].  Voxels are isotropic.

Slicing for a rotational scan at angle θ
-----------------------------------------
The scan plane at angle θ contains the z-axis and the direction
(cos θ, sin θ, 0) in the x-y plane.  A point with depth z and lateral
coordinate L in that plane has world coordinates:

    x_world = L · cos θ
    y_world = L · sin θ
    z_world = z

`slice_at_angle(theta, z_grid, lateral_grid)` samples the impedance field
at these points using trilinear interpolation, returning a (n_z, n_L) array.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class VoxelVolume3D:
    """
    3D acoustic impedance field on a uniform isotropic voxel grid.

    Attributes:
        impedance:  (n_z, n_y, n_x) float32 — local Z = ρ·c_L (Pa·s/m)
        wavespeed:  (n_z, n_y, n_x) float32 — local c_L (m/s)
        voxel_size: Edge length of each voxel (m)
        origin_z:   World z-coordinate of voxel [0, *, *] centre (m)
        origin_y:   World y-coordinate of voxel [*, 0, *] centre (m)
        origin_x:   World x-coordinate of voxel [*, *, 0] centre (m)
    """
    impedance:  np.ndarray   # (n_z, n_y, n_x) float32
    wavespeed:  np.ndarray   # (n_z, n_y, n_x) float32
    voxel_size: float        # m
    origin_z:   float = 0.0
    origin_y:   float = 0.0
    origin_x:   float = 0.0

    @property
    def shape(self) -> Tuple[int, int, int]:
        """(n_z, n_y, n_x)"""
        return self.impedance.shape  # type: ignore[return-value]

    def coords(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        World-coordinate arrays for each axis.

        Returns:
            (z_coords, y_coords, x_coords) — each 1-D, length = voxel count
        """
        n_z, n_y, n_x = self.shape
        z = self.origin_z + np.arange(n_z) * self.voxel_size
        y = self.origin_y + np.arange(n_y) * self.voxel_size
        x = self.origin_x + np.arange(n_x) * self.voxel_size
        return z, y, x

    def slice_at_angle(self,
                       theta: float,
                       z_grid: np.ndarray,
                       lateral_grid: np.ndarray) -> np.ndarray:
        """
        Sample the impedance field in the rotated scan plane.

        For each (z_i, L_j) point in the scan plane, the corresponding world
        position is (z_i, L_j · sin θ, L_j · cos θ).  Points that fall
        outside the volume are set to the mean background impedance.

        Args:
            theta:        Scan-plane rotation angle (rad)
            z_grid:       (n_z_out,) depth values to sample (m)
            lateral_grid: (n_L,)    lateral values to sample (m)

        Returns:
            imp_2d: (n_z_out, n_L) float32 impedance map
        """
        from scipy.ndimage import map_coordinates

        cos_t = float(np.cos(theta))
        sin_t = float(np.sin(theta))

        # Build scan-plane sample grid: shape (n_z_out, n_L)
        ZZ, LL = np.meshgrid(z_grid, lateral_grid, indexing='ij')

        # World coordinates of each sample point
        x_world = LL * cos_t          # x = L cos θ
        y_world = LL * sin_t          # y = L sin θ
        # z_world = ZZ  (unchanged)

        # Convert world coords to fractional voxel indices
        iz_f = (ZZ      - self.origin_z) / self.voxel_size
        iy_f = (y_world - self.origin_y) / self.voxel_size
        ix_f = (x_world - self.origin_x) / self.voxel_size

        # Background value: mean impedance of the volume
        bg_Z = float(np.mean(self.impedance))

        coords = np.array([iz_f.ravel(), iy_f.ravel(), ix_f.ravel()])
        imp_flat = map_coordinates(
            self.impedance.astype(np.float64),
            coords,
            order=1,           # trilinear interpolation
            mode='constant',   # out-of-bounds → background
            cval=bg_Z,
        )
        return imp_flat.reshape(ZZ.shape).astype(np.float32)

    def extract_born_scatterers(
        self,
        theta: float,
        z_grid: np.ndarray,
        lateral_grid: np.ndarray,
        background_Z: float,
        threshold: float = 0.005,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract significant scattering points from the scan-plane slice.

        Uses the Born (first-order) approximation:
            amplitude = δZ / (2·Z₀)   where δZ = Z_local − Z₀

        Only voxels where |δZ / (2·Z₀)| > threshold are returned, pruning
        the background interior of each grain and keeping only boundaries.

        Args:
            theta:        Rotation angle of the scan plane (rad)
            z_grid:       Depth sampling grid (m)
            lateral_grid: Lateral sampling grid (m)
            background_Z: Background impedance Z₀ (Pa·s/m)
            threshold:    Minimum relative amplitude to include a scatterer

        Returns:
            (z_s, x_s, amp_s) — each (N_scatterers,):
                z_s:   depth of each scatterer (m)
                x_s:   lateral coordinate in the scan plane (m)
                amp_s: Born scattering amplitude (dimensionless)
        """
        imp_2d = self.slice_at_angle(theta, z_grid, lateral_grid)

        # Born amplitude map
        delta_rel = (imp_2d - background_Z) / (2.0 * background_Z)

        # Keep only significant boundaries
        mask = np.abs(delta_rel) > threshold
        iz, il = np.where(mask)

        z_s   = z_grid[iz].astype(np.float64)
        x_s   = lateral_grid[il].astype(np.float64)
        amp_s = delta_rel[iz, il].astype(np.float64)

        return z_s, x_s, amp_s
