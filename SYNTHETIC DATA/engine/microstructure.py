"""
Synthetic microstructure generators for voxel-based NDT simulation.

Provides functions to populate a VoxelVolume3D with physically motivated
acoustic property fields:

  generate_grain_structure   — polycrystalline metal (Voronoi tessellation)
  embed_geometric_defects    — burn hard defects (voids/inclusions) into a volume

Grain-noise model
-----------------
In a polycrystalline material, each grain has a slightly different
crystallographic orientation.  For longitudinal wave propagation in a cubic
metal (e.g. aluminium), the anisotropy rotates the stiffness tensor, producing
grain-to-grain variations in c_L of roughly ±0.5 % and in impedance of ±1–3 %.
At grain boundaries the impedance jumps → weak reflections (Born scattering).

The Voronoi model approximates this:
  1. Scatter N random seed points through the volume.
  2. Assign each voxel to its nearest seed (Voronoi cell = one grain).
  3. Draw a random impedance offset for each grain from a uniform distribution
     of width ±impedance_variation.

The resulting impedance field is piece-wise constant with sharp jumps at grain
boundaries, which is the dominant scattering mechanism for 5–20 MHz UT.
"""

import numpy as np
from typing import List

from .voxel_volume import VoxelVolume3D
from .config import MaterialProperties


def generate_grain_structure(
    thickness: float,
    width: float,
    depth: float,
    background_material: MaterialProperties,
    mean_grain_size_m: float = 1.5e-3,
    impedance_variation: float = 0.025,
    wavespeed_variation: float = 0.005,
    voxel_size_m: float = 0.5e-3,
    seed: int = 42,
) -> VoxelVolume3D:
    """
    Generate a synthetic polycrystalline microstructure via Voronoi tessellation.

    Args:
        thickness:            Specimen depth (z), m
        width:                Specimen lateral extent (x), m
        depth:                Specimen elevation extent (y), m
        background_material:  Nominal material (sets Z₀ and c₀)
        mean_grain_size_m:    Target mean grain diameter (m).
                              Controls grain density; smaller → more grains.
        impedance_variation:  Fractional amplitude of per-grain Z variation.
                              E.g. 0.025 → each grain Z is Z₀ × (1 ± 2.5 %).
        wavespeed_variation:  Fractional amplitude of per-grain c_L variation.
        voxel_size_m:         Isotropic voxel edge length (m).
                              Should be ≤ mean_grain_size_m / 4 to resolve grain
                              boundaries with at least 2 voxels.
        seed:                 Random number generator seed for reproducibility.

    Returns:
        VoxelVolume3D with the grain impedance and wavespeed fields set.
    """
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        raise ImportError(
            "scipy is required for grain structure generation.  "
            "Install with: pip install scipy"
        )

    rng = np.random.default_rng(seed)
    vs = voxel_size_m

    # Grid dimensions
    n_z = max(2, int(round(thickness / vs)) + 1)
    n_y = max(2, int(round(depth     / vs)) + 1)
    n_x = max(2, int(round(width     / vs)) + 1)

    # Estimate grain count from target grain size
    total_volume_m3 = thickness * depth * width
    grain_volume_m3 = (4.0 / 3.0) * np.pi * (mean_grain_size_m / 2.0) ** 3
    n_grains = max(20, int(round(total_volume_m3 / grain_volume_m3)))

    print(f"  Grain structure: {n_z}×{n_y}×{n_x} voxels, "
          f"{n_grains} grains "
          f"(~{mean_grain_size_m*1e3:.1f} mm mean diameter)")

    # Random grain seed positions in voxel-index space
    seeds = rng.uniform(
        low=[0.0, 0.0, 0.0],
        high=[float(n_z), float(n_y), float(n_x)],
        size=(n_grains, 3),
    ).astype(np.float32)

    # Per-grain material property offsets
    Z0 = background_material.Z_L
    c0 = background_material.c_L
    grain_Z = (Z0 * (1.0 + rng.uniform(
        -impedance_variation, impedance_variation, n_grains
    ))).astype(np.float32)
    grain_c = (c0 * (1.0 + rng.uniform(
        -wavespeed_variation, wavespeed_variation, n_grains
    ))).astype(np.float32)

    # Voronoi tessellation: assign every voxel to its nearest grain seed
    iz, iy, ix = np.mgrid[0:n_z, 0:n_y, 0:n_x]   # (n_z, n_y, n_x) each
    voxel_coords = np.stack(
        [iz.ravel(), iy.ravel(), ix.ravel()], axis=1
    ).astype(np.float32)

    tree = cKDTree(seeds)
    _, grain_idx = tree.query(voxel_coords, workers=-1)   # parallel query
    grain_idx = grain_idx.reshape(n_z, n_y, n_x)

    impedance = grain_Z[grain_idx]   # (n_z, n_y, n_x)
    wavespeed = grain_c[grain_idx]   # (n_z, n_y, n_x)

    return VoxelVolume3D(
        impedance  = impedance,
        wavespeed  = wavespeed,
        voxel_size = vs,
        origin_z   = 0.0,
        origin_y   = -depth  / 2.0,
        origin_x   = -width  / 2.0,
    )


def embed_geometric_defects(
    volume: VoxelVolume3D,
    defects_3d: list,
    void_impedance_factor: float = 0.001,
) -> VoxelVolume3D:
    """
    Burn geometric 3D defects into an existing voxel volume.

    Each defect is rasterized into the impedance/wavespeed fields.
    Voids (air-filled holes) are set to near-zero impedance, giving
    a reflection coefficient R ≈ −1 at the defect surface.

    Args:
        volume:                  Source VoxelVolume3D (may be a grain structure).
        defects_3d:              List of Defect3D objects (SphericalDefect, etc.).
        void_impedance_factor:   Multiplier applied to Z₀ for void voxels.
                                 Default 0.001 → R ≈ −0.998 (effectively total reflection).

    Returns:
        New VoxelVolume3D with defect voxels overwritten.
    """
    # Lazy import to avoid circular dependency
    from .geometry import SphericalDefect, CylindricalDefect, PlanarCrack3D

    imp  = volume.impedance.copy()
    wave = volume.wavespeed.copy()
    vs   = volume.voxel_size

    n_z, n_y, n_x = volume.shape

    # Coordinate grids in world space
    z_c = volume.origin_z + np.arange(n_z) * vs
    y_c = volume.origin_y + np.arange(n_y) * vs
    x_c = volume.origin_x + np.arange(n_x) * vs

    zz, yy, xx = np.meshgrid(z_c, y_c, x_c, indexing='ij')  # (n_z, n_y, n_x)

    void_Z = float(np.mean(imp)) * void_impedance_factor
    void_c = float(np.mean(wave)) * void_impedance_factor

    for defect in defects_3d:
        if isinstance(defect, SphericalDefect):
            mask = (
                (zz - defect.center_z) ** 2
                + (xx - defect.center_x) ** 2
                + (yy - defect.center_y) ** 2
            ) <= defect.radius ** 2

        elif isinstance(defect, CylindricalDefect):
            in_radial = (
                (zz - defect.center_z) ** 2
                + (xx - defect.center_x) ** 2
            ) <= defect.radius ** 2
            in_extent = (yy >= defect.y_start) & (yy <= defect.y_end)
            mask = in_radial & in_extent

        elif isinstance(defect, PlanarCrack3D):
            # Thin slab: 2 voxels wide, otherwise misses under Born approx.
            half_t = vs * 2.0
            in_y = (yy >= defect.y_start) & (yy <= defect.y_end)
            dz_seg = defect.end_z - defect.start_z
            dx_seg = defect.end_x - defect.start_x
            seg_len_sq = dz_seg ** 2 + dx_seg ** 2
            if seg_len_sq < 1e-20:
                dist_sq = (zz - defect.start_z) ** 2 + (xx - defect.start_x) ** 2
            else:
                t = ((zz - defect.start_z) * dz_seg
                     + (xx - defect.start_x) * dx_seg) / seg_len_sq
                t = np.clip(t, 0.0, 1.0)
                dist_sq = (
                    (zz - (defect.start_z + t * dz_seg)) ** 2
                    + (xx - (defect.start_x + t * dx_seg)) ** 2
                )
            mask = in_y & (dist_sq <= half_t ** 2)

        else:
            continue

        imp[mask]  = void_Z
        wave[mask] = void_c

    return VoxelVolume3D(
        impedance  = imp,
        wavespeed  = wave,
        voxel_size = vs,
        origin_z   = volume.origin_z,
        origin_y   = volume.origin_y,
        origin_x   = volume.origin_x,
    )
