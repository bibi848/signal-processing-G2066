#!/usr/bin/env python3
"""
3D visualisation of the two microstructure models side by side:
    - Voronoi tessellation (log-normal grain size, Al 6061-T6)
    - Gaussian-smoothed random field (tuned to match Voronoi grain sizes)

Renders each volume as a cube showing its three visible faces.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter

from engine.materials import ALUMINUM

# ── PARAMETERS ───────────────────────────────────────────────────────
CUBE_SIZE     = 3.0e-3     # 3 mm cube (m) — shows ~25 grains per face
VOXEL_SIZE    = 15e-6      # 15 µm voxels → 200³ grid (~8 voxels/grain)

# Aluminium 6061-T6 log-normal grain size (same as 2D comparison)
GRAIN_MEDIAN_UM = 120.0
GRAIN_LOG_STD   = 0.45
Z_VARIATION     = 0.025

SEED   = 42
OUTPUT = 'output/plots/microstructure_3d.png'
# ─────────────────────────────────────────────────────────────────────

rng = np.random.default_rng(SEED)
vs  = VOXEL_SIZE
n   = int(round(CUBE_SIZE / vs))
Z0  = ALUMINUM.density * ALUMINUM.c_L

print(f"Grid: {n}³ voxels  ({CUBE_SIZE*1e3:.0f} mm cube, "
      f"voxel = {vs*1e6:.0f} µm)")


# =====================================================================
# 1. VORONOI (matches log-normal grain distribution)
# =====================================================================

volume_m3 = CUBE_SIZE ** 3
median_d  = GRAIN_MEDIAN_UM * 1e-6
mean_grain_vol = (4.0 / 3.0) * np.pi * (median_d / 2) ** 3
n_grains = max(50, int(volume_m3 / mean_grain_vol))

seed_z = rng.uniform(0, n, n_grains).astype(np.float32)
seed_y = rng.uniform(0, n, n_grains).astype(np.float32)
seed_x = rng.uniform(0, n, n_grains).astype(np.float32)
seeds  = np.column_stack([seed_z, seed_y, seed_x])

grain_Z = (Z0 * (1.0 + rng.uniform(-Z_VARIATION, Z_VARIATION, n_grains))).astype(np.float32)

iz, iy, ix = np.mgrid[0:n, 0:n, 0:n]
voxel_coords = np.column_stack([iz.ravel(), iy.ravel(), ix.ravel()]).astype(np.float32)
tree = cKDTree(seeds)
_, grain_idx = tree.query(voxel_coords, workers=-1)
grain_idx = grain_idx.reshape(n, n, n)
voronoi_vol = grain_Z[grain_idx]

# Measure Voronoi cell volumes → equivalent diameters
_, counts = np.unique(grain_idx, return_counts=True)
vor_diam_m = 2.0 * ((3.0 * counts * vs**3) / (4.0 * np.pi)) ** (1.0 / 3.0)
print(f"Voronoi: {n_grains} grains, diameter "
      f"{vor_diam_m.min()*1e6:.0f}–{vor_diam_m.max()*1e6:.0f} µm, "
      f"median {np.median(vor_diam_m)*1e6:.0f} µm")


# =====================================================================
# 2. GAUSSIAN-SMOOTHED FIELD (tuned so the grain size matches Voronoi)
# =====================================================================
# A Gaussian-smoothed white-noise field has characteristic feature size
# ≈ 2σ (full-width at half-max).  Setting σ = median_grain_radius gives
# matching length scales.  We then normalise to the same impedance std.

sigma_vox = (median_d / 2.0) / vs
noise = rng.standard_normal((n, n, n)).astype(np.float32)
smoothed = gaussian_filter(noise, sigma=sigma_vox, mode='wrap')
smoothed = smoothed / smoothed.std() * (Z_VARIATION * Z0)
gauss_vol = (Z0 + smoothed).astype(np.float32)

print(f"Gaussian: σ = {sigma_vox:.1f} voxels "
      f"({sigma_vox * vs * 1e6:.0f} µm)")


# =====================================================================
# 3D CUBE RENDERING — three visible faces
# =====================================================================

def render_cube(ax, vol, title):
    """Render the three visible faces of a 3-D impedance cube."""
    rel = (vol - Z0) / Z0

    vmin, vmax = -Z_VARIATION, Z_VARIATION
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap('RdBu_r')

    L = CUBE_SIZE * 1e3   # mm

    # Grid coordinates for the three visible faces
    u = np.linspace(0, L, n)
    v = np.linspace(0, L, n)
    U, V = np.meshgrid(u, v, indexing='ij')

    # Top face: z = 0 (top of the cube, viewer looks down)
    top_slice = rel[0, :, :]                # (n_y, n_x)
    ax.plot_surface(
        U, V, np.full_like(U, L),
        rstride=1, cstride=1,
        facecolors=cmap(norm(top_slice)),
        shade=False, linewidth=0, antialiased=False,
    )

    # Front face: y = 0
    front_slice = rel[:, 0, :]              # (n_z, n_x)
    ax.plot_surface(
        U, np.zeros_like(U), L - V,
        rstride=1, cstride=1,
        facecolors=cmap(norm(front_slice)),
        shade=False, linewidth=0, antialiased=False,
    )

    # Right face: x = L
    right_slice = rel[:, :, -1]             # (n_z, n_y)
    ax.plot_surface(
        np.full_like(U, L), U, L - V,
        rstride=1, cstride=1,
        facecolors=cmap(norm(right_slice)),
        shade=False, linewidth=0, antialiased=False,
    )

    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_zlim(0, L)
    ax.set_xlabel('x (mm)', labelpad=-2)
    ax.set_ylabel('y (mm)', labelpad=-2)
    ax.set_zlabel('z (mm)', labelpad=-2)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=22, azim=-55)
    ax.set_title(title, fontsize=13, pad=10)

    # Clean up tick labels for a report-quality look
    ax.tick_params(axis='both', which='major', labelsize=8, pad=-2)
    return cmap, norm


fig = plt.figure(figsize=(13, 6.5))

ax1 = fig.add_subplot(1, 2, 1, projection='3d')
cmap, norm = render_cube(ax1, voronoi_vol, 'Voronoi Tessellation')

ax2 = fig.add_subplot(1, 2, 2, projection='3d')
render_cube(ax2, gauss_vol, 'Gaussian Smoothing')

# Shared colourbar
cax = fig.add_axes([0.35, 0.07, 0.32, 0.025])
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cb = fig.colorbar(sm, cax=cax, orientation='horizontal')
cb.set_label(r'Relative impedance perturbation $\Delta Z / Z_0$', fontsize=10)
cb.ax.tick_params(labelsize=8)

fig.suptitle(
    f'3D Microstructure Models — Aluminium 6061-T6  '
    f'(median grain ∅ = {GRAIN_MEDIAN_UM:.0f} µm)',
    fontsize=14, y=0.97,
)

plt.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.14, wspace=0.05)
plt.savefig(OUTPUT, dpi=220, bbox_inches='tight')
print(f"\nSaved → {OUTPUT}")
