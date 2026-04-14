#!/usr/bin/env python3
"""
Compare two microstructure generation methods for synthetic NDT volumes:
  1. Voronoi tessellation with log-normal grain size distribution
  2. Gaussian-smoothed random impedance field

Produces a multi-panel figure showing impedance maps, grain size
distributions, spatial autocorrelation, and Born scatterer statistics.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree, Voronoi
from scipy.ndimage import gaussian_filter, label
from scipy.signal import fftconvolve

from engine.config import MaterialProperties
from engine.materials import ALUMINUM

# ── PARAMETERS ───────────────────────────────────────────────────────
# Specimen (2D slice for visualisation)
THICKNESS   = 0.040       # z extent (m)
WIDTH       = 0.040       # x extent (m)
VOXEL_SIZE  = 0.020e-3    # 20 µm voxels — resolves ~100 µm grains

# Aluminum 6061-T6 grain size distribution (log-normal)
# Wrought Al 6061-T6: literature reports mean grain diameter ~80–200 µm
# with a log-normal spread. We use:
#   median diameter = 120 µm,  geometric std = 0.45
# which gives a mean ~130 µm and 95 % of grains between 50–300 µm.
GRAIN_MEDIAN_UM = 120.0   # median grain diameter (µm)
GRAIN_LOG_STD   = 0.45    # log-normal sigma (dimensionless)

# Impedance variation per grain (crystallographic texture in Al)
Z_VARIATION = 0.025       # ±2.5 %
C_VARIATION = 0.005       # ±0.5 %

# Gaussian smoothing: correlation length matched to mean grain size
GAUSS_SIGMA_UM = 130.0    # smoothing kernel std (µm)

SEED   = 42
OUTPUT = 'output/plots/microstructure_comparison.png'
# ─────────────────────────────────────────────────────────────────────

rng = np.random.default_rng(SEED)
vs  = VOXEL_SIZE

n_z = int(round(THICKNESS / vs))
n_x = int(round(WIDTH / vs))

Z0 = ALUMINUM.density * ALUMINUM.c_L
c0 = ALUMINUM.c_L

print(f"Grid: {n_z} × {n_x} voxels  ({THICKNESS*1e3:.0f} × {WIDTH*1e3:.0f} mm,"
      f"  voxel = {vs*1e6:.0f} µm)")


# =====================================================================
# 1. VORONOI with log-normal grain diameters
# =====================================================================

# Sample grain diameters from log-normal (metres)
area_m2 = THICKNESS * WIDTH
median_d = GRAIN_MEDIAN_UM * 1e-6
mean_area_grain = np.pi * (median_d / 2) ** 2   # approximate 2D grain area
n_grains_est = int(area_m2 / mean_area_grain)
n_grains = max(50, n_grains_est)

grain_diameters = rng.lognormal(
    mean=np.log(median_d),
    sigma=GRAIN_LOG_STD,
    size=n_grains,
)

# Place seeds with density proportional to 1/grain_area so that large
# grains get more space.  Simple approach: uniform random seeds, then
# the Voronoi cell sizes will have some spread.  For a controlled
# log-normal distribution we use a different strategy:
#   - generate seeds on a jittered grid
#   - assign each seed a target radius from the log-normal draw
#   - use iterative Lloyd relaxation weighted by target area
# For simplicity (and because the Voronoi cell size distribution from
# uniform seeds is already close to log-normal), we use uniform seeds
# and verify the resulting distribution.
seed_z = rng.uniform(0, n_z, size=n_grains).astype(np.float32)
seed_x = rng.uniform(0, n_x, size=n_grains).astype(np.float32)
seeds  = np.column_stack([seed_z, seed_x])

# Per-grain impedance offsets
grain_Z = (Z0 * (1.0 + rng.uniform(-Z_VARIATION, Z_VARIATION, n_grains))).astype(np.float32)
grain_c = (c0 * (1.0 + rng.uniform(-C_VARIATION, C_VARIATION, n_grains))).astype(np.float32)

# Voronoi assignment
iz, ix = np.mgrid[0:n_z, 0:n_x]
voxel_coords = np.column_stack([iz.ravel(), ix.ravel()]).astype(np.float32)
tree = cKDTree(seeds)
_, grain_idx = tree.query(voxel_coords, workers=-1)
grain_idx = grain_idx.reshape(n_z, n_x)

voronoi_Z = grain_Z[grain_idx]
voronoi_c = grain_c[grain_idx]

# Measure actual Voronoi cell areas → equivalent diameters
grain_ids, grain_counts = np.unique(grain_idx, return_counts=True)
voronoi_areas_m2 = grain_counts * (vs ** 2)
voronoi_diameters_m = 2.0 * np.sqrt(voronoi_areas_m2 / np.pi)

print(f"Voronoi: {n_grains} grains, "
      f"diameter range {voronoi_diameters_m.min()*1e6:.0f}–"
      f"{voronoi_diameters_m.max()*1e6:.0f} µm, "
      f"mean {voronoi_diameters_m.mean()*1e6:.0f} µm")


# =====================================================================
# 2. GAUSSIAN-SMOOTHED random field
# =====================================================================

sigma_vox = (GAUSS_SIGMA_UM * 1e-6) / vs   # kernel std in voxel units

# White noise scaled to the same impedance variation
noise = rng.standard_normal((n_z, n_x)).astype(np.float32)
smoothed = gaussian_filter(noise, sigma=sigma_vox, mode='wrap')

# Normalise to have the same impedance std as the Voronoi field
target_std = Z_VARIATION * Z0
smoothed = smoothed / smoothed.std() * target_std
gauss_Z = (Z0 + smoothed).astype(np.float32)

print(f"Gaussian: σ_kernel = {GAUSS_SIGMA_UM:.0f} µm = {sigma_vox:.1f} voxels")


# =====================================================================
# Helper: spatial autocorrelation (normalised, radial average)
# =====================================================================

def radial_autocorrelation(field: np.ndarray, max_lag_voxels: int = 400):
    """Compute radially averaged normalised autocorrelation."""
    f = field - field.mean()
    acf_2d = fftconvolve(f, f[::-1, ::-1], mode='full')
    acf_2d /= acf_2d.max()

    cy, cx = np.array(acf_2d.shape) // 2
    y, x = np.ogrid[-cy:acf_2d.shape[0]-cy, -cx:acf_2d.shape[1]-cx]
    r = np.sqrt(x**2 + y**2).astype(int)

    max_r = min(max_lag_voxels, r.max())
    radial = np.zeros(max_r + 1)
    counts = np.zeros(max_r + 1)
    mask = r <= max_r
    np.add.at(radial, r[mask], acf_2d[mask])
    np.add.at(counts, r[mask], 1)
    counts[counts == 0] = 1
    radial /= counts
    return radial


# =====================================================================
# Helper: Born scatterer extraction (depth gradient)
# =====================================================================

def extract_born(Z_field, Z0, threshold=0.005):
    delta = np.diff(Z_field, axis=0, prepend=Z_field[:1, :])
    delta_rel = delta / (2.0 * Z0)
    mask = np.abs(delta_rel) > threshold
    return delta_rel, mask


# =====================================================================
# FIGURE
# =====================================================================

fig = plt.figure(figsize=(16, 14))
gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.35)

z_mm = np.arange(n_z) * vs * 1e3
x_mm = np.arange(n_x) * vs * 1e3
extent = [x_mm[0], x_mm[-1], z_mm[-1], z_mm[0]]

# Shared colour limits (relative impedance perturbation)
vmin_rel = -Z_VARIATION * 1.2
vmax_rel =  Z_VARIATION * 1.2

# ── Row 0: impedance maps ──
ax0 = fig.add_subplot(gs[0, 0:2])
im0 = ax0.imshow((voronoi_Z - Z0) / Z0, extent=extent, aspect='equal',
                  cmap='RdBu_r', vmin=vmin_rel, vmax=vmax_rel)
ax0.set_title('Voronoi Tessellation', fontsize=11)
ax0.set_xlabel('x (mm)')
ax0.set_ylabel('z (mm)')
plt.colorbar(im0, ax=ax0, label='ΔZ / Z₀', shrink=0.85)

ax1 = fig.add_subplot(gs[0, 2:4])
im1 = ax1.imshow((gauss_Z - Z0) / Z0, extent=extent, aspect='equal',
                  cmap='RdBu_r', vmin=vmin_rel, vmax=vmax_rel)
ax1.set_title('Gaussian Smoothing', fontsize=11)
ax1.set_xlabel('x (mm)')
ax1.set_ylabel('z (mm)')
plt.colorbar(im1, ax=ax1, label='ΔZ / Z₀', shrink=0.85)

# ── Row 1, col 0-1: grain size distribution (Voronoi only, + literature) ──
ax2 = fig.add_subplot(gs[1, 0:2])
bins_um = np.linspace(0, 500, 80)

# Voronoi measured diameters
ax2.hist(voronoi_diameters_m * 1e6, bins=bins_um, density=True,
         alpha=0.6, color='C0', label='Voronoi (measured)')

# Theoretical log-normal for wrought Al 6061-T6
d_um = np.linspace(1, 500, 500)
mu_ln = np.log(GRAIN_MEDIAN_UM)
pdf_ln = (1 / (d_um * GRAIN_LOG_STD * np.sqrt(2 * np.pi))
          * np.exp(-0.5 * ((np.log(d_um) - mu_ln) / GRAIN_LOG_STD) ** 2))
ax2.plot(d_um, pdf_ln, 'k-', linewidth=1.5, label=f'Log-normal target\n'
         f'(median={GRAIN_MEDIAN_UM:.0f} µm, σ={GRAIN_LOG_STD})')

ax2.set_xlabel('Equivalent grain diameter (µm)')
ax2.set_ylabel('Probability density')
ax2.set_title('Grain Size Distribution', fontsize=11)
ax2.legend(fontsize=8)
ax2.set_xlim(0, 500)

# ── Row 1, col 2-3: spatial autocorrelation comparison ──
ax3 = fig.add_subplot(gs[1, 2:4])
max_lag = min(500, n_z // 2)
acf_voronoi = radial_autocorrelation(voronoi_Z, max_lag)
acf_gauss   = radial_autocorrelation(gauss_Z, max_lag)
lag_mm = np.arange(len(acf_voronoi)) * vs * 1e3

ax3.plot(lag_mm, acf_voronoi, label='Voronoi', linewidth=1.3)
ax3.plot(lag_mm, acf_gauss, label='Gaussian', linewidth=1.3)
ax3.axhline(0, color='grey', linewidth=0.5, linestyle='--')
# Mark correlation length (first zero crossing)
for acf, name, color in [(acf_voronoi, 'Voronoi', 'C0'),
                          (acf_gauss, 'Gaussian', 'C1')]:
    zeros = np.where(acf[1:] <= 0)[0]
    if len(zeros) > 0:
        corr_len = zeros[0] * vs * 1e3
        ax3.axvline(corr_len, color=color, linestyle=':', alpha=0.7)
        ax3.annotate(f'{corr_len:.2f} mm', xy=(corr_len, 0.05),
                     fontsize=7, color=color)

ax3.set_xlabel('Lag (mm)')
ax3.set_ylabel('Normalised autocorrelation')
ax3.set_title('Spatial Autocorrelation', fontsize=11)
ax3.legend(fontsize=8)
ax3.set_xlim(0, lag_mm[max_lag] if max_lag < len(lag_mm) else lag_mm[-1])

# ── Row 2, col 0-1: Born scatterer map comparison ──
born_vor, mask_vor = extract_born(voronoi_Z, Z0)
born_gau, mask_gau = extract_born(gauss_Z, Z0)

ax4 = fig.add_subplot(gs[2, 0])
ax4.imshow(np.abs(born_vor), extent=extent, aspect='equal',
           cmap='hot', vmin=0, vmax=0.03)
ax4.set_title('Born |amplitude| — Voronoi', fontsize=9)
ax4.set_xlabel('x (mm)')
ax4.set_ylabel('z (mm)')

ax5 = fig.add_subplot(gs[2, 1])
ax5.imshow(np.abs(born_gau), extent=extent, aspect='equal',
           cmap='hot', vmin=0, vmax=0.03)
ax5.set_title('Born |amplitude| — Gaussian', fontsize=9)
ax5.set_xlabel('x (mm)')
ax5.set_ylabel('z (mm)')

# ── Row 2, col 2: Born amplitude histograms ──
ax6 = fig.add_subplot(gs[2, 2])
born_bins = np.linspace(-0.04, 0.04, 120)
ax6.hist(born_vor[mask_vor].ravel(), bins=born_bins, density=True,
         alpha=0.55, label='Voronoi')
ax6.hist(born_gau[mask_gau].ravel(), bins=born_bins, density=True,
         alpha=0.55, label='Gaussian')
ax6.set_xlabel('Born amplitude ΔZ / (2Z₀)')
ax6.set_ylabel('Density')
ax6.set_title('Born Amplitude Distribution', fontsize=9)
ax6.legend(fontsize=7)

# ── Row 2, col 3: summary table ──
ax7 = fig.add_subplot(gs[2, 3])
ax7.axis('off')

n_born_vor = mask_vor.sum()
n_born_gau = mask_gau.sum()

rows = [
    ['Grid size', f'{n_z} × {n_x}', f'{n_z} × {n_x}'],
    ['Voxel size', f'{vs*1e6:.0f} µm', f'{vs*1e6:.0f} µm'],
    ['Z std / Z₀', f'{(voronoi_Z.std()/Z0)*100:.2f} %',
                    f'{(gauss_Z.std()/Z0)*100:.2f} %'],
    ['Born scatterers', f'{n_born_vor:,}', f'{n_born_gau:,}'],
    ['Born |amp| mean', f'{np.abs(born_vor[mask_vor]).mean():.4f}',
                        f'{np.abs(born_gau[mask_gau]).mean():.4f}'],
    ['Grain boundaries', 'Sharp (step)', 'Smooth (gradient)'],
    ['Grain size control', 'Direct (seeds)', 'Indirect (σ)'],
]

table = ax7.table(
    cellText=rows,
    colLabels=['', 'Voronoi', 'Gaussian'],
    loc='center', cellLoc='center',
)
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1.0, 1.6)
ax7.set_title('Summary', fontsize=9, pad=12)

fig.suptitle(
    'Microstructure Model Comparison — Aluminum 6061-T6\n'
    f'(median grain ∅ = {GRAIN_MEDIAN_UM:.0f} µm,  '
    f'Z variation = ±{Z_VARIATION*100:.1f} %,  '
    f'Gaussian σ = {GAUSS_SIGMA_UM:.0f} µm)',
    fontsize=13, y=0.995,
)

plt.savefig(OUTPUT, dpi=200, bbox_inches='tight')
print(f"\nFigure saved → {OUTPUT}")
