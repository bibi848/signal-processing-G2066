#!/usr/bin/env python3
"""
Standalone viewer for the synthetic B-scans produced by
compare_experimental_vs_synthetic.py (which in turn calls scan_volume_3d()).

Loads bscan_0000.npy from the two compare_tmp_* folders and shows them
side by side in a larger figure with proper axes and a shared colourbar.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── PARAMETERS ───────────────────────────────────────────────────────
GRAIN_DIR  = 'output/comparisons/compare_tmp_grain'
DEFECT_DIR = 'output/comparisons/compare_tmp_defect'
OUTPUT     = 'output/plots/synthetic_scans.png'

DB_FLOOR   = -20           # dB display floor
CMAP       = 'viridis'     # 'hot', 'viridis', 'gray', ...
# ─────────────────────────────────────────────────────────────────────


def load_frame(scan_dir: str):
    """Load bscan_0000.npy and its companion scan_meta.npy."""
    bscan = np.load(os.path.join(scan_dir, 'bscan_0000.npy'))
    meta  = np.load(os.path.join(scan_dir, 'scan_meta.npy'),
                    allow_pickle=True).item()
    z_start = meta['tfm_z_start_m'] * 1e3
    z_end   = meta['tfm_z_end_m']   * 1e3
    half_w  = meta['array_aperture_m'] / 2 * 1e3
    extent = [-half_w, half_w, z_end, z_start]   # (xmin, xmax, zmax, zmin)
    return bscan, extent, meta


grain_img,  grain_ext,  grain_meta  = load_frame(GRAIN_DIR)
defect_img, defect_ext, defect_meta = load_frame(DEFECT_DIR)

# ── Figure ──
fig, axes = plt.subplots(1, 2, figsize=(13, 6))

im0 = axes[0].imshow(grain_img, cmap=CMAP,
                      vmin=DB_FLOOR, vmax=0,
                      extent=grain_ext, aspect='equal')
axes[0].set_title('Synthetic — grain noise only', fontsize=12)
axes[0].set_xlabel('x (mm)')
axes[0].set_ylabel('z (mm)')

im1 = axes[1].imshow(defect_img, cmap=CMAP,
                      vmin=DB_FLOOR, vmax=0,
                      extent=defect_ext, aspect='equal')
axes[1].set_title('Synthetic — grain noise + cylindrical defect',
                  fontsize=12)
axes[1].set_xlabel('x (mm)')
axes[1].set_ylabel('z (mm)')

cbar_ax = fig.add_axes([0.25, 0.06, 0.5, 0.025])
cb = fig.colorbar(im1, cax=cbar_ax, orientation='horizontal')
cb.set_label('TFM amplitude (dB, normalised to frame peak)', fontsize=10)

fig.suptitle(
    f'Synthetic TFM B-scans — {grain_meta["tfm_n_pixels"]}×'
    f'{grain_meta["tfm_n_pixels"]} grid, '
    f'aperture {grain_meta["array_aperture_m"]*1e3:.1f} mm',
    fontsize=13, y=0.97,
)
plt.tight_layout(rect=[0, 0.12, 1, 0.94])
plt.savefig(OUTPUT, dpi=200, bbox_inches='tight')
print(f"Saved → {OUTPUT}")
print(f"  grain:  range [{grain_img.min():.1f}, {grain_img.max():.1f}] dB")
print(f"  defect: range [{defect_img.min():.1f}, {defect_img.max():.1f}] dB")
