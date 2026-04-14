#!/usr/bin/env python3
"""
Compare intensity distributions of two B-scan images (.npy).
Edit the paths below, then run:  python compare_intensity.py
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

# ── EDIT THESE ───────────────────────────────────────────────────────
FILE1    = '../SYNTHETIC DATA/output/datasets/dataset_20260317_133035/pos_000/bscan_0000.npy'
FILE2    = '../DATA/1D NPY Data/Al Pure 10MHz 18032026 Vol1/bscan_0000.npy'
DB_FLOOR = None          # None = auto (shallower of the two floors), or e.g. -20.0
OUTPUT   = 'output/plots/intensity_comparison.png'
# ─────────────────────────────────────────────────────────────────────


def load_bscan(path: str) -> tuple[np.ndarray, dict | None]:
    """Load a B-scan .npy and its companion scan_meta.npy if present."""
    img = np.load(path).astype(np.float32)
    if img.ndim != 2:
        sys.exit(f"Error: {path} is not a 2D array (shape={img.shape})")

    meta_path = os.path.join(os.path.dirname(path), 'scan_meta.npy')
    meta = None
    if os.path.exists(meta_path):
        meta = np.load(meta_path, allow_pickle=True).item()
    return img, meta


def label_from_path(path: str) -> str:
    """Derive a short label from the file path."""
    parts = os.path.normpath(path).split(os.sep)
    if len(parts) >= 2:
        return f"{parts[-2]}/{parts[-1]}"
    return parts[-1]


def intensity_stats(img: np.ndarray) -> dict:
    """Compute summary statistics on the pixel intensity distribution."""
    flat = img.ravel()
    return {
        'min':      float(np.min(flat)),
        'max':      float(np.max(flat)),
        'mean':     float(np.mean(flat)),
        'median':   float(np.median(flat)),
        'std':      float(np.std(flat)),
        'skewness': float(stats.skew(flat)),
        'kurtosis': float(stats.kurtosis(flat)),
    }


# ── load ──
img1, meta1 = load_bscan(FILE1)
img2, meta2 = load_bscan(FILE2)
label1 = label_from_path(FILE1)
label2 = label_from_path(FILE2)

# ── detect dB floors and clip to common range ──
floor1, floor2 = float(img1.min()), float(img2.min())
common_floor = DB_FLOOR if DB_FLOOR is not None else max(floor1, floor2)

img1_clip = np.clip(img1, common_floor, 0.0)
img2_clip = np.clip(img2, common_floor, 0.0)

s1 = intensity_stats(img1_clip)
s2 = intensity_stats(img2_clip)

# ── print summary ──
print(f"\n{'Statistic':<14}  {'File 1':>10}  {'File 2':>10}")
print('-' * 40)
for key in s1:
    print(f"  {key:<12}  {s1[key]:>10.3f}  {s2[key]:>10.3f}")
print(f"\n  Raw dB range   [{floor1:.1f}, 0]      [{floor2:.1f}, 0]")
print(f"  Common floor   {common_floor:.1f} dB")

# KS test
ks_stat, ks_p = stats.ks_2samp(img1_clip.ravel(), img2_clip.ravel())
print(f"\n  KS statistic   {ks_stat:.4f}  (p = {ks_p:.2e})")

# ── figure ──
fig, axes = plt.subplots(2, 2, figsize=(11, 9))
fig.suptitle('B-Scan Intensity Distribution Comparison', fontsize=13, y=0.97)

bins = np.linspace(common_floor, 0, 120)

# (0,0) — overlaid histograms
ax = axes[0, 0]
ax.hist(img1_clip.ravel(), bins=bins, density=True, alpha=0.55, label=label1)
ax.hist(img2_clip.ravel(), bins=bins, density=True, alpha=0.55, label=label2)
ax.set_xlabel('Intensity (dB)')
ax.set_ylabel('Probability density')
ax.set_title('Histogram')
ax.legend(fontsize=7)

# (0,1) — empirical CDFs
ax = axes[0, 1]
for img_c, lab in [(img1_clip, label1), (img2_clip, label2)]:
    sorted_vals = np.sort(img_c.ravel())
    cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
    step = max(1, len(sorted_vals) // 2000)
    ax.plot(sorted_vals[::step], cdf[::step], label=lab, linewidth=1.2)
ax.set_xlabel('Intensity (dB)')
ax.set_ylabel('Cumulative probability')
ax.set_title('Empirical CDF')
ax.legend(fontsize=7)

# (1,0) — Q-Q plot
ax = axes[1, 0]
n_qq = min(5000, len(img1_clip.ravel()), len(img2_clip.ravel()))
q_levels = np.linspace(0, 100, n_qq)
q1 = np.percentile(img1_clip, q_levels)
q2 = np.percentile(img2_clip, q_levels)
ax.scatter(q1, q2, s=1, alpha=0.4)
lim = [common_floor, 0]
ax.plot(lim, lim, 'k--', linewidth=0.8, label='y = x')
ax.set_xlabel(f'Quantiles — {label1} (dB)')
ax.set_ylabel(f'Quantiles — {label2} (dB)')
ax.set_title('Q–Q Plot')
ax.legend(fontsize=7)
ax.set_aspect('equal')

# (1,1) — statistics table
ax = axes[1, 1]
ax.axis('off')
rows = ['min (dB)', 'max (dB)', 'mean (dB)', 'median (dB)',
        'std (dB)', 'skewness', 'kurtosis']
keys = list(s1.keys())
table_data = [[f'{s1[k]:.3f}', f'{s2[k]:.3f}'] for k in keys]
table_data.append([f'{ks_stat:.4f}', f'p = {ks_p:.2e}'])
rows.append('KS test')
table = ax.table(
    cellText=table_data,
    rowLabels=rows,
    colLabels=['File 1', 'File 2'],
    loc='center',
    cellLoc='center',
)
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1.0, 1.4)
ax.set_title('Summary Statistics', pad=15)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(OUTPUT, dpi=180)
print(f"\n  Figure saved → {OUTPUT}\n")
