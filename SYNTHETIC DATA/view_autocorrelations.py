"""
View autocorrelations produced by compute_autocorrelations.py.

Usage:
    python "SYNTHETIC DATA/view_autocorrelations.py" <path>

<path> can be:
  - a single autocorr_*.npy           → plot orthogonal slices + 1D profiles
  - a directory containing many ACFs  → print a summary table of correlation
                                        lengths; pass --plot to also show a
                                        grid of profile plots.

Correlation length along an axis is the smallest |lag| at which the 1D profile
through zero lag drops below THRESHOLD (default 1/e ≈ 0.368), with linear
interpolation between samples.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


THRESHOLD = 1.0 / np.e                   # e-folding — change to 0.5 for FWHM/2


def load_acf(path: Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Load an ACF .npy and its (dz, dy, dx) voxel size (m) from sidecar JSON."""
    acf = np.load(path)
    meta_path = path.with_suffix('.json')
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        dz, dy, dx = meta['voxel_size_m']
    else:
        dz = dy = dx = 1.0  # unitless fallback
    return acf, (dz, dy, dx)


def corr_length(profile: np.ndarray, d: float, threshold: float = THRESHOLD
                ) -> float:
    """Distance from zero-lag centre where |profile| first crosses `threshold`.

    Uses linear interpolation between samples. Returns NaN if the profile never
    crosses the threshold within the stored window.
    """
    c = profile.size // 2
    half = profile[c:]                   # 0, +d, +2d, …
    below = np.where(np.abs(half) < threshold)[0]
    if below.size == 0:
        return float('nan')
    i = below[0]
    if i == 0:
        return 0.0
    y0, y1 = abs(half[i - 1]), abs(half[i])
    frac = (y0 - threshold) / (y0 - y1) if y0 != y1 else 0.0
    return (i - 1 + frac) * d


def central_profiles(acf: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """1D profiles through zero lag along each axis (z, y, x)."""
    cz, cy, cx = [s // 2 for s in acf.shape]
    return acf[:, cy, cx], acf[cz, :, cx], acf[cz, cy, :]


def plot_single(path: Path) -> None:
    """Three orthogonal central slices + three 1D profiles for one ACF."""
    acf, (dz, dy, dx) = load_acf(path)
    Nz, Ny, Nx = acf.shape
    cz, cy, cx = Nz // 2, Ny // 2, Nx // 2

    z_axis = (np.arange(Nz) - cz) * dz * 1e3
    y_axis = (np.arange(Ny) - cy) * dy * 1e3
    x_axis = (np.arange(Nx) - cx) * dx * 1e3

    pz, py, px = central_profiles(acf)
    lz = corr_length(pz, dz) * 1e3
    ly = corr_length(py, dy) * 1e3
    lx = corr_length(px, dx) * 1e3

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    fig.suptitle(f"{path.name}  — e-folding lengths: "
                 f"z={lz:.2f} mm, y={ly:.2f} mm, x={lx:.2f} mm")

    def _imshow(ax, img, xax, yax, xlabel, ylabel, title):
        extent = [xax[0], xax[-1], yax[-1], yax[0]]
        ax.imshow(img, extent=extent, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        ax.axhline(0, color='k', lw=0.5); ax.axvline(0, color='k', lw=0.5)
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)

    _imshow(axes[0, 0], acf[cz, :, :], x_axis, y_axis,
            'x lag [mm]', 'y lag [mm]', 'Z=0 slice (x–y)')
    _imshow(axes[0, 1], acf[:, cy, :], x_axis, z_axis,
            'x lag [mm]', 'z lag [mm]', 'Y=0 slice (x–z)')
    _imshow(axes[0, 2], acf[:, :, cx], y_axis, z_axis,
            'y lag [mm]', 'z lag [mm]', 'X=0 slice (y–z)')

    for ax, axis, prof, label, L in [
        (axes[1, 0], x_axis, px, 'x', lx),
        (axes[1, 1], y_axis, py, 'y', ly),
        (axes[1, 2], z_axis, pz, 'z', lz),
    ]:
        ax.plot(axis, prof)
        ax.axhline(THRESHOLD, color='r', ls='--', lw=0.8, label=f'{THRESHOLD:.3f}')
        ax.axhline(-THRESHOLD, color='r', ls='--', lw=0.8)
        if np.isfinite(L):
            ax.axvline(L, color='g', ls=':', lw=1.0, label=f'L={L:.2f} mm')
            ax.axvline(-L, color='g', ls=':', lw=1.0)
        ax.set_xlabel(f'{label} lag [mm]'); ax.set_ylabel('ACF')
        ax.set_title(f'{label}-profile through 0'); ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.show()


def summarize_folder(folder: Path, plot: bool) -> None:
    paths = sorted(folder.rglob('autocorr_*.npy'))
    if not paths:
        print(f"No autocorr_*.npy under {folder}", file=sys.stderr)
        sys.exit(1)

    print(f"{'file':<60} {'Lz [mm]':>10} {'Ly [mm]':>10} {'Lx [mm]':>10}")
    print('-' * 92)
    rows = []
    for p in paths:
        acf, (dz, dy, dx) = load_acf(p)
        pz, py, px = central_profiles(acf)
        lz = corr_length(pz, dz) * 1e3
        ly = corr_length(py, dy) * 1e3
        lx = corr_length(px, dx) * 1e3
        rel = p.relative_to(folder)
        print(f"{str(rel):<60} {lz:>10.3f} {ly:>10.3f} {lx:>10.3f}")
        rows.append((rel, pz, py, px, (dz, dy, dx), (lz, ly, lx)))

    if not plot:
        return

    n = len(rows)
    cols = 3
    rows_fig = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows_fig, cols, figsize=(4.5 * cols, 3.2 * rows_fig),
                             squeeze=False)
    for i, (rel, pz, py, px, (dz, dy, dx), (lz, ly, lx)) in enumerate(rows):
        ax = axes[i // cols][i % cols]
        cx_ = px.size // 2; cy_ = py.size // 2; cz_ = pz.size // 2
        ax.plot((np.arange(px.size) - cx_) * dx * 1e3, px, label=f'x (L={lx:.2f})')
        ax.plot((np.arange(py.size) - cy_) * dy * 1e3, py, label=f'y (L={ly:.2f})')
        ax.plot((np.arange(pz.size) - cz_) * dz * 1e3, pz, label=f'z (L={lz:.2f})')
        ax.axhline(THRESHOLD, color='r', ls='--', lw=0.6)
        ax.set_title(str(rel), fontsize=8)
        ax.set_xlabel('lag [mm]'); ax.grid(alpha=0.3); ax.legend(fontsize=7)
    for j in range(n, rows_fig * cols):
        axes[j // cols][j % cols].axis('off')
    plt.tight_layout()
    plt.show()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('path', type=Path, help='autocorr_*.npy file or folder')
    ap.add_argument('--plot', action='store_true',
                    help='when path is a folder, also show profile plots')
    args = ap.parse_args()

    if args.path.is_file():
        plot_single(args.path)
    elif args.path.is_dir():
        summarize_folder(args.path, plot=args.plot)
    else:
        print(f"Not a file or directory: {args.path}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
