"""
Compare autocorrelations between two (or more) folders.

Each positional argument is a folder containing autocorr_*.npy files (searched
recursively). Each folder becomes one "group" — e.g.

    python "SYNTHETIC DATA/compare_autocorrelations.py" \
        "SYNTHETIC DATA/output/autocorrelations" \
        "SYNTHETIC DATA/output/autocorrelations_Cu_experimental"

Outputs:
  1. A table of mean ± std correlation length (Lz, Ly, Lx) per group.
  2. A 1×3 figure showing the mean 1D ACF profile (±1σ band) along each axis,
     interpolated onto a common lag grid so groups with different voxel
     spacing can be directly overlaid.

Options:
  --label LABEL      Custom group label (pass once per folder).
  --exclude SUBSTR   Skip any file whose name contains this substring
                     (repeatable). E.g. --exclude Calibration.
  --no-plot          Print the table only.
  --max-lag MM       Common lag axis half-extent (default 3 mm).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from view_autocorrelations import (                # noqa: E402
    load_acf, central_profiles, corr_length, THRESHOLD,
)


N_POINTS_DEFAULT = 401


def gather(folders: list[Path], label: str, exclude: list[str],
           common_lag_mm: float, n_points: int) -> dict | None:
    paths: list[Path] = []
    for folder in folders:
        paths.extend(sorted(folder.rglob('autocorr_*.npy')))
    if exclude:
        paths = [p for p in paths if not any(e in p.name for e in exclude)]
    if not paths:
        return None

    common = np.linspace(-common_lag_mm, common_lag_mm, n_points)
    prof = {'z': [], 'y': [], 'x': []}
    Ls   = {'z': [], 'y': [], 'x': []}

    for p in paths:
        acf, (dz, dy, dx) = load_acf(p)
        pz, py, px = central_profiles(acf)
        for axis, profile, d in [('z', pz, dz), ('y', py, dy), ('x', px, dx)]:
            c = profile.size // 2
            lags_mm = (np.arange(profile.size) - c) * d * 1e3
            prof[axis].append(np.interp(common, lags_mm, profile,
                                        left=np.nan, right=np.nan))
            Ls[axis].append(corr_length(profile, d) * 1e3)

    return {
        'label': label,
        'folders': folders,
        'n': len(paths),
        'common_lag_mm': common,
        'profiles': {k: np.array(v) for k, v in prof.items()},
        'Ls': {k: np.array(v) for k, v in Ls.items()},
    }


def print_table(groups: list[dict]) -> None:
    header = f"{'group':<45} {'n':>4}   {'Lz [mm]':>13} {'Ly [mm]':>13} {'Lx [mm]':>13}"
    print(header)
    print('-' * len(header))
    for g in groups:
        row = f"{g['label']:<45} {g['n']:>4}   "
        for ax in ('z', 'y', 'x'):
            arr = g['Ls'][ax]
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                row += f"{'—':>12}  "
            else:
                row += f"  {arr.mean():5.2f} ± {arr.std():4.2f}"
        print(row)


def plot_comparison(groups: list[dict]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    for ax, axis_label in zip(axes, ('z', 'y', 'x')):
        for g in groups:
            common = g['common_lag_mm']
            prof = g['profiles'][axis_label]
            mean = np.nanmean(prof, axis=0)
            std  = np.nanstd(prof, axis=0)
            line, = ax.plot(common, mean, lw=1.8, label=g['label'])
            ax.fill_between(common, mean - std, mean + std,
                            alpha=0.18, color=line.get_color(), linewidth=0)
        ax.axhline(THRESHOLD, color='k', ls='--', lw=0.6, label=f'1/e = {THRESHOLD:.3f}'
                   if axis_label == 'z' else None)
        ax.axvline(0, color='k', lw=0.4)
        ax.set_title(f'{axis_label}-axis mean ACF  (±1σ band)')
        ax.set_xlabel('lag [mm]')
        ax.grid(alpha=0.3)
        if axis_label == 'z':
            ax.set_ylabel('autocorrelation')
            ax.legend(fontsize=8, loc='upper right')
    plt.tight_layout()
    plt.show()


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('folders', nargs='+', type=str,
                    help='one group per positional arg. A group is a folder of '
                         'autocorr_*.npy files, OR a comma-separated list of '
                         'folders that are merged into one group.')
    ap.add_argument('--label', action='append', default=None,
                    help='custom label for each group (repeat per group)')
    ap.add_argument('--exclude', action='append', default=[],
                    help='skip files whose name contains this substring')
    ap.add_argument('--no-plot', action='store_true')
    ap.add_argument('--max-lag', type=float, default=3.0,
                    help='common lag axis half-extent in mm (default 3.0)')
    ap.add_argument('--n-points', type=int, default=N_POINTS_DEFAULT)
    args = ap.parse_args()

    groups_spec = [[Path(p) for p in spec.split(',')] for spec in args.folders]
    labels = args.label or [' + '.join(f.name for f in grp) for grp in groups_spec]
    if len(labels) != len(groups_spec):
        sys.exit(f"{len(groups_spec)} groups but {len(labels)} labels")

    groups = [gather(grp, lbl, args.exclude, args.max_lag, args.n_points)
              for grp, lbl in zip(groups_spec, labels)]
    groups = [g for g in groups if g is not None]
    if not groups:
        sys.exit("No autocorr_*.npy files found in any folder")

    print_table(groups)
    if not args.no_plot:
        plot_comparison(groups)


if __name__ == '__main__':
    main()
