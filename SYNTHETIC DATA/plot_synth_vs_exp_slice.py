"""
Side-by-side 2-D B-scan cross-section: synthetic vs experimental.

Picks a representative XZ slice (constant y) from the synthetic volume,
then searches all experimental 3-D TFM volumes (and their y-slices) for
the slice whose intensity distribution best matches the synthetic one
(minimum 2-sample KS distance). Saves the matched pair side-by-side.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

SYN_PATHS = [
    HERE / 'output' / 'engine_3d_tuned' / 'volume_0000.npz',
    HERE / 'output' / 'engine_3d' / 'volume_0000.npz',
    HERE / 'output' / 'engine_3d_overlap_sweep' / 'ovlp_080' / 'volume_A.npz',
    HERE / 'output' / 'engine_3d_overlap_sweep' / 'ovlp_080' / 'volume_B.npz',
]
EXP_DIR  = ROOT / 'DATA' / '2D TFM Data' / 'Cu Pure 7.5MHz Ex 15042026 Filtered'
OUT_PATH = HERE / 'output' / 'plots' / 'synth_vs_exp_slice.png'

# Physical pixel sizes (from Params.txt)
EXP_DZ_MM, EXP_DX_MM, EXP_DY_MM = 0.040, 0.039, 0.040
EXP_Z_MIN_MM, EXP_Z_MAX_MM = 15.0, 35.0
DB_FLOOR = -20.0


def corr_length_1d(profile_2d, axis: int, dx_mm: float) -> float:
    """Mean 1/e correlation length (mm) along `axis` of a 2-D image.

    Subtracts the row/column mean, autocorrelates each line, normalises by
    the zero-lag value, then averages across lines. Returns the lag at
    which the mean ACF first drops below 1/e."""
    img = profile_2d - profile_2d.mean(axis=axis, keepdims=True)
    n = img.shape[axis]
    F = np.fft.rfft(img, n=2 * n, axis=axis)
    acf = np.fft.irfft(F * np.conj(F), n=2 * n, axis=axis)[:n] if axis == 0 \
          else np.fft.irfft(F * np.conj(F), n=2 * n, axis=axis)[:, :n]
    # Reduce: average along the orthogonal axis, normalise by zero-lag
    other = 1 - axis
    mean_acf = acf.mean(axis=other)
    if mean_acf[0] <= 0:
        return float('nan')
    mean_acf = mean_acf / mean_acf[0]
    below = np.where(mean_acf < np.exp(-1))[0]
    return float((below[0] if below.size else n) * dx_mm)


def slice_features(sl, dz_mm, dx_mm):
    """Return (flat_intensity, Lz_mm, Lx_mm) for one 2-D slice."""
    return sl.ravel(), corr_length_1d(sl, 0, dz_mm), corr_length_1d(sl, 1, dx_mm)


N_SAMPLE = 4000     # subsample size for KS test (fast and accurate enough)


def collect_syn(syn_paths, db_floor=DB_FLOOR, stride=10):
    """Extract candidate slices from synthetic volumes; free volume after."""
    rng = np.random.default_rng(0)
    out = []
    for p in syn_paths:
        if not p.exists():
            continue
        d = np.load(p)
        img = np.asarray(d['img_db'], dtype=np.float32)
        x = d['x'] * 1e3
        z = d['z'] * 1e3
        dx_mm = float(abs(x[1] - x[0]))
        dz_mm = float(abs(z[1] - z[0]))
        Ny = img.shape[1]
        extent = [float(x[0]), float(x[-1]), float(z[-1]), float(z[0])]
        for iy in range(stride, Ny - stride, stride):
            sl = np.clip(img[:, iy, :], db_floor, 0.0)
            flat = sl.ravel()
            samp = np.sort(rng.choice(flat, size=min(N_SAMPLE, flat.size),
                                      replace=False))
            out.append({
                'slice': sl.copy(), 'extent': extent, 'samp': samp,
                'Lz': corr_length_1d(sl, 0, dz_mm),
                'Lx': corr_length_1d(sl, 1, dx_mm),
                'src': f'{p.parent.name}/{p.name}', 'iy': iy,
            })
        del img
    return out


def collect_exp(exp_files, db_floor=DB_FLOOR, stride=10):
    rng = np.random.default_rng(1)
    out = []
    for f in exp_files:
        vol = np.clip(np.load(f).astype(np.float32), db_floor, 0.0)
        Ny = vol.shape[1]
        for iy in range(stride, Ny - stride, stride):
            sl = vol[:, iy, :]
            flat = sl.ravel()
            samp = np.sort(rng.choice(flat, size=min(N_SAMPLE, flat.size),
                                      replace=False))
            out.append({
                'slice': sl.copy(), 'samp': samp,
                'Lz': corr_length_1d(sl, 0, EXP_DZ_MM),
                'Lx': corr_length_1d(sl, 1, EXP_DX_MM),
                'file': f, 'iy': iy,
            })
        del vol
    return out


def ks_from_sorted(a_sorted, b_sorted):
    """KS distance from two pre-sorted arrays — same answer as ks_2samp."""
    grid = np.concatenate([a_sorted, b_sorted])
    grid.sort()
    cdf_a = np.searchsorted(a_sorted, grid, side='right') / a_sorted.size
    cdf_b = np.searchsorted(b_sorted, grid, side='right') / b_sorted.size
    return float(np.max(np.abs(cdf_a - cdf_b)))


def best_pair(syn_paths, exp_files,
              syn_stride=10, exp_stride=10,
              alpha_L=0.6, L_scale_mm=0.4):
    """Score = KS_D + alpha_L * (|ΔLz| + |ΔLx|) / L_scale_mm."""
    print('  loading synthetic volumes...')
    syn_list = collect_syn(syn_paths, stride=syn_stride)
    print(f'    {len(syn_list)} synthetic slices')
    print('  loading experimental volumes...')
    exp_list = collect_exp(exp_files, stride=exp_stride)
    print(f'    {len(exp_list)} experimental slices')
    print(f'  comparing {len(syn_list) * len(exp_list)} pairs...')

    best = {'score': np.inf}
    for s in syn_list:
        for e in exp_list:
            D = ks_from_sorted(s['samp'], e['samp'])
            dL = (abs(s['Lz'] - e['Lz']) + abs(s['Lx'] - e['Lx'])) / L_scale_mm
            score = D + alpha_L * dL
            if score < best['score']:
                best = {
                    'score': float(score), 'D': float(D),
                    'syn_slice': s['slice'], 'syn_extent': s['extent'],
                    'syn_src': s['src'], 'syn_iy': s['iy'],
                    'syn_Lz': s['Lz'], 'syn_Lx': s['Lx'],
                    'exp_slice': e['slice'], 'exp_file': e['file'],
                    'exp_iy': e['iy'],
                    'exp_Lz': e['Lz'], 'exp_Lx': e['Lx'],
                }
    return best


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    exp_files = sorted(EXP_DIR.glob('*_filtered_3D_TFM.npy'))
    if not exp_files:
        raise SystemExit(f'No experimental volumes found in {EXP_DIR}')
    syn_paths = [p for p in SYN_PATHS if p.exists()]
    if not syn_paths:
        raise SystemExit('No synthetic volumes found')
    print(f'searching {len(syn_paths)} synthetic vs {len(exp_files)} '
          f'experimental volumes...')

    best = best_pair(syn_paths, exp_files)

    syn_sl, syn_ext = best['syn_slice'], best['syn_extent']
    exp_sl = best['exp_slice']
    Nz, Nx = exp_sl.shape
    exp_x_half = (Nx * EXP_DX_MM) / 2.0
    exp_ext = [-exp_x_half, exp_x_half, EXP_Z_MAX_MM, EXP_Z_MIN_MM]
    print(f'best pair: score={best["score"]:.4f}  KS D={best["D"]:.4f}')
    print(f'  synthetic    : {best["syn_src"]}  iy={best["syn_iy"]}  '
          f'Lz={best["syn_Lz"]:.2f}mm Lx={best["syn_Lx"]:.2f}mm')
    print(f'  experimental : {best["exp_file"].name}  iy={best["exp_iy"]}  '
          f'Lz={best["exp_Lz"]:.2f}mm Lx={best["exp_Lx"]:.2f}mm')

    AX, TICK, TITLE = 18, 15, 19

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13, 6.5),
                                     gridspec_kw={'wspace': 0.20})

    ax_a.imshow(syn_sl, extent=syn_ext, origin='upper',
                cmap='gray', vmin=DB_FLOOR, vmax=0.0, aspect='equal')
    ax_a.set_title('Synthetic', fontsize=TITLE)
    ax_a.set_xlabel('x (mm)', fontsize=AX)
    ax_a.set_ylabel('z (mm)', fontsize=AX)
    ax_a.tick_params(labelsize=TICK)

    im_b = ax_b.imshow(exp_sl, extent=exp_ext, origin='upper',
                       cmap='gray', vmin=DB_FLOOR, vmax=0.0, aspect='equal')
    ax_b.set_title('Experimental', fontsize=TITLE)
    ax_b.set_xlabel('x (mm)', fontsize=AX)
    ax_b.set_ylabel('z (mm)', fontsize=AX)
    ax_b.tick_params(labelsize=TICK)

    cbar = fig.colorbar(im_b, ax=[ax_a, ax_b], shrink=0.85, pad=0.02)
    cbar.set_label('Intensity (dB)', fontsize=AX)
    cbar.ax.tick_params(labelsize=TICK)

    fig.savefig(OUT_PATH, dpi=180, bbox_inches='tight')
    print(f'wrote {OUT_PATH}')


if __name__ == '__main__':
    main()
