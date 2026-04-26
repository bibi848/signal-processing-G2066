"""
Two-panel report figure: (a) slice from the legacy random-fill + Gaussian-blur
synthetic volume vs (b) experimental TFM B-scan of pure aluminum.

Both panels are dB-normalised (re panel max) on the same scale so the speckle
character can be compared directly.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.signal import hilbert

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'build' / 'CPP' / 'TFM'))
sys.path.insert(0, str(ROOT))
import tfm_cpp  # noqa: E402
from Classes.Stitch3D import normalised_correlation_3D  # noqa: E402


SYNTH_VOL_PATH = ROOT / 'SYNTHETIC DATA' / 'legacy' / 'synthetic_volume_clean.npy'
EXP_FRAME_PATH = (ROOT / 'DATA' / '1D Processed Data'
                  / 'Al Pure 10MHz 18032026 Filtered' / '1_15_filtered')

NCC_EXP_CSV   = Path(__file__).resolve().parent / 'output' / 'Set_1.csv'
NCC_SYNTH_CSV = Path(__file__).resolve().parent / 'output' / 'Set_2.csv'

OUT_PATH     = Path(__file__).resolve().parent / 'output' / 'initial_vs_experimental.png'
OUT_NCC_PATH = Path(__file__).resolve().parent / 'output' / 'ncc_synth_vs_experimental.png'

C_AL    = 6300.0
Z_MIN_M = 10e-3
Z_MAX_M = 40e-3
N_PIX   = 400
DB_FLOOR = -20.0

# Synthetic 3D stitching test parameters --------------------------------------
# Volume axis order is (z, x, y) to match Classes/Stitch3D convention.
SYNTH_VOL_SHAPE      = (80, 400, 40)    # (z, x, y)
SYNTH_VOL_SIGMA      = 2.5
SYNTH_OVERLAP_FRAC   = 0.79             # tuned so peak ≈ experimental peak (~70 vox)
SYNTH_NCC_MAX_SHIFT  = 180


def synth_slice_db(shape=(N_PIX, N_PIX), smoothing_sigma=2.5, seed=0):
    """Reproduce the initial method: uniform random fill + isotropic Gaussian
    blur. The legacy generator filled the volume with low-amplitude uniform
    noise and blurred it; here we keep the same recipe but raise the field to
    a higher power before dB-normalising so the speckle dynamic range matches
    the experimental envelope display."""
    rng = np.random.default_rng(seed)
    base = rng.random(shape, dtype=np.float32)
    field = gaussian_filter(base, sigma=smoothing_sigma)
    field = field - field.min()
    field = field / field.max()
    field = field ** 4
    field = field / field.max()
    return 20.0 * np.log10(np.maximum(field, 10 ** (DB_FLOOR / 20.0)))


def synth_volume_3d(shape=SYNTH_VOL_SHAPE, smoothing_sigma=SYNTH_VOL_SIGMA, seed=0):
    """Legacy method 3D: uniform random fill + isotropic Gaussian blur."""
    rng = np.random.default_rng(seed)
    base = rng.random(shape, dtype=np.float32)
    return gaussian_filter(base, sigma=smoothing_sigma)


def synth_overlap_pair(volume, overlap_frac=SYNTH_OVERLAP_FRAC):
    """Slice a volume (z, x, y) along x into two equal subvolumes that share
    `overlap_frac` of their x-extent. Returns (vol1, vol2, expected_shift)."""
    Nx_total = volume.shape[1]
    sub_x = int(Nx_total / (2 - overlap_frac))
    offset = sub_x - int(sub_x * overlap_frac)
    vol1 = volume[:, :sub_x, :]
    vol2 = volume[:, offset:offset + sub_x, :]
    return vol1, vol2, offset


def experimental_tfm_db():
    metadata = pd.read_csv(EXP_FRAME_PATH / 'metadata.csv')
    time_sec = pd.read_csv(EXP_FRAME_PATH / 'time.csv')['time_seconds'].values
    tx_rx    = pd.read_csv(EXP_FRAME_PATH / 'tx_rx.csv')
    geometry = pd.read_csv(EXP_FRAME_PATH / 'array_geometry.csv')
    with h5py.File(EXP_FRAME_PATH / 'time_data.h5', 'r') as h:
        time_data = h['time_data'][:]

    tx0 = tx_rx['tx'].values.astype(int) - 1
    rx0 = tx_rx['rx'].values.astype(int) - 1
    xc  = geometry['el_xc'].values
    zc  = geometry['el_zc'].values

    x_img = np.linspace(xc.min(), xc.max(), N_PIX)
    z_img = np.linspace(Z_MIN_M, Z_MAX_M, N_PIX)
    X, Z  = np.meshgrid(x_img, z_img)

    img = tfm_cpp.tfm1D(time_data, time_sec, tx0, rx0, xc, zc, X, Z, C_AL)
    env = np.abs(hilbert(img, axis=0))
    db  = 20.0 * np.log10(np.maximum(env / env.max(), 10 ** (DB_FLOOR / 20.0)))
    return db, x_img * 1e3, z_img * 1e3


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    synth_db = synth_slice_db()
    exp_db, exp_x_mm, exp_z_mm = experimental_tfm_db()

    # In-script synthetic NCC (legacy method, computed live) + experimental CSV
    synth_vol = synth_volume_3d()
    vol1, vol2, expected_shift = synth_overlap_pair(synth_vol)
    _, shifts_live, ncc_live = normalised_correlation_3D(
        vol1, vol2, axis='x', max_shift=SYNTH_NCC_MAX_SHIFT)
    shifts_live = np.asarray(list(shifts_live))
    ncc_exp_csv = pd.read_csv(NCC_EXP_CSV)

    AX, TICK, LET = 20, 17, 28

    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(16, 11))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.0, 0.55],
                  hspace=0.02, wspace=0.20,
                  left=0.07, right=0.97, top=0.99, bottom=0.09)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])

    # (a) Initial method synthetic slice (same mm extent as (b)) ----------
    ext_a = [exp_x_mm[0], exp_x_mm[-1], exp_z_mm[-1], exp_z_mm[0]]
    im_a = ax_a.imshow(synth_db, extent=ext_a, origin='upper',
                       cmap='gray', vmin=DB_FLOOR, vmax=0.0,
                       aspect='equal')
    ax_a.set_xlabel('x (mm)', fontsize=AX)
    ax_a.set_ylabel('z (mm)', fontsize=AX)
    ax_a.tick_params(labelsize=TICK)
    ax_a.text(0.02, 0.98, '(a)', transform=ax_a.transAxes,
              fontsize=LET, fontweight='bold', va='top', ha='left',
              color='white',
              bbox=dict(facecolor='black', alpha=0.55,
                        edgecolor='none', pad=4))

    # (b) Experimental TFM (standard NDT orientation: z vertical, z=10 at top) ----
    ext_b = [exp_x_mm[0], exp_x_mm[-1], exp_z_mm[-1], exp_z_mm[0]]
    im_b = ax_b.imshow(exp_db, extent=ext_b, origin='upper',
                       cmap='gray', vmin=DB_FLOOR, vmax=0.0,
                       aspect='equal')
    ax_b.set_xlabel('x (mm)', fontsize=AX)
    ax_b.set_ylabel('z (mm)', fontsize=AX)
    ax_b.tick_params(labelsize=TICK)
    ax_b.text(0.02, 0.98, '(b)', transform=ax_b.transAxes,
              fontsize=LET, fontweight='bold', va='top', ha='left',
              color='white',
              bbox=dict(facecolor='black', alpha=0.55,
                        edgecolor='none', pad=4))

    # (c) NCC: in-script legacy synthetic + experimental CSV --------------
    def _norm01(y):
        y = np.asarray(y, dtype=np.float64)
        return (y - y.min()) / (y.max() - y.min())

    ax_c.plot(shifts_live, _norm01(ncc_live),
              color='tab:blue', linewidth=2.0,
              label='Synthetic')
    ax_c.plot(ncc_exp_csv['shift'], _norm01(ncc_exp_csv['correlation']),
              color='tab:orange', linewidth=2.0,
              label='Experimental')
    ax_c.set_xlabel('Lateral shift (voxels)', fontsize=AX)
    ax_c.set_ylabel('NCC (normalised)', fontsize=AX)
    ax_c.tick_params(labelsize=TICK)
    x_lo = min(shifts_live[0], ncc_exp_csv['shift'].min())
    x_hi = max(shifts_live[-1], ncc_exp_csv['shift'].max())
    ax_c.set_xlim(x_lo, x_hi)
    ax_c.set_ylim(0.0, 1.05)
    ax_c.legend(fontsize=TICK, loc='upper right', framealpha=0.9)
    ax_c.text(0.01, 0.98, '(c)', transform=ax_c.transAxes,
              fontsize=LET, fontweight='bold', va='top', ha='left')
    ax_c.grid(True, alpha=0.3)

    fig.savefig(OUT_PATH, dpi=150)
    plt.close(fig)
    print(f'wrote {OUT_PATH}')
    print(f'in-script synth: expected shift = {expected_shift} vox, '
          f'recovered = {shifts_live[int(np.argmax(ncc_live))]} vox')

    # Separate NCC figure: legacy synthetic vs experimental ---------------
    ncc_synth = pd.read_csv(NCC_SYNTH_CSV)
    ncc_exp   = pd.read_csv(NCC_EXP_CSV)

    fig2, ax_ncc = plt.subplots(1, 1, figsize=(12, 6),
                                gridspec_kw={'left': 0.10, 'right': 0.97,
                                             'top': 0.95, 'bottom': 0.14})
    ax_ncc.plot(ncc_synth['shift'], _norm01(ncc_synth['correlation']),
                color='tab:blue', linewidth=2.0, label='Synthetic')
    ax_ncc.plot(ncc_exp['shift'], _norm01(ncc_exp['correlation']),
                color='tab:orange', linewidth=2.0, label='Experimental')
    ax_ncc.set_xlabel('Lateral shift (voxels)', fontsize=AX)
    ax_ncc.set_ylabel('NCC (normalised)', fontsize=AX)
    ax_ncc.tick_params(labelsize=TICK)
    x_lo = min(ncc_synth['shift'].min(), ncc_exp['shift'].min())
    x_hi = max(ncc_synth['shift'].max(), ncc_exp['shift'].max())
    ax_ncc.set_xlim(x_lo, x_hi)
    ax_ncc.set_ylim(0.0, 1.05)
    ax_ncc.legend(fontsize=TICK, loc='upper right', framealpha=0.9)
    ax_ncc.grid(True, alpha=0.3)
    fig2.savefig(OUT_NCC_PATH, dpi=150)
    plt.close(fig2)
    print(f'wrote {OUT_NCC_PATH}')


if __name__ == '__main__':
    main()
