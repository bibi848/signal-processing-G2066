"""
Report figures from a Radon-validation output directory.

Produces, in output/<subdir>/ (default subdir: 'radon_validation'):
  - mip_N###.png           max-intensity-projection triptych per N.
  - vol3d_N###.png         3D scatter of voxels above the dB floor.
  - psf_sweep.png          3x3 panel (scatterers x axes) showing 1-D line
                           profiles through each reconstructed peak.
  - radon_pipeline.png     TFM slices -> sinogram -> inverse-Radon schematic.

Run from the SYNTHETIC DATA directory:
    python tests/plot_report_figures.py                       # clean validation
    python tests/plot_report_figures.py radon_grain_validation # grain volume
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import iradon

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from test_radon_validation import (
    SCATTERERS, TFM_N_PIXELS, TFM_Z_START, TFM_Z_END,
    N_ELEMENTS, ELEMENT_PITCH, ROI_MM,
)

# Optional CLI arg picks a different output subdir (e.g. radon_grain_validation).
_SUBDIR = sys.argv[1] if len(sys.argv) > 1 else 'radon_validation'
OUTPUT_DIR = HERE / 'output' / _SUBDIR


def axes_mm():
    z = np.linspace(TFM_Z_START, TFM_Z_END, TFM_N_PIXELS) * 1e3
    half = (N_ELEMENTS - 1) * ELEMENT_PITCH / 2 * 1e3
    xy = np.linspace(-half, half, TFM_N_PIXELS)
    return z, xy, half


def find_peak(volume, z_mm, xy_mm, truth_xyz_mm, roi_mm=ROI_MM):
    sx, sy, sz = truth_xyz_mm
    dz  = z_mm[1]  - z_mm[0]
    dxy = xy_mm[1] - xy_mm[0]
    rz  = int(round(roi_mm / dz))
    rxy = int(round(roi_mm / dxy))
    iz0 = int(np.argmin(np.abs(z_mm  - sz)))
    iy0 = int(np.argmin(np.abs(xy_mm - sy)))
    ix0 = int(np.argmin(np.abs(xy_mm - sx)))
    zs = slice(max(iz0 - rz, 0),  min(iz0 + rz + 1, volume.shape[0]))
    ys = slice(max(iy0 - rxy, 0), min(iy0 + rxy + 1, volume.shape[1]))
    xs = slice(max(ix0 - rxy, 0), min(ix0 + rxy + 1, volume.shape[2]))
    roi = volume[zs, ys, xs]
    piz, piy, pix = np.unravel_index(int(np.argmax(roi)), roi.shape)
    return zs.start + piz, ys.start + piy, xs.start + pix


DB_FLOOR = -20.0


def _to_db(arr, ref):
    eps = ref * 10 ** (DB_FLOOR / 20.0)
    return 20.0 * np.log10(np.maximum(arr, eps) / ref)


def mip_triptych(volume, N, z_mm, xy_mm, half_mm, out_path):
    """Three MIPs in dB (re volume max): xy (top-down), xz (side), yz (front)."""
    ref = float(volume.max())
    mip_xy = _to_db(volume.max(axis=0), ref)
    mip_xz = _to_db(volume.max(axis=1), ref)
    mip_yz = _to_db(volume.max(axis=2), ref)

    vmin, vmax = DB_FLOOR, 0.0
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6),
                             gridspec_kw={'wspace': 0.28})

    ext_xy = [-half_mm, +half_mm, -half_mm, +half_mm]
    im0 = axes[0].imshow(mip_xy, extent=ext_xy, origin='lower',
                         cmap='gray', vmin=vmin, vmax=vmax, aspect='equal')
    axes[0].set(xlabel='x (mm)', ylabel='y (mm)',
                title=f'top-down  (z = {z_mm[0]:.0f} to {z_mm[-1]:.0f} mm)')

    ext_xz = [-half_mm, +half_mm, z_mm[-1], z_mm[0]]
    im1 = axes[1].imshow(mip_xz, extent=ext_xz, origin='upper',
                         cmap='gray', vmin=vmin, vmax=vmax, aspect='equal')
    axes[1].set(xlabel='x (mm)', ylabel='z (mm)',
                title=f'side  (y = {-half_mm:+.0f} to {+half_mm:+.0f} mm)')

    ext_yz = [-half_mm, +half_mm, z_mm[-1], z_mm[0]]
    im2 = axes[2].imshow(mip_yz, extent=ext_yz, origin='upper',
                         cmap='gray', vmin=vmin, vmax=vmax, aspect='equal')
    axes[2].set(xlabel='y (mm)', ylabel='z (mm)',
                title=f'front  (x = {-half_mm:+.0f} to {+half_mm:+.0f} mm)')

    step_deg = 180.0 / N
    fig.suptitle(
        f'Reconstructed volume - N = {N} scans  '
        f'(angle step = {step_deg:.2f} deg)',
        fontsize=12,
    )
    fig.colorbar(im2, ax=axes, shrink=0.82, label='amplitude (dB re max)')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def radon_pipeline_figure(out_path, z0_target_mm=20.0):
    """Pedagogical 3-panel figure showing the Radon re-indexing:
         (a) stack of 2D TFM slices indexed by angle,
         (b) sinogram at one depth z0 (re-indexed rows),
         (c) inverse-Radon reconstruction at z0.
    Reads the cached complex B-scans written by test_radon_validation.py."""
    bscans_path = OUTPUT_DIR / 'bscans_complex.npy'
    angles_path = OUTPUT_DIR / 'angles_rad.npy'
    if not (bscans_path.exists() and angles_path.exists()):
        print(f"Skipping pipeline figure; missing cache in {OUTPUT_DIR}")
        return

    bscans = np.abs(np.load(bscans_path)).astype(np.float32)   # (n_ang, n_z, n_x)
    angles = np.load(angles_path)
    z_mm, xy_mm, _ = axes_mm()

    iz0 = int(np.argmin(np.abs(z_mm - z0_target_mm)))
    z0  = float(z_mm[iz0])

    # Scatterers within ~2 mm of the chosen depth (these show up at this slice)
    at_z0 = [(sx * 1e3, sy * 1e3, sz * 1e3) for sx, sy, sz, _ in SCATTERERS
             if abs(sz * 1e3 - z0) < 2.0]

    # Angles closest to 0, 45, 90, 135 deg
    target_deg = [0.0, 45.0, 90.0, 135.0]
    sel_idx = [int(np.argmin(np.abs(np.degrees(angles) - t))) for t in target_deg]

    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.25],
                          hspace=0.45, wspace=0.4)

    # ---- (a) TFM slices at selected angles ----
    for k, i in enumerate(sel_idx):
        ax = fig.add_subplot(gs[0, k])
        img = bscans[i]
        img_db = _to_db(img, float(img.max()))
        ax.imshow(img_db,
                  extent=[xy_mm[0], xy_mm[-1], z_mm[-1], z_mm[0]],
                  origin='upper', cmap='gray',
                  vmin=DB_FLOOR, vmax=0.0, aspect='auto')
        ax.axhline(z0, color='yellow', ls='--', lw=0.9, alpha=0.9)
        ax.set_xlabel(r'$x_\theta$ (mm)')
        if k == 0:
            ax.set_ylabel('z (mm)')
        ax.set_title(r'$\theta$ = ' + f'{np.degrees(angles[i]):.0f}' + r'$\degree$',
                     fontsize=10)
    fig.text(0.5, 0.965,
             '(a)  TFM slices, one per scan angle '
             '(yellow dashed line = depth $z_0$ sampled for sinogram)',
             ha='center', fontsize=12, fontweight='bold')

    # ---- (b) Sinogram at z0 ----
    ax_b = fig.add_subplot(gs[1, 0:2])
    sino = bscans[:, iz0, :]                                   # (n_ang, n_x)
    sino_db = _to_db(sino, float(sino.max()))
    ang_deg = np.degrees(angles)
    ax_b.imshow(sino_db.T,
                extent=[ang_deg[0], ang_deg[-1], xy_mm[0], xy_mm[-1]],
                origin='lower', cmap='gray',
                vmin=DB_FLOOR, vmax=0.0, aspect='auto')
    theta_smooth = np.linspace(0.0, np.pi, 400)
    trace_colors = ['tab:red', 'tab:cyan', 'tab:orange']
    for c, (sx, sy, _sz) in enumerate(at_z0):
        rho = sx * np.cos(theta_smooth) + sy * np.sin(theta_smooth)
        ax_b.plot(np.degrees(theta_smooth), rho,
                  color=trace_colors[c % len(trace_colors)], lw=1.4,
                  label=f'(x, y) = ({sx:+.0f}, {sy:+.0f}) mm')
    ax_b.set_xlabel(r'$\theta$ (deg)')
    ax_b.set_ylabel(r'$x_\theta$ (mm)')
    ax_b.set_title(f'(b)  Sinogram at $z_0$ = {z0:.1f} mm  '
                   r'(traces: $x_\theta = x\cos\theta + y\sin\theta$)',
                   fontsize=12, fontweight='bold')
    ax_b.legend(loc='upper right', fontsize=9)

    # ---- (c) Reconstructed slice at z0 ----
    ax_c = fig.add_subplot(gs[1, 2:4])
    recon = iradon(sino.T, theta=ang_deg, filter_name='ramp',
                   circle=True, output_size=sino.shape[1])
    recon = recon[::-1, :]
    recon_db = _to_db(recon, float(recon.max()))
    ax_c.imshow(recon_db,
                extent=[xy_mm[0], xy_mm[-1], xy_mm[0], xy_mm[-1]],
                origin='lower', cmap='gray',
                vmin=DB_FLOOR, vmax=0.0, aspect='equal')
    ax_c.set_xlabel('x (mm)')
    ax_c.set_ylabel('y (mm)')
    ax_c.set_title(f'(c)  Inverse-Radon reconstruction at $z_0$ = {z0:.1f} mm',
                   fontsize=12, fontweight='bold')

    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def volume_3d(volume, N, z_mm, xy_mm, out_path, stride=4):
    """3D scatter of voxels above DB_FLOOR; alpha scales with intensity so
    values at DB_FLOOR are fully transparent."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3d projection)

    ref = float(volume.max())
    db  = _to_db(volume, ref)

    # Stride-downsample for tractable scatter count
    db_ds = db[::stride, ::stride, ::stride]
    z_ds  = z_mm[::stride]
    xy_ds = xy_mm[::stride]

    mask = db_ds > DB_FLOOR
    iz, iy, ix = np.where(mask)
    vals = db_ds[mask]

    # Intensity in [0, 1] from DB_FLOOR..0; alpha tracks it so bright = opaque.
    normed = (vals - DB_FLOOR) / (-DB_FLOOR)
    normed = np.clip(normed, 0.0, 1.0)
    rgba   = plt.get_cmap('viridis')(normed)
    rgba[:, 3] = normed

    fig = plt.figure(figsize=(8, 7))
    ax  = fig.add_subplot(111, projection='3d')
    ax.scatter(xy_ds[ix], xy_ds[iy], z_ds[iz],
               c=rgba, s=10, marker='o', edgecolors='none', depthshade=False)

    ax.set(xlabel='x (mm)', ylabel='y (mm)', zlabel='z (mm)')
    ax.set_xlim(xy_mm[0], xy_mm[-1])
    ax.set_ylim(xy_mm[0], xy_mm[-1])
    ax.set_zlim(z_mm[-1], z_mm[0])       # z increases downward (ultrasound)
    ax.view_init(elev=20, azim=-60)

    step_deg = 180.0 / N
    ax.set_title(
        f'Reconstructed volume - N = {N} scans  (angle step = {step_deg:.2f} deg)',
        fontsize=11,
    )

    # Colorbar proxy for the amplitude scale
    import matplotlib as mpl
    sm = mpl.cm.ScalarMappable(
        norm=mpl.colors.Normalize(vmin=DB_FLOOR, vmax=0.0),
        cmap='viridis',
    )
    sm.set_array([])
    fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.1,
                 label='amplitude (dB re max)')

    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def psf_sweep(volumes_by_N, z_mm, xy_mm, out_path):
    """3×3 panel: rows = scatterers, cols = (x-line, y-line, z-line through peak).
    Each line is a reconstructed 1-D profile, one curve per N."""
    n_sc = len(SCATTERERS)
    fig, axes = plt.subplots(n_sc, 3, figsize=(13, 3.2 * n_sc),
                             sharex='col')
    Ns = sorted(volumes_by_N.keys())
    cmap = plt.get_cmap('viridis')
    colors = {N: cmap(i / max(len(Ns) - 1, 1)) for i, N in enumerate(Ns)}

    for j, (sx, sy, sz, _) in enumerate(SCATTERERS):
        truth_mm = (sx * 1e3, sy * 1e3, sz * 1e3)
        for N in Ns:
            vol = volumes_by_N[N]
            iz, iy, ix = find_peak(vol, z_mm, xy_mm, truth_mm)
            lx = vol[iz, iy, :]
            ly = vol[iz, :, ix]
            lz = vol[:, iy, ix]
            peak = max(float(lx.max()), float(ly.max()), float(lz.max()))
            lbl = f'N={N}  (angle step={180.0/N:.2f} deg)'
            axes[j, 0].plot(xy_mm, _to_db(lx, peak), color=colors[N], label=lbl)
            axes[j, 1].plot(xy_mm, _to_db(ly, peak), color=colors[N], label=lbl)
            axes[j, 2].plot(z_mm,  _to_db(lz, peak), color=colors[N], label=lbl)

        axes[j, 0].axvline(truth_mm[0], color='k', ls='--', lw=0.8)
        axes[j, 1].axvline(truth_mm[1], color='k', ls='--', lw=0.8)
        axes[j, 2].axvline(truth_mm[2], color='k', ls='--', lw=0.8)

        axes[j, 0].set_ylabel(
            f's{j} @ ({truth_mm[0]:+.0f},{truth_mm[1]:+.0f},{truth_mm[2]:.0f}) mm\n'
            'amplitude (dB re peak)'
        )
        for c in range(3):
            axes[j, c].grid(True, alpha=0.3)
            axes[j, c].set_ylim(DB_FLOOR, 2.0)

    axes[-1, 0].set_xlabel('x (mm)')
    axes[-1, 1].set_xlabel('y (mm)')
    axes[-1, 2].set_xlabel('z (mm)')
    axes[0, 0].set_title('line through peak along x')
    axes[0, 1].set_title('line through peak along y')
    axes[0, 2].set_title('line through peak along z')
    axes[0, 0].legend(fontsize=8, loc='upper right')

    # ROI-limited x-range so profiles aren't dwarfed by empty space
    for j, (sx, sy, sz, _) in enumerate(SCATTERERS):
        axes[j, 0].set_xlim(sx * 1e3 - 3 * ROI_MM, sx * 1e3 + 3 * ROI_MM)
        axes[j, 1].set_xlim(sy * 1e3 - 3 * ROI_MM, sy * 1e3 + 3 * ROI_MM)
        axes[j, 2].set_xlim(sz * 1e3 - 3 * ROI_MM, sz * 1e3 + 3 * ROI_MM)

    fig.suptitle('PSF line profiles through each reconstructed peak', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    files = sorted(OUTPUT_DIR.glob('volume_N*.npy'))
    if not files:
        print(f"No volume_N*.npy in {OUTPUT_DIR}")
        return
    print(f"Plotting volumes from {OUTPUT_DIR}  ({len(files)} files)")

    z_mm, xy_mm, half_mm = axes_mm()

    volumes_by_N = {}
    for f in files:
        N = int(re.search(r'volume_N(\d+)', f.stem).group(1))
        vol = np.load(f).astype(np.float32)
        volumes_by_N[N] = vol

        out = OUTPUT_DIR / f'mip_N{N:03d}.png'
        mip_triptych(vol, N, z_mm, xy_mm, half_mm, out)
        print(f"Wrote {out}")

        out3d = OUTPUT_DIR / f'vol3d_N{N:03d}.png'
        volume_3d(vol, N, z_mm, xy_mm, out3d)
        print(f"Wrote {out3d}")

    out = OUTPUT_DIR / 'psf_sweep.png'
    psf_sweep(volumes_by_N, z_mm, xy_mm, out)
    print(f"Wrote {out}")

    out = OUTPUT_DIR / 'radon_pipeline.png'
    radon_pipeline_figure(out)
    print(f"Wrote {out}")


if __name__ == '__main__':
    main()
