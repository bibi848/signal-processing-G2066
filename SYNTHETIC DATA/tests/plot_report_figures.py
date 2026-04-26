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


def _draw_geometry_schematic(ax):
    """3D schematic for panel (a): linear array on top of specimen, rotation
    about the vertical z axis, and the three SCATTERERS drawn inside."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    L, D = 18.0, 35.0  # mm half-width and depth of specimen

    pts = np.array([
        [-L, -L, 0], [+L, -L, 0], [+L, +L, 0], [-L, +L, 0],
        [-L, -L, D], [+L, -L, D], [+L, +L, D], [-L, +L, D],
    ])
    edges = [(0, 1), (1, 2), (2, 3), (3, 0),
             (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]
    for a, b in edges:
        xs, ys, zs = zip(pts[a], pts[b])
        ax.plot(xs, ys, zs, color='gray', lw=0.6, alpha=0.6)
    ax.add_collection3d(Poly3DCollection(
        [[pts[0], pts[1], pts[2], pts[3]]],
        facecolor='lightblue', alpha=0.12, edgecolor='gray', lw=0.5))

    array_len, array_w, z_arr = 28.0, 2.5, -0.4

    def array_quad(theta):
        ct, st = np.cos(theta), np.sin(theta)
        local = np.array([[-array_len / 2, -array_w / 2],
                          [+array_len / 2, -array_w / 2],
                          [+array_len / 2, +array_w / 2],
                          [-array_len / 2, +array_w / 2]])
        rot = local @ np.array([[ct, -st], [st, ct]]).T
        return [(rot[i, 0], rot[i, 1], z_arr) for i in range(4)]

    ax.add_collection3d(Poly3DCollection(
        [array_quad(0.0)], facecolor='crimson',
        alpha=0.95, edgecolor='darkred', lw=1.5))
    ax.add_collection3d(Poly3DCollection(
        [array_quad(np.radians(60))], facecolor='crimson',
        alpha=0.30, edgecolor='darkred', lw=1.0))

    arc_r = 11.0
    arc_t = np.linspace(np.radians(8), np.radians(65), 30)
    arc_x = arc_r * np.cos(arc_t)
    arc_y = arc_r * np.sin(arc_t)
    arc_z = np.full_like(arc_t, -3.0)
    ax.plot(arc_x, arc_y, arc_z, color='black', lw=2.0)
    ax.quiver(arc_x[-2], arc_y[-2], arc_z[-2],
              arc_x[-1] - arc_x[-2], arc_y[-1] - arc_y[-2], 0.0,
              color='black', length=2.5, arrow_length_ratio=0.8, normalize=True)
    ax.text(arc_r * np.cos(np.radians(36)) + 1.5,
            arc_r * np.sin(np.radians(36)) + 0.5,
            -3.0, r'$\theta$', fontsize=24, fontweight='bold')

    ax.plot([0, 0], [0, 0], [-5, D], color='black', ls=':', lw=1.0)

    sc_pts = [
        (0.0, 0.0, 20.0, 'red',     r'$s_0$'),
        (5.0, 3.0, 20.0, 'cyan',    r'$s_1$'),
        (0.0, 8.0, 25.0, 'orange',  r'$s_2$'),
    ]
    for sx, sy, sz, color, lbl in sc_pts:
        ax.scatter([sx], [sy], [sz], c=color, s=160, edgecolor='black',
                   lw=1.0, depthshade=False)
        ax.text(sx + 1.5, sy + 1.0, sz, lbl, fontsize=20, fontweight='bold')

    ax.set_xlim(-L, L)
    ax.set_ylim(-L, L)
    ax.set_zlim(-7, D)
    ax.invert_zaxis()
    ax.set_xlabel('x (mm)', fontsize=18, labelpad=6)
    ax.set_ylabel('y (mm)', fontsize=18, labelpad=6)
    ax.set_zlabel('z (mm)', fontsize=18, labelpad=6)
    ax.tick_params(labelsize=15)
    ax.view_init(elev=18, azim=-58)
    try:
        ax.set_box_aspect((1.0, 1.0, 1.0))
    except Exception:
        pass


def radon_pipeline_figure(volumes_by_N, out_path, z0_target_mm=20.0):
    """Five-panel figure for the report (letters only, no subplot titles):
         (a) 3D scan-geometry schematic
         (b) cropped TFM slices at θ = 0°, 60°, 120°
         (c) sinogram at depth z0
         (d) inverse-Radon reconstruction at z0
         (e) MIPs of the stacked-per-depth reconstructed volume
    Reads the cached complex B-scans written by test_radon_validation.py
    and the per-N reconstructed volumes built in main()."""
    bscans_path = OUTPUT_DIR / 'bscans_complex.npy'
    angles_path = OUTPUT_DIR / 'angles_rad.npy'
    if not (bscans_path.exists() and angles_path.exists()):
        print(f"Skipping pipeline figure; missing cache in {OUTPUT_DIR}")
        return
    if not volumes_by_N:
        print("Skipping pipeline figure; no reconstructed volumes provided")
        return

    bscans = np.abs(np.load(bscans_path)).astype(np.float32)   # (n_ang, n_z, n_x)
    angles = np.load(angles_path)
    z_mm, xy_mm, half_mm = axes_mm()

    iz0 = int(np.argmin(np.abs(z_mm - z0_target_mm)))
    z0  = float(z_mm[iz0])

    at_z0 = [(sx * 1e3, sy * 1e3, sz * 1e3) for sx, sy, sz, _ in SCATTERERS
             if abs(sz * 1e3 - z0) < 2.0]

    target_deg = [0.0, 60.0, 120.0]
    sel_idx = [int(np.argmin(np.abs(np.degrees(angles) - t))) for t in target_deg]

    N_best = max(volumes_by_N.keys())
    volume = volumes_by_N[N_best]

    cxy = 15.0                # ±mm in-plane crop for (b), (d), (e)
    cz_lo, cz_hi = 14.0, 30.0 # z crop (mm) for (b), (e)

    AX, TICK, LET, LEG = 20, 17, 28, 16

    fig = plt.figure(figsize=(22, 14))
    gs = fig.add_gridspec(2, 15, height_ratios=[1.0, 1.0],
                          hspace=0.28, wspace=0.55,
                          left=0.05, right=0.92, top=0.96, bottom=0.06)

    # ---- (a) 3D geometry schematic ----
    ax_a = fig.add_subplot(gs[0, 0:6], projection='3d')
    _draw_geometry_schematic(ax_a)
    ax_a.text2D(-0.05, 1.02, '(a)', transform=ax_a.transAxes,
                fontsize=LET, fontweight='bold', va='top')

    # ---- (b) Cropped TFM slices at θ = 0°, 60°, 120° ----
    sino = bscans[:, iz0, :]                                   # (n_ang, n_x)
    for k, i in enumerate(sel_idx):
        ax = fig.add_subplot(gs[0, 6 + 3 * k:6 + 3 * (k + 1)])
        img_db = _to_db(bscans[i], float(bscans[i].max()))
        ax.imshow(img_db, extent=[xy_mm[0], xy_mm[-1], z_mm[-1], z_mm[0]],
                  origin='upper', cmap='gray', vmin=DB_FLOOR, vmax=0.0,
                  aspect='auto')
        ax.axhline(z0, color='yellow', ls='--', lw=1.0, alpha=0.95)
        ax.set_xlim(-cxy, cxy)
        ax.set_ylim(cz_hi, cz_lo)
        ax.set_xlabel(r'$x_\theta$ (mm)', fontsize=AX)
        ax.tick_params(labelsize=TICK)
        if k == 0:
            ax.set_ylabel('z (mm)', fontsize=AX)
            ax.text(-0.30, 1.0, '(b)', transform=ax.transAxes,
                    fontsize=LET, fontweight='bold', va='top')
        ax.text(0.5, 0.97,
                fr'$\theta = {np.degrees(angles[i]):.0f}\degree$',
                transform=ax.transAxes, color='white', ha='center', va='top',
                fontsize=AX, fontweight='bold',
                bbox=dict(facecolor='black', alpha=0.5,
                          edgecolor='none', pad=2))
        for sx, sy, _sz in at_z0:
            xth = sx * np.cos(angles[i]) + sy * np.sin(angles[i])
            on_axis = (sx == 0.0 and sy == 0.0)
            color = 'red' if on_axis else 'cyan'
            ax.plot(xth, z0, marker='o', mfc='none', mec=color,
                    mew=1.8, ms=14, alpha=0.95)
            if k == 0:
                lbl = r'$s_0$' if on_axis else r'$s_1$'
                ax.annotate(lbl, xy=(xth, z0),
                            xytext=(xth + 4, z0 - 3),
                            color=color, fontsize=AX, fontweight='bold',
                            arrowprops=dict(arrowstyle='->', color=color, lw=1.0))

    # ---- (c) Sinogram at z0 ----
    ax_c = fig.add_subplot(gs[1, 0:7])
    sino_db = _to_db(sino, float(sino.max()))
    ang_deg = np.degrees(angles)
    ax_c.imshow(sino_db.T,
                extent=[ang_deg[0], ang_deg[-1], xy_mm[0], xy_mm[-1]],
                origin='lower', cmap='gray', vmin=DB_FLOOR, vmax=0.0,
                aspect='auto')
    theta_smooth = np.linspace(0.0, np.pi, 400)
    trace_colors = ['red', 'cyan', 'orange']
    for c, (sx, sy, _sz) in enumerate(at_z0):
        rho = sx * np.cos(theta_smooth) + sy * np.sin(theta_smooth)
        ax_c.plot(np.degrees(theta_smooth), rho,
                  color=trace_colors[c % len(trace_colors)], lw=1.8,
                  label=fr'$(x, y) = ({sx:+.0f}, {sy:+.0f})$ mm')
    ax_c.set_xlabel(r'$\theta$ (deg)', fontsize=AX)
    ax_c.set_ylabel(r'$x_\theta$ (mm)', fontsize=AX)
    ax_c.tick_params(labelsize=TICK)
    ax_c.legend(loc='upper right', fontsize=LEG, framealpha=0.85)
    ax_c.text(-0.07, 1.02, '(c)', transform=ax_c.transAxes,
              fontsize=LET, fontweight='bold', va='top')

    # ---- (d) Reconstructed-volume 3D scatter (matches (a) framing) ----
    import matplotlib as mpl
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    ref = float(volume.max())
    db_vol = _to_db(volume, ref)

    db_floor_e = -12.0  # tighter floor → cleaner blobs in 3D
    mask = db_vol > db_floor_e
    iz, iy, ix = np.where(mask)
    vals = db_vol[mask]
    xs = xy_mm[ix]; ys = xy_mm[iy]; zs = z_mm[iz]

    order = np.argsort(vals)  # plot brightest last so they stay on top
    xs, ys, zs, vals = xs[order], ys[order], zs[order], vals[order]

    normed = np.clip((vals - db_floor_e) / (-db_floor_e), 0.0, 1.0)
    rgba = plt.get_cmap('viridis')(normed)
    rgba[:, 3] = 0.35 + 0.65 * normed

    ax_d = fig.add_subplot(gs[1, 7:14], projection='3d')

    L_d, D_d = 18.0, 35.0
    box_pts = np.array([
        [-L_d, -L_d, 0], [+L_d, -L_d, 0], [+L_d, +L_d, 0], [-L_d, +L_d, 0],
        [-L_d, -L_d, D_d], [+L_d, -L_d, D_d], [+L_d, +L_d, D_d], [-L_d, +L_d, D_d],
    ])
    box_edges = [(0, 1), (1, 2), (2, 3), (3, 0),
                 (4, 5), (5, 6), (6, 7), (7, 4),
                 (0, 4), (1, 5), (2, 6), (3, 7)]
    for a, b in box_edges:
        ex, ey, ez = zip(box_pts[a], box_pts[b])
        ax_d.plot(ex, ey, ez, color='gray', lw=0.6, alpha=0.6)
    ax_d.add_collection3d(Poly3DCollection(
        [[box_pts[0], box_pts[1], box_pts[2], box_pts[3]]],
        facecolor='lightblue', alpha=0.12, edgecolor='gray', lw=0.5))

    ax_d.scatter(xs, ys, zs, c=rgba, s=40 + 60 * normed, marker='o',
                 edgecolors='none', depthshade=False)

    ax_d.set_xlim(-L_d, L_d)
    ax_d.set_ylim(-L_d, L_d)
    ax_d.set_zlim(-7, D_d)
    ax_d.invert_zaxis()
    ax_d.set_xlabel('x (mm)', fontsize=AX, labelpad=6)
    ax_d.set_ylabel('y (mm)', fontsize=AX, labelpad=6)
    ax_d.set_zlabel('z (mm)', fontsize=AX, labelpad=6)
    ax_d.tick_params(labelsize=TICK)
    ax_d.view_init(elev=18, azim=-58)
    try:
        ax_d.set_box_aspect((1.0, 1.0, 1.0))
    except Exception:
        pass
    ax_d.text2D(-0.02, 1.05, '(d)', transform=ax_d.transAxes,
                fontsize=LET, fontweight='bold', va='top')

    ax_cb = fig.add_subplot(gs[1, 14:15])
    sm = mpl.cm.ScalarMappable(
        norm=mpl.colors.Normalize(vmin=db_floor_e, vmax=0.0), cmap='viridis')
    sm.set_array([])
    cb = fig.colorbar(sm, cax=ax_cb)
    cb.set_label('amplitude (dB re max)', fontsize=AX)
    cb.ax.tick_params(labelsize=TICK)

    fig.savefig(out_path, dpi=150)
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
    radon_pipeline_figure(volumes_by_N, out)
    print(f"Wrote {out}")


if __name__ == '__main__':
    main()
