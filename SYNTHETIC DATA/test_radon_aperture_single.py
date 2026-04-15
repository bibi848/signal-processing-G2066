"""
Radon reconstruction of a single off-centre scatterer using the finite
elevation aperture.

Setup:
  - One high-impedance voxel at world (x=1 mm, y=2 mm, z=20 mm)
  - Array rotates θ ∈ [-π/2, +π/2] (64 frames)
  - elevation_aperture = 8 mm (scatterer at radius 2.24 mm is always in the slab)

The reconstructed 3D volume should show a bright blob at (x≈1, y≈2, z≈20).
"""

from __future__ import annotations

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from engine.config import (
    SimulationConfig, SpecimenConfig, ArrayConfig, ScanPlanConfig,
)
from engine.geometry import Specimen3D
from engine.materials import ALUMINUM
from engine.voxel_volume import VoxelVolume3D

from run_engine import scan_volume_3d
from test_radon_reconstruction import load_and_reconstruct
from Classes.Reconstruct3D import compute_reconstruction_coords, view_reconstruction_napari


# Configuration
SCAT_X = 1e-3
SCAT_Y = 2e-3
SCAT_Z = 20e-3
APERTURE = 5e-3       # elevation aperture (real probe element height)
N_SCANS = 64
THICKNESS = 40e-3
WIDTH = 40e-3
DEPTH = 30e-3
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'output', 'radon_tests', 'aperture_single_scatterer')


def build_single_scatterer_volume(voxel_size: float = 0.5e-3,
                                  contrast: float = 0.5) -> VoxelVolume3D:
    n_z = int(THICKNESS / voxel_size) + 1
    n_y = int(DEPTH     / voxel_size) + 1
    n_x = int(WIDTH     / voxel_size) + 1
    imp = np.full((n_z, n_y, n_x), ALUMINUM.Z_L, dtype=np.float32)

    origin_z = 0.0
    origin_y = -DEPTH / 2
    origin_x = -WIDTH / 2

    iz = int(round((SCAT_Z - origin_z) / voxel_size))
    iy = int(round((SCAT_Y - origin_y) / voxel_size))
    ix = int(round((SCAT_X - origin_x) / voxel_size))
    imp[iz, iy, ix] = ALUMINUM.Z_L * (1.0 + contrast)

    return VoxelVolume3D(
        impedance=imp,
        wavespeed=np.full_like(imp, ALUMINUM.c_L),
        voxel_size=voxel_size,
        origin_z=origin_z, origin_y=origin_y, origin_x=origin_x,
    )


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Scatterer (x, y, z) = ({SCAT_X*1e3:.1f}, {SCAT_Y*1e3:.1f}, "
          f"{SCAT_Z*1e3:.1f}) mm   |  aperture = {APERTURE*1e3:.1f} mm")

    # 1. Build the voxel volume
    vol = build_single_scatterer_volume()

    # 2. Configure scan with elevation aperture
    cfg = SimulationConfig(
        specimen=SpecimenConfig(thickness=THICKNESS, width=WIDTH),
        array=ArrayConfig(
            num_elements=128, element_pitch=0.3e-3, element_width=0.27e-3,
            element_height=5e-3, frequency=10e6, bandwidth=0.6,
            elevation_aperture=APERTURE, n_elevation_slices=17,
        ),
        scan_plan=ScanPlanConfig(n_scans=N_SCANS,
                                 theta_start=-np.pi / 2,
                                 theta_end=np.pi / 2),
        max_bounces=2,
        mode_conversion=False,
        wall_echoes=False,
    )
    print(cfg.summary())

    # 3. Simulate rotational scan
    specimen = Specimen3D(thickness=THICKNESS, width=WIDTH, depth=DEPTH)
    scan_volume_3d(
        specimen=specimen,
        defects_3d=[],
        cfg=cfg,
        scan_plan=cfg.scan_plan,
        output_dir=OUT_DIR,
        voxel_volume=vol,
        born_threshold=1e-5,
        tfm_z_start=5e-3,
        tfm_z_end=THICKNESS - 5e-3,
        tfm_n_pixels=400,
    )

    # 4. Radon reconstruct
    print("\nReconstructing via inverse Radon...")
    recon = load_and_reconstruct(OUT_DIR, N_SCANS, filter_name=None)
    print(f"Reconstructed volume shape: {recon.shape}")

    # 5. Map reconstruction grid to world coords to locate scatterer slice
    meta = np.load(os.path.join(OUT_DIR, 'scan_meta.npy'),
                   allow_pickle=True).item()
    z_coords, y_coords, x_coords = compute_reconstruction_coords(
        meta, recon.shape[1])

    # 5b. Maximum valid reconstruction radius
    # r_max = l / (2 sin(Δθ/2))  where l = aperture, Δθ = angular step.
    # Beyond r_max adjacent scan angles do not overlap and the voxel was
    # never illuminated by the array — must be masked out.
    l_ap = meta['array_aperture_m']
    dtheta = meta['angle_step_rad']
    r_max = l_ap / (2.0 * np.sin(abs(dtheta) / 2.0)) if dtheta != 0 else np.inf
    # The array chord also limits the absolute radius to l/2.
    r_max = min(r_max, l_ap / 2.0)
    print(f"\nAperture l = {l_ap*1e3:.2f} mm   Δθ = {np.degrees(dtheta):.2f}°")
    print(f"Valid reconstruction radius r_max = l / (2 sin(Δθ/2)) = "
          f"{r_max*1e3:.2f} mm  (capped at l/2)")

    # Mask voxels outside r_max
    YY, XX = np.meshgrid(y_coords, x_coords, indexing='ij')
    valid_xy = (XX**2 + YY**2) <= r_max**2
    recon = recon * valid_xy[np.newaxis, :, :]

    # 5c. Save and plot the sinogram at the scatterer depth
    bscans_list = []
    for i in range(N_SCANS):
        p = os.path.join(OUT_DIR, f'bscan_complex_{i:04d}.npy')
        bscans_list.append(np.load(p))
    bscans = np.stack(bscans_list, axis=0)            # (n_scans, n_z, n_lateral) complex
    # Rearrange (n_scans, n_z, n_lateral) → (n_z, n_lateral, n_scans) for per-z indexing
    sinos = np.transpose(bscans, (1, 2, 0))

    # Find depth row nearest to SCAT_Z in the B-scan grid
    n_z_b = bscans.shape[1]
    n_l_b = bscans.shape[2]
    z_img = np.linspace(meta['tfm_z_start_m'], meta['tfm_z_end_m'], n_z_b)
    half_w = meta['array_aperture_m'] / 2
    x_img = np.linspace(-half_w, half_w, n_l_b)
    iz_bscan = int(np.argmin(np.abs(z_img - SCAT_Z)))
    angles_deg = np.degrees(meta['angles_rad'])

    sino_at_z = sinos[iz_bscan]                       # (n_lateral, n_scans) complex
    sino_env = np.abs(sino_at_z)

    # Expected projection: L(θ) = x0 cos θ + y0 sin θ
    L_theory = SCAT_X * np.cos(meta['angles_rad']) + SCAT_Y * np.sin(meta['angles_rad'])

    fig_s, axes_s = plt.subplots(1, 3, figsize=(18, 5))
    im0 = axes_s[0].imshow(np.real(sino_at_z), aspect='auto', cmap='seismic',
                           extent=[angles_deg[0], angles_deg[-1],
                                   x_img[-1]*1e3, x_img[0]*1e3],
                           vmin=-np.abs(sino_at_z).max(), vmax=np.abs(sino_at_z).max())
    axes_s[0].plot(angles_deg, L_theory*1e3, 'g--', lw=1.2, label='L(θ)=x₀cosθ+y₀sinθ')
    axes_s[0].set_title(f'Sinogram (real)  z={z_img[iz_bscan]*1e3:.1f} mm')
    axes_s[0].set_xlabel('θ (deg)'); axes_s[0].set_ylabel('lateral L (mm)')
    axes_s[0].legend(loc='upper right', fontsize=8)
    plt.colorbar(im0, ax=axes_s[0])

    im1 = axes_s[1].imshow(sino_env, aspect='auto', cmap='inferno',
                           extent=[angles_deg[0], angles_deg[-1],
                                   x_img[-1]*1e3, x_img[0]*1e3])
    axes_s[1].plot(angles_deg, L_theory*1e3, 'c--', lw=1.2, label='expected L(θ)')
    axes_s[1].set_title('Sinogram envelope |p(L,θ)|')
    axes_s[1].set_xlabel('θ (deg)'); axes_s[1].set_ylabel('lateral L (mm)')
    axes_s[1].legend(loc='upper right', fontsize=8)
    plt.colorbar(im1, ax=axes_s[1])

    # Peak lateral position per angle
    peak_L = x_img[np.argmax(sino_env, axis=0)]
    axes_s[2].plot(angles_deg, L_theory*1e3, 'g-', lw=1.5, label='theory L(θ)')
    axes_s[2].plot(angles_deg, peak_L*1e3, 'r.', ms=4, label='measured peak')
    axes_s[2].set_xlabel('θ (deg)'); axes_s[2].set_ylabel('lateral L (mm)')
    axes_s[2].set_title('Peak lateral vs angle')
    axes_s[2].legend(); axes_s[2].grid(alpha=0.3)

    fig_s.suptitle(f'Sinogram at scatterer depth — scatterer (x,y)=({SCAT_X*1e3:.1f},{SCAT_Y*1e3:.1f}) mm',
                   fontsize=13)
    fig_s.tight_layout()
    sino_path = os.path.join(OUT_DIR, 'sinogram_at_scatterer.png')
    fig_s.savefig(sino_path, dpi=150)
    np.save(os.path.join(OUT_DIR, 'sinogram_at_scatterer.npy'), sino_at_z)
    print(f"Saved sinogram → {sino_path}")

    iz = int(np.argmin(np.abs(z_coords - SCAT_Z)))
    iy = int(np.argmin(np.abs(y_coords - SCAT_Y)))
    ix = int(np.argmin(np.abs(x_coords - SCAT_X)))
    print(f"\nNearest grid indices → z[{iz}]={z_coords[iz]*1e3:.2f} mm, "
          f"y[{iy}]={y_coords[iy]*1e3:.2f} mm, x[{ix}]={x_coords[ix]*1e3:.2f} mm")

    env = np.abs(recon)
    vmax = env.max() + 1e-12

    # 6. Plot three orthogonal slices through the expected scatterer location
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    im0 = axes[0].imshow(env[iz], cmap='inferno', origin='lower',
                         vmin=0, vmax=vmax,
                         extent=[x_coords[0]*1e3, x_coords[-1]*1e3,
                                 y_coords[0]*1e3, y_coords[-1]*1e3])
    axes[0].plot(SCAT_X*1e3, SCAT_Y*1e3, 'c+', markersize=16, markeredgewidth=2)
    circ_phi = np.linspace(0, 2*np.pi, 200)
    axes[0].plot(r_max*np.cos(circ_phi)*1e3, r_max*np.sin(circ_phi)*1e3,
                 'w--', linewidth=1, alpha=0.6, label=f'r_max = {r_max*1e3:.1f} mm')
    axes[0].legend(loc='upper right', fontsize=8)
    axes[0].set_title(f'xy-slice at z = {z_coords[iz]*1e3:.1f} mm')
    axes[0].set_xlabel('x (mm)')
    axes[0].set_ylabel('y (mm)')
    plt.colorbar(im0, ax=axes[0], label='amplitude')

    im1 = axes[1].imshow(env[:, iy, :], cmap='inferno', origin='upper',
                         vmin=0, vmax=vmax,
                         extent=[x_coords[0]*1e3, x_coords[-1]*1e3,
                                 z_coords[-1]*1e3, z_coords[0]*1e3], aspect='auto')
    axes[1].plot(SCAT_X*1e3, SCAT_Z*1e3, 'c+', markersize=16, markeredgewidth=2)
    axes[1].set_title(f'xz-slice at y = {y_coords[iy]*1e3:.1f} mm')
    axes[1].set_xlabel('x (mm)')
    axes[1].set_ylabel('z (mm)')
    plt.colorbar(im1, ax=axes[1], label='amplitude')

    im2 = axes[2].imshow(env[:, :, ix], cmap='inferno', origin='upper',
                         vmin=0, vmax=vmax,
                         extent=[y_coords[0]*1e3, y_coords[-1]*1e3,
                                 z_coords[-1]*1e3, z_coords[0]*1e3], aspect='auto')
    axes[2].plot(SCAT_Y*1e3, SCAT_Z*1e3, 'c+', markersize=16, markeredgewidth=2)
    axes[2].set_title(f'yz-slice at x = {x_coords[ix]*1e3:.1f} mm')
    axes[2].set_xlabel('y (mm)')
    axes[2].set_ylabel('z (mm)')
    plt.colorbar(im2, ax=axes[2], label='amplitude')

    fig.suptitle(
        f'Radon reconstruction of single scatterer at '
        f'({SCAT_X*1e3:.0f}, {SCAT_Y*1e3:.0f}, {SCAT_Z*1e3:.0f}) mm   |   '
        f'aperture = {APERTURE*1e3:.0f} mm,  N_scans = {N_SCANS}',
        fontsize=13,
    )
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, 'radon_aperture_single.png')
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved → {out_path}")

    # 7. Peak location check
    peak_idx = np.unravel_index(np.argmax(env), env.shape)
    peak_z = z_coords[peak_idx[0]]
    peak_y = y_coords[peak_idx[1]]
    peak_x = x_coords[peak_idx[2]]
    print(f"\nPeak of reconstruction at "
          f"(x={peak_x*1e3:+.2f}, y={peak_y*1e3:+.2f}, z={peak_z*1e3:+.2f}) mm "
          f"(expected ({SCAT_X*1e3:+.2f}, {SCAT_Y*1e3:+.2f}, {SCAT_Z*1e3:+.2f}))")

    # 8. STREAK INTENSITY ANALYSIS — linear peak-to-streak ratios
    print("\n" + "=" * 70)
    print("STREAK INTENSITY ANALYSIS (linear ratio vs peak)")
    print("=" * 70)

    i_mid = N_SCANS // 2
    bscan_mid = bscans[i_mid]
    env_b = np.abs(bscan_mid)
    peak_b = env_b.max()
    off_b = np.median(env_b)
    print(f"\n[1] Raw TFM B-scan  (θ=0)")
    print(f"    peak = {peak_b:.3e}    median off-peak = {off_b:.3e}    "
          f"ratio = {off_b/max(peak_b,1e-30):.3e}")

    env_s = np.abs(sino_at_z)
    peak_s = env_s.max()
    L_theory_pix = np.argmin(np.abs(x_img[:, None] - L_theory[None, :]), axis=0)
    on_track = np.zeros_like(env_s, dtype=bool)
    for j, ip in enumerate(L_theory_pix):
        on_track[max(0, ip - 2):ip + 3, j] = True
    off_track = env_s[~on_track]
    print(f"\n[2] Sinogram at z = {z_img[iz_bscan]*1e3:.1f} mm")
    print(f"    peak on sine track = {peak_s:.3e}")
    print(f"    median off-track   = {np.median(off_track):.3e}    "
          f"ratio = {np.median(off_track)/max(peak_s,1e-30):.3e}")
    print(f"    95th pct off-track = {np.percentile(off_track, 95):.3e}    "
          f"ratio = {np.percentile(off_track,95)/max(peak_s,1e-30):.3e}")

    slice_xy = env[iz]
    peak_r = slice_xy.max()
    cy, cx = peak_idx[1], peak_idx[2]
    r_line = np.arange(20, min(slice_xy.shape) // 2)
    ang_streak = np.deg2rad(135.0)
    yy = (cy + r_line * np.sin(ang_streak)).astype(int)
    xx = (cx + r_line * np.cos(ang_streak)).astype(int)
    ok = (yy >= 0) & (yy < slice_xy.shape[0]) & (xx >= 0) & (xx < slice_xy.shape[1])
    streak_profile = slice_xy[yy[ok], xx[ok]]
    streak_peak = streak_profile.max() if streak_profile.size else 0.0
    print(f"\n[3] Reconstruction slice at z = {z_coords[iz]*1e3:.1f} mm")
    print(f"    recon peak = {peak_r:.3e}")
    print(f"    max streak = {streak_peak:.3e}    ratio = {streak_peak/max(peak_r,1e-30):.3e}")
    print(f"    median streak = {np.median(streak_profile):.3e}    "
          f"ratio = {np.median(streak_profile)/max(peak_r,1e-30):.3e}")

    # Diagnostic figure: linear amplitude at each stage
    fig_d, axd = plt.subplots(1, 3, figsize=(18, 5))
    imA = axd[0].imshow(env_b, aspect='auto', cmap='inferno',
                        vmin=0, vmax=peak_b,
                        extent=[x_img[0]*1e3, x_img[-1]*1e3,
                                z_img[-1]*1e3, z_img[0]*1e3])
    axd[0].set_title('[1] Raw TFM B-scan (θ=0), linear')
    axd[0].set_xlabel('lateral L (mm)'); axd[0].set_ylabel('z (mm)')
    plt.colorbar(imA, ax=axd[0], label='amplitude')

    imB = axd[1].imshow(env_s, aspect='auto', cmap='inferno',
                        vmin=0, vmax=peak_s,
                        extent=[angles_deg[0], angles_deg[-1],
                                x_img[-1]*1e3, x_img[0]*1e3])
    axd[1].plot(angles_deg, L_theory*1e3, 'c--', lw=1, alpha=0.6)
    axd[1].set_title('[2] Sinogram at scatterer depth, linear')
    axd[1].set_xlabel('θ (deg)'); axd[1].set_ylabel('lateral L (mm)')
    plt.colorbar(imB, ax=axd[1], label='amplitude')

    imC = axd[2].imshow(slice_xy, cmap='inferno', origin='lower',
                        vmin=0, vmax=peak_r,
                        extent=[x_coords[0]*1e3, x_coords[-1]*1e3,
                                y_coords[0]*1e3, y_coords[-1]*1e3])
    axd[2].plot(SCAT_X*1e3, SCAT_Y*1e3, 'c+', ms=14, mew=2)
    axd[2].set_title('[3] Reconstruction xy-slice, linear')
    axd[2].set_xlabel('x (mm)'); axd[2].set_ylabel('y (mm)')
    plt.colorbar(imC, ax=axd[2], label='amplitude')

    fig_d.suptitle('Streak propagation through the pipeline (linear)', fontsize=13)
    fig_d.tight_layout()
    diag_path = os.path.join(OUT_DIR, 'streak_intensity_stages.png')
    fig_d.savefig(diag_path, dpi=150)
    print(f"\nSaved diagnostic figure → {diag_path}")

    # 9. Interactive 3D view — linear amplitude
    try:
        import napari
        dz = (z_coords[-1] - z_coords[0]) / max(len(z_coords) - 1, 1) * 1e3
        dy = (y_coords[-1] - y_coords[0]) / max(len(y_coords) - 1, 1) * 1e3
        dx = (x_coords[-1] - x_coords[0]) / max(len(x_coords) - 1, 1) * 1e3
        viewer = napari.Viewer(title='Radon reconstruction (linear)')
        viewer.add_image(
            env, name='Reconstruction',
            scale=(dz, dy, dx), colormap='inferno',
            contrast_limits=(0.0, float(vmax)),
        )
        napari.run()
    except ImportError:
        pass


if __name__ == '__main__':
    main()
