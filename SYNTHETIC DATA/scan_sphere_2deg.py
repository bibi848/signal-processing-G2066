#!/usr/bin/env python3
"""
Rotational scan of a volume containing a single OFF-CENTRE spherical defect.

Generates TFM B-scan slices every 2 degrees over [-90°, +90°] using the same
FMC + filter + TFM pipeline as run_engine.py.

Outputs per frame into output/scans/scan_sphere_2deg/:
    fmc_<i>.npy           filtered FMC
    bscan_<i>.npy         TFM dB image
    bscan_complex_<i>.npy complex TFM image (when img_output='complex')
    scan_meta.npy         scan metadata
    bscan_grid.png        grid preview
    bscan_anim.gif        animated sweep
    volume_preview.png    3D specimen preview
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from engine.config import (
    SimulationConfig, SpecimenConfig, ArrayConfig, ScanPlanConfig,
    AcquisitionConfig,
)
from engine.geometry import Specimen3D, SphericalDefect, CylindricalDefect
from engine.materials import ALUMINUM

from run_engine import (
    preview_volume_3d,
    scan_volume_3d,
    visualize_scans,
)


def main():
    # ---- 3D specimen: 50 mm cube matches reconstruction_experiment_radon.py ----
    specimen = Specimen3D(
        thickness=50e-3,   # z: depth
        width=50e-3,       # x: along array
        depth=50e-3,       # y: elevation
    )

    # ---- Defects: two off-centre spherical voids ----
    defects_3d = [
        SphericalDefect(center_x=-10e-3, center_y=-5e-3, center_z=20e-3, radius=2e-3),
        SphericalDefect(center_x=+12e-3, center_y=+7e-3, center_z=32e-3, radius=2e-3),
    ]

    frequency = 10e6

    # ---- Scan plan: every 2° over [-90°, +90°] ----
    step_deg = 2.0
    theta_start = -np.pi / 2
    theta_end = np.pi / 2
    n_scans = int(round((theta_end - theta_start) / np.radians(step_deg))) + 1
    scan_plan = ScanPlanConfig(
        n_scans=n_scans,
        theta_start=theta_start,
        theta_end=theta_end,
    )
    
    print(f"Scan plan: {n_scans} slices, step = {step_deg}°")

    # ---- Simulation config ----
    # Every FMC signal component is set explicitly so it's clear what is
    # and isn't being simulated in each A-scan.
    cfg = SimulationConfig(
        specimen=SpecimenConfig(thickness=specimen.thickness, width=specimen.width),
        array=ArrayConfig(
            num_elements=128,
            element_pitch=0.3e-3,
            frequency=frequency,   # 10 MHz centre frequency
            bandwidth=0.8,         # 80 % fractional bandwidth (Gabor pulse width)
            elevation_aperture=10e-3,   # physical element height in y (m)
            n_elevation_slices=5,       # how many slabs to integrate across it
        ),

        scan_plan=scan_plan,

        # --- Physics contributions to the FMC signal -------------------
        wall_echoes=False,      # front- and back-wall echoes (+ reverberations)
        max_bounces=1,          # 1 = direct defect only
                                # 2 = adds skip (TX→BW→defect→RX) and
                                #     corner-trap (TX→defect→BW→RX)
        mode_conversion=False,  # L→S conversions at back wall and defect

        # --- Acquisition noise + bandpass filter -----------------------
        acquisition=AcquisitionConfig(
            snr_db=200.0,             # electronic Gaussian noise level (high = clean)
            grain_noise_level=0.0,    # structured band-limited "grain" jitter
            filter_alpha=1.0,         # Tukey taper for bandpass (1 = Hann)
            hanning_bool=False,       # pre-window A-scan with Hanning before FFT
        ),

        gel_thickness=0.075e-3,  # couplant layer (front-wall echo delay)
    )
    print(cfg.summary())

    output_dir = os.path.join(
        os.path.dirname(__file__), 'output', 'scans', 'scan_sphere_2deg'
    )
    os.makedirs(output_dir, exist_ok=True)

    # Preview the 3D volume
    preview_volume_3d(specimen, defects_3d, scan_plan,
                      os.path.join(output_dir, 'volume_preview.png'))

    # Pure Kirchhoff: geometric defect only, no grain noise
    scan_volume_3d(
        specimen, defects_3d, cfg, scan_plan, output_dir,
        voxel_volume=None,
    )

    # ---- Save each B-scan as a PNG and drop the .npy files ----
    meta = np.load(os.path.join(output_dir, 'scan_meta.npy'),
                   allow_pickle=True).item()
    angles_rad = meta.get('angles_rad', scan_plan.angles)

    for i, theta in enumerate(angles_rad):
        tag = f"{i:04d}"
        bscan_path = os.path.join(output_dir, f'bscan_{tag}.npy')
        if not os.path.exists(bscan_path):
            continue
        img = np.load(bscan_path)
        # If this is a real-valued TFM output, envelope+dB for display.
        if np.iscomplexobj(img):
            env = np.abs(img)
            img_db = 20 * np.log10(env / (env.max() + 1e-10) + 1e-10)
        elif img.ndim == 2 and img.min() >= -200 and img.max() <= 5:
            img_db = img  # already dB
        else:
            env = np.abs(img)
            img_db = 20 * np.log10(env / (env.max() + 1e-10) + 1e-10)

        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(img_db, cmap='hot', vmin=-20, vmax=0,
                       aspect='auto', origin='upper')
        ax.set_title(f'θ = {np.degrees(theta):+.1f}°')
        ax.set_xlabel('X pixel')
        ax.set_ylabel('Z pixel')
        plt.colorbar(im, ax=ax, label='dB', shrink=0.8)
        plt.tight_layout()
        png_path = os.path.join(output_dir, f'bscan_{tag}.png')
        plt.savefig(png_path, dpi=150, bbox_inches='tight')
        plt.close()

        # Remove the .npy frames, keep only PNGs + metadata
        os.remove(bscan_path)
        fmc_path = os.path.join(output_dir, f'fmc_{tag}.npy')
        if os.path.exists(fmc_path):
            os.remove(fmc_path)
        cplx_path = os.path.join(output_dir, f'bscan_complex_{tag}.npy')
        if os.path.exists(cplx_path):
            os.remove(cplx_path)

    print(f"  Saved {len(angles_rad)} PNG B-scans to {output_dir}/")

    # ---- Build a grid figure from the already-saved PNGs (no recompute) ----
    png_paths = sorted([
        os.path.join(output_dir, f)
        for f in os.listdir(output_dir)
        if f.startswith('bscan_') and f.endswith('.png')
    ])
    n = len(png_paths)
    if n > 0:
        cols = min(8, n)
        rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols,
                                 figsize=(cols * 2.2, rows * 2.5),
                                 squeeze=False)
        for ax in axes.flat:
            ax.axis('off')
        for idx, (p, theta) in enumerate(zip(png_paths, angles_rad)):
            r, c = divmod(idx, cols)
            ax = axes[r][c]
            ax.imshow(plt.imread(p))
            ax.set_title(f'{np.degrees(theta):+.1f}°', fontsize=7)
            ax.axis('off')
        fig.suptitle('TFM B-scans — 3D rotational scan', fontsize=10)
        plt.tight_layout()
        grid_path = os.path.join(output_dir, 'bscan_grid.png')
        plt.savefig(grid_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved grid: {grid_path}")

    print(f"\nDone — {n_scans} B-scans in {output_dir}/")


if __name__ == '__main__':
    main()
