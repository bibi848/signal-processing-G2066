"""
Grain Size Comparison — Single B-scan
======================================

Runs a single B-scan simulation for several grain sizes on the same
specimen geometry.  Uses ``scan_volume_3d`` from run_engine.py directly
so the FMC → noise → filter → TFM pipeline is **identical** to the
main engine.

Usage:
    python compare_grain_sizes.py
    python compare_grain_sizes.py --grain-sizes 0.2e-3 0.5e-3 1.0e-3 2.0e-3
    python compare_grain_sizes.py --frequency 15e6 --no-defect
"""

import os
import sys
import shutil
import argparse
import tempfile
import time as timer
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from engine.config import SimulationConfig, SpecimenConfig, ArrayConfig, ScanPlanConfig
from engine.geometry import Specimen3D, CylindricalDefect
from engine.materials import ALUMINUM, STEEL_MILD, STEEL_STAINLESS
from engine.microstructure import generate_grain_structure, embed_geometric_defects

from run_engine import scan_volume_3d

MATERIAL_PRESETS = {
    'ALUMINUM': ALUMINUM,
    'STEEL_MILD': STEEL_MILD,
    'STEEL_STAINLESS': STEEL_STAINLESS,
}


def simulate_single_bscan(
    grain_size: float,
    material=ALUMINUM,
    thickness: float = 50e-3,
    width: float = 50e-3,
    depth: float = 30e-3,
    frequency: float = 10e6,
    num_elements: int = 64,
    element_pitch: float = 0.6e-3,
    bandwidth: float = 0.6,
    impedance_variation: float = 0.025,
    wavespeed_variation: float = 0.005,
    defect_type: str = 'cylinder',
    defect_z: float = 20e-3,
    defect_radius: float = 1e-3,
    snr_db: float = 35.0,
    filter_alpha: float = 1.0,
    hanning_bool: bool = False,
    max_bounces: int = 2,
    tfm_z_start: float = 10e-3,
    tfm_z_end: float = None,
    tfm_n_pixels: int = 800,
    born_threshold: float = 0.005,
    seed: int = 42,
):
    """
    Build a grain volume and simulate one B-scan at theta=0
    using the exact same pipeline as ``scan_volume_3d``.

    Returns:
        (img_db, x_img, z_img, grain_contrast)
    """
    specimen = Specimen3D(thickness=thickness, width=width, depth=depth)

    # Defect
    if defect_type == 'cylinder':
        defects_3d = [
            CylindricalDefect(
                center_z=defect_z, center_x=0.0, radius=defect_radius,
                y_start=-depth / 2, y_end=depth / 2,
            ),
        ]
    else:
        defects_3d = []

    # Grain volume
    wavelength = material.c_L / frequency
    voxel_size = wavelength / 3

    grain_vol = generate_grain_structure(
        thickness=thickness, width=width, depth=depth,
        background_material=material,
        mean_grain_size_m=grain_size,
        impedance_variation=impedance_variation,
        wavespeed_variation=wavespeed_variation,
        voxel_size_m=voxel_size,
        seed=seed,
    )
    voxel_volume = embed_geometric_defects(grain_vol, defects_3d)

    # Config — identical to what sweep_datasets / run_engine use
    cfg = SimulationConfig(
        material=material,
        specimen=SpecimenConfig(thickness=thickness, width=width),
        array=ArrayConfig(
            num_elements=num_elements,
            element_pitch=element_pitch,
            frequency=frequency,
            bandwidth=bandwidth,
        ),
        max_bounces=max_bounces,
    )
    cfg.acquisition.snr_db = snr_db
    cfg.acquisition.filter_alpha = filter_alpha
    cfg.acquisition.hanning_bool = hanning_bool

    # Single-angle scan plan at theta=0
    scan_plan = ScanPlanConfig(n_scans=1, theta_start=0.0, theta_end=0.0)

    # Default z_end: match scan_volume_3d default (thickness − 5 mm)
    if tfm_z_end is None:
        tfm_z_end = thickness - 5e-3

    # Run the canonical pipeline into a temp directory
    tmp_dir = tempfile.mkdtemp(prefix='grain_cmp_')
    try:
        scan_volume_3d(
            specimen=specimen,
            defects_3d=defects_3d,
            cfg=cfg,
            scan_plan=scan_plan,
            output_dir=tmp_dir,
            voxel_volume=voxel_volume,
            born_threshold=born_threshold,
            tfm_z_start=tfm_z_start,
            tfm_z_end=tfm_z_end,
            tfm_n_pixels=tfm_n_pixels,
        )
        img_db = np.load(os.path.join(tmp_dir, 'bscan_0000.npy'))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # Reconstruct pixel grids for plotting (same logic as reconstruct_tfm)
    half_w = cfg.array.aperture / 2
    x_img = np.linspace(-half_w, half_w, tfm_n_pixels)
    z_img = np.linspace(tfm_z_start, tfm_z_end, tfm_n_pixels)

    # Grain contrast slice for diagnostic visualisation
    z_vis = np.linspace(tfm_z_start, tfm_z_end, tfm_n_pixels)
    l_vis = np.linspace(-half_w, half_w, tfm_n_pixels)
    imp_slice = voxel_volume.slice_at_angle(0.0, z_vis, l_vis)
    Z0 = float(np.median(voxel_volume.impedance))
    grain_contrast = (imp_slice - Z0) / Z0

    return img_db, x_img, z_img, grain_contrast


def main():
    parser = argparse.ArgumentParser(
        description='Compare B-scans for different grain sizes',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--grain-sizes', type=float, nargs='+',
                        default=[0.3e-3, 0.5e-3, 1.0e-3, 2.0e-3],
                        help='Grain sizes to compare (m)')
    parser.add_argument('--material', type=str, default='ALUMINUM',
                        choices=list(MATERIAL_PRESETS.keys()))
    parser.add_argument('--frequency', type=float, default=10e6,
                        help='Centre frequency (Hz)')
    parser.add_argument('--num-elements', type=int, default=64)
    parser.add_argument('--element-pitch', type=float, default=0.6e-3)
    parser.add_argument('--bandwidth', type=float, default=0.6,
                        help='Fractional bandwidth (0.6 = 60%%)')
    parser.add_argument('--impedance-variation', type=float, default=0.025)
    parser.add_argument('--snr-db', type=float, default=35.0,
                        help='Signal-to-noise ratio (dB)')
    parser.add_argument('--filter-alpha', type=float, default=1.0,
                        help='Tukey filter alpha (0=rect, 1=Hann)')
    parser.add_argument('--hanning', action='store_true',
                        help='Enable Hanning pre-window')
    parser.add_argument('--with-defect', action='store_true',
                        help='Include a cylindrical defect (default: grain-only)')
    parser.add_argument('--tfm-n-pixels', type=int, default=800)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default=None,
                        help='Output PNG path')

    args = parser.parse_args()
    material = MATERIAL_PRESETS[args.material]
    grain_sizes = args.grain_sizes
    n = len(grain_sizes)
    defect_type = 'cylinder' if args.with_defect else 'none'

    print(f"\nComparing {n} grain sizes: "
          f"{[f'{g*1e3:.1f} mm' for g in grain_sizes]}")
    print(f"Material: {args.material}, Frequency: {args.frequency/1e6:.0f} MHz")
    print(f"Defect: {defect_type}\n")

    results = []
    for i, gs in enumerate(grain_sizes):
        print(f"[{i+1}/{n}] grain = {gs*1e3:.1f} mm ...")
        t0 = timer.time()
        img_db, x_img, z_img, grain_contrast = simulate_single_bscan(
            grain_size=gs,
            material=material,
            frequency=args.frequency,
            num_elements=args.num_elements,
            element_pitch=args.element_pitch,
            bandwidth=args.bandwidth,
            impedance_variation=args.impedance_variation,
            snr_db=args.snr_db,
            filter_alpha=args.filter_alpha,
            hanning_bool=args.hanning,
            defect_type=defect_type,
            tfm_n_pixels=args.tfm_n_pixels,
            seed=args.seed,
        )
        elapsed = timer.time() - t0
        print(f"  Done in {elapsed:.1f}s")
        results.append((gs, img_db, x_img, z_img, grain_contrast))

    # Plot: 2 rows x n columns — top: grain structure, bottom: TFM B-scan
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 10))
    if n == 1:
        axes = axes.reshape(2, 1)

    for col, (gs, img_db, x_img, z_img, grain_contrast) in enumerate(results):
        ext = [x_img[0]*1e3, x_img[-1]*1e3, z_img[-1]*1e3, z_img[0]*1e3]

        # Top: grain structure
        ax_g = axes[0, col]
        grain_lim = 0.06
        im_g = ax_g.imshow(
            grain_contrast, extent=ext, aspect='auto',
            cmap='RdBu_r', vmin=-grain_lim, vmax=grain_lim,
        )
        ax_g.set_title(f'Grain {gs*1e3:.1f} mm')
        ax_g.set_xlabel('x (mm)')
        if col == 0:
            ax_g.set_ylabel('Depth (mm)')
        plt.colorbar(im_g, ax=ax_g, label='ΔZ/Z₀', shrink=0.8)

        # Bottom: TFM B-scan
        ax_b = axes[1, col]
        vmin_db = np.percentile(img_db, 5)
        vmax_db = np.percentile(img_db, 99)
        im_b = ax_b.imshow(
            img_db, extent=ext, aspect='auto',
            cmap='hot', vmin=vmin_db, vmax=vmax_db,
        )
        ax_b.set_title(f'TFM B-scan')
        ax_b.set_xlabel('x (mm)')
        if col == 0:
            ax_b.set_ylabel('Depth (mm)')
        plt.colorbar(im_b, ax=ax_b, label='dB', shrink=0.8)

    fig.suptitle(
        f'Grain Size Comparison — {args.material} {args.frequency/1e6:.0f} MHz '
        f'({defect_type})',
        fontweight='bold', fontsize=14,
    )
    fig.tight_layout()

    if args.output:
        out_path = args.output
    else:
        out_dir = os.path.join(os.path.dirname(__file__), 'output', 'plots')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, 'grain_size_comparison.png')

    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nFigure saved: {out_path}")


if __name__ == '__main__':
    main()
