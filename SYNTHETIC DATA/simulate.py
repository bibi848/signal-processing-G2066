"""
Synthetic NDT Simulation — Simple Interface
============================================

Scan a 3D volume and reconstruct it using inverse Radon transform.

Simplest usage (Python):
    from simulate import scan_and_reconstruct
    from engine.geometry import Specimen3D

    specimen = Specimen3D(thickness=50e-3, width=50e-3, depth=30e-3)
    volume, metrics, out = scan_and_reconstruct(specimen, frequency=10e6)

With defects:
    from engine.geometry import CylindricalDefect, SphericalDefect

    defects = [SphericalDefect(center_z=25e-3, center_x=0, center_y=0, radius=2e-3)]
    volume, metrics, out = scan_and_reconstruct(
        specimen, defects=defects,
        frequency=10e6, bandwidth=0.6, n_scans=32,
    )

CLI usage:
    python simulate.py
    python simulate.py --frequency 5e6 --n-scans 8 --grain-size 1e-3
    python simulate.py --material STEEL_MILD --bandwidth 0.3 --snr-db 40
    python simulate.py --thickness 40 --width 50 --depth 30 --no-reconstruct
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import sys
import os
from datetime import datetime
from typing import List, Optional, Tuple

# Add parent directory for Classes/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from engine.config import (
    SimulationConfig, SpecimenConfig, ArrayConfig, ScanPlanConfig,
)
from engine.geometry import (
    Specimen3D, SphericalDefect, CylindricalDefect, PlanarCrack3D,
)
from engine.materials import ALUMINUM, STEEL_MILD, STEEL_STAINLESS, WATER, NDT_GEL, MaterialProperties
from engine.voxel_volume import VoxelVolume3D
from engine.microstructure import generate_grain_structure, embed_geometric_defects

# Import utility functions from run_engine (avoid duplicating code)
from run_engine import (
    add_noise,
    apply_bandpass_filter,
    reconstruct_tfm,
    visualize,
    preview_volume_3d,
    rasterize_volume,
    view_in_napari,
    visualize_scans,
    scan_volume_3d,
)
from reconstruct_3d import reconstruct_and_compare, save_ground_truth

MATERIAL_PRESETS = {
    'ALUMINUM': ALUMINUM,
    'STEEL_MILD': STEEL_MILD,
    'STEEL_STAINLESS': STEEL_STAINLESS,
}


# ── Specimen & defect setup ─────────────────────────────────────────

def build_specimen(
    thickness: float = 50e-3,
    width: float = 50e-3,
    depth: float = 30e-3,
) -> Specimen3D:
    """Create a 3D specimen block.

    Args:
        thickness: Specimen depth (z), metres.
        width:     Lateral extent (x, along array), metres.
        depth:     Elevation extent (y, mechanical scan axis), metres.
    """
    return Specimen3D(thickness=thickness, width=width, depth=depth)


def build_defects(specimen: Specimen3D) -> List:
    """Return the default set of 3D geometric defects.

    Defects:
        - Spherical pore at (z=25 mm, x=0, y=0), r=2 mm
        - SDH cylinder running full y-depth at (z=15 mm, x=8 mm), r=1 mm
    """
    return [
        CylindricalDefect(
            center_z=0, center_x=0, radius=1e-3,
            y_start=-specimen.depth / 2, y_end=specimen.depth / 2,
        ),
    ]


# ── Grain / voxel volume ────────────────────────────────────────────

def build_grain_volume(
    specimen: Specimen3D,
    defects_3d: Optional[List] = None,
    frequency: float = 10e6,
    mean_grain_size_m: float = 0.5e-3,
    impedance_variation: float = 0.025,
    wavespeed_variation: float = 0.005,
    voxel_fraction: float = 1 / 3,
    material=ALUMINUM,
    seed: int = 42,
) -> VoxelVolume3D:
    """Generate a Voronoi grain volume and optionally embed defects.

    Args:
        specimen:            3D specimen geometry.
        defects_3d:          Geometric defects to burn in as voids.
                             Pass [] for grain-only (no defects).
        frequency:           Array centre frequency (Hz).
        mean_grain_size_m:   Target mean grain diameter (m).
        impedance_variation: Fractional per-grain Z spread (e.g. 0.025 = +/-2.5 %).
        wavespeed_variation: Fractional per-grain c_L spread.
        voxel_fraction:      Voxel size as a fraction of wavelength (default lambda/3).
        material:            Background material preset.
        seed:                RNG seed for reproducibility.

    Returns:
        VoxelVolume3D with grain structure (and embedded defects if provided).
    """
    wavelength = material.c_L / frequency
    voxel_size = wavelength * voxel_fraction

    print(f"\nGenerating voxel grain structure "
          f"(lambda = {wavelength*1e3:.2f} mm, "
          f"voxel = {voxel_size*1e3:.2f} mm ~ lambda/{1/voxel_fraction:.0f})...")

    grain_vol = generate_grain_structure(
        thickness=specimen.thickness,
        width=specimen.width,
        depth=specimen.depth,
        background_material=material,
        mean_grain_size_m=mean_grain_size_m,
        impedance_variation=impedance_variation,
        wavespeed_variation=wavespeed_variation,
        voxel_size_m=voxel_size,
        seed=seed,
    )

    if defects_3d is None:
        defects_3d = []
    volume = embed_geometric_defects(grain_vol, defects_3d)
    print(f"  Voxel volume shape: {volume.shape}")
    return volume


# ── Simulation config ────────────────────────────────────────────────

def build_config(
    specimen: Specimen3D,
    frequency: float = 10e6,
    num_elements: int = 64,
    element_pitch: float = 0.6e-3,
    bandwidth: float = 0.6,
    n_scans: int = 16,
    theta_start: float = -np.pi / 2,
    theta_end: float = np.pi / 2,
    max_bounces: int = 2,
    mode_conversion: bool = True,
) -> tuple:
    """Build SimulationConfig and ScanPlanConfig.

    Returns:
        (cfg, scan_plan) tuple.
    """
    scan_plan = ScanPlanConfig(
        n_scans=n_scans,
        theta_start=theta_start,
        theta_end=theta_end,
    )
    cfg = SimulationConfig(
        specimen=SpecimenConfig(
            thickness=specimen.thickness,
            width=specimen.width,
        ),
        array=ArrayConfig(
            num_elements=num_elements,
            element_pitch=element_pitch,
            frequency=frequency,
            bandwidth=bandwidth,
        ),
        scan_plan=scan_plan,
        max_bounces=max_bounces,
        mode_conversion=mode_conversion,
    )
    return cfg, scan_plan


# ── Run scan ─────────────────────────────────────────────────────────

def run_scan(
    specimen: Specimen3D,
    defects_3d: List,
    cfg: SimulationConfig,
    scan_plan: ScanPlanConfig,
    output_dir: Optional[str] = None,
    voxel_volume: Optional[VoxelVolume3D] = None,
    use_voxel_world: bool = True,
    show_preview: bool = True,
    show_visualisation: bool = True,
    tfm_z_start: float = 10e-3,
    tfm_z_end: Optional[float] = None,
    tfm_n_pixels: int = 800,
) -> str:
    """Execute the full scan pipeline: preview -> FMC -> TFM -> visualise.

    Args:
        specimen:           3D specimen geometry.
        defects_3d:         List of 3D geometric defects.
        cfg:                Simulation configuration.
        scan_plan:          Rotational scan plan.
        output_dir:         Where to save results. Defaults to
                            ``SYNTHETIC DATA/output/scans/scan_3d/``.
        voxel_volume:       Grain/defect voxel volume (Born scattering).
        use_voxel_world:    If True, Born scattering handles defects
                            (no Kirchhoff). If False, geometric defects
                            are passed to the Kirchhoff engine.
        show_preview:       Render a 3-view volume preview before scanning.
        show_visualisation: Generate grid PNG and animated GIF after scanning.
        tfm_z_start:        TFM reconstruction start depth (m).
        tfm_z_end:          TFM reconstruction end depth (m).
                            Default: thickness - 5 mm.
        tfm_n_pixels:       TFM pixel grid size (square).

    Returns:
        Path to the output directory.
    """
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), 'output', 'scans', 'scan_3d')

    print(cfg.summary())

    # Preview
    if show_preview:
        preview_path = os.path.join(output_dir, 'volume_preview.png')
        preview_volume_3d(specimen, defects_3d, scan_plan, preview_path)

    # Scan
    geom_defects = [] if use_voxel_world else defects_3d
    scan_volume_3d(
        specimen, geom_defects, cfg, scan_plan, output_dir,
        voxel_volume=voxel_volume,
        tfm_z_start=tfm_z_start,
        tfm_z_end=tfm_z_end,
        tfm_n_pixels=tfm_n_pixels,
    )

    print(f"\n{'#'*70}")
    print(f"# SCAN COMPLETE -- {scan_plan.n_scans} B-scans saved to {output_dir}/")
    print(f"{'#'*70}\n")

    # Visualise
    if show_visualisation:
        visualize_scans(output_dir)

    return output_dir


# ── Top-level API ────────────────────────────────────────────────────

def _resolve_material(material) -> MaterialProperties:
    """Accept a MaterialProperties object or a string name."""
    if isinstance(material, str):
        key = material.upper()
        if key not in MATERIAL_PRESETS:
            raise ValueError(
                f"Unknown material '{material}'. "
                f"Choose from: {list(MATERIAL_PRESETS.keys())}"
            )
        return MATERIAL_PRESETS[key]
    return material


def scan_and_reconstruct(
    specimen: Specimen3D,
    defects: Optional[List] = None,
    material='ALUMINUM',
    # Grain structure
    grain_size: float = 0.5e-3,
    impedance_variation: float = 0.025,
    wavespeed_variation: float = 0.005,
    seed: int = 42,
    # Array / imaging
    frequency: float = 10e6,
    num_elements: int = 64,
    element_pitch: float = 0.6e-3,
    bandwidth: float = 0.6,
    # Scan plan
    n_scans: int = 16,
    theta_start_deg: float = -90.0,
    theta_end_deg: float = 90.0,
    # Filtering & noise
    snr_db: float = 35.0,
    filter_alpha: float = 1.0,
    hanning: bool = False,
    # TFM
    tfm_n_pixels: int = 800,
    tfm_z_start: float = 10e-3,
    tfm_z_end: Optional[float] = None,
    # Radon reconstruction
    reconstruct: bool = True,
    radon_filter: str = 'hann',
    crop_to_cube: bool = False,
    # Output
    output_dir: Optional[str] = None,
    save_figures: bool = True,
    # Advanced
    max_bounces: int = 2,
    mode_conversion: bool = False,
) -> Tuple[Optional[np.ndarray], Optional[dict], str]:
    """
    Scan a 3D volume and optionally reconstruct it via inverse Radon transform.

    This is the main entry point. It chains the full pipeline:
        grain structure -> FMC simulation -> TFM B-scans -> 3D reconstruction

    Args:
        specimen:            Specimen3D geometry.
        defects:             List of 3D defect objects (default: no defects).
        material:            'ALUMINUM', 'STEEL_MILD', 'STEEL_STAINLESS',
                             or a MaterialProperties object.

        grain_size:          Mean grain diameter (m).
        impedance_variation: Per-grain Z spread fraction (e.g. 0.025 = +/-2.5%).
        wavespeed_variation: Per-grain c_L spread fraction.
        seed:                RNG seed for grain structure reproducibility.

        frequency:           Array centre frequency (Hz).
        num_elements:        Number of array elements.
        element_pitch:       Element centre-to-centre spacing (m).
        bandwidth:           Fractional bandwidth (e.g. 0.6 = 60%).

        n_scans:             Number of rotational scan angles.
        theta_start_deg:     Start angle in degrees (default -90).
        theta_end_deg:       End angle in degrees (default +90).

        snr_db:              Signal-to-noise ratio (dB).
        filter_alpha:        Tukey filter taper (0=rectangular, 1=Hann).
        hanning:             Apply Hanning pre-window before FFT.

        tfm_n_pixels:        TFM grid size (square, pixels per side).
        tfm_z_start:         TFM start depth (m).
        tfm_z_end:           TFM end depth (m). Default: thickness - 5 mm.

        reconstruct:         If True, run inverse Radon 3D reconstruction.
        radon_filter:        FBP filter for iradon ('hann', 'ramp', etc.).
        crop_to_cube:        Crop cylindrical reconstruction to inscribed cube.

        output_dir:          Where to save results. Auto-generated if None.
        save_figures:        Save diagnostic PNGs.

        max_bounces:         Number of wall echo bounces to simulate.
        mode_conversion:     Enable L->S mode conversion at back wall.

    Returns:
        (recon_volume, metrics, output_dir)
        recon_volume: 3D numpy array (or None if reconstruct=False)
        metrics:      Comparison metrics dict (or None if no ground truth)
        output_dir:   Path where all results are saved
    """
    material = _resolve_material(material)
    if defects is None:
        defects = []

    # Output directory
    if output_dir is None:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join(
            os.path.dirname(__file__), 'output', f'scan_{ts}',
        )

    print(f"\n{'#'*70}")
    print(f"# SCAN AND RECONSTRUCT")
    print(f"#   Material: {material.name}")
    print(f"#   Specimen: {specimen.thickness*1e3:.0f} x "
          f"{specimen.width*1e3:.0f} x {specimen.depth*1e3:.0f} mm")
    print(f"#   Frequency: {frequency/1e6:.0f} MHz, "
          f"Bandwidth: {bandwidth*100:.0f}%")
    print(f"#   Grain size: {grain_size*1e3:.2f} mm")
    print(f"#   Scans: {n_scans} angles "
          f"[{theta_start_deg:.0f} deg, {theta_end_deg:.0f} deg]")
    print(f"{'#'*70}\n")

    # 1. Build grain volume with embedded defects
    voxel_volume = build_grain_volume(
        specimen,
        defects_3d=defects,
        frequency=frequency,
        mean_grain_size_m=grain_size,
        impedance_variation=impedance_variation,
        wavespeed_variation=wavespeed_variation,
        material=material,
        seed=seed,
    )

    # 2. Build simulation config
    theta_start_rad = np.radians(theta_start_deg)
    theta_end_rad = np.radians(theta_end_deg)

    cfg, scan_plan = build_config(
        specimen,
        frequency=frequency,
        num_elements=num_elements,
        element_pitch=element_pitch,
        bandwidth=bandwidth,
        n_scans=n_scans,
        theta_start=theta_start_rad,
        theta_end=theta_end_rad,
        max_bounces=max_bounces,
        mode_conversion=mode_conversion,
    )
    cfg.acquisition.snr_db = snr_db
    cfg.acquisition.filter_alpha = filter_alpha
    cfg.acquisition.hanning_bool = hanning

    # 3. Run the scan (all defects embedded in voxel volume)
    run_scan(
        specimen,
        defects_3d=defects,
        cfg=cfg,
        scan_plan=scan_plan,
        output_dir=output_dir,
        voxel_volume=voxel_volume,
        use_voxel_world=True,
        show_preview=True,
        show_visualisation=save_figures,
        tfm_z_start=tfm_z_start,
        tfm_z_end=tfm_z_end,
        tfm_n_pixels=tfm_n_pixels,
    )

    # 4. Save ground truth
    gt_path = os.path.join(output_dir, 'ground_truth.npz')
    save_ground_truth(voxel_volume, gt_path)

    # 5. Reconstruct 3D volume from B-scans
    recon_volume = None
    metrics = None

    if reconstruct:
        recon_volume, metrics = reconstruct_and_compare(
            scan_dir=output_dir,
            voxel_volume=voxel_volume,
            filter_name=radon_filter,
            crop_to_cube=crop_to_cube,
            show_napari=False,
            save_figures=save_figures,
        )

    print(f"\n{'#'*70}")
    print(f"# ALL DONE -- results saved to {output_dir}/")
    print(f"{'#'*70}\n")

    return recon_volume, metrics, output_dir


# ── Demo: Al Hole 15MHz experiment ──────────────────────────────────

def demo_al_hole(no_defects: bool = False):
    """
    Single-scan test matched to Al Hole 15MHz 26012026 experiment.

    Real experiment parameters (from DATA/ Params.txt):
      Array:  Imasonic 1D 64 elements, 0.63 mm pitch, 15 MHz
      TFM:    c=6700 m/s, z=0..40 mm, 400x600 pixels, vmin=-5 dB
      Filter: alpha=0.2, bandwidth=0.2% (MHz_percentage=0.1), hanning=True
      Sample: 40 mm thick aluminium block with a cylindrical hole (SDH)
    """
    specimen = Specimen3D(thickness=40e-3, width=50e-3, depth=30e-3)

    if no_defects:
        defects = []
    else:
        defects = [
            CylindricalDefect(
                center_z=20e-3, center_x=0.0, radius=1e-3,
                y_start=-specimen.depth / 2, y_end=specimen.depth / 2,
            ),
        ]

    suffix = 'no_defects' if no_defects else 'with_defect'
    out = os.path.join(os.path.dirname(__file__), 'output', 'scans', f'scan_3d_{suffix}')

    scan_and_reconstruct(
        specimen,
        defects=defects,
        material=ALUMINUM,
        frequency=15e6,
        num_elements=128,
        element_pitch=0.3e-3,
        bandwidth=0.002,
        n_scans=32,
        filter_alpha=0.2,
        hanning=True,
        snr_db=35.0,
        tfm_z_start=0.0,
        tfm_z_end=40e-3,
        tfm_n_pixels=400,
        output_dir=out,
    )


# ── CLI ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Scan a 3D volume and reconstruct via inverse Radon',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Specimen geometry (mm for convenience)
    g = parser.add_argument_group('Specimen (dimensions in mm)')
    g.add_argument('--thickness', type=float, default=50,
                   help='Specimen thickness / z-depth (mm)')
    g.add_argument('--width', type=float, default=50,
                   help='Lateral extent along array (mm)')
    g.add_argument('--depth', type=float, default=30,
                   help='Elevation / y extent (mm)')

    # Material
    g = parser.add_argument_group('Material')
    g.add_argument('--material', type=str, default='ALUMINUM',
                   choices=list(MATERIAL_PRESETS.keys()),
                   help='Material preset')

    # Grain structure
    g = parser.add_argument_group('Grain structure')
    g.add_argument('--grain-size', type=float, default=0.5e-3,
                   help='Mean grain diameter (m)')
    g.add_argument('--impedance-variation', type=float, default=0.025,
                   help='Per-grain Z spread fraction')
    g.add_argument('--seed', type=int, default=42,
                   help='RNG seed for grain structure')

    # Array
    g = parser.add_argument_group('Array')
    g.add_argument('--frequency', type=float, default=10e6,
                   help='Centre frequency (Hz)')
    g.add_argument('--num-elements', type=int, default=64)
    g.add_argument('--element-pitch', type=float, default=0.6e-3,
                   help='Element pitch (m)')
    g.add_argument('--bandwidth', type=float, default=0.6,
                   help='Fractional bandwidth (0.6 = 60%%)')

    # Scan plan
    g = parser.add_argument_group('Scan plan')
    g.add_argument('--n-scans', type=int, default=16,
                   help='Number of rotational scan angles')
    g.add_argument('--theta-start', type=float, default=-90,
                   help='Start angle (degrees)')
    g.add_argument('--theta-end', type=float, default=90,
                   help='End angle (degrees)')

    # Filtering & noise
    g = parser.add_argument_group('Filtering & noise')
    g.add_argument('--snr-db', type=float, default=35.0)
    g.add_argument('--filter-alpha', type=float, default=1.0,
                   help='Tukey taper (0=rect, 1=Hann)')
    g.add_argument('--hanning', action='store_true',
                   help='Hanning pre-window before FFT')

    # TFM
    g = parser.add_argument_group('TFM reconstruction')
    g.add_argument('--tfm-n-pixels', type=int, default=800)
    g.add_argument('--tfm-z-start', type=float, default=10e-3,
                   help='TFM start depth (m)')
    g.add_argument('--tfm-z-end', type=float, default=None,
                   help='TFM end depth (m). Default: thickness - 5mm')

    # 3D reconstruction
    g = parser.add_argument_group('3D Radon reconstruction')
    g.add_argument('--no-reconstruct', action='store_true',
                   help='Skip 3D reconstruction (B-scans only)')
    g.add_argument('--radon-filter', type=str, default='hann',
                   choices=['hann', 'ramp', 'shepp-logan', 'hamming'],
                   help='FBP filter for inverse Radon')
    g.add_argument('--crop-to-cube', action='store_true',
                   help='Crop cylindrical reconstruction to inscribed cube')

    # Output
    g = parser.add_argument_group('Output')
    g.add_argument('--output', type=str, default=None,
                   help='Output directory (auto-generated if omitted)')

    # Advanced
    g = parser.add_argument_group('Advanced physics')
    g.add_argument('--max-bounces', type=int, default=2)
    g.add_argument('--mode-conversion', action='store_true',
                   help='Enable L->S mode conversion')

    # Demo mode
    parser.add_argument('--demo', action='store_true',
                        help='Run the Al Hole 15MHz demo instead')
    parser.add_argument('--no-defects', action='store_true',
                        help='(demo mode) Run grain-only, no defects')

    args = parser.parse_args()

    if args.demo:
        demo_al_hole(no_defects=args.no_defects)
        return

    # Build specimen (CLI takes mm, convert to m)
    specimen = Specimen3D(
        thickness=args.thickness * 1e-3,
        width=args.width * 1e-3,
        depth=args.depth * 1e-3,
    )

    scan_and_reconstruct(
        specimen=specimen,
        material=args.material,
        grain_size=args.grain_size,
        impedance_variation=args.impedance_variation,
        seed=args.seed,
        frequency=args.frequency,
        num_elements=args.num_elements,
        element_pitch=args.element_pitch,
        bandwidth=args.bandwidth,
        n_scans=args.n_scans,
        theta_start_deg=args.theta_start,
        theta_end_deg=args.theta_end,
        snr_db=args.snr_db,
        filter_alpha=args.filter_alpha,
        hanning=args.hanning,
        tfm_n_pixels=args.tfm_n_pixels,
        tfm_z_start=args.tfm_z_start,
        tfm_z_end=args.tfm_z_end,
        reconstruct=not args.no_reconstruct,
        radon_filter=args.radon_filter,
        crop_to_cube=args.crop_to_cube,
        output_dir=args.output,
        max_bounces=args.max_bounces,
        mode_conversion=args.mode_conversion,
    )


if __name__ == '__main__':
    main()
