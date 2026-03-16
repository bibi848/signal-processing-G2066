"""
Synthetic NDT Simulation — Callable Interface
==============================================

Provides reusable functions to configure and run the full synthetic NDT
pipeline (specimen → grain structure → FMC → TFM → visualisation).

All parameters that were hard-coded in run_engine.main() are exposed as
function arguments with sensible defaults so the pipeline can be driven
from notebooks, parameter sweeps, or other scripts.

Usage — standalone (identical to run_engine.py):
    python simulate.py

Usage — as a library:
    from simulate import build_specimen, build_grain_volume, build_config, run_scan

    specimen, defects = build_specimen()
    vol = build_grain_volume(specimen, defects, frequency=10e6)
    cfg, scan_plan = build_config(specimen, frequency=10e6)
    run_scan(specimen, defects, cfg, scan_plan, voxel_volume=vol)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import sys
import os
from typing import List, Optional

# Add parent directory for Classes/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from engine.config import (
    SimulationConfig, SpecimenConfig, ArrayConfig, ScanPlanConfig,
)
from engine.geometry import (
    Specimen3D, SphericalDefect, CylindricalDefect, PlanarCrack3D,
)
from engine.materials import ALUMINUM, STEEL_MILD, STEEL_STAINLESS, WATER, NDT_GEL
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
        impedance_variation: Fractional per-grain Z spread (e.g. 0.025 = ±2.5 %).
        wavespeed_variation: Fractional per-grain c_L spread.
        voxel_fraction:      Voxel size as a fraction of wavelength (default λ/3).
        material:            Background material preset.
        seed:                RNG seed for reproducibility.

    Returns:
        VoxelVolume3D with grain structure (and embedded defects if provided).
    """
    wavelength = material.c_L / frequency
    voxel_size = wavelength * voxel_fraction

    print(f"\nGenerating voxel grain structure "
          f"(λ = {wavelength*1e3:.2f} mm, "
          f"voxel = {voxel_size*1e3:.2f} mm ≈ λ/{1/voxel_fraction:.0f})...")

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
    """Execute the full scan pipeline: preview → FMC → TFM → visualise.

    Args:
        specimen:           3D specimen geometry.
        defects_3d:         List of 3D geometric defects.
        cfg:                Simulation configuration.
        scan_plan:          Rotational scan plan.
        output_dir:         Where to save results. Defaults to
                            ``SYNTHETIC DATA/output/scan_3d/``.
        voxel_volume:       Grain/defect voxel volume (Born scattering).
        use_voxel_world:    If True, Born scattering handles defects
                            (no Kirchhoff). If False, geometric defects
                            are passed to the Kirchhoff engine.
        show_preview:       Render a 3-view volume preview before scanning.
        show_visualisation: Generate grid PNG and animated GIF after scanning.
        tfm_z_start:        TFM reconstruction start depth (m).
        tfm_z_end:          TFM reconstruction end depth (m).
                            Default: thickness − 5 mm.
        tfm_n_pixels:       TFM pixel grid size (square).

    Returns:
        Path to the output directory.
    """
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), 'output', 'scan_3d')

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
    print(f"# SCAN COMPLETE — {scan_plan.n_scans} B-scans saved to {output_dir}/")
    print(f"{'#'*70}\n")

    # Visualise
    if show_visualisation:
        visualize_scans(output_dir)

    return output_dir


# ── Main ─────────────────────────────────────────────────────────────

def main():
    """Run a simulation matching the experimental Al 10 MHz setup."""
    print(f"\n{'#'*70}")
    print(f"# PHYSICS-ACCURATE 3D NDT SYNTHETIC DATA ENGINE")
    print(f"{'#'*70}\n")

    # Specimen: 70 mm thick aluminium block (wider than array aperture)
    # Array aperture = 127 × 0.3 mm = 38.1 mm → specimen width 50 mm gives margin
    specimen = build_specimen(
        thickness=70e-3,   # 70 mm deep
        width=50e-3,       # 50 mm wide (array aperture = 38.1 mm fits inside)
        depth=30e-3,       # 30 mm elevation (rotational scan axis)
    )
    defects_3d = build_defects(specimen)

    frequency = 10e6  # 10 MHz — matches experimental array

    # Grain volume with embedded defects
    voxel_volume = build_grain_volume(
        specimen,
        defects_3d=defects_3d,
        frequency=frequency,
    )

    # 128-element, 0.3 mm pitch array — matches experimental hardware
    cfg, scan_plan = build_config(
        specimen,
        frequency=frequency,
        num_elements=128,
        element_pitch=0.3e-3,
    )

    # TFM z_range: 10–65 mm (skip near-surface, stop before 70 mm back wall)
    # x_range is automatically clipped to array aperture (±19.05 mm)
    output_dir = run_scan(
        specimen, defects_3d, cfg, scan_plan,
        voxel_volume=voxel_volume,
        use_voxel_world=True,
        tfm_z_start=10e-3,
        tfm_z_end=65e-3,
    )

    # Save ground truth for reconstruction comparison
    gt_path = os.path.join(output_dir, 'ground_truth.npz')
    save_ground_truth(voxel_volume, gt_path)

    # Reconstruct 3D volume from B-scans and compare to ground truth
    reconstruct_and_compare(
        scan_dir=output_dir,
        voxel_volume=voxel_volume,
        show_napari=True,
        save_figures=True,
    )


if __name__ == '__main__':
    main()
