"""
Reconstruction Validation Pipeline
===================================

Creates a synthetic volume with known geometry, simulates rotational TFM
B-scans, reconstructs via inverse Radon, and quantifies reconstruction
errors against the ground truth.

Usage — single run:
    python validate_reconstruction.py

Usage — parameter sweep:
    python validate_reconstruction.py --sweep

Usage — as a library:
    from validate_reconstruction import validate, sweep_validation
    metrics = validate(n_scans=32, defect_type='cylinder')
"""

import os
import sys
import json
import time as timer
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Optional, List

# Add parent directory for Classes/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from engine.config import SimulationConfig, SpecimenConfig, ArrayConfig, ScanPlanConfig
from engine.geometry import Specimen3D, CylindricalDefect, SphericalDefect
from engine.materials import ALUMINUM, STEEL_MILD
from engine.voxel_volume import VoxelVolume3D
from engine.microstructure import generate_grain_structure, embed_geometric_defects

from run_engine import scan_volume_3d
from reconstruct_3d import (
    reconstruct_and_compare,
    save_ground_truth,
    load_bscans,
)


# ── Build synthetic scene ─────────────────────────────────────────────

def build_scene(
    thickness: float = 40e-3,
    width: float = 50e-3,
    depth: float = 30e-3,
    defect_type: str = 'cylinder',
    defect_radius: float = 1e-3,
    defect_z: float = 20e-3,
    defect_x: float = 0.0,
    material=ALUMINUM,
    mean_grain_size_m: float = 0.5e-3,
    impedance_variation: float = 0.025,
    frequency: float = 5e6,
    seed: int = 42,
):
    """
    Create a synthetic specimen with a single known defect and grain structure.

    Args:
        thickness:           Specimen thickness (m).
        width:               Specimen width along array (m).
        depth:               Specimen depth along rotation axis (m).
        defect_type:         'cylinder', 'sphere', or 'none'.
        defect_radius:       Defect radius (m).
        defect_z:            Defect centre depth (m).
        defect_x:            Defect centre lateral position (m).
        material:            Background material.
        mean_grain_size_m:   Voronoi grain size (m).
        impedance_variation: Per-grain impedance spread.
        frequency:           Centre frequency for voxel sizing.
        seed:                RNG seed.

    Returns:
        (specimen, defects_3d, voxel_volume)
    """
    specimen = Specimen3D(thickness=thickness, width=width, depth=depth)

    if defect_type == 'cylinder':
        defects_3d = [
            CylindricalDefect(
                center_z=defect_z, center_x=defect_x, radius=defect_radius,
                y_start=-depth / 2, y_end=depth / 2,
            ),
        ]
    elif defect_type == 'sphere':
        defects_3d = [
            SphericalDefect(
                center_z=defect_z, center_x=defect_x, center_y=0.0,
                radius=defect_radius,
            ),
        ]
    elif defect_type == 'none':
        defects_3d = []
    else:
        raise ValueError(f"Unknown defect_type: {defect_type}")

    # Build grain volume
    wavelength = material.c_L / frequency
    voxel_size = wavelength / 3

    print(f"Building grain volume (voxel={voxel_size*1e3:.2f} mm, "
          f"grain={mean_grain_size_m*1e3:.1f} mm)...")

    grain_vol = generate_grain_structure(
        thickness=thickness,
        width=width,
        depth=depth,
        background_material=material,
        mean_grain_size_m=mean_grain_size_m,
        impedance_variation=impedance_variation,
        voxel_size_m=voxel_size,
        seed=seed,
    )
    voxel_volume = embed_geometric_defects(grain_vol, defects_3d)
    print(f"  Volume shape: {voxel_volume.shape}")

    return specimen, defects_3d, voxel_volume


# ── Simulate scans ────────────────────────────────────────────────────

def simulate_scans(
    specimen: Specimen3D,
    defects_3d: list,
    voxel_volume: VoxelVolume3D,
    output_dir: str,
    frequency: float = 5e6,
    num_elements: int = 64,
    element_pitch: float = 0.63e-3,
    bandwidth: float = 0.6,
    n_scans: int = 32,
    snr_db: float = 35.0,
    max_bounces: int = 2,
    tfm_z_start: float = 0.0,
    tfm_z_end: Optional[float] = None,
    tfm_n_pixels: int = 400,
):
    """
    Run rotational TFM simulation and save B-scans.

    Returns:
        Path to output directory.
    """
    if tfm_z_end is None:
        tfm_z_end = specimen.thickness

    scan_plan = ScanPlanConfig(
        n_scans=n_scans,
        theta_start=-np.pi / 2,
        theta_end=np.pi / 2,
    )

    cfg = SimulationConfig(
        material=ALUMINUM,
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
    )
    cfg.acquisition.snr_db = snr_db

    print(f"\nSimulating {n_scans} rotational scans...")
    t0 = timer.time()

    scan_volume_3d(
        specimen, [], cfg, scan_plan, output_dir,
        voxel_volume=voxel_volume,
        tfm_z_start=tfm_z_start,
        tfm_z_end=tfm_z_end,
        tfm_n_pixels=tfm_n_pixels,
    )

    elapsed = timer.time() - t0
    print(f"Simulation complete in {elapsed:.1f}s")

    return output_dir


# ── Diagnostic: grain slice vs B-scan ─────────────────────────────────

def plot_slices_vs_bscans(
    voxel_volume: VoxelVolume3D,
    scan_dir: str,
    output_path: Optional[str] = None,
    n_show: int = 4,
):
    """
    Plot grain structure slices alongside the corresponding TFM B-scans.

    For each selected angle, shows:
      - Left:  impedance slice through the voxel volume at that angle
      - Right: TFM B-scan produced by the simulation at that angle

    This allows visual verification that the ray-tracing engine is
    producing B-scans that match the underlying grain structure.

    Args:
        voxel_volume: The ground-truth VoxelVolume3D.
        scan_dir:     Directory containing bscan_*.npy and scan_meta.npy.
        output_path:  Where to save the figure. Default: scan_dir/slices_vs_bscans.png.
        n_show:       Number of angles to display (evenly spaced).
    """
    if output_path is None:
        output_path = os.path.join(scan_dir, 'slices_vs_bscans.png')

    # Load B-scans and metadata
    bscans_db, meta = load_bscans(scan_dir)
    angles_rad = meta['angles_rad']
    n_scans = len(angles_rad)

    # Build the TFM imaging grid (matches what scan_volume_3d used)
    z_start = meta.get('tfm_z_start_m', 0.0)
    z_end = meta.get('tfm_z_end_m', meta['specimen_thickness_m'])
    n_pixels = meta.get('tfm_n_pixels', bscans_db.shape[1])
    half_ap = meta['array_aperture_m'] / 2.0

    z_grid = np.linspace(z_start, z_end, bscans_db.shape[1])
    l_grid = np.linspace(-half_ap, half_ap, bscans_db.shape[2])

    # Pick evenly spaced scan indices
    indices = np.linspace(0, n_scans - 1, n_show, dtype=int)

    fig, axes = plt.subplots(n_show, 3, figsize=(15, 4 * n_show))
    if n_show == 1:
        axes = axes.reshape(1, 3)

    # Get background impedance for contrast display
    Z0 = float(np.median(voxel_volume.impedance))

    for row, idx in enumerate(indices):
        theta = angles_rad[idx]
        angle_deg = np.degrees(theta)

        # Slice the grain volume at this angle
        imp_slice = voxel_volume.slice_at_angle(theta, z_grid, l_grid)
        # Convert to contrast: (Z - Z0) / Z0
        contrast = (imp_slice - Z0) / Z0

        # B-scan (dB scale)
        bscan = bscans_db[idx]
        ext = [l_grid[0]*1e3, l_grid[-1]*1e3, z_grid[-1]*1e3, z_grid[0]*1e3]

        # Column 0: Grain structure — clipped to grain scale so grains are visible
        # Defect voids (contrast ≈ -1) saturate to dark blue
        ax_grain = axes[row, 0]
        grain_lim = 0.06  # ±6% — shows grain boundaries clearly
        im0 = ax_grain.imshow(
            contrast, extent=ext, aspect='auto',
            cmap='RdBu_r', vmin=-grain_lim, vmax=grain_lim,
        )
        ax_grain.set_title(f'Grain structure θ={angle_deg:+.1f}°')
        ax_grain.set_xlabel('Lateral (mm)')
        ax_grain.set_ylabel('Depth (mm)')
        plt.colorbar(im0, ax=ax_grain, label='ΔZ/Z₀')

        # Column 1: Same slice — full range showing defect as void
        ax_defect = axes[row, 1]
        im1 = ax_defect.imshow(
            imp_slice, extent=ext, aspect='auto', cmap='gray',
        )
        ax_defect.set_title(f'Impedance (full range) θ={angle_deg:+.1f}°')
        ax_defect.set_xlabel('Lateral (mm)')
        ax_defect.set_ylabel('Depth (mm)')
        plt.colorbar(im1, ax=ax_defect, label='Z (Pa·s/m)')

        # Column 2: TFM B-scan
        ax_bscan = axes[row, 2]
        b_vmin = np.percentile(bscan, 5)
        b_vmax = np.percentile(bscan, 99)
        im2 = ax_bscan.imshow(
            bscan, extent=ext, aspect='auto',
            cmap='hot', vmin=b_vmin, vmax=b_vmax,
        )
        ax_bscan.set_title(f'TFM B-scan #{idx} θ={angle_deg:+.1f}°')
        ax_bscan.set_xlabel('Lateral (mm)')
        ax_bscan.set_ylabel('Depth (mm)')
        plt.colorbar(im2, ax=ax_bscan, label='dB')

    fig.suptitle('Grain Structure vs Simulated TFM B-scans', fontweight='bold', y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Slice comparison saved: {output_path}")


# ── Full validation run ───────────────────────────────────────────────

def validate(
    # Scene
    thickness: float = 40e-3,
    width: float = 50e-3,
    depth: float = 30e-3,
    defect_type: str = 'cylinder',
    defect_radius: float = 1e-3,
    defect_z: float = 20e-3,
    defect_x: float = 0.0,
    mean_grain_size_m: float = 0.5e-3,
    impedance_variation: float = 0.025,
    # Array
    frequency: float = 5e6,
    num_elements: int = 64,
    element_pitch: float = 0.63e-3,
    bandwidth: float = 0.6,
    # Scan
    n_scans: int = 32,
    snr_db: float = 35.0,
    max_bounces: int = 2,
    # TFM
    tfm_z_start: float = 0.0,
    tfm_z_end: Optional[float] = None,
    tfm_n_pixels: int = 400,
    # Output
    output_dir: Optional[str] = None,
    seed: int = 42,
    show_napari: bool = False,
) -> dict:
    """
    End-to-end validation: build scene → simulate → reconstruct → compare.

    Returns:
        dict with all metrics and parameters.
    """
    if output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join(
            os.path.dirname(__file__), 'output',
            f'validation_{timestamp}',
        )
    os.makedirs(output_dir, exist_ok=True)

    params = {
        'thickness': thickness, 'width': width, 'depth': depth,
        'defect_type': defect_type, 'defect_radius': defect_radius,
        'defect_z': defect_z, 'defect_x': defect_x,
        'mean_grain_size_m': mean_grain_size_m,
        'impedance_variation': impedance_variation,
        'frequency': frequency, 'num_elements': num_elements,
        'element_pitch': element_pitch, 'bandwidth': bandwidth,
        'n_scans': n_scans, 'snr_db': snr_db,
        'max_bounces': max_bounces,
        'tfm_z_start': tfm_z_start,
        'tfm_z_end': tfm_z_end or thickness,
        'tfm_n_pixels': tfm_n_pixels,
        'seed': seed,
    }

    print(f"\n{'='*70}")
    print(f"RECONSTRUCTION VALIDATION")
    print(f"{'='*70}")
    for k, v in params.items():
        print(f"  {k}: {v}")
    print()

    t_total = timer.time()

    # 1. Build scene
    specimen, defects_3d, voxel_volume = build_scene(
        thickness=thickness, width=width, depth=depth,
        defect_type=defect_type, defect_radius=defect_radius,
        defect_z=defect_z, defect_x=defect_x,
        mean_grain_size_m=mean_grain_size_m,
        impedance_variation=impedance_variation,
        frequency=frequency, seed=seed,
    )

    # 2. Save ground truth
    gt_path = os.path.join(output_dir, 'ground_truth.npz')
    save_ground_truth(voxel_volume, gt_path)

    # 3. Simulate scans
    simulate_scans(
        specimen, defects_3d, voxel_volume, output_dir,
        frequency=frequency,
        num_elements=num_elements,
        element_pitch=element_pitch,
        bandwidth=bandwidth,
        n_scans=n_scans,
        snr_db=snr_db,
        max_bounces=max_bounces,
        tfm_z_start=tfm_z_start,
        tfm_z_end=tfm_z_end,
        tfm_n_pixels=tfm_n_pixels,
    )

    # 4. Diagnostic: grain structure slices vs B-scans
    plot_slices_vs_bscans(voxel_volume, output_dir)

    # 5. Reconstruct and compare
    volume, metrics = reconstruct_and_compare(
        scan_dir=output_dir,
        voxel_volume=voxel_volume,
        show_napari=show_napari,
        save_figures=True,
    )

    elapsed_total = timer.time() - t_total

    # 6. Save results
    result = {
        'params': params,
        'metrics': {k: float(v) for k, v in metrics.items()
                    if not isinstance(v, np.ndarray)},
        'elapsed_s': elapsed_total,
        'output_dir': output_dir,
        'volume_shape': list(volume.shape),
    }

    results_path = os.path.join(output_dir, 'validation_results.json')
    with open(results_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n{'='*70}")
    print(f"VALIDATION COMPLETE — {elapsed_total:.1f}s total")
    print(f"Results saved to {results_path}")
    print(f"{'='*70}\n")

    return result


# ── Parameter sweep ───────────────────────────────────────────────────

def sweep_validation(
    sweep_params: dict,
    base_params: Optional[dict] = None,
    output_root: Optional[str] = None,
) -> list:
    """
    Run validate() over a grid of parameter values.

    Args:
        sweep_params: Dict mapping parameter names to lists of values.
                      e.g. {'n_scans': [8, 16, 32, 64], 'snr_db': [20, 35, 50]}
        base_params:  Fixed parameters (passed to validate()).
        output_root:  Root directory for all runs.

    Returns:
        List of result dicts.
    """
    import itertools

    if base_params is None:
        base_params = {}

    if output_root is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_root = os.path.join(
            os.path.dirname(__file__), 'output',
            f'validation_sweep_{timestamp}',
        )
    os.makedirs(output_root, exist_ok=True)

    # Build parameter grid
    keys = list(sweep_params.keys())
    values = list(sweep_params.values())
    combos = list(itertools.product(*values))

    print(f"\n{'#'*70}")
    print(f"# VALIDATION SWEEP — {len(combos)} runs")
    print(f"# Sweeping: {keys}")
    print(f"# Output: {output_root}")
    print(f"{'#'*70}\n")

    results = []
    for i, combo in enumerate(combos):
        run_params = dict(base_params)
        label_parts = []
        for k, v in zip(keys, combo):
            run_params[k] = v
            label_parts.append(f"{k}={v}")

        run_label = f"run_{i:03d}_{'_'.join(label_parts)}"
        run_dir = os.path.join(output_root, run_label)

        print(f"\n{'─'*70}")
        print(f"Run {i+1}/{len(combos)}: {run_label}")
        print(f"{'─'*70}")

        result = validate(output_dir=run_dir, **run_params)
        results.append(result)

    # Save sweep summary
    summary = {
        'sweep_params': {k: [_to_json(x) for x in v]
                         for k, v in sweep_params.items()},
        'base_params': {k: _to_json(v) for k, v in base_params.items()},
        'n_runs': len(results),
        'results': [
            {
                'params': r['params'],
                'metrics': r['metrics'],
                'elapsed_s': r['elapsed_s'],
            }
            for r in results
        ],
    }

    summary_path = os.path.join(output_root, 'sweep_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    # Plot sweep results
    _plot_sweep(results, keys, output_root)

    print(f"\n{'#'*70}")
    print(f"# SWEEP COMPLETE — {len(results)} runs")
    print(f"# Summary: {summary_path}")
    print(f"{'#'*70}\n")

    return results


def _to_json(v):
    """Convert numpy types to JSON-serialisable."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


def _plot_sweep(results: list, sweep_keys: list, output_dir: str):
    """Plot reconstruction metrics vs swept parameters."""
    metric_names = ['ssim_mean', 'nrmse', 'pearson_r', 'cnr_recon',
                    'defect_centroid_error_mm']
    metric_labels = ['SSIM', 'NRMSE', 'Pearson r', 'CNR (recon)',
                     'Centroid error (mm)']

    for key in sweep_keys:
        x_vals = [r['params'][key] for r in results]
        unique_x = sorted(set(x_vals))

        if len(unique_x) < 2:
            continue

        fig, axes = plt.subplots(1, len(metric_names), figsize=(4 * len(metric_names), 4))

        for ax, mname, mlabel in zip(axes, metric_names, metric_labels):
            y_vals = [r['metrics'].get(mname, np.nan) for r in results]

            # Group by x value (in case of repeated params from other sweep dims)
            grouped = {}
            for x, y in zip(x_vals, y_vals):
                grouped.setdefault(x, []).append(y)

            x_plot = sorted(grouped.keys())
            y_mean = [np.mean(grouped[x]) for x in x_plot]
            y_std = [np.std(grouped[x]) for x in x_plot]

            ax.errorbar(x_plot, y_mean, yerr=y_std, marker='o', capsize=4)
            ax.set_xlabel(key)
            ax.set_ylabel(mlabel)
            ax.grid(True, alpha=0.3)

        fig.suptitle(f'Reconstruction quality vs {key}', fontweight='bold')
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f'sweep_{key}.png'), dpi=150)
        plt.close(fig)
        print(f"  Plot saved: sweep_{key}.png")


# ── CLI ───────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Validate inverse Radon reconstruction against synthetic ground truth'
    )
    parser.add_argument('--sweep', action='store_true',
                        help='Run a parameter sweep instead of a single validation')
    parser.add_argument('--n-scans', type=int, default=32,
                        help='Number of rotational scans (default: 32)')
    parser.add_argument('--defect', type=str, default='cylinder',
                        choices=['cylinder', 'sphere', 'none'],
                        help='Defect type (default: cylinder)')
    parser.add_argument('--frequency', type=float, default=5e6,
                        help='Centre frequency in Hz (default: 5e6)')
    parser.add_argument('--tfm-n-pixels', type=int, default=400,
                        help='TFM pixel grid size (default: 400)')
    parser.add_argument('--napari', action='store_true',
                        help='Open napari viewer after reconstruction')

    args = parser.parse_args()

    if args.sweep:
        # Default sweep: vary n_scans to study angular sampling
        sweep_validation(
            sweep_params={
                'n_scans': [8, 16, 32, 64],
            },
            base_params={
                'defect_type': args.defect,
                'frequency': args.frequency,
                'tfm_n_pixels': args.tfm_n_pixels,
                'show_napari': False,
            },
        )
    else:
        validate(
            n_scans=args.n_scans,
            defect_type=args.defect,
            frequency=args.frequency,
            tfm_n_pixels=args.tfm_n_pixels,
            show_napari=args.napari,
        )


if __name__ == '__main__':
    main()
