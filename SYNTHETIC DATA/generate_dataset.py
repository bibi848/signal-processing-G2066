"""
Multi-Position Synthetic Data Generation
=========================================

Generates multiple overlapping scan datasets from a single large specimen
for stitching validation. Supports 2D (B-scans only) and 3D (B-scans +
inverse Radon reconstruction) modes. Ground truth is recorded at every
scan position.

Usage -- standalone:
    python generate_dataset.py

Usage -- as a library:
    from generate_dataset import generate_dataset
    generate_dataset(
        width_total=100e-3, depth_total=60e-3,
        n_positions_x=3, n_positions_y=2,
        overlap_fraction=0.3, mode='3d',
    )
"""

import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
from datetime import datetime
from typing import List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from engine.config import SimulationConfig, SpecimenConfig, ArrayConfig, ScanPlanConfig
from engine.geometry import (
    Specimen3D, SphericalDefect, CylindricalDefect, PlanarCrack3D,
)
from engine.materials import ALUMINUM, MaterialProperties
from engine.voxel_volume import VoxelVolume3D
from engine.microstructure import generate_grain_structure, embed_geometric_defects
from run_engine import scan_volume_3d
from reconstruct_3d import (
    reconstruct_and_compare, save_ground_truth, load_ground_truth_file,
)


# ── Large specimen creation ──────────────────────────────────────────

def create_large_specimen(
    width_total: float = 100e-3,
    depth_total: float = 60e-3,
    thickness: float = 50e-3,
    defects_3d: Optional[List] = None,
    frequency: float = 10e6,
    mean_grain_size_m: float = 0.5e-3,
    impedance_variation: float = 0.025,
    wavespeed_variation: float = 0.005,
    voxel_fraction: float = 1 / 3,
    material: MaterialProperties = ALUMINUM,
    seed: int = 42,
) -> Tuple[Specimen3D, VoxelVolume3D]:
    """
    Build one large specimen with continuous grain structure.

    Args:
        width_total:  Full x-extent of the specimen (m).
        depth_total:  Full y-extent (m).
        thickness:    z-thickness (m).
        defects_3d:   Defects placed anywhere in the volume.
        frequency:    Array centre frequency (Hz).
        mean_grain_size_m: Target mean grain diameter.
        impedance_variation: Per-grain Z spread fraction.
        wavespeed_variation: Per-grain c_L spread fraction.
        voxel_fraction: Voxel size as fraction of wavelength.
        material:     Background material preset.
        seed:         RNG seed for reproducibility.

    Returns:
        (specimen, large_volume) tuple.
    """
    wavelength = material.c_L / frequency
    voxel_size = wavelength * voxel_fraction

    specimen = Specimen3D(
        thickness=thickness, width=width_total, depth=depth_total,
    )

    print(f"\n{'='*60}")
    print(f"  CREATING LARGE SPECIMEN")
    print(f"  {width_total*1e3:.0f} x {depth_total*1e3:.0f} x {thickness*1e3:.0f} mm")
    print(f"  voxel = {voxel_size*1e6:.0f} um, grain = {mean_grain_size_m*1e3:.1f} mm")
    print(f"{'='*60}\n")

    grain_vol = generate_grain_structure(
        thickness=thickness,
        width=width_total,
        depth=depth_total,
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
    print(f"  Volume shape: {volume.shape}")
    return specimen, volume


# ── Scan position grid ───────────────────────────────────────────────

def compute_scan_positions(
    width_total: float,
    depth_total: float,
    aperture: float,
    n_positions_x: int = 3,
    n_positions_y: int = 2,
    overlap_fraction: float = 0.3,
) -> Tuple[List[Tuple[float, float]], dict]:
    """
    Compute a grid of scan positions with overlap.

    The field of view (FOV) at each position is the array aperture.
    Adjacent positions overlap by ``overlap_fraction * aperture``.

    Args:
        width_total:      Full specimen x-extent (m).
        depth_total:      Full specimen y-extent (m).
        aperture:         Array aperture (m).
        n_positions_x:    Number of positions along x.
        n_positions_y:    Number of positions along y.
        overlap_fraction: Fraction of aperture that overlaps (0-1).

    Returns:
        (positions, info) where positions is a list of (px, py) in metres
        and info is a dict with step sizes, overlap, and validation.
    """
    # Enforce minimum 20% overlap
    if overlap_fraction < 0.2:
        print(f"  WARNING: overlap_fraction={overlap_fraction:.2f} < 0.2, "
              f"clamping to 0.2")
        overlap_fraction = 0.2

    # After cube cropping, the effective FOV is aperture / sqrt(2)
    cube_side = aperture / np.sqrt(2)
    step = cube_side * (1.0 - overlap_fraction)
    overlap_m = cube_side * overlap_fraction

    # Center the grid on the specimen
    if n_positions_x > 1:
        range_x = (n_positions_x - 1) * step
        px_values = np.linspace(-range_x / 2, range_x / 2, n_positions_x)
    else:
        px_values = np.array([0.0])

    if n_positions_y > 1:
        range_y = (n_positions_y - 1) * step
        py_values = np.linspace(-range_y / 2, range_y / 2, n_positions_y)
    else:
        py_values = np.array([0.0])

    # Validate: FOV must stay within specimen bounds
    max_px = np.max(np.abs(px_values)) + aperture / 2
    max_py = np.max(np.abs(py_values)) + aperture / 2
    x_ok = max_px <= width_total / 2 + 1e-6
    y_ok = max_py <= depth_total / 2 + 1e-6

    if not x_ok:
        print(f"  WARNING: scan grid extends {max_px*1e3:.1f} mm in x "
              f"but specimen is +/-{width_total/2*1e3:.1f} mm")
    if not y_ok:
        print(f"  WARNING: scan grid extends {max_py*1e3:.1f} mm in y "
              f"but specimen is +/-{depth_total/2*1e3:.1f} mm")

    positions = []
    for py in py_values:
        for px in px_values:
            positions.append((float(px), float(py)))

    info = {
        'n_positions_x': n_positions_x,
        'n_positions_y': n_positions_y,
        'n_total': len(positions),
        'step_x_m': float(step),
        'step_y_m': float(step),
        'overlap_fraction': overlap_fraction,
        'overlap_m': float(overlap_m),
        'aperture_m': float(aperture),
        'cube_side_m': float(cube_side),
        'bounds_valid': bool(x_ok and y_ok),
    }

    print(f"\n  Scan grid: {n_positions_x} x {n_positions_y} = {len(positions)} positions")
    print(f"  Step = {step*1e3:.1f} mm, overlap = {overlap_m*1e3:.1f} mm "
          f"({overlap_fraction*100:.0f}%)")

    return positions, info


# ── Origin-shifted volume view ───────────────────────────────────────

def create_shifted_volume(
    large_vol: VoxelVolume3D,
    px: float,
    py: float,
) -> VoxelVolume3D:
    """
    Create an origin-shifted view of the large volume.

    Shifting the origin makes the engine (which always scans at x=0, y=0)
    sample the region centered at (px, py) in the original volume.
    The numpy arrays are shared by reference — zero memory cost.
    """
    return VoxelVolume3D(
        impedance=large_vol.impedance,
        wavespeed=large_vol.wavespeed,
        voxel_size=large_vol.voxel_size,
        origin_z=large_vol.origin_z,
        origin_y=large_vol.origin_y - py,
        origin_x=large_vol.origin_x - px,
    )


# ── Ground truth sub-volume extraction ───────────────────────────────

def extract_ground_truth_subvolume(
    large_vol: VoxelVolume3D,
    px: float,
    py: float,
    fov_x: float,
    fov_y: float,
) -> VoxelVolume3D:
    """
    Extract a sub-volume around position (px, py) from the large volume.

    Args:
        large_vol: Full VoxelVolume3D.
        px, py:    Scan center position (m).
        fov_x:     Field of view in x (m).
        fov_y:     Field of view in y (m).

    Returns:
        New VoxelVolume3D containing only the sub-region (copied data).
    """
    vs = large_vol.voxel_size
    n_z, n_y, n_x = large_vol.shape

    # Voxel index ranges for the sub-region
    ix_start = int(round((px - fov_x / 2 - large_vol.origin_x) / vs))
    ix_end = int(round((px + fov_x / 2 - large_vol.origin_x) / vs))
    iy_start = int(round((py - fov_y / 2 - large_vol.origin_y) / vs))
    iy_end = int(round((py + fov_y / 2 - large_vol.origin_y) / vs))

    # Clip to volume bounds
    ix_start = max(0, ix_start)
    ix_end = min(n_x, ix_end)
    iy_start = max(0, iy_start)
    iy_end = min(n_y, iy_end)

    sub_imp = large_vol.impedance[:, iy_start:iy_end, ix_start:ix_end].copy()
    sub_ws = large_vol.wavespeed[:, iy_start:iy_end, ix_start:ix_end].copy()

    sub_origin_x = large_vol.origin_x + ix_start * vs
    sub_origin_y = large_vol.origin_y + iy_start * vs

    return VoxelVolume3D(
        impedance=sub_imp,
        wavespeed=sub_ws,
        voxel_size=vs,
        origin_z=large_vol.origin_z,
        origin_y=sub_origin_y,
        origin_x=sub_origin_x,
    )


# ── Scan at one position ─────────────────────────────────────────────

def scan_at_position(
    specimen_local: Specimen3D,
    shifted_vol: VoxelVolume3D,
    cfg: SimulationConfig,
    scan_plan: ScanPlanConfig,
    pos_dir: str,
    pos_index: int,
    px: float,
    py: float,
    tfm_z_start: float = 10e-3,
    tfm_z_end: Optional[float] = None,
    tfm_n_pixels: int = 800,
) -> None:
    """Run the rotational scan pipeline at one position."""
    print(f"\n{'#'*60}")
    print(f"  POSITION {pos_index}  —  "
          f"({px*1e3:+.1f}, {py*1e3:+.1f}) mm")
    print(f"{'#'*60}")

    scan_volume_3d(
        specimen_local,
        defects_3d=[],           # all defects embedded in voxel volume
        cfg=cfg,
        scan_plan=scan_plan,
        output_dir=pos_dir,
        voxel_volume=shifted_vol,
        tfm_z_start=tfm_z_start,
        tfm_z_end=tfm_z_end,
        tfm_n_pixels=tfm_n_pixels,
    )


# ── Defect serialisation for JSON ────────────────────────────────────

def _defect_to_dict(defect) -> dict:
    """Convert a defect object to a JSON-serialisable dict."""
    d = {'type': type(defect).__name__}
    for attr in ['center_z', 'center_x', 'center_y', 'radius',
                 'y_start', 'y_end', 'half_length', 'tilt_rad']:
        if hasattr(defect, attr):
            d[attr] = float(getattr(defect, attr))
    return d


# ── Top-level orchestrator ───────────────────────────────────────────

def generate_dataset(
    # Specimen
    width_total: float = 100e-3,
    depth_total: float = 60e-3,
    thickness: float = 50e-3,
    defects_3d: Optional[List] = None,
    material: MaterialProperties = ALUMINUM,
    # Scan grid
    n_positions_x: int = 3,
    n_positions_y: int = 2,
    overlap_fraction: float = 0.3,
    # Array
    num_elements: int = 64,
    element_pitch: float = 0.6e-3,
    frequency: float = 10e6,
    bandwidth: float = 0.6,
    snr_db: float = 35.0,
    # Scan plan
    n_scans: int = 32,
    theta_start: float = -np.pi / 2,
    theta_end: float = np.pi / 2,
    # Mode
    mode: str = '3d',
    # TFM
    tfm_z_start: float = 10e-3,
    tfm_z_end: Optional[float] = None,
    tfm_n_pixels: int = 800,
    # Grain structure
    mean_grain_size_m: float = 0.5e-3,
    impedance_variation: float = 0.025,
    wavespeed_variation: float = 0.005,
    voxel_fraction: float = 1 / 3,
    seed: int = 42,
    # Output
    output_root: Optional[str] = None,
    save_full_volume: bool = False,
    show_napari: bool = True,
) -> str:
    """
    Generate a multi-position synthetic scan dataset.

    Creates one large specimen, then scans at multiple overlapping
    positions. Each position gets its own B-scans and ground truth.
    In '3d' mode, inverse Radon reconstruction is also performed.

    Args:
        width_total:       Full specimen x-extent (m).
        depth_total:       Full specimen y-extent (m).
        thickness:         Specimen z-thickness (m).
        defects_3d:        3D defects to embed in the volume.
        material:          Material preset.
        n_positions_x:     Grid positions along x.
        n_positions_y:     Grid positions along y.
        overlap_fraction:  Fraction of aperture that overlaps (0-1).
        num_elements:      Array element count.
        element_pitch:     Element pitch (m).
        frequency:         Centre frequency (Hz).
        bandwidth:         Fractional bandwidth.
        n_scans:           Angular scans per position.
        theta_start:       Start angle (rad).
        theta_end:         End angle (rad).
        mode:              '2d' (B-scans only) or '3d' (+ reconstruction).
        tfm_z_start:       TFM start depth (m).
        tfm_z_end:         TFM end depth (m). Default: thickness - 5mm.
        tfm_n_pixels:      TFM pixel grid size.
        mean_grain_size_m: Grain diameter (m).
        impedance_variation: Per-grain Z spread.
        wavespeed_variation: Per-grain c_L spread.
        voxel_fraction:    Voxel size as fraction of wavelength.
        seed:              RNG seed.
        output_root:       Output directory. Auto-generated if None.
        save_full_volume:  Save the full large ground truth volume.

    Returns:
        Path to the dataset output directory.
    """
    assert mode in ('2d', '3d'), f"mode must be '2d' or '3d', got '{mode}'"

    # Output directory
    if output_root is None:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_root = os.path.join(
            os.path.dirname(__file__), 'output', f'dataset_{ts}',
        )
    os.makedirs(output_root, exist_ok=True)

    if tfm_z_end is None:
        tfm_z_end = thickness - 5e-3

    # 1. Create the large specimen + grain volume
    if defects_3d is None:
        defects_3d = []

    specimen, large_vol = create_large_specimen(
        width_total=width_total,
        depth_total=depth_total,
        thickness=thickness,
        defects_3d=defects_3d,
        frequency=frequency,
        mean_grain_size_m=mean_grain_size_m,
        impedance_variation=impedance_variation,
        wavespeed_variation=wavespeed_variation,
        voxel_fraction=voxel_fraction,
        material=material,
        seed=seed,
    )

    # 2. Compute array aperture and scan positions
    aperture = (num_elements - 1) * element_pitch
    positions, grid_info = compute_scan_positions(
        width_total, depth_total, aperture,
        n_positions_x, n_positions_y, overlap_fraction,
    )

    # 3. Build simulation config (local specimen sized to array FOV)
    local_width = aperture * 1.3     # 30% margin beyond aperture
    local_depth = aperture * 1.3
    scan_plan = ScanPlanConfig(
        n_scans=n_scans,
        theta_start=theta_start,
        theta_end=theta_end,
    )
    cfg = SimulationConfig(
        specimen=SpecimenConfig(thickness=thickness, width=local_width),
        array=ArrayConfig(
            num_elements=num_elements,
            element_pitch=element_pitch,
            frequency=frequency,
            bandwidth=bandwidth,
        ),
        scan_plan=scan_plan,
        max_bounces=2,
        mode_conversion=True,
    )
    cfg.acquisition.snr_db = snr_db
    specimen_local = Specimen3D(
        thickness=thickness, width=local_width, depth=local_depth,
    )

    # 4. Save dataset metadata
    meta = {
        'timestamp': datetime.now().isoformat(),
        'mode': mode,
        'specimen': {
            'width_total_m': width_total,
            'depth_total_m': depth_total,
            'thickness_m': thickness,
            'material': material.name,
        },
        'array': {
            'num_elements': num_elements,
            'element_pitch_m': element_pitch,
            'frequency_Hz': frequency,
            'bandwidth': bandwidth,
            'aperture_m': aperture,
        },
        'scan_grid': {
            **grid_info,
            'positions': [
                {'index': i, 'px_m': px, 'py_m': py}
                for i, (px, py) in enumerate(positions)
            ],
        },
        'scan_plan': {
            'n_scans': n_scans,
            'theta_start_rad': theta_start,
            'theta_end_rad': theta_end,
        },
        'tfm': {
            'z_start_m': tfm_z_start,
            'z_end_m': tfm_z_end,
            'n_pixels': tfm_n_pixels,
        },
        'grain': {
            'mean_grain_size_m': mean_grain_size_m,
            'impedance_variation': impedance_variation,
            'wavespeed_variation': wavespeed_variation,
            'voxel_fraction': voxel_fraction,
            'seed': seed,
        },
        'defects': [_defect_to_dict(d) for d in defects_3d],
    }

    meta_path = os.path.join(output_root, 'dataset_meta.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"\n  Metadata saved: {meta_path}")

    # 5. Optionally save the full ground truth volume
    if save_full_volume:
        gt_full_path = os.path.join(output_root, 'ground_truth_full.npz')
        save_ground_truth(large_vol, gt_full_path)

    # 6. Scan at each position
    cube_side = aperture / np.sqrt(2)  # inscribed square after cylinder crop
    fov = cube_side  # field of view per position (cube, not cylinder)

    for i, (px, py) in enumerate(positions):
        pos_dir = os.path.join(output_root, f'pos_{i:03d}')
        os.makedirs(pos_dir, exist_ok=True)

        # Origin-shifted volume view (zero-copy)
        shifted_vol = create_shifted_volume(large_vol, px, py)

        # Run rotational scan
        scan_at_position(
            specimen_local, shifted_vol, cfg, scan_plan,
            pos_dir, i, px, py,
            tfm_z_start=tfm_z_start,
            tfm_z_end=tfm_z_end,
            tfm_n_pixels=tfm_n_pixels,
        )

        # Extract and save ground truth sub-volume
        sub_vol = extract_ground_truth_subvolume(
            large_vol, px, py, fov, fov,
        )
        gt_path = os.path.join(pos_dir, 'ground_truth.npz')
        save_ground_truth(sub_vol, gt_path)

        # Save per-position metadata
        pos_meta = {
            'index': i, 'px_m': px, 'py_m': py,
            'fov_x_m': fov, 'fov_y_m': fov,
            'cube_side_m': float(cube_side),
        }

        # 3D reconstruction (optional)
        if mode == '3d':
            # Use the shifted volume for ground truth comparison
            recon_vol, metrics = reconstruct_and_compare(
                scan_dir=pos_dir,
                voxel_volume=shifted_vol,
                show_napari=False,
                save_figures=True,
                crop_to_cube=True,
            )
            # Save reconstruction in (z, x, y) order for Stitch3D compat
            recon_zxy = recon_vol.transpose(0, 2, 1)
            np.save(os.path.join(pos_dir, 'recon_volume_zxy.npy'), recon_zxy)

            pos_meta['metrics'] = {
                k: float(v) if isinstance(v, (float, np.floating)) else None
                for k, v in metrics.items()
                if not isinstance(v, np.ndarray)
            }

        pos_meta_path = os.path.join(pos_dir, 'position_meta.json')
        with open(pos_meta_path, 'w') as f:
            json.dump(pos_meta, f, indent=2)

    # 7. Summary
    print(f"\n{'#'*60}")
    print(f"  DATASET COMPLETE — {len(positions)} positions")
    print(f"  Mode: {mode}")
    print(f"  Output: {output_root}")
    print(f"{'#'*60}\n")

    # 8. Visualise all volumes in napari (skipped during sweeps)
    if mode == '3d' and show_napari:
        view_dataset_napari(output_root, layer='both')

    return output_root


# ── Napari visualisation of all positions ─────────────────────────────

def view_dataset_napari(dataset_dir: str, layer: str = 'reconstruction') -> None:
    """
    Open a napari viewer with all position volumes placed at their
    correct spatial offsets.

    Args:
        dataset_dir: Path to dataset output directory.
        layer:       What to show per position:
                     'reconstruction' — reconstructed volumes (recon_volume.npy)
                     'ground_truth'   — ground truth sub-volumes (ground_truth.npz)
                     'both'           — both layers per position
                     'overlay'        — grain structure (cyan) + signal (magenta) overlay
    """
    try:
        import napari
    except ImportError:
        print("napari not installed — skipping viewer")
        return

    # Load dataset metadata
    meta_path = os.path.join(dataset_dir, 'dataset_meta.json')
    with open(meta_path) as f:
        meta = json.load(f)

    positions = meta['scan_grid']['positions']
    aperture = meta['array']['aperture_m']
    tfm_z_start = meta['tfm']['z_start_m']
    tfm_z_end = meta['tfm']['z_end_m']

    # After cube cropping, effective FOV is aperture / sqrt(2)
    cube_side = meta['scan_grid'].get('cube_side_m', aperture / np.sqrt(2))

    viewer = napari.Viewer(title=f'Dataset — {len(positions)} positions')

    colours = ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow',
               'orange', 'pink', 'lime', 'teal']

    for pos_info in positions:
        i = pos_info['index']
        px = pos_info['px_m']
        py = pos_info['py_m']
        pos_dir = os.path.join(dataset_dir, f'pos_{i:03d}')

        if not os.path.isdir(pos_dir):
            continue

        show_recon = layer in ('reconstruction', 'both', 'overlay')
        show_gt = layer in ('ground_truth', 'both', 'overlay')

        # Compute translation offset in mm
        # The reconstruction is centered at (px, py) in world coords
        # napari translate: (z_offset, y_offset, x_offset) in mm
        z_offset_mm = tfm_z_start * 1e3
        x_offset_mm = (px - cube_side / 2) * 1e3
        y_offset_mm = (py - cube_side / 2) * 1e3

        if show_recon:
            recon_path = os.path.join(pos_dir, 'recon_volume.npy')
            if os.path.exists(recon_path):
                vol = np.load(recon_path)
                n_z_vol, n_y_vol, n_x_vol = vol.shape
                # z spans tfm_z_start..tfm_z_end, x/y spans cube_side
                dz_mm = (tfm_z_end - tfm_z_start) / max(n_z_vol - 1, 1) * 1e3
                dxy_mm = cube_side / max(n_x_vol - 1, 1) * 1e3
                recon_cmap = 'magenta' if layer == 'overlay' else 'hot'
                recon_opacity = 0.6 if layer == 'overlay' else 0.8
                # Normalise to [0, 1] for overlay mode
                if layer == 'overlay':
                    v_max = vol.max() if vol.max() > 0 else 1.0
                    vol = vol / v_max
                viewer.add_image(
                    vol,
                    name=f'Recon pos_{i:03d} ({px*1e3:+.0f},{py*1e3:+.0f})',
                    scale=(dz_mm, dxy_mm, dxy_mm),
                    translate=(z_offset_mm, y_offset_mm, x_offset_mm),
                    colormap=recon_cmap,
                    opacity=recon_opacity,
                    blending='additive',
                )

        if show_gt:
            gt_path = os.path.join(pos_dir, 'ground_truth.npz')
            if os.path.exists(gt_path):
                gt_vol = load_ground_truth_file(gt_path)
                # Convert impedance to contrast for visualisation
                bg_Z = float(np.mean(gt_vol.impedance))
                contrast = np.abs(
                    (gt_vol.impedance - bg_Z) / bg_Z
                ).astype(np.float32)

                gt_dz = gt_vol.voxel_size * 1e3
                gt_translate = (
                    gt_vol.origin_z * 1e3,
                    gt_vol.origin_y * 1e3,
                    gt_vol.origin_x * 1e3,
                )
                gt_cmap = 'cyan' if layer == 'overlay' else 'viridis'
                gt_opacity = 0.6 if layer == 'overlay' else 0.5
                viewer.add_image(
                    contrast,
                    name=f'GT pos_{i:03d} ({px*1e3:+.0f},{py*1e3:+.0f})',
                    scale=(gt_dz, gt_dz, gt_dz),
                    translate=gt_translate,
                    colormap=gt_cmap,
                    opacity=gt_opacity,
                    blending='additive',
                    visible=(layer in ('ground_truth', 'overlay')),
                )

    # Also load full ground truth if available
    gt_full_path = os.path.join(dataset_dir, 'ground_truth_full.npz')
    if os.path.exists(gt_full_path):
        gt_full = load_ground_truth_file(gt_full_path)
        bg_Z = float(np.mean(gt_full.impedance))
        contrast_full = np.abs(
            (gt_full.impedance - bg_Z) / bg_Z
        ).astype(np.float32)
        gt_dz = gt_full.voxel_size * 1e3
        viewer.add_image(
            contrast_full,
            name='Full ground truth',
            scale=(gt_dz, gt_dz, gt_dz),
            translate=(
                gt_full.origin_z * 1e3,
                gt_full.origin_y * 1e3,
                gt_full.origin_x * 1e3,
            ),
            colormap='gray',
            opacity=0.3,
            blending='additive',
            visible=False,
        )

    viewer.dims.axis_labels = ('z — depth (mm)', 'y (mm)', 'x (mm)')
    print(f"napari viewer open with {len(positions)} positions — "
          f"close the window to continue")
    napari.run()


# ── Main ──────────────────────────────────────────────────────────────

def main():
    """Generate a default dataset with defects for stitching validation."""

    # Defects spanning the large specimen
    defects = [
        # SphericalDefect(
        #     center_z=25e-3, center_x=10e-3, center_y=5e-3, radius=2e-3,
        # ),
        # CylindricalDefect(
        #     center_z=15e-3, center_x=-5e-3, radius=1e-3,
        #     y_start=-30e-3, y_end=30e-3,
        # ),
    ]

    generate_dataset(
        width_total=100e-3,
        depth_total=60e-3,
        thickness=50e-3,
        defects_3d=defects,
        n_positions_x=3,
        n_positions_y=0,
        overlap_fraction=0.3,
        num_elements=64,
        element_pitch=0.6e-3,
        frequency=10e6,
        n_scans=32,
        mode='3d',
        tfm_z_start=10e-3,
        tfm_n_pixels=400,       # lower resolution for faster generation
        save_full_volume=True,
    )


if __name__ == '__main__':
    main()
