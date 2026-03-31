"""
3D Volume Reconstruction from Rotational B-scans
=================================================

Reconstructs a 3D volume from the 2D TFM B-scan stack produced by
``scan_volume_3d()`` using slice-by-slice inverse Radon transform.

Optionally compares the reconstruction quantitatively against the
ground truth ``VoxelVolume3D`` and visualises both in napari.

Usage -- standalone:
    python reconstruct_3d.py

Usage -- as a library:
    from reconstruct_3d import reconstruct_and_compare
    volume, metrics = reconstruct_and_compare('output/scan_3d/',
                                               ground_truth_path='ground_truth.npz',
                                               show_napari=True)
"""

import os
import sys
import numpy as np
from typing import Optional

from scipy.ndimage import map_coordinates
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity as ssim

# Allow imports from parent directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Generic reconstruction functions (shared with experimental pipeline)
from Classes.Reconstruct3D import (
    load_bscans,
    db_to_linear,
    _apply_lateral_taper,
    build_sinograms,
    _subtract_angular_mean,
    reconstruct_volume,
    compute_reconstruction_coords,
    _circle_mask,
    _soft_circle_apodise,
    crop_cylinder_to_cube,
    view_reconstruction_napari,
    save_reconstruction_summary,
    reconstruct_scan,
)

# Synthetic-data-specific imports
from engine.voxel_volume import VoxelVolume3D
from engine.materials import ALUMINUM


# ── Ground truth I/O ──────────────────────────────────────────────────

def save_ground_truth(volume: VoxelVolume3D, path: str) -> None:
    """Save VoxelVolume3D to a compressed .npz file."""
    np.savez_compressed(
        path,
        impedance=volume.impedance,
        wavespeed=volume.wavespeed,
        voxel_size=np.float64(volume.voxel_size),
        origin_z=np.float64(volume.origin_z),
        origin_y=np.float64(volume.origin_y),
        origin_x=np.float64(volume.origin_x),
    )
    size_mb = os.path.getsize(path) / (1024 ** 2)
    print(f"Ground truth saved: {path} ({size_mb:.1f} MB)")


def load_ground_truth_file(path: str) -> VoxelVolume3D:
    """Load VoxelVolume3D from a .npz file."""
    data = np.load(path)
    vol = VoxelVolume3D(
        impedance=data['impedance'],
        wavespeed=data['wavespeed'],
        voxel_size=float(data['voxel_size']),
        origin_z=float(data['origin_z']),
        origin_y=float(data['origin_y']),
        origin_x=float(data['origin_x']),
    )
    print(f"Ground truth loaded: {path}, shape {vol.shape}")
    return vol


# ── Extract ground truth contrast ────────────────────────────────────

def extract_ground_truth_contrast(
    volume: VoxelVolume3D,
    z_coords: np.ndarray,
    y_coords: np.ndarray,
    x_coords: np.ndarray,
    background_Z: float,
) -> np.ndarray:
    """
    Resample ground truth impedance onto the reconstruction grid
    as normalised impedance contrast |dZ / Z0|.

    Args:
        volume:       Ground truth VoxelVolume3D
        z_coords:     (n_z,) depth sample points (m)
        y_coords:     (n_y,) y sample points (m)
        x_coords:     (n_x,) x sample points (m)
        background_Z: Background impedance Z0 (Pa*s/m)

    Returns:
        contrast: (n_z, n_y, n_x) float32
    """
    # Build 3D meshgrid
    ZZ, YY, XX = np.meshgrid(z_coords, y_coords, x_coords, indexing='ij')

    # Convert to fractional voxel indices
    iz = (ZZ - volume.origin_z) / volume.voxel_size
    iy = (YY - volume.origin_y) / volume.voxel_size
    ix = (XX - volume.origin_x) / volume.voxel_size

    coords = np.array([iz.ravel(), iy.ravel(), ix.ravel()])
    bg_Z = float(np.mean(volume.impedance))

    Z_sampled = map_coordinates(
        volume.impedance.astype(np.float64),
        coords,
        order=1,
        mode='constant',
        cval=bg_Z,
    ).reshape(ZZ.shape)

    contrast = np.abs((Z_sampled - background_Z) / background_Z).astype(np.float32)
    return contrast


# ── Quantitative comparison ──────────────────────────────────────────

def compare_volumes(
    recon: np.ndarray,
    ground_truth: np.ndarray,
    voxel_size_mm: Optional[tuple] = None,
) -> dict:
    """
    Quantitative comparison between reconstruction and ground truth.

    Both volumes must have the same shape (n_z, n_y, n_x).
    Both should be non-negative (linear scale).

    Returns dict with:
        ssim_mean, ssim_per_slice, nrmse, pearson_r,
        cnr_recon, cnr_gt, defect_centroid_error_mm
    """
    assert recon.shape == ground_truth.shape, (
        f"Shape mismatch: recon {recon.shape} vs gt {ground_truth.shape}")

    n_z, n_y, n_x = recon.shape
    mask = _circle_mask(n_y)  # (n_y, n_x) — same for all z

    # Normalise both to [0, 1] within the circle mask
    def _normalise(vol):
        masked = vol[:, mask]
        vmin, vmax = masked.min(), masked.max()
        if vmax - vmin < 1e-12:
            return vol * 0.0
        return (vol - vmin) / (vmax - vmin)

    r_norm = _normalise(recon)
    g_norm = _normalise(ground_truth)

    # --- SSIM per slice ---
    ssim_per_slice = np.zeros(n_z)
    for z in range(n_z):
        r_slice = r_norm[z] * mask
        g_slice = g_norm[z] * mask
        ssim_per_slice[z] = ssim(g_slice, r_slice, data_range=1.0)
    ssim_mean = float(np.mean(ssim_per_slice))

    # --- Normalised RMSE (within circle) ---
    diff = (r_norm[:, mask] - g_norm[:, mask])
    nrmse = float(np.sqrt(np.mean(diff ** 2)))

    # --- Pearson correlation ---
    r_flat = r_norm[:, mask].ravel()
    g_flat = g_norm[:, mask].ravel()
    corr, _ = pearsonr(r_flat, g_flat)

    # --- CNR (contrast-to-noise ratio) ---
    # Defect = voxels where ground truth contrast > 0.1
    # Background = everything else within circle
    gt_masked = ground_truth[:, mask]
    r_masked = recon[:, mask]
    defect_thresh = np.percentile(gt_masked, 95)

    gt_defect = gt_masked > defect_thresh
    gt_bg = ~gt_defect

    if gt_defect.sum() > 0 and gt_bg.sum() > 0:
        cnr_gt = float((gt_masked[gt_defect].mean() - gt_masked[gt_bg].mean())
                       / max(gt_masked[gt_bg].std(), 1e-12))
        cnr_recon = float((r_masked[gt_defect].mean() - r_masked[gt_bg].mean())
                          / max(r_masked[gt_bg].std(), 1e-12))
    else:
        cnr_gt = 0.0
        cnr_recon = 0.0

    # --- Defect centroid error ---
    defect_error_mm = float('nan')
    if voxel_size_mm is not None:
        dz, dy, dx = voxel_size_mm
        # Ground truth defect centroid (top 1% intensity)
        gt_thresh_high = np.percentile(ground_truth[:, mask], 99)
        recon_thresh_high = np.percentile(recon[:, mask], 99)

        gt_defect_mask = ground_truth > gt_thresh_high
        r_defect_mask = recon > recon_thresh_high

        if gt_defect_mask.sum() > 0 and r_defect_mask.sum() > 0:
            gt_coords = np.argwhere(gt_defect_mask).mean(axis=0)
            r_coords = np.argwhere(r_defect_mask).mean(axis=0)
            diff_mm = (gt_coords - r_coords) * np.array([dz, dy, dx])
            defect_error_mm = float(np.linalg.norm(diff_mm))

    metrics = {
        'ssim_mean': ssim_mean,
        'ssim_per_slice': ssim_per_slice,
        'nrmse': nrmse,
        'pearson_r': float(corr),
        'cnr_recon': cnr_recon,
        'cnr_gt': cnr_gt,
        'defect_centroid_error_mm': defect_error_mm,
    }
    return metrics


# ── Top-level pipeline ───────────────────────────────────────────────

def reconstruct_and_compare(
    scan_dir: str,
    voxel_volume: Optional[VoxelVolume3D] = None,
    ground_truth_path: Optional[str] = None,
    filter_name: str = 'hann',
    circle: bool = True,
    output_size: Optional[int] = None,
    background_Z: Optional[float] = None,
    show_napari: bool = False,
    save_figures: bool = True,
    output_dir: Optional[str] = None,
    crop_to_cube: bool = False,
) -> tuple:
    """
    Full reconstruction pipeline: load -> reconstruct -> compare -> visualise.

    Args:
        scan_dir:          Directory with bscan_*.npy and scan_meta.npy
        voxel_volume:      In-memory ground truth VoxelVolume3D
        ground_truth_path: Path to saved .npz ground truth file
        filter_name:       FBP filter for iradon
        circle:            Truncated projections flag
        output_size:       Reconstruction grid size (default: n_lateral)
        background_Z:      Background impedance (default: ALUMINUM Z_L)
        show_napari:       Open interactive napari viewer
        save_figures:      Save static comparison PNG
        output_dir:        Where to save outputs (default: scan_dir)

    Returns:
        (volume_recon, metrics): reconstructed volume and comparison dict
    """
    if output_dir is None:
        output_dir = scan_dir
    if background_Z is None:
        background_Z = ALUMINUM.density * ALUMINUM.c_L

    # 1. Load B-scans
    bscans_db, meta = load_bscans(scan_dir)

    # 2. Convert to linear amplitude
    data_fmt = meta.get('data_format', 'db')
    if data_fmt == 'linear_envelope':
        # Already linear (e.g. from fmc_to_npy.py) — use directly
        bscans_lin = bscans_db.copy()
    else:
        # dB-scale data (from png_to_npy.py or synthetic pipeline)
        bscans_lin = db_to_linear(bscans_db)
        vmin_db = meta.get('vmin_db', None)
        if vmin_db is not None:
            floor = np.float32(10.0 ** (vmin_db / 20.0))
            bscans_lin = np.maximum(bscans_lin - floor, 0.0)

    # Taper lateral edges and normalise
    bscans_lin = _apply_lateral_taper(bscans_lin, taper_fraction=0.1)
    global_max = bscans_lin.max()
    if global_max > 0:
        bscans_lin /= global_max

    # 3. Build sinograms and remove wall echoes (rotationally invariant component)
    sinograms = build_sinograms(bscans_lin)
    sinograms = _subtract_angular_mean(sinograms)

    # 4. Convert angles to degrees
    angles_deg = np.degrees(meta['angles_rad'])

    # 5. Reconstruct
    if output_size is None:
        output_size = sinograms.shape[1]
    volume_recon = reconstruct_volume(
        sinograms, angles_deg,
        filter_name=filter_name, circle=circle, output_size=output_size,
    )

    # 5b. Soft circular apodisation to remove ring artifact, then crop
    volume_recon = _soft_circle_apodise(volume_recon, rolloff_fraction=0.08)
    if crop_to_cube:
        volume_recon = crop_cylinder_to_cube(volume_recon)
        print(f"  Cropped to cube: {volume_recon.shape}")
        # Adjust output_size for coordinate computation
        output_size = volume_recon.shape[1]

    # Save reconstructed volume
    recon_path = os.path.join(output_dir, 'recon_volume.npy')
    np.save(recon_path, volume_recon)
    print(f"Reconstructed volume saved: {recon_path}")

    # 6. Ground truth comparison (optional)
    z_coords, y_coords, x_coords = compute_reconstruction_coords(
        meta, output_size)

    gt_contrast = None
    metrics = {}

    # Load ground truth
    gt_vol = None
    if voxel_volume is not None:
        gt_vol = voxel_volume
    elif ground_truth_path is not None and os.path.exists(ground_truth_path):
        gt_vol = load_ground_truth_file(ground_truth_path)

    if gt_vol is not None:
        print("Extracting ground truth contrast on reconstruction grid...")
        gt_contrast = extract_ground_truth_contrast(
            gt_vol, z_coords, y_coords, x_coords, background_Z)

        # Compute voxel size in mm for centroid error
        dz_mm = (z_coords[-1] - z_coords[0]) / max(len(z_coords) - 1, 1) * 1e3
        dy_mm = (y_coords[-1] - y_coords[0]) / max(len(y_coords) - 1, 1) * 1e3
        dx_mm = (x_coords[-1] - x_coords[0]) / max(len(x_coords) - 1, 1) * 1e3

        print("Computing comparison metrics...")
        metrics = compare_volumes(
            volume_recon, gt_contrast,
            voxel_size_mm=(dz_mm, dy_mm, dx_mm),
        )

        print(f"\n{'='*50}")
        print(f"  RECONSTRUCTION QUALITY METRICS")
        print(f"{'='*50}")
        print(f"  SSIM (mean):         {metrics['ssim_mean']:.4f}")
        print(f"  Normalised RMSE:     {metrics['nrmse']:.4f}")
        print(f"  Pearson correlation: {metrics['pearson_r']:.4f}")
        print(f"  CNR (reconstruction):{metrics['cnr_recon']:.2f}")
        print(f"  CNR (ground truth):  {metrics['cnr_gt']:.2f}")
        print(f"  Defect centroid err: {metrics['defect_centroid_error_mm']:.2f} mm")
        print(f"{'='*50}\n")

    # 7. Save reconstruction summary figure
    if save_figures:
        fig_path = os.path.join(output_dir, 'reconstruction_summary.png')
        save_reconstruction_summary(
            volume_recon, bscans_db, sinograms, meta, fig_path,
        )

    # 8. Napari viewer
    if show_napari:
        view_reconstruction_napari(
            volume_recon, gt_contrast,
            z_coords, y_coords, x_coords,
            metrics=metrics or None,
        )

    return volume_recon, metrics


# ── Main ──────────────────────────────────────────────────────────────

def main():
    """Reconstruct from the default output directory."""
    scan_dir = os.path.join(os.path.dirname(__file__), 'output', 'scan_3d')
    gt_path = os.path.join(scan_dir, 'ground_truth.npz')

    if not os.path.isdir(scan_dir):
        print(f"Scan directory not found: {scan_dir}")
        print("Run simulate.py first to generate B-scan data.")
        return

    volume, metrics = reconstruct_and_compare(
        scan_dir=scan_dir,
        ground_truth_path=gt_path if os.path.exists(gt_path) else None,
        show_napari=True,
        save_figures=True,
    )

    if metrics:
        print("\nReconstruction complete.")
    else:
        print("\nReconstruction complete (no ground truth for comparison).")


if __name__ == '__main__':
    main()
