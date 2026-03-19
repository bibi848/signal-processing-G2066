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
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Optional

from scipy.ndimage import map_coordinates
from scipy.stats import pearsonr
from skimage.transform import iradon
from skimage.metrics import structural_similarity as ssim

# Allow imports from parent directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from engine.voxel_volume import VoxelVolume3D
from engine.materials import ALUMINUM


# ── Load B-scans ─────────────────────────────────────────────────────

def load_bscans(scan_dir: str) -> tuple:
    """
    Load B-scan stack and metadata from a scan_volume_3d output directory.

    Returns:
        bscans_db: (n_scans, n_z, n_lateral) float32, dB scale
        meta:      dict with scan parameters
    """
    meta_path = os.path.join(scan_dir, 'scan_meta.npy')
    meta = np.load(meta_path, allow_pickle=True).item()

    # Backward compatibility: infer missing fields
    if 'tfm_z_start_m' not in meta:
        warnings.warn("scan_meta.npy missing tfm fields — using defaults. "
                       "Re-run scan_volume_3d() to fix.")
        meta.setdefault('tfm_z_start_m', 10e-3)
        meta.setdefault('tfm_z_end_m', meta['specimen_thickness_m'] - 5e-3)
        meta.setdefault('array_aperture_m', meta['specimen_width_m'] * 0.7)

    # Load all bscan_*.npy files (sorted by index)
    files = sorted(f for f in os.listdir(scan_dir)
                   if f.startswith('bscan_') and f.endswith('.npy'))
    if not files:
        raise FileNotFoundError(f"No bscan_*.npy files found in {scan_dir}")

    bscans = []
    for f in files:
        bscans.append(np.load(os.path.join(scan_dir, f)))
    bscans_db = np.stack(bscans, axis=0).astype(np.float32)

    if 'tfm_n_pixels' not in meta:
        meta['tfm_n_pixels'] = bscans_db.shape[1]

    assert bscans_db.shape[0] == meta['n_scans'], (
        f"Loaded {bscans_db.shape[0]} B-scans but metadata says {meta['n_scans']}")

    print(f"Loaded {bscans_db.shape[0]} B-scans, shape per frame: "
          f"{bscans_db.shape[1]}x{bscans_db.shape[2]}")
    return bscans_db, meta


# ── dB to linear ─────────────────────────────────────────────────────

def db_to_linear(bscans_db: np.ndarray) -> np.ndarray:
    """Convert dB-scale TFM images to linear amplitude.

    Radon transform assumes linear superposition, so reconstruction
    must operate on linear data.
    """
    return np.float32(10.0 ** (bscans_db / 20.0))


# ── Lateral taper ────────────────────────────────────────────────────

def _apply_lateral_taper(
    bscans: np.ndarray,
    taper_fraction: float = 0.1,
) -> np.ndarray:
    """
    Apply a Tukey (tapered cosine) window along the lateral axis
    of each B-scan so values decay smoothly to zero at the edges.

    This prevents the hard lateral boundary from creating bright-edge
    artifacts in the FBP reconstruction.

    Args:
        bscans:         (n_scans, n_z, n_lateral)
        taper_fraction: Fraction of each edge that is tapered (0-0.5).

    Returns:
        Tapered copy, same shape.
    """
    from scipy.signal.windows import tukey
    n_lateral = bscans.shape[2]
    window = tukey(n_lateral, alpha=2 * taper_fraction).astype(np.float32)
    # Broadcast: (1, 1, n_lateral) over (n_scans, n_z, n_lateral)
    return bscans * window[np.newaxis, np.newaxis, :]


# ── Build sinograms ──────────────────────────────────────────────────

def build_sinograms(bscans_linear: np.ndarray) -> np.ndarray:
    """
    Rearrange B-scan stack into sinograms for iradon.

    Args:
        bscans_linear: (n_scans, n_z, n_lateral) linear amplitude

    Returns:
        sinograms: (n_z, n_lateral, n_scans) -- one sinogram per depth row
    """
    return np.transpose(bscans_linear, (1, 2, 0))


# ── Angular mean subtraction ─────────────────────────────────────────

def _subtract_angular_mean(sinograms: np.ndarray) -> np.ndarray:
    """
    Remove the rotationally invariant component from each sinogram.

    Wall echoes (backwall/frontwall reflections) appear identically at
    every rotation angle. In the sinogram, this means each detector row
    has a constant offset across all angles. Subtracting the angular
    mean removes this DC component, eliminating concentric ring artifacts
    in the reconstruction while preserving angle-dependent grain structure.

    Args:
        sinograms: (n_z, n_detectors, n_angles)

    Returns:
        Sinograms with angular mean subtracted, same shape.
    """
    # Mean across angles for each (depth, detector) pair
    angular_mean = sinograms.mean(axis=2, keepdims=True)
    result = sinograms - angular_mean
    # Clip negative values — amplitude should be non-negative
    return np.maximum(result, 0.0).astype(np.float32)


# ── Inverse Radon reconstruction ─────────────────────────────────────

def reconstruct_volume(
    sinograms: np.ndarray,
    angles_deg: np.ndarray,
    filter_name: str = 'hann',
    circle: bool = True,
    output_size: Optional[int] = None,
) -> np.ndarray:
    """
    Inverse Radon reconstruction, slice by slice.

    Args:
        sinograms:   (n_z, n_detectors, n_angles)
        angles_deg:  (n_angles,) projection angles in degrees
        filter_name: FBP filter ('ramp', 'shepp-logan', 'hamming', 'hann')
        circle:      If True, assume projections cover only the inscribed circle
        output_size: Output image side length. Default: n_detectors.

    Returns:
        volume: (n_z, output_size, output_size) float32
    """
    n_z, n_det, n_ang = sinograms.shape
    if output_size is None:
        output_size = n_det

    volume = np.zeros((n_z, output_size, output_size), dtype=np.float32)

    print(f"Reconstructing {n_z} depth slices "
          f"({n_det} detectors, {n_ang} angles, "
          f"filter='{filter_name}', circle={circle})...")

    for z in range(n_z):
        sinogram = sinograms[z]  # (n_detectors, n_angles)
        recon = iradon(
            sinogram,
            theta=angles_deg,
            filter_name=filter_name,
            circle=circle,
            output_size=output_size,
        )
        volume[z] = recon.astype(np.float32)

    # Clip negative values (reconstruction artifacts)
    volume = np.maximum(volume, 0.0)

    print(f"  Reconstructed volume shape: {volume.shape}")
    return volume


# ── Coordinate computation ────────────────────────────────────────────

def compute_reconstruction_coords(
    meta: dict,
    output_size: int,
) -> tuple:
    """
    Physical coordinates for the reconstructed volume.

    If output_size is smaller than tfm_n_pixels (e.g. after cube cropping),
    the lateral extent is scaled proportionally.

    Returns:
        (z_coords, y_coords, x_coords) -- 1D arrays in metres
    """
    full_size = meta.get('tfm_n_pixels', output_size)
    half_ap = meta['array_aperture_m'] / 2.0
    # Scale lateral extent if cropped
    half_extent = half_ap * output_size / full_size
    z_coords = np.linspace(meta['tfm_z_start_m'], meta['tfm_z_end_m'],
                           meta['tfm_n_pixels'])
    x_coords = np.linspace(-half_extent, half_extent, output_size)
    y_coords = np.linspace(-half_extent, half_extent, output_size)
    return z_coords, y_coords, x_coords


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


# ── Circle mask & cube cropping ───────────────────────────────────────

def _circle_mask(size: int) -> np.ndarray:
    """Boolean mask for the inscribed circle of a square grid."""
    center = (size - 1) / 2.0
    y, x = np.ogrid[:size, :size]
    r_sq = (y - center) ** 2 + (x - center) ** 2
    return r_sq <= (size / 2.0) ** 2


def _soft_circle_apodise(volume: np.ndarray, rolloff_fraction: float = 0.05) -> np.ndarray:
    """
    Apply a soft circular apodisation to each x-y slice, replacing
    iradon's hard circular mask with a smooth cosine rolloff.

    This eliminates the bright ring artifact at the circle boundary.

    Args:
        volume:           (n_z, n_y, n_x)
        rolloff_fraction: Fraction of radius over which the taper falls to 0.

    Returns:
        Apodised volume, same shape.
    """
    n_z, n_y, n_x = volume.shape
    center_y = (n_y - 1) / 2.0
    center_x = (n_x - 1) / 2.0
    radius = min(n_y, n_x) / 2.0

    y, x = np.ogrid[:n_y, :n_x]
    r = np.sqrt((y - center_y) ** 2 + (x - center_x) ** 2)

    # Cosine rolloff from (1-rolloff)*radius to radius
    inner = radius * (1.0 - rolloff_fraction)
    mask = np.ones((n_y, n_x), dtype=np.float32)
    transition = (r > inner) & (r <= radius)
    mask[transition] = (0.5 * (1.0 + np.cos(
        np.pi * (r[transition] - inner) / (radius - inner)
    ))).astype(np.float32)
    mask[r > radius] = 0.0

    return volume * mask[np.newaxis, :, :]


def crop_cylinder_to_cube(volume: np.ndarray) -> np.ndarray:
    """
    Crop a cylindrical reconstruction to the largest inscribed cube.

    The inverse Radon with circle=True produces a cylinder (circle in x-y,
    full extent in z). The inscribed square has side = diameter / sqrt(2).
    This crops each x-y slice to that square, giving a cuboidal volume.

    Args:
        volume: (n_z, n_y, n_x) — cylindrical reconstruction

    Returns:
        (n_z, side, side) — cropped cube
    """
    n_z, n_y, n_x = volume.shape
    diameter = min(n_y, n_x)
    side = int(diameter / np.sqrt(2))

    y_start = (n_y - side) // 2
    x_start = (n_x - side) // 2

    cropped = volume[:, y_start:y_start + side, x_start:x_start + side]
    return cropped


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


# ── Napari viewer ─────────────────────────────────────────────────────

def view_reconstruction_napari(
    recon: np.ndarray,
    ground_truth: Optional[np.ndarray],
    z_coords: np.ndarray,
    y_coords: np.ndarray,
    x_coords: np.ndarray,
    metrics: Optional[dict] = None,
) -> None:
    """Open napari viewer with reconstruction and ground truth side by side."""
    try:
        import napari
    except ImportError:
        print("napari not installed -- skipping interactive viewer")
        return

    n_z = len(z_coords)
    n_y = len(y_coords)
    dz = (z_coords[-1] - z_coords[0]) / max(n_z - 1, 1) * 1e3  # mm
    dy = (y_coords[-1] - y_coords[0]) / max(n_y - 1, 1) * 1e3
    dx = (x_coords[-1] - x_coords[0]) / max(len(x_coords) - 1, 1) * 1e3
    scale = (dz, dy, dx)

    title = 'NDT 3D Reconstruction'
    if metrics is not None:
        title += f'  |  SSIM={metrics["ssim_mean"]:.3f}  r={metrics["pearson_r"]:.3f}'

    viewer = napari.Viewer(title=title)

    viewer.add_image(
        recon, name='Reconstruction',
        scale=scale, colormap='hot', opacity=0.9,
    )

    if ground_truth is not None:
        viewer.add_image(
            ground_truth, name='Ground truth |dZ/Z0|',
            scale=scale, colormap='hot', opacity=0.9, visible=False,
        )
        # Difference layer
        r_max = recon.max() if recon.max() > 0 else 1.0
        g_max = ground_truth.max() if ground_truth.max() > 0 else 1.0
        diff = np.abs(recon / r_max - ground_truth / g_max)
        viewer.add_image(
            diff, name='|Difference|',
            scale=scale, colormap='turbo', opacity=0.7, visible=False,
        )
        # Overlay: grain structure (cyan) + signal (magenta) via additive blending
        viewer.add_image(
            ground_truth / g_max, name='Overlay — grain structure',
            scale=scale, colormap='cyan', opacity=0.6,
            blending='additive', visible=False,
        )
        viewer.add_image(
            recon / r_max, name='Overlay — signal',
            scale=scale, colormap='magenta', opacity=0.6,
            blending='additive', visible=False,
        )

    viewer.dims.axis_labels = ('z - depth (mm)', 'y (mm)', 'x (mm)')
    print("napari viewer open -- close the window to continue")
    napari.run()


# ── Static comparison figure ─────────────────────────────────────────

def save_reconstruction_summary(
    recon: np.ndarray,
    bscans_db: np.ndarray,
    sinograms: np.ndarray,
    meta: dict,
    output_path: str,
) -> None:
    """
    Save a reconstruction summary figure showing cross-sections,
    example B-scans, and sinogram diagnostics.

    Args:
        recon:      (n_z, n_y, n_x) reconstructed volume
        bscans_db:  (n_scans, n_z, n_lateral) B-scans in dB
        sinograms:  (n_z, n_detectors, n_angles) sinogram stack
        meta:       scan metadata dict
        output_path: where to save the PNG
    """
    n_z, n_y, n_x = recon.shape
    n_scans = bscans_db.shape[0]
    angles = meta.get('angles_rad', np.linspace(-np.pi/2, np.pi/2, n_scans))

    fig, axes = plt.subplots(3, 3, figsize=(14, 12))
    r_vmax = np.percentile(recon[recon > 0], 99) if recon.max() > 0 else 1.0

    # ── Row 1: Reconstruction cross-sections at 25%, 50%, 75% depth ──
    fractions = [0.25, 0.50, 0.75]
    for i, frac in enumerate(fractions):
        z_idx = min(int(frac * n_z), n_z - 1)
        axes[0, i].imshow(recon[z_idx], cmap='hot', vmin=0, vmax=r_vmax,
                          origin='lower', aspect='equal')
        axes[0, i].set_title(f'Depth slice z={z_idx}/{n_z} ({frac:.0%})')
        axes[0, i].axis('off')

    # ── Row 2: B-scans at 3 rotation angles ──
    bscan_indices = [0, n_scans // 2, n_scans - 1]
    b_vmin = np.percentile(bscans_db, 5)
    b_vmax = np.percentile(bscans_db, 99)
    for i, idx in enumerate(bscan_indices):
        angle_deg = np.degrees(angles[idx])
        axes[1, i].imshow(bscans_db[idx], cmap='hot', vmin=b_vmin, vmax=b_vmax,
                          origin='lower', aspect='auto')
        axes[1, i].set_title(f'B-scan #{idx} ({angle_deg:+.0f}\u00b0)')
        axes[1, i].set_xlabel('Lateral (px)')
        axes[1, i].set_ylabel('Depth (px)')

    # ── Row 3: Sinogram at mid-depth + amplitude profile + stats ──
    mid_z = n_z // 2
    sino_slice = sinograms[mid_z]  # (n_detectors, n_angles)
    s_vmax = np.percentile(sino_slice, 99) if sino_slice.max() > 0 else 1.0
    axes[2, 0].imshow(sino_slice, cmap='hot', vmin=0, vmax=s_vmax,
                      origin='lower', aspect='auto')
    axes[2, 0].set_title(f'Sinogram at z={mid_z}')
    axes[2, 0].set_xlabel('Angle index')
    axes[2, 0].set_ylabel('Detector (px)')

    # Mean amplitude per depth slice
    mean_per_z = np.mean(recon, axis=(1, 2))
    axes[2, 1].plot(mean_per_z, 'b-', linewidth=1)
    axes[2, 1].set_xlabel('Depth slice')
    axes[2, 1].set_ylabel('Mean amplitude')
    axes[2, 1].set_title('Signal vs depth')
    axes[2, 1].grid(True, alpha=0.3)

    # Stats text
    axes[2, 2].axis('off')
    dynamic_range = 20 * np.log10(recon.max() / (recon[recon > 0].min() + 1e-30)) \
        if recon.max() > 0 else 0
    stats = [
        f'Volume shape:     {recon.shape}',
        f'N scans:          {n_scans}',
        f'Angular range:    {np.degrees(angles[0]):+.0f}\u00b0 to '
        f'{np.degrees(angles[-1]):+.0f}\u00b0',
        f'Mean amplitude:   {recon.mean():.4f}',
        f'Max amplitude:    {recon.max():.4f}',
        f'Dynamic range:    {dynamic_range:.1f} dB',
        f'Non-zero voxels:  {(recon > 0).sum() / recon.size:.1%}',
    ]
    axes[2, 2].text(0.05, 0.5, '\n'.join(stats),
                    transform=axes[2, 2].transAxes,
                    fontsize=10, family='monospace',
                    verticalalignment='center')
    axes[2, 2].set_title('Reconstruction stats')

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Reconstruction summary saved: {output_path}")


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

    # 2. Convert to linear, taper lateral edges, re-normalise globally
    bscans_lin = db_to_linear(bscans_db)
    bscans_lin = _apply_lateral_taper(bscans_lin, taper_fraction=0.1)
    global_max = bscans_lin.max()
    if global_max > 0:
        bscans_lin /= global_max

    # 3. Build sinograms and remove rotationally invariant component (wall echoes)
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
