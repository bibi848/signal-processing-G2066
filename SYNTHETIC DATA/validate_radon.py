"""
Radon Reconstruction Validation
================================

Side-by-side comparison:
  A) Forward Radon on ground truth xy slices → inverse Radon → reconstruct
     (proves the Radon math is correct)
  B) B-scan sinograms → inverse Radon → reconstruct
     (the actual pipeline — shows where it breaks)

Usage:
    python validate_radon.py
    python validate_radon.py --scan-dir output/radon_tests/compound_test
    python validate_radon.py --scan-dir output/radon_tests/radon_test --filter shepp-logan
"""

import os
import sys
import argparse
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from skimage.transform import radon, iradon
from skimage.metrics import structural_similarity as ssim
from scipy.ndimage import map_coordinates

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from Classes.Reconstruct3D import (
    load_bscans,
    db_to_linear,
    _apply_lateral_taper,
    build_sinograms,
    _subtract_angular_mean,
    reconstruct_volume,
    reconstruct_volume_polar,
    _soft_circle_apodise,
    _circle_mask,
    compute_reconstruction_coords,
)
from engine.voxel_volume import VoxelVolume3D
from engine.materials import ALUMINUM


# ── Ground truth loading ─────────────────────────────────────────────

def load_ground_truth(path: str) -> VoxelVolume3D:
    data = np.load(path)
    return VoxelVolume3D(
        impedance=data['impedance'],
        wavespeed=data['wavespeed'],
        voxel_size=float(data['voxel_size']),
        origin_z=float(data['origin_z']),
        origin_y=float(data['origin_y']),
        origin_x=float(data['origin_x']),
    )


def extract_ground_truth_contrast(
    volume: VoxelVolume3D,
    z_coords: np.ndarray,
    y_coords: np.ndarray,
    x_coords: np.ndarray,
    background_Z: float,
) -> np.ndarray:
    """Resample ground truth onto reconstruction grid as |dZ/Z0|."""
    ZZ, YY, XX = np.meshgrid(z_coords, y_coords, x_coords, indexing='ij')
    iz = (ZZ - volume.origin_z) / volume.voxel_size
    iy = (YY - volume.origin_y) / volume.voxel_size
    ix = (XX - volume.origin_x) / volume.voxel_size
    coords = np.array([iz.ravel(), iy.ravel(), ix.ravel()])
    bg_Z = float(np.mean(volume.impedance))
    Z_sampled = map_coordinates(
        volume.impedance.astype(np.float64), coords,
        order=1, mode='constant', cval=bg_Z,
    ).reshape(ZZ.shape)
    return np.abs((Z_sampled - background_Z) / background_Z).astype(np.float32)


# ── Main validation ──────────────────────────────────────────────────

def validate(
    scan_dir: str,
    filter_name: str = 'shepp-logan',
    output_dir: str = None,
    show_napari: bool = False,
):
    if output_dir is None:
        output_dir = scan_dir

    background_Z = ALUMINUM.density * ALUMINUM.c_L

    # ── 1. Load B-scans and metadata ──
    print("=" * 60)
    print("  STEP 1: Loading data")
    print("=" * 60)
    meta = np.load(os.path.join(scan_dir, 'scan_meta.npy'),
                   allow_pickle=True).item()
    angles_rad = meta['angles_rad']
    angles_deg = np.degrees(angles_rad)
    n_scans = meta['n_scans']

    # Load only non-complex B-scans (bscan_NNNN.npy, not bscan_complex_NNNN.npy)
    import re
    files = sorted(
        f for f in os.listdir(scan_dir)
        if re.match(r'^bscan_\d{4}\.npy$', f)
    )
    bscans_db = np.stack(
        [np.load(os.path.join(scan_dir, f)).real.astype(np.float32)
         for f in files[:n_scans]],
        axis=0,
    )
    n_scans, n_z, n_lateral = bscans_db.shape
    print(f"  B-scans: {n_scans} angles, {n_z} depth x {n_lateral} lateral")
    print(f"  Angles: {angles_deg[0]:.1f}° to {angles_deg[-1]:.1f}°")

    # ── 2. Load ground truth and sample on reconstruction grid ──
    print("\n" + "=" * 60)
    print("  STEP 2: Extracting ground truth on reconstruction grid")
    print("=" * 60)
    gt_path = os.path.join(scan_dir, 'ground_truth.npz')
    gt_vol = load_ground_truth(gt_path)
    print(f"  Ground truth shape: {gt_vol.shape}")

    output_size = n_lateral
    z_coords, y_coords, x_coords = compute_reconstruction_coords(meta, output_size)
    gt_contrast = extract_ground_truth_contrast(
        gt_vol, z_coords, y_coords, x_coords, background_Z)
    print(f"  Contrast volume shape: {gt_contrast.shape}")
    print(f"  Contrast range: [{gt_contrast.min():.6f}, {gt_contrast.max():.6f}]")

    # ── 3A. ROUNDTRIP: forward Radon on ground truth → inverse Radon ──
    print("\n" + "=" * 60)
    print("  STEP 3A: Forward Radon on ground truth xy slices")
    print("=" * 60)
    n_z_gt = gt_contrast.shape[0]

    # Forward Radon: for each depth z, radon(xy_slice) → sinogram
    sinograms_gt = np.zeros((n_z_gt, gt_contrast.shape[1], len(angles_deg)),
                            dtype=np.float32)
    for z in range(n_z_gt):
        sinograms_gt[z] = radon(
            gt_contrast[z], theta=angles_deg, circle=True,
        ).astype(np.float32)
    print(f"  Forward Radon sinograms: {sinograms_gt.shape}")

    # Inverse Radon on true sinograms
    print("  Inverse Radon on true sinograms...")
    recon_roundtrip = np.zeros_like(gt_contrast)
    for z in range(n_z_gt):
        recon_roundtrip[z] = np.abs(iradon(
            sinograms_gt[z], theta=angles_deg,
            filter_name=filter_name, circle=True, output_size=output_size,
        )).astype(np.float32)
    recon_roundtrip = _soft_circle_apodise(recon_roundtrip, rolloff_fraction=0.08)

    # ── 3B. B-SCAN PIPELINE: build sinograms from B-scans → inverse Radon ──
    print("\n" + "=" * 60)
    print("  STEP 3B: B-scan sinograms → inverse Radon")
    print("=" * 60)

    # Preprocess B-scans (same as current pipeline)
    data_fmt = meta.get('data_format', 'db')
    if data_fmt == 'linear_envelope':
        bscans_lin = bscans_db.copy()
    else:
        bscans_lin = db_to_linear(bscans_db)
        vmin_db = meta.get('vmin_db', None)
        if vmin_db is not None:
            floor = np.float32(10.0 ** (vmin_db / 20.0))
            bscans_lin = np.maximum(bscans_lin - floor, 0.0)

    bscans_lin = _apply_lateral_taper(bscans_lin, taper_fraction=0.1)
    global_max = bscans_lin.max()
    if global_max > 0:
        bscans_lin /= global_max

    sinograms_bscan = build_sinograms(bscans_lin)
    sinograms_bscan = _subtract_angular_mean(sinograms_bscan)
    print(f"  B-scan sinograms: {sinograms_bscan.shape}")

    # Inverse Radon on B-scan sinograms
    recon_bscan_iradon = reconstruct_volume(
        sinograms_bscan, angles_deg,
        filter_name=filter_name, circle=True, output_size=output_size,
    )
    recon_bscan_iradon = _soft_circle_apodise(recon_bscan_iradon, rolloff_fraction=0.08)

    # ── 3C. B-SCAN PIPELINE: polar gridding (correct method) ──
    # Use same sinograms with angular mean subtraction — removes wall echoes
    print("\n" + "=" * 60)
    print("  STEP 3C: B-scan sinograms → polar gridding")
    print("=" * 60)
    recon_bscan_polar = reconstruct_volume_polar(
        sinograms_bscan, angles_rad, output_size=output_size,
    )
    recon_bscan_polar = _soft_circle_apodise(recon_bscan_polar, rolloff_fraction=0.08)

    # ── 4. Metrics ──
    print("\n" + "=" * 60)
    print("  STEP 4: Comparison metrics")
    print("=" * 60)
    mask = _circle_mask(output_size)

    def compute_ssim(recon, ref, m):
        """Mean SSIM within circle mask."""
        def _norm(v, m):
            vals = v[:, m]
            vmin, vmax = vals.min(), vals.max()
            if vmax - vmin < 1e-12:
                return v * 0.0
            return (v - vmin) / (vmax - vmin)
        rn = _norm(recon, m)
        gn = _norm(ref, m)
        vals = [ssim(gn[z] * m, rn[z] * m, data_range=1.0)
                for z in range(rn.shape[0])]
        return float(np.mean(vals))

    ssim_roundtrip = compute_ssim(recon_roundtrip, gt_contrast, mask)
    ssim_iradon = compute_ssim(recon_bscan_iradon, gt_contrast, mask)
    ssim_polar = compute_ssim(recon_bscan_polar, gt_contrast, mask)

    print(f"  {'Method':<30s} {'SSIM':>8s}")
    print(f"  {'-'*30} {'-'*8}")
    print(f"  {'Radon roundtrip (ground truth)':<30s} {ssim_roundtrip:>8.4f}")
    print(f"  {'B-scan → iradon':<30s} {ssim_iradon:>8.4f}")
    print(f"  {'B-scan → polar gridding':<30s} {ssim_polar:>8.4f}")

    # ── 5. Diagnostic figure ──
    print("\n" + "=" * 60)
    print("  STEP 5: Saving diagnostic figure")
    print("=" * 60)

    mid_z = n_z_gt // 2

    fig, axes = plt.subplots(3, 4, figsize=(20, 14))

    # Row 1: xy slices at mid-depth — ground truth + 3 methods
    gt_vmax = np.percentile(gt_contrast[mid_z], 99) if gt_contrast[mid_z].max() > 0 else 1
    axes[0, 0].imshow(gt_contrast[mid_z], cmap='hot', vmin=0, vmax=gt_vmax)
    axes[0, 0].set_title(f'Ground truth (z={mid_z})')
    axes[0, 0].axis('off')

    rt_vmax = np.percentile(recon_roundtrip[mid_z], 99) if recon_roundtrip[mid_z].max() > 0 else 1
    axes[0, 1].imshow(recon_roundtrip[mid_z], cmap='hot', vmin=0, vmax=rt_vmax)
    axes[0, 1].set_title(f'Radon roundtrip (SSIM={ssim_roundtrip:.3f})')
    axes[0, 1].axis('off')

    ir_vmax = np.percentile(recon_bscan_iradon[mid_z], 99) if recon_bscan_iradon[mid_z].max() > 0 else 1
    axes[0, 2].imshow(recon_bscan_iradon[mid_z], cmap='hot', vmin=0, vmax=ir_vmax)
    axes[0, 2].set_title(f'B-scan iradon (SSIM={ssim_iradon:.3f})')
    axes[0, 2].axis('off')

    po_vmax = np.percentile(recon_bscan_polar[mid_z], 99) if recon_bscan_polar[mid_z].max() > 0 else 1
    axes[0, 3].imshow(recon_bscan_polar[mid_z], cmap='hot', vmin=0, vmax=po_vmax)
    axes[0, 3].set_title(f'B-scan polar (SSIM={ssim_polar:.3f})')
    axes[0, 3].axis('off')

    # Row 2: Sinograms + sinogram difference
    sino_gt_mid = sinograms_gt[mid_z]
    sino_bs_mid = sinograms_bscan[mid_z]

    axes[1, 0].imshow(sino_gt_mid, cmap='gray', aspect='auto',
                      extent=[angles_deg[0], angles_deg[-1],
                              sino_gt_mid.shape[0], 0])
    axes[1, 0].set_title('True Radon sinogram')
    axes[1, 0].set_xlabel('Angle (deg)')
    axes[1, 0].set_ylabel('Detector position')

    axes[1, 1].imshow(sino_bs_mid, cmap='gray', aspect='auto',
                      extent=[angles_deg[0], angles_deg[-1],
                              sino_bs_mid.shape[0], 0])
    axes[1, 1].set_title('B-scan sinogram')
    axes[1, 1].set_xlabel('Angle (deg)')
    axes[1, 1].set_ylabel('Lateral position')

    # Sinogram difference (normalised)
    sg_vmax = np.percentile(np.abs(sino_gt_mid), 99) if np.abs(sino_gt_mid).max() > 0 else 1
    sb_vmax = np.percentile(np.abs(sino_bs_mid), 99) if np.abs(sino_bs_mid).max() > 0 else 1
    sg_norm = sino_gt_mid / sg_vmax if sg_vmax > 0 else sino_gt_mid
    sb_norm = sino_bs_mid / sb_vmax if sb_vmax > 0 else sino_bs_mid
    min_rows = min(sg_norm.shape[0], sb_norm.shape[0])
    min_cols = min(sg_norm.shape[1], sb_norm.shape[1])
    diff = np.abs(sg_norm[:min_rows, :min_cols] - sb_norm[:min_rows, :min_cols])
    axes[1, 2].imshow(diff, cmap='hot', aspect='auto',
                      extent=[angles_deg[0], angles_deg[-1], min_rows, 0])
    axes[1, 2].set_title('|Sinogram difference|')
    axes[1, 2].set_xlabel('Angle (deg)')
    axes[1, 2].set_ylabel('Position')

    axes[1, 3].axis('off')
    axes[1, 3].text(0.1, 0.5,
                    f"SSIM Comparison\n"
                    f"{'─'*25}\n"
                    f"Radon roundtrip: {ssim_roundtrip:.4f}\n"
                    f"B-scan → iradon: {ssim_iradon:.4f}\n"
                    f"B-scan → polar:  {ssim_polar:.4f}\n\n"
                    f"Filter: {filter_name}\n"
                    f"Angles: {n_scans}",
                    transform=axes[1, 3].transAxes,
                    fontsize=12, family='monospace',
                    verticalalignment='center')

    # Row 3: xz cross-sections through mid-y
    mid_y = output_size // 2
    axes[2, 0].imshow(gt_contrast[:, mid_y, :], cmap='hot', aspect='auto')
    axes[2, 0].set_title('Ground truth (x-z, mid-y)')
    axes[2, 0].set_ylabel('Depth z')

    axes[2, 1].imshow(recon_roundtrip[:, mid_y, :], cmap='hot', aspect='auto')
    axes[2, 1].set_title('Radon roundtrip (x-z)')
    axes[2, 1].set_ylabel('Depth z')

    axes[2, 2].imshow(recon_bscan_iradon[:, mid_y, :], cmap='hot', aspect='auto')
    axes[2, 2].set_title('B-scan iradon (x-z)')
    axes[2, 2].set_ylabel('Depth z')

    axes[2, 3].imshow(recon_bscan_polar[:, mid_y, :], cmap='hot', aspect='auto')
    axes[2, 3].set_title('B-scan polar (x-z)')
    axes[2, 3].set_ylabel('Depth z')

    fig.suptitle(f'Radon Validation — {n_scans} angles, filter: {filter_name}\n'
                 f'Roundtrip={ssim_roundtrip:.4f}  |  '
                 f'iradon={ssim_iradon:.4f}  |  '
                 f'polar={ssim_polar:.4f}',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()

    fig_path = os.path.join(output_dir, 'radon_validation.png')
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Figure saved: {fig_path}")

    results = {
        'ssim_roundtrip': ssim_roundtrip,
        'ssim_iradon': ssim_iradon,
        'ssim_polar': ssim_polar,
        'recon_roundtrip': recon_roundtrip,
        'recon_bscan_iradon': recon_bscan_iradon,
        'recon_bscan_polar': recon_bscan_polar,
        'gt_contrast': gt_contrast,
        'sinograms_gt': sinograms_gt,
        'sinograms_bscan': sinograms_bscan,
    }

    # ── 6. Napari viewer ──
    if show_napari:
        print("\n" + "=" * 60)
        print("  STEP 6: Opening napari viewer")
        print("=" * 60)
        try:
            import napari

            z_coords, y_coords, x_coords = compute_reconstruction_coords(
                meta, output_size)
            n_z_r = len(z_coords)
            n_y_r = len(y_coords)
            dz = (z_coords[-1] - z_coords[0]) / max(n_z_r - 1, 1) * 1e3
            dy = (y_coords[-1] - y_coords[0]) / max(n_y_r - 1, 1) * 1e3
            dx = (x_coords[-1] - x_coords[0]) / max(len(x_coords) - 1, 1) * 1e3
            scale = (dz, dy, dx)

            viewer = napari.Viewer(
                title=f'Radon Validation  |  roundtrip={ssim_roundtrip:.3f}  '
                      f'iradon={ssim_iradon:.3f}  polar={ssim_polar:.3f}')

            viewer.add_image(
                gt_contrast, name='Ground truth |dZ/Z0|',
                scale=scale, colormap='gray', opacity=0.9,
            )
            viewer.add_image(
                recon_roundtrip,
                name=f'Radon roundtrip (SSIM={ssim_roundtrip:.3f})',
                scale=scale, colormap='hot', opacity=0.9, visible=False,
            )
            viewer.add_image(
                recon_bscan_iradon,
                name=f'B-scan iradon (SSIM={ssim_iradon:.3f})',
                scale=scale, colormap='hot', opacity=0.9, visible=False,
            )
            viewer.add_image(
                recon_bscan_polar,
                name=f'B-scan polar (SSIM={ssim_polar:.3f})',
                scale=scale, colormap='hot', opacity=0.9,
            )

            viewer.dims.axis_labels = ('z - depth (mm)', 'y (mm)', 'x (mm)')
            print("  napari viewer open — close the window to continue")
            napari.run()

        except ImportError:
            print("  napari not installed — skipping interactive viewer")

    return results


# ── CLI ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Validate Radon reconstruction: roundtrip vs B-scan pipeline')
    parser.add_argument('--scan-dir', type=str,
                        default=os.path.join(os.path.dirname(__file__),
                                             'output', 'radon_tests', 'radon_test'),
                        help='Directory with bscan_*.npy, scan_meta.npy, ground_truth.npz')
    parser.add_argument('--filter', type=str, default='shepp-logan',
                        help='FBP filter (default: shepp-logan)')
    parser.add_argument('--show-napari', action='store_true',
                        help='Open napari viewer with all reconstructions')
    args = parser.parse_args()

    validate(args.scan_dir, filter_name=args.filter, show_napari=args.show_napari)


if __name__ == '__main__':
    main()
