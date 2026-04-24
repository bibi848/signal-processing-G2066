"""
Test Spatial Compounding Reconstruction vs Number of Projections
================================================================

Simulates a cuboid specimen with a cylindrical defect using the full
FMC/TFM pipeline, then reconstructs via spatial compounding (direct
back-projection into 3D volume) with varying numbers of B-scans.

Unlike inverse Radon, spatial compounding places each B-scan into 3D
space at its rotation angle and averages overlapping contributions.
This correctly handles TFM B-scans which are focused images, not
line-integral projections.

Usage:
    python test_spatial_compounding.py
    python test_spatial_compounding.py --n-scans 4 8 16 32 64 --show-napari
    python test_spatial_compounding.py --skip-sim --n-scans 8 16 64 --show-napari
"""

import sys
import os
import argparse
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy.interpolate import RegularGridInterpolator
from skimage.metrics import structural_similarity as ssim
from scipy.stats import pearsonr

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from Classes.Reconstruct3D import (
    db_to_linear,
    _circle_mask,
    compute_reconstruction_coords,
)

# Synthetic data engine
from engine.geometry import Specimen3D, CylindricalDefect
from engine.config import (
    SimulationConfig, SpecimenConfig, ArrayConfig, ScanPlanConfig,
)
from engine.materials import ALUMINUM
from engine.microstructure import generate_grain_structure, embed_geometric_defects
from engine.voxel_volume import VoxelVolume3D
from run_engine import scan_volume_3d
from reconstruct_3d import (
    save_ground_truth,
    load_ground_truth_file,
    extract_ground_truth_contrast,
)


# ── Simulation ──────────────────────────────────────────────────────────

def simulate_specimen(
    max_scans: int = 64,
    thickness: float = 50e-3,
    width: float = 50e-3,
    depth: float = 30e-3,
    cylinder_radius: float = 1e-3,
    cylinder_x: float = 5e-3,
    num_elements: int = 32,
    element_pitch: float = 0.6e-3,
    frequency: float = 5e6,
    use_grain_noise: bool = False,
    mean_grain_size_m: float = 0.5e-3,
    tfm_n_pixels: int = 400,
    output_dir: str = 'output/radon_tests/compound_test',
    seed: int = 42,
) -> str:
    """
    Simulate a cuboid specimen with a cylindrical defect.

    Returns:
        output_dir path where B-scans and metadata are saved.
    """
    specimen = Specimen3D(thickness=thickness, width=width, depth=depth)

    defects_3d = [
        CylindricalDefect(
            center_z=thickness / 2,
            center_x=cylinder_x,
            radius=cylinder_radius,
            y_start=-depth / 2,
            y_end=depth / 2,
        ),
    ]

    # Always create ground truth volume
    wavelength = ALUMINUM.c_L / frequency
    voxel_size = wavelength / 3
    print(f"Generating ground truth volume (voxel = {voxel_size*1e3:.2f} mm)...")

    if use_grain_noise:
        grain_vol = generate_grain_structure(
            thickness=thickness, width=width, depth=depth,
            background_material=ALUMINUM,
            mean_grain_size_m=mean_grain_size_m,
            voxel_size_m=voxel_size, seed=seed,
        )
    else:
        n_z = int(np.ceil(thickness / voxel_size))
        n_y = int(np.ceil(depth / voxel_size))
        n_x = int(np.ceil(width / voxel_size))
        Z0 = ALUMINUM.density * ALUMINUM.c_L
        grain_vol = VoxelVolume3D(
            impedance=np.full((n_z, n_y, n_x), Z0, dtype=np.float32),
            wavespeed=np.full((n_z, n_y, n_x), ALUMINUM.c_L, dtype=np.float32),
            voxel_size=voxel_size,
            origin_z=0.0, origin_y=-depth / 2.0, origin_x=-width / 2.0,
        )

    gt_volume = embed_geometric_defects(grain_vol, defects_3d)
    os.makedirs(output_dir, exist_ok=True)
    save_ground_truth(gt_volume, os.path.join(output_dir, 'ground_truth.npz'))

    voxel_volume = gt_volume if use_grain_noise else None

    scan_plan = ScanPlanConfig(
        n_scans=max_scans,
        theta_start=-np.pi / 2,
        theta_end=np.pi / 2,
    )
    cfg = SimulationConfig(
        specimen=SpecimenConfig(thickness=thickness, width=width),
        array=ArrayConfig(
            num_elements=num_elements,
            element_pitch=element_pitch,
            frequency=frequency,
        ),
        scan_plan=scan_plan, max_bounces=2, mode_conversion=False,
    )
    print(cfg.summary())

    geom_defects = [] if use_grain_noise else defects_3d
    scan_volume_3d(
        specimen, geom_defects, cfg, scan_plan, output_dir,
        voxel_volume=voxel_volume, tfm_n_pixels=tfm_n_pixels,
    )

    print(f"\nSimulation complete — {max_scans} B-scans saved to {output_dir}/")
    return output_dir


# ── Spatial compounding reconstruction ──────────────────────────────────

def spatial_compound(
    scan_dir: str,
    n_scans: int,
    output_size: int = 256,
) -> np.ndarray:
    """
    Reconstruct a 3D volume by spatial compounding of B-scans.

    Each B-scan pixel at (z, L) and angle θ is placed at:
        x = L · cos(θ)
        y = L · sin(θ)
        z = z

    The volume is built by interpolating each B-scan onto the output
    grid and averaging all contributions.

    Args:
        scan_dir:    Directory with bscan_*.npy and scan_meta.npy.
        n_scans:     Number of B-scans to use (evenly spaced subset).
        output_size: Lateral grid size (n_y = n_x = output_size).

    Returns:
        volume: (n_z, output_size, output_size) float32
    """
    meta = np.load(os.path.join(scan_dir, 'scan_meta.npy'),
                   allow_pickle=True).item()
    total_scans = meta['n_scans']
    all_angles_rad = meta['angles_rad']
    half_ap = meta['array_aperture_m'] / 2.0

    # Select evenly-spaced subset
    indices = np.round(np.linspace(0, total_scans - 1, n_scans)).astype(int)
    angles = all_angles_rad[indices]

    # Load first B-scan to get dimensions
    bscan0 = np.load(os.path.join(scan_dir, f'bscan_{indices[0]:04d}.npy'))
    n_z_bscan, n_lateral = bscan0.shape

    # B-scan axes
    z_start = meta.get('tfm_z_start_m', 10e-3)
    z_end = meta.get('tfm_z_end_m', meta['specimen_thickness_m'] - 5e-3)
    z_axis = np.linspace(z_start, z_end, n_z_bscan)
    L_axis = np.linspace(-half_ap, half_ap, n_lateral)

    # Output volume grid
    y_out = np.linspace(-half_ap, half_ap, output_size)
    x_out = np.linspace(-half_ap, half_ap, output_size)

    # Accumulation buffers
    volume_sum = np.zeros((n_z_bscan, output_size, output_size), dtype=np.float64)
    weight_sum = np.zeros((n_z_bscan, output_size, output_size), dtype=np.float64)

    # Output meshgrid (y, x)
    YY, XX = np.meshgrid(y_out, x_out, indexing='ij')

    for i, (idx, theta) in enumerate(zip(indices, angles)):
        bscan_db = np.load(os.path.join(scan_dir, f'bscan_{idx:04d}.npy'))

        # Convert dB to linear
        data_fmt = meta.get('data_format', 'db')
        if data_fmt == 'linear_envelope':
            bscan_lin = bscan_db.copy()
        else:
            bscan_lin = np.float32(10.0 ** (bscan_db / 20.0))
            vmin_db = meta.get('vmin_db', None)
            if vmin_db is not None:
                floor = np.float32(10.0 ** (vmin_db / 20.0))
                bscan_lin = np.maximum(bscan_lin - floor, 0.0)

        # For each output (y, x), compute the lateral position L in this B-scan:
        #   x = L cos θ,  y = L sin θ  →  L = x cos θ + y sin θ
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        L_map = XX * cos_t + YY * sin_t  # (output_size, output_size)

        # Mask: only include points within the B-scan lateral range
        in_range = (L_map >= L_axis[0]) & (L_map <= L_axis[-1])

        # Interpolate each depth slice
        for z in range(n_z_bscan):
            interp = RegularGridInterpolator(
                (L_axis,), bscan_lin[z],
                method='linear', bounds_error=False, fill_value=0.0,
            )
            vals = interp(L_map.ravel()).reshape(output_size, output_size)
            volume_sum[z] += vals * in_range
            weight_sum[z] += in_range.astype(np.float64)

        if (i + 1) % 10 == 0 or i == 0 or i == len(indices) - 1:
            print(f"  Compounded {i+1}/{len(indices)} B-scans "
                  f"(θ = {np.degrees(theta):+6.1f}°)")

    # Average where we have contributions
    mask = weight_sum > 0
    volume = np.zeros_like(volume_sum, dtype=np.float32)
    volume[mask] = (volume_sum[mask] / weight_sum[mask]).astype(np.float32)

    print(f"  Spatial compounding complete: {volume.shape}")
    return volume


# ── Metrics ─────────────────────────────────────────────────────────────

def compute_metrics(recon: np.ndarray, reference: np.ndarray, mask: np.ndarray) -> dict:
    """Compute SSIM, NRMSE, and Pearson r between two volumes within mask."""
    def _normalise(vol, m):
        vals = vol[:, m]
        vmin, vmax = vals.min(), vals.max()
        if vmax - vmin < 1e-12:
            return vol * 0.0
        return (vol - vmin) / (vmax - vmin)

    rn = _normalise(recon, mask)
    gn = _normalise(reference, mask)

    ssim_vals = [ssim(gn[z], rn[z], data_range=1.0) for z in range(rn.shape[0])]
    mean_ssim = float(np.mean(ssim_vals))

    diff = rn[:, mask] - gn[:, mask]
    nrmse = float(np.sqrt(np.mean(diff ** 2)))

    r, _ = pearsonr(rn[:, mask].ravel(), gn[:, mask].ravel())

    return {'ssim': mean_ssim, 'nrmse': nrmse, 'pearson_r': float(r)}


# ── Sweep ───────────────────────────────────────────────────────────────

def sweep_n_scans(
    scan_dir: str,
    n_scans_list: list,
    output_size: int = 256,
    save_volumes: bool = False,
    output_dir: str = '.',
) -> list:
    """Reconstruct via spatial compounding for each n_scans value."""
    n_scans_list = sorted(n_scans_list)
    results = []

    for n in n_scans_list:
        print(f"\n{'='*60}")
        print(f"  Spatial compounding with n_scans = {n}")
        print(f"{'='*60}")

        recon = spatial_compound(scan_dir, n, output_size=output_size)

        if save_volumes:
            vol_path = os.path.join(output_dir, f'compound_n{n:04d}.npy')
            np.save(vol_path, recon)
            print(f"  Volume saved to {vol_path}")

        mid_z = recon.shape[0] // 2
        results.append({
            'n_scans': n,
            'recon_mid_slice': recon[mid_z].copy(),
            'volume': recon,
        })

    # Metrics against best reconstruction
    reference = results[-1]['volume']
    mask = _circle_mask(reference.shape[1])

    for r in results:
        r['metrics'] = compute_metrics(r['volume'], reference, mask)
        m = r['metrics']
        print(f"  n_scans={r['n_scans']:>4d}  |  "
              f"SSIM={m['ssim']:.4f}  |  "
              f"NRMSE={m['nrmse']:.4f}  |  "
              f"Pearson r={m['pearson_r']:.4f}")

    return results


# ── Plotting ────────────────────────────────────────────────────────────

def plot_results(results: list, output_path: str) -> None:
    """Save a summary figure: reconstruction slices + metric curves."""
    n_results = len(results)
    if n_results <= 6:
        show_idx = list(range(n_results))
    else:
        show_idx = np.linspace(0, n_results - 1, 6, dtype=int).tolist()

    n_show = len(show_idx)
    fig, axes = plt.subplots(2, max(n_show, 2), figsize=(4 * max(n_show, 2), 8))

    for col, idx in enumerate(show_idx):
        r = results[idx]
        axes[0, col].imshow(r['recon_mid_slice'], cmap='hot')
        axes[0, col].set_title(f"N = {r['n_scans']}")
        axes[0, col].axis('off')

    for col in range(n_show, axes.shape[1]):
        axes[0, col].axis('off')

    n_vals = [r['n_scans'] for r in results]
    ssim_vals = [r['metrics']['ssim'] for r in results]
    nrmse_vals = [r['metrics']['nrmse'] for r in results]

    axes[1, 0].plot(n_vals, ssim_vals, 'o-', color='tab:blue', linewidth=2)
    axes[1, 0].set_xscale('log', base=2)
    axes[1, 0].set_xlabel('Number of scans')
    axes[1, 0].set_ylabel('SSIM')
    axes[1, 0].set_title('SSIM vs N scans')
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(n_vals, nrmse_vals, 's-', color='tab:red', linewidth=2)
    axes[1, 1].set_xscale('log', base=2)
    axes[1, 1].set_xlabel('Number of scans')
    axes[1, 1].set_ylabel('NRMSE')
    axes[1, 1].set_title('NRMSE vs N scans')
    axes[1, 1].grid(True, alpha=0.3)

    for col in range(2, axes.shape[1]):
        axes[1, col].axis('off')

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"\nFigure saved to {output_path}")


# ── Napari viewer ───────────────────────────────────────────────────────

COLORMAPS = ['viridis', 'magma', 'inferno', 'plasma', 'turbo', 'hot']

def view_in_napari(results: list, scan_dir: str) -> None:
    """Open napari with ground truth and all reconstructed volumes."""
    try:
        import napari
    except ImportError:
        print("napari not installed -- skipping interactive viewer")
        return

    viewer = napari.Viewer(title='Spatial Compounding Sweep')

    # Ground truth
    gt_path = os.path.join(scan_dir, 'ground_truth.npz')
    if os.path.exists(gt_path):
        gt_vol = load_ground_truth_file(gt_path)
        meta = np.load(os.path.join(scan_dir, 'scan_meta.npy'),
                       allow_pickle=True).item()
        recon_size = results[0]['volume'].shape[1]
        z_coords, y_coords, x_coords = compute_reconstruction_coords(
            meta, recon_size)
        background_Z = ALUMINUM.density * ALUMINUM.c_L
        gt_contrast = extract_ground_truth_contrast(
            gt_vol, z_coords, y_coords, x_coords, background_Z)
        viewer.add_image(
            gt_contrast, name='Ground Truth |dZ/Z0|',
            colormap='gray', opacity=0.9,
        )

    for i, r in enumerate(results):
        n = r['n_scans']
        m = r['metrics']
        cmap = COLORMAPS[i % len(COLORMAPS)]
        viewer.add_image(
            r['volume'],
            name=f'N={n}  SSIM={m["ssim"]:.3f}',
            colormap=cmap,
            opacity=0.9,
            visible=(i == len(results) - 1),
        )

    viewer.dims.axis_labels = ('z', 'y', 'x')
    print("napari viewer open -- close the window to continue")
    napari.run()


# ── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Test spatial compounding reconstruction vs number of B-scans')

    # Simulation
    parser.add_argument('--max-scans', type=int, default=64)
    parser.add_argument('--thickness', type=float, default=50e-3)
    parser.add_argument('--width', type=float, default=50e-3)
    parser.add_argument('--depth', type=float, default=30e-3)
    parser.add_argument('--cylinder-radius', type=float, default=1e-3,
                        help='Cylinder defect radius in m (default: 1e-3)')
    parser.add_argument('--cylinder-x', type=float, default=5e-3,
                        help='Cylinder x-offset from centre in m (default: 5e-3)')
    parser.add_argument('--frequency', type=float, default=5e6)
    parser.add_argument('--num-elements', type=int, default=32)
    parser.add_argument('--grain-noise', action='store_true')
    parser.add_argument('--tfm-n-pixels', type=int, default=400)

    # Reconstruction
    parser.add_argument('--n-scans', type=int, nargs='+',
                        default=[4, 8, 16, 32, 64])
    parser.add_argument('--output-size', type=int, default=256,
                        help='Lateral grid size for reconstruction (default: 256)')

    # Output
    parser.add_argument('--output-dir', type=str,
                        default=os.path.join(os.path.dirname(__file__),
                                             'output', 'radon_tests', 'compound_test'))
    parser.add_argument('--save-volumes', action='store_true')
    parser.add_argument('--show-napari', action='store_true')
    parser.add_argument('--skip-sim', action='store_true',
                        help='Skip simulation, use existing B-scans')

    args = parser.parse_args()
    max_scans = max(args.max_scans, max(args.n_scans))
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Simulate
    if not args.skip_sim:
        print(f"\n{'#'*60}")
        print(f"  SIMULATING: {max_scans} B-scans")
        print(f"  Cylinder: r={args.cylinder_radius*1e3:.1f} mm "
              f"at x={args.cylinder_x*1e3:.1f} mm")
        print(f"{'#'*60}\n")

        simulate_specimen(
            max_scans=max_scans,
            thickness=args.thickness, width=args.width, depth=args.depth,
            cylinder_radius=args.cylinder_radius,
            cylinder_x=args.cylinder_x,
            num_elements=args.num_elements, frequency=args.frequency,
            use_grain_noise=args.grain_noise,
            tfm_n_pixels=args.tfm_n_pixels,
            output_dir=args.output_dir,
        )
    else:
        print(f"Skipping simulation — using existing B-scans in {args.output_dir}/")

    # 2. Reconstruction sweep
    print(f"\n{'#'*60}")
    print(f"  SPATIAL COMPOUNDING SWEEP: {args.n_scans}")
    print(f"{'#'*60}\n")

    results = sweep_n_scans(
        args.output_dir, sorted(args.n_scans),
        output_size=args.output_size,
        save_volumes=args.save_volumes,
        output_dir=args.output_dir,
    )

    # 3. Summary table
    print(f"\n{'='*60}")
    print(f"  {'n_scans':>8} | {'SSIM':>8} | {'NRMSE':>8} | {'Pearson r':>10}")
    print(f"  {'-'*8} | {'-'*8} | {'-'*8} | {'-'*10}")
    for r in results:
        m = r['metrics']
        print(f"  {r['n_scans']:>8d} | {m['ssim']:>8.4f} | {m['nrmse']:>8.4f} | "
              f"{m['pearson_r']:>10.4f}")
    print(f"{'='*60}")

    # 4. Figure
    output_path = os.path.join(args.output_dir, 'spatial_compounding_test.png')
    plot_results(results, output_path)

    # 5. Napari
    if args.show_napari:
        view_in_napari(results, args.output_dir)


if __name__ == '__main__':
    main()
