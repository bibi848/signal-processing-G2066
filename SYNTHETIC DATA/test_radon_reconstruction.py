"""
Test Radon Reconstruction Quality vs Number of Projections
===========================================================

Simulates a cuboid aluminum specimen with a cylindrical defect (side-drilled
hole along y through the centre) using the full FMC/TFM pipeline, then
reconstructs the 3D volume with varying numbers of B-scan projections to
compare reconstruction quality.

The simulation runs once at the maximum number of angles.  For each n_scans
value, an evenly-spaced subset of B-scans is selected and reconstructed via
the inverse Radon pipeline in Classes/Reconstruct3D.py.

Usage:
    python test_radon_reconstruction.py
    python test_radon_reconstruction.py --n-scans 4 8 16 32 64 --show-napari
    python test_radon_reconstruction.py --max-scans 128 --n-scans 8 16 32 64 128 --save-volumes
"""

import sys
import os
import argparse
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from skimage.metrics import structural_similarity as ssim
from scipy.stats import pearsonr

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from Classes.Reconstruct3D import (
    has_complex_bscans,
    reconstruct_volume,
    compute_reconstruction_coords,
)


def _circle_mask(size: int) -> np.ndarray:
    c = (size - 1) / 2.0
    y, x = np.ogrid[:size, :size]
    return (y - c) ** 2 + (x - c) ** 2 <= (size / 2.0) ** 2

# Synthetic data engine
from engine.geometry import Specimen3D, CylindricalDefect
from engine.config import (
    SimulationConfig, SpecimenConfig, ArrayConfig, ScanPlanConfig,
)
from engine.materials import ALUMINUM
from engine.microstructure import generate_grain_structure, embed_geometric_defects
from engine.voxel_volume import VoxelVolume3D
from run_engine import scan_volume_3d
from scipy.ndimage import map_coordinates


def save_ground_truth(volume: VoxelVolume3D, path: str) -> None:
    np.savez_compressed(
        path,
        impedance=volume.impedance, wavespeed=volume.wavespeed,
        voxel_size=np.float64(volume.voxel_size),
        origin_z=np.float64(volume.origin_z),
        origin_y=np.float64(volume.origin_y),
        origin_x=np.float64(volume.origin_x),
    )


def load_ground_truth_file(path: str) -> VoxelVolume3D:
    d = np.load(path)
    return VoxelVolume3D(
        impedance=d['impedance'], wavespeed=d['wavespeed'],
        voxel_size=float(d['voxel_size']),
        origin_z=float(d['origin_z']),
        origin_y=float(d['origin_y']),
        origin_x=float(d['origin_x']),
    )


def extract_ground_truth_contrast(volume, z_coords, y_coords, x_coords, background_Z):
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


# ── Simulation ──────────────────────────────────────────────────────────

def simulate_specimen(
    max_scans: int = 64,
    thickness: float = 50e-3,
    width: float = 50e-3,
    depth: float = 30e-3,
    cylinder_radius: float = 1e-3,
    num_elements: int = 64,
    element_pitch: float = 0.6e-3,
    frequency: float = 5e6,
    use_grain_noise: bool = False,
    mean_grain_size_m: float = 0.5e-3,
    tfm_n_pixels: int = 400,
    output_dir: str = 'output/radon_tests/radon_test',
    seed: int = 42,
) -> str:
    """
    Simulate a cuboid specimen with a centred cylindrical defect.

    The cylinder runs along y (elevation) through the specimen centre
    at (center_z=thickness/2, center_x=0).

    Returns:
        output_dir path where B-scans and metadata are saved.
    """
    specimen = Specimen3D(
        thickness=thickness,
        width=width,
        depth=depth,
    )

    defects_3d = [
        CylindricalDefect(
            center_z=thickness / 2,
            center_x=5e-3,  # 5 mm off-axis
            radius=cylinder_radius,
            y_start=-depth / 2,
            y_end=depth / 2,
        ),
    ]

    # Always create a ground truth voxel volume for visualisation
    wavelength = ALUMINUM.c_L / frequency
    voxel_size = wavelength / 3
    print(f"Generating ground truth volume (voxel = {voxel_size*1e3:.2f} mm)...")

    if use_grain_noise:
        grain_vol = generate_grain_structure(
            thickness=thickness,
            width=width,
            depth=depth,
            background_material=ALUMINUM,
            mean_grain_size_m=mean_grain_size_m,
            voxel_size_m=voxel_size,
            seed=seed,
        )
    else:
        # Uniform background (no grain noise)
        n_z = int(np.ceil(thickness / voxel_size))
        n_y = int(np.ceil(depth / voxel_size))
        n_x = int(np.ceil(width / voxel_size))
        Z0 = ALUMINUM.density * ALUMINUM.c_L
        grain_vol = VoxelVolume3D(
            impedance=np.full((n_z, n_y, n_x), Z0, dtype=np.float32),
            wavespeed=np.full((n_z, n_y, n_x), ALUMINUM.c_L, dtype=np.float32),
            voxel_size=voxel_size,
            origin_z=0.0,
            origin_y=-depth / 2.0,
            origin_x=-width / 2.0,
        )

    gt_volume = embed_geometric_defects(grain_vol, defects_3d)
    gt_path = os.path.join(output_dir, 'ground_truth.npz')
    os.makedirs(output_dir, exist_ok=True)
    save_ground_truth(gt_volume, gt_path)

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
        scan_plan=scan_plan,
        max_bounces=2,
        mode_conversion=False,
    )
    print(cfg.summary())

    geom_defects = [] if use_grain_noise else defects_3d
    scan_volume_3d(
        specimen, geom_defects, cfg, scan_plan, output_dir,
        voxel_volume=voxel_volume,
        tfm_n_pixels=tfm_n_pixels,
    )

    print(f"\nSimulation complete — {max_scans} B-scans saved to {output_dir}/")
    return output_dir


# ── Load & reconstruct from subset ─────────────────────────────────────

def load_and_reconstruct(
    scan_dir: str,
    n_scans: int,
    filter_name: str = 'shepp-logan',
) -> np.ndarray:
    """
    Load a subset of B-scans from scan_dir and reconstruct via inverse Radon.

    Uses the complex analytic pipeline (matching MATLAB approach) when
    complex B-scan files are available, otherwise falls back to dB envelope.

    Selects n_scans evenly-spaced frames from the full set.

    Returns:
        volume: (n_z, output_size, output_size) float32
    """
    meta = np.load(os.path.join(scan_dir, 'scan_meta.npy'), allow_pickle=True).item()
    total_scans = meta['n_scans']
    all_angles_rad = meta['angles_rad']

    # Select evenly-spaced subset
    indices = np.round(np.linspace(0, total_scans - 1, n_scans)).astype(int)
    angles_rad = all_angles_rad[indices]
    angles_deg = np.degrees(angles_rad)

    if has_complex_bscans(scan_dir):
        bscans = np.stack([
            np.load(os.path.join(scan_dir, f'bscan_complex_{i:04d}.npy'))
            for i in indices
        ], axis=0)  # (n_scans, n_z, n_lateral) complex
    else:
        bscans_db = np.stack([
            np.load(os.path.join(scan_dir, f'bscan_{i:04d}.npy'))
            for i in indices
        ], axis=0)
        bscans = np.float32(10.0 ** (bscans_db / 20.0))

    volume = reconstruct_volume(
        bscans, angles_deg,
        filter_name=filter_name, circle=True,
    )
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
    filter_name: str = 'hann',
    save_volumes: bool = False,
    output_dir: str = '.',
) -> list:
    """
    Reconstruct from subsets of B-scans and collect metrics.

    Uses the reconstruction with the most scans as the reference for
    comparison (best available reconstruction).

    Returns:
        List of dicts with keys: n_scans, metrics, recon_mid_slice, volume.
    """
    n_scans_list = sorted(n_scans_list)
    results = []

    # Reconstruct all
    for n in n_scans_list:
        print(f"\n{'='*60}")
        print(f"  Reconstructing with n_scans = {n}")
        print(f"{'='*60}")

        recon = load_and_reconstruct(scan_dir, n, filter_name=filter_name)

        if save_volumes:
            vol_path = os.path.join(output_dir, f'recon_n{n:04d}.npy')
            np.save(vol_path, recon)
            print(f"  Volume saved to {vol_path}")

        mid_z = recon.shape[0] // 2
        results.append({
            'n_scans': n,
            'recon_mid_slice': recon[mid_z].copy(),
            'volume': recon,
        })

    # Use best reconstruction (most scans) as reference
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

    # Top row: reconstruction slices
    for col, idx in enumerate(show_idx):
        r = results[idx]
        axes[0, col].imshow(r['recon_mid_slice'], cmap='hot')
        axes[0, col].set_title(f"N = {r['n_scans']}")
        axes[0, col].axis('off')

    for col in range(n_show, axes.shape[1]):
        axes[0, col].axis('off')

    # Bottom row: metric curves
    n_vals = [r['n_scans'] for r in results]
    ssim_vals = [r['metrics']['ssim'] for r in results]
    nrmse_vals = [r['metrics']['nrmse'] for r in results]

    ax_ssim = axes[1, 0]
    ax_ssim.plot(n_vals, ssim_vals, 'o-', color='tab:blue', linewidth=2)
    ax_ssim.set_xscale('log', base=2)
    ax_ssim.set_xlabel('Number of scans')
    ax_ssim.set_ylabel('SSIM')
    ax_ssim.set_title('SSIM vs N scans')
    ax_ssim.grid(True, alpha=0.3)

    ax_nrmse = axes[1, 1]
    ax_nrmse.plot(n_vals, nrmse_vals, 's-', color='tab:red', linewidth=2)
    ax_nrmse.set_xscale('log', base=2)
    ax_nrmse.set_xlabel('Number of scans')
    ax_nrmse.set_ylabel('NRMSE')
    ax_nrmse.set_title('NRMSE vs N scans')
    ax_nrmse.grid(True, alpha=0.3)

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

    viewer = napari.Viewer(title='Radon Reconstruction Sweep')

    # Load and display ground truth
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
        description='Test Radon reconstruction quality vs number of projections '
                    'using the full FMC/TFM simulation pipeline')

    # Simulation parameters
    parser.add_argument('--max-scans', type=int, default=64,
                        help='Total number of B-scans to simulate (default: 64)')
    parser.add_argument('--thickness', type=float, default=50e-3,
                        help='Specimen thickness in m (default: 50e-3)')
    parser.add_argument('--width', type=float, default=50e-3,
                        help='Specimen width in m (default: 50e-3)')
    parser.add_argument('--depth', type=float, default=30e-3,
                        help='Specimen depth/elevation in m (default: 30e-3)')
    parser.add_argument('--cylinder-radius', type=float, default=1e-3,
                        help='Cylinder defect radius in m (default: 1e-3)')
    parser.add_argument('--frequency', type=float, default=5e6,
                        help='Array frequency in Hz (default: 5e6)')
    parser.add_argument('--num-elements', type=int, default=32,
                        help='Number of array elements (default: 32)')
    parser.add_argument('--grain-noise', action='store_true',
                        help='Include Voronoi grain structure')
    parser.add_argument('--tfm-n-pixels', type=int, default=400,
                        help='TFM grid size (default: 400)')

    # Reconstruction sweep
    parser.add_argument('--n-scans', type=int, nargs='+',
                        default=[4, 8, 16, 32, 64],
                        help='List of scan counts to test (default: 4 8 16 32 64)')
    parser.add_argument('--filter', type=str, default='shepp-logan',
                        help='FBP filter name (default: shepp-logan)')

    # Output
    parser.add_argument('--output-dir', type=str,
                        default=os.path.join(os.path.dirname(__file__),
                                             'output', 'radon_tests', 'radon_test'),
                        help='Output directory')
    parser.add_argument('--save-volumes', action='store_true',
                        help='Save each reconstructed volume as .npy')
    parser.add_argument('--show-napari', action='store_true',
                        help='Open napari viewer with all reconstructed volumes')
    parser.add_argument('--skip-sim', action='store_true',
                        help='Skip simulation, use existing B-scans in output-dir')

    args = parser.parse_args()

    # Ensure max-scans covers all requested n_scans values
    max_scans = max(args.max_scans, max(args.n_scans))

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Simulate (or skip if B-scans already exist)
    if not args.skip_sim:
        print(f"\n{'#'*60}")
        print(f"  SIMULATING: {max_scans} B-scans")
        print(f"  Specimen: {args.thickness*1e3:.0f} x {args.width*1e3:.0f} x "
              f"{args.depth*1e3:.0f} mm")
        print(f"  Cylinder: r={args.cylinder_radius*1e3:.1f} mm at centre")
        print(f"{'#'*60}\n")

        simulate_specimen(
            max_scans=max_scans,
            thickness=args.thickness,
            width=args.width,
            depth=args.depth,
            cylinder_radius=args.cylinder_radius,
            num_elements=args.num_elements,
            frequency=args.frequency,
            use_grain_noise=args.grain_noise,
            tfm_n_pixels=args.tfm_n_pixels,
            output_dir=args.output_dir,
        )
    else:
        print(f"Skipping simulation — using existing B-scans in {args.output_dir}/")

    # 2. Reconstruction sweep
    print(f"\n{'#'*60}")
    print(f"  RECONSTRUCTION SWEEP: {args.n_scans}")
    print(f"{'#'*60}\n")

    results = sweep_n_scans(
        args.output_dir, sorted(args.n_scans),
        filter_name=args.filter,
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
    output_path = os.path.join(args.output_dir, 'radon_reconstruction_test.png')
    plot_results(results, output_path)

    # 5. Napari
    if args.show_napari:
        view_in_napari(results, args.output_dir)


if __name__ == '__main__':
    main()
