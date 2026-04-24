#!/usr/bin/env python3
"""
Translation B-scan Stacking + Interpolation Test
=================================================

Translates a 1D array along the y-axis at θ=0, acquiring parallel B-scans
at each y-position.  B-scans are stacked and interpolated along y to
reconstruct a 3D volume.

Tests multiple B-scan spacings to determine how spacing affects
reconstruction quality (SSIM, Pearson r, NRMSE vs ground truth).

Usage:
    python test_translation_reconstruction.py
    python test_translation_reconstruction.py --spacings 0.5 1.0 2.0 5.0
    python test_translation_reconstruction.py --skip-sim   # reuse saved B-scans
    python test_translation_reconstruction.py --show-napari
"""

import argparse
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity as ssim

# ── Imports from the engine ──────────────────────────────────────────
from engine.config import SimulationConfig, SpecimenConfig, ArrayConfig
from engine.geometry import Specimen3D
from engine.fmc_engine import FMCEngine
from engine.materials import ALUMINUM
from engine.voxel_volume import VoxelVolume3D
from engine.microstructure import generate_grain_structure

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from Classes.Filter import filter_signal
from Classes.TFM1D import CTFM1D
from scipy.signal import hilbert

# Reuse noise / filter / TFM helpers from run_engine
from run_engine import add_noise, apply_bandpass_filter, reconstruct_tfm

# Ground truth comparison
from reconstruct_3d import extract_ground_truth_contrast, save_ground_truth

# ── Parameters ───────────────────────────────────────────────────────
SPECIMEN_THICKNESS = 50e-3   # 50 mm (z)
SPECIMEN_WIDTH     = 50e-3   # 50 mm (x, along array)
SPECIMEN_DEPTH     = 30e-3   # 30 mm (y, translation axis)

FREQUENCY       = 10e6
NUM_ELEMENTS    = 64
ELEMENT_PITCH   = 0.6e-3
WAVELENGTH      = ALUMINUM.c_L / FREQUENCY
MEAN_GRAIN_SIZE = 0.5e-3
IMP_VARIATION   = 0.025
VOXEL_SIZE      = WAVELENGTH / 3          # λ/3 ≈ 0.21 mm

TFM_N_PIXELS   = 800
TFM_Z_START    = 10e-3
TFM_Z_END      = 45e-3
BORN_THRESHOLD = 0.005

DEFAULT_SPACINGS_MM = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output', 'radon_tests', 'translation_test')


# ── Helpers: slicing at y-offset ─────────────────────────────────────

def slice_at_y_offset(volume: VoxelVolume3D, y_offset: float,
                      z_grid: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
    """Sample impedance at (z, y_offset, x) — a plane parallel to x-z."""
    ZZ, XX = np.meshgrid(z_grid, x_grid, indexing='ij')
    iz_f = (ZZ - volume.origin_z) / volume.voxel_size
    iy_f = (y_offset - volume.origin_y) / volume.voxel_size
    ix_f = (XX - volume.origin_x) / volume.voxel_size
    coords = np.array([iz_f.ravel(),
                        np.full(iz_f.size, iy_f),
                        ix_f.ravel()])
    bg_Z = float(np.mean(volume.impedance))
    imp_flat = map_coordinates(volume.impedance.astype(np.float64), coords,
                               order=0, mode='constant', cval=bg_Z)
    return imp_flat.reshape(ZZ.shape).astype(np.float32)


def extract_born_at_y_offset(volume: VoxelVolume3D, y_offset: float,
                              z_grid: np.ndarray, x_grid: np.ndarray,
                              background_Z: float,
                              threshold: float = 0.005):
    """Extract Born scatterers for a fixed y-offset plane (θ=0)."""
    imp_2d = slice_at_y_offset(volume, y_offset, z_grid, x_grid)
    delta_z = np.diff(imp_2d, axis=0, prepend=imp_2d[:1, :])
    delta_rel = delta_z / (2.0 * background_Z)
    mask = np.abs(delta_rel) > threshold
    iz, ix = np.where(mask)
    return (z_grid[iz].astype(np.float64),
            x_grid[ix].astype(np.float64),
            delta_rel[iz, ix].astype(np.float64))


# ── Simulation ───────────────────────────────────────────────────────

def run_translation_scan(voxel_volume: VoxelVolume3D,
                          y_positions: np.ndarray,
                          output_dir: str):
    """
    Simulate FMC + TFM at each y-position and save B-scans.
    Array is at θ=0, aligned along x.
    """
    os.makedirs(output_dir, exist_ok=True)

    specimen = Specimen3D(thickness=SPECIMEN_THICKNESS,
                          width=SPECIMEN_WIDTH,
                          depth=SPECIMEN_DEPTH)
    cfg = SimulationConfig(
        specimen=SpecimenConfig(thickness=SPECIMEN_THICKNESS,
                                width=SPECIMEN_WIDTH),
        array=ArrayConfig(num_elements=NUM_ELEMENTS,
                          element_pitch=ELEMENT_PITCH,
                          frequency=FREQUENCY),
        max_bounces=2,
        mode_conversion=False,
    )

    half_w = cfg.array.aperture / 2

    # Born scattering grids
    gate_z = ALUMINUM.c_L * 2e-6 / 2
    born_z_start = max(gate_z * 1.2, 1e-3)
    born_step = voxel_volume.voxel_size
    born_z_grid = np.linspace(born_z_start, SPECIMEN_THICKNESS,
                              max(2, int((SPECIMEN_THICKNESS - born_z_start) / born_step) + 1))
    born_x_grid = np.linspace(-half_w, half_w,
                              max(2, int(cfg.array.aperture / born_step) + 1))

    n_pos = len(y_positions)
    print(f"\n{'='*60}")
    print(f"  TRANSLATION SCAN — {n_pos} positions along y")
    print(f"  y = [{y_positions[0]*1e3:.1f} mm, {y_positions[-1]*1e3:.1f} mm]"
          f"  step = {(y_positions[1]-y_positions[0])*1e3:.2f} mm")
    print(f"  Output → {output_dir}")
    print(f"{'='*60}\n")

    bscans = []
    for i, y_pos in enumerate(y_positions):
        print(f"  Position {i+1:>3}/{n_pos}  (y = {y_pos*1e3:+.2f} mm)", end="  ")

        engine = FMCEngine(cfg)

        # Born scatterers at this y-offset
        z_s, x_s, amp_s = extract_born_at_y_offset(
            voxel_volume, y_pos, born_z_grid, born_x_grid,
            background_Z=ALUMINUM.Z_L, threshold=BORN_THRESHOLD,
        )
        if len(z_s) > 0:
            rng = np.random.default_rng(seed=i)
            dz = born_z_grid[1] - born_z_grid[0] if len(born_z_grid) > 1 else 1e-4
            dx = born_x_grid[1] - born_x_grid[0] if len(born_x_grid) > 1 else 1e-4
            z_s = z_s + rng.uniform(-dz/2, dz/2, size=z_s.shape)
            x_s = x_s + rng.uniform(-dx/2, dx/2, size=x_s.shape)
            engine.set_born_scatterers(z_s, x_s, amp_s)

        print(f"({len(z_s)} Born scatterers)")

        result = engine.simulate()
        fmc = result['fmc_data']
        time_axis = result['time_axis']
        elem_x = result['element_positions']

        # Gate out front-wall echo
        gate_samples = int(2e-6 / cfg.dt)
        fmc[:, :, :gate_samples] = 0.0

        fmc = add_noise(fmc, snr_db=cfg.acquisition.snr_db,
                        grain_noise_level=cfg.acquisition.grain_noise_level)
        fmc = apply_bandpass_filter(fmc, cfg.dt, cfg.array.frequency,
                                    bandwidth_fraction=cfg.array.bandwidth,
                                    filter_alpha=cfg.acquisition.filter_alpha,
                                    hanning_bool=cfg.acquisition.hanning_bool)

        img, x_img, z_img = reconstruct_tfm(
            fmc, time_axis, elem_x, ALUMINUM.c_L,
            x_range=(-half_w, half_w),
            z_range=(TFM_Z_START, TFM_Z_END),
            n_pixels=TFM_N_PIXELS,
        )

        # Convert to envelope (dB)
        if np.iscomplexobj(img):
            img_envelope = np.abs(img)
        else:
            img_envelope = np.abs(hilbert(img, axis=0))
        img_db = 20 * np.log10(img_envelope / (img_envelope.max() + 1e-10) + 1e-10)

        np.save(os.path.join(output_dir, f'bscan_{i:04d}.npy'),
                img_db.astype(np.float32))
        bscans.append(img_db.astype(np.float32))

    # Save metadata
    meta = {
        'n_positions': n_pos,
        'y_positions_m': y_positions,
        'y_step_m': float(y_positions[1] - y_positions[0]),
        'specimen_thickness_m': SPECIMEN_THICKNESS,
        'specimen_width_m': SPECIMEN_WIDTH,
        'specimen_depth_m': SPECIMEN_DEPTH,
        'tfm_z_start_m': TFM_Z_START,
        'tfm_z_end_m': TFM_Z_END,
        'tfm_n_pixels': TFM_N_PIXELS,
        'array_aperture_m': cfg.array.aperture,
    }
    np.save(os.path.join(output_dir, 'scan_meta.npy'), meta, allow_pickle=True)

    # Save ground truth
    save_ground_truth(voxel_volume,
                      os.path.join(output_dir, 'ground_truth.npz'))

    print(f"\n  Done — {n_pos} B-scans saved to {output_dir}/")
    return np.stack(bscans), x_img, z_img


# ── Reconstruction by interpolation ─────────────────────────────────

def reconstruct_volume(bscans: np.ndarray,
                        y_positions: np.ndarray,
                        y_output: np.ndarray,
                        method: str = 'linear') -> np.ndarray:
    """
    Stack B-scans and interpolate along y.

    Args:
        bscans:      (n_y_sparse, n_z, n_x) dB images
        y_positions: (n_y_sparse,) y-coords of each B-scan
        y_output:    (n_y_out,) y-coords to interpolate onto
        method:      'linear' or 'cubic'

    Returns:
        volume: (n_z, n_y_out, n_x)
    """
    n_sparse, n_z, n_x = bscans.shape
    z_idx = np.arange(n_z)
    x_idx = np.arange(n_x)

    # RegularGridInterpolator expects (y, z, x) ordering
    interp = RegularGridInterpolator(
        (y_positions, z_idx, x_idx),
        bscans,
        method=method,
        bounds_error=False,
        fill_value=np.nan,
    )

    # Build output grid: (n_y_out, n_z, n_x) → then transpose to (n_z, n_y_out, n_x)
    YY, ZZ, XX = np.meshgrid(y_output, z_idx, x_idx, indexing='ij')
    pts = np.stack([YY.ravel(), ZZ.ravel(), XX.ravel()], axis=-1)
    volume = interp(pts).reshape(len(y_output), n_z, n_x)

    # Transpose to (n_z, n_y, n_x)
    volume = np.transpose(volume, (1, 0, 2))
    return volume


# ── Metrics ──────────────────────────────────────────────────────────

def compute_metrics(recon: np.ndarray, gt: np.ndarray):
    """Compute SSIM, Pearson r, NRMSE between reconstruction and ground truth."""
    # Normalise both to [0, 1]
    r_min, r_max = np.nanmin(recon), np.nanmax(recon)
    g_min, g_max = gt.min(), gt.max()
    if r_max - r_min < 1e-12:
        r_norm = np.zeros_like(recon)
    else:
        r_norm = (recon - r_min) / (r_max - r_min)
    g_norm = (gt - g_min) / (g_max - g_min) if (g_max - g_min) > 1e-12 else np.zeros_like(gt)

    # Mask out NaN from interpolation (out-of-bounds)
    valid = ~np.isnan(r_norm)
    r_flat = r_norm[valid].ravel()
    g_flat = g_norm[valid].ravel()

    ssim_val = ssim(g_norm, np.nan_to_num(r_norm), data_range=1.0)
    pearson_r, _ = pearsonr(r_flat, g_flat)
    nrmse = np.sqrt(np.mean((r_flat - g_flat)**2)) / (g_flat.max() - g_flat.min() + 1e-12)

    return {'ssim': ssim_val, 'pearson_r': pearson_r, 'nrmse': nrmse}


# ── Plotting ─────────────────────────────────────────────────────────

def plot_results(all_results: dict, gt_volume: np.ndarray,
                 z_img: np.ndarray, y_out: np.ndarray, x_img: np.ndarray,
                 output_path: str):
    """
    Create comparison figure.

    all_results: {spacing_mm: {'volume': ..., 'metrics_linear': ..., 'metrics_cubic': ..., ...}}
    """
    spacings = sorted(all_results.keys())
    n_spacings = len(spacings)

    # ── Slice comparison figure ──
    fig, axes = plt.subplots(n_spacings + 1, 3, figsize=(15, 4 * (n_spacings + 1)))
    if n_spacings + 1 == 1:
        axes = axes[np.newaxis, :]

    # Ground truth row
    iz_mid = gt_volume.shape[0] // 2
    iy_mid = gt_volume.shape[1] // 2
    ix_mid = gt_volume.shape[2] // 2

    ax = axes[0]
    ax[0].imshow(gt_volume[:, iy_mid, :], aspect='auto', cmap='hot',
                  extent=[x_img[0]*1e3, x_img[-1]*1e3, z_img[-1]*1e3, z_img[0]*1e3])
    ax[0].set_title('Ground Truth — x-z (mid y)')
    ax[0].set_ylabel('z (mm)')

    ax[1].imshow(gt_volume[:, :, ix_mid], aspect='auto', cmap='hot',
                  extent=[y_out[0]*1e3, y_out[-1]*1e3, z_img[-1]*1e3, z_img[0]*1e3])
    ax[1].set_title('Ground Truth — y-z (mid x)')

    ax[2].imshow(gt_volume[iz_mid, :, :], aspect='auto', cmap='hot',
                  extent=[x_img[0]*1e3, x_img[-1]*1e3, y_out[-1]*1e3, y_out[0]*1e3])
    ax[2].set_title('Ground Truth — x-y (mid z)')
    ax[2].set_ylabel('y (mm)')

    for row, sp in enumerate(spacings, start=1):
        vol = all_results[sp]['volume_linear']
        ax = axes[row]

        ax[0].imshow(vol[:, iy_mid, :], aspect='auto', cmap='hot',
                      extent=[x_img[0]*1e3, x_img[-1]*1e3, z_img[-1]*1e3, z_img[0]*1e3])
        ax[0].set_title(f'{sp:.1f} mm — x-z')
        ax[0].set_ylabel('z (mm)')

        ax[1].imshow(vol[:, :, ix_mid], aspect='auto', cmap='hot',
                      extent=[y_out[0]*1e3, y_out[-1]*1e3, z_img[-1]*1e3, z_img[0]*1e3])
        ax[1].set_title(f'{sp:.1f} mm — y-z')

        ax[2].imshow(vol[iz_mid, :, :], aspect='auto', cmap='hot',
                      extent=[x_img[0]*1e3, x_img[-1]*1e3, y_out[-1]*1e3, y_out[0]*1e3])
        ax[2].set_title(f'{sp:.1f} mm — x-y')
        ax[2].set_ylabel('y (mm)')

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Slice comparison saved: {output_path}")
    plt.close(fig)

    # ── Metrics vs spacing ──
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ssim_lin  = [all_results[s]['metrics_linear']['ssim'] for s in spacings]
    ssim_cub  = [all_results[s]['metrics_cubic']['ssim'] for s in spacings]
    pr_lin    = [all_results[s]['metrics_linear']['pearson_r'] for s in spacings]
    pr_cub    = [all_results[s]['metrics_cubic']['pearson_r'] for s in spacings]
    nrmse_lin = [all_results[s]['metrics_linear']['nrmse'] for s in spacings]
    nrmse_cub = [all_results[s]['metrics_cubic']['nrmse'] for s in spacings]

    ax1.plot(spacings, ssim_lin, 'o-', label='SSIM (linear)')
    ax1.plot(spacings, ssim_cub, 's--', label='SSIM (cubic)')
    ax1.plot(spacings, pr_lin, 'o-', label='Pearson r (linear)')
    ax1.plot(spacings, pr_cub, 's--', label='Pearson r (cubic)')
    ax1.axvline(MEAN_GRAIN_SIZE * 1e3, color='gray', ls=':', label=f'Mean grain ({MEAN_GRAIN_SIZE*1e3:.1f} mm)')
    ax1.set_xlabel('B-scan spacing (mm)')
    ax1.set_ylabel('Score')
    ax1.set_title('Reconstruction Quality vs Spacing')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.plot(spacings, nrmse_lin, 'o-', label='NRMSE (linear)')
    ax2.plot(spacings, nrmse_cub, 's--', label='NRMSE (cubic)')
    ax2.axvline(MEAN_GRAIN_SIZE * 1e3, color='gray', ls=':', label=f'Mean grain ({MEAN_GRAIN_SIZE*1e3:.1f} mm)')
    ax2.set_xlabel('B-scan spacing (mm)')
    ax2.set_ylabel('NRMSE')
    ax2.set_title('Error vs Spacing')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    metrics_path = output_path.replace('.png', '_metrics.png')
    fig2.savefig(metrics_path, dpi=150, bbox_inches='tight')
    print(f"  Metrics plot saved: {metrics_path}")
    plt.close(fig2)


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Translation B-scan stacking test')
    parser.add_argument('--spacings', type=float, nargs='+', default=DEFAULT_SPACINGS_MM,
                        help='B-scan spacings to test (mm)')
    parser.add_argument('--skip-sim', action='store_true',
                        help='Load previously saved B-scans')
    parser.add_argument('--show-napari', action='store_true',
                        help='Show napari 3D viewer')
    args = parser.parse_args()

    spacings_mm = sorted(args.spacings)
    finest_spacing_m = spacings_mm[0] * 1e-3

    # y-range: scan across the specimen depth, leaving a small margin
    margin = 2e-3
    y_min = -SPECIMEN_DEPTH / 2 + margin
    y_max =  SPECIMEN_DEPTH / 2 - margin
    y_range = y_max - y_min

    # All y-positions at finest spacing
    n_finest = int(y_range / finest_spacing_m) + 1
    y_positions_finest = np.linspace(y_min, y_max, n_finest)

    print(f"\n{'#'*70}")
    print(f"# TRANSLATION B-SCAN STACKING TEST")
    print(f"# Spacings: {spacings_mm} mm")
    print(f"# Finest: {spacings_mm[0]} mm → {n_finest} positions")
    print(f"{'#'*70}\n")

    # ── Step 1: Generate grain structure ──
    if not args.skip_sim:
        print("Generating grain structure ...")
        grain_vol = generate_grain_structure(
            thickness=SPECIMEN_THICKNESS,
            width=SPECIMEN_WIDTH,
            depth=SPECIMEN_DEPTH,
            background_material=ALUMINUM,
            mean_grain_size_m=MEAN_GRAIN_SIZE,
            impedance_variation=IMP_VARIATION,
            wavespeed_variation=0.005,
            voxel_size_m=VOXEL_SIZE,
        )
        print(f"  Volume shape: {grain_vol.shape}")

        # ── Step 2: Simulate at finest spacing ──
        bscans_all, x_img, z_img = run_translation_scan(
            grain_vol, y_positions_finest, OUTPUT_DIR)

        # Save x_img, z_img for reconstruction
        np.save(os.path.join(OUTPUT_DIR, 'x_img.npy'), x_img)
        np.save(os.path.join(OUTPUT_DIR, 'z_img.npy'), z_img)
    else:
        print("Loading saved B-scans ...")
        meta = np.load(os.path.join(OUTPUT_DIR, 'scan_meta.npy'),
                       allow_pickle=True).item()
        y_positions_finest = meta['y_positions_m']
        n_finest = len(y_positions_finest)
        bscans_list = []
        for i in range(n_finest):
            bscans_list.append(np.load(os.path.join(OUTPUT_DIR, f'bscan_{i:04d}.npy')))
        bscans_all = np.stack(bscans_list)
        x_img = np.load(os.path.join(OUTPUT_DIR, 'x_img.npy'))
        z_img = np.load(os.path.join(OUTPUT_DIR, 'z_img.npy'))
        grain_vol = None

    # Load ground truth
    from reconstruct_3d import load_ground_truth_file
    grain_vol_loaded = load_ground_truth_file(
        os.path.join(OUTPUT_DIR, 'ground_truth.npz'))

    # ── Step 3: Dense output y-grid for reconstruction ──
    y_out = np.linspace(y_positions_finest[0], y_positions_finest[-1],
                        max(n_finest, 60))

    # Ground truth on the same grid
    gt = extract_ground_truth_contrast(
        grain_vol_loaded, z_img, y_out, x_img,
        background_Z=ALUMINUM.Z_L,
    )

    # ── Step 4: Reconstruct at each spacing ──
    print(f"\n{'='*60}")
    print(f"  RECONSTRUCTION & METRICS")
    print(f"{'='*60}")
    print(f"{'Spacing':>10s} {'Interp':>8s} {'SSIM':>8s} {'Pearson':>8s} {'NRMSE':>8s}")
    print(f"{'-'*46}")

    all_results = {}
    for sp_mm in spacings_mm:
        sp_m = sp_mm * 1e-3
        # Subsample from finest
        step_ratio = max(1, round(sp_m / finest_spacing_m))
        indices = np.arange(0, n_finest, step_ratio)
        y_sub = y_positions_finest[indices]
        bscans_sub = bscans_all[indices]

        result_entry = {}

        for method in ['linear', 'cubic']:
            if len(y_sub) < 4 and method == 'cubic':
                # Need at least 4 points for cubic
                result_entry[f'volume_{method}'] = result_entry['volume_linear']
                result_entry[f'metrics_{method}'] = result_entry['metrics_linear']
                continue

            vol = reconstruct_volume(bscans_sub, y_sub, y_out, method=method)
            metrics = compute_metrics(vol, gt)

            result_entry[f'volume_{method}'] = vol
            result_entry[f'metrics_{method}'] = metrics

            print(f"{sp_mm:>8.1f}mm {method:>8s} {metrics['ssim']:>8.4f} "
                  f"{metrics['pearson_r']:>8.4f} {metrics['nrmse']:>8.4f}")

        all_results[sp_mm] = result_entry

    # ── Step 5: Plot ──
    print()
    out_fig = os.path.join(os.path.dirname(OUTPUT_DIR),
                           'translation_reconstruction_test.png')
    plot_results(all_results, gt, z_img, y_out, x_img, out_fig)

    # ── Optional: napari ──
    if args.show_napari:
        try:
            import napari
            best_sp = spacings_mm[0]
            best_vol = all_results[best_sp]['volume_linear']
            viewer = napari.Viewer(title='Translation Reconstruction')
            viewer.add_image(gt, name='Ground Truth', colormap='hot')
            viewer.add_image(np.nan_to_num(best_vol), name=f'Recon {best_sp}mm',
                             colormap='hot', opacity=0.7)
            napari.run()
        except ImportError:
            print("  napari not installed — skipping 3D viewer")

    print(f"\n{'#'*70}")
    print(f"# DONE")
    print(f"{'#'*70}\n")


if __name__ == '__main__':
    main()
