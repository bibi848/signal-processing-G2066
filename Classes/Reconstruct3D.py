"""
3D Volume Reconstruction — minimal module.

Inverse Radon reconstruction from a rotational B-scan stack. One
``iradon`` call per z-plane; no sinogram pre-construction, no
preprocessing, no alternative methods.
"""

import os
import re
import warnings
import numpy as np
from typing import Optional

from skimage.transform import iradon


# ── Load B-scans ─────────────────────────────────────────────────────

def _load_meta(scan_dir: str) -> dict:
    meta = np.load(os.path.join(scan_dir, 'scan_meta.npy'),
                   allow_pickle=True).item()
    if 'tfm_z_start_m' not in meta:
        warnings.warn("scan_meta.npy missing tfm fields — using defaults.")
        meta.setdefault('tfm_z_start_m', 10e-3)
        meta.setdefault('tfm_z_end_m', meta['specimen_thickness_m'] - 5e-3)
        meta.setdefault('array_aperture_m', meta['specimen_width_m'] * 0.7)
    return meta


def load_bscans(scan_dir: str) -> tuple:
    """Load dB B-scan stack: (n_scans, n_z, n_lateral)."""
    meta = _load_meta(scan_dir)
    files = sorted(f for f in os.listdir(scan_dir)
                   if re.match(r'^bscan_\d{4}\.npy$', f))
    if not files:
        raise FileNotFoundError(f"No bscan_*.npy files in {scan_dir}")
    bscans = np.stack([np.load(os.path.join(scan_dir, f)) for f in files],
                      axis=0).astype(np.float32)
    meta.setdefault('tfm_n_pixels', bscans.shape[1])
    print(f"Loaded {bscans.shape[0]} B-scans, shape {bscans.shape[1]}x{bscans.shape[2]}")
    return bscans, meta


def load_bscans_complex(scan_dir: str) -> tuple:
    """Load complex analytic B-scan stack: (n_scans, n_z, n_lateral)."""
    meta = _load_meta(scan_dir)
    files = sorted(f for f in os.listdir(scan_dir)
                   if re.match(r'^bscan_complex_\d{4}\.npy$', f))
    if not files:
        raise FileNotFoundError(
            f"No bscan_complex_*.npy in {scan_dir}. "
            f"Re-run with img_output='complex'.")
    bscans = np.stack([np.load(os.path.join(scan_dir, f)) for f in files],
                      axis=0).astype(np.complex64)
    meta.setdefault('tfm_n_pixels', bscans.shape[1])
    print(f"Loaded {bscans.shape[0]} complex B-scans, shape {bscans.shape[1]}x{bscans.shape[2]}")
    return bscans, meta


def has_complex_bscans(scan_dir: str) -> bool:
    return any(re.match(r'^bscan_complex_\d{4}\.npy$', f)
               for f in os.listdir(scan_dir))


# ── Inverse Radon ────────────────────────────────────────────────────

def reconstruct_volume(
    bscans: np.ndarray,
    angles_deg: np.ndarray,
    filter_name: str = 'hann',
    circle: bool = False,
    output_size: Optional[int] = None,
) -> np.ndarray:
    """
    Inverse Radon per z-plane.

    Args:
        bscans: (n_scans, n_z, n_lateral) real or complex.
        angles_deg: (n_scans,) projection angles in degrees.

    Returns:
        (n_z, output_size, output_size) float32 magnitude volume.
    """
    n_scans, n_z, n_lat = bscans.shape
    if output_size is None:
        output_size = n_lat

    is_complex = np.iscomplexobj(bscans)
    dtype = np.complex64 if is_complex else np.float32
    volume = np.zeros((n_z, output_size, output_size), dtype=dtype)

    print(f"Reconstructing {n_z} z-slices ({n_lat} detectors, {n_scans} angles, "
          f"filter='{filter_name}', circle={circle}, complex={is_complex})")

    for z in range(n_z):
        sino = bscans[:, z, :].T  # (n_lateral, n_scans)
        if is_complex:
            re = iradon(sino.real, theta=angles_deg,
                        filter_name=filter_name, interpolation='linear',
                        circle=circle, output_size=output_size)
            im = iradon(sino.imag, theta=angles_deg,
                        filter_name=filter_name, interpolation='linear',
                        circle=circle, output_size=output_size)
            volume[z] = (re + 1j * im).astype(np.complex64)
        else:
            recon = iradon(sino, theta=angles_deg,
                           filter_name=filter_name, interpolation='linear',
                           circle=circle, output_size=output_size)
            volume[z] = recon.astype(np.float32)

    # iradon returns image-convention rows (y increases downward);
    # flip so axis 1 matches ascending y_coords.
    volume = volume[:, ::-1, :]
    return volume


# ── Coordinates ──────────────────────────────────────────────────────

def compute_reconstruction_coords(meta: dict, output_size: int) -> tuple:
    """Physical (z, y, x) coordinate axes in metres."""
    full_size = meta.get('tfm_n_pixels', output_size)
    half_ap = meta['array_aperture_m'] / 2.0
    half_extent = half_ap * output_size / full_size
    z_coords = np.linspace(meta['tfm_z_start_m'], meta['tfm_z_end_m'],
                           meta['tfm_n_pixels'])
    x_coords = np.linspace(-half_extent, half_extent, output_size)
    y_coords = np.linspace(-half_extent, half_extent, output_size)
    return z_coords, y_coords, x_coords


# ── Napari viewer ────────────────────────────────────────────────────

def view_reconstruction_napari(
    recon: np.ndarray,
    ground_truth: Optional[np.ndarray],
    z_coords: np.ndarray,
    y_coords: np.ndarray,
    x_coords: np.ndarray,
    metrics: Optional[dict] = None,
) -> None:
    try:
        import napari
    except ImportError:
        print("napari not installed — skipping viewer")
        return

    dz = (z_coords[-1] - z_coords[0]) / max(len(z_coords) - 1, 1) * 1e3
    dy = (y_coords[-1] - y_coords[0]) / max(len(y_coords) - 1, 1) * 1e3
    dx = (x_coords[-1] - x_coords[0]) / max(len(x_coords) - 1, 1) * 1e3
    scale = (dz, dy, dx)

    title = 'NDT 3D Reconstruction'
    if metrics is not None:
        title += f'  |  SSIM={metrics["ssim_mean"]:.3f}  r={metrics["pearson_r"]:.3f}'

    viewer = napari.Viewer(title=title)
    viewer.add_image(recon, name='Reconstruction',
                     scale=scale, colormap='hot', opacity=0.9)
    if ground_truth is not None:
        viewer.add_image(ground_truth, name='Ground truth',
                         scale=scale, colormap='hot', opacity=0.9, visible=False)
    viewer.dims.axis_labels = ('z (mm)', 'y (mm)', 'x (mm)')
    napari.run()


# ── Top-level pipeline ───────────────────────────────────────────────

def reconstruct_scan(
    scan_dir: str,
    filter_name: str = 'shepp-logan',
    circle: bool = False,
    output_size: Optional[int] = None,
    show_napari: bool = False,
    output_dir: Optional[str] = None,
) -> np.ndarray:
    """Load → iradon per z-plane → save → optional napari."""
    if output_dir is None:
        output_dir = scan_dir

    if has_complex_bscans(scan_dir):
        bscans, meta = load_bscans_complex(scan_dir)
    else:
        bscans_db, meta = load_bscans(scan_dir)
        bscans = np.float32(10.0 ** (bscans_db / 20.0))

    angles_deg = np.degrees(meta['angles_rad'])
    print(f"Angular range: {angles_deg[0]:+.1f} to {angles_deg[-1]:+.1f} deg "
          f"({len(angles_deg)} projections)")

    volume = reconstruct_volume(
        bscans, angles_deg,
        filter_name=filter_name, circle=circle, output_size=output_size,
    )

    recon_path = os.path.join(output_dir, 'recon_volume.npy')
    np.save(recon_path, volume)
    print(f"Saved: {recon_path}")

    if show_napari:
        z, y, x = compute_reconstruction_coords(meta, volume.shape[1])
        view_reconstruction_napari(volume, None, z, y, x)

    return volume
