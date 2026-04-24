"""
3D volume reconstruction via filtered backprojection.

Faithful to Driver (2024), eq. 16:
    f(x, y) = ∫₀^π F1⁻¹{P(ω, φ) · |ω|} dφ

One skimage.iradon call per depth slice, ramp filter, nothing else.
"""

import os
import re
import numpy as np
from skimage.transform import iradon


def load_meta(scan_dir):
    return np.load(os.path.join(scan_dir, 'scan_meta.npy'),
                   allow_pickle=True).item()


def _load_bscans(scan_dir):
    """Load TFM amplitude stack as (n_scans, n_z, n_lateral) float32.

    Uses complex analytic files when present (takes magnitude) else the
    real dB files as-is.
    """
    complex_files = sorted(f for f in os.listdir(scan_dir)
                           if re.match(r'^bscan_complex_\d{4}\.npy$', f))
    real_files = sorted(f for f in os.listdir(scan_dir)
                        if re.match(r'^bscan_\d{4}\.npy$', f))
    if complex_files:
        stack = np.stack([np.abs(np.load(os.path.join(scan_dir, f)))
                          for f in complex_files], axis=0)
    elif real_files:
        stack = np.stack([np.load(os.path.join(scan_dir, f))
                          for f in real_files], axis=0)
    else:
        raise FileNotFoundError(f"No bscan_*.npy files in {scan_dir}")
    return stack.astype(np.float32)


def reconstruct_scan(scan_dir, show_napari=False):
    """Load B-scans → iradon per z-plane with |ω| filter → save → (napari)."""
    meta = load_meta(scan_dir)
    bscans = _load_bscans(scan_dir)                       # (n_scans, n_z, n_lat)
    angles_deg = np.degrees(meta['angles_rad'])

    n_scans, n_z, n_lat = bscans.shape
    volume = np.zeros((n_z, n_lat, n_lat), dtype=np.float32)
    for z in range(n_z):
        sino = bscans[:, z, :].T                          # (n_lat, n_scans)
        recon = iradon(sino, theta=angles_deg, filter_name='ramp',
                       circle=True, output_size=n_lat)
        volume[z] = recon[::-1, :].astype(np.float32)     # iradon y → ascending

    np.save(os.path.join(scan_dir, 'recon_volume.npy'), volume)
    print(f"Saved recon_volume.npy  shape={volume.shape}")

    if show_napari:
        view_napari(volume, meta)
    return volume


def view_napari(volume, meta):
    try:
        import napari
    except ImportError:
        print("napari not installed — skipping viewer")
        return
    n_z, _, n_x = volume.shape
    half = meta['array_aperture_m'] / 2
    dz  = (meta['tfm_z_end_m'] - meta['tfm_z_start_m']) / max(n_z - 1, 1) * 1e3
    dxy = (2 * half) / max(n_x - 1, 1) * 1e3
    viewer = napari.Viewer(title='3D Radon Reconstruction')
    viewer.add_image(volume, name='Reconstruction',
                     scale=(dz, dxy, dxy), colormap='hot')
    viewer.dims.axis_labels = ('z (mm)', 'y (mm)', 'x (mm)')
    napari.run()
