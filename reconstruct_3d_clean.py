"""
3D Volume Reconstruction from Rotational B-scans (Clean Implementation)
========================================================================

Reconstructs a 3D volume from a stack of 2D TFM B-scans acquired at
different rotation angles around the depth (z) axis.

Physics
-------
A 1D phased array lies on the specimen surface. At rotation angle theta,
the TFM image gives intensity I(s, z) where:
    s = x*cos(theta) + y*sin(theta)   (lateral projection coordinate)
    z = depth into the specimen

At each fixed depth z, the 1D lateral profile I(s, z, theta) is a Radon
projection of the 2D cross-section I(x, y, z). Collecting profiles across
all angles forms a sinogram. Applying the inverse Radon transform (iradon)
to each sinogram recovers the cross-section. Stacking all depths gives
the full 3D volume.

Pipeline (when run directly)
----------------------------
1. Generate a synthetic 3D specimen with defects using the physics engine
2. Simulate rotational FMC acquisition at multiple angles
3. Compute TFM B-scans for each angle
4. Build sinograms from the B-scan stack
5. Apply inverse Radon transform slice-by-slice
6. Save the reconstructed volume + diagnostic PNGs
"""

import os
import re
import sys
import warnings
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal.windows import tukey
from skimage.transform import iradon


# ---------------------------------------------------------------------------
#  Configuration
# ---------------------------------------------------------------------------

@dataclass
class ReconConfig:
    """Parameters controlling the reconstruction pipeline."""

    filter_name: str = 'ramp'
    """FBP filter for iradon. Options: 'ramp', 'shepp-logan', 'hamming', 'hann'.
    'ramp' is the pure |ω| filter from the paper (Driver 2024, eq. 16)."""

    circle: bool = True
    """If True, iradon limits the reconstruction to the inscribed circle.
    Only the inscribed circle contains valid information — corners contain
    backprojection artifacts."""

    output_size: Optional[int] = None
    """Side length of the reconstructed x-y grid. Default: auto from iradon."""

    taper_fraction: float = 0.0
    """Fraction of each lateral edge tapered to zero (Tukey window).
    Set >0 only if edge artifacts appear."""

    subtract_angular_mean: bool = True
    """If True, subtract the mean across angles from each sinogram row.
    Removes rotationally symmetric features (wall echoes) that would
    otherwise appear as ring artifacts around the rotation axis."""

    rolloff_fraction: float = 0.0
    """Circle apodisation rolloff. 0.0 = disabled (matches MATLAB).
    Only relevant when circle=True."""


# ---------------------------------------------------------------------------
#  Section 2: Data Loading
# ---------------------------------------------------------------------------

def load_precomputed(scan_dir: str) -> Tuple[np.ndarray, dict, str]:
    """
    Load a stack of pre-computed B-scans and metadata from a scan directory.

    Detects data type automatically:
      1. Complex analytic (bscan_complex_NNNN.npy) -> 'complex'
      2. Linear envelope or dB (bscan_NNNN.npy) -> 'linear' or 'db'

    Args:
        scan_dir: Directory containing bscan_*.npy and scan_meta.npy.

    Returns:
        bscans:    (n_scans, n_z, n_lateral) array
        meta:      dict with scan parameters
        data_type: 'complex', 'linear', or 'db'
    """
    meta_path = os.path.join(scan_dir, 'scan_meta.npy')
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"scan_meta.npy not found in {scan_dir}")
    meta = np.load(meta_path, allow_pickle=True).item()

    # Backward compatibility defaults
    meta.setdefault('tfm_z_start_m', 10e-3)
    meta.setdefault('tfm_z_end_m', meta.get('specimen_thickness_m', 40e-3) - 5e-3)
    meta.setdefault('array_aperture_m', meta.get('specimen_width_m', 38e-3) * 0.7)

    # Try complex files first
    complex_files = sorted(
        f for f in os.listdir(scan_dir)
        if re.match(r'^bscan_complex_\d{4}\.npy$', f)
    )
    if complex_files:
        bscans = np.stack(
            [np.load(os.path.join(scan_dir, f)) for f in complex_files],
            axis=0,
        ).astype(np.complex64)

        meta.setdefault('tfm_n_pixels', bscans.shape[2])
        data_type = 'complex'
        print(f"Loaded {bscans.shape[0]} complex B-scans, "
              f"shape per frame: {bscans.shape[1]}x{bscans.shape[2]}")
        return bscans, meta, data_type

    # Fall back to real-valued B-scans
    real_files = sorted(
        f for f in os.listdir(scan_dir)
        if re.match(r'^bscan_\d{4}\.npy$', f)
    )
    if not real_files:
        raise FileNotFoundError(f"No bscan_*.npy files found in {scan_dir}")

    bscans = np.stack(
        [np.load(os.path.join(scan_dir, f)) for f in real_files],
        axis=0,
    ).astype(np.float32)

    meta.setdefault('tfm_n_pixels', bscans.shape[2])
    data_type = meta.get('data_format', 'db')
    if data_type not in ('linear', 'linear_envelope'):
        data_type = 'db'
    else:
        data_type = 'linear'

    print(f"Loaded {bscans.shape[0]} B-scans ({data_type}), "
          f"shape per frame: {bscans.shape[1]}x{bscans.shape[2]}")
    return bscans, meta, data_type


def load_fmc_and_compute_tfm(
    input_dir: str,
    c: Optional[float] = None,
    z_start: float = 0.0,
    z_end: Optional[float] = None,
    x_pixels: int = 400,
    z_pixels: int = 400,
) -> Tuple[np.ndarray, dict, str]:
    """
    Load raw FMC data from scan subfolders, compute TFM, return B-scans.

    Each subfolder must contain: array_geometry.csv, time.csv, tx_rx.csv,
    time_data.h5, metadata.csv.

    Args:
        input_dir: Directory containing scan subfolders.
        c:         Wave speed (m/s). If None, inferred from directory name.
        z_start:   TFM start depth (m).
        z_end:     TFM end depth (m). If None, inferred from time axis.
        x_pixels:  Lateral pixel count.
        z_pixels:  Depth pixel count.

    Returns:
        bscans:    (n_scans, n_z, n_lateral) float32 linear envelope
        meta:      dict with scan parameters
        data_type: 'linear'
    """
    import pandas as pd
    import h5py

    # Import run_tfm from the project's fmc_to_npy module
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from fmc_to_npy import run_tfm

    def _natural_key(s):
        return [int(c_) if c_.isdigit() else c_.lower()
                for c_ in re.split(r'(\d+)', s)]

    scan_folders = sorted(
        [f for f in os.listdir(input_dir)
         if os.path.isdir(os.path.join(input_dir, f))],
        key=_natural_key,
    )
    if not scan_folders:
        raise FileNotFoundError(f"No scan subfolders found in {input_dir}")

    print(f"Found {len(scan_folders)} scans in {input_dir}")

    bscans = []
    xc_ref = None

    for i, fol in enumerate(scan_folders):
        folder_path = os.path.join(input_dir, fol)

        metadata = pd.read_csv(os.path.join(folder_path, "metadata.csv"))
        time_sec = pd.read_csv(
            os.path.join(folder_path, "time.csv"))["time_seconds"].values
        tx_rx = pd.read_csv(os.path.join(folder_path, "tx_rx.csv"))
        geometry = pd.read_csv(
            os.path.join(folder_path, "array_geometry.csv"))

        with h5py.File(os.path.join(folder_path, "time_data.h5"), "r") as h5f:
            time_data = h5f["time_data"][:]

        tx = tx_rx["tx"].values.astype(int)
        rx = tx_rx["rx"].values.astype(int)
        xc = geometry["el_xc"].values
        zc = geometry["el_zc"].values

        # First scan: set up parameters
        if i == 0:
            xc_ref = xc
            if c is None:
                dirname = os.path.basename(input_dir).lower()
                if 'al' in dirname:
                    c = 6320.0
                elif 'cu' in dirname:
                    c = 4700.0
                elif 'steel' in dirname:
                    c = 5960.0
                else:
                    c = 6320.0
                print(f"  Wave speed: {c:.0f} m/s (inferred)")

            if z_end is None:
                z_end = min(c * time_sec[-1] / 2.0, 50e-3)

            x_img = np.linspace(xc.min(), xc.max(), x_pixels)
            z_img = np.linspace(z_start, z_end, z_pixels)
            print(f"  Depth: {z_start*1e3:.1f} - {z_end*1e3:.1f} mm")
            print(f"  Grid: {z_pixels} x {x_pixels} pixels\n")

        envelope = run_tfm(
            time_data, time_sec, tx, rx, xc, zc, c,
            x_img, z_img, engine='python',
        )
        bscans.append(envelope)
        print(f"  [{i+1}/{len(scan_folders)}] {fol}  "
              f"max={envelope.max():.4f}")

    bscans = np.stack(bscans, axis=0).astype(np.float32)
    n_scans = bscans.shape[0]
    angles_rad = np.linspace(0, np.pi, n_scans, endpoint=False)

    meta = {
        'n_scans': n_scans,
        'angles_rad': angles_rad.astype(np.float64),
        'tfm_z_start_m': float(z_start),
        'tfm_z_end_m': float(z_end),
        'tfm_n_pixels': int(x_pixels),
        'array_aperture_m': float(xc_ref.max() - xc_ref.min()),
        'specimen_width_m': float(xc_ref.max() - xc_ref.min()),
        'specimen_thickness_m': float(z_end),
        'wave_speed': float(c),
        'data_format': 'linear_envelope',
    }

    print(f"\nComputed {n_scans} B-scans, shape: {bscans.shape[1]}x{bscans.shape[2]}")
    return bscans, meta, 'linear'


# ---------------------------------------------------------------------------
#  Section 3: Preprocessing
# ---------------------------------------------------------------------------

def preprocess(
    bscans: np.ndarray,
    meta: dict,
    data_type: str,
    config: ReconConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare B-scans for Radon reconstruction.

    For complex data (matching MATLAB): no preprocessing at all.
    The raw complex TFM data goes directly into iradon.

    For envelope/dB data: convert to linear, optionally taper and normalize.

    Args:
        bscans:    (n_scans, n_z, n_lateral) raw B-scans
        meta:      scan metadata dict
        data_type: 'complex', 'linear', or 'db'
        config:    ReconConfig instance

    Returns:
        bscans_ready: preprocessed B-scans, same shape
        angles_deg:   (n_scans,) projection angles in degrees
    """
    bscans = bscans.copy()
    angles_deg = np.degrees(meta['angles_rad'])

    if data_type == 'complex':
        # Match MATLAB: no preprocessing on complex data
        print(f"Complex data — no preprocessing (matches MATLAB). "
              f"Angles {angles_deg[0]:+.1f} to {angles_deg[-1]:+.1f} deg "
              f"({len(angles_deg)} projections)")
        return bscans, angles_deg

    # dB to linear conversion
    if data_type == 'db':
        vmin_db = meta.get('vmin_db', None)
        db_range = bscans.max() - bscans.min()
        if db_range < 20:
            warnings.warn(
                f"B-scan dB range is only {db_range:.1f} dB. "
                "Heavily clipped data will produce poor reconstructions."
            )
        bscans = np.float32(10.0 ** (bscans / 20.0))
        if vmin_db is not None:
            floor = np.float32(10.0 ** (vmin_db / 20.0))
            bscans = np.maximum(bscans - floor, 0.0)

    # Optional lateral taper
    if config.taper_fraction > 0:
        n_lateral = bscans.shape[2]
        window = tukey(n_lateral, alpha=2 * config.taper_fraction)
        window = window.astype(np.float32)
        bscans = bscans * window[np.newaxis, np.newaxis, :]

    # Optional normalize
    global_max = np.abs(bscans).max()
    if global_max > 0:
        bscans = bscans / global_max

    print(f"Preprocessed: {data_type} data, "
          f"angles {angles_deg[0]:+.1f} to {angles_deg[-1]:+.1f} deg "
          f"({len(angles_deg)} projections)")
    return bscans, angles_deg


# ---------------------------------------------------------------------------
#  Section 4: Sinogram Construction and Reconstruction
# ---------------------------------------------------------------------------

def build_sinograms(bscans: np.ndarray) -> np.ndarray:
    """
    Rearrange B-scan stack into sinograms for iradon.

    Each B-scan has shape (n_z, n_lateral). At each depth z, the lateral
    profile across all angles forms one sinogram.

    Args:
        bscans: (n_scans, n_z, n_lateral)

    Returns:
        sinograms: (n_z, n_lateral, n_scans)
            sinograms[z] is the sinogram at depth z, with shape
            (n_detectors, n_angles) — ready for iradon.
    """
    # (n_scans, n_z, n_lateral) -> (n_z, n_lateral, n_scans)
    return np.transpose(bscans, (1, 2, 0))


def reconstruct_volume(
    sinograms: np.ndarray,
    angles_deg: np.ndarray,
    config: ReconConfig,
    is_complex: bool = False,
) -> np.ndarray:
    """
    Inverse Radon reconstruction, applied slice-by-slice at each depth z.

    Matches the MATLAB implementation:
        for iz = 1:length(z)
            tmp = squeeze(result_tfm(iz,:,:));
            tmp_xy = iradon(real(tmp), theta, 'pchip', 'Shepp-Logan')
                   + 1j * iradon(imag(tmp), theta, 'pchip', 'Shepp-Logan');
            result_tfm_3D(:,:,iz) = tmp_xy;
        end

    Key: no abs(), no circle constraint, no preprocessing on complex data.
    The complex result is kept as-is; the caller takes abs() for display.

    Args:
        sinograms:  (n_z, n_lateral, n_angles) sinogram stack
        angles_deg: (n_angles,) projection angles in degrees
        config:     ReconConfig instance
        is_complex: True if sinograms contain complex data

    Returns:
        volume: (n_z, Ny, Ny) complex64 (if is_complex) or float32
                Ny is determined by iradon (may differ from n_lateral).
    """
    n_z, n_det, n_ang = sinograms.shape

    # Let iradon determine output size on the first slice (matches MATLAB),
    # unless explicitly overridden
    output_size = config.output_size  # None = let iradon decide

    print(f"Reconstructing {n_z} depth slices "
          f"({n_det} detectors, {n_ang} angles, "
          f"filter='{config.filter_name}', circle={config.circle})...")

    volume = None  # allocated after first slice determines Ny

    for z in range(n_z):
        sino = sinograms[z]  # (n_lateral, n_angles)

        # Optional: subtract angular mean to remove wall echoes
        if config.subtract_angular_mean:
            sino = sino - sino.mean(axis=1, keepdims=True)

        if is_complex:
            # iradon(real) + 1j * iradon(imag) — iradon is linear
            recon_re = iradon(
                sino.real.astype(np.float64),
                theta=angles_deg,
                filter_name=config.filter_name,
                circle=config.circle,
                output_size=output_size,
            )
            recon_im = iradon(
                sino.imag.astype(np.float64),
                theta=angles_deg,
                filter_name=config.filter_name,
                circle=config.circle,
                output_size=output_size,
            )
            recon_slice = (recon_re + 1j * recon_im).astype(np.complex64)
        else:
            recon_slice = iradon(
                sino.astype(np.float64),
                theta=angles_deg,
                filter_name=config.filter_name,
                circle=config.circle,
                output_size=output_size,
            ).astype(np.float32)

        # iradon returns image-convention rows (y-decreasing-downward);
        # flip axis 0 so axis 1 of the volume matches ascending y_coords.
        recon_slice = recon_slice[::-1, :]

        # Allocate volume on first slice (Ny determined by iradon)
        if volume is None:
            Ny = recon_slice.shape[0]
            dtype = np.complex64 if is_complex else np.float32
            volume = np.zeros((n_z, Ny, Ny), dtype=dtype)
            print(f"  iradon output size: {Ny} x {Ny}")

        volume[z] = recon_slice

        if (z + 1) % 50 == 0 or z == 0 or z == n_z - 1:
            print(f"  Slice {z+1}/{n_z}")

    print(f"  Volume shape: {volume.shape}")
    return volume


# ---------------------------------------------------------------------------
#  Section 5: Post-processing and Coordinates
# ---------------------------------------------------------------------------

def soft_circle_apodise(
    volume: np.ndarray,
    rolloff_fraction: float = 0.08,
) -> np.ndarray:
    """
    Replace iradon's hard circular mask with a smooth cosine rolloff.

    With circle=True, iradon zeros everything outside the inscribed circle,
    creating a sharp boundary that appears as a bright ring. This function
    replaces that hard edge with a smooth cosine taper.

    Args:
        volume:           (n_z, n_y, n_x)
        rolloff_fraction: fraction of radius over which the taper falls.

    Returns:
        Apodised volume, same shape.
    """
    n_z, n_y, n_x = volume.shape
    center_y = (n_y - 1) / 2.0
    center_x = (n_x - 1) / 2.0
    radius = min(n_y, n_x) / 2.0

    y, x = np.ogrid[:n_y, :n_x]
    r = np.sqrt((y - center_y) ** 2 + (x - center_x) ** 2)

    inner = radius * (1.0 - rolloff_fraction)
    mask = np.ones((n_y, n_x), dtype=np.float32)
    transition = (r > inner) & (r <= radius)
    mask[transition] = (0.5 * (1.0 + np.cos(
        np.pi * (r[transition] - inner) / (radius - inner)
    ))).astype(np.float32)
    mask[r > radius] = 0.0

    return volume * mask[np.newaxis, :, :]


def compute_coordinates(
    meta: dict,
    volume_shape: Tuple[int, int, int],
    n_lateral: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute physical coordinates for the reconstructed volume.

    Matches the MATLAB approach:
        xp = x(1:Ny); xp = xp - xp(ceil(Ny/2));
    i.e. take the first Ny points of the lateral axis and re-center.

    Args:
        meta:         scan metadata dict
        volume_shape: (n_z, Ny, Ny)
        n_lateral:    original lateral pixel count of the B-scans

    Returns:
        z_coords: (n_z,) depth coordinates in metres
        y_coords: (Ny,) y coordinates in metres (re-centered)
        x_coords: (Ny,) x coordinates in metres (re-centered)
    """
    n_z, Ny, _ = volume_shape
    z_coords = np.linspace(
        meta['tfm_z_start_m'], meta['tfm_z_end_m'], n_z)

    # Build the original lateral axis
    half_ap = meta['array_aperture_m'] / 2.0
    x_full = np.linspace(-half_ap, half_ap, n_lateral)

    # Take the first Ny points and re-center (MATLAB: xp = x(1:Ny) - x(ceil(Ny/2)))
    xp = x_full[:Ny].copy()
    xp = xp - xp[Ny // 2]   # ceil(Ny/2) in 0-based indexing = Ny // 2

    return z_coords, xp, xp.copy()


# ---------------------------------------------------------------------------
#  Section 6: Visualization and PNG Saving
# ---------------------------------------------------------------------------

def save_bscan_pngs(
    bscans: np.ndarray,
    angles_deg: np.ndarray,
    output_dir: str,
    cmap: str = 'hot',
) -> None:
    """Save each B-scan as a PNG for visual inspection."""
    os.makedirs(output_dir, exist_ok=True)
    vmin = np.percentile(np.abs(bscans), 1)
    vmax = np.percentile(np.abs(bscans), 99)
    if np.iscomplexobj(bscans):
        data = np.abs(bscans)
    else:
        data = bscans

    for i in range(data.shape[0]):
        path = os.path.join(output_dir, f'bscan_{i:04d}.png')
        plt.imsave(path, data[i], cmap=cmap, vmin=vmin, vmax=vmax,
                   origin='lower')
    print(f"Saved {data.shape[0]} B-scan PNGs to {output_dir}")


def save_sinogram_pngs(
    sinograms: np.ndarray,
    angles_deg: np.ndarray,
    output_dir: str,
    n_theta_interp: int = 360,
    cmap: str = 'hot',
) -> None:
    """
    Save sinograms at 25%, 50%, 75% depth as interpolated PNGs.

    The raw sinogram has shape (n_lateral, n_angles) which may be very
    narrow (e.g. 800 x 16). This function interpolates along the angle
    axis to produce a proper 2D image in (s, theta) space, where
    sinusoidal patterns from defects become visible.

    Args:
        sinograms:      (n_z, n_lateral, n_angles)
        angles_deg:     (n_angles,) projection angles in degrees
        output_dir:     where to save PNGs
        n_theta_interp: number of interpolated angle samples (default 360)
        cmap:           colormap
    """
    from scipy.interpolate import RegularGridInterpolator

    os.makedirs(output_dir, exist_ok=True)
    n_z, n_lateral, n_angles = sinograms.shape
    data = np.abs(sinograms) if np.iscomplexobj(sinograms) else sinograms
    vmax = np.percentile(np.abs(data), 99)

    # Interpolate along the angle axis for display
    theta_fine = np.linspace(angles_deg[0], angles_deg[-1], n_theta_interp)
    s_axis = np.arange(n_lateral)

    for frac in [0.25, 0.50, 0.75]:
        z_idx = min(int(frac * n_z), n_z - 1)
        sino_raw = data[z_idx]  # (n_lateral, n_angles)

        # Interpolate: (n_lateral, n_angles) -> (n_lateral, n_theta_interp)
        interp = RegularGridInterpolator(
            (s_axis, angles_deg), sino_raw.astype(np.float64),
            method='cubic', bounds_error=False, fill_value=0.0,
        )
        ss, tt = np.meshgrid(s_axis, theta_fine, indexing='ij')
        sino_interp = interp((ss, tt)).astype(np.float32)

        # Save with proper axes using imshow
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(
            sino_interp, cmap=cmap, vmin=0, vmax=vmax,
            origin='lower', aspect='auto',
            extent=[angles_deg[0], angles_deg[-1], 0, n_lateral],
        )
        ax.set_xlabel('Angle (deg)')
        ax.set_ylabel('Lateral position s (px)')
        ax.set_title(f'Sinogram at z={z_idx}/{n_z} ({frac:.0%} depth)')
        fig.savefig(
            os.path.join(output_dir, f'sinogram_z{z_idx:04d}.png'),
            dpi=150, bbox_inches='tight',
        )
        plt.close(fig)

    print(f"Saved interpolated sinogram PNGs to {output_dir}")


def save_reconstruction_pngs(
    volume: np.ndarray,
    output_dir: str,
    n_slices: int = 10,
    cmap: str = 'hot',
) -> None:
    """Save reconstructed x-y cross-sections at evenly spaced depths."""
    os.makedirs(output_dir, exist_ok=True)
    n_z = volume.shape[0]
    vmax = np.percentile(volume[volume > 0], 99) if volume.max() > 0 else 1.0

    for i in range(n_slices):
        z_idx = min(int((i + 0.5) / n_slices * n_z), n_z - 1)
        path = os.path.join(output_dir, f'recon_z{z_idx:04d}.png')
        plt.imsave(path, volume[z_idx], cmap=cmap, vmin=0, vmax=vmax,
                   origin='lower')
    print(f"Saved {n_slices} reconstruction PNGs to {output_dir}")


def plot_diagnostics(
    bscans: np.ndarray,
    sinograms: np.ndarray,
    volume: np.ndarray,
    angles_deg: np.ndarray,
    meta: dict,
    save_path: str,
) -> None:
    """
    Save a 3x3 diagnostic figure showing the full reconstruction pipeline.

    Row 1: Three B-scans at different angles.
    Row 2: Three sinograms at 25%, 50%, 75% depth.
    Row 3: Three reconstructed x-y cross-sections at the same depths.
    """
    n_scans = bscans.shape[0]
    n_z = sinograms.shape[0]
    bscan_data = np.abs(bscans) if np.iscomplexobj(bscans) else bscans
    sino_data = np.abs(sinograms) if np.iscomplexobj(sinograms) else sinograms

    fig, axes = plt.subplots(3, 3, figsize=(14, 12))

    # Row 1: B-scans
    bscan_indices = [0, n_scans // 2, n_scans - 1]
    b_vmax = np.percentile(bscan_data, 99)
    for i, idx in enumerate(bscan_indices):
        axes[0, i].imshow(bscan_data[idx], cmap='hot', vmin=0, vmax=b_vmax,
                          origin='lower', aspect='auto')
        axes[0, i].set_title(
            f'B-scan #{idx} ({angles_deg[idx]:+.1f}\u00b0)')
        axes[0, i].set_xlabel('Lateral (px)')
        axes[0, i].set_ylabel('Depth (px)')

    # Row 2: Sinograms (interpolated for display)
    from scipy.interpolate import RegularGridInterpolator
    fractions = [0.25, 0.50, 0.75]
    s_vmax = np.percentile(np.abs(sino_data), 99)
    n_lateral_sino = sino_data.shape[1]
    s_axis = np.arange(n_lateral_sino)
    theta_fine = np.linspace(angles_deg[0], angles_deg[-1], 360)
    for i, frac in enumerate(fractions):
        z_idx = min(int(frac * n_z), n_z - 1)
        sino_raw = sino_data[z_idx]
        interp = RegularGridInterpolator(
            (s_axis, angles_deg), sino_raw.astype(np.float64),
            method='cubic', bounds_error=False, fill_value=0.0,
        )
        ss, tt = np.meshgrid(s_axis, theta_fine, indexing='ij')
        sino_interp = interp((ss, tt))
        axes[1, i].imshow(
            sino_interp, cmap='hot', vmin=0, vmax=s_vmax,
            origin='lower', aspect='auto',
            extent=[angles_deg[0], angles_deg[-1], 0, n_lateral_sino],
        )
        axes[1, i].set_title(f'Sinogram at z={z_idx}/{n_z} ({frac:.0%})')
        axes[1, i].set_xlabel('Angle (deg)')
        axes[1, i].set_ylabel('Lateral position s (px)')

    # Row 3: Reconstructed cross-sections
    r_vmax = np.percentile(volume[volume > 0], 99) if volume.max() > 0 else 1.0
    for i, frac in enumerate(fractions):
        z_idx = min(int(frac * n_z), n_z - 1)
        axes[2, i].imshow(volume[z_idx], cmap='hot', vmin=0, vmax=r_vmax,
                          origin='lower', aspect='equal')
        axes[2, i].set_title(
            f'Reconstruction z={z_idx}/{n_z} ({frac:.0%})')
        axes[2, i].axis('off')

    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Diagnostic figure saved: {save_path}")


def view_napari(
    volume: np.ndarray,
    z_coords: np.ndarray,
    y_coords: np.ndarray,
    x_coords: np.ndarray,
) -> None:
    """Open napari viewer with the reconstructed volume."""
    try:
        import napari
    except ImportError:
        print("napari not installed -- skipping interactive viewer")
        return

    n_z = len(z_coords)
    n_y = len(y_coords)
    dz = (z_coords[-1] - z_coords[0]) / max(n_z - 1, 1) * 1e3
    dy = (y_coords[-1] - y_coords[0]) / max(n_y - 1, 1) * 1e3
    dx = (x_coords[-1] - x_coords[0]) / max(len(x_coords) - 1, 1) * 1e3
    scale = (dz, dy, dx)

    viewer = napari.Viewer(title='3D Reconstruction')
    viewer.add_image(
        volume, name='Reconstruction',
        scale=scale, colormap='hot', opacity=0.9,
    )
    viewer.dims.axis_labels = ('z - depth (mm)', 'y (mm)', 'x (mm)')
    print("napari viewer open -- close the window to continue")
    napari.run()


# ---------------------------------------------------------------------------
#  Section 7: Pipeline and CLI
# ---------------------------------------------------------------------------

def reconstruct(
    scan_dir: str,
    config: Optional[ReconConfig] = None,
    from_fmc: bool = False,
    c: Optional[float] = None,
    z_start: float = 0.0,
    z_end: Optional[float] = None,
    x_pixels: int = 400,
    z_pixels: int = 400,
    save_dir: Optional[str] = None,
    show_napari: bool = False,
) -> np.ndarray:
    """
    Full reconstruction pipeline: load -> preprocess -> reconstruct -> save.

    Args:
        scan_dir:   Directory with B-scan .npy files or FMC subfolders.
        config:     ReconConfig instance (default parameters if None).
        from_fmc:   If True, treat scan_dir as FMC data directory.
        c:          Wave speed for FMC processing (m/s).
        z_start:    TFM start depth for FMC (m).
        z_end:      TFM end depth for FMC (m).
        x_pixels:   Lateral pixels for FMC.
        z_pixels:   Depth pixels for FMC.
        save_dir:   Output directory (default: scan_dir).
        show_napari: Open interactive napari viewer.

    Returns:
        volume: (n_z, output_size, output_size) float32 reconstructed volume.
    """
    if config is None:
        config = ReconConfig()

    if save_dir is None:
        save_dir = scan_dir
    os.makedirs(save_dir, exist_ok=True)

    # --- Load data ---
    print("=" * 60)
    print("3D VOLUME RECONSTRUCTION (Clean Implementation)")
    print("=" * 60)

    if from_fmc:
        bscans, meta, data_type = load_fmc_and_compute_tfm(
            scan_dir, c=c, z_start=z_start, z_end=z_end,
            x_pixels=x_pixels, z_pixels=z_pixels,
        )
    else:
        bscans, meta, data_type = load_precomputed(scan_dir)

    is_complex = (data_type == 'complex')
    n_lateral = bscans.shape[2]  # original lateral pixel count

    # --- Save input B-scan PNGs ---
    png_dir = os.path.join(save_dir, 'pngs')
    save_bscan_pngs(
        bscans, np.degrees(meta['angles_rad']),
        os.path.join(png_dir, 'bscans'),
    )

    # --- Preprocess ---
    # For complex data this is a no-op (matches MATLAB)
    bscans_ready, angles_deg = preprocess(bscans, meta, data_type, config)

    # --- Build sinograms ---
    # result_tfm(iz, :, :) in MATLAB = sinograms[z] here
    sinograms = build_sinograms(bscans_ready)
    print(f"Sinograms shape: {sinograms.shape} "
          f"(n_z={sinograms.shape[0]}, "
          f"n_lateral={sinograms.shape[1]}, "
          f"n_angles={sinograms.shape[2]})")

    # Save sinogram PNGs
    save_sinogram_pngs(sinograms, angles_deg, os.path.join(png_dir, 'sinograms'))

    # --- Reconstruct (matches MATLAB iradon loop) ---
    volume_complex = reconstruct_volume(
        sinograms, angles_deg, config, is_complex)

    # --- Post-process ---
    if config.circle and config.rolloff_fraction > 0:
        volume_complex = soft_circle_apodise(
            np.abs(volume_complex), config.rolloff_fraction)

    # --- Take magnitude for display/saving ---
    # MATLAB keeps the complex result in result_tfm_3D; we save both
    volume_abs = np.abs(volume_complex).astype(np.float32)

    # --- Compute coordinates (matches MATLAB: xp = x(1:Ny) - x(ceil(Ny/2))) ---
    z_coords, y_coords, x_coords = compute_coordinates(
        meta, volume_abs.shape, n_lateral)

    # --- Save outputs ---
    np.save(os.path.join(save_dir, 'recon_volume.npy'), volume_abs)
    if is_complex:
        np.save(os.path.join(save_dir, 'recon_volume_complex.npy'),
                volume_complex)
    print(f"Volume saved: {os.path.join(save_dir, 'recon_volume.npy')}")

    # Save reconstruction slice PNGs (using magnitude)
    save_reconstruction_pngs(
        volume_abs, os.path.join(png_dir, 'reconstruction'))

    # Save diagnostic figure
    plot_diagnostics(
        bscans, sinograms, volume_abs, angles_deg, meta,
        os.path.join(save_dir, 'reconstruction_summary.png'),
    )

    # --- Interactive viewer ---
    if show_napari:
        view_napari(volume_abs, z_coords, y_coords, x_coords)

    print("=" * 60)
    print("DONE")
    print(f"  Volume shape:  {volume_abs.shape}")
    print(f"  Depth range:   {z_coords[0]*1e3:.1f} - {z_coords[-1]*1e3:.1f} mm")
    print(f"  Lateral range: {x_coords[0]*1e3:.1f} - {x_coords[-1]*1e3:.1f} mm")
    print(f"  Output:        {save_dir}")
    print("=" * 60)

    return volume_abs


def generate_synthetic_bscans(
    output_dir: str,
    n_scans: int = 16,
    num_elements: int = 64,
    element_pitch: float = 0.6e-3,
    frequency: float = 10e6,
    specimen_thickness: float = 50e-3,
    specimen_width: float = 50e-3,
    specimen_depth: float = 30e-3,
    defects_3d: Optional[list] = None,
    use_voxel_world: bool = True,
    mean_grain_size: float = 0.5e-3,
    impedance_variation: float = 0.025,
    tfm_z_start: float = 10e-3,
    tfm_z_end: Optional[float] = None,
    tfm_n_pixels: int = 800,
    snr_db: float = 35.0,
    max_bounces: int = 2,
) -> str:
    """
    Generate synthetic B-scans by simulating a 3D rotational scan.

    Creates a 3D specimen with defects and grain structure, then simulates
    FMC acquisition at each rotation angle and computes TFM B-scans.

    Args:
        output_dir:           Where to save bscan_*.npy and scan_meta.npy.
        n_scans:              Number of rotation angles evenly spaced over [0°, 180°) (endpoint excluded).
        num_elements:         Number of array elements.
        element_pitch:        Element spacing (m).
        frequency:            Centre frequency (Hz).
        specimen_thickness:   Specimen thickness / depth into material (m).
        specimen_width:       Specimen width along array (m).
        specimen_depth:       Specimen depth along elevation / y-axis (m).
        defects_3d:           List of 3D defect objects (SphericalDefect, etc.).
                              If None, a default spherical pore is created.
        use_voxel_world:      If True, generate Voronoi grain structure with
                              defects embedded as impedance contrasts.
        mean_grain_size:      Mean grain diameter for Voronoi generation (m).
        impedance_variation:  Per-grain impedance spread (fraction, e.g. 0.025 = +/-2.5%).
        tfm_z_start:          TFM imaging start depth (m).
        tfm_z_end:            TFM imaging end depth (m). Default: thickness - 5 mm.
        tfm_n_pixels:         TFM pixel grid size (square).
        snr_db:               Signal-to-noise ratio for FMC noise (dB).
        max_bounces:          Maximum number of ray bounces.

    Returns:
        Path to the output directory containing the generated B-scans.
    """
    import time as timer

    # Import the synthetic engine modules
    script_dir = os.path.dirname(os.path.abspath(__file__))
    synth_dir = os.path.join(script_dir, 'SYNTHETIC DATA')
    if synth_dir not in sys.path:
        sys.path.insert(0, synth_dir)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    from engine.config import SimulationConfig, SpecimenConfig, ArrayConfig, ScanPlanConfig
    from engine.geometry import Specimen3D, SphericalDefect, CylindricalDefect
    from engine.fmc_engine import FMCEngine
    from engine.materials import ALUMINUM
    from engine.microstructure import generate_grain_structure, embed_geometric_defects

    # Import TFM and signal processing from run_engine
    sys.path.insert(0, synth_dir)
    from run_engine import (
        reconstruct_tfm, add_noise, apply_bandpass_filter, _elevation_offsets,
    )

    print(f"\n{'#'*70}")
    print(f"# SYNTHETIC DATA GENERATION")
    print(f"{'#'*70}\n")

    # ---- Specimen ----
    specimen = Specimen3D(
        thickness=specimen_thickness,
        width=specimen_width,
        depth=specimen_depth,
    )
    print(f"Specimen: {specimen_thickness*1e3:.0f} x "
          f"{specimen_width*1e3:.0f} x {specimen_depth*1e3:.0f} mm")

    # ---- Defects ----
    if defects_3d is None:
        defects_3d = [
            SphericalDefect(
                center_z=25e-3, center_x=0.0, center_y=0.0, radius=2e-3),
            CylindricalDefect(
                center_z=15e-3, center_x=8e-3, radius=1e-3,
                y_start=-specimen.depth / 2, y_end=specimen.depth / 2),
        ]
    print(f"Defects: {len(defects_3d)}")
    for d in defects_3d:
        print(f"  - {d}")

    # ---- Voxel grain structure ----
    wavelength = ALUMINUM.c_L / frequency
    voxel_size = wavelength / 3
    voxel_volume = None

    if use_voxel_world:
        print(f"\nGenerating voxel grain structure "
              f"(lambda = {wavelength*1e3:.2f} mm, voxel = {voxel_size*1e3:.2f} mm)...")
        t0 = timer.time()
        grain_vol = generate_grain_structure(
            thickness=specimen.thickness,
            width=specimen.width,
            depth=specimen.depth,
            background_material=ALUMINUM,
            mean_grain_size_m=mean_grain_size,
            impedance_variation=impedance_variation,
            wavespeed_variation=0.005,
            voxel_size_m=voxel_size,
        )
        voxel_volume = embed_geometric_defects(grain_vol, defects_3d)
        print(f"  Voxel volume shape: {voxel_volume.shape}  "
              f"({timer.time() - t0:.1f}s)")

    # ---- Simulation config ----
    # Angles span [0, π) with endpoint excluded — paper convention, no
    # duplicate projection between first and last frame.
    scan_plan = ScanPlanConfig(n_scans=n_scans)
    cfg = SimulationConfig(
        specimen=SpecimenConfig(thickness=specimen.thickness, width=specimen.width),
        array=ArrayConfig(
            num_elements=num_elements,
            element_pitch=element_pitch,
            frequency=frequency,
            element_height=5e-3,  # 5 mm elevation (from array_geometry.csv)
        ),
        scan_plan=scan_plan,
        max_bounces=max_bounces,
        mode_conversion=False,
    )
    print(cfg.summary())

    # ---- Run scan ----
    os.makedirs(output_dir, exist_ok=True)
    angles = scan_plan.angles
    half_w = cfg.array.aperture / 2

    if tfm_z_end is None:
        tfm_z_end = specimen.thickness - 5e-3

    # Save metadata
    meta = {
        'n_scans': n_scans,
        'angles_rad': angles,
        'angle_step_rad': scan_plan.angle_step,
        'specimen_thickness_m': specimen.thickness,
        'specimen_width_m': specimen.width,
        'specimen_depth_m': specimen.depth,
        'tfm_z_start_m': tfm_z_start,
        'tfm_z_end_m': tfm_z_end,
        'tfm_n_pixels': tfm_n_pixels,
        'array_aperture_m': cfg.array.aperture,
        'has_complex_data': True,
    }
    np.save(os.path.join(output_dir, 'scan_meta.npy'), meta, allow_pickle=True)

    # Born scattering grids
    assert cfg.material is not None
    gate_z = cfg.material.c_L * 2e-6 / 2
    born_z_start = max(gate_z * 1.2, 1e-3)
    born_step = voxel_volume.voxel_size if voxel_volume is not None else 5e-4
    born_z_grid = np.linspace(
        born_z_start, specimen.thickness,
        max(2, int((specimen.thickness - born_z_start) / born_step) + 1))
    born_l_grid = np.linspace(
        -half_w, half_w,
        max(2, int(cfg.array.aperture / born_step) + 1))

    print(f"\n{'='*60}")
    print(f"  3D ROTATIONAL SCAN  --  {n_scans} frames")
    print(f"  theta = [{np.degrees(angles[0]):.1f} deg, "
          f"{np.degrees(angles[-1]):.1f} deg]  "
          f"step = {np.degrees(scan_plan.angle_step):.2f} deg")
    print(f"  Output -> {output_dir}")
    print(f"{'='*60}\n")

    geom_defects = [] if use_voxel_world else defects_3d

    for i, theta in enumerate(angles):
        t0 = timer.time()
        print(f"  Frame {i+1:>3}/{n_scans}  "
              f"(theta = {np.degrees(theta):+.1f} deg)", end="  ")

        # Slice 3D defects into 2D for this angle
        engine = FMCEngine(cfg)
        active = 0
        dy_offsets = _elevation_offsets(
            cfg.array.element_height, cfg.array.n_elevation_slices)
        for d3 in geom_defects:
            for dy in dy_offsets:
                d2 = d3.slice_at_angle(theta, dy_offset=dy)
                if d2 is not None:
                    engine.add_defect(d2)
                    active += 1

        # Born scatterers from voxel volume
        n_born = 0
        if voxel_volume is not None:
            z_s, x_s, amp_s = voxel_volume.extract_born_scatterers(
                theta, born_z_grid, born_l_grid,
                background_Z=cfg.material.Z_L,
                threshold=0.005,
                element_height=cfg.array.element_height,
                n_slices=cfg.array.n_elevation_slices,
            )
            if len(z_s) > 0:
                dz = born_z_grid[1] - born_z_grid[0] if len(born_z_grid) > 1 else 1e-4
                dl = born_l_grid[1] - born_l_grid[0] if len(born_l_grid) > 1 else 1e-4
                rng = np.random.default_rng(seed=i)
                z_s = z_s + rng.uniform(-dz / 2, dz / 2, size=z_s.shape)
                x_s = x_s + rng.uniform(-dl / 2, dl / 2, size=x_s.shape)
                engine.set_born_scatterers(z_s, x_s, amp_s)
                n_born = len(z_s)

        print(f"({active} defect(s), {n_born} Born scatterers)")

        # Simulate FMC
        result = engine.simulate()
        fmc = result['fmc_data']
        time_axis = result['time_axis']
        elem_x = result['element_positions']

        # Gate out front-wall echo
        gate_samples = int(2e-6 / cfg.dt)
        fmc[:, :, :gate_samples] = 0.0

        # Add noise and filter
        fmc = add_noise(fmc, snr_db=snr_db,
                        grain_noise_level=cfg.acquisition.grain_noise_level)
        fmc = apply_bandpass_filter(
            fmc, cfg.dt, cfg.array.frequency,
            bandwidth_fraction=cfg.array.bandwidth,
            filter_alpha=cfg.acquisition.filter_alpha,
            hanning_bool=cfg.acquisition.hanning_bool)

        # TFM reconstruction
        img, _, _ = reconstruct_tfm(
            fmc, time_axis, elem_x, cfg.material.c_L,
            x_range=(-half_w, half_w),
            z_range=(tfm_z_start, tfm_z_end),
            n_pixels=tfm_n_pixels,
        )

        # Save B-scans
        tag = f"{i:04d}"
        # Complex analytic signal
        np.save(os.path.join(output_dir, f'bscan_complex_{tag}.npy'),
                img.astype(np.complex64))
        # dB version for visualization
        img_envelope = np.abs(img)
        img_db = 20 * np.log10(
            img_envelope / (img_envelope.max() + 1e-10) + 1e-10)
        np.save(os.path.join(output_dir, f'bscan_{tag}.npy'),
                img_db.astype(np.float32))

        print(f"    -> bscan_{tag}.npy  ({timer.time() - t0:.1f}s)")

    print(f"\n  Done -- {n_scans} frames saved to {output_dir}/")
    return output_dir


# =========================================================================
#  MAIN: Generate synthetic data + reconstruct
# =========================================================================

if __name__ == '__main__':

    # Add project paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    synth_dir = os.path.join(script_dir, 'SYNTHETIC DATA')
    sys.path.insert(0, synth_dir)
    sys.path.insert(0, script_dir)

    from engine.geometry import SphericalDefect, CylindricalDefect

    # =====================================================================
    #  USER PARAMETERS — edit these to configure the simulation
    # =====================================================================

    # Output directory for the generated B-scans and reconstruction
    output_dir = os.path.join(script_dir, 'output', 'recon_3d_clean')

    # Specimen geometry (metres)
    specimen_thickness = 50e-3   # 50 mm
    specimen_width     = 50e-3   # 50 mm (along array)
    specimen_depth     = 30e-3   # 30 mm (elevation / y-axis)

    # Array parameters
    num_elements  = 128
    element_pitch = 0.3e-3       # 0.6 mm
    frequency     = 10e6         # 10 MHz
    element_height= 5e-3

    # Scan parameters
    n_scans = 32                 # Rotation frames over [0°, 180°), endpoint excluded

    # TFM imaging parameters
    tfm_z_start = 10e-3          # Start depth (m)
    tfm_z_end   = 45e-3          # End depth (m)
    tfm_n_pixels = 400           # Pixel grid size (square)

    # Grain structure
    use_voxel_world      = False     # No grain noise — single scatterer test
    mean_grain_size      = 0.5e-3    # 500 um (unused when use_voxel_world=False)
    impedance_variation  = 0.025     # +/- 2.5% (unused)

    # 3D defects to embed in the specimen — single off-center spherical pore
    defects_3d = [
        SphericalDefect(
            center_z=25e-3, center_x=5e-3, center_y=5e-3, radius=2e-3),
    ]

    # Reconstruction config — defaults match the paper (ramp |ω| filter,
    # circle mask, DC subtraction). Override here if needed.
    config = ReconConfig(
        taper_fraction=0.1,
        rolloff_fraction=0.08,
    )

    # =====================================================================
    #  STEP 1: Generate synthetic B-scans
    # =====================================================================

    scan_dir = os.path.join(output_dir, 'scan_data')

    generate_synthetic_bscans(
        output_dir=scan_dir,
        n_scans=n_scans,
        num_elements=num_elements,
        element_pitch=element_pitch,
        frequency=frequency,
        specimen_thickness=specimen_thickness,
        specimen_width=specimen_width,
        specimen_depth=specimen_depth,
        defects_3d=defects_3d,
        use_voxel_world=use_voxel_world,
        mean_grain_size=mean_grain_size,
        impedance_variation=impedance_variation,
        tfm_z_start=tfm_z_start,
        tfm_z_end=tfm_z_end,
        tfm_n_pixels=tfm_n_pixels,
    )

    # =====================================================================
    #  STEP 2: Reconstruct the 3D volume from the B-scans
    # =====================================================================

    volume = reconstruct(
        scan_dir=scan_dir,
        config=config,
        save_dir=output_dir,
        show_napari=False,
    )
