"""
3D Volume Reconstruction — Shared Module
=========================================

Generic inverse Radon reconstruction from rotational B-scan stacks.
No dependency on the synthetic data engine — usable for both synthetic
and experimental data.

Usage:
    from Classes.Reconstruct3D import reconstruct_scan
    volume = reconstruct_scan('DATA/1D NPY Data/Al Hole 5MHz 02022026/')
"""

import os
import re
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Optional

from scipy.signal.windows import tukey
from skimage.transform import iradon


# ── Load B-scans ─────────────────────────────────────────────────────

def load_bscans(scan_dir: str) -> tuple:
    """
    Load B-scan stack and metadata from a scan directory.

    Expects:
        scan_dir/scan_meta.npy   — dict with scan parameters
        scan_dir/bscan_NNNN.npy  — one file per rotation angle

    Returns:
        bscans_db: (n_scans, n_z, n_lateral) float32
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

    # Load only bscan_NNNN.npy (exclude bscan_complex_NNNN.npy)
    files = sorted(f for f in os.listdir(scan_dir)
                   if re.match(r'^bscan_\d{4}\.npy$', f))
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


def load_bscans_complex(scan_dir: str) -> tuple:
    """
    Load complex analytic B-scan stack and metadata.

    Expects:
        scan_dir/scan_meta.npy             — dict with scan parameters
        scan_dir/bscan_complex_NNNN.npy    — one complex file per angle

    Returns:
        bscans_complex: (n_scans, n_z, n_lateral) complex64
        meta:           dict with scan parameters
    """
    meta_path = os.path.join(scan_dir, 'scan_meta.npy')
    meta = np.load(meta_path, allow_pickle=True).item()

    if 'tfm_z_start_m' not in meta:
        warnings.warn("scan_meta.npy missing tfm fields — using defaults. "
                       "Re-run scan_volume_3d() to fix.")
        meta.setdefault('tfm_z_start_m', 10e-3)
        meta.setdefault('tfm_z_end_m', meta['specimen_thickness_m'] - 5e-3)
        meta.setdefault('array_aperture_m', meta['specimen_width_m'] * 0.7)

    files = sorted(f for f in os.listdir(scan_dir)
                   if re.match(r'^bscan_complex_\d{4}\.npy$', f))
    if not files:
        raise FileNotFoundError(
            f"No bscan_complex_*.npy files in {scan_dir}. "
            f"Re-run scan_volume_3d() with img_output='complex'.")

    bscans = [np.load(os.path.join(scan_dir, f)) for f in files]
    bscans_complex = np.stack(bscans, axis=0).astype(np.complex64)

    if 'tfm_n_pixels' not in meta:
        meta['tfm_n_pixels'] = bscans_complex.shape[1]

    assert bscans_complex.shape[0] == meta['n_scans'], (
        f"Loaded {bscans_complex.shape[0]} complex B-scans "
        f"but metadata says {meta['n_scans']}")

    print(f"Loaded {bscans_complex.shape[0]} complex B-scans, shape per frame: "
          f"{bscans_complex.shape[1]}x{bscans_complex.shape[2]}")
    return bscans_complex, meta


def has_complex_bscans(scan_dir: str) -> bool:
    """Check whether complex B-scan files exist in scan_dir."""
    return any(re.match(r'^bscan_complex_\d{4}\.npy$', f)
               for f in os.listdir(scan_dir))


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

    Returns signed deviations — iradon is a linear inversion and handles
    negative values correctly. Clipping or taking abs destroys the sign
    information needed to reconstruct voids/holes (negative features).

    Args:
        sinograms: (n_z, n_detectors, n_angles)

    Returns:
        Sinograms with angular mean subtracted (signed), same shape.
    """
    # Mean across angles for each (depth, detector) pair
    angular_mean = sinograms.mean(axis=2, keepdims=True)
    return (sinograms - angular_mean).astype(sinograms.dtype)


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

    volume = np.abs(volume)
    # Match ascending y_coords (iradon output has y increasing downward).
    volume = volume[:, ::-1, :]

    print(f"  Reconstructed volume shape: {volume.shape}")
    return volume


def reconstruct_volume_complex(
    sinograms: np.ndarray,
    angles_deg: np.ndarray,
    filter_name: str = 'shepp-logan',
    circle: bool = True,
    output_size: Optional[int] = None,
) -> np.ndarray:
    """
    Inverse Radon reconstruction from complex analytic sinograms.

    Applies iradon separately to real and imaginary parts, then takes
    the magnitude of the combined complex result.  This matches the
    MATLAB approach and preserves phase information through the
    reconstruction.

    Args:
        sinograms:   (n_z, n_detectors, n_angles) complex
        angles_deg:  (n_angles,) projection angles in degrees
        filter_name: FBP filter (default 'shepp-logan' to match MATLAB)
        circle:      If True, assume projections cover inscribed circle
        output_size: Output image side length. Default: n_detectors.

    Returns:
        volume: (n_z, output_size, output_size) float32 (magnitude)
    """
    n_z, n_det, n_ang = sinograms.shape
    if output_size is None:
        output_size = n_det

    volume = np.zeros((n_z, output_size, output_size), dtype=np.float32)

    print(f"Reconstructing {n_z} depth slices (complex) "
          f"({n_det} detectors, {n_ang} angles, "
          f"filter='{filter_name}', circle={circle})...")

    for z in range(n_z):
        sino = sinograms[z]
        recon_real = iradon(
            sino.real, theta=angles_deg,
            filter_name=filter_name, circle=circle, output_size=output_size,
        )
        recon_imag = iradon(
            sino.imag, theta=angles_deg,
            filter_name=filter_name, circle=circle, output_size=output_size,
        )
        volume[z] = np.abs(recon_real + 1j * recon_imag).astype(np.float32)

    # iradon returns images in image convention (row index increases
    # downward). y_coords is ascending, so flip row axis to match.
    volume = volume[:, ::-1, :]

    print(f"  Reconstructed volume shape: {volume.shape}")
    return volume


# ── Polar gridding (spatial compounding) reconstruction ──────────────

def reconstruct_volume_polar(
    sinograms: np.ndarray,
    angles_rad: np.ndarray,
    output_size: Optional[int] = None,
    beam_width: Optional[float] = None,
) -> np.ndarray:
    """
    Reconstruct volume by polar-to-Cartesian interpolation with
    finite beam width weighting.

    Each B-scan is a 2D slice through the 3D volume. At angle θ, the
    scan plane contains points where the perpendicular distance
    d = -x·sinθ + y·cosθ ≈ 0.  A point at (x, y) is only imaged by
    this B-scan if it lies close to the scan plane.

    For each output voxel (x, y, z):
      1. Compute lateral position: L = x·cosθ + y·sinθ
      2. Compute perpendicular distance: d = -x·sinθ + y·cosθ
      3. Weight by Gaussian: w = exp(-d²/(2σ²)) where σ = beam_width
      4. Interpolate B-scan value at (z, L) and accumulate with weight w

    This correctly limits each B-scan's contribution to voxels near
    its scan plane, producing localised 3D reconstructions (e.g. a
    sphere appears as a sphere, not a star pattern).

    Args:
        sinograms:   (n_z, n_lateral, n_angles)
        angles_rad:  (n_angles,) rotation angles in radians
        output_size: Output image side length. Default: n_lateral.
        beam_width:  Gaussian beam half-width in normalised coords.
                     Default: angular spacing × radius, i.e. the
                     distance between adjacent scan planes at the edge.

    Returns:
        volume: (n_z, output_size, output_size) float32
    """
    from scipy.interpolate import RegularGridInterpolator

    n_z, n_lateral, n_angles = sinograms.shape
    if output_size is None:
        output_size = n_lateral

    # Lateral axis: normalised to [-1, 1]
    L_axis = np.linspace(-1.0, 1.0, n_lateral)

    # Default beam width: 2× the angular step — provides overlap between
    # adjacent scan planes while preserving localisation
    if beam_width is None:
        if n_angles > 1:
            d_theta = np.abs(angles_rad[1] - angles_rad[0])
        else:
            d_theta = np.pi
        beam_width = float(2.0 * d_theta)

    sigma_sq = beam_width ** 2

    # Output grid
    xy = np.linspace(-1.0, 1.0, output_size)
    YY, XX = np.meshgrid(xy, xy, indexing='ij')

    z_axis = np.arange(n_z, dtype=np.float64)

    volume_sum = np.zeros((n_z, output_size, output_size), dtype=np.float64)
    weight_sum = np.zeros((n_z, output_size, output_size), dtype=np.float64)

    print(f"Polar gridding: {n_z} depth slices, "
          f"{n_lateral} lateral, {n_angles} angles, "
          f"beam_width={beam_width:.4f}...")

    for i in range(n_angles):
        theta = angles_rad[i]
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        # Lateral position and perpendicular distance for each output point
        L_map = XX * cos_t + YY * sin_t   # (output_size, output_size)
        d_map = -XX * sin_t + YY * cos_t  # perpendicular distance

        # Gaussian weight: points on the scan plane get weight 1,
        # points far from it get weight → 0
        w_map = np.exp(-0.5 * d_map ** 2 / sigma_sq)

        # Also require L within B-scan lateral range
        in_range = (L_map >= L_axis[0]) & (L_map <= L_axis[-1])
        w_map *= in_range

        # Build 2D interpolator over (z, L) for this angle's B-scan
        bscan_data = sinograms[:, :, i]  # (n_z, n_lateral)
        interp = RegularGridInterpolator(
            (z_axis, L_axis), bscan_data.astype(np.float64),
            method='linear', bounds_error=False, fill_value=0.0,
        )

        # Sample points
        ZZ_out = np.broadcast_to(z_axis[:, None, None],
                                 (n_z, output_size, output_size))
        L_out = np.broadcast_to(L_map[None, :, :],
                                (n_z, output_size, output_size))

        pts = np.stack([ZZ_out.ravel(), L_out.ravel()], axis=-1)
        vals = interp(pts).reshape(n_z, output_size, output_size)

        w_3d = np.broadcast_to(w_map[None, :, :],
                               (n_z, output_size, output_size))
        volume_sum += vals * w_3d
        weight_sum += w_3d

        if (i + 1) % 10 == 0 or i == 0 or i == n_angles - 1:
            print(f"  Compounded {i+1}/{n_angles} B-scans "
                  f"(\u03b8 = {np.degrees(theta):+6.1f}\u00b0)")

    # Weighted average
    mask = weight_sum > 1e-12
    volume = np.zeros((n_z, output_size, output_size), dtype=np.float32)
    volume[mask] = (volume_sum[mask] / weight_sum[mask]).astype(np.float32)

    # Absolute value for signed data (from angular mean subtraction)
    volume = np.abs(volume)

    print(f"  Polar gridding complete: {volume.shape}")
    return volume


# ── ZX-plane iradon reconstruction (CT-scanner geometry) ─────────────

def _subtract_angular_mean_zx(bscans: np.ndarray) -> np.ndarray:
    """
    Remove rotationally invariant component from B-scans directly.

    Companion to ``_subtract_angular_mean`` for the zx-plane method.
    Wall echoes at the same (depth, lateral) across all rotation angles
    form the DC component along axis 0 of the B-scan stack.

    Args:
        bscans: (n_scans, n_z, n_lateral) linear amplitude

    Returns:
        Signed deviations, same shape.
    """
    return (bscans - bscans.mean(axis=0, keepdims=True)).astype(bscans.dtype)


def reconstruct_volume_zx(
    bscans_linear: np.ndarray,
    angles_deg: np.ndarray,
    filter_name: str = 'hann',
    circle: bool = True,
    output_size: Optional[int] = None,
) -> np.ndarray:
    """
    Reconstruct volume by applying iradon to zx-plane sinograms.

    Each B-scan is a 2D zx slice of the volume taken at rotation angle θ.
    For each lateral index l (fixed radial distance from the rotation
    axis), the stack of B-scan columns across rotation angles forms a
    sinogram with z as the detector axis:

        sinogram[l] = bscans[:, :, l].T  → (n_z, n_scans)

    iradon then reconstructs a 2D image at each lateral offset l whose
    two spatial axes are (z, z'): the rotationally-swept cross-section
    of the volume at that lateral position. This simulates a CT-scanner
    with the 1D array rotating around the z-axis.

    Args:
        bscans_linear: (n_scans, n_z, n_lateral) linear amplitude
        angles_deg:    (n_scans,) rotation angles in degrees
        filter_name:   FBP filter ('ramp', 'shepp-logan', 'hamming', 'hann')
        circle:        If True, assume projections cover only inscribed circle
        output_size:   Output image side length. Default: n_z.

    Returns:
        volume: (n_lateral, output_size, output_size) float32
    """
    n_scans, n_z, n_lateral = bscans_linear.shape
    if output_size is None:
        output_size = n_z

    volume = np.zeros((n_lateral, output_size, output_size), dtype=np.float32)

    print(f"Reconstructing {n_lateral} lateral slices (zx iradon) "
          f"({n_z} z-detectors, {n_scans} angles, "
          f"filter='{filter_name}', circle={circle})...")

    for l in range(n_lateral):
        sinogram = bscans_linear[:, :, l].T  # (n_z, n_scans)
        recon = iradon(
            sinogram,
            theta=angles_deg,
            filter_name=filter_name,
            circle=circle,
            output_size=output_size,
        )
        volume[l] = recon.astype(np.float32)

    volume = np.abs(volume)

    print(f"  Reconstructed volume shape: {volume.shape}")
    return volume


def compute_reconstruction_coords_zx(
    meta: dict,
    output_size: int,
) -> tuple:
    """
    Physical coordinates for a zx-plane reconstructed volume.

    The volume shape is (n_lateral, output_size, output_size). The first
    axis corresponds to lateral position along the array aperture; the
    two reconstructed axes span the rotational sweep in the z direction.

    Returns:
        (lateral_coords, z_coords, zp_coords) -- 1D arrays in metres
    """
    n_lateral = meta.get('tfm_n_pixels', output_size)
    half_ap = meta['array_aperture_m'] / 2.0
    lateral_coords = np.linspace(-half_ap, half_ap, n_lateral)
    # Span of the z axis in the tfm image
    z_extent = meta['tfm_z_end_m'] - meta['tfm_z_start_m']
    half_z = z_extent / 2.0
    z_center = 0.5 * (meta['tfm_z_start_m'] + meta['tfm_z_end_m'])
    z_coords = np.linspace(z_center - half_z, z_center + half_z, output_size)
    zp_coords = np.linspace(-half_z, half_z, output_size)
    return lateral_coords, z_coords, zp_coords


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


# ── Napari viewer ─────────────────────────────────────────────────────

def view_reconstruction_napari(
    recon: np.ndarray,
    ground_truth: Optional[np.ndarray],
    z_coords: np.ndarray,
    y_coords: np.ndarray,
    x_coords: np.ndarray,
    metrics: Optional[dict] = None,
) -> None:
    """Open napari viewer with reconstruction and optional ground truth."""
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


# ── Top-level pipeline (no ground truth) ─────────────────────────────

def reconstruct_scan(
    scan_dir: str,
    method: str = 'complex',
    filter_name: str = 'shepp-logan',
    circle: bool = True,
    output_size: Optional[int] = None,
    crop_to_cube: bool = False,
    show_napari: bool = False,
    save_figures: bool = True,
    output_dir: Optional[str] = None,
) -> np.ndarray:
    """
    Full reconstruction pipeline: load -> reconstruct -> visualise.

    Uses complex analytic TFM data by default (matching MATLAB approach):
    iradon(real) + 1j*iradon(imag) per depth slice, no preprocessing.

    Falls back to dB envelope pipeline for datasets without complex data.

    Args:
        scan_dir:     Directory with bscan_*.npy and scan_meta.npy
        method:       'complex' (default — requires bscan_complex_*.npy),
                      'iradon' (dB envelope fallback),
                      'polar' (spatial compounding on envelope data), or
                      'zx_iradon' (CT-scanner geometry per lateral index)
        filter_name:  FBP filter (default 'shepp-logan' to match MATLAB)
        circle:       Truncated projections flag (iradon only)
        output_size:  Reconstruction grid size (default: n_lateral)
        crop_to_cube: Crop cylindrical volume to inscribed cube
        show_napari:  Open interactive napari viewer
        save_figures: Save static summary PNG
        output_dir:   Where to save outputs (default: scan_dir)

    Returns:
        volume_recon: (n_z, output_size, output_size) float32
    """
    if output_dir is None:
        output_dir = scan_dir

    # Auto-detect: if 'complex' requested but no complex files, fall back
    use_complex = (method == 'complex')
    if use_complex and not has_complex_bscans(scan_dir):
        warnings.warn("No bscan_complex_*.npy files found — "
                       "falling back to dB envelope iradon pipeline.")
        use_complex = False
        method = 'iradon'

    if use_complex:
        # ── Complex pipeline (matches MATLAB supervisor code) ──
        # No dB conversion, no taper, no angular mean subtraction.
        bscans_complex, meta = load_bscans_complex(scan_dir)

        angles_rad = meta['angles_rad']
        angles_deg = np.degrees(angles_rad)
        print(f"Angular range: {angles_deg[0]:+.1f} to {angles_deg[-1]:+.1f} deg "
              f"({len(angles_deg)} projections), method=complex")

        sinograms = build_sinograms(bscans_complex)

        if output_size is None:
            output_size = sinograms.shape[1]

        volume_recon = reconstruct_volume_complex(
            sinograms, angles_deg,
            filter_name=filter_name, circle=circle, output_size=output_size,
        )

        # Also load dB B-scans for the summary figure
        bscans_db, _ = load_bscans(scan_dir)
        sinograms_diag = build_sinograms(
            np.abs(bscans_complex).astype(np.float32))
    else:
        # ── Envelope fallback pipeline ──
        bscans_db, meta = load_bscans(scan_dir)

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

        angles_rad = meta['angles_rad']
        angles_deg = np.degrees(angles_rad)
        print(f"Angular range: {angles_deg[0]:+.1f} to {angles_deg[-1]:+.1f} deg "
              f"({len(angles_deg)} projections), method={method}")

        if method == 'zx_iradon':
            bscans_signed = _subtract_angular_mean_zx(bscans_lin)
            if output_size is None:
                output_size = bscans_signed.shape[1]
            volume_recon = reconstruct_volume_zx(
                bscans_signed, angles_deg,
                filter_name=filter_name, circle=circle, output_size=output_size,
            )
            sinograms_diag = build_sinograms(bscans_lin)
        else:
            sinograms = build_sinograms(bscans_lin)
            sinograms = _subtract_angular_mean(sinograms)
            sinograms_diag = sinograms

            if output_size is None:
                output_size = sinograms.shape[1]

            if method == 'polar':
                volume_recon = reconstruct_volume_polar(
                    sinograms, angles_rad, output_size=output_size,
                )
            else:
                volume_recon = reconstruct_volume(
                    sinograms, angles_deg,
                    filter_name=filter_name, circle=circle,
                    output_size=output_size,
                )

    # Soft circular apodisation to remove ring artifact, then crop
    volume_recon = _soft_circle_apodise(volume_recon, rolloff_fraction=0.08)
    if crop_to_cube:
        if method == 'zx_iradon':
            warnings.warn("crop_to_cube is not supported for zx_iradon method; skipping.")
        else:
            volume_recon = crop_cylinder_to_cube(volume_recon)
            print(f"  Cropped to cube: {volume_recon.shape}")
            output_size = volume_recon.shape[1]

    # Save reconstructed volume
    recon_path = os.path.join(output_dir, 'recon_volume.npy')
    np.save(recon_path, volume_recon)
    print(f"Reconstructed volume saved: {recon_path}")

    # Save reconstruction summary figure
    if save_figures:
        fig_path = os.path.join(output_dir, 'reconstruction_summary.png')
        save_reconstruction_summary(
            volume_recon, bscans_db, sinograms_diag, meta, fig_path,
        )

    # Napari viewer
    if show_napari:
        if method == 'zx_iradon':
            axis0, axis1, axis2 = compute_reconstruction_coords_zx(
                meta, output_size)
        else:
            axis0, axis1, axis2 = compute_reconstruction_coords(
                meta, output_size)
        view_reconstruction_napari(
            volume_recon, None,
            axis0, axis1, axis2,
        )

    return volume_recon
