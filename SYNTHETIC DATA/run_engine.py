#!/usr/bin/env python3
"""
Physics-Accurate 2D NDT Synthetic Data Engine
==============================================

Uses the new ray-tracing engine with:
- Front-wall and back-wall echoes
- Angle-dependent Fresnel coefficients (Zoeppritz)
- Mode conversion (L→S) at the back wall
- Kirchhoff surface scattering from defect geometry
- Skip paths and corner trap for crack detection
- Back-wall reverberations
- Gabor wavelet pulse synthesis (not spike model)

Output: FMC data → noise → bandpass filter → TFM B-scan

Usage:
    python run_engine.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from typing import Any
import sys
import os
import time

# Add parent directory for Classes/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from Classes.Filter import filter_signal
from Classes.TFM1D import CTFM1D

from engine.config import (
    SimulationConfig, SpecimenConfig, ArrayConfig, ScanPlanConfig,
)
from engine.geometry import (
    Specimen3D, SphericalDefect, CylindricalDefect, PlanarCrack3D,
)
from engine.fmc_engine import FMCEngine
from engine.materials import ALUMINUM, STEEL_MILD, STEEL_STAINLESS, WATER, NDT_GEL
from engine.voxel_volume import VoxelVolume3D
from engine.microstructure import generate_grain_structure, embed_geometric_defects


def add_noise(fmc_data: np.ndarray, snr_db: float = 35.0,
              grain_noise_level: float = 0.03) -> np.ndarray:
    """
    Add realistic noise to FMC data.

    Args:
        fmc_data: (num_tx, num_rx, time_samples)
        snr_db: Target signal-to-noise ratio
        grain_noise_level: Grain scattering amplitude relative to signal

    Returns:
        Noisy FMC data
    """
    signal_power = np.mean(fmc_data**2)
    if signal_power < 1e-30:
        return fmc_data

    noise_std = np.sqrt(signal_power / 10**(snr_db / 10))

    # Gaussian electronic noise
    noisy = fmc_data + np.random.normal(0, noise_std, fmc_data.shape).astype(np.float32)

    # Structured grain noise (band-limited random)
    num_tx, num_rx, n_t = fmc_data.shape
    grain = np.random.normal(0, grain_noise_level * noise_std, fmc_data.shape)
    # Smooth grain noise to simulate grain scattering bandwidth
    from scipy.ndimage import uniform_filter1d
    grain = uniform_filter1d(grain, size=5, axis=2).astype(np.float32)
    noisy += grain

    return noisy


def apply_bandpass_filter(fmc_data: np.ndarray, dt: float,
                           frequency: float,
                           bandwidth_fraction: float = 0.6) -> np.ndarray:
    """
    Apply bandpass filter to all A-scans in FMC data.

    Args:
        fmc_data: (num_tx, num_rx, time_samples)
        dt: Time step (s)
        frequency: Center frequency (Hz)
        bandwidth_fraction: Fractional bandwidth (e.g. 0.6 = 60%)

    Returns:
        Filtered FMC data
    """
    f_center = frequency / 1e6  # MHz
    f_start = f_center * (1 - bandwidth_fraction / 2)
    f_end = f_center * (1 + bandwidth_fraction / 2)
    f_start = max(f_start, 0.1)  # Avoid zero

    num_tx, num_rx, n_t = fmc_data.shape
    filtered = np.zeros_like(fmc_data)

    for tx in range(num_tx):
        for rx in range(num_rx):
            filtered[tx, rx, :] = filter_signal(
                fmc_data[tx, rx, :], dt, f_start, f_end,
                filter_alpha=0.5, hanning_bool=False
            )

    return filtered


def reconstruct_tfm(fmc_data: np.ndarray, time_axis: np.ndarray,
                     elem_x: np.ndarray, c: float,
                     x_range: tuple, z_range: tuple,
                     n_pixels: int = 300) -> tuple:
    """
    TFM reconstruction using existing CTFM1D.

    Args:
        fmc_data: (num_tx, num_rx, time_samples)
        time_axis: (time_samples,)
        elem_x: (num_elements,) element x-positions
        c: Wave speed (m/s)
        x_range: (x_min, x_max) in meters
        z_range: (z_min, z_max) in meters
        n_pixels: Number of pixels per axis

    Returns:
        (img_db, x_img, z_img): dB image and axis arrays
    """
    num_el = fmc_data.shape[0]

    # Create TX/RX index arrays (1-based for CTFM1D)
    tx_arr = np.repeat(np.arange(1, num_el + 1), num_el)
    rx_arr = np.tile(np.arange(1, num_el + 1), num_el)

    # Flatten FMC to (N_fmc, N_t)
    fmc_flat = fmc_data.reshape(-1, fmc_data.shape[-1])

    # Element positions
    xc = elem_x
    zc = np.zeros_like(xc)

    # Image grid
    x_img = np.linspace(x_range[0], x_range[1], n_pixels)
    z_img = np.linspace(z_range[0], z_range[1], n_pixels)

    print(f"  TFM reconstruction: {n_pixels}×{n_pixels} pixels...")
    t0 = time.time()
    img_db = CTFM1D(fmc_flat, time_axis, tx_arr, rx_arr, xc, zc, c, x_img, z_img,
                     output_db=True)
    print(f"  TFM complete: {time.time() - t0:.1f}s")

    return img_db, x_img, z_img


def visualize(img_db: np.ndarray, x_img: np.ndarray, z_img: np.ndarray,
              fmc_raw: np.ndarray, fmc_filtered: np.ndarray,
              time_axis: np.ndarray,
              defects: list, cfg: SimulationConfig,
              output_path: str = 'engine_result.png'):
    """Create a multi-panel visualization of the simulation results."""
    from scipy.signal import hilbert as scipy_hilbert
    fig, axes = plt.subplots(2, 3, figsize=(22, 10))
    ax1, ax2, ax3 = axes[0]
    ax4, ax5, ax6 = axes[1]

    assert cfg.material is not None
    assert cfg.couplant is not None
    center = cfg.array.num_elements // 2
    a_raw  = fmc_raw[center, center, :]
    a_filt = fmc_filtered[center, center, :]
    fw_tof = 2.0 * cfg.gel_thickness / cfg.couplant.c_L
    bw_tof = 2.0 * cfg.specimen.thickness / cfg.material.c_L
    dt = time_axis[1] - time_axis[0]

    # Panel 1: Full A-scan — 0 to 20% past back wall
    ax1.plot(time_axis * 1e6, a_raw,  color='0.65', linewidth=0.6, label='Pre-filter (raw)')
    ax1.plot(time_axis * 1e6, a_filt, 'k-', linewidth=0.8, label='Post-filter')
    ax1.axvline(fw_tof * 1e6, color='b', ls='--', alpha=0.7, label=f'Front wall ({fw_tof*1e6:.2f} μs)')
    ax1.axvline(bw_tof * 1e6, color='r', ls='--', alpha=0.7, label=f'Back wall ({bw_tof*1e6:.2f} μs)')
    ax1.set_xlim(0, bw_tof * 1e6 * 1.2)
    ax1.set_xlabel('Time (μs)')
    ax1.set_ylabel('Amplitude')
    ax1.set_title(f'A-scan full view (TX=RX={center})')
    ax1.legend(fontsize=7)

    # Panel 2: Zoomed A-scan — coupling-layer echo
    zoom_end = max(fw_tof * 1e6 * 20, 2.0)
    ax2.plot(time_axis * 1e6, a_raw,  color='0.65', linewidth=0.8, label='Pre-filter (raw)')
    ax2.plot(time_axis * 1e6, a_filt, 'k-', linewidth=1.0, label='Post-filter')
    ax2.axvline(fw_tof * 1e6, color='b', ls='--', alpha=0.8, label=f'Front wall ({fw_tof*1e6:.3f} μs)')
    ax2.set_xlim(0, zoom_end)
    ax2.set_xlabel('Time (μs)')
    ax2.set_ylabel('Amplitude')
    ax2.set_title(f'A-scan zoomed — coupling layer (0–{zoom_end:.1f} μs)')
    ax2.legend(fontsize=7)

    # Panel 3: Frequency spectrum (FFT of center A-scan)
    n = len(a_raw)
    freqs = np.fft.rfftfreq(n, d=dt) / 1e6  # MHz
    spec_raw  = np.abs(np.fft.rfft(a_raw))
    spec_filt = np.abs(np.fft.rfft(a_filt))
    # Normalise to dB relative to peak of raw spectrum
    peak = np.max(spec_raw) + 1e-30
    spec_raw_db  = 20 * np.log10(spec_raw  / peak + 1e-12)
    spec_filt_db = 20 * np.log10(spec_filt / peak + 1e-12)
    ax3.plot(freqs, spec_raw_db,  color='0.65', linewidth=0.8, label='Pre-filter')
    ax3.plot(freqs, spec_filt_db, 'k-', linewidth=1.0, label='Post-filter')
    ax3.axvline(cfg.array.frequency / 1e6, color='r', ls='--', alpha=0.7,
                label=f'Centre freq ({cfg.array.frequency/1e6:.0f} MHz)')
    ax3.set_xlim(0, cfg.array.frequency / 1e6 * 3)
    ax3.set_ylim(-60, 5)
    ax3.set_xlabel('Frequency (MHz)')
    ax3.set_ylabel('Magnitude (dB)')
    ax3.set_title('Frequency spectrum')
    ax3.legend(fontsize=7)
    ax3.grid(True, alpha=0.3)

    # Panel 4: Hilbert envelope of filtered A-scan
    envelope = np.abs(scipy_hilbert(a_filt))
    ax4.plot(time_axis * 1e6, a_filt,   color='0.65', linewidth=0.6, label='RF signal')
    ax4.plot(time_axis * 1e6, envelope,  'r-', linewidth=1.2, label='Hilbert envelope')
    ax4.plot(time_axis * 1e6, -envelope, 'r-', linewidth=1.2)  # Mirror for clarity
    ax4.axvline(fw_tof * 1e6, color='b', ls='--', alpha=0.6, label=f'Front wall')
    ax4.axvline(bw_tof * 1e6, color='g', ls='--', alpha=0.6, label=f'Back wall')
    ax4.set_xlim(0, bw_tof * 1e6 * 1.2)
    ax4.set_xlabel('Time (μs)')
    ax4.set_ylabel('Amplitude')
    ax4.set_title('Hilbert envelope (post-filter)')
    ax4.legend(fontsize=7)

    # Panel 5: TFM B-scan
    extent = [x_img[0]*1e3, x_img[-1]*1e3, z_img[-1]*1e3, z_img[0]*1e3]
    im = ax5.imshow(img_db, extent=extent, aspect='auto', cmap='hot', vmin=-40, vmax=0)
    ax5.set_xlabel('X (mm)')
    ax5.set_ylabel('Depth (mm)')
    ax5.set_title('TFM B-scan (dB)')
    plt.colorbar(im, ax=ax5, label='dB', shrink=0.8)
    for d in defects:
        c_pos = d.center
        marker = 'co' if hasattr(d, 'radius') else 'c^'
        ax5.plot(c_pos[1]*1e3, c_pos[0]*1e3, marker, markersize=10,
                 markerfacecolor='none', linewidth=2)

    # Panel 6: FMC B-scan (TX=center, all RX)
    bscan = np.abs(fmc_filtered[center, :, :])
    bscan_db = 20 * np.log10(bscan / np.max(bscan) + 1e-12)
    extent_b = [0, cfg.array.num_elements - 1, time_axis[-1]*1e6, 0]
    ax6.imshow(bscan_db.T, extent=extent_b, aspect='auto', cmap='hot', vmin=-40, vmax=0)
    ax6.set_xlabel('RX Element')
    ax6.set_ylabel('Time (μs)')
    ax6.set_title(f'FMC B-scan (TX={center})')
    ax6.set_ylim(bw_tof * 1e6 * 1.2, 0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def preview_volume_3d(
    specimen: Specimen3D,
    defects_3d: list,
    scan_plan: ScanPlanConfig,
    output_path: str,
) -> None:
    """
    Render a 3-view preview of the 3D specimen + defects before scanning.

    Views:
        - Perspective (elevation=25°, azimuth=-55°)
        - Top-down    (looking along −z, i.e. into the specimen from the array)
        - Front       (looking along +y, i.e. the B-scan plane at θ=0°)

    Defect colours:   red = sphere, blue = cylinder (SDH), green = crack
    Specimen box:     thin grey wireframe
    Scan planes:      transparent fan lines in the x-y surface

    Args:
        specimen:    3D specimen geometry
        defects_3d:  List of Defect3D instances
        scan_plan:   Angular scan plan (used to draw the rotation fan)
        output_path: Path to save the PNG
    """
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # type: ignore[import]

    # Convenient mm conversion
    def mm(v: float) -> float:
        return v * 1e3

    W  = mm(specimen.width)       # x-extent (mm)
    D  = mm(specimen.depth)       # y-extent (mm)
    TH = mm(specimen.thickness)   # z-extent/depth (mm)
    x0, x1 = -W / 2, W / 2
    y0, y1 = -D / 2, D / 2
    z0, z1 = 0.0, TH

    def _draw_box(ax: Any) -> None:
        """Draw specimen as a thin grey wireframe box."""
        edges = [
            ([x0, x1], [y0, y0], [z0, z0]), ([x0, x1], [y1, y1], [z0, z0]),
            ([x0, x1], [y0, y0], [z1, z1]), ([x0, x1], [y1, y1], [z1, z1]),
            ([x0, x0], [y0, y1], [z0, z0]), ([x1, x1], [y0, y1], [z0, z0]),
            ([x0, x0], [y0, y1], [z1, z1]), ([x1, x1], [y0, y1], [z1, z1]),
            ([x0, x0], [y0, y0], [z0, z1]), ([x1, x1], [y0, y0], [z0, z1]),
            ([x0, x0], [y1, y1], [z0, z1]), ([x1, x1], [y1, y1], [z0, z1]),
        ]
        for xs, ys, zs in edges:
            ax.plot(xs, ys, zs, color='steelblue', alpha=0.25, linewidth=0.8)

    def _draw_array(ax: Any) -> None:
        """Draw the 1D array as a bold line at z=0, oriented along x (θ=0)."""
        ax.plot([x0, x1], [0, 0], [0, 0], 'k-', linewidth=2.5, zorder=5)

    def _draw_scan_fan(ax: Any) -> None:
        """Show the rotation fan as thin lines on the front surface (z=0)."""
        r = W / 2
        for theta in scan_plan.angles:
            xe = r * np.cos(theta)
            ye = r * np.sin(theta)
            ax.plot([0, xe], [0, ye], [0, 0],
                    color='grey', alpha=0.3, linewidth=0.6)
        # Bold lines for first and last angle
        for theta, lw in [(scan_plan.angles[0], 1.2), (scan_plan.angles[-1], 1.2)]:
            xe = r * np.cos(theta)
            ye = r * np.sin(theta)
            ax.plot([0, xe], [0, ye], [0, 0], 'k-', alpha=0.7, linewidth=lw)

    _DEFECT_COLORS = {
        'SphericalDefect':   'red',
        'CylindricalDefect': 'royalblue',
        'PlanarCrack3D':     'limegreen',
    }

    def _draw_defects(ax: Any) -> None:
        for d3 in defects_3d:
            col = _DEFECT_COLORS.get(type(d3).__name__, 'orange')

            if isinstance(d3, SphericalDefect):
                u = np.linspace(0, 2 * np.pi, 18)
                v = np.linspace(0, np.pi, 12)
                r = mm(d3.radius)
                xs = mm(d3.center_x) + r * np.outer(np.cos(u), np.sin(v))
                ys = mm(d3.center_y) + r * np.outer(np.sin(u), np.sin(v))
                zs = mm(d3.center_z) + r * np.outer(np.ones_like(u), np.cos(v))
                ax.plot_surface(xs, ys, zs, color=col, alpha=0.55, linewidth=0,
                                antialiased=True)

            elif isinstance(d3, CylindricalDefect):
                # Cylinder: axis along y, cross-section circle in z-x plane
                alpha = np.linspace(0, 2 * np.pi, 20)
                y_cyl = np.linspace(mm(d3.y_start), mm(d3.y_end), 6)
                r = mm(d3.radius)
                # x = cx + r*sin(a),  z = cz + r*cos(a)
                xs = mm(d3.center_x) + r * np.outer(np.sin(alpha), np.ones_like(y_cyl))
                ys = np.outer(np.ones_like(alpha), y_cyl)
                zs = mm(d3.center_z) + r * np.outer(np.cos(alpha), np.ones_like(y_cyl))
                ax.plot_surface(xs, ys, zs, color=col, alpha=0.55, linewidth=0,
                                antialiased=True)
                # End caps
                for y_cap in [mm(d3.y_start), mm(d3.y_end)]:
                    x_cap = mm(d3.center_x) + r * np.sin(alpha)
                    z_cap = mm(d3.center_z) + r * np.cos(alpha)
                    ax.plot(x_cap, [y_cap] * len(alpha), z_cap,
                            color=col, alpha=0.4, linewidth=0.6)

            elif isinstance(d3, PlanarCrack3D):
                verts = [[
                    [mm(d3.start_x), mm(d3.y_start), mm(d3.start_z)],
                    [mm(d3.end_x),   mm(d3.y_start), mm(d3.end_z)],
                    [mm(d3.end_x),   mm(d3.y_end),   mm(d3.end_z)],
                    [mm(d3.start_x), mm(d3.y_end),   mm(d3.start_z)],
                ]]
                poly = Poly3DCollection(verts, alpha=0.55,
                                        facecolor=col, edgecolor=col, linewidth=0.8)
                ax.add_collection3d(poly)

    def _style_ax(ax: Any, elev: float, azim: float, title: str) -> None:
        ax.set_xlabel('x (mm)', fontsize=7, labelpad=2)
        ax.set_ylabel('y (mm)', fontsize=7, labelpad=2)
        ax.set_zlabel('depth (mm)', fontsize=7, labelpad=2)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_zlim(z0, z1)
        ax.invert_zaxis()   # depth increases downward (intuitive for NDT)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title, fontsize=9)
        ax.tick_params(labelsize=6)

    fig = plt.figure(figsize=(18, 6))
    views = [
        (131, 25, -55, 'Perspective'),
        (132, 90, -90, 'Top  (array surface, −z)'),
        (133,  0, -90, 'Front  (B-scan plane, θ=0°)'),
    ]
    for pos, elev, azim, title in views:
        ax = fig.add_subplot(pos, projection='3d')
        _draw_box(ax)
        _draw_scan_fan(ax)
        _draw_defects(ax)
        _draw_array(ax)
        _style_ax(ax, elev, azim, title)

    # Legend patches
    import matplotlib.patches as mpatches
    patches = [
        mpatches.Patch(color='red',        label='Spherical void'),
        mpatches.Patch(color='royalblue',  label='Cylindrical void (SDH)'),
        mpatches.Patch(color='limegreen',  label='Planar crack'),
        mpatches.Patch(color='steelblue',  label='Specimen', alpha=0.4),
    ]
    fig.legend(handles=patches, loc='lower center', ncol=4,
               fontsize=8, frameon=False)
    fig.suptitle(
        f'3D Volume Preview — {len(defects_3d)} defect(s)  |  '
        f'{scan_plan.n_scans} scan planes  '
        f'[{np.degrees(scan_plan.theta_start):.0f}° → {np.degrees(scan_plan.theta_end):.0f}°]',
        fontsize=10,
    )
    plt.tight_layout(rect=(0, 0.05, 1, 0.95))
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved volume preview: {output_path}")


def rasterize_volume(specimen: 'Specimen3D', defects_3d: list,
                     voxel_size_mm: float = 0.5) -> np.ndarray:
    """
    Convert the geometric 3D volume into a voxel labels array.

    Each voxel is tested against every defect and assigned the label of the
    first (lowest-index) defect it belongs to, or 0 for background.

    Coordinate mapping:
        axis 0 → z  (depth,     0 … thickness)
        axis 1 → y  (elevation, -depth/2 … +depth/2)
        axis 2 → x  (lateral,   -width/2 … +width/2)

    Args:
        specimen:      Specimen3D instance
        defects_3d:    List of Defect3D objects (SphericalDefect, etc.)
        voxel_size_mm: Voxel edge length in millimetres

    Returns:
        (n_z, n_y, n_x) uint8 array — 0 = background, 1..N = defect labels
    """
    vs = voxel_size_mm * 1e-3  # metres per voxel

    n_z = max(2, int(round(specimen.thickness / vs)) + 1)
    n_y = max(2, int(round(specimen.depth    / vs)) + 1)
    n_x = max(2, int(round(specimen.width    / vs)) + 1)

    z = np.linspace(0,                  specimen.thickness, n_z)
    y = np.linspace(-specimen.depth / 2, specimen.depth / 2, n_y)
    x = np.linspace(-specimen.width / 2, specimen.width / 2, n_x)

    zz, yy, xx = np.meshgrid(z, y, x, indexing='ij')   # (n_z, n_y, n_x)

    labels = np.zeros((n_z, n_y, n_x), dtype=np.uint8)

    for label_idx, defect in enumerate(defects_3d, start=1):
        if isinstance(defect, SphericalDefect):
            mask = (
                (zz - defect.center_z) ** 2
                + (xx - defect.center_x) ** 2
                + (yy - defect.center_y) ** 2
            ) <= defect.radius ** 2

        elif isinstance(defect, CylindricalDefect):
            # Cylinder axis runs along y
            in_radial = (
                (zz - defect.center_z) ** 2
                + (xx - defect.center_x) ** 2
            ) <= defect.radius ** 2
            in_extent = (yy >= defect.y_start) & (yy <= defect.y_end)
            mask = in_radial & in_extent

        elif isinstance(defect, PlanarCrack3D):
            # Represent crack as a thin slab (1.5 voxel half-thickness)
            half_t = vs * 1.5

            in_y = (yy >= defect.y_start) & (yy <= defect.y_end)

            # Closest-point distance from (zz, xx) to the crack line segment
            dz_seg = defect.end_z - defect.start_z
            dx_seg = defect.end_x - defect.start_x
            seg_len_sq = dz_seg ** 2 + dx_seg ** 2

            if seg_len_sq < 1e-20:
                dist_sq = (zz - defect.start_z) ** 2 + (xx - defect.start_x) ** 2
            else:
                t = ((zz - defect.start_z) * dz_seg
                     + (xx - defect.start_x) * dx_seg) / seg_len_sq
                t = np.clip(t, 0.0, 1.0)
                cz = defect.start_z + t * dz_seg
                cx = defect.start_x + t * dx_seg
                dist_sq = (zz - cz) ** 2 + (xx - cx) ** 2

            mask = in_y & (dist_sq <= half_t ** 2)

        else:
            continue

        # Only label voxels not already claimed by an earlier defect
        labels[mask & (labels == 0)] = label_idx

    print(f"  Rasterized volume: {n_z}×{n_y}×{n_x} voxels "
          f"({n_z*vs*1e3:.0f}×{n_y*vs*1e3:.0f}×{n_x*vs*1e3:.0f} mm), "
          f"{np.count_nonzero(labels)} defect voxels")
    return labels


def view_in_napari(specimen: 'Specimen3D', defects_3d: list,
                   voxel_size_mm: float = 0.5) -> None:
    """
    Open the rasterized 3D volume in a napari viewer.

    Axis order in the viewer:
        axis 0 → z  (depth downward, mm)
        axis 1 → y  (elevation, mm)
        axis 2 → x  (lateral, mm)

    The specimen boundary is shown as a translucent image layer.
    Defect voxels are shown as a colour-coded labels layer.

    Args:
        specimen:      Specimen3D instance
        defects_3d:    List of Defect3D objects
        voxel_size_mm: Voxel edge length (mm) — trades resolution for speed
    """
    try:
        import napari
    except ImportError:
        print("napari is not installed.  Run:  pip install 'napari[all]'")
        return

    print(f"\nRasterizing volume at {voxel_size_mm} mm/voxel...")
    labels = rasterize_volume(specimen, defects_3d, voxel_size_mm)

    vs = voxel_size_mm  # viewer scale is in mm
    scale = (vs, vs, vs)

    viewer = napari.Viewer(title='NDT Synthetic Volume — napari')

    # Background specimen as a uniform intensity layer (outlines the bounding box)
    background = np.ones_like(labels, dtype=np.float32)
    viewer.add_image(
        background,
        name='Specimen boundary',
        scale=scale,
        colormap='gray',
        opacity=0.05,
        blending='additive',
    )

    # Defect labels layer
    defect_names = {
        0: 'background',
    }
    for i, d in enumerate(defects_3d, start=1):
        defect_names[i] = f"{type(d).__name__} #{i}"

    viewer.add_labels(
        labels,
        name='Defects',
        scale=scale,
        opacity=0.8,
    )

    # Axis labels and physical units
    viewer.dims.axis_labels = ('z — depth (mm)', 'y — elevation (mm)', 'x — lateral (mm)')

    print(f"  Opening napari viewer  (close the window to continue)...")
    print(f"  Defect legend:")
    for i, d in enumerate(defects_3d, start=1):
        print(f"    [{i}] {type(d).__name__}")

    napari.run()


def visualize_scans(scan_dir: str, db_range: float = -20.0) -> None:
    """
    Visualize B-scan frames saved by scan_volume_3d().

    Loads all bscan_*.npy files from scan_dir and produces:
      - bscan_grid.png  — grid of every B-scan with its y-position label
      - bscan_anim.gif  — animated GIF cycling through frames

    The metadata file scan_meta.npy is used for y-position labels if present.

    Args:
        scan_dir:  Directory containing bscan_*.npy (and scan_meta.npy)
        db_range:  Colour-scale lower limit in dB (e.g. -40)
    """
    from matplotlib.animation import FuncAnimation, PillowWriter

    # Load frames in sorted order — avoid glob, which mis-interprets '[' in path names
    paths = sorted([
        os.path.join(scan_dir, f)
        for f in os.listdir(scan_dir)
        if f.startswith('bscan_') and f.endswith('.npy')
    ])
    if not paths:
        print(f"No bscan_*.npy files found in {scan_dir}")
        return

    frames = [np.load(p) for p in paths]
    n = len(frames)

    # Load angle labels from metadata if available
    meta_path = os.path.join(scan_dir, 'scan_meta.npy')
    if os.path.exists(meta_path):
        meta = np.load(meta_path, allow_pickle=True).item()
        if 'angles_rad' in meta:
            labels = [f"{np.degrees(a):+.1f}°" for a in meta['angles_rad']]
        else:
            labels = [f"frame {k}" for k in range(n)]
    else:
        labels = [f"frame {k}" for k in range(n)]

    # ---- Grid PNG ----
    cols = min(8, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols,
                             figsize=(cols * 2.2, rows * 2.5),
                             squeeze=False)
    for ax in axes.flat:
        ax.axis('off')

    for idx, (frame, lbl) in enumerate(zip(frames, labels)):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        ax.imshow(frame, cmap='hot', vmin=db_range, vmax=0,
                  aspect='auto', origin='upper')
        ax.set_title(lbl, fontsize=7)
        ax.axis('on')
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle('TFM B-scans — 3D rotational scan', fontsize=10)
    plt.tight_layout()
    grid_path = os.path.join(scan_dir, 'bscan_grid.png')
    plt.savefig(grid_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved grid:  {grid_path}")

    # ---- Animated GIF ----
    fig_a, ax_a = plt.subplots(figsize=(5, 4))
    im = ax_a.imshow(frames[0], cmap='hot', vmin=db_range, vmax=0,
                     aspect='auto', origin='upper')
    title = ax_a.set_title(f'{labels[0]}  (frame 1/{n})', fontsize=9)
    ax_a.set_xlabel('X pixel')
    ax_a.set_ylabel('Z pixel')
    plt.colorbar(im, ax=ax_a, label='dB', shrink=0.8)
    plt.tight_layout()

    def _update(i: int):
        im.set_data(frames[i])
        title.set_text(f'{labels[i]}  (frame {i+1}/{n})')
        return im, title

    anim = FuncAnimation(fig_a, _update, frames=n, interval=200, blit=True)
    gif_path = os.path.join(scan_dir, 'bscan_anim.gif')
    anim.save(gif_path, writer=PillowWriter(fps=5))
    plt.close()
    print(f"  Saved animation: {gif_path}")


def scan_volume_3d(
    specimen: Specimen3D,
    defects_3d: list,
    cfg: SimulationConfig,
    scan_plan: ScanPlanConfig,
    output_dir: str,
    voxel_volume: 'VoxelVolume3D | None' = None,
    born_threshold: float = 0.005,
) -> None:
    """
    Scan a 3D specimen with a 1D array rotating around the array centre.

    For each angle the 3D content (geometric defects and/or a voxel volume)
    is sliced in the rotated scan plane.  The physics engine produces a full
    FMC acquisition per frame.  Filtered FMC data and TFM B-scans are saved
    as individual .npy files.

    Reconstruction of the 3D volume from the B-scans is a separate step.

    File outputs per frame (zero-padded index i):
        fmc_<i>.npy   — filtered FMC array  (num_tx, num_rx, n_t)
        bscan_<i>.npy — TFM dB image        (n_z, n_x)

    Args:
        specimen:      3D specimen geometry
        defects_3d:    List of Defect3D instances (may be empty)
        cfg:           SimulationConfig (array / acquisition / reconstruction)
        scan_plan:     Rotational scan plan
        output_dir:    Directory for saving per-frame files
        voxel_volume:  Optional VoxelVolume3D for grain-noise / voxel defects.
                       If provided, its Born scatterers are added to every frame.
        born_threshold: Minimum |δZ / 2Z₀| to include a voxel as a scatterer.
    """
    os.makedirs(output_dir, exist_ok=True)
    angles = scan_plan.angles
    n_scans = scan_plan.n_scans
    rc = cfg.reconstruction
    half_w = specimen.width / 2
    z_start = rc.z_start if rc.z_start > 0 else 1e-3

    assert cfg.material is not None, "SimulationConfig.material must be set before scanning"

    # Save scan metadata alongside the frames
    meta = {
        'n_scans': n_scans,
        'angles_rad': angles,
        'angle_step_rad': scan_plan.angle_step,
        'specimen_thickness_m': specimen.thickness,
        'specimen_width_m': specimen.width,
        'specimen_depth_m': specimen.depth,
    }
    np.save(os.path.join(output_dir, 'scan_meta.npy'), meta, allow_pickle=True)  # type: ignore[arg-type]

    print(f"\n{'='*60}")
    print(f"  3D ROTATIONAL SCAN  —  {n_scans} frames")
    print(f"  θ = [{np.degrees(angles[0]):.1f}°, {np.degrees(angles[-1]):.1f}°]"
          f"  step = {np.degrees(scan_plan.angle_step):.2f}°")
    print(f"  Output → {output_dir}")
    print(f"{'='*60}\n")

    # Pre-build Born scattering grids (shared across all frames).
    # Start z at the gate depth to avoid near-surface voxels where the
    # geometric spreading 1/r diverges, which would blow up the FMC amplitude.
    gate_z = cfg.material.c_L * 2e-6 / 2   # depth of 2 µs gate (≈ 6.3 mm for Al)
    born_z_start = max(gate_z * 1.2, 1e-3)  # 20 % margin above gate, at least 1 mm
    born_z_grid = np.linspace(born_z_start, specimen.thickness,
                              max(2, int((specimen.thickness - born_z_start) / 5e-4) + 1))
    born_l_grid = np.linspace(
        -specimen.width / 2, specimen.width / 2,
        max(2, int(specimen.width / 5e-4) + 1),
    )

    for i, theta in enumerate(angles):
        print(f"  Frame {i+1:>3}/{n_scans}  (θ = {np.degrees(theta):+.1f}°)", end="  ")

        # Slice each 3D geometric defect in the rotated scan plane
        engine = FMCEngine(cfg)
        active = 0
        for d3 in defects_3d:
            d2 = d3.slice_at_angle(theta)
            if d2 is not None:
                engine.add_defect(d2)
                active += 1

        # Extract Born scatterers from voxel volume (grain noise / voxel defects)
        n_born = 0
        if voxel_volume is not None:
            assert cfg.material is not None
            z_s, x_s, amp_s = voxel_volume.extract_born_scatterers(
                theta, born_z_grid, born_l_grid,
                background_Z=cfg.material.Z_L,
                threshold=born_threshold,
            )
            if len(z_s) > 0:
                engine.set_born_scatterers(z_s, x_s, amp_s)
                n_born = len(z_s)

        print(f"({active} defect(s), {n_born} Born scatterers)")

        result = engine.simulate()
        fmc = result['fmc_data']
        time_axis = result['time_axis']
        elem_x = result['element_positions']

        # Gate out front-wall echo before noise / filter
        gate_samples = int(2e-6 / cfg.dt)
        fmc[:, :, :gate_samples] = 0.0

        fmc = add_noise(fmc, snr_db=cfg.acquisition.snr_db,
                        grain_noise_level=cfg.acquisition.grain_noise_level)
        fmc = apply_bandpass_filter(fmc, cfg.dt, cfg.array.frequency,
                                    bandwidth_fraction=cfg.array.bandwidth)

        # Exclude back wall from TFM z_range so it doesn't become the 0 dB reference
        z_end_tfm = specimen.thickness - 3e-3  # 3 mm before back wall
        img_db, _, _ = reconstruct_tfm(
            fmc, time_axis, elem_x, cfg.material.c_L,
            x_range=(-half_w, half_w),
            z_range=(z_start, z_end_tfm),
            n_pixels=300,
        )

        # Save this frame
        tag = f"{i:04d}"
        np.save(os.path.join(output_dir, f'fmc_{tag}.npy'), fmc)
        np.save(os.path.join(output_dir, f'bscan_{tag}.npy'), img_db.astype(np.float32))

    print(f"\n  Done — {n_scans} frames saved to {output_dir}/")


def main():
    """
    3D volume scan demo.

    Defines a 3D specimen with 3D defects, then steps a 1D array along the
    elevation (y) axis.  At each y-position the 3D defects are sliced to
    their 2D cross-sections and the physics engine produces a B-scan.
    Each frame's FMC data and TFM B-scan are saved separately.
    3D volume reconstruction from the B-scans is a separate future step.
    """
    print(f"\n{'#'*70}")
    print(f"# PHYSICS-ACCURATE 3D NDT SYNTHETIC DATA ENGINE")
    print(f"{'#'*70}\n")

    # ---- 3D specimen ----
    specimen = Specimen3D(
        thickness=50e-3,   # 50 mm deep
        width=50e-3,       # 50 mm wide  (x, along array)
        depth=30e-3,       # 30 mm tall  (y, mechanical scan axis)
    )

    # ---- 3D geometric defects ----
    defects_3d = [
        # Spherical pore at (z=25mm, x=0, y=0), r=2mm
        SphericalDefect(center_z=25e-3, center_x=0.0, center_y=0.0, radius=2e-3),
        # Side-drilled hole running full depth of specimen (cylinder along y)
        CylindricalDefect(
            center_z=15e-3, center_x=8e-3, radius=1e-3,
            y_start=-specimen.depth / 2, y_end=specimen.depth / 2,
        ),
        # Planar crack in a central y-slab
        PlanarCrack3D(
            start_z=35e-3, start_x=-5e-3,
            end_z=35e-3,   end_x=5e-3,
            y_start=-5e-3, y_end=5e-3,
        ),
    ]

    # ---- Voxel volume — unified grain + defect world ----
    #
    # USE_VOXEL_WORLD = True  → single Born-scattering world:
    #   • Grain noise: Voronoi cells, each ±2.5 % impedance variation (amplitude ≈ ±0.0125)
    #   • Defects embedded as near-zero-impedance voxels (amplitude ≈ −0.5, i.e. R ≈ −1)
    #   • Amplitude ratio defect/grain ≈ 40:1 (≈ 32 dB) — automatic relative filtering:
    #       – With defects present:  defects at 0 dB, grain noise at −32 dB (barely visible)
    #       – With no defects:       grain structure IS the brightest signal → fully visible
    #   • No Kirchhoff scattering used; Born handles everything via the amplitude ratio.
    #
    # USE_VOXEL_WORLD = False → pure Kirchhoff mode (geometric defects only, no grain noise)
    #
    USE_VOXEL_WORLD = True
    voxel_volume = None
    if USE_VOXEL_WORLD:
        print("\nGenerating voxel grain structure...")
        grain_vol = generate_grain_structure(
            thickness           = specimen.thickness,
            width               = specimen.width,
            depth               = specimen.depth,
            background_material = ALUMINUM,
            mean_grain_size_m   = 2.0e-3,   # 2 mm mean grain diameter
            impedance_variation = 0.025,     # ±2.5 % per-grain Z spread → grain amp ≈ ±0.0125
            wavespeed_variation = 0.005,     # ±0.5 % per-grain c_L spread
            voxel_size_m        = 0.5e-3,    # 0.5 mm voxels
        )
        # Burn defects into the grain volume as near-zero impedance (R ≈ −1, amp ≈ −0.5)
        # This gives defect/grain amplitude ratio ≈ 40:1 ≈ 32 dB
        # Set to [] for a grain-only (no defect) test run
        voxel_volume = embed_geometric_defects(grain_vol, [])
        print(f"  Voxel volume shape: {voxel_volume.shape}")

    # ---- Simulation config ----
    # Rotate 180° around the array centre (−90° to +90°)
    scan_plan = ScanPlanConfig(n_scans=16,
                               theta_start=-np.pi / 2,
                               theta_end=np.pi / 2)
    cfg = SimulationConfig(
        specimen=SpecimenConfig(thickness=specimen.thickness, width=specimen.width),
        array=ArrayConfig(num_elements=64, element_pitch=0.6e-3, frequency=10e6),
        scan_plan=scan_plan,
        max_bounces=2,
        mode_conversion=False,
    )
    print(cfg.summary())

    output_dir = os.path.join(os.path.dirname(__file__), 'output', 'scan_3d')

    # ---- Preview volume before scanning ----
    preview_path = os.path.join(output_dir, 'volume_preview.png')
    preview_volume_3d(specimen, defects_3d, scan_plan, preview_path)

    # ---- Interactive 3D napari view (blocks until window is closed) ----
    # Disabled for automated test run; re-enable to inspect volume interactively.
    # view_in_napari(specimen, defects_3d, voxel_size_mm=0.5)

    # ---- Run 3D scan ----
    # Voxel world: defects embedded in grain volume → Born handles everything.
    # Kirchhoff mode: defects_3d passed directly, no grain noise.
    geom_defects = [] if USE_VOXEL_WORLD else defects_3d
    scan_volume_3d(specimen, geom_defects, cfg, scan_plan, output_dir,
                   voxel_volume=voxel_volume)

    print(f"\n{'#'*70}")
    print(f"# SCAN COMPLETE — {scan_plan.n_scans} B-scans saved to {output_dir}/")
    print(f"{'#'*70}\n")

    # ---- Visualize saved frames ----
    visualize_scans(output_dir)


if __name__ == '__main__':
    main()
