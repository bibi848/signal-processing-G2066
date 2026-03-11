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
import sys
import os
import time

# Add parent directory for Classes/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from Classes.Filter import filter_signal
from Classes.TFM1D import CTFM1D

from engine.config import SimulationConfig, SpecimenConfig, ArrayConfig, AcquisitionConfig
from engine.geometry import CircularDefect, CrackDefect, FlatBottomHole
from engine.fmc_engine import FMCEngine
from engine.materials import ALUMINUM, STEEL_MILD, STEEL_STAINLESS, WATER, NDT_GEL


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


def main():
    """Main entry point for the physics engine simulation."""
    print(f"\n{'#'*70}")
    print(f"# PHYSICS-ACCURATE 2D NDT SYNTHETIC DATA ENGINE")
    print(f"{'#'*70}\n")

    # ---- Configuration ----
    cfg = SimulationConfig(
        specimen=SpecimenConfig(thickness=50e-3, width=50e-3),
        array=ArrayConfig(num_elements=64, element_pitch=0.6e-3, frequency=10e6),
        max_bounces=2,
        mode_conversion=False,
    )

    # ---- Defects ----
    defects = [
        # Side-drilled hole at 25mm depth, centered
        CircularDefect(center_z=25e-3, center_x=0.0, radius=2e-3),
        # Small SDH at 15mm depth, offset
        CircularDefect(center_z=15e-3, center_x=8e-3, radius=1e-3),
    ]

    # ---- Simulate ----
    engine = FMCEngine(cfg)
    for d in defects:
        engine.add_defect(d)

    result = engine.simulate()
    fmc_raw = result['fmc_data']
    time_axis = result['time_axis']
    elem_x = result['element_positions']

    # Preserve raw A-scan for visualization before gating destroys the front-wall echo
    fmc_for_ascan = fmc_raw.copy()

    # ---- Time gate (remove front-wall echo for TFM) ----
    gate_samples = int(2e-6 / cfg.dt)
    fmc_raw[:, :, :gate_samples] = 0.0

    # ---- Add noise ----
    fmc_noisy = add_noise(fmc_raw, snr_db=35.0, grain_noise_level=0.03)

    # ---- Bandpass filter ----
    fmc_filtered = apply_bandpass_filter(
        fmc_noisy, cfg.dt, cfg.array.frequency, bandwidth_fraction=0.9
    )

    # ---- TFM reconstruction ----
    half_w = cfg.specimen.width / 2
    img_db, x_img, z_img = reconstruct_tfm(
        fmc_filtered, time_axis, elem_x, cfg.material.c_L,
        x_range=(-half_w, half_w),
        z_range=(1e-3, cfg.specimen.thickness),
        n_pixels=300,
    )

    # ---- Visualize ----
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)

    viz_path = os.path.join(output_dir, 'engine_result.png')
    visualize(img_db, x_img, z_img, fmc_for_ascan, fmc_filtered, time_axis,
              defects, cfg, output_path=viz_path)

    # ---- Save FMC data ----
    fmc_path = os.path.join(output_dir, 'engine_fmc_data.npy')
    np.save(fmc_path, fmc_filtered)
    print(f"  Saved: {fmc_path} ({fmc_filtered.shape})")

    print(f"\n{'#'*70}")
    print(f"# SIMULATION COMPLETE")
    print(f"{'#'*70}\n")


if __name__ == '__main__':
    main()
