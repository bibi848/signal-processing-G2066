"""
Single scatterer → rotational FMC with 3D engine + 1D array → Radon.

Pipeline:
  1. Build a 1D linear array along world x (y = 0 fixed).
  2. For each θ ∈ [0, π), place the single spherical defect at its
     array-frame position (rotate by −θ around z). The 3D Born engine
     naturally models the off-elevation (y≠0) component of the delay.
  3. TFM-reconstruct a 2D (z, x) slice at y = 0 using the C++ 3D-distance
     kernel → complex B-scan at angle θ.
  4. Save as bscan_complex_NNNN.npy + scan_meta.npy.
  5. Inverse Radon per z-plane via Classes.Reconstruct3D.reconstruct_scan.

Run from the SYNTHETIC DATA directory:
    python tests/test_radon_single_scatterer.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
import numpy as np
from scipy.signal import hilbert

HERE    = Path(__file__).resolve().parent
SD_ROOT = HERE.parent
REPO    = SD_ROOT.parent
sys.path.insert(0, str(SD_ROOT))
sys.path.insert(0, str(REPO))

from engine_3d import (
    SimulationConfig3D, ArrayConfig3D, SpecimenConfig3D, AcquisitionConfig3D,
    SphericalDefect, FMCEngine3D, ALUMINUM,
)
from Classes.Filter import filter_signal
from Classes.Reconstruct3D import reconstruct_scan, load_meta, view_napari

# tfm_cpp for 3D-distance TFM focusing
sys.path.insert(0, str(REPO / "build" / "CPP" / "TFM"))
import tfm_cpp


# ---- Array ----
N_ELEMENTS   = 128
ELEMENT_PITCH = 0.3e-3
FREQUENCY    = 10e6
BANDWIDTH    = 0.8
TIME_SAMPLES = 2048

# ---- Specimen ----
THICKNESS = 50e-3
WIDTH     = 50e-3
DEPTH     = 50e-3

# ---- Defect (world frame, before rotation) ----
DEFECT_X  = +6e-3
DEFECT_Y  = +4e-3
DEFECT_Z  =  25e-3
DEFECT_R  = 0.8e-3
DEFECT_N_POINTS = 600

# ---- Scan ----
N_SCANS = 60

# ---- TFM gate ----
TFM_N_PIXELS = 400
TFM_Z_START  = 5e-3
TFM_Z_END    = 45e-3

# ---- Output ----
OUTPUT_DIR = HERE / 'output' / 'radon_single_scatterer'


def build_cfg() -> SimulationConfig3D:
    """1D array along x, all elements at y=0."""
    x = (np.arange(N_ELEMENTS) - (N_ELEMENTS - 1) / 2) * ELEMENT_PITCH
    positions = np.stack([x, np.zeros_like(x)], axis=1)
    return SimulationConfig3D(
        material=ALUMINUM,
        array=ArrayConfig3D(
            custom_positions=positions,
            pitch_x=ELEMENT_PITCH, pitch_y=ELEMENT_PITCH,
            element_width_x=ELEMENT_PITCH * 0.9,
            element_width_y=ELEMENT_PITCH * 0.9,
            frequency=FREQUENCY, bandwidth=BANDWIDTH,
        ),
        specimen=SpecimenConfig3D(thickness=THICKNESS, width=WIDTH, depth=DEPTH),
        acquisition=AcquisitionConfig3D(
            time_samples=TIME_SAMPLES,
            snr_db=40.0, grain_noise_level=0.0, add_noise=False,
        ),
    )


def bandpass_fmc(fmc: np.ndarray, dt: float) -> np.ndarray:
    f_c = FREQUENCY / 1e6
    f_lo = max(f_c * (1 - BANDWIDTH / 2), 0.1)
    f_hi = f_c * (1 + BANDWIDTH / 2)
    out = np.zeros_like(fmc)
    n_tx, n_rx, _ = fmc.shape
    for t in range(n_tx):
        for r in range(n_rx):
            out[t, r, :] = filter_signal(fmc[t, r, :], dt, f_lo, f_hi,
                                         filter_alpha=1.0, hanning_bool=False)
    return out


def tfm_slice_y0(result: dict) -> np.ndarray:
    """2D (z, x) TFM at y=0 using 3D element-to-pixel distances. Returns
    a complex analytic B-scan of shape (n_z, n_x)."""
    fmc = result['fmc_data']
    elem = result['element_positions_xyz']          # (n_el, 3) cols (z, x, y)
    time_axis = result['time_axis']
    cfg: SimulationConfig3D = result['config']
    c = float(cfg.material.c_L)

    n_el, _, n_t = fmc.shape
    n_fmc = n_el * n_el

    tx, rx = np.meshgrid(np.arange(n_el, dtype=np.int32),
                         np.arange(n_el, dtype=np.int32), indexing='ij')
    tx0 = tx.ravel()
    rx0 = rx.ravel()
    time_data = fmc.reshape(n_fmc, n_t).astype(np.float64, copy=False)

    zc = elem[:, 0].astype(np.float64, copy=False)
    xc = elem[:, 1].astype(np.float64, copy=False)
    yc = elem[:, 2].astype(np.float64, copy=False)

    x_img = np.linspace(-(N_ELEMENTS - 1) * ELEMENT_PITCH / 2,
                        +(N_ELEMENTS - 1) * ELEMENT_PITCH / 2, TFM_N_PIXELS)
    z_img = np.linspace(TFM_Z_START, TFM_Z_END, TFM_N_PIXELS)
    y_img = np.array([0.0])
    Z, Y, X = np.meshgrid(z_img, y_img, x_img, indexing='ij')

    img = tfm_cpp.tfm2D(
        time_data, time_axis.astype(np.float64),
        tx0, rx0, xc, yc, zc, X, Y, Z, c,
    )
    img2d = np.squeeze(img, axis=1)                  # (n_z, n_x)
    return hilbert(img2d, axis=0).astype(np.complex64)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cfg = build_cfg()
    print(cfg.summary())
    print(f"\nDefect (world): ({DEFECT_X*1e3:+.1f}, {DEFECT_Y*1e3:+.1f}, "
          f"{DEFECT_Z*1e3:.1f}) mm  r={DEFECT_R*1e3:.1f} mm")
    print(f"Angles: {N_SCANS}  step {180/N_SCANS:.2f}°  [0°, 180°)\n")

    angles = np.linspace(0.0, np.pi, N_SCANS, endpoint=False)
    aperture = (N_ELEMENTS - 1) * ELEMENT_PITCH

    t_total = time.time()
    for i, theta in enumerate(angles):
        # Rotate defect by -θ around z → array-frame position
        ct, st = np.cos(theta), np.sin(theta)
        x_arr = DEFECT_X * ct + DEFECT_Y * st
        y_arr = -DEFECT_X * st + DEFECT_Y * ct

        engine = FMCEngine3D(cfg)
        engine.add_defect(
            SphericalDefect(center_x=x_arr, center_y=y_arr,
                            center_z=DEFECT_Z, radius=DEFECT_R),
            n_points=DEFECT_N_POINTS,
        )
        t0 = time.time()
        result = engine.simulate(tx_chunk=1, verbose=False)
        result['fmc_data'] = bandpass_fmc(result['fmc_data'], cfg.dt)
        bscan = tfm_slice_y0(result)
        np.save(OUTPUT_DIR / f'bscan_complex_{i:04d}.npy', bscan)
        print(f"  frame {i+1:>3}/{N_SCANS}  θ={np.degrees(theta):+6.1f}°  "
              f"{time.time() - t0:5.1f}s")

    meta = {
        'n_scans': N_SCANS,
        'angles_rad': angles,
        'angle_step_rad': np.pi / N_SCANS,
        'specimen_thickness_m': THICKNESS,
        'specimen_width_m': WIDTH,
        'specimen_depth_m': DEPTH,
        'tfm_z_start_m': TFM_Z_START,
        'tfm_z_end_m': TFM_Z_END,
        'tfm_n_pixels': TFM_N_PIXELS,
        'array_aperture_m': aperture,
        'has_complex_data': True,
    }
    np.save(OUTPUT_DIR / 'scan_meta.npy', meta, allow_pickle=True)
    print(f"\nScan complete in {time.time() - t_total:.1f}s")

    # ---- Inverse Radon per z-plane ----
    volume = reconstruct_scan(str(OUTPUT_DIR), show_napari=False)

    # Peak location vs ground truth
    meta = load_meta(str(OUTPUT_DIR))
    n_z, _, n_x = volume.shape
    z = np.linspace(meta['tfm_z_start_m'], meta['tfm_z_end_m'], n_z)
    half = meta['array_aperture_m'] / 2
    xy = np.linspace(-half, half, n_x)
    iz, iy, ix = np.unravel_index(np.argmax(volume), volume.shape)
    print(f"\nPeak voxel:   (x={xy[ix]*1e3:+.2f}, y={xy[iy]*1e3:+.2f}, "
          f"z={z[iz]*1e3:.2f}) mm")
    print(f"Ground truth: (x={DEFECT_X*1e3:+.2f}, y={DEFECT_Y*1e3:+.2f}, "
          f"z={DEFECT_Z*1e3:.2f}) mm")

    view_napari(volume, meta)


if __name__ == '__main__':
    main()
