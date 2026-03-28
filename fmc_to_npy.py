"""
FMC → TFM → NPY: Full-precision B-scan generation for 3D reconstruction
========================================================================

Runs TFM imaging directly from FMC time-domain data (HDF5) and saves
the linear-amplitude envelope as .npy files — no PNG intermediate,
no dB clipping, no colormap quantisation.

Output is compatible with ``reconstruct_3d.py``.

Usage:
    python fmc_to_npy.py --input "DATA/1D Processed Data/Al Hole 5MHz 02022026 Filtered" \
                         --output "DATA/1D NPY Data/Al Hole 5MHz 02022026 Raw"

    python fmc_to_npy.py --input "DATA/1D Processed Data/Al Hole 5MHz 02022026 Filtered" \
                         --output "DATA/1D NPY Data/Al Hole 5MHz 02022026 Raw" \
                         --engine python   # if C++ module not available
"""

import os
import re
import sys
import argparse
import time as timer
import numpy as np
import pandas as pd
import h5py
from scipy.signal import hilbert


def _natural_key(s):
    """Sort key for natural ordering (scan_2 before scan_10)."""
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]


def _load_cpp_engine():
    """Try to load the C++ TFM module."""
    build_dir = os.path.join(os.path.dirname(__file__), "build", "CPP", "TFM")
    sys.path.insert(0, build_dir)
    try:
        import tfm_cpp
        return tfm_cpp
    except ImportError:
        return None


def run_tfm(time_data, time_sec, tx, rx, xc, zc, c, x_img, z_img,
            engine='cpp', tfm_cpp_mod=None):
    """
    Run TFM and return the linear envelope (no dB conversion).

    Args:
        time_data: (N_fmc, N_t) float32 FMC data
        time_sec:  (N_t,) time axis in seconds
        tx, rx:    (N_fmc,) 1-based transmit/receive element indices
        xc, zc:    element positions in metres
        c:         wave speed (m/s)
        x_img:     (n_x,) lateral image coordinates
        z_img:     (n_z,) depth image coordinates
        engine:    'cpp' or 'python'
        tfm_cpp_mod: preloaded tfm_cpp module (for engine='cpp')

    Returns:
        (n_z, n_x) float32 linear envelope image
    """
    X, Z = np.meshgrid(x_img, z_img)

    if engine == 'cpp' and tfm_cpp_mod is not None:
        tx0 = tx - 1
        rx0 = rx - 1
        img = tfm_cpp_mod.tfm1D(time_data, time_sec, tx0, rx0, xc, zc, X, Z, c)
    else:
        # Pure Python fallback
        Nf, Nt = time_data.shape
        dt = time_sec[1] - time_sec[0]
        t0 = time_sec[0]
        tx0 = tx - 1
        rx0 = rx - 1
        img = np.zeros_like(X)
        for i in range(Nf):
            d_tx = np.sqrt((X - xc[tx0[i]])**2 + (Z - zc[tx0[i]])**2)
            d_rx = np.sqrt((X - xc[rx0[i]])**2 + (Z - zc[rx0[i]])**2)
            idx_f = ((d_tx + d_rx) / c - t0) / dt
            i0 = np.floor(idx_f).astype(int)
            w = idx_f - i0
            valid = (i0 >= 0) & (i0 < Nt - 1)
            i0_clipped = np.clip(i0, 0, Nt - 2)
            s0 = time_data[i, i0_clipped]
            s1 = time_data[i, i0_clipped + 1]
            img += valid * ((1.0 - w) * s0 + w * s1)

    # Hilbert envelope (linear amplitude)
    img_analytic = hilbert(img, axis=0)
    img_envelope = np.abs(img_analytic).astype(np.float32)
    return img_envelope


def convert_dataset(
    input_dir: str,
    output_dir: str,
    c: float = None,
    z_start: float = None,
    z_end: float = None,
    x_pixels: int = 400,
    z_pixels: int = 400,
    engine: str = 'cpp',
) -> str:
    """
    Run TFM on all scans in a processed data directory and save .npy files.

    Args:
        input_dir:  Directory with scan subfolders (each containing time_data.h5 etc.)
        output_dir: Where to save bscan_XXXX.npy + scan_meta.npy
        c:          Wave speed (m/s). If None, read from first scan's metadata.
        z_start:    TFM start depth (m). Default: 0.
        z_end:      TFM end depth (m). Default: inferred from time axis.
        x_pixels:   Lateral pixel count.
        z_pixels:   Depth pixel count.
        engine:     'cpp' or 'python'.

    Returns:
        Path to output directory.
    """
    # Find scan subfolders
    scan_folders = sorted(
        [f for f in os.listdir(input_dir)
         if os.path.isdir(os.path.join(input_dir, f))],
        key=_natural_key,
    )
    if not scan_folders:
        raise FileNotFoundError(f"No scan subfolders found in {input_dir}")

    print(f"Found {len(scan_folders)} scans in {input_dir}")

    # Load C++ engine if requested
    tfm_cpp_mod = None
    if engine == 'cpp':
        tfm_cpp_mod = _load_cpp_engine()
        if tfm_cpp_mod is None:
            print("Warning: C++ TFM module not found, falling back to Python")
            engine = 'python'
        else:
            print("Using C++ TFM engine")

    os.makedirs(output_dir, exist_ok=True)
    bscans = []
    source_files = []

    for i, fol in enumerate(scan_folders):
        t_start = timer.time()
        folder_path = os.path.join(input_dir, fol)

        # Load data
        metadata = pd.read_csv(os.path.join(folder_path, "metadata.csv"))
        time_sec = pd.read_csv(os.path.join(folder_path, "time.csv"))["time_seconds"].values
        tx_rx = pd.read_csv(os.path.join(folder_path, "tx_rx.csv"))
        geometry = pd.read_csv(os.path.join(folder_path, "array_geometry.csv"))

        with h5py.File(os.path.join(folder_path, "time_data.h5"), "r") as h5f:
            time_data = h5f["time_data"][:]

        tx = tx_rx["tx"].values.astype(int)
        rx = tx_rx["rx"].values.astype(int)
        xc = geometry["el_xc"].values
        zc = geometry["el_zc"].values

        # Infer parameters from first scan
        if i == 0:
            if c is None:
                freq = float(metadata.loc[
                    metadata['Field'] == 'centre_frequency_Hz', 'Value'
                ].iloc[0])
                # Guess wave speed from directory name
                dirname = os.path.basename(input_dir).lower()
                if 'al' in dirname:
                    c = 6320.0
                elif 'cu' in dirname:
                    c = 4700.0
                elif 'steel' in dirname:
                    c = 5960.0
                else:
                    c = 6320.0
                print(f"  Wave speed: {c:.0f} m/s (inferred from directory name)")

            if z_start is None:
                z_start = 0.0
            if z_end is None:
                max_depth = c * time_sec[-1] / 2.0
                z_end = min(max_depth, 50e-3)  # cap at 50mm
                print(f"  Depth range: {z_start*1e3:.1f} – {z_end*1e3:.1f} mm")

            x_min, x_max = xc.min(), xc.max()
            x_img = np.linspace(x_min, x_max, x_pixels)
            z_img = np.linspace(z_start, z_end, z_pixels)
            print(f"  Image grid: {z_pixels} x {x_pixels} pixels")
            print()

        # Run TFM
        envelope = run_tfm(
            time_data, time_sec, tx, rx, xc, zc, c,
            x_img, z_img, engine=engine, tfm_cpp_mod=tfm_cpp_mod,
        )
        bscans.append(envelope)
        source_files.append(fol)

        # Save
        npy_name = f'bscan_{i:04d}.npy'
        np.save(os.path.join(output_dir, npy_name), envelope)

        elapsed = timer.time() - t_start
        print(f"  [{i+1}/{len(scan_folders)}] {fol} → {npy_name}  "
              f"shape={envelope.shape}  max={envelope.max():.4f}  ({elapsed:.1f}s)")

    # Build scan_meta.npy
    n_scans = len(bscans)
    angles_rad = np.linspace(0, np.pi, n_scans, endpoint=False)

    meta = {
        'n_scans': n_scans,
        'angles_rad': angles_rad.astype(np.float64),
        'specimen_thickness_m': float(z_end),
        'specimen_width_m': float(x_max - x_min),
        'tfm_z_start_m': float(z_start),
        'tfm_z_end_m': float(z_end),
        'array_aperture_m': float(x_max - x_min),
        'tfm_n_pixels': int(z_pixels),
        'source_dir': os.path.abspath(input_dir),
        'source_files': source_files,
        'wave_speed': float(c),
        'data_format': 'linear_envelope',
    }

    np.save(os.path.join(output_dir, 'scan_meta.npy'), meta)

    print(f"\nSaved {n_scans} B-scans + scan_meta.npy to {output_dir}")
    print(f"  Shape per frame: {bscans[0].shape}")
    print(f"  Data format: linear envelope (float32)")
    print(f"  Angles: {np.rad2deg(angles_rad[0]):.1f}° to {np.rad2deg(angles_rad[-1]):.1f}°")

    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description='Run TFM from FMC data and save full-precision .npy B-scans'
    )
    parser.add_argument('--input', '-i', required=True,
                        help='Input directory with scan subfolders')
    parser.add_argument('--output', '-o', required=True,
                        help='Output directory for .npy files')
    parser.add_argument('--c', type=float, default=None,
                        help='Wave speed in m/s (default: inferred from material)')
    parser.add_argument('--z-start', type=float, default=None,
                        help='TFM start depth in metres')
    parser.add_argument('--z-end', type=float, default=None,
                        help='TFM end depth in metres')
    parser.add_argument('--x-pixels', type=int, default=400,
                        help='Lateral pixel count (default: 400)')
    parser.add_argument('--z-pixels', type=int, default=400,
                        help='Depth pixel count (default: 400)')
    parser.add_argument('--engine', choices=['cpp', 'python'], default='cpp',
                        help='TFM engine (default: cpp)')

    args = parser.parse_args()

    convert_dataset(
        input_dir=args.input,
        output_dir=args.output,
        c=args.c,
        z_start=args.z_start,
        z_end=args.z_end,
        x_pixels=args.x_pixels,
        z_pixels=args.z_pixels,
        engine=args.engine,
    )


if __name__ == '__main__':
    main()
