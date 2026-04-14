"""
Export synthetic scan data to .mat for the MATLAB iradon reconstruction script.

Produces a .mat file with:
    result_tfm  — (Nz, Nx, N_theta) complex128  [z, x, theta]
    theta       — (N_theta,) float64             angles in degrees
    x           — (Nx,) float64                  lateral coords in metres
    z           — (Nz,) float64                  depth coords in metres

Usage:
    python export_for_matlab.py                          # default scan dir
    python export_for_matlab.py /path/to/scan_data       # custom scan dir
"""

import os
import sys
import numpy as np
import scipy.io as sio


def find_numbered_files(scan_path, prefix):
    """Find files like prefix_0000.npy, prefix_0001.npy, ..."""
    files = []
    for fname in sorted(os.listdir(scan_path)):
        if fname.startswith(prefix + "_") and fname.endswith(".npy"):
            stem = fname[len(prefix) + 1 : -4]
            if stem.isdigit():
                files.append(os.path.join(scan_path, fname))
    return files


def export(scan_dir: str, out_path: str | None = None):
    # --- Load metadata ---
    meta = np.load(
        os.path.join(scan_dir, "scan_meta.npy"), allow_pickle=True
    ).item()

    angles_rad = meta["angles_rad"]
    theta_deg = np.degrees(angles_rad).astype(np.float64)
    n_scans = meta["n_scans"]
    tfm_z_start = meta["tfm_z_start_m"]
    tfm_z_end = meta["tfm_z_end_m"]
    tfm_n_pixels = meta["tfm_n_pixels"]
    aperture = meta["array_aperture_m"]

    # --- Coordinate vectors (match MATLAB conventions) ---
    half_w = aperture / 2
    x = np.linspace(-half_w, half_w, tfm_n_pixels, dtype=np.float64)
    z = np.linspace(tfm_z_start, tfm_z_end, tfm_n_pixels, dtype=np.float64)

    # --- Load complex B-scans ---
    bscan_files = find_numbered_files(scan_dir, "bscan_complex")
    if not bscan_files:
        raise FileNotFoundError("No bscan_complex_*.npy files found")

    print(f"Loading {len(bscan_files)} complex B-scans from {scan_dir}")
    bscans = np.stack([np.load(f) for f in bscan_files])
    # bscans shape: (n_scans, Nz, Nx)
    print(f"  bscans shape: {bscans.shape}  dtype: {bscans.dtype}")

    # --- Build result_tfm as (Nz, Nx, N_theta) ---
    # MATLAB: result_tfm(iz, :, :) -> squeeze -> (Nx, N_theta)
    result_tfm = np.transpose(bscans, (1, 2, 0)).astype(np.complex128)
    print(f"  result_tfm shape: {result_tfm.shape}  [Nz, Nx, N_theta]")

    # --- Save ---
    if out_path is None:
        out_path = os.path.join(
            os.path.dirname(scan_dir), "result_tfm_for_matlab.mat"
        )

    sio.savemat(
        out_path,
        {
            "result_tfm": result_tfm,
            "theta": theta_deg,
            "x": x,
            "z": z,
        },
        do_compression=True,
    )
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"\nSaved: {out_path}  ({size_mb:.1f} MB)")
    print(f"  result_tfm: {result_tfm.shape} complex128")
    print(f"  theta:      {theta_deg.shape}  [{theta_deg[0]:.1f}° .. {theta_deg[-1]:.1f}°]")
    print(f"  x:          {x.shape}  [{x[0]*1e3:.2f} .. {x[-1]*1e3:.2f}] mm")
    print(f"  z:          {z.shape}  [{z[0]*1e3:.2f} .. {z[-1]*1e3:.2f}] mm")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_scan_dir = os.path.join(
        script_dir, "output", "recon_3d_clean", "scan_data"
    )
    scan_dir = sys.argv[1] if len(sys.argv) > 1 else default_scan_dir
    export(scan_dir)
