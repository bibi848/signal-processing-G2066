"""
Convert .npy scan data directories into .mat files for MATLAB.

Each scan directory produces one .mat file containing:
  - bscans:          (n_scans, Nz, Nx) float32        — dB B-scans
  - bscans_complex:  (n_scans, Nz, Nx) complex64      — complex B-scans (if present)
  - fmc:             (n_scans, Ntx, Nrx, Nt) float32  — FMC data
  - All fields from scan_meta.npy (angles_rad, specimen dims, etc.)
  - ground_truth_impedance / ground_truth_wavespeed (if ground_truth.npz exists)
"""

import numpy as np
import scipy.io as sio
import os
import sys

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
MAT_DIR = os.path.join(OUTPUT_DIR, "mat_exports")

# Directories to convert — add/remove as needed
# Paths are relative to OUTPUT_DIR
SCAN_DIRS = [
    "radon_tests/radon_test",
    "radon_tests/compound_test",
    "radon_tests/radon_test_sphere2mm",
    "radon_tests/radon_test_sphere_close",
    "scans/scan_3d",
    "scans/scan_3d_with_defect",
]


def find_numbered_files(scan_path, prefix):
    """Find files like prefix_0000.npy, prefix_0001.npy, ... without glob
    (glob breaks on paths containing bracket characters like [3])."""
    files = []
    for fname in sorted(os.listdir(scan_path)):
        if fname.startswith(prefix + "_") and fname.endswith(".npy"):
            # Check that the part after prefix_ is numeric
            stem = fname[len(prefix) + 1 : -4]
            if stem.isdigit():
                files.append(os.path.join(scan_path, fname))
    return files


def convert_scan_dir(scan_dir_name):
    scan_path = os.path.join(OUTPUT_DIR, scan_dir_name)
    if not os.path.isdir(scan_path):
        print(f"  SKIP  {scan_dir_name} — directory not found")
        return

    meta_path = os.path.join(scan_path, "scan_meta.npy")
    if not os.path.exists(meta_path):
        print(f"  SKIP  {scan_dir_name} — no scan_meta.npy")
        return

    meta = np.load(meta_path, allow_pickle=True).item()
    n_scans = meta["n_scans"]
    print(f"  Converting {scan_dir_name}  ({n_scans} scans) …")

    mat_dict = {}

    # --- Metadata fields ---
    for key, val in meta.items():
        mat_dict[key] = val

    # --- B-scans (dB) ---
    bscan_files = find_numbered_files(scan_path, "bscan")
    if bscan_files:
        arrays = [np.load(f) for f in bscan_files]
        # Check for mixed shapes (e.g. resolution changed mid-scan)
        shapes = set(a.shape for a in arrays)
        if len(shapes) == 1:
            bscans = np.stack(arrays)
            mat_dict["bscans"] = bscans
            print(f"    bscans:         {bscans.shape}  {bscans.dtype}")
        else:
            # Keep only the most common shape
            from collections import Counter
            shape_counts = Counter(a.shape for a in arrays)
            dominant_shape = shape_counts.most_common(1)[0][0]
            filtered = [a for a in arrays if a.shape == dominant_shape]
            bscans = np.stack(filtered)
            mat_dict["bscans"] = bscans
            print(f"    bscans:         {bscans.shape}  {bscans.dtype}  (kept {len(filtered)}/{len(arrays)} with shape {dominant_shape})")

    # --- Complex B-scans ---
    bscan_complex_files = find_numbered_files(scan_path, "bscan_complex")
    if bscan_complex_files:
        bscans_c = np.stack([np.load(f) for f in bscan_complex_files])
        mat_dict["bscans_complex_real"] = bscans_c.real
        mat_dict["bscans_complex_imag"] = bscans_c.imag
        print(f"    bscans_complex: {bscans_c.shape}  {bscans_c.dtype}")

    # --- Real B-scans (if saved separately) ---
    bscan_real_files = find_numbered_files(scan_path, "bscan_real")
    if bscan_real_files:
        bscans_r = np.stack([np.load(f) for f in bscan_real_files])
        mat_dict["bscans_real"] = bscans_r
        print(f"    bscans_real:    {bscans_r.shape}  {bscans_r.dtype}")

    # --- FMC data ---
    fmc_files = find_numbered_files(scan_path, "fmc")
    if fmc_files:
        fmc = np.stack([np.load(f) for f in fmc_files])
        # v5 .mat has a 4 GB per-element limit
        if fmc.nbytes < 4 * 1024**3:
            mat_dict["fmc"] = fmc
            print(f"    fmc:            {fmc.shape}  {fmc.dtype}")
        else:
            size_gb = fmc.nbytes / 1e9
            print(f"    fmc:            {fmc.shape}  {fmc.dtype}  — {size_gb:.1f} GB, splitting per-frame …")
            # Save each FMC frame as a separate .mat
            fmc_dir = os.path.join(MAT_DIR, f"{os.path.basename(scan_dir_name)}_fmc")
            os.makedirs(fmc_dir, exist_ok=True)
            for idx in range(fmc.shape[0]):
                fmc_path = os.path.join(fmc_dir, f"fmc_{idx:04d}.mat")
                sio.savemat(fmc_path, {"fmc": fmc[idx]}, do_compression=True)
            print(f"                    Saved {fmc.shape[0]} files in {scan_dir_name}_fmc/")

    # --- Ground truth ---
    gt_path = os.path.join(scan_path, "ground_truth.npz")
    if os.path.exists(gt_path):
        gt = np.load(gt_path)
        for key in gt.files:
            mat_dict[f"ground_truth_{key}"] = gt[key]
        print(f"    ground_truth:   keys={gt.files}")

    # --- Save .mat ---
    os.makedirs(MAT_DIR, exist_ok=True)
    out_path = os.path.join(MAT_DIR, f"{os.path.basename(scan_dir_name)}.mat")
    sio.savemat(out_path, mat_dict, do_compression=True)
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"    Saved: {out_path}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    dirs = sys.argv[1:] if len(sys.argv) > 1 else SCAN_DIRS
    print(f"Converting {len(dirs)} scan directories to .mat …\n")
    for d in dirs:
        convert_scan_dir(d)
    print("\nDone.")
