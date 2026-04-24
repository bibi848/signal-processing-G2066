"""
Radon reconstruction on a polycrystalline grain volume with embedded defects.

Same scatterer positions as test_radon_validation.py, but the specimen is a
Voronoi grain structure. Grain-boundary impedance jumps emit Born scatterers,
so each TFM slice contains coherent speckle on top of the defect signal. The
purpose of this script is visual: after filtered backprojection the defects
should still localise, while grain speckle partially averages across angles.

Pipeline:
  1. Generate grain volume (aluminum, Voronoi tessellation).
  2. Embed the same 3 spherical defects as the clean validation.
  3. Extract 3D Born scatterer cloud (signed per-axis gradient / 2Z0).
  4. Cache (cloud + B-scans). For each angle theta, rotate the cloud by -theta
     around z and simulate FMC + TFM (y=0 slice) -> complex B-scan.
  5. iradon per z-plane at several N values in N_SWEEP.

Run from the SYNTHETIC DATA directory:
    python tests/test_radon_grain_validation.py

Then view with plot_report_figures:
    python tests/plot_report_figures.py radon_grain_validation
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
import numpy as np

HERE    = Path(__file__).resolve().parent
SD_ROOT = HERE.parent
REPO    = SD_ROOT.parent
sys.path.insert(0, str(SD_ROOT))
sys.path.insert(0, str(REPO))

from engine_3d import (
    SphericalDefect, FMCEngine3D, ALUMINUM, extract_born_scatterers_3d,
)
from engine.microstructure import (
    generate_grain_structure, embed_geometric_defects,
)

# Reuse everything shared with the clean validation
from test_radon_validation import (
    SCATTERERS,
    N_ELEMENTS, ELEMENT_PITCH,
    THICKNESS, WIDTH, DEPTH,
    TFM_N_PIXELS, TFM_Z_START, TFM_Z_END,
    build_cfg, bandpass_fmc, tfm_slice_y0,
    reconstruct, measure, axes_zxy,
)

# ---- Grain structure ----
GRAIN_SIZE_M         = 2.0e-3     # mean grain diameter
IMPEDANCE_VARIATION  = 0.025      # +/- per-grain fractional Z variation
WAVESPEED_VARIATION  = 0.005      # +/- per-grain fractional c_L variation
VOXEL_SIZE_M         = 0.5e-3     # voxel edge (needs to resolve grains)
BORN_THRESHOLD       = 0.02       # emit a scatterer when |dZ / 2Z0| exceeds this
GRAIN_SEED           = 42
VOID_IMPEDANCE_FACTOR = 0.001     # embedded defect impedance = Z0 * this

# Safety cap — if extraction yields more than this, randomly subsample so the
# FMC sim stays tractable. Set to None to disable.
MAX_SCATTERERS       = 40_000

# ---- Angular sampling ----
N_MAX   = 60                      # expensive with grains; keep modest
N_SWEEP = [12, 30, 60]            # 15 deg, 6 deg, 3 deg steps

# ---- Output ----
OUTPUT_DIR = HERE / 'output' / 'radon_grain_validation'


def build_grain_cloud(seed: int = GRAIN_SEED) -> tuple[np.ndarray, np.ndarray,
                                                         np.ndarray, np.ndarray]:
    """Grain volume -> embed defects -> extract Born scatterers. Returns
    (z_s, x_s, y_s, amp_s) in world coordinates."""
    print("\nGenerating grain volume...")
    t0 = time.time()
    vol = generate_grain_structure(
        thickness=THICKNESS, width=WIDTH, depth=DEPTH,
        background_material=ALUMINUM,
        mean_grain_size_m=GRAIN_SIZE_M,
        impedance_variation=IMPEDANCE_VARIATION,
        wavespeed_variation=WAVESPEED_VARIATION,
        voxel_size_m=VOXEL_SIZE_M,
        seed=seed,
    )
    print(f"  {time.time() - t0:.1f}s")

    print("Embedding defects...")
    defects = [SphericalDefect(center_x=sx, center_y=sy, center_z=sz, radius=sr)
               for sx, sy, sz, sr in SCATTERERS]
    vol = embed_geometric_defects(vol, defects,
                                  void_impedance_factor=VOID_IMPEDANCE_FACTOR)

    print("Extracting Born scatterers...")
    t0 = time.time()
    z_s, x_s, y_s, amp_s = extract_born_scatterers_3d(
        vol, background_Z=ALUMINUM.Z_L, threshold=BORN_THRESHOLD,
    )
    n = len(z_s)
    print(f"  {n} scatterers  ({time.time() - t0:.1f}s)")

    if MAX_SCATTERERS is not None and n > MAX_SCATTERERS:
        rng = np.random.default_rng(seed + 1)
        keep = rng.choice(n, MAX_SCATTERERS, replace=False)
        z_s, x_s, y_s, amp_s = z_s[keep], x_s[keep], y_s[keep], amp_s[keep]
        print(f"  subsampled to {MAX_SCATTERERS}  (cap)")

    return z_s, x_s, y_s, amp_s


def simulate_all_angles_grain(cfg, angles, cloud):
    """Rotate the cloud by -theta around z per angle; simulate FMC + TFM."""
    from scipy.signal import hilbert  # noqa: F401 (indirectly via tfm_slice_y0)

    z_s, x_s, y_s, amp_s = cloud
    bscans = np.empty((len(angles), TFM_N_PIXELS, TFM_N_PIXELS),
                      dtype=np.complex64)
    t_total = time.time()
    for i, theta in enumerate(angles):
        ct, st = np.cos(theta), np.sin(theta)
        x_arr =  x_s * ct + y_s * st
        y_arr = -x_s * st + y_s * ct

        engine = FMCEngine3D(cfg)
        engine.set_born_scatterers(z_s, x_arr, y_arr, amp_s)
        t0 = time.time()
        result = engine.simulate(tx_chunk=1, verbose=False)
        result['fmc_data'] = bandpass_fmc(result['fmc_data'], cfg.dt)
        bscans[i] = tfm_slice_y0(result)
        print(f"  frame {i+1:>3}/{len(angles)}  "
              f"theta={np.degrees(theta):+6.1f} deg  "
              f"{time.time() - t0:5.1f}s")
    print(f"\nSim total: {time.time() - t_total:.1f}s")
    return bscans


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for N in N_SWEEP:
        assert 1 <= N <= N_MAX, f"N={N} out of range [1, {N_MAX}]"

    cfg = build_cfg()
    print(cfg.summary())
    print("\nScatterers (x, y, z) mm  r mm:")
    for sx, sy, sz, sr in SCATTERERS:
        print(f"  ({sx*1e3:+.1f}, {sy*1e3:+.1f}, {sz*1e3:.1f})  r={sr*1e3:.1f}")
    print(f"\nGrain size {GRAIN_SIZE_M*1e3:.1f} mm  "
          f"dZ = +/-{IMPEDANCE_VARIATION*100:.1f}%  "
          f"voxel {VOXEL_SIZE_M*1e3:.2f} mm  "
          f"threshold {BORN_THRESHOLD}")
    print(f"N_MAX={N_MAX}  sweep={N_SWEEP}")

    # Clean up stale per-N volumes so plot scripts only see the current sweep
    for f in OUTPUT_DIR.glob('volume_N*.npy'):
        f.unlink()

    bscans_path = OUTPUT_DIR / 'bscans_complex.npy'
    angles_path = OUTPUT_DIR / 'angles_rad.npy'
    cloud_path  = OUTPUT_DIR / 'grain_cloud.npz'

    if bscans_path.exists() and angles_path.exists():
        bscans_full = np.load(bscans_path)
        angles_full = np.load(angles_path)
        assert bscans_full.shape[0] == N_MAX and len(angles_full) == N_MAX, (
            "Cached B-scans don't match N_MAX; delete "
            f"{bscans_path.name}/{angles_path.name} to regenerate"
        )
        print(f"\nLoaded cached B-scans  shape={bscans_full.shape}")
    else:
        if cloud_path.exists():
            d = np.load(cloud_path)
            cloud = (d['z_s'], d['x_s'], d['y_s'], d['amp_s'])
            print(f"\nLoaded cached cloud  {len(cloud[0])} scatterers")
        else:
            cloud = build_grain_cloud()
            np.savez(cloud_path,
                     z_s=cloud[0], x_s=cloud[1], y_s=cloud[2], amp_s=cloud[3])

        angles_full = np.linspace(0.0, np.pi, N_MAX, endpoint=False)
        print(f"\nSimulating {N_MAX} angles...")
        bscans_full = simulate_all_angles_grain(cfg, angles_full, cloud)
        np.save(bscans_path, bscans_full)
        np.save(angles_path, angles_full)

    bscans_mag = np.abs(bscans_full).astype(np.float32)
    z_axis, xy_axis = axes_zxy()

    for N in N_SWEEP:
        idx = np.round(np.linspace(0, N_MAX, N, endpoint=False)).astype(int)
        idx = np.clip(idx, 0, N_MAX - 1)
        sub_bs  = bscans_mag[idx]
        sub_ang = angles_full[idx]
        print(f"\n--- N={N} (angle step {180.0/N:.2f} deg) ---")
        t0 = time.time()
        volume = reconstruct(sub_bs, sub_ang)
        print(f"  iradon: {time.time() - t0:.1f}s")
        np.save(OUTPUT_DIR / f'volume_N{N:03d}.npy', volume)

        metrics = measure(volume, z_axis, xy_axis)
        amp0 = metrics[0]['peak_amp']
        for j, m in enumerate(metrics):
            tx, ty, tz = m['truth']
            px, py, pz = m['peak_xyz']
            ratio = (m['peak_amp'] / amp0) if amp0 > 0 else float('nan')
            print(f"  s{j}  truth ({tx*1e3:+.1f},{ty*1e3:+.1f},{tz*1e3:.1f})  "
                  f"peak ({px*1e3:+.1f},{py*1e3:+.1f},{pz*1e3:.1f})  "
                  f"err={m['loc_err']*1e3:.2f} mm  "
                  f"FWHM(x,y,z)=({m['fwhm_x']*1e3:.2f},"
                  f"{m['fwhm_y']*1e3:.2f},{m['fwhm_z']*1e3:.2f}) mm  "
                  f"amp/s0={ratio:.3f}")

    print(f"\nDone. View with: python tests/plot_report_figures.py "
          f"{OUTPUT_DIR.name}")


if __name__ == '__main__':
    main()
