"""
Radon reconstruction validation — three scatterers, sweep over N_angles.

Pipeline:
  1. Simulate the rotational scan at N_MAX angles (once, cached to disk).
     Three spherical scatterers:
       s0 = (0, 0, 20) mm — on-axis, in-plane
       s1 = (5, 3, 20) mm — off-axis, in-plane
       s2 = (0, 8, 25) mm — far off-elevation (probes directivity)
  2. For N ∈ N_SWEEP (must divide N_MAX): stride-subsample the cached
     B-scan stack, run inverse Radon per z-plane (Classes.Reconstruct3D
     math: iradon, ramp, circle=True, axis flip).
  3. Per scatterer & per N, measure
       - localisation error (argmax in 3 mm ROI),
       - FWHM along each axis (linear half-max interpolation),
       - peak amplitude & ratio to s0.
  4. Write validation_metrics.csv + validation_summary.png.

Run from the SYNTHETIC DATA directory:
    python tests/test_radon_validation.py
"""

from __future__ import annotations

import csv
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
from skimage.transform import iradon

sys.path.insert(0, str(REPO / "build" / "CPP" / "TFM"))
import tfm_cpp


# ---- Array ----
N_ELEMENTS    = 128
ELEMENT_PITCH = 0.3e-3
FREQUENCY     = 10e6
BANDWIDTH     = 0.8
TIME_SAMPLES  = 2048

# ---- Specimen ----
THICKNESS = 50e-3
WIDTH     = 50e-3
DEPTH     = 50e-3

# ---- Scatterers: (x, y, z, r) in metres, world frame (pre-rotation) ----
SCATTERERS = [
    (0.0e-3, 0.0e-3, 20.0e-3, 0.8e-3),   # s0: on-axis
    (5.0e-3, 3.0e-3, 20.0e-3, 0.8e-3),   # s1: off-axis in-plane
    (0.0e-3, 8.0e-3, 25.0e-3, 0.8e-3),   # s2: far off-elevation
]
DEFECT_N_POINTS = 600

# ---- Angular sampling ----
# Sweep values are picked as ~uniform indices into the N_MAX cache; they do
# NOT need to divide N_MAX (angles end up approximately but not exactly
# uniform in [0, pi), which iradon handles fine via its `theta` argument).
N_MAX   = 240
N_SWEEP = [12, 18]                        # angle step 15 deg, 10 deg

# ---- TFM gate ----
TFM_N_PIXELS = 400
TFM_Z_START  = 5e-3
TFM_Z_END    = 45e-3

# ---- Metrics ----
ROI_MM = 3.0                             # ±ROI around truth for argmax

# ---- Output ----
OUTPUT_DIR = HERE / 'output' / 'radon_validation'


def build_cfg() -> SimulationConfig3D:
    """1D linear array along world x, y = 0."""
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
    f_c  = FREQUENCY / 1e6
    f_lo = max(f_c * (1 - BANDWIDTH / 2), 0.1)
    f_hi = f_c * (1 + BANDWIDTH / 2)
    out  = np.zeros_like(fmc)
    n_tx, n_rx, _ = fmc.shape
    for t in range(n_tx):
        for r in range(n_rx):
            out[t, r, :] = filter_signal(fmc[t, r, :], dt, f_lo, f_hi,
                                         filter_alpha=1.0, hanning_bool=False)
    return out


def tfm_slice_y0(result: dict) -> np.ndarray:
    """2D (z, x) TFM at y=0 in array frame, 3D element-to-pixel distances.
    Returns complex analytic B-scan of shape (n_z, n_x)."""
    fmc       = result['fmc_data']
    elem      = result['element_positions_xyz']   # (n_el, 3) cols (z, x, y)
    time_axis = result['time_axis']
    cfg       = result['config']
    c         = float(cfg.material.c_L)

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
    img2d = np.squeeze(img, axis=1)
    return hilbert(img2d, axis=0).astype(np.complex64)


def simulate_all_angles(cfg: SimulationConfig3D,
                        angles: np.ndarray) -> np.ndarray:
    """Rotate ALL scatterers by -θ around z (equivalent to +θ array rotation),
    simulate FMC, then TFM-slice at y=0. Returns (n_angles, n_z, n_x) cplx."""
    bscans = np.empty((len(angles), TFM_N_PIXELS, TFM_N_PIXELS),
                      dtype=np.complex64)
    t0 = time.time()
    for i, theta in enumerate(angles):
        ct, st = np.cos(theta), np.sin(theta)
        engine = FMCEngine3D(cfg)
        for sx, sy, sz, sr in SCATTERERS:
            x_arr = sx * ct + sy * st
            y_arr = -sx * st + sy * ct
            engine.add_defect(
                SphericalDefect(center_x=x_arr, center_y=y_arr,
                                center_z=sz, radius=sr),
                n_points=DEFECT_N_POINTS,
            )
        t1 = time.time()
        result = engine.simulate(tx_chunk=1, verbose=False)
        result['fmc_data'] = bandpass_fmc(result['fmc_data'], cfg.dt)
        bscans[i] = tfm_slice_y0(result)
        print(f"  frame {i+1:>3}/{len(angles)}  θ={np.degrees(theta):+6.1f}°  "
              f"{time.time() - t1:5.1f}s")
    print(f"\nSim total: {time.time() - t0:.1f}s")
    return bscans


def reconstruct(bscans_mag: np.ndarray,
                angles: np.ndarray) -> np.ndarray:
    """iradon per z-plane. bscans_mag: (n_angles, n_z, n_lat) float."""
    angles_deg = np.degrees(angles)
    _, n_z, n_lat = bscans_mag.shape
    volume = np.zeros((n_z, n_lat, n_lat), dtype=np.float32)
    for z in range(n_z):
        sino = bscans_mag[:, z, :].T                     # (n_lat, n_angles)
        recon = iradon(sino, theta=angles_deg, filter_name='ramp',
                       circle=True, output_size=n_lat)
        volume[z] = recon[::-1, :].astype(np.float32)
    return volume


def axes_zxy():
    z_axis  = np.linspace(TFM_Z_START, TFM_Z_END, TFM_N_PIXELS)
    half    = (N_ELEMENTS - 1) * ELEMENT_PITCH / 2
    xy_axis = np.linspace(-half, half, TFM_N_PIXELS)
    return z_axis, xy_axis


def _fwhm_along(line: np.ndarray, peak_idx: int,
                axis: np.ndarray) -> float:
    """Linear-interpolated full-width-half-max around peak_idx.
    Returns nan if the half-max crossing can't be bracketed."""
    peak = float(line[peak_idx])
    if peak <= 0:
        return float('nan')
    half = peak / 2.0
    n = len(line)

    il = peak_idx
    while il > 0 and line[il - 1] > half:
        il -= 1
    if il == 0 and line[0] > half:
        return float('nan')
    t = (half - line[il - 1]) / (line[il] - line[il - 1])
    xl = axis[il - 1] + t * (axis[il] - axis[il - 1])

    ir = peak_idx
    while ir < n - 1 and line[ir + 1] > half:
        ir += 1
    if ir == n - 1 and line[-1] > half:
        return float('nan')
    t = (half - line[ir]) / (line[ir + 1] - line[ir])
    xr = axis[ir] + t * (axis[ir + 1] - axis[ir])

    return float(xr - xl)


def measure(volume: np.ndarray,
            z_axis: np.ndarray, xy_axis: np.ndarray) -> list[dict]:
    dz  = z_axis[1]  - z_axis[0]
    dxy = xy_axis[1] - xy_axis[0]
    rz  = int(round(ROI_MM * 1e-3 / dz))
    rxy = int(round(ROI_MM * 1e-3 / dxy))

    out = []
    for sx, sy, sz, _ in SCATTERERS:
        iz0 = int(np.argmin(np.abs(z_axis  - sz)))
        iy0 = int(np.argmin(np.abs(xy_axis - sy)))
        ix0 = int(np.argmin(np.abs(xy_axis - sx)))
        zs = slice(max(iz0 - rz, 0),  min(iz0 + rz + 1, volume.shape[0]))
        ys = slice(max(iy0 - rxy, 0), min(iy0 + rxy + 1, volume.shape[1]))
        xs = slice(max(ix0 - rxy, 0), min(ix0 + rxy + 1, volume.shape[2]))
        roi = volume[zs, ys, xs]
        piz, piy, pix = np.unravel_index(int(np.argmax(roi)), roi.shape)
        iz = zs.start + piz
        iy = ys.start + piy
        ix = xs.start + pix
        peak = float(volume[iz, iy, ix])

        loc_err = float(np.sqrt(
            (z_axis[iz]  - sz) ** 2 +
            (xy_axis[iy] - sy) ** 2 +
            (xy_axis[ix] - sx) ** 2
        ))

        out.append({
            'truth': (sx, sy, sz),
            'peak_xyz': (float(xy_axis[ix]), float(xy_axis[iy]), float(z_axis[iz])),
            'peak_amp': peak,
            'loc_err':  loc_err,
            'fwhm_x':   _fwhm_along(volume[iz, iy, :], ix, xy_axis),
            'fwhm_y':   _fwhm_along(volume[iz, :, ix], iy, xy_axis),
            'fwhm_z':   _fwhm_along(volume[:, iy, ix], iz, z_axis),
        })
    return out


def write_csv(rows: list[dict], path: Path) -> None:
    keys = list(rows[0].keys())
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def plot_summary(rows: list[dict], path: Path) -> None:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    Ns = N_SWEEP
    for j, sc in enumerate(SCATTERERS):
        loc  = [r['loc_err_mm']      for r in rows if r['scatterer'] == j]
        fwhm = [np.sqrt(r['fwhm_x_mm']**2 + r['fwhm_y_mm']**2 + r['fwhm_z_mm']**2)
                for r in rows if r['scatterer'] == j]
        amp_db = [20.0 * np.log10(max(r['amp_ratio_to_s0'], 1e-6))
                  for r in rows if r['scatterer'] == j]
        lbl  = f"s{j} ({sc[0]*1e3:+.0f},{sc[1]*1e3:+.0f},{sc[2]*1e3:.0f}) mm"
        axes[0].plot(Ns, loc,    'o-', label=lbl)
        axes[1].plot(Ns, fwhm,   'o-', label=lbl)
        axes[2].plot(Ns, amp_db, 'o-', label=lbl)
    axes[0].set(xlabel='N scans  (angle step)', ylabel='localisation error (mm)',
                title='Localisation error')
    axes[1].set(xlabel='N scans  (angle step)', ylabel='FWHM combined (mm)',
                title='PSF width (sqrt(FWHM_x^2 + FWHM_y^2 + FWHM_z^2))')
    axes[2].set(xlabel='N scans  (angle step)', ylabel='peak / peak(s0)  (dB)',
                title='Amplitude ratio to s0')
    tick_labels = [f'{N}\n({180.0/N:.2f} deg)' for N in Ns]
    for a in axes:
        a.set_xticks(Ns)
        a.set_xticklabels(tick_labels)
        a.grid(True, alpha=0.3)
        a.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for N in N_SWEEP:
        assert 1 <= N <= N_MAX, f"N={N} out of range [1, {N_MAX}]"

    cfg = build_cfg()
    print(cfg.summary())
    print("\nScatterers (x, y, z) mm  r mm:")
    for sx, sy, sz, sr in SCATTERERS:
        print(f"  ({sx*1e3:+.1f}, {sy*1e3:+.1f}, {sz*1e3:.1f})  r={sr*1e3:.1f}")
    print(f"\nN_MAX={N_MAX}, sweep={N_SWEEP}")

    # ---- Simulate (cached) ----
    bscans_path = OUTPUT_DIR / 'bscans_complex.npy'
    angles_path = OUTPUT_DIR / 'angles_rad.npy'
    if bscans_path.exists() and angles_path.exists():
        bscans_full = np.load(bscans_path)
        angles_full = np.load(angles_path)
        assert bscans_full.shape[0] == N_MAX and len(angles_full) == N_MAX, (
            "Cached B-scans don't match N_MAX — delete "
            f"{bscans_path.name}/{angles_path.name} to regenerate"
        )
        print(f"\nLoaded cached B-scans  shape={bscans_full.shape}")
    else:
        angles_full = np.linspace(0.0, np.pi, N_MAX, endpoint=False)
        print(f"\nSimulating {N_MAX} angles…")
        bscans_full = simulate_all_angles(cfg, angles_full)
        np.save(bscans_path, bscans_full)
        np.save(angles_path, angles_full)

    bscans_mag = np.abs(bscans_full).astype(np.float32)
    z_axis, xy_axis = axes_zxy()

    # ---- Sweep N ----
    rows = []
    for N in N_SWEEP:
        idx      = np.round(np.linspace(0, N_MAX, N, endpoint=False)).astype(int)
        idx      = np.clip(idx, 0, N_MAX - 1)
        sub_bs   = bscans_mag[idx]
        sub_ang  = angles_full[idx]
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
            rows.append({
                'N':                N,
                'scatterer':        j,
                'truth_x_mm':       tx * 1e3,
                'truth_y_mm':       ty * 1e3,
                'truth_z_mm':       tz * 1e3,
                'peak_x_mm':        px * 1e3,
                'peak_y_mm':        py * 1e3,
                'peak_z_mm':        pz * 1e3,
                'loc_err_mm':       m['loc_err'] * 1e3,
                'fwhm_x_mm':        m['fwhm_x']  * 1e3,
                'fwhm_y_mm':        m['fwhm_y']  * 1e3,
                'fwhm_z_mm':        m['fwhm_z']  * 1e3,
                'peak_amp':         m['peak_amp'],
                'amp_ratio_to_s0':  ratio,
            })

    csv_path  = OUTPUT_DIR / 'validation_metrics.csv'
    plot_path = OUTPUT_DIR / 'validation_summary.png'
    write_csv(rows, csv_path)
    plot_summary(rows, plot_path)
    print(f"\nWrote {csv_path}")
    print(f"Wrote {plot_path}")


if __name__ == '__main__':
    main()
