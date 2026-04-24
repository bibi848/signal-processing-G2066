"""
Compute 3D autocorrelations for every TFM volume in a folder (recursive).

For each volume matching VOLUME_GLOB under INPUT_DIR:
    1. Load the volume. If it's in dB, convert to linear amplitude.
    2. Subtract the mean (so we autocorrelate fluctuations, not DC).
    3. Compute the 3D autocorrelation via FFT (Wiener–Khinchin).
    4. Normalize so the zero-lag peak = 1.
    5. Crop to ±MAX_LAG_MM around zero lag (if set).
    6. Save as `autocorr_<name>.npy` under OUTPUT_DIR, mirroring the input tree.

Voxel spacing is resolved in this order:
    1. sibling `meta.json` (synthetic engine output), or
    2. sibling `Params.txt` / `Parameters.txt` (experimental folders), or
    3. VOXEL_SIZE_MM_FALLBACK.
"""

import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
HERE_REPO = HERE.parent


# =============================================================================
# USER SETTINGS
# =============================================================================

# --- Synthetic overlap sweep with absolute-depth crop (current default) ---
INPUT_DIR    = HERE / 'output' / 'engine_3d_overlap_sweep'
OUTPUT_DIR   = HERE / 'output' / 'autocorrelations_synthetic_15_35mm'
VOLUME_GLOB  = 'volume_*.npy'
PATH_FILTER  = None

# --- Experimental copper data (no crop, comment out above, uncomment below) ---
# INPUT_DIR    = HERE_REPO / 'DATA' / '2D TFM Data'
# OUTPUT_DIR   = HERE / 'output' / 'autocorrelations_Cu_experimental'
# VOLUME_GLOB  = '*_3D_TFM.npy'
# PATH_FILTER  = 'Cu'

DATA_IS_DB          = True       # True → convert 10^(v/20) before autocorrelating
SUBTRACT_MEAN       = True
NORMALIZE_ZERO_LAG  = True       # peak at zero lag becomes 1

# Absolute-depth Z crop. Keeps voxels whose physical depth lies within
# [Z_DEPTH_KEEP_MIN_MM, Z_DEPTH_KEEP_MAX_MM]. Requires sibling meta.json to
# resolve the volume's absolute z-origin — if absent the crop is skipped
# (the volume is kept as-is), so experimental data pointed at this script
# is left untouched regardless of these settings.
APPLY_DEPTH_CROP      = True
Z_DEPTH_KEEP_MIN_MM   = 15.0
Z_DEPTH_KEEP_MAX_MM   = 35.0       # 20 mm Z span to match experimental volumes
MIN_Z_MM_AFTER_CROP   = 2.0

# Crop radius around zero lag. None → keep full autocorrelation (large!).
# For a 400×200×200 volume the full ACF is 799×399×399 ≈ 1 GB.
MAX_LAG_MM = 5.0                 # e.g. 5 mm window → (lags within ±5 mm)

VOXEL_SIZE_MM_FALLBACK = (0.04, 0.04, 0.039)   # (dz, dy, dx) if no meta/Params

OVERWRITE = False                # skip volumes whose output already exists


# =============================================================================
# Core
# =============================================================================

def autocorr_3d_fft(vol: np.ndarray) -> np.ndarray:
    """3D linear (non-circular) autocorrelation via zero-padded FFT.

    Returns an array of shape (2*Nz-1, 2*Ny-1, 2*Nx-1) with zero lag at the
    centre of each axis.
    """
    Nz, Ny, Nx = vol.shape
    pad_shape = (2 * Nz - 1, 2 * Ny - 1, 2 * Nx - 1)
    # rfftn wants real input — use fftn on complex for symmetry w/ ifftshift
    F = np.fft.rfftn(vol, s=pad_shape)
    acf = np.fft.irfftn(F * np.conj(F), s=pad_shape)
    # Shift so zero lag is at the centre
    return np.fft.fftshift(acf)


def crop_to_lag(acf: np.ndarray, voxel_size_m: tuple[float, float, float],
                max_lag_m: float) -> np.ndarray:
    """Keep only lags within ±max_lag_m along every axis."""
    cz, cy, cx = [s // 2 for s in acf.shape]
    dz, dy, dx = voxel_size_m
    rz = int(round(max_lag_m / dz))
    ry = int(round(max_lag_m / dy))
    rx = int(round(max_lag_m / dx))
    rz = min(rz, cz); ry = min(ry, cy); rx = min(rx, cx)
    return acf[cz - rz:cz + rz + 1,
               cy - ry:cy + ry + 1,
               cx - rx:cx + rx + 1]


def voxel_size_from_meta(volume_path: Path) -> tuple[float, float, float] | None:
    """Read voxel spacing (dz, dy, dx) in metres from sibling meta.json (synthetic)."""
    meta_path = volume_path.parent / 'meta.json'
    if not meta_path.exists():
        return None
    try:
        with open(meta_path) as f:
            meta = json.load(f)
    except Exception:
        return None

    tfm_px  = meta.get('tfm_pixels')                 # [Nz, Ny, Nx]
    half    = meta.get('recon_half_extent_m')
    z_range = meta.get('tfm_z_range_m')
    if not (tfm_px and half and z_range):
        return None
    Nz, Ny, Nx = tfm_px
    dz = (z_range[1] - z_range[0]) / Nz
    dy = (2 * half) / Ny
    dx = (2 * half) / Nx
    return (dz, dy, dx)


_PARAMS_PATTERNS = {
    'x': re.compile(r'X-dir pixel size:\s*([-\d.eE+]+)\s*mm'),
    'y': re.compile(r'Y-dir pixel size:\s*([-\d.eE+]+)\s*mm'),
    'z': re.compile(r'Z-dir pixel size:\s*([-\d.eE+]+)\s*mm'),
    'lateral': re.compile(r'Lateral pixel size:\s*([-\d.eE+]+)\s*mm'),
    'depth':   re.compile(r'Depth pixel size:\s*([-\d.eE+]+)\s*mm'),
}


def voxel_size_from_params(volume_path: Path) -> tuple[float, float, float] | None:
    """Read voxel spacing (dz, dy, dx) in metres from sibling Params.txt.

    Supports both 3D ("X/Y/Z-dir pixel size: 0.04 mm") and 2D-legacy
    ("Lateral / Depth pixel size") formats.
    """
    for name in ('Params.txt', 'Parameters.txt'):
        p = volume_path.parent / name
        if p.exists():
            text = p.read_text(errors='ignore')
            break
    else:
        return None

    found = {k: float(m.group(1)) for k, pat in _PARAMS_PATTERNS.items()
             if (m := pat.search(text))}
    if 'x' in found and 'y' in found and 'z' in found:
        return (found['z'] * 1e-3, found['y'] * 1e-3, found['x'] * 1e-3)
    if 'lateral' in found and 'depth' in found:
        return (found['depth'] * 1e-3, found['lateral'] * 1e-3, found['lateral'] * 1e-3)
    return None


def resolve_voxel_size(volume_path: Path) -> tuple[float, float, float]:
    return (voxel_size_from_meta(volume_path)
            or voxel_size_from_params(volume_path)
            or tuple(v * 1e-3 for v in VOXEL_SIZE_MM_FALLBACK))


def depth_crop_indices(volume_path: Path, Nz: int) -> tuple[int, int] | None:
    """Return (z_start, z_stop) voxel indices keeping physical depth in
    [Z_DEPTH_KEEP_MIN_MM, Z_DEPTH_KEEP_MAX_MM], using sibling meta.json.

    Returns None when meta.json is absent (the crop is then skipped and the
    volume is used in full — this is the experimental-data path).
    """
    meta_path = volume_path.parent / 'meta.json'
    if not meta_path.exists():
        return None
    try:
        with open(meta_path) as f:
            meta = json.load(f)
    except Exception:
        return None
    z_range = meta.get('tfm_z_range_m')
    if not z_range:
        return None
    z0_mm, z1_mm = z_range[0] * 1e3, z_range[1] * 1e3
    dz_mm = (z1_mm - z0_mm) / Nz
    z_start = int(round((Z_DEPTH_KEEP_MIN_MM - z0_mm) / dz_mm))
    z_stop  = int(round((Z_DEPTH_KEEP_MAX_MM - z0_mm) / dz_mm))
    z_start = max(0, z_start)
    z_stop  = min(Nz, z_stop)
    return (z_start, z_stop)


def process_one(volume_path: Path, out_path: Path) -> dict | None:
    volume_path = volume_path.resolve()
    vol = np.load(volume_path)

    voxel_size = resolve_voxel_size(volume_path)
    dz = voxel_size[0]

    # Depth-band crop (synthetic only — skipped for experimental without meta.json).
    Nz_full = vol.shape[0]
    crop_info = {'cropped': False, 'z_start_idx': 0, 'z_stop_idx': Nz_full}
    if APPLY_DEPTH_CROP:
        idx = depth_crop_indices(volume_path, Nz_full)
        if idx is not None:
            z_start, z_stop = idx
            kept_mm = (z_stop - z_start) * dz * 1e3
            if z_stop - z_start < 1 or kept_mm < MIN_Z_MM_AFTER_CROP:
                print(f"       SKIP — depth crop "
                      f"[{Z_DEPTH_KEEP_MIN_MM},{Z_DEPTH_KEEP_MAX_MM}] mm "
                      f"leaves only {kept_mm:.1f} mm (< {MIN_Z_MM_AFTER_CROP} mm)")
                return None
            vol = vol[z_start:z_stop, :, :]
            crop_info = {'cropped': True, 'z_start_idx': z_start, 'z_stop_idx': z_stop}
    kept_mm = vol.shape[0] * dz * 1e3

    if DATA_IS_DB:
        vol = np.power(10.0, vol / 20.0)
    vol = vol.astype(np.float64, copy=False)

    if SUBTRACT_MEAN:
        vol = vol - vol.mean()

    acf = autocorr_3d_fft(vol)

    if NORMALIZE_ZERO_LAG:
        peak = acf[acf.shape[0] // 2, acf.shape[1] // 2, acf.shape[2] // 2]
        if peak > 1e-30:
            acf = acf / peak

    if MAX_LAG_MM is not None:
        acf = crop_to_lag(acf, voxel_size, MAX_LAG_MM * 1e-3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, acf.astype(np.float32))

    try:
        rel_source = str(volume_path.relative_to(INPUT_DIR.resolve()))
    except ValueError:
        rel_source = str(volume_path)
    meta = {
        'source_volume': rel_source,
        'acf_shape': list(acf.shape),
        'voxel_size_m': list(voxel_size),
        'max_lag_mm': MAX_LAG_MM,
        'data_was_db': DATA_IS_DB,
        'mean_subtracted': SUBTRACT_MEAN,
        'normalized_zero_lag': NORMALIZE_ZERO_LAG,
        'apply_depth_crop': APPLY_DEPTH_CROP,
        'z_depth_keep_mm': [Z_DEPTH_KEEP_MIN_MM, Z_DEPTH_KEEP_MAX_MM],
        'depth_cropped': crop_info['cropped'],
        'z_start_idx': int(crop_info['z_start_idx']),
        'z_stop_idx':  int(crop_info['z_stop_idx']),
        'z_kept_mm':   float(kept_mm),
        'nz_full':     int(Nz_full),
        'nz_kept':     int(vol.shape[0]),
    }
    with open(out_path.with_suffix('.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    return meta


def main() -> None:
    if not INPUT_DIR.exists():
        print(f"INPUT_DIR does not exist: {INPUT_DIR}", file=sys.stderr)
        sys.exit(1)

    volumes = sorted(INPUT_DIR.rglob(VOLUME_GLOB))
    if PATH_FILTER:
        volumes = [p for p in volumes if PATH_FILTER in str(p.relative_to(INPUT_DIR))]
    if not volumes:
        msg = f"No files matching {VOLUME_GLOB!r} under {INPUT_DIR}"
        if PATH_FILTER:
            msg += f" (with PATH_FILTER={PATH_FILTER!r})"
        print(msg, file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(volumes)} volume(s) under {INPUT_DIR}"
          + (f" matching {PATH_FILTER!r}" if PATH_FILTER else ""))
    print(f"Writing autocorrelations to {OUTPUT_DIR}")
    if MAX_LAG_MM is not None:
        print(f"Cropping to ±{MAX_LAG_MM} mm around zero lag")

    for i, vpath in enumerate(volumes, 1):
        rel = vpath.relative_to(INPUT_DIR)
        out_name = f"autocorr_{vpath.stem.removeprefix('volume_')}.npy"
        out_path = OUTPUT_DIR / rel.parent / out_name

        if out_path.exists() and not OVERWRITE:
            print(f"  [{i}/{len(volumes)}] skip (exists) {rel}")
            continue

        print(f"  [{i}/{len(volumes)}] {rel}")
        meta = process_one(vpath, out_path)
        if meta is None:
            continue
        print(f"       → {out_path.relative_to(OUTPUT_DIR)}  "
              f"shape={meta['acf_shape']}  (kept {meta['z_kept_mm']:.1f} mm Z)")

    print("\nDone.")


if __name__ == '__main__':
    main()
