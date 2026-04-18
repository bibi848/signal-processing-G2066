#!/usr/bin/env python3
"""
Open a previously-saved 3D TFM volume (and optional grain volume) in napari.

Reads the sidecar files written by run_engine_3d.py:
  fmc_3d_tfm.npz    — img_db + (x, y, z) axes
  fmc_3d_grain.npz  — impedance, voxel_size, origin_{z,y,x}  (optional)

Pair mode: if the argument is a directory containing volume_*.npz (as written
by generate_overlap_pairs.py), all TFM volumes are loaded together. When a
meta.json with `shift_m` is present, volumes tagged 'A' and 'B' are translated
by ∓shift/2 in x so the physical overlap is visible in the viewer.

Run:  python "SYNTHETIC DATA/view_engine_3d.py"  [path]
"""

import json
import sys
from pathlib import Path

import numpy as np
import napari

HERE = Path(__file__).resolve().parent

TFM_NPZ   = HERE / "fmc_3d_tfm.npz"
TFM_NPY   = HERE / "fmc_3d_tfm.npy"         # legacy format
GRAIN_PATH = HERE / "fmc_3d_grain.npz"
SCAT_PATH  = HERE / "fmc_3d_scatterers.npz"
PAIR_DIR: Path | None = None

if len(sys.argv) > 1:
    arg = Path(sys.argv[1])
    if arg.is_dir():
        PAIR_DIR = arg
        GRAIN_PATH = arg / "grain_volume.npz"
        SCAT_PATH  = arg / "fmc_3d_scatterers.npz"   # unlikely in pair dirs
    elif arg.suffix == ".npz":
        TFM_NPZ = arg
        GRAIN_PATH = arg.with_name(arg.stem.replace("_tfm", "_grain") + ".npz")
        SCAT_PATH  = arg.with_name(arg.stem.replace("_tfm", "_scatterers") + ".npz")
    else:
        TFM_NPY = arg
        GRAIN_PATH = arg.with_name(arg.stem.replace("_tfm", "_grain") + ".npz")
        SCAT_PATH  = arg.with_name(arg.stem.replace("_tfm", "_scatterers") + ".npz")


def _load_tfm(path: Path, meta: dict | None = None
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Load a TFM volume. .npz carries axes inline; .npy reconstructs them from meta."""
    if path.suffix == ".npy":
        if meta is None:
            raise SystemExit(f"{path.name} is .npy but no meta.json was loaded")
        img_db = np.load(path)
        n_z, n_y, n_x = img_db.shape
        ap_x = float(meta['aperture_x_m'])
        ap_y = float(meta.get('aperture_y_m', 1e-3))
        z_lo, z_hi = meta.get('tfm_z_range_m', [0.0, 35e-3])
        x_axis = np.linspace(-ap_x / 2, ap_x / 2, n_x)
        y_axis = np.linspace(-ap_y / 2, ap_y / 2, n_y)
        z_axis = np.linspace(float(z_lo), float(z_hi), n_z)
        return img_db, x_axis, y_axis, z_axis, 20.0
    tfm = np.load(path)
    return (tfm['img_db'], tfm['x'], tfm['y'], tfm['z'],
            float(tfm['db_range']) if 'db_range' in tfm.files else 20.0)


# Look up meta.json / shift_m first so .npy loaders can reconstruct axes.
pair_meta: dict | None = None
shift_m = 0.0
if PAIR_DIR is not None:
    meta_path = PAIR_DIR / "meta.json"
    if meta_path.exists():
        pair_meta = json.loads(meta_path.read_text())
        shift_m = float(pair_meta.get("shift_m", 0.0))
        print(f"Pair shift = {shift_m*1e3:.2f} mm")

# Collect (tag, path) pairs — directory-mode yields several; single-file mode one.
tfm_entries: list[tuple[str, Path]] = []
if PAIR_DIR is not None:
    for vp in sorted(list(PAIR_DIR.glob("volume_*.npy"))
                     + list(PAIR_DIR.glob("volume_*.npz"))):
        tfm_entries.append((vp.stem.split("_")[-1], vp))
    if not tfm_entries:
        raise SystemExit(f"No volume_*.npy or volume_*.npz found in {PAIR_DIR}")
elif TFM_NPZ.exists():
    tfm_entries.append(("", TFM_NPZ))

if tfm_entries:
    # Use the first volume to size napari (all pair volumes share scales/axes).
    img_db, x_axis, y_axis, z_axis, db_range = _load_tfm(tfm_entries[0][1], pair_meta)
elif TFM_NPY.exists():
    # Legacy: reconstruct axes from the run_engine_3d.py defaults (X/Y from
    # the CSV aperture, Z from 0 to Z_MAX_MM).
    print(f"Legacy .npy detected at {TFM_NPY}; reconstructing axes from "
          "run_engine_3d.py defaults. Re-run the script to save an .npz "
          "sidecar with embedded axes.")
    img_db = np.load(TFM_NPY)
    n_z, n_y, n_x = img_db.shape
    ARRAY_CSV = (HERE.parent / "DATA" / "2D Processed Data"
                 / "Cu Pure 7.5MHz Ex 15042026 Filtered"
                 / "11_filtered" / "array_geometry.csv")
    data = np.genfromtxt(ARRAY_CSV, delimiter=',', names=True)
    xc = data['el_xc'].astype(float)
    yc = data['el_yc'].astype(float)
    x_axis = np.linspace(xc.min(), xc.max(), n_x)
    y_axis = np.linspace(yc.min(), yc.max(), n_y)
    z_axis = np.linspace(0.0, 35e-3, n_z)
    db_range = 20.0
else:
    raise SystemExit(
        f"No TFM volume found. Looked for:\n  {TFM_NPZ}\n  {TFM_NPY}"
    )

dz = float(z_axis[1] - z_axis[0]) * 1e3
dy = float(y_axis[1] - y_axis[0]) * 1e3
dx = float(x_axis[1] - x_axis[0]) * 1e3
tz = float(z_axis[0]) * 1e3
ty = float(y_axis[0]) * 1e3
tx = float(x_axis[0]) * 1e3

viewer = napari.Viewer()

if GRAIN_PATH.exists():
    g = np.load(GRAIN_PATH)
    vs_mm = float(g['voxel_size']) * 1e3
    viewer.add_image(
        g['impedance'],
        name='Grain impedance',
        scale=(vs_mm, vs_mm, vs_mm),
        translate=(float(g['origin_z']) * 1e3,
                   float(g['origin_y']) * 1e3,
                   float(g['origin_x']) * 1e3),
        colormap='gray',
        rendering='attenuated_mip',
        opacity=0.5,
        visible=False,
    )
    print(f"Loaded grain volume {g['impedance'].shape} from {GRAIN_PATH}")
else:
    print(f"No grain sidecar at {GRAIN_PATH} — TFM only.")

if len(tfm_entries) > 1:
    # Pair mode: shift A by +shift/2 and B by -shift/2 in x so physical
    # overlap is visible (tag letter drives sign; unknown tags sit at 0).
    dx_for_tag = {'A': +shift_m / 2 * 1e3, 'B': -shift_m / 2 * 1e3}
    for tag, path in tfm_entries:
        img, xa, ya, za, dbr = _load_tfm(path, pair_meta)
        tdx = dx_for_tag.get(tag, 0.0)
        viewer.add_image(
            img,
            name=f'TFM (dB) — {tag}',
            scale=(dz, dy, dx),
            translate=(float(za[0]) * 1e3, float(ya[0]) * 1e3,
                       float(xa[0]) * 1e3 + tdx),
            contrast_limits=(-dbr, 0.0),
            colormap='inferno',
            rendering='mip',
            opacity=0.6,
            blending='additive',
        )
        print(f"Loaded TFM {tag}: {img.shape} from {path.name} "
              f"(x offset = {tdx:+.2f} mm)")
else:
    viewer.add_image(
        img_db,
        name='3D TFM (dB)',
        scale=(dz, dy, dx),
        translate=(tz, ty, tx),
        contrast_limits=(-db_range, 0.0),
        colormap='inferno',
        rendering='mip',
    )

if SCAT_PATH.exists():
    s = np.load(SCAT_PATH)
    coords = np.stack([s['z'], s['y'], s['x']], axis=1) * 1e3   # mm
    amps = np.abs(s['amp'].astype(np.float64))
    # Map |amp| → point size (0.05–0.8 mm) so strong scatterers read as larger.
    amp_max = amps.max() if amps.size else 1.0
    sizes = 0.05 + 0.75 * (amps / max(amp_max, 1e-30))
    viewer.add_points(
        coords,
        name=f"Born scatterers ({coords.shape[0]:,})",
        size=sizes,
        face_color='cyan',
        edge_color='transparent',
        opacity=0.6,
        visible=False,
    )
    total = int(s['total_count']) if 'total_count' in s.files else coords.shape[0]
    print(f"Loaded {coords.shape[0]:,}/{total:,} scatterers from {SCAT_PATH}")
else:
    print(f"No scatterer sidecar at {SCAT_PATH}")

viewer.dims.axis_labels = ('z (mm)', 'y (mm)', 'x (mm)')
napari.run()
