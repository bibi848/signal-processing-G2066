#!/usr/bin/env python3
"""
2D matrix-array entry point for the 3D Born FMC engine.

Loads the real 2D probe geometry from DATA/2D Processed Data/..., runs a
single full-aperture 3D FMC acquisition, and reconstructs a volumetric
(z, y, x) TFM using the project's C++ kernel.

Edit the USER SETTINGS block below, then
    python "SYNTHETIC DATA/run_engine_2d.py"
"""

import sys
import time
from pathlib import Path

import numpy as np
from scipy.signal import hilbert

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from engine_3d import (
    SimulationConfig3D, ArrayConfig3D, SpecimenConfig3D, AcquisitionConfig3D,
    SphericalDefect, FMCEngine3D, ALUMINUM,
    generate_grain_structure, embed_geometric_defects_3d,
    extract_born_scatterers_3d,
)

# =============================================================================
# USER SETTINGS — edit these
# =============================================================================

# 2D probe geometry CSV (element centres in metres)
ARRAY_CSV = (REPO / "DATA" / "2D Processed Data"
             / "Al Pure 15MHz 12022026"
             / "Al_70_1_1" / "array_geometry.csv")

# Acquisition
FREQUENCY_MHZ = 15.0
BANDWIDTH     = 0.8
TIME_SAMPLES  = 2048
TX_CHUNK      = 1

# Specimen (m)
SPECIMEN_THICKNESS = 50e-3
SPECIMEN_WIDTH     = 50e-3
SPECIMEN_DEPTH     = 50e-3

# Single spherical defect (set RADIUS_MM = 0 to disable)
DEFECT_CENTER_Z_MM = 20.0
DEFECT_CENTER_X_MM = 0.0
DEFECT_CENTER_Y_MM = 0.0
DEFECT_RADIUS_MM   = 0.0
DEFECT_N_POINTS    = 1200

# Grain background (Voronoi voxel volume)
USE_GRAIN       = True
GRAIN_SIZE_MM   = 1.0
GRAIN_VOXEL_MM  = 0.2

# TFM — full 3D volume
RUN_TFM    = True
X_PIXELS   = 200
Y_PIXELS   = 200
Z_PIXELS   = 400
X_MIN_MM   = None             # None = array aperture min
X_MAX_MM   = None             # None = array aperture max
Y_MIN_MM   = None             # None = array aperture min
Y_MAX_MM   = None             # None = array aperture max
Z_MIN_MM   = None             # None = 0
Z_MAX_MM   = 35               # None = specimen thickness
TFM_DB_RANGE = 20.0

# Output / display
OUT_NAME        = "fmc_2d.npy"
SHOW_ARRAY_PLOT = True       # scatter of element centres before simulating
SHOW_NAPARI     = True

# =============================================================================
# Helpers
# =============================================================================


def load_array_geometry_csv(path: Path) -> tuple[np.ndarray, float, float]:
    data = np.genfromtxt(path, delimiter=',', names=True)
    xc = data['el_xc'].astype(float)
    yc = data['el_yc'].astype(float)
    width_x = float(np.median(np.abs(data['el_x2'] - data['el_x1'])))
    width_y = float(np.median(np.abs(data['el_y2'] - data['el_y1'])))
    return np.stack([xc, yc], axis=1), width_x, width_y


def _infer_pitch(coords: np.ndarray) -> float:
    uniq = np.unique(np.round(coords, 9))
    diffs = np.diff(uniq)
    diffs = diffs[diffs > 1e-9]
    return float(diffs.min()) if diffs.size else 0.0


def build_config() -> SimulationConfig3D:
    positions, w_x, w_y = load_array_geometry_csv(ARRAY_CSV)
    if np.allclose(positions[:, 1], 0.0, atol=1e-6):
        raise ValueError(
            f"{ARRAY_CSV.name} looks like a 1D probe (all y=0). "
            f"Use run_engine_1d.py instead."
        )
    return SimulationConfig3D(
        material=ALUMINUM,
        array=ArrayConfig3D(
            custom_positions=positions,
            pitch_x=_infer_pitch(positions[:, 0]),
            pitch_y=_infer_pitch(positions[:, 1]),
            element_width_x=w_x,
            element_width_y=w_y,
            frequency=FREQUENCY_MHZ * 1e6,
            bandwidth=BANDWIDTH,
        ),
        specimen=SpecimenConfig3D(
            thickness=SPECIMEN_THICKNESS,
            width=SPECIMEN_WIDTH,
            depth=SPECIMEN_DEPTH,
        ),
        acquisition=AcquisitionConfig3D(time_samples=TIME_SAMPLES),
    )


def _load_tfm_cpp():
    build_dir = str(REPO / "build" / "CPP" / "TFM")
    if build_dir not in sys.path:
        sys.path.insert(0, build_dir)
    import tfm_cpp
    return tfm_cpp


def reconstruct_volume(result: dict, n_x: int, n_y: int, n_z: int,
                       x_range: tuple, y_range: tuple, z_range: tuple,
                       db_range: float):
    """Volumetric TFM via the project's C++ kernel. Returns (nz, ny, nx) dB."""
    tfm_cpp = _load_tfm_cpp()

    fmc = result['fmc_data']
    elem_xyz = result['element_positions_xyz']
    time_axis = result['time_axis']
    cfg: SimulationConfig3D = result['config']
    c = float(cfg.material.c_L)

    n_el, _, n_t = fmc.shape
    n_fmc = n_el * n_el

    tx_grid, rx_grid = np.meshgrid(
        np.arange(n_el, dtype=np.int32),
        np.arange(n_el, dtype=np.int32),
        indexing='ij',
    )
    tx0 = tx_grid.ravel()
    rx0 = rx_grid.ravel()
    time_data = fmc.reshape(n_fmc, n_t).astype(np.float64, copy=False)

    zc = elem_xyz[:, 0].astype(np.float64, copy=False)
    xc = elem_xyz[:, 1].astype(np.float64, copy=False)
    yc = elem_xyz[:, 2].astype(np.float64, copy=False)

    x_img = np.linspace(x_range[0], x_range[1], n_x)
    y_img = np.linspace(y_range[0], y_range[1], n_y)
    z_img = np.linspace(z_range[0], z_range[1], n_z)

    Z, Y, X = np.meshgrid(z_img, y_img, x_img, indexing='ij')

    dx_mm = (x_img[-1] - x_img[0]) * 1e3 / max(n_x - 1, 1)
    dy_mm = (y_img[-1] - y_img[0]) * 1e3 / max(n_y - 1, 1)
    dz_mm = (z_img[-1] - z_img[0]) * 1e3 / max(n_z - 1, 1)
    print(f"  TFM grid: {n_z} (z) × {n_y} (y) × {n_x} (x) "
          f"= {Z.size:,} voxels")
    print(f"  Pixel size: dx={dx_mm:.3f} mm, dy={dy_mm:.3f} mm, "
          f"dz={dz_mm:.3f} mm")
    print(f"  FMC pairs: {n_fmc:,}, samples: {n_t}")
    print("  Running tfm_cpp.tfm2D …")

    t0 = time.time()
    img = tfm_cpp.tfm2D(
        time_data, time_axis.astype(np.float64),
        tx0, rx0, xc, yc, zc, X, Y, Z, c,
    )
    print(f"  TFM done in {time.time() - t0:.2f}s, shape {img.shape}")

    env = np.abs(hilbert(img, axis=0))
    env_max = float(env.max())
    img_db = 20.0 * np.log10(env / max(env_max, 1e-30) + 1e-10)
    img_db = np.clip(img_db, -db_range, 0.0)

    return img_db, {'x': x_img, 'y': y_img, 'z': z_img}


def plot_array_geometry(cfg: SimulationConfig3D) -> None:
    """Scatter element centres and draw their active footprints."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    elem = cfg.array.element_positions()           # cols (z, x, y)
    x_mm = elem[:, 1] * 1e3
    y_mm = elem[:, 2] * 1e3
    w_x_mm = cfg.array.element_width_x * 1e3
    w_y_mm = cfg.array.element_width_y * 1e3

    fig, ax = plt.subplots(figsize=(6, 6))
    for xc, yc in zip(x_mm, y_mm):
        ax.add_patch(Rectangle(
            (xc - w_x_mm / 2, yc - w_y_mm / 2), w_x_mm, w_y_mm,
            edgecolor='C0', facecolor='C0', alpha=0.25, linewidth=0.6,
        ))
    ax.scatter(x_mm, y_mm, s=8, c='C0', zorder=3)
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_title(f"2D array — {elem.shape[0]} elements, "
                 f"footprint {w_x_mm:.3f} × {w_y_mm:.3f} mm")
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()


def show_napari(img_db: np.ndarray, axes: dict, db_range: float):
    import napari

    dz = float(axes['z'][1] - axes['z'][0]) * 1e3
    dy = float(axes['y'][1] - axes['y'][0]) * 1e3
    dx = float(axes['x'][1] - axes['x'][0]) * 1e3

    viewer = napari.Viewer()
    viewer.add_image(
        img_db,
        name='2D-array TFM (dB)',
        scale=(dz, dy, dx),
        contrast_limits=(-db_range, 0.0),
        colormap='inferno',
        rendering='mip',
    )
    viewer.dims.axis_labels = ('z (mm)', 'y (mm)', 'x (mm)')
    napari.run()


# =============================================================================
# Main
# =============================================================================


cfg = build_config()
print(cfg.summary())

if SHOW_ARRAY_PLOT:
    plot_array_geometry(cfg)

engine = FMCEngine3D(cfg)

defect = None
if DEFECT_RADIUS_MM > 0:
    defect = SphericalDefect(
        center_z=DEFECT_CENTER_Z_MM * 1e-3,
        center_x=DEFECT_CENTER_X_MM * 1e-3,
        center_y=DEFECT_CENTER_Y_MM * 1e-3,
        radius=DEFECT_RADIUS_MM * 1e-3,
    )
    engine.add_defect(defect, n_points=DEFECT_N_POINTS)

if USE_GRAIN:
    assert cfg.material is not None
    vol = generate_grain_structure(
        thickness=cfg.specimen.thickness,
        width=cfg.specimen.width,
        depth=cfg.specimen.depth,
        background_material=cfg.material,
        mean_grain_size_m=GRAIN_SIZE_MM * 1e-3,
        voxel_size_m=GRAIN_VOXEL_MM * 1e-3,
    )
    if defect is not None:
        vol = embed_geometric_defects_3d(vol, [defect])
    z_s, x_s, y_s, amp_s, c_s = extract_born_scatterers_3d(
        vol, background_Z=cfg.material.Z_L, threshold=0.02,
    )
    engine.set_born_scatterers(z_s, x_s, y_s, amp_s, c_s=c_s)

result = engine.simulate(tx_chunk=TX_CHUNK)

out_path = HERE / OUT_NAME
np.save(out_path, result['fmc_data'])
print(f"Saved FMC ({result['fmc_data'].shape}) to {out_path}")

if RUN_TFM:
    r = cfg.reconstruction
    elem = cfg.array.element_positions()
    ap_x_min, ap_x_max = float(elem[:, 1].min()), float(elem[:, 1].max())
    ap_y_min, ap_y_max = float(elem[:, 2].min()), float(elem[:, 2].max())
    x_min = ap_x_min if X_MIN_MM is None else X_MIN_MM * 1e-3
    x_max = ap_x_max if X_MAX_MM is None else X_MAX_MM * 1e-3
    y_min = ap_y_min if Y_MIN_MM is None else Y_MIN_MM * 1e-3
    y_max = ap_y_max if Y_MAX_MM is None else Y_MAX_MM * 1e-3
    z_min = r.z_start if Z_MIN_MM is None else Z_MIN_MM * 1e-3
    z_max = r.z_end   if Z_MAX_MM is None else Z_MAX_MM * 1e-3

    img_db, axes = reconstruct_volume(
        result,
        n_x=X_PIXELS, n_y=Y_PIXELS, n_z=Z_PIXELS,
        x_range=(x_min, x_max),
        y_range=(y_min, y_max),
        z_range=(z_min, z_max),
        db_range=TFM_DB_RANGE,
    )
    vol_path = out_path.with_name(out_path.stem + "_tfm.npy")
    np.save(vol_path, img_db)
    print(f"Saved TFM volume ({img_db.shape}) to {vol_path}")

    if SHOW_NAPARI:
        show_napari(img_db, axes, TFM_DB_RANGE)
