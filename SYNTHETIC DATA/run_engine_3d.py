#!/usr/bin/env python3
"""
Born-only 3D NDT synthetic data engine — driver.

Dimensional lift of run_engine.py: same module layout, same helpers, same
overall flow. Differences are confined to the genuinely 3D physics:
    - 2D matrix phased array (n_x × n_y) instead of 1D linear
    - 1/r geometric spreading per leg
    - Rectangular-element 2D-sinc directivity
    - Full 3D voxel-gradient Born scatterer extraction
    - 3D TFM volume reconstruction via tfm_cpp.tfm2D
    - napari for 3D volume visualization

Output: 3D FMC data → noise → bandpass filter → 3D TFM volume.
"""

import sys
import os
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from scipy.signal import hilbert

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from engine_3d import (
    SimulationConfig3D, ArrayConfig3D, SpecimenConfig3D, AcquisitionConfig3D,
    SphericalDefect, CylindricalDefect, PlanarCrack3D,
    FMCEngine3D, ALUMINUM, COPPER, STEEL_MILD,
    extract_born_scatterers_3d,
)
from engine.microstructure import generate_grain_structure, embed_geometric_defects

# External Functions (OD)
sys.path.insert(0, str(REPO))
from Classes.Filter import filter_signal
import platform

# TFM Hardware Parameters
program_language = 'cpp'        # cpp or python or gpu
img_output       = 'envelope'   # envelope or db (3D path uses these only)

if program_language == 'cpp':
    if platform.system() == 'Windows':
        build_dir = str(REPO / "build" / "CPP" / "TFM" / "Debug")
    else:
        build_dir = str(REPO / "build" / "CPP" / "TFM")
    sys.path.insert(0, build_dir)
    import tfm_cpp
    print('CPP Available')

elif program_language == 'gpu':
    build_dir = str(REPO / "build" / "CPP" / "TFM_GPU")
    sys.path.insert(0, build_dir)
    import tfm_gpu
    print('GPU Available')


# =============================================================================
# USER SETTINGS
# =============================================================================

# Array
ARRAY_MODE         = "csv"   # "matrix" or "csv"
N_ELEMENTS_X       = 16
N_ELEMENTS_Y       = 16
PITCH_X_MM         = 0.6
PITCH_Y_MM         = 0.6
ELEMENT_WIDTH_X_MM = 0.54
ELEMENT_WIDTH_Y_MM = 0.54
ARRAY_CSV          = (REPO / "DATA" / "2D Processed Data"
                      / "Cu Pure 7.5MHz Ex 15042026 Filtered"
                      / "11_filtered" / "array_geometry.csv")

# Acquisition
FREQUENCY_MHZ      = 7.5
BANDWIDTH          = 0.8
TIME_SAMPLES       = 1024
SAMPLING_MHZ       = 50       # None → 4 × frequency
TX_CHUNK           = 1          # tx elements per inner loop (memory ↔ speed)

# Specimen + material
MATERIAL           = COPPER
SPECIMEN_THICKNESS_MM = 40.0
SPECIMEN_WIDTH_MM     = 20.0
SPECIMEN_DEPTH_MM     = 20.0

# Defects (3D). Empty list disables defects.
defects_3d = [
    #SphericalDefect(center_z=20e-3, center_x=0.0, center_y=0.0, radius=1e-3),
]
DEFECT_N_POINTS    = 1200

# Grain microstructure (Voronoi voxel volume)
USE_GRAIN          = True
GRAIN_SIZE_MM      = 0.5
GRAIN_VOXEL_MM     = 0.5
GRAIN_IMP_VAR      = 0.025
BORN_THRESHOLD     = 0.005

# TFM
RUN_TFM            = True
X_PIXELS           = 200
Y_PIXELS           = 200
Z_PIXELS           = 400
Z_MIN_MM           = 0.0        # crop near-surface coupling/front-wall band
Z_MAX_MM           = 35.0       # crop deep back-wall echo (None = thickness)
TFM_DB_RANGE       = 40.0

# Scan plan — list of rotation angles (degrees) about the z axis.
# A single [0.0] is a static frame; multiple values run a rotational scan.
SCAN_ANGLES_DEG    = [0.0]

# Output / display
OUTPUT_DIR         = HERE / 'output' / 'engine_3d'
OUT_NAME           = "fmc_3d.npy"
SHOW_NAPARI        = True


# =============================================================================
# Helpers
# =============================================================================


def add_noise(fmc_data: np.ndarray, snr_db: float = 35.0,
              grain_noise_level: float = 0.03) -> np.ndarray:
    """Add Gaussian electronic noise + band-limited grain noise."""
    signal_power = np.mean(fmc_data ** 2)
    if signal_power < 1e-30:
        return fmc_data

    noise_std = np.sqrt(signal_power / 10 ** (snr_db / 10))
    noisy = fmc_data + np.random.normal(0, noise_std, fmc_data.shape).astype(np.float32)

    grain = np.random.normal(0, grain_noise_level * noise_std, fmc_data.shape)
    from scipy.ndimage import uniform_filter1d
    grain = uniform_filter1d(grain, size=5, axis=2).astype(np.float32)
    noisy += grain
    return noisy


def apply_bandpass_filter(fmc_data: np.ndarray, dt: float, frequency: float,
                           bandwidth_fraction: float = 0.6,
                           filter_alpha: float = 1.0,
                           hanning_bool: bool = False) -> np.ndarray:
    """Apply bandpass filter to every A-scan in the FMC cube."""
    f_center = frequency / 1e6
    f_start = max(f_center * (1 - bandwidth_fraction / 2), 0.1)
    f_end = f_center * (1 + bandwidth_fraction / 2)

    num_tx, num_rx, _ = fmc_data.shape
    filtered = np.zeros_like(fmc_data)
    for tx in range(num_tx):
        for rx in range(num_rx):
            filtered[tx, rx, :] = filter_signal(
                fmc_data[tx, rx, :], dt, f_start, f_end,
                filter_alpha=filter_alpha, hanning_bool=hanning_bool,
            )
    return filtered


def reconstruct_tfm_3d(fmc_data: np.ndarray, time_axis: np.ndarray,
                        elem_xyz: np.ndarray, c: float,
                        x_range: Tuple[float, float],
                        y_range: Tuple[float, float],
                        z_range: Tuple[float, float],
                        n_x: int, n_y: int, n_z: int,
                        db_range: float = 40.0) -> Tuple[np.ndarray, dict]:
    """
    3D TFM reconstruction via the project's OpenMP C++ kernel.

    Args:
        fmc_data:  (n_el, n_el, n_t) float32 FMC cube
        time_axis: (n_t,) sampling instants in seconds
        elem_xyz:  (n_el, 3) element positions, columns (z, x, y)
        c:         Wave speed (m/s)
        x_range, y_range, z_range: image bounds in metres
        n_x, n_y, n_z: pixel counts per axis
        db_range:  dB clip range for envelope output

    Returns:
        (img_db, axes) — (n_z, n_y, n_x) dB volume and 1D axis dict.
    """
    n_el, _, n_t = fmc_data.shape
    n_fmc = n_el * n_el

    tx_grid, rx_grid = np.meshgrid(
        np.arange(n_el, dtype=np.int32),
        np.arange(n_el, dtype=np.int32),
        indexing='ij',
    )
    tx0 = tx_grid.ravel()
    rx0 = rx_grid.ravel()
    time_data = fmc_data.reshape(n_fmc, n_t).astype(np.float64, copy=False)

    zc = elem_xyz[:, 0].astype(np.float64, copy=False)
    xc = elem_xyz[:, 1].astype(np.float64, copy=False)
    yc = elem_xyz[:, 2].astype(np.float64, copy=False)

    x_img = np.linspace(x_range[0], x_range[1], n_x)
    y_img = np.linspace(y_range[0], y_range[1], n_y)
    z_img = np.linspace(z_range[0], z_range[1], n_z)
    Z, Y, X = np.meshgrid(z_img, y_img, x_img, indexing='ij')

    print(f"  TFM grid: {n_z} (z) × {n_y} (y) × {n_x} (x) "
          f"= {Z.size:,} voxels")
    print(f"  FMC pairs: {n_fmc:,}, samples: {n_t}")

    t0 = time.time()
    if program_language == 'cpp':
        img = tfm_cpp.tfm2D(
            time_data, time_axis.astype(np.float64),
            tx0, rx0, xc, yc, zc, X, Y, Z, c,
        )
    elif program_language == 'gpu':
        img = tfm_gpu.tfm2D_GPU(
            time_data, time_axis.astype(np.float64),
            tx0, rx0, xc, yc, zc, X, Y, Z, c, 512,
        )
    else:
        raise NotImplementedError(
            "Pure-Python 3D TFM is not available; build tfm_cpp or tfm_gpu."
        )
    print(f"  TFM done in {time.time() - t0:.2f}s, shape {img.shape}")

    if img_output == 'envelope':
        env = np.abs(hilbert(img, axis=0))
        env_max = float(env.max())
        img_db = 20.0 * np.log10(env / max(env_max, 1e-30) + 1e-10)
        img_db = np.clip(img_db, -db_range, 0.0)
    else:
        env = np.abs(hilbert(img, axis=0))
        env_max = float(env.max())
        img_db = 20.0 * np.log10(env / max(env_max, 1e-30) + 1e-10)
        img_db = np.clip(img_db, -db_range, 0.0)

    return img_db, {'x': x_img, 'y': y_img, 'z': z_img}


def show_napari(img_db: np.ndarray, axes: dict, db_range: float,
                grain_volume=None) -> None:
    """Open the 3D TFM volume (and optional grain volume) in napari."""
    import napari

    dz = float(axes['z'][1] - axes['z'][0]) * 1e3
    dy = float(axes['y'][1] - axes['y'][0]) * 1e3
    dx = float(axes['x'][1] - axes['x'][0]) * 1e3
    tz = float(axes['z'][0]) * 1e3
    ty = float(axes['y'][0]) * 1e3
    tx = float(axes['x'][0]) * 1e3

    viewer = napari.Viewer()

    if grain_volume is not None:
        vs_mm = float(grain_volume.voxel_size) * 1e3
        viewer.add_image(
            grain_volume.impedance,
            name='Grain impedance',
            scale=(vs_mm, vs_mm, vs_mm),
            translate=(float(grain_volume.origin_z) * 1e3,
                       float(grain_volume.origin_y) * 1e3,
                       float(grain_volume.origin_x) * 1e3),
            colormap='gray',
            rendering='attenuated_mip',
            opacity=0.5,
            visible=True,
        )

    viewer.add_image(
        img_db,
        name='3D TFM (dB)',
        scale=(dz, dy, dx),
        translate=(tz, ty, tx),
        contrast_limits=(-db_range, 0.0),
        colormap='inferno',
        rendering='mip',
    )
    viewer.dims.axis_labels = ('z (mm)', 'y (mm)', 'x (mm)')
    napari.run()


def view_in_napari(output_dir: Path = OUTPUT_DIR,
                   db_range: float = TFM_DB_RANGE) -> None:
    """
    Open every saved volume in `output_dir` in napari.

    Loads:
        - grain_volume.npz   → "Grain impedance" image layer (if present)
        - volume_<i>.npz     → one "TFM (dB) — i" layer per scan frame

    Run as a standalone script after `main()` has produced the files:
        python -c "from run_engine_3d import view_in_napari; view_in_napari()"
    """
    import napari

    output_dir = Path(output_dir)
    if not output_dir.exists():
        raise FileNotFoundError(f"Output directory not found: {output_dir}")

    viewer = napari.Viewer(title=f'Saved volumes — {output_dir.name}')

    grain_path = output_dir / 'grain_volume.npz'
    if grain_path.exists():
        g = np.load(grain_path)
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
            visible=True,
        )
        print(f"  Loaded grain volume: {g['impedance'].shape} from {grain_path.name}")

    vol_paths = sorted(output_dir.glob('volume_*.npz'))
    if not vol_paths:
        print(f"  No volume_*.npz files found in {output_dir}")
    for vp in vol_paths:
        v = np.load(vp)
        x = v['x']; y = v['y']; z = v['z']
        dz = float(z[1] - z[0]) * 1e3
        dy = float(y[1] - y[0]) * 1e3
        dx = float(x[1] - x[0]) * 1e3
        tag = vp.stem.split('_')[-1]
        viewer.add_image(
            v['img_db'],
            name=f'TFM (dB) — {tag}',
            scale=(dz, dy, dx),
            translate=(float(z[0]) * 1e3, float(y[0]) * 1e3, float(x[0]) * 1e3),
            contrast_limits=(-db_range, 0.0),
            colormap='inferno',
            rendering='mip',
        )
        print(f"  Loaded TFM volume {tag}: {v['img_db'].shape} from {vp.name}")

    viewer.dims.axis_labels = ('z (mm)', 'y (mm)', 'x (mm)')
    napari.run()


def load_array_geometry_csv(path: Path) -> Tuple[np.ndarray, float, float]:
    """Load element centres + widths from an array_geometry.csv."""
    data = np.genfromtxt(path, delimiter=',', names=True)
    xc = data['el_xc'].astype(float)
    yc = data['el_yc'].astype(float)
    width_x = float(np.median(np.abs(data['el_x2'] - data['el_x1'])))
    width_y = float(np.median(np.abs(data['el_y2'] - data['el_y1'])))
    return np.stack([xc, yc], axis=1), width_x, width_y


def _infer_pitch(coords: np.ndarray) -> float:
    """Smallest non-zero gap between sorted unique coordinate values."""
    uniq = np.unique(np.round(coords, 9))
    diffs = np.diff(uniq)
    diffs = diffs[diffs > 1e-9]
    return float(diffs.min()) if diffs.size else 0.0


def build_config() -> SimulationConfig3D:
    if ARRAY_MODE == "matrix":
        array_cfg = ArrayConfig3D(
            n_elements_x=N_ELEMENTS_X,
            n_elements_y=N_ELEMENTS_Y,
            pitch_x=PITCH_X_MM * 1e-3,
            pitch_y=PITCH_Y_MM * 1e-3,
            element_width_x=ELEMENT_WIDTH_X_MM * 1e-3,
            element_width_y=ELEMENT_WIDTH_Y_MM * 1e-3,
            frequency=FREQUENCY_MHZ * 1e6,
            bandwidth=BANDWIDTH,
        )
    elif ARRAY_MODE == "csv":
        positions, w_x, w_y = load_array_geometry_csv(ARRAY_CSV)
        array_cfg = ArrayConfig3D(
            n_elements_x=positions.shape[0],
            n_elements_y=1,
            pitch_x=_infer_pitch(positions[:, 0]),
            pitch_y=_infer_pitch(positions[:, 1]),
            element_width_x=w_x,
            element_width_y=w_y,
            frequency=FREQUENCY_MHZ * 1e6,
            bandwidth=BANDWIDTH,
            custom_positions=positions,
        )
    else:
        raise ValueError(f"ARRAY_MODE must be 'matrix' or 'csv', got {ARRAY_MODE!r}")

    acq_cfg = AcquisitionConfig3D(
        time_samples=TIME_SAMPLES,
        sampling_frequency=(SAMPLING_MHZ * 1e6) if SAMPLING_MHZ is not None else None,
    )
    return SimulationConfig3D(
        material=MATERIAL,
        array=array_cfg,
        specimen=SpecimenConfig3D(
            thickness=SPECIMEN_THICKNESS_MM * 1e-3,
            width=SPECIMEN_WIDTH_MM * 1e-3,
            depth=SPECIMEN_DEPTH_MM * 1e-3,
        ),
        acquisition=acq_cfg,
    )


def scan_volume_3d_rotational(
    cfg: SimulationConfig3D,
    defects_3d: list,
    angles_deg: list,
    output_dir: Path,
    voxel_volume=None,
    born_threshold: float = 0.005,
    defect_n_points: int = 1200,
) -> None:
    """
    Rotational 3D scan: full 3D FMC + 3D TFM volume per angle.

    The scatterer cloud is rotated around the z axis between frames (cheap
    coordinate transform), instead of re-voxelising the volume per angle.
    Saves one fmc_<i>.npy and one volume_<i>.npz per angle.

    Args:
        cfg:            3D simulation configuration
        defects_3d:     Geometric defects (combined with the voxel volume)
        angles_deg:     Rotation angles about z (degrees)
        output_dir:     Per-frame output directory
        voxel_volume:   Optional VoxelVolume3D for grain background
        born_threshold: Min |ΔZ / 2Z₀| for voxel-Born scatterers
        defect_n_points: Surface samples per defect
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    assert cfg.material is not None
    Z0 = cfg.material.Z_L

    # Persist the grain volume so view_in_napari() can load it later.
    if voxel_volume is not None:
        grain_path = output_dir / 'grain_volume.npz'
        np.savez_compressed(
            grain_path,
            impedance=voxel_volume.impedance,
            voxel_size=np.float64(voxel_volume.voxel_size),
            origin_z=np.float64(voxel_volume.origin_z),
            origin_y=np.float64(voxel_volume.origin_y),
            origin_x=np.float64(voxel_volume.origin_x),
        )
        print(f"  Saved grain volume ({voxel_volume.impedance.shape}) to {grain_path.name}")

    # Pre-extract the un-rotated scatterer cloud once (defects + voxel volume).
    z0, x0, y0, a0 = _gather_scatterers(
        cfg, defects_3d, voxel_volume, born_threshold, defect_n_points,
    )
    print(f"\n  Total un-rotated scatterers: {z0.size:,}")

    n = len(angles_deg)
    for i, theta_deg in enumerate(angles_deg):
        theta = np.deg2rad(theta_deg)
        ct, st = np.cos(theta), np.sin(theta)
        x_rot = x0 * ct - y0 * st
        y_rot = x0 * st + y0 * ct

        print(f"\n  Frame {i+1}/{n}  θ = {theta_deg:+.1f}°")

        engine = FMCEngine3D(cfg)
        engine.set_born_scatterers(z0, x_rot, y_rot, a0)
        result = engine.simulate(tx_chunk=TX_CHUNK)

        fmc = result['fmc_data']
        time_axis = result['time_axis']
        elem_xyz = result['element_positions_xyz']

        fmc = add_noise(fmc, snr_db=cfg.acquisition.snr_db,
                        grain_noise_level=cfg.acquisition.grain_noise_level)
        fmc = apply_bandpass_filter(fmc, cfg.dt, cfg.array.frequency,
                                     bandwidth_fraction=cfg.array.bandwidth,
                                     filter_alpha=cfg.acquisition.filter_alpha,
                                     hanning_bool=cfg.acquisition.hanning_bool)

        tag = f"{i:04d}"
        fmc_path = output_dir / f"fmc_{tag}.npy"
        np.save(fmc_path, fmc)
        print(f"  Saved FMC ({fmc.shape}) to {fmc_path.name}")

        if RUN_TFM:
            ap_x = float(elem_xyz[:, 1].max() - elem_xyz[:, 1].min())
            ap_y = float(elem_xyz[:, 2].max() - elem_xyz[:, 2].min())
            x_min, x_max = -ap_x / 2, ap_x / 2
            y_min, y_max = -ap_y / 2, ap_y / 2
            if ap_y == 0.0:
                # 1D linear array: give it a non-zero elevation slab
                y_min, y_max = -1e-3, 1e-3
            z_min = (Z_MIN_MM * 1e-3) if Z_MIN_MM is not None else 0.0
            z_max = (Z_MAX_MM * 1e-3) if Z_MAX_MM is not None else cfg.specimen.thickness

            img_db, axes = reconstruct_tfm_3d(
                fmc, time_axis, elem_xyz, float(cfg.material.c_L),
                x_range=(x_min, x_max),
                y_range=(y_min, y_max),
                z_range=(z_min, z_max),
                n_x=X_PIXELS, n_y=Y_PIXELS, n_z=Z_PIXELS,
                db_range=TFM_DB_RANGE,
            )
            vol_path = output_dir / f"volume_{tag}.npz"
            np.savez_compressed(
                vol_path,
                img_db=img_db,
                x=axes['x'], y=axes['y'], z=axes['z'],
                db_range=np.float64(TFM_DB_RANGE),
                theta_deg=np.float64(theta_deg),
            )
            print(f"  Saved TFM volume ({img_db.shape}) to {vol_path.name}")

            if SHOW_NAPARI and n == 1:
                show_napari(img_db, axes, TFM_DB_RANGE,
                            grain_volume=voxel_volume)


def _gather_scatterers(cfg: SimulationConfig3D, defects_3d: list,
                        voxel_volume, born_threshold: float,
                        defect_n_points: int) -> Tuple[np.ndarray, np.ndarray,
                                                         np.ndarray, np.ndarray]:
    """Combine voxel-Born + defect-surface scatterers into a single cloud."""
    from engine_3d.geometry import defect_to_born_scatterers_3d

    z_parts, x_parts, y_parts, a_parts = [], [], [], []
    if voxel_volume is not None:
        assert cfg.material is not None
        z_s, x_s, y_s, a_s = extract_born_scatterers_3d(
            voxel_volume, background_Z=cfg.material.Z_L,
            threshold=born_threshold,
        )
        if z_s.size > 0:
            z_parts.append(z_s); x_parts.append(x_s)
            y_parts.append(y_s); a_parts.append(a_s)
            print(f"  Voxel-Born scatterers: {z_s.size:,}")
    for d in defects_3d:
        zd, xd, yd, ad = defect_to_born_scatterers_3d(d, n_points=defect_n_points)
        if zd.size > 0:
            z_parts.append(zd); x_parts.append(xd)
            y_parts.append(yd); a_parts.append(ad)
            print(f"  {type(d).__name__} surface scatterers: {zd.size:,}")
    if not z_parts:
        empty = np.empty(0, dtype=np.float64)
        return empty, empty, empty, empty
    return (np.concatenate(z_parts), np.concatenate(x_parts),
            np.concatenate(y_parts), np.concatenate(a_parts))


# =============================================================================
# Main
# =============================================================================


def main():
    print(f"\n{'#'*70}")
    print(f"# BORN-ONLY 3D NDT SYNTHETIC DATA ENGINE")
    print(f"{'#'*70}\n")

    cfg = build_config()
    print(cfg.summary())

    voxel_volume = None
    if USE_GRAIN:
        assert cfg.material is not None
        voxel_volume = generate_grain_structure(
            thickness=cfg.specimen.thickness,
            width=cfg.specimen.width,
            depth=cfg.specimen.depth,
            background_material=cfg.material,
            mean_grain_size_m=GRAIN_SIZE_MM * 1e-3,
            voxel_size_m=GRAIN_VOXEL_MM * 1e-3,
            impedance_variation=GRAIN_IMP_VAR,
        )
        if defects_3d:
            voxel_volume = embed_geometric_defects(voxel_volume, defects_3d)
        print(f"  Voxel volume shape: {voxel_volume.impedance.shape}")

    # Geometric-defect surfaces are added explicitly when no voxel volume is
    # provided. With a voxel volume, the defects are already burned into the
    # impedance field by embed_geometric_defects(), so we don't double-count.
    geom_defects = [] if voxel_volume is not None else defects_3d

    scan_volume_3d_rotational(
        cfg=cfg,
        defects_3d=geom_defects,
        angles_deg=SCAN_ANGLES_DEG,
        output_dir=OUTPUT_DIR,
        voxel_volume=voxel_volume,
        born_threshold=BORN_THRESHOLD,
        defect_n_points=DEFECT_N_POINTS,
    )

    print(f"\n{'#'*70}")
    print(f"# SCAN COMPLETE — {len(SCAN_ANGLES_DEG)} frame(s) saved to "
          f"{OUTPUT_DIR}/")
    print(f"{'#'*70}\n")


if __name__ == '__main__':
    main()
