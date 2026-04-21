"""
Generate a 2D grid of 3D scans with a uniform overlap in x and y for
testing 2D stitching.

For a GRID_NX × GRID_NY sequence at a single OVERLAP fraction:
    1. Build one grain volume wide enough to host every scan window.
    2. For each (ix, iy) tile, shift the grain origin by (dx, dy) so the
       array sees that tile.
    3. Save FMC + TFM volume per tile with a name encoding (ix, iy).

    overlap = 1.0  → shift = 0           (all tiles identical)
    overlap = 0.0  → shift = aperture    (adjacent, non-overlapping)
"""

import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from engine_3d import (
    SimulationConfig3D, ArrayConfig3D, SpecimenConfig3D, AcquisitionConfig3D,
    FMCEngine3D, COPPER, extract_born_scatterers_3d,
)
from engine.microstructure import generate_grain_structure
from engine.voxel_volume import VoxelVolume3D

import run_engine_3d
from run_engine_3d import (
    add_noise, apply_bandpass_filter, reconstruct_tfm_3d,
    load_array_geometry_csv, _infer_pitch,
)

HERE_REPO = HERE.parent


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
ARRAY_CSV          = (HERE_REPO / "DATA" / "2D Processed Data"
                      / "Cu Pure 7.5MHz Ex 15042026 Filtered"
                      / "11_filtered" / "array_geometry.csv")

# Acquisition
FREQUENCY_MHZ      = 7.5
BANDWIDTH          = 0.8
TIME_SAMPLES       = 1024
SAMPLING_MHZ       = 50
TX_CHUNK           = 1

# Specimen + material
MATERIAL              = replace(COPPER, attenuation_L=0.3, attenuation_S=1.2)
SPECIMEN_THICKNESS_MM = 40.0
MARGIN_MM             = 10.0          # grain padding beyond the scan window

# Grain microstructure
GRAIN_SIZE_MM      = 0.5
GRAIN_VOXEL_MM     = 0.4
GRAIN_IMP_VAR      = 0.05
BORN_THRESHOLD     = 0.005

# TFM
PROGRAM_LANGUAGE   = "cpp"   # "cpp" or "gpu"
RUN_TFM            = True
X_PIXELS           = 200
Y_PIXELS           = 200
Z_PIXELS           = 400
Z_MIN_MM           = 15.0
Z_MAX_MM           = 35.0
TFM_DB_RANGE       = 20.0

# Sequence grid
OVERLAP            = 0.50
GRID_NX            = 3
GRID_NY            = 3
SEED               = 1000

OUTPUT_DIR         = HERE / 'output' / 'engine_3d_overlap_sequence'


# =============================================================================
# Pipeline
# =============================================================================


def _select_tfm_backend(lang: str) -> None:
    """Monkey-patch run_engine_3d so reconstruct_tfm_3d dispatches to the chosen backend."""
    if lang not in ("cpp", "gpu"):
        raise ValueError(f"PROGRAM_LANGUAGE must be 'cpp' or 'gpu', got {lang!r}")
    if lang == run_engine_3d.program_language:
        return
    if lang == "gpu":
        gpu_build = HERE_REPO / "build" / "CPP" / "TFM_GPU"
        if str(gpu_build) not in sys.path:
            sys.path.insert(0, str(gpu_build))
        import tfm_gpu
        run_engine_3d.tfm_gpu = tfm_gpu
        print("GPU Available (overlap-sequence override)")
    run_engine_3d.program_language = lang


def build_array_config() -> ArrayConfig3D:
    if ARRAY_MODE == "matrix":
        return ArrayConfig3D(
            n_elements_x=N_ELEMENTS_X,
            n_elements_y=N_ELEMENTS_Y,
            pitch_x=PITCH_X_MM * 1e-3,
            pitch_y=PITCH_Y_MM * 1e-3,
            element_width_x=ELEMENT_WIDTH_X_MM * 1e-3,
            element_width_y=ELEMENT_WIDTH_Y_MM * 1e-3,
            frequency=FREQUENCY_MHZ * 1e6,
            bandwidth=BANDWIDTH,
        )
    if ARRAY_MODE == "csv":
        positions, w_x, w_y = load_array_geometry_csv(ARRAY_CSV)
        return ArrayConfig3D(
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
    raise ValueError(f"ARRAY_MODE must be 'matrix' or 'csv', got {ARRAY_MODE!r}")


def build_config(array_cfg: ArrayConfig3D,
                 width_x_m: float, depth_y_m: float) -> SimulationConfig3D:
    acq_cfg = AcquisitionConfig3D(
        time_samples=TIME_SAMPLES,
        sampling_frequency=SAMPLING_MHZ * 1e6,
    )
    return SimulationConfig3D(
        material=MATERIAL,
        array=array_cfg,
        specimen=SpecimenConfig3D(
            thickness=SPECIMEN_THICKNESS_MM * 1e-3,
            width=width_x_m,
            depth=depth_y_m,
        ),
        acquisition=acq_cfg,
    )


def scan_one(cfg: SimulationConfig3D, voxel_volume: VoxelVolume3D,
             tag: str, out_dir: Path, x_half: float, y_half: float) -> None:
    """Run FMC → noise → filter → TFM for a single tile."""
    assert cfg.material is not None
    z_s, x_s, y_s, a_s = extract_born_scatterers_3d(
        voxel_volume, background_Z=cfg.material.Z_L, threshold=BORN_THRESHOLD,
    )
    print(f"    [{tag}] scatterers: {z_s.size:,}")

    engine = FMCEngine3D(cfg)
    engine.set_born_scatterers(z_s, x_s, y_s, a_s)
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

    fmc_path = out_dir / f"fmc_{tag}.npy"
    np.save(fmc_path, fmc)
    print(f"    [{tag}] saved FMC {fmc.shape} → {fmc_path.name}")

    if RUN_TFM:
        z_min = Z_MIN_MM * 1e-3
        z_max = (Z_MAX_MM * 1e-3) if Z_MAX_MM is not None else cfg.specimen.thickness

        img_db, axes = reconstruct_tfm_3d(
            fmc, time_axis, elem_xyz, float(cfg.material.c_L),
            x_range=(-x_half, x_half),
            y_range=(-y_half, y_half),
            z_range=(z_min, z_max),
            n_x=X_PIXELS, n_y=Y_PIXELS, n_z=Z_PIXELS,
            db_range=TFM_DB_RANGE,
        )
        vol_path = out_dir / f"volume_{tag}.npy"
        np.save(vol_path, img_db)
        print(f"    [{tag}] saved TFM {img_db.shape} → {vol_path.name}")


def main() -> None:
    _select_tfm_backend(PROGRAM_LANGUAGE)
    array_cfg = build_array_config()
    aperture_x = array_cfg.aperture_x
    aperture_y = max(array_cfg.aperture_y, 1e-3)
    L = max(aperture_x, aperture_y)
    x_half = y_half = L / 2
    margin = MARGIN_MM * 1e-3

    shift = L * (1.0 - OVERLAP)
    grain_width_x = L + (GRID_NX - 1) * shift + 2 * margin
    grain_width_y = L + (GRID_NY - 1) * shift + 2 * margin

    seq_dir = OUTPUT_DIR / f"seq_{GRID_NX}x{GRID_NY}_ovlp{int(round(OVERLAP * 100)):03d}"
    seq_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#'*70}")
    print(f"# 2D OVERLAP-SEQUENCE DATASET GENERATOR")
    print(f"# aperture_x = {aperture_x*1e3:.2f} mm, "
          f"aperture_y = {aperture_y*1e3:.2f} mm  (recon side = {L*1e3:.2f} mm)")
    print(f"# grid       = {GRID_NX} x {GRID_NY}   overlap = {OVERLAP:.2f}   "
          f"shift = {shift*1e3:.2f} mm")
    print(f"# grain      = {grain_width_x*1e3:.1f} x {grain_width_y*1e3:.1f} mm")
    print(f"# output     = {seq_dir}")
    print(f"{'#'*70}")

    cfg = build_config(array_cfg, grain_width_x, grain_width_y)

    t0 = time.time()
    grain = generate_grain_structure(
        thickness=SPECIMEN_THICKNESS_MM * 1e-3,
        width=grain_width_x,
        depth=grain_width_y,
        background_material=MATERIAL,
        mean_grain_size_m=GRAIN_SIZE_MM * 1e-3,
        voxel_size_m=GRAIN_VOXEL_MM * 1e-3,
        impedance_variation=GRAIN_IMP_VAR,
        seed=SEED,
    )
    grain.origin_x = -grain_width_x / 2
    grain.origin_y = -grain_width_y / 2

    np.savez_compressed(
        seq_dir / 'grain_volume.npz',
        impedance=grain.impedance,
        voxel_size=np.float64(grain.voxel_size),
        origin_z=np.float64(grain.origin_z),
        origin_y=np.float64(grain.origin_y),
        origin_x=np.float64(grain.origin_x),
    )

    total_tiles = GRID_NX * GRID_NY
    tile_counter = 0
    tiles_meta = []

    for iy in range(GRID_NY):
        for ix in range(GRID_NX):
            tile_counter += 1
            dx = (ix - (GRID_NX - 1) / 2) * shift
            dy = (iy - (GRID_NY - 1) / 2) * shift
            tag = f"ix{ix:02d}_iy{iy:02d}"

            print(f"\n  tile {tile_counter}/{total_tiles}  {tag}  "
                  f"(dx={dx*1e3:+.2f} mm, dy={dy*1e3:+.2f} mm)")

            shifted = VoxelVolume3D(
                impedance=grain.impedance,
                wavespeed=grain.wavespeed,
                voxel_size=grain.voxel_size,
                origin_z=grain.origin_z,
                origin_y=grain.origin_y + dy,
                origin_x=grain.origin_x + dx,
            )
            scan_one(cfg, shifted, tag, seq_dir, x_half, y_half)

            tiles_meta.append({
                'ix': int(ix),
                'iy': int(iy),
                'dx_m': float(dx),
                'dy_m': float(dy),
                'tag': tag,
            })

    meta = {
        'overlap_fraction': float(OVERLAP),
        'shift_m': float(shift),
        'grid_nx': int(GRID_NX),
        'grid_ny': int(GRID_NY),
        'aperture_x_m': float(array_cfg.aperture_x),
        'aperture_y_m': float(array_cfg.aperture_y),
        'recon_half_extent_m': float(x_half),
        'grain_width_x_m': float(grain_width_x),
        'grain_width_y_m': float(grain_width_y),
        'seed': int(SEED),
        'tiles': tiles_meta,
        'tfm_pixels': [Z_PIXELS, Y_PIXELS, X_PIXELS] if RUN_TFM else None,
        'tfm_z_range_m': [Z_MIN_MM * 1e-3, Z_MAX_MM * 1e-3] if RUN_TFM else None,
        'array': {
            'mode': ARRAY_MODE,
            'n_elements_x': int(array_cfg.n_elements_x),
            'n_elements_y': int(array_cfg.n_elements_y),
            'pitch_x_m': float(array_cfg.pitch_x),
            'pitch_y_m': float(array_cfg.pitch_y),
            'frequency_Hz': float(array_cfg.frequency),
        },
        'material': MATERIAL.name,
    }
    with open(seq_dir / 'meta.json', 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\n{'#'*70}")
    print(f"# SEQUENCE COMPLETE — {total_tiles} tiles in {time.time() - t0:.1f}s "
          f"→ {seq_dir}")
    print(f"{'#'*70}\n")


if __name__ == '__main__':
    main()
