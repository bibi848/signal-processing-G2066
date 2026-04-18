"""
Generate pairs of 3D scans with a configurable lateral (x) overlap for
testing the stitching algorithm.

For each overlap fraction in OVERLAPS:
    1. Build a fresh grain volume wide enough to host both scan windows.
    2. Take scan A with array centred at grain x=0.
    3. Take scan B with the volume origin shifted by `shift` in x, so the
       array sees a region offset by `shift` relative to scan A.
    4. Save FMC, TFM volume, grain volume, and meta.json per pair.

    overlap_fraction = 1.0  → shift = 0           (identical scans)
    overlap_fraction = 0.0  → shift = aperture_x  (adjacent, non-overlapping)
"""

import json
import sys
import time
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
MATERIAL              = COPPER
SPECIMEN_THICKNESS_MM = 40.0
MARGIN_MM             = 2.0          # grain padding beyond the scan window

# Grain microstructure
GRAIN_SIZE_MM      = 0.5
GRAIN_VOXEL_MM     = 0.5
GRAIN_IMP_VAR      = 0.025
BORN_THRESHOLD     = 0.005

# TFM
PROGRAM_LANGUAGE   = "cpp"   # "cpp" or "gpu"  (matches run_engine_3d)
RUN_TFM            = True
X_PIXELS           = 200
Y_PIXELS           = 200
Z_PIXELS           = 400
Z_MIN_MM           = 0.0
Z_MAX_MM           = 35.0
TFM_DB_RANGE       = 40.0

# Overlap sweep
OVERLAPS             = [0.8]
N_PAIRS_PER_OVERLAP  = 1
BASE_SEED            = 1000

OUTPUT_DIR         = HERE / 'output' / 'engine_3d_overlap_sweep'


# =============================================================================
# Pipeline
# =============================================================================


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
             tag: str, out_dir: Path) -> None:
    """Run FMC → noise → filter → TFM for a single array position."""
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
        ap_x = float(elem_xyz[:, 1].max() - elem_xyz[:, 1].min())
        ap_y = float(elem_xyz[:, 2].max() - elem_xyz[:, 2].min())
        if ap_y == 0.0:
            ap_y = 1e-3
        z_min = Z_MIN_MM * 1e-3
        z_max = (Z_MAX_MM * 1e-3) if Z_MAX_MM is not None else cfg.specimen.thickness

        img_db, axes = reconstruct_tfm_3d(
            fmc, time_axis, elem_xyz, float(cfg.material.c_L),
            x_range=(-ap_x / 2, ap_x / 2),
            y_range=(-ap_y / 2, ap_y / 2),
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
    aperture_y = max(array_cfg.aperture_y, 1e-3)   # 1D arrays report 0
    margin = MARGIN_MM * 1e-3

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#'*70}")
    print(f"# OVERLAP-SWEEP DATASET GENERATOR")
    print(f"# aperture_x = {aperture_x*1e3:.2f} mm, "
          f"aperture_y = {aperture_y*1e3:.2f} mm")
    print(f"# overlaps  = {OVERLAPS}")
    print(f"# output    = {OUTPUT_DIR}")
    print(f"{'#'*70}")

    total_pairs = len(OVERLAPS) * N_PAIRS_PER_OVERLAP
    pair_counter = 0

    for i, overlap in enumerate(OVERLAPS):
        shift = aperture_x * (1.0 - overlap)
        grain_width_x = aperture_x + shift + 2 * margin
        grain_width_y = aperture_y + 2 * margin

        overlap_dir = OUTPUT_DIR / f"ovlp_{int(round(overlap * 100)):03d}"
        overlap_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n── overlap {i+1}/{len(OVERLAPS)} = {overlap:.2f}  "
              f"shift={shift*1e3:.2f} mm  "
              f"grain=({grain_width_x*1e3:.1f}×{grain_width_y*1e3:.1f} mm) ──")

        cfg = build_config(array_cfg, grain_width_x, grain_width_y)

        for j in range(N_PAIRS_PER_OVERLAP):
            pair_counter += 1
            seed = BASE_SEED + i * N_PAIRS_PER_OVERLAP + j
            pair_dir = overlap_dir / f"pair_{j:02d}"
            pair_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n  pair {pair_counter}/{total_pairs}  "
                  f"(overlap={overlap:.2f}, seed={seed}) → {pair_dir.name}")
            t0 = time.time()

            grain = generate_grain_structure(
                thickness=cfg.specimen.thickness,
                width=grain_width_x,
                depth=grain_width_y,
                background_material=cfg.material,
                mean_grain_size_m=GRAIN_SIZE_MM * 1e-3,
                voxel_size_m=GRAIN_VOXEL_MM * 1e-3,
                impedance_variation=GRAIN_IMP_VAR,
                seed=seed,
            )
            grain.origin_x = -grain_width_x / 2
            grain.origin_y = -grain_width_y / 2

            np.savez_compressed(
                pair_dir / 'grain_volume.npz',
                impedance=grain.impedance,
                voxel_size=np.float64(grain.voxel_size),
                origin_z=np.float64(grain.origin_z),
                origin_y=np.float64(grain.origin_y),
                origin_x=np.float64(grain.origin_x),
            )

            for tag, dx in [('A', -shift / 2), ('B', +shift / 2)]:
                shifted = VoxelVolume3D(
                    impedance=grain.impedance,
                    wavespeed=grain.wavespeed,
                    voxel_size=grain.voxel_size,
                    origin_z=grain.origin_z,
                    origin_y=grain.origin_y,
                    origin_x=grain.origin_x + dx,
                )
                scan_one(cfg, shifted, tag, pair_dir)

            meta = {
                'overlap_fraction': float(overlap),
                'pair_index': int(j),
                'shift_m': float(shift),
                'aperture_x_m': float(aperture_x),
                'aperture_y_m': float(aperture_y),
                'grain_width_x_m': float(grain_width_x),
                'grain_width_y_m': float(grain_width_y),
                'seed': int(seed),
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
            with open(pair_dir / 'meta.json', 'w') as f:
                json.dump(meta, f, indent=2)

            print(f"  pair done in {time.time() - t0:.1f}s")

    print(f"\n{'#'*70}")
    print(f"# SWEEP COMPLETE — {total_pairs} pair(s) "
          f"({len(OVERLAPS)} overlap × {N_PAIRS_PER_OVERLAP}) in {OUTPUT_DIR}")
    print(f"{'#'*70}\n")


if __name__ == '__main__':
    main()
