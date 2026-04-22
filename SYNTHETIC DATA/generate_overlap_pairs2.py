"""
Can later delete this file when presenting the github
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
    apply_bandpass_filter, reconstruct_tfm_3d,
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
PROGRAM_LANGUAGE   = "gpu" # "cpp" or "gpu" 
RUN_TFM            = True
X_PIXELS           = 200
Y_PIXELS           = 200
Z_PIXELS           = 400
Z_MIN_MM           = 15.0
Z_MAX_MM           = 35.0
TFM_DB_RANGE       = 20.0

# Overlap sweep
OVERLAPS             = [0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 
                        0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
N_PAIRS_PER_OVERLAP  = 10
BASE_SEED            = 120

# Array rotations
ROTATE_SCANS = False
N_ROTATIONS  = 4

DETERMINISTIC_NOISE = True
INTER_VOLUME_NOISE_SCALE = 1.0
OUTPUT_DIR  = HERE / 'output' / 'engine_3d_overlap_sweep'

# Pipeline

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
        print("GPU Available")
    run_engine_3d.program_language = lang


def _rotate_array_cfg(base_cfg: ArrayConfig3D, k: int) -> ArrayConfig3D:
    """Rotate element positions by k*90° CCW about world origin (k ∈ {0,1,2,3}).

    Rotating the probe in place is physically equivalent to rotating its element
    positions about the array centre; the grain stays fixed. Element widths and
    pitches also swap for odd k (the element's long axis rotates with it).
    """
    k = k % 4
    base_pos = (base_cfg.custom_positions
                if base_cfg.custom_positions is not None
                else base_cfg.element_positions()[:, 1:3])
    if k == 0:
        rot_pos = base_pos.copy()
    elif k == 1:
        rot_pos = np.stack([-base_pos[:, 1], base_pos[:, 0]], axis=1)
    elif k == 2:
        rot_pos = -base_pos
    else:
        rot_pos = np.stack([base_pos[:, 1], -base_pos[:, 0]], axis=1)
    swap = (k % 2 == 1)
    w_x = base_cfg.element_width_y if swap else base_cfg.element_width_x
    w_y = base_cfg.element_width_x if swap else base_cfg.element_width_y
    p_x = base_cfg.pitch_y        if swap else base_cfg.pitch_x
    p_y = base_cfg.pitch_x        if swap else base_cfg.pitch_y
    return ArrayConfig3D(
        n_elements_x=rot_pos.shape[0], n_elements_y=1,
        pitch_x=p_x, pitch_y=p_y,
        element_width_x=w_x, element_width_y=w_y,
        frequency=base_cfg.frequency, bandwidth=base_cfg.bandwidth,
        z_position=base_cfg.z_position,
        custom_positions=rot_pos,
    )


def _array_stats(arr: np.ndarray) -> dict:
    """Small JSON-friendly summary for checking scan strength."""
    arr64 = arr.astype(np.float64, copy=False)
    return {
        'min': float(arr64.min()),
        'max': float(arr64.max()),
        'mean_abs': float(np.mean(np.abs(arr64))),
        'rms': float(np.sqrt(np.mean(arr64 ** 2))),
    }


def _add_noise_to_fmc(fmc_data: np.ndarray, snr_db: float,
                      grain_noise_level: float, seed: int | None,
                      noise_scale: float = 1.0) -> np.ndarray:
    """Same noise model as run_engine_3d.add_noise, with a local RNG."""
    signal_power = np.mean(fmc_data ** 2)
    if signal_power < 1e-30:
        return fmc_data

    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
    noise_std = np.sqrt(signal_power / 10 ** (snr_db / 10)) * float(noise_scale)
    noisy = fmc_data + rng.normal(0, noise_std, fmc_data.shape).astype(np.float32)

    grain = rng.normal(0, grain_noise_level * noise_std, fmc_data.shape)
    from scipy.ndimage import uniform_filter1d
    grain = uniform_filter1d(grain, size=5, axis=2).astype(np.float32)
    noisy += grain
    return noisy


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
             tag: str, out_dir: Path, x_half: float, y_half: float,
             noise_seed: int | None = None) -> dict:
    """Run FMC → noise → filter → TFM for a single array position/rotation.

    TFM reconstructs in a fixed world box [±x_half, ±y_half] so that every
    rotation at a given position shares the same spatial frame.
    """
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
    sim_stats = _array_stats(fmc)

    noise_mode = 'none'
    actual_noise_seed = None
    noise_scale = 0.0
    if cfg.acquisition.add_noise:
        if DETERMINISTIC_NOISE:
            noise_mode = 'deterministic_seeded'
            actual_noise_seed = noise_seed
            noise_scale = 1.0
        else:
            noise_mode = 'random_inter_volume'
            actual_noise_seed = None
            noise_scale = INTER_VOLUME_NOISE_SCALE

        fmc = _add_noise_to_fmc(
            fmc, snr_db=cfg.acquisition.snr_db,
            grain_noise_level=cfg.acquisition.grain_noise_level,
            seed=actual_noise_seed,
            noise_scale=noise_scale,
        )
    noisy_stats = _array_stats(fmc)

    fmc = apply_bandpass_filter(fmc, cfg.dt, cfg.array.frequency,
                                 bandwidth_fraction=cfg.array.bandwidth,
                                 filter_alpha=cfg.acquisition.filter_alpha,
                                 hanning_bool=cfg.acquisition.hanning_bool)
    filtered_stats = _array_stats(fmc)

    fmc_path = out_dir / f"fmc_{tag}.npy"
    # np.save(fmc_path, fmc)
    # print(f"    [{tag}] saved FMC {fmc.shape} → {fmc_path.name}")

    tfm_stats = None
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
        tfm_stats = _array_stats(img_db)
        print(f"    [{tag}] saved TFM {img_db.shape} → {vol_path.name}")

    return {
        'tag': tag,
        'noise_mode': noise_mode,
        'noise_seed': actual_noise_seed,
        'noise_scale': float(noise_scale),
        'scatterers': int(z_s.size),
        'fmc_simulated': sim_stats,
        'fmc_after_noise': noisy_stats,
        'fmc_after_filter': filtered_stats,
        'tfm_db': tfm_stats,
    }


def main() -> None:
    _select_tfm_backend(PROGRAM_LANGUAGE)
    array_cfg = build_array_config()
    aperture_x = array_cfg.aperture_x
    aperture_y = max(array_cfg.aperture_y, 1e-3)   # 1D arrays report 0
    rotation_indices = list(range(N_ROTATIONS)) if ROTATE_SCANS else [0]
    # After rotation the aperture spans max(ap_x, ap_y) along whichever axis
    # the long side has swung to — size the recon box + grain so every
    # rotation fits.
    L = max(aperture_x, aperture_y)
    x_half = y_half = L / 2
    margin = MARGIN_MM * 1e-3

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#'*70}")
    print(f"# OVERLAP-SWEEP DATASET GENERATOR")
    print(f"# aperture_x = {aperture_x*1e3:.2f} mm, "
          f"aperture_y = {aperture_y*1e3:.2f} mm  (recon side = {L*1e3:.2f} mm)")
    print(f"# overlaps   = {OVERLAPS}")
    print(f"# rotations  = {'on' if ROTATE_SCANS else 'off'}  (angles = "
          f"{[k*90 for k in rotation_indices]} deg)")
    print(f"# output     = {OUTPUT_DIR}")
    print(f"{'#'*70}")

    total_pairs = len(OVERLAPS) * N_PAIRS_PER_OVERLAP
    pair_counter = 0

    for i, overlap in enumerate(OVERLAPS):
        shift = aperture_x * (1.0 - overlap)
        grain_width_x = L + shift + 2 * margin
        grain_width_y = L + 2 * margin

        overlap_dir = OUTPUT_DIR / f"ovlp_{int(round(overlap * 100)):03d}"
        overlap_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n── overlap {i+1}/{len(OVERLAPS)} = {overlap:.2f}  "
              f"shift={shift*1e3:.2f} mm  "
              f"grain=({grain_width_x*1e3:.1f}×{grain_width_y*1e3:.1f} mm) ──")

        for j in range(N_PAIRS_PER_OVERLAP):
            pair_counter += 1
            seed = BASE_SEED + i * N_PAIRS_PER_OVERLAP + j
            pair_dir = overlap_dir / f"pair_{j:02d}"
            pair_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n  pair {pair_counter}/{total_pairs}  "
                  f"(overlap={overlap:.2f}, seed={seed}) → {pair_dir.name}")
            t0 = time.time()

            rotated_cfgs = {
                k: build_config(_rotate_array_cfg(array_cfg, k),
                                grain_width_x, grain_width_y)
                for k in rotation_indices
            }
            scan_tags = [('A', -shift / 2), ('B', +shift / 2)]
            tag_index = {tag: n for n, (tag, _) in enumerate(scan_tags)}
            same_noise_for_ab = np.isclose(shift, 0.0)
            rotation_noise_offset = {
                k: seed_offset
                for seed_offset, k in enumerate(rotation_indices)
            }

            def noise_seed_for(tag: str, k: int) -> int | None:
                if not DETERMINISTIC_NOISE:
                    return None
                tag_offset = 0 if same_noise_for_ab else tag_index[tag] * 100
                return int(seed * 1000 + tag_offset + rotation_noise_offset[k])

            grain = generate_grain_structure(
                thickness=SPECIMEN_THICKNESS_MM * 1e-3,
                width=grain_width_x,
                depth=grain_width_y,
                background_material=MATERIAL,
                mean_grain_size_m=GRAIN_SIZE_MM * 1e-3,
                voxel_size_m=GRAIN_VOXEL_MM * 1e-3,
                impedance_variation=GRAIN_IMP_VAR,
                seed=seed,
            )
            grain.origin_x = -grain_width_x / 2
            grain.origin_y = -grain_width_y / 2

            # np.savez_compressed(
            #     pair_dir / 'grain_volume.npz',
            #     impedance=grain.impedance,
            #     voxel_size=np.float64(grain.voxel_size),
            #     origin_z=np.float64(grain.origin_z),
            #     origin_y=np.float64(grain.origin_y),
            #     origin_x=np.float64(grain.origin_x),
            # )

            scan_diagnostics = []
            for tag, dx in scan_tags:
                shifted = VoxelVolume3D(
                    impedance=grain.impedance,
                    wavespeed=grain.wavespeed,
                    voxel_size=grain.voxel_size,
                    origin_z=grain.origin_z,
                    origin_y=grain.origin_y,
                    origin_x=grain.origin_x + dx,
                )
                for k in rotation_indices:
                    scan_diagnostics.append(
                        scan_one(rotated_cfgs[k], shifted, f"{tag}_r{k}",
                                 pair_dir, x_half, y_half,
                                 noise_seed=noise_seed_for(tag, k))
                    )

            meta = {
                'overlap_fraction': float(overlap),
                'pair_index': int(j),
                'shift_m': float(shift),
                'aperture_x_m': float(array_cfg.aperture_x),
                'aperture_y_m': float(array_cfg.aperture_y),
                'recon_half_extent_m': float(x_half),
                'grain_width_x_m': float(grain_width_x),
                'grain_width_y_m': float(grain_width_y),
                'seed': int(seed),
                'rotate_scans': bool(ROTATE_SCANS),
                'n_rotations': int(len(rotation_indices)),
                'rotation_angles_deg': [k * 90 for k in rotation_indices],
                'deterministic_noise': bool(DETERMINISTIC_NOISE),
                'inter_volume_noise_scale': float(INTER_VOLUME_NOISE_SCALE),
                'tfm_pixels': [Z_PIXELS, Y_PIXELS, X_PIXELS] if RUN_TFM else None,
                'tfm_z_range_m': [Z_MIN_MM * 1e-3, Z_MAX_MM * 1e-3] if RUN_TFM else None,
                'scan_diagnostics': scan_diagnostics,
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
