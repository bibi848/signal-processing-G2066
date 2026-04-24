#!/usr/bin/env python3
"""
Generate two grain-only 2D TFM B-scans in the same scan plane (θ = 0°),
with a configurable lateral overlap fraction — for testing 2D stitching.

    OVERLAP_FRACTION = 1.0 → identical window
    OVERLAP_FRACTION = 0.0 → adjacent, non-overlapping tiles
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from engine.config import (
    SimulationConfig, SpecimenConfig, ArrayConfig, AcquisitionConfig,
)
from engine.geometry import Specimen3D
from engine.fmc_engine import FMCEngine
from engine.materials import ALUMINUM
from engine.microstructure import generate_grain_structure
from engine.voxel_volume import VoxelVolume3D

from run_engine import add_noise, apply_bandpass_filter, reconstruct_tfm


# ---- User-tunable parameters ------------------------------------------------
OVERLAP_FRACTION = 0.99
N_PAIRS             = 1

FREQUENCY           = 10e6
NUM_ELEMENTS        = 128
ELEMENT_PITCH       = 0.3e-3
BANDWIDTH           = 0.8
ELEVATION_APERTURE  = 16e-3
N_ELEVATION_SLICES  = 7

SPECIMEN_THICKNESS  = 50e-3
SPECIMEN_WIDTH      = 200e-3
SPECIMEN_DEPTH      = 200e-3

MEAN_GRAIN_SIZE     = 0.5e-3
IMPEDANCE_VARIATION = 0.025
WAVESPEED_VARIATION = 0.005

SNR_DB              = 200.0

TFM_Z_RANGE         = (10e-3, 45e-3)
TFM_N_PIXELS        = 400
# -----------------------------------------------------------------------------


def main():
    cfg = SimulationConfig(
        specimen=SpecimenConfig(thickness=SPECIMEN_THICKNESS, width=SPECIMEN_WIDTH),
        array=ArrayConfig(
            num_elements=NUM_ELEMENTS,
            element_pitch=ELEMENT_PITCH,
            frequency=FREQUENCY,
            bandwidth=BANDWIDTH,
            elevation_aperture=ELEVATION_APERTURE,
            n_elevation_slices=N_ELEVATION_SLICES,
        ),
        scan_plan=None,
        wall_echoes=False,
        max_bounces=1,
        mode_conversion=False,
        acquisition=AcquisitionConfig(
            snr_db=SNR_DB, grain_noise_level=0.0,
            filter_alpha=1.0, hanning_bool=False,
        ),
        gel_thickness=0.075e-3,
    )
    aperture = cfg.array.aperture
    half_w = aperture / 2
    shift = (1.0 - OVERLAP_FRACTION) * aperture

    # Grain volume wide enough to cover both shifted windows
    voxel_size = (ALUMINUM.c_L / FREQUENCY) / (3 * 2 ** (1 / 3))
    volume_width = aperture + 2 * abs(shift) + 4e-3
    print(f"overlap = {OVERLAP_FRACTION:.2f}, shift = {shift*1e3:.2f} mm, "
          f"grain volume width = {volume_width*1e3:.2f} mm")

    out_dir = os.path.join(os.path.dirname(__file__), 'output', 'compare')
    os.makedirs(out_dir, exist_ok=True)

    for pair_idx in range(N_PAIRS):
        print(f"\n######## pair {pair_idx} ########")
        grain_vol = generate_grain_structure(
            thickness=SPECIMEN_THICKNESS, width=volume_width, depth=SPECIMEN_DEPTH,
            background_material=ALUMINUM,
            mean_grain_size_m=MEAN_GRAIN_SIZE,
            impedance_variation=IMPEDANCE_VARIATION,
            wavespeed_variation=WAVESPEED_VARIATION,
            voxel_size_m=voxel_size,
            seed=1000 + pair_idx,
        )
        # Centre the grain volume on x = 0 so the scan window [-half_w, +half_w]
        # lies fully inside the volume at every shift.
        grain_vol.origin_x = -volume_width / 2

        for tag, origin_shift in [('A', 0.0), ('B', -shift)]:
            print(f"\n==== pair {pair_idx} scan {tag} ====")
            vol = VoxelVolume3D(
                impedance=grain_vol.impedance, wavespeed=grain_vol.wavespeed,
                voxel_size=grain_vol.voxel_size,
                origin_z=grain_vol.origin_z, origin_y=grain_vol.origin_y,
                origin_x=grain_vol.origin_x + origin_shift,
            )
            engine = FMCEngine(cfg)

            step = vol.voxel_size
            gate_z = ALUMINUM.c_L * 2e-6 / 2
            z_start = max(gate_z * 1.2, 1e-3)
            z_grid = np.linspace(z_start, SPECIMEN_THICKNESS,
                                 max(2, int((SPECIMEN_THICKNESS - z_start) / step) + 1))
            l_grid = np.linspace(-half_w, half_w,
                                 max(2, int(aperture / step) + 1))
            z_s, x_s, amp_s = vol.extract_born_scatterers(
                0.0, z_grid, l_grid,
                background_Z=ALUMINUM.Z_L, threshold=0.005,
                elevation_aperture=ELEVATION_APERTURE,
                n_slices=N_ELEVATION_SLICES,
            )
            if len(z_s) > 0:
                dz = z_grid[1] - z_grid[0]
                dl = l_grid[1] - l_grid[0]
                rng = np.random.default_rng(seed=pair_idx)
                z_s = z_s + rng.uniform(-dz / 2, dz / 2, size=z_s.shape)
                x_s = x_s + rng.uniform(-dl / 2, dl / 2, size=x_s.shape)
                engine.set_born_scatterers(z_s, x_s, amp_s)

            result = engine.simulate()
            fmc = result['fmc_data']
            time_axis = result['time_axis']
            elem_x = result['element_positions']

            gate_samples = int(2e-6 / cfg.dt)
            fmc[:, :, :gate_samples] = 0.0
            fmc = add_noise(fmc, snr_db=SNR_DB, grain_noise_level=0.0)
            fmc = apply_bandpass_filter(fmc, cfg.dt, FREQUENCY,
                                        bandwidth_fraction=BANDWIDTH,
                                        filter_alpha=1.0, hanning_bool=False)

            img, x_img, z_img = reconstruct_tfm(
                fmc, time_axis, elem_x, ALUMINUM.c_L,
                x_range=(-half_w, half_w), z_range=TFM_Z_RANGE,
                n_pixels=TFM_N_PIXELS,
            )
            env = np.abs(img)
            img_db = 20 * np.log10(env / (env.max() + 1e-10) + 1e-10)

            out_path = os.path.join(out_dir,
                                    f'grain_overlap_{pair_idx:02d}_{tag}.png')
            plt.imsave(out_path, img_db, cmap='hot', vmin=-20, vmax=0)
            print(f"Saved → {out_path}")


if __name__ == '__main__':
    main()
