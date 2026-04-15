#!/usr/bin/env python3
"""
Compare TFM B-scans generated under four different physics configurations
to isolate the contribution of each signal component.

Variants rendered at θ = 0°:
    1. grain_only         — grain Voronoi volume, no defect
    2. defect_kirchhoff   — geometric spheres only, no walls, no grain
    3. all_contributions  — walls + mode conv + skip/corner + Kirchhoff + Born
    4. born_only_defect   — defect burned into a uniform voxel volume
                            (Born scattering only, no Kirchhoff, no grain)

Output: SYNTHETIC DATA/output/compare/model_comparison.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from engine.config import (
    SimulationConfig, SpecimenConfig, ArrayConfig, AcquisitionConfig,
)
from engine.geometry import Specimen3D, SphericalDefect
from engine.fmc_engine import FMCEngine
from engine.materials import ALUMINUM
from engine.microstructure import generate_grain_structure, embed_geometric_defects

from run_engine import add_noise, apply_bandpass_filter, reconstruct_tfm


# ---------------------------------------------------------------------------
# Shared scene
# ---------------------------------------------------------------------------
SPECIMEN = Specimen3D(thickness=50e-3, width=50e-3, depth=50e-3)
SPHERES_3D = [
    SphericalDefect(center_x=-10e-3, center_y=-5e-3, center_z=20e-3, radius=2e-3),
    SphericalDefect(center_x=+12e-3, center_y=+7e-3, center_z=32e-3, radius=2e-3),
]
THETA = 0.0
FREQUENCY = 10e6


def make_array_config() -> ArrayConfig:
    return ArrayConfig(
        num_elements=128,
        element_pitch=0.3e-3,
        frequency=FREQUENCY,
        bandwidth=0.6,
        elevation_aperture=16e-3,   # must exceed 2·max|center_y| of the spheres
        n_elevation_slices=7,
    )


def base_sim_config(
    *,
    wall_echoes: bool,
    max_bounces: int,
    mode_conversion: bool,
    snr_db: float = 200.0,
    grain_noise_level: float = 0.0,
) -> SimulationConfig:
    return SimulationConfig(
        specimen=SpecimenConfig(thickness=SPECIMEN.thickness, width=SPECIMEN.width),
        array=make_array_config(),
        scan_plan=None,
        wall_echoes=wall_echoes,
        max_bounces=max_bounces,
        mode_conversion=mode_conversion,
        acquisition=AcquisitionConfig(
            snr_db=snr_db,
            grain_noise_level=grain_noise_level,
            filter_alpha=1.0,
            hanning_bool=False,
        ),
        gel_thickness=0.075e-3,
    )


# ---------------------------------------------------------------------------
# Single-frame runner
# ---------------------------------------------------------------------------
def run_variant(
    name: str,
    cfg: SimulationConfig,
    kirchhoff_spheres_3d: list,
    voxel_volume,
    tfm_z_start: float = 10e-3,
    tfm_z_end: float = 45e-3,
    tfm_n_pixels: int = 400,
) -> tuple:
    """Run one FMC → filter → TFM pipeline at θ=0° and return (img_db, x, z)."""
    print(f"\n==== {name} ====")
    engine = FMCEngine(cfg)

    # 2D slices of geometric defects at θ=0° (optionally with slab integration)
    aperture = cfg.array.elevation_aperture or cfg.array.element_height
    if aperture and cfg.array.n_elevation_slices > 1:
        dy_offsets = np.linspace(-aperture / 2, aperture / 2,
                                 cfg.array.n_elevation_slices)
        slab_weight = 1.0 / len(dy_offsets)
    else:
        dy_offsets = [0.0]
        slab_weight = 1.0

    for sph in kirchhoff_spheres_3d:
        for dy in dy_offsets:
            d2 = sph.slice_at_angle(THETA, dy_offset=float(dy))
            if d2 is not None:
                engine.add_defect(d2, amplitude_scale=slab_weight)

    # Born scatterers from voxel volume
    if voxel_volume is not None:
        half_w = cfg.array.aperture / 2
        gate_z = ALUMINUM.c_L * 2e-6 / 2
        z_start = max(gate_z * 1.2, 1e-3)
        step = voxel_volume.voxel_size
        z_grid = np.linspace(z_start, SPECIMEN.thickness,
                             max(2, int((SPECIMEN.thickness - z_start) / step) + 1))
        l_grid = np.linspace(-half_w, half_w,
                             max(2, int(cfg.array.aperture / step) + 1))
        z_s, x_s, amp_s = voxel_volume.extract_born_scatterers(
            THETA, z_grid, l_grid,
            background_Z=ALUMINUM.Z_L,
            threshold=0.005,
            elevation_aperture=aperture,
            n_slices=cfg.array.n_elevation_slices,
        )
        if len(z_s) > 0:
            # De-grid jitter to kill vertical streak artifacts
            dz = z_grid[1] - z_grid[0]
            dl = l_grid[1] - l_grid[0]
            rng = np.random.default_rng(seed=0)
            z_s = z_s + rng.uniform(-dz / 2, dz / 2, size=z_s.shape)
            x_s = x_s + rng.uniform(-dl / 2, dl / 2, size=x_s.shape)
            engine.set_born_scatterers(z_s, x_s, amp_s)

    result = engine.simulate()
    fmc = result['fmc_data']
    time_axis = result['time_axis']
    elem_x = result['element_positions']

    # Front-wall gate (identical to scan_volume_3d)
    gate_samples = int(2e-6 / cfg.dt)
    fmc[:, :, :gate_samples] = 0.0

    fmc = add_noise(fmc,
                    snr_db=cfg.acquisition.snr_db,
                    grain_noise_level=cfg.acquisition.grain_noise_level)
    fmc = apply_bandpass_filter(fmc, cfg.dt, cfg.array.frequency,
                                 bandwidth_fraction=cfg.array.bandwidth,
                                 filter_alpha=cfg.acquisition.filter_alpha,
                                 hanning_bool=cfg.acquisition.hanning_bool)

    half_w = cfg.array.aperture / 2
    img, x_img, z_img = reconstruct_tfm(
        fmc, time_axis, elem_x, ALUMINUM.c_L,
        x_range=(-half_w, half_w),
        z_range=(tfm_z_start, tfm_z_end),
        n_pixels=tfm_n_pixels,
    )

    # run_engine.reconstruct_tfm returns complex when img_output='complex';
    # envelope → dB normalised to the panel's peak.
    env = np.abs(img)
    img_db = 20 * np.log10(env / (env.max() + 1e-10) + 1e-10)
    return img_db, x_img, z_img


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # ---- Build the two voxel volumes once ----
    wavelength = ALUMINUM.c_L / FREQUENCY
    voxel_size = wavelength / (3 * 2 ** (1 / 3))  # 2x total voxel count

    print("Building grain voxel volume ...")
    grain_vol = generate_grain_structure(
        thickness=SPECIMEN.thickness,
        width=SPECIMEN.width,
        depth=SPECIMEN.depth,
        background_material=ALUMINUM,
        mean_grain_size_m=0.5e-3,
        impedance_variation=0.025,
        wavespeed_variation=0.005,
        voxel_size_m=voxel_size,
    )
    grain_vol_grain_only    = embed_geometric_defects(grain_vol, [])
    grain_vol_with_defects  = embed_geometric_defects(grain_vol, SPHERES_3D)

    print("Building uniform voxel volume (defect-as-Born-only) ...")
    uniform_vol_base = generate_grain_structure(
        thickness=SPECIMEN.thickness,
        width=SPECIMEN.width,
        depth=SPECIMEN.depth,
        background_material=ALUMINUM,
        mean_grain_size_m=0.5e-3,
        impedance_variation=0.0,
        wavespeed_variation=0.0,
        voxel_size_m=voxel_size,
    )
    uniform_vol_with_defects = embed_geometric_defects(uniform_vol_base, SPHERES_3D)

    # ---- Variant configurations ----
    variants = []

    # 1. Grain only — no defects anywhere
    cfg = base_sim_config(wall_echoes=False, max_bounces=1, mode_conversion=False)
    variants.append(('grain_only',
                     'walls OFF · defect OFF · grain Born ON',
                     cfg, [], grain_vol_grain_only))

    # 2. Defect-only Kirchhoff — geometric spheres, no walls, no grain
    cfg = base_sim_config(wall_echoes=False, max_bounces=1, mode_conversion=False)
    variants.append(('defect_kirchhoff',
                     'walls OFF · Kirchhoff spheres · no grain',
                     cfg, SPHERES_3D, None))

    # 3. All contributions — everything on
    cfg = base_sim_config(wall_echoes=True, max_bounces=3, mode_conversion=True)
    variants.append(('all_contributions',
                     'walls ON · mode conv · bounces=3 · Kirchhoff + grain',
                     cfg, SPHERES_3D, grain_vol_with_defects))

    # 4. Born-only defect — defect burned into a uniform voxel volume,
    #    NO Kirchhoff defect, NO grain, NO walls
    cfg = base_sim_config(wall_echoes=False, max_bounces=1, mode_conversion=False)
    variants.append(('born_only_defect',
                     'walls OFF · no Kirchhoff · Born defect only',
                     cfg, [], uniform_vol_with_defects))

    # ---- Run all variants ----
    results = []
    for name, subtitle, cfg, kdefs, vol in variants:
        img_db, x, z = run_variant(name, cfg, kdefs, vol)
        results.append((name, subtitle, img_db, x, z))

    # ---- Figure: 1×4 grid ----
    fig, axes = plt.subplots(1, 4, figsize=(22, 5.5))
    for ax, (name, subtitle, img_db, x, z) in zip(axes, results):
        extent = [x[0] * 1e3, x[-1] * 1e3, z[-1] * 1e3, z[0] * 1e3]
        im = ax.imshow(img_db, extent=extent, aspect='auto',
                       cmap='hot', vmin=-20, vmax=0)
        for sph in SPHERES_3D:
            ax.plot(sph.center_x * 1e3, sph.center_z * 1e3,
                    'co', markersize=10, markerfacecolor='none', linewidth=1.5)
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Depth (mm)')
        ax.set_title(f'{name}\n{subtitle}', fontsize=9)
        plt.colorbar(im, ax=ax, label='dB', shrink=0.8)

    fig.suptitle('Physics-contribution comparison — TFM B-scan at θ = 0°',
                 fontsize=12)
    plt.tight_layout(rect=(0, 0, 1, 0.95))

    out_dir = os.path.join(os.path.dirname(__file__), 'output', 'compare')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'model_comparison.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved → {out_path}")


if __name__ == '__main__':
    main()
