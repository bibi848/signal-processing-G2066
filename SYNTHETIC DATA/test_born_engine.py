#!/usr/bin/env python3
"""
Side-by-side sanity test for the Born-only engine.

Builds two single-frame B-scans:
  (A) defect only  — one geometric defect, no grain noise
  (B) grains only  — Voronoi grain volume, no defects

Both go through the same FMC → noise → bandpass → TFM pipeline so the
two TFM images are directly comparable. Parameters at the top of main().

Usage:
    python test_born_engine.py
"""

from __future__ import annotations

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from engine.config import SimulationConfig, ArrayConfig, SpecimenConfig
from engine.fmc_engine import FMCEngine
from engine.geometry import SphericalDefect, CylindricalDefect
from engine.materials import ALUMINUM
from engine.microstructure import generate_grain_structure
from engine.voxel_volume import VoxelVolume3D

# Reuse noise / filter / TFM helpers from the main runner.
from run_engine import add_noise, apply_bandpass_filter, reconstruct_tfm


# --------------------------------------------------------------------------
# Per-case simulation
# --------------------------------------------------------------------------

def simulate_case(
    cfg: SimulationConfig,
    *,
    defect=None,
    voxel_volume: VoxelVolume3D | None = None,
    born_threshold: float = 0.005,
    tfm_z_range: tuple = (5e-3, 45e-3),
    tfm_n_pixels: int = 400,
    label: str = "",
) -> tuple:
    """Run one B-scan: optional defect + optional grain volume, return TFM image."""
    print(f"\n--- Case: {label} ---")

    engine = FMCEngine(cfg)
    if defect is not None:
        d2 = defect.slice_at_angle(0.0)
        if d2 is None:
            raise ValueError("Defect does not intersect the scan plane at θ=0")
        engine.add_defect(d2)

    if voxel_volume is not None:
        half_w = cfg.array.aperture / 2
        z_grid = np.linspace(1e-3, cfg.specimen.thickness,
                             max(2, int(cfg.specimen.thickness / voxel_volume.voxel_size) + 1))
        l_grid = np.linspace(-half_w, half_w,
                             max(2, int(cfg.array.aperture / voxel_volume.voxel_size) + 1))
        z_s, x_s, amp_s = voxel_volume.extract_born_scatterers(
            theta=0.0,
            z_grid=z_grid,
            lateral_grid=l_grid,
            background_Z=cfg.material.Z_L,
            threshold=born_threshold,
        )
        print(f"  Born scatterers extracted: {len(z_s)}")
        if len(z_s) > 0:
            engine.set_born_scatterers(z_s, x_s, amp_s)

    result = engine.simulate()
    fmc = result["fmc_data"]
    time_axis = result["time_axis"]
    elem_x = result["element_positions"]

    # Same post-processing as run_engine.scan_volume_3d
    gate_samples = int(2e-6 / cfg.dt)
    fmc[:, :, :gate_samples] = 0.0
    fmc = add_noise(fmc, snr_db=cfg.acquisition.snr_db,
                    grain_noise_level=cfg.acquisition.grain_noise_level)
    fmc = apply_bandpass_filter(fmc, cfg.dt, cfg.array.frequency,
                                bandwidth_fraction=cfg.array.bandwidth,
                                filter_alpha=cfg.acquisition.filter_alpha,
                                hanning_bool=cfg.acquisition.hanning_bool)

    half_w = cfg.array.aperture / 2
    img, x_img, z_img = reconstruct_tfm(
        fmc, time_axis, elem_x, cfg.material.c_L,
        x_range=(-half_w, half_w),
        z_range=tfm_z_range,
        n_pixels=tfm_n_pixels,
    )
    return img, x_img, z_img


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------

def to_db(img: np.ndarray) -> np.ndarray:
    """Convert (real or complex) TFM image to dB envelope normalised to its own peak."""
    if np.iscomplexobj(img):
        env = np.abs(img)
    else:
        from scipy.signal import hilbert
        env = np.abs(hilbert(img, axis=0))
    peak = env.max() + 1e-30
    return 20 * np.log10(env / peak + 1e-12)


def plot_side_by_side(
    img_a, img_b, x_img, z_img,
    defect_marker_xz: tuple | None,
    output_path: str,
    db_range: float = -30.0,
):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    extent = [x_img[0] * 1e3, x_img[-1] * 1e3,
              z_img[-1] * 1e3, z_img[0] * 1e3]

    for ax, img, title in [
        (axes[0], img_a, "Defect only (no grains)"),
        (axes[1], img_b, "Grains only (no defect)"),
    ]:
        im = ax.imshow(to_db(img), extent=extent, aspect="auto",
                       cmap="hot", vmin=db_range, vmax=0)
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Depth (mm)")
        ax.set_title(title)
        plt.colorbar(im, ax=ax, label="dB", shrink=0.85)

    if defect_marker_xz is not None:
        x_mm, z_mm = defect_marker_xz
        axes[0].plot(x_mm, z_mm, "co", markersize=12,
                     markerfacecolor="none", linewidth=2, label="True defect")
        axes[0].legend(loc="upper right", fontsize=9)

    fig.suptitle("Born-only engine — defect vs. grain noise", fontsize=12)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved comparison: {output_path}")


# --------------------------------------------------------------------------
# Main — knobs at the top
# --------------------------------------------------------------------------

def main() -> None:
    # ---- Array parameters ----
    NUM_ELEMENTS    = 128
    ELEMENT_PITCH   = 0.3e-3
    FREQUENCY       = 10e6           # Hz
    BANDWIDTH       = 0.8           # fractional

    # ---- Specimen parameters ----
    THICKNESS       = 50e-3
    WIDTH           = 50e-3
    DEPTH           = 30e-3         # elevation extent (only used for grain volume)

    # ---- Defect parameters ----
    DEFECT = SphericalDefect(
        center_z=25e-3, center_x=5.0e-3, center_y=0.0, radius=0.5e-3,
    )
    # Try also: CylindricalDefect(center_z=15e-3, center_x=8e-3, radius=1e-3,
    #                              y_start=-DEPTH/2, y_end=DEPTH/2)

    # ---- Grain parameters ----
    MEAN_GRAIN_SIZE     = 0.5e-3
    IMPEDANCE_VARIATION = 0.025      # ±2.5 %
    WAVESPEED_VARIATION = 0.005
    GRAIN_SEED          = 42

    # ---- TFM parameters ----
    TFM_N_PIXELS    = 800
    TFM_Z_RANGE     = (5e-3, THICKNESS - 5e-3)
    DB_RANGE        = -20.0
    BORN_THRESHOLD  = 0.005

    # ---- Build common config ----
    cfg = SimulationConfig(
        material=ALUMINUM,
        array=ArrayConfig(
            num_elements=NUM_ELEMENTS,
            element_pitch=ELEMENT_PITCH,
            frequency=FREQUENCY,
            bandwidth=BANDWIDTH,
        ),
        specimen=SpecimenConfig(thickness=THICKNESS, width=WIDTH),
    )

    # ---- Grain volume for case B ----
    wavelength = ALUMINUM.c_L / FREQUENCY
    voxel_size = wavelength / 5
    print(f"Generating grain volume (λ={wavelength*1e3:.2f} mm, "
          f"voxel={voxel_size*1e3:.2f} mm)...")
    grain_vol = generate_grain_structure(
        thickness=THICKNESS,
        width=WIDTH,
        depth=DEPTH,
        background_material=ALUMINUM,
        mean_grain_size_m=MEAN_GRAIN_SIZE,
        impedance_variation=IMPEDANCE_VARIATION,
        wavespeed_variation=WAVESPEED_VARIATION,
        voxel_size_m=voxel_size,
        seed=GRAIN_SEED,
    )

    # ---- Run both cases ----
    img_a, x_img, z_img = simulate_case(
        cfg, defect=DEFECT, voxel_volume=None,
        tfm_z_range=TFM_Z_RANGE, tfm_n_pixels=TFM_N_PIXELS,
        label="defect only",
    )
    img_b, _, _ = simulate_case(
        cfg, defect=None, voxel_volume=grain_vol,
        born_threshold=BORN_THRESHOLD,
        tfm_z_range=TFM_Z_RANGE, tfm_n_pixels=TFM_N_PIXELS,
        label="grains only",
    )

    # ---- Plot ----
    d2 = DEFECT.slice_at_angle(0.0)
    marker = (d2.center_x * 1e3, d2.center_z * 1e3) if d2 is not None else None
    out = os.path.join(HERE, "output", "born_engine_test.png")
    plot_side_by_side(img_a, img_b, x_img, z_img, marker, out, db_range=DB_RANGE)


if __name__ == "__main__":
    main()
