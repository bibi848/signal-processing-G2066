"""
Single off-centre defect → rotational FMC scan → inverse-Radon volume → napari.

Pipeline:
  1. Build a 3D specimen containing one SphericalDefect off-axis.
  2. Rotate the 1D array around its own centre over [0, 180) deg, taking
     an FMC + TFM B-scan at each angle.
  3. Run iradon per depth slice to backproject the B-scan stack into a
     3D volume (Classes.Reconstruct3D.reconstruct_scan).
  4. Open the reconstructed volume in napari.

Knobs at the top of main().
"""

from __future__ import annotations

import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SD_ROOT = os.path.dirname(HERE)
REPO_ROOT = os.path.dirname(SD_ROOT)
sys.path.insert(0, SD_ROOT)
sys.path.insert(0, REPO_ROOT)

from engine.config import (
    SimulationConfig, ArrayConfig, SpecimenConfig,
    AcquisitionConfig, ScanPlanConfig,
)
from engine.geometry import Specimen3D, SphericalDefect
from engine.materials import ALUMINUM

import run_engine
from run_engine import scan_volume_3d
from Classes.Reconstruct3D import (
    reconstruct_scan, compute_reconstruction_coords, view_reconstruction_napari,
    _load_meta,
)


def main() -> None:
    # ---- Array ----
    NUM_ELEMENTS  = 128
    ELEMENT_PITCH = 0.3e-3
    FREQUENCY     = 10e6
    BANDWIDTH     = 0.8

    # ---- Specimen ----
    THICKNESS = 50e-3
    WIDTH     = 50e-3
    DEPTH     = 50e-3      # elevation extent

    # ---- Defect (off-centre on purpose) ----
    DEFECT = SphericalDefect(
        center_z=25e-3,
        center_x=8e-3,
        center_y=5e-3,
        radius=1.0e-3,
    )

    # ---- Scan plan: [0°, 180°) — paper convention, endpoint excluded ----
    N_SCANS     = 60
    THETA_START = 0.0
    THETA_END   = np.pi          # exclusive — ScanPlanConfig uses endpoint=False

    # ---- TFM ----
    TFM_N_PIXELS = 400
    TFM_Z_START  = 5e-3
    TFM_Z_END    = THICKNESS - 5e-3

    # ---- Output ----
    OUTPUT_DIR = os.path.join(HERE, 'output', 'radon_volume_recon')

    # ---- Build cfg ----
    cfg = SimulationConfig(
        material=ALUMINUM,
        array=ArrayConfig(
            num_elements=NUM_ELEMENTS,
            element_pitch=ELEMENT_PITCH,
            frequency=FREQUENCY,
            bandwidth=BANDWIDTH,
        ),
        specimen=SpecimenConfig(thickness=THICKNESS, width=WIDTH),
        acquisition=AcquisitionConfig(snr_db=40.0, grain_noise_level=0.0),
    )
    scan_plan = ScanPlanConfig(
        n_scans=N_SCANS, theta_start=THETA_START, theta_end=THETA_END,
    )
    specimen = Specimen3D(thickness=THICKNESS, width=WIDTH, depth=DEPTH)

    # Tell run_engine to save complex B-scans (needed for proper Radon backprojection)
    run_engine.img_output = 'complex'

    print(cfg.summary())
    print(f"Defect: sphere r={DEFECT.radius*1e3:.1f} mm at "
          f"(x={DEFECT.center_x*1e3:+.1f}, y={DEFECT.center_y*1e3:+.1f}, "
          f"z={DEFECT.center_z*1e3:.1f}) mm")

    # ---- Acquire ----
    scan_volume_3d(
        specimen=specimen,
        defects_3d=[DEFECT],
        cfg=cfg,
        scan_plan=scan_plan,
        output_dir=OUTPUT_DIR,
        voxel_volume=None,
        tfm_z_start=TFM_Z_START,
        tfm_z_end=TFM_Z_END,
        tfm_n_pixels=TFM_N_PIXELS,
    )

    # ---- Reconstruct (complex), then view envelope in napari ----
    volume = reconstruct_scan(
        scan_dir=OUTPUT_DIR,
        show_napari=False,
    )
    envelope = np.abs(volume).astype(np.float32)
    meta = _load_meta(OUTPUT_DIR)
    z, y, x = compute_reconstruction_coords(meta, envelope.shape[1])
    view_reconstruction_napari(envelope, None, z, y, x)


if __name__ == '__main__':
    main()
