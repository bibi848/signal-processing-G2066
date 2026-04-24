"""Monotonic sensitivity sweep: defect radius vs peak FMC echo amplitude.

Drops a single CircularDefect at fixed (z, x), sweeps its radius, and for
each run takes the max envelope across all (tx, rx, t) samples within a
TOF window around the expected two-way travel time. A correct engine
should give a monotonically increasing curve.
"""
from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from engine.config import SimulationConfig, ArrayConfig, SpecimenConfig, AcquisitionConfig
from engine.fmc_engine import FMCEngine
from engine.geometry import DEFAULT_VOID_BORN_AMP
from engine.materials import ALUMINUM

DEFECT_Z       = 20e-3
DEFECT_X       = 0.0
RADII_MM       = np.array([0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0])
FREQUENCY_HZ   = 7.5e6
N_ELEMENTS     = 128
PITCH_M        = 0.6e-3
TIME_SAMPLES   = 2048
SAMPLING_HZ    = 50e6

cfg = SimulationConfig(
    material=ALUMINUM,
    array=ArrayConfig(num_elements=N_ELEMENTS, element_pitch=PITCH_M,
                      frequency=FREQUENCY_HZ, bandwidth=0.6),
    specimen=SpecimenConfig(thickness=40e-3, width=60e-3),
    acquisition=AcquisitionConfig(time_samples=TIME_SAMPLES,
                                  sampling_frequency=SAMPLING_HZ,
                                  add_noise=False),
)

expected_tof = 2.0 * DEFECT_Z / ALUMINUM.c_L
dt = cfg.dt
window_half = 2e-6  # ±2 µs around expected TOF
i_lo = max(0, int((expected_tof - window_half) / dt))
i_hi = min(TIME_SAMPLES, int((expected_tof + window_half) / dt))

# Constant arc-length sampling: one scatterer every λ/10 so discretization
# density (not count) is fixed across radii.
wavelength = ALUMINUM.c_L / FREQUENCY_HZ
arc_step = wavelength / 10.0

peaks = []
for r_mm in RADII_MM:
    r = r_mm * 1e-3
    n_points = max(8, int(np.ceil(2 * np.pi * r / arc_step)))
    angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    z_s = DEFECT_Z + r * np.cos(angles)
    x_s = DEFECT_X + r * np.sin(angles)
    amp_s = np.full(n_points, DEFAULT_VOID_BORN_AMP, dtype=np.float64)

    engine = FMCEngine(cfg)
    engine.set_born_scatterers(z_s, x_s, amp_s)
    fmc = engine.simulate()['fmc_data']
    env = np.abs(hilbert(fmc[:, :, i_lo:i_hi], axis=-1))
    peaks.append(env.max())
    print(f"  r = {r_mm:.2f} mm  (n_pts={n_points:4d})  →  peak envelope = {peaks[-1]:.3e}")

peaks = np.array(peaks)

fig, ax = plt.subplots(figsize=(7, 5))
ax.loglog(RADII_MM, peaks, "o-", label="engine")
ax.loglog(RADII_MM, peaks[-1] * (RADII_MM / RADII_MM[-1]) ** 1, "--", color="0.5",
          label="∝ r (reference)")
ax.loglog(RADII_MM, peaks[-1] * (RADII_MM / RADII_MM[-1]) ** 2, ":", color="0.5",
          label="∝ r² (reference)")
ax.set_xlabel("defect radius (mm)")
ax.set_ylabel("peak FMC envelope (a.u.)")
ax.set_title(f"Defect-size sensitivity at z={DEFECT_Z*1e3:.0f} mm, f={FREQUENCY_HZ/1e6:.0f} MHz")
ax.grid(True, which="both", alpha=0.3)
ax.legend()
plt.tight_layout()
plt.show()
