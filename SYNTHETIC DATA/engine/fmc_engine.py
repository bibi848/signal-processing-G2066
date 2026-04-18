"""
Born-only FMC acquisition simulation engine.

Single physical primitive: every contribution is a Born point scatterer
(z, x, amplitude). Defects, grain noise, and any other content are reduced
to scatterer triples and rendered through one vectorised loop. Single
longitudinal mode at c_L. No walls, no Kirchhoff, no mode conversion.
"""

import numpy as np
import time as time_module
import sys
import os
from typing import List, Optional

# Add parent directory so we can import Classes/TFM1D
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from Classes.TFM1D import CTFM1D  # noqa: F401  (re-exported for callers)

from .config import SimulationConfig
from .geometry import Specimen2D, Defect2D
from .propagation import element_directivity_array


class FMCEngine:
    """
    Born-only physics-based FMC simulator.

    Sources of scatterers, all combined into a single (z, x, amp) cloud:
      1. Arrays passed to set_born_scatterers(...)  (e.g. voxel grain extract)
      2. Each registered defect's .to_born_scatterers()
    """

    def __init__(self, config: SimulationConfig):
        self.cfg = config
        self.specimen = Specimen2D(
            thickness=config.specimen.thickness,
            width=config.specimen.width,
            front_wall_z=config.specimen.front_wall_z,
        )
        self.defects: List[Defect2D] = []
        self._defect_scales: List[float] = []

        # Scatterers set via set_born_scatterers()
        self._born_z:   Optional[np.ndarray] = None
        self._born_x:   Optional[np.ndarray] = None
        self._born_amp: Optional[np.ndarray] = None

    def add_defect(self, defect: Defect2D, amplitude_scale: float = 1.0):
        """Register a 2D defect. Its .to_born_scatterers() output is scaled
        by amplitude_scale before being merged into the global cloud
        (used for elevation slab averaging)."""
        self.defects.append(defect)
        self._defect_scales.append(float(amplitude_scale))

    def set_born_scatterers(self,
                             z_s: np.ndarray,
                             x_s: np.ndarray,
                             amp_s: np.ndarray) -> None:
        """Register externally-derived Born scatterers (e.g. voxel grain extract)."""
        self._born_z   = np.asarray(z_s,   dtype=np.float64)
        self._born_x   = np.asarray(x_s,   dtype=np.float64)
        self._born_amp = np.asarray(amp_s, dtype=np.float64)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def simulate(self) -> dict:
        cfg = self.cfg
        num_el = cfg.array.num_elements
        n_t = cfg.acquisition.time_samples
        time_axis = cfg.time_axis
        elem_x = cfg.array.element_positions

        fmc_data = np.zeros((num_el, num_el, n_t), dtype=np.float32)

        # Collect scatterers from defects + external arrays into one cloud.
        z_parts, x_parts, a_parts = [], [], []
        if self._born_z is not None and len(self._born_z) > 0:
            z_parts.append(self._born_z)
            x_parts.append(self._born_x)
            a_parts.append(self._born_amp)

        for defect, scale in zip(self.defects, self._defect_scales):
            z_d, x_d, a_d = defect.to_born_scatterers()
            if len(z_d) == 0:
                continue
            z_parts.append(np.asarray(z_d, dtype=np.float64))
            x_parts.append(np.asarray(x_d, dtype=np.float64))
            a_parts.append(np.asarray(a_d, dtype=np.float64) * scale)

        n_defect_pts = sum(len(d.to_born_scatterers()[0]) for d in self.defects) \
            if self.defects else 0
        n_external = len(self._born_z) if self._born_z is not None else 0
        print(cfg.summary())
        print(f"  Defects: {len(self.defects)} ({n_defect_pts} surface pts)  "
              f"|  External Born scatterers: {n_external}")
        print(f"  Simulating Born FMC acquisition...")

        t_start = time_module.time()

        if z_parts:
            z_all = np.concatenate(z_parts)
            x_all = np.concatenate(x_parts)
            a_all = np.concatenate(a_parts)
            self._compute_born_scattering(fmc_data, elem_x, time_axis,
                                          z_all, x_all, a_all)

        elapsed = time_module.time() - t_start
        print(f"  FMC simulation complete: {elapsed:.1f}s")
        print(f"  FMC shape: {fmc_data.shape}")
        print(f"  Signal range: [{fmc_data.min():.2e}, {fmc_data.max():.2e}]")

        return {
            'fmc_data': fmc_data,
            'time_axis': time_axis,
            'element_positions': elem_x,
            'config': cfg,
        }

    # ------------------------------------------------------------------
    # Born scattering — the only forward-modelling path
    # ------------------------------------------------------------------

    def _compute_born_scattering(self,
                                  fmc_data: np.ndarray,
                                  elem_x: np.ndarray,
                                  time_axis: np.ndarray,
                                  born_z: np.ndarray,
                                  born_x: np.ndarray,
                                  born_amp: np.ndarray) -> None:
        """
        Vectorised Born-approximation rendering.

        For every (tx, rx, scatterer) triple compute the arrival sample,
        scatter-add weighted impulses into a per-pair impulse train, then
        batch-convolve with a Gabor wavelet kernel.

        Includes geometric spreading (1/√r per leg), material attenuation,
        and element directivity. All in float32.
        """
        from scipy.ndimage import convolve1d

        cfg = self.cfg
        assert cfg.material is not None
        c_L  = cfg.material.c_L
        freq = cfg.array.frequency
        bw   = cfg.array.bandwidth
        sigma = 1.0 / (np.pi * freq * bw)

        n_el = len(elem_x)
        n_t  = len(time_axis)
        n_s  = len(born_z)
        dt   = float(time_axis[1] - time_axis[0])
        alpha_np_per_m = float(cfg.material.attenuation_L * (freq / 1e6))

        # Gabor kernel centred at τ = 0
        n_half  = int(np.ceil(5.0 * sigma / dt))
        tau     = np.arange(-n_half, n_half + 1) * dt
        gabor_k = (np.exp(-0.5 * (tau / sigma) ** 2)
                   * np.cos(2.0 * np.pi * freq * tau)).astype(np.float32)

        born_z32 = born_z.astype(np.float32, copy=False)
        born_x32 = born_x.astype(np.float32, copy=False)
        born_a32 = born_amp.astype(np.float32, copy=False)

        elem_x32 = elem_x.astype(np.float32, copy=False)
        dz = born_z32[np.newaxis, :]
        dx = born_x32[np.newaxis, :] - elem_x32[:, np.newaxis]
        r  = np.sqrt(dz ** 2 + dx ** 2).astype(np.float32)

        theta_el = np.arctan2(np.abs(dx), np.abs(dz))
        wavelength = c_L / freq
        dir_factor = element_directivity_array(
            theta_el, cfg.array.element_width, wavelength
        ).astype(np.float32)

        r_tx = r[:, np.newaxis, :]
        r_rx = r[np.newaxis, :, :]

        total_dist = (r_tx + r_rx).astype(np.float32)
        tof = total_dist * np.float32(1.0 / c_L)

        spread = np.float32(1.0) / np.sqrt(
            np.maximum(r_tx * r_rx, np.float32(1e-20))
        )
        directivity = (dir_factor[:, np.newaxis, :]
                        * dir_factor[np.newaxis, :, :])
        atten = np.exp((-alpha_np_per_m) * total_dist, dtype=np.float32)
        spread *= directivity
        spread *= atten
        del directivity, atten
        amp = born_a32[np.newaxis, np.newaxis, :] * spread
        del spread

        tof_samp = np.round(tof / dt).astype(np.int32)
        valid    = (tof_samp >= 0) & (tof_samp < n_t)

        n_pairs  = n_el * n_el
        f_flat   = np.zeros((n_pairs, n_t), dtype=np.float32)

        pair_idx = np.broadcast_to(
            np.arange(n_pairs, dtype=np.int32).reshape(n_el, n_el, 1),
            (n_el, n_el, n_s),
        ).ravel()
        t_idx = tof_samp.clip(0, n_t - 1).ravel()
        a_val = amp.ravel()
        v     = valid.ravel()

        np.add.at(f_flat, (pair_idx[v], t_idx[v]), a_val[v])

        result = convolve1d(f_flat, gabor_k, axis=1,
                            mode='constant', cval=0.0)

        fmc_data += result.reshape(n_el, n_el, n_t)
