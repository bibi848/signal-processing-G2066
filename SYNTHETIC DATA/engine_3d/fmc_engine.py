"""
Born-only 3D FMC acquisition simulator for a 2D matrix phased array.

The forward model mirrors engine.fmc_engine.FMCEngine but lives in full 3D:
    - Scatterers are (z, x, y, amplitude) point clouds
    - Element positions live on a tensor-product (iy, ix) grid
    - Distance r = √(dz² + dx² + dy²)
    - Geometric spreading is 1/r per leg (spherical), not 1/√r
    - Element directivity is the product of two sinc patterns, sized by
      (element_width_x, element_width_y)

Memory strategy
---------------
A 32×32 matrix array has 1024 elements → 1024² = ~1M pairs. A naive
(n_tx, n_rx, n_scat) tensor is impossible, so this engine loops over tx
and vectorises (rx, scatterer) internally. For each tx chunk we still use
the impulse-train + convolve1d trick from the 2D engine — it generalises
verbatim.
"""

import time as time_module
from typing import List, Optional, Tuple
import numpy as np

from .config import SimulationConfig3D
from .geometry import (
    SphericalDefect,
    CylindricalDefect,
    PlanarCrack3D,
    defect_to_born_scatterers_3d,
)
from .propagation import element_directivity_3d_array


class FMCEngine3D:
    """
    Born-only 3D FMC simulator with a 2D matrix array.

    Scatterer sources, combined into a single (z, x, y, amp) cloud:
        1. Arrays passed to set_born_scatterers(...)
           (e.g. 3D voxel-gradient extract)
        2. Each registered 3D defect via defect_to_born_scatterers_3d(...)
    """

    def __init__(self, config: SimulationConfig3D):
        self.cfg = config
        self.defects: List = []
        self._defect_scales: List[float] = []
        self._defect_n_points: List[int] = []

        self._born_z:   Optional[np.ndarray] = None
        self._born_x:   Optional[np.ndarray] = None
        self._born_y:   Optional[np.ndarray] = None
        self._born_amp: Optional[np.ndarray] = None

    def add_defect(self, defect, amplitude_scale: float = 1.0,
                    n_points: int = 600):
        """Register a 3D defect. Its surface cloud is scaled by amplitude_scale."""
        if not isinstance(defect, (SphericalDefect, CylindricalDefect,
                                     PlanarCrack3D)):
            raise TypeError(
                f"FMCEngine3D only accepts 3D defects "
                f"(SphericalDefect/CylindricalDefect/PlanarCrack3D), "
                f"got {type(defect).__name__}"
            )
        self.defects.append(defect)
        self._defect_scales.append(float(amplitude_scale))
        self._defect_n_points.append(int(n_points))

    def set_born_scatterers(self,
                             z_s: np.ndarray,
                             x_s: np.ndarray,
                             y_s: np.ndarray,
                             amp_s: np.ndarray) -> None:
        """Register externally-derived 3D Born scatterers."""
        self._born_z   = np.asarray(z_s,   dtype=np.float64)
        self._born_x   = np.asarray(x_s,   dtype=np.float64)
        self._born_y   = np.asarray(y_s,   dtype=np.float64)
        self._born_amp = np.asarray(amp_s, dtype=np.float64)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def simulate(self, tx_chunk: int = 1, verbose: bool = True) -> dict:
        cfg = self.cfg
        n_el = cfg.array.n_elements_total
        n_t  = cfg.acquisition.time_samples
        time_axis = cfg.time_axis
        elem_xyz = cfg.array.element_positions()   # (n_el, 3) columns z, x, y

        fmc_data = np.zeros((n_el, n_el, n_t), dtype=np.float32)

        z_s, x_s, y_s, amp_s = self._gather_scatterers()
        n_scat = len(z_s)

        if verbose:
            print(cfg.summary())
            print(f"  Defects: {len(self.defects)}  |  "
                  f"External Born scatterers: "
                  f"{0 if self._born_z is None else len(self._born_z)}")
            print(f"  Total scatterers: {n_scat}")
            print(f"  Simulating 3D Born FMC acquisition...")

        if n_scat == 0:
            return self._pack_result(fmc_data, time_axis, elem_xyz)

        t_start = time_module.time()
        self._compute_born_scattering_3d(
            fmc_data, elem_xyz, time_axis,
            z_s, x_s, y_s, amp_s,
            tx_chunk=tx_chunk, verbose=verbose,
        )

        if verbose:
            elapsed = time_module.time() - t_start
            print(f"  FMC simulation complete: {elapsed:.1f}s")
            print(f"  FMC shape: {fmc_data.shape}")
            print(f"  Signal range: [{fmc_data.min():.2e}, {fmc_data.max():.2e}]")

        return self._pack_result(fmc_data, time_axis, elem_xyz)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _pack_result(self, fmc_data, time_axis, elem_xyz) -> dict:
        return {
            'fmc_data': fmc_data,
            'time_axis': time_axis,
            'element_positions_xyz': elem_xyz,
            'config': self.cfg,
        }

    def _gather_scatterers(self) -> Tuple[np.ndarray, np.ndarray,
                                            np.ndarray, np.ndarray]:
        z_parts, x_parts, y_parts, a_parts = [], [], [], []
        if self._born_z is not None and len(self._born_z) > 0:
            z_parts.append(self._born_z)
            x_parts.append(self._born_x)
            y_parts.append(self._born_y)
            a_parts.append(self._born_amp)
        for defect, scale, n_pts in zip(self.defects, self._defect_scales,
                                         self._defect_n_points):
            zd, xd, yd, ad = defect_to_born_scatterers_3d(defect, n_points=n_pts)
            if len(zd) == 0:
                continue
            z_parts.append(zd)
            x_parts.append(xd)
            y_parts.append(yd)
            a_parts.append(ad * scale)
        if not z_parts:
            empty = np.empty(0, dtype=np.float64)
            return empty, empty, empty, empty
        return (np.concatenate(z_parts), np.concatenate(x_parts),
                np.concatenate(y_parts), np.concatenate(a_parts))

    # ------------------------------------------------------------------
    # Born scattering — vectorised over rx and scatterers, chunked over tx
    # ------------------------------------------------------------------

    def _compute_born_scattering_3d(self,
                                     fmc_data: np.ndarray,
                                     elem_xyz: np.ndarray,
                                     time_axis: np.ndarray,
                                     z_s: np.ndarray,
                                     x_s: np.ndarray,
                                     y_s: np.ndarray,
                                     amp_s: np.ndarray,
                                     tx_chunk: int = 1,
                                     verbose: bool = True) -> None:
        from scipy.ndimage import convolve1d

        cfg = self.cfg
        assert cfg.material is not None
        c_L  = cfg.material.c_L
        freq = cfg.array.frequency
        bw   = cfg.array.bandwidth
        sigma = 1.0 / (np.pi * freq * bw)
        wavelength = c_L / freq
        w_x = cfg.array.element_width_x
        w_y = cfg.array.element_width_y

        n_el = elem_xyz.shape[0]
        n_t  = len(time_axis)
        dt   = float(time_axis[1] - time_axis[0])
        alpha_np_per_m = float(cfg.material.attenuation_L * (freq / 1e6))
        inv_c = np.float32(1.0 / c_L)

        # Gabor kernel centred at τ = 0
        n_half = int(np.ceil(5.0 * sigma / dt))
        tau = np.arange(-n_half, n_half + 1) * dt
        gabor_k = (np.exp(-0.5 * (tau / sigma) ** 2)
                   * np.cos(2.0 * np.pi * freq * tau)).astype(np.float32)

        # Cast scatterers to float32
        z_s32 = z_s.astype(np.float32, copy=False)
        x_s32 = x_s.astype(np.float32, copy=False)
        y_s32 = y_s.astype(np.float32, copy=False)
        amp32 = amp_s.astype(np.float32, copy=False)
        elem32 = elem_xyz.astype(np.float32, copy=False)

        # Precompute rx-side distances & directivities once (shared across tx)
        d_rx_z = z_s32[np.newaxis, :] - elem32[:, 0:1]
        d_rx_x = x_s32[np.newaxis, :] - elem32[:, 1:2]
        d_rx_y = y_s32[np.newaxis, :] - elem32[:, 2:3]
        r_rx_all = np.sqrt(d_rx_z ** 2 + d_rx_x ** 2 + d_rx_y ** 2)
        r_rx_safe = np.maximum(r_rx_all, np.float32(1e-10))

        # Separable rectangular-element directivity needs the ray's projected
        # angle onto the x-z plane and the y-z plane independently.
        r_xz = np.maximum(np.sqrt(d_rx_z ** 2 + d_rx_x ** 2), np.float32(1e-10))
        r_yz = np.maximum(np.sqrt(d_rx_z ** 2 + d_rx_y ** 2), np.float32(1e-10))
        sin_rx_x = np.clip(d_rx_x / r_xz, -1.0, 1.0)
        sin_rx_y = np.clip(d_rx_y / r_yz, -1.0, 1.0)
        theta_rx_x = np.arcsin(sin_rx_x)
        theta_rx_y = np.arcsin(sin_rx_y)
        dir_rx_all = element_directivity_3d_array(
            theta_rx_x, theta_rx_y, w_x, w_y, wavelength
        ).astype(np.float32)
        del sin_rx_x, sin_rx_y, theta_rx_x, theta_rx_y, r_xz, r_yz
        del d_rx_z, d_rx_x, d_rx_y

        f_flat = np.zeros((n_el, n_t), dtype=np.float32)

        for tx0 in range(0, n_el, tx_chunk):
            tx1 = min(tx0 + tx_chunk, n_el)
            self._render_tx_chunk(
                fmc_data, tx0, tx1, elem32,
                z_s32, x_s32, y_s32, amp32, inv_c,
                r_rx_all, r_rx_safe, dir_rx_all,
                w_x, w_y, wavelength, alpha_np_per_m,
                dt, n_t, gabor_k, convolve1d, f_flat,
            )
            if verbose and ((tx0 // tx_chunk) % max(1, n_el // (10 * tx_chunk)) == 0):
                print(f"    tx {tx1}/{n_el}")

    def _render_tx_chunk(self, fmc_data, tx0, tx1, elem32,
                          z_s32, x_s32, y_s32, amp32, inv_c,
                          r_rx_all, r_rx_safe, dir_rx_all,
                          w_x, w_y, wavelength, alpha_np_per_m,
                          dt, n_t, gabor_k, convolve1d, f_flat):
        n_el = elem32.shape[0]
        n_scat = z_s32.shape[0]
        d_tx_z = z_s32[np.newaxis, :] - elem32[tx0:tx1, 0:1]
        d_tx_x = x_s32[np.newaxis, :] - elem32[tx0:tx1, 1:2]
        d_tx_y = y_s32[np.newaxis, :] - elem32[tx0:tx1, 2:3]
        r_tx = np.sqrt(d_tx_z ** 2 + d_tx_x ** 2 + d_tx_y ** 2)
        r_tx_safe = np.maximum(r_tx, np.float32(1e-10))
        r_tx_xz = np.maximum(np.sqrt(d_tx_z ** 2 + d_tx_x ** 2), np.float32(1e-10))
        r_tx_yz = np.maximum(np.sqrt(d_tx_z ** 2 + d_tx_y ** 2), np.float32(1e-10))
        sin_tx_x = np.clip(d_tx_x / r_tx_xz, -1.0, 1.0)
        sin_tx_y = np.clip(d_tx_y / r_tx_yz, -1.0, 1.0)
        theta_tx_x = np.arcsin(sin_tx_x)
        theta_tx_y = np.arcsin(sin_tx_y)
        dir_tx = element_directivity_3d_array(
            theta_tx_x, theta_tx_y, w_x, w_y, wavelength
        ).astype(np.float32)
        del sin_tx_x, sin_tx_y, theta_tx_x, theta_tx_y, r_tx_xz, r_tx_yz
        del d_tx_z, d_tx_x, d_tx_y

        pair_idx = np.broadcast_to(
            np.arange(n_el, dtype=np.int32).reshape(n_el, 1),
            (n_el, n_scat),
        ).ravel()

        for local_i, tx_idx in enumerate(range(tx0, tx1)):
            r_tx_i = r_tx[local_i]
            r_tx_safe_i = r_tx_safe[local_i]
            dir_tx_i = dir_tx[local_i]

            total_r = r_tx_i[np.newaxis, :] + r_rx_all
            tof = total_r * inv_c
            spread = np.float32(1.0) / (
                r_tx_safe_i[np.newaxis, :] * r_rx_safe
            )
            atten = np.exp((-alpha_np_per_m) * total_r, dtype=np.float32)
            amp = (amp32[np.newaxis, :]
                   * dir_tx_i[np.newaxis, :]
                   * dir_rx_all
                   * spread
                   * atten)
            del total_r, spread, atten

            tof_samp = np.round(tof / dt).astype(np.int32)
            valid = (tof_samp >= 0) & (tof_samp < n_t)

            f_flat.fill(0.0)
            t_idx = tof_samp.clip(0, n_t - 1).ravel()
            a_val = amp.ravel()
            v = valid.ravel()

            np.add.at(f_flat, (pair_idx[v], t_idx[v]), a_val[v])

            result = convolve1d(f_flat, gabor_k, axis=1,
                                mode='constant', cval=0.0)
            fmc_data[tx_idx] += result
