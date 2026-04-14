"""
FMC acquisition simulation engine.

Orchestrates the physics-based generation of Full Matrix Capture data
by computing ray paths, evaluating physics, and synthesizing A-scans.
Includes TFM reconstruction using the existing CTFM1D algorithm.
"""

import numpy as np
import time as time_module
import sys
import os
from typing import List, Optional

# Add parent directory so we can import Classes/TFM1D
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from Classes.TFM1D import CTFM1D

from .config import SimulationConfig
from .geometry import Specimen2D, Defect2D
from .waveforms import Arrival, synthesize_ascan_vectorized
from .interfaces import (
    reflection_coefficient_normal,
    fresnel_solid_free_surface,
)
from .propagation import (
    geometric_spreading_2d,
    geometric_spreading_2d_array,
    material_attenuation,
    material_attenuation_array,
    element_directivity,
    element_directivity_array,
    incidence_angle,
)
from .materials import acoustic_impedance, AIR
from .rays import (
    compute_backwall_mode_converted_tof,
    compute_skip_path_tof,
    compute_corner_trap_tof,
)


class FMCEngine:
    """
    Physics-based FMC data simulator.

    Generates synthetic Full Matrix Capture data by computing:
    - Front-wall echoes (couplant → specimen interface)
    - Back-wall echoes (specimen → air interface)
    - Defect echoes (via Kirchhoff surface scattering)
    - Skip paths and corner trap (via mirror-image method)
    - Mode-converted paths (LL, LS, SL, SS)
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

        # Voxel Born scatterers set via set_born_scatterers()
        self._born_z:   Optional[np.ndarray] = None   # (N_s,) scatterer depths
        self._born_x:   Optional[np.ndarray] = None   # (N_s,) scatterer laterals
        self._born_amp: Optional[np.ndarray] = None   # (N_s,) Born amplitudes

    def add_defect(self, defect: Defect2D, amplitude_scale: float = 1.0):
        """Register a geometric defect in the simulation.

        amplitude_scale multiplies the Kirchhoff scattered-field amplitude.
        Used to weight individual elevation-aperture sub-slices so their
        superposition represents the slab-averaged cross-section.
        """
        self.defects.append(defect)
        self._defect_scales.append(float(amplitude_scale))

    def set_born_scatterers(self,
                             z_s: np.ndarray,
                             x_s: np.ndarray,
                             amp_s: np.ndarray) -> None:
        """
        Register Born-approximation scatterers derived from a voxel volume.

        These represent grain boundaries or other weak impedance contrasts.
        They are added on top of any geometric defects.

        Args:
            z_s:   (N_s,) depth coordinates (m)
            x_s:   (N_s,) lateral coordinates in the scan plane (m)
            amp_s: (N_s,) Born amplitudes = δZ / (2·Z₀)
        """
        self._born_z   = np.asarray(z_s,   dtype=np.float64)
        self._born_x   = np.asarray(x_s,   dtype=np.float64)
        self._born_amp = np.asarray(amp_s, dtype=np.float64)

    def simulate(self) -> dict:
        """
        Run the full FMC acquisition simulation.

        Returns:
            dict with keys:
                'fmc_data': (num_tx, num_rx, time_samples) float32 array
                'time_axis': (time_samples,) array in seconds
                'element_positions': (num_elements,) array in meters
                'config': the SimulationConfig used
        """
        cfg = self.cfg
        num_el = cfg.array.num_elements
        n_t = cfg.acquisition.time_samples
        time_axis = cfg.time_axis
        elem_x = cfg.array.element_positions  # (num_el,)

        fmc_data = np.zeros((num_el, num_el, n_t), dtype=np.float32)

        n_born = len(self._born_z) if self._born_z is not None else 0
        print(cfg.summary())
        print(f"  Defects: {len(self.defects)}  |  Born scatterers: {n_born}")
        print(f"  Simulating FMC acquisition...")

        t_start = time_module.time()

        # --- Wall echoes (vectorized over all TX-RX pairs) ---
        wall_arrivals = None
        if cfg.wall_echoes:
            wall_arrivals = self._compute_wall_echoes(elem_x, time_axis)
            self._add_arrivals_to_fmc(fmc_data, wall_arrivals, elem_x, time_axis)

            # --- Mode-converted back-wall echo (L→S) ---
            if cfg.mode_conversion:
                self._add_mode_converted_backwall(fmc_data, elem_x, time_axis)

            # --- Back-wall reverberations (2nd, 3rd echoes) ---
            if cfg.max_bounces >= 2:
                self._add_reverberations(fmc_data, wall_arrivals, elem_x, time_axis)

        # --- Defect echoes (direct LL path) ---
        if self.defects:
            defect_arrivals = self._compute_defect_echoes(elem_x, time_axis)
            self._add_defect_arrivals_to_fmc(fmc_data, defect_arrivals,
                                              elem_x, time_axis)

        # --- Skip paths and corner trap ---
        if self.defects and cfg.max_bounces >= 2:
            self._add_skip_and_corner_trap(fmc_data, elem_x, time_axis)

        # --- Born scattering from voxel volume (grain noise / voxel defects) ---
        if self._born_z is not None and len(self._born_z) > 0:
            self._compute_born_scattering(fmc_data, elem_x, time_axis)

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
    # Wall echoes
    # ------------------------------------------------------------------

    def _compute_wall_echoes(self, elem_x: np.ndarray,
                              time_axis: np.ndarray) -> dict:
        """
        Compute front-wall and back-wall echo arrivals for all TX-RX pairs.

        Returns dict with 'front_wall' and 'back_wall' entries, each containing
        arrays of (tof, amplitude, phase) for every TX-RX pair.
        """
        cfg = self.cfg
        mat = cfg.material
        couplant = cfg.couplant
        num_el = len(elem_x)

        # --- Front-wall echo ---
        # The front wall is at z=0 (array contact surface).
        # For contact (gel) testing, the gel layer is << 1 wavelength thick
        # (~0.05-0.1 mm), so it acts purely as a coupling medium.
        #
        # The impedance mismatch determines how much energy enters the specimen:
        #   Z_gel  ≈ 1050 × 1500 = 1.575 MRayl
        #   Z_alum ≈ 2700 × 6320 = 17.06 MRayl
        #   R = (Z2 - Z1) / (Z2 + Z1) ≈ 0.83  (strong reflection)
        #
        # In contact testing, the front-wall echo overlaps with the TX pulse
        # ring-down. It arrives at t ≈ 2 × d_gel / c_gel ≈ 0.07–0.13 μs.
        # This creates the "dead zone" / noise band at the top of B-scans.
        Z_couplant = acoustic_impedance(couplant, 'L')
        Z_specimen = acoustic_impedance(mat, 'L')
        R_front = reflection_coefficient_normal(Z_couplant, Z_specimen)

        # TOF: round-trip through gel layer (d_gel = 0.05–0.1 mm typical)
        front_wall_tof = 2.0 * cfg.gel_thickness / couplant.c_L  # ≈ 0.1 μs

        # Front-wall amplitude — large due to high impedance mismatch
        front_wall_amp = abs(R_front)

        # --- Back-wall echo (L→L reflection at free surface) ---
        bw_z = self.specimen.back_wall_z

        tx_x, rx_x = np.meshgrid(elem_x, elem_x, indexing='ij')  # (num_el, num_el)
        dx = rx_x - tx_x

        # Reflection point: x_r = (x_tx + x_rx)/2 for specular on flat wall
        # TX/RX leg distances
        tx_leg = np.sqrt(bw_z**2 + (dx / 2)**2)
        rx_leg = tx_leg  # Symmetric for flat wall
        backwall_dist = tx_leg + rx_leg  # Total path length
        backwall_tof = backwall_dist / mat.c_L

        # Incidence angle at back wall
        theta_bw = np.arctan2(np.abs(dx / 2), bw_z)  # (num_el, num_el)

        # Angle-dependent Fresnel coefficients at free surface (solid→air)
        # Vectorize over all TX-RX pairs
        R_LL_bw = np.zeros_like(theta_bw)
        R_LS_bw = np.zeros_like(theta_bw)
        for i in range(num_el):
            for j in range(num_el):
                R_LL_bw[i, j], R_LS_bw[i, j] = fresnel_solid_free_surface(
                    theta_bw[i, j], mat.c_L, mat.c_S, incident_mode='L'
                )

        # 2D cylindrical spreading
        spreading = 1.0 / np.sqrt(np.maximum(tx_leg * rx_leg, 1e-20))

        # Material attenuation
        atten = material_attenuation_array(backwall_dist, cfg.array.frequency,
                                            mat.attenuation_L)

        # Element directivity
        wavelength = mat.c_L / cfg.array.frequency
        theta_el = np.arctan2(np.abs(dx / 2), bw_z)
        arg_tx = np.pi * cfg.array.element_width * np.sin(theta_el) / wavelength
        with np.errstate(invalid='ignore', divide='ignore'):
            dir_tx = np.where(np.abs(arg_tx) < 1e-10, 1.0,
                              np.abs(np.sin(arg_tx) / arg_tx))
        dir_rx = dir_tx

        # L→L back-wall echo amplitude (dominant)
        backwall_amp = np.abs(R_LL_bw) * spreading * atten * dir_tx * dir_rx

        # Phase: sign of R_LL determines phase
        backwall_phase = np.where(R_LL_bw < 0, np.pi, 0.0)

        # Store L→S mode-converted back-wall echo for Milestone 5
        self._backwall_LS = {
            'R_LS': R_LS_bw,
            'theta_bw': theta_bw,
            'tx_leg': tx_leg,
            'spreading': spreading,
            'atten': atten,
            'dir_tx': dir_tx,
            'dir_rx': dir_rx,
        }

        return {
            'front_wall_tof': front_wall_tof,
            'front_wall_amp': front_wall_amp,
            'front_wall_phase': np.pi if R_front < 0 else 0.0,
            'backwall_tof': backwall_tof,
            'backwall_amp': backwall_amp,
            'backwall_phase': backwall_phase,
        }

    def _add_arrivals_to_fmc(self, fmc_data: np.ndarray,
                              wall_arrivals: dict,
                              elem_x: np.ndarray,
                              time_axis: np.ndarray):
        """Add wall echo wavelets into the FMC data array."""
        cfg = self.cfg
        num_el = len(elem_x)
        freq = cfg.array.frequency
        bw = cfg.array.bandwidth

        # backwall_phase may be a scalar or per-element array
        bw_phase = wall_arrivals['backwall_phase']
        bw_phase_is_array = isinstance(bw_phase, np.ndarray)

        for tx_idx in range(num_el):
            for rx_idx in range(num_el):
                bw_ph = bw_phase[tx_idx, rx_idx] if bw_phase_is_array else bw_phase
                tofs = np.array([
                    wall_arrivals['front_wall_tof'],
                    wall_arrivals['backwall_tof'][tx_idx, rx_idx],
                ])
                amps = np.array([
                    wall_arrivals['front_wall_amp'],
                    wall_arrivals['backwall_amp'][tx_idx, rx_idx],
                ])
                phases = np.array([
                    wall_arrivals['front_wall_phase'],
                    bw_ph,
                ])

                a_scan = synthesize_ascan_vectorized(
                    time_axis, tofs, amps, phases, freq, bw
                )
                fmc_data[tx_idx, rx_idx, :] += a_scan

    # ------------------------------------------------------------------
    # Mode-converted back-wall echo (L→S)
    # ------------------------------------------------------------------

    def _add_mode_converted_backwall(self, fmc_data: np.ndarray,
                                      elem_x: np.ndarray,
                                      time_axis: np.ndarray):
        """
        Add L→S mode-converted back-wall echo.

        When an L-wave hits the back wall, part of the energy converts
        to an S-wave (via Fresnel coefficients). The S-wave returns at
        c_S < c_L, arriving later than the LL back-wall echo.
        """
        cfg = self.cfg
        mat = cfg.material
        num_el = len(elem_x)

        # Compute L→S TOF using Snell's law solver
        tof_LS, x_refl = compute_backwall_mode_converted_tof(
            elem_x, elem_x, self.specimen.back_wall_z, mat.c_L, mat.c_S
        )

        # Use stored Fresnel data from wall echo computation
        ls_data = self._backwall_LS
        R_LS = ls_data['R_LS']

        # Amplitude: |R_LS| × spreading × attenuation × directivity
        # Recompute leg distances with the correct reflection point
        bw_z = self.specimen.back_wall_z
        tx_x, rx_x = np.meshgrid(elem_x, elem_x, indexing='ij')
        d_tx_leg = np.sqrt(bw_z**2 + (x_refl - tx_x)**2)  # L-wave leg
        d_rx_leg = np.sqrt(bw_z**2 + (rx_x - x_refl)**2)   # S-wave leg

        spreading = 1.0 / np.sqrt(np.maximum(d_tx_leg * d_rx_leg, 1e-20))
        # Mode-converted path: L-wave on TX leg, S-wave on RX leg → each leg
        # needs its own attenuation coefficient.
        atten_L = material_attenuation_array(
            d_tx_leg, cfg.array.frequency, mat.attenuation_L
        )
        atten_S = material_attenuation_array(
            d_rx_leg, cfg.array.frequency, mat.attenuation_S
        )
        atten = atten_L * atten_S

        amp_LS = np.abs(R_LS) * spreading * atten
        phase_LS = np.where(R_LS < 0, np.pi, 0.0)

        # Add to FMC
        freq = cfg.array.frequency
        bw = cfg.array.bandwidth
        for tx_idx in range(num_el):
            for rx_idx in range(num_el):
                if abs(amp_LS[tx_idx, rx_idx]) < 1e-20:
                    continue
                a_scan = synthesize_ascan_vectorized(
                    time_axis,
                    np.array([tof_LS[tx_idx, rx_idx]]),
                    np.array([amp_LS[tx_idx, rx_idx]]),
                    np.array([phase_LS[tx_idx, rx_idx]]),
                    freq, bw
                )
                fmc_data[tx_idx, rx_idx, :] += a_scan

    # ------------------------------------------------------------------
    # Defect echoes (direct path, Kirchhoff scattering)
    # ------------------------------------------------------------------

    def _compute_defect_echoes(self, elem_x: np.ndarray,
                                time_axis: np.ndarray) -> list:
        """
        Compute defect echo contributions using Kirchhoff surface scattering.

        For each defect:
        1. Discretize the defect surface into points
        2. For each TX-RX pair, compute the contribution from each surface point
        3. Sum contributions with proper amplitude and phase

        Returns:
            List of dicts, one per defect, each containing arrays of
            shape (num_tx, num_rx, n_surface_points) for tof, amplitude, phase.
        """
        cfg = self.cfg
        mat = cfg.material
        num_el = len(elem_x)
        freq = cfg.array.frequency
        wavelength = mat.c_L / freq

        defect_data = []

        for defect_idx, defect in enumerate(self.defects):
            # Discretize the defect surface
            surface_pts, surface_normals = defect.discretize_surface(n_points=120)
            n_pts = len(surface_pts)

            # Element positions as [z, x]: all at z=0
            elem_pos = np.zeros((num_el, 2))
            elem_pos[:, 1] = elem_x

            # --- Vectorized distance computation ---
            # TX to surface points: (num_el, n_pts)
            d_tx = elem_pos[:, np.newaxis, :] - surface_pts[np.newaxis, :, :]
            dist_tx = np.linalg.norm(d_tx, axis=2)  # (num_el, n_pts)

            # RX to surface points: same as TX (symmetric)
            dist_rx = dist_tx  # (num_el, n_pts)

            # Total distance for each TX-RX-surface_point combination
            # dist_total[tx, rx, pt] = dist_tx[tx, pt] + dist_rx[rx, pt]
            # TOF = dist_total / c_L
            dist_total = dist_tx[:, np.newaxis, :] + dist_rx[np.newaxis, :, :]  # (num_el, num_el, n_pts)
            tof = dist_total / mat.c_L

            # --- Amplitude factors ---
            # Geometric spreading: 1/√(d_tx × d_rx) per surface point
            spread = 1.0 / np.sqrt(
                np.maximum(dist_tx[:, np.newaxis, :] * dist_rx[np.newaxis, :, :], 1e-20)
            )

            # Material attenuation
            atten = np.exp(-mat.attenuation_L * (freq / 1e6) * dist_total)

            # Obliquity factor: cos(θ) at the surface point
            # θ = angle between incident ray and surface normal
            # For TX leg: direction = (surface_pt - tx_pos) / distance
            dir_tx = d_tx / np.maximum(dist_tx[:, :, np.newaxis], 1e-15)  # (num_el, n_pts, 2) — wait, need to negate
            # Actually d_tx = elem_pos - surface_pts, so direction TO surface = -d_tx
            dir_to_surface = -d_tx / np.maximum(dist_tx[:, :, np.newaxis], 1e-15)  # (num_el, n_pts, 2)
            # cos(θ) = |dot(dir_to_surface, -normal)| (inward normal)
            # normal points outward from defect, so inward = -normal
            cos_theta_tx = np.abs(np.sum(
                dir_to_surface * (-surface_normals[np.newaxis, :, :]), axis=2
            ))  # (num_el, n_pts)

            # Same for RX (by reciprocity, it's the same geometry)
            cos_theta_rx = cos_theta_tx  # (num_el, n_pts)

            # Combined obliquity: cos(θ_tx) × cos(θ_rx)
            obliquity = cos_theta_tx[:, np.newaxis, :] * cos_theta_rx[np.newaxis, :, :]

            # Reflection coefficient: for a void, R ≈ -1 (total reflection)
            R = 1.0  # Magnitude; phase handled separately

            # Element directivity for TX and RX legs
            # Angle from element normal (z-axis) to surface point
            theta_tx = np.arctan2(np.abs(d_tx[:, :, 1]), np.abs(d_tx[:, :, 0]))  # (num_el, n_pts)
            dir_tx_factor = element_directivity_array(theta_tx, cfg.array.element_width, wavelength)

            theta_rx = theta_tx  # Same geometry
            dir_rx_factor = dir_tx_factor

            directivity = dir_tx_factor[:, np.newaxis, :] * dir_rx_factor[np.newaxis, :, :]

            # Surface element arc length (2D Kirchhoff integral)
            if n_pts > 1:
                ds = np.linalg.norm(np.diff(surface_pts, axis=0), axis=1)
                ds = np.append(ds, ds[-1])  # Repeat last for same length
            else:
                ds = np.ones(1)

            # Kirchhoff normalization for 2D:
            # The scattered field from a surface element ds is proportional to
            # √(k / (2π)) × ds × R × obliquity / √(d_tx × d_rx)
            # where k = 2πf/c is the wavenumber.
            k = 2.0 * np.pi * freq / mat.c_L
            kirchhoff_factor = np.sqrt(k / (2.0 * np.pi))

            # Combined amplitude per surface point
            amp_scale = (self._defect_scales[defect_idx]
                         if defect_idx < len(self._defect_scales) else 1.0)
            amplitude = (amp_scale * R * kirchhoff_factor * spread * atten * obliquity
                        * directivity * ds[np.newaxis, np.newaxis, :])

            # Phase: π for void reflection (free surface) + propagation phase
            # Propagation phase = 2πf × TOF (but this is captured by the wavelet timing)
            # The reflection phase inversion is the key contribution
            phase = np.full_like(amplitude, np.pi)  # Phase inversion for void

            defect_data.append({
                'tof': tof,           # (num_el, num_el, n_pts)
                'amplitude': amplitude,
                'phase': phase,
            })

        return defect_data

    def _add_defect_arrivals_to_fmc(self, fmc_data: np.ndarray,
                                     defect_data: list,
                                     elem_x: np.ndarray,
                                     time_axis: np.ndarray):
        """Add defect echo wavelets into the FMC data array."""
        cfg = self.cfg
        num_el = len(elem_x)
        freq = cfg.array.frequency
        bw = cfg.array.bandwidth

        for dd in defect_data:
            tof = dd['tof']       # (num_el, num_el, n_pts)
            amp = dd['amplitude']
            phase = dd['phase']

            for tx_idx in range(num_el):
                for rx_idx in range(num_el):
                    a_scan = synthesize_ascan_vectorized(
                        time_axis,
                        tof[tx_idx, rx_idx, :],
                        amp[tx_idx, rx_idx, :],
                        phase[tx_idx, rx_idx, :],
                        freq, bw
                    )
                    fmc_data[tx_idx, rx_idx, :] += a_scan

    # ------------------------------------------------------------------
    # Born scattering from voxel volume
    # ------------------------------------------------------------------

    def _compute_born_scattering(self,
                                  fmc_data: np.ndarray,
                                  elem_x: np.ndarray,
                                  time_axis: np.ndarray) -> None:
        """
        Add Born-approximation scattering from voxel-derived scatterers.

        Algorithm — impulse-train convolution:
        1. For every (tx, rx, scatterer) triple, compute the arrival sample
           index tof_samp = round(tof / dt) and weighted amplitude.
        2. Scatter-add amplitudes into a (n_pairs, n_t) impulse-train matrix
           using np.add.at — one column spike per arrival.
        3. Batch-convolve all n_pairs impulse trains with the Gabor wavelet
           kernel using scipy.ndimage.convolve1d (single vectorised call).

        Complexity: O(n_el² × n_s)  scatter  +  O(n_el² × n_t × n_kernel) conv
        For 64 elements, 6000 scatterers, 2048 samples, ~60-sample kernel:
          ~25 M  (scatter)  +  ~500 M (conv with FFT opt.) ≈ 1–3 s total.
        This is ~100× faster than the per-pair synthesize_ascan_vectorized loop.

        Args:
            fmc_data:  (num_el, num_el, n_t) FMC array, updated in-place
            elem_x:    (num_el,) element x-positions
            time_axis: (n_t,) time axis
        """
        from scipy.ndimage import convolve1d

        assert self._born_z is not None
        cfg  = self.cfg
        assert cfg.material is not None
        c_L  = cfg.material.c_L
        freq = cfg.array.frequency
        bw   = cfg.array.bandwidth
        sigma = 1.0 / (np.pi * freq * bw)

        n_el = len(elem_x)
        n_t  = len(time_axis)
        n_s  = len(self._born_z)
        dt   = float(time_axis[1] - time_axis[0])
        alpha_np_per_m = float(cfg.material.attenuation_L * (freq / 1e6))

        # Build Gabor kernel centred at τ = 0
        n_half  = int(np.ceil(5.0 * sigma / dt))
        tau     = np.arange(-n_half, n_half + 1) * dt
        gabor_k = (np.exp(-0.5 * (tau / sigma) ** 2)
                   * np.cos(2.0 * np.pi * freq * tau)).astype(np.float32)

        # Cast scatterer coords to float32 up front so that every (n_el², n_s)
        # array downstream is float32, not float64 — halves peak memory.
        born_z = self._born_z.astype(np.float32, copy=False)
        born_x = self._born_x.astype(np.float32, copy=False)
        born_amp = self._born_amp.astype(np.float32, copy=False)

        # Distances: (n_el, n_s)
        elem_x32 = elem_x.astype(np.float32, copy=False)
        dz = born_z[np.newaxis, :]
        dx = born_x[np.newaxis, :] - elem_x32[:, np.newaxis]
        r  = np.sqrt(dz ** 2 + dx ** 2).astype(np.float32)

        # Element directivity: sinc pattern suppresses off-axis contributions
        theta_el = np.arctan2(np.abs(dx), np.abs(dz))   # (n_el, n_s)
        wavelength = c_L / freq
        dir_factor = element_directivity_array(
            theta_el, cfg.array.element_width, wavelength
        ).astype(np.float32)

        # TOF and amplitude: (n_el, n_el, n_s), built in float32.
        r_tx = r[:, np.newaxis, :]
        r_rx = r[np.newaxis, :, :]

        # total_dist = r_tx + r_rx  → reused for tof, spread weighting, and
        # the attenuation exponent, so we only materialise it once.
        total_dist = (r_tx + r_rx).astype(np.float32)      # (n_el, n_el, n_s)
        tof = total_dist * np.float32(1.0 / c_L)

        # Combined amplitude weight: spreading × directivity × material
        # attenuation.  Born scattering must decay with path length the same
        # way every other path in the engine does — otherwise deep grain
        # noise sits at full amplitude and inverts the apparent depth
        # gradient in the image.
        spread = np.float32(1.0) / np.sqrt(
            np.maximum(r_tx * r_rx, np.float32(1e-20))
        )
        directivity = (dir_factor[:, np.newaxis, :]
                        * dir_factor[np.newaxis, :, :])
        # Exp into a pre-allocated buffer, then fold into spread in-place.
        atten = np.exp((-alpha_np_per_m) * total_dist, dtype=np.float32)
        spread *= directivity
        spread *= atten
        del directivity, atten                           # release ASAP
        amp = born_amp[np.newaxis, np.newaxis, :] * spread
        del spread

        # Sample indices: (n_el, n_el, n_s)
        tof_samp = np.round(tof / dt).astype(np.int32)
        valid    = (tof_samp >= 0) & (tof_samp < n_t)

        # Build impulse trains: (n_pairs, n_t)
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

        # Batch convolve all impulse trains with the Gabor kernel
        result = convolve1d(f_flat, gabor_k, axis=1,
                            mode='constant', cval=0.0)

        fmc_data += result.reshape(n_el, n_el, n_t)

    # ------------------------------------------------------------------
    # Back-wall reverberations
    # ------------------------------------------------------------------

    def _add_reverberations(self, fmc_data: np.ndarray,
                             wall_arrivals: dict,
                             elem_x: np.ndarray,
                             time_axis: np.ndarray):
        """
        Add back-wall reverberations (2nd, 3rd multiple echoes).

        In real NDT, the pulse bounces back and forth between front and
        back walls. Each round trip attenuates by R_front × R_back and
        doubles the path length. Typically 2-3 reverberations are visible.
        """
        cfg = self.cfg
        mat = cfg.material
        num_el = len(elem_x)
        freq = cfg.array.frequency
        bw = cfg.array.bandwidth
        t_max = time_axis[-1]

        # First back-wall echo data
        bw_tof_1 = wall_arrivals['backwall_tof']   # (num_el, num_el)
        bw_amp_1 = wall_arrivals['backwall_amp']
        bw_phase = wall_arrivals['backwall_phase']

        # Round-trip attenuation factor per additional bounce
        # Each additional round trip: R_front × R_back × attenuation × spreading
        from .materials import acoustic_impedance
        Z_couplant = acoustic_impedance(cfg.couplant, 'L')
        Z_specimen = acoustic_impedance(mat, 'L')
        R_front = reflection_coefficient_normal(Z_specimen, Z_couplant)
        # R_back ≈ -1 (already in Fresnel), use average for reverberations
        round_trip_factor = abs(R_front) * 0.95  # Slight loss per bounce

        bw_z = self.specimen.back_wall_z
        round_trip_atten = material_attenuation_array(
            2 * bw_z * np.ones(1), freq, mat.attenuation_L
        )[0]

        for n in range(2, min(cfg.max_bounces + 1, 4)):
            # nth echo arrives at n × first-echo TOF
            tof_n = n * bw_tof_1
            # Amplitude decays with each round trip
            amp_factor = (round_trip_factor * round_trip_atten) ** (n - 1)
            amp_n = bw_amp_1 * amp_factor
            # Phase alternates with each reflection
            phase_n = bw_phase  # Additional π per bounce handled by R factors

            for tx_idx in range(num_el):
                for rx_idx in range(num_el):
                    t_arr = tof_n[tx_idx, rx_idx]
                    if t_arr > t_max:
                        continue
                    bw_ph = phase_n[tx_idx, rx_idx] if isinstance(phase_n, np.ndarray) else phase_n
                    a_scan = synthesize_ascan_vectorized(
                        time_axis,
                        np.array([t_arr]),
                        np.array([amp_n[tx_idx, rx_idx]]),
                        np.array([bw_ph]),
                        freq, bw
                    )
                    fmc_data[tx_idx, rx_idx, :] += a_scan

    # ------------------------------------------------------------------
    # Skip paths and corner trap
    # ------------------------------------------------------------------

    def _add_skip_and_corner_trap(self, fmc_data: np.ndarray,
                                    elem_x: np.ndarray,
                                    time_axis: np.ndarray):
        """
        Add skip-path and corner-trap echoes for each defect.

        Skip path: TX → back wall → defect → RX
          The wave bounces off the back wall first, then hits the defect.
          Important for detecting defects near the back wall.

        Corner trap: TX → defect → back wall → RX
          The wave hits the defect first, then bounces off the back wall.
          Critical for vertical crack detection — the classic corner echo.
        """
        cfg = self.cfg
        mat = cfg.material
        num_el = len(elem_x)
        freq = cfg.array.frequency
        bw = cfg.array.bandwidth
        bw_z = self.specimen.back_wall_z
        t_max = time_axis[-1]
        wavelength = mat.c_L / freq
        k = 2.0 * np.pi * freq / mat.c_L

        for defect_idx, defect in enumerate(self.defects):
            # Use defect center as the scattering point for skip/corner
            defect_pos = defect.center
            amp_scale = (self._defect_scales[defect_idx]
                         if defect_idx < len(self._defect_scales) else 1.0)

            # --- Skip path: TX → back wall → defect → RX ---
            skip_tof, refl_x = compute_skip_path_tof(
                elem_x, defect_pos, elem_x, bw_z, mat.c_L
            )

            # Amplitude for skip path: two reflections (back wall + defect)
            # Each with geometric spreading over total path
            z_mirror = 2 * bw_z
            for tx_idx in range(num_el):
                for rx_idx in range(num_el):
                    t_arr = skip_tof[tx_idx, rx_idx]
                    if t_arr > t_max:
                        continue

                    # Leg distances
                    d_tx_bw = np.sqrt(bw_z**2 + (refl_x[tx_idx] - elem_x[tx_idx])**2)
                    d_bw_def = np.sqrt((bw_z - defect_pos[0])**2 + (refl_x[tx_idx] - defect_pos[1])**2)
                    d_def_rx = np.sqrt(defect_pos[0]**2 + (elem_x[rx_idx] - defect_pos[1])**2)
                    total_dist = d_tx_bw + d_bw_def + d_def_rx

                    # Spreading over total path (simplified)
                    spread = 1.0 / np.sqrt(max(total_dist * defect_pos[0], 1e-20))
                    atten = np.exp(-mat.attenuation_L * (freq / 1e6) * total_dist)
                    # Defect scattering cross-section (simplified: proportional to size)
                    defect_size = getattr(defect, 'radius', 1e-3) * 2
                    scatter_factor = np.sqrt(k * defect_size / (2 * np.pi))

                    amp = amp_scale * scatter_factor * spread * atten * 0.5  # 0.5 for skip attenuation
                    phase = np.pi  # Phase inversion at void

                    a_scan = synthesize_ascan_vectorized(
                        time_axis,
                        np.array([t_arr]),
                        np.array([amp]),
                        np.array([phase]),
                        freq, bw
                    )
                    fmc_data[tx_idx, rx_idx, :] += a_scan

            # --- Corner trap: TX → defect → back wall → RX ---
            corner_tof = compute_corner_trap_tof(
                elem_x, defect_pos, elem_x, bw_z, mat.c_L
            )

            for tx_idx in range(num_el):
                for rx_idx in range(num_el):
                    t_arr = corner_tof[tx_idx, rx_idx]
                    if t_arr > t_max:
                        continue

                    d_tx_def = np.sqrt(defect_pos[0]**2 + (elem_x[tx_idx] - defect_pos[1])**2)
                    d_def_bw = bw_z - defect_pos[0]
                    d_bw_rx = np.sqrt(bw_z**2 + (elem_x[rx_idx] - defect_pos[1])**2)
                    total_dist = d_tx_def + d_def_bw + d_bw_rx

                    spread = 1.0 / np.sqrt(max(total_dist * defect_pos[0], 1e-20))
                    atten = np.exp(-mat.attenuation_L * (freq / 1e6) * total_dist)
                    defect_size = getattr(defect, 'radius', 1e-3) * 2
                    scatter_factor = np.sqrt(k * defect_size / (2 * np.pi))

                    amp = amp_scale * scatter_factor * spread * atten * 0.5
                    phase = 0.0  # Double reflection: π + π = 0

                    a_scan = synthesize_ascan_vectorized(
                        time_axis,
                        np.array([t_arr]),
                        np.array([amp]),
                        np.array([phase]),
                        freq, bw
                    )
                    fmc_data[tx_idx, rx_idx, :] += a_scan
