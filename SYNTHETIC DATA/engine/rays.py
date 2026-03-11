"""
Ray path data structures and enumeration for multi-mode NDT simulation.

Enumerates physically significant ray paths between TX elements,
interfaces/defects, and RX elements. Each path tracks mode (L/S)
per leg, total time-of-flight, amplitude, and phase.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class RayLeg:
    """A single straight segment of a ray path."""
    start: np.ndarray     # [z, x] in meters
    end: np.ndarray       # [z, x] in meters
    mode: str             # 'L' or 'S'
    speed: float          # Wave speed for this mode (m/s)

    @property
    def distance(self) -> float:
        return float(np.linalg.norm(self.end - self.start))

    @property
    def tof(self) -> float:
        return self.distance / self.speed

    @property
    def direction(self) -> np.ndarray:
        d = self.end - self.start
        dist = np.linalg.norm(d)
        if dist < 1e-15:
            return np.array([1.0, 0.0])
        return d / dist


@dataclass
class RayPath:
    """
    A complete ray path from TX to RX, possibly via reflections.

    The path_type string encodes the mode sequence:
    - 'LL': direct L→L (TX emits L, defect reflects L, RX receives L)
    - 'LS': L→S mode conversion at the scatterer
    - 'SL': S→L mode conversion
    - 'SS': S→S throughout
    """
    legs: List[RayLeg]
    path_type: str           # e.g. 'LL', 'LS', 'skip_LLL', 'corner_LLL'
    reflection_coeff: float = 1.0  # Product of all Fresnel coefficients
    phase_shift: float = 0.0       # Accumulated phase (radians)

    @property
    def total_tof(self) -> float:
        return sum(leg.tof for leg in self.legs)

    @property
    def total_distance(self) -> float:
        return sum(leg.distance for leg in self.legs)


def compute_backwall_mode_converted_tof(elem_x_tx: np.ndarray,
                                         elem_x_rx: np.ndarray,
                                         bw_z: float,
                                         c_L: float,
                                         c_S: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute TOF for L→S mode-converted back-wall echo.

    Path: TX emits L-wave → hits back wall → converts to S-wave → returns to RX.
    The reflection point is NOT at (x_tx + x_rx)/2 because the two legs
    travel at different speeds. We solve for the correct reflection point
    using Snell's law: sin(θ_L)/c_L = sin(θ_S)/c_S.

    For the small angles in typical NDT (array aperture << specimen thickness),
    we use an iterative solver starting from the midpoint.

    Args:
        elem_x_tx: (num_el,) TX element x-positions
        elem_x_rx: (num_el,) RX element x-positions
        bw_z: Back-wall z-position (m)
        c_L: Longitudinal speed (m/s)
        c_S: Shear speed (m/s)

    Returns:
        tof: (num_tx, num_rx) time-of-flight array
        x_refl: (num_tx, num_rx) reflection point x-position
    """
    num_tx = len(elem_x_tx)
    num_rx = len(elem_x_rx)

    tx_x, rx_x = np.meshgrid(elem_x_tx, elem_x_rx, indexing='ij')
    dx = rx_x - tx_x

    # For L→S at back wall with Snell's law:
    # The L-wave leg from TX at (0, x_tx) to reflection point (bw_z, x_r)
    # has angle θ_L = arctan(|x_r - x_tx| / bw_z)
    # The S-wave leg from (bw_z, x_r) to RX at (0, x_rx)
    # has angle θ_S = arctan(|x_rx - x_r| / bw_z)
    # Snell's law: sin(θ_L)/c_L = sin(θ_S)/c_S

    # For small angles, start with the midpoint and iterate
    # x_r = x_tx + fraction * dx, where fraction accounts for speed ratio
    # Initial guess: weighted by speeds (L-leg is faster, so it covers more lateral distance)
    fraction = c_L / (c_L + c_S)  # ~0.67 for aluminum
    x_r = tx_x + fraction * dx

    # Newton's method to satisfy Snell's law (3 iterations sufficient)
    for _ in range(5):
        dx_tx = x_r - tx_x
        dx_rx = rx_x - x_r
        d_tx = np.sqrt(bw_z**2 + dx_tx**2)
        d_rx = np.sqrt(bw_z**2 + dx_rx**2)
        sin_L = dx_tx / d_tx
        sin_S = dx_rx / d_rx

        # Snell's residual: sin(θ_L)/c_L - sin(θ_S)/c_S = 0
        residual = sin_L / c_L - sin_S / c_S

        # Derivative of residual w.r.t. x_r
        cos_L_sq = bw_z**2 / d_tx**2
        cos_S_sq = bw_z**2 / d_rx**2
        d_residual = (cos_L_sq / (c_L * d_tx)) + (cos_S_sq / (c_S * d_rx))

        # Update
        x_r = x_r - residual / np.maximum(d_residual, 1e-20)

    # Compute final TOF
    d_tx = np.sqrt(bw_z**2 + (x_r - tx_x)**2)
    d_rx = np.sqrt(bw_z**2 + (rx_x - x_r)**2)
    tof = d_tx / c_L + d_rx / c_S

    return tof, x_r


def compute_skip_path_tof(elem_x_tx: np.ndarray,
                           defect_pos: np.ndarray,
                           elem_x_rx: np.ndarray,
                           bw_z: float,
                           c: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute TOF for single-skip path: TX → back wall → defect → RX.

    Uses the mirror-image method: reflect TX across the back wall,
    then the path TX_mirror → defect is a straight line whose length
    gives the TOF for the TX → backwall → defect portion.

    Args:
        elem_x_tx: (num_el,) TX element x-positions
        defect_pos: [z, x] defect position
        elem_x_rx: (num_el,) RX element x-positions
        bw_z: Back-wall z-position
        c: Wave speed for this mode (m/s)

    Returns:
        tof: (num_tx, num_rx) TOF array
        refl_x: (num_tx,) back-wall reflection point x-positions
    """
    dz, dx = defect_pos

    # Mirror TX across back wall: z_mirror = 2*bw_z - 0 = 2*bw_z
    z_mirror = 2 * bw_z

    # TX_mirror to defect distances
    # TX_mirror at (z_mirror, elem_x_tx[i])
    d_mirror_to_defect = np.sqrt((z_mirror - dz)**2 + (elem_x_tx - dx)**2)

    # Defect to RX distances
    d_defect_to_rx = np.sqrt(dz**2 + (elem_x_rx - dx)**2)

    # Total distance: TX→backwall→defect + defect→RX
    # shape: (num_tx, num_rx) via broadcasting
    total_dist = d_mirror_to_defect[:, np.newaxis] + d_defect_to_rx[np.newaxis, :]
    tof = total_dist / c

    # Back-wall reflection point (for reference)
    # On the line from TX_mirror to defect, the back-wall intersection
    t_param = (bw_z - 0.0) / (z_mirror - dz + 1e-15)  # parametric along mirror→defect
    refl_x = elem_x_tx + t_param * (dx - elem_x_tx)

    return tof, refl_x


def compute_corner_trap_tof(elem_x_tx: np.ndarray,
                             defect_pos: np.ndarray,
                             elem_x_rx: np.ndarray,
                             bw_z: float,
                             c: float) -> np.ndarray:
    """
    Compute TOF for corner trap path: TX → defect → back wall → RX.

    The wave first hits the defect, then bounces off the back wall
    back to the receiver. This is the primary mechanism for detecting
    vertical cracks near the back wall.

    Uses mirror image: reflect RX across back wall.

    Args:
        elem_x_tx: (num_el,) TX x-positions
        defect_pos: [z, x] defect position
        elem_x_rx: (num_el,) RX x-positions
        bw_z: Back-wall z-position
        c: Wave speed (m/s)

    Returns:
        tof: (num_tx, num_rx) TOF array
    """
    dz, dx = defect_pos
    z_mirror_rx = 2 * bw_z

    # TX to defect
    d_tx_to_defect = np.sqrt(dz**2 + (elem_x_tx - dx)**2)

    # Defect to RX_mirror (via back wall)
    d_defect_to_rx_mirror = np.sqrt((z_mirror_rx - dz)**2 + (elem_x_rx - dx)**2)

    total_dist = d_tx_to_defect[:, np.newaxis] + d_defect_to_rx_mirror[np.newaxis, :]
    return total_dist / c
