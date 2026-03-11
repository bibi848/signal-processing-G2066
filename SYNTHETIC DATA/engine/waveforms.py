"""
Ultrasonic pulse generation and A-scan synthesis.

Replaces the spike model with realistic oscillatory Gabor wavelets
that match the frequency content of real NDT transducers.
"""

import numpy as np
from dataclasses import dataclass
from typing import List


@dataclass
class Arrival:
    """
    A single echo arrival to be placed into an A-scan.

    Attributes:
        tof: Time-of-flight / arrival time (s)
        amplitude: Peak amplitude (arbitrary units, includes all loss factors)
        phase: Phase shift (radians). π for free-surface reflection.
        path_label: Descriptive label (e.g. 'direct_LL', 'backwall', 'skip_LS')
    """
    tof: float
    amplitude: float
    phase: float = 0.0
    path_label: str = ""


def gabor_pulse(t: np.ndarray, t0: float, frequency: float,
                bandwidth: float = 0.6, phase: float = 0.0) -> np.ndarray:
    """
    Generate a Gabor wavelet (Gaussian-windowed sinusoid).

    This models a realistic broadband ultrasonic pulse as emitted
    by a piezoelectric transducer.

    Args:
        t: Time axis array (s)
        t0: Pulse center time / arrival time (s)
        frequency: Center frequency (Hz)
        bandwidth: Fractional bandwidth (0-1). 0.6 = 60% bandwidth at -6dB.
                   Controls pulse duration: higher BW = shorter pulse.
        phase: Initial phase (radians). 0 = cosine pulse, π = inverted.

    Returns:
        Wavelet array with same shape as t
    """
    # Gaussian envelope width from bandwidth
    # BW = Δf/f₀, and σ_t ≈ 1/(π·f₀·BW) for a Gabor wavelet
    sigma = 1.0 / (np.pi * frequency * bandwidth)

    # Time relative to arrival
    dt = t - t0

    # Gaussian envelope
    envelope = np.exp(-0.5 * (dt / sigma) ** 2)

    # Carrier with phase
    carrier = np.cos(2.0 * np.pi * frequency * dt + phase)

    return envelope * carrier


def synthesize_ascan(time_axis: np.ndarray, arrivals: List[Arrival],
                     frequency: float, bandwidth: float = 0.6) -> np.ndarray:
    """
    Build a complete A-scan by summing Gabor wavelets for all arrivals.

    Each arrival produces an oscillatory pulse at its time-of-flight,
    scaled by its amplitude and shifted by its phase.

    Args:
        time_axis: Time axis array (s), shape (N_t,)
        arrivals: List of Arrival objects to superpose
        frequency: Center frequency (Hz) of the transducer
        bandwidth: Fractional bandwidth of the transducer

    Returns:
        A-scan signal array, shape (N_t,)
    """
    a_scan = np.zeros_like(time_axis)

    t_max = time_axis[-1]

    for arrival in arrivals:
        # Skip arrivals outside the time window
        if arrival.tof < 0 or arrival.tof > t_max:
            continue
        # Skip negligible arrivals
        if abs(arrival.amplitude) < 1e-20:
            continue

        pulse = gabor_pulse(time_axis, arrival.tof, frequency,
                           bandwidth, arrival.phase)
        a_scan += arrival.amplitude * pulse

    return a_scan


def synthesize_ascan_vectorized(time_axis: np.ndarray,
                                tofs: np.ndarray,
                                amplitudes: np.ndarray,
                                phases: np.ndarray,
                                frequency: float,
                                bandwidth: float = 0.6) -> np.ndarray:
    """
    Vectorized A-scan synthesis for many arrivals at once.

    More efficient than synthesize_ascan when there are many arrivals
    (e.g. from Kirchhoff surface discretization).

    Args:
        time_axis: (N_t,) time axis
        tofs: (N_arrivals,) time-of-flight for each arrival
        amplitudes: (N_arrivals,) amplitude for each arrival
        phases: (N_arrivals,) phase for each arrival
        frequency: Center frequency (Hz)
        bandwidth: Fractional bandwidth

    Returns:
        A-scan signal array, shape (N_t,)
    """
    sigma = 1.0 / (np.pi * frequency * bandwidth)

    # Filter valid arrivals (within time window and non-negligible)
    t_max = time_axis[-1]
    valid = (tofs >= 0) & (tofs <= t_max) & (np.abs(amplitudes) > 1e-20)

    if not np.any(valid):
        return np.zeros_like(time_axis)

    tofs = tofs[valid]
    amplitudes = amplitudes[valid]
    phases = phases[valid]

    # Broadcasting: time_axis (N_t, 1) vs tofs (1, N_arrivals)
    # dt shape: (N_t, N_arrivals)
    dt = time_axis[:, np.newaxis] - tofs[np.newaxis, :]

    # Gaussian envelope: (N_t, N_arrivals)
    envelope = np.exp(-0.5 * (dt / sigma) ** 2)

    # Carrier: (N_t, N_arrivals)
    carrier = np.cos(2.0 * np.pi * frequency * dt + phases[np.newaxis, :])

    # Weighted sum across arrivals
    a_scan = np.sum(amplitudes[np.newaxis, :] * envelope * carrier, axis=1)

    return a_scan
