"""
Validation tests for the 3D Born engine (engine_3d).

These tests run small problems so they complete in seconds:
  1. Analytic TOF for a single off-plane point scatterer
  2. 1/r² round-trip amplitude scaling (vs the 2D engine's 1/r)
  3. Reciprocity: fmc[tx, rx] ≈ fmc[rx, tx]
  4. No regression in the 2D engine (its tests still import cleanly)
"""

from __future__ import annotations

import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from engine_3d import (
    SimulationConfig3D, ArrayConfig3D, SpecimenConfig3D, AcquisitionConfig3D,
    FMCEngine3D, ALUMINUM,
)
from engine_3d.propagation import (
    geometric_spreading_3d, element_directivity_3d_array,
)
from engine_3d.voxel_volume import VoxelVolume3D, extract_born_scatterers_3d
from engine.propagation import element_directivity_array


def _make_small_config(freq: float = 5e6, n_x: int = 4, n_y: int = 4,
                        time_samples: int = 1024) -> SimulationConfig3D:
    return SimulationConfig3D(
        material=ALUMINUM,
        array=ArrayConfig3D(
            n_elements_x=n_x, n_elements_y=n_y,
            pitch_x=0.6e-3, pitch_y=0.6e-3,
            element_width_x=0.54e-3, element_width_y=0.54e-3,
            frequency=freq, bandwidth=0.6,
        ),
        specimen=SpecimenConfig3D(thickness=40e-3, width=20e-3, depth=20e-3),
        acquisition=AcquisitionConfig3D(time_samples=time_samples),
    )


def _peak_sample(ascan: np.ndarray) -> int:
    """Sample index of the absolute-peak in an A-scan."""
    return int(np.argmax(np.abs(ascan)))


def test_analytic_tof_offplane_point():
    """TOF for a known (z, x, y) scatterer matches the expected (r_tx + r_rx)/c."""
    cfg = _make_small_config()
    engine = FMCEngine3D(cfg)

    # Put a single scatterer clearly off the z-axis
    z0, x0, y0 = 15e-3, 2e-3, 1.5e-3
    engine.set_born_scatterers(
        np.array([z0]), np.array([x0]), np.array([y0]), np.array([1.0]),
    )

    result = engine.simulate(verbose=False)
    fmc = result['fmc_data']
    elem = result['element_positions_xyz']
    dt = cfg.dt
    c_L = cfg.material.c_L

    # Check three randomly chosen tx/rx pairs
    rng = np.random.default_rng(0)
    idx_pairs = rng.integers(0, cfg.array.n_elements_total, size=(3, 2))
    for tx, rx in idx_pairs:
        pt = np.array([z0, x0, y0])
        r_tx = np.linalg.norm(pt - elem[tx])
        r_rx = np.linalg.norm(pt - elem[rx])
        expected_tof = (r_tx + r_rx) / c_L
        expected_sample = int(round(expected_tof / dt))
        actual_sample = _peak_sample(fmc[tx, rx])
        assert abs(actual_sample - expected_sample) <= 1, (
            f"tx={tx} rx={rx}: expected sample {expected_sample}, "
            f"got {actual_sample}"
        )


def test_spreading_is_inverse_r_squared():
    """
    Move a point scatterer from range r to 2r (normal-incidence, on the central
    element's axis). With 1/r spreading per leg the round-trip intensity ratio
    is ~1/4 in amplitude. 2D (1/√r) would give ~1/2.
    """
    # Use a 1×1 array at origin so element directivity is effectively flat
    cfg = _make_small_config(n_x=1, n_y=1)
    c_L = cfg.material.c_L
    freq = cfg.array.frequency
    alpha = float(cfg.material.attenuation_L * (freq / 1e6))

    # Two scatterers straight below the element at z=r and z=2r
    r = 10e-3
    engine_near = FMCEngine3D(cfg)
    engine_near.set_born_scatterers(np.array([r]), np.array([0.0]),
                                     np.array([0.0]), np.array([1.0]))
    engine_far = FMCEngine3D(cfg)
    engine_far.set_born_scatterers(np.array([2 * r]), np.array([0.0]),
                                    np.array([0.0]), np.array([1.0]))

    fmc_near = engine_near.simulate(verbose=False)['fmc_data'][0, 0]
    fmc_far = engine_far.simulate(verbose=False)['fmc_data'][0, 0]

    amp_near = np.max(np.abs(fmc_near))
    amp_far = np.max(np.abs(fmc_far))

    # Analytic ratio, including attenuation:
    # amp ∝ (1/r_tx)(1/r_rx) · exp(-α (r_tx + r_rx))
    #      = (1/r²) exp(-2αr) for normal incidence
    ratio_expected = (
        (1.0 / (2 * r) ** 2) * np.exp(-alpha * 2 * (2 * r))
    ) / (
        (1.0 / r ** 2) * np.exp(-alpha * 2 * r)
    )
    ratio_actual = amp_far / amp_near
    assert abs(ratio_actual / ratio_expected - 1.0) < 0.05, (
        f"expected ratio {ratio_expected:.3f}, got {ratio_actual:.3f}"
    )


def test_reciprocity():
    """
    For a linear forward model with rectangular-separable directivity the FMC
    must be reciprocal: fmc[i, j, :] == fmc[j, i, :].
    """
    cfg = _make_small_config(n_x=3, n_y=3)
    engine = FMCEngine3D(cfg)
    engine.set_born_scatterers(
        np.array([15e-3, 25e-3]),
        np.array([1e-3, -0.5e-3]),
        np.array([0.5e-3, -1e-3]),
        np.array([1.0, -0.8]),
    )
    fmc = engine.simulate(verbose=False)['fmc_data']
    diff = np.max(np.abs(fmc - np.transpose(fmc, (1, 0, 2))))
    # Compare against signal scale; float32 + scatter-add ordering produces
    # ~1e-7 relative error, which is well below any physical threshold.
    scale = float(np.max(np.abs(fmc)))
    assert diff / max(scale, 1e-30) < 1e-5, (
        f"reciprocity broken, max |Δ|={diff:.2e}, scale={scale:.2e}"
    )


def test_spreading_helper_monotonic():
    """Sanity: 1/r is monotonic decreasing and positive."""
    rs = np.linspace(1e-4, 50e-3, 10)
    vals = np.array([geometric_spreading_3d(r) for r in rs])
    assert np.all(np.diff(vals) < 0.0)
    assert np.all(vals > 0.0)


def test_directivity_shape():
    """Directivity is 1 at normal incidence and < 1 off-axis."""
    thetas = np.array([0.0, 0.3, 0.8])
    thetas_y = np.zeros_like(thetas)
    d = element_directivity_3d_array(
        thetas, thetas_y, 0.5e-3, 0.5e-3, 6320.0 / 5e6,
    )
    assert abs(d[0] - 1.0) < 1e-6
    assert d[1] < d[0] and d[2] < d[1]


def test_directivity_separability():
    """
    With the y-aperture collapsed (w_y → 0), the 3D rectangular directivity
    must reduce to the 1D x-only sinc pattern. The 2D engine returns |sinc|,
    so compare absolute values.
    """
    wavelength = 6320.0 / 5e6
    w_x = 0.5e-3
    w_y_tiny = 1e-12
    thetas_x = np.linspace(-0.9, 0.9, 11)
    thetas_y = np.zeros_like(thetas_x)
    d_3d = element_directivity_3d_array(
        thetas_x, thetas_y, w_x, w_y_tiny, wavelength,
    )
    d_1d = element_directivity_array(thetas_x, w_x, wavelength)
    assert np.allclose(np.abs(d_3d), d_1d, atol=1e-6)


def test_voxel_extractor_lateral_boundary():
    """
    A voxel volume with a pure x-direction impedance step (no z or y
    variation) must yield at least one Born scatterer. The previous
    z-only gradient returned zero for this case.
    """
    n_z, n_y, n_x = 4, 4, 8
    imp = np.ones((n_z, n_y, n_x), dtype=np.float32) * 1.0e7
    imp[:, :, n_x // 2:] = 1.1e7            # 10% impedance step along x
    wavespeed = np.full_like(imp, 6320.0)
    vol = VoxelVolume3D(
        impedance=imp, wavespeed=wavespeed,
        voxel_size=0.1e-3, origin_z=0.0, origin_y=0.0, origin_x=0.0,
    )
    z_s, x_s, y_s, amp_s = extract_born_scatterers_3d(
        vol, background_Z=1.0e7, threshold=0.001,
    )
    assert z_s.size > 0, "lateral impedance step must produce scatterers"
    # The step is at ix = n_x // 2; with voxel-centre coordinates (no jitter)
    # lateral scatterers sit exactly at x = origin_x + (n_x // 2) * voxel_size.
    step_x = vol.origin_x + (n_x // 2) * vol.voxel_size
    in_bin = np.isclose(x_s, step_x, atol=1e-12)
    assert np.any(in_bin), (
        f"expected scatterers at x = {step_x}; "
        f"got x range [{x_s.min()}, {x_s.max()}]"
    )


if __name__ == "__main__":
    test_spreading_helper_monotonic()
    print("ok: spreading helper")
    test_directivity_shape()
    print("ok: directivity shape")
    test_directivity_separability()
    print("ok: directivity separability")
    test_voxel_extractor_lateral_boundary()
    print("ok: voxel extractor lateral boundary")
    test_analytic_tof_offplane_point()
    print("ok: analytic TOF off-plane")
    test_spreading_is_inverse_r_squared()
    print("ok: 1/r² spreading")
    test_reciprocity()
    print("ok: reciprocity")
    print("All engine_3d tests passed.")
