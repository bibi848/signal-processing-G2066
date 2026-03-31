"""
Parametric Sweep for Synthetic Dataset Generation
==================================================

Generates multiple datasets by sweeping over parameter combinations
to study their effect on stitching performance.

Usage -- grid sweep (Cartesian product of all values):
    python sweep_datasets.py

Usage -- as a library:
    from sweep_datasets import sweep_datasets

    sweep_datasets(
        sweep_params={
            'overlap_fraction': [0.2, 0.3, 0.5],
            'impedance_variation': [0.01, 0.025, 0.05],
        },
        base_params={
            'num_elements': 64, 'n_scans': 32, 'mode': '3d',
        },
    )

Usage -- explicit run list (non-rectangular designs):
    sweep_datasets(
        run_list=[
            {'overlap_fraction': 0.2, 'snr_db': 30},
            {'overlap_fraction': 0.5, 'snr_db': 40},
        ],
        base_params={...},
    )
"""

import itertools
import json
import os
import sys
import traceback
from datetime import datetime
from typing import Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from engine.materials import ALUMINUM, STEEL_MILD, STEEL_STAINLESS
from generate_dataset import generate_dataset

# String-to-object lookup for materials
MATERIAL_PRESETS = {
    'ALUMINUM': ALUMINUM,
    'STEEL_MILD': STEEL_MILD,
    'STEEL_STAINLESS': STEEL_STAINLESS,
}


def _build_run_label(index: int, varied_params: dict) -> str:
    """Build a short directory name from the varied parameters."""
    parts = [f'run_{index:03d}']
    for key, val in varied_params.items():
        short_key = key.replace('_fraction', '').replace('_variation', '_var')
        short_key = short_key.replace('impedance', 'Z').replace('wavespeed', 'c')
        short_key = short_key.replace('mean_grain_size_m', 'grain')
        short_key = short_key.replace('num_elements', 'nelem')
        short_key = short_key.replace('frequency', 'freq')
        short_key = short_key.replace('overlap', 'ovlp')
        short_key = short_key.replace('grain_noise_level', 'gnoise')
        short_key = short_key.replace('time_samples', 'tsamp')
        short_key = short_key.replace('sampling', 'fs')
        short_key = short_key.replace('max_bounces', 'bounces')
        short_key = short_key.replace('mode_conversion', 'mconv')
        short_key = short_key.replace('tfm_db_range', 'dbrange')
        short_key = short_key.replace('tfm_n_pixels', 'tfmpx')
        short_key = short_key.replace('n_scans', 'nscans')
        short_key = short_key.replace('filter_alpha', 'alpha')
        short_key = short_key.replace('snr_db', 'snr')
        if key == 'seed':
            short_key = 'seed'
        if isinstance(val, float):
            parts.append(f'{short_key}{val:g}')
        else:
            parts.append(f'{short_key}{val}')
    return '_'.join(parts)


def _resolve_material(params: dict) -> dict:
    """Convert string material names to MaterialProperties objects."""
    if 'material' in params and isinstance(params['material'], str):
        name = params['material'].upper()
        if name not in MATERIAL_PRESETS:
            raise ValueError(
                f"Unknown material '{params['material']}'. "
                f"Available: {list(MATERIAL_PRESETS.keys())}"
            )
        params = dict(params)
        params['material'] = MATERIAL_PRESETS[name]
    return params


def _make_json_serialisable(params: dict) -> dict:
    """Convert params dict to JSON-safe types for the index."""
    out = {}
    for k, v in params.items():
        if hasattr(v, 'name'):  # MaterialProperties
            out[k] = v.name
        elif isinstance(v, list):
            out[k] = [
                x.name if hasattr(x, 'name') else x for x in v
            ]
        else:
            out[k] = v
    return out


def _load_existing_index(index_path: str) -> dict:
    """Load existing sweep index for resume capability."""
    if os.path.exists(index_path):
        with open(index_path) as f:
            return json.load(f)
    return None


def _save_index(index_path: str, index_data: dict) -> None:
    """Write sweep index to disk."""
    with open(index_path, 'w') as f:
        json.dump(index_data, f, indent=2)


def sweep_datasets(
    sweep_params: Optional[Dict[str, list]] = None,
    run_list: Optional[List[dict]] = None,
    base_params: Optional[dict] = None,
    output_root: Optional[str] = None,
    n_realisations: int = 1,
    dry_run: bool = False,
    show_napari: bool = False,
) -> str:
    """
    Generate multiple datasets by sweeping over parameter combinations.

    Provide exactly one of ``sweep_params`` (grid mode) or ``run_list``
    (explicit mode).

    Args:
        sweep_params: Dict mapping parameter names to lists of values.
                      All combinations are generated (Cartesian product).
                      Example: {'overlap_fraction': [0.2, 0.3], 'snr_db': [30, 40]}
                      → 4 runs.
        run_list:     Explicit list of parameter dicts. Each dict contains
                      only the parameters that differ from base_params.
        base_params:  Default values for all non-swept parameters.
                      These are passed directly to generate_dataset().
        output_root:  Parent directory for all runs.
                      Default: output/sweep_<timestamp>/
        n_realisations: Number of grain structure realisations per parameter
                      combination. Each realisation uses a different RNG seed
                      (0, 1, 2, ...) so the grain geometry varies while all
                      other parameters stay fixed. Default 1 (no repeats).
        dry_run:      If True, print the plan without generating anything.
        show_napari:  If True, open napari viewer after each run (blocks).
                      Default False for unattended overnight runs.

    Returns:
        Path to the sweep output directory.
    """
    # Validate inputs
    if (sweep_params is None) == (run_list is None):
        raise ValueError(
            "Provide exactly one of sweep_params (grid) or run_list (explicit)"
        )

    if base_params is None:
        base_params = {}

    # Build list of per-run parameter overrides
    if sweep_params is not None:
        param_names = list(sweep_params.keys())
        param_values = [sweep_params[k] for k in param_names]
        overrides = [
            dict(zip(param_names, combo))
            for combo in itertools.product(*param_values)
        ]
        swept_keys = set(param_names)
    else:
        overrides = run_list
        swept_keys = set()
        for run in run_list:
            swept_keys.update(run.keys())

    # Expand each parameter combo into n_realisations with different seeds
    if n_realisations > 1:
        swept_keys.add('seed')
        expanded = []
        for override in overrides:
            for r in range(n_realisations):
                entry = dict(override)
                entry['seed'] = r
                expanded.append(entry)
        overrides = expanded

    n_runs = len(overrides)

    # Output directory
    if output_root is None:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_root = os.path.join(
            os.path.dirname(__file__), 'output', f'sweep_{ts}',
        )

    # Print plan
    print(f"\n{'='*60}")
    print(f"  PARAMETER SWEEP — {n_runs} runs")
    if n_realisations > 1:
        n_combos = n_runs // n_realisations
        print(f"  ({n_combos} parameter combos × {n_realisations} grain realisations)")
    print(f"  Output: {output_root}")
    if sweep_params is not None:
        print(f"  Swept parameters:")
        for k, v in sweep_params.items():
            print(f"    {k}: {v}")
    print(f"{'='*60}\n")

    for i, override in enumerate(overrides):
        varied = {k: override[k] for k in override if k in swept_keys}
        label = _build_run_label(i, varied)
        print(f"  {label}: {varied}")

    if dry_run:
        print(f"\n  DRY RUN — no datasets generated\n")
        return output_root

    # Create output directory
    os.makedirs(output_root, exist_ok=True)

    # Check for existing index (resume)
    index_path = os.path.join(output_root, 'sweep_index.json')
    existing_index = _load_existing_index(index_path)
    completed_indices = set()
    if existing_index is not None:
        for run_info in existing_index.get('runs', []):
            if run_info.get('status') == 'complete':
                completed_indices.add(run_info['index'])
        if completed_indices:
            print(f"\n  Resuming: {len(completed_indices)} runs already complete")

    # Initialise index
    index_data = {
        'timestamp': datetime.now().isoformat(),
        'sweep_params': _make_json_serialisable(sweep_params or {}),
        'base_params': _make_json_serialisable(base_params),
        'n_realisations': n_realisations,
        'n_runs': n_runs,
        'runs': existing_index['runs'] if existing_index else [],
    }

    # Run each combination
    for i, override in enumerate(overrides):
        if i in completed_indices:
            print(f"\n  Skipping run {i} (already complete)")
            continue

        varied = {k: override[k] for k in override if k in swept_keys}
        label = _build_run_label(i, varied)
        run_dir = os.path.join(output_root, label)

        # Merge base + override
        run_params = dict(base_params)
        run_params.update(override)
        run_params['output_root'] = run_dir
        run_params['show_napari'] = show_napari
        run_params = _resolve_material(run_params)

        print(f"\n{'#'*60}")
        print(f"  RUN {i}/{n_runs - 1}  —  {label}")
        print(f"{'#'*60}")

        run_entry = {
            'index': i,
            'label': label,
            'params': _make_json_serialisable(run_params),
            'output_dir': run_dir,
            'status': 'running',
        }

        try:
            generate_dataset(**run_params)
            run_entry['status'] = 'complete'
        except Exception as e:
            run_entry['status'] = 'failed'
            run_entry['error'] = str(e)
            print(f"\n  ERROR in run {i}: {e}")
            traceback.print_exc()

        # Update index incrementally
        # Replace existing entry if resuming, otherwise append
        found = False
        for j, existing in enumerate(index_data['runs']):
            if existing['index'] == i:
                index_data['runs'][j] = run_entry
                found = True
                break
        if not found:
            index_data['runs'].append(run_entry)

        _save_index(index_path, index_data)

    # Summary
    n_complete = sum(1 for r in index_data['runs'] if r['status'] == 'complete')
    n_failed = sum(1 for r in index_data['runs'] if r['status'] == 'failed')

    print(f"\n{'='*60}")
    print(f"  SWEEP COMPLETE")
    print(f"  {n_complete}/{n_runs} runs succeeded"
          + (f", {n_failed} failed" if n_failed else ""))
    print(f"  Index: {index_path}")
    print(f"{'='*60}\n")

    return output_root


# ── Sweepable parameters reference ────────────────────────────────────
#
# Any parameter accepted by generate_dataset() can be swept.
# The full list:
#
#   SPECIMEN GEOMETRY
#     width_total          (m)    e.g. 100e-3
#     depth_total          (m)    e.g. 60e-3
#     thickness            (m)    e.g. 50e-3
#
#   MATERIAL & GRAIN STRUCTURE
#     material             str    'ALUMINUM', 'STEEL_MILD', 'STEEL_STAINLESS'
#     mean_grain_size_m    (m)    e.g. 0.3e-3, 0.5e-3, 1.0e-3, 2.0e-3
#     impedance_variation  frac   e.g. 0.01, 0.025, 0.05 (±Z spread per grain)
#     wavespeed_variation  frac   e.g. 0.002, 0.005, 0.01 (±c_L spread per grain)
#     voxel_fraction       frac   voxel size as fraction of wavelength (e.g. 1/3)
#     seed                 int    RNG seed for grain structure reproducibility
#
#   ARRAY
#     num_elements         int    e.g. 32, 64, 128
#     element_pitch        (m)    e.g. 0.3e-3, 0.6e-3, 1.0e-3
#     element_width        (m)    active element width (default: 0.9 × pitch)
#     frequency            (Hz)   e.g. 5e6, 10e6, 15e6
#     bandwidth            frac   e.g. 0.03, 0.6, 0.9  (real data: 0.03–0.9)
#
#   FMC ACQUISITION
#     snr_db               (dB)   e.g. 20, 30, 40
#     add_noise            bool   toggle noise on/off (default True)
#     grain_noise_level    frac   grain scattering amplitude relative to signal
#     time_samples         int    number of time samples per A-scan (e.g. 2048)
#     sampling_frequency   (Hz)   sample rate (default: 4× centre frequency)
#
#   FILTERING
#     filter_alpha         frac   Tukey taper (0=rect, 1=Hann). Real data: 0.2–1.0
#     hanning_bool         bool   Pre-window with Hanning. Real data: True for Al Hole
#
#   PHYSICS
#     max_bounces          int    ray bounces (default 2)
#     mode_conversion      bool   L→S mode conversion at back wall (default True)
#
#   SCAN PLAN
#     n_scans              int    angular frames per rotation (e.g. 16, 32, 64)
#     theta_start          (rad)  e.g . -np.pi/2
#     theta_end            (rad)  e.g.  np.pi/2
#
#   SCAN GRID
#     n_positions_x        int    grid positions along x
#     n_positions_y        int    grid positions along y (0 = 1D line)
#     overlap_fraction     frac   overlap between adjacent cubes (min 0.2)
#
#   TFM RECONSTRUCTION
#     mode                 str    '2d' (B-scans only) or '3d' (+ iradon recon)
#     tfm_z_start          (m)    TFM start depth
#     tfm_z_end            (m)    TFM end depth (None = thickness - 5mm)
#     tfm_n_pixels         int    TFM grid size (e.g. 400, 800)
#     tfm_db_range         (dB)   display dynamic range (e.g. -40, -20)
#
#   OUTPUT
#     save_full_volume     bool   save the full large ground truth volume


# ── Main ──────────────────────────────────────────────────────────────

def main():
    """
    Sweep matched to real experimental parameters.

    Real data configurations (from DATA/ Params.txt files):
      Al Pure 10MHz:  c=6700, 64 els, 0.63mm pitch, BW~3%, alpha=1.0
      Al Pure 15MHz:  c=6700, 128 els, 0.21mm pitch, BW~3%, alpha=1.0
      Al Hole 5MHz:   c=6700, 64 els, 0.63mm pitch, BW~0.2%, alpha=0.2, hanning=True
      Cu Pure 10MHz:  c=4700, 64 els, 0.63mm pitch, BW~3%, alpha=1.0
      Cu Pure 7.5MHz: c=4700, 128 els, 0.77mm pitch, BW~0.8%, alpha=1.0

    We sweep over the parameters that most affect stitching performance:
      - bandwidth (narrow vs wide filtering)
      - grain size (scattering regime)
      - overlap fraction
    with multiple grain realisations per combo for statistical confidence.
    """
    sweep_datasets(
        show_napari=False,
        # Multiple grain realisations per parameter combo to test
        # stitching robustness across different microstructures
        n_realisations=5,

        sweep_params={
            # Stitching overlap — primary variable for robustness study
            'overlap_fraction': [0.2, 0.3, 0.5],
        },

        # Base params aligned with run_engine.py defaults
        base_params={
            # Specimen
            'width_total': 100e-3,
            'depth_total': 60e-3,
            'thickness': 50e-3,
            # Material
            'material': 'ALUMINUM',
            # Array — matches run_engine.py defaults
            'num_elements': 64,
            'element_pitch': 0.6e-3,
            'frequency': 10e6,
            'bandwidth': 0.6,
            'snr_db': 35.0,
            # Filtering
            'filter_alpha': 1.0,
            'hanning_bool': False,
            # Grain structure
            'impedance_variation': 0.025,
            'mean_grain_size_m': 0.5e-3,
            # Scan
            'n_positions_x': 2,
            'n_positions_y': 0,
            'n_scans': 32,
            'mode': '3d',
            'tfm_z_start': 10e-3,
            'tfm_n_pixels': 800,
            'save_full_volume': True,
        },
    )


if __name__ == '__main__':
    main()
