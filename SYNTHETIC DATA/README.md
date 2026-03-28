# SYNTHETIC DATA — NDT Simulation & Dataset Generation

Phased-array ultrasonic simulation pipeline for generating synthetic FMC/TFM data with realistic grain microstructure. Designed for validating 3D stitching algorithms.

## Pipeline Overview

```
engine/          Ray-tracing physics engine (FMC + Born scattering)
    |
run_engine.py    Entry point — single scan or rotational 3D scan
    |
simulate.py      Quick single-scan demo (calls run_engine)
    |
reconstruct_3d.py   Inverse Radon transform: rotational B-scans → 3D volume
    |
generate_dataset.py  Multi-position dataset generation (overlapping cubes)
    |
sweep_datasets.py    Parameter sweeps across grain size, frequency, overlap, etc.
    |
view_dataset.py      Napari viewer for existing datasets
```

## Scripts

### `run_engine.py`
Entry point for the simulation engine. Runs FMC acquisition + TFM imaging for either a single 2D B-scan or a full 3D rotational scan (`scan_volume_3d()`).

### `simulate.py`
Quick-start demo — generates a single 3D rotational scan with default ALUMINUM parameters, then reconstructs and compares to ground truth.

### `reconstruct_3d.py`
Reconstructs a 3D volume from rotational B-scans using slice-by-slice inverse Radon transform (filtered back-projection). Includes:
- Lateral Tukey taper to suppress edge artifacts
- Soft circular apodisation to eliminate ring artifacts from `iradon(circle=True)`
- Cylinder-to-cube cropping for stitching compatibility
- Quantitative comparison to ground truth (SSIM, NRMSE, Pearson r, CNR)
- Napari visualisation with overlay modes

### `generate_dataset.py`
Generates multi-position overlapping scan datasets from a single large specimen. Uses the origin-shift trick (zero-copy shared numpy arrays) to scan different regions.

### `sweep_datasets.py`
Runs `generate_dataset()` over a grid of parameter combinations for systematic performance analysis. Supports resume from interruption.

### `view_dataset.py`
Standalone napari viewer for previously generated datasets. Loads all position volumes with correct spatial offsets. Supports `--layer` flag: `reconstruction`, `ground_truth`, `both`, or `overlay`.

## Engine Modules (`engine/`)

| Module | Description |
|--------|-------------|
| `config.py` | Dataclass configs (MaterialProperties, ArrayConfig, SpecimenConfig, SimulationConfig, ScanPlanConfig) |
| `materials.py` | Material presets (ALUMINUM, STEEL_MILD, WATER, AIR), acoustic impedance |
| `waveforms.py` | Gabor pulse generation, vectorised A-scan synthesis |
| `interfaces.py` | Snell's law, Fresnel coefficients, Zoeppritz (Auld formulation) |
| `propagation.py` | 2D geometric spreading (1/sqrt(r)), attenuation, element directivity |
| `geometry.py` | Specimen2D/3D, CircularDefect, CrackDefect, SphericalDefect, CylindricalDefect, PlanarCrack3D |
| `scattering.py` | Kirchhoff surface scattering (physical optics approximation) |
| `rays.py` | Ray path data structures, skip/corner-trap TOF with mirror-image method |
| `fmc_engine.py` | FMCEngine: geometric defects + Born scattering from voxels |
| `voxel_volume.py` | VoxelVolume3D: 3D impedance grid, `slice_at_angle()`, `extract_born_scatterers()` |
| `microstructure.py` | `generate_grain_structure()` (Voronoi), `embed_geometric_defects()` |

---

## Using sweep_datasets()

### Quick Start

```bash
cd "SYNTHETIC DATA"

# Generate a single dataset (3 positions, default parameters)
python generate_dataset.py

# Run a parameter sweep (multiple datasets)
python sweep_datasets.py

# Preview sweep plan without generating
python sweep_datasets.py --dry-run
```

### Grid Mode (Cartesian product of all values)

```python
from sweep_datasets import sweep_datasets

sweep_datasets(
    sweep_params={
        'overlap_fraction': [0.3, 0.5],
        'mean_grain_size_m': [0.5e-3, 1.0e-3],
        'snr_db': [25.0, 40.0],
    },
    base_params={
        'width_total': 100e-3,
        'depth_total': 60e-3,
        'thickness': 50e-3,
        'material': 'ALUMINUM',
        'num_elements': 64,
        'element_pitch': 0.63e-3,
        'frequency': 10e6,
        'n_positions_x': 3,
        'n_positions_y': 0,
        'n_scans': 32,
        'mode': '3d',
        'tfm_n_pixels': 400,
        'save_full_volume': True,
    },
    n_realisations=3,     # 3 grain seeds per combo
    show_napari=False,     # no GUI for overnight runs
)
# Total: 2 x 2 x 2 x 3 = 24 datasets
```

### Explicit Mode (non-rectangular designs)

```python
sweep_datasets(
    run_list=[
        {'overlap_fraction': 0.2, 'snr_db': 30, 'bandwidth': 0.03},
        {'overlap_fraction': 0.5, 'snr_db': 40, 'bandwidth': 0.9},
    ],
    base_params={...},
)
# Total: exactly 2 datasets
```

### Multiple Grain Realisations

Use `n_realisations` to generate the same parameter set with different grain structures (seeds 0, 1, 2, ...). This gives statistical confidence in stitching performance metrics.

```python
sweep_datasets(
    sweep_params={'overlap_fraction': [0.3, 0.5]},
    base_params={...},
    n_realisations=5,   # 2 overlaps x 5 seeds = 10 datasets
)
```

### Resume Capability

If a sweep crashes mid-run, re-run the same command with the same `output_root`. The sweep reads `sweep_index.json` and skips completed runs automatically.

```python
sweep_datasets(
    sweep_params={...},
    base_params={...},
    output_root='output/sweep_20260323_120000',  # same dir as before
)
```

---

## Sweepable Parameters

Any parameter below can be placed in `sweep_params` or `run_list`.
Fixed values go in `base_params`.

### Specimen Geometry

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `width_total` | m | 100e-3 | Full specimen x-extent |
| `depth_total` | m | 60e-3 | Full specimen y-extent |
| `thickness` | m | 50e-3 | Specimen z-thickness |
| `material` | str | 'ALUMINUM' | 'ALUMINUM', 'STEEL_MILD', 'STEEL_STAINLESS' |

### Grain Structure

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mean_grain_size_m` | m | 0.5e-3 | Mean Voronoi grain diameter |
| `impedance_variation` | frac | 0.025 | Per-grain impedance spread (e.g. 0.025 = +/-2.5%) |
| `wavespeed_variation` | frac | 0.005 | Per-grain wave speed spread |
| `voxel_fraction` | frac | 1/3 | Voxel size as fraction of wavelength |
| `seed` | int | 42 | RNG seed (use `n_realisations` to auto-vary) |

### Array

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_elements` | int | 64 | Number of array elements |
| `element_pitch` | m | 0.6e-3 | Centre-to-centre element spacing |
| `element_width` | m | None | Active element width (default: 0.9 x pitch) |
| `frequency` | Hz | 10e6 | Centre frequency |
| `bandwidth` | frac | 0.6 | Fractional bandwidth (0.03 = 3%, 0.9 = 90%) |

### FMC Acquisition

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `snr_db` | dB | 35.0 | Signal-to-noise ratio |
| `add_noise` | bool | True | Toggle noise on/off |
| `grain_noise_level` | frac | 0.05 | Grain scattering amplitude relative to signal |
| `time_samples` | int | 2048 | Number of time samples per A-scan |
| `sampling_frequency` | Hz | None | Sample rate (default: 4x centre frequency) |

### Filtering

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `filter_alpha` | frac | 1.0 | Tukey window taper (0=rectangular, 1=Hann) |
| `hanning_bool` | bool | False | Pre-window signal with Hanning before FFT |

### Physics

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_bounces` | int | 2 | Number of ray bounces to simulate |
| `mode_conversion` | bool | True | Enable L-to-S mode conversion at back wall |

### Scan Plan

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_scans` | int | 32 | Angular frames per rotational scan |
| `theta_start` | rad | -pi/2 | Start angle |
| `theta_end` | rad | +pi/2 | End angle |

### Scan Grid

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_positions_x` | int | 3 | Grid positions along x |
| `n_positions_y` | int | 2 | Grid positions along y (0 = 1D line) |
| `overlap_fraction` | frac | 0.3 | Overlap between adjacent cubes (min 0.2) |

### TFM Reconstruction

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mode` | str | '3d' | '2d' (B-scans only) or '3d' (+ inverse Radon) |
| `tfm_z_start` | m | 10e-3 | TFM start depth |
| `tfm_z_end` | m | None | TFM end depth (default: thickness - 5mm) |
| `tfm_n_pixels` | int | 800 | TFM pixel grid size (square) |
| `tfm_db_range` | dB | -40.0 | Display dynamic range |

### Output

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `save_full_volume` | bool | False | Save the full large specimen ground truth |

---

## Output Structure

```
output/
  sweep_YYYYMMDD_HHMMSS/
    sweep_index.json                    # Sweep progress (resume-capable)

    run_000_ovlp0.3_grain0.0005_seed0/
      dataset_meta.json                 # All parameters (reproducibility)
      ground_truth_full.npz             # Full specimen volume (optional)

      pos_000/
        bscan_0000.npy ... bscan_0031.npy   # TFM B-scans (dB scale)
        fmc_0000.npy ... fmc_0031.npy        # Raw FMC data
        scan_meta.npy                        # Angles, depths, aperture
        ground_truth.npz                     # Position ground truth
        recon_volume.npy                     # 3D reconstruction (z, y, x)
        recon_volume_zxy.npy                 # Transposed for Stitch3D
        position_meta.json                   # Position coords + metrics

      pos_001/ ... pos_NNN/

  dataset_YYYYMMDD_HHMMSS/              # Single dataset (from generate_dataset.py)
    (same structure as above)
```

## Loading Results for Stitching

```python
import numpy as np
import json

# Load two adjacent reconstructed volumes
vol1 = np.load('output/.../pos_000/recon_volume_zxy.npy')
vol2 = np.load('output/.../pos_001/recon_volume_zxy.npy')

# Stitch using cross-correlation
from Classes.Stitch3D import normalised_correlation_3D, stitch_volumes
best_shift, shifts, corr = normalised_correlation_3D(vol1, vol2, axis='x')
stitched = stitch_volumes(vol1, vol2, best_shift, axis='x')

# Compare to ground truth positions from metadata
with open('output/.../dataset_meta.json') as f:
    meta = json.load(f)
positions = meta['scan_grid']['positions']
expected_shift_m = positions[1]['px_m'] - positions[0]['px_m']
```

## Real Experimental Configurations

Parameters from DATA/ Params.txt files for reference:

| Experiment | Elements | Pitch | Freq | BW | alpha | Hanning |
|-----------|---------|-------|------|-----|-------|---------|
| Al Pure 10MHz | 64 | 0.63mm | 10MHz | 3% | 1.0 | False |
| Al Pure 15MHz | 128 | 0.3mm | 15MHz | 3% | 1.0 | False |
| Al Hole 5MHz | 64 | 0.63mm | 5MHz | 0.2% | 0.2 | True |
| Cu Pure 10MHz | 64 | 0.63mm | 10MHz | 3% | 1.0 | False |
| Cu Pure 7.5MHz | 128 | 0.77mm | 7.5MHz | 0.8% | 1.0 | False |

---
*Last updated: 23 Mar 2026*
