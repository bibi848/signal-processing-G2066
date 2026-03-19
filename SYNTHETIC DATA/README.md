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
Generates multi-position overlapping scan datasets from a single large specimen. Uses the origin-shift trick (zero-copy shared numpy arrays) to scan different regions. Configurable parameters:
- **Specimen**: dimensions, material, defects
- **Array**: element count, pitch, frequency, bandwidth, SNR
- **Grain structure**: mean grain size, impedance/wavespeed variation, voxel resolution
- **Scan grid**: number of positions, overlap fraction (min 20%), angular samples per rotation
- **Mode**: `'2d'` (B-scans only) or `'3d'` (B-scans + inverse Radon reconstruction)

### `sweep_datasets.py`
Runs `generate_dataset()` over a grid of parameter combinations for systematic performance analysis. Supports resume from interruption. Example sweep axes: overlap fraction, grain size, frequency, SNR, element count.

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

## Quick Start

### Single 3D scan with reconstruction
```bash
python simulate.py
```

### Multi-position dataset for stitching
```bash
python generate_dataset.py
```

### View existing dataset in napari
```bash
python view_dataset.py                          # auto-finds latest
python view_dataset.py output/dataset_xxx/      # specific dataset
python view_dataset.py --layer overlay           # grain + signal overlay
```

### Parameter sweep
```bash
python sweep_datasets.py
```

## Output Structure

```
output/
  dataset_YYYYMMDD_HHMMSS/
    dataset_meta.json          # Full parameter record
    ground_truth_full.npz      # Large specimen ground truth (optional)
    pos_000/
      bscan_000.npy ... bscan_031.npy   # Rotational B-scans (dB)
      scan_meta.npy                      # Per-scan metadata
      ground_truth.npz                   # Sub-volume ground truth
      recon_volume.npy                   # 3D reconstruction (z, y, x)
      recon_volume_zxy.npy               # Transposed for Stitch3D (z, x, y)
      reconstruction_summary.png         # Diagnostic figure
      position_meta.json                 # Position coords + metrics
    pos_001/
      ...
```

## Key Parameters (defaults)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_elements` | 64 | Array elements |
| `element_pitch` | 0.6 mm | Element spacing |
| `frequency` | 10 MHz | Centre frequency |
| `n_scans` | 32 | Angular samples per rotation |
| `mean_grain_size_m` | 0.5 mm | Voronoi grain diameter |
| `impedance_variation` | 0.025 | Per-grain Z spread |
| `overlap_fraction` | 0.3 | Minimum 20% enforced |
| `material` | ALUMINUM | c_L=6320, c_S=3130, rho=2700 |

---
*Last updated: 18 Mar 2026*
