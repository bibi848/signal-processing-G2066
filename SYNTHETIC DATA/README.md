# SYNTHETIC DATA Directory

## Active Scripts

### Main Scripts
- **ray_tracing_ndt_2d.py** **NEW** - 2D NDT simulation with 1D linear array (B-scan imaging using FMC+TFM)
- **ray_tracing_ndt.py** - 3D NDT simulation using 2D matrix array (realistic time-domain acquisition)
- **3d synthetic data v2.py** - Legacy synthetic data generator with angular imaging effects
- **open_synthetic_data.py** - Utility script to load and visualize synthetic .npy volumes in Napari

### Notebooks
- **3D Synthetic data.ipynb** - Jupyter notebook for interactive synthetic data generation

## Data Files

### Generated Volumes (.npy files)
- `synthetic_volume_clean.npy` - Clean volume without artifacts
- `synthetic_volume_db.npy` - dB-scaled volume
- `synthetic_volume_medium_artifacts.npy` - Volume with medium artifacts
- `synthetic_volume_with_artifacts.npy` - Volume with artifacts

## Directories

- **SYNTHETIC NPY/** - Additional synthetic data files organized by type (stitching tests, etc.)
- **_archived_test_files/** - Archived test scripts, guides, and backup files (kept for reference)

## Usage

### Generate 2D B-scan with FMC+TFM (RECOMMENDED - Fast):
```bash
python ray_tracing_ndt_2d.py
```
**Output**: B-scan image showing defects in 2D cross-section (~3 min runtime)

### Generate 3D volume with realistic NDT data:
```bash
python ray_tracing_ndt.py
```
**Output**: 3D subvolumes with TFM reconstruction (slower, for advanced simulations)

### Visualize existing .npy volumes:
```bash
python open_synthetic_data.py
```

---
*Last updated: 20 Feb 2026*
