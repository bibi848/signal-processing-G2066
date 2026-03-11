#!/usr/bin/env python3
"""Quick test to verify ghost cylinders are being created."""

import numpy as np
import sys
from importlib import import_module

# Import module
module = import_module('3d synthetic data v2')
gen = module.SyntheticVolumeGenerator(dimensions=(50, 50, 100), seed=42)

# Add a cylinder along Y-axis
gen.add_cylindrical_void(center_pos=25, other_pos=50, radius=5, intensity=0.9, axis='y')

# Create volume
vol = gen.generate(base_intensity_range=(0.05, 0.10), smoothing_sigma=1.0)

# Add ghost cylinders directly
vol_with_ghosts = gen.add_cylindrical_surface_markers(
    vol,
    top_intensity=0.3,
    bottom_intensity=0.2, 
    bloom_offset=8,
    bloom_thickness=3
)

# Analyze results
diff = vol_with_ghosts - vol
print(f'Volume range before ghosts: {vol.min():.4f} - {vol.max():.4f}')
print(f'Volume range after ghosts: {vol_with_ghosts.min():.4f} - {vol_with_ghosts.max():.4f}')
print(f'Max intensity added: {diff.max():.4f}')
print(f'Voxels with added intensity > 0.15: {np.sum(diff > 0.15)}')
print(f'Voxels with added intensity > 0.25: {np.sum(diff > 0.25)}')

# Check specific locations where ghosts should be
# Cylinder is at z=25, x=50, extends along Y (all Y values)
# Top ghost should be at z=25+8=33
# Bottom ghost should be at z=25-8=17

# Check a slice at z=33 (top ghost)
slice_top = diff[33, :, :]
print(f'\nTop ghost slice (z=33):')
print(f'  Max value: {slice_top.max():.4f}')
print(f'  Non-zero voxels: {np.sum(slice_top > 0.01)}')

# Check a slice at z=17 (bottom ghost)
slice_bottom = diff[17, :, :]
print(f'\nBottom ghost slice (z=17):')
print(f'  Max value: {slice_bottom.max():.4f}')
print(f'  Non-zero voxels: {np.sum(slice_bottom > 0.01)}')

# Check that ghosts extend along Y
if slice_top.max() > 0.1:
    # Find where intensity is added
    y_coords, x_coords = np.where(slice_top > 0.1)
    print(f'\nTop ghost extent:')
    print(f'  Y range: {y_coords.min()} to {y_coords.max()} (should span most of 0-49)')
    print(f'  X range: {x_coords.min()} to {x_coords.max()} (should be near 50)')
