#!/usr/bin/env python3
"""
Simple test to verify ghost cylinders work and create blooming.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from importlib import import_module
import importlib

# Force reload
if '3d synthetic data v2' in sys.modules:
    importlib.reload(sys.modules['3d synthetic data v2'])
module = import_module('3d synthetic data v2')
SyntheticVolumeGenerator = module.SyntheticVolumeGenerator

print("="*80)
print("GHOST CYLINDER VERIFICATION TEST")
print("="*80)

# Create small test volume
gen = SyntheticVolumeGenerator(dimensions=(60, 60, 120), seed=42)

# Add ONE cylinder along Y-axis for clarity
print("\nAdding cylinder along Y-axis...")
print("  Position: z=30, x=60, radius=6")
gen.add_cylindrical_void(center_pos=30, other_pos=60, radius=6, intensity=0.92, axis='y')

# Generate clean volume
vol_clean = gen.generate(base_intensity_range=(0.05, 0.15), smoothing_sigma=1.5)
print(f"Clean volume range: {vol_clean.min():.4f} - {vol_clean.max():.4f}")

# Add artifacts WITHOUT blooming
print("\n--- WITHOUT ghost cylinders ---")
vol_no_ghosts = gen.add_ultrasonic_artifacts(
    vol_clean,
    electronic_noise_level=0.01,
    grain_noise_level=0.02,
    depth_attenuation=0.2,
    speckle_noise_level=0.03,
    blur_sigma=(0.5, 1.0, 2.5),
    cylindrical_bloom=False  # NO ghosts
)

# Add artifacts WITH blooming
print("\n--- WITH ghost cylinders ---")
vol_with_ghosts = gen.add_ultrasonic_artifacts(
    vol_clean,
    electronic_noise_level=0.01,
    grain_noise_level=0.02,
    depth_attenuation=0.2,
    speckle_noise_level=0.03,
    blur_sigma=(0.5, 1.0, 2.5),
    cylindrical_bloom=True,  # Add ghosts
    cylindrical_bloom_params={
        'top_intensity': 0.35,
        'bottom_intensity': 0.20,
        'bloom_offset': 8,
        'bloom_thickness': 3
    }
)

# Check difference BEFORE Hilbert
diff_before = vol_with_ghosts - vol_no_ghosts
print(f"\nDifference before Hilbert:")
print(f"  Max added intensity: {diff_before.max():.4f}")
print(f"  Voxels with added intensity > 0.1: {np.sum(diff_before > 0.1)}")

# Check specific locations
# Cylinder at z=30, ghosts should be at z=30±8 = z=38 and z=22
print(f"\nGhost locations (cylinder at z=30, offset=8):")
print(f"  Top ghost (z=38) max: {diff_before[38, :, :].max():.4f}")
print(f"  Bottom ghost (z=22) max: {diff_before[22, :, :].max():.4f}")
print(f"  Real cylinder (z=30) should be 0: {diff_before[30, :, :].max():.4f}")

# Now apply Hilbert
print("\n--- Applying Hilbert envelope ---")
vol_no_ghosts_hilbert = gen.apply_hilbert_envelope(vol_no_ghosts)
vol_with_ghosts_hilbert = gen.apply_hilbert_envelope(vol_with_ghosts)

# Convert to dB
vol_no_ghosts_db = gen.convert_to_db(vol_no_ghosts_hilbert, vmin=-40.0, vmax=0.0)
vol_with_ghosts_db = gen.convert_to_db(vol_with_ghosts_hilbert, vmin=-40.0, vmax=0.0)

# Check difference AFTER Hilbert
diff_after = vol_with_ghosts_db - vol_no_ghosts_db
print(f"\nDifference after Hilbert (dB scale):")
print(f"  Max difference: {diff_after.max():.4f} dB")
print(f"  Min difference: {diff_after.min():.4f} dB")
print(f"  Mean absolute difference: {np.abs(diff_after).mean():.4f} dB")

# Check if blooming spread from ghost locations
print(f"\nBlooming at ghost locations (dB scale):")
print(f"  Top ghost region (z=38): {diff_after[38, :, :].max():.2f} dB")
print(f"  Bottom ghost region (z=22): {diff_after[22, :, :].max():.2f} dB")

print("\n" + "="*80)
if diff_before.max() > 0.15:
    print("✓ Ghost cylinders were added successfully!")
else:
    print("✗ Ghost cylinders were NOT added!")

if np.abs(diff_after).mean() > 0.5:
    print("✓ Blooming visible after Hilbert transform!")
else:
    print("✗ No significant blooming after Hilbert transform!")
print("="*80)
