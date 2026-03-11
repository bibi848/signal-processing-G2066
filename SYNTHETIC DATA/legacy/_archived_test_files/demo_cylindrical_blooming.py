#!/usr/bin/env python3
"""
Demonstrate the new cylindrical blooming behavior.

Shows how surface markers at top/bottom of cylindrical defects create
blooming when Hilbert envelope is applied, simulating experimental ultrasound data.
"""

import sys
import os

# Qt configuration
try:
    import site
    qt_plugin_path = None
    site_packages = site.getsitepackages()
    
    for sp in site_packages:
        potential_path = os.path.join(sp, 'PyQt5', 'Qt5', 'plugins')
        if os.path.exists(potential_path):
            qt_plugin_path = potential_path
            break
    
    if qt_plugin_path:
        os.environ['QT_PLUGIN_PATH'] = qt_plugin_path
        print(f"✓ Qt configured")
except Exception as e:
    print(f"Warning: {e}")

import numpy as np
import napari
from importlib import import_module
import importlib

# Import the generator - FORCE RELOAD to avoid cache
sys.path.insert(0, os.path.dirname(__file__))
if '3d synthetic data v2' in sys.modules:
    importlib.reload(sys.modules['3d synthetic data v2'])
    module = sys.modules['3d synthetic data v2']
else:
    module = import_module('3d synthetic data v2')
SyntheticVolumeGenerator = module.SyntheticVolumeGenerator

print("\n" + "="*80)
print("CYLINDRICAL BLOOMING DEMONSTRATION")
print("="*80)

# Create a simple volume with cylindrical defects
print("\nCreating volume with cylindrical defects...")
generator = SyntheticVolumeGenerator(dimensions=(100, 100, 200), seed=42)

# Add cylindrical defects in different orientations
# These will show elongated blooming artifacts running along their length
print("  - Cylinder along Y-axis (vertical)")
generator.add_cylindrical_void(center_pos=40, other_pos=100, radius=8, intensity=0.95, axis='y')
print("  - Cylinder along X-axis (horizontal)")
generator.add_cylindrical_void(center_pos=60, other_pos=100, radius=6, intensity=0.92, axis='x')

# Generate clean volume
volume_clean = generator.generate(base_intensity_range=(0.05, 0.15), smoothing_sigma=2.0)

print("\nComparing blooming approaches...")
print("-" * 80)

# Approach 1: WITH surface blooming at top/bottom of cylinders
print("\n1. WITH direct 2D Gaussian blooming...")
volume_with_ghosts = generator.add_ultrasonic_artifacts(
    volume_clean,
    electronic_noise_level=0.01,
    grain_noise_level=0.02,
    depth_attenuation=0.2,
    speckle_noise_level=0.03,
    blur_sigma=(0.5, 1.0, 2.5),
    cylindrical_bloom=True,  # Add direct blooming
    cylindrical_bloom_params={
        'top_intensity': 0.60,      # Strong at top surface (closer to array)
        'bottom_intensity': 0.40,   # Weaker at bottom surface (further from array)
        'bloom_sigma': 15.0         # Gaussian sigma for X-Y spreading (pixels)
    }
)

# Apply Hilbert envelope - this spreads blooming in X-Y plane
volume_with_bloom_hilbert = generator.apply_hilbert_envelope(volume_with_ghosts)
volume_with_bloom_db = generator.convert_to_db(volume_with_bloom_hilbert, vmin=-40.0, vmax=0.0)

# Approach 2: WITHOUT surface markers (for comparison)
print("2. WITHOUT surface markers (standard Hilbert)...")
volume_without_markers = generator.add_ultrasonic_artifacts(
    volume_clean,
    electronic_noise_level=0.01,
    grain_noise_level=0.02,
    depth_attenuation=0.2,
    speckle_noise_level=0.03,
    blur_sigma=(0.5, 1.0, 2.5),
    cylindrical_bloom=False  # NO surface markers
)

volume_without_bloom_hilbert = generator.apply_hilbert_envelope(volume_without_markers)
volume_without_bloom_db = generator.convert_to_db(volume_without_bloom_hilbert, vmin=-40.0, vmax=0.0)

print("\n" + "="*80)
print("VISUALIZING IN NAPARI")
print("="*80)
print("\nLayers:")
print("  1. 'With Blooming' - DIRECT 2D Gaussian blooming from cylinder surfaces")
print("     - Blooming created DIRECTLY in X-Y plane (not relying on Hilbert spreading)")  
print("     - Top surface (z = center + radius): Strong Gaussian bloom (sigma=15 pixels)")
print("     - Bottom surface (z = center - radius): Weaker Gaussian bloom")
print("     - Extends along ENTIRE cylinder length")
print("")
print("  2. 'Without Blooming' - Standard processing (no blooming)")
print("\nToggle layers on/off to compare!")
print("Look for bright blooming regions extending from cylinder top/bottom surfaces.")
print("These should spread visibly in X and Y directions.")
print("Close the napari window to exit.")
print("="*80)

# Visualize side-by-side
viewer = napari.Viewer()

viewer.add_image(
    volume_with_bloom_db,
    name="With Blooming (NEW - experimental)",
    colormap="viridis",
    contrast_limits=(-40, 0),
    blending='additive',
    opacity=1.0
)

viewer.add_image(
    volume_without_bloom_db,
    name="Without Blooming (standard)",
    colormap="inferno",
    contrast_limits=(-40, 0),
    blending='additive',
    opacity=0.0,  # Start hidden
    visible=False
)

print("\nNapari tips:")
print("  - Use left panel to toggle layer visibility")
print("  - Adjust opacity sliders to compare")
print("  - Rotate view to see blooming at cylinder edges\n")

napari.run()

print("\n✓ Demonstration complete!")
