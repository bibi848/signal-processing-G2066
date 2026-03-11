#!/usr/bin/env python3
"""Simple test to verify visualize_sub_volumes works"""

import sys
import os

# Qt configuration (same as main script)
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
        print(f"✓ Qt configured: {qt_plugin_path}")
except Exception as e:
    print(f"Warning: {e}")

import numpy as np
from importlib import import_module

# Import the generator class
sys.path.insert(0, os.path.dirname(__file__))
module = import_module('3d synthetic data v2')
SyntheticVolumeGenerator = module.SyntheticVolumeGenerator

# Create simple test data
print("\nCreating test subvolumes...")
test_subvolumes = []

for i in range(3):
    sub_vol = {
        'index': (0, 0, i),
        'volume': np.random.rand(50, 50, 50) * 0.2,  # Small random volume
        'origin': (0, 0, i * 40),  # Offset along x-axis
        'overlap': (0, 0, 10),
        'subvolume_id': i
    }
    test_subvolumes.append(sub_vol)
    print(f"  Created subvolume {i}: origin={sub_vol['origin']}, shape={sub_vol['volume'].shape}")

# Create generator instance
generator = SyntheticVolumeGenerator(dimensions=(50, 50, 150))

# Test visualization
print("\nTesting visualization...")
generator.visualize_sub_volumes(test_subvolumes)

print("\nTest complete!")
