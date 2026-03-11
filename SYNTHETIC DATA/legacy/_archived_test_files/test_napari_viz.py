#!/usr/bin/env python3
"""Test napari visualization"""

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
        print(f"✓ Set QT_PLUGIN_PATH: {qt_plugin_path}")
except Exception as e:
    print(f"Warning: Could not configure Qt paths: {e}")

import numpy as np
import napari

print("Creating test volume...")
# Create a simple test volume
test_vol = np.random.rand(50, 50, 50)

print("Creating napari viewer...")
viewer = napari.Viewer()

print("Adding image layer...")
viewer.add_image(test_vol, name="Test Volume")

print("Starting napari event loop...")
napari.run()

print("Napari closed successfully")
