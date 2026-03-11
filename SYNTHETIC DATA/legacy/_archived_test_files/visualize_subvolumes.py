#!/usr/bin/env python3
"""
Test napari visualization of subvolumes.
Run this script to verify that napari visualization works correctly.
Close the napari window to exit.
"""

import sys
import os
import numpy as np

# Qt configuration (must be done before importing napari)
print("Configuring Qt...")
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
    else:
        print("WARNING: Qt plugin path not found")
except Exception as e:
    print(f"Warning: {e}")

print("\nImporting napari...")
import napari

print("\nLoading saved subvolumes from SYNTHETIC NPY/stitching_manual/...")

# Load the saved subvolumes
save_dir = "SYNTHETIC NPY/stitching_manual"

if not os.path.exists(save_dir):
    print(f"ERROR: Directory {save_dir} not found!")
    print("Please run '3d synthetic data v2.py' first to generate the subvolumes.")
    sys.exit(1)

# Find all subvolume files
subvol_files = sorted([f for f in os.listdir(save_dir) if f.startswith('subvol_') and f.endswith('.npy') and 'meta' not in f])

if not subvol_files:
    print(f"ERROR: No subvolume files found in {save_dir}")
    sys.exit(1)

print(f"Found {len(subvol_files)} subvolume files")

# Load subvolumes and their metadata
sub_volumes = []
for f in subvol_files:
    # Load volume
    vol_path = os.path.join(save_dir, f)
    volume = np.load(vol_path)
    
    # Load metadata
    meta_path = vol_path.replace('.npy', '_meta.npy')
    meta_data = np.load(meta_path + 'z', allow_pickle=True)
    
    # Extract metadata
    meta = {k: meta_data[k].item() if meta_data[k].ndim == 0 else meta_data[k] 
            for k in meta_data.files}
    meta['volume'] = volume
    
    sub_volumes.append(meta)
    print(f"  Loaded {f}: shape={volume.shape}, origin={meta['origin']}")

# Create napari viewer
print("\nCreating napari viewer...")
viewer = napari.Viewer()

colormaps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'turbo']

# Detect if CTFM was applied (negative values indicate dB scale)
sample_vol = sub_volumes[0]['volume']
is_db_scale = sample_vol.min() < 0
contrast_limits = (-40, 0) if is_db_scale else (0.0, 1.0)

print(f"Data scale: {'dB (CTFM applied)' if is_db_scale else 'Linear'}")
print(f"Contrast limits: {contrast_limits}")

# Add each subvolume as a layer
for i, sv in enumerate(sub_volumes):
    idx = sv['index']
    origin = sv['origin']
    vol_shape = sv['volume'].shape
    
    print(f"  Adding SubVol {idx}: shape={vol_shape}, origin={origin}")
    
    viewer.add_image(
        sv['volume'],
        name=f"SubVol_{idx[0]}_{idx[1]}_{idx[2]}",
        colormap=colormaps[i % len(colormaps)],
        contrast_limits=contrast_limits,
        blending='additive',
        opacity=0.6,
        translate=origin  # Spatial offset to position correctly
    )

print("\n" + "="*80)
print("Napari viewer opened!")
print("- Use mouse to rotate/zoom the 3D view")
print("- Toggle layer visibility in the left panel")
print("- Adjust opacity/contrast in the layer controls")
print("- Close the window to exit")
print("="*80)

napari.run()

print("\n✓ Visualization test complete!")
