#!/usr/bin/env python3
"""Check subvolume sizes from the generated data"""

import numpy as np
import os

save_dir = "SYNTHETIC NPY/stitching_manual"

if not os.path.exists(save_dir):
    print(f"ERROR: Directory {save_dir} not found!")
    print("Please run '3d synthetic data v2.py' first to generate the subvolumes.")
    exit(1)

# Load reconstruction info
recon_info = np.load(os.path.join(save_dir, 'reconstruction_info.npz'), allow_pickle=True)
print("="*80)
print("SUBVOLUME SIZE ANALYSIS")
print("="*80)

print(f"\nOriginal volume shape: {tuple(recon_info['original_shape'])}")
print(f"Number of splits: {tuple(recon_info['num_splits'])}")

if 'subvol_shape' in recon_info:
    print(f"Target fixed subvolume shape: {tuple(recon_info['subvol_shape'])}")

if 'varied_overlaps' in recon_info:
    overlaps = recon_info['varied_overlaps'].item()
    print(f"\nVaried overlaps:")
    print(f"  Z overlaps: {overlaps['z_overlaps']}")
    print(f"  Y overlaps: {overlaps['y_overlaps']}")
    print(f"  X overlaps: {overlaps['x_overlaps']}")

# Find all subvolume files
subvol_files = sorted([f for f in os.listdir(save_dir) 
                       if f.startswith('subvol_') and f.endswith('.npy') and 'meta' not in f])

print(f"\n{'='*80}")
print("ACTUAL SUBVOLUME SIZES:")
print(f"{'='*80}")

sizes = []
for f in subvol_files:
    # Load volume
    vol_path = os.path.join(save_dir, f)
    volume = np.load(vol_path)
    
    # Load metadata
    meta_path = vol_path.replace('.npy', '_meta.npy.npz')
    meta_data = np.load(meta_path, allow_pickle=True)
    
    index = tuple(meta_data['index'])
    origin = tuple(meta_data['origin'])
    overlap = tuple(meta_data['overlap'])
    
    print(f"\nSubvolume {index}:")
    print(f"  Actual shape: {volume.shape}")
    print(f"  Origin: {origin}")
    print(f"  Overlap with next: {overlap}")
    
    sizes.append(volume.shape)

# Check if all sizes are the same
print(f"\n{'='*80}")
if len(set(sizes)) == 1:
    print("✓ ALL SUBVOLUMES HAVE THE SAME SIZE")
    print(f"  Size: {sizes[0]}")
else:
    print("⚠ SUBVOLUMES HAVE DIFFERENT SIZES")
    unique_sizes = set(sizes)
    for size in unique_sizes:
        count = sizes.count(size)
        print(f"  {count} subvolume(s) with size: {size}")
    
    # Explain why
    print("\nNote: The last subvolume may be smaller due to boundary clipping.")
    print("This is expected when the volume dimension is not perfectly divisible.")
print("="*80)
