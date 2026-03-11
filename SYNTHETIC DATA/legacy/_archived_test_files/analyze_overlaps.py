#!/usr/bin/env python3
"""Show detailed overlap analysis for variable overlap subvolumes"""

import numpy as np
import os

save_dir = "SYNTHETIC NPY/stitching_manual"

if not os.path.exists(save_dir):
    print(f"ERROR: Directory {save_dir} not found!")
    exit(1)

# Load reconstruction info
recon_info = np.load(os.path.join(save_dir, 'reconstruction_info.npz'), allow_pickle=True)
print("="*80)
print("OVERLAP ANALYSIS")
print("="*80)

print(f"\nOriginal volume shape: {tuple(recon_info['original_shape'])}")
print(f"Number of splits: {tuple(recon_info['num_splits'])}")
if 'subvol_shape' in recon_info:
    print(f"Fixed subvolume shape: {tuple(recon_info['subvol_shape'])}")

if 'varied_overlaps' in recon_info:
    overlaps = recon_info['varied_overlaps'].item()
    print(f"\nSpecified overlaps:")
    print(f"  X overlaps: {overlaps['x_overlaps']}")

# Load all subvolumes
subvol_files = sorted([f for f in os.listdir(save_dir) 
                       if f.startswith('subvol_') and f.endswith('.npy') and 'meta' not in f])

print(f"\n{'='*80}")
print("SUBVOLUME POSITIONING AND ACTUAL OVERLAPS:")
print(f"{'='*80}")

subvolumes = []
for f in subvol_files:
    vol_path = os.path.join(save_dir, f)
    volume = np.load(vol_path)
    
    meta_path = vol_path.replace('.npy', '_meta.npy.npz')
    meta_data = np.load(meta_path, allow_pickle=True)
    
    subvolumes.append({
        'index': tuple(meta_data['index']),
        'origin': tuple(meta_data['origin']),
        'shape': volume.shape,
        'specified_overlap': tuple(meta_data['overlap'])
    })

# Analyze along x-axis (dimension with splits)
x_dim = recon_info['original_shape'][2]
if 'subvol_shape' in recon_info:
    subvol_x_size = recon_info['subvol_shape'][2]
else:
    # Infer from actual subvolumes
    subvol_x_size = subvolumes[0]['shape'][2] if subvolumes else 0

for i, sv in enumerate(subvolumes):
    idx = sv['index']
    origin = sv['origin']
    shape = sv['shape']
    spec_overlap = sv['specified_overlap'][2]
    
    x_start = origin[2]
    x_end = x_start + shape[2]
    
    print(f"\nSubvolume {i} {idx}:")
    print(f"  Position: x={x_start} to x={x_end} (size={shape[2]})")
    print(f"  Specified overlap with next: {spec_overlap} pixels")
    
    # Calculate actual overlap with next subvolume
    if i < len(subvolumes) - 1:
        next_sv = subvolumes[i + 1]
        next_x_start = next_sv['origin'][2]
        actual_overlap = x_end - next_x_start
        
        print(f"  ACTUAL overlap with next: {actual_overlap} pixels", end="")
        
        if actual_overlap != spec_overlap:
            diff = actual_overlap - spec_overlap
            print(f" ({diff:+d} from specified)")
        else:
            print(" ✓")
        
        # Calculate step size
        step = next_x_start - x_start
        print(f"  Step size to next: {step} pixels (= {shape[2]} - {actual_overlap})")
    else:
        print(f"  Last subvolume - ends at volume boundary ({x_end})")

print(f"\n{'='*80}")
print("COVERAGE VERIFICATION:")
print(f"{'='*80}")

# Check coverage
coverage = [False] * x_dim
for sv in subvolumes:
    x_start = sv['origin'][2]
    x_end = x_start + sv['shape'][2]
    for i in range(x_start, x_end):
        coverage[i] = True

gaps = [i for i in range(x_dim) if not coverage[i]]
if gaps:
    print(f"⚠ WARNING: {len(gaps)} pixels not covered: {gaps[:10]}...")
else:
    print(f"✓ COMPLETE COVERAGE: All {x_dim} pixels covered")

# Count overlapping regions
overlap_count = [0] * x_dim
for sv in subvolumes:
    x_start = sv['origin'][2]
    x_end = x_start + sv['shape'][2]
    for i in range(x_start, x_end):
        overlap_count[i] += 1

max_overlap = max(overlap_count)
print(f"Maximum overlap depth: {max_overlap}x")
for depth in range(1, max_overlap + 1):
    count = sum(1 for c in overlap_count if c == depth)
    print(f"  {count} pixels covered {depth}x")

print("="*80)
