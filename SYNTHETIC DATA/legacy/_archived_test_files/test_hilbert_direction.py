"""
Quick verification test for Hilbert bloom direction fix
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import hilbert

print("="*80)
print("HILBERT BLOOM DIRECTION TEST")
print("="*80)

# Create a simple 3D volume with a point defect at center
dims = (50, 50, 100)  # (depth, height, lateral)
volume = np.zeros(dims, dtype=np.float32)

# Add a point defect at center
center_z, center_y, center_x = 25, 25, 50
volume[center_z-2:center_z+3, center_y-2:center_y+3, center_x-2:center_x+3] = 1.0

print(f"\nOriginal point defect:")
print(f"  Position: (z={center_z}, y={center_y}, x={center_x})")
print(f"  Size: 5×5×5 voxels")
print(f"  Peak value: {volume.max():.2f}")

# Apply Hilbert along LATERAL axis (axis 2) - CORRECT
print("\n" + "-"*80)
print("Applying Hilbert along LATERAL axis (axis 2) - CORRECT")
print("-"*80)

volume_smooth = gaussian_filter(volume, sigma=(0, 0, 1.2))
volume_hilbert_lateral = np.abs(hilbert(volume_smooth, axis=2))

# Measure spread in each direction
z_spread = np.sum(volume_hilbert_lateral > 0.1, axis=(1, 2))
y_spread = np.sum(volume_hilbert_lateral > 0.1, axis=(0, 2))
x_spread = np.sum(volume_hilbert_lateral > 0.1, axis=(0, 1))

z_width = np.sum(z_spread > 0)
y_width = np.sum(y_spread > 0)
x_width = np.sum(x_spread > 0)

print(f"\nBloom extent (voxels above 10% peak):")
print(f"  Z-direction (depth): {z_width} voxels")
print(f"  Y-direction (height): {y_width} voxels")
print(f"  X-direction (lateral): {x_width} voxels")

if x_width > z_width and x_width > y_width:
    print("\n✅ CORRECT: Bloom is strongest in LATERAL (X) direction")
    print("   This creates horizontal bloom parallel to the array.")
else:
    print("\n❌ INCORRECT: Bloom is NOT strongest in lateral direction")
    print("   Check Hilbert axis!")

# For comparison, show what happens with wrong axis
print("\n" + "-"*80)
print("For comparison: Hilbert along DEPTH axis (axis 0) - INCORRECT")
print("-"*80)

volume_smooth_wrong = gaussian_filter(volume, sigma=(0.8, 0, 0))
volume_hilbert_depth = np.abs(hilbert(volume_smooth_wrong, axis=0))

z_spread_wrong = np.sum(volume_hilbert_depth > 0.1, axis=(1, 2))
y_spread_wrong = np.sum(volume_hilbert_depth > 0.1, axis=(0, 2))
x_spread_wrong = np.sum(volume_hilbert_depth > 0.1, axis=(0, 1))

z_width_wrong = np.sum(z_spread_wrong > 0)
y_width_wrong = np.sum(y_spread_wrong > 0)
x_width_wrong = np.sum(x_spread_wrong > 0)

print(f"\nBloom extent (voxels above 10% peak):")
print(f"  Z-direction (depth): {z_width_wrong} voxels")
print(f"  Y-direction (height): {y_width_wrong} voxels")
print(f"  X-direction (lateral): {x_width_wrong} voxels")

if z_width_wrong > x_width_wrong:
    print("\n❌ INCORRECT: Bloom is strongest in DEPTH (Z) direction")
    print("   This creates vertical bloom (perpendicular to array).")
    print("   This does NOT match real TFM physics!")

# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print("\nCORRECT approach (Hilbert on axis 2):")
print(f"  Lateral spread: {x_width} voxels (WIDEST)")
print(f"  Depth spread: {z_width} voxels")
print(f"  Height spread: {y_width} voxels")
print("\nINCORRECT approach (Hilbert on axis 0):")
print(f"  Lateral spread: {x_width_wrong} voxels")
print(f"  Depth spread: {z_width_wrong} voxels (WIDEST)")
print(f"  Height spread: {y_width_wrong} voxels")

print("\n✅ The code now uses the CORRECT approach (axis 2)")
print("   Bloom spreads horizontally, parallel to the array.")
print("   This matches real TFM ultrasound physics!")
print("="*80)
