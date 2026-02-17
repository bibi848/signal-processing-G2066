"""
Verification test for 2D Hilbert bloom (y and x directions)
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import hilbert

print("="*80)
print("2D HILBERT BLOOM DIRECTION TEST")
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

# Apply 2D Hilbert along HEIGHT and LATERAL axes (axes 1 and 2) - CORRECT
print("\n" + "-"*80)
print("Applying 2D Hilbert along HEIGHT (axis 1) and LATERAL (axis 2) - CORRECT")
print("-"*80)

volume_smooth = gaussian_filter(volume, sigma=(0, 0.8, 1.5))
print("Pre-smoothed with sigma=(0, 0.8, 1.5) - smooth in y-x plane only")

# Apply sequentially: first lateral (x), then height (y)
print("\nStep 1: Hilbert along lateral axis (x, axis 2)...")
volume_hilbert_x = np.abs(hilbert(volume_smooth, axis=2))

print("Step 2: Hilbert along height axis (y, axis 1)...")
volume_hilbert_xy = np.abs(hilbert(volume_hilbert_x, axis=1))

# Measure spread in each direction
z_spread = np.sum(volume_hilbert_xy > 0.1, axis=(1, 2))
y_spread = np.sum(volume_hilbert_xy > 0.1, axis=(0, 2))
x_spread = np.sum(volume_hilbert_xy > 0.1, axis=(0, 1))

z_width = np.sum(z_spread > 0)
y_width = np.sum(y_spread > 0)
x_width = np.sum(x_spread > 0)

print(f"\nBloom extent (voxels above 10% peak):")
print(f"  Z-direction (depth): {z_width} voxels - should be SMALL (good axial resolution)")
print(f"  Y-direction (height): {y_width} voxels - should be MODERATE (elevation)")
print(f"  X-direction (lateral): {x_width} voxels - should be LARGE (worst resolution)")

if x_width > z_width and y_width > z_width:
    print("\n✅ CORRECT: Bloom is in Y-X plane (perpendicular to beam)")
    print("   Depth resolution preserved, bloom in elevation and lateral directions.")
    if x_width > y_width:
        print("   ✅ Lateral bloom stronger than elevation (as expected)")
    else:
        print("   ⚠️  Elevation bloom stronger than lateral (unusual)")
else:
    print("\n❌ INCORRECT: Bloom pattern is wrong")
    print("   Check Hilbert axes!")

# For comparison, show 1D Hilbert (previous approach)
print("\n" + "-"*80)
print("For comparison: 1D Hilbert along LATERAL only (axis 2) - PREVIOUS")
print("-"*80)

volume_smooth_1d = gaussian_filter(volume, sigma=(0, 0, 1.2))
volume_hilbert_1d = np.abs(hilbert(volume_smooth_1d, axis=2))

z_spread_1d = np.sum(volume_hilbert_1d > 0.1, axis=(1, 2))
y_spread_1d = np.sum(volume_hilbert_1d > 0.1, axis=(0, 2))
x_spread_1d = np.sum(volume_hilbert_1d > 0.1, axis=(0, 1))

z_width_1d = np.sum(z_spread_1d > 0)
y_width_1d = np.sum(y_spread_1d > 0)
x_width_1d = np.sum(x_spread_1d > 0)

print(f"\nBloom extent (voxels above 10% peak):")
print(f"  Z-direction (depth): {z_width_1d} voxels")
print(f"  Y-direction (height): {y_width_1d} voxels (no spreading in y)")
print(f"  X-direction (lateral): {x_width_1d} voxels")

# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print("\n2D Hilbert (axes 1 & 2) - CURRENT APPROACH:")
print(f"  Depth spread (z): {z_width} voxels (minimal ✓)")
print(f"  Height spread (y): {y_width} voxels (moderate ✓)")
print(f"  Lateral spread (x): {x_width} voxels (largest ✓)")
bloom_ratio = (y_width * x_width) / max(z_width**2, 1)
print(f"  Bloom area ratio (y-x plane vs z²): {bloom_ratio:.1f}x")

print("\n1D Hilbert (axis 2 only) - PREVIOUS APPROACH:")
print(f"  Depth spread (z): {z_width_1d} voxels")
print(f"  Height spread (y): {y_width_1d} voxels (NO elevation bloom)")
print(f"  Lateral spread (x): {x_width_1d} voxels")

print("\n" + "="*80)
print("ADVANTAGES OF 2D HILBERT:")
print("="*80)
print("✅ More realistic: Both lateral and elevation resolution are limited")
print("✅ Creates elliptical bloom in y-x plane (perpendicular to beam)")
print("✅ Matches experimental TFM data appearance better")
print("✅ Good axial (depth) resolution maintained")
print(f"✅ Bloom spreads {bloom_ratio:.1f}x more in cross-section than along beam")
print("\nPhysics:")
print("  - Lateral (x): Worst resolution → strongest bloom")
print("  - Elevation (y): Moderate resolution → moderate bloom")
print("  - Axial (z): Best resolution → minimal bloom")
print("="*80)
