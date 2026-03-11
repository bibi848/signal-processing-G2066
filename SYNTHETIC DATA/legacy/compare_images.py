#!/usr/bin/env python3
"""
Compare synthetic TFM images with experimental data
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
import os

# Create comparison figure
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('COMPARISON: Synthetic (top) vs Experimental (bottom)', 
             fontsize=16, fontweight='bold')

# Load synthetic result
synthetic = imread('ndt_2d_results.png')

# Show full synthetic result
axes[0, 0].imshow(synthetic)
axes[0, 0].set_title('Synthetic: Full 4-Panel Output', fontsize=12)
axes[0, 0].axis('off')

# Extract just the TFM B-scan panel (approx top-right quadrant)
h, w = synthetic.shape[:2]
tfm_panel = synthetic[:h//2, w//2:, :]
axes[0, 1].imshow(tfm_panel)
axes[0, 1].set_title('Synthetic: TFM B-Scan Panel (zoomed)', fontsize=12)
axes[0, 1].axis('off')

# Show ground truth
gt_panel = synthetic[:h//2, :w//2, :]
axes[0, 2].imshow(gt_panel)
axes[0, 2].set_title('Synthetic: Ground Truth Panel', fontsize=12)
axes[0, 2].axis('off')

# Load experimental images with holes (5MHz, aluminum)
exp_paths = [
    '../DATA/1D TFM Data/Al Hole 5MHz 28012026/Al_Hole_30_3_TFM.png',
    '../DATA/1D TFM Data/Al Hole 5MHz 28012026/Al_Hole_60_4_TFM.png',
    '../DATA/1D TFM Data/Al Hole 5MHz 02022026/Al_Hole_5_3_TFM.png',
]

print("\nCOMPARISON ANALYSIS")
print("=" * 80)

for idx, path in enumerate(exp_paths):
    try:
        exp_img = imread(path)
        axes[1, idx].imshow(exp_img)
        filename = path.split('/')[-1]
        axes[1, idx].set_title(f'Experimental: {filename}', fontsize=10)
        axes[1, idx].axis('off')
        print(f'\n{idx+1}. Loaded: {filename}')
        print(f'   Shape: {exp_img.shape}')
    except Exception as e:
        axes[1, idx].text(0.5, 0.5, f'Failed to load\n{path.split("/")[-1]}', 
                          ha='center', va='center', transform=axes[1, idx].transAxes)
        axes[1, idx].axis('off')
        print(f'\nError loading {path}: {e}')

plt.tight_layout()
plt.savefig('comparison_synthetic_vs_experimental.png', dpi=150, bbox_inches='tight')
print('\n' + '=' * 80)
print('✓ Saved: comparison_synthetic_vs_experimental.png')
print('=' * 80)

# Print detailed analysis
print("\n\nDETAILED ANALYSIS:")
print("=" * 80)

print("\n1. SYNTHETIC IMAGE CHARACTERISTICS:")
print("   - Uses ray-tracing simulation with FMC+TFM workflow")
print("   - Hilbert envelope detection applied (removes aliasing)")
print("   - Bandpass filtering at 4.5-5.5 MHz")
print("   - dB scale visualization (-60 to 0 dB)")
print("   - Clean, well-defined circular and rectangular defects")

print("\n2. TYPICAL EXPERIMENTAL IMAGE CHARACTERISTICS:")
print("   - Real ultrasonic array data with environmental noise")
print("   - Hardware-dependent signal characteristics")
print("   - Material variations and grain structure")
print("   - Geometric uncertainties in defect positioning")

print("\n3. KEY DIFFERENCES TO CHECK:")
print("   a) Signal-to-noise ratio (SNR)")
print("   b) Defect sharpness/clarity")
print("   c) Background texture")
print("   d) Color scale range and distribution")
print("   e) Presence of artifacts (side lobes, grating lobes)")
print("   f) Defect shape fidelity")

print("\n" + "=" * 80)
