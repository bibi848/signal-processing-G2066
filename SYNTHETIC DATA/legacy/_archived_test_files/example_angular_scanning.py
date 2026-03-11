"""
Example: Simulating Angular-Dependent Defect Imaging with Array Movement

This demonstrates how defects appear differently as the ultrasound array moves
across the sample, showing realistic off-axis imaging effects.
"""

import sys
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import hilbert

# Add parent directory to path to import the generator
sys.path.insert(0, '.')

# Note: Import from the main module when running standalone
# For this demo, we'll just show the usage pattern

print("="*80)
print("ANGULAR-DEPENDENT IMAGING EXAMPLE")
print("="*80)

print("""
USAGE PATTERN:
--------------

from SyntheticVolumeGenerator import SyntheticVolumeGenerator

# 1. Create generator with initial array position
generator = SyntheticVolumeGenerator(
    dimensions=(200, 150, 600),
    seed=42,
    array_position=(75, 300)  # Center of sample
)

# 2. Add defects (fixed positions)
generator.add_spherical_void(center=(80, 75, 200), radius=12, intensity=0.95)
generator.add_spherical_void(center=(100, 75, 400), radius=10, intensity=0.93)

# 3. Generate volume with angular effects enabled (default)
volume = generator.generate(use_angular_effects=True)

# Defects directly under the array will appear bright (max intensity)
# Defects off to the side will appear dimmer (reduced intensity)

# 4. Move the array to a different position
generator.set_array_position(y_pos=75, x_pos=150)  # Shift left

# 5. Re-generate to see how defects look from new position
volume_new_position = generator.generate(use_angular_effects=True)

# Now defect at x=200 is closer to array (brighter)
# And defect at x=400 is farther from array (dimmer)

# 6. Disable angular effects for comparison (all defects at max intensity)
volume_no_angular = generator.generate(use_angular_effects=False)


PHYSICS OF ANGULAR IMAGING:
---------------------------

The intensity reduction follows realistic ultrasound physics:

1. Reflectivity Angular Response:
   - Spherical defects: cos^1.5(θ) response (moderate falloff)
   - Cylindrical defects: cos^1.0(θ) response (broader, less sensitive)

2. Beam Divergence:
   - Gaussian beam profile with ~50° -6dB beam width
   - Signal gradually decreases away from beam center

3. Combined Effect:
   - At 0° (directly below): 100% intensity
   - At 20°: ~80% intensity  
   - At 30°: ~63% intensity
   - At 40°: ~47% intensity
   - At 50°: ~33% intensity
   - Minimum floor: 5% (even at extreme angles, some signal remains)

4. Angular Calculation:
   angle = arctan(lateral_offset / depth)
   - Deep defects have smaller angles for same lateral offset
   - Shallow defects show stronger angular effects


TYPICAL USE CASES:
-----------------

1. Simulate Array Scanning:
   Generate multiple volumes with different array positions to simulate
   a scan sequence as the array moves across the sample.

2. Study Detection Limits:
   Understand how defects become harder to detect as they move away
   from the array's direct line of sight.

3. Optimize Scan Overlap:
   Determine required overlap between adjacent scans to ensure all
   defects are imaged with sufficient intensity.

4. Realistic Training Data:
   Generate synthetic data that matches real ultrasound scans where
   defects vary in brightness based on their position relative to the probe.


PARAMETERS:
-----------

array_position: (y, x) tuple
    - Y: vertical position of array (0 = top, height = bottom)
    - X: lateral position of array (0 = left, width = right)
    - Default: center of volume (dimensions[1]//2, dimensions[2]//2)

use_angular_effects: bool
    - True: Apply angular attenuation (realistic)
    - False: All defects at full intensity (idealized)


PERFORMANCE NOTE:
-----------------
Angular effects add minimal computational overhead (~1-2% slower)
as the calculation is vectorized and only done once per defect.
""")

print("="*80)
print("See '3d synthetic data v2.py' for the full implementation!")
print("Check the bottom of the file for a complete working example.")
print("="*80)
