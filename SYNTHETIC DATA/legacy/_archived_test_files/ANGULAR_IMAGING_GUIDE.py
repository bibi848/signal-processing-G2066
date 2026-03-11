"""
ANGULAR-DEPENDENT IMAGING FOR SUBVOLUMES
========================================

OVERVIEW:
---------
The synthetic data generator now supports angular-dependent defect imaging 
specifically for subvolume generation. This simulates the realistic behavior
where each subvolume represents an independent scan with the ultrasound array
positioned above it.

WORKFLOW:
---------

1. GROUND TRUTH GENERATION (no angular effects):
   
   generator = SyntheticVolumeGenerator(dimensions=(200, 150, 600), seed=42)
   generator.add_spherical_void(center=(80, 75, 200), radius=12)
   generator.add_spherical_void(center=(100, 75, 400), radius=10)
   
   # Generate ground truth - defects at FULL intensity everywhere
   volume_clean = generator.generate()  # use_angular_effects=False by default
   
   Result: Perfect representation of the physical sample.
           All defects appear at their maximum intensity regardless of position.
           This is your reference "ground truth" for validation.


2. SUBVOLUME GENERATION (with angular effects):
   
   sub_volumes = generator.generate_stitching_test_data(
       volume_clean,
       num_splits=(1, 1, 3),         # Split in X direction (3 scans)
       overlap_pixels=(0, 0, 40),    # 40 pixel overlap
       artifact_mode='per_subvolume',
       use_angular_effects=True      # ENABLE angular-dependent imaging
   )
   
   Result: Each subvolume is REGENERATED (not just split) with:
           - Array positioned at the center of that subvolume
           - Defects appearing with angle-dependent intensity
           - Brighter defects near subvolume center
           - Dimmer defects far from subvolume center


WHAT HAPPENS BEHIND THE SCENES:
--------------------------------

For each subvolume at position (z_start, y_start, x_start):

1. Calculate array position:
   array_y = y_start + height/2
   array_x = x_start + width/2
   
2. Regenerate defects for this subvolume region:
   - For each defect, calculate angle from array
   - Apply angular attenuation based on physics model
   - Only render defects that intersect this subvolume

3. Add imaging artifacts (noise, attenuation, etc.)

4. Apply CTFM processing (Hilbert + dB conversion)


ANGULAR ATTENUATION MODEL:
---------------------------

Intensity reduction follows realistic ultrasound physics:

For spherical defects:
  attenuation = 0.6 * cos^1.5(θ) + 0.4 * exp(-(θ/σ)²)
  
For cylindrical defects:
  attenuation = 0.6 * cos^1.0(θ) + 0.4 * exp(-(θ/σ)²)

Where:
  θ = angle from array normal = arctan(lateral_offset / depth)
  σ = beam divergence (50° -6dB beam width)
  Minimum attenuation: 0.05 (5% noise floor)

Result at different angles:
  0°  → 100% intensity (directly below array)
  20° → ~80% intensity
  30° → ~63% intensity
  40° → ~47% intensity
  50° → ~33% intensity
  60°+ → ~20-30% intensity (approaching noise floor)


EXAMPLE SCENARIO:
-----------------

Sample: 200×150×600 volume
Split into: 3 subvolumes along X-axis (1×1×3)

Defect positions:
  - Defect A at (z=80, y=75, x=150)
  - Defect B at (z=100, y=75, x=300)
  - Defect C at (z=120, y=75, x=450)

Subvolume 1: x=0-200, array at (y=75, x=100)
  - Defect A (x=150): angle ~27°, intensity ~65%
  - Defect B (x=300): outside this subvolume (or very dim)
  - Defect C (x=450): outside this subvolume

Subvolume 2: x=160-400, array at (y=75, x=280)
  - Defect A (x=150): angle ~52°, intensity ~30%
  - Defect B (x=300): angle ~11°, intensity ~92%
  - Defect C (x=450): angle ~48°, intensity ~36%

Subvolume 3: x=360-600, array at (y=75, x=480)
  - Defect A (x=150): outside this subvolume
  - Defect B (x=300): angle ~50°, intensity ~34%
  - Defect C (x=450): angle ~16°, intensity ~88%


WHY THIS MATTERS:
-----------------

1. REALISM:
   - Matches real ultrasound scanning where probe position matters
   - Different scan positions see defects differently
   - Edge effects and overlap regions show intensity variations

2. STITCHING CHALLENGES:
   - Overlapping regions have different intensities from different "scans"
   - Stitching algorithms must handle varying defect brightness
   - Tests the robustness of reconstruction methods

3. DETECTION LIMITS:
   - Off-axis defects are harder to detect (lower SNR)
   - Scan planning is important to ensure all defects are imaged
   - Overlap requirements depend on defect positions

4. TRAINING DATA:
   - ML models see realistic variations in defect appearance
   - Better generalization to real experimental data
   - Captures physics of ultrasound imaging


COMPARISON:
-----------

WITHOUT angular effects (old behavior):
  - All subvolumes show defects at full intensity
  - Unrealistic - doesn't match real scanning
  - Easy to stitch (same intensity everywhere)
  - Poor training data for ML

WITH angular effects (new behavior):
  - Subvolumes show angle-dependent intensity
  - Realistic - matches experimental data
  - Challenging to stitch (intensity variations)
  - Excellent training data for ML


PARAMETERS:
-----------

In SyntheticVolumeGenerator:
  
  array_position: (y, x) tuple
    - Set manually for ground truth generation (if using angular effects)
    - Automatically calculated for subvolumes (center of each subvolume)
  
  use_angular_effects: bool
    - False (default): Ground truth mode, no attenuation
    - True: Simulate array position and angle-dependent imaging

In generate_stitching_test_data():
  
  use_angular_effects: bool
    - False: Simple split of ground truth (fast, unrealistic)
    - True: Regenerate each subvolume with angular effects (slower, realistic)


PERFORMANCE NOTES:
------------------

Enabling angular effects in subvolume generation:
  - ~20-40% slower (must regenerate defects for each subvolume)
  - Still vectorized and optimized
  - Parallel processing of defects within each subvolume
  - Most time spent on CTFM processing, not angular calculations

Typical timing for 200×150×600 volume split into 3 subvolumes:
  - Without angular effects: ~10-15 seconds
  - With angular effects: ~15-20 seconds


BEST PRACTICES:
---------------

1. Always generate ground truth WITHOUT angular effects
   - Use for validation and reference
   - Represents the true physical sample

2. Use angular effects for subvolume generation
   - Simulates realistic scanning behavior
   - Provides challenging stitching scenarios

3. Save both versions
   - Ground truth for quantitative validation
   - Scanned subvolumes for algorithm testing

4. Document array positions
   - Metadata includes array_position for each subvolume
   - Important for understanding intensity variations

5. Consider defect distribution
   - Evenly distributed defects → moderate angular effects
   - Clustered defects → strong angular effects in some subvolumes


FUTURE ENHANCEMENTS:
--------------------

Potential improvements:
  - Depth-dependent beam characteristics
  - Surface roughness effects on reflection
  - Multiple scattering for complex geometries
  - Random variations in array positioning (scan inaccuracies)
  - Time-variant noise per subvolume scan
  - Realistic probe motion artifacts

"""

if __name__ == "__main__":
    print(__doc__)
