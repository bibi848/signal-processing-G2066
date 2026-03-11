"""Debug script to check why TFM B-scan doesn't show defects"""
import numpy as np
import matplotlib.pyplot as plt

# Parameters
pixel_size = 0.2e-3  # 0.2 mm
depth, width = 200, 300
c = 6320.0  # m/s
frequency = 5e6  # Hz

# Ground truth
ground_truth = np.random.uniform(0.05, 0.15, size=(depth, width))

# Add circular void
z_c, x_c = 60, 100
radius = 15
Z, X = np.meshgrid(np.arange(depth), np.arange(width), indexing='ij')
dist = np.sqrt((Z - z_c)**2 + (X - x_c)**2)
void_mask = dist <= radius
ground_truth[void_mask] = 0.0

shell_thickness_px = 0.8e-3 / pixel_size
shell_mask = (dist > radius) & (dist <= radius + shell_thickness_px)
ground_truth[shell_mask] = np.random.uniform(0.8, 1.0, size=np.sum(shell_mask))

print("="*80)
print("GROUND TRUTH")
print("="*80)
print(f"Shape: {ground_truth.shape}")
print(f"Range: {ground_truth.min():.3f} - {ground_truth.max():.3f}")
print(f"Void pixels: {np.sum(void_mask)}")
print(f"Shell pixels: {np.sum(shell_mask)}")
print(f"Shell at position (60, 100), radius {radius}")
print(f"Physical position: ({60*pixel_size*1e3:.1f} mm, {(100-width/2)*pixel_size*1e3:.1f} mm)")

# Check scatterers
threshold = 0.1
scatterer_mask = ground_truth > threshold
scatterer_count = np.sum(scatterer_mask)
print(f"\nScatterers (reflectivity > {threshold}): {scatterer_count}")

# Check shell scatterers specifically
shell_scatterers = np.sum(shell_mask)
print(f"Shell scatterers: {shell_scatterers}")

# Simulate simple case: center element to center pixel
print("\n" + "="*80)
print("TESTING TOF CALCULATION")
print("="*80)

# Array setup
num_elements = 32
element_pitch = 0.6e-3
elem_positions = (np.arange(num_elements) - (num_elements - 1) / 2) * element_pitch
print(f"Array: {num_elements} elements")
print(f"Element positions range: {elem_positions.min()*1e3:.2f} to {elem_positions.max()*1e3:.2f} mm")
print(f"Array aperture: {(elem_positions.max() - elem_positions.min())*1e3:.2f} mm")

# Test pixel at defect location
test_z_px, test_x_px = 60, 100
test_z_m = test_z_px * pixel_size
test_x_m = (test_x_px - width/2) * pixel_size

print(f"\nTest pixel: ({test_z_px}, {test_x_px})")
print(f"Physical position: ({test_z_m*1e3:.2f}, {test_x_m*1e3:.2f}) mm")
print(f"Reflectivity at this pixel: {ground_truth[test_z_px, test_x_px]:.3f}")

# Center element
center_elem = num_elements // 2
elem_x = elem_positions[center_elem]
elem_pos = np.array([0.0, elem_x])
pixel_pos = np.array([test_z_m, test_x_m])

dist_to_pixel = np.linalg.norm(pixel_pos - elem_pos)
tof_pulse_echo = 2 * dist_to_pixel / c

print(f"\nCenter element position: {elem_x*1e3:.2f} mm")
print(f"Distance to test pixel: {dist_to_pixel*1e3:.2f} mm")
print(f"TOF (pulse-echo): {tof_pulse_echo*1e6:.2f} μs")

# Check FMC time window
sampling_freq = 4 * frequency
time_samples = 2048
dt = 1.0 / sampling_freq
max_time = time_samples * dt
max_depth = c * max_time / 2

print(f"\nFMC parameters:")
print(f"Sampling frequency: {sampling_freq/1e6:.1f} MHz")
print(f"Time samples: {time_samples}")
print(f"Time step: {dt*1e6:.3f} μs")
print(f"Max time: {max_time*1e6:.1f} μs")
print(f"Max depth (pulse-echo): {max_depth*1e3:.1f} mm")
print(f"Image depth: {depth*pixel_size*1e3:.1f} mm")

# Calculate which time bin this would be in
time_bin = int(tof_pulse_echo / dt)
print(f"\nTime bin for defect echo: {time_bin} / {time_samples}")

if time_bin >= time_samples:
    print("WARNING: Defect echo is OUTSIDE the time window!")
else:
    print("OK: Defect echo is within time window")

# Visualize ground truth
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

extent_mm = [-width/2 * pixel_size * 1e3, width/2 * pixel_size * 1e3, 
             depth * pixel_size * 1e3, 0]

ax = axes[0]
im = ax.imshow(ground_truth, aspect='auto', cmap='hot', extent=extent_mm)
ax.axvline(test_x_m*1e3, color='cyan', linestyle='--', label='Test pixel X')
ax.axhline(test_z_m*1e3, color='cyan', linestyle='--', label='Test pixel Z')
ax.plot(test_x_m*1e3, test_z_m*1e3, 'c*', markersize=15, label='Test pixel')
ax.set_xlabel('Lateral Position (mm)')
ax.set_ylabel('Depth (mm)')
ax.set_title('Ground Truth Reflectivity')
ax.legend()
plt.colorbar(im, ax=ax, label='Reflectivity')

# Show defect Shell locations
ax = axes[1]
shell_image = np.zeros_like(ground_truth)
shell_image[shell_mask] = 1.0
im2 = ax.imshow(shell_image, aspect='auto', cmap='Reds', extent=extent_mm)
ax.set_xlabel('Lateral Position (mm)')
ax.set_ylabel('Depth (mm)')
ax.set_title('Defect Shell (High Reflectivity)')
plt.colorbar(im2, ax=ax, label='Shell')

plt.tight_layout()
plt.savefig('debug_ground_truth.png', dpi=150)
print(f"\nSaved: debug_ground_truth.png")

plt.show()
