"""Debug FMC acquisition to see if echoes are captured"""
import numpy as np
import matplotlib.pyplot as plt

# Parameters
pixel_size = 0.2e-3
depth, width = 200, 300
c = 6320.0
frequency = 5e6
num_elements = 32
element_pitch = 0.6e-3

# Create ground truth with ONE defect
ground_truth = np.ones((depth, width)) * 0.1  # Uniform background

# Add ONE circular shell for clarity
z_c, x_c = 60, 150  # Center of image laterally
radius = 15
Z, X = np.meshgrid(np.arange(depth), np.arange(width), indexing='ij')
dist = np.sqrt((Z - z_c)**2 + (X - x_c)**2)

# Clear void
void_mask = dist <= radius
ground_truth[void_mask] = 0.0

# Bright shell
shell_thickness_px = 4
shell_mask = (dist > radius) & (dist <= radius + shell_thickness_px)
ground_truth[shell_mask] = 1.0  # Maximum reflectivity

print("="*80)
print("TEST CONFIGURATION")
print("="*80)
print(f"Image: {depth} × {width} pixels ({depth*pixel_size*1e3:.1f} × {width*pixel_size*1e3:.1f} mm)")
print(f"Defect center: pixel ({z_c}, {x_c})")
z_phys = z_c * pixel_size * 1e3
x_phys = (x_c - width/2) * pixel_size * 1e3
print(f"Defect center: physical ({z_phys:.1f}, {x_phys:.1f}) mm")
print(f"Defect radius: {radius} pixels ({radius*pixel_size*1e3:.1f} mm)")
print(f"Shell thickness: {shell_thickness_px} pixels")
print(f"Shell pixels: {np.sum(shell_mask)}")

# Array setup
elem_positions = (np.arange(num_elements) - (num_elements - 1) / 2) * element_pitch
print(f"\nArray: {num_elements} elements, pitch = {element_pitch*1e3:.2f} mm")
print(f"Aperture: {(elem_positions[-1] - elem_positions[0])*1e3:.1f} mm")

# FMC setup
sampling_freq = 4 * frequency
time_samples = 2048
dt = 1.0 / sampling_freq
time_axis = np.arange(time_samples) * dt

# Find scatterers
threshold = 0.5  # Only strong scatterers
scatterer_mask = ground_truth > threshold
scatterer_coords = np.argwhere(scatterer_mask)
scatterer_reflectivity = ground_truth[scatterer_mask]
scatterer_positions = scatterer_coords.astype(float) * pixel_size

print(f"\nScatterers (>0.5): {len(scatterer_coords)}")
print(f"First 5 scatterers:")
for i in range(min(5, len(scatterer_coords))):
    z_px, x_px = scatterer_coords[i]
    z_m, x_m = scatterer_positions[i]
    print(f"  Pixel ({z_px:3d}, {x_px:3d}) → ({z_m*1e3:6.2f}, {x_m*1e3:6.2f}) mm, R={scatterer_reflectivity[i]:.2f}")

# Simulate FMC for ONE TX element (center element)
tx_idx = num_elements // 2
tx_pos = np.array([0.0, elem_positions[tx_idx]])

print(f"\n" + "="*80)
print(f"SIMULATING FMC FOR TX ELEMENT {tx_idx}")
print(f"TX position: {tx_pos[1]*1e3:.2f} mm")
print("="*80)

# Create A-scans for all RX elements
fmc_row = np.zeros((num_elements, time_samples), dtype=np.float32)

for rx_idx in range(num_elements):
    rx_pos = np.array([0.0, elem_positions[rx_idx]])
    a_scan = np.zeros(time_samples, dtype=np.float32)
    
    # Add echo from each scatterer
    for scat_idx, scat_pos in enumerate(scatterer_positions):
        tx_distance = np.linalg.norm(scat_pos - tx_pos)
        rx_distance = np.linalg.norm(scat_pos - rx_pos)
        total_distance = tx_distance + rx_distance
        tof = total_distance / c
        
        time_bin = int(tof / dt)
        
        if time_bin < time_samples:
            reflectivity = scatterer_reflectivity[scat_idx]
            geometric = 1.0 / (tx_distance * rx_distance + 1e-10)
            attenuation = np.exp(-0.03 * (frequency / 1e6) * total_distance)
            amplitude = reflectivity * geometric * attenuation
            a_scan[time_bin] += amplitude
    
    fmc_row[rx_idx, :] = a_scan

# Check if we got any signal
max_signal = fmc_row.max()
nonzero_bins = np.sum(fmc_row > 0)

print(f"\nFMC results for TX element {tx_idx}:")
print(f"  Max signal: {max_signal:.2e}")
print(f"  Non-zero time bins: {nonzero_bins}")
print(f"  Total time bins: {time_samples * num_elements}")

if max_signal > 0:
    # Find which RX has strongest signal
    max_rx = np.argmax(fmc_row.max(axis=1))
    max_time_bin = np.argmax(fmc_row[max_rx, :])
    max_tof = max_time_bin * dt
    print(f"  Strongest RX: element {max_rx} (position {elem_positions[max_rx]*1e3:.2f} mm)")
    print(f"  Strongest echo time: {max_tof*1e6:.2f} μs (bin {max_time_bin})")
    print(f"  Amplitude: {fmc_row[max_rx, max_time_bin]:.2e}")

# Plot results
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Ground truth
ax = axes[0, 0]
extent = [-width/2*pixel_size*1e3, width/2*pixel_size*1e3, depth*pixel_size*1e3, 0]
im = ax.imshow(ground_truth, aspect='auto', cmap='hot', extent=extent, vmin=0, vmax=1)
ax.axhline(z_phys, color='cyan', linestyle='--', alpha=0.5)
ax.axvline(x_phys, color='cyan', linestyle='--', alpha=0.5)
ax.set_xlabel('Lateral (mm)')
ax.set_ylabel('Depth (mm)')
ax.set_title('Ground Truth')
plt.colorbar(im, ax=ax)

# A-scans
ax = axes[0, 1]
for rx_idx in range(0, num_elements, 2):  # Every other RX
    ax.plot(time_axis*1e6, fmc_row[rx_idx, :], alpha=0.6, linewidth=0.8, 
            label=f'RX {rx_idx}' if rx_idx % 8 == 0 else '')
ax.set_xlabel('Time (μs)')
ax.set_ylabel('Amplitude')
ax.set_title(f'A-Scans (TX element {tx_idx})')
ax.grid(True, alpha=0.3)
ax.legend()

# FMC matrix
ax = axes[1, 0]
im2 = ax.imshow(fmc_row, aspect='auto', cmap='viridis', extent=[0, time_axis[-1]*1e6, num_elements, 0])
ax.set_xlabel('Time (μs)')
ax.set_ylabel('RX Element')
ax.set_title(f'A-Scans Image (TX={tx_idx})')
plt.colorbar(im2, ax=ax)

# Max A-scan envelope
ax = axes[1, 1]
max_ascan = fmc_row.max(axis=0)
ax.plot(time_axis*1e6, max_ascan, 'b-', linewidth=2)
ax.set_xlabel('Time (μs)')
ax.set_ylabel('Max Amplitude (over all RX)')
ax.set_title('Maximum A-Scan Envelope')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('debug_fmc_acquisition.png', dpi=150)
print(f"\nSaved: debug_fmc_acquisition.png")

plt.show()

print("\n" + "="*80)
if max_signal < 1e-10:
    print("WARNING: NO SIGNAL DETECTED! FMC acquisition may be broken.")
    print("Check:")
    print("  1. Are scatterers being found?")
    print("  2. Are TOF calculations correct?")
    print("  3. Are amplitudes too small?")
else:
    print("FMC acquisition appears to be working - signal detected!")
print("="*80)
