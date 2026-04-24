'''
Generate rotated, thickness-averaged slices through a synthetic 3D volume.

Builds a simple white-on-black cube with an obvious spherical defect at the
centre, then extracts 2D cross-sectional slices at various rotation angles
about the z-axis.  For each angle, several parallel slices are averaged
across the array width to mimic the finite elevation aperture of a
phased-array probe.  All averaged slices are stored in a single 3D matrix
of shape (z, x, theta) and also saved as individual PNGs.

'''
#%%
# Imports
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

#%%
# Parameters, all lengths in mm

# Block dimensions
block_height = 50
block_length = 50   # square x-y footprint

# Volume resolution
nx, ny, nz = 400, 400, 400   

# Single spherical defect, off-centre
# [x, y, z, r] in mm
sphere_coordinates = [
    [block_length/2 - 10, block_length/2 + 8, block_height/2, 2],
]

# Array geometry
array_length = 50          # lateral extent along the scan line
# Radon-mode elevation: integrate across the full block so each scan-line
# slice is a true line integral.  array_width ≥ block diagonal (≈ 71 mm)
# and n_thickness_samples dense enough to cover every voxel perpendicular to
# the scan line (≳ array_width / voxel_size ≈ 285 for 71 mm / 0.25 mm).
array_width  = 75
n_thickness_samples = 300  # parallel sub-slices integrated per angle

# Rotation sweep
rotational_angle_deg = 2
angles_deg = np.arange(0, 180, rotational_angle_deg)   # 90 angles

print(f'Volume:  {nz} x {ny} x {nx}  ({block_height} x {block_length} x {block_length} mm)')
print(f'Array:   length {array_length} mm, width {array_width} mm, '
      f'{n_thickness_samples} thickness samples')
print(f'Angles:  {len(angles_deg)} from {angles_deg[0]}° to {angles_deg[-1]}° '
      f'(step {rotational_angle_deg}°)')

#%%
# Build ground-truth volume

x = np.linspace(0, block_length, nx)
y = np.linspace(0, block_length, ny)
z = np.linspace(0, block_height, nz)

X, Y, Z = np.meshgrid(x, y, z, indexing='xy')
X = np.transpose(X, (2, 0, 1))
Y = np.transpose(Y, (2, 0, 1))
Z = np.transpose(Z, (2, 0, 1))

volume = np.zeros((nz, ny, nx), dtype=np.uint8)

for sx, sy, sz, r in sphere_coordinates:
    mask = (X - sx)**2 + (Y - sy)**2 + (Z - sz)**2 <= r**2
    volume[mask] = 255

script_dir = Path(__file__).parent
np.save(script_dir / 'ground_truth_cube.npy', volume)
print(f'Saved ground_truth_cube.npy  (shape {volume.shape}, dtype {volume.dtype})')

#%%
# Extract one thickness-averaged rotated slice

def extract_rotated_slice_averaged(volume, theta_deg,
                                   array_length_mm, array_width_mm,
                                   n_thickness_samples,
                                   block_length_mm, block_height_mm):
    nz, ny, nx = volume.shape

    n_line_samples = nx
    s = np.linspace(-array_length_mm / 2, array_length_mm / 2, n_line_samples)
    w = np.linspace(-array_width_mm  / 2, array_width_mm  / 2, n_thickness_samples)

    theta = np.deg2rad(theta_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    x_c = block_length_mm / 2
    y_c = block_length_mm / 2

    slice_sum = np.zeros((nz, n_line_samples), dtype=np.float32)

    for w_off in w:
        # Rotated scan line, shifted perpendicular by w_off
        x_line = x_c + s * cos_t - w_off * sin_t
        y_line = y_c + s * sin_t + w_off * cos_t

        x_idx = np.round(x_line / block_length_mm * (nx - 1)).astype(int)
        y_idx = np.round(y_line / block_length_mm * (ny - 1)).astype(int)

        valid = (x_idx >= 0) & (x_idx < nx) & (y_idx >= 0) & (y_idx < ny)

        sub_slice = np.zeros((nz, n_line_samples), dtype=np.float32)
        for i in range(n_line_samples):
            if valid[i]:
                sub_slice[:, i] = volume[:, y_idx[i], x_idx[i]]

        slice_sum += sub_slice

    return slice_sum / n_thickness_samples

#%%
# Sweep angles, stack into single 3D matrix, save PNGs

n_theta = len(angles_deg)
ns = nx   # n_line_samples

# Single 3D matrix (z, x, theta)
slice_stack = np.zeros((nz, ns, n_theta), dtype=np.float32)

output_dir = script_dir / 'slices_averaged'
output_dir.mkdir(exist_ok=True)

for i, angle in enumerate(angles_deg):
    s2d = extract_rotated_slice_averaged(
        volume, angle,
        array_length, array_width, n_thickness_samples,
        block_length, block_height,
    )
    slice_stack[:, :, i] = s2d

    plt.imsave(output_dir / f'slice_{i:03d}.png',
               s2d, cmap='gray', vmin=0, vmax=255)

np.save(script_dir / 'rotated_slices_stack.npy', slice_stack)
np.savez(script_dir / 'rotated_slices.npz',
         stack=slice_stack,
         angles_deg=angles_deg,
         block_length=block_length,
         block_height=block_height,
         array_length=array_length,
         array_width=array_width)

print(f'slice_stack.shape = {slice_stack.shape}   # (z, x, theta)')
print(f'Saved {n_theta} PNGs to {output_dir}')

#%%
# Sanity plot

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

axes[0].imshow(slice_stack[:, :, 0],
               cmap='gray', vmin=0, vmax=255, aspect='equal')
axes[0].set_title(f'θ = {angles_deg[0]:.1f}°')
axes[0].set_xlabel('x [px]')
axes[0].set_ylabel('z [px]')

mid = n_theta // 2
axes[1].imshow(slice_stack[:, :, mid],
               cmap='gray', vmin=0, vmax=255, aspect='equal')
axes[1].set_title(f'θ = {angles_deg[mid]:.1f}°')
axes[1].set_xlabel('x [px]')
axes[1].set_ylabel('z [px]')

axes[2].imshow(slice_stack[nz // 2, :, :],
               cmap='gray', vmin=0, vmax=255, aspect='auto')
axes[2].set_title('Mid-depth sinogram  (x vs θ)')
axes[2].set_xlabel('θ index')
axes[2].set_ylabel('x [px]')

plt.tight_layout()
plt.savefig(script_dir / 'rotated_slices_overview.png', dpi=120)
plt.show()

#%%
# Reconstruct 3D volume: inverse Radon transform per z-slice
#
# slice_stack has shape (nz, ns, n_theta) = (z, x, theta).
# For each z0 the 2D input to iradon is slice_stack[z0, :, :] of shape
# (ns, n_theta) = (x, theta) — matching iradon's expected (n_detectors, n_angles).

from skimage.transform import iradon

recon_volume = np.zeros((nz, nx, nx), dtype=np.float32)

for z0 in range(nz):
    sinogram_at_z0 = slice_stack[z0, :, :]                # (x, theta)
    assert sinogram_at_z0.shape == (nx, n_theta), (
        f'expected (x, theta) = ({nx}, {n_theta}), got {sinogram_at_z0.shape}'
    )
    # iradon's output uses y-increasing-up; flip to match our row-as-y-down volume.
    recon_volume[z0] = iradon(
        sinogram_at_z0,
        theta=angles_deg,
        filter_name='ramp',
        circle=True,
        output_size=nx,
    )[::-1, :].astype(np.float32)

np.save(script_dir / 'reconstructed_volume.npy', recon_volume)
print(f'recon_volume.shape = {recon_volume.shape}   # (z, y, x)')
print(f'value range: [{recon_volume.min():.3f}, {recon_volume.max():.3f}]')

#%%
# View the reconstructed volume

import napari

viewer = napari.Viewer()
viewer.add_image(volume,        name='Ground truth', colormap='gray')
viewer.add_image(recon_volume,  name='Reconstruction', colormap='gray')

napari.run()
