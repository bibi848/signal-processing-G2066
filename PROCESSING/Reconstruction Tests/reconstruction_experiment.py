'''
This script investigates the difficulties of reconstructing 3D volumes from 2D images taken with ultrasonic arrays. 
Prior to trying with microstructural backscatter, it is tried with 3D white on black volumes.
'''
#%%
# Functions
import numpy as np
import matplotlib.pyplot as plt
import napari
import matplotlib.image as mpimg
from pathlib import Path

# See paper notes for the definitions and derivations of these equations.
def effective_length_calc(x, theta):
    return (x) / (2*np.sin(theta/2))

def optimal_angle_calc(x, h):
    return 2 * np.arcsin((x) / (2 * h))

#%%
# Parameters, & all lengths are in mm
block_height = 50 # 50mm
block_length = 50 # Block has a square x-y plane, so the length and width are identical. 

# [[x, y, z, r], [x, y, z, r]]... etc
sphere_coordinates = [
    [10, 10, 10, 4],
    [20, 20, 20, 4],
    [30, 30, 30, 4],
    [40, 40, 40, 4],
    [15, 35, 25, 3],
    [35, 15, 25, 3],
]

# [x, z, r]
cylinder_coordinates  = [15, 25, 4]
cylinder_coordinates2 = [25, 15, 3]

# Full cube resolution
nx, ny, nz = 200, 200, 200

#%%
# Calculating effective volume from array parameters
array_length = 50
array_width  = 5 # also element length
rotational_angle_deg = 3

rotational_angle_rad = np.deg2rad(rotational_angle_deg)
effective_length = effective_length_calc(array_width, rotational_angle_rad)

optimal_angle_rad = optimal_angle_calc(array_width, array_length/2)
optimal_angle_deg = np.rad2deg(optimal_angle_rad)

print()
print(f'Effective length: {effective_length:.3f} mm using {rotational_angle_deg} degrees')
print(f'Optimal angle: {optimal_angle_deg:.3f} deg')

# Graphing effective length vs choice of rotational angle
theta_deg = np.linspace(5, 40, 100)
theta_rad = np.deg2rad(theta_deg)
effective_lengths = effective_length_calc(array_width, theta_rad)

plt.plot(theta_deg, effective_lengths, c='b')
plt.scatter(optimal_angle_deg, array_length/2, c='r', label='Optimal')
plt.xlabel('Rotational Angle [deg]')
plt.ylabel('Effective Length [mm]')
plt.grid(True)
plt.legend()

#%%
# Creating 'Ground Truth Cube'

x = np.linspace(0, block_length, nx)
y = np.linspace(0, block_length, ny)
z = np.linspace(0, block_height, nz)

# Coordinate grid
X, Y, Z = np.meshgrid(x, y, z, indexing='xy')
X = np.transpose(X, (2, 0, 1))
Y = np.transpose(Y, (2, 0, 1))
Z = np.transpose(Z, (2, 0, 1))

# Create empty volume
volume = np.zeros((nz, ny, nx), dtype=np.uint8)

# Add spheres
for sphere in sphere_coordinates:
    sx, sy, sz, r = sphere
    mask = (X - sx)**2 + (Y - sy)**2 + (Z - sz)**2 <= r**2
    volume[mask] = 255

# Cylindrical defect
cx, cz, r_cyl = cylinder_coordinates
cylinder_mask = (X - cx)**2 + (Z - cz)**2 <= r_cyl**2
volume[cylinder_mask] = 255

cy2, cz2, r2 = cylinder_coordinates2
cylinder2_mask = (Y - cy2)**2 + (Z - cz2)**2 <= r2**2
volume[cylinder2_mask] = 255

# Save volume
output_filename = 'ground_truth_cube.npy'
np.save(output_filename, volume)

viewer = napari.Viewer()
viewer.add_image(volume, name='Sphere Volume')

napari.run()

#%%
# Generate rotated 2D slices through the 3D volume and save as PNGs

def extract_rotated_slice(volume, theta_deg, array_length_mm, block_length_mm, block_height_mm):

    nz, ny, nx = volume.shape

    # Physical coordinate grids in mm
    x_coords = np.linspace(0, block_length_mm, nx)
    y_coords = np.linspace(0, block_length_mm, ny)
    z_coords = np.linspace(0, block_height_mm, nz)

    # Centre of the block
    x_c = block_length_mm / 2
    y_c = block_length_mm / 2

    # Parameter along the array line
    n_line_samples = nx
    s = np.linspace(-array_length_mm / 2, array_length_mm / 2, n_line_samples)

    theta = np.deg2rad(theta_deg)

    # Rotated line in x-y
    x_line = x_c + s * np.cos(theta)
    y_line = y_c + s * np.sin(theta)

    # Convert physical coordinates to nearest pixel indices
    x_idx = np.round((x_line / block_length_mm) * (nx - 1)).astype(int)
    y_idx = np.round((y_line / block_length_mm) * (ny - 1)).astype(int)

    # Valid points that lie inside the volume
    valid = (x_idx >= 0) & (x_idx < nx) & (y_idx >= 0) & (y_idx < ny)

    # Build slice
    slice_2d = np.zeros((nz, n_line_samples), dtype=volume.dtype)

    for i in range(n_line_samples):
        if valid[i]:
            slice_2d[:, i] = volume[:, y_idx[i], x_idx[i]]

    return slice_2d


angles_deg = np.arange(0, 180, rotational_angle_deg)
output_dir = Path(__file__).parent / 'slices'
output_dir.mkdir(exist_ok=True)

all_slices = []
for i, angle in enumerate(angles_deg):
    slice_2d = extract_rotated_slice(
        volume=volume,
        theta_deg=angle,
        array_length_mm=array_length,
        block_length_mm=block_length,
        block_height_mm=block_height
    )

    all_slices.append(slice_2d)

    output_path = output_dir / f'slice_{i:03d}.png'
    plt.imsave(output_path, slice_2d, cmap='gray', vmin=0, vmax=255)

#%%
# Reconstruct 3D volume from rotated central slices

def reconstruct_volume_from_rotated_slices(all_slices, angles_deg,
                                           nx, ny, nz,
                                           block_length_mm, block_height_mm,
                                           array_length_mm):
    angles_deg = np.asarray(angles_deg)
    slice_stack = np.stack(all_slices, axis=0)

    n_angles, nz_s, ns = slice_stack.shape

    recon_volume = np.zeros((nz, ny, nx), dtype=slice_stack.dtype)

    # Physical coordinates of output grid
    x = np.linspace(0, block_length_mm, nx)
    y = np.linspace(0, block_length_mm, ny)
    z = np.linspace(0, block_height_mm, nz)

    x_c = block_length_mm / 2
    y_c = block_length_mm / 2
    radius_max = array_length_mm / 2

    # Slice coordinate along the array
    s_coords = np.linspace(-radius_max, radius_max, ns)

    # Build x-y coordinate arrays
    X, Y = np.meshgrid(x, y, indexing='xy')   # (ny, nx)
    dx = X - x_c
    dy = Y - y_c
    rho = np.sqrt(dx**2 + dy**2)

    # Angle of each voxel in x-y plane mapped to [0, 180)
    phi_deg = (np.rad2deg(np.arctan2(dy, dx)) + 360) % 360
    theta_target = np.where(phi_deg < 180, phi_deg, phi_deg - 180)

    # Inside cylinder
    inside = rho <= radius_max

    # For each (x,y) choose nearest measured angle
    angle_step = np.abs(angles_deg[1] - angles_deg[0]) if len(angles_deg) > 1 else 180
    angle_idx = np.round(theta_target / angle_step).astype(int)
    angle_idx = np.clip(angle_idx, 0, len(angles_deg) - 1)

    # Use the actual chosen angle to compute signed position s
    chosen_theta_deg = angles_deg[angle_idx]
    chosen_theta_rad = np.deg2rad(chosen_theta_deg)

    s = dx * np.cos(chosen_theta_rad) + dy * np.sin(chosen_theta_rad)

    # Convert s to nearest slice column
    s_idx = np.round((s - s_coords[0]) / (s_coords[-1] - s_coords[0]) * (ns - 1)).astype(int)
    s_idx = np.clip(s_idx, 0, ns - 1)

    # Fill volume
    for iz in range(nz):
        plane = np.zeros((ny, nx), dtype=recon_volume.dtype)

        valid_angle_idx = angle_idx[inside]
        valid_s_idx = s_idx[inside]

        plane_vals = slice_stack[valid_angle_idx, iz, valid_s_idx]
        plane[inside] = plane_vals

        recon_volume[iz] = plane

    return recon_volume

reconstructed_volume = reconstruct_volume_from_rotated_slices(
    all_slices=all_slices,
    angles_deg=angles_deg,
    nx=volume.shape[2],
    ny=volume.shape[1],
    nz=volume.shape[0],
    block_length_mm=block_length,
    block_height_mm=block_height,
    array_length_mm=array_length
)

np.save('reconstructed_volume.npy', reconstructed_volume)

viewer = napari.Viewer()
viewer.add_image(reconstructed_volume, name='Reconstructed Volume')

napari.run()

#%%
# Experimental Data

data_path = Path('../../DATA/1D TFM Data/Al Hole 5MHz 02022026 Filtered')
png_files = sorted(data_path.glob('*.png'))

all_slices = []

for file in png_files:
    img = mpimg.imread(file)

    # Convert to grayscale
    gray = 0.2989 * img[..., 0] + 0.5870 * img[..., 1] + 0.1140 * img[..., 2]
    gray = gray.astype(np.float32)

    # Normalise
    gray_min = gray.min()
    gray_max = gray.max()

    if gray_max > gray_min:
        gray = (gray - gray_min) / (gray_max - gray_min)
    else:
        gray = np.zeros_like(gray)

    gray = (gray * 255).astype(np.uint8)

    all_slices.append(gray)

print(f'{len(all_slices)} experimental slices')

angles_deg = np.linspace(0, 180, len(all_slices), endpoint=False)
slice_nz, slice_ns = all_slices[0].shape

reconstructed_exp_volume = reconstruct_volume_from_rotated_slices(
    all_slices=all_slices,
    angles_deg=angles_deg,
    nx=slice_ns,              
    ny=slice_ns,              
    nz=slice_nz,              
    block_length_mm=block_length,
    block_height_mm=block_height,
    array_length_mm=array_length
)

np.save('reconstructed_experimental_volume.npy', reconstructed_exp_volume)

viewer = napari.Viewer()
viewer.add_image(reconstructed_exp_volume, name='Reconstructed Experimental Volume')
napari.run()

#%%