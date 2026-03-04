#%%
# Imports
import numpy as np
import matplotlib.pyplot as plt

#%%
# Filter Parameters
theta_deg = 20
min_els   = 30

# Array geometry
min_x = -15e-3
max_x =  15e-3
no_elements = 128

# Search Grid
x_resolution = 300
z_resolution = 300
z_depth = 50e-3

#%%
# Calculated Parameters
theta_rad = np.deg2rad(theta_deg)

x_range = np.linspace(min_x, max_x, x_resolution)
z_range = np.linspace(0, z_depth, z_resolution)

X, Z = np.meshgrid(x_range, z_range)
element_x = np.linspace(min_x, max_x, no_elements)

#%%
# Binary Image
x_spread = Z * np.tan(theta_rad)
element_x_3d = element_x[np.newaxis, np.newaxis, :]
illumination_mask = np.abs(element_x_3d - X[:, :, np.newaxis]) <= x_spread[:, :, np.newaxis]

# Count illuminating elements
num_elements = np.sum(illumination_mask, axis=2)

# Full cone inside array aperture
x_left  = X - x_spread
x_right = X + x_spread

cone_fits = (x_left >= min_x) & (x_right <= max_x)

# Binary image
binary_image = (
    (num_elements >= min_els) &
    cone_fits
).astype(int)

#%%
# Visualisation
plt.imshow(binary_image,
           extent=[min_x, max_x, z_depth, 0],
           aspect='auto',
           cmap='grey')

plt.xlabel("x (m)")
plt.ylabel("z (m)")
plt.title("Angular Illumination Filter")
plt.colorbar()
plt.show()
#%%