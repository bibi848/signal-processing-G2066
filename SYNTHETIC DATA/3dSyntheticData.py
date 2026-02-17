"""
3D Synthetic Data Generation and Reconstruction for Ultrasonic NDT
This module provides a class for generating synthetic 3D volumes with defects.

Methodology:
1. Generate synthetic 3D volume with customizable defects matching experimental TFM data
2. Save as .npy file
3. Visualize using napari
"""

import os
import sys
import numpy as np
from scipy.ndimage import gaussian_filter
import napari

# # Configure QT environment for macOS
# os.environ['QT_QPA_PLATFORM'] = 'cocoa'
# os.environ['QT_MAC_WANTS_LAYER'] = '1'

# TODO: add noise generation for volumes, add imaging artefact around defects, maybe add attenuation 
# properties (more noise at depth) 

class SyntheticVolumeGenerator:
    """
    Generator for synthetic 3D ultrasonic volumes matching experimental TFM data.
    
    In ultrasonic TFM imaging:
    - Base material (homogeneous) = VERY LOW intensity (0.0-0.1, almost black)
    - Defects (voids, cracks) = HIGH intensity (0.85-1.0, bright spots)
    """
    
    def __init__(self, dimensions=(100, 100, 100), seed=None):
        """
        Initialize the synthetic volume generator.
        
        Parameters:
            dimensions: Tuple of (depth, height, width) in pixels
            seed: Random seed for reproducibility
        """
        self.dimensions = dimensions
        self.defects = []
        
        if seed is not None:
            np.random.seed(seed)
        
        print(f"Initialized SyntheticVolumeGenerator with dimensions: {dimensions}")
    
    def add_spherical_void(self, center, radius, intensity=0.95):
        """
        Add a spherical void (high reflectivity).
        """
        self.defects.append({
            'type': 'spherical_void',
            'center': center,
            'radius': radius,
            'intensity': intensity
        })
        print(f"Added spherical void at {center}, radius={radius}px, intensity={intensity}")
        return self
    
    def add_spherical_inclusion(self, center, radius, intensity=0.90):
        """
        Add a spherical inclusion (high reflectivity).
        """
        self.defects.append({
            'type': 'spherical_inclusion',
            'center': center,
            'radius': radius,
            'intensity': intensity
        })
        print(f"Added spherical inclusion at {center}, radius={radius}px, intensity={intensity}")
        return self
    
    def add_cylindrical_void(self, center_x, center_y, radius, intensity=0.95, axis='z'):
        """
        Add a cylindrical void (high reflectivity).
        
        Parameters:
            center_x: X coordinate of cylinder center (or Z if axis='x'/'y' depending on orientation logic)
            center_y: Y coordinate of cylinder center
            radius: Radius in pixels
            intensity: Reflectivity value (0.85-1.0)
            axis: Axis along which cylinder extends ('x', 'y', or 'z')
        """
        self.defects.append({
            'type': 'cylindrical_void',
            'center_x': center_x,
            'center_y': center_y,
            'radius': radius,
            'intensity': intensity,
            'axis': axis
        })
        print(f"Added cylindrical void at x={center_x}, y={center_y}, radius={radius}px, axis={axis}")
        return self
    
    def add_planar_crack(self, depth, thickness, y_range, x_range, intensity=0.98):
        """
        Add a planar crack (high reflectivity).
        """
        self.defects.append({
            'type': 'planar_crack',
            'depth': depth,
            'thickness': thickness,
            'y_range': y_range,
            'x_range': x_range,
            'intensity': intensity
        })
        print(f"Added planar crack at depth={depth}, thickness={thickness}px")
        return self
    
    def generate(self, base_intensity_range=(0.05, 0.1), smoothing_sigma=2.0):
        """
        Generate the synthetic volume with all added defects.
        """
        print(f"\nGenerating synthetic volume with {len(self.defects)} defects...")
        
        # Create base material with VERY LOW intensity
        base_min, base_max = base_intensity_range
        base_material = np.random.rand(*self.dimensions) * (base_max - base_min) + base_min
        volume = gaussian_filter(base_material, sigma=smoothing_sigma)
        
        # Create coordinate grids
        z, y, x = np.ogrid[:self.dimensions[0], :self.dimensions[1], :self.dimensions[2]]
        
        # Apply each defect
        for defect in self.defects:
            if defect['type'] == 'spherical_void' or defect['type'] == 'spherical_inclusion':
                center = defect['center']
                radius = defect['radius']
                
                distances = np.sqrt((z - center[0])**2 + (y - center[1])**2 + (x - center[2])**2)
                
                mask = distances <= radius
                volume[mask] = defect['intensity']
                
                # Soft edge
                edge_mask = (distances > radius) & (distances <= radius + 2)
                volume[edge_mask] = np.maximum(volume[edge_mask], 
                                               defect['intensity'] * (1 - (distances[edge_mask] - radius) / 2))
            
            elif defect['type'] == 'cylindrical_void':
                axis = defect.get('axis', 'z')
                if axis == 'z':
                    y_grid, x_grid = np.mgrid[0:self.dimensions[1], 0:self.dimensions[2]]
                    mask = (y_grid - defect['center_y'])**2 + (x_grid - defect['center_x'])**2 <= defect['radius']**2
                    volume[:, mask] = defect['intensity']
                
                elif axis == 'x':
                    z_grid, y_grid = np.mgrid[0:self.dimensions[0], 0:self.dimensions[1]]
                    mask = (z_grid - defect['center_x'])**2 + (y_grid - defect['center_y'])**2 <= defect['radius']**2
                    volume[mask, :] = defect['intensity']
                
                elif axis == 'y':
                    z_grid, x_grid = np.mgrid[0:self.dimensions[0], 0:self.dimensions[2]]
                    mask = (z_grid - defect['center_y'])**2 + (x_grid - defect['center_x'])**2 <= defect['radius']**2
                    volume[:, :, :][mask[:, np.newaxis, :].repeat(self.dimensions[1], axis=1)] = defect['intensity']

            elif defect['type'] == 'planar_crack':
                d = defect['depth']
                t = defect['thickness']
                y_start, y_end = defect['y_range']
                x_start, x_end = defect['x_range']
                volume[d:d+t, y_start:y_end, x_start:x_end] = defect['intensity']
        
        # Ensure values stay in valid range
        volume = np.clip(volume, 0.0, 1.0)
        
        print(f"Volume generated successfully. Min: {volume.min():.3f}, Max: {volume.max():.3f}")
        return volume
    
    def save_volume(self, volume, filename="synthetic_volume.npy"):
        np.save(filename, volume)
        print(f"\nSaved: {filename}")
    
    def visualize(self, volume):
        viewer = napari.Viewer()
        viewer.add_image(
            volume,
            name="Synthetic Volume",
            colormap="viridis",
            contrast_limits=(0.0, 1.0),
            blending='translucent',
            rendering='iso',
            iso_threshold=1,
            opacity=0.7
        )
        napari.run()

    def split_volume_for_stitching(self, volume, num_splits=(2, 2, 2), overlap_pixels=(20, 20, 20), save_dir=None):
        """
        Split volume into overlapping sub-volumes of IDENTICAL dimensions.
        """
        print(f"\nSplitting volume into {num_splits[0]}x{num_splits[1]}x{num_splits[2]} sub-volumes...")
        print(f"Overlap pixels between adjacent volumes: {overlap_pixels}")
        
        z_dim, y_dim, x_dim = volume.shape
        z_splits, y_splits, x_splits = num_splits
        overlap_z, overlap_y, overlap_x = overlap_pixels
        
        # Calculate sub-volume size
        subvol_z = (z_dim + (z_splits - 1) * overlap_z) // z_splits
        subvol_y = (y_dim + (y_splits - 1) * overlap_y) // y_splits
        subvol_x = (x_dim + (x_splits - 1) * overlap_x) // x_splits
        
        subvol_shape = (subvol_z, subvol_y, subvol_x)
        print(f"Each sub-volume shape: {subvol_shape}")
        
        # Calculate step size
        step_z = subvol_z - overlap_z
        step_y = subvol_y - overlap_y
        step_x = subvol_x - overlap_x
        
        # Check padding
        expected_z = (z_splits - 1) * step_z + subvol_z
        expected_y = (y_splits - 1) * step_y + subvol_y
        expected_x = (x_splits - 1) * step_x + subvol_x
        
        if expected_z > z_dim or expected_y > y_dim or expected_x > x_dim:
            print(f"Warning: Volume padded from ({z_dim},{y_dim},{x_dim}) to ({expected_z},{expected_y},{expected_x})")
            padded = np.zeros((expected_z, expected_y, expected_x), dtype=volume.dtype)
            padded[:z_dim, :y_dim, :x_dim] = volume
            volume = padded
        
        sub_volumes = []
        
        for zi in range(z_splits):
            for yi in range(y_splits):
                for xi in range(x_splits):
                    z_start = zi * step_z
                    y_start = yi * step_y
                    x_start = xi * step_x
                    
                    sub_vol = volume[
                        z_start:z_start + subvol_z,
                        y_start:y_start + subvol_y,
                        x_start:x_start + subvol_x
                    ].copy()
                    
                    metadata = {
                        'index': (zi, yi, xi),
                        'volume': sub_vol,
                        'shape': sub_vol.shape,
                        'origin': (z_start, y_start, x_start),
                        'subvol_shape': subvol_shape,
                        'overlap': overlap_pixels,
                        'step': (step_z, step_y, step_x)
                    }
                    sub_volumes.append(metadata)
                    print(f"  Sub-volume [{zi},{yi},{xi}]: origin={metadata['origin']}, shape={sub_vol.shape}")
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            for sv in sub_volumes:
                idx = sv['index']
                filename = os.path.join(save_dir, f"subvol_{idx[0]}_{idx[1]}_{idx[2]}.npy")
                np.save(filename, sv['volume'])
                
                meta = {k: v for k, v in sv.items() if k != 'volume'}
                meta_filename = os.path.join(save_dir, f"subvol_{idx[0]}_{idx[1]}_{idx[2]}_meta.npy")
                np.savez(meta_filename, **meta)
            
            recon_info = {
                'original_shape': (z_dim, y_dim, x_dim),
                'num_splits': num_splits,
                'overlap_pixels': overlap_pixels,
                'subvol_shape': subvol_shape,
                'step': (step_z, step_y, step_x),
                'num_subvolumes': len(sub_volumes)
            }
            np.savez(os.path.join(save_dir, 'reconstruction_info.npz'), **recon_info)
            print(f"\nSaved {len(sub_volumes)} sub-volumes to: {save_dir}")
        
        return sub_volumes
    
    @staticmethod
    def reconstruct_volume(sub_volumes, original_shape=None):
        """
        Reconstruct original volume using weighted blending.
        """
        print("\nReconstructing volume from sub-volumes...")
        subvol_shape = sub_volumes[0]['subvol_shape']
        
        if original_shape is None:
            max_z = max(sv['origin'][0] + subvol_shape[0] for sv in sub_volumes)
            max_y = max(sv['origin'][1] + subvol_shape[1] for sv in sub_volumes)
            max_x = max(sv['origin'][2] + subvol_shape[2] for sv in sub_volumes)
            original_shape = (max_z, max_y, max_x)
        
        reconstructed = np.zeros(original_shape, dtype=np.float64)
        weights = np.zeros(original_shape, dtype=np.float64)
        
        for sv in sub_volumes:
            origin = sv['origin']
            vol = sv['volume']
            
            z_s, y_s, x_s = origin
            z_e = min(z_s + vol.shape[0], original_shape[0])
            y_e = min(y_s + vol.shape[1], original_shape[1])
            x_e = min(x_s + vol.shape[2], original_shape[2])
            
            vol_z = z_e - z_s
            vol_y = y_e - y_s
            vol_x = x_e - x_s
            
            reconstructed[z_s:z_e, y_s:y_e, x_s:x_e] += vol[:vol_z, :vol_y, :vol_x]
            weights[z_s:z_e, y_s:y_e, x_s:x_e] += 1.0
            
        weights[weights == 0] = 1
        reconstructed /= weights
        print("Reconstruction complete!")
        return reconstructed
    
    @staticmethod
    def load_sub_volumes(load_dir):
        print(f"\nLoading sub-volumes from: {load_dir}")
        recon_info = np.load(os.path.join(load_dir, 'reconstruction_info.npy'), allow_pickle=True).item()
        sub_volumes = []
        for filename in sorted(os.listdir(load_dir)):
            if filename.startswith('subvol_') and filename.endswith('.npy') and '_meta' not in filename:
                vol_path = os.path.join(load_dir, filename)
                volume = np.load(vol_path)
                meta_path = vol_path.replace('.npy', '_meta.npy')
                metadata = np.load(meta_path, allow_pickle=True).item()
                metadata['volume'] = volume
                sub_volumes.append(metadata)
        return sub_volumes, recon_info
    
    def verify_reconstruction(self, original, reconstructed):
        if original.shape != reconstructed.shape:
            print(f"Shape mismatch: {original.shape} vs {reconstructed.shape}")
            return False
        diff = np.abs(original - reconstructed)
        print(f"Max difference: {diff.max():.10f}")
        if diff.max() < 1e-10:
            print("  ✓ Perfect reconstruction!")
            return True
        else:
            print("  ✗ Reconstruction has differences")
            return False

    def visualize_sub_volumes(self, sub_volumes):
        """
        Visualize all sub-volumes in napari.
        Corrects rotation issue by using (z,y,x) for translate.
        """
        viewer = napari.Viewer()
        colormaps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'turbo']
        
        for i, sv in enumerate(sub_volumes):
            idx = sv['index']
            viewer.add_image(
                sv['volume'],
                name=f"SubVol {idx}",
                colormap=colormaps[i % len(colormaps)],
                contrast_limits=(0.0, 1.0),
                blending='translucent',
                rendering='iso',
                iso_threshold=1,
                opacity=0.5,
                translate=sv['origin']
            )
        napari.run()

# ============================================================================
# Example Usage
# ============================================================================

# Create generator
generator = SyntheticVolumeGenerator(dimensions=(100, 100, 110), seed=42)

# Add defects (using cylindrical_void instead of inclusion)
generator.add_spherical_void(center=(30, 50, 50), radius=8, intensity=0.95)
generator.add_cylindrical_void(center_x=70, center_y=50, radius=5, intensity=0.92, axis='y')

# Generate volume
volume = generator.generate(base_intensity_range=(0.9, 0.9), smoothing_sigma=2.0)
generator.save_volume(volume, "synthetic_volume.npy")

# Split into sub-volumes with overlapped identical dimensions
sub_volumes = generator.split_volume_for_stitching(
    volume,
    num_splits=(1, 1, 2),
    overlap_pixels=(0, 0, 80),
    save_dir="SYNTHETIC NPY/stitching_test"
)

# Reconstruct
reconstructed = SyntheticVolumeGenerator.reconstruct_volume(sub_volumes, volume.shape)

# Verify
generator.verify_reconstruction(volume, reconstructed)

# Visualize
generator.visualize(volume)
generator.visualize_sub_volumes(sub_volumes)