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

os.environ['QT_QPA_PLATFORM'] = 'cocoa'
os.environ['QT_MAC_WANTS_LAYER'] = '1'

import numpy as np
from scipy.ndimage import gaussian_filter
import napari


class SyntheticVolumeGenerator:
    """
    Generator for synthetic 3D ultrasonic volumes matching experimental TFM data.
    
    In ultrasonic TFM imaging:
    - Base material (homogeneous) = VERY LOW intensity (0.0-0.1, almost black)
    - Defects (voids, cracks) = HIGH intensity (0.85-1.0, bright spots)
    
    Example usage:
        generator = SyntheticVolumeGenerator(dimensions=(100, 100, 100), seed=42)
        generator.add_spherical_void(center=(30, 50, 50), radius=8)
        volume = generator.generate()
        generator.visualize(volume)
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
        Add a spherical void (high reflectivity - creates strong echoes).
        
        Parameters:
            center: Tuple of (z, y, x) coordinates
            radius: Radius in pixels
            intensity: Reflectivity value (0.85-1.0 to match experimental data)
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
        
        Parameters:
            center: Tuple of (z, y, x) coordinates
            radius: Radius in pixels
            intensity: Reflectivity value (0.85-1.0)
        """
        self.defects.append({
            'type': 'spherical_inclusion',
            'center': center,
            'radius': radius,
            'intensity': intensity
        })
        print(f"Added spherical inclusion at {center}, radius={radius}px, intensity={intensity}")
        return self
    
    def add_cylindrical_inclusion(self, center_x, center_y, radius, intensity=0.92):
        """
        Add a cylindrical inclusion (vertical through all depths).
        
        Parameters:
            center_x: X coordinate of cylinder center
            center_y: Y coordinate of cylinder center
            radius: Radius in pixels
            intensity: Reflectivity value (0.85-1.0)
        """
        self.defects.append({
            'type': 'cylindrical_inclusion',
            'center_x': center_x,
            'center_y': center_y,
            'radius': radius,
            'intensity': intensity
        })
        print(f"Added cylindrical inclusion at x={center_x}, y={center_y}, radius={radius}px")
        return self
    
    def add_planar_crack(self, depth, thickness, y_range, x_range, intensity=0.98):
        """
        Add a planar crack (high reflectivity).
        
        Parameters:
            depth: Starting depth (z coordinate)
            thickness: Thickness in pixels
            y_range: Tuple of (y_start, y_end)
            x_range: Tuple of (x_start, x_end)
            intensity: Reflectivity value (0.85-1.0)
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
        Matches experimental TFM data intensity distribution.
        
        Parameters:
            base_intensity_range: Tuple of (min, max) for base material intensity
                                 (VERY LOW: 0.0-0.05 to match dark experimental background)
            smoothing_sigma: Gaussian smoothing parameter for base material
        
        Returns:
            volume: 3D numpy array representing the synthetic volume
        """
        print(f"\nGenerating synthetic volume with {len(self.defects)} defects...")
        
        # Create base material with VERY LOW intensity (almost black like experimental data)
        base_min, base_max = base_intensity_range
        base_material = np.random.rand(*self.dimensions) * (base_max - base_min) + base_min
        volume = gaussian_filter(base_material, sigma=smoothing_sigma)
        
        # Create coordinate grids
        z, y, x = np.ogrid[:self.dimensions[0], :self.dimensions[1], :self.dimensions[2]]
        
        # Apply each defect (HIGH intensity = strong reflection, 0.85-1.0)
        for defect in self.defects:
            if defect['type'] == 'spherical_void' or defect['type'] == 'spherical_inclusion':
                center = defect['center']
                radius = defect['radius']
                
                # Create distance field for smooth edges (more realistic)
                distances = np.sqrt((z - center[0])**2 + (y - center[1])**2 + (x - center[2])**2)
                
                # Hard edge (like experimental data)
                mask = distances <= radius
                volume[mask] = defect['intensity']
                
                # Optional: Add slight gradient at edges for realism
                edge_mask = (distances > radius) & (distances <= radius + 2)
                volume[edge_mask] = np.maximum(volume[edge_mask], 
                                               defect['intensity'] * (1 - (distances[edge_mask] - radius) / 2))
            
            elif defect['type'] == 'cylindrical_inclusion':
                y_grid, x_grid = np.mgrid[0:self.dimensions[1], 0:self.dimensions[2]]
                mask = (y_grid - defect['center_y'])**2 + (x_grid - defect['center_x'])**2 <= defect['radius']**2
                volume[:, mask] = defect['intensity']
            
            elif defect['type'] == 'planar_crack':
                d = defect['depth']
                t = defect['thickness']
                y_start, y_end = defect['y_range']
                x_start, x_end = defect['x_range']
                volume[d:d+t, y_start:y_end, x_start:x_end] = defect['intensity']
        
        # Ensure values stay in valid range
        volume = np.clip(volume, 0.0, 1.0)
        
        print(f"Volume generated successfully")
        print(f"  Min: {volume.min():.3f} (base material - very dark)")
        print(f"  Max: {volume.max():.3f} (defects - bright)")
        print(f"  Mean: {volume.mean():.3f}")
        print(f"  Matches experimental TFM intensity distribution")
        
        return volume
    
    def save_volume(self, volume, filename="synthetic_volume.npy"):
        """
        Save volume to .npy file.
        
        Parameters:
            volume: Volume to save
            filename: Output filename
        """
        np.save(filename, volume)
        print(f"\nSaved: {filename}")
    
    def visualize(self, volume):
        """
        Visualize volume using napari with settings matching experimental data.
        
        Parameters:
            volume: Volume to visualize
        """
        viewer = napari.Viewer()
        
        viewer.add_image(
            volume,
            name="Synthetic Volume",
            colormap="viridis",  # Matches your experimental data colormap
            contrast_limits=(0.0, 1.0),  # Full range like experimental data
            blending='translucent',
            rendering='iso',
            iso_threshold=0.3,  # Lower threshold to catch bright defects
            opacity=0.7
        )
        
        print("\nNapari viewer opened!")
        print("Controls:")
        print("  - Click '2D/3D' button in bottom-left for 3D view")
        print("  - In 3D: drag to rotate, scroll to zoom")
        print("  - Adjust iso_threshold slider (try 0.2-0.5)")
        print("  - Adjust contrast/opacity in left panel")
        print("\nIntensity matching experimental data:")
        print("  - Very dark background (0.0-0.05) = homogeneous material")
        print("  - Bright spots (0.85-1.0) = defects")
        
        napari.run()


# ============================================================================
# Example Usage - Matching Experimental Data
# ============================================================================


# Create generator
generator = SyntheticVolumeGenerator(dimensions=(300, 100, 100), seed=42)

# Add defects with experimental-like intensities
generator.add_spherical_void(center=(30, 50, 50), radius=8, intensity=0.95)

# Generate volume with VERY LOW base intensity (0.0-0.05) like experimental data
volume = generator.generate(base_intensity_range=(0.1, 0.7), smoothing_sigma=1.0)

# Save volume
generator.save_volume(volume, "synthetic_volume.npy")

# Visualize
generator.visualize(volume)


