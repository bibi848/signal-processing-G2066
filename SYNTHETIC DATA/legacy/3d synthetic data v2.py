"""
3D Synthetic Data Generation and Reconstruction for Ultrasonic NDT - OPTIMIZED
This module provides a class for generating synthetic 3D volumes with defects
and realistic imaging artifacts.

COORDINATE SYSTEM (matching real TFM setup):
- Axis 0 (depth/z): Penetration depth into material (array scans from the SIDE) - BEST RESOLUTION
- Axis 1 (height/y): Vertical direction (perpendicular to array surface) - MODERATE RESOLUTION
- Axis 2 (width/x): Lateral direction (along array length) - WORST RESOLUTION

Array is positioned on the SIDE of the sample, scanning horizontally into the material.
The characteristic TFM "bloom" spreads in the Y-X PLANE (perpendicular to beam propagation):
- 2D Hilbert envelope applied to axes 1 (y) and 2 (x)
- Creates elliptical bloom: strongest laterally, moderate in elevation, minimal in depth

Methodology:
1. Generate synthetic 3D volume with customizable defects matching experimental TFM data
2. Add realistic ultrasonic imaging artifacts (noise, attenuation, anisotropic PSF, etc.)
3. Apply 2D Hilbert envelope along height and lateral axes (creates realistic bloom in y-x plane)
4. Save as .npy file
5. Visualize using napari

OPTIMIZATIONS:
- Vectorized cylindrical void generation (no loops)
- Efficient convolution using FFT for large kernels
- Reduced redundant operations
- Memory-efficient processing
- 2D Hilbert transform (axes 1 & 2) for realistic TFM bloom in y-x plane
"""

# Configure Qt for macOS before importing napari
import os
import sys

# Configure Qt for macOS before importing napari
if sys.platform == 'darwin':  # macOS
    import sysconfig
    
    qt_plugin_path = None
    
    # Method 1: Check site-packages
    try:
        import site
        site_packages_list = site.getsitepackages()
        if site_packages_list and len(site_packages_list) > 0:
            for sp in site_packages_list:
                test_path = os.path.join(sp, 'PyQt5', 'Qt5', 'plugins')
                if os.path.exists(test_path):
                    qt_plugin_path = test_path
                    print(f"✓ Found Qt plugins via site.getsitepackages(): {qt_plugin_path}")
                    break
    except Exception as e:
        print(f"Method 1 failed: {e}")
    
    # Method 2: Check virtual environment lib path
    if qt_plugin_path is None:
        try:
            lib_path = sysconfig.get_path('purelib')
            if lib_path:
                test_path = os.path.join(lib_path, 'PyQt5', 'Qt5', 'plugins')
                if os.path.exists(test_path):
                    qt_plugin_path = test_path
                    print(f"✓ Found Qt plugins via sysconfig.get_path(): {qt_plugin_path}")
        except Exception as e:
            print(f"Method 2 failed: {e}")
    
    # Method 3: Try sys.prefix
    if qt_plugin_path is None:
        try:
            base_path = os.path.join(sys.prefix, 'lib', f'python{sys.version_info.major}.{sys.version_info.minor}', 'site-packages')
            test_path = os.path.join(base_path, 'PyQt5', 'Qt5', 'plugins')
            if os.path.exists(test_path):
                qt_plugin_path = test_path
                print(f"✓ Found Qt plugins via sys.prefix: {qt_plugin_path}")
        except Exception as e:
            print(f"Method 3 failed: {e}")
    
    # Set environment variables if path was found
    if qt_plugin_path:
        os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = qt_plugin_path
        os.environ['QT_QPA_PLATFORM'] = 'cocoa'
        os.environ['QT_LOGGING_RULES'] = '*.debug=false;qt.qpa.*=false'
        print(f"✓ Qt environment configured successfully")
    else:
        print("✗ WARNING: Could not find Qt plugin path!")
        print("  Attempting to continue anyway...")
        # Set platform anyway - Qt might find plugins on its own
        os.environ['QT_QPA_PLATFORM'] = 'cocoa'

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import hilbert, fftconvolve
from scipy.fft import fftn, ifftn, fftshift
import napari


class SyntheticVolumeGenerator:
    """
    Generator for synthetic 3D ultrasonic volumes matching experimental TFM data.
    
    COORDINATE CONVENTION:
    - dimensions = (depth, height, width) = (z, y, x)
    - Depth (z, axis 0): Into material, from array surface (0) to far side
    - Height (y, axis 1): Vertical direction
    - Width (x, axis 2): Lateral/horizontal direction along array
    
    In ultrasonic TFM imaging:
    - Base material (homogeneous) = VERY LOW intensity (0.0-0.1, almost black)
    - Defects (voids, cracks) = HIGH intensity (0.85-1.0, bright spots)
    """
    
    def __init__(self, dimensions=(100, 100, 100), seed=None, array_position=None):
        """
        Initialize the synthetic volume generator.
        
        Parameters:
            dimensions: Tuple of (depth, height, width) in pixels
                       depth (z): scanning depth into material (axis 0)
                       height (y): vertical extent (axis 1)
                       width (x): lateral extent (axis 2)
            seed: Random seed for reproducibility
            array_position: Tuple (y, x) position of the ultrasound array
                           If None, defaults to center of volume (y_center, x_center)
                           Array is positioned on the SIDE of the sample (z=0)
        """
        self.dimensions = dimensions
        self.defects = []
        
        # Set array position (defaults to center of y-x plane)
        if array_position is None:
            self.array_position = (dimensions[1] // 2, dimensions[2] // 2)
        else:
            self.array_position = array_position
        
        if seed is not None:
            np.random.seed(seed)
    
    def add_spherical_void(self, center, radius, intensity=0.95):
        """
        Add a spherical void (high reflectivity).
        
        Parameters:
            center: (depth, height, width) = (z, y, x) position
            radius: Radius in pixels
            intensity: Reflectivity value (0.85-1.0)
        """
        self.defects.append({
            'type': 'spherical_void',
            'center': center,
            'radius': radius,
            'intensity': intensity
        })
        return self
    
    def add_cylindrical_void(self, center_pos, other_pos, radius, intensity=0.95, axis='x'):
        """
        Add a cylindrical void (high reflectivity).
        
        Parameters:
            center_pos: Position along first perpendicular axis
            other_pos: Position along second perpendicular axis
            radius: Radius in pixels
            intensity: Reflectivity value (0.85-1.0)
            axis: Axis along which cylinder extends
                  'x' (lateral): cylinder extends horizontally along array
                  'y' (vertical): cylinder extends vertically  
                  'z' (depth): cylinder extends into material depth
        """
        self.defects.append({
            'type': 'cylindrical_void',
            'center_pos': center_pos,
            'other_pos': other_pos,
            'radius': radius,
            'intensity': intensity,
            'axis': axis
        })
        return self
    
    def set_array_position(self, y_pos, x_pos):
        """
        Update the array position to simulate array movement.
        
        Parameters:
            y_pos: Y-coordinate of array position (vertical)
            x_pos: X-coordinate of array position (lateral)
        """
        self.array_position = (y_pos, x_pos)
        return self
    
    def _calculate_angular_attenuation(self, defect_center, defect_type='spherical'):
        """
        Calculate angular-dependent attenuation for a defect based on array position.
        
        Physics:
        - When array is directly above defect: maximum signal return (angle ≈ 0°)
        - When array is off-axis: reduced signal due to:
          1. Reduced reflectivity at oblique angles
          2. Beam divergence (weaker signal off-center)
          3. Reduced aperture overlap
        
        Parameters:
            defect_center: (z, y, x) position of defect center
            defect_type: 'spherical' or 'cylindrical' (different angular responses)
        
        Returns:
            attenuation_factor: float [0, 1] - multiplicative factor for defect intensity
                              1.0 = directly above (no attenuation)
                              <1.0 = off-axis (reduced intensity)
        """
        z_defect, y_defect, x_defect = defect_center
        y_array, x_array = self.array_position
        
        # Calculate lateral offset (in y-x plane)
        lateral_offset = np.sqrt((y_defect - y_array)**2 + (x_defect - x_array)**2)
        
        # Calculate angle from array normal (assuming array points into +z direction)
        # angle ≈ arctan(lateral_offset / depth)
        angle_rad = np.arctan2(lateral_offset, z_defect + 1e-6)  # Add small epsilon to avoid division by zero
        angle_deg = np.degrees(angle_rad)
        
        # Attenuation model based on angle - REALISTIC ULTRASOUND RESPONSE
        # Ultrasound arrays typically have ~60-80° effective beam width
        # For spherical defects: cosine response (Lambertian-like reflector)
        # For cylindrical defects: slightly broader response
        if defect_type == 'spherical':
            # Moderate drop-off for spherical defects (cos^1.5 for gentler falloff)
            attenuation = np.cos(angle_rad) ** 1.5
        else:
            # Broader response for cylindrical defects
            attenuation = np.cos(angle_rad) ** 1.0
        
        # Beam divergence factor (Gaussian beam profile)
        # Assumes -6dB beam width of ~50° (typical for phased array ultrasound)
        # This gives good signal out to ~40° and gradually falls off beyond that
        beam_width_deg = 50.0  # -6dB beam width
        beam_sigma = beam_width_deg / 2.355  # Convert FWHM to sigma (for Gaussian)
        beam_factor = np.exp(-(angle_deg**2) / (2 * beam_sigma**2))
        
        # Combined attenuation (multiplicative)
        # Use weighted combination to avoid over-attenuation
        total_attenuation = 0.6 * attenuation + 0.4 * beam_factor
        
        # Ensure minimum visibility even at extreme angles (noise floor)
        total_attenuation = np.maximum(total_attenuation, 0.05)
        
        return total_attenuation
    
    def generate(self, base_intensity_range=(0.05, 0.1), smoothing_sigma=2.0, use_angular_effects=False):
        """
        Generate the synthetic volume with all added defects - OPTIMIZED.
        
        Parameters:
            base_intensity_range: Intensity range for base material
            smoothing_sigma: Gaussian smoothing for base material
            use_angular_effects: If True, apply angular-dependent intensity modulation
                                based on array position (simulates off-axis imaging)
                                DEFAULT: False (ground truth generation without angular effects)
        """
        # Create base material with VERY LOW intensity
        base_min, base_max = base_intensity_range
        base_material = np.random.rand(*self.dimensions).astype(np.float32) * (base_max - base_min) + base_min
        volume = gaussian_filter(base_material, sigma=smoothing_sigma)
        
        # Create coordinate grids ONCE (reuse for all defects)
        z, y, x = np.ogrid[:self.dimensions[0], :self.dimensions[1], :self.dimensions[2]]
        
        # Apply each defect
        for defect in self.defects:
            if defect['type'] == 'spherical_void':
                center = defect['center']
                radius = defect['radius']
                base_intensity = defect['intensity']
                
                # Apply angular attenuation if enabled
                if use_angular_effects:
                    angular_factor = self._calculate_angular_attenuation(center, defect_type='spherical')
                    effective_intensity = base_intensity * angular_factor
                else:
                    effective_intensity = base_intensity
                
                distances = np.sqrt((z - center[0])**2 + (y - center[1])**2 + (x - center[2])**2)
                
                mask = distances <= radius
                volume[mask] = effective_intensity
                
                # Soft edge
                edge_mask = (distances > radius) & (distances <= radius + 2)
                volume[edge_mask] = np.maximum(volume[edge_mask], 
                                               effective_intensity * (1 - (distances[edge_mask] - radius) / 2))
            
            elif defect['type'] == 'cylindrical_void':
                # Simple uniform cylinder (ground truth)
                axis = defect.get('axis', 'z')
                center_pos = defect['center_pos']
                other_pos = defect['other_pos']
                radius = defect['radius']
                base_intensity = defect['intensity']
                
                # For cylinders, calculate angular attenuation at cylinder center
                if use_angular_effects:
                    if axis == 'x':
                        cylinder_center = (center_pos, other_pos, self.dimensions[2] // 2)
                    elif axis == 'y':
                        cylinder_center = (center_pos, self.dimensions[1] // 2, other_pos)
                    elif axis == 'z':
                        cylinder_center = (self.dimensions[0] // 2, center_pos, other_pos)
                    
                    angular_factor = self._calculate_angular_attenuation(cylinder_center, defect_type='cylindrical')
                    effective_intensity = base_intensity * angular_factor
                else:
                    effective_intensity = base_intensity
                
                if axis == 'x':  # Cylinder extends along lateral direction (x-axis)
                    z_grid = np.arange(self.dimensions[0])[:, None, None]
                    y_grid = np.arange(self.dimensions[1])[None, :, None]
                    distances = np.sqrt((z_grid - center_pos)**2 + (y_grid - other_pos)**2)
                    mask = distances <= radius
                    volume[mask.squeeze()] = effective_intensity
                
                elif axis == 'y':  # Cylinder extends vertically (y-axis)
                    z_grid = np.arange(self.dimensions[0])[:, None, None]
                    x_grid = np.arange(self.dimensions[2])[None, None, :]
                    distances = np.sqrt((z_grid - center_pos)**2 + (x_grid - other_pos)**2)
                    mask = distances <= radius
                    volume[:, :, :][np.broadcast_to(mask, self.dimensions)] = effective_intensity
                
                elif axis == 'z':  # Cylinder extends into depth (z-axis)
                    y_grid = np.arange(self.dimensions[1])[None, :, None]
                    x_grid = np.arange(self.dimensions[2])[None, None, :]
                    distances = np.sqrt((y_grid - center_pos)**2 + (x_grid - other_pos)**2)
                    mask = distances <= radius
                    volume[:, :, :][np.broadcast_to(mask, self.dimensions)] = effective_intensity
        
        # Ensure values stay in valid range
        volume = np.clip(volume, 0.0, 1.0)
        return volume
    
    def add_cylindrical_surface_markers(self, volume, top_intensity=0.45, bottom_intensity=0.30, 
                                         bloom_sigma=12.0):
        """
        Add DIRECT 2D Gaussian blooming in X-Y plane from cylinder's top and bottom surfaces.
        
        This creates the blooming effect DIRECTLY without relying on Hilbert spreading.
        The blooming:
        - Originates AT the cylinder's actual top (z+radius) and bottom (z-radius) surfaces
        - Spreads outward in X-Y directions as a 2D Gaussian distribution
        - Extends along the ENTIRE length of the cylinder
        - Uses asymmetric intensity (stronger at top, weaker at bottom, matching ultrasound physics)
        
        Parameters:
            volume: Input volume with cylindrical defects
            top_intensity: Peak blooming intensity at top surface (default 0.45)
            bottom_intensity: Peak blooming intensity at bottom surface (default 0.30)
            bloom_sigma: Gaussian sigma for bloom spreading in X-Y plane (pixels, default 12.0)
                        Larger values = wider blooming
        """
        bloom_volume = np.zeros_like(volume, dtype=np.float32)
        
        for defect in self.defects:
            if defect['type'] != 'cylindrical_void':
                continue
                
            axis = defect.get('axis', 'z')
            center_pos = defect['center_pos']  # z-coordinate of cylinder center
            other_pos = defect['other_pos']     # other coordinate (x or y)
            radius = defect['radius']
            
            # Create coordinate grids
            y_grid = np.arange(self.dimensions[1])[None, :, None]
            x_grid = np.arange(self.dimensions[2])[None, None, :]
            
            if axis == 'x':
                # Cylinder extends along X-axis (constant Y position, varies in X)
                # Top and bottom surfaces are at z = center_pos ± radius
                
                z_top = int(np.round(center_pos + radius))
                z_bottom = int(np.round(center_pos - radius))
                
                # Ensure within bounds
                z_top = np.clip(z_top, 0, self.dimensions[0] - 1)
                z_bottom = np.clip(z_bottom, 0, self.dimensions[0] - 1)
                
                # Distance from cylinder centerline in Y direction
                y_dist = np.abs(y_grid - other_pos)
                
                # Create 2D Gaussian blooming in Y direction (perpendicular to cylinder)
                # Extends along entire X axis (parallel to cylinder)
                for z_surf, intensity in [(z_top, top_intensity), (z_bottom, bottom_intensity)]:
                    bloom_profile = intensity * np.exp(-(y_dist / bloom_sigma)**2)
                    
                    # Apply to the specific Z surface
                    bloom_volume[z_surf, :, :] = np.maximum(
                        bloom_volume[z_surf, :, :],
                        bloom_profile[0, :, :]
                    )
            
            elif axis == 'y':
                # Cylinder extends along Y-axis (constant X position, varies in Y)
                # Top and bottom surfaces are at z = center_pos ± radius
                
                z_top = int(np.round(center_pos + radius))
                z_bottom = int(np.round(center_pos - radius))
                
                # Ensure within bounds
                z_top = np.clip(z_top, 0, self.dimensions[0] - 1)
                z_bottom = np.clip(z_bottom, 0, self.dimensions[0] - 1)
                
                # Distance from cylinder centerline in X direction
                x_dist = np.abs(x_grid - other_pos)
                
                # Create 2D Gaussian blooming in X direction (perpendicular to cylinder)
                # Extends along entire Y axis (parallel to cylinder)
                for z_surf, intensity in [(z_top, top_intensity), (z_bottom, bottom_intensity)]:
                    bloom_profile = intensity * np.exp(-(x_dist / bloom_sigma)**2)
                    
                    # Apply to the specific Z surface
                    bloom_volume[z_surf, :, :] = np.maximum(
                        bloom_volume[z_surf, :, :],
                        bloom_profile[0, :, :]
                    )
            
            elif axis == 'z':
                # Cylinder extends along Z-axis (parallel to beam)
                # For Z-axis cylinders, create radial blooming
                
                z_grid = np.arange(self.dimensions[0])[:, None, None]
                dist_y = y_grid - center_pos
                dist_x = x_grid - other_pos
                radial_dist = np.sqrt(dist_y**2 + dist_x**2)
                
                # Blooming spreads radially from cylinder edge
                edge_bloom = top_intensity * np.exp(-((radial_dist - radius) / bloom_sigma)**2)
                edge_bloom = np.broadcast_to(edge_bloom, self.dimensions)
                bloom_volume = np.maximum(bloom_volume, edge_bloom)
        
        # Add blooming to volume (additive, not replacement)
        volume_with_bloom = volume + bloom_volume
        return np.clip(volume_with_bloom, 0.0, 1.0)
    
    def add_cylindrical_blooming(self, volume, bloom_sigma_z=3.0, bloom_sigma_radial=5.0, bloom_intensity=0.5):
        """
        [DEPRECATED - Use add_cylindrical_surface_markers instead]
        
        Add blooming at the top and bottom Z-coordinates of cylindrical defects.
        
        This creates bright hotspots where the cylinder surfaces meet the top/bottom
        boundaries along the Z-axis (depth direction).
        
        Parameters:
            volume: Input volume with cylindrical defects
            bloom_sigma_z: Gaussian spread along z-axis (depth)
            bloom_sigma_radial: Gaussian spread in radial direction (x-y plane)
            bloom_intensity: Intensity boost for the blooming effect
        """
        bloom_map = np.zeros_like(volume, dtype=np.float32)
        
        for defect in self.defects:
            if defect['type'] != 'cylindrical_void':
                continue
                
            axis = defect.get('axis', 'z')
            center_pos = defect['center_pos']
            other_pos = defect['other_pos']
            radius = defect['radius']
            
            # Create grids
            z_grid = np.arange(self.dimensions[0])[:, None, None]
            y_grid = np.arange(self.dimensions[1])[None, :, None]
            x_grid = np.arange(self.dimensions[2])[None, None, :]
            
            # Find the z-coordinates of the cylinder's top and bottom surfaces
            if axis == 'x':
                # Cylinder extends along x, circular cross-section in y-z plane
                # Top z: center_pos + radius, Bottom z: center_pos - radius
                z_top = center_pos + radius
                z_bottom = center_pos - radius
                
                # Create hotspots at top and bottom z-coordinates
                for z_surface in [z_top, z_bottom]:
                    # Distance from the circular edge at this z-height
                    dist_from_z = np.abs(z_grid - z_surface)
                    dist_from_axis_y = np.abs(y_grid - other_pos)
                    
                    # Gaussian bloom in z-direction and radially
                    z_bloom = np.exp(-(dist_from_z**2) / (2 * bloom_sigma_z**2))
                    radial_bloom = np.exp(-(dist_from_axis_y**2) / (2 * bloom_sigma_radial**2))
                    
                    bloom_map += z_bloom * radial_bloom * bloom_intensity
            
            elif axis == 'y':
                # Cylinder extends along y, circular cross-section in x-z plane  
                # Top z: center_pos + radius, Bottom z: center_pos - radius
                z_top = center_pos + radius
                z_bottom = center_pos - radius
                
                for z_surface in [z_top, z_bottom]:
                    # Distance from the circular edge at this z-height
                    dist_from_z = np.abs(z_grid - z_surface)
                    dist_from_axis_x = np.abs(x_grid - other_pos)
                    
                    # Gaussian bloom
                    z_bloom = np.exp(-(dist_from_z**2) / (2 * bloom_sigma_z**2))
                    radial_bloom = np.exp(-(dist_from_axis_x**2) / (2 * bloom_sigma_radial**2))
                    
                    bloom_map += z_bloom * radial_bloom * bloom_intensity
            
            elif axis == 'z':
                # Cylinder extends along z (depth)
                # The cylinder itself spans the z-range, but we add blooming along its length
                # at the top/bottom of the circular cross-section (in x-y plane)
                # Top y: center_pos + radius, Bottom y: center_pos - radius
                y_top = center_pos + radius
                y_bottom = center_pos - radius
                
                for y_surface in [y_top, y_bottom]:
                    dist_from_y = np.abs(y_grid - y_surface)
                    dist_from_axis_x = np.abs(x_grid - other_pos)
                    
                    # Bloom along z (along cylinder length) and radially
                    y_bloom = np.exp(-(dist_from_y**2) / (2 * bloom_sigma_radial**2))
                    radial_bloom = np.exp(-(dist_from_axis_x**2) / (2 * bloom_sigma_radial**2))
                    
                    bloom_map += y_bloom * radial_bloom * bloom_intensity
        
        # Add bloom to volume
        volume_with_bloom = volume + bloom_map
        return np.clip(volume_with_bloom, 0.0, 1.0)
    
    def add_ultrasonic_artifacts(self, volume, 
                                  electronic_noise_level=0.02,
                                  grain_noise_level=0.025,
                                  depth_attenuation=0.3,
                                  bloom_radius=0,
                                  bloom_intensity=0.0,
                                  bloom_falloff='gaussian',
                                  speckle_noise_level=0.04,
                                  blur_sigma=(0.5, 1.0, 2.5),
                                  cylindrical_bloom=True,
                                  cylindrical_bloom_params=None):
        """
        Add realistic ultrasonic imaging artifacts - OPTIMIZED.
        
        APPROACH: Artifacts are created primarily through CTFM processing
        (2D Hilbert envelope detection + dB conversion), not artificial bloom.
        
        This method adds basic physical artifacts:
        - Electronic and grain noise
        - Depth-dependent attenuation
        - Speckle noise (coherent interference)
        - PSF blurring (limited aperture) - ANISOTROPIC to match TFM physics
        
        The main "bloom" or halo effect comes from:
        1. 2D Hilbert envelope (spreads in y-x plane perpendicular to beam)
        2. Anisotropic PSF blur (reinforces the resolution differences)
        
        Together these create elliptical bloom in the cross-section plane (y-x) while
        maintaining good axial resolution along the beam direction (z).
        
        Parameters:
            electronic_noise_level: Amplitude of random electronic noise (0-0.03)
            grain_noise_level: Amplitude of grain scattering noise (0-0.05)
            depth_attenuation: Attenuation coefficient with depth (0-0.6)
            bloom_radius: [DEPRECATED] Set to 0 to disable artificial bloom
            bloom_intensity: [DEPRECATED] Set to 0.0 to disable artificial bloom
            bloom_falloff: [DEPRECATED] Type of falloff
            speckle_noise_level: Coherent speckle noise level (0-0.05)
            blur_sigma: Anisotropic PSF blurring - tuple of (sigma_z, sigma_y, sigma_x)
                        or single float for isotropic blur
                        COORDINATE CONVENTION: (depth, height, lateral) = (z, y, x)
                        sigma_z (depth, axis 0): 0.5-1.0 (good axial resolution)
                        sigma_y (height, axis 1): 0.6-1.2 (moderate elevation resolution)
                        sigma_x (lateral, axis 2): 2.0-5.0 (poor lateral resolution)
                        Note: Lateral blur is largest because array has finite aperture
            cylindrical_bloom: Whether to add blooming at cylinder top/bottom surfaces (default True)
            cylindrical_bloom_params: Dictionary with surface blooming parameters:
                        - top_intensity: Peak intensity at top surface (closer to array, default 0.45)
                        - bottom_intensity: Peak intensity at bottom surface (further from array, default 0.30)
                        - bloom_sigma: Gaussian sigma for bloom spreading in X-Y plane (pixels, default 12.0)
        """
        # Work in float32 for speed
        volume_artifacts = volume.astype(np.float32)
        
        # Add surface blooming if requested
        # These will be enhanced when Hilbert envelope is applied later
        if cylindrical_bloom:
            default_marker_params = {
                'top_intensity': 0.45,
                'bottom_intensity': 0.30,
                'bloom_sigma': 12.0
            }
            if cylindrical_bloom_params:
                default_marker_params.update(cylindrical_bloom_params)
            volume_artifacts = self.add_cylindrical_surface_markers(volume_artifacts, **default_marker_params)
        
        # 1. Combined noise generation (single pass)
        if electronic_noise_level > 0 or grain_noise_level > 0:
            # Electronic noise
            if electronic_noise_level > 0:
                volume_artifacts += np.random.randn(*self.dimensions).astype(np.float32) * electronic_noise_level
            
            # Grain noise (smoothed)
            if grain_noise_level > 0:
                grain_noise = np.random.randn(*self.dimensions).astype(np.float32) * grain_noise_level
                grain_noise = gaussian_filter(grain_noise, sigma=1.5)
                volume_artifacts += grain_noise
        
        # 2. Depth-dependent attenuation (vectorized)
        if depth_attenuation > 0:
            depth_factor = np.exp(-depth_attenuation * np.linspace(0, 1, self.dimensions[0], dtype=np.float32))
            depth_factor = depth_factor[:, None, None]  # Broadcasting shape
            volume_artifacts *= depth_factor
        
        # 3. *** BLOOM EFFECT *** - OPTIMIZED with FFT convolution
        if bloom_radius > 0 and bloom_intensity > 0:
            # Find defect locations (threshold) - but weight by intensity
            defect_mask = (volume > 0.7).astype(np.float32)
            defect_mask *= volume  # Weight by actual defect intensity
            
            # Create smaller bloom kernel (efficiency)
            kernel_size = min(int(bloom_radius * 2 + 1), 31)  # Cap kernel size at 31
            bloom_kernel = self._create_bloom_kernel(kernel_size, bloom_falloff)
            
            # Use FFT convolution for large volumes (much faster)
            if np.prod(self.dimensions) > 1e6:  # For volumes larger than 100^3
                bloom_map = fftconvolve(defect_mask, bloom_kernel, mode='same')
            else:
                from scipy.ndimage import convolve
                bloom_map = convolve(defect_mask, bloom_kernel, mode='constant')
            
            # Scale bloom - ensure it's WEAKER than defect
            bloom_map = bloom_map.astype(np.float32) * bloom_intensity
            
            # Cap bloom at much lower level than defect (bloom should never overpower defect)
            bloom_map = np.minimum(bloom_map, volume_artifacts * 0.4)  # Cap at 40% of local intensity
            
            # Only add bloom where it doesn't already exist as a defect
            bloom_mask = volume < 0.7  # Only add bloom outside defects
            volume_artifacts[bloom_mask] += bloom_map[bloom_mask]
        
        # 4. Speckle noise (multiplicative, fast)
        if speckle_noise_level > 0:
            speckle = np.random.rayleigh(scale=1.0, size=self.dimensions).astype(np.float32)
            speckle = (speckle - 1.0) * speckle_noise_level
            speckle = gaussian_filter(speckle, sigma=0.8)
            volume_artifacts *= (1 + speckle)
        
        # 5. Limited aperture effect (anisotropic PSF blur)
        if blur_sigma is not None:
            # Handle both tuple (anisotropic) and float (isotropic) inputs
            if isinstance(blur_sigma, (tuple, list)):
                sigma_z, sigma_y, sigma_x = blur_sigma
            else:
                sigma_z = sigma_y = sigma_x = blur_sigma
            
            volume_artifacts = gaussian_filter(volume_artifacts, sigma=(sigma_z, sigma_y, sigma_x))
        
        # Ensure values stay in valid range
        volume_artifacts = np.clip(volume_artifacts, 0.0, 1.0)
        return volume_artifacts
    
    def _create_bloom_kernel(self, size, falloff='gaussian'):
        """
        Create a 3D bloom kernel (PSF) - OPTIMIZED with ANISOTROPIC bloom.
        
        The bloom is STRONG in the scanning plane (y-x) but WEAK along depth (z).
        This matches the physics of ultrasonic beam focusing:
        - Good lateral/elevation resolution → strong bloom in (y, x)
        - Poor axial resolution → weak bloom along z
        
        Creates SMOOTH, OVAL-shaped bloom (not rectangular).
        
        Returns float32 for speed.
        """
        center = size // 2
        
        # Vectorized distance calculation with ANISOTROPIC scaling
        idx = np.arange(size, dtype=np.float32)
        z, y, x = np.meshgrid(idx, idx, idx, indexing='ij')
        
        if falloff == 'gaussian':
            # Anisotropic Gaussian: SLIM oval in scanning plane, minimal depth
            sigma_x = size / 3.5        # LONGEST - lateral (horizontal streaks)
            sigma_y = size / 8.0        # MODERATE - height  
            sigma_z = size / 45.0       # SHORTEST - depth
                
            # Ellipsoidal kernel with POINTY tips (higher power decay)
            r2_x = (x - center)**2 / (2 * sigma_x**2)
            r2_y = (y - center)**2 / (2 * sigma_y**2)
            r2_z = (z - center)**2 / (2 * sigma_z**2)
            
            # Use power of 1.5 for pointier tips (sharper than Gaussian)
            r_total = np.sqrt(r2_y + r2_z)
            # Minimal smoothing only to prevent aliasing
            kernel = np.exp(-r2_x - r2_y - r2_z) * np.exp(-0.25 * r_total**1.5)
        
        elif falloff == 'exponential':
            # Anisotropic exponential: SLIM oval in scanning plane
            decay_yx = 4.5 / size       # Scanning plane - moderate decay
            decay_z = 25.0 / size       # Depth - very fast decay (thinner)
            
            r_yx = np.sqrt((x - center)**2 + (y - center)**2)
            r_z = np.abs(z - center)
            
            # Power term for pointy tips
            kernel = np.exp(-r_yx * decay_yx - r_z * decay_z)
            kernel *= np.exp(-0.2 * (r_yx + r_z)**1.5)
            
            # Minimal smoothing
            kernel = gaussian_filter(kernel, sigma=0.2)
        
        else:
            raise ValueError(f"Unknown falloff type: {falloff}")
        
        # Normalize
        kernel = kernel.astype(np.float32)
        kernel /= kernel.max()
        
        return kernel
    
    def generate_subvolume_region(self, origin, shape, base_intensity_range=(0.05, 0.1), 
                                   smoothing_sigma=2.0, use_angular_effects=True):
        """
        Generate a specific subvolume region with independent array position.
        This simulates an independent scan of a portion of the full volume.
        
        Parameters:
            origin: (z, y, x) starting position of the subvolume in the full volume
            shape: (depth, height, width) size of the subvolume
            base_intensity_range: Intensity range for base material
            smoothing_sigma: Gaussian smoothing for base material
            use_angular_effects: If True, position array at subvolume center (default: True)
        
        Returns:
            Generated subvolume with defects and angular effects
        """
        z_origin, y_origin, x_origin = origin
        z_size, y_size, x_size = shape
        
        # Calculate array position at the center of this subvolume
        if use_angular_effects:
            array_y = y_origin + y_size // 2
            array_x = x_origin + x_size // 2
            self.set_array_position(array_y, array_x)
        
        # Temporarily set dimensions to subvolume size
        original_dims = self.dimensions
        self.dimensions = shape
        
        # Create base material
        base_min, base_max = base_intensity_range
        base_material = np.random.rand(*shape).astype(np.float32) * (base_max - base_min) + base_min
        volume = gaussian_filter(base_material, sigma=smoothing_sigma)
        
        # Create coordinate grids for the subvolume (in local coordinates)
        z_local, y_local, x_local = np.ogrid[:z_size, :y_size, :x_size]
        
        # Apply each defect (only if it intersects with this subvolume)
        for defect in self.defects:
            if defect['type'] == 'spherical_void':
                center = defect['center']
                radius = defect['radius']
                base_intensity = defect['intensity']
                
                # Convert global defect center to local subvolume coordinates
                center_local = (
                    center[0] - z_origin,
                    center[1] - y_origin,
                    center[2] - x_origin
                )
                
                # Check if defect is within or near this subvolume
                if (center_local[0] < -radius - 5 or center_local[0] > z_size + radius + 5 or
                    center_local[1] < -radius - 5 or center_local[1] > y_size + radius + 5 or
                    center_local[2] < -radius - 5 or center_local[2] > x_size + radius + 5):
                    continue  # Defect too far from this subvolume
                
                # Apply angular attenuation based on GLOBAL position
                if use_angular_effects:
                    angular_factor = self._calculate_angular_attenuation(center, defect_type='spherical')
                    effective_intensity = base_intensity * angular_factor
                else:
                    effective_intensity = base_intensity
                
                # Calculate distances using local coordinates
                distances = np.sqrt(
                    (z_local - center_local[0])**2 + 
                    (y_local - center_local[1])**2 + 
                    (x_local - center_local[2])**2
                )
                
                mask = distances <= radius
                volume[mask] = effective_intensity
                
                # Soft edge
                edge_mask = (distances > radius) & (distances <= radius + 2)
                volume[edge_mask] = np.maximum(volume[edge_mask], 
                                               effective_intensity * (1 - (distances[edge_mask] - radius) / 2))
            
            elif defect['type'] == 'cylindrical_void':
                # Simple uniform cylinder
                axis = defect.get('axis', 'z')
                center_pos = defect['center_pos']
                other_pos = defect['other_pos']
                radius = defect['radius']
                base_intensity = defect['intensity']
                
                # Determine global center for angular calculation
                if axis == 'x':
                    cylinder_center = (center_pos, other_pos, x_origin + x_size // 2)
                elif axis == 'y':
                    cylinder_center = (center_pos, y_origin + y_size // 2, other_pos)
                elif axis == 'z':
                    cylinder_center = (z_origin + z_size // 2, center_pos, other_pos)
                
                if use_angular_effects:
                    angular_factor = self._calculate_angular_attenuation(cylinder_center, defect_type='cylindrical')
                    effective_intensity = base_intensity * angular_factor
                else:
                    effective_intensity = base_intensity
                
                # Convert to local coordinates
                center_pos_local = center_pos - (z_origin if axis in ['x', 'y'] else (y_origin if axis == 'z' else 0))
                other_pos_local = other_pos - (y_origin if axis == 'x' else (x_origin if axis in ['y', 'z'] else 0))
                
                # Check if cylinder intersects this subvolume
                if axis == 'x':
                    if (center_pos_local < -radius - 5 or center_pos_local > z_size + radius + 5 or
                        other_pos_local < -radius - 5 or other_pos_local > y_size + radius + 5):
                        continue
                    
                    z_grid = np.arange(z_size)[:, None, None]
                    y_grid = np.arange(y_size)[None, :, None]
                    distances = np.sqrt((z_grid - center_pos_local)**2 + (y_grid - other_pos_local)**2)
                    mask = distances <= radius
                    volume[mask.squeeze()] = effective_intensity
                
                elif axis == 'y':
                    if (center_pos_local < -radius - 5 or center_pos_local > z_size + radius + 5 or
                        other_pos_local < -radius - 5 or other_pos_local > x_size + radius + 5):
                        continue
                    
                    z_grid = np.arange(z_size)[:, None, None]
                    x_grid = np.arange(x_size)[None, None, :]
                    distances = np.sqrt((z_grid - center_pos_local)**2 + (x_grid - other_pos_local)**2)
                    mask = distances <= radius
                    volume[:, :, :][np.broadcast_to(mask, shape)] = effective_intensity
                
                elif axis == 'z':
                    if (center_pos_local < -radius - 5 or center_pos_local > y_size + radius + 5 or
                        other_pos_local < -radius - 5 or other_pos_local > x_size + radius + 5):
                        continue
                    
                    y_grid = np.arange(y_size)[None, :, None]
                    x_grid = np.arange(x_size)[None, None, :]
                    distances = np.sqrt((y_grid - center_pos_local)**2 + (x_grid - other_pos_local)**2)
                    mask = distances <= radius
                    volume[:, :, :][np.broadcast_to(mask, shape)] = effective_intensity
        
        # Restore original dimensions
        self.dimensions = original_dims
        
        # Ensure values stay in valid range
        volume = np.clip(volume, 0.0, 1.0)
        
        return volume
    
    def apply_hilbert_envelope(self, volume):
        """
        Apply 2D Hilbert transform along LATERAL and HEIGHT axes - uses scipy's optimized implementation.
        
        COORDINATE SYSTEM:
        - Axis 0 (z, depth): penetration into material - GOOD resolution, NO Hilbert
        - Axis 1 (y, height): vertical/elevation direction - MODERATE resolution, Hilbert applied
        - Axis 2 (x, lateral): along array length - POOR resolution, Hilbert applied
        
        In TFM imaging, the characteristic "bloom" spreads in the plane perpendicular to
        the beam propagation (y-x plane) because lateral and elevation resolutions are
        worse than axial resolution. The 2D Hilbert envelope creates this spreading effect.
        
        The transform is applied sequentially:
        1. Along lateral axis (x, axis 2) - primary bloom direction
        2. Along height axis (y, axis 1) - secondary bloom direction
        
        This creates an elliptical bloom in the y-x plane while maintaining good depth resolution.
        
        Note: We apply mild Gaussian blur in the y-x plane BEFORE Hilbert transform
        to smooth out sharp transitions and create realistic TFM appearance.
        """
        # Pre-smooth in the y-x plane (perpendicular to beam propagation)
        # Stronger smoothing in lateral direction (worse resolution)
        volume_smoothed = gaussian_filter(volume, sigma=(0, 0.8, 1.5))  # (depth=0, height, lateral)
        
        # Apply Hilbert along LATERAL axis first (strongest bloom)
        volume_analytic_x = hilbert(volume_smoothed, axis=2)
        volume_envelope_x = np.abs(volume_analytic_x).astype(np.float32)
        
        # Apply Hilbert along HEIGHT axis second (moderate bloom)
        volume_analytic_xy = hilbert(volume_envelope_x, axis=1)
        volume_envelope = np.abs(volume_analytic_xy).astype(np.float32)
        return volume_envelope
    
    def convert_to_db(self, volume, vmin=-40.0, vmax=0.0):
        """
        Convert to decibel scale - optimized with clipping.
        """
        volume_max = np.max(volume)
        volume_db = 20 * np.log10(volume / volume_max + 1e-10, dtype=np.float32)
        volume_db = np.clip(volume_db, vmin, vmax)
        return volume_db
    
    def save_volume(self, volume, filename="synthetic_volume.npy"):
        np.save(filename, volume)
    
    def visualize(self, volume, name="Volume", contrast_limits=None):
        """Visualize a single volume."""
        viewer = napari.Viewer()
        
        if contrast_limits is None:
            contrast_limits = (volume.min(), volume.max())
        
        viewer.add_image(
            volume,
            name=name,
            colormap="viridis",
            contrast_limits=contrast_limits,
            blending='translucent',
            rendering='mip',
            opacity=0.7
        )
        napari.run()

    def split_volume_for_stitching(self, volume, num_splits=(2, 2, 2), overlap_pixels=(20, 20, 20), save_dir=None):
        """
        Split volume into overlapping sub-volumes - OPTIMIZED with views instead of copies.
        """
        z_dim, y_dim, x_dim = volume.shape
        z_splits, y_splits, x_splits = num_splits
        overlap_z, overlap_y, overlap_x = overlap_pixels
        
        # Calculate sub-volume size
        subvol_z = (z_dim + (z_splits - 1) * overlap_z) // z_splits
        subvol_y = (y_dim + (y_splits - 1) * overlap_y) // y_splits
        subvol_x = (x_dim + (x_splits - 1) * overlap_x) // x_splits
        
        subvol_shape = (subvol_z, subvol_y, subvol_x)
        
        # Calculate step size
        step_z = subvol_z - overlap_z
        step_y = subvol_y - overlap_y
        step_x = subvol_x - overlap_x
        
        # Check padding
        expected_z = (z_splits - 1) * step_z + subvol_z
        expected_y = (y_splits - 1) * step_y + subvol_y
        expected_x = (x_splits - 1) * step_x + subvol_x
        
        if expected_z > z_dim or expected_y > y_dim or expected_x > x_dim:
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
                    
                    # Use .copy() only when saving, otherwise just reference
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
        
        return sub_volumes
    
    def generate_varied_overlaps(self, num_splits, base_overlap=(20, 20, 20), variation_percent=50.0, seed=None):
        """
        Generate varied overlap values for each subvolume boundary to simulate realistic scanning variations.
        
        Parameters:
            num_splits: Tuple of (z_splits, y_splits, x_splits)
            base_overlap: Base overlap in pixels (z, y, x)
            variation_percent: Percent variation to apply (e.g., 50.0 means ±50%)
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary with overlap specifications for each subvolume boundary
        """
        if seed is not None:
            np.random.seed(seed)
        
        z_splits, y_splits, x_splits = num_splits
        base_z, base_y, base_x = base_overlap
        variation_factor = variation_percent / 100.0
        
        # Generate overlaps for boundaries between subvolumes
        # For z-axis: between adjacent z slices (z_splits - 1 boundaries)
        # For y-axis: between adjacent y slices (y_splits - 1 boundaries)
        # For x-axis: between adjacent x slices (x_splits - 1 boundaries)
        
        varied_overlaps = {
            'z_overlaps': [],  # List of overlaps between z slices
            'y_overlaps': [],  # List of overlaps between y slices
            'x_overlaps': []   # List of overlaps between x slices
        }
        
        # Generate random overlaps for each boundary
        for _ in range(max(0, z_splits - 1)):
            overlap = int(base_z * (1 + np.random.uniform(-variation_factor, variation_factor)))
            varied_overlaps['z_overlaps'].append(max(0, overlap))  # Ensure non-negative
        
        for _ in range(max(0, y_splits - 1)):
            overlap = int(base_y * (1 + np.random.uniform(-variation_factor, variation_factor)))
            varied_overlaps['y_overlaps'].append(max(0, overlap))
        
        for _ in range(max(0, x_splits - 1)):
            overlap = int(base_x * (1 + np.random.uniform(-variation_factor, variation_factor)))
            varied_overlaps['x_overlaps'].append(max(0, overlap))
        
        return varied_overlaps
    
    def split_volume_with_varied_overlaps(self, volume, num_splits=(2, 2, 2), 
                                           varied_overlaps=None, save_dir=None):
        """
        Split volume with FIXED subvolume size and VARIED overlaps between subvolumes.
        
        Subvolume size stays constant (representing fixed array size), while the overlap
        (and thus step size) varies between acquisitions. This simulates realistic scanning
        where the array size is fixed but positioning varies.
        
        Parameters:
            volume: Input volume to split
            num_splits: Tuple of (z_splits, y_splits, x_splits)
            varied_overlaps: Dictionary with 'z_overlaps', 'y_overlaps', 'x_overlaps' lists
                            If None, uses uniform overlaps
            save_dir: Optional directory to save subvolumes
        """
        z_dim, y_dim, x_dim = volume.shape
        z_splits, y_splits, x_splits = num_splits
        
        # If no varied overlaps provided, create uniform overlaps
        if varied_overlaps is None:
            varied_overlaps = {
                'z_overlaps': [20] * max(0, z_splits - 1),
                'y_overlaps': [20] * max(0, y_splits - 1),
                'x_overlaps': [20] * max(0, x_splits - 1)
            }
        
        # Calculate FIXED subvolume size for each dimension (same as uniform overlap method)
        # This represents the constant array size (hardware constraint)
        def calculate_subvolume_size(dim_size, n_splits, overlaps_list):
            """Calculate fixed subvolume size ensuring complete coverage."""
            if n_splits == 1:
                return dim_size
            # Calculate size such that n_splits subvolumes with avg overlap cover the dimension
            avg_overlap = np.mean(overlaps_list) if overlaps_list else 0
            # Size = (dimension + total_overlap) / n_splits
            total_overlap = (n_splits - 1) * avg_overlap
            return int(np.ceil((dim_size + total_overlap) / n_splits))
        
        z_subvol_size = calculate_subvolume_size(z_dim, z_splits, varied_overlaps['z_overlaps'])
        y_subvol_size = calculate_subvolume_size(y_dim, y_splits, varied_overlaps['y_overlaps'])
        x_subvol_size = calculate_subvolume_size(x_dim, x_splits, varied_overlaps['x_overlaps'])
        
        # Calculate positions with FIXED size and VARIED step
        def calculate_positions(dim_size, n_splits, subvol_size, overlaps_list):
            """
            Calculate start positions for fixed-size subvolumes with varied overlaps.
            
            CRITICAL: ALL subvolumes must have the exact same size (array hardware constraint).
            The last subvolume is positioned to end exactly at the volume boundary,
            which may require additional overlap beyond what's specified.
            """
            if n_splits == 1:
                return [0]
            
            positions = [0]  # First subvolume starts at 0
            current_pos = 0
            
            for i in range(n_splits - 1):
                # Step size = subvolume_size - overlap
                step = subvol_size - overlaps_list[i]
                current_pos += step
                positions.append(current_pos)
            
            # CRITICAL: Ensure last subvolume has exact same size by adjusting its position
            # Last subvolume must end exactly at dim_size, so start = dim_size - subvol_size
            positions[-1] = dim_size - subvol_size
            
            return positions
        
        z_positions = calculate_positions(z_dim, z_splits, z_subvol_size, varied_overlaps['z_overlaps'])
        y_positions = calculate_positions(y_dim, y_splits, y_subvol_size, varied_overlaps['y_overlaps'])
        x_positions = calculate_positions(x_dim, x_splits, x_subvol_size, varied_overlaps['x_overlaps'])
        
        # Fixed subvolume shape (constant array size - all subvolumes identical)
        subvol_shape = (z_subvol_size, y_subvol_size, x_subvol_size)
        
        sub_volumes = []
        subvol_index = 0
        
        for zi, z_start in enumerate(z_positions):
            for yi, y_start in enumerate(y_positions):
                for xi, x_start in enumerate(x_positions):
                    # Extract subvolume with FIXED size (NO clipping - all sizes identical)
                    z_end = z_start + z_subvol_size
                    y_end = y_start + y_subvol_size
                    x_end = x_start + x_subvol_size
                    
                    sub_vol = volume[
                        z_start:z_end,
                        y_start:y_end,
                        x_start:x_end
                    ].copy()
                    
                    # Calculate ACTUAL overlap (may differ from specified for last subvolume)
                    if zi < z_splits - 1 and zi < len(varied_overlaps['z_overlaps']):
                        actual_overlap_z = varied_overlaps['z_overlaps'][zi]
                    else:
                        actual_overlap_z = 0
                        
                    if yi < y_splits - 1 and yi < len(varied_overlaps['y_overlaps']):
                        actual_overlap_y = varied_overlaps['y_overlaps'][yi]
                    else:
                        actual_overlap_y = 0
                        
                    if xi < x_splits - 1 and xi < len(varied_overlaps['x_overlaps']):
                        actual_overlap_x = varied_overlaps['x_overlaps'][xi]
                    else:
                        actual_overlap_x = 0
                    
                    metadata = {
                        'index': (zi, yi, xi),
                        'volume': sub_vol,
                        'subvol_shape': subvol_shape,  # Store constant shape
                        'origin': (z_start, y_start, x_start),
                        'overlap': (actual_overlap_z, actual_overlap_y, actual_overlap_x),
                        'subvolume_id': subvol_index
                    }
                    sub_volumes.append(metadata)
                    subvol_index += 1
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            for sv in sub_volumes:
                idx = sv['index']
                filename = os.path.join(save_dir, f"subvol_{idx[0]}_{idx[1]}_{idx[2]}.npy")
                np.save(filename, sv['volume'])
                
                meta = {k: v for k, v in sv.items() if k != 'volume'}
                meta_filename = os.path.join(save_dir, f"subvol_{idx[0]}_{idx[1]}_{idx[2]}_meta.npy")
                np.savez(meta_filename, **meta)
            
            # Save reconstruction info with varied overlaps
            recon_info = {
                'original_shape': (z_dim, y_dim, x_dim),
                'num_splits': num_splits,
                'subvol_shape': subvol_shape,  # Fixed subvolume shape
                'varied_overlaps': varied_overlaps,
                'variable_overlap_mode': True,
                'num_subvolumes': len(sub_volumes)
            }
            np.savez(os.path.join(save_dir, 'reconstruction_info.npz'), **recon_info)
        
        return sub_volumes
    
    def generate_varied_artifact_params(self, num_subvolumes, base_params=None, variation_percent=10.0, seed=None):
        """
        Generate varied artifact parameters for each subvolume to simulate realistic scan inconsistencies.
        
        Parameters:
            num_subvolumes: Number of subvolumes to generate parameters for
            base_params: Base artifact parameters dict (uses defaults if None)
            variation_percent: Percent variation to apply (e.g., 10.0 means ±10%)
            seed: Random seed for reproducibility
            
        Returns:
            List of parameter dicts, one for each subvolume
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Default base parameters
        default_base = {
            'electronic_noise_level': 0.02,
            'grain_noise_level': 0.025,
            'depth_attenuation': 0.3,
            'bloom_radius': 0,
            'bloom_intensity': 0.0,
            'bloom_falloff': 'gaussian',
            'speckle_noise_level': 0.04,
            'blur_sigma': (0.8, 1.0, 3.0),
            'cylindrical_bloom': True,
            'cylindrical_bloom_params': {
                'top_intensity': 0.45,
                'bottom_intensity': 0.30,
                'bloom_sigma': 12.0
            }
        }
        
        if base_params is not None:
            default_base.update(base_params)
        
        varied_params_list = []
        variation_factor = variation_percent / 100.0
        
        for i in range(num_subvolumes):
            params = {}
            
            for key, value in default_base.items():
                if key in ['bloom_falloff']:  # String parameters - no variation
                    params[key] = value
                elif key == 'blur_sigma':  # Tuple parameters - vary each component
                    if isinstance(value, (tuple, list)):
                        varied = tuple(v * (1 + np.random.uniform(-variation_factor, variation_factor)) 
                                      for v in value)
                        params[key] = varied
                    else:
                        params[key] = value * (1 + np.random.uniform(-variation_factor, variation_factor))
                elif key == 'cylindrical_bloom_params' and isinstance(value, dict):
                    # Vary nested cylindrical bloom parameters
                    varied_bloom = {}
                    for bloom_key, bloom_val in value.items():
                        if isinstance(bloom_val, (int, float)):
                            varied_bloom[bloom_key] = bloom_val * (1 + np.random.uniform(-variation_factor, variation_factor))
                        else:
                            varied_bloom[bloom_key] = bloom_val
                    params[key] = varied_bloom
                elif isinstance(value, (int, float)):  # Numeric parameters - add variation
                    # Ensure values stay non-negative
                    varied_value = value * (1 + np.random.uniform(-variation_factor, variation_factor))
                    params[key] = max(0.0, varied_value)
                else:
                    params[key] = value
            
            varied_params_list.append(params)
        
        return varied_params_list
    
    def generate_stitching_test_data(self, volume_clean, 
                                      num_splits=(1, 1, 3),
                                      overlap_pixels=(0, 0, 30),
                                      varied_overlaps=None,
                                      auto_vary_overlaps=False,
                                      overlap_variation_percent=50.0,
                                      artifact_mode='per_subvolume',
                                      artifact_params=None,
                                      per_subvolume_params=None,
                                      auto_vary_params=False,
                                      variation_percent=10.0,
                                      save_dir=None,
                                      visualize_subvolumes=False,
                                      apply_ctfm=True,
                                      use_angular_effects=True):
        """
        Generate test data for stitching with per-subvolume artifact and overlap variation - OPTIMIZED.
        
        Parameters:
            volume_clean: Clean volume to process (ground truth without angular effects)
            num_splits: Tuple of (z, y, x) splits
            overlap_pixels: Tuple of base overlap in each dimension (used if varied_overlaps=None)
            varied_overlaps: Dictionary with 'z_overlaps', 'y_overlaps', 'x_overlaps' lists
                            for manual per-boundary overlap specification. If provided, creates
                            variable-sized subvolumes ensuring complete coverage.
            auto_vary_overlaps: If True, automatically generate varied overlaps for each boundary
            overlap_variation_percent: Percent variation for overlaps (e.g., 50.0 means ±50%)
            artifact_mode: 'whole_volume' or 'per_subvolume'
            artifact_params: Dict of artifact parameters (used as base if auto_vary_params=True,
                            or applied to all subvolumes if per_subvolume_params=None)
            per_subvolume_params: List of parameter dicts (one per subvolume) for manual control.
                                 Length must match total number of subvolumes (z_splits * y_splits * x_splits).
                                 If provided, overrides artifact_params and auto_vary_params.
            auto_vary_params: If True, automatically generate varied parameters for each subvolume
                             to simulate realistic scan inconsistencies
            variation_percent: Percent variation when auto_vary_params=True (e.g., 10.0 means ±10%)
            save_dir: Directory to save subvolumes
            visualize_subvolumes: Whether to visualize in napari
            apply_ctfm: Whether to apply CTFM processing (Hilbert + dB) to subvolumes
            use_angular_effects: If True, regenerate each subvolume with angular effects
                                based on array position (simulates independent scans)
                                Only applies in 'per_subvolume' mode.
        """
        # Determine overlap mode and generate/validate overlaps
        use_variable_overlaps = (varied_overlaps is not None) or auto_vary_overlaps
        
        if auto_vary_overlaps and varied_overlaps is None:
            # Auto-generate varied overlaps
            varied_overlaps = self.generate_varied_overlaps(
                num_splits,
                base_overlap=overlap_pixels,
                variation_percent=overlap_variation_percent
            )
            print(f"Auto-generated varied overlaps (±{overlap_variation_percent}% variation)")
            print(f"  Z overlaps: {varied_overlaps['z_overlaps']}")
            print(f"  Y overlaps: {varied_overlaps['y_overlaps']}")
            print(f"  X overlaps: {varied_overlaps['x_overlaps']}")
        elif varied_overlaps is not None:
            # Validate manually provided overlaps
            expected_z = max(0, num_splits[0] - 1)
            expected_y = max(0, num_splits[1] - 1)
            expected_x = max(0, num_splits[2] - 1)
            
            if len(varied_overlaps.get('z_overlaps', [])) != expected_z:
                raise ValueError(f"z_overlaps must have {expected_z} entries, got {len(varied_overlaps.get('z_overlaps', []))}")
            if len(varied_overlaps.get('y_overlaps', [])) != expected_y:
                raise ValueError(f"y_overlaps must have {expected_y} entries, got {len(varied_overlaps.get('y_overlaps', []))}")
            if len(varied_overlaps.get('x_overlaps', [])) != expected_x:
                raise ValueError(f"x_overlaps must have {expected_x} entries, got {len(varied_overlaps.get('x_overlaps', []))}")
            print(f"Using manually specified varied overlaps")
        
        # Calculate total number of subvolumes
        total_subvolumes = num_splits[0] * num_splits[1] * num_splits[2]
        # Determine artifact parameters for each subvolume
        if per_subvolume_params is not None:
            # Manual per-subvolume parameters provided
            if len(per_subvolume_params) != total_subvolumes:
                raise ValueError(f"per_subvolume_params must have {total_subvolumes} entries (one per subvolume), got {len(per_subvolume_params)}")
            subvolume_params_list = per_subvolume_params
            print(f"Using manually specified parameters for {total_subvolumes} subvolumes")
        elif auto_vary_params:
            # Auto-generate varied parameters
            base_params = artifact_params if artifact_params is not None else None
            subvolume_params_list = self.generate_varied_artifact_params(
                total_subvolumes, 
                base_params=base_params,
                variation_percent=variation_percent
            )
            print(f"Auto-generated varied parameters for {total_subvolumes} subvolumes (±{variation_percent}% variation)")
        else:
            # Use same parameters for all subvolumes (original behavior)
            default_params = {
                'electronic_noise_level': 0.02,
                'grain_noise_level': 0.025,
                'depth_attenuation': 0.3,
                'bloom_radius': 0,
                'bloom_intensity': 0.0,
                'bloom_falloff': 'gaussian',
                'speckle_noise_level': 0.04,
                'blur_sigma': (0.8, 1.0, 3.0),
                'cylindrical_bloom': True,
                'cylindrical_bloom_params': None
            }
            if artifact_params is not None:
                default_params.update(artifact_params)
            subvolume_params_list = [default_params.copy() for _ in range(total_subvolumes)]
            print(f"Using identical parameters for all {total_subvolumes} subvolumes")
        
        if artifact_mode == 'whole_volume':
            # For whole_volume mode, use first parameter set (or averaged parameters)
            volume_with_artifacts = self.add_ultrasonic_artifacts(volume_clean, **subvolume_params_list[0])
            
            # Apply CTFM processing if requested
            if apply_ctfm:
                volume_with_artifacts = self.apply_hilbert_envelope(volume_with_artifacts)
                volume_with_artifacts = self.convert_to_db(volume_with_artifacts, vmin=-40.0, vmax=0.0)
            
            # Choose splitting method based on overlap mode
            if use_variable_overlaps:
                sub_volumes = self.split_volume_with_varied_overlaps(
                    volume_with_artifacts,
                    num_splits=num_splits,
                    varied_overlaps=varied_overlaps,
                    save_dir=save_dir
                )
            else:
                sub_volumes = self.split_volume_for_stitching(
                    volume_with_artifacts,
                    num_splits=num_splits,
                    overlap_pixels=overlap_pixels,
                    save_dir=save_dir
                )
            
        elif artifact_mode == 'per_subvolume':
            # Choose splitting method based on overlap mode
            if use_variable_overlaps:
                sub_volumes_clean = self.split_volume_with_varied_overlaps(
                    volume_clean,
                    num_splits=num_splits,
                    varied_overlaps=varied_overlaps,
                    save_dir=None
                )
            else:
                sub_volumes_clean = self.split_volume_for_stitching(
                    volume_clean,
                    num_splits=num_splits,
                    overlap_pixels=overlap_pixels,
                    save_dir=None
                )
            
            sub_volumes = []
            
            for i, sv in enumerate(sub_volumes_clean):
                idx = sv['index']
                origin = sv['origin']
                shape = sv['volume'].shape
                
                # Get parameters for this specific subvolume
                current_params = subvolume_params_list[i]
                
                if use_angular_effects:
                    # Regenerate this subvolume with angular effects
                    # Array positioned at the center of the subvolume
                    vol_regenerated = self.generate_subvolume_region(
                        origin=origin,
                        shape=shape,
                        use_angular_effects=True
                    )
                else:
                    # Use the clean split volume (no regeneration)
                    vol_regenerated = sv['volume']
                
                # Temporarily update dimensions for artifact processing
                original_dims = self.dimensions
                self.dimensions = shape
                
                # Add artifacts with subvolume-specific parameters
                vol_with_artifacts = self.add_ultrasonic_artifacts(
                    vol_regenerated, 
                    **current_params
                )
                
                # Apply CTFM processing if requested
                if apply_ctfm:
                    vol_with_artifacts = self.apply_hilbert_envelope(vol_with_artifacts)
                    vol_with_artifacts = self.convert_to_db(vol_with_artifacts, vmin=-40.0, vmax=0.0)
                
                # Restore dimensions
                self.dimensions = original_dims
                
                # Create new metadata
                sv_with_artifacts = sv.copy()
                sv_with_artifacts['volume'] = vol_with_artifacts
                sv_with_artifacts['artifact_params'] = current_params  # Store params used for this subvolume
                if use_angular_effects:
                    sv_with_artifacts['array_position'] = (
                        origin[1] + shape[1] // 2,
                        origin[2] + shape[2] // 2
                    )
                sub_volumes.append(sv_with_artifacts)
            
            # Save if directory provided
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                
                for sv in sub_volumes:
                    idx = sv['index']
                    filename = os.path.join(save_dir, f"subvol_{idx[0]}_{idx[1]}_{idx[2]}.npy")
                    np.save(filename, sv['volume'])
                    
                    meta = {k: v for k, v in sv.items() if k != 'volume'}
                    meta_filename = os.path.join(save_dir, f"subvol_{idx[0]}_{idx[1]}_{idx[2]}_meta.npy")
                    np.savez(meta_filename, **meta)
                
                # Save reconstruction info with per-subvolume parameters
                recon_info = {
                    'original_shape': volume_clean.shape,
                    'num_splits': num_splits,
                    'overlap_pixels': overlap_pixels if not use_variable_overlaps else None,
                    'varied_overlaps': varied_overlaps if use_variable_overlaps else None,
                    'variable_overlap_mode': use_variable_overlaps,
                    'auto_vary_overlaps': auto_vary_overlaps,
                    'overlap_variation_percent': overlap_variation_percent if auto_vary_overlaps else 0.0,
                    'num_subvolumes': len(sub_volumes),
                    'artifact_mode': artifact_mode,
                    'per_subvolume_params': per_subvolume_params is not None,
                    'auto_vary_params': auto_vary_params,
                    'variation_percent': variation_percent if auto_vary_params else 0.0,
                    'ctfm_applied': apply_ctfm,
                    'angular_effects': use_angular_effects
                }
                # Note: subvol_shape and step not included in variable overlap mode
                if not use_variable_overlaps and len(sub_volumes) > 0:
                    recon_info['subvol_shape'] = sub_volumes[0].get('subvol_shape')
                    recon_info['step'] = sub_volumes[0].get('step')
                np.savez(os.path.join(save_dir, 'reconstruction_info.npz'), **recon_info)
                
                # Save individual subvolume parameters (as pickled object)
                params_file = os.path.join(save_dir, 'subvolume_artifact_params.npy')
                np.save(params_file, np.array(subvolume_params_list, dtype=object), allow_pickle=True)
        
        else:
            raise ValueError(f"Unknown artifact_mode: {artifact_mode}")
        
        # Visualize if requested
        if visualize_subvolumes:
            self.visualize_sub_volumes(sub_volumes)
        
        return sub_volumes
    
    @staticmethod
    def reconstruct_volume(sub_volumes, original_shape=None):
        """
        Reconstruct original volume using weighted blending - OPTIMIZED.
        Handles both uniform and variable-sized subvolumes.
        """
        if original_shape is None:
            # Calculate original shape from subvolume positions and sizes
            # Works for both uniform and variable-sized subvolumes
            max_z = max(sv['origin'][0] + sv['volume'].shape[0] for sv in sub_volumes)
            max_y = max(sv['origin'][1] + sv['volume'].shape[1] for sv in sub_volumes)
            max_x = max(sv['origin'][2] + sv['volume'].shape[2] for sv in sub_volumes)
            original_shape = (max_z, max_y, max_x)
        
        # Use float32 for speed
        reconstructed = np.zeros(original_shape, dtype=np.float32)
        weights = np.zeros(original_shape, dtype=np.float32)
        
        for sv in sub_volumes:
            origin = sv['origin']
            vol = sv['volume'].astype(np.float32)
            
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
        return reconstructed
    
    @staticmethod
    def load_sub_volumes(load_dir):
        recon_info = np.load(os.path.join(load_dir, 'reconstruction_info.npz'), allow_pickle=True)
        recon_info = {k: recon_info[k].item() if recon_info[k].ndim == 0 else recon_info[k] 
                      for k in recon_info.files}
        
        sub_volumes = []
        for filename in sorted(os.listdir(load_dir)):
            if filename.startswith('subvol_') and filename.endswith('.npy') and '_meta' not in filename:
                vol_path = os.path.join(load_dir, filename)
                volume = np.load(vol_path)
                meta_path = vol_path.replace('.npy', '_meta.npz')
                metadata_raw = np.load(meta_path, allow_pickle=True)
                metadata = {k: metadata_raw[k].item() if metadata_raw[k].ndim == 0 else metadata_raw[k] 
                           for k in metadata_raw.files}
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
        """Visualize all sub-volumes in napari with spatial positioning."""
        print(f"\nVisualizing {len(sub_volumes)} subvolumes in napari...")
        
        viewer = napari.Viewer()
        colormaps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'turbo']
        
        # Detect if CTFM was applied (negative values indicate dB scale)
        sample_vol = sub_volumes[0]['volume']
        is_db_scale = sample_vol.min() < 0
        contrast_limits = (-40, 0) if is_db_scale else (0.0, 1.0)
        
        print(f"Data scale: {'dB (CTFM applied)' if is_db_scale else 'Linear'}")
        print(f"Contrast limits: {contrast_limits}")
        
        for i, sv in enumerate(sub_volumes):
            idx = sv['index']
            origin = sv['origin']
            vol_shape = sv['volume'].shape
            
            print(f"  Adding SubVol {idx}: shape={vol_shape}, origin={origin}")
            
            # Add image layer with spatial positioning
            viewer.add_image(
                sv['volume'],
                name=f"SubVol_{idx[0]}_{idx[1]}_{idx[2]}",
                colormap=colormaps[i % len(colormaps)],
                contrast_limits=contrast_limits,
                blending='additive',
                opacity=0.6,
                translate=origin  # Spatial offset
            )
        
        print("Opening napari viewer...")
        print("Close the napari window to continue execution.")
        napari.run()


# ============================================================================
# Example Usage
# ============================================================================


# Generate clean volume
generator = SyntheticVolumeGenerator(dimensions=(200, 150, 600), seed=42)

# Add defects
generator.add_spherical_void(center=(40, 75, 150), radius=10, intensity=0.95)
generator.add_cylindrical_void(center_pos=100, other_pos=300, radius=6, intensity=0.92, axis='y')
generator.add_spherical_void(center=(120, 60, 450), radius=8, intensity=0.93)
generator.add_cylindrical_void(center_pos=80, other_pos=100, radius=5, intensity=0.90, axis='x')

volume_clean = generator.generate(base_intensity_range=(0.05, 0.15), smoothing_sigma=2.0)
generator.save_volume(volume_clean, "synthetic_volume_clean.npy")

# Add artifacts
volume_with_artifacts = generator.add_ultrasonic_artifacts(
    volume_clean,
    electronic_noise_level=0.02,
    grain_noise_level=0.025,
    depth_attenuation=0.3,
    bloom_radius=0,
    bloom_intensity=0.0,
    speckle_noise_level=0.04,
    blur_sigma=(0.3, 4.5, 4.5)
)
generator.save_volume(volume_with_artifacts, "synthetic_volume_with_artifacts.npy")

# Apply CTFM processing
volume_envelope = generator.apply_hilbert_envelope(volume_with_artifacts)
volume_db = generator.convert_to_db(volume_envelope, vmin=-40.0, vmax=0.0)
generator.save_volume(volume_db, "synthetic_volume_db.npy")

# Visualize
generator.visualize(volume_clean, name="Clean Volume")
generator.visualize(volume_with_artifacts, name="With Artifacts")
generator.visualize(volume_db, name="CTFM Processed (dB)", contrast_limits=(-40, 0))

# Generate stitching test data with AUTOMATIC parameter variation (±15% variation)
# This simulates realistic inconsistencies between scans
# sub_volumes_realistic = generator.generate_stitching_test_data(
#     volume_clean,
#     num_splits=(1, 1, 3),
#     overlap_pixels=(0, 0, 40),
#     artifact_mode='per_subvolume',
#     artifact_params={
#         'electronic_noise_level': 0.02,
#         'grain_noise_level': 0.025,
#         'depth_attenuation': 0.3,
#         'bloom_radius': 0,
#         'bloom_intensity': 0.0,
#         'speckle_noise_level': 0.04,
#         'blur_sigma': (0.8, 1.0, 3.0)
#     },
#     auto_vary_params=True,          # Enable automatic variation
#     variation_percent=15.0,         # ±15% variation in parameters
#     save_dir="SYNTHETIC NPY/stitching_realistic",
#     visualize_subvolumes=True,
#     apply_ctfm=True,
#     use_angular_effects=True
# )

# ============================================================================
# EXAMPLE 1: Manual per-subvolume parameters with VARIABLE OVERLAPS
# ============================================================================
# Variable overlaps allow you to specify different overlap amounts between each
# pair of adjacent subvolumes. This simulates realistic scanning where the overlap
# between successive scans may vary due to operator inconsistency or equipment drift.
# 
# KEY BEHAVIOR: 
# - Subvolume size (array size) is FIXED for all subvolumes
# - Overlap between subvolumes VARIES as specified
# - Step size varies accordingly: step = array_size - overlap
# - The entire sample is guaranteed to be 100% scanned (lossless reconstruction)

manual_params = [
    # Subvolume 0 - high noise
    {'electronic_noise_level': 0.03, 'grain_noise_level': 0.03, 'speckle_noise_level': 0.05,
     'depth_attenuation': 0.35, 'blur_sigma': (0.9, 1.1, 3.2)},
    # Subvolume 1 - medium quality
    {'electronic_noise_level': 0.02, 'grain_noise_level': 0.025, 'speckle_noise_level': 0.04,
     'depth_attenuation': 0.3, 'blur_sigma': (0.8, 1.0, 3.0)},
    # Subvolume 2 - low noise
    {'electronic_noise_level': 0.015, 'grain_noise_level': 0.02, 'speckle_noise_level': 0.035,
     'depth_attenuation': 0.25, 'blur_sigma': (0.7, 0.9, 2.8)}
]

# Manual specification of overlaps between subvolumes
# For 3 subvolumes along x, we have 2 boundaries (overlaps)
manual_overlaps = {
    'z_overlaps': [],      # No z splits (1, 1, 3)
    'y_overlaps': [],      # No y splits
    'x_overlaps': [30, 50] # First overlap: 30 pixels, Second overlap: 50 pixels
}

sub_volumes_manual = generator.generate_stitching_test_data(
    volume_clean,
    num_splits=(1, 1, 3),
    varied_overlaps=manual_overlaps,    # Use manual overlaps (creates variable-sized subvolumes)
    artifact_mode='per_subvolume',
    per_subvolume_params=manual_params,  # Use manual artifact parameters
    save_dir="SYNTHETIC NPY/stitching_manual",
    visualize_subvolumes=True,           # Visualize in napari (blocks execution until window closed)
    apply_ctfm=True,
    use_angular_effects=True
)

# ============================================================================
# EXAMPLE 2: Automatic parameter AND overlap variation (most realistic)
# ============================================================================
print("\n" + "="*80)
print("Testing automatic overlap and parameter variation...")
print("="*80)
sub_volumes_auto_everything = generator.generate_stitching_test_data(
    volume_clean,
    num_splits=(1, 1, 3),
    overlap_pixels=(0, 0, 40),          # Base overlap
    auto_vary_overlaps=True,            # Vary overlaps automatically
    overlap_variation_percent=50.0,     # ±50% variation in overlaps
    artifact_mode='per_subvolume',
    artifact_params={                   # Base artifact parameters
        'electronic_noise_level': 0.02,
        'grain_noise_level': 0.025,
        'depth_attenuation': 0.3,
        'speckle_noise_level': 0.04,
        'blur_sigma': (0.8, 1.0, 3.0)
    },
    auto_vary_params=True,              # Vary artifact params automatically
    variation_percent=15.0,             # ±15% variation in artifact parameters
    save_dir="SYNTHETIC NPY/stitching_auto_realistic",
    visualize_subvolumes=False,         # Don't visualize to speed up testing
    apply_ctfm=True,
    use_angular_effects=True
)

# Test reconstruction accuracy with clean splits (before artifacts)
# This validates that variable overlaps preserve data correctly
print("\n" + "="*80)
print("Testing reconstruction accuracy (auto-generated variable overlaps)...")
print("="*80)
# Get the auto-generated overlaps from the saved reconstruction info
recon_info_path = "SYNTHETIC NPY/stitching_auto_realistic/reconstruction_info.npz"
if os.path.exists(recon_info_path):
    recon_info = np.load(recon_info_path, allow_pickle=True)
    auto_overlaps = recon_info['varied_overlaps'].item() if 'varied_overlaps' in recon_info else None
    if auto_overlaps:
        sub_volumes_clean_auto = generator.split_volume_with_varied_overlaps(
            volume_clean,
            num_splits=(1, 1, 3),
            varied_overlaps=auto_overlaps,
            save_dir=None
        )
        reconstructed_clean_auto = SyntheticVolumeGenerator.reconstruct_volume(
            sub_volumes_clean_auto,
            volume_clean.shape
        )
        generator.verify_reconstruction(volume_clean, reconstructed_clean_auto)


# ============================================================================
# EXAMPLE 3: Uniform overlaps (original behavior - commented out)
# ============================================================================
"""
sub_volumes_uniform = generator.generate_stitching_test_data(
    volume_clean,
    num_splits=(1, 1, 3),
    overlap_pixels=(0, 0, 40),          # Uniform overlap for all subvolumes
    artifact_mode='per_subvolume',
    artifact_params={
        'electronic_noise_level': 0.02,
        'grain_noise_level': 0.025,
        'depth_attenuation': 0.3,
        'blur_sigma': (0.8, 1.0, 3.0)
    },
    save_dir="SYNTHETIC NPY/stitching_uniform",
    visualize_subvolumes=True,
    apply_ctfm=True,
    use_angular_effects=True
)
"""


# Test reconstruction accuracy with clean splits (before artifacts)
# This validates that variable overlaps preserve data correctly
print("\n" + "="*80)
print("Testing reconstruction accuracy (manual variable overlaps)...")
print("="*80)
sub_volumes_clean_test = generator.split_volume_with_varied_overlaps(
    volume_clean,
    num_splits=(1, 1, 3),
    varied_overlaps=manual_overlaps,
    save_dir=None
)
reconstructed_clean = SyntheticVolumeGenerator.reconstruct_volume(
    sub_volumes_clean_test, 
    volume_clean.shape
)
generator.verify_reconstruction(volume_clean, reconstructed_clean)


# Note: Reconstruction from artifacted subvolumes will NOT match the original clean volume
# (that's expected - artifacts are meant to create differences)

print("\n" + "="*80)
print("ALL DONE!")
print("="*80)

# ============================================================================
# OPTIONAL: Visualize subvolumes in napari
# ============================================================================
# Uncomment the following lines to visualize the generated subvolumes in napari.
# Note: This will block execution until you close the napari window.
"""
print("\n" + "="*80)
print("Visualizing subvolumes (close napari window to exit)...")
print("="*80)
generator.visualize_sub_volumes(sub_volumes_manual)
"""
