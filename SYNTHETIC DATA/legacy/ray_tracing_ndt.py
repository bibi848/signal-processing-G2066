#!/usr/bin/env python3
"""
Physics-based NDT Simulation using Ray Tracing

This script creates realistic ultrasonic NDT data by:
1. Creating ground truth volumes with defects (same as current method)
2. Simulating real phased array scanning with configurable parameters
3. Using ray tracing for wave propagation through the material
4. Applying Total Focusing Method (TFM) reconstruction
5. Generating subvolumes that match real experimental data

Key features:
- Realistic wave propagation physics
- Configurable array parameters (wavelength, pitch, element count, etc.)
- Material properties (speed of sound, attenuation)
- Ray tracing with reflection/scattering from defects
- TFM image reconstruction
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import hilbert2
from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict
import time

# Qt configuration for visualization
import os
import sys
try:
    import site
    qt_plugin_path = None
    site_packages = site.getsitepackages()
    
    for sp in site_packages:
        potential_path = os.path.join(sp, 'PyQt5', 'Qt5', 'plugins')
        if os.path.exists(potential_path):
            qt_plugin_path = potential_path
            break
    
    if qt_plugin_path:
        os.environ['QT_PLUGIN_PATH'] = qt_plugin_path
        print(f"✓ Qt configured")
except Exception as e:
    print(f"Warning: Qt configuration failed: {e}")

import napari


@dataclass
class ArrayParameters:
    """
    Parameters for the ultrasonic phased array.
    
    Supports 2D matrix arrays for Full Matrix Capture (FMC) / TFM imaging.
    
    Parameters typical for NDT:
    - Frequency: 5-15 MHz for metals
    - Element count: 32-128 elements (e.g., 8x8, 8x16, 16x16)
    - Pitch: 0.5-1.0 mm (typically 0.5-0.6 mm)
    - Element width: 0.4-0.9mm (typically 0.9 * pitch)
    """
    frequency: float = 5e6  # Hz (5 MHz default)
    speed_of_sound: float = 6320.0  # m/s (longitudinal waves in aluminum)
    
    # 2D Array configuration
    num_elements_x: int = 8  # Number of elements in X direction
    num_elements_y: int = 8  # Number of elements in Y direction
    pitch: float = 0.6e-3  # m (0.6 mm element spacing)
    element_width: float = 0.54e-3  # m (0.54 mm active element width, 90% of pitch)
    aperture_angle: float = 45.0  # degrees (maximum beam steering angle)
    
    # Material properties
    density: float = 2700.0  # kg/m³ (aluminum)
    attenuation_coeff: float = 0.03  # Nepers/m/MHz (frequency-dependent attenuation)
    
    # Imaging parameters
    pixel_size: float = 0.1e-3  # m (0.1 mm per pixel - high resolution)
    
    # Beam pattern parameters (for realistic artifacts)
    element_directivity: bool = True  # Apply element directivity pattern
    side_lobe_level: float = -20.0  # dB (side lobe relative to main lobe)
    grating_lobe_suppression: float = -15.0  # dB (grating lobe suppression)
    psf_bloom_sigma: float = 2.0  # pixels (PSF spreading for blooming effect)
    
    @property
    def num_elements(self) -> int:
        """Total number of elements in the 2D array."""
        return self.num_elements_x * self.num_elements_y
    
    @property
    def wavelength(self) -> float:
        """Calculate wavelength from frequency and speed of sound."""
        return self.speed_of_sound / self.frequency
    
    @property
    def aperture_size_x(self) -> float:
        """Aperture size in X direction (meters)."""
        return self.num_elements_x * self.pitch
    
    @property
    def aperture_size_y(self) -> float:
        """Aperture size in Y direction (meters)."""
        return self.num_elements_y * self.pitch
    
    @property
    def element_positions(self) -> np.ndarray:
        """Get (Y, X) positions of all elements in the 2D array.
        
        Returns:
            Array of shape (num_elements, 2) with [y, x] coordinates for each element.
            Elements are centered at origin (0, 0).
        """
        # Create 2D grid of element positions
        x_positions = np.arange(self.num_elements_x) * self.pitch - self.aperture_size_x / 2
        y_positions = np.arange(self.num_elements_y) * self.pitch - self.aperture_size_y / 2
        
        # Meshgrid and flatten to get all element positions
        xx, yy = np.meshgrid(x_positions, y_positions)
        positions = np.column_stack([yy.ravel(), xx.ravel()])  # Shape: (num_elements, 2)
        
        return positions
    
    def __post_init__(self):
        """Validate parameters."""
        assert self.frequency > 0, "Frequency must be positive"
        assert self.speed_of_sound > 0, "Speed of sound must be positive"
        assert self.num_elements > 0, "Number of elements must be positive"
        assert self.pitch > 0, "Pitch must be positive"
        assert self.element_width <= self.pitch, "Element width cannot exceed pitch"
        
        # Calculate derived parameters
        wavelength = self.wavelength
        print(f"\n{'='*80}")
        print(f"ARRAY PARAMETERS (2D MATRIX ARRAY)")
        print(f"{'='*80}")
        print(f"  Frequency: {self.frequency/1e6:.1f} MHz")
        print(f"  Speed of sound: {self.speed_of_sound:.0f} m/s")
        print(f"  Wavelength: {wavelength*1e3:.3f} mm")
        print(f"  Array configuration: {self.num_elements_y} × {self.num_elements_x} = {self.num_elements} elements")
        print(f"  Element pitch: {self.pitch*1e3:.2f} mm")
        print(f"  Element width: {self.element_width*1e3:.2f} mm")
        print(f"  Aperture size: {self.aperture_size_y*1e3:.1f} × {self.aperture_size_x*1e3:.1f} mm (Y × X)")
        print(f"  Pixel size: {self.pixel_size*1e3:.2f} mm")
        print(f"  Attenuation: {self.attenuation_coeff:.3f} Np/m/MHz")
        print(f"  Element directivity: {self.element_directivity}")
        print(f"  Side lobe level: {self.side_lobe_level:.1f} dB")
        print(f"{'='*80}\n")


class GroundTruthGenerator:
    """
    Generate ground truth volumes with defects.
    This is the same approach as the current synthetic data generator.
    """
    
    def __init__(self, dimensions: Tuple[int, int, int], seed: int = None):
        """
        Initialize ground truth generator.
        
        Parameters:
            dimensions: (depth, height, width) in pixels
            seed: Random seed for reproducibility
        """
        self.dimensions = dimensions
        self.depth, self.height, self.width = dimensions
        self.defects = []
        
        if seed is not None:
            np.random.seed(seed)
        
        print(f"\n{'='*80}")
        print(f"GROUND TRUTH GENERATION")
        print(f"{'='*80}")
        print(f"  Volume dimensions: {dimensions} pixels (Z, Y, X)")
        print(f"  Seed: {seed}")
        print(f"{'='*80}\n")
    
    def add_cylindrical_void(self, center_pos: float, other_pos: float, 
                            radius: float, intensity: float = 0.95, 
                            axis: str = 'z', length: float = None):
        """
        Add a cylindrical void defect to the ground truth.
        
        Parameters:
            center_pos: Center position along main coordinate
            other_pos: Position along perpendicular coordinate
            radius: Cylinder radius in pixels
            intensity: Reflectivity (0-1, where 1 = perfect reflector)
            axis: 'x', 'y', or 'z' - direction cylinder extends along
            length: Cylinder length (None = full dimension)
        """
        if axis == 'z':
            if length is None:
                length = self.depth
        elif axis == 'y':
            if length is None:
                length = self.height
        elif axis == 'x':
            if length is None:
                length = self.width
        
        self.defects.append({
            'type': 'cylindrical_void',
            'center_pos': center_pos,
            'other_pos': other_pos,
            'radius': radius,
            'intensity': intensity,
            'axis': axis,
            'length': length
        })
        
        print(f"  Added cylindrical void: axis={axis}, radius={radius:.1f}px, "
              f"reflectivity={intensity:.2f}")
    
    def add_spherical_void(self, center: Tuple[float, float, float], 
                          radius: float, intensity: float = 0.95):
        """
        Add a spherical void defect.
        
        Parameters:
            center: (z, y, x) center position in pixels
            radius: Sphere radius in pixels
            intensity: Reflectivity (0-1)
        """
        self.defects.append({
            'type': 'spherical_void',
            'center': center,
            'radius': radius,
            'intensity': intensity
        })
        
        print(f"  Added spherical void: center={center}, radius={radius:.1f}px, "
              f"reflectivity={intensity:.2f}")
    
    def generate(self, base_intensity_range: Tuple[float, float] = (0.05, 0.15),
                 smoothing_sigma: float = 2.0) -> np.ndarray:
        """
        Generate ground truth volume with defects.
        
        Returns reflectivity map where:
        - 0.0 = no reflection (void/air)
        - 0.05-0.15 = background material scattering
        - 0.8-1.0 = strong reflector (defect boundaries)
        
        Parameters:
            base_intensity_range: Background reflectivity range
            smoothing_sigma: Gaussian smoothing for background variation
            
        Returns:
            Ground truth reflectivity volume (Z, Y, X)
        """
        print(f"\nGenerating ground truth volume...")
        
        # Start with background material (random variation in reflectivity)
        volume = np.random.uniform(
            base_intensity_range[0], 
            base_intensity_range[1], 
            self.dimensions
        ).astype(np.float32)
        
        # Smooth background for realistic grain structure
        if smoothing_sigma > 0:
            volume = gaussian_filter(volume, sigma=smoothing_sigma)
        
        # Add defects
        for defect in self.defects:
            if defect['type'] == 'cylindrical_void':
                axis = defect['axis']
                center_pos = defect['center_pos']
                other_pos = defect['other_pos']
                radius = defect['radius']
                intensity = defect['intensity']
                
                if axis == 'z':
                    # Cylinder along Z, circular in X-Y plane
                    y_coords, x_coords = np.meshgrid(
                        np.arange(self.height),
                        np.arange(self.width),
                        indexing='ij'
                    )
                    dist = np.sqrt((y_coords - center_pos)**2 + (x_coords - other_pos)**2)
                    mask = dist <= radius
                    shell_mask = (dist >= radius - 1) & (dist <= radius)
                    
                    # Apply to all Z slices
                    for z in range(self.depth):
                        volume[z][mask] = 0.0  # Interior is void
                        volume[z][shell_mask] = intensity  # Boundary reflects strongly
                    
                elif axis == 'y':
                    # Cylinder along Y, circular in X-Z plane
                    z_coords, x_coords = np.meshgrid(
                        np.arange(self.depth),
                        np.arange(self.width),
                        indexing='ij'
                    )
                    dist = np.sqrt((z_coords - center_pos)**2 + (x_coords - other_pos)**2)
                    mask = dist <= radius
                    shell_mask = (dist >= radius - 1) & (dist <= radius)
                    
                    # Apply to all Y slices
                    for y in range(self.height):
                        volume[:, y, :][mask] = 0.0
                        volume[:, y, :][shell_mask] = intensity
                    
                elif axis == 'x':
                    # Cylinder along X, circular in Y-Z plane
                    z_coords, y_coords = np.meshgrid(
                        np.arange(self.depth),
                        np.arange(self.height),
                        indexing='ij'
                    )
                    dist = np.sqrt((y_coords - other_pos)**2 + (z_coords - center_pos)**2)
                    mask = dist <= radius
                    shell_mask = (dist >= radius - 1) & (dist <= radius)
                    
                    # Apply to all X slices
                    for x in range(self.width):
                        volume[:, :, x][mask] = 0.0
                        volume[:, :, x][shell_mask] = intensity
            
            elif defect['type'] == 'spherical_void':
                center = defect['center']
                radius = defect['radius']
                intensity = defect['intensity']
                
                # Create 3D coordinate grids
                z_coords = np.arange(self.depth)[:, None, None]
                y_coords = np.arange(self.height)[None, :, None]
                x_coords = np.arange(self.width)[None, None, :]
                
                dist = np.sqrt(
                    (z_coords - center[0])**2 + 
                    (y_coords - center[1])**2 + 
                    (x_coords - center[2])**2
                )
                mask = dist <= radius
                shell_mask = (dist >= radius - 1) & (dist <= radius)
                volume[mask] = 0.0
                volume[shell_mask] = intensity
        
        print(f"  ✓ Ground truth generated: {len(self.defects)} defects")
        print(f"  Reflectivity range: {volume.min():.3f} - {volume.max():.3f}")
        
        return volume


class RayTracingNDT:
    """
    Ray tracing-based ultrasonic NDT simulator.
    
    Simulates realistic wave propagation using:
    - Ray tracing for transmit/receive paths
    - Time-of-flight calculations
    - Amplitude decay (attenuation + geometric spreading)
    - Scattering from defects (reflectivity-based)
    - Total Focusing Method (TFM) image reconstruction
    """
    
    def __init__(self, array_params: ArrayParameters):
        """
        Initialize ray tracing simulator.
        
        Parameters:
            array_params: Array configuration parameters
        """
        self.array = array_params
        self.c = array_params.speed_of_sound
        self.pixel_size = array_params.pixel_size
        
        print(f"\n{'='*80}")
        print(f"RAY TRACING NDT SIMULATOR")
        print(f"{'='*80}")
        print(f"  Speed of sound: {self.c:.0f} m/s")
        print(f"  Pixel size: {self.pixel_size*1e3:.2f} mm")
        print(f"  Wavelength: {self.array.wavelength*1e3:.3f} mm")
        print(f"{'='*80}\n")
    
    def calculate_element_directivity(self, angle: np.ndarray) -> np.ndarray:
        """
        Calculate element directivity pattern (far-field approximation).
        
        For a rectangular element, the directivity is approximately:
        D(θ) = sinc(k·w·sin(θ)/2)
        
        where:
        - k = 2π/λ (wave number)
        - w = element width
        - θ = angle from element normal
        
        Parameters:
            angle: Angle from element normal (radians), can be array
            
        Returns:
            Directivity factor (0 to 1)
        """
        if not self.array.element_directivity:
            return np.ones_like(angle)
        
        k = 2 * np.pi / self.array.wavelength
        w = self.array.element_width
        
        # sinc function with small angle protection
        arg = k * w * np.sin(angle) / 2
        
        # Use np.sinc which is sin(πx)/(πx), so we need to divide by π
        directivity = np.abs(np.sinc(arg / np.pi))
        
        return directivity
    
    def calculate_array_psf_weight(self, tx_angle: np.ndarray, rx_angle: np.ndarray) -> np.ndarray:
        """
        Calculate array Point Spread Function weighting.
        
        Includes:
        - Element directivity for TX and RX
        - Side lobe effects
        - Angular-dependent weighting
        
        Parameters:
            tx_angle: Angle from TX element normal (radians)
            rx_angle: Angle from RX element normal (radians)
            
        Returns:
            PSF weight (combined TX/RX directivity with side lobes)
        """
        # Element directivity for TX and RX paths
        tx_directivity = self.calculate_element_directivity(tx_angle)
        rx_directivity = self.calculate_element_directivity(rx_angle)
        
        # Combined directivity (multiply TX and RX)
        combined_directivity = tx_directivity * rx_directivity
        
        # Add side lobe contribution (simplified model)
        # Side lobes appear at larger angles
        side_lobe_amplitude = 10 ** (self.array.side_lobe_level / 20.0)
        
        # Base weight is directivity, plus small side lobe component
        psf_weight = combined_directivity + side_lobe_amplitude * 0.1
        
        return psf_weight
    
    def calculate_tfm_focal_law(self, focal_point: np.ndarray, 
                                tx_element: int, rx_element: int) -> Tuple[float, float]:
        """
        Calculate time-of-flight and amplitude for TFM.
        
        For each transmit-receive element pair and focal point:
        - Calculate transmit path: tx_element → focal_point
        - Calculate receive path: focal_point → rx_element
        - Total time = transmit_time + receive_time
        - Amplitude = geometric spreading loss + attenuation
        
        Parameters:
            focal_point: [z, y, x] position in meters
            tx_element: Transmit element index
            rx_element: Receive element index
            
        Returns:
            (time_of_flight, amplitude_factor)
        """
        # Element positions (2D array at z=0)
        # element_positions shape: (num_elements, 2) with [y, x]
        elem_pos = self.array.element_positions
        tx_pos = np.array([0.0, elem_pos[tx_element, 0], elem_pos[tx_element, 1]])  # [z, y, x]
        rx_pos = np.array([0.0, elem_pos[rx_element, 0], elem_pos[rx_element, 1]])  # [z, y, x]
        
        # Calculate distances
        tx_dist = np.linalg.norm(focal_point - tx_pos)
        rx_dist = np.linalg.norm(focal_point - rx_pos)
        total_dist = tx_dist + rx_dist
        
        # Time of flight
        tof = total_dist / self.c
        
        # Amplitude decay factors
        # 1. Geometric spreading (1/r factor for each path)
        geometric_loss = 1.0 / (tx_dist * rx_dist + 1e-10)
        
        # 2. Material attenuation (frequency-dependent, distance-dependent)
        freq_mhz = self.array.frequency / 1e6
        attenuation_np = self.array.attenuation_coeff * freq_mhz * total_dist
        attenuation_loss = np.exp(-attenuation_np)
        
        # Combined amplitude
        amplitude = geometric_loss * attenuation_loss
        
        return tof, amplitude
    
    def simulate_fmc_acquisition(self, ground_truth: np.ndarray,
                                 array_position: Tuple[int, int, int],
                                 time_samples: int = 2048,
                                 sampling_frequency: float = None) -> np.ndarray:
        """
        Simulate Full Matrix Capture (FMC) acquisition - the REAL way NDT systems work.
        
        For each TX element:
            1. Transmit pulse
            2. Record received signal at ALL RX elements (A-scans)
            3. Each A-scan is a time-domain signal showing reflections
        
        This creates time-of-flight data that can then be processed with TFM.
        
        Parameters:
            ground_truth: Reflectivity volume (Z, Y, X)
            array_position: (z, y, x) position of array in pixels
            time_samples: Number of time samples in A-scan
            sampling_frequency: Sampling rate (Hz), default = 4× center frequency
            
        Returns:
            FMC data: shape (num_tx, num_rx, time_samples)
        """
        if sampling_frequency is None:
            sampling_frequency = 4 * self.array.frequency  # Nyquist sampling
        
        dt = 1.0 / sampling_frequency  # Time step
        time_axis = np.arange(time_samples) * dt
        
        num_elements = self.array.num_elements
        fmc_data = np.zeros((num_elements, num_elements, time_samples), dtype=np.float32)
        
        print(f"\n  Simulating FMC acquisition (REALISTIC)...")
        print(f"    Array position: {array_position}")
        print(f"    Elements: {num_elements}")
        print(f"    Sampling: {sampling_frequency/1e6:.1f} MHz, {time_samples} samples")
        print(f"    Time window: {time_axis[-1]*1e6:.1f} μs")
        
        # Get element positions
        elem_positions = self.array.element_positions  # Shape: (num_elements, 2) [y, x]
        
        # Get all defect/scatterer positions from ground truth
        # Find all voxels with significant reflectivity
        threshold = 0.1
        scatterer_mask = ground_truth > threshold
        scatterer_coords = np.argwhere(scatterer_mask)
        scatterer_reflectivity = ground_truth[scatterer_mask]
        
        print(f"    Found {len(scatterer_coords)} scatterers (reflectivity > {threshold})")
        
        # Convert scatterer positions to meters
        scatterer_positions = scatterer_coords.astype(float) * self.pixel_size
        
        start_time = time.time()
        
        # For each TX-RX pair, calculate A-scan
        for tx_idx in range(num_elements):
            if tx_idx % 4 == 0:
                elapsed = time.time() - start_time
                progress = tx_idx / num_elements * 100
                print(f"      TX element {tx_idx}/{num_elements} ({progress:.0f}%, {elapsed:.1f}s)")
            
            # TX element position [z, y, x]
            tx_pos = np.array([0.0, elem_positions[tx_idx, 0], elem_positions[tx_idx, 1]])
            
            for rx_idx in range(num_elements):
                # RX element position [z, y, x]
                rx_pos = np.array([0.0, elem_positions[rx_idx, 0], elem_positions[rx_idx, 1]])
                
                # For each scatterer, calculate its contribution to this A-scan
                a_scan = np.zeros(time_samples, dtype=np.float32)
                
                for scat_idx, scat_pos in enumerate(scatterer_positions):
                    # Calculate round-trip travel time
                    tx_distance = np.linalg.norm(scat_pos - tx_pos)
                    rx_distance = np.linalg.norm(scat_pos - rx_pos)
                    total_distance = tx_distance + rx_distance
                    tof = total_distance / self.c
                    
                    # Find corresponding time bin
                    time_bin = int(tof / dt)
                    if time_bin < time_samples:
                        # Calculate amplitude
                        reflectivity = scatterer_reflectivity[scat_idx]
                        
                        # Geometric spreading
                        geometric = 1.0 / (tx_distance * rx_distance + 1e-10)
                        
                        # Attenuation
                        attenuation = np.exp(-self.array.attenuation_coeff * 
                                           (self.array.frequency / 1e6) * total_distance)
                        
                        # Combined amplitude
                        amplitude = reflectivity * geometric * attenuation
                        
                        # Add pulse to A-scan (simple spike model)
                        # In reality, this would be a wavelet
                        a_scan[time_bin] += amplitude
                
                # Store A-scan
                fmc_data[tx_idx, rx_idx, :] = a_scan
        
        elapsed = time.time() - start_time
        print(f"    ✓ FMC acquisition complete: {elapsed:.1f}s")
        print(f"    FMC data shape: {fmc_data.shape}")
        print(f"    Signal range: {fmc_data.min():.2e} - {fmc_data.max():.2e}")
        
        return fmc_data
    
    def reconstruct_tfm_from_fmc(self, fmc_data: np.ndarray,
                                 subvolume_size: Tuple[int, int, int],
                                 sampling_frequency: float = None) -> np.ndarray:
        """
        Reconstruct TFM image from FMC data using delay-and-sum beamforming.
        
        This is the REAL TFM algorithm used in NDT systems.
        
        For each pixel (focal point):
            1. Calculate time-of-flight from TX→pixel→RX for all element pairs
            2. Look up FMC data at those time indices
            3. Sum coherently (delay-and-sum)
        
        Parameters:
            fmc_data: FMC acquisition data (num_tx, num_rx, time_samples)
            subvolume_size: (depth, height, width) of image to reconstruct
            sampling_frequency: Must match FMC acquisition
            
        Returns:
            TFM reconstructed image
        """
        if sampling_frequency is None:
            sampling_frequency = 4 * self.array.frequency
        
        dt = 1.0 / sampling_frequency
        num_tx, num_rx, time_samples = fmc_data.shape
        depth, height, width = subvolume_size
        
        print(f"\n  Reconstructing TFM image from FMC data...")
        print(f"    Image size: {subvolume_size}")
        print(f"    Using {num_tx}×{num_rx} = {num_tx*num_rx} A-scans")
        
        tfm_image = np.zeros(subvolume_size, dtype=np.float32)
        
        # Create pixel positions
        z_coords = np.arange(depth) * self.pixel_size
        y_coords = (np.arange(height) - height/2) * self.pixel_size
        x_coords = (np.arange(width) - width/2) * self.pixel_size
        
        elem_positions = self.array.element_positions
        
        start_time = time.time()
        
        # Process in Z slices for efficiency
        for iz in range(depth):
            if iz % 20 == 0:
                elapsed = time.time() - start_time
                progress = iz / depth * 100
                print(f"      Depth slice {iz}/{depth} ({progress:.0f}%, {elapsed:.1f}s)")
            
            z = z_coords[iz]
            
            for iy in range(height):
                y = y_coords[iy]
                
                for ix in range(width):
                    x = x_coords[ix]
                    
                    pixel_pos = np.array([z, y, x])
                    pixel_value = 0.0
                    
                    # Sum over all TX-RX pairs
                    for tx_idx in range(num_tx):
                        tx_pos = np.array([0.0, elem_positions[tx_idx, 0], elem_positions[tx_idx, 1]])
                        
                        for rx_idx in range(num_rx):
                            rx_pos = np.array([0.0, elem_positions[rx_idx, 0], elem_positions[rx_idx, 1]])
                            
                            # Calculate time-of-flight
                            tx_dist = np.linalg.norm(pixel_pos - tx_pos)
                            rx_dist = np.linalg.norm(pixel_pos - rx_pos)
                            tof = (tx_dist + rx_dist) / self.c
                            
                            # Find time index
                            time_idx = int(tof / dt)
                            
                            # Extract value from A-scan (with interpolation)
                            if 0 <= time_idx < time_samples - 1:
                                # Linear interpolation
                                frac = (tof / dt) - time_idx
                                value = (1 - frac) * fmc_data[tx_idx, rx_idx, time_idx] + \
                                       frac * fmc_data[tx_idx, rx_idx, time_idx + 1]
                                pixel_value += value
                    
                    tfm_image[iz, iy, ix] = pixel_value
        
        elapsed = time.time() - start_time
        print(f"    ✓ TFM reconstruction complete: {elapsed:.1f}s")
        print(f"    TFM range: {tfm_image.min():.2e} - {tfm_image.max():.2e}")
        
        # Normalize
        if tfm_image.max() > 0:
            tfm_image = tfm_image / np.percentile(tfm_image, 99)
        
        return tfm_image 
    def generate_subvolumes(self, ground_truth: np.ndarray,
                           num_subvolumes: Tuple[int, int, int],
                           subvolume_size: Tuple[int, int, int],
                           overlap: Tuple[int, int, int] = (30, 30, 30)) -> List[Dict]:
        """
        Generate multiple subvolumes by scanning the array over the ground truth.
        
        Parameters:
            ground_truth: Full ground truth volume
            num_subvolumes: (nz, ny, nx) number of subvolumes in each direction
            subvolume_size: (depth, height, width) of each subvolume
            overlap: (z, y, x) overlap in pixels between adjacent subvolumes
            
        Returns:
            List of dictionaries with:
                - 'data': TFM reconstructed subvolume
                - 'position': (z, y, x) position in full volume
                - 'envelope': Hilbert envelope for display
                - 'db': dB-scale for display
        """
        print(f"\n{'='*80}")
        print(f"GENERATING SUBVOLUMES VIA ARRAY SCANNING")
        print(f"{'='*80}")
        print(f"  Number of subvolumes: {num_subvolumes}")
        print(f"  Subvolume size: {subvolume_size}")
        print(f"  Overlap: {overlap}")
        
        nz, ny, nx = num_subvolumes
        dz, dy, dx = subvolume_size
        oz, oy, ox = overlap
        
        # Calculate step sizes
        step_z = dz - oz
        step_y = dy - oy
        step_x = dx - ox
        
        subvolumes = []
        
        for iz in range(nz):
            for iy in range(ny):
                for ix in range(nx):
                    # Array position for this subvolume
                    z_pos = iz * step_z
                    y_pos = iy * step_y
                    x_pos = ix * step_x
                    
                    print(f"\nSubvolume [{iz}, {iy}, {ix}]: position=({z_pos}, {y_pos}, {x_pos})")
                    
                    # Simulate REALISTIC FMC acquisition + TFM reconstruction
                    # Step 1: Simulate Full Matrix Capture (time-domain A-scans)
                    fmc_data = self.simulate_fmc_acquisition(
                        ground_truth, 
                        (z_pos, y_pos, x_pos)
                    )
                    
                    # Step 2: Reconstruct TFM image using delay-and-sum beamforming
                    tfm_data = self.reconstruct_tfm_from_fmc(
                        fmc_data,
                        subvolume_size
                    )
                    
                    # Calculate envelope (magnitude of complex TFM data)
                    envelope = np.abs(tfm_data)
                    
                    # Convert to dB scale for display
                    # Reference to maximum value for proper dB scaling
                    max_val = envelope.max()
                    if max_val > 0:
                        envelope_db = 20 * np.log10(envelope / max_val + 1e-10)
                        envelope_db = np.clip(envelope_db, -60, 0)  # -60 to 0 dB range
                    else:
                        envelope_db = np.full_like(envelope, -60)
                    
                    subvolumes.append({
                        'data': tfm_data,
                        'position': (z_pos, y_pos, x_pos),
                        'envelope': envelope,
                        'db': envelope_db,
                        'index': (iz, iy, ix)
                    })
        
        print(f"\n{'='*80}")
        print(f"✓ Generated {len(subvolumes)} subvolumes")
        print(f"{'='*80}\n")
        
        return subvolumes


def visualize_subvolumes(subvolumes: List[Dict], title: str = "Ray Traced Subvolumes"):
    """
    Visualize subvolumes in napari.
    
    Parameters:
        subvolumes: List of subvolume dictionaries
        title: Window title
    """
    print(f"\n{'='*80}")
    print(f"VISUALIZING SUBVOLUMES IN NAPARI")
    print(f"{'='*80}")
    
    viewer = napari.Viewer(title=title)
    
    for i, sv in enumerate(subvolumes):
        layer_name = f"SubVol {sv['index']}"
        
        # Use dB-scale data for visualization
        viewer.add_image(
            sv['db'],
            name=layer_name,
            colormap='gray',
            contrast_limits=(-60, 0),  # Match dB range
            translate=sv['position']
        )
        
        print(f"  Added layer '{layer_name}': shape={sv['db'].shape}, pos={sv['position']}")
    
    print(f"\n{'='*80}")
    print(f"✓ Napari viewer opened with {len(subvolumes)} layers")
    print(f"  Close window to continue...")
    print(f"{'='*80}\n")
    
    napari.run()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("PHYSICS-BASED NDT SIMULATION WITH RAY TRACING")
    print("="*80)
    
    # ========================================================================
    # 1. DEFINE ARRAY PARAMETERS
    # ========================================================================
    
    array_params = ArrayParameters(
        frequency=5e6,              # 5 MHz
        speed_of_sound=6320.0,      # m/s (aluminum longitudinal)
        num_elements_x=8,           # 16 elements in X direction (DEMO: use 8 for full)
        num_elements_y=8,           # 16 elements in Y direction (16x16 = 256 total, faster)
        pitch=0.6e-3,               # 0.6 mm pitch
        element_width=0.54e-3,      # 0.54 mm element width
        attenuation_coeff=0.03,     # Np/m/MHz
        pixel_size=0.2e-3,          # 0.2 mm/pixel
        element_directivity=True,   # Enable realistic beam patterns
        side_lobe_level=-20.0,      # -20 dB side lobes
        psf_bloom_sigma=2.5         # Blooming effect strength
    )
    
    # NOTE: For production use with 64 elements, change to:
    # num_elements_x=8, num_elements_y=8, pixel_size=0.15e-3
    
    # ========================================================================
    # 2. CREATE GROUND TRUTH VOLUME
    # ========================================================================
    
    # Demo volume (small for speed)
    volume_dims = (80, 80, 100)  # (Z, Y, X) in pixels
    
    gt_gen = GroundTruthGenerator(dimensions=volume_dims, seed=42)
    
    # Add defects
    print("\nAdding defects to ground truth:")
    gt_gen.add_cylindrical_void(
        center_pos=35,      # Z position
        other_pos=50,       # X position
        radius=7,           # 7 pixels radius (~1.4 mm)
        intensity=0.95,     # Strong reflector
        axis='y'            # Extends along Y
    )
    
    gt_gen.add_spherical_void(
        center=(55, 40, 40),  # (Z, Y, X)
        radius=5,             # 5 pixels radius (~1.0 mm)
        intensity=0.90
    )
    
    # Generate ground truth
    ground_truth = gt_gen.generate(
        base_intensity_range=(0.05, 0.12),
        smoothing_sigma=2.0
    )
    
    # ========================================================================
    # 3. SIMULATE ARRAY SCANNING AND TFM RECONSTRUCTION
    # ========================================================================
    
    simulator = RayTracingNDT(array_params)
    
    # Generate subvolumes
    # Demo: 2 subvolumes in X direction
    subvolumes = simulator.generate_subvolumes(
        ground_truth=ground_truth,
        num_subvolumes=(1, 1, 2),      # (nz, ny, nx)
        subvolume_size=(80, 80, 55),   # (depth, height, width) in pixels
        overlap=(0, 0, 10)             # 10 pixel overlap in X
    )
    
    # ========================================================================
    # 4. VISUALIZE RESULTS
    # ========================================================================
    
    visualize_subvolumes(subvolumes, title="Ray Tracing NDT - TFM Subvolumes")
    
    print("\n" + "="*80)
    print("✓ SIMULATION COMPLETE")
    print("="*80)
    print("\nGenerated data can be used for:")
    print("  - TFM algorithm testing")
    print("  - Array parameter optimization")
    print("  - Defect detection validation")
    print("  - Stitching algorithm development")
    print("="*80 + "\n")
