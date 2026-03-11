#!/usr/bin/env python3
"""
2D RAY TRACING NDT SIMULATION
==============================

Simplified 2D ultrasonic NDT simulation using:
- 1D linear array (elements along X-axis)
- FMC (Full Matrix Capture) acquisition
- TFM (Total Focusing Method) reconstruction
- Generates realistic B-scan images

This is a simplified version for faster testing and development.

Workflow:
1. Create 2D ground truth (Z, X) with defects
2. FMC Acquisition: Simulate A-scans for all TX-RX pairs
3. TFM Reconstruction: Delay-and-sum beamforming → B-scan
4. Visualize 2D image

Author: Generated for NDT simulation
Date: 20 Feb 2026
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, List, Dict
import sys
import os
from scipy.signal import hilbert

# Import filter from Classes
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from Classes.Filter import filter_signal

# Qt configuration for matplotlib
os.environ['QT_API'] = 'pyqt5'

@dataclass
class ArrayParameters1D:
    """
    Parameters for a 1D linear ultrasound array.
    
    Attributes:
        num_elements: Number of elements in the linear array
        element_pitch: Distance between element centers (meters)
        element_width: Width of each element (meters)
        frequency: Center frequency (Hz)
        speed_of_sound: Material sound velocity (m/s)
        attenuation_coeff: Material attenuation coefficient (Np/m/MHz)
    """
    num_elements: int = 32
    element_pitch: float = 0.6e-3  # 0.6 mm
    element_width: float = 0.54e-3  # 0.54 mm (90% of pitch)
    frequency: float = 5e6  # 5 MHz
    speed_of_sound: float = 6320.0  # Aluminum longitudinal wave
    attenuation_coeff: float = 0.03  # Np/m/MHz
    
    @property
    def wavelength(self) -> float:
        """Calculate wavelength."""
        return self.speed_of_sound / self.frequency
    
    @property
    def aperture_size(self) -> float:
        """Total aperture size of the array."""
        return (self.num_elements - 1) * self.element_pitch
    
    @property
    def element_positions(self) -> np.ndarray:
        """
        Get element positions along X-axis.
        
        Returns:
            Array of shape (num_elements,) with X positions centered at 0
        """
        indices = np.arange(self.num_elements)
        positions = (indices - (self.num_elements - 1) / 2) * self.element_pitch
        return positions
    
    def __post_init__(self):
        """Validate and print configuration."""
        print(f"\n{'='*80}")
        print(f"ARRAY PARAMETERS (1D LINEAR ARRAY)")
        print(f"{'='*80}")
        print(f"  Frequency: {self.frequency/1e6:.1f} MHz")
        print(f"  Wavelength: {self.wavelength*1e3:.3f} mm")
        print(f"  Speed of sound: {self.speed_of_sound:.0f} m/s")
        print(f"  Number of elements: {self.num_elements}")
        print(f"  Element pitch: {self.element_pitch*1e3:.2f} mm")
        print(f"  Element width: {self.element_width*1e3:.2f} mm")
        print(f"  Total aperture: {self.aperture_size*1e3:.2f} mm")
        print(f"  Attenuation: {self.attenuation_coeff:.3f} Np/m/MHz")
        print(f"{'='*80}\n")


class GroundTruthGenerator2D:
    """
    Generate 2D ground truth cross-sections with defects.
    
    Creates a 2D plane (Z=depth, X=lateral) representing a material cross-section
    with embedded defects (voids, inclusions, etc.).
    """
    
    def __init__(self, dimensions: Tuple[int, int], pixel_size: float = 0.2e-3, seed: int = None):
        """
        Initialize generator.
        
        Args:
            dimensions: (depth, width) in pixels
            pixel_size: Size of each pixel in meters
            seed: Random seed for reproducibility
        """
        self.depth, self.width = dimensions
        self.pixel_size = pixel_size
        self.defects = []
        
        if seed is not None:
            np.random.seed(seed)
        
        print(f"\n{'='*80}")
        print(f"2D GROUND TRUTH GENERATOR")
        print(f"{'='*80}")
        print(f"  Image size: {self.depth} × {self.width} pixels")
        print(f"  Physical size: {self.depth*pixel_size*1e3:.1f} × {self.width*pixel_size*1e3:.1f} mm")
        print(f"  Pixel size: {pixel_size*1e3:.3f} mm")
        print(f"{'='*80}\n")
    
    def add_circular_void(self, center: Tuple[float, float], radius: float, 
                         shell_thickness: float = 0.8e-3):
        """
        Add a circular void (appears as circle in 2D cross-section).
        
        Args:
            center: (z, x) position in pixels
            radius: Void radius in pixels
            shell_thickness: Thickness of high-reflectivity shell in meters
        """
        self.defects.append({
            'type': 'circle',
            'center': center,
            'radius': radius,
            'shell_thickness': shell_thickness
        })
        
        z_mm = center[0] * self.pixel_size * 1e3
        x_mm = center[1] * self.pixel_size * 1e3
        r_mm = radius * self.pixel_size * 1e3
        
        print(f"  Added circular void: center=({z_mm:.1f}, {x_mm:.1f}) mm, radius={r_mm:.1f} mm")
    
    def add_rectangular_void(self, center: Tuple[float, float], 
                            size: Tuple[float, float],
                            shell_thickness: float = 0.8e-3):
        """
        Add a rectangular void.
        
        Args:
            center: (z, x) position in pixels
            size: (depth, width) in pixels
            shell_thickness: Thickness of shell in meters
        """
        self.defects.append({
            'type': 'rectangle',
            'center': center,
            'size': size,
            'shell_thickness': shell_thickness
        })
        
        z_mm = center[0] * self.pixel_size * 1e3
        x_mm = center[1] * self.pixel_size * 1e3
        d_mm = size[0] * self.pixel_size * 1e3
        w_mm = size[1] * self.pixel_size * 1e3
        
        print(f"  Added rectangular void: center=({z_mm:.1f}, {x_mm:.1f}) mm, size=({d_mm:.1f}×{w_mm:.1f}) mm")
    
    def generate(self, base_intensity_range: Tuple[float, float] = (0.05, 0.15),
                defect_intensity_range: Tuple[float, float] = (0.8, 1.0)) -> np.ndarray:
        """
        Generate the 2D reflectivity map.
        
        Args:
            base_intensity_range: (min, max) reflectivity for background material
            defect_intensity_range: (min, max) reflectivity for defect shells
            
        Returns:
            2D array (depth, width) with reflectivity values [0, 1]
        """
        print(f"\nGenerating 2D ground truth...")
        
        # Initialize with background reflectivity
        reflectivity = np.random.uniform(
            base_intensity_range[0],
            base_intensity_range[1],
            size=(self.depth, self.width)
        )
        
        # Create coordinate grids
        z_coords = np.arange(self.depth)
        x_coords = np.arange(self.width)
        Z, X = np.meshgrid(z_coords, x_coords, indexing='ij')
        
        # Add each defect
        for defect in self.defects:
            if defect['type'] == 'circle':
                z_c, x_c = defect['center']
                radius = defect['radius']
                shell_thickness_px = defect['shell_thickness'] / self.pixel_size
                
                # Distance from center
                dist = np.sqrt((Z - z_c)**2 + (X - x_c)**2)
                
                # Void interior (zero reflectivity)
                void_mask = dist <= radius
                reflectivity[void_mask] = 0.0
                
                # Shell (high reflectivity)
                shell_mask = (dist > radius) & (dist <= radius + shell_thickness_px)
                shell_intensity = np.random.uniform(
                    defect_intensity_range[0],
                    defect_intensity_range[1],
                    size=np.sum(shell_mask)
                )
                reflectivity[shell_mask] = shell_intensity
            
            elif defect['type'] == 'rectangle':
                z_c, x_c = defect['center']
                d_size, w_size = defect['size']
                shell_thickness_px = defect['shell_thickness'] / self.pixel_size
                
                # Distances from edges
                z_dist = np.abs(Z - z_c)
                x_dist = np.abs(X - x_c)
                
                # Void interior
                void_mask = (z_dist <= d_size/2) & (x_dist <= w_size/2)
                reflectivity[void_mask] = 0.0
                
                # Shell
                shell_mask = (
                    (z_dist <= d_size/2 + shell_thickness_px) &
                    (x_dist <= w_size/2 + shell_thickness_px) &
                    ~void_mask
                )
                shell_intensity = np.random.uniform(
                    defect_intensity_range[0],
                    defect_intensity_range[1],
                    size=np.sum(shell_mask)
                )
                reflectivity[shell_mask] = shell_intensity
        
        print(f"  Ground truth generated: {len(self.defects)} defects")
        print(f"  Reflectivity range: {reflectivity.min():.3f} - {reflectivity.max():.3f}")
        
        return reflectivity


class RayTracingNDT2D:
    """
    2D NDT simulation using ray tracing with FMC+TFM.
    
    Implements realistic ultrasonic imaging workflow:
    1. FMC Acquisition: Simulate time-domain A-scans
    2. TFM Reconstruction: Delay-and-sum beamforming
    """
    
    def __init__(self, array_params: ArrayParameters1D, pixel_size: float = 0.2e-3):
        """
        Initialize simulator.
        
        Args:
            array_params: Array configuration
            pixel_size: Pixel size in meters
        """
        self.array = array_params
        self.pixel_size = pixel_size
        self.c = array_params.speed_of_sound
    
    def simulate_fmc_acquisition(self, ground_truth: np.ndarray,
                                 time_samples: int = 2048,
                                 sampling_frequency: float = None) -> np.ndarray:
        """
        Simulate Full Matrix Capture (FMC) acquisition.
        
        For each TX element:
            - Transmit pulse
            - Record A-scans at all RX elements
            - Each A-scan shows echoes from scatterers
        
        Args:
            ground_truth: 2D reflectivity map (depth, width)
            time_samples: Number of time samples per A-scan
            sampling_frequency: Sampling rate (Hz), default = 4× center frequency
            
        Returns:
            FMC data: shape (num_tx, num_rx, time_samples)
        """
        if sampling_frequency is None:
            sampling_frequency = 4 * self.array.frequency
        
        dt = 1.0 / sampling_frequency
        time_axis = np.arange(time_samples) * dt
        
        num_elements = self.array.num_elements
        fmc_data = np.zeros((num_elements, num_elements, time_samples), dtype=np.float32)
        
        print(f"\n{'='*80}")
        print(f"FMC ACQUISITION (2D)")
        print(f"{'='*80}")
        print(f"  Array elements: {num_elements}")
        print(f"  Sampling: {sampling_frequency/1e6:.1f} MHz, {time_samples} samples")
        print(f"  Time window: {time_axis[-1]*1e6:.1f} μs")
        print(f"  Max depth: {self.c * time_axis[-1] / 2 * 1e3:.1f} mm")
        
        # Get element positions
        elem_positions = self.array.element_positions  # Shape: (num_elements,)
        
        # Find scatterers (high reflectivity points)
        threshold = 0.1
        scatterer_mask = ground_truth > threshold
        scatterer_coords = np.argwhere(scatterer_mask)  # (N, 2) array of [z, x]
        scatterer_reflectivity = ground_truth[scatterer_mask]
        
        # Convert to meters with proper centering
        # Z coordinate: starts from 0 (top of image)
        # X coordinate: centered at 0 (middle of image)
        depth, width = ground_truth.shape
        scatterer_positions = np.zeros_like(scatterer_coords, dtype=float)
        scatterer_positions[:, 0] = scatterer_coords[:, 0] * self.pixel_size  # Z
        scatterer_positions[:, 1] = (scatterer_coords[:, 1] - width/2) * self.pixel_size  # X (centered)
        
        print(f"  Found {len(scatterer_coords)} scatterers (reflectivity > {threshold})")
        print(f"  Progress:")
        
        start_time = time.time()
        
        # For each TX-RX pair
        for tx_idx in range(num_elements):
            if tx_idx % 8 == 0:
                elapsed = time.time() - start_time
                progress = tx_idx / num_elements * 100
                print(f"    TX element {tx_idx}/{num_elements} ({progress:.0f}%, {elapsed:.1f}s)")
            
            # TX element position [z=0, x]
            tx_pos = np.array([0.0, elem_positions[tx_idx]])
            
            for rx_idx in range(num_elements):
                # RX element position [z=0, x]
                rx_pos = np.array([0.0, elem_positions[rx_idx]])
                
                # Initialize A-scan
                a_scan = np.zeros(time_samples, dtype=np.float32)
                
                # Add contribution from each scatterer
                for scat_idx, scat_pos in enumerate(scatterer_positions):
                    # Calculate time-of-flight
                    tx_distance = np.linalg.norm(scat_pos - tx_pos)
                    rx_distance = np.linalg.norm(scat_pos - rx_pos)
                    total_distance = tx_distance + rx_distance
                    tof = total_distance / self.c
                    
                    # Find time bin
                    time_bin = int(tof / dt)
                    
                    if time_bin < time_samples:
                        # Calculate amplitude
                        reflectivity = scatterer_reflectivity[scat_idx]
                        
                        # Geometric spreading (spherical wave)
                        geometric = 1.0 / (tx_distance * rx_distance + 1e-10)
                        
                        # Material attenuation
                        attenuation = np.exp(-self.array.attenuation_coeff * 
                                           (self.array.frequency / 1e6) * total_distance)
                        
                        # Combined amplitude
                        amplitude = reflectivity * geometric * attenuation
                        
                        # Add to A-scan (spike model - could use wavelet)
                        a_scan[time_bin] += amplitude
                
                # Store A-scan
                fmc_data[tx_idx, rx_idx, :] = a_scan
        
        elapsed = time.time() - start_time
        print(f"  ✓ FMC acquisition complete: {elapsed:.1f}s")
        print(f"  FMC data shape: {fmc_data.shape}")
        print(f"  Signal range: {fmc_data.min():.2e} - {fmc_data.max():.2e}")
        print(f"{'='*80}\n")
        
        return fmc_data
    
    def add_realistic_noise(self, fmc_data: np.ndarray, 
                           snr_db: float = 35.0,
                           grain_noise_level: float = 0.05) -> np.ndarray:
        """
        Add realistic noise to FMC data to match experimental conditions.
        
        This reduces coherent interference patterns by:
        1. Adding Gaussian electronic noise
        2. Adding grain scattering noise (structured background)
        
        Args:
            fmc_data: Clean FMC data (num_tx, num_rx, time_samples)
            snr_db: Target signal-to-noise ratio in dB (typical: 30-40 dB)
            grain_noise_level: Grain scattering amplitude relative to signal
            
        Returns:
            Noisy FMC data with same shape
        """
        print(f"\n{'='*80}")
        print(f"ADDING REALISTIC NOISE")
        print(f"{'='*80}")
        print(f"  Target SNR: {snr_db:.1f} dB")
        print(f"  Grain scattering level: {grain_noise_level:.3f}")
        
        # Calculate signal power
        signal_power = np.mean(fmc_data**2)
        
        # Calculate noise power from SNR
        snr_linear = 10**(snr_db / 10)
        noise_power = signal_power / snr_linear
        noise_std = np.sqrt(noise_power)
        
        # 1. Add Gaussian electronic noise
        gaussian_noise = np.random.normal(0, noise_std, fmc_data.shape).astype(np.float32)
        
        # 2. Add grain scattering noise (structured, time-correlated)
        # Grain scattering creates random echoes throughout the material
        grain_noise = np.zeros_like(fmc_data)
        num_tx, num_rx, time_samples = fmc_data.shape
        
        # Add random grain echoes (more at later times/deeper)
        for tx_idx in range(num_tx):
            for rx_idx in range(num_rx):
                # Number of grain echoes increases with depth
                num_grains = np.random.randint(50, 150)
                
                for _ in range(num_grains):
                    # Random time position (biased toward middle/late times)
                    time_pos = int(np.random.beta(2, 2) * time_samples)
                    
                    if time_pos < time_samples:
                        # Random amplitude (Rayleigh distributed)
                        amplitude = np.random.rayleigh(grain_noise_level * noise_std)
                        
                        # Add with small spread (simulates grain size)
                        spread = 3
                        for offset in range(-spread, spread+1):
                            idx = time_pos + offset
                            if 0 <= idx < time_samples:
                                weight = np.exp(-offset**2 / 2)
                                grain_noise[tx_idx, rx_idx, idx] += amplitude * weight
        
        # Combine noises
        fmc_noisy = fmc_data + gaussian_noise + grain_noise
        
        # Calculate actual SNR achieved
        noise_added = gaussian_noise + grain_noise
        actual_snr = 10 * np.log10(signal_power / np.mean(noise_added**2))
        
        print(f"  Gaussian noise std: {noise_std:.2e}")
        print(f"  Actual SNR achieved: {actual_snr:.1f} dB")
        print(f"  Noisy data range: {fmc_noisy.min():.2e} - {fmc_noisy.max():.2e}")
        print(f"{'='*80}\n")
        
        return fmc_noisy
    
    def apply_bandpass_filter(self, fmc_data: np.ndarray,
                             sampling_frequency: float = None,
                             bandwidth_percentage: float = 0.1,
                             filter_alpha: float = 0.9,
                             hanning_window: bool = False) -> np.ndarray:
        """
        Apply bandpass filtering to FMC data (same as MATtoCSV.py workflow).
        
        This filters each A-scan to remove noise and focus on signal around
        the center frequency of the transducer.
        
        Args:
            fmc_data: FMC data (num_tx, num_rx, time_samples)
            sampling_frequency: Sampling rate (Hz), default = 4× center frequency
            bandwidth_percentage: Bandwidth as fraction of center frequency (±%)
            filter_alpha: Tukey window taper parameter (0=rect, 1=Hann)
            hanning_window: Apply Hanning window in time domain before FFT
            
        Returns:
            Filtered FMC data with same shape
        """
        if sampling_frequency is None:
            sampling_frequency = 4 * self.array.frequency
        
        dt = 1.0 / sampling_frequency
        num_tx, num_rx, time_samples = fmc_data.shape
        
        # Calculate filter bandwidth
        center_freq_mhz = self.array.frequency / 1e6
        freq_spacing = center_freq_mhz * bandwidth_percentage
        f_start = center_freq_mhz - freq_spacing
        f_end = center_freq_mhz + freq_spacing
        
        print(f"\n{'='*80}")
        print(f"BANDPASS FILTERING")
        print(f"{'='*80}")
        print(f"  Center frequency: {center_freq_mhz:.1f} MHz")
        print(f"  Filter range: {f_start:.2f} - {f_end:.2f} MHz")
        print(f"  Tukey alpha: {filter_alpha}")
        print(f"  Hanning window: {hanning_window}")
        print(f"  Filtering {num_tx * num_rx} A-scans...")
        
        start_time = time.time()
        
        # Apply filtering to each A-scan
        fmc_filtered = np.zeros_like(fmc_data)
        
        for tx_idx in range(num_tx):
            for rx_idx in range(num_rx):
                a_scan = fmc_data[tx_idx, rx_idx, :]
                
                # Apply bandpass filter (from Classes/Filter.py)
                filtered = filter_signal(
                    signal=a_scan,
                    dt=dt,
                    f_start=f_start,
                    f_end=f_end,
                    filter_alpha=filter_alpha,
                    hanning_bool=hanning_window
                )
                
                fmc_filtered[tx_idx, rx_idx, :] = filtered
        
        elapsed = time.time() - start_time
        print(f"  ✓ Filtering complete: {elapsed:.1f}s")
        print(f"  Signal range: {fmc_filtered.min():.2e} - {fmc_filtered.max():.2e}")
        print(f"{'='*80}\n")
        
        return fmc_filtered
    
    def reconstruct_tfm_from_fmc(self, fmc_data: np.ndarray,
                                 image_size: Tuple[int, int],
                                 sampling_frequency: float = None) -> np.ndarray:
        """
        Reconstruct B-scan image using TFM (delay-and-sum beamforming).
        
        For each pixel:
            - Calculate TOF delays for all TX-RX pairs
            - Extract values from A-scans
            - Coherently sum (delay-and-sum)
        
        Args:
            fmc_data: FMC acquisition data (num_tx, num_rx, time_samples)
            image_size: (depth, width) of B-scan in pixels
            sampling_frequency: Must match FMC acquisition
            
        Returns:
            TFM B-scan image (depth, width)
        """
        if sampling_frequency is None:
            sampling_frequency = 4 * self.array.frequency
        
        dt = 1.0 / sampling_frequency
        num_tx, num_rx, time_samples = fmc_data.shape
        depth, width = image_size
        
        print(f"{'='*80}")
        print(f"TFM RECONSTRUCTION (2D)")
        print(f"{'='*80}")
        print(f"  Image size: {depth} × {width} pixels")
        print(f"  Physical size: {depth*self.pixel_size*1e3:.1f} × {width*self.pixel_size*1e3:.1f} mm")
        print(f"  Using {num_tx}×{num_rx} = {num_tx*num_rx} A-scans")
        
        tfm_image = np.zeros((depth, width), dtype=np.float32)
        
        # Create pixel coordinates (meters)
        z_coords = np.arange(depth) * self.pixel_size
        x_coords = (np.arange(width) - width/2) * self.pixel_size
        
        # Element positions
        elem_positions = self.array.element_positions
        
        print(f"  Progress:")
        start_time = time.time()
        
        # For each depth slice
        for iz in range(depth):
            if iz % 50 == 0:
                elapsed = time.time() - start_time
                progress = iz / depth * 100
                print(f"    Depth {iz}/{depth} ({progress:.0f}%, {elapsed:.1f}s)")
            
            z = z_coords[iz]
            
            for ix in range(width):
                x = x_coords[ix]
                pixel_pos = np.array([z, x])
                
                pixel_value = 0.0
                
                # Sum over all TX-RX pairs (delay-and-sum)
                for tx_idx in range(num_tx):
                    tx_pos = np.array([0.0, elem_positions[tx_idx]])
                    
                    for rx_idx in range(num_rx):
                        rx_pos = np.array([0.0, elem_positions[rx_idx]])
                        
                        # Calculate TOF
                        tx_dist = np.linalg.norm(pixel_pos - tx_pos)
                        rx_dist = np.linalg.norm(pixel_pos - rx_pos)
                        tof = (tx_dist + rx_dist) / self.c
                        
                        # Find time index
                        time_idx = int(tof / dt)
                        
                        # Extract value with interpolation
                        if 0 <= time_idx < time_samples - 1:
                            frac = (tof / dt) - time_idx
                            value = (1 - frac) * fmc_data[tx_idx, rx_idx, time_idx] + \
                                   frac * fmc_data[tx_idx, rx_idx, time_idx + 1]
                            pixel_value += value
                
                tfm_image[iz, ix] = pixel_value
        
        elapsed = time.time() - start_time
        print(f"  ✓ TFM reconstruction complete: {elapsed:.1f}s")
        print(f"  TFM range (raw): {tfm_image.min():.2e} - {tfm_image.max():.2e}")
        
        # Apply Hilbert transform for envelope detection (removes aliasing)
        # This is what CTFM1D does - apply along depth axis
        print(f"  Applying Hilbert transform for envelope detection...")
        tfm_analytic = hilbert(tfm_image, axis=0)
        tfm_envelope = np.abs(tfm_analytic).astype(np.float32)
        
        print(f"  Envelope range: {tfm_envelope.min():.2e} - {tfm_envelope.max():.2e}")
        
        # Normalize to 0-1 range using 99th percentile
        percentile_99 = np.percentile(tfm_envelope, 99)
        if percentile_99 > 0:
            tfm_normalized = tfm_envelope / percentile_99
            tfm_normalized = np.clip(tfm_normalized, 0, 1)  # Clip to [0, 1]
        else:
            tfm_normalized = tfm_envelope
        
        print(f"  Normalized range: {tfm_normalized.min():.3f} - {tfm_normalized.max():.3f}")
        print(f"  99th percentile: {percentile_99:.2e}")
        print(f"{'='*80}\n")
        
        return tfm_normalized


def visualize_results(ground_truth: np.ndarray, 
                     tfm_image: np.ndarray,
                     fmc_data: np.ndarray = None,
                     pixel_size: float = 0.2e-3):
    """
    Visualize ground truth, FMC data, and TFM B-scan.
    
    Args:
        ground_truth: Ground truth reflectivity (depth, width)
        tfm_image: TFM reconstructed image (depth, width)
        fmc_data: FMC A-scans (num_tx, num_rx, time_samples), optional
        pixel_size: Pixel size in meters
    """
    print(f"{'='*80}")
    print(f"VISUALIZATION")
    print(f"{'='*80}")
    
    if fmc_data is not None:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        axes = [axes]
    
    depth, width = ground_truth.shape
    extent_mm = [
        -width/2 * pixel_size * 1e3,
        width/2 * pixel_size * 1e3,
        depth * pixel_size * 1e3,
        0
    ]
    
    # Ground truth
    ax = axes[0] if fmc_data is None else axes[0, 0]
    im1 = ax.imshow(ground_truth, aspect='auto', cmap='hot', extent=extent_mm)
    ax.set_xlabel('Lateral Position (mm)')
    ax.set_ylabel('Depth (mm)')
    ax.set_title('Ground Truth Reflectivity')
    plt.colorbar(im1, ax=ax, label='Reflectivity')
    
    # TFM B-scan (dB scale)
    ax = axes[1] if fmc_data is None else axes[0, 1]
    # Convert normalized [0,1] to dB scale
    tfm_db = 20 * np.log10(tfm_image + 1e-10)
    tfm_db = np.clip(tfm_db, -60, 0)  # Clip to -60 to 0 dB range
    im2 = ax.imshow(tfm_db, aspect='auto', cmap='hot', extent=extent_mm, vmin=-60, vmax=0)
    ax.set_xlabel('Lateral Position (mm)')
    ax.set_ylabel('Depth (mm)')
    ax.set_title('TFM B-Scan (dB scale)')
    plt.colorbar(im2, ax=ax, label='Amplitude (dB)')
    
    if fmc_data is not None:
        # FMC A-scan heatmap (all RX for center TX element)
        center_elem = fmc_data.shape[0] // 2
        ax = axes[1, 0]
        time_axis_us = np.arange(fmc_data.shape[2]) / (4 * 5e6) * 1e6
        
        # Show all A-scans for center TX element as heatmap
        a_scans = fmc_data[center_elem, :, :]  # Shape: (num_rx, time_samples)
        
        im3 = ax.imshow(a_scans, aspect='auto', cmap='seismic', 
                       extent=[time_axis_us[0], time_axis_us[-1], 
                              fmc_data.shape[1]-0.5, -0.5],
                       vmin=-np.percentile(np.abs(a_scans), 95),
                       vmax=np.percentile(np.abs(a_scans), 95))
        ax.set_xlabel('Time (μs)')
        ax.set_ylabel('RX Element')
        ax.set_title(f'A-Scans Heatmap (TX element {center_elem})')
        plt.colorbar(im3, ax=ax, label='Amplitude')
        
        # FMC full matrix view
        ax = axes[1, 1]
        fmc_max = np.max(fmc_data, axis=2)  # Max amplitude per TX-RX pair
        
        # Use log scale to reduce diagonal dominance
        fmc_max_db = 20 * np.log10(fmc_max + 1e-10)
        fmc_max_db = np.clip(fmc_max_db, fmc_max_db.max() - 60, fmc_max_db.max())
        
        im4 = ax.imshow(fmc_max_db, aspect='auto', cmap='viridis')
        ax.set_xlabel('RX Element')
        ax.set_ylabel('TX Element')
        ax.set_title('FMC Matrix (Max Amplitude, dB scale)')
        plt.colorbar(im4, ax=ax, label='Amplitude (dB)')
    
    plt.tight_layout()
    plt.savefig('ndt_2d_results.png', dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved results to: ndt_2d_results.png")
    
    plt.show()
    print(f"  ✓ Visualization complete")
    print(f"{'='*80}\n")


def main():
    """Main execution."""
    print(f"\n{'#'*80}")
    print(f"# 2D RAY TRACING NDT SIMULATION")
    print(f"# FMC + TFM Workflow")
    print(f"{'#'*80}\n")
    
    # Configuration
    pixel_size = 0.2e-3  # 0.2 mm pixels
    
    # 1. Configure 1D array
    array_params = ArrayParameters1D(
        num_elements=32,
        element_pitch=0.6e-3,
        frequency=5e6
    )
    
    # 2. Create 2D ground truth
    image_depth = 200  # pixels (40 mm at 0.2mm/pixel)
    image_width = 300  # pixels (60 mm)
    
    generator = GroundTruthGenerator2D(
        dimensions=(image_depth, image_width),
        pixel_size=pixel_size,
        seed=42
    )
    
    # Add defects
    generator.add_circular_void(center=(60, 100), radius=15)  # Left circle
    generator.add_circular_void(center=(80, 200), radius=12)  # Right circle
    generator.add_rectangular_void(center=(120, 150), size=(20, 30))  # Rectangle
    
    ground_truth = generator.generate()
    
    # 3. Initialize simulator
    simulator = RayTracingNDT2D(array_params, pixel_size=pixel_size)
    
    # 4. FMC Acquisition
    fmc_data = simulator.simulate_fmc_acquisition(ground_truth)
    
    # 5. Add Realistic Noise (reduces interference patterns)
    ADD_NOISE = True  # Set to False for idealized "clean" simulation
    
    if ADD_NOISE:
        fmc_data = simulator.add_realistic_noise(
            fmc_data,
            snr_db=35.0,              # Typical experimental SNR: 30-40 dB
            grain_noise_level=0.05    # Grain scattering amplitude
        )
    else:
        print("\n[Noise DISABLED - Clean idealized simulation]\n")
    
    # 6. Apply Bandpass Filter (matches MATtoCSV.py workflow)
    APPLY_FILTERING = True  # Set to False to skip filtering
    
    if APPLY_FILTERING:
        fmc_filtered = simulator.apply_bandpass_filter(
            fmc_data,
            bandwidth_percentage=0.1,  # ±10% of center frequency
            filter_alpha=0.9,          # Tukey taper parameter
            hanning_window=False       # Optional time-domain windowing
        )
    else:
        fmc_filtered = fmc_data
        print("\n[Filtering DISABLED]\n")
    
    # 7. TFM Reconstruction
    tfm_image = simulator.reconstruct_tfm_from_fmc(
        fmc_filtered,
        image_size=(image_depth, image_width)
    )
    
    # 8. Visualize
    visualize_results(ground_truth, tfm_image, fmc_filtered, pixel_size)
    
    print(f"\n{'#'*80}")
    print(f"# SIMULATION COMPLETE")
    print(f"{'#'*80}\n")


if __name__ == "__main__":
    main()
