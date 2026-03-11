"""
Physics-accurate 2D ultrasonic NDT synthetic data engine.

Modules:
    config       - Dataclass configurations for simulation parameters
    materials    - Material presets and acoustic impedance calculations
    waveforms    - Gabor pulse generation and A-scan synthesis
    interfaces   - Snell's law and Fresnel/Zoeppritz coefficients
    propagation  - Geometric spreading, attenuation, element directivity
    geometry     - Specimen and defect geometry definitions
    scattering   - Kirchhoff surface scattering from defect boundaries
    rays         - Ray path enumeration (direct, skip, corner trap)
    fmc_engine   - FMC acquisition simulation orchestrator
"""
