"""
Born-only 2D ultrasonic NDT synthetic data engine.

Modules:
    config         - Dataclass configurations for simulation parameters
    materials      - Material presets and acoustic impedance helpers
    waveforms      - Gabor pulse generation utilities
    propagation    - Geometric spreading, attenuation, element directivity
    geometry       - Specimen and defect geometry (defects emit Born scatterers)
    voxel_volume   - 3D impedance grid, Born-scatterer extraction
    microstructure - Voronoi grain generation
    fmc_engine     - Born-only FMC acquisition simulator
"""
