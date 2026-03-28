"""
PNG → NPY Converter for 3D Reconstruction
==========================================

Converts TFM B-scan PNG images (saved by Imaging.py with a matplotlib
colormap) back to dB-scaled NumPy arrays (.npy) suitable for
``reconstruct_3d.py``.

The conversion reverses the colormap mapping using the vmin/vmax dB
thresholds stored in each dataset's Params.txt file.

Usage:
    python png_to_npy.py --input "DATA/1D TFM Data/Al Pure 10MHz 12022026 Filtered" \
                         --output "SYNTHETIC DATA/output/real_al_10mhz"

    python png_to_npy.py --input "DATA/1D TFM Data/Al Pure 10MHz 12022026 Filtered" \
                         --output "SYNTHETIC DATA/output/real_al_10mhz" \
                         --vmin -20 --vmax 0  # override Params.txt values
"""

import os
import sys
import argparse
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from PIL import Image


def parse_params(params_path: str) -> dict:
    """Parse a TFM Params.txt file into a dict of values.

    Handles lines like:
        vmin = -20.0
        c    = 6700 # m/s
        cmap = 'viridis'
    """
    params = {}
    with open(params_path) as f:
        for line in f:
            line = line.split('#')[0].strip()  # strip comments
            if '=' not in line:
                continue
            key, val = line.split('=', 1)
            key = key.strip()
            val = val.strip().strip("'\"")
            # Try numeric conversion
            try:
                val = int(val)
            except ValueError:
                try:
                    val = float(val)
                except ValueError:
                    pass  # keep as string
            params[key] = val
    return params


def _is_greyscale_cmap(cmap_name: str) -> bool:
    """Check if a colormap is greyscale (gray, grey, Greys, etc.)."""
    return cmap_name.lower().replace('_', '') in (
        'gray', 'grey', 'greys', 'grays', 'gist_yarg', 'binary',
    )


_cmap_tree_cache = {}


def _get_cmap_tree(cmap_name: str):
    """Build and cache a KDTree + scalars for a colormap's 256 entries.

    plt.imsave() maps data through 256 colormap entries, so PNG pixels
    only ever contain those 256 RGB values. We build a small KDTree
    (256 points) once, then use it to map every pixel back to its
    scalar value via batch query.
    """
    if cmap_name in _cmap_tree_cache:
        return _cmap_tree_cache[cmap_name]

    from scipy.spatial import cKDTree

    cmap = plt.get_cmap(cmap_name)
    scalars = np.linspace(0, 1, 256)
    lut_rgb = (cmap(scalars)[:, :3] * 255).astype(np.float64)
    tree = cKDTree(lut_rgb)

    _cmap_tree_cache[cmap_name] = (tree, scalars)
    return tree, scalars


def invert_colormap(rgba: np.ndarray, cmap_name: str = 'viridis') -> np.ndarray:
    """Map RGBA pixel values back to scalar [0, 1] by inverting a colormap.

    For greyscale colormaps, uses direct luminance (instant).
    For colour colormaps, queries a cached KDTree of 256 colormap
    entries — fast batch lookup for all pixels at once.

    Args:
        rgba:     (H, W, 3 or 4) uint8 image array.
        cmap_name: Matplotlib colormap name used when saving.

    Returns:
        (H, W) float64 array with values in [0, 1].
    """
    # Fast path for greyscale colormaps
    if _is_greyscale_cmap(cmap_name):
        grey = np.mean(rgba[:, :, :3].astype(np.float64), axis=2) / 255.0
        cmap = plt.get_cmap(cmap_name)
        if cmap(0.0)[0] > cmap(1.0)[0]:  # inverted
            grey = 1.0 - grey
        return grey

    # Colour colormaps: batch KDTree query (256-point tree, very fast)
    tree, scalars = _get_cmap_tree(cmap_name)
    h, w = rgba.shape[:2]
    rgb = rgba[:, :, :3].astype(np.float64).reshape(-1, 3)
    _, indices = tree.query(rgb)
    return scalars[indices].reshape(h, w)


def png_to_db(png_path: str, vmin: float, vmax: float,
              cmap_name: str = 'viridis') -> np.ndarray:
    """Convert a single PNG file to a dB-scaled numpy array.

    Args:
        png_path:  Path to the PNG image.
        vmin:      Minimum dB value (maps to scalar 0).
        vmax:      Maximum dB value (maps to scalar 1).
        cmap_name: Colormap used when saving the PNG.

    Returns:
        (H, W) float32 array in dB scale [vmin, vmax].
    """
    img = np.array(Image.open(png_path))
    scalar = invert_colormap(img, cmap_name)
    db_array = scalar * (vmax - vmin) + vmin
    return db_array.astype(np.float32)


def convert_dataset(
    input_dir: str,
    output_dir: str,
    vmin: float = None,
    vmax: float = None,
    cmap_name: str = None,
    angles_deg: list = None,
    specimen_thickness_m: float = None,
    specimen_width_m: float = None,
) -> str:
    """Convert all TFM PNGs in a directory to .npy files for reconstruction.

    Reads Params.txt for vmin/vmax/cmap if not provided explicitly.
    Generates bscan_XXXX.npy files and scan_meta.npy.

    Args:
        input_dir:            Directory containing TFM PNGs and Params.txt.
        output_dir:           Where to save .npy files.
        vmin:                 Override dB minimum (default: from Params.txt).
        vmax:                 Override dB maximum (default: from Params.txt).
        cmap_name:            Override colormap (default: from Params.txt).
        angles_deg:           List of rotation angles in degrees for each PNG.
                              If None, assumes uniform spacing over [0, 180).
        specimen_thickness_m: Specimen thickness in metres (from Params.txt z range).
        specimen_width_m:     Specimen width in metres (from Params.txt x range).

    Returns:
        Path to output directory.
    """
    # Read Params.txt
    params_path = os.path.join(input_dir, 'Params.txt')
    params = {}
    if os.path.exists(params_path):
        params = parse_params(params_path)
        print(f"Read parameters from {params_path}")
        for k, v in params.items():
            print(f"  {k} = {v}")
    else:
        print(f"Warning: No Params.txt found in {input_dir}")

    # Resolve dB range
    if vmin is None:
        vmin = params.get('vmin', -20.0)
    if vmax is None:
        vmax = params.get('vmax', 0.0)
    if cmap_name is None:
        cmap_name = params.get('cmap', 'viridis')

    print(f"\nConversion settings: vmin={vmin} dB, vmax={vmax} dB, cmap={cmap_name}")

    # Find PNG files (natural sort so scan_2 comes before scan_10)
    def _natural_key(s):
        return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]

    png_files = sorted(
        (f for f in os.listdir(input_dir) if f.lower().endswith('.png')),
        key=_natural_key,
    )
    if not png_files:
        raise FileNotFoundError(f"No PNG files found in {input_dir}")

    print(f"Found {len(png_files)} PNG files")

    # Convert each PNG
    os.makedirs(output_dir, exist_ok=True)
    bscans = []
    for i, png_file in enumerate(png_files):
        png_path = os.path.join(input_dir, png_file)
        print(f"  [{i+1}/{len(png_files)}] {png_file}", end='')

        db_array = png_to_db(png_path, vmin, vmax, cmap_name)
        bscans.append(db_array)

        npy_name = f'bscan_{i:04d}.npy'
        np.save(os.path.join(output_dir, npy_name), db_array)
        print(f" → {npy_name}  shape={db_array.shape}")

    # Build scan_meta.npy
    n_scans = len(bscans)

    if angles_deg is not None:
        angles_rad = np.deg2rad(angles_deg)
    else:
        # Uniform angular spacing over [0, 180)
        angles_rad = np.linspace(0, np.pi, n_scans, endpoint=False)

    # Infer geometry from Params.txt
    z_min_m = params.get('z_max', 0.0)      # note: z_max in Params.txt is the top
    z_max_m = params.get('z_min', 40e-3)     # z_min in Params.txt is the bottom
    if isinstance(z_min_m, str):
        z_min_m = 0.0
    if isinstance(z_max_m, str):
        z_max_m = 40e-3

    if specimen_thickness_m is None:
        specimen_thickness_m = float(z_max_m)
    if specimen_width_m is None:
        specimen_width_m = specimen_thickness_m  # rough default

    tfm_n_pixels = bscans[0].shape[0]

    meta = {
        'n_scans': n_scans,
        'angles_rad': angles_rad.astype(np.float64),
        'specimen_thickness_m': float(specimen_thickness_m),
        'specimen_width_m': float(specimen_width_m),
        'tfm_z_start_m': float(z_min_m),
        'tfm_z_end_m': float(z_max_m),
        'array_aperture_m': float(specimen_width_m * 0.7),
        'tfm_n_pixels': int(tfm_n_pixels),
        # Extra info for provenance
        'source_dir': os.path.abspath(input_dir),
        'source_files': png_files,
        'vmin_db': float(vmin),
        'vmax_db': float(vmax),
        'cmap': cmap_name,
    }

    meta_path = os.path.join(output_dir, 'scan_meta.npy')
    np.save(meta_path, meta)

    print(f"\nSaved {n_scans} B-scans + scan_meta.npy to {output_dir}")
    print(f"  Shape per frame: {bscans[0].shape}")
    print(f"  dB range: [{vmin}, {vmax}]")
    print(f"  Angles: {np.rad2deg(angles_rad[0]):.1f}° to {np.rad2deg(angles_rad[-1]):.1f}°")

    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description='Convert TFM PNG images to .npy for 3D reconstruction'
    )
    parser.add_argument('--input', '-i', required=True,
                        help='Input directory containing TFM PNGs and Params.txt')
    parser.add_argument('--output', '-o', required=True,
                        help='Output directory for .npy files')
    parser.add_argument('--vmin', type=float, default=None,
                        help='Override dB minimum (default: from Params.txt)')
    parser.add_argument('--vmax', type=float, default=None,
                        help='Override dB maximum (default: from Params.txt)')
    parser.add_argument('--cmap', type=str, default=None,
                        help='Override colormap name (default: from Params.txt)')
    parser.add_argument('--thickness', type=float, default=None,
                        help='Specimen thickness in metres')
    parser.add_argument('--width', type=float, default=None,
                        help='Specimen width in metres')

    args = parser.parse_args()

    convert_dataset(
        input_dir=args.input,
        output_dir=args.output,
        vmin=args.vmin,
        vmax=args.vmax,
        cmap_name=args.cmap,
        specimen_thickness_m=args.thickness,
        specimen_width_m=args.width,
    )


if __name__ == '__main__':
    main()
