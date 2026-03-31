"""
Reconstruct experimental data using inverse Radon transform.

Usage:
    python reconstruct_experimental.py "DATA/1D NPY Data/Al Hole 5MHz 02022026/"
    python reconstruct_experimental.py "DATA/1D NPY Data/Al Pure 10MHz 18032026 Vol1/" --napari
    python reconstruct_experimental.py "DATA/1D NPY Data/Al Pure 10MHz 18032026 Vol2/" --filter ramp --crop
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from Classes.Reconstruct3D import reconstruct_scan


def main():
    parser = argparse.ArgumentParser(
        description='Reconstruct 3D volume from experimental B-scan data '
                    'using inverse Radon transform (filtered back-projection).')
    parser.add_argument(
        'scan_dir',
        help='Directory containing bscan_*.npy and scan_meta.npy')
    parser.add_argument(
        '--filter', dest='filter_name', default='hann',
        choices=['hann', 'ramp', 'shepp-logan', 'hamming'],
        help='FBP reconstruction filter (default: hann)')
    parser.add_argument(
        '--output-size', type=int, default=None,
        help='Output grid size per slice (default: same as lateral pixels). '
             'Use to downsample large datasets.')
    parser.add_argument(
        '--crop', action='store_true',
        help='Crop cylindrical volume to inscribed cube')
    parser.add_argument(
        '--napari', action='store_true',
        help='Open interactive napari 3D viewer')
    parser.add_argument(
        '--no-figures', action='store_true',
        help='Skip saving reconstruction summary PNG')
    parser.add_argument(
        '--save-dir', default=None,
        help='Output directory (default: same as scan_dir)')

    args = parser.parse_args()

    scan_dir = os.path.abspath(args.scan_dir)
    if not os.path.isdir(scan_dir):
        print(f"Error: directory not found: {scan_dir}")
        sys.exit(1)

    print(f"Scan directory: {scan_dir}")

    volume = reconstruct_scan(
        scan_dir=scan_dir,
        filter_name=args.filter_name,
        output_size=args.output_size,
        crop_to_cube=args.crop,
        show_napari=args.napari,
        save_figures=not args.no_figures,
        output_dir=args.save_dir,
    )

    print(f"\nDone. Volume shape: {volume.shape}, "
          f"size: {volume.nbytes / (1024**2):.1f} MB")


if __name__ == '__main__':
    main()
