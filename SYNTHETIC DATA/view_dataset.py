"""
Dataset Viewer — visualise generated volumes in napari
======================================================

Loads an existing dataset directory and opens napari with all
position volumes at their correct spatial offsets.

Usage:
    python view_dataset.py                          # latest dataset
    python view_dataset.py output/datasets/dataset_20260317  # specific dataset
    python view_dataset.py output/datasets/dataset_20260317 --layer ground_truth
    python view_dataset.py output/datasets/dataset_20260317 --layer overlay
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from generate_dataset import view_dataset_napari


def find_latest_dataset(output_dir: str) -> str:
    """Find the most recent dataset_* directory."""
    if not os.path.isdir(output_dir):
        raise FileNotFoundError(f"Output directory not found: {output_dir}")
    dirs = sorted(
        d for d in os.listdir(output_dir)
        if d.startswith('dataset_') and os.path.isdir(os.path.join(output_dir, d))
    )
    if not dirs:
        raise FileNotFoundError(f"No dataset_* directories in {output_dir}")
    return os.path.join(output_dir, dirs[-1])


def main():
    layer = 'both'

    # Parse args
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    flags = [a for a in sys.argv[1:] if a.startswith('--')]

    for flag in flags:
        if flag.startswith('--layer='):
            layer = flag.split('=', 1)[1]
        elif flag == '--layer' and flags.index(flag) + 1 < len(sys.argv):
            # handle --layer ground_truth
            pass

    # Handle --layer as two separate args
    for i, a in enumerate(sys.argv[1:], 1):
        if a == '--layer' and i < len(sys.argv) - 1:
            layer = sys.argv[i + 1]

    if args:
        dataset_dir = args[0]
    else:
        output_dir = os.path.join(os.path.dirname(__file__), 'output')
        dataset_dir = find_latest_dataset(output_dir)

    # Also check if it's a scan_3d single-scan directory (not a dataset)
    if os.path.exists(os.path.join(dataset_dir, 'scan_meta.npy')) and \
       not os.path.exists(os.path.join(dataset_dir, 'dataset_meta.json')):
        # Single scan — use reconstruct_3d viewer instead
        from reconstruct_3d import reconstruct_and_compare
        print(f"Single scan directory detected: {dataset_dir}")
        reconstruct_and_compare(
            scan_dir=dataset_dir, show_napari=True, save_figures=False,
        )
        return

    print(f"Loading dataset: {dataset_dir}")
    print(f"Layer mode: {layer}")
    view_dataset_napari(dataset_dir, layer=layer)


if __name__ == '__main__':
    main()
