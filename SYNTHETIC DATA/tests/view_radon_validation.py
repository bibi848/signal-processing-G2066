"""
Load volume_N###.npy from test_radon_validation output and view in napari.

Each N becomes a layer; toggle visibility in napari to compare. Ground-truth
scatterer positions are added as a Points layer.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from test_radon_validation import (
    SCATTERERS, TFM_N_PIXELS, TFM_Z_START, TFM_Z_END,
    N_ELEMENTS, ELEMENT_PITCH,
)

OUTPUT_DIR = HERE / 'output' / 'radon_validation'


def main() -> None:
    import napari

    files = sorted(OUTPUT_DIR.glob('volume_N*.npy'))
    if not files:
        print(f"No volume_N*.npy in {OUTPUT_DIR}")
        return

    n = TFM_N_PIXELS
    half = (N_ELEMENTS - 1) * ELEMENT_PITCH / 2
    dz_mm  = (TFM_Z_END - TFM_Z_START) / max(n - 1, 1) * 1e3
    dxy_mm = (2 * half) / max(n - 1, 1) * 1e3
    z0_mm = TFM_Z_START * 1e3

    viewer = napari.Viewer(title='Radon validation volumes')

    for i, f in enumerate(files):
        vol = np.load(f).astype(np.float32)
        N = int(re.search(r'volume_N(\d+)', f.stem).group(1))
        viewer.add_image(
            vol, name=f'N={N}',
            scale=(dz_mm, dxy_mm, dxy_mm),
            translate=(z0_mm, -half * 1e3, -half * 1e3),
            colormap='hot', visible=(i == len(files) - 1),
        )

    pts = np.array([[sz * 1e3, sy * 1e3, sx * 1e3]
                    for sx, sy, sz, _ in SCATTERERS])
    viewer.add_points(
        pts, name='Ground truth', size=1.5,
        face_color='cyan', symbol='x',
    )

    viewer.dims.axis_labels = ('z (mm)', 'y (mm)', 'x (mm)')
    napari.run()


if __name__ == '__main__':
    main()
