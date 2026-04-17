#%%
from pathlib import Path
import numpy as np
import napari

from combine_rotations import RotationCombiner
from stitch_variable_simple import VariableStitcher


# ================================
# EXECUTION SETTINGS
# ================================
ROOT = Path.cwd().parent.parent 
INPUT_DIR = ROOT / "DATA" / "2D TFM Data" / "Cu Pure 7.5MHz Ex 15042026 Filtered"
FUSED_DIR = ROOT / "PROCESSING" / "Rotation NPYs"

POSITIONS = None              # None or [1,2,3,4]

# Combining
METHOD = "max"           # "mean", "median", "gradient"
CROP = 40
GRADIENT_SIGMA = 1.0
SAVE_MIP = True

# Stitching
DO_STITCH_TEST = True
STITCH_TEST_PAIR = (3, 4)

DO_FIND_ALL_SHIFTS = True
STITCH_AXIS = "x"
MAX_SHIFT = 200

ADAPTIVE_GRID = True
FIXED_GRID = (4, 4)
GRID_BINARY_THRESHOLD = 0.93
CORR_BINARY_THRESHOLD = 0.93
IGNORE_TOP = 0
MIN_VOXELS = 50
TILE_MULTIPLE = (1.5, 1.5)
MIN_GRID = (4, 4)
MAX_GRID = (45, 45)
OPENING_STRUCTURE = np.ones((3, 3, 3), dtype=bool)
SIZE_STATISTIC = "median"

VIEW_STITCH_RESULT = True

X_PIXEL_SIZE = 0.039e-3
Y_PIXEL_SIZE = 0.040e-3


# ================================
# RUN COMBINING
# ================================
combiner = RotationCombiner(
    input_dir=INPUT_DIR,
    output_dir=FUSED_DIR,
    method=METHOD,
    crop=CROP,
    gradient_sigma=GRADIENT_SIGMA,
    save_mip=SAVE_MIP,
)

fused_paths = combiner.combine_all(positions=POSITIONS)


# ================================
# MAKE STITCHER
# ================================
stitcher = VariableStitcher(
    axis=STITCH_AXIS,
    max_shift=MAX_SHIFT,
    grid=FIXED_GRID,
    adaptive_grid=ADAPTIVE_GRID,
    grid_binary_threshold=GRID_BINARY_THRESHOLD,
    corr_binary_threshold=CORR_BINARY_THRESHOLD,
    ignore_top=IGNORE_TOP,
    min_voxels=MIN_VOXELS,
    tile_multiple=TILE_MULTIPLE,
    min_grid=MIN_GRID,
    max_grid=MAX_GRID,
    opening_structure=OPENING_STRUCTURE,
    size_statistic=SIZE_STATISTIC,
)


# ================================
# OPTIONAL STITCH TEST
# ================================
if DO_STITCH_TEST:
    pa, pb = STITCH_TEST_PAIR

    vol_a = np.load(fused_paths[pa]).astype(np.float32)
    vol_b = np.load(fused_paths[pb]).astype(np.float32)

    result = stitcher.stitch(vol_a, vol_b)

    pixel_size = X_PIXEL_SIZE if STITCH_AXIS == "x" else Y_PIXEL_SIZE
    shift = result["best_shift"]

    print(f"\nStitch test: positions {pa} to {pb}")
    print(f"Stitch axis: {STITCH_AXIS}")
    print(f"Pixel shift: {shift}")
    print(f"Distance: {shift * pixel_size * 1000:.3f} mm")
    print(f"Absolute distance: {abs(shift * pixel_size * 1000):.3f} mm")

    stitcher.print_summary(result)
    stitcher.plot_correlation(result)
    stitcher.plot_vote_map(result)

    if result["diagnostics"]["grid_info"] is not None:
        stitcher.plot_binary_projections(result)

    if VIEW_STITCH_RESULT:
        viewer = napari.Viewer()
        viewer.add_image(
            stitcher.normalise_for_display(result["canvas1"]),
            name=f"Position {pa}",
            colormap="magenta",
            blending="additive",
        )
        viewer.add_image(
            stitcher.normalise_for_display(result["canvas2"]),
            name=f"Position {pb}",
            colormap="cyan",
            blending="additive",
        )
        napari.run()


# ================================
# FIND SHIFTS BETWEEN ALL POSITIONS
# ================================
if DO_FIND_ALL_SHIFTS:
    positions_sorted = sorted(fused_paths.keys())
    pixel_size = X_PIXEL_SIZE if STITCH_AXIS == "x" else Y_PIXEL_SIZE

    print()
    for i in range(len(positions_sorted) - 1):
        pa = positions_sorted[i]
        pb = positions_sorted[i + 1]

        vol_a = np.load(fused_paths[pa]).astype(np.float32)
        vol_b = np.load(fused_paths[pb]).astype(np.float32)

        result = stitcher.stitch(vol_a, vol_b)
        shift = result["best_shift"]

        print(f"Positions {pa} to {pb}")
        print(f"Stitch Axis: {STITCH_AXIS}")
        print(f"Pixel Shift: {shift} pixels")
        print(f"Distance Calculated: {shift * pixel_size * 1000:.3f} mm")
        print(f"Absolute Distance: {abs(shift * pixel_size * 1000):.3f} mm")
        print()