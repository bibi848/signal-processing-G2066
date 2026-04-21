# %%

from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
import napari
from scipy.ndimage import binary_opening, label, find_objects


THIS_FILE = Path(__file__).resolve()
THIS_DIR = THIS_FILE.parent
REPO_ROOT = THIS_DIR.parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Classes.Stitch3D import stitch_volumes

# ============================================================
# SETTINGS
# ============================================================
STITCH_AXIS = "x"
ROTATION = 0
SHOW_NAPARI = False
MAX_SHIFT = 180

data_dir = Path.cwd().parent.parent / "SYNTHETIC DATA" / "ovlp_080" / "pair_00"

# Input file order:
# True  -> input files are (z, y, x), convert to (z, x, y)
# False -> input files already are (z, x, y)
TRANSPOSE_INPUT_TO_ZXY = True

# Optional features
USE_IGNORE_Z = True
IGNORE_TOP = 15
IGNORE_BOTTOM = 10 

USE_CORR_BINARY_MASK = True
CORR_BINARY_THRESHOLD = 0.93

USE_TILED_CORRELATION = True
GRID = (15, 15)

USE_ADAPTIVE_GRID = True
GRID_BINARY_THRESHOLD = CORR_BINARY_THRESHOLD
MIN_VOXELS = 25
TILE_MULTIPLE = (1.5, 1.5)
MIN_GRID = (10, 10)
MAX_GRID = (50, 50)
OPENING_STRUCTURE = np.ones((3, 3, 3), dtype=bool)
SIZE_STATISTIC = "median"


# ============================================================
# HELPERS
# ============================================================
def validate_threshold(value: float, name: str) -> None:
    if not (0 <= value <= 1):
        raise ValueError(f"{name} must be between 0 and 1")


def get_z_bounds(z_dim: int, ignore_top: int = 0, ignore_bottom: int = 0) -> tuple[int, int]:
    z_start = min(ignore_top, z_dim)
    z_end = max(z_start, z_dim - min(ignore_bottom, z_dim - z_start))
    return z_start, z_end


def make_binary_mask(
    vol: np.ndarray,
    binary_threshold: float | None,
    opening_structure=None,
) -> np.ndarray:
    if binary_threshold is None:
        return np.ones_like(vol, dtype=bool)

    validate_threshold(binary_threshold, "binary_threshold")

    v_abs = np.abs(vol)
    v_max = np.max(v_abs)

    if v_max == 0:
        return np.zeros_like(vol, dtype=bool)

    v_norm = v_abs / v_max
    mask = v_norm >= binary_threshold

    if opening_structure is not None:
        mask = binary_opening(mask, structure=opening_structure)

    return mask


def measure_component_sizes_3d(
    binary_vol: np.ndarray,
    ignore_top: int = 0,
    ignore_bottom: int = 0,
    min_voxels: int = 10,
) -> list[dict]:
    z_dim = binary_vol.shape[0]
    z_start, z_end = get_z_bounds(z_dim, ignore_top=ignore_top, ignore_bottom=ignore_bottom)
    binary_crop = binary_vol[z_start:z_end, :, :]

    labeled, _ = label(binary_crop)
    slices = find_objects(labeled)

    component_sizes = []

    for comp_idx, slc in enumerate(slices, start=1):
        if slc is None:
            continue

        z_slc, x_slc, y_slc = slc
        component_mask = labeled[slc] == comp_idx
        voxels = int(np.count_nonzero(component_mask))

        if voxels < min_voxels:
            continue

        component_sizes.append(
            {
                "label": comp_idx,
                "z_size": int(z_slc.stop - z_slc.start),
                "x_size": int(x_slc.stop - x_slc.start),
                "y_size": int(y_slc.stop - y_slc.start),
                "voxels": voxels,
            }
        )

    return component_sizes


def estimate_adaptive_grid(
    vol1: np.ndarray,
    vol2: np.ndarray,
    axis: str,
    grid_binary_threshold: float,
    ignore_top: int,
    ignore_bottom: int,
    min_voxels: int,
    tile_multiple: tuple[float, float],
    min_grid: tuple[int, int],
    max_grid: tuple[int, int],
    opening_structure,
    size_statistic: str = "median",
) -> tuple[tuple[int, int], dict]:
    mask1 = make_binary_mask(vol1, grid_binary_threshold, opening_structure=opening_structure)
    mask2 = make_binary_mask(vol2, grid_binary_threshold, opening_structure=opening_structure)

    sizes1 = measure_component_sizes_3d(
        mask1,
        ignore_top=ignore_top,
        ignore_bottom=ignore_bottom,
        min_voxels=min_voxels,
    )
    sizes2 = measure_component_sizes_3d(
        mask2,
        ignore_top=ignore_top,
        ignore_bottom=ignore_bottom,
        min_voxels=min_voxels,
    )
    all_sizes = sizes1 + sizes2

    if not all_sizes:
        raise ValueError(
            "No binary components found for adaptive grid sizing. "
            "Try lowering GRID_BINARY_THRESHOLD or MIN_VOXELS."
        )

    z_start, z_end = get_z_bounds(vol1.shape[0], ignore_top=ignore_top, ignore_bottom=ignore_bottom)

    if axis == "x":
        size0_vals = np.array([s["z_size"] for s in all_sizes], dtype=float)
        size1_vals = np.array([s["y_size"] for s in all_sizes], dtype=float)
        plane_dim_0 = z_end - z_start
        plane_dim_1 = vol1.shape[2]
    else:
        size0_vals = np.array([s["z_size"] for s in all_sizes], dtype=float)
        size1_vals = np.array([s["x_size"] for s in all_sizes], dtype=float)
        plane_dim_0 = z_end - z_start
        plane_dim_1 = vol1.shape[1]

    if size_statistic == "mean":
        rep0 = float(np.mean(size0_vals))
        rep1 = float(np.mean(size1_vals))
    else:
        rep0 = float(np.median(size0_vals))
        rep1 = float(np.median(size1_vals))

    tile0 = max(1, int(round(tile_multiple[0] * rep0)))
    tile1 = max(1, int(round(tile_multiple[1] * rep1)))

    raw_rows = max(1, plane_dim_0 // tile0)
    raw_cols = max(1, plane_dim_1 // tile1)

    rows = int(np.clip(raw_rows, min_grid[0], max_grid[0]))
    cols = int(np.clip(raw_cols, min_grid[1], max_grid[1]))

    info = {
        "grid": (rows, cols),
        "raw_grid": (raw_rows, raw_cols),
        "representative_size_axis0": rep0,
        "representative_size_axis1": rep1,
        "tile_size_axis0": tile0,
        "tile_size_axis1": tile1,
        "num_components_vol1": len(sizes1),
        "num_components_vol2": len(sizes2),
        "num_components_total": len(all_sizes),
        "mask1": mask1,
        "mask2": mask2,
    }

    return (rows, cols), info


def normalised_correlation_3d_basic(
    vol1: np.ndarray,
    vol2: np.ndarray,
    axis: str = "x",
    max_shift: int = 100,
    binary_threshold: float | None = None,
    ignore_top: int = 0,
    ignore_bottom: int = 0,
) -> tuple[int, np.ndarray, np.ndarray]:
    z1, x1, y1 = vol1.shape
    z2, x2, y2 = vol2.shape

    if axis not in ("x", "y"):
        raise ValueError("axis must be 'x' or 'y'")

    if binary_threshold is not None:
        validate_threshold(binary_threshold, "binary_threshold")

    z1_start, z1_end = get_z_bounds(z1, ignore_top=ignore_top, ignore_bottom=ignore_bottom)
    z2_start, z2_end = get_z_bounds(z2, ignore_top=ignore_top, ignore_bottom=ignore_bottom)

    vol1 = vol1[z1_start:z1_end]
    vol2 = vol2[z2_start:z2_end]

    z1, x1, y1 = vol1.shape
    z2, x2, y2 = vol2.shape

    v1_abs = np.abs(vol1)
    v2_abs = np.abs(vol2)

    v1_max = np.max(v1_abs)
    v2_max = np.max(v2_abs)

    if binary_threshold is not None:
        mask1_full = np.zeros_like(vol1, dtype=bool) if v1_max == 0 else (v1_abs / v1_max) >= binary_threshold
        mask2_full = np.zeros_like(vol2, dtype=bool) if v2_max == 0 else (v2_abs / v2_max) >= binary_threshold
    else:
        mask1_full = None
        mask2_full = None

    shifts = np.arange(-max_shift, max_shift + 1)
    corr_values = []

    for d in shifts:
        if axis == "x":
            a1_start = max(0, d)
            a1_end = min(x1, x2 + d)

            a2_start = max(0, -d)
            a2_end = min(x2, x1 - d)

            if (a1_end - a1_start) <= 0:
                corr_values.append(0.0)
                continue

            region1 = vol1[:, a1_start:a1_end, :]
            region2 = vol2[:, a2_start:a2_end, :]

            if binary_threshold is not None:
                mask1 = mask1_full[:, a1_start:a1_end, :]
                mask2 = mask2_full[:, a2_start:a2_end, :]
        else:
            a1_start = max(0, d)
            a1_end = min(y1, y2 + d)

            a2_start = max(0, -d)
            a2_end = min(y2, y1 - d)

            if (a1_end - a1_start) <= 0:
                corr_values.append(0.0)
                continue

            region1 = vol1[:, :, a1_start:a1_end]
            region2 = vol2[:, :, a2_start:a2_end]

            if binary_threshold is not None:
                mask1 = mask1_full[:, :, a1_start:a1_end]
                mask2 = mask2_full[:, :, a2_start:a2_end]

        if binary_threshold is not None:
            joint_mask = mask1 & mask2

            if not np.any(joint_mask):
                corr_values.append(0.0)
                continue

            r1 = region1[joint_mask]
            r2 = region2[joint_mask]
        else:
            r1 = region1.ravel()
            r2 = region2.ravel()

        numerator = np.sum(r1 * r2)
        denom = np.sqrt(np.sum(r1 ** 2) * np.sum(r2 ** 2))
        corr_values.append(numerator / denom if denom > 0 else 0.0)

    corr_values = np.array(corr_values, dtype=float)
    best_index = int(np.argmax(corr_values))
    best_shift = int(shifts[best_index])

    return best_shift, shifts, corr_values


def normalised_correlation_3d_tiled(
    vol1: np.ndarray,
    vol2: np.ndarray,
    axis: str = "x",
    max_shift: int = 100,
    grid: tuple[int, int] = (4, 4),
    binary_threshold: float | None = None,
    ignore_top: int = 0,
    ignore_bottom: int = 0,
) -> tuple[int, np.ndarray, np.ndarray, dict]:
    z1, x1, y1 = vol1.shape
    z2, x2, y2 = vol2.shape

    rows, cols = grid
    shifts = np.arange(-max_shift, max_shift + 1)
    corr_values = np.zeros_like(shifts, dtype=float)

    tile_vote_map = np.full((rows, cols), np.nan, dtype=float)
    tile_peak_map = np.full((rows, cols), np.nan, dtype=float)
    valid_tile_count = 0

    z_start, z_end = get_z_bounds(z1, ignore_top=ignore_top, ignore_bottom=ignore_bottom)
    z_usable = z_end - z_start

    if z_usable <= 0:
        raise ValueError("No usable z slices remain after applying ignore_top and ignore_bottom")

    if axis == "x":
        if z1 != z2 or y1 != y2:
            raise ValueError("For x stitching, z and y dimensions must match")

        tile_z = z_usable // rows
        tile_y = y1 // cols

        if tile_z <= 0 or tile_y <= 0:
            raise ValueError(f"Grid {grid} is too fine for volume shape {vol1.shape}")

        for r in range(rows):
            for c in range(cols):
                zs = z_start + r * tile_z
                ze = z_start + (r + 1) * tile_z if r < rows - 1 else z_end

                ys = c * tile_y
                ye = (c + 1) * tile_y if c < cols - 1 else y1

                tile1 = vol1[zs:ze, :, ys:ye]
                tile2 = vol2[zs:ze, :, ys:ye]

                if tile1.size == 0 or tile2.size == 0:
                    continue

                _, _, tile_corr = normalised_correlation_3d_basic(
                    tile1,
                    tile2,
                    axis=axis,
                    max_shift=max_shift,
                    binary_threshold=binary_threshold,
                    ignore_top=0,
                    ignore_bottom=0,
                )

                corr_values += tile_corr
                peak_idx = int(np.argmax(tile_corr))
                tile_vote_map[r, c] = shifts[peak_idx]
                tile_peak_map[r, c] = tile_corr[peak_idx]
                valid_tile_count += 1

    else:
        if z1 != z2 or x1 != x2:
            raise ValueError("For y stitching, z and x dimensions must match")

        tile_z = z_usable // rows
        tile_x = x1 // cols

        if tile_z <= 0 or tile_x <= 0:
            raise ValueError(f"Grid {grid} is too fine for volume shape {vol1.shape}")

        for r in range(rows):
            for c in range(cols):
                zs = z_start + r * tile_z
                ze = z_start + (r + 1) * tile_z if r < rows - 1 else z_end

                xs = c * tile_x
                xe = (c + 1) * tile_x if c < cols - 1 else x1

                tile1 = vol1[zs:ze, xs:xe, :]
                tile2 = vol2[zs:ze, xs:xe, :]

                if tile1.size == 0 or tile2.size == 0:
                    continue

                    _, _, tile_corr = normalised_correlation_3d_basic(
                    tile1,
                    tile2,
                    axis=axis,
                    max_shift=max_shift,
                    binary_threshold=binary_threshold,
                    ignore_top=0,
                    ignore_bottom=0,
                )

                corr_values += tile_corr
                peak_idx = int(np.argmax(tile_corr))
                tile_vote_map[r, c] = shifts[peak_idx]
                tile_peak_map[r, c] = tile_corr[peak_idx]
                valid_tile_count += 1

    if valid_tile_count == 0:
        raise ValueError("No valid tiles were found")

    best_index = int(np.argmax(corr_values))
    best_shift = int(shifts[best_index])

    diagnostics = {
        "grid": grid,
        "tile_vote_map": tile_vote_map,
        "tile_peak_map": tile_peak_map,
        "valid_tile_count": valid_tile_count,
    }

    return best_shift, shifts, corr_values, diagnostics


def run_correlation(
    vol1: np.ndarray,
    vol2: np.ndarray,
    *,
    axis: str,
    max_shift: int,
    use_tiled: bool,
    grid: tuple[int, int],
    use_adaptive_grid: bool,
    grid_binary_threshold: float,
    corr_binary_threshold: float | None,
    use_ignore_z: bool,
    ignore_top: int,
    ignore_bottom: int,
    min_voxels: int,
    tile_multiple: tuple[float, float],
    min_grid: tuple[int, int],
    max_grid: tuple[int, int],
    opening_structure,
    size_statistic: str,
):
    actual_ignore_top = ignore_top if use_ignore_z else 0
    actual_ignore_bottom = ignore_bottom if use_ignore_z else 0
    actual_corr_threshold = corr_binary_threshold if USE_CORR_BINARY_MASK else None

    diagnostics = {"mode": "simple", "grid_info": None}

    if use_tiled:
        actual_grid = grid

        if use_adaptive_grid:
            actual_grid, grid_info = estimate_adaptive_grid(
                vol1,
                vol2,
                axis=axis,
                grid_binary_threshold=grid_binary_threshold,
                ignore_top=actual_ignore_top,
                ignore_bottom=actual_ignore_bottom,
                min_voxels=min_voxels,
                tile_multiple=tile_multiple,
                min_grid=min_grid,
                max_grid=max_grid,
                opening_structure=opening_structure,
                size_statistic=size_statistic,
            )
            diagnostics["grid_info"] = grid_info
        else:
            diagnostics["grid_info"] = None

        best_shift, shifts, corr_values, tiled_diag = normalised_correlation_3d_tiled(
            vol1,
            vol2,
            axis=axis,
            max_shift=max_shift,
            grid=actual_grid,
            binary_threshold=actual_corr_threshold,
            ignore_top=actual_ignore_top,
            ignore_bottom=actual_ignore_bottom,
        )
        diagnostics.update(tiled_diag)
        diagnostics["mode"] = "tiled"

    else:
        best_shift, shifts, corr_values = normalised_correlation_3d_basic(
            vol1,
            vol2,
            axis=axis,
            max_shift=max_shift,
            binary_threshold=actual_corr_threshold,
            ignore_top=actual_ignore_top,
            ignore_bottom=actual_ignore_bottom,
        )
        diagnostics["grid"] = None
        diagnostics["tile_vote_map"] = None
        diagnostics["tile_peak_map"] = None
        diagnostics["valid_tile_count"] = None

    return best_shift, shifts, corr_values, diagnostics


# ============================================================
# LOAD DATA
# ============================================================
data_dir = Path.cwd().parent.parent / "PROCESSING" / "Rotation NPYs" / "pair_00"

vol_a_path = data_dir / f"volume_A_r{ROTATION}.npy"
vol_b_path = data_dir / f"volume_B_r{ROTATION}.npy"

if not vol_a_path.exists():
    raise FileNotFoundError(f"Missing file: {vol_a_path}")

if not vol_b_path.exists():
    raise FileNotFoundError(f"Missing file: {vol_b_path}")

print(f"Processing overlap folder {data_dir.name}")
print(f"Loading:\n  {vol_a_path}\n  {vol_b_path}")

vol_a_signal = np.load(vol_a_path).astype(np.float32)
vol_b_signal = np.load(vol_b_path).astype(np.float32)

if TRANSPOSE_INPUT_TO_ZXY:
    vol_a = np.transpose(vol_a_signal, (0, 2, 1))
    vol_b = np.transpose(vol_b_signal, (0, 2, 1))
else:
    vol_a = vol_a_signal
    vol_b = vol_b_signal


# ============================================================
# EXPECTED SHIFT
# ============================================================
parent_name = data_dir.parent.name  # e.g. ovlp_090
overlap_fraction = float(parent_name.split("_")[-1]) / 100.0
_, x_size, _ = vol_a.shape
expected_shift_pixels = (1 - overlap_fraction) * x_size  
print(f"Expected shift {expected_shift_pixels:.1f} pixels")


# ============================================================
# RUN STITCHING
# ============================================================
best_shift, shifts, corr_values, diagnostics = run_correlation(
    vol_a,
    vol_b,
    axis=STITCH_AXIS,
    max_shift=MAX_SHIFT,
    use_tiled=USE_TILED_CORRELATION,
    grid=GRID,
    use_adaptive_grid=USE_ADAPTIVE_GRID,
    grid_binary_threshold=GRID_BINARY_THRESHOLD,
    corr_binary_threshold=CORR_BINARY_THRESHOLD,
    use_ignore_z=USE_IGNORE_Z,
    ignore_top=IGNORE_TOP,
    ignore_bottom=IGNORE_BOTTOM,
    min_voxels=MIN_VOXELS,
    tile_multiple=TILE_MULTIPLE,
    min_grid=MIN_GRID,
    max_grid=MAX_GRID,
    opening_structure=OPENING_STRUCTURE,
    size_statistic=SIZE_STATISTIC,
)

canvas_a, canvas_b = stitch_volumes(
    vol_a,
    vol_b,
    best_shift,
    axis=STITCH_AXIS,
)

print(f"Best shift {best_shift} pixels")
print(f"Mode: {diagnostics['mode']}")

if diagnostics["grid"] is not None:
    print(f"Grid: {diagnostics['grid']}")
    print(f"Valid tiles: {diagnostics['valid_tile_count']}")


# ============================================================
# PLOT CORRELATION
# ============================================================
plt.figure(figsize=(8, 4.5))
plt.plot(shifts, corr_values, linewidth=1.8)
plt.title(f"{data_dir.parent.name} / {data_dir.name}")
plt.xlabel("Pixel Shift")
plt.ylabel("Correlation")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

if diagnostics["tile_vote_map"] is not None:
    plt.figure(figsize=(6, 5))
    heat = np.ma.masked_invalid(diagnostics["tile_vote_map"])
    im = plt.imshow(heat, aspect="auto", interpolation="nearest")
    plt.colorbar(im, label="Tile best shift")
    plt.title("Tile Vote Map")
    plt.xlabel("Tile column")
    plt.ylabel("Tile row")
    plt.tight_layout()
    plt.show()


# ============================================================
# NAPARI VIEWS
# ============================================================
stitched = np.zeros_like(canvas_a, dtype=np.float32)

canvas_a_view = np.transpose(canvas_a, (0, 2, 1))
canvas_b_view = np.transpose(canvas_b, (0, 2, 1))
stitched_view = np.transpose(stitched, (0, 2, 1))

if SHOW_NAPARI:
    viewer = napari.Viewer()

    viewer.add_image(
        vol_a_signal,
        name="A original signal",
        colormap="magenta",
        blending="additive",
        opacity=0.75,
    )
    viewer.add_image(
        vol_b_signal,
        name="B original signal",
        colormap="cyan",
        blending="additive",
        opacity=0.75,
        visible=False,
    )
    viewer.add_image(
        canvas_a_view,
        name="A aligned canvas",
        colormap="magenta",
        blending="additive",
        opacity=0.6,
        visible=False,
    )
    viewer.add_image(
        canvas_b_view,
        name="B aligned canvas",
        colormap="cyan",
        blending="additive",
        opacity=0.6,
        visible=False,
    )
    viewer.add_image(
        stitched_view,
        name="stitched result",
        colormap="inferno",
    )

    viewer.dims.axis_labels = ("z", "y", "x")
    napari.run()