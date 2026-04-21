# %%
from pathlib import Path
import sys
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_opening, label, find_objects
import csv
import json
from datetime import datetime


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
MAX_SHIFT = 180

ROOT_DATA_DIR = Path.cwd().parent.parent / "SYNTHETIC DATA"

# Optional: restrict which overlaps or pairs are used
ONLY_OVERLAPS = None          # e.g. ["ovlp_050", "ovlp_080"] or None
ONLY_PAIRS = None             # e.g. ["pair_00"] or None

# Input file order:
# True  -> input files are (z, y, x), convert to (z, x, y)
# False -> input files already are (z, x, y)
TRANSPOSE_INPUT_TO_ZXY = True

# Optional features
USE_IGNORE_Z = True
IGNORE_TOP = 15
IGNORE_BOTTOM = 10

USE_CORR_BINARY_MASK = False
CORR_BINARY_THRESHOLD = 0.93

USE_TILED_CORRELATION = True
GRID = (15, 15)

USE_ADAPTIVE_GRID = False
GRID_BINARY_THRESHOLD = CORR_BINARY_THRESHOLD
MIN_VOXELS = 25
TILE_MULTIPLE = (1.5, 1.5)
MIN_GRID = (10, 10)
MAX_GRID = (50, 50)
OPENING_STRUCTURE = np.ones((3, 3, 3), dtype=bool)
SIZE_STATISTIC = "median"

EXPORT_RESULTS = True
EXPORT_DIR = Path.cwd() / "correlation_exports"


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
    use_corr_binary_mask: bool,
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
    actual_corr_threshold = corr_binary_threshold if use_corr_binary_mask else None

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


def parse_overlap_from_folder(folder_name: str) -> float:
    match = re.fullmatch(r"ovlp_(\d+)", folder_name)
    if match is None:
        raise ValueError(f"Could not parse overlap from folder name: {folder_name}")
    return float(match.group(1)) / 100.0


def collect_cases(root_dir: Path):
    cases = []

    for ovlp_dir in sorted(root_dir.iterdir()):
        if not ovlp_dir.is_dir():
            continue
        if not ovlp_dir.name.startswith("ovlp_"):
            continue
        if ONLY_OVERLAPS is not None and ovlp_dir.name not in ONLY_OVERLAPS:
            continue

        for pair_dir in sorted(ovlp_dir.iterdir()):
            if not pair_dir.is_dir():
                continue
            if not pair_dir.name.startswith("pair_"):
                continue
            if ONLY_PAIRS is not None and pair_dir.name not in ONLY_PAIRS:
                continue

            vol_a_path = pair_dir / f"volume_B_r{ROTATION}.npy"
            vol_b_path = pair_dir / f"volume_A_r{ROTATION}.npy"

            if vol_a_path.exists() and vol_b_path.exists():
                cases.append(
                    {
                        "ovlp_dir": ovlp_dir,
                        "pair_dir": pair_dir,
                        "vol_a_path": vol_a_path,
                        "vol_b_path": vol_b_path,
                    }
                )

    return cases


def sanitise_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def build_run_settings_dict() -> dict:
    return {
        "STITCH_AXIS": STITCH_AXIS,
        "ROTATION": ROTATION,
        "MAX_SHIFT": MAX_SHIFT,
        "ROOT_DATA_DIR": str(ROOT_DATA_DIR),
        "ONLY_OVERLAPS": ONLY_OVERLAPS,
        "ONLY_PAIRS": ONLY_PAIRS,
        "TRANSPOSE_INPUT_TO_ZXY": TRANSPOSE_INPUT_TO_ZXY,
        "USE_IGNORE_Z": USE_IGNORE_Z,
        "IGNORE_TOP": IGNORE_TOP,
        "IGNORE_BOTTOM": IGNORE_BOTTOM,
        "USE_CORR_BINARY_MASK": USE_CORR_BINARY_MASK,
        "CORR_BINARY_THRESHOLD": CORR_BINARY_THRESHOLD,
        "USE_TILED_CORRELATION": USE_TILED_CORRELATION,
        "GRID": GRID,
        "USE_ADAPTIVE_GRID": USE_ADAPTIVE_GRID,
        "GRID_BINARY_THRESHOLD": GRID_BINARY_THRESHOLD,
        "MIN_VOXELS": MIN_VOXELS,
        "TILE_MULTIPLE": TILE_MULTIPLE,
        "MIN_GRID": MIN_GRID,
        "MAX_GRID": MAX_GRID,
        "OPENING_STRUCTURE_SHAPE": None if OPENING_STRUCTURE is None else tuple(OPENING_STRUCTURE.shape),
        "SIZE_STATISTIC": SIZE_STATISTIC,
    }


def make_json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, tuple):
        return [make_json_safe(v) for v in value]
    if isinstance(value, list):
        return [make_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {k: make_json_safe(v) for k, v in value.items()}
    return value


def export_grouped_correlation_csvs(export_root: Path, grouped_curves: dict) -> None:
    for ovlp_name, payload in grouped_curves.items():
        shifts = payload["shifts"]
        pair_to_curve = payload["pairs"]

        pair_names = sorted(pair_to_curve.keys())
        out_path = export_root / f"{sanitise_name(ovlp_name)}_correlation_vs_shift.csv"

        with out_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["shift"] + pair_names)

            for i, s in enumerate(shifts):
                row = [int(s)]
                for pair_name in pair_names:
                    row.append(float(pair_to_curve[pair_name][i]))
                writer.writerow(row)


def export_grouped_metadata_json(export_root: Path, grouped_metadata: dict) -> None:
    for ovlp_name, payload in grouped_metadata.items():
        out_path = export_root / f"{sanitise_name(ovlp_name)}_metadata.json"
        serialisable = make_json_safe(payload)
        serialisable["exported_at"] = datetime.now().isoformat()

        with out_path.open("w") as f:
            json.dump(serialisable, f, indent=2)


def export_summary_csv(export_root: Path, results: list[dict]) -> None:
    if not results:
        return

    summary_path = export_root / "summary_results.csv"
    fieldnames = [
        "overlap_fraction",
        "overlap_percent",
        "pair_name",
        "expected_shift",
        "best_shift",
        "signed_error",
        "abs_error",
        "peak_corr",
        "status",
    ]

    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row.get(k) for k in fieldnames})


# ============================================================
# RUN ALL CASES
# ============================================================
cases = collect_cases(ROOT_DATA_DIR)

if not cases:
    raise FileNotFoundError("No matching ovlp_*/pair_* cases found.")

if EXPORT_RESULTS:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

results = []
grouped_curves = {}
grouped_metadata = {}

for case in cases:
    ovlp_dir = case["ovlp_dir"]
    pair_dir = case["pair_dir"]
    vol_a_path = case["vol_a_path"]
    vol_b_path = case["vol_b_path"]

    print(f"Processing {ovlp_dir.name} / {pair_dir.name}")

    vol_a_signal = np.load(vol_a_path).astype(np.float32)
    vol_b_signal = np.load(vol_b_path).astype(np.float32)

    if TRANSPOSE_INPUT_TO_ZXY:
        vol_a = np.transpose(vol_a_signal, (0, 2, 1))
        vol_b = np.transpose(vol_b_signal, (0, 2, 1))
    else:
        vol_a = vol_a_signal
        vol_b = vol_b_signal

    overlap_fraction = parse_overlap_from_folder(ovlp_dir.name)

    if STITCH_AXIS == "x":
        stitch_dim = vol_a.shape[1]
    else:
        stitch_dim = vol_a.shape[2]

    expected_shift_pixels = (1.0 - overlap_fraction) * stitch_dim

    try:
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
            use_corr_binary_mask=USE_CORR_BINARY_MASK,
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

        signed_error = best_shift - expected_shift_pixels
        abs_error = abs(signed_error)

        results.append(
            {
                "overlap_fraction": overlap_fraction,
                "overlap_percent": 100.0 * overlap_fraction,
                "pair_name": pair_dir.name,
                "expected_shift": expected_shift_pixels,
                "best_shift": float(best_shift),
                "signed_error": signed_error,
                "abs_error": abs_error,
                "peak_corr": float(np.max(corr_values)),
                "status": "ok",
            }
        )

        if EXPORT_RESULTS:
            ovlp_key = ovlp_dir.name
            pair_key = pair_dir.name

            if ovlp_key not in grouped_curves:
                grouped_curves[ovlp_key] = {
                    "shifts": shifts.copy(),
                    "pairs": {},
                }

            grouped_curves[ovlp_key]["pairs"][pair_key] = corr_values.copy()

            if ovlp_key not in grouped_metadata:
                grouped_metadata[ovlp_key] = {
                    "ovlp_dir_name": ovlp_dir.name,
                    "overlap_fraction": overlap_fraction,
                    "overlap_percent": 100.0 * overlap_fraction,
                    "stitch_dim": int(stitch_dim),
                    "run_settings": build_run_settings_dict(),
                    "pairs": {},
                }

            grouped_metadata[ovlp_key]["pairs"][pair_key] = {
                "pair_name": pair_dir.name,
                "vol_a_path": str(vol_a_path),
                "vol_b_path": str(vol_b_path),
                "expected_shift": float(expected_shift_pixels),
                "best_shift": float(best_shift),
                "signed_error": float(signed_error),
                "abs_error": float(abs_error),
                "peak_corr": float(np.max(corr_values)),
                "status": "ok",
                "diagnostics": make_json_safe(diagnostics),
            }

    except Exception as e:
        print(f"  Failed: {e}")
        results.append(
            {
                "overlap_fraction": overlap_fraction,
                "overlap_percent": 100.0 * overlap_fraction,
                "pair_name": pair_dir.name,
                "expected_shift": expected_shift_pixels,
                "best_shift": np.nan,
                "signed_error": np.nan,
                "abs_error": np.nan,
                "peak_corr": np.nan,
                "status": f"failed: {e}",
            }
        )

        if EXPORT_RESULTS:
            ovlp_key = ovlp_dir.name
            pair_key = pair_dir.name

            if ovlp_key not in grouped_metadata:
                grouped_metadata[ovlp_key] = {
                    "ovlp_dir_name": ovlp_dir.name,
                    "overlap_fraction": overlap_fraction,
                    "overlap_percent": 100.0 * overlap_fraction,
                    "stitch_dim": int(stitch_dim),
                    "run_settings": build_run_settings_dict(),
                    "pairs": {},
                }

            grouped_metadata[ovlp_key]["pairs"][pair_key] = {
                "pair_name": pair_dir.name,
                "vol_a_path": str(vol_a_path),
                "vol_b_path": str(vol_b_path),
                "expected_shift": float(expected_shift_pixels),
                "best_shift": None,
                "signed_error": None,
                "abs_error": None,
                "peak_corr": None,
                "status": f"failed: {e}",
                "diagnostics": None,
            }

if EXPORT_RESULTS:
    export_grouped_correlation_csvs(EXPORT_DIR, grouped_curves)
    export_grouped_metadata_json(EXPORT_DIR, grouped_metadata)
    export_summary_csv(EXPORT_DIR, results)


# ============================================================
# SUMMARISE
# ============================================================
successful = [r for r in results if r["status"] == "ok"]
if not successful:
    raise RuntimeError("All stitching runs failed.")

successful_sorted = sorted(successful, key=lambda r: (r["overlap_percent"], r["pair_name"]))

print("\nResults:")
print(
    f"{'Overlap %':>10}  {'Pair':>8}  {'Expected':>10}  {'Predicted':>10}  "
    f"{'Signed err':>11}  {'Abs err':>9}  {'Peak corr':>10}"
)
for r in successful_sorted:
    print(
        f"{r['overlap_percent']:10.1f}  {r['pair_name']:>8}  "
        f"{r['expected_shift']:10.2f}  {r['best_shift']:10.2f}  "
        f"{r['signed_error']:11.2f}  {r['abs_error']:9.2f}  {r['peak_corr']:10.4f}"
    )


# ============================================================
# PREPARE DATA FOR PLOTTING
# ============================================================
overlaps = np.array([r["overlap_percent"] for r in successful_sorted], dtype=float)
expected_shifts = np.array([r["expected_shift"] for r in successful_sorted], dtype=float)
predicted_shifts = np.array([r["best_shift"] for r in successful_sorted], dtype=float)
signed_errors = np.array([r["signed_error"] for r in successful_sorted], dtype=float)
abs_errors = np.rint(
    np.array([r["abs_error"] for r in successful_sorted], dtype=float)
).astype(int)

mask_0 = abs_errors == 0
mask_1 = abs_errors == 1
mask_2 = abs_errors == 2
mask_3 = abs_errors == 3
mask_4 = abs_errors == 4
mask_gt4 = abs_errors > 4

unique_overlaps = np.unique(overlaps)
mean_abs_error = np.array([np.mean(abs_errors[overlaps == ov]) for ov in unique_overlaps], dtype=float)
std_abs_error = np.array([np.std(abs_errors[overlaps == ov]) for ov in unique_overlaps], dtype=float)
mean_signed_error = np.array([np.mean(signed_errors[overlaps == ov]) for ov in unique_overlaps], dtype=float)


plt.figure(figsize=(8, 5))
plt.scatter(overlaps, abs_errors, alpha=0.2, label="_nolegend_")
plt.scatter(overlaps[mask_0], abs_errors[mask_0], label="0 voxel error")
plt.scatter(overlaps[mask_1], abs_errors[mask_1], label="1 voxel error")
plt.scatter(overlaps[mask_2], abs_errors[mask_2], label="2 voxel error")
plt.scatter(overlaps[mask_3], abs_errors[mask_3], label="3 voxel error")
plt.scatter(overlaps[mask_4], abs_errors[mask_4], label="4 voxel error")
plt.scatter(overlaps[mask_gt4], abs_errors[mask_gt4], alpha=0.5, label=">4 voxels")
plt.plot(unique_overlaps, mean_abs_error, linewidth=2, label="mean abs error")
plt.fill_between(
    unique_overlaps,
    mean_abs_error - std_abs_error,
    mean_abs_error + std_abs_error,
    alpha=0.2,
    label="±1 std",
)
plt.xlabel("Overlap (%)")
plt.ylabel("Absolute stitching error (pixels)")
plt.title("Stitching Error vs Overlap")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 5))
plt.scatter(overlaps, signed_errors, alpha=0.2, label="_nolegend_")
plt.scatter(overlaps[mask_0], signed_errors[mask_0], label="0 voxel error")
plt.scatter(overlaps[mask_1], signed_errors[mask_1], label="1 voxel error")
plt.scatter(overlaps[mask_2], signed_errors[mask_2], label="2 voxel error")
plt.scatter(overlaps[mask_3], signed_errors[mask_3], label="3 voxel error")
plt.scatter(overlaps[mask_4], signed_errors[mask_4], label="4 voxel error")
plt.scatter(overlaps[mask_gt4], signed_errors[mask_gt4], alpha=0.5, label=">4 voxels")
plt.plot(unique_overlaps, mean_signed_error, linewidth=2, label="mean signed error")
plt.axhline(0, linestyle="--", linewidth=1)
plt.xlabel("Overlap (%)")
plt.ylabel("Signed stitching error (pixels)")
plt.title("Signed Stitching Error vs Overlap")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


positive_mask = (expected_shifts >= 0) & (predicted_shifts >= 0)

expected_shifts_pos = expected_shifts[positive_mask]
predicted_shifts_pos = predicted_shifts[positive_mask]
abs_errors_pos = abs_errors[positive_mask]

mask_0 = abs_errors_pos == 0
mask_1 = abs_errors_pos == 1
mask_2 = abs_errors_pos == 2
mask_3 = abs_errors_pos == 3
mask_4 = abs_errors_pos == 4
mask_gt4 = abs_errors_pos > 4

plt.figure(figsize=(6, 6))
plt.scatter(expected_shifts_pos, predicted_shifts_pos, alpha=0.2, label="_nolegend_")
plt.scatter(expected_shifts_pos[mask_0], predicted_shifts_pos[mask_0], label="0 voxel error")
plt.scatter(expected_shifts_pos[mask_1], predicted_shifts_pos[mask_1], label="1 voxel error")
plt.scatter(expected_shifts_pos[mask_2], predicted_shifts_pos[mask_2], label="2 voxel error")
plt.scatter(expected_shifts_pos[mask_3], predicted_shifts_pos[mask_3], label="3 voxel error")
plt.scatter(expected_shifts_pos[mask_4], predicted_shifts_pos[mask_4], label="4 voxel error")
plt.scatter(expected_shifts_pos[mask_gt4], predicted_shifts_pos[mask_gt4], alpha=0.5, label=">4 voxels")

if len(expected_shifts_pos) > 0:
    lo = min(np.min(expected_shifts_pos), np.min(predicted_shifts_pos))
    hi = max(np.max(expected_shifts_pos), np.max(predicted_shifts_pos))
    plt.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)

plt.xlabel("Expected shift (pixels)")
plt.ylabel("Predicted shift (pixels)")
plt.title("Expected vs Predicted Shift")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# OPTIONAL: CHECK CANVAS BUILD FOR LAST SUCCESSFUL CASE
# ============================================================
last = successful_sorted[-1]
last_case = None
for case in cases:
    if case["ovlp_dir"].name == f"ovlp_{int(last['overlap_percent']):03d}" and case["pair_dir"].name == last["pair_name"]:
        last_case = case
        break

if last_case is not None:
    vol_a_signal = np.load(last_case["vol_a_path"]).astype(np.float32)
    vol_b_signal = np.load(last_case["vol_b_path"]).astype(np.float32)

    if TRANSPOSE_INPUT_TO_ZXY:
        vol_a = np.transpose(vol_a_signal, (0, 2, 1))
        vol_b = np.transpose(vol_b_signal, (0, 2, 1))
    else:
        vol_a = vol_a_signal
        vol_b = vol_b_signal

    canvas_a, canvas_b = stitch_volumes(
        vol_a,
        vol_b,
        int(round(last["best_shift"])),
        axis=STITCH_AXIS,
    )