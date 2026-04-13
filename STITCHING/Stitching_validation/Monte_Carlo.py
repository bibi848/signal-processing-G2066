from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import correlate, correlation_lags


# ==========================================
# 0. PRE-PROCESSING
# ==========================================
def apply_db_cutoff(vol: np.ndarray, cutoff_db: float = -5.0) -> tuple[np.ndarray, float]:
    """
    Zero voxels more than cutoff_db below the peak magnitude.
    Returns thresholded volume and sparsity percentage.
    """
    v_abs = np.abs(vol)
    v_max = float(np.max(v_abs))

    if v_max == 0:
        return vol.astype(np.float32), 0.0

    thresh = v_max * (10 ** (cutoff_db / 20.0))
    v_thresh = np.where(v_abs >= thresh, vol, 0)

    sparsity = 100.0 * np.count_nonzero(v_thresh == 0) / v_thresh.size
    return v_thresh.astype(np.float32), float(sparsity)


# ==========================================
# 1. DIAGNOSTIC PLOTTING
# ==========================================
def plot_stitcher_diagnostics(diag: dict[str, Any]) -> None:
    shifts = diag["shift_axis"]
    weighted_scores = diag["weighted_scores"]
    vote_counts = diag["vote_counts"]
    final_shift = diag["final_shift"]
    tile_vote_map = diag["tile_vote_map"]
    tile_distance_map = diag["tile_distance_map"]
    tile_axis_names = diag["tile_axis_names"]

    plt.figure(figsize=(11, 4))
    plt.plot(shifts, weighted_scores, marker="o", linewidth=1.5)
    plt.axvline(final_shift, linestyle="--", linewidth=1.5, label=f"Chosen Shift = {final_shift}")
    plt.title("Weighted Cross-Correlation Score by Shift")
    plt.xlabel("Shift (voxels)")
    plt.ylabel("Sum of tile confidence scores")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.figure(figsize=(11, 4))
    plt.bar(shifts, vote_counts, width=0.9)
    plt.axvline(final_shift, linestyle="--", linewidth=1.5, label=f"Chosen Shift = {final_shift}")
    plt.title("Tile Vote Count by Shift")
    plt.xlabel("Shift (voxels)")
    plt.ylabel("Number of tiles voting for this shift")
    plt.grid(alpha=0.3, axis="y")
    plt.legend()
    plt.tight_layout()

    plt.figure(figsize=(10, 8))
    heat_data = np.ma.masked_invalid(tile_distance_map)
    im = plt.imshow(heat_data, aspect="auto", interpolation="nearest")
    cbar = plt.colorbar(im)
    cbar.set_label("Distance from chosen shift (|tile shift - chosen shift|)")

    rows, cols = tile_vote_map.shape
    for r in range(rows):
        for c in range(cols):
            if np.isnan(tile_vote_map[r, c]):
                plt.text(c, r, "0", ha="center", va="center", fontsize=8)
            elif int(tile_vote_map[r, c]) == final_shift:
                plt.text(c, r, "x", ha="center", va="center", fontsize=8)

    plt.title(
        f"Tile Shift Selection Heat Map\n"
        f"x = tile chose final shift ({final_shift}), 0 = skipped tile"
    )
    plt.xlabel(f"Tile index along {tile_axis_names[1]}")
    plt.ylabel(f"Tile index along {tile_axis_names[0]}")
    plt.tight_layout()
    plt.show()


# ==========================================
# 2. AXIS HELPERS
# ==========================================
def get_axis_setup(axis: int) -> dict[str, Any]:
    """
    Volume storage convention: (z, y, x)

    axis = 2 -> stitch along x, tile over (z, y)
    axis = 1 -> stitch along y, tile over (z, x)
    """
    axis_names = {0: "z", 1: "y", 2: "x"}

    if axis == 2:
        tile_axes = (0, 1)
    elif axis == 1:
        tile_axes = (0, 2)
    else:
        raise ValueError("Only axis=1 (y) and axis=2 (x) are supported.")

    return {
        "stitch_axis": axis,
        "stitch_axis_name": axis_names[axis],
        "tile_axes": tile_axes,
        "tile_axis_names": (axis_names[tile_axes[0]], axis_names[tile_axes[1]]),
    }


def make_tile_slices(
    shape: tuple[int, int, int],
    tile_axes: tuple[int, int],
    grid: tuple[int, int],
    ignore_top: int = 30,
) -> list[tuple[int, int, tuple[slice, slice, slice]]]:
    starts = [0, 0, 0]
    ends = list(shape)

    if tile_axes[0] == 0:
        starts[0] = ignore_top

    size0 = ends[tile_axes[0]] - starts[tile_axes[0]]
    size1 = ends[tile_axes[1]] - starts[tile_axes[1]]

    tile_size0 = size0 // grid[0]
    tile_size1 = size1 // grid[1]

    if tile_size0 <= 0 or tile_size1 <= 0:
        raise ValueError(
            f"Grid {grid} too fine for shape {shape}, tile_axes {tile_axes}, ignore_top {ignore_top}"
        )

    tiles: list[tuple[int, int, tuple[slice, slice, slice]]] = []

    for r in range(grid[0]):
        for c in range(grid[1]):
            s = [slice(None), slice(None), slice(None)]

            a0_start = starts[tile_axes[0]] + r * tile_size0
            a0_end = starts[tile_axes[0]] + (r + 1) * tile_size0

            a1_start = starts[tile_axes[1]] + c * tile_size1
            a1_end = starts[tile_axes[1]] + (c + 1) * tile_size1

            s[tile_axes[0]] = slice(a0_start, a0_end)
            s[tile_axes[1]] = slice(a1_start, a1_end)

            tiles.append((r, c, tuple(s)))

    return tiles


def extract_profile(
    vol: np.ndarray,
    tile_slices: tuple[slice, slice, slice],
    stitch_axis: int,
) -> np.ndarray:
    subvol = np.abs(vol[tile_slices])
    avg_axes = tuple(ax for ax in range(3) if ax != stitch_axis)
    return np.mean(subvol, axis=avg_axes)


# ==========================================
# 3. STITCHER
# ==========================================
def run_stitcher_test(
    vol1: np.ndarray,
    vol2: np.ndarray,
    axis: int = 2,
    grid: tuple[int, int] = (30, 30),
    expected: int = 0,
    tolerance: int = 100,
    cutoff_db: float = -10.0,
    peak_ratio_min: float = 1.05,
    ignore_top: int = 30,
    verbose: bool = False,
) -> tuple[int, np.ndarray, np.ndarray, dict[str, Any], dict[str, Any]]:
    setup = get_axis_setup(axis)
    stitch_axis_name = setup["stitch_axis_name"]
    tile_axes = setup["tile_axes"]
    tile_axis_names = setup["tile_axis_names"]

    v1, s1 = apply_db_cutoff(vol1, cutoff_db)
    v2, s2 = apply_db_cutoff(vol2, cutoff_db)

    if verbose:
        print(f"[Pre-Process] {cutoff_db:.2f} dB cutoff")
        print(f"Vol1 sparsity: {s1:.1f}% | Vol2 sparsity: {s2:.1f}%")
        print(f"Axis: {stitch_axis_name}")
        print(f"Grid: {grid[0]} x {grid[1]}")
        print(f"Peak ratio min: {peak_ratio_min:.3f}")

    tiles = make_tile_slices(
        shape=v1.shape,
        tile_axes=tile_axes,
        grid=grid,
        ignore_top=ignore_top,
    )

    all_shifts: list[int] = []
    all_weights: list[float] = []
    tile_vote_map = np.full(grid, np.nan, dtype=float)

    rejected_prof1 = 0
    rejected_prof2 = 0
    rejected_not_enough_peaks = 0
    rejected_nonfinite_peak = 0
    rejected_ambiguous = 0

    for r, c, tile_slices in tiles:
        prof1 = extract_profile(v1, tile_slices, stitch_axis=axis)
        prof2 = extract_profile(v2, tile_slices, stitch_axis=axis)

        if np.std(prof1) < 1e-6 or np.max(prof1) == 0:
            rejected_prof1 += 1
            continue

        if np.std(prof2) < 1e-6 or np.max(prof2) == 0:
            rejected_prof2 += 1
            continue

        p1_n = (prof1 - np.mean(prof1)) / (np.std(prof1) + 1e-10)
        p2_n = (prof2 - np.mean(prof2)) / (np.std(prof2) + 1e-10)

        corr = correlate(p1_n, p2_n, mode="full")
        lags = correlation_lags(len(p1_n), len(p2_n), mode="full")

        mask = (lags >= expected - tolerance) & (lags <= expected + tolerance)
        if not np.any(mask):
            rejected_not_enough_peaks += 1
            continue

        corr_masked = corr[mask]
        lags_masked = lags[mask]

        finite_corr = corr_masked[np.isfinite(corr_masked)]
        if finite_corr.size < 2:
            rejected_not_enough_peaks += 1
            continue

        peak_idx = int(np.argmax(corr_masked))
        chosen_lag = int(lags_masked[peak_idx])
        best_peak = float(corr_masked[peak_idx])

        if not np.isfinite(best_peak):
            rejected_nonfinite_peak += 1
            continue

        sorted_peaks = np.sort(finite_corr)
        second_peak = float(sorted_peaks[-2])

        peak_ratio = best_peak / (second_peak + 1e-10)
        if peak_ratio < peak_ratio_min:
            rejected_ambiguous += 1
            continue

        all_shifts.append(chosen_lag)
        all_weights.append(float(peak_ratio))
        tile_vote_map[r, c] = chosen_lag

    if not all_shifts:
        raise ValueError("No tiles survived after thresholding and ambiguity rejection.")

    lag_min = int(np.min(all_shifts))
    lag_max = int(np.max(all_shifts))
    shift_axis = np.arange(lag_min, lag_max + 1)

    weighted_scores = np.zeros_like(shift_axis, dtype=float)
    vote_counts = np.zeros_like(shift_axis, dtype=int)

    shift_to_idx = {int(s): i for i, s in enumerate(shift_axis)}

    for s, w in zip(all_shifts, all_weights):
        idx = shift_to_idx[int(s)]
        weighted_scores[idx] += w
        vote_counts[idx] += 1

    final_shift = int(shift_axis[np.argmax(weighted_scores)])

    tile_distance_map = np.full(grid, np.nan, dtype=float)
    valid_mask = ~np.isnan(tile_vote_map)
    tile_distance_map[valid_mask] = np.abs(tile_vote_map[valid_mask] - final_shift)

    if verbose:
        print(f"Chosen shift: {final_shift} voxels")
        print(f"Participating tiles: {len(all_shifts)} / {grid[0] * grid[1]}")
        print("Rejections:")
        print(f"  prof1 empty/flat: {rejected_prof1}")
        print(f"  prof2 empty/flat: {rejected_prof2}")
        print(f"  not enough peaks: {rejected_not_enough_peaks}")
        print(f"  nonfinite best peak: {rejected_nonfinite_peak}")
        print(f"  ambiguous: {rejected_ambiguous}")

    diagnostics = {
        "shift_axis": shift_axis,
        "weighted_scores": weighted_scores,
        "vote_counts": vote_counts,
        "final_shift": final_shift,
        "tile_vote_map": tile_vote_map,
        "tile_distance_map": tile_distance_map,
        "all_shifts": np.array(all_shifts, dtype=int),
        "all_weights": np.array(all_weights, dtype=float),
        "rejected_prof1": rejected_prof1,
        "rejected_prof2": rejected_prof2,
        "rejected_not_enough_peaks": rejected_not_enough_peaks,
        "rejected_nonfinite_peak": rejected_nonfinite_peak,
        "rejected_ambiguous": rejected_ambiguous,
        "peak_ratio_min": peak_ratio_min,
        "stitch_axis": axis,
        "stitch_axis_name": stitch_axis_name,
        "tile_axes": tile_axes,
        "tile_axis_names": tile_axis_names,
        "grid": grid,
        "cutoff_db": cutoff_db,
        "expected": expected,
        "tolerance": tolerance,
        "ignore_top": ignore_top,
        "vol1_sparsity": s1,
        "vol2_sparsity": s2,
    }

    summary = {
        "chosen_shift": int(final_shift),
        "axis": int(axis),
        "axis_name": stitch_axis_name,
        "grid_r": int(grid[0]),
        "grid_c": int(grid[1]),
        "grid_str": f"{grid[0]}x{grid[1]}",
        "expected": int(expected),
        "tolerance": int(tolerance),
        "cutoff_db": float(cutoff_db),
        "peak_ratio_min": float(peak_ratio_min),
        "ignore_top": int(ignore_top),
        "participating_tiles": int(len(all_shifts)),
        "total_tiles": int(grid[0] * grid[1]),
        "participation_frac": float(len(all_shifts) / (grid[0] * grid[1])),
        "tiles_voting_final": int(np.sum(np.array(all_shifts) == final_shift)),
        "n_unique_tile_shifts": int(len(np.unique(all_shifts))),
        "tile_shift_std": float(np.std(all_shifts)) if len(all_shifts) > 0 else np.nan,
        "rejected_prof1": int(rejected_prof1),
        "rejected_prof2": int(rejected_prof2),
        "rejected_not_enough_peaks": int(rejected_not_enough_peaks),
        "rejected_nonfinite_peak": int(rejected_nonfinite_peak),
        "rejected_ambiguous": int(rejected_ambiguous),
        "vol1_sparsity": float(s1),
        "vol2_sparsity": float(s2),
    }

    return final_shift, v1, v2, summary, diagnostics


# ==========================================
# 4. METRIC HELPERS
# ==========================================
def score_at_shift(shift_axis: np.ndarray, weighted_scores: np.ndarray, shift: int) -> float | None:
    idx = np.where(shift_axis == shift)[0]
    if len(idx) == 0:
        return None
    return float(weighted_scores[idx[0]])


def votes_at_shift(shift_axis: np.ndarray, vote_counts: np.ndarray, shift: int) -> int:
    idx = np.where(shift_axis == shift)[0]
    if len(idx) == 0:
        return 0
    return int(vote_counts[idx[0]])


def summarize_against_truth(diagnostics: dict[str, Any], true_shift: int) -> dict[str, Any]:
    shift_axis = diagnostics["shift_axis"]
    weighted_scores = diagnostics["weighted_scores"]
    vote_counts = diagnostics["vote_counts"]
    chosen_shift = diagnostics["final_shift"]
    all_shifts = diagnostics["all_shifts"]
    total_tiles = diagnostics["grid"][0] * diagnostics["grid"][1]

    chosen_score = score_at_shift(shift_axis, weighted_scores, chosen_shift)
    true_score = score_at_shift(shift_axis, weighted_scores, true_shift)

    sorted_idx = np.argsort(weighted_scores)[::-1]
    ranked_shifts = shift_axis[sorted_idx]

    true_rank_idx = np.where(ranked_shifts == true_shift)[0]
    true_rank = int(true_rank_idx[0] + 1) if len(true_rank_idx) else None

    best_score = float(weighted_scores[sorted_idx[0]])
    second_best_score = float(weighted_scores[sorted_idx[1]]) if len(sorted_idx) > 1 else None

    return {
        "true_shift": int(true_shift),
        "shift_error": int(chosen_shift - true_shift),
        "abs_shift_error": int(abs(chosen_shift - true_shift)),
        "weighted_score_chosen": chosen_score,
        "weighted_score_true": true_score,
        "score_margin_chosen_minus_true": (
            np.nan if true_score is None or chosen_score is None else float(chosen_score - true_score)
        ),
        "votes_chosen_shift": votes_at_shift(shift_axis, vote_counts, chosen_shift),
        "votes_true_shift": votes_at_shift(shift_axis, vote_counts, true_shift),
        "true_shift_rank_by_weighted_score": true_rank,
        "best_score": best_score,
        "second_best_score": second_best_score,
        "best_minus_second_best": (
            np.nan if second_best_score is None else float(best_score - second_best_score)
        ),
        "true_shift_in_search_range": bool(true_shift in shift_axis),
        "participating_tiles": int(len(all_shifts)),
        "total_tiles": int(total_tiles),
        "participation_frac": float(len(all_shifts) / total_tiles),
        "n_unique_tile_shifts": int(len(np.unique(all_shifts))) if len(all_shifts) else 0,
        "tile_shift_std": float(np.std(all_shifts)) if len(all_shifts) else np.nan,
    }


# ==========================================
# 5. STORAGE HELPERS
# ==========================================
def save_diag_npz(path: Path, diagnostics: dict[str, Any]) -> None:
    np.savez_compressed(
        path,
        shift_axis=diagnostics["shift_axis"],
        weighted_scores=diagnostics["weighted_scores"],
        vote_counts=diagnostics["vote_counts"],
        tile_vote_map=diagnostics["tile_vote_map"],
        tile_distance_map=diagnostics["tile_distance_map"],
        all_shifts=diagnostics["all_shifts"],
        all_weights=diagnostics["all_weights"],
    )


def flatten_tile_results(run_id: str, diagnostics: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    vote_map = diagnostics["tile_vote_map"]
    dist_map = diagnostics["tile_distance_map"]
    valid_mask = ~np.isnan(vote_map)

    weights_map = np.full(vote_map.shape, np.nan, dtype=float)
    weight_iter = iter(diagnostics["all_weights"])

    for r in range(vote_map.shape[0]):
        for c in range(vote_map.shape[1]):
            if valid_mask[r, c]:
                weights_map[r, c] = next(weight_iter)

    final_shift = int(diagnostics["final_shift"])

    for r in range(vote_map.shape[0]):
        for c in range(vote_map.shape[1]):
            tile_shift = None if np.isnan(vote_map[r, c]) else int(vote_map[r, c])
            rows.append(
                {
                    "run_id": run_id,
                    "tile_r": r,
                    "tile_c": c,
                    "valid": bool(valid_mask[r, c]),
                    "tile_shift": tile_shift,
                    "tile_weight": None if np.isnan(weights_map[r, c]) else float(weights_map[r, c]),
                    "tile_distance_from_final": None if np.isnan(dist_map[r, c]) else float(dist_map[r, c]),
                    "tile_matches_final": bool(tile_shift == final_shift) if tile_shift is not None else False,
                }
            )
    return rows


def save_dataframe_with_fallback(df: pd.DataFrame, parquet_path: Path) -> Path:
    try:
        df.to_parquet(parquet_path, index=False)
        return parquet_path
    except Exception:
        csv_path = parquet_path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        return csv_path


# ==========================================
# 6. FOLDER LOADING
# ==========================================
def natural_sort_key(path: Path):
    text = str(path)
    return [int(tok) if tok.isdigit() else tok.lower() for tok in re.split(r"(\d+)", text)]


def find_volume_files(root_dir: str | Path) -> list[Path]:
    root_dir = Path(root_dir)

    if not root_dir.exists():
        raise FileNotFoundError(f"Input folder does not exist: {root_dir}")

    files = sorted(root_dir.rglob("*.npy"), key=natural_sort_key)

    if len(files) < 2:
        raise FileNotFoundError(f"Need at least 2 .npy files under {root_dir}, found {len(files)}")

    return files


def make_case_from_folder_adjacent_pair(
    rng: np.random.Generator,
    stitch_params: dict[str, Any],
    case_dir: str | Path,
    pair_index: int = 0,
    true_shift: int | None = None,
) -> dict[str, Any]:
    """
    Loads two adjacent .npy files from a folder tree.
    pair_index=0 -> files[0], files[1]
    pair_index=1 -> files[1], files[2]
    """
    volume_files = find_volume_files(case_dir)

    if pair_index < 0 or pair_index + 1 >= len(volume_files):
        raise IndexError(f"pair_index={pair_index} is out of range for {len(volume_files)} files")

    vol1_path = volume_files[pair_index]
    vol2_path = volume_files[pair_index + 1]

    vol1 = np.load(vol1_path)
    vol2 = np.load(vol2_path)

    return {
        "dataset_id": f"{Path(case_dir).name}_pair_{pair_index:03d}",
        "vol1": vol1,
        "vol2": vol2,
        "true_shift": true_shift,
        "vol1_path": str(vol1_path),
        "vol2_path": str(vol2_path),
    }


# ==========================================
# 7. PARAMETER SAMPLING
# ==========================================
def sample_stitch_params(rng: np.random.Generator) -> dict[str, Any]:
    """
    Hybrid Monte Carlo:
    - structural params sampled from sensible discrete choices
    - thresholds sampled from continuous ranges
    """
    grid_options = [
        (15, 15),
        (20, 20),
        (30, 20),
        (30, 30),
        (40, 20),
        (40, 30),
        (50, 25),
        (60, 20),
    ]

    grid = grid_options[int(rng.integers(0, len(grid_options)))]

    return {
        "axis": 2,
        "grid": grid,
        "expected": 0,
        "tolerance": int(rng.integers(50, 201)),
        "cutoff_db": float(rng.uniform(-12.0, -5.0)),
        "peak_ratio_min": float(rng.uniform(1.02, 1.20)),
        "ignore_top": int(rng.choice([0, 10, 20, 30])),
    }


# ==========================================
# 8. ONE TRIAL
# ==========================================
def run_trial(
    *,
    run_id: str,
    seed: int,
    dataset_id: str,
    vol1: np.ndarray,
    vol2: np.ndarray,
    true_shift: int | None,
    stitch_params: dict[str, Any],
    save_artifacts: bool = False,
    artifact_dir: str | Path | None = None,
    save_tile_rows: bool = False,
    verbose: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]] | None, dict[str, Any] | None]:
    t0 = time.perf_counter()

    try:
        _, _, _, summary, diagnostics = run_stitcher_test(
            vol1,
            vol2,
            verbose=verbose,
            **stitch_params,
        )

        runtime_s = time.perf_counter() - t0

        row: dict[str, Any] = {
            "run_id": run_id,
            "seed": int(seed),
            "dataset_id": dataset_id,
            "status": "ok",
            "runtime_s": float(runtime_s),
            **summary,
        }

        if true_shift is not None:
            row.update(summarize_against_truth(diagnostics, true_shift))

        if save_artifacts and artifact_dir is not None:
            artifact_dir = Path(artifact_dir)
            artifact_dir.mkdir(parents=True, exist_ok=True)
            artifact_path = artifact_dir / f"{run_id}_diag.npz"
            save_diag_npz(artifact_path, diagnostics)
            row["artifact_path"] = str(artifact_path)

        tile_rows = flatten_tile_results(run_id, diagnostics) if save_tile_rows else None
        return row, tile_rows, diagnostics

    except Exception as e:
        runtime_s = time.perf_counter() - t0

        row = {
            "run_id": run_id,
            "seed": int(seed),
            "dataset_id": dataset_id,
            "status": "fail",
            "runtime_s": float(runtime_s),
            "error_type": type(e).__name__,
            "error_msg": str(e),
            "true_shift": np.nan if true_shift is None else int(true_shift),
            "axis": int(stitch_params["axis"]),
            "grid_r": int(stitch_params["grid"][0]),
            "grid_c": int(stitch_params["grid"][1]),
            "grid_str": f"{stitch_params['grid'][0]}x{stitch_params['grid'][1]}",
            "expected": int(stitch_params["expected"]),
            "tolerance": int(stitch_params["tolerance"]),
            "cutoff_db": float(stitch_params["cutoff_db"]),
            "peak_ratio_min": float(stitch_params["peak_ratio_min"]),
            "ignore_top": int(stitch_params["ignore_top"]),
        }
        return row, None, None


# ==========================================
# 9. MONTE CARLO DRIVER
# ==========================================
def run_experiments(
    make_case_fn,
    n_runs: int = 100,
    out_dir: str | Path = "stitch_mc_results",
    save_artifacts: bool = True,
    save_tile_rows: bool = False,
    save_all_artifacts: bool = False,
    random_seed: int = 42,
) -> pd.DataFrame:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = out_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(random_seed)

    run_rows: list[dict[str, Any]] = []
    tile_rows: list[dict[str, Any]] = []

    config = {
        "n_runs": n_runs,
        "save_artifacts": save_artifacts,
        "save_tile_rows": save_tile_rows,
        "save_all_artifacts": save_all_artifacts,
        "random_seed": random_seed,
    }
    (out_dir / "experiment_config.json").write_text(json.dumps(config, indent=2))

    for i in range(n_runs):
        seed = int(rng.integers(0, 2**31 - 1))
        rrng = np.random.default_rng(seed)

        stitch_params = sample_stitch_params(rrng)
        case = make_case_fn(rrng, stitch_params)
        run_id = f"run_{i:06d}"

        row, current_tile_rows, diagnostics = run_trial(
            run_id=run_id,
            seed=seed,
            dataset_id=case["dataset_id"],
            vol1=case["vol1"],
            vol2=case["vol2"],
            true_shift=case.get("true_shift"),
            stitch_params=stitch_params,
            save_artifacts=False,
            artifact_dir=artifact_dir,
            save_tile_rows=save_tile_rows,
            verbose=False,
        )

        for k, v in case.items():
            if k not in {"vol1", "vol2", "dataset_id"}:
                row[k] = v

        should_save_artifact = False
        if save_artifacts and diagnostics is not None:
            if save_all_artifacts:
                should_save_artifact = True
            else:
                should_save_artifact = (
                    row["status"] == "fail"
                    or row.get("abs_shift_error", 0) > 2
                    or (i % 20 == 0)
                )

        if should_save_artifact and diagnostics is not None:
            artifact_path = artifact_dir / f"{run_id}_diag.npz"
            save_diag_npz(artifact_path, diagnostics)
            row["artifact_path"] = str(artifact_path)

        run_rows.append(row)

        if current_tile_rows is not None:
            tile_rows.extend(current_tile_rows)

        print(
            f"[{i + 1:4d}/{n_runs}] "
            f"status={row['status']} "
            f"grid={row.get('grid_str', 'n/a')} "
            f"chosen_shift={row.get('chosen_shift', 'n/a')}"
        )

    runs_df = pd.DataFrame(run_rows)
    saved_runs_path = save_dataframe_with_fallback(runs_df, out_dir / "runs.parquet")
    print(f"\nSaved run table to: {saved_runs_path}")

    if tile_rows:
        tiles_df = pd.DataFrame(tile_rows)
        saved_tiles_path = save_dataframe_with_fallback(tiles_df, out_dir / "tiles.parquet")
        print(f"Saved tile table to: {saved_tiles_path}")

    return runs_df


# ==========================================
# 10. ANALYSIS HELPERS
# ==========================================
def add_eval_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "abs_shift_error" in df.columns:
        df["success_within_0px"] = df["abs_shift_error"] <= 0
        df["success_within_1px"] = df["abs_shift_error"] <= 1
        df["success_within_2px"] = df["abs_shift_error"] <= 2
        df["success_within_5px"] = df["abs_shift_error"] <= 5
    return df


def summarize_results(df: pd.DataFrame) -> pd.DataFrame:
    ok = df[df["status"] == "ok"].copy()
    if ok.empty:
        return pd.DataFrame()

    ok = add_eval_columns(ok)

    agg_dict = {
        "n": ("run_id", "count"),
        "mean_runtime_s": ("runtime_s", "mean"),
        "mean_participation": ("participation_frac", "mean"),
    }

    if "abs_shift_error" in ok.columns:
        agg_dict["mean_abs_error"] = ("abs_shift_error", "mean")
        agg_dict["median_abs_error"] = ("abs_shift_error", "median")

    if "success_within_1px" in ok.columns:
        agg_dict["success_1px"] = ("success_within_1px", "mean")

    if "success_within_2px" in ok.columns:
        agg_dict["success_2px"] = ("success_within_2px", "mean")

    if "score_margin_chosen_minus_true" in ok.columns:
        agg_dict["mean_score_margin"] = ("score_margin_chosen_minus_true", "mean")

    summary = (
        ok.groupby(["cutoff_db", "peak_ratio_min", "axis", "grid_r", "grid_c", "grid_str"])
        .agg(**agg_dict)
        .reset_index()
    )

    sort_cols = []
    ascending = []

    if "success_1px" in summary.columns:
        sort_cols.append("success_1px")
        ascending.append(False)

    if "mean_abs_error" in summary.columns:
        sort_cols.append("mean_abs_error")
        ascending.append(True)

    if sort_cols:
        summary = summary.sort_values(sort_cols, ascending=ascending)

    return summary


# ==========================================
# 11. MAIN
# ==========================================
if __name__ == "__main__":
    """
    Edit these values for your run.
    """

    OUT_DIR = Path("stitch_mc_results")

    CASE_DIR = (
        Path.cwd().parent
        / "SYNTHETIC DATA"
        / "output"
        / "sweep_20260330_125331"
        / "run_018_grain0.005_ovlp0.9"
    )

    PAIR_INDEX = 0
    TRUE_SHIFT = None   # replace with an integer when known
    N_RUNS = 50

    df = run_experiments(
        make_case_fn=lambda rng, stitch_params: make_case_from_folder_adjacent_pair(
            rng,
            stitch_params,
            case_dir=CASE_DIR,
            pair_index=PAIR_INDEX,
            true_shift=TRUE_SHIFT,
        ),
        n_runs=N_RUNS,
        out_dir=OUT_DIR,
        save_artifacts=True,
        save_tile_rows=False,
        save_all_artifacts=False,
        random_seed=42,
    )

    summary_df = summarize_results(df)
    if not summary_df.empty:
        print("\nTop parameter groups:")
        print(summary_df.head(10).to_string(index=False))
    else:
        print("\nNo summary table could be built.")
        print("This usually means there were no successful runs or no truth-based metrics were available.")

    if "artifact_path" in df.columns:
        artifact_paths = df["artifact_path"].dropna().tolist()
        if artifact_paths:
            first_artifact = Path(artifact_paths[0])
            print(f"\nExample saved artifact: {first_artifact}")

            data = np.load(first_artifact)
            diag = {
                "shift_axis": data["shift_axis"],
                "weighted_scores": data["weighted_scores"],
                "vote_counts": data["vote_counts"],
                "tile_vote_map": data["tile_vote_map"],
                "tile_distance_map": data["tile_distance_map"],
                "final_shift": int(data["shift_axis"][np.argmax(data["weighted_scores"])]),
                "tile_axis_names": ("tile_axis_0", "tile_axis_1"),
            }

            # Uncomment to plot one saved diagnostic
            # plot_stitcher_diagnostics(diag)