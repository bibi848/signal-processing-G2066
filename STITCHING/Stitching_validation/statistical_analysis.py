#%%
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter


# ============================================================
# Config
# ============================================================

# Peak/baseline settings
EXCLUDE_RADIUS = 20          # your +/- 20 px rule
EDGE_MARGIN = 20             # if peak is within this of boundary -> incomplete/edge case
SECOND_PEAK_EXCLUDE = 5      # exclude +/- around top peak when finding second peak
SMOOTH_WINDOW = 11           # Savitzky-Golay window, must be odd
SMOOTH_POLYORDER = 3
MIN_PEAK_PROMINENCE = 0.0    # leave at 0 first; raise later if needed

# Correctness thresholds for overlap-level summary
SUPPORT_THRESHOLDS = (1, 2, 5)

# Output
PAIR_FEATURES_FILENAME = "pair_features.csv"
OVERLAP_FEATURES_FILENAME = "overlap_features.csv"
PEAK_STD_RADIUS = 20


# ============================================================
# Utilities
# ============================================================

def robust_std(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return 1.4826 * mad


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)

    if len(values) == 0:
        return np.nan
    if np.all(weights <= 0):
        return float(np.median(values))

    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cum = np.cumsum(weights)
    cutoff = 0.5 * np.sum(weights)
    idx = np.searchsorted(cum, cutoff, side="left")
    idx = min(idx, len(values) - 1)
    return float(values[idx])


def safe_div(a: float, b: float, default=np.nan) -> float:
    if b == 0 or np.isnan(b):
        return default
    return a / b


def maybe_smooth(y: np.ndarray, window: int = SMOOTH_WINDOW, polyorder: int = SMOOTH_POLYORDER) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if len(y) < 5:
        return y.copy()

    # Window must be odd and <= len(y)
    window = min(window, len(y) if len(y) % 2 == 1 else len(y) - 1)
    if window < polyorder + 2:
        return y.copy()
    if window % 2 == 0:
        window -= 1
    if window < 3:
        return y.copy()

    return savgol_filter(y, window_length=window, polyorder=polyorder, mode="interp")


def find_crossing_left(x: np.ndarray, y: np.ndarray, peak_idx: int, level: float) -> float:
    """
    Find x location where curve crosses 'level' on left side of peak.
    Returns np.nan if not found.
    Linear interpolation between samples.
    """
    for i in range(peak_idx - 1, -1, -1):
        y0, y1 = y[i], y[i + 1]
        if (y0 <= level <= y1) or (y1 <= level <= y0):
            if y1 == y0:
                return float(x[i])
            t = (level - y0) / (y1 - y0)
            return float(x[i] + t * (x[i + 1] - x[i]))
    return np.nan


def find_crossing_right(x: np.ndarray, y: np.ndarray, peak_idx: int, level: float) -> float:
    """
    Find x location where curve crosses 'level' on right side of peak.
    Returns np.nan if not found.
    Linear interpolation between samples.
    """
    for i in range(peak_idx, len(y) - 1):
        y0, y1 = y[i], y[i + 1]
        if (y0 >= level >= y1) or (y1 >= level >= y0):
            if y1 == y0:
                return float(x[i + 1])
            t = (level - y0) / (y1 - y0)
            return float(x[i] + t * (x[i + 1] - x[i]))
    return np.nan


def infer_shift_column(df: pd.DataFrame) -> str:
    candidates = ["shift", "x_shift", "lag", "offset", "dx", "displacement"]
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in lower:
            return lower[c]

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        raise ValueError("Could not infer shift column")
    return numeric_cols[0]


def infer_pair_columns(df: pd.DataFrame, shift_col: str) -> list[str]:
    pair_cols = []

    for c in df.columns:
        lc = c.lower()
        if c == shift_col:
            continue
        if ("pair" in lc) or ("corr" in lc) or ("score" in lc):
            if pd.api.types.is_numeric_dtype(df[c]):
                pair_cols.append(c)

    if pair_cols:
        return pair_cols

    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != shift_col]
    return numeric_cols


def load_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return json.load(f)


def extract_expected_shift(meta: dict[str, Any]) -> float | None:
    candidates = [
        "expected_shift",
        "expected",
        "expected_dx",
        "expected_offset",
        "prior_shift",
    ]
    for k in candidates:
        if k in meta:
            return meta[k]
    return None


def extract_true_shift(meta: dict[str, Any]) -> float | None:
    candidates = [
        "true_shift",
        "ground_truth_shift",
        "gt_shift",
        "actual_shift",
    ]
    for k in candidates:
        if k in meta:
            return meta[k]
    return None


def extract_overlap_label(path: Path) -> str:
    m = re.search(r"(ovlp_\d+)", path.name)
    return m.group(1) if m else path.stem


# ============================================================
# Feature extraction
# ============================================================

@dataclass
class PairFeatures:
    overlap_id: str
    pair_id: str

    expected_shift: float | None
    true_shift: float | None

    peak1_shift: float
    peak1_value: float
    peak1_idx: int

    peak_std: float | None
    peak_std_norm: float | None
    peak_to_baseline_std_ratio: float | None

    peak2_shift: float | None
    peak2_value: float | None

    peak_ratio: float | None
    peak_margin: float | None

    baseline_median: float
    baseline_mean: float
    baseline_std: float
    baseline_mad_std: float

    psr_median_mad: float | None
    psr_mean_std: float | None

    half_level: float
    level_95: float

    left_cross_50: float | None
    right_cross_50: float | None
    width_50: float | None

    left_cross_95: float | None
    right_cross_95: float | None
    width_95: float | None

    left_width_50: float | None
    right_width_50: float | None
    left_width_95: float | None
    right_width_95: float | None

    prominence: float | None
    prominence_left_base_shift: float | None
    prominence_right_base_shift: float | None

    candidate_peak_count: int

    is_boundary_peak: bool
    incomplete_50: bool
    incomplete_95: bool

    distance_to_expected: float | None
    abs_error_true: float | None
    is_correct_true_1px: bool | None
    is_correct_true_2px: bool | None
    is_correct_true_5px: bool | None


def compute_pair_features(
    x: np.ndarray,
    y_raw: np.ndarray,
    overlap_id: str,
    pair_id: str,
    expected_shift: float | None = None,
    true_shift: float | None = None,
    exclude_radius: int = EXCLUDE_RADIUS,
    edge_margin: int = EDGE_MARGIN,
    second_peak_exclude: int = SECOND_PEAK_EXCLUDE,
    peak_std_radius: int = PEAK_STD_RADIUS,
) -> PairFeatures:
    x = np.asarray(x, dtype=float)
    y_raw = np.asarray(y_raw, dtype=float)
    y = y_raw.copy()

    peak1_idx = int(np.argmax(y))
    peak1_shift = float(x[peak1_idx])
    peak1_value = float(y[peak1_idx])

    # Peak-region std
    lo_p = max(0, peak1_idx - peak_std_radius)
    hi_p = min(len(y), peak1_idx + peak_std_radius + 1)
    peak_region = y[lo_p:hi_p]

    if len(peak_region) > 1:
        peak_std = float(np.std(peak_region))
    else:
        peak_std = np.nan

    peak_std_norm = safe_div(peak_std, peak1_value, default=np.nan)

    # Baseline region using your +/-25 rule
    mask = np.ones(len(y), dtype=bool)
    lo = max(0, peak1_idx - exclude_radius)
    hi = min(len(y), peak1_idx + exclude_radius + 1)
    mask[lo:hi] = False
    outside = y[mask]

    if len(outside) == 0:
        baseline_median = float(np.median(y))
        baseline_mean = float(np.mean(y))
        baseline_std = float(np.std(y))
        baseline_mad_std = float(robust_std(y))
    else:
        baseline_median = float(np.median(outside))
        baseline_mean = float(np.mean(outside))
        baseline_std = float(np.std(outside))
        baseline_mad_std = float(robust_std(outside))

    peak_to_baseline_std_ratio = safe_div(peak_std, baseline_std, default=np.nan)

    psr_median_mad = None
    if baseline_mad_std > 0:
        psr_median_mad = float((peak1_value - baseline_median) / baseline_mad_std)

    psr_mean_std = None
    if baseline_std > 0:
        psr_mean_std = float((peak1_value - baseline_mean) / baseline_std)

    # Relative levels for widths
    half_level = float(baseline_median + 0.5 * (peak1_value - baseline_median))
    level_95 = float(baseline_median + 0.95 * (peak1_value - baseline_median))

    left_cross_50 = find_crossing_left(x, y, peak1_idx, half_level)
    right_cross_50 = find_crossing_right(x, y, peak1_idx, half_level)

    left_cross_95 = find_crossing_left(x, y, peak1_idx, level_95)
    right_cross_95 = find_crossing_right(x, y, peak1_idx, level_95)

    width_50 = None if (np.isnan(left_cross_50) or np.isnan(right_cross_50)) else float(right_cross_50 - left_cross_50)
    width_95 = None if (np.isnan(left_cross_95) or np.isnan(right_cross_95)) else float(right_cross_95 - left_cross_95)

    left_width_50 = None if np.isnan(left_cross_50) else float(peak1_shift - left_cross_50)
    right_width_50 = None if np.isnan(right_cross_50) else float(right_cross_50 - peak1_shift)

    left_width_95 = None if np.isnan(left_cross_95) else float(peak1_shift - left_cross_95)
    right_width_95 = None if np.isnan(right_cross_95) else float(right_cross_95 - peak1_shift)

    incomplete_50 = width_50 is None
    incomplete_95 = width_95 is None

    is_boundary_peak = (peak1_idx <= edge_margin) or (peak1_idx >= len(y) - 1 - edge_margin)

    # Candidate peaks for second-peak stats and prominence
    peak_idxs, props = find_peaks(y, prominence=MIN_PEAK_PROMINENCE)
    if len(peak_idxs) == 0:
        peak_idxs = np.array([peak1_idx], dtype=int)
        prominences = np.array([np.nan], dtype=float)
        left_bases = np.array([np.nan], dtype=float)
        right_bases = np.array([np.nan], dtype=float)
    else:
        prominences = props.get("prominences", np.full(len(peak_idxs), np.nan))
        left_bases = props.get("left_bases", np.full(len(peak_idxs), np.nan))
        right_bases = props.get("right_bases", np.full(len(peak_idxs), np.nan))

    candidate_peak_count = int(len(peak_idxs))

    prominence = None
    prominence_left_base_shift = None
    prominence_right_base_shift = None

    if peak1_idx in peak_idxs:
        winner_loc = int(np.where(peak_idxs == peak1_idx)[0][0])
        if not np.isnan(prominences[winner_loc]):
            prominence = float(prominences[winner_loc])
        lb = left_bases[winner_loc]
        rb = right_bases[winner_loc]
        if not np.isnan(lb):
            prominence_left_base_shift = float(x[int(lb)])
        if not np.isnan(rb):
            prominence_right_base_shift = float(x[int(rb)])
    else:
        prominence = float(peak1_value - baseline_median)

    # second peak outside a small exclusion zone around the top peak
    second_mask = np.ones(len(y), dtype=bool)
    lo2 = max(0, peak1_idx - second_peak_exclude)
    hi2 = min(len(y), peak1_idx + second_peak_exclude + 1)
    second_mask[lo2:hi2] = False
    y_second = y.copy()
    y_second[~second_mask] = -np.inf

    peak2_idx = int(np.argmax(y_second))
    peak2_value = None
    peak2_shift = None
    peak_ratio = None
    peak_margin = None

    if np.isfinite(y_second[peak2_idx]):
        peak2_value = float(y[peak2_idx])
        peak2_shift = float(x[peak2_idx])
        peak_ratio = safe_div(peak1_value, peak2_value, default=np.nan)
        peak_margin = float(peak1_value - peak2_value)

    distance_to_expected = None if expected_shift is None else float(abs(peak1_shift - expected_shift))
    abs_error_true = None if true_shift is None else float(abs(peak1_shift - true_shift))

    is_correct_true_1px = None if abs_error_true is None else bool(abs_error_true <= 1)
    is_correct_true_2px = None if abs_error_true is None else bool(abs_error_true <= 2)
    is_correct_true_5px = None if abs_error_true is None else bool(abs_error_true <= 5)

    return PairFeatures(
        overlap_id=overlap_id,
        pair_id=pair_id,
        expected_shift=expected_shift,
        true_shift=true_shift,
        peak1_shift=peak1_shift,
        peak1_value=peak1_value,
        peak1_idx=peak1_idx,
        peak_std=peak_std,
        peak_std_norm=peak_std_norm,
        peak_to_baseline_std_ratio=peak_to_baseline_std_ratio,
        peak2_shift=peak2_shift,
        peak2_value=peak2_value,
        peak_ratio=peak_ratio,
        peak_margin=peak_margin,
        baseline_median=baseline_median,
        baseline_mean=baseline_mean,
        baseline_std=baseline_std,
        baseline_mad_std=baseline_mad_std,
        psr_median_mad=psr_median_mad,
        psr_mean_std=psr_mean_std,
        half_level=half_level,
        level_95=level_95,
        left_cross_50=None if np.isnan(left_cross_50) else float(left_cross_50),
        right_cross_50=None if np.isnan(right_cross_50) else float(right_cross_50),
        width_50=width_50,
        left_cross_95=None if np.isnan(left_cross_95) else float(left_cross_95),
        right_cross_95=None if np.isnan(right_cross_95) else float(right_cross_95),
        width_95=width_95,
        left_width_50=left_width_50,
        right_width_50=right_width_50,
        left_width_95=left_width_95,
        right_width_95=right_width_95,
        prominence=prominence,
        prominence_left_base_shift=prominence_left_base_shift,
        prominence_right_base_shift=prominence_right_base_shift,
        candidate_peak_count=candidate_peak_count,
        is_boundary_peak=is_boundary_peak,
        incomplete_50=incomplete_50,
        incomplete_95=incomplete_95,
        distance_to_expected=distance_to_expected,
        abs_error_true=abs_error_true,
        is_correct_true_1px=is_correct_true_1px,
        is_correct_true_2px=is_correct_true_2px,
        is_correct_true_5px=is_correct_true_5px,
    )


# ============================================================
# Overlap aggregation
# ============================================================

def aggregate_overlap_features(pair_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for overlap_id, g in pair_df.groupby("overlap_id", sort=True):
        shifts = g["peak1_shift"].to_numpy(dtype=float)

        # Use PSR if available; fallback to ones
        weights = g["psr_median_mad"].to_numpy(dtype=float)
        weights = np.where(np.isfinite(weights) & (weights > 0), weights, 1.0)

        consensus_shift = weighted_median(shifts, weights)

        shift_mad = float(np.median(np.abs(shifts - np.median(shifts)))) if len(shifts) else np.nan
        shift_std = float(np.std(shifts)) if len(shifts) else np.nan

        row: dict[str, Any] = {
            "overlap_id": overlap_id,
            "n_pairs": int(len(g)),
            "expected_shift": g["expected_shift"].dropna().iloc[0] if g["expected_shift"].notna().any() else np.nan,
            "true_shift": g["true_shift"].dropna().iloc[0] if g["true_shift"].notna().any() else np.nan,
            "consensus_shift_weighted_median": consensus_shift,
            "consensus_shift_median": float(np.median(shifts)) if len(shifts) else np.nan,
            "shift_std": shift_std,
            "shift_mad": shift_mad,
            "boundary_fraction": float(np.mean(g["is_boundary_peak"].astype(float))),
            "incomplete_50_fraction": float(np.mean(g["incomplete_50"].astype(float))),
            "incomplete_95_fraction": float(np.mean(g["incomplete_95"].astype(float))),
            "median_peak_ratio": float(np.nanmedian(g["peak_ratio"])) if g["peak_ratio"].notna().any() else np.nan,
            "median_peak_margin": float(np.nanmedian(g["peak_margin"])) if g["peak_margin"].notna().any() else np.nan,
            "median_psr_median_mad": float(np.nanmedian(g["psr_median_mad"])) if g["psr_median_mad"].notna().any() else np.nan,
            "median_psr_mean_std": float(np.nanmedian(g["psr_mean_std"])) if g["psr_mean_std"].notna().any() else np.nan,
            "median_width_50": float(np.nanmedian(g["width_50"])) if g["width_50"].notna().any() else np.nan,
            "median_width_95": float(np.nanmedian(g["width_95"])) if g["width_95"].notna().any() else np.nan,
            "median_prominence": float(np.nanmedian(g["prominence"])) if g["prominence"].notna().any() else np.nan,
            "mean_candidate_peak_count": float(np.nanmean(g["candidate_peak_count"])) if g["candidate_peak_count"].notna().any() else np.nan,
        }

        for thr in SUPPORT_THRESHOLDS:
            row[f"support_within_{thr}px"] = float(np.mean(np.abs(shifts - consensus_shift) <= thr))

        expected_shift = row["expected_shift"]
        if pd.notna(expected_shift):
            row["distance_consensus_to_expected"] = float(abs(consensus_shift - expected_shift))
        else:
            row["distance_consensus_to_expected"] = np.nan

        true_shift = row["true_shift"]
        if pd.notna(true_shift):
            abs_error = abs(consensus_shift - true_shift)
            row["abs_error_true"] = float(abs_error)
            row["is_correct_true_1px"] = bool(abs_error <= 1)
            row["is_correct_true_2px"] = bool(abs_error <= 2)
            row["is_correct_true_5px"] = bool(abs_error <= 5)
        else:
            row["abs_error_true"] = np.nan
            row["is_correct_true_1px"] = np.nan
            row["is_correct_true_2px"] = np.nan
            row["is_correct_true_5px"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)

print(Path.cwd())
# ============================================================
# Main processing
# ============================================================
def get_pair_tile_count(pair_meta: dict[str, Any]) -> int | None:
    diagnostics = pair_meta.get("diagnostics", {}) or {}

    valid_tile_count = diagnostics.get("valid_tile_count")
    if valid_tile_count is not None:
        try:
            valid_tile_count = int(valid_tile_count)
            if valid_tile_count > 0:
                return valid_tile_count
        except (TypeError, ValueError):
            pass

    grid = diagnostics.get("grid")
    if grid is not None and len(grid) == 2:
        try:
            rows, cols = int(grid[0]), int(grid[1])
            tile_count = rows * cols
            if tile_count > 0:
                return tile_count
        except (TypeError, ValueError):
            pass

    return None


def normalise_curve_by_tiles(y: np.ndarray, pair_meta: dict[str, Any]) -> np.ndarray:
    tile_count = get_pair_tile_count(pair_meta)
    if tile_count is None:
        return np.asarray(y, dtype=float)
    return np.asarray(y, dtype=float) / tile_count

def process_correlation_exports(correlation_dir: str | Path, output_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    correlation_dir = Path(correlation_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_paths = sorted(correlation_dir.glob("*_correlation_vs_shift.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No *_correlation_vs_shift.csv files found in {correlation_dir}")

    pair_rows: list[dict[str, Any]] = []

    for csv_path in csv_paths:
        overlap_id = extract_overlap_label(csv_path)
        meta_path = csv_path.with_name(
            csv_path.name.replace("_correlation_vs_shift.csv", "_metadata.json")
        )

        meta = load_metadata(meta_path)
        pairs_meta = meta.get("pairs", {})

        df = pd.read_csv(csv_path)
        shift_col = infer_shift_column(df)
        pair_cols = infer_pair_columns(df, shift_col)

        x = df[shift_col].to_numpy(dtype=float)

        if not pair_cols:
            raise ValueError(f"No pair columns found in {csv_path}")

        for pair_col in pair_cols:
            raw_y = df[pair_col].to_numpy(dtype=float)
            pair_meta = pairs_meta.get(pair_col, {})

            expected_shift = pair_meta.get("expected_shift")

            # synthetic data: true shift = expected shift unless explicitly provided
            true_shift = (
                pair_meta.get("true_shift")
                or pair_meta.get("ground_truth_shift")
                or pair_meta.get("gt_shift")
                or pair_meta.get("actual_shift")
                or expected_shift
            )

            y = normalise_curve_by_tiles(raw_y, pair_meta)

            features = compute_pair_features(
                x=x,
                y_raw=y,
                overlap_id=overlap_id,
                pair_id=pair_col,
                expected_shift=expected_shift,
                true_shift=true_shift,
            )
            pair_rows.append(asdict(features))

    pair_df = pd.DataFrame(pair_rows)
    overlap_df = aggregate_overlap_features(pair_df)

    pair_out = output_dir / PAIR_FEATURES_FILENAME
    overlap_out = output_dir / OVERLAP_FEATURES_FILENAME

    pair_df.to_csv(pair_out, index=False)
    overlap_df.to_csv(overlap_out, index=False)

    print(f"Saved pair features: {pair_out}")
    print(f"Saved overlap features: {overlap_out}")

    return pair_df, overlap_df


# ============================================================
# Optional: quick binned summaries against correctness
# ============================================================

def make_binned_correctness_table(
    df: pd.DataFrame,
    feature_col: str,
    correct_col: str = "is_correct_true_2px",
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    Build a table like:
      peak_ratio bin -> % correct
      width_95 bin   -> % correct
      PSR bin        -> % correct

    Only works when ground-truth correctness is present.
    """
    work = df[[feature_col, correct_col]].copy()
    work = work.dropna()

    if len(work) == 0:
        return pd.DataFrame()

    work["bin"] = pd.qcut(work[feature_col], q=min(n_bins, len(work)), duplicates="drop")
    out = (
        work.groupby("bin", observed=True)
        .agg(
            n=(correct_col, "size"),
            pct_correct=(correct_col, "mean"),
            feature_min=(feature_col, "min"),
            feature_max=(feature_col, "max"),
            feature_median=(feature_col, "median"),
        )
        .reset_index(drop=True)
    )
    return out


if __name__ == "__main__":
    # Change these to match your setup
    # Base directory (where your project lives)
    BASE_DIR = Path.cwd()   # or set manually if needed

    # Input / output directories
    correlation_dir = BASE_DIR/ "STITCHING" / "combine_and_stitch" / "correlation_exports"
    output_dir = BASE_DIR / "STITCHING" / "combine_and_stitch" / "metric_exports"

    # Create output directory if needed
    output_dir.mkdir(parents=True, exist_ok=True)

    # Debug prints (VERY useful)
    print("Working directory:", BASE_DIR)
    print("Looking for data in:", correlation_dir.resolve())
    print("Exists:", correlation_dir.exists())

    print("\nCSV files found:")
    for f in correlation_dir.glob("*.csv"):
        print("  ", f.name)

    pair_df, overlap_df = process_correlation_exports(correlation_dir, output_dir)

    # Optional summary tables if you have true_shift in metadata
    for feature in [
    "peak_ratio",
    "psr_median_mad",
    "width_50",
    "width_95",
    "peak_margin",
    "peak_std",
    "peak_std_norm",
    "peak_to_baseline_std_ratio",
    ]:
        if feature in pair_df.columns and "is_correct_true_2px" in pair_df.columns:
            table = make_binned_correctness_table(pair_df, feature, correct_col="is_correct_true_2px", n_bins=10)
            if not table.empty:
                out_path = Path(output_dir) / f"binned_{feature}_vs_correctness.csv"
                table.to_csv(out_path, index=False)
                print(f"Saved: {out_path}")