#%%

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# SETTINGS
# ============================================================

PAIR_FEATURES_PATH = Path.cwd().parent.parent / "STITCHING" / "combine_and_stitch" / "metric_exports" / "pair_features.csv"
OUTPUT_DIR = Path.cwd().parent.parent / "STITCHING" / "combine_and_stitch" / "metric_exports" / "certainty_maps"

CORRECTNESS_COL = "is_correct_true_1px"   # choose: is_correct_true_1px / 2px / 5px
N_BINS_1D = 12
N_BINS_2D = 5
MIN_COUNT_PER_BIN = 5

METRICS_1D = [
    "peak_ratio",
    "psr_median_mad",
    "width_50",
    "width_95",
    "peak_std_norm",
    "peak_to_baseline_std_ratio",
    "prominence",
]

METRIC_PAIRS_2D = [
    ("peak_ratio", "width_50"),
    ("psr_median_mad", "width_50"),
    ("peak_ratio", "width_95"),
    ("psr_median_mad", "width_95"),
]


# ============================================================
# HELPERS
# ============================================================

def correctness_tag(correctness_col: str) -> str:
    return correctness_col.replace("is_correct_true_", "")


def prepare_df(path: Path, correctness_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.replace([np.inf, -np.inf], np.nan)
    if correctness_col not in df.columns:
        raise ValueError(f"Missing correctness column: {correctness_col}")
    return df


def make_1d_calibration_table(
    df: pd.DataFrame,
    metric_col: str,
    correctness_col: str,
    n_bins: int = 10,
) -> pd.DataFrame:
    work = df[[metric_col, correctness_col]].dropna().copy()
    if len(work) == 0:
        return pd.DataFrame()

    # quantile bins give more balanced counts
    work["bin"] = pd.qcut(work[metric_col], q=min(n_bins, len(work)), duplicates="drop")

    out = (
        work.groupby("bin", observed=True)
        .agg(
            n=(correctness_col, "size"),
            pct_correct=(correctness_col, "mean"),
            metric_min=(metric_col, "min"),
            metric_max=(metric_col, "max"),
            metric_median=(metric_col, "median"),
        )
        .reset_index(drop=True)
    )
    return out


def plot_1d_calibration(
    table: pd.DataFrame,
    metric_col: str,
    output_dir: Path,
    tag: str,
) -> None:
    if table.empty:
        return

    x = table["metric_median"].to_numpy()
    y = table["pct_correct"].to_numpy()
    n = table["n"].to_numpy()

    plt.figure(figsize=(7, 5))
    plt.plot(x, y, marker="o")
    for xi, yi, ni in zip(x, y, n):
        plt.text(xi, yi, str(int(ni)), fontsize=8, ha="center", va="bottom")

    plt.ylim(-0.02, 1.02)
    plt.xlabel(metric_col)
    plt.ylabel("Empirical P(correct)")
    plt.title(f"1D certainty curve: {metric_col} ({tag})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = output_dir / f"calibration_1d_{metric_col}_{tag}.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def make_2d_certainty_table(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    correctness_col: str,
    n_bins: int = 8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = df[[x_col, y_col, correctness_col]].dropna().copy()
    if len(work) == 0:
        return pd.DataFrame(), pd.DataFrame()

    work["x_bin"] = pd.qcut(work[x_col], q=min(n_bins, len(work)), duplicates="drop")
    work["y_bin"] = pd.qcut(work[y_col], q=min(n_bins, len(work)), duplicates="drop")

    prob_table = work.pivot_table(
        index="y_bin",
        columns="x_bin",
        values=correctness_col,
        aggfunc="mean",
        observed=True,
    )

    count_table = work.pivot_table(
        index="y_bin",
        columns="x_bin",
        values=correctness_col,
        aggfunc="size",
        observed=True,
    )

    return prob_table, count_table


def plot_2d_certainty_map(
    prob_table: pd.DataFrame,
    count_table: pd.DataFrame,
    x_col: str,
    y_col: str,
    output_dir: Path,
    tag: str,
    min_count_per_bin: int = 5,
) -> None:
    if prob_table.empty:
        return

    probs = prob_table.copy()
    counts = count_table.reindex_like(prob_table)

    # mask sparse bins
    probs = probs.where(counts >= min_count_per_bin)

    plt.figure(figsize=(8, 6))
    im = plt.imshow(probs.to_numpy(), origin="lower", aspect="auto", vmin=0, vmax=1)
    plt.colorbar(im, label="Empirical P(correct)")

    plt.title(f"2D certainty map: {x_col} vs {y_col} ({tag})")
    plt.xlabel(x_col)
    plt.ylabel(y_col)

    plt.xticks(
        ticks=np.arange(probs.shape[1]),
        labels=[f"{i+1}" for i in range(probs.shape[1])],
        rotation=45,
        ha="right",
    )
    plt.yticks(
        ticks=np.arange(probs.shape[0]),
        labels=[f"{i+1}" for i in range(probs.shape[0])],
    )

    for r in range(probs.shape[0]):
        for c in range(probs.shape[1]):
            p = probs.iat[r, c]
            n = counts.iat[r, c]
            if pd.notna(p):
                plt.text(c, r, f"{p:.2f}\n(n={int(n)})", ha="center", va="center", fontsize=8)

    plt.tight_layout()
    out_path = output_dir / f"certainty_2d_{x_col}_vs_{y_col}_{tag}.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def lookup_empirical_probability_1d(
    calibration_table: pd.DataFrame,
    metric_value: float,
) -> float | None:
    if calibration_table.empty or pd.isna(metric_value):
        return None

    for _, row in calibration_table.iterrows():
        if row["metric_min"] <= metric_value <= row["metric_max"]:
            return float(row["pct_correct"])

    idx = np.argmin(np.abs(calibration_table["metric_median"].to_numpy() - metric_value))
    return float(calibration_table.iloc[idx]["pct_correct"])


# ============================================================
# MAIN
# ============================================================

def main():
    tag = correctness_tag(CORRECTNESS_COL)
    run_output_dir = OUTPUT_DIR / tag
    run_output_dir.mkdir(parents=True, exist_ok=True)

    df = prepare_df(PAIR_FEATURES_PATH, CORRECTNESS_COL)

    # 1D curves
    one_d_tables = {}
    for metric in METRICS_1D:
        if metric not in df.columns:
            print(f"Skipping missing metric: {metric}")
            continue

        table = make_1d_calibration_table(df, metric, CORRECTNESS_COL, n_bins=N_BINS_1D)
        if table.empty:
            print(f"No usable data for metric: {metric}")
            continue

        one_d_tables[metric] = table
        table.to_csv(run_output_dir / f"calibration_1d_{metric}_{tag}.csv", index=False)
        plot_1d_calibration(table, metric, run_output_dir, tag)
        print(f"Saved 1D calibration for {metric} ({tag})")

    # 2D maps
    for x_col, y_col in METRIC_PAIRS_2D:
        if x_col not in df.columns or y_col not in df.columns:
            print(f"Skipping missing pair: {x_col}, {y_col}")
            continue

        prob_table, count_table = make_2d_certainty_table(
            df,
            x_col=x_col,
            y_col=y_col,
            correctness_col=CORRECTNESS_COL,
            n_bins=N_BINS_2D,
        )

        if prob_table.empty:
            print(f"No usable data for pair: {x_col}, {y_col}")
            continue

        prob_table.to_csv(run_output_dir / f"certainty_2d_{x_col}_vs_{y_col}_{tag}_prob.csv")
        count_table.to_csv(run_output_dir / f"certainty_2d_{x_col}_vs_{y_col}_{tag}_count.csv")
        plot_2d_certainty_map(
            prob_table,
            count_table,
            x_col=x_col,
            y_col=y_col,
            output_dir=run_output_dir,
            tag=tag,
            min_count_per_bin=MIN_COUNT_PER_BIN,
        )
        print(f"Saved 2D certainty map for {x_col} vs {y_col} ({tag})")

    # Example: look up certainty for a new case using a 1D metric
    if "peak_ratio" in one_d_tables:
        example_value = 1.8
        p = lookup_empirical_probability_1d(one_d_tables["peak_ratio"], example_value)
        print(
            f"Example lookup ({tag}): peak_ratio={example_value:.2f} -> P(correct)≈{p:.3f}"
            if p is not None else "No lookup available"
        )


if __name__ == "__main__":
    main()
# %%