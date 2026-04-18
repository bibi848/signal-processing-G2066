from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# SETTINGS
# ============================================================

EXPORT_DIR = Path.cwd() / "STITCHING" / "combine_and_stitch" / "correlation_exports"
OUTPUT_DIR = EXPORT_DIR / "plots_by_overlap"

SHOW_PLOTS = False
SAVE_PLOTS = True

PAIR_LINEWIDTH = 1.5
EXPECTED_LINESTYLE = ":"
EXPECTED_LINEWIDTH = 2.0


# ============================================================
# HELPERS
# ============================================================
def find_grouped_csvs(export_dir: Path):
    return sorted(export_dir.glob("ovlp_*_correlation_vs_shift.csv"))


def load_metadata(csv_path: Path):
    meta_path = csv_path.with_name(
        csv_path.name.replace("_correlation_vs_shift.csv", "_metadata.json")
    )
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata: {meta_path}")
    with open(meta_path, "r") as f:
        return json.load(f)


def get_pair_tile_count(pair_meta: dict) -> int | None:
    """
    Prefer the actual number of valid tiles if present.
    Fall back to rows*cols from the grid if needed.
    """
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


def normalise_curve_by_tiles(y, pair_meta: dict):
    """
    Divide summed tiled correlation by the number of contributing tiles
    so the plotted correlation is back on a 0..1 scale.

    If no tile count is available, returns the curve unchanged.
    """
    tile_count = get_pair_tile_count(pair_meta)
    if tile_count is None:
        return y
    return y / tile_count


# ============================================================
# PLOT
# ============================================================
def plot_overlap(csv_path: Path, output_dir: Path):
    df = pd.read_csv(csv_path)
    meta = load_metadata(csv_path)

    if "shift" not in df.columns:
        raise ValueError(f"'shift' column missing in {csv_path}")

    shift = df["shift"].to_numpy()
    pair_columns = [c for c in df.columns if c != "shift"]

    if not pair_columns:
        print(f"Skipping {csv_path.name}")
        return

    plt.figure(figsize=(10, 6))

    # --- Plot each pair ---
    for i, col in enumerate(pair_columns):
        raw_y = df[col].to_numpy()
        pair_meta = meta["pairs"].get(col, {})

        y = normalise_curve_by_tiles(raw_y, pair_meta)
        chosen_shift = pair_meta.get("best_shift")

        tile_count = get_pair_tile_count(pair_meta)
        if chosen_shift is not None and tile_count is not None:
            label = f"{col} (shift={int(chosen_shift)}, tiles={tile_count})"
        elif chosen_shift is not None:
            label = f"{col} (shift={int(chosen_shift)})"
        else:
            label = col

        plt.plot(
            shift,
            y,
            linewidth=PAIR_LINEWIDTH,
            label=label,
        )

    # --- Expected shift (single line) ---
    pairs_meta = meta.get("pairs", {})

    expected_shift = None
    if pairs_meta:
        first_pair = next(iter(pairs_meta.values()))
        expected_shift = first_pair.get("expected_shift")

    if expected_shift is not None:
        plt.axvline(
            expected_shift,
            linestyle=EXPECTED_LINESTYLE,
            linewidth=EXPECTED_LINEWIDTH,
            color="black",
            label=f"Expected shift ({int(expected_shift)})",
        )

    overlap_name = csv_path.stem.replace("_correlation_vs_shift", "")

    plt.xlabel("Shift (pixels)", fontsize=18)
    plt.ylabel("Correlation", fontsize=18)


    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if SAVE_PLOTS:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"{overlap_name}.png"
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"Saved: {out_path}")

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()


# ============================================================
# MAIN
# ============================================================
def main():
    csv_files = find_grouped_csvs(EXPORT_DIR)

    if not csv_files:
        raise FileNotFoundError(f"No grouped CSVs found in {EXPORT_DIR}")

    for csv_path in csv_files:
        print(f"Plotting {csv_path.name}")
        plot_overlap(csv_path, OUTPUT_DIR)


if __name__ == "__main__":
    main()