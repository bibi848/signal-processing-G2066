import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate, correlation_lags
from pathlib import Path
import napari

# ==========================================
# 0. PRE-PROCESSING UTILITIES
# ==========================================
def apply_db_cutoff(vol, cutoff_db=-5):
    """
    Zeros out voxels more than 'cutoff_db' below the peak.
    Returns the thresholded volume and the sparsity percentage.
    """
    v_abs = np.abs(vol)
    v_max = np.max(v_abs)
    if v_max == 0:
        return vol, 0.0

    thresh = v_max * (10 ** (cutoff_db / 20))
    v_thresh = np.where(v_abs >= thresh, vol, 0)
    sparsity = (np.count_nonzero(v_thresh == 0) / v_thresh.size) * 100
    return v_thresh.astype(np.float32), sparsity


# ==========================================
# 1. PLOTTING HELPERS
# ==========================================
def plot_shift_histograms(all_shifts, all_weights, final_shift):
    """
    Plot:
    1) count of tiles choosing each shift
    2) total weighted correlation score per shift
    """
    shifts = np.asarray(all_shifts)
    weights = np.asarray(all_weights, dtype=np.float64)

    unique_shifts = np.arange(shifts.min(), shifts.max() + 1)

    counts = np.array([(shifts == s).sum() for s in unique_shifts], dtype=int)
    weighted_scores = np.array(
        [weights[shifts == s].sum() for s in unique_shifts],
        dtype=np.float64
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    # Tile count histogram
    axes[0].bar(unique_shifts, counts, width=0.8)
    axes[0].axvline(final_shift, color="red", linestyle="--", linewidth=2, label=f"Final shift = {final_shift}")
    axes[0].set_title("Tile Count per Chosen Shift")
    axes[0].set_xlabel("Shift (voxels)")
    axes[0].set_ylabel("Number of tiles")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Weighted score histogram
    axes[1].bar(unique_shifts, weighted_scores, width=0.8)
    axes[1].axvline(final_shift, color="red", linestyle="--", linewidth=2, label=f"Final shift = {final_shift}")
    axes[1].set_title("Total Cross-Correlation Weight per Shift")
    axes[1].set_xlabel("Shift (voxels)")
    axes[1].set_ylabel("Summed peak correlation")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.show()


def plot_tile_maps(tile_shift_map, tile_weight_map, final_shift):
    """
    Plot:
    1) shift chosen by each tile
    2) binary map of tiles that chose the final shift
    3) peak cross-correlation weight per tile
    """
    final_shift_map = np.where(np.isfinite(tile_shift_map), tile_shift_map == final_shift, np.nan)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

    # Shift chosen per tile
    im0 = axes[0].imshow(tile_shift_map, aspect="auto", origin="upper")
    axes[0].set_title("Shift Chosen by Each Tile")
    axes[0].set_xlabel("Tile column")
    axes[0].set_ylabel("Tile row")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # Tiles choosing final shift
    im1 = axes[1].imshow(final_shift_map, aspect="auto", origin="upper", vmin=0, vmax=1)
    axes[1].set_title(f"Tiles Choosing Final Shift ({final_shift})")
    axes[1].set_xlabel("Tile column")
    axes[1].set_ylabel("Tile row")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Peak cross-correlation weight
    im2 = axes[2].imshow(tile_weight_map, aspect="auto", origin="upper")
    axes[2].set_title("Peak Cross-Correlation Score per Tile")
    axes[2].set_xlabel("Tile column")
    axes[2].set_ylabel("Tile row")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.show()


# ==========================================
# 2. THE STITCHER ENGINE (Profile-Based Consensus)
# ==========================================
def run_stitcher_test(vol1, vol2, axis=2, grid=(60, 20), expected=0, tolerance=200, cutoff_db=-5):
    v1, s1 = apply_db_cutoff(vol1, cutoff_db)
    v2, s2 = apply_db_cutoff(vol2, cutoff_db)

    print(f"[Pre-Process] Applied {cutoff_db} dB Cutoff.")
    print(f" -> Vol1 Sparsity: {s1:.1f}% | Vol2 Sparsity: {s2:.1f}%")

    ignore_top = 30
    z_dim, y_dim, x_dim = v1.shape
    z_start, z_end = ignore_top, z_dim

    grid_z, grid_y = grid
    tile_z = (z_end - z_start) // grid_z
    tile_y = y_dim // grid_y

    all_shifts = []
    all_weights = []

    tile_shift_map = np.full((grid_z, grid_y), np.nan, dtype=np.float64)
    tile_weight_map = np.full((grid_z, grid_y), np.nan, dtype=np.float64)

    for r in range(grid_z):
        for c in range(grid_y):
            zs = z_start + (r * tile_z)
            ze = z_start + ((r + 1) * tile_z) if r < grid_z - 1 else z_end
            ys = c * tile_y
            ye = (c + 1) * tile_y if c < grid_y - 1 else y_dim

            prof1 = np.max(np.abs(v1[zs:ze, ys:ye, :]), axis=(0, 1))
            prof2 = np.max(np.abs(v2[zs:ze, ys:ye, :]), axis=(0, 1))

            if np.std(prof1) < 1e-6 or np.max(prof1) == 0:
                continue
            if np.std(prof2) < 1e-6 or np.max(prof2) == 0:
                continue

            p1_n = (prof1 - np.mean(prof1)) / (np.std(prof1) + 1e-10)
            p2_n = (prof2 - np.mean(prof2)) / (np.std(prof2) + 1e-10)

            corr = correlate(p1_n, p2_n, mode='full')
            lags = correlation_lags(len(p1_n), len(p2_n), mode='full')

            # Compute overlap length for each lag
            N = len(p1_n)
            M = len(p2_n)
            overlap = np.minimum(N, M + lags) - np.maximum(0, lags)
            overlap = np.maximum(overlap, 1)

            # Normalize correlation by overlap
            corr = corr / overlap

            mask = (lags >= expected - tolerance) & (lags <= expected + tolerance)
            if not np.any(mask):
                continue
            corr[~mask] = -np.inf

            peak_idx = np.argmax(corr)
            best_shift = lags[peak_idx]
            best_weight = corr[peak_idx]

            all_shifts.append(best_shift)
            all_weights.append(best_weight)

            tile_shift_map[r, c] = best_shift
            tile_weight_map[r, c] = best_weight

    if not all_shifts:
        raise ValueError(f"No features survived the {cutoff_db} dB cutoff.")

    lag_min, lag_max = int(np.min(all_shifts)), int(np.max(all_shifts))
    bins = np.arange(lag_min, lag_max + 2) - 0.5
    counts, bin_edges = np.histogram(all_shifts, bins=bins, weights=all_weights)
    final_shift = int(bin_edges[np.argmax(counts)] + 0.5)

    print(f"\nFinal Shift: {final_shift} voxels")

    results = {
        "final_shift": final_shift,
        "thresholded_volume": v1,
        "all_shifts": np.array(all_shifts),
        "all_weights": np.array(all_weights),
        "tile_shift_map": tile_shift_map,
        "tile_weight_map": tile_weight_map,
    }
    return results


# ==========================================
# 3. STANDALONE STITCHER EXECUTION
# ==========================================
if __name__ == "__main__":
    IN_DIR = Path.cwd() / 'DATA' / '2D TFM Data' / 'FeC Smile 3MHz 04022026 Filtered'

    try:
        vol1_raw = np.load(IN_DIR / "FeC_40_3_filtered_3D_TFM.npy")
        vol2_raw = np.load(IN_DIR / "FeC_40_2_filtered_3D_TFM.npy")
    except FileNotFoundError:
        print("Data not found.")
        raise SystemExit

    results = run_stitcher_test(vol1_raw, vol2_raw, grid=(120, 40), cutoff_db=-5)

    stitch_shift = results["final_shift"]
    v1_thresholded = results["thresholded_volume"]
    all_shifts = results["all_shifts"]
    all_weights = results["all_weights"]
    tile_shift_map = results["tile_shift_map"]
    tile_weight_map = results["tile_weight_map"]

    # Plot histogram / bar charts
    plot_shift_histograms(all_shifts, all_weights, stitch_shift)

    # Plot tile maps
    plot_tile_maps(tile_shift_map, tile_weight_map, stitch_shift)

    # --- ROBUST CONTRAST LIMITS ---
    clim_raw = sorted([float(np.percentile(vol1_raw, 0.1)), float(np.percentile(vol1_raw, 99.9))])
    if clim_raw[0] == clim_raw[1]:
        clim_raw = [clim_raw[0], clim_raw[0] + 1]

    clim_thresh = sorted([float(np.min(v1_thresholded)), float(np.max(v1_thresholded))])
    if clim_thresh[0] == clim_thresh[1]:
        clim_thresh = [0, 1]

    viewer = napari.Viewer(title="Stitcher Result Testing (-5dB Cutoff)")

    # Layer 1: Raw Reference
    viewer.add_image(
        vol1_raw,
        name='Vol 1 (Raw)',
        colormap='cyan',
        contrast_limits=clim_raw,
        opacity=0.5
    )

    # Layer 2: Thresholded hotspots
    viewer.add_image(
        v1_thresholded,
        name='Vol 1 (-5dB Hotspots)',
        colormap='yellow',
        contrast_limits=clim_thresh
    )

    # Layer 3: Shifted Volume 2
    trans = [0, 0, 0]
    trans[2] = stitch_shift
    viewer.add_image(
        vol2_raw,
        name=f'Vol 2 (Shifted {stitch_shift}px)',
        colormap='magenta',
        blending='additive',
        translate=trans,
        contrast_limits=clim_raw
    )

    print(f"\nStitcher complete. Threshold limits used: {clim_thresh}")
    napari.run()