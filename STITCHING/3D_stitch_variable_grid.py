import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate, correlation_lags
from scipy.ndimage import label, find_objects, binary_opening
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
        return vol.astype(np.float32), 0.0

    thresh = v_max * (10 ** (cutoff_db / 20))
    v_thresh = np.where(v_abs >= thresh, vol, 0)

    sparsity = (np.count_nonzero(v_thresh == 0) / v_thresh.size) * 100
    return v_thresh.astype(np.float32), sparsity


def make_binary_hotspot_volume(vol, binary_threshold=0.35, ignore_top=30, opening_structure=None):
    """
    Creates a binary 3D hotspot mask using a normalised intensity threshold
    between 0 and 1.

    Parameters
    ----------
    vol : ndarray
        Input 3D volume.
    binary_threshold : float
        Threshold in the range [0, 1]. Voxels with normalised absolute
        intensity >= binary_threshold are treated as hotspot voxels.
    ignore_top : int
        Number of voxels to crop from the top of z when creating diagnostics.
    opening_structure : ndarray or None
        Optional structuring element for binary opening, applied in 3D.

    Returns
    -------
    binary_vol : 3D boolean array
        Thresholded binary volume.
    yz_mask : 2D boolean array
        y-z projection of the binary volume, for diagnostics only.
    """
    if not (0 <= binary_threshold <= 1):
        raise ValueError("binary_threshold must be between 0 and 1.")

    v_abs = np.abs(vol)
    v_max = np.max(v_abs)

    if v_max == 0:
        binary_vol = np.zeros_like(vol, dtype=bool)
        yz_shape = (max(vol.shape[0] - min(ignore_top, vol.shape[0]), 0), vol.shape[1])
        yz_mask = np.zeros(yz_shape, dtype=bool)
        return binary_vol, yz_mask

    v_norm = v_abs / v_max
    binary_vol = v_norm >= binary_threshold

    # IMPORTANT: cleanup is applied in 3D, before any projection
    if opening_structure is not None:
        binary_vol = binary_opening(binary_vol, structure=opening_structure)

    z_dim = binary_vol.shape[0]
    z_start = min(ignore_top, z_dim)
    yz_mask = np.any(binary_vol[z_start:, :, :], axis=2)

    return binary_vol, yz_mask


def measure_hotspot_sizes_3d(binary_vol, ignore_top=30, min_voxels=10):
    """
    Measures hotspot extents from 3D connected components.

    A hotspot is a connected region of 1s in the binary volume.
    Any 0-valued gap breaks the hotspot.

    Returns
    -------
    hotspot_sizes : list of dicts
        Each dict contains z, y, x extents and voxel count.
    """
    z_dim = binary_vol.shape[0]
    z_start = min(ignore_top, z_dim)
    binary_crop = binary_vol[z_start:, :, :]

    labeled, _ = label(binary_crop)
    slices = find_objects(labeled)

    hotspot_sizes = []

    for comp_idx, slc in enumerate(slices, start=1):
        if slc is None:
            continue

        z_slc, y_slc, x_slc = slc
        component_mask = (labeled[slc] == comp_idx)
        voxels = int(np.count_nonzero(component_mask))

        if voxels < min_voxels:
            continue

        z_size = int(z_slc.stop - z_slc.start)
        y_size = int(y_slc.stop - y_slc.start)
        x_size = int(x_slc.stop - x_slc.start)

        hotspot_sizes.append({
            "label": comp_idx,
            "z_size": z_size,
            "y_size": y_size,
            "x_size": x_size,
            "voxels": voxels,
        })

    return hotspot_sizes


def estimate_adaptive_grid(
    vol1,
    vol2,
    binary_threshold=0.35,
    ignore_top=30,
    tile_multiple=(2.0, 2.0),
    min_hotspot_voxels=10,
    opening_structure=None,
    size_statistic="median",
    min_grid=(20, 20),
    max_grid=(100, 100),
):
    """
    Estimates an adaptive grid based on hotspot sizes measured from 3D
    connected components.

    The tile size is chosen as:
        tile_z = tile_multiple[0] * representative hotspot z extent
        tile_y = tile_multiple[1] * representative hotspot y extent

    Then the resulting grid is clamped between min_grid and max_grid.
    """
    binary_vol_1, yz_mask_1 = make_binary_hotspot_volume(
        vol1,
        binary_threshold=binary_threshold,
        ignore_top=ignore_top,
        opening_structure=opening_structure,
    )
    binary_vol_2, yz_mask_2 = make_binary_hotspot_volume(
        vol2,
        binary_threshold=binary_threshold,
        ignore_top=ignore_top,
        opening_structure=opening_structure,
    )

    sizes_1 = measure_hotspot_sizes_3d(
        binary_vol_1,
        ignore_top=ignore_top,
        min_voxels=min_hotspot_voxels,
    )
    sizes_2 = measure_hotspot_sizes_3d(
        binary_vol_2,
        ignore_top=ignore_top,
        min_voxels=min_hotspot_voxels,
    )
    all_sizes = sizes_1 + sizes_2

    if not all_sizes:
        raise ValueError(
            "No hotspots detected for adaptive grid sizing. "
            "Try lowering binary_threshold or reducing min_hotspot_voxels."
        )

    z_sizes = np.array([h["z_size"] for h in all_sizes], dtype=float)
    y_sizes = np.array([h["y_size"] for h in all_sizes], dtype=float)

    if size_statistic.lower() == "mean":
        rep_z = float(np.mean(z_sizes))
        rep_y = float(np.mean(y_sizes))
    elif size_statistic.lower() == "median":
        rep_z = float(np.median(z_sizes))
        rep_y = float(np.median(y_sizes))
    else:
        raise ValueError("size_statistic must be 'mean' or 'median'.")

    z_dim, y_dim, _ = vol1.shape
    z_usable = z_dim - min(ignore_top, z_dim)

    tile_z_est = max(1, int(round(tile_multiple[0] * rep_z)))
    tile_y_est = max(1, int(round(tile_multiple[1] * rep_y)))

    grid_rows_raw = max(1, z_usable // tile_z_est)
    grid_cols_raw = max(1, y_dim // tile_y_est)

    min_rows, min_cols = min_grid
    max_rows, max_cols = max_grid

    if min_rows > max_rows or min_cols > max_cols:
        raise ValueError(
            f"min_grid {min_grid} must be coarser than or equal to max_grid {max_grid}. "
            "Use fewer tiles for min_grid and more tiles for max_grid."
        )

    grid_rows = int(np.clip(grid_rows_raw, min_rows, max_rows))
    grid_cols = int(np.clip(grid_cols_raw, min_cols, max_cols))
    grid = (grid_rows, grid_cols)

    info = {
        "representative_hotspot_z": rep_z,
        "representative_hotspot_y": rep_y,
        "tile_z_est": tile_z_est,
        "tile_y_est": tile_y_est,
        "grid_raw": (grid_rows_raw, grid_cols_raw),
        "grid": grid,
        "min_grid": min_grid,
        "max_grid": max_grid,
        "binary_threshold": binary_threshold,
        "num_hotspots_vol1": len(sizes_1),
        "num_hotspots_vol2": len(sizes_2),
        "num_hotspots_total": len(all_sizes),
        "yz_mask_1": yz_mask_1,
        "yz_mask_2": yz_mask_2,
        "binary_vol_1": binary_vol_1,
        "binary_vol_2": binary_vol_2,
        "hotspot_sizes_vol1": sizes_1,
        "hotspot_sizes_vol2": sizes_2,
        "size_statistic": size_statistic,
        "tile_multiple": tile_multiple,
        "min_hotspot_voxels": min_hotspot_voxels,
    }

    return grid, info


# ==========================================
# 1. DIAGNOSTIC PLOTTING
# ==========================================
def plot_stitcher_diagnostics(diag):
    """
    Creates:
      1) weighted cross-correlation score vs shift
      2) vote count vs shift
      3) tile heat map showing distance from chosen shift
    """
    shifts = diag["shift_axis"]
    weighted_scores = diag["weighted_scores"]
    vote_counts = diag["vote_counts"]
    final_shift = diag["final_shift"]
    tile_vote_map = diag["tile_vote_map"]
    tile_distance_map = diag["tile_distance_map"]

    plt.figure(figsize=(11, 4))
    plt.plot(shifts, weighted_scores, marker="o", linewidth=1.5)
    plt.axvline(final_shift, linestyle="--", linewidth=1.5, label=f"Chosen Shift = {final_shift}")
    plt.title("Weighted Cross-Correlation Score by Shift")
    plt.xlabel("Shift (voxels)")
    plt.ylabel("Sum of peak cross-correlation scores")
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
    plt.xlabel("Tile column")
    plt.ylabel("Tile row")
    plt.tight_layout()

    plt.show()


def plot_adaptive_grid_diagnostics(grid_info):
    """
    Optional plotting for the adaptive grid preprocessing step.
    Shows the y-z hotspot masks used for diagnostics only.
    """
    yz_mask_1 = grid_info["yz_mask_1"]
    yz_mask_2 = grid_info["yz_mask_2"]

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(yz_mask_1, aspect="auto", interpolation="nearest")
    plt.title("Vol 1 Hotspot Projection (y-z)")
    plt.xlabel("y")
    plt.ylabel("z (cropped)")

    plt.subplot(1, 2, 2)
    plt.imshow(yz_mask_2, aspect="auto", interpolation="nearest")
    plt.title("Vol 2 Hotspot Projection (y-z)")
    plt.xlabel("y")
    plt.ylabel("z (cropped)")

    plt.tight_layout()
    plt.show()


# ==========================================
# 2. THE STITCHER ENGINE (Profile-Based Consensus)
# ==========================================
def run_stitcher_test(
    vol1,
    vol2,
    axis=2,
    grid=None,
    expected=0,
    tolerance=200,
    cutoff_db=-5,
    binary_threshold=0.35,
    ignore_top=30,
    adaptive_grid=True,
    tile_multiple=(2.0, 2.0),
    min_hotspot_voxels=10,
    size_statistic="median",
    min_grid=(20, 20),
    max_grid=(100, 100),
):
    """
    Runs the stitching algorithm.

    If adaptive_grid=True:
      - a binary 3D hotspot mask is created
      - hotspots are measured from 3D connected components
      - tile size is chosen as a multiple of representative hotspot size
      - the resulting grid is clamped between min_grid and max_grid
      - the rest of the stitching algorithm stays unchanged

    If adaptive_grid=False:
      - uses the manually supplied 'grid'
    """
    if axis != 2:
        raise NotImplementedError(
            "This implementation currently assumes stitching along x (axis=2), "
            "with the grid defined in the z-y plane."
        )

    v1, s1 = apply_db_cutoff(vol1, cutoff_db)
    v2, s2 = apply_db_cutoff(vol2, cutoff_db)

    print(f"[Pre-Process] Applied {cutoff_db} dB cutoff.")
    print(f" -> Vol1 Sparsity: {s1:.1f}% | Vol2 Sparsity: {s2:.1f}%")

    if adaptive_grid:
        grid, grid_info = estimate_adaptive_grid(
            vol1,
            vol2,
            binary_threshold=binary_threshold,
            ignore_top=ignore_top,
            tile_multiple=tile_multiple,
            min_hotspot_voxels=min_hotspot_voxels,
            opening_structure=np.ones((3, 3, 3), dtype=bool),
            size_statistic=size_statistic,
            min_grid=min_grid,
            max_grid=max_grid,
        )

        print("\n[Adaptive Grid]")
        print(f" -> Binary hotspot threshold: {grid_info['binary_threshold']:.3f}")
        print(
            f" -> Representative hotspot size ({grid_info['size_statistic']}) "
            f"(z, y): ({grid_info['representative_hotspot_z']:.2f}, "
            f"{grid_info['representative_hotspot_y']:.2f})"
        )
        print(f" -> Tile multiple (z, y): {grid_info['tile_multiple']}")
        print(
            f" -> Estimated tile size from hotspot sizing (z, y): "
            f"({grid_info['tile_z_est']}, {grid_info['tile_y_est']})"
        )
        print(f" -> Raw grid from hotspot sizing: {grid_info['grid_raw']}")
        print(f" -> Grid limits: min={grid_info['min_grid']} max={grid_info['max_grid']}")
        print(f" -> Final clamped grid (rows, cols): {grid}")
        print(
            f" -> Hotspots detected: {grid_info['num_hotspots_total']} "
            f"(Vol1={grid_info['num_hotspots_vol1']}, Vol2={grid_info['num_hotspots_vol2']})"
        )
    else:
        if grid is None:
            raise ValueError("Provide 'grid' when adaptive_grid=False.")
        grid_info = None

    z_dim, y_dim, _ = v1.shape
    z_start, z_end = ignore_top, z_dim

    tile_z = (z_end - z_start) // grid[0]
    tile_y = y_dim // grid[1]

    if tile_z <= 0 or tile_y <= 0:
        raise ValueError(
            f"Grid {grid} is too fine for volume shape {v1.shape} after ignore_top={ignore_top}."
        )

    all_shifts = []
    all_weights = []
    tile_vote_map = np.full(grid, np.nan, dtype=float)

    for r in range(grid[0]):
        for c in range(grid[1]):
            zs = z_start + (r * tile_z)
            ze = z_start + ((r + 1) * tile_z) if r < grid[0] - 1 else z_end

            ys = c * tile_y
            ye = (c + 1) * tile_y if c < grid[1] - 1 else y_dim

            tile1 = v1[zs:ze, ys:ye, :]
            tile2 = v2[zs:ze, ys:ye, :]

            if tile1.size == 0 or tile2.size == 0:
                continue

            prof1 = np.mean(np.abs(tile1), axis=(0, 1))
            prof2 = np.mean(np.abs(tile2), axis=(0, 1))

            if np.std(prof1) < 1e-6 or np.max(prof1) == 0:
                continue
            if np.std(prof2) < 1e-6 or np.max(prof2) == 0:
                continue

            p1_n = (prof1 - np.mean(prof1)) / (np.std(prof1) + 1e-10)
            p2_n = (prof2 - np.mean(prof2)) / (np.std(prof2) + 1e-10)

            corr = correlate(p1_n, p2_n, mode="full")
            lags = correlation_lags(len(p1_n), len(p2_n), mode="full")

            mask = (lags >= expected - tolerance) & (lags <= expected + tolerance)
            if not np.any(mask):
                continue

            corr[~mask] = -np.inf

            peak_idx = np.argmax(corr)
            chosen_lag = int(lags[peak_idx])
            chosen_weight = float(corr[peak_idx])

            if not np.isfinite(chosen_weight):
                continue

            all_shifts.append(chosen_lag)
            all_weights.append(chosen_weight)
            tile_vote_map[r, c] = chosen_lag

    if not all_shifts:
        raise ValueError(f"No valid tile features survived the {cutoff_db} dB cutoff.")

    lag_min, lag_max = np.min(all_shifts), np.max(all_shifts)
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

    print(f"\nShift: {final_shift} voxels")
    print(f"Participating tiles: {len(all_shifts)} / {grid[0] * grid[1]}")
    print(f"Tiles voting for chosen shift: {np.sum(np.array(all_shifts) == final_shift)}")
    print(f"Weighted score at chosen shift: {weighted_scores[shift_to_idx[final_shift]]:.3f}")

    diagnostics = {
        "shift_axis": shift_axis,
        "weighted_scores": weighted_scores,
        "vote_counts": vote_counts,
        "final_shift": final_shift,
        "tile_vote_map": tile_vote_map,
        "tile_distance_map": tile_distance_map,
        "all_shifts": np.array(all_shifts, dtype=int),
        "all_weights": np.array(all_weights, dtype=float),
        "grid": grid,
        "tile_z": tile_z,
        "tile_y": tile_y,
        "adaptive_grid_info": grid_info,
    }

    return final_shift, v1, v2, diagnostics

def plot_tile_signal_alignment(v1, v2, diagnostics, num_tiles=5):
    """
    Plots 1D signals from tiles that voted for the final shift only.
    Overlays signals after applying the computed stitch shift.
    """
    grid = diagnostics["grid"]
    final_shift = diagnostics["final_shift"]

    z_dim, y_dim, x_dim = v1.shape
    ignore_top = 30
    z_start, z_end = ignore_top, z_dim

    tile_z = diagnostics["tile_z"]
    tile_y = diagnostics["tile_y"]

    tile_votes = diagnostics["tile_vote_map"]

    # 🔥 Only keep tiles that chose the final shift
    matching_tiles = np.argwhere(tile_votes == final_shift)

    if len(matching_tiles) == 0:
        print("No tiles voted for the final shift.")
        return

    # Random selection if too many
    np.random.shuffle(matching_tiles)
    selected_tiles = matching_tiles[:num_tiles]

    plt.figure(figsize=(12, 3 * len(selected_tiles)))

    for i, (r, c) in enumerate(selected_tiles):
        zs = z_start + (r * tile_z)
        ze = z_start + ((r + 1) * tile_z) if r < grid[0] - 1 else z_end

        ys = c * tile_y
        ye = (c + 1) * tile_y if c < grid[1] - 1 else y_dim

        tile1 = v1[zs:ze, ys:ye, :]
        tile2 = v2[zs:ze, ys:ye, :]

        prof1 = np.mean(np.abs(tile1), axis=(0, 1))
        prof2 = np.mean(np.abs(tile2), axis=(0, 1))

        # Normalise
        p1_n = (prof1 - np.mean(prof1)) / (np.std(prof1) + 1e-10)
        p2_n = (prof2 - np.mean(prof2)) / (np.std(prof2) + 1e-10)

        x = np.arange(len(p1_n))
        shifted_x = x + final_shift

        plt.subplot(len(selected_tiles), 1, i + 1)
        plt.plot(x, p1_n, label="Vol 1", linewidth=2)
        plt.plot(shifted_x, p2_n, label="Vol 2 (shifted)", linestyle="--")

        plt.xlabel("x (voxels)")
        plt.ylabel("Normalised Intensity")
        plt.legend()
        plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()
# ==========================================
# 3. STANDALONE STITCHER EXECUTION
# ==========================================
if __name__ == "__main__":
    IN_DIR = Path.cwd() / "DATA" / "2D TFM Data" / "FeC Smile 3MHz 04022026 Filtered"

    vol1_raw = np.load(IN_DIR / "FeC_40_2_filtered_3D_TFM.npy")
    vol2_raw = np.load(IN_DIR / "FeC_40_3_filtered_3D_TFM.npy")

    stitch_shift, v1_thresholded, v2_thresholded, diagnostics = run_stitcher_test(
        vol1_raw,
        vol2_raw,
        cutoff_db=-10,             # threshold used for stitching signal cleanup
        binary_threshold=0.90,     # threshold used for 3D hotspot mask generation (0 to 1)
        adaptive_grid=True,
        tile_multiple=(2.0, 2.0),
        min_hotspot_voxels=10,
        ignore_top=30,
        size_statistic="median",
        min_grid=(10, 10),         # coarsest allowed grid
        max_grid=(100, 100),       # finest allowed grid
        expected=0,
        tolerance=200,
    )

    plot_stitcher_diagnostics(diagnostics)
    plot_tile_signal_alignment(v1_thresholded, v2_thresholded, diagnostics, num_tiles=1)

    if diagnostics["adaptive_grid_info"] is not None:
        plot_adaptive_grid_diagnostics(diagnostics["adaptive_grid_info"])

    clim_raw_1 = sorted([
        float(np.percentile(vol1_raw, 0.1)),
        float(np.percentile(vol1_raw, 99.9)),
    ])
    if clim_raw_1[0] == clim_raw_1[1]:
        clim_raw_1 = [clim_raw_1[0], clim_raw_1[0] + 1]

    clim_raw_2 = sorted([
        float(np.percentile(vol2_raw, 0.1)),
        float(np.percentile(vol2_raw, 99.9)),
    ])
    if clim_raw_2[0] == clim_raw_2[1]:
        clim_raw_2 = [clim_raw_2[0], clim_raw_2[0] + 1]

    clim_thresh_1 = sorted([
        float(np.min(v1_thresholded)),
        float(np.max(v1_thresholded)),
    ])
    if clim_thresh_1[0] == clim_thresh_1[1]:
        clim_thresh_1 = [0, 1]

    clim_thresh_2 = sorted([
        float(np.min(v2_thresholded)),
        float(np.max(v2_thresholded)),
    ])
    if clim_thresh_2[0] == clim_thresh_2[1]:
        clim_thresh_2 = [0, 1]

    viewer = napari.Viewer(title="Stitcher Result Testing with Adaptive Grid")

    viewer.add_image(
        vol1_raw,
        name="Vol 1 (Raw)",
        colormap="cyan",
        contrast_limits=clim_raw_1,
        opacity=0.35,
    )
    viewer.add_image(
        v1_thresholded,
        name="Vol 1 (Thresholded)",
        colormap="yellow",
        contrast_limits=clim_thresh_1,
        opacity=1.0,
    )

    trans = [0, 0, 0]
    trans[2] = stitch_shift

    viewer.add_image(
        vol2_raw,
        name=f"Vol 2 Raw (Shifted {stitch_shift}px)",
        colormap="magenta",
        blending="additive",
        translate=trans,
        contrast_limits=clim_raw_2,
        opacity=0.35,
    )
    viewer.add_image(
        v2_thresholded,
        name=f"Vol 2 Thresholded (Shifted {stitch_shift}px)",
        colormap="red",
        blending="additive",
        translate=trans,
        contrast_limits=clim_thresh_2,
        opacity=1.0,
    )

    print("\nStitcher complete.")
    print(f"Final adaptive grid: {diagnostics['grid']}")
    print(f"Actual tile size used: ({diagnostics['tile_z']}, {diagnostics['tile_y']})")
    print(f"Vol 1 threshold limits: {clim_thresh_1}")
    print(f"Vol 2 threshold limits: {clim_thresh_2}")


    napari.run()