import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate, correlation_lags
from pathlib import Path
import napari

# ==========================================
# 0. PRE-PROCESSING UTILITIES
# ==========================================
def apply_db_cutoff(vol, cutoff_db=-5):
    v_abs = np.abs(vol)
    v_max = np.max(v_abs)
    if v_max == 0:
        return vol, 0.0
    thresh = v_max * (10 ** (cutoff_db / 20))
    v_thresh = np.where(v_abs >= thresh, vol, 0)
    sparsity = (np.count_nonzero(v_thresh == 0) / v_thresh.size) * 100
    return v_thresh.astype(np.float32), sparsity


# ==========================================
# 1. THE STITCHER ENGINE
# ==========================================
def run_stitcher_test(vol1, vol2, axis=2, grid=(60, 20), expected=0, tolerance=200, cutoff_db=-5):
    """Finds shift using filtered volumes."""
    v1, _ = apply_db_cutoff(vol1, cutoff_db)
    v2, _ = apply_db_cutoff(vol2, cutoff_db)

    ignore_top = 30
    z_dim, y_dim, x_dim = v1.shape
    tile_z = (z_dim - ignore_top) // grid[0]
    tile_y = y_dim // grid[1]

    all_shifts, all_weights = [], []

    for r in range(grid[0]):
        for c in range(grid[1]):
            zs, ze = ignore_top + (r * tile_z), ignore_top + ((r + 1) * tile_z)
            ys, ye = c * tile_y, (c + 1) * tile_y

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

            all_shifts.append(lags[np.argmax(corr)])
            all_weights.append(np.max(corr))

    if not all_shifts:
        raise ValueError(f"No features survived the {cutoff_db} dB cutoff.")

    bins = np.arange(np.min(all_shifts), np.max(all_shifts) + 2) - 0.5
    counts, bin_edges = np.histogram(all_shifts, bins=bins, weights=all_weights)
    return int(bin_edges[np.argmax(counts)] + 0.5)


# ==========================================
# 2. OVERLAP EXTRACTION
# ==========================================
def extract_overlap_after_shift(vol1, vol2, shift, axis=2):
    """
    Extract matching overlapping sub-volumes after shifting vol2 by 'shift' pixels
    along the specified axis.

    Positive shift means vol2 moves toward larger indices along 'axis'.
    """
    if axis != 2:
        raise NotImplementedError("This helper currently assumes stitching along axis=2 (x-axis).")

    if vol1.shape != vol2.shape:
        raise ValueError("Volumes must have the same shape for overlap extraction.")

    x_dim = vol1.shape[2]

    if shift >= 0:
        if shift >= x_dim:
            raise ValueError("Shift is too large; no overlap remains.")
        ov1 = vol1[:, :, shift:]
        ov2 = vol2[:, :, :x_dim - shift]
    else:
        s = -shift
        if s >= x_dim:
            raise ValueError("Shift is too large; no overlap remains.")
        ov1 = vol1[:, :, :x_dim - s]
        ov2 = vol2[:, :, s:]

    return ov1, ov2


# ==========================================
# 3. VALIDATION METRICS
# ==========================================
def compute_difference_metrics(vol1, vol2, eps=1e-10):
    """
    Compute global percentage difference metrics between two volumes.
    """

    if vol1.shape != vol2.shape:
        raise ValueError("Volumes must have identical shapes")

    v1 = vol1.astype(np.float64)
    v2 = vol2.astype(np.float64)

    diff = np.abs(v1 - v2)

    reference = (np.abs(v1) + np.abs(v2)) / 2.0

    percent_diff = 100 * diff / (reference + eps)

    rms_percent = np.sqrt(np.mean(percent_diff ** 2))
    mad_percent = np.median(percent_diff)
    max_percent = np.max(percent_diff)

    return {
        "rms_percent": float(rms_percent),
        "mad_percent": float(mad_percent),
        "max_percent": float(max_percent),
    }

def compute_tile_metrics(vol1, vol2, grid=(60,20), ignore_top=30, eps=1e-10):

    z_dim, y_dim, x_dim = vol1.shape
    grid_z, grid_y = grid

    usable_z = z_dim - ignore_top
    tile_z = usable_z // grid_z
    tile_y = y_dim // grid_y

    rms_map = np.zeros((grid_z, grid_y))
    mad_map = np.zeros((grid_z, grid_y))
    max_map = np.zeros((grid_z, grid_y))

    for r in range(grid_z):
        for c in range(grid_y):

            zs = ignore_top + r * tile_z
            ze = ignore_top + (r + 1) * tile_z

            ys = c * tile_y
            ye = (c + 1) * tile_y

            t1 = vol1[zs:ze, ys:ye, :]
            t2 = vol2[zs:ze, ys:ye, :]

            diff = np.abs(t1 - t2)
            ref = (np.abs(t1) + np.abs(t2)) / 2.0

            percent = 100 * diff / (ref + eps)

            rms_map[r,c] = np.sqrt(np.mean(percent**2))
            mad_map[r,c] = np.median(percent)
            max_map[r,c] = np.max(percent)

    return {
        "rmsd_map": rms_map,
        "mad_map": mad_map,
        "max_map": max_map
    }


# ==========================================
# 4. ERROR VOLUMES FOR NAPARI
# ==========================================
def compute_error_volumes(vol1, vol2):
    """
    Compute voxelwise error volumes for 3D visualization.
    """
    if vol1.shape != vol2.shape:
        raise ValueError("Volumes must have identical shapes for error-volume computation.")

    diff = vol1.astype(np.float32) - vol2.astype(np.float32)
    abs_diff = np.abs(diff)
    sq_diff = diff ** 2

    return {
        "diff": diff,
        "abs_diff": abs_diff,
        "sq_diff": sq_diff,
    }


def expand_tile_map_to_volume(tile_map, vol_shape, grid=(60, 20), ignore_top=30):
    """
    Expand a 2D tile metric map back into a 3D volume so it can be viewed in napari.
    Each tile value is repeated across the tile's z-y region and across all x.
    """
    z_dim, y_dim, x_dim = vol_shape
    grid_z, grid_y = grid

    usable_z = z_dim - ignore_top
    tile_z = usable_z // grid_z
    tile_y = y_dim // grid_y

    volume = np.zeros(vol_shape, dtype=np.float32)

    for r in range(grid_z):
        for c in range(grid_y):
            zs = ignore_top + r * tile_z
            ze = ignore_top + (r + 1) * tile_z if r < grid_z - 1 else z_dim
            ys = c * tile_y
            ye = (c + 1) * tile_y if c < grid_y - 1 else y_dim

            value = tile_map[r, c]
            if np.isnan(value):
                value = 0.0

            volume[zs:ze, ys:ye, :] = value

    return volume


# ==========================================
# 5. PLOTTING
# ==========================================
def safe_contrast_limits(arr, percentile=99.5, default_high=1.0):
    """
    Return napari-safe contrast limits.
    If the data are constant or invalid, fall back to [0, default_high].
    """
    arr = np.asarray(arr, dtype=np.float32)

    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return [0.0, default_high]

    high = float(np.percentile(finite, percentile))
    low = 0.0

    if high <= low:
        max_val = float(np.max(finite))
        min_val = float(np.min(finite))

        if max_val > min_val:
            return [min_val, max_val]

        if max_val > 0:
            return [0.0, max_val]

        return [0.0, default_high]

    return [low, high]

def plot_metric_heatmaps(tile_metrics, title_prefix=""):
    """
    Plot heatmaps for RMSD, MAD, and max difference.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

    metric_info = [
    ("rmsd_map","RMS % difference"),
    ("mad_map","Median % difference"),
    ("max_map","Max % difference")
    ]

    for ax, (key, label) in zip(axes, metric_info):
        data = tile_metrics[key]
        im = ax.imshow(data, aspect="auto", origin="upper")
        ax.set_title(f"{title_prefix} {label}".strip())
        ax.set_xlabel("Tile column (y)")
        ax.set_ylabel("Tile row (z)")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.show()


def print_global_metrics(name, metrics):
    print(f"\n{name} GLOBAL METRICS")
    print("-" * (len(name) + 15))
    print(f"RMS % diff : {metrics['rms_percent']:.2f}%")
    print(f"MAD % diff : {metrics['mad_percent']:.2f}%")
    print(f"Max % diff : {metrics['max_percent']:.2f}%")


def print_worst_tiles(name, tile_metrics, top_n=10):
    print(f"\n{name} WORST TILES")
    print("-" * (len(name) + 12))

    for metric_key, metric_label in [
        ("rmsd_map", "RMSD"),
        ("mad_map", "MAD"),
        ("max_map", "Max |ΔI|"),
    ]:
        data = tile_metrics[metric_key]
        flat_idx = np.argsort(data, axis=None)[::-1]

        print(f"\nTop {top_n} tiles by {metric_label}:")
        count = 0
        for idx in flat_idx:
            value = data.flat[idx]
            if np.isnan(value):
                continue
            r, c = np.unravel_index(idx, data.shape)
            print(f"  Tile (row={r}, col={c}) : {value:.6g}")
            count += 1
            if count >= top_n:
                break


# ==========================================
# 6. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    DATA_ROOT = Path.cwd() / "DATA" / "2D TFM Data"
    FILTERED_DIR = DATA_ROOT / "FeC Smile 3MHz 04022026 Filtered"
    RAW_DIR = DATA_ROOT / "FeC Smile 3MHz 04022026"

    # Use the same grid for stitching and tile validation
    grid = (120, 40)
    cutoff_db = -10
    ignore_top = 30

    try:
        # LOAD BOTH DATASETS
        v1_filt = np.load(FILTERED_DIR / "FeC_40_3_filtered_3D_TFM.npy")
        v2_filt = np.load(FILTERED_DIR / "FeC_40_4_filtered_3D_TFM.npy")

        v1_raw = np.load(RAW_DIR / "FeC_40_3_3D_TFM.npy")
        v2_raw = np.load(RAW_DIR / "FeC_40_4_3D_TFM.npy")

        # --- SHAPE CHECK ---
        if v1_filt.shape != v1_raw.shape:
            raise ValueError(
                f"DIMENSION MISMATCH DETECTED!\n"
                f"Filtered Data: {v1_filt.shape}\n"
                f"Raw Data:      {v1_raw.shape}\n"
                "The shift from filtered data cannot be applied to raw data of a different size."
            )

        if v1_raw.shape != v2_raw.shape:
            raise ValueError(
                f"RAW VOLUME SHAPE MISMATCH!\n"
                f"Vol 1 Raw: {v1_raw.shape}\n"
                f"Vol 2 Raw: {v2_raw.shape}"
            )

        print(f"[Check] Dimensions match: {v1_raw.shape}. Proceeding...")

    except FileNotFoundError as e:
        print(f"Data not found: {e}")
        raise SystemExit
    except ValueError as e:
        print(e)
        raise SystemExit

    # 1. Calculate shift using the filtered volumes
    stitch_shift = run_stitcher_test(v1_filt, v2_filt, grid=grid, cutoff_db=cutoff_db)
    print(f"\nCalculated Shift: {stitch_shift} px")

    # 2. Prepare filtered validation volumes
    v1_filt_cut, sparsity1 = apply_db_cutoff(v1_filt, cutoff_db=cutoff_db)
    v2_filt_cut, sparsity2 = apply_db_cutoff(v2_filt, cutoff_db=cutoff_db)

    # 3. Extract overlap after shift for RAW and FILTERED
    ov1_raw, ov2_raw = extract_overlap_after_shift(v1_raw, v2_raw, stitch_shift, axis=2)
    ov1_filt, ov2_filt = extract_overlap_after_shift(v1_filt_cut, v2_filt_cut, stitch_shift, axis=2)

    print(f"Raw overlap shape      : {ov1_raw.shape}")
    print(f"Filtered overlap shape : {ov1_filt.shape}")
    print(f"Filtered sparsity Vol 1: {sparsity1:.2f}%")
    print(f"Filtered sparsity Vol 2: {sparsity2:.2f}%")

    # 4. Global metrics
    raw_global = compute_difference_metrics(ov1_raw, ov2_raw)
    filt_global = compute_difference_metrics(ov1_filt, ov2_filt)

    print_global_metrics("RAW", raw_global)
    print_global_metrics("FILTERED", filt_global)

    # 5. Tile metrics
    raw_tiles = compute_tile_metrics(ov1_raw, ov2_raw, grid=grid, ignore_top=ignore_top)
    filt_tiles = compute_tile_metrics(ov1_filt, ov2_filt, grid=grid, ignore_top=ignore_top)

    print_worst_tiles("RAW", raw_tiles, top_n=10)
    print_worst_tiles("FILTERED", filt_tiles, top_n=10)

    # 6. Plot heatmaps
    plot_metric_heatmaps(raw_tiles, title_prefix="RAW")
    plot_metric_heatmaps(filt_tiles, title_prefix="FILTERED")

    # 7. Error volumes for napari
    raw_error = compute_error_volumes(ov1_raw, ov2_raw)
    filt_error = compute_error_volumes(ov1_filt, ov2_filt)

    raw_rmsd_vol = expand_tile_map_to_volume(raw_tiles["rmsd_map"], ov1_raw.shape, grid=grid, ignore_top=ignore_top)
    raw_mad_vol = expand_tile_map_to_volume(raw_tiles["mad_map"], ov1_raw.shape, grid=grid, ignore_top=ignore_top)
    raw_max_vol = expand_tile_map_to_volume(raw_tiles["max_map"], ov1_raw.shape, grid=grid, ignore_top=ignore_top)

    filt_rmsd_vol = expand_tile_map_to_volume(filt_tiles["rmsd_map"], ov1_filt.shape, grid=grid, ignore_top=ignore_top)
    filt_mad_vol = expand_tile_map_to_volume(filt_tiles["mad_map"], ov1_filt.shape, grid=grid, ignore_top=ignore_top)
    filt_max_vol = expand_tile_map_to_volume(filt_tiles["max_map"], ov1_filt.shape, grid=grid, ignore_top=ignore_top)

    # 8. Visualize stitched raw volumes plus 3D error maps in napari
    clim_raw = sorted([float(np.percentile(v1_raw, 0.1)), float(np.percentile(v1_raw, 99.9))])
    if clim_raw[0] == clim_raw[1]:
        clim_raw = [0, 1]

    # Contrast limits for overlap-based error layers
    raw_abs_clim = safe_contrast_limits(raw_error["abs_diff"])
    raw_sq_clim = safe_contrast_limits(raw_error["sq_diff"])
    filt_abs_clim = safe_contrast_limits(filt_error["abs_diff"])
    filt_sq_clim = safe_contrast_limits(filt_error["sq_diff"])

    raw_rmsd_clim = safe_contrast_limits(raw_rmsd_vol)
    raw_mad_clim = safe_contrast_limits(raw_mad_vol)
    raw_max_clim = safe_contrast_limits(raw_max_vol)

    filt_rmsd_clim = safe_contrast_limits(filt_rmsd_vol)
    filt_mad_clim = safe_contrast_limits(filt_mad_vol)
    filt_max_clim = safe_contrast_limits(filt_max_vol)

    viewer = napari.Viewer(title="Cross-Dimensionality Stitching + 3D Validation")

    # Original stitched raw volumes
    viewer.add_image(
        v1_raw,
        name="Raw Vol 1",
        colormap="cyan",
        contrast_limits=clim_raw,
    )

    trans = [0, 0, 0]
    trans[2] = stitch_shift
    viewer.add_image(
        v2_raw,
        name="Raw Vol 2 Shifted",
        colormap="magenta",
        blending="additive",
        translate=trans,
        contrast_limits=clim_raw,
    )

    # Raw overlap error volumes
    viewer.add_image(
        raw_error["abs_diff"],
        name="Raw |ΔI|",
        colormap="inferno",
        contrast_limits=raw_abs_clim,
        visible=False,
    )
    viewer.add_image(
        raw_error["sq_diff"],
        name="Raw ΔI²",
        colormap="magma",
        contrast_limits=raw_sq_clim,
        visible=False,
    )

    # Raw tile-expanded metric volumes
    viewer.add_image(
        raw_rmsd_vol,
        name="Raw Tile RMSD",
        colormap="viridis",
        contrast_limits=raw_rmsd_clim,
        visible=False,
    )
    viewer.add_image(
        raw_mad_vol,
        name="Raw Tile MAD",
        colormap="plasma",
        contrast_limits=raw_mad_clim,
        visible=False,
    )
    viewer.add_image(
        raw_max_vol,
        name="Raw Tile Max |ΔI|",
        colormap="turbo",
        contrast_limits=raw_max_clim,
        visible=False,
    )

    # Filtered overlap error volumes
    viewer.add_image(
        filt_error["abs_diff"],
        name="Filtered |ΔI|",
        colormap="inferno",
        contrast_limits=filt_abs_clim,
        visible=False,
    )
    viewer.add_image(
        filt_error["sq_diff"],
        name="Filtered ΔI²",
        colormap="magma",
        contrast_limits=filt_sq_clim,
        visible=False,
    )

    # Filtered tile-expanded metric volumes
    viewer.add_image(
        filt_rmsd_vol,
        name="Filtered Tile RMSD",
        colormap="viridis",
        contrast_limits=filt_rmsd_clim,
        visible=False,
    )
    viewer.add_image(
        filt_mad_vol,
        name="Filtered Tile MAD",
        colormap="plasma",
        contrast_limits=filt_mad_clim,
        visible=False,
    )
    viewer.add_image(
        filt_max_vol,
        name="Filtered Tile Max |ΔI|",
        colormap="turbo",
        contrast_limits=filt_max_clim,
        visible=False,
    )

    print(f"\nViewing High-Dimensionality assembly. Shift applied: {stitch_shift} px.")
    napari.run()