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

            corr = correlate(p1_n, p2_n, mode="full")
            lags = correlation_lags(len(p1_n), len(p2_n), mode="full")

            N = len(p1_n)
            M = len(p2_n)
            overlap = np.minimum(N, M + lags) - np.maximum(0, lags)
            overlap = np.maximum(overlap, 1)
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
# 2. OVERLAP EXTRACTION FOR FSC
# ==========================================
def extract_overlap_after_shift(vol1, vol2, shift, axis=2):
    """
    Extract matching overlapping sub-volumes after shifting vol2 by 'shift' pixels
    along the specified axis.

    Positive shift means vol2 moves toward larger indices along 'axis'.
    """
    if axis != 2:
        raise NotImplementedError("This helper currently assumes stitching along axis=2.")

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


def moving_average_nan(x, window=5):
    """Simple NaN-aware moving average."""
    if window <= 1:
        return x.copy()

    x = np.asarray(x, dtype=np.float64)
    out = np.full_like(x, np.nan)
    half = window // 2

    for i in range(len(x)):
        lo = max(0, i - half)
        hi = min(len(x), i + half + 1)
        vals = x[lo:hi]
        vals = vals[np.isfinite(vals)]
        if len(vals) > 0:
            out[i] = np.mean(vals)

    return out


# ==========================================
# 3. FSC COMPUTATION
# ==========================================
def compute_fsc(
    vol1,
    vol2,
    voxel_size=1.0,
    threshold=1 / 7,
    apply_hann=True,
    min_shell_count=50,
    smooth_window=5,
    max_freq_fraction=0.95,
):
    """
    Compute Fourier Shell Correlation (FSC) between two 3D volumes.

    Improvements:
    - ignores shells with too few Fourier samples
    - ignores the extreme high-frequency edge where FSC is unstable
    - uses a smoothed curve for threshold crossing
    """
    if vol1.shape != vol2.shape:
        raise ValueError("FSC volumes must have identical shape.")

    v1 = np.asarray(vol1, dtype=np.float32)
    v2 = np.asarray(vol2, dtype=np.float32)

    # Mean-center
    v1 = v1 - np.mean(v1)
    v2 = v2 - np.mean(v2)

    if apply_hann:
        wz = np.hanning(v1.shape[0])[:, None, None]
        wy = np.hanning(v1.shape[1])[None, :, None]
        wx = np.hanning(v1.shape[2])[None, None, :]
        window = wz * wy * wx
        v1 = v1 * window
        v2 = v2 * window

    F1 = np.fft.fftn(v1)
    F2 = np.fft.fftn(v2)

    nz, ny, nx = v1.shape
    kz = np.fft.fftfreq(nz, d=voxel_size)
    ky = np.fft.fftfreq(ny, d=voxel_size)
    kx = np.fft.fftfreq(nx, d=voxel_size)

    KZ, KY, KX = np.meshgrid(kz, ky, kx, indexing="ij")
    kr = np.sqrt(KZ**2 + KY**2 + KX**2)

    dk = min(
        abs(kz[1] - kz[0]) if nz > 1 else np.inf,
        abs(ky[1] - ky[0]) if ny > 1 else np.inf,
        abs(kx[1] - kx[0]) if nx > 1 else np.inf,
    )
    if not np.isfinite(dk) or dk <= 0:
        raise ValueError("Invalid voxel spacing or volume shape for FSC.")

    shell_idx = np.floor(kr / dk).astype(np.int32)
    n_shells = shell_idx.max() + 1

    num = np.zeros(n_shells, dtype=np.complex128)
    p1 = np.zeros(n_shells, dtype=np.float64)
    p2 = np.zeros(n_shells, dtype=np.float64)
    counts = np.zeros(n_shells, dtype=np.int64)

    f1_flat = F1.ravel()
    f2_flat = F2.ravel()
    shell_flat = shell_idx.ravel()

    for s in range(n_shells):
        mask = shell_flat == s
        if not np.any(mask):
            continue
        a = f1_flat[mask]
        b = f2_flat[mask]
        num[s] = np.sum(a * np.conj(b))
        p1[s] = np.sum(np.abs(a) ** 2)
        p2[s] = np.sum(np.abs(b) ** 2)
        counts[s] = mask.sum()

    freqs = np.arange(n_shells) * dk
    nyquist = 1.0 / (2.0 * voxel_size)
    max_valid_freq = max_freq_fraction * nyquist

    valid = (
        (counts >= min_shell_count)
        & (p1 > 0)
        & (p2 > 0)
        & (freqs <= max_valid_freq)
    )

    fsc_raw = np.full(n_shells, np.nan, dtype=np.float64)
    fsc_raw[valid] = np.real(num[valid] / np.sqrt(p1[valid] * p2[valid]))

    # Smoothed FSC used for threshold crossing only
    fsc_smooth = moving_average_nan(fsc_raw, window=smooth_window)

    # Resolution from smoothed curve, ignoring DC shell
    resolution = None
    cutoff_freq = None
    for i in range(1, len(fsc_smooth)):
        if np.isfinite(fsc_smooth[i]) and fsc_smooth[i] < threshold:
            if freqs[i] > 0:
                cutoff_freq = freqs[i]
                resolution = 1.0 / cutoff_freq
            break

    return {
        "freqs": freqs,
        "fsc_raw": fsc_raw,
        "fsc_smooth": fsc_smooth,
        "counts": counts,
        "resolution": resolution,
        "cutoff_freq": cutoff_freq,
        "nyquist": nyquist,
    }


def plot_fsc_comparison(
    result_before,
    result_after,
    threshold=1 / 7,
    title="FSC Comparison: Before vs After Stitching",
):
    plt.figure(figsize=(9, 6))

    freqs_b = result_before["freqs"]
    freqs_a = result_after["freqs"]

    # Raw curves
    plt.plot(freqs_b, result_before["fsc_raw"], alpha=0.35, lw=1.5, label="Before (raw)")
    plt.plot(freqs_a, result_after["fsc_raw"], alpha=0.35, lw=1.5, label="After (raw)")

    # Smoothed curves
    plt.plot(freqs_b, result_before["fsc_smooth"], lw=2.5, label="Before (smoothed)")
    plt.plot(freqs_a, result_after["fsc_smooth"], lw=2.5, label="After (smoothed)")

    plt.axhline(threshold, color="red", linestyle="--", label=f"Threshold = {threshold:.3f}")

    if result_before["cutoff_freq"] is not None:
        plt.axvline(
            result_before["cutoff_freq"],
            linestyle=":",
            linewidth=2,
            label=f"Before res = {result_before['resolution']:.2f}",
        )

    if result_after["cutoff_freq"] is not None:
        plt.axvline(
            result_after["cutoff_freq"],
            linestyle=":",
            linewidth=2,
            label=f"After res = {result_after['resolution']:.2f}",
        )

    plt.xlabel("Spatial frequency")
    plt.ylabel("FSC")
    plt.title(title)
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ==========================================
# 4. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    DATA_ROOT = Path.cwd() / "DATA" / "2D TFM Data"
    FILTERED_DIR = DATA_ROOT / "FeC Smile 3MHz 04022026 Filtered"
    RAW_DIR = DATA_ROOT / "FeC Smile 3MHz 04022026"

    try:
        v1_filt = np.load(FILTERED_DIR / "FeC_40_3_filtered_3D_TFM.npy")
        v2_filt = np.load(FILTERED_DIR / "FeC_40_4_filtered_3D_TFM.npy")

        v1_raw = np.load(RAW_DIR / "FeC_40_3_3D_TFM.npy")
        v2_raw = np.load(RAW_DIR / "FeC_40_4_3D_TFM.npy")

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

        print(f"[Check] Dimensions match: {v1_raw.shape}")

    except FileNotFoundError as e:
        print(f"Data not found: {e}")
        raise SystemExit
    except ValueError as e:
        print(e)
        raise SystemExit

    # 1. Calculate shift using filtered volumes
    stitch_shift = run_stitcher_test(v1_filt, v2_filt, grid=(120, 40), cutoff_db=-10)
    print(f"\nCalculated Shift: {stitch_shift} px")

    # 2. Overlap BEFORE stitching
    ov1_before, ov2_before = extract_overlap_after_shift(v1_raw, v2_raw, 0, axis=2)

    # 3. Overlap AFTER stitching
    ov1_after, ov2_after = extract_overlap_after_shift(v1_raw, v2_raw, stitch_shift, axis=2)
    print(f"Overlap volume shape after stitching: {ov1_after.shape}")

    # 4. Compute FSC before and after
    voxel_size = 1.0  # replace with real voxel spacing if known

    fsc_before = compute_fsc(
        ov1_before,
        ov2_before,
        voxel_size=voxel_size,
        threshold=1 / 7,
        apply_hann=True,
        min_shell_count=50,
        smooth_window=5,
        max_freq_fraction=0.95,
    )

    fsc_after = compute_fsc(
        ov1_after,
        ov2_after,
        voxel_size=voxel_size,
        threshold=1 / 7,
        apply_hann=True,
        min_shell_count=50,
        smooth_window=5,
        max_freq_fraction=0.95,
    )

    if fsc_before["resolution"] is None:
        print("Before stitching: FSC did not cross threshold.")
    else:
        print(f"Before stitching FSC resolution: {fsc_before['resolution']:.2f} voxels")

    if fsc_after["resolution"] is None:
        print("After stitching: FSC did not cross threshold.")
    else:
        print(f"After stitching FSC resolution: {fsc_after['resolution']:.2f} voxels")

    plot_fsc_comparison(
        fsc_before,
        fsc_after,
        threshold=1 / 7,
        title="FSC Comparison of Overlap Region",
    )

    # 5. Visualize stitched volumes
    clim_raw = sorted([float(np.percentile(v1_raw, 0.1)), float(np.percentile(v1_raw, 99.9))])
    if clim_raw[0] == clim_raw[1]:
        clim_raw = [0, 1]

    viewer = napari.Viewer(title="Cross-Dimensionality Stitching + FSC Validation")
    viewer.add_image(v1_raw, name="Vol 1 (High-Res)", colormap="cyan", contrast_limits=clim_raw)

    trans = [0, 0, 0]
    trans[2] = stitch_shift
    viewer.add_image(
        v2_raw,
        name="Vol 2 (High-Res Shifted)",
        colormap="magenta",
        blending="additive",
        translate=trans,
        contrast_limits=clim_raw,
    )

    print(f"Viewing High-Dimensionality assembly. Shift applied: {stitch_shift} px.")
    napari.run()