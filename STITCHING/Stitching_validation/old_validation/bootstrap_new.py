#%%

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 18,
    "axes.labelsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 14,
})


# ==========================================
# 0. STITCHING FUNCTIONS
# ==========================================
def normalised_correlation_3D(vol1, vol2, axis='x', max_shift=100):
    """
    Compute normalized cross-correlation between two volumes.

    Parameters
    ----------
    vol1, vol2 : np.ndarray
        Volumes with shape (z, x, y)
    axis : str
        Axis to shift along: 'x' or 'y'
    max_shift : int
        Maximum shift to test in both directions

    Returns
    -------
    best_shift : int
        Shift that gives maximum correlation
    shifts : np.ndarray
        All tested shifts
    corr_values : np.ndarray
        Correlation values for each shift
    """
    z1, x1, y1 = vol1.shape
    z2, x2, y2 = vol2.shape

    shifts = np.arange(-max_shift, max_shift + 1)
    corr_values = []

    for d in shifts:
        if axis == 'x':
            a1_start = max(0, d)
            a1_end = min(x1, x2 + d)

            a2_start = max(0, -d)
            a2_end = min(x2, x1 - d)

            if (a1_end - a1_start) <= 0:
                corr_values.append(0.0)
                continue

            region1 = vol1[:, a1_start:a1_end, :]
            region2 = vol2[:, a2_start:a2_end, :]

        elif axis == 'y':
            a1_start = max(0, d)
            a1_end = min(y1, y2 + d)

            a2_start = max(0, -d)
            a2_end = min(y2, y1 - d)

            if (a1_end - a1_start) <= 0:
                corr_values.append(0.0)
                continue

            region1 = vol1[:, :, a1_start:a1_end]
            region2 = vol2[:, :, a2_start:a2_end]

        else:
            raise ValueError("axis must be 'x' or 'y'")

        numerator = np.sum(region1 * region2)
        denom = np.sqrt(np.sum(region1 ** 2) * np.sum(region2 ** 2))

        corr_values.append(numerator / denom if denom > 0 else 0.0)

    corr_values = np.asarray(corr_values, dtype=float)
    best_index = int(np.argmax(corr_values))
    best_shift = int(shifts[best_index])

    return best_shift, shifts, corr_values


def stitch_volumes(vol1, vol2, shift, axis='x'):
    z1, x1, y1 = vol1.shape
    z2, x2, y2 = vol2.shape

    if axis == 'x':
        left_offset = max(0, -shift)
        right_extent = max(x1, x2 + shift)
        total_x = left_offset + right_extent

        canvas1 = np.zeros((z1, total_x, y1), dtype=vol1.dtype)
        canvas2 = np.zeros((z1, total_x, y1), dtype=vol2.dtype)

        canvas1[:, left_offset:left_offset + x1, :] = vol1

        x2_start = left_offset + shift
        canvas2[:, x2_start:x2_start + x2, :] = vol2

    elif axis == 'y':
        left_offset = max(0, -shift)
        right_extent = max(y1, y2 + shift)
        total_y = left_offset + right_extent

        canvas1 = np.zeros((z1, x1, total_y), dtype=vol1.dtype)
        canvas2 = np.zeros((z1, x1, total_y), dtype=vol2.dtype)

        canvas1[:, :, left_offset:left_offset + y1] = vol1

        y2_start = left_offset + shift
        canvas2[:, :, y2_start:y2_start + y2] = vol2

    else:
        raise ValueError("axis must be 'x' or 'y'")

    return canvas1, canvas2


# ==========================================
# 1. BOOTSTRAP UTILITIES
# ==========================================
def make_z_band_slices(z_dim, ignore_top=0, ignore_bottom=0, band_size=20):
    """
    Split the usable z-range into equal-size bootstrap bands.
    Any leftover remainder is preserved unchanged.
    """
    if band_size <= 0:
        raise ValueError("band_size must be > 0")

    z_start = min(ignore_top, z_dim)
    z_end = max(z_start, z_dim - min(ignore_bottom, z_dim - z_start))

    if z_start >= z_end:
        raise ValueError(
            f"No usable z-range after ignore_top={ignore_top}, "
            f"ignore_bottom={ignore_bottom}, z_dim={z_dim}"
        )

    usable = z_end - z_start
    n_full_bands = usable // band_size
    remainder = usable % band_size

    if n_full_bands == 0:
        raise ValueError(
            f"band_size={band_size} is larger than usable z-depth={usable}"
        )

    band_slices = []
    for i in range(n_full_bands):
        zs = z_start + i * band_size
        ze = zs + band_size
        band_slices.append(slice(zs, ze))

    remainder_slice = None
    if remainder > 0:
        remainder_slice = slice(z_start + n_full_bands * band_size, z_end)

    return band_slices, remainder_slice


def resample_volumes_by_z_band(vol1, vol2, band_slices, remainder_slice, rng):
    """
    Resample equal-size z-bands with replacement.
    Top ignored region, bottom ignored region, and any remainder are preserved.
    """
    if vol1.shape != vol2.shape:
        raise ValueError(f"Volume shapes do not match: {vol1.shape} vs {vol2.shape}")

    z_dim = vol1.shape[0]
    z_start = band_slices[0].start
    z_boot_end = band_slices[-1].stop

    sampled_indices = rng.integers(0, len(band_slices), size=len(band_slices))

    v1_parts = []
    v2_parts = []

    if z_start > 0:
        v1_parts.append(vol1[:z_start])
        v2_parts.append(vol2[:z_start])

    for idx in sampled_indices:
        s = band_slices[idx]
        v1_parts.append(vol1[s])
        v2_parts.append(vol2[s])

    if remainder_slice is not None:
        v1_parts.append(vol1[remainder_slice])
        v2_parts.append(vol2[remainder_slice])

    bottom_start = remainder_slice.stop if remainder_slice is not None else z_boot_end
    if bottom_start < z_dim:
        v1_parts.append(vol1[bottom_start:])
        v2_parts.append(vol2[bottom_start:])

    v1_boot = np.concatenate(v1_parts, axis=0)
    v2_boot = np.concatenate(v2_parts, axis=0)

    if v1_boot.shape != vol1.shape or v2_boot.shape != vol2.shape:
        raise RuntimeError(
            f"Bootstrapped volumes changed shape: "
            f"{v1_boot.shape}, {v2_boot.shape}, expected {vol1.shape}"
        )

    return v1_boot, v2_boot, sampled_indices


# ==========================================
# 2. BOOTSTRAP RUNNER
# ==========================================
def bootstrap_stitch_shift(
    vol1,
    vol2,
    n_boot=1000,
    *,
    axis="x",
    max_shift=100,
    ignore_top=0,
    ignore_bottom=0,
    band_size=20,
    rng_seed=42,
    verbose=True,
):
    """
    Bootstrap stitch shift by resampling z-bands and rerunning
    normalised_correlation_3D on each replicate.
    """
    rng = np.random.default_rng(rng_seed)

    if vol1.shape != vol2.shape:
        raise ValueError(f"Volume shapes do not match: {vol1.shape} vs {vol2.shape}")

    z_dim = vol1.shape[0]
    band_slices, remainder_slice = make_z_band_slices(
        z_dim,
        ignore_top=ignore_top,
        ignore_bottom=ignore_bottom,
        band_size=band_size,
    )

    original_shift, original_shifts_axis, original_corr_values = normalised_correlation_3D(
        vol1,
        vol2,
        axis=axis,
        max_shift=max_shift,
    )

    boot_shifts = []
    sampled_band_indices = []

    for i in range(n_boot):
        v1_boot, v2_boot, sampled_idx = resample_volumes_by_z_band(
            vol1, vol2, band_slices, remainder_slice, rng
        )

        shift_boot, _, _ = normalised_correlation_3D(
            v1_boot,
            v2_boot,
            axis=axis,
            max_shift=max_shift,
        )

        boot_shifts.append(shift_boot)
        sampled_band_indices.append(sampled_idx)

        if verbose and ((i + 1) % 100 == 0 or i == n_boot - 1):
            print(f"{i+1}/{n_boot} complete")

    boot_shifts = np.asarray(boot_shifts, dtype=int)
    sampled_band_indices = np.asarray(sampled_band_indices, dtype=int)

    results = {
        "original_shift": int(original_shift),
        "original_shifts_axis": original_shifts_axis,
        "original_corr_values": original_corr_values,
        "bootstrap_shifts": boot_shifts,
        "bootstrap_mean": float(np.mean(boot_shifts)),
        "bootstrap_median": float(np.median(boot_shifts)),
        "bootstrap_std": float(np.std(boot_shifts, ddof=1)) if len(boot_shifts) > 1 else 0.0,
        "ci_95": tuple(np.percentile(boot_shifts, [2.5, 97.5])),
        "n_boot_requested": int(n_boot),
        "n_boot_successful": int(len(boot_shifts)),
        "band_size": int(band_size),
        "n_bands": int(len(band_slices)),
        "band_slices": band_slices,
        "sampled_band_indices": sampled_band_indices,
        "axis": axis,
        "max_shift": max_shift,
        "ignore_top": ignore_top,
        "ignore_bottom": ignore_bottom,
    }

    return results


# ==========================================
# 3. PLOTS
# ==========================================
def plot_original_correlation(results, title="Original correlation curve"):
    shifts = results["original_shifts_axis"]
    corr_values = results["original_corr_values"]
    original_shift = results["original_shift"]

    plt.figure(figsize=(8, 4.5))
    plt.plot(shifts, corr_values, linewidth=1.8)
    plt.axvline(original_shift, linestyle="--", linewidth=1.5, label=f"Best shift = {original_shift}")
    plt.xlabel("Pixel Shift")
    plt.ylabel("Correlation")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_bootstrap_distribution(results):
    shifts = results["bootstrap_shifts"]
    original_shift = results["original_shift"]
    ci_low, ci_high = results["ci_95"]

    unique, counts = np.unique(shifts, return_counts=True)

    plt.figure(figsize=(10, 5))
    plt.bar(unique, counts)
    plt.axvline(original_shift, linestyle="--", linewidth=2, label=f"Original = {original_shift}")
    plt.axvline(ci_low, linestyle=":", linewidth=2, label=f"2.5% = {ci_low:.2f}")
    plt.axvline(ci_high, linestyle=":", linewidth=2, label=f"97.5% = {ci_high:.2f}")
    plt.xlabel("Shift")
    plt.ylabel("Count")
    plt.title("Bootstrap Shift Distribution")
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("\nBootstrap summary")
    print(f"Original shift: {results['original_shift']}")
    print(f"Bootstrap mean: {results['bootstrap_mean']:.3f}")
    print(f"Bootstrap median: {results['bootstrap_median']:.3f}")
    print(f"Bootstrap std: {results['bootstrap_std']:.3f}")
    print(f"95% CI: [{ci_low:.3f}, {ci_high:.3f}]")
    print(f"Successful replicates: {results['n_boot_successful']} / {results['n_boot_requested']}")
    print(f"Band size: {results['band_size']} voxels")
    print(f"Number of bands: {results['n_bands']}")


# ==========================================
# 4. MAIN
# ==========================================
if __name__ == "__main__":
    STITCH_AXIS = "x"
    ROTATION = 0
    MAX_SHIFT = 180
    TRANSPOSE_INPUT_TO_ZXY = True

    IGNORE_TOP = 15
    IGNORE_BOTTOM = 10

    data_dir = Path.cwd().parent.parent / "DATA" / "2D TFM Data" / "2nd_experiment"

    vol_a_path = data_dir / "r21_filtered_3D_TFM.npy"
    vol_b_path = data_dir / "r22_filtered_3D_TFM.npy"

    if not vol_a_path.exists():
        raise FileNotFoundError(f"Missing file: {vol_a_path}")
    if not vol_b_path.exists():
        raise FileNotFoundError(f"Missing file: {vol_b_path}")

    print(f"Loading:\n  {vol_a_path}\n  {vol_b_path}")

    vol_a_signal = np.load(vol_a_path).astype(np.float32)
    vol_b_signal = np.load(vol_b_path).astype(np.float32)

    if TRANSPOSE_INPUT_TO_ZXY:
        vol_a = np.transpose(vol_a_signal, (0, 2, 1))
        vol_b = np.transpose(vol_b_signal, (0, 2, 1))
    else:
        vol_a = vol_a_signal
        vol_b = vol_b_signal

    z_dim = vol_a.shape[0]
    usable_z = z_dim - IGNORE_TOP - IGNORE_BOTTOM

    results = bootstrap_stitch_shift(
        vol_a,
        vol_b,
        n_boot=1000,
        axis=STITCH_AXIS,
        max_shift=MAX_SHIFT,
        ignore_top=IGNORE_TOP,
        ignore_bottom=IGNORE_BOTTOM,
        band_size=max(1, usable_z // 40),
        rng_seed=42,
        verbose=True,
    )

    plot_original_correlation(results, title=f"{data_dir.parent.name} / {data_dir.name}")
    plot_bootstrap_distribution(results)
# %%
