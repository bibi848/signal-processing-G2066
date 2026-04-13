import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate, correlation_lags
from pathlib import Path


# ==========================================
# 0. PRE-PROCESSING UTILITIES
# ==========================================
def apply_db_cutoff(vol, cutoff_db=-5):
    v_abs = np.abs(vol)
    v_max = np.max(v_abs)
    if v_max == 0:
        return vol.astype(np.float32), 0.0

    thresh = v_max * (10 ** (cutoff_db / 20))
    v_thresh = np.where(v_abs >= thresh, vol, 0)

    sparsity = (np.count_nonzero(v_thresh == 0) / v_thresh.size) * 100
    return v_thresh.astype(np.float32), sparsity


# ==========================================
# 1. STITCHER ENGINE (NO VISUAL OUTPUT)
# ==========================================
def run_stitcher_test(
    vol1,
    vol2,
    grid=(40, 20),
    expected=0,
    tolerance=200,
    cutoff_db=-10,
    ignore_top=30,
    verbose=False
):
    v1, _ = apply_db_cutoff(vol1, cutoff_db)
    v2, _ = apply_db_cutoff(vol2, cutoff_db)

    z_dim, y_dim, _ = v1.shape
    z_start, z_end = ignore_top, z_dim

    tile_z = (z_end - z_start) // grid[0]
    tile_y = y_dim // grid[1]

    if tile_z <= 0 or tile_y <= 0:
        raise ValueError("Grid too fine for volume.")

    all_shifts = []
    all_weights = []

    for r in range(grid[0]):
        for c in range(grid[1]):
            zs, ze = z_start + r * tile_z, z_start + (r + 1) * tile_z
            ys, ye = c * tile_y, (c + 1) * tile_y

            prof1 = np.mean(np.abs(v1[zs:ze, ys:ye, :]), axis=(0, 1))
            prof2 = np.mean(np.abs(v2[zs:ze, ys:ye, :]), axis=(0, 1))

            if np.std(prof1) < 1e-6 or np.max(prof1) == 0:
                continue
            if np.std(prof2) < 1e-6 or np.max(prof2) == 0:
                continue

            p1_n = (prof1 - np.mean(prof1)) / (np.std(prof1) + 1e-10)
            p2_n = (prof2 - np.mean(prof2)) / (np.std(prof2) + 1e-10)

            corr = correlate(p1_n, p2_n, mode='full')
            lags = correlation_lags(len(p1_n), len(p2_n), mode='full')

            mask = (lags >= expected - tolerance) & (lags <= expected + tolerance)
            if not np.any(mask):
                continue

            corr[~mask] = -np.inf

            peak_idx = np.argmax(corr)
            lag = int(lags[peak_idx])
            weight = float(corr[peak_idx])

            if np.isfinite(weight):
                all_shifts.append(lag)
                all_weights.append(weight)

    if not all_shifts:
        raise ValueError("No valid tiles.")

    # weighted consensus
    shifts = np.array(all_shifts)
    weights = np.array(all_weights)

    unique = np.arange(shifts.min(), shifts.max() + 1)
    score = np.zeros_like(unique, dtype=float)

    idx_map = {s: i for i, s in enumerate(unique)}
    for s, w in zip(shifts, weights):
        score[idx_map[s]] += w

    final_shift = int(unique[np.argmax(score)])

    if verbose:
        print(f"Shift: {final_shift}")

    return final_shift


# ==========================================
# 2. BOOTSTRAP UTILITIES
# ==========================================
def make_z_band_slices(z_dim, ignore_top=30, band_size=20):
    bands = []
    z = ignore_top
    while z < z_dim:
        z_next = min(z + band_size, z_dim)
        bands.append(slice(z, z_next))
        z = z_next
    return bands


def resample_volumes(vol1, vol2, band_slices, rng):
    ignore_top = band_slices[0].start
    idx = rng.integers(0, len(band_slices), size=len(band_slices))

    v1_parts = [vol1[:ignore_top]]
    v2_parts = [vol2[:ignore_top]]

    for i in idx:
        s = band_slices[i]
        v1_parts.append(vol1[s])
        v2_parts.append(vol2[s])

    v1_boot = np.concatenate(v1_parts, axis=0)
    v2_boot = np.concatenate(v2_parts, axis=0)

    return v1_boot, v2_boot


def bootstrap_shift(
    vol1,
    vol2,
    n_boot=1000,
    grid=(40, 20),
    cutoff_db=-10,
    ignore_top=30,
    band_size=25,
    rng_seed=42
):
    rng = np.random.default_rng(rng_seed)

    z_dim = vol1.shape[0]
    band_slices = make_z_band_slices(z_dim, ignore_top, band_size)

    original_shift = run_stitcher_test(
        vol1, vol2,
        grid=grid,
        cutoff_db=cutoff_db,
        ignore_top=ignore_top
    )

    shifts = []

    for i in range(n_boot):
        v1b, v2b = resample_volumes(vol1, vol2, band_slices, rng)

        try:
            s = run_stitcher_test(
                v1b, v2b,
                grid=grid,
                cutoff_db=cutoff_db,
                ignore_top=ignore_top
            )
            shifts.append(s)
        except ValueError:
            continue

        if (i + 1) % 100 == 0:
            print(f"{i+1}/{n_boot}")

    shifts = np.array(shifts)

    return {
        "original": original_shift,
        "shifts": shifts,
        "mean": float(np.mean(shifts)),
        "std": float(np.std(shifts, ddof=1)),
        "ci": tuple(np.percentile(shifts, [2.5, 97.5]))
    }


# ==========================================
# 3. FINAL PLOT ONLY
# ==========================================
def plot_distribution(results):
    shifts = results["shifts"]

    plt.figure(figsize=(10, 5))
    plt.hist(shifts, bins='auto')
    plt.axvline(results["original"], linestyle='--', label="original")
    plt.axvline(results["ci"][0], linestyle=':', label="2.5%")
    plt.axvline(results["ci"][1], linestyle=':', label="97.5%")
    plt.xlabel("Shift")
    plt.ylabel("Count")
    plt.title("Bootstrap Shift Distribution")
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("\nResults:")
    print(results)


# ==========================================
# 4. MAIN
# ==========================================
if __name__ == "__main__":
    IN_DIR = Path.cwd() / 'DATA' / '2D TFM Data' / "FeC Smile 3MHz 04022026 Filtered"

    vol1 = np.load(IN_DIR / "FeC_40_6_filtered_3D_TFM.npy")
    vol2 = np.load(IN_DIR / "FeC_40_5_filtered_3D_TFM.npy")

    results = bootstrap_shift(
        vol1,
        vol2,
        n_boot=1000,
        grid=(40, 20),
        cutoff_db=-10,
        band_size=30
    )

    plot_distribution(results)