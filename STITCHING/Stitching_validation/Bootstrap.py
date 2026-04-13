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
        raise ValueError(
            f"Grid {grid} is too fine for volume shape {v1.shape} after ignore_top={ignore_top}."
        )

    all_shifts = []
    all_weights = []
    tile_vote_map = np.full(grid, np.nan, dtype=float)

    rejected_prof1 = 0
    rejected_prof2 = 0
    rejected_not_enough_peaks = 0
    rejected_nonfinite_peak = 0
    rejected_ambiguous = 0

    for r in range(grid[0]):
        for c in range(grid[1]):
            zs, ze = z_start + (r * tile_z), z_start + ((r + 1) * tile_z)
            ys, ye = c * tile_y, (c + 1) * tile_y

            prof1 = np.mean(np.abs(v1[zs:ze, ys:ye, :]), axis=(0, 1))
            prof2 = np.mean(np.abs(v2[zs:ze, ys:ye, :]), axis=(0, 1))

            if np.std(prof1) < 1e-6 or np.max(prof1) == 0:
                rejected_prof1 += 1
                continue

            if np.std(prof2) < 1e-6 or np.max(prof2) == 0:
                rejected_prof2 += 1
                continue

            p1_n = (prof1 - np.mean(prof1)) / (np.std(prof1) + 1e-10)
            p2_n = (prof2 - np.mean(prof2)) / (np.std(prof2) + 1e-10)

            corr = correlate(p1_n, p2_n, mode='full')
            lags = correlation_lags(len(p1_n), len(p2_n), mode='full')

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

            peak_idx = np.argmax(corr_masked)
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

            chosen_weight = peak_ratio

            all_shifts.append(chosen_lag)
            all_weights.append(chosen_weight)
            tile_vote_map[r, c] = chosen_lag

    if not all_shifts:
        raise ValueError("No tiles survived after cutoff + ambiguity rejection.")

    all_shifts = np.asarray(all_shifts, dtype=int)
    all_weights = np.asarray(all_weights, dtype=float)

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

    if verbose:
        print(f"\nShift: {final_shift} voxels")
        print(f"Participating tiles: {len(all_shifts)} / {grid[0] * grid[1]}")
        print(f"Tiles voting for chosen shift: {np.sum(all_shifts == final_shift)}")
        print(f"Weighted score at chosen shift: {weighted_scores[shift_to_idx[final_shift]]:.3f}")
        print("\nTile rejection summary:")
        print(f" - Rejected (flat/empty prof1): {rejected_prof1}")
        print(f" - Rejected (flat/empty prof2): {rejected_prof2}")
        print(f" - Rejected (insufficient finite peaks): {rejected_not_enough_peaks}")
        print(f" - Rejected (non-finite best peak): {rejected_nonfinite_peak}")
        print(f" - Rejected (ambiguous peak ratio): {rejected_ambiguous}")

    diagnostics = {
        "shift_axis": shift_axis,
        "weighted_scores": weighted_scores,
        "vote_counts": vote_counts,
        "final_shift": final_shift,
        "tile_vote_map": tile_vote_map,
        "tile_distance_map": tile_distance_map,
        "all_shifts": all_shifts,
        "all_weights": all_weights,
        "rejected_prof1": rejected_prof1,
        "rejected_prof2": rejected_prof2,
        "rejected_not_enough_peaks": rejected_not_enough_peaks,
        "rejected_nonfinite_peak": rejected_nonfinite_peak,
        "rejected_ambiguous": rejected_ambiguous,
        "peak_ratio_min": peak_ratio_min,
        "ignore_top": ignore_top,
        "grid": grid,
        "expected": expected,
        "tolerance": tolerance,
        "cutoff_db": cutoff_db,
    }

    return final_shift, v1, v2, diagnostics


# ==========================================
# 2. BOOTSTRAP UTILITIES
# ==========================================
def make_z_band_slices(z_dim, ignore_top=30, band_size=20):
    """
    Split the usable z-range [ignore_top:z_dim] into bootstrap bands.
    """
    if band_size <= 0:
        raise ValueError("band_size must be > 0")

    if ignore_top >= z_dim:
        raise ValueError(f"ignore_top={ignore_top} is too large for z_dim={z_dim}")

    bands = []
    z = ignore_top
    while z < z_dim:
        z_next = min(z + band_size, z_dim)
        bands.append(slice(z, z_next))
        z = z_next

    if not bands:
        raise ValueError("No z-bands created.")

    return bands


def resample_volumes_by_z_band(vol1, vol2, band_slices, rng):
    """
    Resample z-bands with replacement and rebuild new volumes.
    The ignored top region is preserved unchanged.
    """
    if vol1.shape != vol2.shape:
        raise ValueError(f"Volume shapes do not match: {vol1.shape} vs {vol2.shape}")

    ignore_top = band_slices[0].start
    sampled_indices = rng.integers(0, len(band_slices), size=len(band_slices))

    v1_parts = [vol1[:ignore_top]]
    v2_parts = [vol2[:ignore_top]]

    for idx in sampled_indices:
        s = band_slices[idx]
        v1_parts.append(vol1[s])
        v2_parts.append(vol2[s])

    v1_boot = np.concatenate(v1_parts, axis=0)
    v2_boot = np.concatenate(v2_parts, axis=0)

    if v1_boot.shape != vol1.shape or v2_boot.shape != vol2.shape:
        raise RuntimeError(
            f"Bootstrapped volumes changed shape: {v1_boot.shape}, {v2_boot.shape}, expected {vol1.shape}"
        )

    return v1_boot, v2_boot, sampled_indices


# ==========================================
# 3. BOOTSTRAP RUNNER
# ==========================================
def bootstrap_stitch_shift(
    vol1,
    vol2,
    n_boot=1000,
    grid=(40, 20),
    expected=0,
    tolerance=200,
    cutoff_db=-10,
    peak_ratio_min=1.05,
    ignore_top=30,
    band_size=20,
    rng_seed=42,
    verbose=True,
):
    """
    Bootstrap the final consensus stitch shift by resampling z-bands.

    Returns
    -------
    results : dict
    """
    rng = np.random.default_rng(rng_seed)

    if vol1.shape != vol2.shape:
        raise ValueError(f"Volume shapes do not match: {vol1.shape} vs {vol2.shape}")

    z_dim = vol1.shape[0]
    band_slices = make_z_band_slices(z_dim, ignore_top=ignore_top, band_size=band_size)

    original_shift, _, _, original_diag = run_stitcher_test(
        vol1,
        vol2,
        grid=grid,
        expected=expected,
        tolerance=tolerance,
        cutoff_db=cutoff_db,
        peak_ratio_min=peak_ratio_min,
        ignore_top=ignore_top,
        verbose=True,
    )

    boot_shifts = []
    sampled_band_indices = []
    fails = 0

    for i in range(n_boot):
        v1_boot, v2_boot, sampled_idx = resample_volumes_by_z_band(
            vol1, vol2, band_slices, rng
        )

        try:
            shift_boot, _, _, _ = run_stitcher_test(
                v1_boot,
                v2_boot,
                grid=grid,
                expected=expected,
                tolerance=tolerance,
                cutoff_db=cutoff_db,
                peak_ratio_min=peak_ratio_min,
                ignore_top=ignore_top,
                verbose=False,
            )
            boot_shifts.append(shift_boot)
            sampled_band_indices.append(sampled_idx)

        except ValueError:
            fails += 1

        if verbose and ((i + 1) % 100 == 0 or i == n_boot - 1):
            print(f"{i+1}/{n_boot} complete | successes={len(boot_shifts)} | failures={fails}")

    if len(boot_shifts) == 0:
        raise RuntimeError("All bootstrap runs failed.")

    boot_shifts = np.asarray(boot_shifts, dtype=int)
    sampled_band_indices = np.asarray(sampled_band_indices, dtype=int)

    results = {
        "original_shift": int(original_shift),
        "bootstrap_shifts": boot_shifts,
        "bootstrap_mean": float(np.mean(boot_shifts)),
        "bootstrap_std": float(np.std(boot_shifts, ddof=1)) if len(boot_shifts) > 1 else 0.0,
        "ci_95": tuple(np.percentile(boot_shifts, [2.5, 97.5])),
        "n_boot_requested": int(n_boot),
        "n_boot_successful": int(len(boot_shifts)),
        "n_boot_failed": int(fails),
        "band_size": int(band_size),
        "n_bands": int(len(band_slices)),
        "band_slices": band_slices,
        "sampled_band_indices": sampled_band_indices,
        "grid": grid,
        "expected": expected,
        "tolerance": tolerance,
        "cutoff_db": cutoff_db,
        "peak_ratio_min": peak_ratio_min,
        "ignore_top": ignore_top,
        "original_diagnostics": original_diag,
    }

    return results


# ==========================================
# 4. FINAL PLOT
# ==========================================
def plot_bootstrap_distribution(results):
    shifts = results["bootstrap_shifts"]
    original_shift = results["original_shift"]
    ci_low, ci_high = results["ci_95"]

    unique, counts = np.unique(shifts, return_counts=True)

    plt.figure(figsize=(10, 5))
    plt.bar(unique, counts)
    plt.axvline(original_shift, linestyle='--', linewidth=2, label=f'Original = {original_shift}')
    plt.axvline(ci_low, linestyle=':', linewidth=2, label=f'2.5% = {ci_low:.2f}')
    plt.axvline(ci_high, linestyle=':', linewidth=2, label=f'97.5% = {ci_high:.2f}')
    plt.xlabel("Shift")
    plt.ylabel("Count")
    plt.title("Bootstrap Shift Distribution")
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("\nBootstrap summary")
    print(f"Original shift: {results['original_shift']}")
    print(f"Bootstrap mean: {results['bootstrap_mean']:.3f}")
    print(f"Bootstrap std: {results['bootstrap_std']:.3f}")
    print(f"95% CI: [{ci_low:.3f}, {ci_high:.3f}]")
    print(f"Successful replicates: {results['n_boot_successful']} / {results['n_boot_requested']}")
    print(f"Failures: {results['n_boot_failed']}")
    print(f"Band size: {results['band_size']} voxels")
    print(f"Number of bands: {results['n_bands']}")


# ==========================================
# 5. MAIN
# ==========================================
if __name__ == "__main__":
    IN_DIR = Path.cwd().parent / "SYNTHETIC DATA" / "output" / "sweep_20260327_105215" / "run_000"

    vol1_raw = np.load(IN_DIR / "pos_000" / "recon_volume_zxy.npy")
    vol2_raw = np.load(IN_DIR / "pos_001" / "recon_volume_zxy.npy")

    z_dim = vol1_raw.shape[0]

    results = bootstrap_stitch_shift(
        vol1_raw,
        vol2_raw,
        n_boot=1000,
        grid=(50, 50),
        expected=0,
        tolerance=180,
        cutoff_db=-5,
        peak_ratio_min=1.10,
        ignore_top=30,
        band_size=max(1, (z_dim - 30) // 40),
        rng_seed=42,
        verbose=True,
    )

    plot_bootstrap_distribution(results)
