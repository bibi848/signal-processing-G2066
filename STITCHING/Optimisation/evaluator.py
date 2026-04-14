import time
import numpy as np
from config import *
from dataio import load_case
from ground_truth import true_shift_vox, half_wavelength_vox

# IMPORT YOUR STITCHER HERE
from STITCHING.stitch_variable_grid_3D import run_stitcher_test


def evaluate_case(run_dir, params):
    try:
        vol1, vol2, meta = load_case(run_dir)

        true_shift = TRUE_SHIFT_SIGN * int(round(true_shift_vox(meta)))
        tol = half_wavelength_vox(meta, WAVESPEED_M_S)

        t0 = time.time()

        shift, _, _, diag = run_stitcher_test(
            vol1,
            vol2,
            cutoff_db=params["cutoff_db"],
            binary_threshold=params["binary_threshold"],
            tile_multiple=(params["tile_multiple_z"], params["tile_multiple_y"]),
            min_hotspot_voxels=params["min_hotspot_voxels"],
            size_statistic=params["size_statistic"],
            **FIXED_STITCHER_PARAMS,
        )

        runtime = time.time() - t0

        error = abs(shift - true_shift)
        success = error <= tol

        return {
            "run_dir": run_dir,
            "true_shift": true_shift,
            "estimated_shift": shift,
            "error": error,
            "success": success,
            "tol": tol,
            "runtime": runtime,
            "final_score": float(np.max(diag["weighted_scores"])),
            "n_tiles": len(diag["all_shifts"]),
            "status": "ok",
        }

    except Exception as e:
        return {
            "run_dir": run_dir,
            "status": "fail",
            "error_msg": str(e),
        }


def evaluate_paramset(run_dirs, params):
    results = []

    for rd in run_dirs:
        res = evaluate_case(rd, params)
        res.update(params)
        results.append(res)

    return results