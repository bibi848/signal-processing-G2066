from pathlib import Path

# =========================
# DATA
# =========================
SWEEP_ROOT = Path("PATH/TO/YOUR/SWEEP_OUTPUT")

POS1_NAME = "pos_000"
POS2_NAME = "pos_001"

VOLUME_FILENAME = "recon_volume_zxy.npy"
META_FILENAME = "dataset_meta.json"

# =========================
# STITCHER SETTINGS
# =========================
STITCH_AXIS = 2
TRUE_SHIFT_SIGN = +1

FIXED_STITCHER_PARAMS = {
    "axis": 2,
    "adaptive_grid": True,
    "ignore_top": 30,
    "expected": 0,
    "tolerance": 200,
    "min_grid": (10, 10),
    "max_grid": (100, 100),
}

# =========================
# EVALUATION
# =========================
WAVESPEED_M_S = 6700.0  # aluminium approx

# =========================
# SEARCH SPACE
# =========================
SEARCH_SPACE = {
    "cutoff_db": [-20, -15, -10, -7, -5],
    "binary_threshold": [0.3, 0.5, 0.7, 0.9],
    "tile_multiple_z": [1.5, 2.0, 2.5],
    "tile_multiple_y": [1.5, 2.0, 2.5],
    "min_hotspot_voxels": [5, 10, 20],
    "size_statistic": ["median", "mean"],
}

N_RANDOM_SAMPLES = 30

# =========================
# OUTPUT
# =========================
OUTPUT_DIR = Path("stitch_eval_output")