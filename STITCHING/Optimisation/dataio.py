import json
import numpy as np
import pandas as pd
from pathlib import Path
from config import *

def find_runs():
    return sorted([p for p in SWEEP_ROOT.glob("run_*") if p.is_dir()])

def load_case(run_dir):
    pos1 = run_dir / POS1_NAME
    pos2 = run_dir / POS2_NAME

    vol1 = np.load(pos1 / VOLUME_FILENAME)
    vol2 = np.load(pos2 / VOLUME_FILENAME)

    with open(pos1 / META_FILENAME) as f:
        meta = json.load(f)

    return vol1, vol2, meta

def build_manifest():
    rows = []

    for run_dir in find_runs():
        try:
            _, _, meta = load_case(run_dir)

            row = {
                "run_dir": str(run_dir),
                "overlap_fraction": meta["scan_grid"]["overlap_fraction"],
                "step_x_m": meta["scan_grid"]["step_x_m"],
                "cube_side_m": meta["scan_grid"]["cube_side_m"],
                "n_pixels": meta["tfm"]["n_pixels"],
                "frequency_Hz": meta["array"]["frequency_Hz"],
                "seed": meta["grain"]["seed"],
            }

            rows.append(row)

        except Exception as e:
            print(f"Skipping {run_dir}: {e}")

    return pd.DataFrame(rows)