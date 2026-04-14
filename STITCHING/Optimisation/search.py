import random
import pandas as pd
from config import *

def sample_params():
    return {
        "cutoff_db": random.choice(SEARCH_SPACE["cutoff_db"]),
        "binary_threshold": random.choice(SEARCH_SPACE["binary_threshold"]),
        "tile_multiple_z": random.choice(SEARCH_SPACE["tile_multiple_z"]),
        "tile_multiple_y": random.choice(SEARCH_SPACE["tile_multiple_y"]),
        "min_hotspot_voxels": random.choice(SEARCH_SPACE["min_hotspot_voxels"]),
        "size_statistic": random.choice(SEARCH_SPACE["size_statistic"]),
    }

def summarise(df):
    ok = df[df["status"] == "ok"]

    return {
        "success_rate": ok["success"].mean(),
        "mean_error": ok["error"].mean(),
        "fail_rate": 1 - (len(ok) / len(df)),
    }

def run_random_search(run_dirs, evaluator):
    all_results = []
    summaries = []

    for i in range(N_RANDOM_SAMPLES):
        params = sample_params()

        results = evaluator(run_dirs, params)
        df = pd.DataFrame(results)

        summary = summarise(df)
        summary.update(params)

        all_results.append(df)
        summaries.append(summary)

        print(f"[{i}] success={summary['success_rate']:.3f}")

    return pd.concat(all_results), pd.DataFrame(summaries)