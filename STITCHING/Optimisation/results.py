import json
from config import OUTPUT_DIR

def save_all(trials, summary):
    OUTPUT_DIR.mkdir(exist_ok=True)

    trials.to_parquet(OUTPUT_DIR / "trials.parquet")
    summary.to_csv(OUTPUT_DIR / "summary.csv", index=False)

    best = summary.sort_values(
        ["success_rate", "mean_error"],
        ascending=[False, True]
    ).iloc[0]

    with open(OUTPUT_DIR / "best_params.json", "w") as f:
        json.dump(best.to_dict(), f, indent=2)