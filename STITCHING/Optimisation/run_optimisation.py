from dataio import find_runs
from evaluator import evaluate_paramset
from search import run_random_search
from results import save_all

def main():
    run_dirs = find_runs()

    trials, summary = run_random_search(
        run_dirs,
        evaluate_paramset
    )

    save_all(trials, summary)

    print("\nBest params:")
    print(summary.sort_values(
        ["success_rate", "mean_error"],
        ascending=[False, True]
    ).head(1))


if __name__ == "__main__":
    main()