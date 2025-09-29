import argparse
import json
import pandas as pd
from utilities.ml_processes import select_best_model
from utilities.mlflow_processes import get_candidates_for_current_run, score_on_test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--selection_criteria", type=str, required=True, help="JSON with selection criteria")
    parser.add_argument("--metrics_output", type=str, required=True)
    parser.add_argument("--best_model_pointer_file", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    # Load data
    test_df = pd.read_csv(args.test_data)

    # Parse configs
    candidates = get_candidates_for_current_run()
    selection_criteria = json.loads(args.selection_criteria)

    # Step 1: select champion
    best = select_best_model(candidates, selection_criteria)
    model_uri = best["model_uri"]

    # Step 2: score on test set
    metrics = score_on_test(model_uri, test_df, threshold=args.threshold)

    # Step 3: write outputs
    with open(args.metrics_output, "w") as f:
        json.dump(metrics, f, indent=2)

    with open(args.best_model_pointer_file, "w") as f:
        f.write(model_uri)

    print(f" Best model: {best['model']} @ {model_uri}")
    print(f" Test metrics: {metrics}")


if __name__ == "__main__":
    main()
