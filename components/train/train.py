import argparse
import json
import pandas as pd

from utilities.ml_processes import build_preprocessor
from utilities.model_factory import build_pipelines
from utilities.mlflow_processes import train_and_register_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", type=str, help="Path to preprocessed training dataset (CSV)")
    parser.add_argument("--candidates", type=str, required=True, help="JSON string with candidate models and hyperparameters")
    parser.add_argument("--feature_groups", type=str, required=True, help="JSON string with feature group definitions")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--cv_folds", type=int, default=5, help="Number of cross-validation folds")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for classification metrics")
    args = parser.parse_args()

    # --- Load dataset ---
    df = pd.read_csv(args.input_data)
    X = df.drop(columns=["CreditRisk"])
    y = df["CreditRisk"]

    # --- Parse configs ---
    candidates = json.loads(args.candidates)      # list of {model, params, tags, cv_score, ...}
    feature_groups = json.loads(args.feature_groups)

    # --- Build preprocessor + pipelines ---
    preprocessor = build_preprocessor(feature_groups)
    pipelines = build_pipelines(candidates, preprocessor)

    # --- Train, log, and register each candidate ---
    for cand in candidates:
        name = cand["model"]
        pipeline = pipelines[name]

        metrics = train_and_register_model(
            name=name,
            pipeline=pipeline,
            X_train=X,
            y_train=y,
            params=cand.get("params", {}),
            tags=cand.get("tags", {}),
            cv_folds=args.cv_folds,
            threshold=args.threshold
        )

        print(f"Finished training {name}. Logged metrics: {metrics}")


if __name__ == "__main__":
    main()
