# train.py
import argparse
import json
import os
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

import mlflow
from utilities.ml_processes import make_learning_pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", type=str, help="Path to preprocessed dataset (CSV)")
    parser.add_argument("--output_dir", type=str, help="Output directory for artifacts")
    parser.add_argument("--num_cols", type=str, help="Numerical columns (comma-separated)")
    parser.add_argument("--simple_cat_cols", type=str, help="Low-cardinality categorical columns (comma-separated)")
    parser.add_argument("--complex_cat_cols", type=str, help="High-cardinality categorical columns (comma-separated)")
    parser.add_argument("--passthrough_cols", type=str, default=None, help="Columns to passthrough (comma-separated)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset
    df = pd.read_csv(args.input_data)
    X = df.drop("target", axis=1)
    y = df["target"]

    # Parse columns
    num_cols = args.num_cols.split(",")
    simple_cat_cols = args.simple_cat_cols.split(",")
    complex_cat_cols = args.complex_cat_cols.split(",")
    passthrough_cols = args.passthrough_cols.split(",") if args.passthrough_cols else None

    # Candidate models with fixed params (from experimentation)
    candidates = {
        "random_forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        "log_reg": LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs"),
        "xgb": XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            use_label_encoder=False,
            eval_metric="logloss"
        )
    }

    metrics = {}

    for name, model in candidates.items():
        pipeline = make_learning_pipeline(
            num_cols=num_cols,
            simple_cat_cols=simple_cat_cols,
            complex_cat_cols=complex_cat_cols,
            model=model,
            passthrough_cols=passthrough_cols
        )

        pipeline.fit(X, y)
        score = pipeline.score(X, y)
        metrics[f"{name}_train_accuracy"] = score

        # Log metrics & params with AML’s MLflow (implicit run)
        mlflow.log_param(f"{name}_params", model.get_params())
        mlflow.log_metric(f"{name}_train_accuracy", score)

        # Log full model pipeline to AML registry
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path=name,
            registered_model_name=f"credit_model_{name}"  # appears in AML registry
        )

        print(f"Logged & registered: credit_model_{name} (acc={score:.4f})")

    # Save metrics as JSON (optional artifact)
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f)


if __name__ == "__main__":
    main()
