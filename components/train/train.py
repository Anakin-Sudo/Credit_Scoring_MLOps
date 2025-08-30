import argparse
import os
import json
import yaml
import pandas as pd

from utilities.ml_processes import make_learning_pipeline, load_model
from utilities.mlflow_processes import train_and_register_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", type=str, help="Path to preprocessed dataset (CSV)")
    parser.add_argument("--output_dir", type=str, help="Output directory for artifacts")
    parser.add_argument("--num_cols", type=str, help="Numerical columns (comma-separated)")
    parser.add_argument("--simple_cat_cols", type=str, help="Low-cardinality categorical columns (comma-separated)")
    parser.add_argument("--complex_cat_cols", type=str, help="High-cardinality categorical columns (comma-separated)")
    parser.add_argument("--passthrough_cols", type=str, default=None, help="Columns to passthrough (comma-separated)")
    parser.add_argument("--config", type=str, required=True, help="YAML config file with models/params")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset
    df = pd.read_csv(args.input_data)
    X = df.drop("target", axis=1)
    y = df["target"]

    # Parse feature groups
    num_cols = args.num_cols.split(",")
    simple_cat_cols = args.simple_cat_cols.split(",")
    complex_cat_cols = args.complex_cat_cols.split(",")
    passthrough_cols = args.passthrough_cols.split(",") if args.passthrough_cols else None

    # Load models from config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    metrics = {}

    for name, spec in config["models"].items():
        # Instantiate model dynamically
        model = load_model(spec["class"], spec["params"])

        # Build preprocessing + model pipeline
        pipeline = make_learning_pipeline(
            num_cols=num_cols,
            simple_cat_cols=simple_cat_cols,
            complex_cat_cols=complex_cat_cols,
            model=model,
            passthrough_cols=passthrough_cols
        )

        # Train + log + register with MLflow
        train_and_register_model(
            pipeline=pipeline,
            model_name=name,
            X_train=X, y_train=y,    # Right now train=test, split later
            X_test=X, y_test=y,
            params=spec["params"],
            tags={"type": name}
        )

        # Collect metrics for local artifact
        metrics[f"{name}_train_accuracy"] = pipeline.score(X, y)

    # Save metrics as JSON artifact
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f)


if __name__ == "__main__":
    main()
