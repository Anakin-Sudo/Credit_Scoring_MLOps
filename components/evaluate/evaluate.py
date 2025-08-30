import argparse
import os
import json
import mlflow
from mlflow.tracking import MlflowClient


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_output", type=str, help="Path to save metrics JSON")
    parser.add_argument("--best_model_uri", type=str, help="Path to save best model URI (TXT)")
    args = parser.parse_args()

    client = MlflowClient()

    all_metrics = {}
    best_model = None
    best_version = None
    best_auc = -1.0

    # List all registered models
    registered_models = client.list_registered_models()

    for model in registered_models:
        model_name = model.name

        # Look at all versions for this model
        versions = client.search_model_versions(f"name='{model_name}'")
        for v in versions:
            run_id = v.run_id
            run = client.get_run(run_id)

            auc = run.data.metrics.get("roc_auc", None)
            if auc is not None:
                auc = float(auc)
                key = f"{model_name}:v{v.version}"
                all_metrics[key] = {"roc_auc": auc}

                if auc > best_auc:
                    best_auc = auc
                    best_model = model_name
                    best_version = v.version

    if best_model is None:
        raise ValueError("No models with roc_auc metric found in the registry")

    # Save metrics JSON
    os.makedirs(os.path.dirname(args.metrics_output), exist_ok=True)
    with open(args.metrics_output, "w") as f:
        json.dump(all_metrics, f, indent=2)

    # Save best model URI
    best_model_uri = f"models:/{best_model}/{best_version}"
    os.makedirs(os.path.dirname(args.best_model_uri), exist_ok=True)
    with open(args.best_model_uri, "w") as f:
        f.write(best_model_uri)

    print(f"✅ Best model selected: {best_model_uri} (roc_auc={best_auc:.4f})")


if __name__ == "__main__":
    main()
