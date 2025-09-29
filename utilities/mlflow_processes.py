import mlflow
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.metrics import (
    roc_auc_score, precision_score, accuracy_score, recall_score, f1_score,
    brier_score_loss, log_loss
)
import os


def train_and_register_model(
        name: str,
        pipeline,
        X_train, y_train,
        params: dict = None,
        tags: dict = None,
        cv_folds: int = 5,
        threshold: float = 0.5
):
    """
    Train a candidate pipeline, log metrics and params to MLflow,
    and register the model in the MLflow/AML registry.

    Workflow:
      - Cross-validation for selection metrics (AUC-ROC primarily)
      - Final fit for calibration + threshold-based metrics
    """
    with mlflow.start_run():
        metrics = {}

        # --- 1. Cross-validation for selection metrics ---
        scoring = {"auc": "roc_auc"}  # we keep AUC as primary
        cv_results = cross_validate(
            pipeline, X_train, y_train,
            cv=cv_folds,
            scoring=scoring,
            return_train_score=False,
            n_jobs=-1
        )

        auc_mean = np.mean(cv_results["test_auc"])
        auc_std = np.std(cv_results["test_auc"])
        mlflow.log_metric("cv_auc_mean", auc_mean)
        mlflow.log_metric("cv_auc_std", auc_std)
        metrics["cv_auc_mean"] = auc_mean
        metrics["cv_auc_std"] = auc_std

        # --- 2. Final fit for calibration + threshold metrics ---
        pipeline.fit(X_train, y_train)

        try:
            y_proba = pipeline.predict_proba(X_train)[:, 1]

            # Calibration metrics
            brier = brier_score_loss(y_train, y_proba)
            ll = log_loss(y_train, y_proba)
            mlflow.log_metric("brier_score", brier)
            mlflow.log_metric("log_loss", ll)
            metrics["brier_score"] = brier
            metrics["log_loss"] = ll

            # Threshold-based classification metrics
            y_pred = (y_proba >= threshold).astype(int)
            precision = precision_score(y_train, y_pred, pos_label=1)
            recall = recall_score(y_train, y_pred, pos_label=1)
            f1 = f1_score(y_train, y_pred, pos_label=1)

            mlflow.log_metric(f"precision_pos@{threshold}", precision)
            mlflow.log_metric(f"recall_pos@{threshold}", recall)
            mlflow.log_metric(f"f1_pos@{threshold}", f1)

            metrics[f"precision_pos@{threshold}"] = precision
            metrics[f"recall_pos@{threshold}"] = recall
            metrics[f"f1_pos@{threshold}"] = f1

        except AttributeError:
            metrics["brier_score"] = None
            metrics["log_loss"] = None

        # --- 3. Log hyperparameters and tags ---
        if params:
            mlflow.log_params(params)
        mlflow.log_param("model_key", name)
        if tags:
            mlflow.set_tags(tags)

        # --- 4. Register model ---
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path=name,
            registered_model_name=f"credit_model_{name}"
        )


def get_candidates_for_current_run():
    parent_run_id = os.environ.get("AZUREML_PARENT_RUN_ID")
    if not parent_run_id:
        raise RuntimeError("AZUREML_PARENT_RUN_ID not found in environment")

    client = mlflow.tracking.MlflowClient()
    all_models = []

    for rm in client.search_registered_models():
        for v in rm.latest_versions:
            # Only models tagged with current pipeline run id
            if v.tags.get("pipeline_run_id") == parent_run_id and v.tags.get("candidate") == "True":
                run = client.get_run(v.run_id)
                metrics = run.data.metrics
                all_models.append({
                    "name": v.name,
                    "version": v.version,
                    "run_id": v.run_id,
                    "model_uri": f"models:/{v.name}/{v.version}",
                    "metrics": metrics,
                })
    return all_models


def score_on_test(model_uri, test_df, threshold=0.5):
    """
    Load model from MLflow registry and score it on test set.
    """
    model = mlflow.sklearn.load_model(model_uri)

    X_test = test_df.drop(columns=["CreditRisk"])
    y_test = test_df["CreditRisk"]

    # Predict
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    # Metrics
    metrics = {
        "auc_roc": roc_auc_score(y_test, y_proba),
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "brier_score": brier_score_loss(y_test, y_proba),
    }
    return metrics
