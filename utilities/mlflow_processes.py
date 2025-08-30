import mlflow
import mlflow.sklearn
from sklearn.metrics import roc_auc_score, precision_score, accuracy_score, recall_score, f1_score

import mlflow
import mlflow.sklearn
from sklearn.metrics import roc_auc_score, precision_score, accuracy_score, recall_score, f1_score


def train_and_log_binary_classifier(
        model,
        model_name: str,
        X_train, y_train,
        X_test, y_test,
        params: dict = None,
        tags: dict = None,
        log_model=True
):
    """
    Lightweight helper for experimentation in notebooks.
    Trains one model, logs metrics, params, tags, and optionally saves model.
    """
    with mlflow.start_run():
        # Train
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Try to get probabilities if possible
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_proba)
            mlflow.log_metric("roc_auc", auc)
        except AttributeError:
            print("Model does not give prediction probability, can't compute AUC-ROC")

        # Log error metrics
        for avg in ['macro', 'micro', 'weighted']:
            precision = round(precision_score(y_test, y_pred, average=avg), 2)
            recall = round(recall_score(y_test, y_pred, average=avg), 2)
            f1 = round(f1_score(y_test, y_pred, average=avg), 2)

            mlflow.log_metric(f'precision_{avg}', precision)
            mlflow.log_metric(f'recall_{avg}', recall)
            mlflow.log_metric(f'f1_{avg}', f1)

        accuracy = round(accuracy_score(y_test, y_pred), 2)
        mlflow.log_metric('accuracy', accuracy)

        # Log hyperparameters
        if params:
            mlflow.log_params(params)
        mlflow.log_param('model_name', model_name)

        # Log tags
        if tags:
            mlflow.set_tags(tags)

        # Log model itself
        if log_model:
            mlflow.sklearn.log_model(model, artifact_path="model")


def train_and_register_model(
        pipeline,
        model_name: str,
        X_train, y_train,
        X_test, y_test,
        params: dict = None,
        tags: dict = None,
        register: bool = True
):
    """
    Production-oriented helper.
    Trains a full pipeline (preprocessing + model),
    logs metrics and params, and registers the model in MLflow/AML.
    """
    with mlflow.start_run():
        # Train
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        # Try to get probabilities if possible
        try:
            y_proba = pipeline.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_proba)
            mlflow.log_metric("roc_auc", auc)
        except AttributeError:
            print(f"Pipeline {model_name} has no predict_proba, can't compute AUC-ROC")

        # Log metrics
        accuracy = round(accuracy_score(y_test, y_pred), 2)
        mlflow.log_metric("accuracy", accuracy)

        for avg in ["macro", "micro", "weighted"]:
            precision = round(precision_score(y_test, y_pred, average=avg), 2)
            recall = round(recall_score(y_test, y_pred, average=avg), 2)
            f1 = round(f1_score(y_test, y_pred, average=avg), 2)

            mlflow.log_metric(f"precision_{avg}", precision)
            mlflow.log_metric(f"recall_{avg}", recall)
            mlflow.log_metric(f"f1_{avg}", f1)

        # Log hyperparameters
        if params:
            mlflow.log_params(params)
        mlflow.log_param("model_name", model_name)

        # Log tags
        if tags:
            mlflow.set_tags(tags)

        # Log + (optionally) register model
        if register:
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                artifact_path=model_name,
                registered_model_name=f"credit_model_{model_name}"
            )
        else:
            mlflow.sklearn.log_model(pipeline, artifact_path=model_name)



