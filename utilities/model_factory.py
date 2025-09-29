from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def build_pipelines(candidates: list, preprocessor):
    """
    Build pipelines for all candidate models.

    Args:
        candidates (list): list of dicts, each with:
            - model: "logreg", "rf", "xgb"
            - params: hyperparameters dict
        preprocessor: fitted ColumnTransformer from build_preprocessor()

    Returns:
        dict {model_name: sklearn Pipeline}
    """

    pipelines = {}

    for cand in candidates:
        name = cand["model"]
        params = cand["params"]

        if name == "logreg":
            estimator = LogisticRegression(**params)
        elif name == "rf":
            estimator = RandomForestClassifier(**params)
        elif name == "xgb":
            estimator = XGBClassifier(**params)
        else:
            raise ValueError(f"Unknown model name: {name}")

        pipelines[name] = Pipeline([
            ("preprocessor", preprocessor),
            ("model", estimator)
        ])

    return pipelines
