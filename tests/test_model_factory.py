import pandas as pd
from utilities.ml_processes import build_preprocessor
from utilities.model_factory import build_pipelines


def test_build_pipelines_and_preprocessor():
    """Verify preprocessing transformer and pipelines can be built and fitted."""

    # Synthetic dataset
    df = pd.DataFrame({
        "Duration": [6, 12, 24, 36, 48, 60],
        "CreditAmount": [1000, 2000, 1500, 3000, 2500, 3500],
        "Housing": ["own", "rent", "free", "own", "rent", "free"],
        "Job": ["skilled", "unskilled", "management", "skilled", "unskilled", "management"],
        "CreditRisk": [0, 1, 0, 1, 0, 1],
    })

    feature_groups = {
        "num_cols": ["Duration", "CreditAmount"],
        "simple_cat_cols": ["Housing"],
        "complex_cat_cols": ["Job"],
    }

    # Build preprocessor
    preprocessor = build_preprocessor(feature_groups)
    names = [name for name, _, _ in preprocessor.transformers]
    assert "num" in names
    assert "simple_cat" in names
    assert "complex_cat" in names

    # Candidate models
    candidates = [
        {"model": "logreg", "params": {"max_iter": 50}},
        {"model": "rf", "params": {"n_estimators": 5, "max_depth": 2}},
    ]
    pipelines = build_pipelines(candidates, preprocessor)

    assert "logreg" in pipelines
    assert "rf" in pipelines

    # Fit pipelines on synthetic data
    X = df.drop(columns=["CreditRisk"])
    y = df["CreditRisk"]

    for pipeline in pipelines.values():
        pipeline.fit(X, y)  # Should not raise
