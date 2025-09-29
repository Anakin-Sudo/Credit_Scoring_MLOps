from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from category_encoders import TargetEncoder
import pandas as pd
from sklearn.model_selection import train_test_split
from importlib import import_module


def build_preprocessor(feature_groups: dict):
    """
    Build a preprocessing ColumnTransformer given feature groups.

    Args:
        feature_groups (dict): dictionary with keys:
            - num_cols
            - simple_cat_cols
            - complex_cat_cols

    Returns:
        sklearn ColumnTransformer
    """
    num_cols = feature_groups.get("num_cols", [])
    simple_cat_cols = feature_groups.get("simple_cat_cols", [])
    complex_cat_cols = feature_groups.get("complex_cat_cols", [])

    # Pipelines
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    simple_cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore"))
    ])

    complex_cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encode', TargetEncoder())
    ])

    return ColumnTransformer([
        ("num", num_pipeline, num_cols),
        ("simple_cat", simple_cat_pipeline, simple_cat_cols),
        ("complex_cat", complex_cat_pipeline, complex_cat_cols)
    ])


def run_preprocessing_df(
    df: pd.DataFrame,
    dropna_cols=None,
    drop_duplicates=True,
    rename_map=None,
    dtype_map=None,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify_col: str = "target"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply preprocessing to an in-memory DataFrame and split into train/test sets.

    Args:
        df (pd.DataFrame): Input dataset.
        dropna_cols (list[str], optional): Columns on which to drop NA rows.
        drop_duplicates (bool): Whether to drop duplicates.
        rename_map (dict, optional): Dict for renaming columns.
        dtype_map (dict, optional): Dict for casting dtypes.
        test_size (float): Proportion for test split.
        random_state (int): Seed for reproducibility.
        stratify_col (str): Column name to stratify on (e.g., "target").

    Returns:
        train_df (pd.DataFrame): Preprocessed training set.
        test_df (pd.DataFrame): Preprocessed test set.
    """

    # --- Cleaning ---
    if dropna_cols:
        df = df.dropna(subset=dropna_cols)

    if drop_duplicates:
        df = df.drop_duplicates()

    if rename_map:
        df = df.rename(columns=rename_map)

    if dtype_map:
        df = df.astype(dtype_map)

    # --- Splitting ---
    stratify_vals = df[stratify_col] if stratify_col and stratify_col in df.columns else None

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_vals
    )

    return train_df, test_df


def load_model(class_path: str, params: dict):
    """
    Dynamically import and instantiate a model class from its string path.

    Example:
        load_model("sklearn.ensemble.RandomForestClassifier", {"n_estimators": 200})
    """
    try:
        module_name, class_name = class_path.rsplit(".", 1)
        module = import_module(module_name)
        cls = getattr(module, class_name)
        return cls(**params)
    except Exception as e:
        raise ValueError(f"Could not load model {class_path} with params {params}") from e


def select_best_model(candidates, selection_criteria):
    """
    Select the best model based on CV metrics and tie-breaking rules.

    Args:
        candidates (list[dict]): List of candidates with 'metrics' dicts.
        selection_criteria (dict): Config block with 'primary', 'tiebreaker', 'min_threshold'.
    Returns:
        dict: Best candidate (with metrics, uri, etc.)
    """
    primary_metric = selection_criteria["primary"]
    min_threshold = selection_criteria.get("min_threshold", 0.0)
    tiebreakers = selection_criteria.get("tiebreaker", [])
    equality_threshold = tiebreakers[0].get("equality_threshold", 0.0) if tiebreakers else 0.0

    # 1. Filter out candidates below min_threshold on the primary metric
    valid = [c for c in candidates if c["metrics"].get(primary_metric, 0) >= min_threshold]
    if not valid:
        raise ValueError(f"No candidate reached min_threshold {min_threshold} on {primary_metric}")

    # 2. Sort candidates by primary metric
    valid.sort(key=lambda c: c["metrics"].get(primary_metric, 0), reverse=True)
    best, second = valid[0], valid[1] if len(valid) > 1 else None

    # 3. Apply tie-breakers if top two are "too close"
    if second:
        diff = abs(
            best["metrics"].get(primary_metric, 0) -
            second["metrics"].get(primary_metric, 0)
        )
        if diff <= equality_threshold:
            for tb in tiebreakers:
                metric = tb["metric"]
                best_metric = best["metrics"].get(metric, 0)
                second_metric = second["metrics"].get(metric, 0)
                if second_metric > best_metric:
                    best = second
                    break

    return best
