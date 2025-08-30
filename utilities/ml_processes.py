from sklearn.pipeline import Pipeline as Sk_pipeline
from imblearn.pipeline import Pipeline as Imb_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from category_encoders import TargetEncoder
import pandas as pd
from sklearn.model_selection import train_test_split
from importlib import import_module


def make_learning_pipeline(num_cols, simple_cat_cols, complex_cat_cols, model, passthrough_cols=None):
    """
    Build a preprocessing + modeling pipeline.

    Parameters
    ----------
    num_cols : list
        List of numerical column names.
    simple_cat_cols : list
        List of low-cardinality categorical column names (one-hot encoding).
    complex_cat_cols : list
        List of high-cardinality categorical column names (target encoding).
    passthrough_cols : list, optional
        Columns to leave untouched (default: None).

    Returns
    -------
    pipeline : Imb_pipeline
        Full pipeline with preprocessing and model.
    """

    # numeric preprocessing: impute missing with mean, then standardize
    num_prep = Sk_pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # simple categorical preprocessing: impute with most frequent, then one-hot encode
    simple_cat_prep = Sk_pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encode', OneHotEncoder(handle_unknown='ignore'))
    ])

    # complex categorical preprocessing: impute with most frequent, then target encode
    complex_cat_prep = Sk_pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encode', TargetEncoder())
    ])

    # column transformer to combine all preprocessing steps
    transformers = [
        ('num_prep', num_prep, num_cols),
        ('simple_cat_prep', simple_cat_prep, simple_cat_cols),
        ('complex_cat_prep', complex_cat_prep, complex_cat_cols)
    ]

    if passthrough_cols:
        # passthrough columns remain as is
        transformers.append(('passthrough', 'passthrough', passthrough_cols))

    preprocessor = ColumnTransformer(transformers, remainder='drop')

    # final pipeline: preprocessing + model
    pipeline = Imb_pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    return pipeline


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