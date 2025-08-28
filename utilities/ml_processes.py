from sklearn.pipeline import Pipeline as Sk_pipeline
from imblearn.pipeline import Pipeline as Imb_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from category_encoders import TargetEncoder
import pandas as pd


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


def preprocess_dataset(df, dropna_cols=None, drop_duplicates=True, rename_map=None, dtype_map=None, feature_eng_func=None):
    """
    Basic data engineering preprocessing: schema cleanup and deterministic features.

    Parameters
    ----------
    df : pd.DataFrame
        Raw input dataset.
    dropna_cols : list, optional
        Drop rows with NA in these columns (default: None).
    drop_duplicates : bool, optional
        Whether to drop duplicate rows (default: True).
    rename_map : dict, optional
        Dictionary to rename columns, e.g. {'OldName': 'new_name'} (default: None).
    dtype_map : dict, optional
        Dictionary to cast dtypes, e.g. {'col1': 'int64', 'col2': 'category'} (default: None).
    feature_eng_func : callable, optional
        A function that takes df and returns df with engineered features (default: None).

    Returns
    -------
    df : pd.DataFrame
        Cleaned dataset ready for ML preprocessing/training.
    """

    # drop duplicates
    if drop_duplicates:
        df = df.drop_duplicates()

    # drop rows with missing values in specified columns
    if dropna_cols:
        df = df.dropna(subset=dropna_cols)

    # rename columns
    if rename_map:
        df = df.rename(columns=rename_map)

    # enforce data types
    if dtype_map:
        df = df.astype(dtype_map)

    # apply custom feature engineering if provided
    if feature_eng_func:
        df = feature_eng_func(df)

    return df
