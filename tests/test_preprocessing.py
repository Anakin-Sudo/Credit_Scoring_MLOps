import pandas as pd
from utilities.ml_processes import run_preprocessing_df


def test_run_preprocessing_df_split_and_stratify():
    """Ensure the preprocessing function splits and stratifies correctly."""
    # Create a small synthetic dataset
    df = pd.DataFrame({
        "Duration": [6, 12, 24, 36, 48, 60, 72, 84],
        "CreditAmount": [1000, 2000, 1500, 3000, 2500, 3500, 4000, 5000],
        "CreditRisk": [0, 1, 0, 1, 0, 1, 0, 1],
    })

    total_len = len(df)
    class_counts = df["CreditRisk"].value_counts(normalize=True)

    train_df, test_df = run_preprocessing_df(
        df=df,
        dropna_cols=None,
        drop_duplicates=True,
        rename_map=None,
        dtype_map=None,
        test_size=0.25,
        random_state=123,
        stratify_col="CreditRisk",
    )

    # Train/test sizes add up
    assert len(train_df) + len(test_df) == total_len

    # Stratification: class proportions preserved within ±2%
    train_counts = train_df["CreditRisk"].value_counts(normalize=True)
    test_counts = test_df["CreditRisk"].value_counts(normalize=True)

    for cls in class_counts.index:
        assert abs(train_counts.get(cls, 0) - class_counts[cls]) < 0.02
        assert abs(test_counts.get(cls, 0) - class_counts[cls]) < 0.02
