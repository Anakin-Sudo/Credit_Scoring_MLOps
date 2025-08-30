import argparse
import pandas as pd
import json
from utilities.ml_processes import run_preprocessing_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", type=str, help="Path to raw input dataset (CSV)")
    parser.add_argument("--train_output", type=str, help="Path to save preprocessed train dataset (CSV)")
    parser.add_argument("--test_output", type=str, help="Path to save preprocessed test dataset (CSV)")
    parser.add_argument("--dropna_cols", type=str, default=None, help="Columns to drop NA values (comma-separated)")
    parser.add_argument("--drop_duplicates", type=bool, default=True, help="Whether to drop duplicate rows")
    parser.add_argument("--rename_map", type=str, default=None, help="Column rename map (JSON string)")
    parser.add_argument("--dtype_map", type=str, default=None, help="Column dtype map (JSON string)")
    parser.add_argument("--test_size", type=float, default=0.2, help="Proportion of test split")
    parser.add_argument("--random_state", type=int, default=42, help="Seed for reproducibility")
    parser.add_argument("--stratify_col", type=str, default="target", help="Column name to stratify on (default 'target')")

    args = parser.parse_args()

    # Load dataset
    df = pd.read_csv(args.input_data)

    # Parse optional args
    dropna_cols = args.dropna_cols.split(",") if args.dropna_cols else None
    rename_map = json.loads(args.rename_map) if args.rename_map else None
    dtype_map = json.loads(args.dtype_map) if args.dtype_map else None

    # Call reusable preprocessing function (returns train/test DFs)
    train_df, test_df = run_preprocessing_df(
        df=df,
        dropna_cols=dropna_cols,
        drop_duplicates=args.drop_duplicates,
        rename_map=rename_map,
        dtype_map=dtype_map,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify_col=args.stratify_col
    )

    # Save results
    train_df.to_csv(args.train_output, index=False)
    test_df.to_csv(args.test_output, index=False)


if __name__ == "__main__":
    main()
