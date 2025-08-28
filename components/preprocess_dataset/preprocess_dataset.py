# azure_ml_preprocessing_component.py
import argparse
import pandas as pd
from utilities.ml_processes import preprocess_dataset
from utilities.azure_storage import download_csv_from_blob
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
import os


def main():

    # Load local env for non-sensitive config (like container name, workspace info)
    load_dotenv()

    # Get container name from .env (non-sensitive)
    container_name = os.getenv("AZURE_STORAGE_CONTAINER_NAME")
    storage_account_secret_name = os.getenv("STORAGE_ACCOUNT_CONN_STR_SECRET_NAME")

    # Get connection string securely from Key Vault
    keyvault_uri = os.getenv("KEYVAULT_URI")
    credential = DefaultAzureCredential()
    client = SecretClient(vault_url=keyvault_uri, credential=credential)
    conn_str = client.get_secret(storage_account_secret_name).value

    parser = argparse.ArgumentParser()
    parser.add_argument("--blob_name", type=str, help="Name of the blob in Azure Storage")
    parser.add_argument("--output_data", type=str, help="Path to save preprocessed dataset (CSV)")
    parser.add_argument("--dropna_cols", type=str, default=None, help="Columns to drop NA values (comma-separated)")
    parser.add_argument("--drop_duplicates", type=bool, default=True, help="Whether to drop duplicate rows")
    parser.add_argument("--rename_map", type=str, default=None, help="Column rename map (JSON string)")
    parser.add_argument("--dtype_map", type=str, default=None, help="Column dtype map (JSON string)")

    args = parser.parse_args()

    # Load dataset
    df = download_csv_from_blob(args.blob_name, conn_str, container_name)

    # Parse optional arguments
    dropna_cols = args.dropna_cols.split(",") if args.dropna_cols else None
    rename_map = eval(args.rename_map) if args.rename_map else None
    dtype_map = eval(args.dtype_map) if args.dtype_map else None

    # Preprocess dataset
    preprocessed_df = preprocess_dataset(
        df,
        dropna_cols=dropna_cols,
        drop_duplicates=args.drop_duplicates,
        rename_map=rename_map,
        dtype_map=dtype_map
    )

    # Save preprocessed dataset
    preprocessed_df.to_csv(args.output_data, index=False)


if __name__ == "__main__":
    main()