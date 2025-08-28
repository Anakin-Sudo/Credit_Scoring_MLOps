# azure_ml_ingestion_component.py
# Not implemented yet, but this component will handle data ingestion from meaningful Azure sources.
import argparse
import os
from dotenv import load_dotenv
from utilities.azure_storage import download_csv_from_blob


def main():
    # Load environment variables
    load_dotenv()
    conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    container_name = os.getenv("AZURE_STORAGE_CONTAINER_NAME")

    parser = argparse.ArgumentParser()
    parser.add_argument("--blob_name", type=str, help="Name of the blob in Azure Storage")
    parser.add_argument("--output_csv", type=str, help="Path to save downloaded CSV file")

    args = parser.parse_args()

    # Download CSV from Azure Blob Storage
    df = download_csv_from_blob(args.blob_name, conn_str, container_name)

    # Save the DataFrame to the specified output path
    df.to_csv(args.output_csv, index=False)
    print(f"Downloaded CSV from blob '{args.blob_name}' and saved to '{args.output_csv}'.")


if __name__ == "__main__":
    main()