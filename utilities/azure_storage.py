import json
import pandas as pd
from azure.storage.blob import BlobServiceClient
from PyPDF2 import PdfReader
from io import BytesIO


def download_json_from_blob(blob_name: str, conn_str, container_name) -> list[dict]:

    blob_service = BlobServiceClient.from_connection_string(conn_str)
    container_client = blob_service.get_container_client(container_name)
    blob_client = container_client.get_blob_client(blob_name)

    if not blob_client.exists():
        raise FileNotFoundError(f"Blob {blob_name} not found.")

    download_stream = blob_client.download_blob()
    content = download_stream.readall().decode("utf-8")

    try:
        documents = json.loads(content)
        assert isinstance(documents, list)
    except (json.JSONDecodeError, AssertionError) as e:
        raise ValueError(f"Blob {blob_name} does not contain a valid JSON list.") from e

    print(f'Downloaded {len(documents)} documents from {blob_name}')
    return documents


def upload_json_to_blob(documents, blob_name: str, conn_str, container_name):

    blob_service = BlobServiceClient.from_connection_string(conn_str)
    container_client = blob_service.get_container_client(container_name)

    if not container_client.exists():
        container_client.create_container()

    blob_client = container_client.get_blob_client(blob_name)

    buffer = BytesIO()
    buffer.write((json.dumps(documents, ensure_ascii=False, indent=2)).encode("utf-8"))
    buffer.seek(0)

    blob_client.upload_blob(buffer, overwrite=True)
    print(f'Uploaded {blob_name} to Azure Blob Storage.')


def download_csv_from_blob(blob_name: str, conn_str, container_name) -> pd.DataFrame:
    """
    Reads a CSV file from Azure Blob Storage and returns it as a Pandas DataFrame.

    Parameters:
    - blob_name: Name of the blob in Azure Storage.
    - conn_str: Azure Blob Storage connection string.
    - container_name: Name of the container in Azure Storage.

    Returns:
    - A Pandas DataFrame containing the CSV data.
    """
    blob_service = BlobServiceClient.from_connection_string(conn_str)
    container_client = blob_service.get_container_client(container_name)
    blob_client = container_client.get_blob_client(blob_name)

    if not blob_client.exists():
        raise FileNotFoundError(f"Blob {blob_name} not found.")

    download_stream = blob_client.download_blob()
    csv_data = pd.read_csv(BytesIO(download_stream.readall()))
    print(f"Read CSV data from {blob_name}")
    return csv_data


def upload_csv_to_blob(df: pd.DataFrame, blob_name: str, conn_str, container_name):
    """
    Uploads a Pandas DataFrame as a CSV file to Azure Blob Storage.

    Parameters:
    - df: The Pandas DataFrame to upload.
    - blob_name: Name of the blob in Azure Storage.
    - conn_str: Azure Blob Storage connection string.
    - container_name: Name of the container in Azure Storage.
    """
    blob_service = BlobServiceClient.from_connection_string(conn_str)
    container_client = blob_service.get_container_client(container_name)

    if not container_client.exists():
        container_client.create_container()

    blob_client = container_client.get_blob_client(blob_name)

    buffer = BytesIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)

    blob_client.upload_blob(buffer, overwrite=True)
    print(f"Uploaded DataFrame as CSV to {blob_name} in Azure Blob Storage.")


def download_pdf_from_blob(blob_name: str, conn_str, container_name) -> str:
    """
    Reads a PDF file from Azure Blob Storage and returns its content as text.

    Parameters:
    - blob_name: Name of the blob in Azure Storage.
    - conn_str: Azure Blob Storage connection string.
    - container_name: Name of the container in Azure Storage.

    Returns:
    - A string containing the text content of the PDF.
    """
    blob_service = BlobServiceClient.from_connection_string(conn_str)
    container_client = blob_service.get_container_client(container_name)
    blob_client = container_client.get_blob_client(blob_name)

    if not blob_client.exists():
        raise FileNotFoundError(f"Blob {blob_name} not found.")

    download_stream = blob_client.download_blob()
    pdf_reader = PdfReader(BytesIO(download_stream.readall()))
    pdf_text = ""
    for page in pdf_reader.pages:
        pdf_text += page.extract_text()
    print(f"Read PDF content from {blob_name}")
    return pdf_text


def upload_pdf_to_blob(pdf_data: BytesIO, blob_name: str, conn_str, container_name):
    """
    Uploads a PDF file to Azure Blob Storage.

    Parameters:
    - pdf_data: A BytesIO object containing the PDF data.
    - blob_name: Name of the blob in Azure Storage.
    - conn_str: Azure Blob Storage connection string.
    - container_name: Name of the container in Azure Storage.
    """
    blob_service = BlobServiceClient.from_connection_string(conn_str)
    container_client = blob_service.get_container_client(container_name)

    if not container_client.exists():
        container_client.create_container()

    blob_client = container_client.get_blob_client(blob_name)

    pdf_data.seek(0)  # Ensure the buffer is at the beginning
    blob_client.upload_blob(pdf_data, overwrite=True)
    print(f"Uploaded PDF to {blob_name} in Azure Blob Storage.")