# app/document_loader.py

import os
from PyPDF2 import PdfReader
from azure.storage.blob import BlobServiceClient
from io import BytesIO


def extract_text_from_local(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text



def extract_text_from_bytes(file_bytes: bytes) -> str:
    reader = PdfReader(BytesIO(file_bytes))  # <-- This is the key
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text



def extract_text_from_blob(blob_name):
    connect_str = os.getenv("BLOB_CONNECTION_STRING")
    container_name = os.getenv("BLOB_CONTAINER_NAME")

    blob_service_client = BlobServiceClient.from_connection_string(connect_str)
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)

    download_stream = blob_client.download_blob()
    data = download_stream.readall()

    temp_file = "temp.pdf"
    with open(temp_file, "wb") as f:
        f.write(data)

    text = extract_text_from_local(temp_file)
    os.remove(temp_file)

    return text

# def extract_text(mode="local", filename=None):
#     if mode == "local":
#         file_path = os.path.join("data", filename)
#         return extract_text_from_local(file_path)
#     elif mode == "cloud":
#         return extract_text_from_blob(filename)
#     else:
#         raise ValueError("Invalid mode. Use 'local' or 'cloud'.")




def extract_text(mode="local", filename=None, file_bytes=None):
    if mode == "local":
        file_path = os.path.join("data", filename)
        return extract_text_from_local(file_path)
    
    elif mode == "cloud":
        return extract_text_from_blob(filename)
    
    elif mode == "memory":
        if not file_bytes:
            raise ValueError("file_bytes must be provided for memory mode")
        reader = PdfReader(BytesIO(file_bytes))
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
        return text
    
    else:
        raise ValueError("Invalid mode. Use 'local', 'cloud', or 'memory'.")

