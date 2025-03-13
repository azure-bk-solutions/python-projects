from azure.storage.blob import BlobServiceClient
from config import AZURE_STORAGE_CONNECTION_STRING, AZURE_STORAGE_CONTAINER_NAME

class AzureBlobStorage:
    def __init__(self):
        self.blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
        self.container_client = self.blob_service_client.get_container_client(AZURE_STORAGE_CONTAINER_NAME)

    def upload_document(self, file_path, blob_name):
        """Uploads a document to Azure Blob Storage."""
        with open(file_path, "rb") as data:
            self.container_client.upload_blob(blob_name, data, overwrite=True)
            

    def download_document(self, blob_name):
        """Downloads a document from Azure Blob Storage."""
        blob_client = self.container_client.get_blob_client(blob_name)
        return blob_client.download_blob().readall()

    def delete_document(self, blob_name):
        """Deletes a document from Azure Blob Storage."""
        blob_client = self.container_client.get_blob_client(blob_name)
        blob_client.delete_blob()
