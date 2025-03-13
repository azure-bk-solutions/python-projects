import time
from change_handler import ChangeHandler
from azure_blob_storage import AzureBlobStorage
import os

change_handler = ChangeHandler()
blob_storage = AzureBlobStorage()


file_path = "sample.txt"  # Or whatever path you *think* it is
absolute_path = os.path.abspath(file_path)
print(f"The absolute path being used is: {absolute_path}")

# Simulate Uploading a Document
file_path = "sample.txt"
blob_name = "sample.txt"
blob_storage.upload_document(file_path, blob_name)

# Simulate Change Event
change_handler.simulate_change_event(blob_name, "create", "This is a sample document.")

# Process Change Log (Trigger Event Handler)
try:
    print("\nProcessing Change Log...\n")
    change_handler.process_change_log()
except Exception as e:
    print(f"An error occurred while processing the change log: {e}")
