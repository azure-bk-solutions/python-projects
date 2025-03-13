import hashlib
import json
import time
from azure_blob_storage import AzureBlobStorage
from pinecone_handler import PineconeHandler
from config import CHANGE_LOG_FILE

class ChangeHandler:
    def __init__(self):
        self.blob_storage = AzureBlobStorage()
        self.vector_db = PineconeHandler()

    def compute_hash(self, content):
        """Computes SHA256 hash for document content."""
        return hashlib.sha256(content.encode()).hexdigest()

    def process_change_log(self):
        """Processes changes from the change log."""
        with open(CHANGE_LOG_FILE, "r") as file:
            for line in file:
                event = json.loads(line.strip())
                doc_id, change_type, timestamp, doc_hash = event.values()

                if change_type in ["create", "update"]:
                    content = self.blob_storage.download_document(doc_id).decode("utf-8")
                    self.vector_db.upsert_vector(doc_id, content)
                    print(f"✅ Processed {change_type} for {doc_id}")

                elif change_type == "delete":
                    self.vector_db.delete_vector(doc_id)
                    print(f"❌ Deleted {doc_id} from vector DB")

        # Clear log after processing
        open(CHANGE_LOG_FILE, "w").close()
        print(f"Change log file '{CHANGE_LOG_FILE}' cleared.")  # Add this line

    def simulate_change_event(self, doc_id, change_type, content=""):
        """Simulates a change event by writing to the log file."""
        event = {
            "DocumentId": doc_id,
            "ChangeType": change_type,
            "Timestamp": time.time(),
            "Hash": self.compute_hash(content) if content else None
        }
        with open(CHANGE_LOG_FILE, "a") as file:
            file.write(json.dumps(event) + "\n")
