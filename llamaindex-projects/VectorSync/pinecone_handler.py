from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
#from llama_index.embeddings.openai import OpenAIEmbedding
#from llama_index.core.schema import TextNode
from llama_index.embeddings.azure_inference import AzureAIEmbeddingsModel
from llama_index.core.schema import TextNode
from config import PINECONE_API_KEY,  PINECONE_INDEX_NAME, AZURE_EMBED_MODEL, AZURE_INFERENCE_CREDENTIAL, AZURE_EMBED_API_VERSION, AZURE_EMBED_ENDPOINT
from typing import List
    
class PineconeHandler:
    def __init__(self):
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
         # Create the index if it doesn't exist
        existing_indexes: List[str] = self.pc.list_indexes().names
        if PINECONE_INDEX_NAME not in existing_indexes():
            self.pc.create_index(
                PINECONE_INDEX_NAME,
                dimension=1536,  # Ensure it matches your embedding model
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
        # Connect to the Pinecone index
        self.pinecone_index = self.pc.Index(PINECONE_INDEX_NAME)
       # self.pc.index = pc.create_index(PINECONE_INDEX_NAME, dimension=3072, metric="cosine", spec=ServerlessSpec(cloud="aws", region="us-east-1"))
        em = AzureAIEmbeddingsModel(
            endpoint=AZURE_EMBED_ENDPOINT,
            credential=AZURE_INFERENCE_CREDENTIAL,
            api_version=AZURE_EMBED_API_VERSION,
            model_name=AZURE_EMBED_MODEL)
        self.embed_model = em

    def generate_embedding(self, text):
        """Generates an embedding vector from text content."""
        return self.embed_model.get_text_embedding(text)

    def upsert_vector(self, document_id, text):
        """Generates an embedding and upserts into Pinecone."""
        vector = self.generate_embedding(text)
        self.pinecone_index.upsert(vectors=[(document_id, vector)])

    def delete_vector(self, document_id):
        """Deletes a vector from Pinecone."""
        self.pinecone_index.delete(ids=[document_id])
