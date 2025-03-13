import os

# 🔹 Required Secrets (Must be Set) - These will throw an error if missing
try:
    AZURE_LLM_ENDPOINT = os.environ['AZURE_LLM_END_POINT']
    AZURE_LLM_API_VERSION = os.environ['AZURE_LLM_API_VERSION']
    AZURE_LLM_MODEL_NAME = os.environ['AZURE_LLM_MODEL_NAME']
    AZURE_LLM_ENGINE = os.environ['AZURE_LLM_ENGINE']
    AZURE_EMBED_ENDPOINT = os.environ['AZURE_EMBED_END_POINT']
    AZURE_EMBED_MODEL = os.environ['AZURE_EMBED_MODEL']
    AZURE_EMBED_API_VERSION = os.environ['AZURE_EMBED_API_VERSION']
    AZURE_INFERENCE_CREDENTIAL = os.environ['AZURE_OPENAI_KEY']
    PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
except KeyError as e:
    raise EnvironmentError(f"Missing required environment variable: {e}")

# 🔹 Optional Configs (Provide Defaults for Local Testing)
AZURE_STORAGE_ACCOUNT_NAME = os.getenv("AZURE_STORAGE_ACCOUNT_NAME", "default_account")
AZURE_STORAGE_CONTAINER_NAME = os.getenv("AZURE_STORAGE_CONTAINER_NAME", "default_container")
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "default_connection_string")

# 🔹 Pinecone Configuration (Optional Defaults)
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "blooming-cypress")

# 🔹 Change Log File (Simulating a Message Queue)
CHANGE_LOG_FILE = "change_log.txt"
