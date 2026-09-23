import os
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = "qwen3:latest"
EMBEDDING_MODEL_NAME = "bge-m3"

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_EMBED_URL = f"{OLLAMA_BASE_URL}/api/embed"
OLLAMA_CHAT_URL = f"{OLLAMA_BASE_URL}/api/chat"

DATABASE_PATH = "database/data"
VECTOR_DB_PATH = "database/vector_db"

MAX_CATEGORICAL_SAMPLES = 10  
DEFAULT_TOP_K = 5

MAX_SQL_RETRIES = 3
MAX_CHART_RETRIES = 3
