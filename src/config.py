import os
from pathlib import Path

from sentence_transformers import SentenceTransformer

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_DB_PATH = "vectorstore"
LLM_MODEL = "gpt-4o"
TOP_K = 5       
TEMPERATURE = 0.7
SOURCE_DIRECTORY = "data"
PERSIST_DIRECTORY = "vectorstore"
MAX_INPUT_SIZE = 4096
MAX_TOTAL_TOKENS = 8192

query = "What is the primary objective of ISO/IEC 27001:2022?"