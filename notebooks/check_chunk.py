from config import EMBEDDING_MODEL, VECTOR_DB_PATH     
from src.ingestion.ingestion import chunks
# Load and process documents

for i, chunk in enumerate(chunks[:3]):
    print(f"Chunk {i}:")
    print(chunk.page_content)
    print(chunk.metadata)
