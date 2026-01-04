from src.ingestion.loader import load_documents_unstructured
from src.ingestion.cleaner import clean_documents
from src.ingestion.chunker import hybrid_chunk_documents
from src.ingestion.index import create_vectorstore


def run_ingestion(file_path: str):
    docs = load_documents_unstructured(file_path)
    docs = clean_documents(docs)
    chunks = hybrid_chunk_documents(docs)
    create_vectorstore(chunks)


if __name__ == "__main__":
    try:
        run_ingestion(
            "/Users/atharvawakade/Documents/VS_Projects/RAG_ATHARVA/data/iso27001.pdf"
        )
        print("Ingestion completed successfully.")
    except Exception as e:
        print("Ingestion failed:", e)
        raise
