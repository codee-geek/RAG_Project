from src.ingestion.loader import load_documents_unstructured
from src.ingestion.cleaner import clean_documents
from src.ingestion.chunker import chunk_large_sections
from src.ingestion.index import create_vectorstore


def run_ingestion(file_path: str):
    docs = load_documents_unstructured(file_path)
    docs = clean_documents(docs)
    chunks = chunk_large_sections(docs)
    create_vectorstore(chunks)
    return chunks 



if __name__ == "__main__":
    try:
        run_ingestion(
            "data/lunamatic_fall.txt"
        )
        print("Ingestion completed successfully.")
    except Exception as e:
        print("Ingestion failed:", e)
        raise
