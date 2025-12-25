from ingestion.loader import load_documents
from ingestion.cleaner import clean_documents
from ingestion.chunker import chunk_documents
from ingestion.index import create_vectorstore


def run_ingestion(path: str):
    docs = load_documents(path)
    docs = clean_documents(docs)
    chunks = chunk_documents(docs)
    create_vectorstore(chunks)


if __name__ == "__main__":
    run_ingestion("data/sample.pdf")
