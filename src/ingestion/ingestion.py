from src.ingestion.loader import load_documents
from src.ingestion.cleaner import clean_documents
from src.ingestion.chunker import chunk_documents
from src.ingestion.index import create_vectorstore

def run_ingestion(path: str):
    docs = load_documents(path)
    docs = clean_documents(docs)
    chunks = chunk_documents(docs)
    create_vectorstore(chunks)

    # Show first 10 chunks for inspection
    # for i, chunk in enumerate(chunks[:10]):
    #     print(f"Chunk {i}:")
    #     print(chunk.page_content)
    #     print(chunk.metadata)
    # print(f"Total chunks: {len(chunks)}")
    
if __name__ == "__main__":
    run_ingestion("data/iso27001.pdf")
    
