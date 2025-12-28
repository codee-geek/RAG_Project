from src.ingestion.loader import load_documents
from src.ingestion.cleaner import clean_documents
from src.ingestion.chunker import chunk_documents
from src.ingestion.index import create_vectorstore



def run_ingestion(path: str):
    docs = load_documents(path)
    docs = clean_documents(docs)
    chunks = chunk_documents(docs)

    # Show first 10 chunks for inspection
    # for i, chunk in enumerate(chunks[:10]):
    #     print(f"Chunk {i}:")
    #     print(chunk.page_content)
    #     print(chunk.metadata)
    # print(f"Total chunks: {len(chunks)}")

    # Only create the vectorstore if the OpenAI API key is set
    from src.config import OPENAI_API_KEY
    if OPENAI_API_KEY:
        print("OPENAI_API_KEY found — creating vectorstore (this will call OpenAI embeddings).")
        create_vectorstore(chunks)
    else:
        print("OPENAI_API_KEY not found — skipping vectorstore creation. Set OPENAI_API_KEY to enable embeddings and vectorstore creation.")
    
     
if __name__ == "__main__":
    run_ingestion("data/iso27001.pdf")
    
