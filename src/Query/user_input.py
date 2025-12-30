from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from src.config import VECTOR_DB_PATH, EMBEDDING_MODEL

def run_query(user_query: str):
    
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True}
    )

    vectorstore = FAISS.load_local(
        VECTOR_DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    results = vectorstore.similarity_search(user_query, k=5)

    for i, doc in enumerate(results):
        print(f"\n--- Chunk {i} ---")
        print(doc.page_content)

if __name__ == "__main__":
    user_query = input("Enter your query: ")
    run_query(user_query)
