from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from core.config import VECTOR_DB_PATH, EMBEDDING_MODEL
from query.rerankers import cross_encoder_rerank

# Load ONCE
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    encode_kwargs={"normalize_embeddings": True}
)

def run_query(user_query: str, k: int = 30):
    
    vectorstore = FAISS.load_local(
    VECTOR_DB_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)
    results = vectorstore.similarity_search_with_score(user_query, k=k)

    docs = []
    for doc, score in results:
        doc.metadata["faiss_score"] = float(score)
        docs.append(doc)

    reranked_docs = cross_encoder_rerank(
        query=user_query,
        docs=docs,
        top_n=10
    )

    return reranked_docs



if __name__ == "__main__":
    user_query = input("Enter your query: ")
    final_docs = run_query(user_query)

    for i, doc in enumerate(final_docs):
        print(f"\n--- Reranked Chunk {i} ---")
        print(f"Rerank Score: {doc.metadata['rerank_score']}")
        print(doc.page_content)
