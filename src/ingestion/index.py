from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from src.config import EMBEDDING_MODEL, VECTOR_DB_PATH, OPENAI_API_KEY


def create_vectorstore(chunks):
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Set it in your environment (export OPENAI_API_KEY=\"sk-...\") or add it to a .env file at the project root and install python-dotenv."
        )

    # Pass the key explicitly to ensure the client uses the correct credential
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=OPENAI_API_KEY)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(VECTOR_DB_PATH)
    return vectorstore
