from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from config import EMBEDDING_MODEL, VECTOR_DB_PATH


def create_vectorstore(chunks):
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(VECTOR_DB_PATH)
    return vectorstore
