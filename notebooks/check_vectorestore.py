from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from src.config import EMBEDDING_MODEL, VECTOR_DB_PATH 

vectorstore = FAISS.load_local(
    VECTOR_DB_PATH,
    OpenAIEmbeddings(model=EMBEDDING_MODEL),
    allow_dangerous_deserialization=True
)

# Number of vectors
print(vectorstore.index.ntotal)

# Peek at stored documents
docs = vectorstore.docstore._dict
print(list(docs.values())[:2])
