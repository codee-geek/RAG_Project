from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from core.config import EMBEDDING_MODEL, VECTOR_DB_PATH


def create_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True}
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(VECTOR_DB_PATH)

    return vectorstore

import shutil
from pathlib import Path

VECTORSTORE_DIR = Path("vectorstore")


from pathlib import Path
import shutil
from core.config import VECTOR_DB_PATH

def reset_vectorstore():
    path = Path(VECTOR_DB_PATH)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
