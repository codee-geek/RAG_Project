from pathlib import Path
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
)

def load_documents(path: str):
    suffix = Path(path).suffix.lower()

    if suffix == ".pdf":
        return PyPDFLoader(path).load()
    elif suffix == ".txt":
        return TextLoader(path).load()
    elif suffix == ".csv":
        return CSVLoader(path).load()
    else:
        raise ValueError(f"Unsupported file type: {suffix}")