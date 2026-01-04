from unstructured.partition.auto import partition
from langchain_core.documents import Document
from pathlib import Path


def load_documents_unstructured(file_path: str) -> list[Document]:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(file_path)

    elements = partition(filename=str(path))

    docs = []
    current_section = "unknown"

    for el in elements:
        text = el.text.strip() if el.text else ""
        if not text:
            continue

        if el.category in ("Title", "Heading"):
            current_section = text
            continue

        docs.append(
            Document(
                page_content=text,
                metadata={
                    "file_name": path.name,
                    "file_path": str(path),
                    "file_type": path.suffix[1:],
                    "semantic_section": current_section,
                    "element_type": el.category,
                },
            )
        )

    return docs
