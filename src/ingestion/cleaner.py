import re
from langchain_core.documents import Document


def clean_documents(docs: list[Document]) -> list[Document]:
    cleaned = []

    for doc in docs:
        text = doc.page_content

        text = re.sub(r"\s+", " ", text).strip()

        if len(text) < 20:
            continue

        cleaned.append(
            Document(
                page_content=text,
                metadata=doc.metadata,
            )
        )

    return cleaned
