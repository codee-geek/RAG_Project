def load_documents_unstructured(file_path: str) -> list[Document]:
    path = Path(file_path)
    elements = partition(filename=str(path))

    docs = []
    current_section = None
    buffer = []

    for el in elements:
        text = el.text.strip() if el.text else ""
        if not text:
            continue

        if el.category in ("Title", "Heading"):
            # Flush previous section
            if buffer and current_section:
                docs.append(
                    Document(
                        page_content="\n".join(buffer),
                        metadata={
                            "semantic_section": current_section,
                            "file_name": path.name,
                            "file_path": str(path),
                        },
                    )
                )
                buffer = []

            current_section = text
            continue

        buffer.append(text)

    # Flush last section
    if buffer and current_section:
        docs.append(
            Document(
                page_content="\n".join(buffer),
                metadata={
                    "semantic_section": current_section,
                    "file_name": path.name,
                    "file_path": str(path),
                },
            )
        )

    return docs
