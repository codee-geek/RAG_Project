from pathlib import Path
from typing import List, Optional
import re
from langchain_core.documents import Document
from unstructured.partition.auto import partition

DEFAULT_MIN_CHARS = 400
DEFAULT_MAX_CHARS = 1200

def load_generic_unstructured(
    file_path: str,
    *,
    corpus_id: str,
    min_chars: int = DEFAULT_MIN_CHARS,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> List[Document]:
    """
    Generic loader for arbitrary unstructured documents.
    Suitable for PDFs, DOCX, TXT, stories, articles, notes.
    """

    path = Path(file_path)
    files = [path] if path.is_file() else list(path.glob("*"))

    docs: List[Document] = []

    for file in files:
        if not file.is_file():
            continue

        elements = partition(filename=str(file))

        buffer = []
        current_pages = set()

        for el in elements:
            text = el.text.strip() if el.text else ""
            if not text:
                continue

            category = getattr(el, "category", "")
            if category in {"Header", "Footer"}:
                continue

            # Skip standalone page numbers
            if re.fullmatch(r"\d{1,4}", text):
                continue

            page = getattr(el.metadata, "page_number", None)
            if page is not None:
                current_pages.add(page)
            buffer.append(text)
            current_len = sum(len(x) for x in buffer)

            # Flush when max size reached
            if current_len >= max_chars:
                docs.append(
                    _emit_chunk(
                        buffer,
                        file=file,
                        corpus_id=corpus_id,
                        pages=current_pages,
                    )
                )
                buffer = []
                current_pages = set()

        # Final flush (even if small — NO SILENT DROP)
        if buffer:
            docs.append(
                _emit_chunk(    
                    buffer,
                    file=file,
                    corpus_id=corpus_id,
                    pages=current_pages,
                )
            )
    return docs

def _emit_chunk(buffer, *, file: Path, corpus_id: str, pages: set) -> Document:
    text = "\n".join(buffer)
    return Document(
        page_content=text,
        metadata={
            "doc_id": file.stem,
            "corpus_id": corpus_id,
            "doc_family": "generic_unstructured",
            "source": str(file),
            "file_name": file.name,
            "pages": sorted(pages),
            "char_count": len(text),
        },
    )
