from pathlib import Path
from typing import List, Optional
import re
from langchain_core.documents import Document
from unstructured.partition.auto import partition


def is_iso_clause_number(text: str) -> bool:
    return bool(re.fullmatch(r"\d+(\.\d+)+", text))


def load_iso_standard(
    file_path: str,
    *,
    standard_id: str,
    year: Optional[str] = None,
    corpus_id: str,
) -> List[Document]:
    """
    Load ISO/IEC/NIST-style standards into semantic sections.
    """
    path = Path(file_path)
    files = [path] if path.is_file() else list(path.glob("*"))

    doc_uid = standard_id if year is None else f"{standard_id}-{year}"

    all_docs = []

    for file in files:
        if not file.is_file():
            continue

        elements = partition(filename=str(file))

        current_section = "PREFACE"
        current_clause = None
        buffer = []
        current_pages = set()

        for el in elements:
            text = el.text.strip() if el.text else ""
            if not text:
                continue

            category = getattr(el, "category", "")
            if category in {"Header", "Footer"} or re.fullmatch(r"\d{1,3}", text):
                continue

            page = getattr(el.metadata, "page_number", None)
            if page is not None:
                current_pages.add(page)

            if category in {"Title", "Heading"}:
                if is_iso_clause_number(text):
                    current_clause = text
                    continue

                if buffer:
                    section_text = "\n".join(buffer)
                    all_docs.append(
                        Document(
                            page_content=section_text,
                            metadata={
                                "doc_id": doc_uid,
                                "corpus_id": corpus_id,
                                "doc_family": "iso_standard",
                                "section_title": current_section,
                                "semantic_section": f"{doc_uid} :: {current_section}",
                                "pages": sorted(current_pages),
                                "source": str(file),
                                "file_name": file.name,
                                "char_count": len(section_text),
                            },
                        )
                    )

                current_section = (
                    f"{current_clause} {text}".strip() if current_clause else text
                )
                current_clause = None
                buffer = []
                current_pages = set()
                continue

            buffer.append(text)

        if buffer:
            section_text = "\n".join(buffer)
            all_docs.append(
                Document(
                    page_content=section_text,
                    metadata={
                        "doc_id": doc_uid,
                        "corpus_id": corpus_id,
                        "doc_family": "iso_standard",
                        "section_title": current_section,
                        "semantic_section": f"{doc_uid} :: {current_section}",
                        "pages": sorted(current_pages),
                        "source": str(file),
                        "file_name": file.name,
                        "char_count": len(section_text),
                    },
                )
            )

    return all_docs
