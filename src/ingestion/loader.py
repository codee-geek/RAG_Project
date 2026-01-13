from pathlib import Path
from typing import List
import re
from langchain_core.documents import Document
from unstructured.partition.auto import partition


def is_clause_number(text: str) -> bool:
    """Check if text is just a clause number like 8.34 or 5.2.1"""
    return bool(re.fullmatch(r"\d+(\.\d+)+", text))


def load_documents_unstructured(file_path: str) -> List[Document]:
    """
    Load and parse ISO standard PDF into semantic sections.
    Handles clause numbers and section titles properly.
    """
    path = Path(file_path)
    elements = partition(filename=str(path))

    docs = []
    current_section = "PREFACE"
    current_clause = None  # Store clause number separately
    buffer = []
    current_pages = set()

    for el in elements:
        text = el.text.strip() if el.text else ""
        if not text:
            continue

        # Skip headers, footers, and standalone page numbers
        category = el.category if hasattr(el, 'category') else ""
        if category in ("Header", "Footer") or re.fullmatch(r"\d{1,3}", text):
            continue

        # Track page numbers
        page = getattr(el.metadata, "page_number", None)
        if page is not None:
            current_pages.add(page)

        # Handle headings and titles
        if category in ("Title", "Heading"):
            # Is it a clause number? Store but don't flush
            if is_clause_number(text):
                current_clause = text
                continue
            
            # Real section title - flush previous section
            if buffer and len(" ".join(buffer)) > 200:  # Only save substantial sections
                docs.append(
                    Document(
                        page_content="\n".join(buffer),
                        metadata={
                            "doc_id": "ISO27001-2022",
                            "doc_type": "standard",
                            "section_title": current_section,
                            "semantic_section": current_section,
                            "pages": sorted(current_pages),
                            "source": str(path),
                            "file_name": path.name,
                            "language": "en",
                            "char_count": len(" ".join(buffer)),
                        },
                    )
                )
            
            # Start new section with clause number if available
            current_section = f"{current_clause} {text}".strip() if current_clause else text
            current_clause = None
            buffer = []
            current_pages = set()
            continue

        # Regular content
        buffer.append(text)

    # Final flush
    if buffer and len(" ".join(buffer)) > 200:
        docs.append(
            Document(
                page_content="\n".join(buffer),
                metadata={
                    "doc_id": "ISO27001-2022",
                    "doc_type": "standard",
                    "section_title": current_section,
                    "semantic_section": current_section,
                    "pages": sorted(current_pages),
                    "source": str(path),
                    "file_name": path.name,
                    "language": "en",
                    "char_count": len(" ".join(buffer)),
                },
            )
        )

    return docs


def debug_print_documents(docs: List[Document], preview_chars: int = 400):
    """Print document details for inspection"""
    print(f"\n{'='*90}")
    print(f"TOTAL DOCUMENTS: {len(docs)}")
    print(f"{'='*90}\n")

    for i, doc in enumerate(docs, 1):
        print(f"DOCUMENT #{i}")
        print("-" * 90)
        
        print("METADATA:")
        for k, v in doc.metadata.items():
            print(f"  {k}: {v}")
        
        print(f"\nCONTENT PREVIEW ({preview_chars} chars):")
        preview = doc.page_content[:preview_chars]
        print(preview + ("..." if len(doc.page_content) > preview_chars else ""))
        print(f"\nFull length: {len(doc.page_content)} characters\n")


# Usage
if __name__ == "__main__":
    docs = load_documents_unstructured("data/iso27001.pdf")
    debug_print_documents(docs, preview_chars=300)
    
    # Summary
    print(f"\n{'='*90}")
    print("SUMMARY")
    print(f"{'='*90}")
    for i, doc in enumerate(docs, 1):
        print(f"{i}. {doc.metadata['section_title'][:60]:<60} | "
        f"Pages: {doc.metadata['pages'][0]}-{doc.metadata['pages'][-1]} | "
        f"{doc.metadata['char_count']} chars")