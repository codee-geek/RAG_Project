from pathlib import Path
from typing import List
import re
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from unstructured.partition.auto import partition


# =========================
# STAGE 1: LOAD
# =========================
def load_pdf(file_path: str) -> List[Document]:
    """Load PDF and convert to documents"""
    path = Path(file_path)
    elements = partition(filename=str(path))
    
    docs = []
    for el in elements:
        text = el.text.strip() if el.text else ""
        if not text:
            continue
        
        metadata = {
            "source": str(path),
            "filename": path.name,
            "category": el.category if hasattr(el, 'category') else "Unknown",
        }
        
        page = getattr(el.metadata, "page_number", None)
        if page is not None:
            metadata["page_number"] = page
        
        docs.append(Document(page_content=text, metadata=metadata))
    
    return docs


# =========================
# STAGE 2: CLEAN
# =========================
def clean_documents(docs: List[Document]) -> List[Document]:
    """
    Clean documents while preserving meaningful structure.
    - Remove headers/footers
    - Normalize whitespace but keep paragraph breaks
    - Filter noise
    """
    cleaned = []
    
    for doc in docs:
        text = doc.page_content
        category = doc.metadata.get("category", "")
        
        # Skip headers, footers, page numbers
        if category in ("Header", "Footer"):
            continue
        if re.fullmatch(r"\d{1,3}", text.strip()):
            continue
        
        # Normalize whitespace but preserve paragraph structure
        # Replace multiple spaces with one, but keep single newlines
        text = re.sub(r" +", " ", text)  # Multiple spaces → one space
        text = re.sub(r"\n\n+", "\n\n", text)  # Multiple newlines → double newline
        text = text.strip()
        
        # Filter very short fragments (likely noise)
        if len(text) < 20:
            continue
        
        # Remove common PDF artifacts
        text = re.sub(r"--``,,,,,``````,,,,,`,`,`,`,,`,-`-`,,`,,`,`,,`---", "", text)
        text = text.strip()
        
        if text:  # Ensure still has content after cleaning
            cleaned.append(
                Document(
                    page_content=text,
                    metadata=doc.metadata,
                )
            )
    
    return cleaned


# =========================
# STAGE 3: SEMANTIC SECTIONING
# =========================
def is_clause_number(text: str) -> bool:
    """Check if text is a clause number like 8.34"""
    return bool(re.fullmatch(r"\d+(\.\d+)+", text))


def split_into_semantic_sections(docs: List[Document]) -> List[Document]:
    """
    Group cleaned elements into semantic sections based on headings.
    This is your main chunking strategy.
    """
    if not docs:
        return []
    
    source_metadata = docs[0].metadata
    file_path = source_metadata.get("source", "unknown")
    file_name = source_metadata.get("filename", "unknown")
    
    sections = []
    current_section = "PREFACE"
    current_clause = None
    buffer = []
    current_pages = set()
    
    for doc in docs:
        text = doc.page_content.strip()
        if not text:
            continue
        
        category = doc.metadata.get("category", "")
        page = doc.metadata.get("page_number")
        if page:
            current_pages.add(page)
        
        # Detect headings
        if category in ("Title", "Heading"):
            # Clause number? Store it
            if is_clause_number(text):
                current_clause = text
                continue
            
            # Real heading - flush previous section
            if buffer and len(" ".join(buffer)) > 200:
                sections.append(
                    Document(
                        page_content="\n\n".join(buffer),  # Use double newline for paragraphs
                        metadata={
                            "doc_id": "ISO27001-2022",
                            "doc_type": "standard",
                            "section_title": current_section,
                            "pages": sorted(current_pages),
                            "source": file_path,
                            "file_name": file_name,
                            "language": "en",
                            "chunk_type": "semantic_section",
                        },
                    )
                )
            
            # Start new section
            current_section = f"{current_clause} {text}".strip() if current_clause else text
            current_clause = None
            buffer = []
            current_pages = set()
            continue
        
        # Regular content
        buffer.append(text)
    
    # Final flush
    if buffer and len(" ".join(buffer)) > 200:
        sections.append(
            Document(
                page_content="\n\n".join(buffer),
                metadata={
                    "doc_id": "ISO27001-2022",
                    "doc_type": "standard",
                    "section_title": current_section,
                    "pages": sorted(current_pages),
                    "source": file_path,
                    "file_name": file_name,
                    "language": "en",
                    "chunk_type": "semantic_section",
                },
            )
        )
    
    return sections


# =========================
# STAGE 4: CHUNK LARGE SECTIONS (Optional)
# =========================
def chunk_large_sections(docs: List[Document], max_size: int = 2000) -> List[Document]:
    """
    Only chunk sections that are TOO large.
    Most ISO sections should fit within reasonable size.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,  # Larger than your 800
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    
    final_docs = []
    
    for doc in docs:
        # If section is reasonably sized, keep as-is
        if len(doc.page_content) <= max_size:
            final_docs.append(doc)
        else:
            # Split large sections
            chunks = splitter.split_documents([doc])
            for i, chunk in enumerate(chunks):
                # Preserve original metadata and add chunk info
                chunk.metadata.update({
                    "chunk_type": "split_from_large_section",
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                })
                final_docs.append(chunk)
    
    return final_docs


# =========================
# COMPLETE PIPELINE
# =========================
def process_iso_document(file_path: str, force_chunk: bool = False) -> List[Document]:
    """
    Complete RAG pipeline for ISO standards.
    
    Args:
        file_path: Path to PDF
        force_chunk: If True, chunk ALL sections. If False, only chunk large ones.
    """
    print("Stage 1: Loading PDF...")
    raw_docs = load_pdf(file_path)
    print(f"  → Loaded {len(raw_docs)} elements")
    
    print("Stage 2: Cleaning...")
    cleaned_docs = clean_documents(raw_docs)
    print(f"  → {len(cleaned_docs)} elements after cleaning")
    
    print("Stage 3: Semantic sectioning...")
    sections = split_into_semantic_sections(cleaned_docs)
    print(f"  → Created {len(sections)} semantic sections")
    
    if force_chunk:
        print("Stage 4: Chunking all sections...")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        final_docs = splitter.split_documents(sections)
        print(f"  → Split into {len(final_docs)} chunks")
    else:
        print("Stage 4: Chunking only large sections...")
        final_docs = chunk_large_sections(sections, max_size=2000)
        print(f"  → {len(final_docs)} final documents")
    
    return final_docs


# =========================
# DEBUG UTILITIES
# =========================
def analyze_document_sizes(docs: List[Document]):
    """Show distribution of document sizes"""
    sizes = [len(d.page_content) for d in docs]
    
    print("\n" + "="*90)
    print("DOCUMENT SIZE ANALYSIS")
    print("="*90)
    print(f"Total documents: {len(docs)}")
    print(f"Average size: {sum(sizes)/len(sizes):.0f} chars")
    print(f"Min size: {min(sizes)} chars")
    print(f"Max size: {max(sizes)} chars")
    print(f"Median size: {sorted(sizes)[len(sizes)//2]} chars")
    
    # Size distribution
    ranges = [(0, 500), (500, 1000), (1000, 2000), (2000, 5000), (5000, float('inf'))]
    print("\nSize distribution:")
    for low, high in ranges:
        count = sum(1 for s in sizes if low <= s < high)
        label = f"{low}-{high}" if high != float('inf') else f"{low}+"
        print(f"  {label:12} chars: {count:3} documents")


# =========================
# USAGE
# =========================
if __name__ == "__main__":
    # Option 1: Keep semantic sections intact (RECOMMENDED for ISO)
    docs = process_iso_document("data/iso27001.pdf", force_chunk=False)
    
    # Option 2: Force chunk everything
    # docs = process_iso_document("data/iso27001.pdf", force_chunk=True)
    
    analyze_document_sizes(docs)
    
    # Preview
    print("\n" + "="*90)
    print("SAMPLE DOCUMENTS")
    print("="*90)
    for i, doc in enumerate(docs[:3], 1):
        print(f"\n{i}. {doc.metadata.get('section_title', 'Unknown')}")
        print(f"   Pages: {doc.metadata.get('pages', [])}")
        print(f"   Type: {doc.metadata.get('chunk_type', 'unknown')}")
        print(f"   Size: {len(doc.page_content)} chars")
        print(f"   Preview: {doc.page_content[:200]}...")