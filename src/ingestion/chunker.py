from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
import re
from langchain_core.documents import Document

# =========================
# STAGE 4: CHUNK LARGE SECTIONS (Optional)
# =========================
def chunk_large_sections(docs: List[Document], max_size: int = 2000) -> List[Document]:
    """
    Only chunk sections that are TOO large.
    Most ISO sections should fit within reasonable size.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,  # Larger than your 800
        chunk_overlap=120,
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


