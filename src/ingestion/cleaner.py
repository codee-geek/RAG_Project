from typing import List
import re
from langchain_core.documents import Document


def clean_documents(docs: List[Document]) -> List[Document]:
    """
    Clean documents while preserving meaningful structure.
    Works for any document type (PDFs, Word, etc.)
    """
    cleaned = []
    
    for doc in docs:
        text = (doc.page_content or "").strip()
        if not text:
            continue
        
        category = doc.metadata.get("category", "")
        
        # Remove standalone page numbers (1, 2, 3, etc.)
        if re.fullmatch(r"\d{1,3}", text):
            continue
        
        # GARBAGE FILTER: Remove lines with too many special characters
        # (Like: --``,,,,,``````,,,,`,,`,`,-,,`,`,`,,`---)
        lines = text.split('\n')
        clean_lines = []
        
        for line in lines:
            line_stripped = line.strip()
            if not line_stripped:
                clean_lines.append(line)  # Keep empty lines for structure
                continue
            
            # Count alphanumeric characters
            alphanumeric = len(re.findall(r'[a-zA-Z0-9]', line_stripped))
            total = len(line_stripped)
            
            # Keep lines with at least 30% real content
            if total > 0 and (alphanumeric / total) >= 0.30:
                clean_lines.append(line)
        
        text = '\n'.join(clean_lines)
        
        # Fix hyphenated line breaks (from v1 - this is GOOD)
        # Example: "informa-\ntion" → "information"
        text = re.sub(r"-\n(?=[a-z])", "", text)
        
        # Normalize whitespace (but preserve paragraph breaks)
        text = re.sub(r"[ \t]+", " ", text)  # Multiple spaces → one space
        text = re.sub(r"\n{3,}", "\n\n", text)  # Triple+ newlines → double
        text = text.strip()
        
        # Filter very short fragments (but allow meaningful short text)
        if len(text) < 15:
            continue
        
        if text:
            cleaned.append(
                Document(
                    page_content=text,
                    metadata=dict(doc.metadata),  # Safe copy (from v1)
                )
            )
    
    return cleaned