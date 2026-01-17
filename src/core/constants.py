from enum import Enum

class DocumentType(str, Enum):
    UNSTRUCTURED = "unstructured"
    GENERAL_STRUCTURED = "general_structured"
    ISO_STRUCTURED = "iso_structured"
