from dataclasses import dataclass
from typing import Dict, Any

@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    source: str
    text: str
    metadata: Dict[str, Any]

@dataclass(frozen=True)
class Chunk:
    chunk_id: int
    doc_id: str
    title: str
    text: str
    metadata: Dict[str, Any]
