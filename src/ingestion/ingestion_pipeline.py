from typing import List

from core.constants import DocumentType
from core.exception import PipelineError
from core.schema import Document, Chunk

from ingestion.loader.registry import LOADER_REGISTRY
from ingestion.cleaner import clean_documents
from ingestion.chunker import chunk_large_sections
from ingestion.index import (
    create_vectorstore,
    reset_vectorstore,
)


def run_ingestion(
    upload_dir: str,
    document_type: DocumentType,
    reset_index: bool = True,
) -> List[Chunk]:
    """
    Orchestrates the full ingestion pipeline.

    Responsibilities:
    - Select loader based on explicit document type
    - Run cleaning, chunking, and indexing
    - Stay completely ignorant of document semantics (ISO/general/etc.)

    Parameters
    ----------
    upload_dir : str
        Directory containing uploaded documents
    document_type : DocumentType
        Explicit user-selected document type
    reset_index : bool
        Whether to reset the vectorstore before ingestion

    Returns
    -------
    List[Chunk]
        Final chunks that were indexed
    """

    # -----------------------------
    # 1. Resolve loader (NO branching logic here)
    # -----------------------------
    try:
        loader_func = LOADER_REGISTRY[document_type]
    except KeyError:
        raise PipelineError(f"Unsupported document type: {document_type}")

    # -----------------------------
    # 2. Reset vectorstore (optional)
    # -----------------------------
    if reset_index:
        reset_vectorstore()

    # -----------------------------
    # 3. Load documents (strategy injected)
    # -----------------------------
    try:
        documents: List[Document] = loader_func(upload_dir)
    except Exception as e:
        raise PipelineError(f"Document loading failed: {e}") from e

    if not documents:
        raise PipelineError("Loader returned zero documents")

    # -----------------------------
    # 4. Clean documents
    # -----------------------------
    try:
        cleaned_documents = clean_documents(documents)
    except Exception as e:
        raise PipelineError(f"Document cleaning failed: {e}") from e

    # -----------------------------
    # 5. Chunk documents
    # -----------------------------
    try:
        chunks: List[Chunk] = chunk_large_sections(cleaned_documents)
    except Exception as e:
        raise PipelineError(f"Document chunking failed: {e}") from e

    if not chunks:
        raise PipelineError("Chunker produced zero chunks")

    # -----------------------------
    # 6. Index chunks
    # -----------------------------
    try:
        create_vectorstore(chunks)
    except Exception as e:
        raise PipelineError(f"Vectorstore creation failed: {e}") from e

    return chunks
