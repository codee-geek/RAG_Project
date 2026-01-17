class RAGError(Exception):
    """
    Base exception for the RAG system.
    All custom exceptions MUST inherit from this.
    """
    pass


# =====================
# INGESTION ERRORS
# =====================

class DocumentLoadError(RAGError):
    """Raised when a document cannot be loaded or parsed."""
    pass


class CleaningError(RAGError):
    """Raised when text cleaning or normalization fails."""
    pass


class ChunkingError(RAGError):
    """Raised when document chunking fails or produces invalid chunks."""
    pass


class IndexingError(RAGError):
    """Raised when vector indexing or storage fails."""
    pass


# =====================
# RETRIEVAL ERRORS
# =====================

class RetrievalError(RAGError):
    """Raised when retrieval from vector store fails."""
    pass


class RerankingError(RAGError):
    """Raised when reranking logic fails."""
    pass


# =====================
# LLM ERRORS
# =====================

class LLMInitializationError(RAGError):
    """Raised when an LLM fails to load or initialize."""
    pass


class LLMGenerationError(RAGError):
    """Raised when the LLM fails during text generation."""
    pass


# =====================
# PIPELINE ERRORS
# =====================

class PipelineError(RAGError):
    """Raised when a high-level pipeline step fails."""
    pass


class InvalidSchemaError(RAGError):
    """Raised when Document or Chunk schema validation fails."""
    pass
