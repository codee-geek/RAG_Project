# ingestion/loader/registry.py
from core.constants import DocumentType
from ingestion.loader.loader import load_generic_unstructured
from ingestion.loader.ISO_loader import load_iso_standard

LOADER_REGISTRY = {
    DocumentType.UNSTRUCTURED: lambda path: load_generic_unstructured(path, corpus_id="unstructured"),
    DocumentType.GENERAL_STRUCTURED: lambda path: load_generic_unstructured(path, corpus_id="general"),
    DocumentType.ISO_STRUCTURED: lambda path: load_iso_standard(path, standard_id="ISO27001", year="2022", corpus_id="ISO27001-2022"),
}
