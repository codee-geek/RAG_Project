from sentence_transformers import CrossEncoder
from langchain_core.documents import Document
from typing import List

cross_encoder = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    max_length=512
)

def cross_encoder_rerank(
    query: str,
    docs: List[Document],
    top_n: int = 10
) -> List[Document]:

    if not docs:
        return []

    # 🔥 FIX STARTS HERE — enrich reranker input
    pairs = []
    for doc in docs:
        section = doc.metadata.get("semantic_section", "Unknown Section")

        enriched_text = (
            f"Section: {section}\n"
            f"Content: {doc.page_content}"
        )

        pairs.append((query, enriched_text))
    # 🔥 FIX ENDS HERE

    rerank_scores = cross_encoder.predict(pairs)

    scored_docs = []

    for doc, rerank_score in zip(docs, rerank_scores):
        faiss_score = doc.metadata.get("faiss_score", 1.0)

        # FAISS returns distance → invert
        final_score = (
            0.7 * float(rerank_score)
            + 0.3 * (1 - float(faiss_score))
        )

        doc.metadata["rerank_score"] = float(rerank_score)
        doc.metadata["final_score"] = final_score

        scored_docs.append(doc)

    scored_docs.sort(
        key=lambda d: d.metadata["final_score"],
        reverse=True
    )

    return scored_docs[:top_n]
