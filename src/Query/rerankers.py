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
    top_n: int = 5
) -> List[Document]:

    if not docs:
        return []

    pairs = [(query, doc.page_content) for doc in docs]
    scores = cross_encoder.predict(pairs)

    scored_docs = list(zip(docs, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)

    reranked_docs = []
    for doc, score in scored_docs[:top_n]:
        doc.metadata["rerank_score"] = float(score)
        reranked_docs.append(doc)

    return reranked_docs
