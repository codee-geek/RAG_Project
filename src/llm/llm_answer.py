from llm.openai import generate_phi3, load_llm
from query.retriever import run_query

# =========================
# CONTEXT BUILDER
# =========================
def build_context(docs, max_chars: int = 3000) -> str:
    """
    Build grounded context from retrieved documents.
    Preserves section boundaries.
    """
    blocks = []
    used = 0

    for doc in docs:
        section = doc.metadata.get("section_title", "Unknown Section")
        text = doc.page_content.strip()

        block = f"[{section}]\n{text}\n"
        if used + len(block) > max_chars:
            break

        blocks.append(block)
        used += len(block)

    return "\n---\n".join(blocks)


# =========================
# PROMPT BUILDER (FIXED)
# =========================
def build_prompt(query: str, context: str) -> str:
    return f"""You are an expert assistant.

Follow this decision process strictly:
1. Check whether the context contains information relevant to the question.
2. If relevant information exists:
   - Answer using only that information.
   - If the question is broad, summarize the document purpose and scope.
   - Cite clause numbers when present (e.g., Clauses 4–10).
3. If no relevant information exists at all:
   - Respond exactly with: "Not specified in the document."

Rules:
- Use ONLY the provided context.
- Answer ONLY the question asked.
- Do NOT add external knowledge.
- Be concise, precise, and technical.
- Start directly with the answer.

Context:
{context}

Question:
{query}

Answer:
"""



# =========================
# MAIN ANSWER FUNCTION
# =========================
def answer_query(query: str):
    # 1. Retrieve + rerank
    docs = run_query(query)

    # 2. Build grounded context
    context = build_context(docs)

    # 3. Build prompt (llama.cpp safe)
    prompt = build_prompt(query, context)

    # 4. Generate answer
    answer = generate_phi3(prompt)

    return {
        "answer": answer,
        "chunks": docs
    }


# =========================
# CLI ENTRY
# =========================
# if __name__ == "__main__":

#     # LOAD MODEL ONCE
#     load_llm()
#     print("Model loaded. Ready for queries.")

#     while True:
#         user_query = input("\nEnter your query (or 'exit'): ")
#         if user_query.lower() == "exit":
#             break

#         answer, sources = answer_query(user_query)

#         print("\nANSWER:\n")
#         print(answer)

#         print("\nSOURCES:\n")
#         for doc in sources:
#             print(
#                 f"- {doc.metadata.get('section_title')}"
#                 f" (pages {doc.metadata.get('pages')})"
#             )
