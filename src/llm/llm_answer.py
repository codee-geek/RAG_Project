from llm.phi3 import generate_phi3, load_llm
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

Rules:
- Answer ONLY from the provided context.
- Answer ONLY the Question asked.
- Do NOT explain related concepts.
- Do NOT answer implied or follow-up questions.
- Do NOT generate new questions.
- Use ONLY the given context.
- Do NOT add external knowledge.
- If the answer is not present, say exactly: "Not specified in the document."
- Be concise, precise, and technical.
- Start directly with the answer.
- Use a numbered list only if actions are requested.

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
