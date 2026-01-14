from src.llm.phi3 import generate_phi3
from src.Query.user_input import run_query
from src.llm.phi3 import load_llm
# =========================
# PROMPT
# =========================
SYSTEM_PROMPT = """You are an expert assistant.
Answer ONLY from the provided context.
Rules:
Answer ONLY the Question asked.
Do NOT explain related concepts.
Do NOT answer implied or follow-up questions.
Do NOT generate new questions.
If multiple actions are listed, enumerate them.
- Use ONLY the given context
- Do NOT add external knowledge
- If the answer is not present, say exactly:
   "Not specified in the document."
- Be concise, precise, and technical
Answer format:
- Start directly with the answer.
- Use a numbered list if actions are requested.
- Do not include headings, questions, or explanations.

"""

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


def build_prompt(query: str, context: str) -> str:
    return f"""<|system|>
{SYSTEM_PROMPT}
<|user|>
{context}

{query}
<|assistant|>
"""

# =========================
# MAIN ANSWER FUNCTION
# =========================
def answer_query(query: str) -> str:
    # 1. Retrieve + rerank
    docs = run_query(query)

    # 2. Build grounded context
    context = build_context(docs)

    # 3. Build Phi-3 prompt
    prompt = build_prompt(query, context)

    # 4. Generate answer
    output = generate_phi3(prompt)

    # 5. Remove prompt echo if present
    answer = output
    return answer, docs

# =========================
# CLI ENTRY
# =========================

if __name__ == "__main__":

    # LOAD ONCE
    load_llm()
    print("Model loaded. Ready for queries.")

    while True:
        user_query = input("\nEnter your query (or 'exit'): ")
        if user_query.lower() == "exit":
            break

        answer, sources = answer_query(user_query)

        print("\nANSWER:\n")
        print(answer)

        print("\nSOURCES:\n")
        for doc in sources:
            print(
                f"- {doc.metadata.get('section_title')} "
                f"(pages {doc.metadata.get('pages')})"
            )