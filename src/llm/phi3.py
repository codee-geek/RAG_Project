from llama_cpp import Llama
from query.retriever import run_query

# =========================
# LOAD MODEL (ONCE)
# =========================

_llm = None

def load_llm():
    global _llm

    if _llm is None:
        _llm = Llama(
            model_path="models/phi3-q4.gguf",   # adjust if needed
            n_ctx=4096,
            n_threads=8,        # physical cores
            n_batch=512,
            temperature=0.0,
            repeat_penalty=1.1,
            stop=[
                "\nQuestion:",
                "\nEnter your query",
                "\nwhat shall",
            ],
            verbose=False,
        )

    return _llm


# =========================
# TEXT GENERATION
# =========================
def generate_phi3(prompt: str, max_new_tokens: int = 256) -> str:
    llm = load_llm()

    out = llm(
        prompt,
        max_tokens=max_new_tokens,
    )

    text = out["choices"][0]["text"].strip()

    # Optional safety trim (same intent as before)
    if "\n\n" in text:
        text = text.split("\n\n")[0]

    return text
