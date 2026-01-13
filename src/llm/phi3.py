import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.Query.user_input import run_query

_model = None
_tokenizer = None
# =========================
# CONFIG
# =========================
MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"

# =========================
# LOAD MODEL (ONCE)
# =========================
def load_llm():
    global _model, _tokenizer

    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-3-mini-4k-instruct")
        _model = AutoModelForCausalLM.from_pretrained(
            "microsoft/phi-3-mini-4k-instruct",
            torch_dtype=torch.float16,   # still slow, we fix later
            device_map="cpu"
        )

    return _model, _tokenizer


# =========================
# TEXT GENERATION
# =========================
def generate_phi3(prompt: str, max_new_tokens=526):
    model, tokenizer = load_llm()

    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)



