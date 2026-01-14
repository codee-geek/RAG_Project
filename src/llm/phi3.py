from pyexpat import model
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

#=========================
#stoping criteria   
#=========================
from transformers import StoppingCriteria, StoppingCriteriaList

class StopOnNextTurn(StoppingCriteria):
    def __init__(self, tokenizer):
        self.stop_ids = [
            tokenizer.encode("\nWhat", add_special_tokens=False),
            tokenizer.encode("\nEnter your query", add_special_tokens=False),
            tokenizer.encode("\nwhat shall", add_special_tokens=False),
        ]

    def __call__(self, input_ids, scores, **kwargs):
        for stop in self.stop_ids:
            if input_ids[0][-len(stop):].tolist() == stop:
                return True
        return False

# =========================
# TEXT GENERATION
# =========================
def generate_phi3(prompt: str, max_new_tokens=256):
    model, tokenizer = load_llm()

    inputs = tokenizer(prompt, return_tensors="pt")

    stopping_criteria = StoppingCriteriaList(
        [StopOnNextTurn(tokenizer)]
    )

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        stopping_criteria=stopping_criteria,
    )

    # 🔴 Decode ONLY generated tokens
    input_len = inputs["input_ids"].shape[-1]
    generated_ids = outputs[0][input_len:]

    text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True
    ).strip()

    # Optional safety trim
    if "\n\n" in text:
        text = text.split("\n\n")[0]

    return text




