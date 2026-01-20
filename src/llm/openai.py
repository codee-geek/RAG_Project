from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from core.config import LLM_MODEL, OPENAI_API_KEY, TEMPERATURE

_llm = None

def load_llm():
    global _llm

    if _llm is None:
        _llm = ChatOpenAI(
            model=LLM_MODEL,
            openai_api_key=OPENAI_API_KEY,
            temperature=TEMPERATURE,
        )

    return _llm

def generate_phi3(prompt: str, max_new_tokens: int = 256) -> str:
    llm = load_llm()

    messages = [HumanMessage(content=prompt)]
    response = llm.invoke(messages)
    text = response.content.strip()

    # Optional safety trim
    if "\n\n" in text:
        text = text.split("\n\n")[0]

    return text
