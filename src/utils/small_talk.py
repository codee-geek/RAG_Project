# utils/small_talk.py

SMALL_TALK_KEYWORDS = {
    "hi", "hello", "hey",
    "good morning", "good night",
    "thanks", "thank you",
    "ok", "okay",
    "i don't understand",
    "can you repeat",
    "help", "yes", "no"
}

def is_small_talk(query: str) -> bool:
    q = query.lower().strip()
    return (
        len(q.split()) <= 4 and
        any(phrase in q for phrase in SMALL_TALK_KEYWORDS)
    )

def small_talk_response(query: str) -> str:
    q = query.lower()

    if "good morning" in q:
        return "Good morning. How can I help you with the documents?"
    if "good night" in q:
        return "Good night. You can ask me about the uploaded documents anytime."
    if "thank" in q:
        return "You’re welcome."
    if "i don't understand" in q:
        return "No problem. Tell me what you want me to explain."
    if q in {"ok", "okay"}:
        return "Alright."

    return "How can I help you with the documents?"
