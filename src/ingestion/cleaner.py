def clean_documents(docs):
    cleaned = []
    for doc in docs:
        text = doc.page_content.strip()
        if len(text) < 20:
            continue
        doc.page_content = text
        cleaned.append(doc)
    return cleaned
