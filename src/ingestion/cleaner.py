import re
def clean_documents(docs):
    cleaned = []
    for doc in docs:
        text = doc.page_content or ""
        # Normalize line endings
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        cleaned_lines = []
        for line in text.split('\n'):
            s = line.strip()
            if not s:
                continue
            # Skip very short garbage lines
            if len(s) < 5:
                continue
            # Drop lines that are mostly punctuation/symbols
            alnum = sum(1 for c in s if c.isalnum())
            if len(s) > 0 and (alnum / len(s)) < 0.3:
                continue
            # Remove page number lines like 'Page 1' or just '1'
            if re.match(r'^(page\s*)?\d+$', s.lower()):
                continue
            # If the line is mostly dots/dashes, skip it; otherwise replace long runs of punctuation with a space
            if re.fullmatch(r'[\.\-`,_\s]+', s):
                continue
            s = re.sub(r'[\.\-`,_]{3,}', ' ', s)
            # Collapse repeated punctuation like '!!!' -> '!'
            s = re.sub(r'([^\w\s])\1{2,}', r"\1", s)
            cleaned_lines.append(s)
        new_text = ' '.join(cleaned_lines)
        # Normalize whitespace
        new_text = re.sub(r'\s+', ' ', new_text).strip()
        if len(new_text) < 20:
            continue
        doc.page_content = new_text
        cleaned.append(doc)
    return cleaned