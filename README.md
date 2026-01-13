✅ What CAN be set in loader.py

✔ doc_id
✔ doc_type
✔ page
✔ source_path
✔ domain / standard name

✅ This should be done after:

title detection
heading parsing
semantic merge
chunking

add answer confidence scoring

retrieved_docs = run_query(query)

temp_chunks = []
for doc in retrieved_docs:
    temp_chunks.extend(
        splitter.split_text(doc.page_content)
    )

# lightweight scoring
scores = mini_embeddings.similarity(query, temp_chunks)

final_context = top_k(temp_chunks, scores)
