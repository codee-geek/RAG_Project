# RAG_Project

src/
├── ingestion/
│   ├── loader.py
│   ├── cleaner.py
│   ├── chunker.py
│   ├── index.py
│
│── Query/
│   ├──user_input.py
│
├── config.py
├── ingest.py
|
|── vectorstore#


to run query file: python3 -m src.Query.user_input  
to run injestion file: python3 -m src.ingestion.ingestion


In this project we have requirement.txt plus pyproject.toml right now just for the ease of installing libraries. Once the project is completed perfectly only uv files will be kept and requirement.txt will be deleted adding all the important dependencies in the pyproject.toml files.


Stage 1 (Hybrid Retrieval): Expand your search to retrieve 20–50 chunks using a combination of vector search (semantic) and BM25 (keyword matching).
Stage 2 (Reranking): Pass those 50 chunks through a Reranker (like BGE-Reranker or Cohere Rerank). The Reranker will reorder them, moving the "hidden" best answers to the top.
Final Generation: Take only the Top 3–5 chunks from the reranked list and provide them to the LLM. 

chuncking unstructured problem is chunking layer wise abrubly 
so logic is to implement a min character aswell :

Merge texts within the same semantic section
Flush only when:
section changes, or
buffer has reached enough size and we want to cap growth

File path
 └─ Unstructured.partition()          ← semantic structure happens HERE
     └─ LangChain Documents
         └─ Cleaning / normalization
             └─ Chunking (Recursive / hybrid)
                 └─ Embeddings
                     └─ FAISS


Reranker-only budget

Out of total latency, reranking should consume:
Cross-encoder: 150–400 ms (top 20–50 docs)
Hybrid reranker: 80–250 ms
Bi-encoder rerank: 5–20 ms
If your reranker alone is >500 ms, your system will feel slow unless it’s an offline or research workflow.