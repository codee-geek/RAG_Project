# RAG_Project

src/
├── ingestion/
│   ├── loader.py
│   ├── cleaner.py
│   ├── chunker.py
│   ├── index.py
│
├── config.py
├── ingest.py



to run query file: python3 -m src.Query.user_input  
to run injestion file: python3 -m src.ingestion.ingestion


In this project we have requirement.txt plus pyproject.toml right now just for the ease of installing libraries. Once the project is completed perfectly only uv files will be kept and requirement.txt will be deleted adding all the important dependencies in the pyproject.toml files.