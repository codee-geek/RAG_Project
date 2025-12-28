import os
from pathlib import Path

# Optional: load variables from a .env file if python-dotenv is installed
try:
    from dotenv import load_dotenv
    this_dir = Path(__file__).resolve().parent
    project_root = this_dir.parent
    # load .env from src/ (if present) then project root (if present)
    load_dotenv(this_dir / ".env")
    load_dotenv(project_root / ".env")
except Exception:
    # dotenv not installed or failed to load; environment variables will be used
    pass

# If the project root .env uses `export OPENAI_API_KEY="..."` style, parse it as a fallback
if "OPENAI_API_KEY" not in os.environ:
    env_path = project_root / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if "OPENAI_API_KEY" in line:
                parts = line.split("=", 1)
                if len(parts) == 2:
                    val = parts[1].strip().strip('"').strip("'")
                    val = val.removeprefix("export ").strip()
                    os.environ["OPENAI_API_KEY"] = val
                break

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

EMBEDDING_MODEL = "text-embedding-3-large"
VECTOR_DB_PATH = "vectorstore"
LLM_MODEL = "gpt-4o"
TOP_K = 5       
TEMPERATURE = 0.7
SOURCE_DIRECTORY = "data"
PERSIST_DIRECTORY = "vectorstore"
MAX_INPUT_SIZE = 4096
MAX_TOTAL_TOKENS = 8192

# Read OpenAI API key from environment (or .env if present)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

query = "What is the primary objective of ISO/IEC 27001:2022?"