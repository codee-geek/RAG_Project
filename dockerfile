FROM python:3.11-slim

WORKDIR /app

# 1️⃣ Install system-level build tools (REQUIRED for ML libs)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 2️⃣ Install uv
RUN pip install --no-cache-dir uv

# 3️⃣ Copy dependency files
COPY pyproject.toml uv.lock ./

# 4️⃣ Install Python dependencies
RUN uv sync --no-cache

# 5️⃣ Copy source code
COPY src ./src

# 6️⃣ Run as module
CMD ["python", "-m", "src.main"]
