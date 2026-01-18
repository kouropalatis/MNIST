# 1. Base image with uv
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# 2. System dependencies
RUN apt-get update && \
    apt-get install --no-install-recommends -y build-essential gcc && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 3. Set a specific working directory
WORKDIR /app

# 4. Copy blueprints
COPY pyproject.toml uv.lock ./

# 5. Helper config to install into system python if desired, 
# but sticking to uv venv is usually safer. 
# We will just run uv sync.
RUN uv sync --frozen --no-cache

# 6. Copy the project
COPY src/ src/
COPY data/ data/

# 7. Install the project logic (re-syncing to install the root project)
RUN uv sync --frozen --no-cache

ENTRYPOINT ["uv", "run", "python", "-u", "src/mnist/evaluate.py"]