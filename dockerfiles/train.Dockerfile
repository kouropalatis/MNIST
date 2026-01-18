# Base image with uv pre-installed
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

RUN apt-get update && \
    apt-get install --no-install-recommends -y build-essential gcc && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy configuration
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen --no-cache

# Copy project files
COPY src/ src/
COPY data/ data/

# Install the project itself
RUN uv sync --frozen --no-cache

# Set working directory
WORKDIR /app

# Run using uv
ENTRYPOINT ["uv", "run", "python", "-u", "src/mnist/train.py"]