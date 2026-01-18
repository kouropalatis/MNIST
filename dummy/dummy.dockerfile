FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

# install dependencies
RUN apt-get update && \
    apt-get install --no-install-recommends -y build-essential gcc && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-cache

COPY dummy/main.py main.py
WORKDIR /

ENTRYPOINT ["uv", "run", "python", "-u", "main.py"]