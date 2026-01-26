FROM python:3.12-slim AS base

RUN pip install uv

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml
COPY LICENSE LICENSE
COPY README.md README.md

RUN uv sync --frozen --no-install-project

COPY src src/

RUN uv sync --frozen

EXPOSE 8080
CMD ["sh", "-c", "uv run streamlit run src/mnist/frontend.py --server.port ${PORT:-8080} --server.address 0.0.0.0"]
