# 1. Base image
FROM python:3.12-slim

# 2. System dependencies
RUN apt-get update && \
    apt-get install --no-install-recommends -y build-essential gcc && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 3. Set a specific working directory
WORKDIR /app

# 4. Copy blueprints
COPY requirements.txt .
COPY pyproject.toml .

# 5. Optimized PIP Install using Cache Mount
# We remove --no-cache-dir here so it can actually USE the cache mount
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# 6. Copy the project
COPY src/ src/
COPY data/ data/

# 7. Install the project logic
RUN pip install . --no-deps --no-cache-dir

ENTRYPOINT ["python", "-u", "src/mnist/evaluate.py"]