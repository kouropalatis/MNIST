# Use uv base image
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

# Set the working directory inside the container
WORKDIR /code

# Copy the project files first
COPY pyproject.toml uv.lock ./

# Install dependencies using uv sync
# --no-dev: Excludes testing tools to keep the image small
RUN uv sync --frozen --no-cache --no-dev

# Copy the entire src directory (maintains your mnist/ package)
COPY ./src /code/src

# Install the project itself (without dev dependencies)
RUN uv sync --frozen --no-cache --no-dev

# Expose port 80 for web traffic
EXPOSE 80

# Run the app using uv run
CMD ["uv", "run", "uvicorn", "ml_app:app", "--host", "0.0.0.0", "--port", "80", "--app-dir", "src/mnist"]
