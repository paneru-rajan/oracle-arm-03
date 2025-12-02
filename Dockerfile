FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Set working directory
WORKDIR /app

# Copy dependency definitions
COPY pyproject.toml .

# Install dependencies
RUN uv sync --frozen --no-install-project || uv sync --no-install-project

# Copy the application code
COPY embed ./embed

# Run the application
CMD ["uv", "run", "uvicorn", "embed.main:app", "--host", "0.0.0.0", "--port", "8000"]
