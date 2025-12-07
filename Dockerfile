FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /home
COPY uv.lock pyproject.toml ./
RUN uv sync

COPY . /home

ENV PYTHONPATH=/home/app
WORKDIR /home/app

CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]