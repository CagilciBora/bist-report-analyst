# One image for both the API and the UI (they share the code).
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /usr/local/bin/uv
WORKDIR /app
# Install dependencies first (cached layer), then copy the code.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY . .
RUN uv sync --frozen --no-dev
ENV PATH="/app/.venv/bin:$PATH" HF_HOME=/hf-cache
EXPOSE 8000 8501
CMD ["uvicorn", "kaprag.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
