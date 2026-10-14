# syntax=docker/dockerfile:1
# Build: dependencies come from uv.lock only (--locked), so the image cannot drift from CI.
FROM python:3.12-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.13.0 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --no-install-project
COPY src ./src
RUN uv sync --locked --no-dev

FROM python:3.12-slim
# Run as an unprivileged user; nothing in this service needs to write to disk.
RUN useradd --system --uid 10001 --no-create-home --shell /usr/sbin/nologin agent
WORKDIR /app
COPY --from=build --chown=agent:agent /app /app
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1 LANGSMITH_TRACING=false
USER agent
EXPOSE 8090
HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8090/health', timeout=2).status == 200 else 1)"
CMD ["uvicorn", "rag_agent.main:app", "--host", "0.0.0.0", "--port", "8090"]
