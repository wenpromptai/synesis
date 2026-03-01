# ---- Builder: install deps + project with uv ----
FROM python:3.12-slim-trixie AS builder

COPY --from=ghcr.io/astral-sh/uv:0.10 /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

WORKDIR /app

# Install dependencies (cached layer — only re-runs when lock changes)
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project

# Copy source and install project
COPY . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-editable

# ---- Runtime: clean image with just .venv ----
FROM python:3.12-slim-trixie

RUN groupadd --system --gid 999 app \
 && useradd --system --gid 999 --uid 999 --create-home app

WORKDIR /app

# Copy built venv from builder
COPY --from=builder --chown=app:app /app/.venv /app/.venv

ENV PATH="/app/.venv/bin:$PATH"

USER app

EXPOSE 7337

CMD ["uvicorn", "synesis.main:app", "--host", "0.0.0.0", "--port", "7337"]
