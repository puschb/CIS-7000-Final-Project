FROM nvcr.io/nvidia/pytorch:25.08-py3
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV UV_BREAK_SYSTEM_PACKAGES=1

COPY pyproject.toml /tmp/pyproject.toml
RUN uv pip install --system -r /tmp/pyproject.toml && rm /tmp/pyproject.toml

WORKDIR /app
