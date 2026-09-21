FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim@sha256:e5b65587bce7de595f299855d7385fe7fca39b8a74baa261ba1b7147afa78e58

WORKDIR /opt/openhands-adapter

ARG OPENHANDS_SDK_VERSION=0.31.2
RUN uv pip install --system --no-cache \
        "openhands-agent-sdk==${OPENHANDS_SDK_VERSION}" \
        "pydantic>=2.0" \
        "tomli"

COPY entrypoint.py /opt/openhands-adapter/entrypoint.py
COPY openhands.config.toml /opt/openhands-adapter/openhands.config.toml

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python3", "/opt/openhands-adapter/entrypoint.py"]
