# openhands-runtime: the OpenHands SDK as one self-contained tree.
#
# Not an image any task runs IN. overlay.yaml starts it as a sidecar whose only
# job is to hand /opt/openhands-runtime to `main` through a named volume, so the
# agent gets its SDK without the bundle's Dockerfile knowing OpenHands exists
# and without anything being installed at trial time -- the agent phase runs
# with egress closed, so a pip install there would fail.
#
# Self-contained means the interpreter lives INSIDE the tree too (uv's managed
# CPython under python/, the venv under venv/). A venv built against this
# image's system Python would point at a /usr/local/bin/python3 that main may
# not have, or may have at another version. Mounted at the same path in main,
# every symlink in the venv still resolves.
#
# Built by scripts/run_task.sh's preflight (ensure_image), or by hand:
#   docker build -t openhands-runtime:latest tools/openhands_agent
#
# Base: bookworm, the glibc the task images here are built on (python:3.12-slim).
# Wheels are resolved for this libc, so a main image with an OLDER glibc than
# bookworm's, or a musl (alpine) one, cannot load them.
FROM debian:bookworm-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.13 /uv /usr/local/bin/uv

ENV UV_PYTHON_INSTALL_DIR=/opt/openhands-runtime/python \
    UV_PYTHON_PREFERENCE=only-managed \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_NO_CACHE=1

COPY requirements.lock /opt/openhands-runtime/requirements.lock

# uv carries its own TLS roots, so the base needs no ca-certificates package.
RUN uv python install 3.12 \
    && uv venv /opt/openhands-runtime/venv --python 3.12 \
    && uv pip install --python /opt/openhands-runtime/venv/bin/python \
         --require-hashes -r /opt/openhands-runtime/requirements.lock

# Fail the build, not the trial, if the tree cannot import what the runner uses.
RUN OPENHANDS_SUPPRESS_BANNER=1 LITELLM_LOCAL_MODEL_COST_MAP=True \
    /opt/openhands-runtime/venv/bin/python -c "import openhands.sdk, openhands.tools.terminal, openhands.tools.file_editor, openhands.tools.task_tracker, openhands.sdk.mcp; print('openhands-sdk', openhands.sdk.__version__)"

# The sidecar only has to stay up while main mounts the volume; overlay.yaml
# runs it with init so it stops on SIGTERM instead of waiting out the timeout.
CMD ["sleep", "infinity"]
