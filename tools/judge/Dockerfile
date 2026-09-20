# The judge container: every scored channel runs in here, and nothing else does.
#
#   rubric        codex (gpt-5.6-sol) over /tests/rubric.json
#   channel A     pytest over /tests/test_outputs.py
#   state Rc/Rb   /tests/state_dump.py, over MCP against light-servers
#   ledger        /harness/scoring/tests/grade.py, over the bundle's weights
#
# The grading scripts are the harness's, mounted with the rest of
# services/scoring, so the numbers cannot drift from what main used to produce or
# between one bundle and the next. What this image supplies is exactly what they
# need: python, pytest, the mcp client, and codex.
#
# Build with the scoring tree as a named context -- the shared graders are baked
# in, not mounted:
#
#   docker build --build-context scoring=services/scoring -t codex-judge:latest tools/judge
#
# scripts/run_task.sh (ensure_image) and `make build-codex-judge` both pass it.

# Stage 1: npm ships codex as a node launcher plus a native binary. Only the
# binary is kept, so the final image carries no node runtime (~300 MB saved).
# The vendor directory is named per architecture; the glob copies whichever one
# this build produced, so amd64 and arm64 build from the same line.
FROM node:22-bookworm-slim@sha256:83f487e0a63425e5b4d146fb5e5be574bcbe1b7b843d3ebafdd95eaf7767a7e5 AS codex
ARG CODEX_VERSION=0.154.0
RUN npm install -g "@openai/codex@${CODEX_VERSION}" \
    && mkdir -p /out \
    && cp /usr/local/lib/node_modules/@openai/codex/node_modules/@openai/codex-linux-*/vendor/*/bin/codex /out/codex \
    && cp /usr/local/lib/node_modules/@openai/codex/node_modules/@openai/codex-linux-*/vendor/*/codex-path/rg /out/rg \
    && /out/codex --version

FROM python:3.12-slim@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea

# pytest runs the bundles' Channel A suite; mcp is what tests/state_dump.py
# speaks to light-servers. Nothing else: the graders import only these and the
# standard library (checked across every bundle).
RUN pip install --no-cache-dir --root-user-action=ignore pytest mcp

# Optional grader-path Headroom (services/scoring/grader_compress.py). OFF by
# default: the library pulls litellm and friends, ~440 MB on an image every run
# uses, and measured 0 to 3% saving on real trajectories. Built as its own tag
# when GRADER_HEADROOM_ENABLED=true, by scripts/run_task.sh or:
#
#   docker build --build-context scoring=services/scoring \
#     --build-arg WITH_HEADROOM=1 -t codex-judge-headroom:latest tools/judge
#
# The tiktoken vocab is baked in the same layer because the judge cannot
# download it: its squid allows chatgpt.com and auth.openai.com only.
ARG WITH_HEADROOM=0
RUN if [ "$WITH_HEADROOM" = "1" ]; then \
        pip install --no-cache-dir --root-user-action=ignore "headroom-ai>=0.37,<0.38" \
        && mkdir -p /opt/tiktoken \
        && TIKTOKEN_CACHE_DIR=/opt/tiktoken python3 -c \
           "import tiktoken; tiktoken.get_encoding('o200k_base'); tiktoken.get_encoding('cl100k_base')" \
        && chmod -R a+rX /opt/tiktoken; \
    fi

COPY --from=codex /out/codex /usr/local/bin/codex
COPY --from=codex /out/rg /usr/local/bin/rg
COPY --from=scoring rubric_judge_cli.py grader_compress.py /judge/
COPY codexbridge.py /judge/

RUN useradd --create-home --uid 1000 judge \
    && mkdir -p /logs/verifier /tests /workspace \
    && chown judge:judge /logs/verifier
USER judge
ENV CODEX_HOME=/home/judge/.codex \
    JUDGE_PORT=8770 \
    JUDGE_MODEL=gpt-5.6-sol \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# What headroom needs to run offline, carried by the image rather than by the
# caller. These used to live only in overlay-judge-headroom.yaml, which
# scripts/run_task.sh applies and which does NOT travel with a delivered
# bundle. A recipient who set GRADER_HEADROOM_ENABLED=true on a headroom build
# therefore got a library that reaches for HuggingFace and litellm's price list
# on import, against a squid that allows chatgpt.com and auth.openai.com only.
# Here, GRADER_HEADROOM_ENABLED is the single switch, which is what it reads
# like from the outside.
#
# Inert on a WITH_HEADROOM=0 build: nothing that reads them is installed.
# GRADER_HEADROOM_ENABLED is deliberately NOT set -- compression stays opt-in
# (services/scoring/grader_compress.py: "Default OFF. Compression is opted
# into, never inherited.").
ENV TIKTOKEN_CACHE_DIR=/opt/tiktoken \
    HEADROOM_OFFLINE=1 \
    HEADROOM_TELEMETRY=off \
    HEADROOM_UPDATE_CHECK=off \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    LITELLM_LOCAL_MODEL_COST_MAP=True
EXPOSE 8770
ENTRYPOINT ["python3", "/judge/codexbridge.py"]
