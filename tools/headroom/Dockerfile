# The agent's prompt compressor: compress_proxy.py plus the headroom library.
#
# NOT `headroom proxy`. That rewrites the whole request and deleted Claude
# Code's deferred tool loading, which ended two runs with
# "400 Tool reference 'tool_search_tool_regex' not found in available tools".
# compress_proxy.py touches `messages` and nothing else, and only when the
# result passes a structural check -- see its docstring.
#
# The library, not the [proxy] extra: that extra pulls transformers, onnxruntime
# and numpy for features nothing here uses, and cost ~400 MB of image and the
# same again in a container the docker VM did not have to spare.
#
# Build with the scoring tree as a named context, like the judge, so
# grader_compress.py is baked in rather than mounted:
#
#   docker build --build-context scoring=services/scoring -t headroom-compress:latest tools/headroom
#
# scripts/run_task.sh (stage_preflight) and `make build-headroom-compress` do.

FROM python:3.12-slim@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea

RUN pip install --no-cache-dir --root-user-action=ignore "headroom-ai>=0.37,<0.38" \
    && mkdir -p /opt/tiktoken \
    && TIKTOKEN_CACHE_DIR=/opt/tiktoken python3 -c \
       "import tiktoken; tiktoken.get_encoding('o200k_base'); tiktoken.get_encoding('cl100k_base')" \
    && chmod -R a+rX /opt/tiktoken

COPY --from=scoring grader_compress.py /app/
COPY compress_proxy.py /app/

# The vocab is baked because the container cannot fetch it: squid allows
# api.anthropic.com only. Without it token counts fall back to estimation.
# The offline flags stop headroom reaching for HuggingFace, litellm's price
# list and an onnxruntime telemetry host on import; squid denies them anyway,
# but each one costs retries and startup seconds.
ENV TIKTOKEN_CACHE_DIR=/opt/tiktoken \
    HEADROOM_OFFLINE=1 \
    HEADROOM_TELEMETRY=off \
    HEADROOM_UPDATE_CHECK=off \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    LITELLM_LOCAL_MODEL_COST_MAP=True \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8787

RUN useradd --create-home --uid 1000 compress
USER compress
EXPOSE 8787
ENTRYPOINT ["python3", "/app/compress_proxy.py"]
