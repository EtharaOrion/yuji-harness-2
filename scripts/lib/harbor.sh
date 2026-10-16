#!/usr/bin/env bash
# Single source of truth for which harbor this harness runs on, and for how to
# install it on THIS machine. Sourced by scripts/run_task.sh so the pin and the
# "how to fix it" message can never drift apart again. Run directly to install:
#
#   bash scripts/lib/harbor.sh install
#
# PINNED on purpose, and pinned by hash. scripts/patch_harbor.py edits harbor's
# own source by matching exact anchor strings measured against this release,
# and refuses any other version outright. A version string alone is not an
# immutable revision, so the install resolves nothing: it installs the closure
# in harbor-requirements.lock under `--require-hashes`, which refuses any file
# whose sha256 differs. `uv tool install` and `pipx install` both ignore hashes
# (measured: each installed harbor with a deliberately wrong one), which is why
# neither installs harbor here any more.
#
# To raise it: bump HARBOR_VERSION, regenerate the lock with
#   uv pip compile <(echo harbor==X.Y.Z) --universal --python-version 3.12 \
#     --generate-hashes --no-header -o scripts/lib/harbor-requirements.lock
# run `python3 scripts/patch_harbor.py --audit` under the new harbor, and
# re-anchor whatever it reports.
HARBOR_VERSION="0.24.0"
HARBOR_LOCK="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/harbor-requirements.lock"
HARBOR_VENV="${HARBOR_VENV:-$HOME/.local/share/harbor-pinned/$HARBOR_VERSION}"
HARBOR_SHIM_DIR="${HARBOR_SHIM_DIR:-$HOME/.local/bin}"

# Which tool owns the harbor on PATH. A uv-tool or pipx harbor was installed
# without hash checks; naming it lets the refusal say why it is being replaced.
harbor_install_tool() {
  local bin real
  if bin="$(command -v harbor 2>/dev/null)"; then
    # BSD readlink has no -f, so fall back to python for the realpath.
    real="$(readlink -f "$bin" 2>/dev/null)" \
      || real="$(python3 -c 'import os,sys;print(os.path.realpath(sys.argv[1]))' "$bin" 2>/dev/null)" \
      || real=""
    case "$real" in
      */harbor-pinned/*) echo pinned; return 0 ;;
      */uv/tools/*)      echo uv;     return 0 ;;
      */pipx/*)          echo pipx;   return 0 ;;
    esac
  fi
  command -v uv >/dev/null 2>&1 && { echo none; return 0; }
  echo no-uv
}

# The exact command that fixes this machine, for error messages.
harbor_install_cmd() {
  echo "bash scripts/lib/harbor.sh install"
}

harbor_install() {
  command -v uv >/dev/null 2>&1 || {
    echo "[harbor] uv is required to install the pinned harbor: https://docs.astral.sh/uv/" >&2
    return 1
  }
  [ -f "$HARBOR_LOCK" ] || { echo "[harbor] lock missing: $HARBOR_LOCK" >&2; return 1; }
  case "$(harbor_install_tool)" in
    uv|pipx) echo "[harbor] replacing the $(harbor_install_tool)-installed harbor on PATH: it was installed without hash checks" >&2 ;;
  esac
  echo "[harbor] installing harbor==$HARBOR_VERSION into $HARBOR_VENV (--require-hashes)"
  uv venv --quiet --allow-existing --python 3.12 "$HARBOR_VENV" \
    && uv pip install --quiet --python "$HARBOR_VENV/bin/python" --require-hashes -r "$HARBOR_LOCK" \
    || { echo "[harbor] install failed; nothing on PATH was changed" >&2; return 1; }
  mkdir -p "$HARBOR_SHIM_DIR"
  ln -sf "$HARBOR_VENV/bin/harbor" "$HARBOR_SHIM_DIR/harbor"
  echo "[harbor] $("$HARBOR_VENV/bin/harbor" --version) -> $HARBOR_SHIM_DIR/harbor"
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  case "${1:-}" in
    install) harbor_install ;;
    version) echo "$HARBOR_VERSION" ;;
    *) echo "usage: bash scripts/lib/harbor.sh install|version" >&2; exit 2 ;;
  esac
fi
