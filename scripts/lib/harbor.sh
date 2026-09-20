#!/usr/bin/env bash
# Single source of truth for which harbor this harness runs on, and for how to
# install it on THIS machine. Sourced by setup.sh and scripts/run_task.sh so the
# pin and the "how to fix it" message can never drift apart again.
#
# PINNED on purpose. scripts/patch_harbor.py edits harbor's own source by
# matching exact anchor strings, so a harbor that moves those lines silently
# drops patches: the 0.23.0 flag rewrite (CliFlag lists -> pydantic fields)
# killed two of them at once, one of which had been failing quietly for
# releases. Unpinned, a fresh machine installs whatever is newest that day and
# inherits that breakage with no warning.
#
# To raise it: bump the number, run `python3 scripts/patch_harbor.py --audit`,
# re-anchor whatever it reports, then run one task end to end before pushing.
HARBOR_VERSION="0.23.0"

# Which tool owns the harbor on PATH. The fleet is mixed: HARNESS.md and
# run_task.sh:864 say uv, setup.sh historically used pipx, and both drop their
# shim in ~/.local/bin, so the shim NAME cannot tell them apart -- only where it
# resolves to can. Installing with the other tool does not upgrade anything, it
# just shadows one copy with a second one, which is how a box ends up running a
# harbor nobody thinks is installed.
harbor_install_tool() {
  local bin real
  if bin="$(command -v harbor 2>/dev/null)"; then
    # BSD readlink has no -f, so fall back to python for the realpath.
    real="$(readlink -f "$bin" 2>/dev/null)" \
      || real="$(python3 -c 'import os,sys;print(os.path.realpath(sys.argv[1]))' "$bin" 2>/dev/null)" \
      || real=""
    case "$real" in
      */uv/tools/*) echo uv;   return 0 ;;
      */pipx/*)     echo pipx; return 0 ;;
    esac
  fi
  # Nothing installed yet. uv first: HARNESS.md lists it as a prerequisite.
  command -v uv   >/dev/null 2>&1 && { echo uv;   return 0; }
  command -v pipx >/dev/null 2>&1 && { echo pipx; return 0; }
  echo none
}

# The exact command that fixes this machine, for error messages. Printing a
# pipx command to a uv box is what sent one operator into installing a second
# harbor instead of upgrading the one in use.
harbor_install_cmd() {
  case "$(harbor_install_tool)" in
    uv)   echo "uv tool install --force 'harbor==${HARBOR_VERSION}'" ;;
    pipx) echo "pipx install --force 'harbor==${HARBOR_VERSION}'" ;;
    *)    echo "uv tool install --force 'harbor==${HARBOR_VERSION}'   # install uv first" ;;
  esac
}

harbor_install() {
  local spec="harbor==${HARBOR_VERSION}"
  case "$(harbor_install_tool)" in
    uv)   echo "[harbor] uv tool install --force $spec";  uv tool install --force "$spec" ;;
    pipx) echo "[harbor] pipx install --force $spec";     pipx install --force "$spec" ;;
    *)    echo "[harbor] neither uv nor pipx found; install one, then run: $(harbor_install_cmd)" >&2
          return 1 ;;
  esac
}
