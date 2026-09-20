#!/usr/bin/env bash
# Tear down the harness background services and reap Docker leftovers.
#
# Images and build cache are deliberately KEPT: they are the warm cache that
# makes a second run fast. This reaps harbor's containers and the volumes they
# leave behind -- named and anonymous both -- and nothing else.
#
#   bash scripts/stop_harness.sh [--dry-run] [--all] [--force]
#
#   --dry-run  print what would be killed/removed, change nothing
#   --all      blanket `docker container/volume prune` instead of the harness-
#              scoped sweep. Removes stopped containers from UNRELATED Docker
#              projects on this machine too.
#   --force    proceed even when a run looks live. Kills the model route out
#              from under it; the trial dies mid-flight.
set -uo pipefail

DRY=0; ALL=0; FORCE=0
for a in "$@"; do
  case "$a" in
    --dry-run|-n) DRY=1 ;;
    --all)        ALL=1 ;;
    --force)      FORCE=1 ;;
    -h|--help)    sed -n '2,14p' "$0"; exit 0 ;;
    *) echo "unknown flag: $a (want --dry-run|--all|--force)" >&2; exit 2 ;;
  esac
done

say()  { echo "[stop-harness] $*"; }
run()  { if [ "$DRY" = 1 ]; then echo "[dry-run]      $*"; else "$@"; fi; }

# Ports the harness binds. zbridge is shared by every concurrent run_task.sh on
# this machine, which is why the live-run guard below exists at all.
PORTS=(8766 4000 4001 8787 8788)

cmd_of()  { ps -o command= -p "$1" 2>/dev/null; }
ppid_of() { ps -o ppid= -p "$1" 2>/dev/null | tr -d ' '; }

# --- who is a service, who is only holding one ------------------------------
# `(nohup uv run … &)` leaves a 3-deep chain: an orphaned bash that still shows
# in pgrep as `bash scripts/run_task.sh <task>`, the `uv run` supervisor, and
# the real server. Only the last does any work; the other two are dead weight
# that makes it look like a run is still going.
SERVICE_PIDS=()
for pat in 'python -m zbridge' 'cbridge\.py' 'zbridge_adapter\.py' 'headroom.*proxy'; do
  while read -r p; do [ -n "$p" ] && SERVICE_PIDS+=("$p"); done < <(pgrep -f "$pat" 2>/dev/null)
done

# Ancestors worth reaping alongside a service: the `uv run` supervisor, and an
# orphaned run_task.sh (PPID 1 — its real parent already exited). A run_task.sh
# with a living parent is a REAL run and is never touched here.
WRAPPER_PIDS=()
for svc in "${SERVICE_PIDS[@]:-}"; do
  [ -n "${svc:-}" ] || continue
  p="$(ppid_of "$svc")"
  while [ -n "${p:-}" ] && [ "$p" -gt 1 ] 2>/dev/null; do
    c="$(cmd_of "$p")"
    case "$c" in
      *"uv run"*)        WRAPPER_PIDS+=("$p") ;;
      *run_task.sh*)     [ "$(ppid_of "$p")" = "1" ] && WRAPPER_PIDS+=("$p") ;;
    esac
    p="$(ppid_of "$p")"
  done
done

# --- live-run guard ----------------------------------------------------------
# A run_task.sh that is NOT one of the orphaned wrappers above, or a live
# `harbor run`, means a trial is in flight and shares these bridges.
LIVE=()
while read -r p; do
  [ -n "$p" ] || continue
  skip=0
  for w in "${WRAPPER_PIDS[@]:-}"; do [ "$p" = "$w" ] && skip=1; done
  [ "$skip" = 0 ] && LIVE+=("$p")
done < <(pgrep -f 'scripts/run_task.sh' 2>/dev/null)
while read -r p; do [ -n "$p" ] && LIVE+=("$p"); done < <(pgrep -f 'harbor run' 2>/dev/null)

if [ "${#LIVE[@]}" -gt 0 ] && [ "$FORCE" != 1 ]; then
  say "REFUSING: a run looks live. These bridges are shared machine-wide."
  for p in "${LIVE[@]}"; do echo "    $p  $(cmd_of "$p")"; done
  say "wait for it, or re-run with --force to kill it mid-trial."
  exit 1
fi
[ "${#LIVE[@]}" -gt 0 ] && say "--force: proceeding over ${#LIVE[@]} live run(s)"

# --- kill ---------------------------------------------------------------------
KILL=("${SERVICE_PIDS[@]:-}" "${WRAPPER_PIDS[@]:-}")
KILL=($(printf '%s\n' "${KILL[@]:-}" | grep -E '^[0-9]+$' | sort -un))

if [ "${#KILL[@]}" -eq 0 ]; then
  say "no harness services running"
else
  for p in "${KILL[@]}"; do say "kill $p  $(cmd_of "$p")"; done
  run kill "${KILL[@]}" 2>/dev/null
  if [ "$DRY" != 1 ]; then
    for _ in 1 2 3 4 5 6 7 8 9 10; do
      still=0; for p in "${KILL[@]}"; do kill -0 "$p" 2>/dev/null && still=1; done
      [ "$still" = 0 ] && break
      sleep 0.5
    done
    for p in "${KILL[@]}"; do
      kill -0 "$p" 2>/dev/null && { say "SIGKILL $p (ignored SIGTERM)"; kill -9 "$p" 2>/dev/null; }
    done
  fi
fi

busy="$(lsof -nP $(printf -- '-iTCP:%s ' "${PORTS[@]}") -sTCP:LISTEN 2>/dev/null | tail -n +2)"
if [ -n "$busy" ]; then say "WARNING: ports still bound:"; echo "$busy"; else say "ports free: ${PORTS[*]}"; fi

# --- docker -------------------------------------------------------------------
if ! docker info >/dev/null 2>&1; then
  say "docker not reachable; skipping container/volume sweep"
  exit 0
fi

before="$(docker system df --format '{{.Type}} {{.Reclaimable}}' 2>/dev/null)"

if [ "$ALL" = 1 ]; then
  say "--all: pruning every stopped container and unused volume on this machine"
  run docker container prune -f
  run docker volume prune -f
else
  # Scoped to harbor's own compose projects so unrelated Docker projects on this
  # machine are left alone.
  #
  # This used to scope on the name infix "__env-", from a harbor that named its
  # compose project "<trial>__env". It does not any more -- the project is just
  # the lowercased trial name, e.g. "sakshi_lydbury-departmental-oper__3hthstd",
  # so containers are "<project>-<service>-1" and volumes "<project>_<name>".
  # Neither filter had matched anything for a long time, so this script was a
  # no-op on the very containers it exists to remove: a stuck light-servers held
  # most of Docker's memory until every later run's egress-proxy was OOM-killed
  # (exit 137) and never started `main`, and dangling volumes leaked alongside.
  #
  # Containers are selected by COMPOSE LABEL rather than by name: harbor may
  # rename its projects again, but a container it created always carries
  # com.docker.compose.project, and harbor's trial names always contain "__".
  # Volumes carry the same label and `docker volume ls` does expose it, both as
  # `--filter label=` and as `{{.Label "..."}}` in --format, so they are scoped
  # the same way rather than by the name infix.
  _harness_containers() {  # _harness_containers <status>... -> ids on stdout
    local _st _args=()
    for _st in "$@"; do _args+=(--filter "status=$_st"); done
    docker ps -a --filter "label=com.docker.compose.project" "${_args[@]}" \
        --format '{{.Label "com.docker.compose.project"}}	{{.ID}}' 2>/dev/null \
      | awk -F'\t' '$1 ~ /__/ {print $2}'
  }
  # Volume names mounted by the given containers, named and anonymous alike.
  _volumes_of() {
    [ "$#" -gt 0 ] || return 0
    docker inspect "$@" \
        --format '{{range .Mounts}}{{if eq .Type "volume"}}{{println .Name}}{{end}}{{end}}' \
        2>/dev/null | grep . | sort -u
  }

  # What to remove: every volume a swept container mounts, plus every volume
  # labelled with a harbor compose project, plus unused ANONYMOUS volumes --
  # minus anything a surviving container still holds.
  #
  # The compose label alone is not enough, and this is where the rest of the
  # leak lived. egress-proxy's base image (ubuntu/squid) declares
  # VOLUME /var/log/squid and VOLUME /var/spool/squid, so every proxy container
  # gets two volumes with 64-hex names and NO labels at all -- four per run,
  # counting the judge's proxy. `docker compose down --volumes` takes them when
  # a run ends normally. A run that is interrupted leaves them, and once their
  # container is gone no label or name can attribute them to harbor again.
  #
  # So they are matched on shape instead: a 64-hex name is Docker's own marker
  # for an anonymous volume (compose names its own "<project>_<volume>"), and an
  # anonymous volume attached to nothing cannot be addressed by anybody. That is
  # the same line `docker volume prune` draws by default -- it removes unused
  # anonymous volumes and needs --all before it touches a named one.
  #
  # Takes the ids the sweep is removing and treats them as already gone. Under
  # --dry-run they are NOT actually removed, so without this every volume would
  # still read as in use and a dry run would promise to remove nothing -- the
  # exact blind spot that hid this bug.
  _harness_volumes() {  # _harness_volumes <container-id-being-removed>...
    local c _x _s _keep=() _cand _live
    _cand="$( { _volumes_of "$@"
                docker volume ls --filter "label=com.docker.compose.project" \
                    --format '{{.Label "com.docker.compose.project"}}	{{.Name}}' 2>/dev/null \
                  | awk -F'\t' '$1 ~ /__/ {print $2}'
                docker volume ls -q 2>/dev/null | grep -E '^[0-9a-f]{64}$'
              } | grep . | sort -u )"
    [ -n "$_cand" ] || return 0
    while read -r c; do
      [ -n "$c" ] || continue
      _s=0
      for _x in "$@"; do [ "$c" = "$_x" ] && _s=1; done
      [ "$_s" = 0 ] && _keep+=("$c")
    done < <(docker ps -aq 2>/dev/null)
    _live="$(_volumes_of ${_keep[@]+"${_keep[@]}"})"
    printf '%s\n' "$_cand" | while IFS= read -r v; do
      printf '%s\n' "$_live" | grep -qxF -- "$v" || printf '%s\n' "$v"
    done
  }

  # RUNNING containers are swept when nothing owns them. The live-run guard
  # above already worked that out: LIVE holds every run_task.sh and `harbor run`
  # still on this machine. With LIVE empty no process is left to own a trial, so
  # every harness container is an orphan of a run that died -- a Ctrl-C, a
  # SIGKILL, a harbor that never reached its own teardown -- and leaving it up
  # is the whole bug this script exists to fix.
  #
  # This used to be stopped-only unless --force was given, on the grounds that a
  # bare `harbor run` owns no run_task.sh wrapper to match. It does own a
  # `harbor run` process, which the guard matches directly, so the LIVE check
  # covers that case too. The old behaviour meant a plain `make stop-harness`
  # removed whichever containers happened to be in `created` and left every
  # running one behind -- and, because those held the volume open, reported "no
  # unused harness volumes" and removed nothing at all.
  STATUSES=(exited created dead)
  RM=(docker rm)
  if [ "$FORCE" = 1 ] || [ "${#LIVE[@]}" -eq 0 ]; then
    STATUSES+=(running restarting removing)
    # A running container needs -f. Without it the batch below is guaranteed to
    # fail and every sweep would print an error before the per-container retry
    # rescued it. Only set here, where we have already decided these containers
    # belong to nothing.
    RM=(docker rm -f)
  fi
  CTRS=()
  while read -r c; do [ -n "$c" ] && CTRS+=("$c"); done < <(_harness_containers "${STATUSES[@]}")
  if [ "${#CTRS[@]}" -gt 0 ]; then
    for c in "${CTRS[@]}"; do say "rm container $c  $(docker inspect -f '{{.Name}} {{.State.Status}}' "$c" 2>/dev/null)"; done
    # Batch first, then a per-container -f retry for whatever survived. One
    # unremovable container used to fail the whole batch and take the rest of
    # the sweep's exit code with it, so the leak was reported as a success.
    if ! run "${RM[@]}" "${CTRS[@]}"; then
      for c in "${CTRS[@]}"; do
        docker inspect "$c" >/dev/null 2>&1 || continue   # already gone
        say "rm -f container $c (plain rm failed)"
        run docker rm -f "$c" || say "WARNING: could not remove $c -- remove it by hand"
      done
    fi
  else
    say "no stopped harness containers"
  fi

  # Volumes only AFTER containers, and selected by COMPOSE PROJECT rather than
  # by `dangling=true`.
  #
  # dangling means "attached to no container at all", which a harness volume is
  # only once every container of its project is gone. While the sweep above
  # could not touch running containers that never happened, so this filter
  # matched nothing on the very volumes it exists to remove. Ask instead for the
  # volumes harbor's compose projects own, and drop any still attached to a
  # container that survived the sweep -- docker would refuse those anyway, and
  # one refusal used to take the whole sweep's exit code with it.
  VOLS=()
  while read -r v; do [ -n "$v" ] && VOLS+=("$v"); done \
    < <(_harness_volumes ${CTRS[@]+"${CTRS[@]}"})
  if [ "${#VOLS[@]}" -gt 0 ]; then
    for v in "${VOLS[@]}"; do say "rm volume $v"; done
    run docker volume rm "${VOLS[@]}"
  else
    say "no unused harness volumes"
  fi
fi

say "images and build cache left intact (warm cache for the next run)"
[ "$DRY" != 1 ] && { echo; docker system df; }
exit 0
