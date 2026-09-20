#!/bin/sh
# Squid runs as the unprivileged 'proxy' user and so cannot write to the
# container's stdout directly. It logs to a file instead; this tail is what puts
# denials into `docker logs egress-proxy`, which is where an operator debugging a
# blocked run will look first.
set -e

mkdir -p /var/log/squid /var/run/squid
: > /var/log/squid/access.log
chown -R proxy:proxy /var/log/squid /var/run/squid

# Two sinks, one tail.
#
#   stdout      `docker logs egress-proxy`, for an operator debugging a blocked
#               run while the container is still up.
#   /egress-out the trial's own agent-log dir on the host (see overlay.yaml),
#               so the record outlives the container and reaches the audit.
#
# Squid drops privileges to the 'proxy' user before opening its logs and cannot
# write a root-owned bind mount, which is why it still logs to a file it owns
# and this tail does the copying. Adding a second `access_log` directive
# pointing at /egress-out would mean chowning the trial's log dir to 'proxy',
# which harbor also writes claude-code.txt into -- not worth the collision.
#
# The mount is absent when the proxy is driven outside harbor (the runtime test,
# a hand `compose up`), so tee only when it is there.
if [ -d /egress-out ]; then
  tail -F /var/log/squid/access.log 2>/dev/null | tee -a /egress-out/egress-access.log &
else
  tail -F /var/log/squid/access.log 2>/dev/null &
fi

exec /usr/local/bin/entrypoint.sh "$@"
