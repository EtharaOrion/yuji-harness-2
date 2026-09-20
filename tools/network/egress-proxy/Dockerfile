# The only containers in a task project with a route off the compose bridge:
# egress-proxy for the agent (squid.conf) and, in bundles with a judge service,
# judge-proxy for the judge (squid-judge.conf, see overlay-judge.yaml).
# See squid.conf for why this exists rather than a Harbor network_mode.
# Pinned by digest, not by :latest. This container IS the enforcement boundary:
# an unpinned tag means the thing deciding what the agent may reach can change
# between two runs of the same benchmark, silently. Every other image in this
# repo pins a version, and squid.conf already argues the reproducibility case
# for disabling the cache -- the same argument applies to the proxy itself.
#
# ubuntu/squid, Squid Cache 6.13. To move: docker pull ubuntu/squid:latest &&
# docker image inspect ubuntu/squid:latest --format '{{index .RepoDigests 0}}'
FROM ubuntu/squid@sha256:6a097f68bae708cedbabd6188d68c7e2e7a38cedd05a176e1cc0ba29e3bbe029

COPY squid.conf /etc/squid/squid.conf
COPY squid-judge.conf /etc/squid/squid-judge.conf
COPY entrypoint.sh /usr/local/bin/egress-entrypoint.sh

RUN chmod +x /usr/local/bin/egress-entrypoint.sh

# Fail the build, not the run, on a malformed ACL. A squid that exits at
# container start looks exactly like a network outage from inside main, and
# costs an agent phase to diagnose.
RUN squid -k parse -f /etc/squid/squid.conf
RUN squid -k parse -f /etc/squid/squid-judge.conf

EXPOSE 3128

ENTRYPOINT ["/usr/local/bin/egress-entrypoint.sh"]
CMD ["-f", "/etc/squid/squid.conf", "-NYC"]
