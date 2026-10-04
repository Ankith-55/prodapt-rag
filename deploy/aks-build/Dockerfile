# State image for the cluster demo: the base application image plus a copy of the pre-built ticket state.
# Immutable and versioned together with the code, so every replica serves identical data and rollback is one command.
# Built by deploy/deploy_aks.ps1, which stages the files into ./state first and passes the real BASE image.
ARG BASE=ticketrag:local
FROM ${BASE}
COPY --chown=10001:10001 state/ /data/state/
