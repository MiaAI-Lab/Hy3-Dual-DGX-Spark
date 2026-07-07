#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Cluster network — must match start.sh
# =============================================================================
WORKER_IP="10.0.0.2"

HEAD_NAME="${HEAD_NAME:-hy3-head}"
WORKER_NAME="${WORKER_NAME:-hy3-worker}"
REMOTE_USER="${REMOTE_USER:-$(id -un)}"
SSH_KEY="${SSH_KEY:-}"

SSH_OPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o GlobalKnownHostsFile=/dev/null -o LogLevel=ERROR -o BatchMode=yes -o ConnectTimeout=10)
if [[ -n "${SSH_KEY}" && -f "${SSH_KEY}" ]]; then
  SSH_OPTS+=(-i "${SSH_KEY}" -o IdentitiesOnly=yes -o IdentityAgent=none)
fi

echo "Stopping ${HEAD_NAME} on the host..."
docker rm -f "$HEAD_NAME" >/dev/null 2>&1 || true

echo "Stopping ${WORKER_NAME} on ${WORKER_IP}..."
ssh "${SSH_OPTS[@]}" "${REMOTE_USER}@${WORKER_IP}" "docker rm -f ${WORKER_NAME} >/dev/null 2>&1 || true"

echo "Stopped."
