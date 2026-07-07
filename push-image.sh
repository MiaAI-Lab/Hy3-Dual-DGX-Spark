#!/usr/bin/env bash
# Push the vLLM image to GitHub Container Registry (needs GITHUB_TOKEN with write:packages).
set -euo pipefail

LOCAL_IMAGE="${LOCAL_IMAGE:-vllm-node-tf5-glm52-b12x:probe-modded}"
GHCR_IMAGE="${GHCR_IMAGE:-ghcr.io/miaai-lab/hy3-dual-dgx-spark:vllm-probe-modded}"
GHCR_USER="${GHCR_USER:-MiaAI-Lab}"

if [[ -z "${GITHUB_TOKEN:-}" ]]; then
  echo "Set GITHUB_TOKEN with write:packages scope." >&2
  exit 1
fi

if ! docker image inspect "$LOCAL_IMAGE" >/dev/null 2>&1; then
  echo "Missing local image: $LOCAL_IMAGE" >&2
  exit 1
fi

echo "Logging in to ghcr.io as ${GHCR_USER}..."
echo "${GITHUB_TOKEN}" | docker login ghcr.io -u "${GHCR_USER}" --password-stdin

echo "Tagging ${LOCAL_IMAGE} -> ${GHCR_IMAGE}"
docker tag "${LOCAL_IMAGE}" "${GHCR_IMAGE}"

echo "Pushing ${GHCR_IMAGE} (~19 GB, this takes a while)..."
docker push "${GHCR_IMAGE}"

echo "Done. Pull on each node with:"
echo "  docker pull ${GHCR_IMAGE}"
echo "  IMAGE=${GHCR_IMAGE} ./start.sh"