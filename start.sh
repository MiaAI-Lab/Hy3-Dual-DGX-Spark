#!/usr/bin/env bash
# Hy3-295B NVFP4 · 2× DGX Spark TP=2 — model sync, Ray cluster, vLLM serve (all-in-one).
set -euo pipefail

# =============================================================================
# Cluster network — edit these for your setup
# =============================================================================
HEAD_IP="10.0.0.1"           # this node's fabric/cluster IP
WORKER_IP="10.0.0.2"         # remote worker fabric/cluster IP
CLUSTER_IFACE="enp1s0f1np1"  # NIC carrying HEAD_IP/WORKER_IP (Gloo + NCCL socket bootstrap)
NCCL_IB_HCA="roceP2p1s0f0"   # RoCE HCA for NCCL GPU traffic (not the socket-bootstrap NIC)

MODEL_REPO="${MODEL_REPO:-kodelow/Hy3-NVFP4-W4A16}"
MODEL_HOST_DIR="${MODEL_HOST_DIR:-}"
MODEL_REMOTE_DIR="${MODEL_REMOTE_DIR:-}"
MODEL_PATH="${MODEL_PATH:-}"
SERVED_NAME="${SERVED_NAME:-hy3}"
PORT="${PORT:-8888}"
IMAGE="${IMAGE:-ghcr.io/miaai-lab/hy3-dual-dgx-spark:vllm-probe-modded}"
RAY_PORT="${RAY_PORT:-26480}"
REMOTE_USER="${REMOTE_USER:-$(id -un)}"
SSH_KEY="${SSH_KEY:-/etc/kamiwaza/ssl/cluster.key}"

docker_common=(
  --network host --ipc host --privileged --security-opt label=disable --gpus all
  --ulimit memlock=-1 --ulimit stack=67108864
  -e RAY_memory_usage_threshold=0.99 -e RAY_memory_monitor_refresh_ms=0
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e CUDA_DEVICE_MAX_CONNECTIONS=32
  -e NCCL_SOCKET_IFNAME="${CLUSTER_IFACE}" -e GLOO_SOCKET_IFNAME="${CLUSTER_IFACE}"
  -e NCCL_IB_DISABLE=0 -e NCCL_IB_HCA="${NCCL_IB_HCA}"
  -e NCCL_MAX_NCHANNELS=4 -e NCCL_MIN_NCHANNELS=4
  -e VLLM_USE_B12X_FP8_GEMM=0 -e VLLM_USE_B12X_MOE=0 -e VLLM_USE_B12X_SPARSE_INDEXER=0
)

SSH_OPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o GlobalKnownHostsFile=/dev/null -o LogLevel=ERROR -o BatchMode=yes -o ConnectTimeout=10)
if [[ -f "$SSH_KEY" ]]; then
  SSH_OPTS+=(-i "$SSH_KEY" -o IdentitiesOnly=yes -o IdentityAgent=none)
fi
REMOTE_TARGET="${REMOTE_USER}@${WORKER_IP}"

echo "== ensure Docker image (${IMAGE}) =="
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "Pulling on head..."
  docker pull "$IMAGE"
fi
if ! ssh "${SSH_OPTS[@]}" "$REMOTE_TARGET" "docker image inspect '$IMAGE' >/dev/null 2>&1"; then
  echo "Copying image to worker (${WORKER_IP}) via docker save | ssh docker load..."
  docker save "$IMAGE" | ssh "${SSH_OPTS[@]}" "$REMOTE_TARGET" "docker load"
fi

echo "== download / locate model =="
MODEL_CACHE_DIR="$(
python3 - "$MODEL_REPO" <<'PY'
import sys
from huggingface_hub import snapshot_download

repo_id = sys.argv[1]
print(snapshot_download(repo_id=repo_id))
PY
)"

MODEL_HOST_DIR="$(dirname "$(dirname "$MODEL_CACHE_DIR")")"
MODEL_REMOTE_DIR="${MODEL_REMOTE_DIR:-$HOME/.cache/huggingface/hub/$(basename "$MODEL_HOST_DIR")}"
MODEL_REVISION="$(basename "$MODEL_CACHE_DIR")"
MODEL_PATH="/models/snapshots/${MODEL_REVISION}"

mkdir -p "$MODEL_HOST_DIR"

RSYNC_RSH="ssh ${SSH_OPTS[*]}"

echo "== rsync model to worker (${WORKER_IP}) =="
ssh "${SSH_OPTS[@]}" "$REMOTE_TARGET" "mkdir -p ${MODEL_REMOTE_DIR}"
rsync -a --delete --info=progress2 -e "$RSYNC_RSH" "$MODEL_HOST_DIR/" "${REMOTE_TARGET}:${MODEL_REMOTE_DIR}/"

echo "== teardown any prior hy3 containers =="
docker rm -f hy3-head >/dev/null 2>&1 || true
ssh "${SSH_OPTS[@]}" "$REMOTE_TARGET" "docker rm -f hy3-worker >/dev/null 2>&1 || true"

echo "== start Ray head (${HEAD_IP}) =="
docker run -d --name hy3-head "${docker_common[@]}" \
  -v "${MODEL_HOST_DIR}:/models:ro" \
  -e VLLM_HOST_IP="${HEAD_IP}" "${IMAGE}" \
  ray start --head --node-ip-address="${HEAD_IP}" --port="${RAY_PORT}" \
    --num-gpus=1 --object-store-memory=134217728 --disable-usage-stats --block

echo "== start Ray worker (${WORKER_IP}) =="
# shellcheck disable=SC2029
ssh "${SSH_OPTS[@]}" "$REMOTE_TARGET" "docker run -d --name hy3-worker \
  --network host --ipc host --privileged --security-opt label=disable --gpus all \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v ${MODEL_REMOTE_DIR}:/models:ro \
  -e RAY_memory_usage_threshold=0.99 -e RAY_memory_monitor_refresh_ms=0 \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e CUDA_DEVICE_MAX_CONNECTIONS=32 \
  -e NCCL_SOCKET_IFNAME=${CLUSTER_IFACE} -e GLOO_SOCKET_IFNAME=${CLUSTER_IFACE} \
  -e NCCL_IB_DISABLE=0 -e NCCL_IB_HCA=${NCCL_IB_HCA} \
  -e NCCL_MAX_NCHANNELS=4 -e NCCL_MIN_NCHANNELS=4 \
  -e VLLM_USE_B12X_FP8_GEMM=0 -e VLLM_USE_B12X_MOE=0 -e VLLM_USE_B12X_SPARSE_INDEXER=0 \
  -e VLLM_HOST_IP=${WORKER_IP} ${IMAGE} \
  ray start --address=${HEAD_IP}:${RAY_PORT} --node-ip-address=${WORKER_IP} \
    --num-gpus=1 --object-store-memory=134217728 --disable-usage-stats --block"

echo "== wait for 2 Ray nodes =="
for attempt in $(seq 1 30); do
  nodes="$(docker exec hy3-head ray status 2>/dev/null | grep -c 'node_' || true)"
  echo "[${attempt}/30] ray nodes: ${nodes:-0}"
  if [[ "${nodes:-0}" -ge 2 ]]; then
    break
  fi
  sleep 5
done

if [[ "${nodes:-0}" -lt 2 ]]; then
  echo "Ray never reported 2 nodes; aborting." >&2
  exit 1
fi

echo "== launch vLLM serve (detached inside head) =="
SERVE_SCRIPT="$(mktemp)"
trap 'rm -f "$SERVE_SCRIPT"' EXIT
cat >"$SERVE_SCRIPT" <<'SERVE_EOF'
#!/usr/bin/env bash
exec > /tmp/hy3-serve.log 2>&1
export PYTHONUNBUFFERED=1
# bug #3 fix: checkpoint emits :opensource-suffixed special tokens; vLLM parsers hardcode bare forms.
RP=/usr/local/lib/python3.12/dist-packages/vllm/reasoning/hy_v3_reasoning_parser.py
TP=/usr/local/lib/python3.12/dist-packages/vllm/tool_parsers/hy_v3_tool_parser.py
sed -i 's|"<think>"|"<think:opensource>"|g; s|"</think>"|"</think:opensource>"|g' "$RP"
sed -i 's|"<tool_calls>"|"<tool_calls:opensource>"|g; s|"</tool_calls>"|"</tool_calls:opensource>"|g; s|"<tool_call>"|"<tool_call:opensource>"|g; s|"</tool_call>"|"</tool_call:opensource>"|g; s|"<tool_sep>"|"<tool_sep:opensource>"|g; s|"<arg_key>"|"<arg_key:opensource>"|g; s|"</arg_key>"|"</arg_key:opensource>"|g; s|"<arg_value>"|"<arg_value:opensource>"|g; s|"</arg_value>"|"</arg_value:opensource>"|g' "$TP"
echo "PATCH-CHECK reasoning:"; grep -c "opensource" "$RP"
echo "PATCH-CHECK tool:"; grep -c "opensource" "$TP"
exec vllm serve "${MODEL_PATH}" \
  --served-model-name "${SERVED_NAME}" --port "${PORT}" \
  --tensor-parallel-size 2 --distributed-executor-backend ray \
  --max-model-len 131072 --max-num-seqs 6 \
  --kv-cache-dtype fp8_e4m3 --gpu-memory-utilization 0.90 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}' \
  --tool-call-parser hy_v3 --reasoning-parser hy_v3 --enable-auto-tool-choice \
  --trust-remote-code --enforce-eager
SERVE_EOF

docker cp "$SERVE_SCRIPT" hy3-head:/serve-tools-hy3.sh
docker exec -d \
  -e VLLM_HOST_IP="${HEAD_IP}" \
  -e MODEL_PATH="${MODEL_PATH}" \
  -e SERVED_NAME="${SERVED_NAME}" \
  -e PORT="${PORT}" \
  -e NCCL_SOCKET_IFNAME="${CLUSTER_IFACE}" \
  -e GLOO_SOCKET_IFNAME="${CLUSTER_IFACE}" \
  -e NCCL_IB_DISABLE=0 \
  -e NCCL_IB_HCA="${NCCL_IB_HCA}" \
  -e NCCL_MAX_NCHANNELS=4 \
  -e NCCL_MIN_NCHANNELS=4 \
  hy3-head bash /serve-tools-hy3.sh

echo "Started vLLM on ${HEAD_IP}:${PORT}"

if [[ "${FOLLOW_LOGS:-1}" == "1" && "${1:-}" != "--no-follow" ]]; then
  echo "Following vLLM logs (Ctrl+C detaches; container keeps running)..."
  echo "Stop with: ./stop.sh"
  for _ in $(seq 1 30); do
    if docker exec hy3-head test -s /tmp/hy3-serve.log 2>/dev/null; then
      break
    fi
    sleep 1
  done
  exec docker exec hy3-head tail -f /tmp/hy3-serve.log
fi

echo "Tail vLLM logs: docker exec hy3-head tail -f /tmp/hy3-serve.log"
echo "Ray logs:        docker logs -f hy3-head"