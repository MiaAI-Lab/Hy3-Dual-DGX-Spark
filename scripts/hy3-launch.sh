#!/usr/bin/env bash
# Hy3-295B NVFP4-W4A16 · 2x Spark TP=2 (Bluey head + Reddie worker) — Kai 2026-07-07
# Adapted from m9e-matt-glm52 launch-ray.sh (Ray-in-docker over the 200G fabric).
set -euo pipefail

IMAGE="vllm-node-tf5-glm52-b12x:probe-modded"        # vLLM 0.23.1rc1.dev190
MODEL_HOST_DIR="$HOME/models/hy3-nvfp4-w4a16"
HEAD_IP="192.168.192.1"
WORKER_IP="192.168.192.2"
RAY_PORT=26480
SSH_KEY="/etc/kamiwaza/ssl/cluster.key"
HS_IFACE="enP2p1s0f0np0"
PORT=8600

docker_common=(
  --network host --ipc host --privileged --security-opt label=disable --gpus all
  --ulimit memlock=-1 --ulimit stack=67108864
  -v "${MODEL_HOST_DIR}:/models:ro"
  -e RAY_memory_usage_threshold=0.99 -e RAY_memory_monitor_refresh_ms=0
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e CUDA_DEVICE_MAX_CONNECTIONS=32
  -e NCCL_SOCKET_IFNAME="${HS_IFACE}" -e GLOO_SOCKET_IFNAME="${HS_IFACE}"
  -e NCCL_IB_DISABLE=0 -e NCCL_IB_HCA="roceP2p1s0f0"
  -e NCCL_MAX_NCHANNELS=4 -e NCCL_MIN_NCHANNELS=4
  # b12x custom paths OFF — Hy3 is not a b12x-tuned model; want vanilla vLLM+marlin
  -e VLLM_USE_B12X_FP8_GEMM=0 -e VLLM_USE_B12X_MOE=0 -e VLLM_USE_B12X_SPARSE_INDEXER=0
)

sshw=(ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i "$SSH_KEY"
      -o IdentitiesOnly=yes -o IdentityAgent=none -o BatchMode=yes -o ConnectTimeout=10)

echo "== teardown any prior hy3 containers =="
docker rm -f hy3-head 2>/dev/null || true
"${sshw[@]}" "tonyspark2@${WORKER_IP}" "docker rm -f hy3-worker 2>/dev/null || true"

echo "== start Ray head (Bluey) =="
docker run -d --name hy3-head "${docker_common[@]}" \
  -e VLLM_HOST_IP="${HEAD_IP}" "${IMAGE}" \
  ray start --head --node-ip-address="${HEAD_IP}" --port="${RAY_PORT}" \
    --num-gpus=1 --object-store-memory=134217728 --disable-usage-stats --block

echo "== start Ray worker (Reddie) =="
"${sshw[@]}" "tonyspark2@${WORKER_IP}" "docker run -d --name hy3-worker \
  --network host --ipc host --privileged --security-opt label=disable --gpus all \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v \$HOME/models/hy3-nvfp4-w4a16:/models:ro \
  -e RAY_memory_usage_threshold=0.99 -e RAY_memory_monitor_refresh_ms=0 \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e CUDA_DEVICE_MAX_CONNECTIONS=32 \
  -e NCCL_SOCKET_IFNAME=${HS_IFACE} -e GLOO_SOCKET_IFNAME=${HS_IFACE} \
  -e NCCL_IB_DISABLE=0 -e NCCL_IB_HCA=roceP2p1s0f0 \
  -e NCCL_MAX_NCHANNELS=4 -e NCCL_MIN_NCHANNELS=4 \
  -e VLLM_USE_B12X_FP8_GEMM=0 -e VLLM_USE_B12X_MOE=0 -e VLLM_USE_B12X_SPARSE_INDEXER=0 \
  -e VLLM_HOST_IP=${WORKER_IP} ${IMAGE} \
  ray start --address=${HEAD_IP}:${RAY_PORT} --node-ip-address=${WORKER_IP} \
    --num-gpus=1 --object-store-memory=134217728 --disable-usage-stats --block"

echo "== wait for 2 ray nodes =="
for i in $(seq 1 30); do
  N=$(docker exec hy3-head ray status 2>/dev/null | grep -c "node_" || true)
  echo "  ray nodes: $N"
  [ "$N" -ge 2 ] && break
  sleep 5
done

echo "== launch vllm serve (detached inside head) =="
docker exec -d hy3-head bash -c "
  export VLLM_HOST_IP=${HEAD_IP}
  vllm serve /models \
    --served-model-name hy3 --port ${PORT} \
    --tensor-parallel-size 2 \
    --max-model-len 131072 \
    --max-num-seqs 6 \
    --kv-cache-dtype fp8_e4m3 \
    --gpu-memory-utilization 0.92 \
    --speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":1}' \
    --tool-call-parser hy_v3 --reasoning-parser hy_v3 --enable-auto-tool-choice \
    --trust-remote-code --enforce-eager \
    > /tmp/hy3-serve.log 2>&1"
echo "LAUNCHED — tail with: docker exec hy3-head tail -f /tmp/hy3-serve.log"
