# Hy3-295B · NVFP4 · 2× DGX Spark

Serve [Tencent Hunyuan 3 (295B MoE)](https://huggingface.co/kodelow/Hy3-NVFP4-W4A16) at **tensor-parallel size 2** across two NVIDIA DGX Spark nodes (GB10), with MTP speculative decoding and Hy3 tool-call support.

Everything runs through two scripts:

| Script | Purpose |
|---|---|
| `./start.sh` | Download model, sync to worker, start Ray cluster, launch vLLM |
| `./stop.sh` | Tear down head and worker containers |

<p>
<a href="https://x.com/MiaAI_lab" target="_blank">
  <img src="https://img.shields.io/badge/Follow%20me%20on%20X-000000?style=for-the-badge&logo=x&logoColor=white" alt="Follow Mia on X" />
</a>
</p>
<p>
<a href='https://ko-fi.com/Z8Z3SPLOD' target='_blank'><img height='36' style='border:0px;height:36px;' src='https://storage.ko-fi.com/cdn/kofi6.png?v=6' border='0' alt='Buy Me a Coffee at ko-fi.com' /></a>
</p>

---

## Quick start

### 1. Download the Docker image

Run on **both** nodes (head and worker). The image is ~19 GB and public — no login required.

```bash
docker pull ghcr.io/miaai-lab/hy3-dual-dgx-spark:vllm-probe-modded
```

Verify the image is present:

```bash
docker images ghcr.io/miaai-lab/hy3-dual-dgx-spark:vllm-probe-modded
```

### 2. Prerequisites

On **both** nodes:

- Docker with GPU support
- Image pulled (step 1)
- ~181 GB free disk for model weights

On the **head** node (where you run `start.sh`):

- Passwordless SSH to the worker as `REMOTE_USER` (default: your login)
- Optional cluster key at `/etc/kamiwaza/ssl/cluster.key` (used automatically if present)
- `python3` with `huggingface_hub` installed

### 3. Configure network

Edit the block at the top of `start.sh`:

```bash
HEAD_IP="10.0.0.1"           # this node's fabric/cluster IP
WORKER_IP="10.0.0.2"         # remote worker fabric/cluster IP
CLUSTER_IFACE="enp1s0f1np1"  # NIC carrying HEAD_IP/WORKER_IP
NCCL_IB_HCA="roceP2p1s0f0"   # RoCE HCA for NCCL GPU traffic
```

**How to find these on a DGX Spark:**

```bash
# Cluster IPs — the /24 used between your two nodes
ip -4 -o addr show

# Which NIC carries HEAD_IP
ip -o -4 addr show to 10.0.0.1/32

# RoCE device name
ibdev2netdev
```

`CLUSTER_IFACE` must be the interface that has your `10.0.0.x` addresses. This is used for Gloo and NCCL socket bootstrap. Do **not** point it at the RoCE NIC (`enP2p1s0f0np0`) — that interface typically has no IP and will cause distributed init to fail.

`NCCL_IB_HCA` is the RoCE device for GPU-to-GPU traffic over the 200G fabric.

Set the same `WORKER_IP` at the top of `stop.sh`.

### 4. Start

```bash
./start.sh
```

The script will:

1. Download or locate the model from HuggingFace (`kodelow/Hy3-NVFP4-W4A16`)
2. rsync weights to the worker over SSH
3. Remove any stale `hy3-head` / `hy3-worker` containers
4. Start a Ray head on this node and a Ray worker on the remote node
5. Wait until Ray reports 2 nodes
6. Launch vLLM inside the head container (tool-parser patches + MTP spec decode)
7. Follow vLLM logs in your terminal

Model load takes roughly 10 minutes. When ready, the API is at:

```
http://<HEAD_IP>:8888/v1
```

### 5. Stop

```bash
./stop.sh
```

Removes `hy3-head` locally and `hy3-worker` on the worker via SSH. Edit `WORKER_IP` at the top of `stop.sh` if you changed it in `start.sh`.

---

## What `start.sh` runs

### Ray cluster

Two Docker containers, both using host networking and the same NCCL/Gloo settings:

- `hy3-head` — Ray head on `HEAD_IP`
- `hy3-worker` — Ray worker on `WORKER_IP`, started over SSH

### vLLM serve

Launched detached inside `hy3-head` with:

```
vllm serve <model-snapshot> \
  --served-model-name hy3 --port 8888 \
  --tensor-parallel-size 2 --distributed-executor-backend ray \
  --max-model-len 131072 --max-num-seqs 6 \
  --kv-cache-dtype fp8_e4m3 --gpu-memory-utilization 0.90 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}' \
  --tool-call-parser hy_v3 --reasoning-parser hy_v3 --enable-auto-tool-choice \
  --trust-remote-code --enforce-eager
```

On startup it also patches vLLM's `hy_v3` parsers in-place so `:opensource`-suffixed tokenizer tokens match the checkpoint.

---

## Logs

| What | Command |
|---|---|
| **vLLM** (use this) | `docker exec hy3-head tail -f /tmp/hy3-serve.log` |
| Ray startup only | `docker logs -f hy3-head` |

Ray is PID 1 in the container, so vLLM output does not appear in `docker logs`. `start.sh` follows `/tmp/hy3-serve.log` by default.

Detach from log follow without stopping the server: **Ctrl+C**.

Start without following logs:

```bash
./start.sh --no-follow
# or
FOLLOW_LOGS=0 ./start.sh
```

---

## Optional configuration

Edit the network block at the top of `start.sh`, or override at runtime via environment variables:

### Environment variables

| Variable | Default | Used by |
|---|---|---|
| `IMAGE` | `ghcr.io/miaai-lab/hy3-dual-dgx-spark:vllm-probe-modded` | `start.sh` |
| `REMOTE_USER` | `$(id -un)` | `start.sh`, `stop.sh` |
| `SSH_KEY` | `/etc/kamiwaza/ssl/cluster.key` | `start.sh`, `stop.sh` |
| `FOLLOW_LOGS` | `1` | `start.sh` |
| `MODEL_REPO` | `kodelow/Hy3-NVFP4-W4A16` | `start.sh` |

`WORKER_IP` is set at the top of both `start.sh` and `stop.sh` — keep them in sync when you edit your network config.

---

## Troubleshooting

**`Unable to find address for: enP2p1s0f0np0`**

`CLUSTER_IFACE` is set to the RoCE NIC instead of the NIC with your cluster IPs. Use the interface that carries `10.0.0.1` / `10.0.0.2` (typically `enp1s0f1np1`).

**`Ray never reported 2 nodes`**

- Check SSH: `ssh <user>@<WORKER_IP> docker info`
- Check worker container: `ssh <user>@<WORKER_IP> docker ps -a --filter name=hy3-worker`
- Verify `HEAD_IP` / `WORKER_IP` match your fabric addresses

**Serve hangs or fails after a previous run**

Kill everything and restart cleanly. A crashed serve can leave GPU memory held on the worker:

```bash
./stop.sh && ./start.sh
```

**`Missing Docker image`**

Pull the image on both nodes (see step 1), then rerun `./start.sh`.

**Health check**

```bash
curl http://<HEAD_IP>:8888/health
```

Returns `200` once model load completes.

---

## File layout

```
start.sh      # all-in-one launcher (edit network block at top)
stop.sh       # tear down both containers (WORKER_IP must match start.sh)
README.md
.gitignore
```

---

## Model

- **Weights:** [kodelow/Hy3-NVFP4-W4A16](https://huggingface.co/kodelow/Hy3-NVFP4-W4A16) — 181 GB, MARLIN W4A16, MTP layer preserved
- **Stack:** vLLM 0.23.x, Ray TP=2, FP8 KV cache, MTP speculative decoding (`num_speculative_tokens: 1`)
- **Hardware:** 2× NVIDIA DGX Spark (GB10), 200GbE RoCE fabric

---

## Credits

This recipe was built upon the work of [tonyd2wild](https://github.com/tonyd2wild), whose [Hy3-295B NVFP4 MTP 2× DGX Spark](https://github.com/tonyd2wild/Hy3-295B-NVFP4-MTP-2x-DGX-Spark) bring-up established the core serving stack, benchmarks, and the bugs-to-avoid playbook that this repo streamlines into `start.sh` / `stop.sh`.
