# Hy3-295B · NVFP4 · 2× DGX Spark

Serve [Tencent Hunyuan 3 (295B MoE)](https://huggingface.co/kodelow/Hy3-NVFP4-W4A16) at **tensor-parallel size 2** across two NVIDIA DGX Spark nodes (GB10), with MTP speculative decoding and Hy3 tool-call support.

<p>
<a href="https://x.com/MiaAI_lab" target="_blank">
  <img src="https://img.shields.io/badge/Follow%20me%20on%20X-000000?style=for-the-badge&logo=x&logoColor=white" alt="Follow Mia on X" />
</a>
</p>
<p>
<a href='https://ko-fi.com/Z8Z3SPLOD' target='_blank'><img height='36' style='border:0px;height:36px;' src='https://storage.ko-fi.com/cdn/kofi6.png?v=6' border='0' alt='Buy Me a Coffee at ko-fi.com' /></a>
</p>


Everything runs through two scripts:

| Script | Purpose |
|---|---|
| `./start.sh` | Download model, sync to worker, start Ray cluster, launch vLLM |
| `./stop.sh` | Tear down head and worker containers |


---

## Quick start

### 1. Prerequisites

#### Hardware

- **2× NVIDIA DGX Spark** (GB10), connected over the **200GbE RoCE fabric**
- Cluster/fabric IPs reachable between the two nodes (e.g. `10.0.0.1` ↔ `10.0.0.2`)

#### Both nodes

| Requirement | Notes |
|---|---|
| Docker with GPU support | `docker run --gpus all` must work |
| Disk space | ~181 GB for model weights + ~19 GB for the Docker image (first run) |
| `rsync` | Installed on both nodes (model sync uses `rsync` over SSH) |

#### Head node only

This is where you run `./start.sh`.

| Requirement | Notes |
|---|---|
| This repo | Clone or copy `start.sh` and `stop.sh` onto the head |
| `bash`, `docker`, `rsync`, `python3` | Used by `start.sh` |
| `huggingface_hub` | `pip install huggingface_hub` — downloads the model on first run |
| Outbound `ghcr.io` | Pulls the Docker image (~19 GB, public — no login) |
| Outbound `huggingface.co` | Downloads [kodelow/Hy3-NVFP4-W4A16](https://huggingface.co/kodelow/Hy3-NVFP4-W4A16) |
| Passwordless SSH to worker | As `REMOTE_USER` (default: your login) — used for image copy, model rsync, and starting the worker container |
| Cluster SSH key (optional) | `/etc/kamiwaza/ssl/cluster.key` — used automatically if present; override with `SSH_KEY` |

The worker does **not** need outbound internet. The head pulls the image and streams it over SSH.

**Verify on the head before starting:**

```bash
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi
python3 -c "import huggingface_hub"
ssh <user>@<WORKER_IP> docker info
ssh <user>@<WORKER_IP> rsync --version
```

#### Worker node only

| Requirement | Notes |
|---|---|
| Docker with GPU support | Head starts containers here via `ssh … docker run` |
| SSH from head | Passwordless login for `REMOTE_USER` |
| Writable model cache path | Default: `~/.cache/huggingface/hub/` (created by `start.sh`) |

#### Network configuration (required before first run)

Edit the block at the top of `start.sh` (see step 2). Set the same `WORKER_IP` in `stop.sh`.

### 2. Configure network

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

### 3. Disable earlyoom (DGX Spark — required)

GB10 uses **unified memory**: Hy3 weights (~85 GB) plus KV cache and Ray leave very little free RAM. DGX Spark ships with `earlyoom` configured to **prefer killing** `vllm`, `ray`, and `python` when available memory drops below **2%**. That SIGTERM's Ray workers and takes down vLLM — often right after load or on the first request.

**Before every `./start.sh`, on the head node:**

```bash
sudo systemctl stop earlyoom
```

Verify it is stopped:

```bash
systemctl is-active earlyoom   # should print "inactive"
```

**Permanent fix** (optional) — edit `/etc/default/earlyoom` and move inference processes from `--prefer` to `--avoid`:

```bash
# Before (kills inference first):
EARLYOOM_ARGS="-m 2 -s 80 --prefer '(vllm|VLLM|...|ray|python3|python)' --avoid '(systemd|sshd|dockerd|...)'

# After (protect inference):
EARLYOOM_ARGS="-m 2 -s 80 --avoid '(vllm|VLLM|sglang|llama-server|llama-cli|trtllm|tritonserver|ray|python3|python|dockerd|containerd|systemd|sshd|dbus-daemon|NetworkManager)'"
```

Then:

```bash
sudo systemctl restart earlyoom
```

`start.sh` prints a warning if earlyoom is still running with the default `--prefer` config.

### 4. Start

```bash
./start.sh
```

The script will:

1. Pull the Docker image on head if missing (~19 GB, public GHCR — no login required), then copy it to the worker over SSH if the worker does not have it
2. Download or locate the model from HuggingFace (`kodelow/Hy3-NVFP4-W4A16`)
3. rsync weights to the worker over SSH
4. Remove any stale `hy3-head` / `hy3-worker` containers
5. Start a Ray head on this node and a Ray worker on the remote node
6. Wait until Ray reports 2 nodes
7. Launch vLLM inside the head container (tool-parser patches + MTP spec decode)
8. Follow vLLM logs in your terminal

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

## Pi Agent (recommended harness settings)

vLLM enables reasoning via `--reasoning-parser hy_v3`, but **Pi must also request thinking** in the chat template. Hy3 defaults to `reasoning_effort: no_think` unless Pi sends `chat_template_kwargs.reasoning_effort` (`low` or `high`).

Add a `Hy3 Local` provider to `~/.pi/agent/models.json` and set defaults in `~/.pi/agent/settings.json`.

### `~/.pi/agent/settings.json`

```json
{
  "defaultProvider": "Hy3 Local",
  "defaultModel": "hy3",
  "defaultThinkingLevel": "high"
}
```

Use `"high"` for full reasoning, `"low"` for lighter thinking, or `"off"` for `no_think`.

### `~/.pi/agent/models.json` — `Hy3 Local` provider

Point `baseUrl` at your head node if Pi runs elsewhere (`http://<HEAD_IP>:8888/v1`).

```json
"Hy3 Local": {
  "baseUrl": "http://localhost:8888/v1",
  "api": "openai-completions",
  "apiKey": "dummy",
  "compat": {
    "supportsDeveloperRole": false,
    "supportsReasoningEffort": false,
    "maxTokensField": "max_tokens"
  },
  "models": [
    {
      "id": "hy3",
      "name": "Hy3 295B NVFP4 MTP",
      "reasoning": true,
      "input": ["text"],
      "contextWindow": 131072,
      "maxTokens": 32768,
      "thinkingLevelMap": {
        "off": "no_think",
        "minimal": "low",
        "low": "low",
        "medium": "high",
        "high": "high",
        "xhigh": "high"
      },
      "compat": {
        "requiresReasoningContentOnAssistantMessages": true,
        "thinkingFormat": "chat-template",
        "chatTemplateKwargs": {
          "reasoning_effort": {
            "$var": "thinking.effort",
            "omitWhenOff": true
          }
        }
      }
    }
  ]
}
```

| Pi thinking level | Sent to Hy3 as | Effect |
|---|---|---|
| `off` | `no_think` | No `</think>` block |
| `low` / `minimal` | `low` | Light reasoning |
| `medium` / `high` / `xhigh` | `high` | Full reasoning |

Restart Pi or start a new session after editing these files.

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

**`docker pull` fails on head**

- Confirm outbound network access to `ghcr.io` on the head node

**Model download fails (`huggingface_hub` / `snapshot_download`)**

- Confirm outbound access to `huggingface.co` on the head node
- Install: `pip install huggingface_hub`
- For gated models, run `huggingface-cli login` first (this checkpoint is public)

**`rsync` fails during model sync**

- Install `rsync` on both nodes
- Verify SSH: `ssh <user>@<WORKER_IP> rsync --version`
- Check disk space on the worker (~181 GB)

**Image copy to worker fails**

- Verify SSH from head: `ssh <user>@<WORKER_IP> docker info`
- Re-copy manually: `docker save ghcr.io/miaai-lab/hy3-dual-dgx-spark:vllm-probe-modded | ssh <user>@<WORKER_IP> docker load`

**Health check**

```bash
curl http://<HEAD_IP>:8888/health
```

Returns `200` once model load completes.

**vLLM shuts down right after load or first request (`EngineDeadError`, `RayWorkerProc died`, `[shutdown] MPClient`)**

Almost always **earlyoom** on DGX Spark. Check:

```bash
journalctl --since "10 min ago" | grep -E "earlyoom|SIGTERM|ray::RayWorker"
free -h
```

If you see `sending SIGTERM to process ... ray::RayWorkerP` and available memory is near **2%**, stop earlyoom before serving:

```bash
sudo systemctl stop earlyoom
./stop.sh && ./start.sh
```

See [step 3 — Disable earlyoom](#3-disable-earlyoom-dgx-spark--required).

**Pi shows no thinking / `reasoning: 0` tokens**

Confirm `defaultThinkingLevel` is not `"off"` in `~/.pi/agent/settings.json` and the `Hy3 Local` model entry includes `thinkingLevelMap` + `chatTemplateKwargs` (see [Pi Agent](#pi-agent-recommended-harness-settings)).

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
